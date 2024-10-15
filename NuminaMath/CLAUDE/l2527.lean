import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2527_252738

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + c
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s : ℝ, ∀ x : ℝ, f x = 0 → (∃ y : ℝ, f y = 0 ∧ y ≠ x ∧ x + y = s)) →
  s = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let f := fun x : ℝ => x^2 - 5*x + 6 - 9
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s : ℝ, ∀ x : ℝ, f x = 0 → (∃ y : ℝ, f y = 0 ∧ y ≠ x ∧ x + y = s)) →
  s = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2527_252738


namespace NUMINAMATH_CALUDE_percentage_problem_l2527_252736

theorem percentage_problem : 
  ∃ P : ℝ, (0.45 * 60 = P * 40 + 13) ∧ P = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2527_252736


namespace NUMINAMATH_CALUDE_total_wheels_l2527_252754

theorem total_wheels (bicycles tricycles : ℕ) 
  (bicycle_wheels tricycle_wheels : ℕ) : 
  bicycles = 24 → 
  tricycles = 14 → 
  bicycle_wheels = 2 → 
  tricycle_wheels = 3 → 
  bicycles * bicycle_wheels + tricycles * tricycle_wheels = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_l2527_252754


namespace NUMINAMATH_CALUDE_debby_text_messages_l2527_252798

theorem debby_text_messages 
  (total_messages : ℕ) 
  (before_noon_messages : ℕ) 
  (h1 : total_messages = 39) 
  (h2 : before_noon_messages = 21) : 
  total_messages - before_noon_messages = 18 := by
sorry

end NUMINAMATH_CALUDE_debby_text_messages_l2527_252798


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2527_252773

/-- A geometric sequence with common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n ↦ q ^ (n - 1)

theorem geometric_sequence_properties (q : ℝ) (h_q : 0 < q ∧ q < 1) :
  let a := geometric_sequence q
  (∀ n : ℕ, a (n + 1) < a n) ∧
  (∃ k : ℕ+, a (k + 1) = (a k + a (k + 2)) / 2 → q = (1 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2527_252773


namespace NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l2527_252786

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem f_of_2_eq_neg_2 : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l2527_252786


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2527_252769

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 + 4 * x + 1 = 0 ∧ 
   (k - 1) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 5 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2527_252769


namespace NUMINAMATH_CALUDE_switch_pairs_relation_l2527_252760

/-- Represents a row in the sequence --/
structure Row where
  switchPairs : ℕ
  oddBlocks : ℕ

/-- The relationship between switch pairs and odd blocks in a row --/
axiom switch_pairs_odd_blocks (r : Row) : r.switchPairs = 2 * r.oddBlocks

/-- The existence of at least one switch pair above each odd block --/
axiom switch_pair_above_odd_block (rn : Row) (rn_minus_1 : Row) :
  rn.oddBlocks ≤ rn_minus_1.switchPairs

/-- Theorem: The number of switch pairs in row n is at most twice 
    the number of switch pairs in row n-1 --/
theorem switch_pairs_relation (rn : Row) (rn_minus_1 : Row) :
  rn.switchPairs ≤ 2 * rn_minus_1.switchPairs := by
  sorry

end NUMINAMATH_CALUDE_switch_pairs_relation_l2527_252760


namespace NUMINAMATH_CALUDE_subtraction_value_problem_l2527_252776

theorem subtraction_value_problem (x y : ℝ) : 
  ((x - 5) / 7 = 7) → ((x - y) / 10 = 3) → y = 24 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_value_problem_l2527_252776


namespace NUMINAMATH_CALUDE_greendale_points_are_130_l2527_252715

/-- Calculates the total points for Greendale High School in a basketball tournament -/
def greendalePoints : ℕ :=
  let rooseveltFirstGame : ℕ := 30
  let rooseveltSecondGame : ℕ := rooseveltFirstGame / 2
  let rooseveltThirdGame : ℕ := rooseveltSecondGame * 3
  let rooseveltTotalBeforeBonus : ℕ := rooseveltFirstGame + rooseveltSecondGame + rooseveltThirdGame
  let rooseveltBonus : ℕ := 50
  let rooseveltTotal : ℕ := rooseveltTotalBeforeBonus + rooseveltBonus
  let pointDifference : ℕ := 10
  rooseveltTotal - pointDifference

/-- Theorem stating that Greendale High School's total points are 130 -/
theorem greendale_points_are_130 : greendalePoints = 130 := by
  sorry

end NUMINAMATH_CALUDE_greendale_points_are_130_l2527_252715


namespace NUMINAMATH_CALUDE_room_occupancy_l2527_252752

theorem room_occupancy (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (2 : ℚ) / 3 * chairs ∧ 
  chairs - (2 : ℚ) / 3 * chairs = 8 →
  people = 27 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l2527_252752


namespace NUMINAMATH_CALUDE_hari_investment_is_8280_l2527_252758

/-- Represents the business partnership between Praveen and Hari --/
structure Partnership where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  total_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's investment given the partnership details --/
def calculate_hari_investment (p : Partnership) : ℕ :=
  (3 * p.praveen_investment * p.total_months) / (2 * p.hari_months)

/-- Theorem stating that Hari's investment is 8280 Rs given the specific partnership conditions --/
theorem hari_investment_is_8280 :
  let p : Partnership := {
    praveen_investment := 3220,
    praveen_months := 12,
    hari_months := 7,
    total_months := 12,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  calculate_hari_investment p = 8280 := by
  sorry


end NUMINAMATH_CALUDE_hari_investment_is_8280_l2527_252758


namespace NUMINAMATH_CALUDE_math_team_selection_ways_l2527_252717

def num_boys : ℕ := 6
def num_girls : ℕ := 8
def team_size : ℕ := 4
def min_boys : ℕ := 2

theorem math_team_selection_ways :
  (Finset.sum (Finset.range (team_size - min_boys + 1))
    (fun k => Nat.choose num_boys (min_boys + k) * Nat.choose num_girls (team_size - (min_boys + k)))) = 595 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_ways_l2527_252717


namespace NUMINAMATH_CALUDE_speed_of_northern_cyclist_l2527_252732

/-- Theorem: Speed of northern cyclist
Given two cyclists starting from the same place in opposite directions,
with one going north at speed v kmph and the other going south at 40 kmph,
if they are 50 km apart after 0.7142857142857143 hours, then v = 30 kmph. -/
theorem speed_of_northern_cyclist (v : ℝ) : 
  v > 0 → -- Assuming positive speed
  50 = (v + 40) * 0.7142857142857143 →
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_northern_cyclist_l2527_252732


namespace NUMINAMATH_CALUDE_neither_odd_nor_even_and_increasing_l2527_252730

-- Define the function f(x) = |x + 1|
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem neither_odd_nor_even_and_increasing :
  (¬ ∀ x, f (-x) = -f x) ∧  -- not odd
  (¬ ∀ x, f (-x) = f x) ∧  -- not even
  (∀ x y, 0 < x → x < y → f x < f y) -- monotonically increasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_neither_odd_nor_even_and_increasing_l2527_252730


namespace NUMINAMATH_CALUDE_circle_radius_condition_l2527_252765

theorem circle_radius_condition (x y c : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 - 6*y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l2527_252765


namespace NUMINAMATH_CALUDE_solve_equation_l2527_252793

theorem solve_equation (x : ℝ) : 3639 + 11.95 - x = 3054 → x = 596.95 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2527_252793


namespace NUMINAMATH_CALUDE_reciprocal_equation_l2527_252789

theorem reciprocal_equation (x : ℝ) : 
  (((5 * x - 1) / 6 - 2)⁻¹ = 3) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l2527_252789


namespace NUMINAMATH_CALUDE_distinct_values_of_3_3_3_3_l2527_252701

-- Define a function to represent the expression with different parenthesizations
def exprParenthesization : List (ℕ → ℕ → ℕ → ℕ) :=
  [ (λ a b c => a^(b^(c^c))),
    (λ a b c => a^((b^c)^c)),
    (λ a b c => ((a^b)^c)^c),
    (λ a b c => (a^(b^c))^c),
    (λ a b c => (a^b)^(c^c)) ]

-- Define a function to evaluate the expression for a given base
def evaluateExpr (base : ℕ) : List ℕ :=
  exprParenthesization.map (λ f => f base base base)

-- Theorem statement
theorem distinct_values_of_3_3_3_3 :
  (evaluateExpr 3).toFinset.card = 3 := by sorry


end NUMINAMATH_CALUDE_distinct_values_of_3_3_3_3_l2527_252701


namespace NUMINAMATH_CALUDE_vertical_motion_time_relation_l2527_252712

/-- Represents the vertical motion of a ball thrown upward and returning to its starting point. -/
structure VerticalMotion where
  V₀ : ℝ  -- Initial velocity
  g : ℝ   -- Gravitational acceleration
  t₁ : ℝ  -- Time to reach maximum height
  H : ℝ   -- Maximum height
  t : ℝ   -- Total time of motion

/-- The theorem stating the relationship between initial velocity, gravity, and total time of motion. -/
theorem vertical_motion_time_relation (motion : VerticalMotion)
  (h_positive_V₀ : 0 < motion.V₀)
  (h_positive_g : 0 < motion.g)
  (h_max_height : motion.H = (1/2) * motion.g * motion.t₁^2)
  (h_symmetry : motion.t = 2 * motion.t₁) :
  motion.t = 2 * motion.V₀ / motion.g :=
by sorry

end NUMINAMATH_CALUDE_vertical_motion_time_relation_l2527_252712


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2527_252790

def B (x : ℕ) : ℕ := 3 * x

def S (x : ℕ) : ℕ := x + 100

def game_sequence (N : ℕ) : ℕ := B (S (B (S (B N))))

theorem smallest_winning_number :
  ∀ N : ℕ, 0 ≤ N ∧ N ≤ 1999 →
    (∀ M : ℕ, 0 ≤ M ∧ M < N → S (B (S (B M))) ≤ 2000) ∧
    2000 < game_sequence N ∧
    S (B (S (B N))) ≤ 2000 →
    N = 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2527_252790


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2527_252747

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), r > 0 → 4 * π * r^2 = 9 * π → (4 / 3) * π * r^3 = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2527_252747


namespace NUMINAMATH_CALUDE_smallest_number_l2527_252795

def binary_to_decimal (n : ℕ) : ℕ := n

def base_6_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

def octal_to_decimal (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := binary_to_decimal 111111
  let b := base_6_to_decimal 210
  let c := base_4_to_decimal 1000
  let d := octal_to_decimal 101
  a < b ∧ a < c ∧ a < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2527_252795


namespace NUMINAMATH_CALUDE_fish_population_approximation_l2527_252716

/-- Represents the fish population in an ocean reserve --/
structure FishPopulation where
  initialPopulation : ℕ
  taggedFish : ℕ
  secondCatchSize : ℕ
  taggedInSecondCatch : ℕ
  monthlyMigration : ℕ
  monthlyDeaths : ℕ
  months : ℕ

/-- Calculates the approximate number of fish in the ocean reserve after a given number of months --/
def approximateFishPopulation (fp : FishPopulation) : ℕ :=
  let totalChange := fp.months * (fp.monthlyMigration - fp.monthlyDeaths)
  let finalPopulation := (fp.secondCatchSize * fp.taggedFish) / fp.taggedInSecondCatch + totalChange
  finalPopulation

/-- Theorem stating that the approximate number of fish in the ocean reserve after three months is 71429 --/
theorem fish_population_approximation (fp : FishPopulation) 
  (h1 : fp.taggedFish = 1000)
  (h2 : fp.secondCatchSize = 500)
  (h3 : fp.taggedInSecondCatch = 7)
  (h4 : fp.monthlyMigration = 150)
  (h5 : fp.monthlyDeaths = 200)
  (h6 : fp.months = 3) :
  approximateFishPopulation fp = 71429 := by
  sorry

#eval approximateFishPopulation {
  initialPopulation := 0,  -- Not used in the calculation
  taggedFish := 1000,
  secondCatchSize := 500,
  taggedInSecondCatch := 7,
  monthlyMigration := 150,
  monthlyDeaths := 200,
  months := 3
}

end NUMINAMATH_CALUDE_fish_population_approximation_l2527_252716


namespace NUMINAMATH_CALUDE_no_real_distinct_roots_l2527_252723

theorem no_real_distinct_roots (k : ℝ) : 
  ¬∃ (x y : ℝ), x ≠ y ∧ x^2 + 2*k*x + 3*k^2 = 0 ∧ y^2 + 2*k*y + 3*k^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_distinct_roots_l2527_252723


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_theorem_l2527_252783

theorem lcm_gcd_sum_theorem : 
  (Nat.lcm 12 18 * Nat.gcd 12 18) + (Nat.lcm 10 15 * Nat.gcd 10 15) = 366 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_theorem_l2527_252783


namespace NUMINAMATH_CALUDE_cheryl_mms_l2527_252741

/-- The number of m&m's Cheryl ate after lunch -/
def lunch_mms : ℕ := 7

/-- The number of m&m's Cheryl ate after dinner -/
def dinner_mms : ℕ := 5

/-- The number of m&m's Cheryl gave to her sister -/
def sister_mms : ℕ := 13

/-- The total number of m&m's Cheryl had at the beginning -/
def total_mms : ℕ := lunch_mms + dinner_mms + sister_mms

theorem cheryl_mms : total_mms = 25 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_mms_l2527_252741


namespace NUMINAMATH_CALUDE_random_events_l2527_252782

-- Define a type for the events
inductive Event
  | addition
  | subtraction
  | multiplication
  | division

-- Define a function to check if an event is random
def is_random (e : Event) : Prop :=
  match e with
  | Event.addition => ∃ (a b : ℝ), a * b < 0 ∧ a + b < 0
  | Event.subtraction => ∃ (a b : ℝ), a * b < 0 ∧ a - b > 0
  | Event.multiplication => false
  | Event.division => true

-- Theorem stating which events are random
theorem random_events :
  (is_random Event.addition) ∧
  (is_random Event.subtraction) ∧
  (¬ is_random Event.multiplication) ∧
  (¬ is_random Event.division) := by
  sorry

end NUMINAMATH_CALUDE_random_events_l2527_252782


namespace NUMINAMATH_CALUDE_problem_statement_l2527_252797

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem statement
theorem problem_statement :
  (∃ m : ℝ, m > 0 ∧
    (Set.Icc (-2 : ℝ) 2 = { x | f (x + 1/2) ≤ 2*m + 1 }) ∧
    m = 3/2) ∧
  (∀ x y : ℝ, f x ≤ 2^y + 4/2^y + |2*x + 3|) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2527_252797


namespace NUMINAMATH_CALUDE_football_player_average_increase_l2527_252746

theorem football_player_average_increase (goals_fifth_match : ℕ) (total_goals : ℕ) :
  goals_fifth_match = 2 →
  total_goals = 8 →
  (total_goals / 5 : ℚ) - ((total_goals - goals_fifth_match) / 4 : ℚ) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_football_player_average_increase_l2527_252746


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2527_252703

theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 60 →
  B = 85 →
  C = D →
  E = 2 * C + 15 →
  A + B + C + D + E = 540 →
  max A (max B (max C (max D E))) = 205 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2527_252703


namespace NUMINAMATH_CALUDE_initial_men_count_l2527_252799

theorem initial_men_count (M : ℝ) : 
  M * 17 = (M + 320) * 14.010989010989011 → M = 1500 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2527_252799


namespace NUMINAMATH_CALUDE_problem_1_l2527_252720

theorem problem_1 : Real.sqrt 8 / Real.sqrt 2 + (Real.sqrt 5 + 3) * (Real.sqrt 5 - 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2527_252720


namespace NUMINAMATH_CALUDE_union_complement_equality_l2527_252791

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set M
def M : Set Nat := {0, 1, 2}

-- Define set N
def N : Set Nat := {0, 2, 3}

-- Theorem statement
theorem union_complement_equality : M ∪ (U \ N) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l2527_252791


namespace NUMINAMATH_CALUDE_michael_spending_l2527_252785

def fair_spending (initial_amount snack_cost : ℕ) : ℕ :=
  let game_cost := 3 * snack_cost
  let total_spent := snack_cost + game_cost
  initial_amount - total_spent

theorem michael_spending :
  fair_spending 80 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_michael_spending_l2527_252785


namespace NUMINAMATH_CALUDE_initial_strawberry_jelly_beans_proof_initial_strawberry_jelly_beans_l2527_252744

theorem initial_strawberry_jelly_beans : ℕ → ℕ → Prop :=
  fun s g =>
    s = 3 * g ∧ (s - 15 = 4 * (g - 15)) → s = 135

-- The proof is omitted
theorem proof_initial_strawberry_jelly_beans :
  ∀ s g : ℕ, initial_strawberry_jelly_beans s g :=
sorry

end NUMINAMATH_CALUDE_initial_strawberry_jelly_beans_proof_initial_strawberry_jelly_beans_l2527_252744


namespace NUMINAMATH_CALUDE_davids_biology_marks_l2527_252755

/-- Given David's marks in four subjects and his average marks across five subjects,
    proves that his marks in Biology are 90. -/
theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℚ)
  (h1 : english = 74)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 75.6)
  (h6 : average = (english + mathematics + physics + chemistry + biology) / 5) :
  biology = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_davids_biology_marks_l2527_252755


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_min_beta_delta_value_l2527_252722

open Complex

/-- The function g(z) as defined in the problem -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*I)*z^2 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum :
  ∃ (β δ : ℂ), 
    (g β δ 1).im = 0 ∧ 
    (g β δ (-I)).im = 0 ∧
    ∀ (β' δ' : ℂ), (g β' δ' 1).im = 0 → (g β' δ' (-I)).im = 0 → 
      abs β + abs δ ≤ abs β' + abs δ' :=
by
  sorry

/-- The actual minimum value of |β| + |δ| -/
theorem min_beta_delta_value :
  ∃ (β δ : ℂ), 
    (g β δ 1).im = 0 ∧ 
    (g β δ (-I)).im = 0 ∧
    abs β + abs δ = Real.sqrt 40 :=
by
  sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_min_beta_delta_value_l2527_252722


namespace NUMINAMATH_CALUDE_expression_equality_l2527_252700

theorem expression_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x = y * z) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (x + y + y * z)⁻¹ * ((x * y)⁻¹ + (y * z)⁻¹ + (x * z)⁻¹) = 
  1 / (y^3 * z^3 * (y + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_expression_equality_l2527_252700


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l2527_252792

/-- Represents the number of cans in each pack size -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | PackSize.small => 8
  | PackSize.medium => 16
  | PackSize.large => 32

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a pack combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Theorem: The minimum number of packs to buy 120 cans is 5 -/
theorem min_packs_for_120_cans :
  ∃ (c : PackCombination),
    totalCans c = 120 ∧
    totalPacks c = 5 ∧
    (∀ (c' : PackCombination), totalCans c' = 120 → totalPacks c' ≥ 5) :=
by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l2527_252792


namespace NUMINAMATH_CALUDE_revenue_maximizing_price_l2527_252704

/-- Revenue function for the bookstore --/
def revenue (p : ℝ) : ℝ := p * (200 - 6 * p)

/-- The maximum price constraint --/
def max_price : ℝ := 30

/-- Theorem stating the price that maximizes revenue --/
theorem revenue_maximizing_price :
  ∃ (p : ℝ), p ≤ max_price ∧ 
  ∀ (q : ℝ), q ≤ max_price → revenue q ≤ revenue p ∧
  p = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_revenue_maximizing_price_l2527_252704


namespace NUMINAMATH_CALUDE_root_expression_value_l2527_252725

theorem root_expression_value (a : ℝ) : 
  2 * a^2 - 7 * a - 1 = 0 → a * (2 * a - 7) + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_expression_value_l2527_252725


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2527_252718

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 40 > 0 ↔ x < -5 ∨ x > 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2527_252718


namespace NUMINAMATH_CALUDE_sqrt5_irrational_l2527_252781

theorem sqrt5_irrational : Irrational (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_irrational_l2527_252781


namespace NUMINAMATH_CALUDE_least_common_denominator_l2527_252742

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l2527_252742


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2527_252753

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  QS : ℝ
  PS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of the specific tetrahedron is 24/√737 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    QR := 5,
    QS := 5,
    PS := 4,
    RS := 15/4 * Real.sqrt 2
  }
  tetrahedronVolume t = 24 / Real.sqrt 737 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2527_252753


namespace NUMINAMATH_CALUDE_exists_abc_sum_product_l2527_252733

def NatPos := {n : ℕ | n > 0}

def A : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m}
def B : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m - 1}
def C : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m - 2}

theorem exists_abc_sum_product (a : ℕ) (b : ℕ) (c : ℕ) 
  (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  ∃ a b c, a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ 2006 = a + b * c :=
by sorry

end NUMINAMATH_CALUDE_exists_abc_sum_product_l2527_252733


namespace NUMINAMATH_CALUDE_volume_difference_rectangular_prisms_volume_difference_specific_bowls_l2527_252775

/-- The volume difference between two rectangular prisms with the same width and length
    but different heights is equal to the product of the width, length, and the difference in heights. -/
theorem volume_difference_rectangular_prisms
  (w : ℝ) (l : ℝ) (h₁ : ℝ) (h₂ : ℝ)
  (hw : w > 0) (hl : l > 0) (hh₁ : h₁ > 0) (hh₂ : h₂ > 0) :
  w * l * h₁ - w * l * h₂ = w * l * (h₁ - h₂) :=
by sorry

/-- The volume difference between two specific bowls -/
theorem volume_difference_specific_bowls :
  (16 : ℝ) * 14 * 9 - (16 : ℝ) * 14 * 4 = 1120 :=
by sorry

end NUMINAMATH_CALUDE_volume_difference_rectangular_prisms_volume_difference_specific_bowls_l2527_252775


namespace NUMINAMATH_CALUDE_certain_number_proof_l2527_252761

def smallest_number : ℕ := 3153
def increase : ℕ := 3
def divisor1 : ℕ := 70
def divisor2 : ℕ := 25
def divisor3 : ℕ := 21

theorem certain_number_proof :
  ∃ (n : ℕ), n > 0 ∧
  (smallest_number + increase) % n = 0 ∧
  n % divisor1 = 0 ∧
  n % divisor2 = 0 ∧
  n % divisor3 = 0 ∧
  ∀ (m : ℕ), m > 0 →
    (smallest_number + increase) % m = 0 →
    m % divisor1 = 0 →
    m % divisor2 = 0 →
    m % divisor3 = 0 →
    n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2527_252761


namespace NUMINAMATH_CALUDE_parabola_translation_l2527_252768

-- Define the base parabola
def base_parabola (x : ℝ) : ℝ := x^2

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := (x + 4)^2 - 5

-- Theorem stating the translation process
theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola x = base_parabola (x + 4) - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2527_252768


namespace NUMINAMATH_CALUDE_system_solution_l2527_252735

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 2 * y = 8) ∧ 
  (x + 3 * y = 9) ∧ 
  (x = 42 / 11) ∧ 
  (y = 19 / 11) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2527_252735


namespace NUMINAMATH_CALUDE_tammy_mountain_climb_l2527_252762

/-- Tammy's mountain climbing problem -/
theorem tammy_mountain_climb 
  (total_time : ℝ) 
  (second_day_speed : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) :
  total_time = 14 →
  second_day_speed = 4 →
  speed_difference = 0.5 →
  time_difference = 2 →
  ∃ (first_day_time second_day_time : ℝ),
    first_day_time + second_day_time = total_time ∧
    second_day_time = first_day_time - time_difference ∧
    ∃ (first_day_speed : ℝ),
      first_day_speed = second_day_speed - speed_difference ∧
      first_day_speed * first_day_time + second_day_speed * second_day_time = 52 :=
by sorry

end NUMINAMATH_CALUDE_tammy_mountain_climb_l2527_252762


namespace NUMINAMATH_CALUDE_biff_break_even_hours_l2527_252767

/-- Calculates the number of hours required to break even on a bus trip. -/
def hours_to_break_even (ticket_cost snacks_cost headphones_cost hourly_rate wifi_cost : ℚ) : ℚ :=
  let total_expenses := ticket_cost + snacks_cost + headphones_cost
  let net_hourly_rate := hourly_rate - wifi_cost
  total_expenses / net_hourly_rate

/-- Proves that given Biff's expenses and earnings, the number of hours required to break even on a bus trip is 3 hours. -/
theorem biff_break_even_hours :
  hours_to_break_even 11 3 16 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_biff_break_even_hours_l2527_252767


namespace NUMINAMATH_CALUDE_set_01_proper_subset_N_l2527_252778

-- Define the set of natural numbers
def N : Set ℕ := Set.univ

-- Define the set {0,1}
def set_01 : Set ℕ := {0, 1}

-- Theorem to prove
theorem set_01_proper_subset_N : set_01 ⊂ N := by sorry

end NUMINAMATH_CALUDE_set_01_proper_subset_N_l2527_252778


namespace NUMINAMATH_CALUDE_parabola_no_x_intersection_l2527_252709

/-- The parabola defined by y = -2x^2 + x - 1 has no intersection with the x-axis -/
theorem parabola_no_x_intersection :
  ∀ x : ℝ, -2 * x^2 + x - 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_no_x_intersection_l2527_252709


namespace NUMINAMATH_CALUDE_mother_triple_daughter_age_l2527_252763

/-- Represents the age difference between mother and daughter -/
def age_difference : ℕ := 42 - 8

/-- Represents the current age of the mother -/
def mother_age : ℕ := 42

/-- Represents the current age of the daughter -/
def daughter_age : ℕ := 8

/-- The number of years until the mother is three times as old as her daughter -/
def years_until_triple : ℕ := 9

theorem mother_triple_daughter_age :
  mother_age + years_until_triple = 3 * (daughter_age + years_until_triple) :=
sorry

end NUMINAMATH_CALUDE_mother_triple_daughter_age_l2527_252763


namespace NUMINAMATH_CALUDE_dans_initial_money_l2527_252740

def candy_price : ℕ := 2
def chocolate_price : ℕ := 3

theorem dans_initial_money :
  ∀ (initial_money : ℕ),
  (initial_money = candy_price + chocolate_price) ∧
  (chocolate_price - candy_price = 1) →
  initial_money = 5 := by
sorry

end NUMINAMATH_CALUDE_dans_initial_money_l2527_252740


namespace NUMINAMATH_CALUDE_tens_digit_of_subtraction_l2527_252777

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hun_less_than_tens : hundreds = tens - 3
  tens_double_units : tens = 2 * units
  is_three_digit : hundreds ≥ 1 ∧ hundreds ≤ 9

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

def ThreeDigitNumber.reversed (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

theorem tens_digit_of_subtraction (n : ThreeDigitNumber) :
  (n.toNat - n.reversed) / 10 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_subtraction_l2527_252777


namespace NUMINAMATH_CALUDE_gcd_90_252_l2527_252727

theorem gcd_90_252 : Nat.gcd 90 252 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_252_l2527_252727


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l2527_252711

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^2 < 2000 → x ≤ 44 ∧ ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^2 < 2000 ∧ y = 44 := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l2527_252711


namespace NUMINAMATH_CALUDE_treehouse_planks_l2527_252707

theorem treehouse_planks (initial_planks : ℕ) (total_planks : ℕ) (planks_from_forest : ℕ) :
  initial_planks = 15 →
  total_planks = 35 →
  total_planks = initial_planks + 2 * planks_from_forest →
  planks_from_forest = 10 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_planks_l2527_252707


namespace NUMINAMATH_CALUDE_profit_maximization_l2527_252728

/-- The profit function for a product with cost 20 yuan per kilogram -/
def profit_function (x : ℝ) : ℝ := (x - 20) * (-x + 150)

/-- The sales volume function -/
def sales_volume (x : ℝ) : ℝ := -x + 150

theorem profit_maximization :
  ∃ (max_price max_profit : ℝ),
    (∀ x : ℝ, 20 ≤ x ∧ x ≤ 90 → profit_function x ≤ max_profit) ∧
    max_price = 85 ∧
    max_profit = 4225 ∧
    profit_function max_price = max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l2527_252728


namespace NUMINAMATH_CALUDE_subset_sum_theorem_l2527_252714

theorem subset_sum_theorem (A : Finset ℤ) (h_card : A.card = 4) 
  (h_order : ∃ (a₁ a₂ a₃ a₄ : ℤ), A = {a₁, a₂, a₃, a₄} ∧ a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄) 
  (h_subset_sums : (A.powerset.filter (fun s => s.card = 3)).image (fun s => s.sum id) = {-1, 3, 5, 8}) :
  A = {-3, 0, 2, 6} := by
sorry

end NUMINAMATH_CALUDE_subset_sum_theorem_l2527_252714


namespace NUMINAMATH_CALUDE_coefficient_without_x_is_70_l2527_252708

/-- The coefficient of the term without x in (xy - 1/x)^8 -/
def coefficientWithoutX : ℕ :=
  Nat.choose 8 4

/-- Theorem: The coefficient of the term without x in (xy - 1/x)^8 is 70 -/
theorem coefficient_without_x_is_70 : coefficientWithoutX = 70 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_without_x_is_70_l2527_252708


namespace NUMINAMATH_CALUDE_star_operation_result_l2527_252702

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem star_operation_result :
  star (star Element.three Element.one) (star Element.four Element.two) = Element.three := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l2527_252702


namespace NUMINAMATH_CALUDE_class_average_l2527_252766

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (high_score : ℕ) 
  (zero_scorers : ℕ) (rest_average : ℕ) : 
  total_students = 27 →
  high_scorers = 5 →
  high_score = 95 →
  zero_scorers = 3 →
  rest_average = 45 →
  (total_students - high_scorers - zero_scorers) * rest_average + 
    high_scorers * high_score = 1330 →
  (1330 : ℚ) / total_students = 1330 / 27 := by
sorry

end NUMINAMATH_CALUDE_class_average_l2527_252766


namespace NUMINAMATH_CALUDE_repeating_decimal_one_point_foursix_equals_fraction_l2527_252721

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to its rational representation. -/
def repeating_decimal_to_rational (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (99 : ℚ)

theorem repeating_decimal_one_point_foursix_equals_fraction :
  repeating_decimal_to_rational ⟨1, 46⟩ = 145 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_one_point_foursix_equals_fraction_l2527_252721


namespace NUMINAMATH_CALUDE_percentage_five_digit_numbers_with_repeated_digits_l2527_252796

theorem percentage_five_digit_numbers_with_repeated_digits :
  let total_five_digit_numbers : ℕ := 90000
  let five_digit_numbers_without_repeats : ℕ := 27216
  let five_digit_numbers_with_repeats : ℕ := total_five_digit_numbers - five_digit_numbers_without_repeats
  let percentage : ℚ := (five_digit_numbers_with_repeats : ℚ) / (total_five_digit_numbers : ℚ) * 100
  ∃ (ε : ℚ), abs (percentage - 69.8) < ε ∧ ε ≤ 0.05 :=
by sorry

end NUMINAMATH_CALUDE_percentage_five_digit_numbers_with_repeated_digits_l2527_252796


namespace NUMINAMATH_CALUDE_sally_quarters_remaining_l2527_252770

def initial_quarters : ℕ := 760
def first_purchase : ℕ := 418
def second_purchase : ℕ := 192

theorem sally_quarters_remaining :
  initial_quarters - first_purchase - second_purchase = 150 := by sorry

end NUMINAMATH_CALUDE_sally_quarters_remaining_l2527_252770


namespace NUMINAMATH_CALUDE_price_difference_per_can_l2527_252787

/-- Proves that the difference in price per can between the grocery store and bulk warehouse is 25 cents -/
theorem price_difference_per_can (bulk_price bulk_quantity grocery_price grocery_quantity : ℚ) : 
  bulk_price = 12 →
  bulk_quantity = 48 →
  grocery_price = 6 →
  grocery_quantity = 12 →
  (grocery_price / grocery_quantity - bulk_price / bulk_quantity) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_per_can_l2527_252787


namespace NUMINAMATH_CALUDE_no_valid_domino_placement_without_2x2_square_l2527_252788

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Represents a domino placement on the chessboard -/
def DominoPlacement := List (Fin 8 × Fin 8 × Bool)

/-- Checks if a domino placement is valid (covers the entire board without overlaps) -/
def isValidPlacement (board : Chessboard) (placement : DominoPlacement) : Prop :=
  sorry

/-- Checks if a domino placement forms a 2x2 square -/
def forms2x2Square (placement : DominoPlacement) : Prop :=
  sorry

/-- The main theorem: it's impossible to cover an 8x8 chessboard with 2x1 dominoes
    without forming a 2x2 square -/
theorem no_valid_domino_placement_without_2x2_square :
  ¬ ∃ (board : Chessboard) (placement : DominoPlacement),
    isValidPlacement board placement ∧ ¬ forms2x2Square placement :=
  sorry

end NUMINAMATH_CALUDE_no_valid_domino_placement_without_2x2_square_l2527_252788


namespace NUMINAMATH_CALUDE_spa_nail_polish_problem_l2527_252739

/-- The number of girls who went to the spa -/
def num_girls : ℕ := 8

/-- The number of fingers on each limb -/
def fingers_per_limb : ℕ := 5

/-- The total number of fingers polished -/
def total_fingers_polished : ℕ := 40

/-- The number of limbs polished per girl -/
def limbs_per_girl : ℕ := total_fingers_polished / (num_girls * fingers_per_limb)

theorem spa_nail_polish_problem :
  limbs_per_girl = 1 :=
by sorry

end NUMINAMATH_CALUDE_spa_nail_polish_problem_l2527_252739


namespace NUMINAMATH_CALUDE_no_three_color_solution_exists_seven_color_solution_l2527_252731

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an equilateral triangle
structure EqTriangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define a coloring function type
def ColoringFunction (n : ℕ) := ℝ × ℝ → Fin n

-- Define what it means for a circle to be contained in another circle
def containedIn (c1 c2 : Circle) : Prop :=
  (c1.radius + ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2).sqrt ≤ c2.radius)

-- Define what it means for a triangle to be inscribed in a circle
def inscribedIn (t : EqTriangle) (c : Circle) : Prop := sorry

-- Define what it means for a coloring to be good for a circle
def goodColoring (f : ColoringFunction n) (c : Circle) : Prop :=
  ∀ t : EqTriangle, inscribedIn t c → (f (t.vertices 0) ≠ f (t.vertices 1) ∧ 
                                       f (t.vertices 0) ≠ f (t.vertices 2) ∧ 
                                       f (t.vertices 1) ≠ f (t.vertices 2))

-- Main theorem for part 1
theorem no_three_color_solution :
  ¬ ∃ (f : ColoringFunction 3), 
    ∀ (c : Circle), 
      containedIn c (Circle.mk (0, 0) 2) → c.radius ≥ 1 → goodColoring f c :=
sorry

-- Main theorem for part 2
theorem exists_seven_color_solution :
  ∃ (g : ColoringFunction 7), 
    ∀ (c : Circle), 
      containedIn c (Circle.mk (0, 0) 2) → c.radius ≥ 1 → goodColoring g c :=
sorry

end NUMINAMATH_CALUDE_no_three_color_solution_exists_seven_color_solution_l2527_252731


namespace NUMINAMATH_CALUDE_coal_piles_weights_l2527_252771

theorem coal_piles_weights (pile1 pile2 : ℕ) : 
  pile1 = pile2 + 80 →
  pile1 * 80 / 100 = pile2 - 50 →
  pile1 = 650 ∧ pile2 = 570 := by
sorry

end NUMINAMATH_CALUDE_coal_piles_weights_l2527_252771


namespace NUMINAMATH_CALUDE_train_speed_l2527_252749

/-- A train journey with two segments and a given average speed -/
structure TrainJourney where
  x : ℝ  -- distance of the first segment
  V : ℝ  -- speed of the train in the first segment
  avg_speed : ℝ  -- average speed for the entire journey

/-- The train journey satisfies the given conditions -/
def valid_journey (j : TrainJourney) : Prop :=
  j.x > 0 ∧ j.V > 0 ∧ j.avg_speed = 16 ∧
  (j.x / j.V + (2 * j.x) / 20) = (3 * j.x) / j.avg_speed

theorem train_speed (j : TrainJourney) (h : valid_journey j) : j.V = 40 / 7 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2527_252749


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2527_252756

theorem sqrt_difference_equality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) : ℝ) = Real.sqrt m - Real.sqrt n →
  m + n = 2011 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2527_252756


namespace NUMINAMATH_CALUDE_midpoint_property_l2527_252743

/-- Given two points A and B in the plane, if C is their midpoint,
    then 2x - 4y = 0, where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B : ℝ × ℝ) (hA : A = (20, 10)) (hB : B = (10, 5)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l2527_252743


namespace NUMINAMATH_CALUDE_hydrogen_moles_formed_l2527_252774

/-- Represents a chemical element --/
structure Element where
  name : String
  atomic_mass : Float

/-- Represents a chemical compound --/
structure Compound where
  formula : String
  elements : List (Element × Nat)

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List (Compound × Float)
  products : List (Compound × Float)

/-- Calculate the molar mass of a compound --/
def molar_mass (c : Compound) : Float :=
  c.elements.foldl (fun acc (elem, count) => acc + elem.atomic_mass * count.toFloat) 0

/-- Calculate the number of moles given mass and molar mass --/
def moles (mass : Float) (molar_mass : Float) : Float :=
  mass / molar_mass

/-- The main theorem --/
theorem hydrogen_moles_formed
  (carbon : Element)
  (hydrogen : Element)
  (benzene : Compound)
  (methane : Compound)
  (toluene : Compound)
  (h2 : Compound)
  (reaction : Reaction)
  (benzene_mass : Float) :
  carbon.atomic_mass = 12.01 →
  hydrogen.atomic_mass = 1.008 →
  benzene.elements = [(carbon, 6), (hydrogen, 6)] →
  methane.elements = [(carbon, 1), (hydrogen, 4)] →
  toluene.elements = [(carbon, 7), (hydrogen, 8)] →
  h2.elements = [(hydrogen, 2)] →
  reaction.reactants = [(benzene, 1), (methane, 1)] →
  reaction.products = [(toluene, 1), (h2, 1)] →
  benzene_mass = 156 →
  moles benzene_mass (molar_mass benzene) = 2 →
  moles benzene_mass (molar_mass benzene) = moles 2 (molar_mass h2) :=
by sorry

end NUMINAMATH_CALUDE_hydrogen_moles_formed_l2527_252774


namespace NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_relation_l2527_252726

/-- Regular octahedron with side length 2√2 -/
def octahedron : Real → Set (Fin 3 → ℝ) := sorry

/-- Tetrahedron with vertices at the centers of octahedron faces -/
def tetrahedron (O : Set (Fin 3 → ℝ)) : Set (Fin 3 → ℝ) := sorry

/-- Volume of a set in ℝ³ -/
def volume (S : Set (Fin 3 → ℝ)) : ℝ := sorry

theorem octahedron_tetrahedron_volume_relation :
  let O := octahedron (2 * Real.sqrt 2)
  let T := tetrahedron O
  volume O = 4 * volume T →
  volume T = (4 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_relation_l2527_252726


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l2527_252745

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (h1 : total_clips = 81) (h2 : clips_per_box = 9) :
  total_clips / clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l2527_252745


namespace NUMINAMATH_CALUDE_infinite_non_representable_l2527_252750

/-- A natural number is representable if it can be written as p + n^(2k) for some prime p and natural numbers n and k. -/
def Representable (m : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ m = p + n^(2*k)

/-- The set of non-representable natural numbers is infinite. -/
theorem infinite_non_representable :
  {m : ℕ | ¬Representable m}.Infinite :=
sorry

end NUMINAMATH_CALUDE_infinite_non_representable_l2527_252750


namespace NUMINAMATH_CALUDE_max_value_f_l2527_252759

theorem max_value_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  let f := fun (a b c : ℝ) => (1 - b*c + c) * (1 - a*c + a) * (1 - a*b + b)
  (∀ a b c, a > 0 → b > 0 → c > 0 → a * b * c = 1 → f a b c ≤ 1) ∧
  f x y z = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_l2527_252759


namespace NUMINAMATH_CALUDE_wallet_cost_l2527_252713

theorem wallet_cost (W : ℝ) : 
  W / 2 + 15 + 2 * 15 + 5 = W → W = 100 := by
  sorry

end NUMINAMATH_CALUDE_wallet_cost_l2527_252713


namespace NUMINAMATH_CALUDE_binary_1101011_equals_107_l2527_252706

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1101011_equals_107 :
  binary_to_decimal [true, true, false, true, false, true, true] = 107 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101011_equals_107_l2527_252706


namespace NUMINAMATH_CALUDE_max_value_constraint_l2527_252780

theorem max_value_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2*x*y*Real.sqrt 6 + 9*y*z ≤ Real.sqrt 87 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2527_252780


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l2527_252748

theorem max_value_x_plus_y :
  ∃ (x y : ℝ),
    (2 * Real.sin x - 1) * (2 * Real.cos y - Real.sqrt 3) = 0 ∧
    x ∈ Set.Icc 0 (3 * Real.pi / 2) ∧
    y ∈ Set.Icc Real.pi (2 * Real.pi) ∧
    ∀ (x' y' : ℝ),
      (2 * Real.sin x' - 1) * (2 * Real.cos y' - Real.sqrt 3) = 0 →
      x' ∈ Set.Icc 0 (3 * Real.pi / 2) →
      y' ∈ Set.Icc Real.pi (2 * Real.pi) →
      x + y ≥ x' + y' ∧
    x + y = 8 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l2527_252748


namespace NUMINAMATH_CALUDE_smallest_result_is_zero_l2527_252737

def S : Set ℕ := {2, 4, 6, 8, 10, 12}

def operation (a b c : ℕ) : ℕ := ((a + b - c) * c)

theorem smallest_result_is_zero :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  operation a b c = 0 ∧
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  operation x y z ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_result_is_zero_l2527_252737


namespace NUMINAMATH_CALUDE_friday_snowfall_l2527_252757

-- Define the snowfall amounts
def total_snowfall : Float := 0.89
def wednesday_snowfall : Float := 0.33
def thursday_snowfall : Float := 0.33

-- Define the theorem
theorem friday_snowfall :
  total_snowfall - (wednesday_snowfall + thursday_snowfall) = 0.23 := by
  sorry

end NUMINAMATH_CALUDE_friday_snowfall_l2527_252757


namespace NUMINAMATH_CALUDE_los_angeles_women_ratio_l2527_252779

/-- The ratio of women to the total population in Los Angeles -/
def women_ratio (total_population women_in_retail : ℕ) (retail_fraction : ℚ) : ℚ :=
  (women_in_retail / retail_fraction) / total_population

/-- Proof that the ratio of women to the total population in Los Angeles is 1/2 -/
theorem los_angeles_women_ratio :
  women_ratio 6000000 1000000 (1/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_los_angeles_women_ratio_l2527_252779


namespace NUMINAMATH_CALUDE_students_with_both_pets_l2527_252729

theorem students_with_both_pets (total : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total = 50)
  (h2 : dog_owners = 35)
  (h3 : cat_owners = 40)
  (h4 : dog_owners + cat_owners - total ≤ dog_owners)
  (h5 : dog_owners + cat_owners - total ≤ cat_owners) :
  dog_owners + cat_owners - total = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_with_both_pets_l2527_252729


namespace NUMINAMATH_CALUDE_movie_profit_calculation_l2527_252719

def actor_cost : ℕ := 1200
def num_people : ℕ := 50
def food_cost_per_person : ℕ := 3
def movie_selling_price : ℕ := 10000

def total_food_cost : ℕ := num_people * food_cost_per_person
def actors_and_food_cost : ℕ := actor_cost + total_food_cost
def equipment_rental_cost : ℕ := 2 * actors_and_food_cost
def total_cost : ℕ := actors_and_food_cost + equipment_rental_cost

theorem movie_profit_calculation :
  movie_selling_price - total_cost = 5950 := by
  sorry

end NUMINAMATH_CALUDE_movie_profit_calculation_l2527_252719


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l2527_252724

theorem last_two_digits_sum (n : ℕ) : n = 25 → (6^n + 14^n) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l2527_252724


namespace NUMINAMATH_CALUDE_cherry_pie_count_l2527_252710

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) : 
  total_pies = 36 →
  apple_ratio = 2 →
  blueberry_ratio = 5 →
  cherry_ratio = 4 →
  (cherry_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_count_l2527_252710


namespace NUMINAMATH_CALUDE_product_of_odd_is_even_correct_propositions_count_l2527_252772

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The product of two odd functions is even -/
theorem product_of_odd_is_even (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsOdd g) :
    IsEven (fun x ↦ f x * g x) := by
  sorry

/-- There are exactly two correct propositions among the original, converse, negation, and contrapositive -/
theorem correct_propositions_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_product_of_odd_is_even_correct_propositions_count_l2527_252772


namespace NUMINAMATH_CALUDE_garden_feet_is_117_l2527_252794

/-- The number of feet in a garden with various animals --/
def garden_feet : ℕ :=
  let normal_dog_count : ℕ := 5
  let normal_cat_count : ℕ := 3
  let normal_bird_count : ℕ := 6
  let duck_count : ℕ := 2
  let insect_count : ℕ := 10
  let three_legged_dog_count : ℕ := 1
  let three_legged_cat_count : ℕ := 1
  let three_legged_bird_count : ℕ := 1
  let dog_legs : ℕ := normal_dog_count * 4 + three_legged_dog_count * 3
  let cat_legs : ℕ := normal_cat_count * 4 + three_legged_cat_count * 3
  let bird_legs : ℕ := normal_bird_count * 2 + three_legged_bird_count * 3
  let duck_legs : ℕ := duck_count * 2
  let insect_legs : ℕ := insect_count * 6
  dog_legs + cat_legs + bird_legs + duck_legs + insect_legs

theorem garden_feet_is_117 : garden_feet = 117 := by
  sorry

end NUMINAMATH_CALUDE_garden_feet_is_117_l2527_252794


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2527_252784

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq_20 : x + y + z = 20)
  (first_eq_four_times_sum_others : x = 4 * (y + z))
  (second_eq_seven_times_third : y = 7 * z) : 
  x * y * z = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2527_252784


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2527_252705

def expression : ℕ := 16^4 + 2 * 16^2 + 1 - 15^4

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2527_252705


namespace NUMINAMATH_CALUDE_smallest_n_after_tax_l2527_252764

theorem smallest_n_after_tax : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ (104 * m = 100 * 100 * n)) ∧ 
  (∀ (k : ℕ), k > 0 → k < n → ¬∃ (j : ℕ), j > 0 ∧ (104 * j = 100 * 100 * k)) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_after_tax_l2527_252764


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2527_252751

theorem rationalize_denominator : 7 / Real.sqrt 200 = (7 * Real.sqrt 2) / 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2527_252751


namespace NUMINAMATH_CALUDE_electrician_wage_l2527_252734

theorem electrician_wage (total_hours : ℝ) (bricklayer_wage : ℝ) (total_payment : ℝ) (individual_hours : ℝ)
  (h1 : total_hours = 90)
  (h2 : bricklayer_wage = 12)
  (h3 : total_payment = 1350)
  (h4 : individual_hours = 22.5) :
  (total_payment - bricklayer_wage * individual_hours) / individual_hours = 48 := by
sorry

end NUMINAMATH_CALUDE_electrician_wage_l2527_252734
