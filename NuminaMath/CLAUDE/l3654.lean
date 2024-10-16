import Mathlib

namespace NUMINAMATH_CALUDE_sally_earnings_l3654_365466

def earnings_per_house : ℕ := 25
def houses_cleaned : ℕ := 96

theorem sally_earnings :
  (earnings_per_house * houses_cleaned) / 12 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sally_earnings_l3654_365466


namespace NUMINAMATH_CALUDE_megans_work_hours_l3654_365459

/-- Megan's work problem -/
theorem megans_work_hours
  (hourly_rate : ℝ)
  (days_per_month : ℕ)
  (total_earnings : ℝ)
  (h : hourly_rate = 7.5)
  (d : days_per_month = 20)
  (e : total_earnings = 2400) :
  ∃ (hours_per_day : ℝ),
    hours_per_day * hourly_rate * (2 * days_per_month) = total_earnings ∧
    hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_megans_work_hours_l3654_365459


namespace NUMINAMATH_CALUDE_right_triangle_proof_l3654_365426

open Real

theorem right_triangle_proof (A B C : ℝ) (a b c : ℝ) (h1 : b ≠ 1) 
  (h2 : C / A = 2) (h3 : sin B / sin A = 2) (h4 : A + B + C = π) :
  A = π / 6 ∧ B = π / 2 ∧ C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_proof_l3654_365426


namespace NUMINAMATH_CALUDE_smallest_x_for_g_equality_l3654_365461

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem smallest_x_for_g_equality (g : ℝ → ℝ) : 
  (∀ (x : ℝ), x > 0 → g (4 * x) = 4 * g x) →
  (∀ (x : ℝ), 2 ≤ x ∧ x ≤ 4 → g x = 1 - |x - 3|) →
  (∀ (x : ℝ), x ≥ 0 ∧ g x = g 2048 → x ≥ 2) ∧
  g 2 = g 2048 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_g_equality_l3654_365461


namespace NUMINAMATH_CALUDE_f_increasing_interval_l3654_365470

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x

-- State the theorem
theorem f_increasing_interval :
  ∀ x y : ℝ, x ≥ 3 → y > x → f y > f x :=
sorry

end NUMINAMATH_CALUDE_f_increasing_interval_l3654_365470


namespace NUMINAMATH_CALUDE_exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary_l3654_365458

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins -/
def TwoCoinsOutcome := CoinOutcome × CoinOutcome

/-- The event of exactly one head facing up -/
def exactlyOneHead (outcome : TwoCoinsOutcome) : Prop :=
  (outcome.1 = CoinOutcome.Heads ∧ outcome.2 = CoinOutcome.Tails) ∨
  (outcome.1 = CoinOutcome.Tails ∧ outcome.2 = CoinOutcome.Heads)

/-- The event of exactly two heads facing up -/
def exactlyTwoHeads (outcome : TwoCoinsOutcome) : Prop :=
  outcome.1 = CoinOutcome.Heads ∧ outcome.2 = CoinOutcome.Heads

/-- The sample space of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads),
   (CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads),
   (CoinOutcome.Tails, CoinOutcome.Tails)}

theorem exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary :
  (∀ (outcome : TwoCoinsOutcome), ¬(exactlyOneHead outcome ∧ exactlyTwoHeads outcome)) ∧
  (∃ (outcome : TwoCoinsOutcome), ¬exactlyOneHead outcome ∧ ¬exactlyTwoHeads outcome) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary_l3654_365458


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l3654_365413

theorem square_sum_geq_product (x y z : ℝ) : x + y + z ≥ x * y * z → x^2 + y^2 + z^2 ≥ x * y * z := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l3654_365413


namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_exists_minimum_l3654_365454

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f (-x) + f (x + 5)

-- Theorem for part (I)
theorem inequality_solution (x : ℝ) : f x > 2 ↔ x > 3 ∨ x < -1 := by sorry

-- Theorem for part (II)
theorem minimum_value : ∀ x, g x ≥ 3 := by sorry

-- Theorem to show that 3 is indeed the minimum value
theorem exists_minimum : ∃ x, g x = 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_exists_minimum_l3654_365454


namespace NUMINAMATH_CALUDE_circle_trajectory_l3654_365493

/-- Given two circles and their symmetry, prove the trajectory of a third circle's center -/
theorem circle_trajectory (a l : ℝ) :
  -- Circle 1: x^2 + y^2 - ax + 2y + 1 = 0
  (∀ x y : ℝ, x^2 + y^2 - a*x + 2*y + 1 = 0 → 
    -- Circle 2: x^2 + y^2 = 1
    ∃ x' y' : ℝ, x'^2 + y'^2 = 1 ∧ 
    -- Symmetry condition
    ∃ m : ℝ, y' - y = m*(x' - x) ∧ y' + y = 2*(x' + x - l)) →
  -- Circle P passing through C(-a, a) and tangent to y-axis
  (∀ x y : ℝ, (x + a)^2 + (y - a)^2 = x^2) →
  -- Trajectory equation
  (∀ x y : ℝ, (x + a)^2 + (y - a)^2 = x^2 → y^2 + 4*x - 4*y + 8 = 0) :=
by sorry


end NUMINAMATH_CALUDE_circle_trajectory_l3654_365493


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l3654_365440

theorem modulus_of_complex_power :
  Complex.abs ((2 + 2*Complex.I)^6) = 512 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l3654_365440


namespace NUMINAMATH_CALUDE_factorial_simplification_l3654_365499

theorem factorial_simplification : (12 : ℕ).factorial / ((10 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 1320 / 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l3654_365499


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3654_365429

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 4) :
  a 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3654_365429


namespace NUMINAMATH_CALUDE_lottery_winnings_l3654_365497

/-- Calculates the total money won in a lottery given the number of tickets, winning numbers per ticket, and value per winning number. -/
def total_money_won (num_tickets : ℕ) (winning_numbers_per_ticket : ℕ) (value_per_winning_number : ℕ) : ℕ :=
  num_tickets * winning_numbers_per_ticket * value_per_winning_number

/-- Proves that with 3 lottery tickets, 5 winning numbers per ticket, and $20 per winning number, the total money won is $300. -/
theorem lottery_winnings :
  total_money_won 3 5 20 = 300 := by
  sorry

#eval total_money_won 3 5 20

end NUMINAMATH_CALUDE_lottery_winnings_l3654_365497


namespace NUMINAMATH_CALUDE_modulo_thirteen_residue_l3654_365473

theorem modulo_thirteen_residue :
  (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_modulo_thirteen_residue_l3654_365473


namespace NUMINAMATH_CALUDE_ali_age_difference_l3654_365498

/-- Given the ages of Ali and Umar, and the relationship between Umar and Yusaf's ages,
    prove that Ali is 3 years older than Yusaf. -/
theorem ali_age_difference (ali_age umar_age : ℕ) (h1 : ali_age = 8) (h2 : umar_age = 10)
  (h3 : umar_age = 2 * (umar_age / 2)) : ali_age - (umar_age / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ali_age_difference_l3654_365498


namespace NUMINAMATH_CALUDE_triangle_area_problem_l3654_365432

theorem triangle_area_problem (base_small : ℝ) (base_large : ℝ) (area_small : ℝ) :
  base_small = 14 →
  base_large = 24 →
  area_small = 35 →
  let height_small := (2 * area_small) / base_small
  let height_large := (height_small * base_large) / base_small
  (1/2 : ℝ) * base_large * height_large = 144 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l3654_365432


namespace NUMINAMATH_CALUDE_min_matches_to_reach_target_l3654_365407

def victory_points : ℕ := 3
def draw_points : ℕ := 1
def defeat_points : ℕ := 0

def initial_games : ℕ := 5
def initial_points : ℕ := 8
def target_points : ℕ := 40
def min_additional_wins : ℕ := 9

def min_total_matches : ℕ := 16

theorem min_matches_to_reach_target :
  ∀ (total_matches : ℕ),
    total_matches ≥ initial_games ∧
    total_matches ≥ initial_games + min_additional_wins ∧
    (total_matches - initial_games) * victory_points + initial_points ≥ target_points →
    total_matches ≥ min_total_matches :=
by sorry

end NUMINAMATH_CALUDE_min_matches_to_reach_target_l3654_365407


namespace NUMINAMATH_CALUDE_absolute_value_integral_l3654_365496

theorem absolute_value_integral : ∫ x in (0:ℝ)..4, |x - 2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l3654_365496


namespace NUMINAMATH_CALUDE_distance_between_foci_l3654_365492

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 7)^2) = 26

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 7)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3654_365492


namespace NUMINAMATH_CALUDE_opposite_of_miss_both_is_hit_at_least_once_l3654_365467

-- Define the sample space
def Ω : Type := Unit

-- Define the event of missing the target on both shots
def miss_both : Set Ω := sorry

-- Define the event of hitting the target at least once
def hit_at_least_once : Set Ω := sorry

-- Theorem stating that the complement of missing both shots is hitting at least once
theorem opposite_of_miss_both_is_hit_at_least_once : 
  (miss_both)ᶜ = hit_at_least_once := by sorry

end NUMINAMATH_CALUDE_opposite_of_miss_both_is_hit_at_least_once_l3654_365467


namespace NUMINAMATH_CALUDE_complement_A_in_B_union_equality_implies_m_range_l3654_365486

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | x > m}

-- Theorem 1: Complement of A in B when m = -1
theorem complement_A_in_B : 
  {x : ℝ | x ∈ B (-1) ∧ x ∉ A} = {x : ℝ | x ≥ 3} := by sorry

-- Theorem 2: Range of m when A ∪ B = B
theorem union_equality_implies_m_range (m : ℝ) : 
  A ∪ B m = B m → m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_A_in_B_union_equality_implies_m_range_l3654_365486


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l3654_365436

theorem rationalize_denominator_sqrt5 :
  ∃ (A B C : ℤ),
    (A = -9 ∧ B = -4 ∧ C = 5) ∧
    (A * B * C = 180) ∧
    ∃ (x : ℝ),
      x = (2 + Real.sqrt 5) / (2 - Real.sqrt 5) ∧
      x = A + B * Real.sqrt C := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l3654_365436


namespace NUMINAMATH_CALUDE_distribute_5_4_l3654_365410

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_4 : distribute 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_4_l3654_365410


namespace NUMINAMATH_CALUDE_exists_78_lines_1992_intersections_l3654_365414

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ

/-- Theorem: There exists a configuration of 78 lines with exactly 1992 intersection points -/
theorem exists_78_lines_1992_intersections :
  ∃ (config : LineConfiguration), config.num_lines = 78 ∧ config.num_intersections = 1992 :=
sorry

end NUMINAMATH_CALUDE_exists_78_lines_1992_intersections_l3654_365414


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l3654_365478

theorem largest_four_digit_divisible_by_24 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 24 = 0 → n ≤ 9984 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l3654_365478


namespace NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_three_equals_two_l3654_365424

theorem sqrt_twelve_div_sqrt_three_equals_two : Real.sqrt 12 / Real.sqrt 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_three_equals_two_l3654_365424


namespace NUMINAMATH_CALUDE_dilution_proof_l3654_365405

/-- Proves that adding 6 ounces of water to 12 ounces of a 60% alcohol solution results in a 40% alcohol solution. -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) 
  (water_added : ℝ) (h1 : initial_volume = 12) (h2 : initial_concentration = 0.6) 
  (h3 : target_concentration = 0.4) (h4 : water_added = 6) : 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_proof_l3654_365405


namespace NUMINAMATH_CALUDE_min_value_of_f_l3654_365437

/-- The function f(x) = 3/x + 1/(1-3x) has a minimum value of 16 on the interval (0, 1/3) -/
theorem min_value_of_f (x : ℝ) (hx : 0 < x ∧ x < 1/3) : 3/x + 1/(1-3*x) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3654_365437


namespace NUMINAMATH_CALUDE_function_properties_l3654_365457

noncomputable def f (a b x : ℝ) : ℝ := (a * x) / (Real.exp x + 1) + b * Real.exp (-x)

theorem function_properties (a b k : ℝ) :
  (f a b 0 = 1) →
  (HasDerivAt (f a b) (-1/2) 0) →
  (∀ x ≠ 0, f a b x > x / (Real.exp x - 1) + k * Real.exp (-x)) →
  (a = 1 ∧ b = 1 ∧ k ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3654_365457


namespace NUMINAMATH_CALUDE_school_year_work_hours_l3654_365418

/-- Amy's work schedule and earnings -/
structure WorkSchedule where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_year_weeks : ℕ
  school_year_target_earnings : ℕ

/-- Calculate the required hours per week during school year -/
def required_school_year_hours_per_week (schedule : WorkSchedule) : ℚ :=
  let hourly_wage : ℚ := schedule.summer_earnings / (schedule.summer_weeks * schedule.summer_hours_per_week)
  let total_hours_needed : ℚ := schedule.school_year_target_earnings / hourly_wage
  total_hours_needed / schedule.school_year_weeks

/-- Theorem stating the required hours per week during school year -/
theorem school_year_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.summer_weeks = 8)
  (h2 : schedule.summer_hours_per_week = 40)
  (h3 : schedule.summer_earnings = 3200)
  (h4 : schedule.school_year_weeks = 32)
  (h5 : schedule.school_year_target_earnings = 4000) :
  required_school_year_hours_per_week schedule = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_school_year_work_hours_l3654_365418


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3654_365452

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_10th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a4 : a 4 = 6) :
  a 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3654_365452


namespace NUMINAMATH_CALUDE_custom_deck_probability_l3654_365474

/-- A custom deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (new_ranks : ℕ)

/-- The probability of drawing a specific type of card -/
def draw_probability (d : Deck) (favorable_cards : ℕ) : ℚ :=
  favorable_cards / d.total_cards

/-- Our specific deck configuration -/
def custom_deck : Deck :=
  { total_cards := 60
  , ranks := 15
  , suits := 4
  , cards_per_suit := 15
  , new_ranks := 2 }

theorem custom_deck_probability :
  let d := custom_deck
  let diamond_cards := d.cards_per_suit
  let new_rank_cards := d.new_ranks * d.suits
  let favorable_cards := diamond_cards + new_rank_cards - d.new_ranks
  draw_probability d favorable_cards = 7 / 20 := by
  sorry


end NUMINAMATH_CALUDE_custom_deck_probability_l3654_365474


namespace NUMINAMATH_CALUDE_sandwich_problem_solution_l3654_365411

/-- Represents the sandwich shop problem -/
def sandwich_problem (sandwich_price : ℝ) (delivery_fee : ℝ) (tip_percentage : ℝ) (total_received : ℝ) : Prop :=
  ∃ (num_sandwiches : ℝ),
    sandwich_price * num_sandwiches + delivery_fee + 
    (sandwich_price * num_sandwiches + delivery_fee) * tip_percentage = total_received ∧
    num_sandwiches = 18

/-- Theorem stating the solution to the sandwich problem -/
theorem sandwich_problem_solution :
  sandwich_problem 5 20 0.1 121 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_problem_solution_l3654_365411


namespace NUMINAMATH_CALUDE_balloon_radius_increase_l3654_365451

theorem balloon_radius_increase (C₁ C₂ r₁ r₂ Δr : ℝ) : 
  C₁ = 20 → 
  C₂ = 25 → 
  C₁ = 2 * Real.pi * r₁ → 
  C₂ = 2 * Real.pi * r₂ → 
  Δr = r₂ - r₁ → 
  Δr = 5 / (2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_balloon_radius_increase_l3654_365451


namespace NUMINAMATH_CALUDE_father_age_and_pen_cost_l3654_365421

/-- Xiao Ming's age -/
def xiao_ming_age : ℕ := 9

/-- The factor by which Xiao Ming's father's age is greater than Xiao Ming's -/
def father_age_factor : ℕ := 5

/-- The cost of one pen in yuan -/
def pen_cost : ℕ := 2

/-- The number of pens to be purchased -/
def pen_quantity : ℕ := 60

theorem father_age_and_pen_cost :
  (xiao_ming_age * father_age_factor = 45) ∧
  (pen_cost * pen_quantity = 120) := by
  sorry


end NUMINAMATH_CALUDE_father_age_and_pen_cost_l3654_365421


namespace NUMINAMATH_CALUDE_largest_initial_number_prove_largest_initial_number_l3654_365439

theorem largest_initial_number : ℕ → Prop :=
  fun n => n = 189 ∧
    ∃ (a b c d e : ℕ),
      n + a + b + c + d + e = 200 ∧
      a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
      ¬(n % a = 0) ∧ ¬(n % b = 0) ∧ ¬(n % c = 0) ∧ ¬(n % d = 0) ∧ ¬(n % e = 0) ∧
      ∀ m : ℕ, m > n →
        ¬∃ (a' b' c' d' e' : ℕ),
          m + a' + b' + c' + d' + e' = 200 ∧
          a' ≥ 2 ∧ b' ≥ 2 ∧ c' ≥ 2 ∧ d' ≥ 2 ∧ e' ≥ 2 ∧
          ¬(m % a' = 0) ∧ ¬(m % b' = 0) ∧ ¬(m % c' = 0) ∧ ¬(m % d' = 0) ∧ ¬(m % e' = 0)

theorem prove_largest_initial_number : ∃ n : ℕ, largest_initial_number n := by
  sorry

end NUMINAMATH_CALUDE_largest_initial_number_prove_largest_initial_number_l3654_365439


namespace NUMINAMATH_CALUDE_function_decomposition_l3654_365471

/-- A non-negative function defined on [-3, 3] -/
def NonNegativeFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 0}

/-- An even function defined on [-3, 3] -/
def EvenFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x = f (-x)}

/-- An odd function defined on [-3, 3] -/
def OddFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x = -f (-x)}

theorem function_decomposition
  (f : EvenFunction) (g : OddFunction)
  (h : ∀ x ∈ Set.Icc (-3 : ℝ) 3, f.val x + g.val x ≥ 2007 * x * Real.sqrt (9 - x^2) + x^2006) :
  ∃ p : NonNegativeFunction,
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f.val x = x^2006 + (p.val x + p.val (-x)) / 2) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, g.val x = 2007 * x * Real.sqrt (9 - x^2) + (p.val x - p.val (-x)) / 2) :=
sorry

end NUMINAMATH_CALUDE_function_decomposition_l3654_365471


namespace NUMINAMATH_CALUDE_positive_A_value_l3654_365450

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h1 : hash A 7 = 130) (h2 : A > 0) : A = 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l3654_365450


namespace NUMINAMATH_CALUDE_no_prime_satisfies_equation_l3654_365465

theorem no_prime_satisfies_equation : 
  ¬ ∃ (q : ℕ), Nat.Prime q ∧ 
  (1 * q^3 + 0 * q^2 + 1 * q + 2) + 
  (3 * q^2 + 0 * q + 7) + 
  (1 * q^2 + 1 * q + 4) + 
  (1 * q^2 + 2 * q + 6) + 
  7 = 
  (1 * q^2 + 4 * q + 3) + 
  (2 * q^2 + 7 * q + 2) + 
  (3 * q^2 + 6 * q + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_equation_l3654_365465


namespace NUMINAMATH_CALUDE_eco_park_cherry_sample_l3654_365404

/-- Represents the number of cherry trees in a stratified sample -/
def cherry_trees_in_sample (total_trees : ℕ) (total_cherry_trees : ℕ) (sample_size : ℕ) : ℕ :=
  (total_cherry_trees * sample_size) / total_trees

/-- Theorem stating the number of cherry trees in the sample for the given eco-park -/
theorem eco_park_cherry_sample :
  cherry_trees_in_sample 60000 4000 300 = 20 := by
  sorry

end NUMINAMATH_CALUDE_eco_park_cherry_sample_l3654_365404


namespace NUMINAMATH_CALUDE_inequality_properties_l3654_365468

theorem inequality_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (b / a ≤ (b + c^2) / (a + c^2)) ∧ (a + b < Real.sqrt (2 * (a^2 + b^2))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3654_365468


namespace NUMINAMATH_CALUDE_train_speed_l3654_365438

/-- Proves that a train of length 480 meters crossing a telegraph post in 16 seconds has a speed of 108 km/h -/
theorem train_speed (train_length : Real) (crossing_time : Real) (speed_kmh : Real) : 
  train_length = 480 ∧ 
  crossing_time = 16 ∧ 
  speed_kmh = (train_length / crossing_time) * 3.6 → 
  speed_kmh = 108 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l3654_365438


namespace NUMINAMATH_CALUDE_fraction_simplification_l3654_365417

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3654_365417


namespace NUMINAMATH_CALUDE_infinitely_many_squares_l3654_365403

theorem infinitely_many_squares (k : ℕ+) :
  ∀ (B : ℕ), ∃ (n m : ℕ), n > B ∧ m > B ∧ (2 * k.val * n - 7 = m^2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_l3654_365403


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l3654_365415

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) 
  (h1 : selling_price = 720)
  (h2 : num_balls_sold = 17)
  (h3 : num_balls_loss = 5) : 
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧ 
    cost_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l3654_365415


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3654_365485

theorem inequality_equivalence (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ -1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3654_365485


namespace NUMINAMATH_CALUDE_circle_square_area_difference_l3654_365427

theorem circle_square_area_difference :
  let square_side : ℝ := 12
  let circle_diameter : ℝ := 16
  let π : ℝ := 3
  let square_area := square_side ^ 2
  let circle_area := π * (circle_diameter / 2) ^ 2
  circle_area - square_area = 48 := by sorry

end NUMINAMATH_CALUDE_circle_square_area_difference_l3654_365427


namespace NUMINAMATH_CALUDE_jinas_mascots_l3654_365449

/-- The number of mascots Jina has -/
def total_mascots (original_teddies bunny_to_teddy_ratio koala_bears additional_teddies_per_bunny : ℕ) : ℕ :=
  let bunnies := original_teddies * bunny_to_teddy_ratio
  let additional_teddies := bunnies * additional_teddies_per_bunny
  original_teddies + bunnies + koala_bears + additional_teddies

/-- Theorem stating the total number of mascots Jina has -/
theorem jinas_mascots : total_mascots 5 3 1 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_jinas_mascots_l3654_365449


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3654_365462

def OA : ℝ × ℝ := (2, 2)
def OB : ℝ × ℝ := (5, 3)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_difference_magnitude : 
  Real.sqrt ((2 * OA.1 - OB.1)^2 + (2 * OA.2 - OB.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3654_365462


namespace NUMINAMATH_CALUDE_ellipse_foci_l3654_365443

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop := y^2 / 3 + x^2 / 2 = 1

/-- The coordinates of a point -/
def Point := ℝ × ℝ

/-- The foci of an ellipse -/
def are_foci (p1 p2 : Point) : Prop :=
  p1 = (0, -1) ∧ p2 = (0, 1)

/-- Theorem: The foci of the given ellipse are (0, -1) and (0, 1) -/
theorem ellipse_foci :
  ∃ (p1 p2 : Point), (∀ x y : ℝ, is_ellipse x y → are_foci p1 p2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l3654_365443


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3654_365419

theorem max_value_of_expression (a b : ℝ) 
  (h : 17 * (a^2 + b^2) - 30 * a * b - 16 = 0) : 
  ∃ (x : ℝ), x = Real.sqrt (16 * a^2 + 4 * b^2 - 16 * a * b - 12 * a + 6 * b + 9) ∧ 
  x ≤ 7 ∧ 
  ∃ (a₀ b₀ : ℝ), 17 * (a₀^2 + b₀^2) - 30 * a₀ * b₀ - 16 = 0 ∧ 
    Real.sqrt (16 * a₀^2 + 4 * b₀^2 - 16 * a₀ * b₀ - 12 * a₀ + 6 * b₀ + 9) = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3654_365419


namespace NUMINAMATH_CALUDE_john_mean_score_l3654_365408

def john_scores : List ℝ := [100, 95, 90, 84, 92, 75]

theorem john_mean_score :
  (john_scores.sum / john_scores.length : ℝ) = 89.333 := by
  sorry

end NUMINAMATH_CALUDE_john_mean_score_l3654_365408


namespace NUMINAMATH_CALUDE_range_of_a_l3654_365423

theorem range_of_a : ∀ a : ℝ, 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) →
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3654_365423


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3654_365469

theorem roots_of_polynomial : ∀ x : ℝ,
  (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) →
  8 * x^5 - 45 * x^4 + 84 * x^3 - 84 * x^2 + 45 * x - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3654_365469


namespace NUMINAMATH_CALUDE_sphere_radius_when_volume_equals_surface_area_l3654_365475

theorem sphere_radius_when_volume_equals_surface_area :
  ∀ r : ℝ,
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_when_volume_equals_surface_area_l3654_365475


namespace NUMINAMATH_CALUDE_prob_three_different_suits_value_l3654_365428

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := standard_deck_size / number_of_suits

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
def prob_three_different_suits : ℚ :=
  (cards_per_suit * (number_of_suits - 1) : ℚ) / (standard_deck_size - 1) *
  (cards_per_suit * (number_of_suits - 2) : ℚ) / (standard_deck_size - 2)

theorem prob_three_different_suits_value : 
  prob_three_different_suits = 169 / 425 := by sorry

end NUMINAMATH_CALUDE_prob_three_different_suits_value_l3654_365428


namespace NUMINAMATH_CALUDE_distance_to_origin_l3654_365488

/-- The distance from the point corresponding to the complex number 2i/(1-i) to the origin in the complex plane is √2. -/
theorem distance_to_origin : Complex.abs (2 * Complex.I / (1 - Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3654_365488


namespace NUMINAMATH_CALUDE_divisible_by_eight_expression_l3654_365479

theorem divisible_by_eight_expression :
  ∃ (A B C : ℕ), (A % 8 ≠ 0) ∧ (B % 8 ≠ 0) ∧ (C % 8 ≠ 0) ∧
    (∀ n : ℕ, (A * 5^n + B * 3^(n-1) + C) % 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_eight_expression_l3654_365479


namespace NUMINAMATH_CALUDE_five_digit_divisible_count_l3654_365456

theorem five_digit_divisible_count : 
  let lcm := Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))
  let lower_bound := ((10000 + lcm - 1) / lcm) * lcm
  let upper_bound := (99999 / lcm) * lcm
  (upper_bound - lower_bound) / lcm + 1 = 179 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_count_l3654_365456


namespace NUMINAMATH_CALUDE_line_equation_l3654_365431

/-- A line passing through (1,1) and intersecting the circle (x-2)^2 + (y-3)^2 = 9 at two points A and B -/
def Line : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (k : ℝ), p.2 = k * (p.1 - 1) + 1}

/-- The circle (x-2)^2 + (y-3)^2 = 9 -/
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 9}

/-- The line passes through (1,1) -/
axiom line_passes_through : (1, 1) ∈ Line

/-- The line intersects the circle at two points A and B -/
axiom line_intersects_circle : ∃ (A B : ℝ × ℝ), A ∈ Line ∩ Circle ∧ B ∈ Line ∩ Circle ∧ A ≠ B

/-- The distance between A and B is 4 -/
axiom distance_AB : ∀ (A B : ℝ × ℝ), A ∈ Line ∩ Circle → B ∈ Line ∩ Circle → A ≠ B →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16

/-- The equation of the line is x + 2y - 3 = 0 -/
theorem line_equation : Line = {p : ℝ × ℝ | p.1 + 2 * p.2 - 3 = 0} :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3654_365431


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3654_365444

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 6)^2 + (z - 8)^2 = 0 → 2*x + 2*y + 2*z = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3654_365444


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_parallel_planes_l3654_365446

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : contained_in b β)
  (h3 : parallel α β) :
  line_perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_parallel_planes_l3654_365446


namespace NUMINAMATH_CALUDE_smallest_marble_set_marble_set_existence_l3654_365442

theorem smallest_marble_set (n : ℕ) : n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 210 := by
  sorry

theorem marble_set_existence : ∃ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 210 := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_set_marble_set_existence_l3654_365442


namespace NUMINAMATH_CALUDE_short_trees_planted_l3654_365455

/-- The number of short trees planted in a park. -/
theorem short_trees_planted (current : ℕ) (final : ℕ) (planted : ℕ) : 
  current = 3 → final = 12 → planted = final - current → planted = 9 := by
sorry

end NUMINAMATH_CALUDE_short_trees_planted_l3654_365455


namespace NUMINAMATH_CALUDE_percentage_problem_l3654_365480

theorem percentage_problem (total : ℝ) (part : ℝ) (h1 : total = 300) (h2 : part = 75) :
  (part / total) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3654_365480


namespace NUMINAMATH_CALUDE_arithmetic_expression_proof_l3654_365472

theorem arithmetic_expression_proof : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_proof_l3654_365472


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3654_365435

/-- 
Given two points A(m,3) and B(4,n) that are symmetric about the y-axis,
prove that (m+n)^2015 = -1
-/
theorem symmetric_points_sum_power (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (m, 3) ∧ B = (4, n) ∧ 
   A.1 = -B.1 ∧ A.2 = B.2) → 
  (m + n)^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3654_365435


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3654_365481

theorem trigonometric_identity : 
  Real.cos (6 * π / 180) * Real.cos (36 * π / 180) + 
  Real.sin (6 * π / 180) * Real.cos (54 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3654_365481


namespace NUMINAMATH_CALUDE_f_is_linear_l3654_365430

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation -x - 3 = 4 -/
def f (x : ℝ) : ℝ := -x - 3

/-- Theorem stating that f is a linear equation -/
theorem f_is_linear : is_linear_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_linear_l3654_365430


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3654_365401

theorem largest_prime_divisor_of_factorial_sum : 
  ∃ p : ℕ, Prime p ∧ p ∣ (Nat.factorial 12 + Nat.factorial 13) ∧ 
  ∀ q : ℕ, Prime q → q ∣ (Nat.factorial 12 + Nat.factorial 13) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3654_365401


namespace NUMINAMATH_CALUDE_ad_cost_per_square_inch_l3654_365483

/-- Proves that the cost per square inch for advertising is $8 --/
theorem ad_cost_per_square_inch :
  let page_length : ℝ := 9
  let page_width : ℝ := 12
  let full_page_area : ℝ := page_length * page_width
  let ad_area : ℝ := full_page_area / 2
  let total_cost : ℝ := 432
  let cost_per_square_inch : ℝ := total_cost / ad_area
  cost_per_square_inch = 8 := by
  sorry

end NUMINAMATH_CALUDE_ad_cost_per_square_inch_l3654_365483


namespace NUMINAMATH_CALUDE_files_remaining_l3654_365491

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 16)
  (h2 : video_files = 48)
  (h3 : deleted_files = 30) :
  music_files + video_files - deleted_files = 34 :=
by sorry

end NUMINAMATH_CALUDE_files_remaining_l3654_365491


namespace NUMINAMATH_CALUDE_wire_length_difference_l3654_365406

theorem wire_length_difference (total_length first_part : ℕ) 
  (h1 : total_length = 180)
  (h2 : first_part = 106) :
  first_part - (total_length - first_part) = 32 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_difference_l3654_365406


namespace NUMINAMATH_CALUDE_karls_trip_distance_l3654_365484

-- Define the problem parameters
def miles_per_gallon : ℚ := 30
def tank_capacity : ℚ := 16
def initial_distance : ℚ := 420
def gas_bought : ℚ := 10
def final_tank_fraction : ℚ := 3/4

-- Theorem statement
theorem karls_trip_distance :
  let initial_gas_used : ℚ := initial_distance / miles_per_gallon
  let remaining_gas : ℚ := tank_capacity - initial_gas_used
  let gas_after_refill : ℚ := remaining_gas + gas_bought
  let final_gas : ℚ := tank_capacity * final_tank_fraction
  gas_after_refill = final_gas →
  initial_distance = 420 := by
sorry

end NUMINAMATH_CALUDE_karls_trip_distance_l3654_365484


namespace NUMINAMATH_CALUDE_negation_equivalence_l3654_365402

theorem negation_equivalence (a b : ℝ) : 
  ¬(a * b = 0 → a = 0 ∨ b = 0) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3654_365402


namespace NUMINAMATH_CALUDE_cube_space_diagonal_length_l3654_365420

/-- The length of a space diagonal in a cube with side length 15 -/
theorem cube_space_diagonal_length :
  ∀ (s : ℝ), s = 15 →
  ∃ (d : ℝ), d = s * Real.sqrt 3 ∧ d^2 = 3 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_space_diagonal_length_l3654_365420


namespace NUMINAMATH_CALUDE_three_digit_number_property_l3654_365453

theorem three_digit_number_property : 
  ∃ (N : ℕ), 
    (100 ≤ N ∧ N < 1000) ∧ 
    (N % 11 = 0) ∧ 
    (N / 11 = (N / 100)^2 + ((N / 10) % 10)^2 + (N % 10)^2) ∧
    (N = 550 ∨ N = 803) ∧
    (∀ (M : ℕ), 
      (100 ≤ M ∧ M < 1000) ∧ 
      (M % 11 = 0) ∧ 
      (M / 11 = (M / 100)^2 + ((M / 10) % 10)^2 + (M % 10)^2) →
      (M = 550 ∨ M = 803)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l3654_365453


namespace NUMINAMATH_CALUDE_power_multiplication_l3654_365489

theorem power_multiplication (t : ℝ) : t^3 * t^4 = t^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3654_365489


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l3654_365422

/-- Represents the retirement eligibility rule for a company -/
structure RetirementRule :=
  (hire_year : ℕ)
  (hire_age : ℕ)
  (retirement_year : ℕ)

/-- Calculates the required total of age and years of employment for retirement -/
def retirement_total (rule : RetirementRule) : ℕ :=
  (rule.retirement_year - rule.hire_year) + rule.hire_age + (rule.retirement_year - rule.hire_year)

/-- Theorem stating the required total for retirement is 70 -/
theorem retirement_total_is_70 (rule : RetirementRule) 
  (h1 : rule.hire_year = 1986)
  (h2 : rule.hire_age = 30)
  (h3 : rule.retirement_year = 2006) :
  retirement_total rule = 70 := by
  sorry

end NUMINAMATH_CALUDE_retirement_total_is_70_l3654_365422


namespace NUMINAMATH_CALUDE_basketball_team_score_lower_bound_l3654_365464

theorem basketball_team_score_lower_bound (n : ℕ) (player_scores : Fin n → ℕ) 
  (h1 : n = 12) 
  (h2 : ∀ i, player_scores i ≥ 7) 
  (h3 : ∀ i, player_scores i ≤ 23) : 
  (Finset.sum Finset.univ player_scores) ≥ 84 := by
  sorry

#check basketball_team_score_lower_bound

end NUMINAMATH_CALUDE_basketball_team_score_lower_bound_l3654_365464


namespace NUMINAMATH_CALUDE_stream_speed_l3654_365400

/-- Proves that the speed of the stream is 5 km/hr given the conditions of the problem -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 125 →
  downstream_time = 5 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l3654_365400


namespace NUMINAMATH_CALUDE_factors_of_2520_l3654_365482

/-- The number of distinct, positive factors of 2520 -/
def num_factors_2520 : ℕ :=
  (Finset.filter (· ∣ 2520) (Finset.range 2521)).card

/-- Theorem stating that the number of distinct, positive factors of 2520 is 48 -/
theorem factors_of_2520 : num_factors_2520 = 48 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2520_l3654_365482


namespace NUMINAMATH_CALUDE_system_solution_l3654_365477

theorem system_solution :
  let solutions : List (ℝ × ℝ × ℝ) := [
    (1, 2, 3), (1, 5, -3), (3, -2, 5),
    (3, 3, -5), (6, -5, 2), (6, -3, -2)
  ]
  ∀ (x y z : ℝ),
    (3*x + 2*y + z = 10 ∧
     3*x^2 + 4*x*y + 2*x*z + y^2 + y*z = 27 ∧
     x^3 + 2*x^2*y + x^2*z + x*y^2 + x*y*z = 18) ↔
    (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3654_365477


namespace NUMINAMATH_CALUDE_total_weight_moved_tom_total_weight_l3654_365433

/-- Calculate the total weight Tom is moving with. -/
theorem total_weight_moved (tom_weight : ℝ) (hand_weight_ratio : ℝ) (vest_weight_ratio : ℝ) : ℝ :=
  let vest_weight := vest_weight_ratio * tom_weight
  let hand_weight := hand_weight_ratio * tom_weight
  let total_hand_weight := 2 * hand_weight
  total_hand_weight + vest_weight

/-- Prove that Tom is moving a total weight of 525 kg. -/
theorem tom_total_weight :
  total_weight_moved 150 1.5 0.5 = 525 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_moved_tom_total_weight_l3654_365433


namespace NUMINAMATH_CALUDE_johnson_family_seating_l3654_365409

def num_sons : ℕ := 5
def num_daughters : ℕ := 4
def total_children : ℕ := num_sons + num_daughters

def total_arrangements : ℕ := Nat.factorial total_children

def arrangements_without_bbg : ℕ := Nat.factorial 7 * 4

theorem johnson_family_seating :
  total_arrangements - arrangements_without_bbg = 342720 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l3654_365409


namespace NUMINAMATH_CALUDE_max_collisions_l3654_365445

/-- Represents an ant with a position and velocity -/
structure Ant where
  position : ℝ
  velocity : ℝ

/-- The configuration of n ants on a line -/
def AntConfiguration (n : ℕ) := Fin n → Ant

/-- Predicate to check if the total number of collisions is finite -/
def HasFiniteCollisions (config : AntConfiguration n) : Prop := sorry

/-- The number of collisions that occur in a given configuration -/
def NumberOfCollisions (config : AntConfiguration n) : ℕ := sorry

/-- Theorem stating the maximum number of collisions for n ants -/
theorem max_collisions (n : ℕ) (h : n > 0) :
  ∃ (config : AntConfiguration n),
    HasFiniteCollisions config ∧
    NumberOfCollisions config = n * (n - 1) / 2 ∧
    ∀ (other_config : AntConfiguration n),
      HasFiniteCollisions other_config →
      NumberOfCollisions other_config ≤ n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_collisions_l3654_365445


namespace NUMINAMATH_CALUDE_simplify_expression_l3654_365448

theorem simplify_expression (s : ℝ) : 105 * s - 63 * s = 42 * s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3654_365448


namespace NUMINAMATH_CALUDE_item_cost_calculation_l3654_365425

theorem item_cost_calculation (total_items : ℕ) (total_cost : ℕ) : 
  total_items = 15 → total_cost = 30 → (total_cost / total_items : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_item_cost_calculation_l3654_365425


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l3654_365460

theorem smallest_factorizable_b : ∃ (b : ℕ),
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2016 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b →
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 2016 = (x + p) * (x + q))) ∧
  b = 90 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l3654_365460


namespace NUMINAMATH_CALUDE_point_same_side_condition_l3654_365490

/-- A point on a line is on the same side as the origin with respect to another line -/
def same_side_as_origin (k b : ℝ) : Prop :=
  ∀ x : ℝ, (x - (k * x + b) + 2) * 2 > 0

/-- Theorem: If a point on y = kx + b is on the same side as the origin
    with respect to x - y + 2 = 0, then k = 1 and b < 2 -/
theorem point_same_side_condition (k b : ℝ) :
  same_side_as_origin k b → k = 1 ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_same_side_condition_l3654_365490


namespace NUMINAMATH_CALUDE_best_discount_l3654_365447

def original_price : ℝ := 100

def discount_a (price : ℝ) : ℝ := price * 0.8

def discount_b (price : ℝ) : ℝ := price * 0.9 * 0.9

def discount_c (price : ℝ) : ℝ := price * 0.85 * 0.95

def discount_d (price : ℝ) : ℝ := price * 0.95 * 0.85

theorem best_discount :
  discount_a original_price < discount_b original_price ∧
  discount_a original_price < discount_c original_price ∧
  discount_a original_price < discount_d original_price :=
sorry

end NUMINAMATH_CALUDE_best_discount_l3654_365447


namespace NUMINAMATH_CALUDE_mass_equivalence_l3654_365412

-- Define symbols as real numbers representing their masses
variable (circle square triangle zero : ℝ)

-- Define the balanced scales conditions
axiom scale1 : 3 * circle = 2 * triangle
axiom scale2 : square + circle + triangle = 2 * square

-- Define the mass of the left side of the equation to prove
def left_side : ℝ := circle + 3 * triangle

-- Define the mass of the right side of the equation to prove
def right_side : ℝ := 3 * zero + square

-- Theorem to prove
theorem mass_equivalence : left_side = right_side :=
sorry

end NUMINAMATH_CALUDE_mass_equivalence_l3654_365412


namespace NUMINAMATH_CALUDE_parallelepipeds_from_four_points_l3654_365476

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A predicate that checks if four points are coplanar -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    a * p1.x + b * p1.y + c * p1.z + d = 0 ∧
    a * p2.x + b * p2.y + c * p2.z + d = 0 ∧
    a * p3.x + b * p3.y + c * p3.z + d = 0 ∧
    a * p4.x + b * p4.y + c * p4.z + d = 0

/-- A function that counts the number of distinct parallelepipeds -/
def count_parallelepipeds (p1 p2 p3 p4 : Point3D) : ℕ :=
  sorry -- The actual implementation is not needed for the theorem statement

/-- Theorem stating that 4 non-coplanar points form 29 distinct parallelepipeds -/
theorem parallelepipeds_from_four_points (p1 p2 p3 p4 : Point3D) 
  (h : ¬coplanar p1 p2 p3 p4) : 
  count_parallelepipeds p1 p2 p3 p4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_parallelepipeds_from_four_points_l3654_365476


namespace NUMINAMATH_CALUDE_decrement_value_theorem_l3654_365441

theorem decrement_value_theorem (n : ℕ) (original_mean new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : new_mean = 185) :
  let decrement := (n * original_mean - n * new_mean) / n
  decrement = 15 := by
sorry

end NUMINAMATH_CALUDE_decrement_value_theorem_l3654_365441


namespace NUMINAMATH_CALUDE_no_solution_iff_p_equals_seven_l3654_365463

theorem no_solution_iff_p_equals_seven :
  ∀ p : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - p) / (x - 8)) ↔ p = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_p_equals_seven_l3654_365463


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3654_365487

theorem min_value_of_expression (a b : ℕ) (ha : 0 < a ∧ a < 9) (hb : 0 < b ∧ b < 9) :
  ∃ (m : ℤ), m = -5 ∧ ∀ (x y : ℕ), (0 < x ∧ x < 9) → (0 < y ∧ y < 9) → m ≤ (3 * x^2 - x * y : ℤ) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3654_365487


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l3654_365495

/-- Represents a 5x6 grid of integers -/
def Grid := Matrix (Fin 5) (Fin 6) ℕ

/-- Checks if a row in the grid has no repeating numbers -/
def rowNoRepeats (g : Grid) (row : Fin 5) : Prop :=
  ∀ i j : Fin 6, i ≠ j → g row i ≠ g row j

/-- Checks if a column in the grid has no repeating numbers -/
def colNoRepeats (g : Grid) (col : Fin 6) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i col ≠ g j col

/-- Checks if all numbers in the grid are between 1 and 6 -/
def validNumbers (g : Grid) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 6, 1 ≤ g i j ∧ g i j ≤ 6

/-- Checks if the sums of specific digits match the given constraints -/
def validSums (g : Grid) : Prop :=
  g 0 0 * 100 + g 0 1 * 10 + g 0 2 = 669 ∧
  g 0 3 * 10 + g 0 4 = 44

/-- The main theorem stating that 41244 satisfies all conditions -/
theorem solution_satisfies_conditions : ∃ (g : Grid),
  (∀ row : Fin 5, rowNoRepeats g row) ∧
  (∀ col : Fin 6, colNoRepeats g col) ∧
  validNumbers g ∧
  validSums g ∧
  g 0 0 = 4 ∧ g 0 1 = 1 ∧ g 0 2 = 2 ∧ g 0 3 = 4 ∧ g 0 4 = 4 :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_conditions_l3654_365495


namespace NUMINAMATH_CALUDE_heart_ratio_l3654_365494

def heart (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_ratio : (heart 2 4) / (heart 4 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_l3654_365494


namespace NUMINAMATH_CALUDE_souvenir_profit_maximization_l3654_365434

/-- Represents the problem of maximizing profit for a souvenir seller --/
theorem souvenir_profit_maximization
  (cost_price : ℕ)
  (initial_price : ℕ)
  (initial_sales : ℕ)
  (price_increase : ℕ → ℕ)
  (sales_decrease : ℕ → ℕ)
  (profit : ℕ → ℕ)
  (h_cost : cost_price = 5)
  (h_initial_price : initial_price = 9)
  (h_initial_sales : initial_sales = 32)
  (h_price_increase : ∀ x, price_increase x = x)
  (h_sales_decrease : ∀ x, sales_decrease x = 4 * x)
  (h_profit : ∀ x, profit x = (initial_price + price_increase x - cost_price) * (initial_sales - sales_decrease x)) :
  ∃ (optimal_increase : ℕ),
    optimal_increase = 2 ∧
    ∀ x, x ≠ optimal_increase → profit x ≤ profit optimal_increase ∧
    profit optimal_increase = 144 := by
  sorry


end NUMINAMATH_CALUDE_souvenir_profit_maximization_l3654_365434


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l3654_365416

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define basis vectors
variable (e₁ e₂ : V)

-- Define points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- State the theorem
theorem collinear_points_k_value
  (h_basis : LinearIndependent ℝ ![e₁, e₂])
  (h_AB : B - A = e₁ - k • e₂)
  (h_CB : B - C = 2 • e₁ - e₂)
  (h_CD : D - C = 3 • e₁ - 3 • e₂)
  (h_collinear : ∃ (t : ℝ), D - A = t • (B - A)) :
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l3654_365416
