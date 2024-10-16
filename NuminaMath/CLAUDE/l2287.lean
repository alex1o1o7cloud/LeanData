import Mathlib

namespace NUMINAMATH_CALUDE_president_and_committee_choices_l2287_228702

/-- The number of ways to choose a president and committee from a group --/
def choose_president_and_committee (total_group : ℕ) (senior_members : ℕ) (committee_size : ℕ) : ℕ :=
  let non_senior_members := total_group - senior_members
  let president_choices := non_senior_members
  let remaining_for_committee := total_group - 1
  president_choices * (Nat.choose remaining_for_committee committee_size)

/-- Theorem stating the number of ways to choose a president and committee --/
theorem president_and_committee_choices :
  choose_president_and_committee 10 4 3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_choices_l2287_228702


namespace NUMINAMATH_CALUDE_power_ratio_equals_nine_l2287_228722

theorem power_ratio_equals_nine (a b : ℝ) 
  (h1 : 3^(a-2) + a = 1/2) 
  (h2 : (1/3) * b^3 + Real.log b / Real.log 3 = -1/2) : 
  3^a / b^3 = 9 := by
sorry

end NUMINAMATH_CALUDE_power_ratio_equals_nine_l2287_228722


namespace NUMINAMATH_CALUDE_otimes_nested_equality_l2287_228797

/-- The custom operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 + 3 - y

/-- Theorem stating that k ⊗ (k ⊗ (k ⊗ k)) = k^3 + 3 - k -/
theorem otimes_nested_equality (k : ℝ) : otimes k (otimes k (otimes k k)) = k^3 + 3 - k := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_equality_l2287_228797


namespace NUMINAMATH_CALUDE_sqrt_x_plus_3_meaningful_l2287_228746

theorem sqrt_x_plus_3_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_3_meaningful_l2287_228746


namespace NUMINAMATH_CALUDE_mycoplasma_pneumonia_relation_l2287_228779

-- Define the contingency table
def a : ℕ := 40  -- infected with mycoplasma pneumonia and with chronic disease
def b : ℕ := 20  -- infected with mycoplasma pneumonia and without chronic disease
def c : ℕ := 60  -- not infected with mycoplasma pneumonia and with chronic disease
def d : ℕ := 80  -- not infected with mycoplasma pneumonia and without chronic disease
def n : ℕ := a + b + c + d

-- Define the K^2 statistic
def K_squared : ℚ := (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.5% confidence level
def critical_value : ℚ := 7.879

-- Define the number of cases with exactly one person having chronic disease
def favorable_cases : ℕ := 8
def total_cases : ℕ := 15

theorem mycoplasma_pneumonia_relation :
  K_squared > critical_value ∧ (favorable_cases : ℚ) / total_cases = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mycoplasma_pneumonia_relation_l2287_228779


namespace NUMINAMATH_CALUDE_salary_change_result_l2287_228737

def initial_salary : ℝ := 2500

def raise_percentage : ℝ := 0.10

def cut_percentage : ℝ := 0.25

def final_salary : ℝ := initial_salary * (1 + raise_percentage) * (1 - cut_percentage)

theorem salary_change_result :
  final_salary = 2062.5 := by sorry

end NUMINAMATH_CALUDE_salary_change_result_l2287_228737


namespace NUMINAMATH_CALUDE_marbles_in_larger_container_l2287_228741

/-- Given a container of volume v1 that can hold m1 marbles,
    calculate the number of marbles (m2) that can be held by a container of volume v2,
    assuming a linear relationship between volume and marble capacity. -/
def marbles_in_container (v1 v2 m1 : ℚ) : ℚ :=
  (v2 * m1) / v1

/-- Theorem stating that a 72 cm³ container will hold 90 marbles
    given that a 24 cm³ container holds 30 marbles. -/
theorem marbles_in_larger_container :
  marbles_in_container 24 72 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_marbles_in_larger_container_l2287_228741


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2287_228794

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (4 * x - 8 ≤ 0) ∧ ((x + 3) / 2 > 3 - x)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 < x ∧ x ≤ 2

-- Theorem statement
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2287_228794


namespace NUMINAMATH_CALUDE_f_750_value_l2287_228783

/-- A function satisfying f(xy) = f(x)/y for positive reals -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 ∧ y > 0 → f (x * y) = f x / y

theorem f_750_value (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 1000 = 4) :
  f 750 = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_f_750_value_l2287_228783


namespace NUMINAMATH_CALUDE_simplify_complex_square_l2287_228725

theorem simplify_complex_square : 
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 25 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_square_l2287_228725


namespace NUMINAMATH_CALUDE_no_number_divisible_by_1998_with_small_digit_sum_l2287_228736

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem no_number_divisible_by_1998_with_small_digit_sum :
  ∀ n : ℕ, n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end NUMINAMATH_CALUDE_no_number_divisible_by_1998_with_small_digit_sum_l2287_228736


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l2287_228700

/-- Given a function f(x) = ax - a/x - 2ln(x) where a ≥ 0, if f(x) is monotonically increasing
    on its domain (0, +∞), then a > 1. -/
theorem monotonic_increasing_condition (a : ℝ) (h_a : a ≥ 0) :
  (∀ x : ℝ, x > 0 → Monotone (fun x => a * x - a / x - 2 * Real.log x)) →
  a > 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l2287_228700


namespace NUMINAMATH_CALUDE_max_metro_speed_l2287_228780

/-- Represents the metro system and the students' travel scenario -/
structure MetroSystem where
  v : ℕ  -- Speed of metro trains in km/h
  S : ℝ  -- Distance between two nearest metro stations
  R : ℝ  -- Distance from home to nearest station

/-- Conditions for the metro system -/
def validMetroSystem (m : MetroSystem) : Prop :=
  m.S > 0 ∧ m.R > 0 ∧ m.R < m.S / 2

/-- Yegor's travel condition -/
def yegorCondition (m : MetroSystem) : Prop :=
  m.S / 24 > m.R / m.v

/-- Nikita's travel condition -/
def nikitaCondition (m : MetroSystem) : Prop :=
  m.S / 12 < (m.R + m.S) / m.v

/-- The maximum speed theorem -/
theorem max_metro_speed :
  ∃ (m : MetroSystem),
    validMetroSystem m ∧
    yegorCondition m ∧
    nikitaCondition m ∧
    (∀ (m' : MetroSystem),
      validMetroSystem m' ∧ yegorCondition m' ∧ nikitaCondition m' →
      m'.v ≤ m.v) ∧
    m.v = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_metro_speed_l2287_228780


namespace NUMINAMATH_CALUDE_smallest_n_value_l2287_228757

theorem smallest_n_value (r g b : ℕ+) (h : 10 * r = 18 * g ∧ 18 * g = 20 * b) :
  ∃ (n : ℕ+), 30 * n = 10 * r ∧ ∀ (m : ℕ+), 30 * m = 10 * r → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2287_228757


namespace NUMINAMATH_CALUDE_min_value_of_a_l2287_228781

theorem min_value_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ (1 + a * x) / (x * 2^x) ≥ 1) → 
  a ≥ 7/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2287_228781


namespace NUMINAMATH_CALUDE_equation_solution_l2287_228744

theorem equation_solution : ∃! x : ℚ, (x - 30) / 3 = (3 * x + 4) / 8 ∧ x = -252 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2287_228744


namespace NUMINAMATH_CALUDE_gbp_share_change_l2287_228789

/-- The change in the share of British pounds in the National Wealth Fund -/
theorem gbp_share_change (
  total : ℝ)
  (initial_share : ℝ)
  (other_amounts : List ℝ)
  (h_total : total = 794.26)
  (h_initial : initial_share = 8.2)
  (h_other : other_amounts = [39.84, 34.72, 600.3, 110.54, 0.31]) :
  ∃ (δ : ℝ), abs (δ + 7) < 0.5 ∧ 
  δ = (total - (other_amounts.sum)) / total * 100 - initial_share :=
sorry

end NUMINAMATH_CALUDE_gbp_share_change_l2287_228789


namespace NUMINAMATH_CALUDE_family_income_problem_l2287_228747

/-- The number of initial earning members in a family -/
def initial_members : ℕ := 4

/-- The initial average monthly income -/
def initial_average : ℚ := 735

/-- The new average monthly income after one member's death -/
def new_average : ℚ := 650

/-- The income of the deceased member -/
def deceased_income : ℚ := 990

theorem family_income_problem :
  initial_members * initial_average - (initial_members - 1) * new_average = deceased_income :=
by sorry

end NUMINAMATH_CALUDE_family_income_problem_l2287_228747


namespace NUMINAMATH_CALUDE_product_of_roots_l2287_228792

theorem product_of_roots (x : ℝ) : 
  (x^2 + 2*x - 35 = 0) → 
  ∃ y : ℝ, (y^2 + 2*y - 35 = 0) ∧ (x * y = -35) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2287_228792


namespace NUMINAMATH_CALUDE_solution_equation_l2287_228796

theorem solution_equation : ∃ x : ℝ, 0.4 * x + (0.3 * 0.2) = 0.26 ∧ x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_solution_equation_l2287_228796


namespace NUMINAMATH_CALUDE_value_of_a_l2287_228787

theorem value_of_a (M : Set ℝ) (a : ℝ) : 
  M = {0, 1, a + 1} → -1 ∈ M → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2287_228787


namespace NUMINAMATH_CALUDE_no_common_elements_except_one_l2287_228704

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) + 2 * sequence_a n

def sequence_b : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * sequence_b (n + 1) + 3 * sequence_b n

theorem no_common_elements_except_one :
  ∀ n : ℕ, n > 0 → sequence_a n ≠ sequence_b n :=
by sorry

end NUMINAMATH_CALUDE_no_common_elements_except_one_l2287_228704


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2287_228731

/-- The probability of a coin landing heads. -/
def p_heads : ℚ := 3/5

/-- The probability of a coin landing tails. -/
def p_tails : ℚ := 1 - p_heads

/-- The number of times the coin is flipped. -/
def num_flips : ℕ := 8

/-- The number of initial flips that should be heads. -/
def num_heads : ℕ := 3

/-- The number of final flips that should be tails. -/
def num_tails : ℕ := num_flips - num_heads

/-- The probability of getting heads on the first 3 flips and tails on the last 5 flips. -/
def prob_specific_sequence : ℚ := p_heads^num_heads * p_tails^num_tails

theorem coin_flip_probability : prob_specific_sequence = 864/390625 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2287_228731


namespace NUMINAMATH_CALUDE_neg_sufficient_but_not_necessary_l2287_228770

-- Define the propositions p and q
variable (p q : Prop)

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_but_not_necessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬(q → p)

-- State the theorem
theorem neg_sufficient_but_not_necessary 
  (h : sufficient_but_not_necessary p q) : 
  sufficient_but_not_necessary (¬q) (¬p) := by
  sorry


end NUMINAMATH_CALUDE_neg_sufficient_but_not_necessary_l2287_228770


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2287_228703

theorem quadratic_roots_theorem (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (p + 3 * Complex.I) ^ 2 - (16 + 9 * Complex.I) * (p + 3 * Complex.I) + (40 + 57 * Complex.I) = 0 →
  (q + 6 * Complex.I) ^ 2 - (16 + 9 * Complex.I) * (q + 6 * Complex.I) + (40 + 57 * Complex.I) = 0 →
  p = 9.5 ∧ q = 6.5 := by
sorry


end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2287_228703


namespace NUMINAMATH_CALUDE_expression_order_l2287_228714

theorem expression_order (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧
  Real.sqrt (a * b) < (a + b) / 2 ∧
  (a + b) / 2 < Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_order_l2287_228714


namespace NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l2287_228763

/-- Given a square piece of paper with side length 4 inches that is folded in half vertically
    and then cut in half parallel to the fold, the ratio of the perimeter of one of the resulting
    small rectangles to the perimeter of the large rectangle is 5/6. -/
theorem paper_folding_perimeter_ratio :
  let initial_side_length : ℝ := 4
  let small_rectangle_length : ℝ := initial_side_length
  let small_rectangle_width : ℝ := initial_side_length / 4
  let large_rectangle_length : ℝ := initial_side_length
  let large_rectangle_width : ℝ := initial_side_length / 2
  let small_perimeter : ℝ := 2 * (small_rectangle_length + small_rectangle_width)
  let large_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  small_perimeter / large_perimeter = 5 / 6 := by
sorry


end NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l2287_228763


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2287_228769

theorem sqrt_equation_solution :
  ∃ t : ℝ, t = 37/10 ∧ Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2287_228769


namespace NUMINAMATH_CALUDE_second_meeting_time_is_four_minutes_l2287_228799

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming pool scenario --/
structure PoolScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting --/
def secondMeetingTime (scenario : PoolScenario) : ℝ :=
  sorry

/-- Theorem stating that the second meeting occurs 4 minutes after starting --/
theorem second_meeting_time_is_four_minutes (scenario : PoolScenario) 
    (h1 : scenario.poolLength = 50)
    (h2 : scenario.swimmer1.startPosition = 0)
    (h3 : scenario.swimmer2.startPosition = 50)
    (h4 : scenario.firstMeetingTime = 2)
    (h5 : scenario.firstMeetingPosition = 20) :
    secondMeetingTime scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_time_is_four_minutes_l2287_228799


namespace NUMINAMATH_CALUDE_cone_curved_surface_area_l2287_228715

/-- The curved surface area of a cone with given slant height and base radius -/
theorem cone_curved_surface_area 
  (slant_height : ℝ) 
  (base_radius : ℝ) 
  (h1 : slant_height = 10) 
  (h2 : base_radius = 5) : 
  π * base_radius * slant_height = 50 * π := by
sorry

end NUMINAMATH_CALUDE_cone_curved_surface_area_l2287_228715


namespace NUMINAMATH_CALUDE_three_fifths_of_negative_twelve_sevenths_l2287_228786

theorem three_fifths_of_negative_twelve_sevenths :
  (3 : ℚ) / 5 * (-12 : ℚ) / 7 = -36 / 35 := by sorry

end NUMINAMATH_CALUDE_three_fifths_of_negative_twelve_sevenths_l2287_228786


namespace NUMINAMATH_CALUDE_smallest_f_one_l2287_228748

/-- A cubic polynomial f(x) with specific properties -/
noncomputable def f (r s : ℝ) (x : ℝ) : ℝ := (x - r) * (x - s) * (x - (r + s) / 2)

/-- The theorem stating the smallest value of f(1) -/
theorem smallest_f_one (r s : ℝ) :
  (r ≠ s) →
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f r s (f r s x₁) = 0 ∧ f r s (f r s x₂) = 0 ∧ f r s (f r s x₃) = 0) →
  (∀ (x : ℝ), f r s (f r s x) = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∀ (r' s' : ℝ), r' ≠ s' → f r' s' 1 ≥ 3/8) ∧
  (∃ (r₀ s₀ : ℝ), r₀ ≠ s₀ ∧ f r₀ s₀ 1 = 3/8) := by
  sorry


end NUMINAMATH_CALUDE_smallest_f_one_l2287_228748


namespace NUMINAMATH_CALUDE_cheaper_lens_price_l2287_228740

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) : 
  original_price = 300 →
  discount_rate = 0.2 →
  savings = 20 →
  original_price * (1 - discount_rate) - savings = 220 := by
sorry

end NUMINAMATH_CALUDE_cheaper_lens_price_l2287_228740


namespace NUMINAMATH_CALUDE_range_invariant_under_shift_l2287_228726

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of having a range (-1, 1)
def hasRange_openInterval_neg1_1 (g : ℝ → ℝ) : Prop :=
  ∀ y, y ∈ Set.range g ↔ -1 < y ∧ y < 1

-- State the theorem
theorem range_invariant_under_shift :
  hasRange_openInterval_neg1_1 (fun x ↦ f (x + 2011)) →
  hasRange_openInterval_neg1_1 f :=
by sorry

end NUMINAMATH_CALUDE_range_invariant_under_shift_l2287_228726


namespace NUMINAMATH_CALUDE_tennis_players_count_l2287_228766

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  neither_players : ℕ
  both_players : ℕ

/-- Calculate the number of tennis players in the sports club -/
def tennis_players (club : SportsClub) : ℕ :=
  club.total_members - club.neither_players - (club.badminton_players - club.both_players)

/-- Theorem stating the number of tennis players in the given club configuration -/
theorem tennis_players_count (club : SportsClub) 
  (h1 : club.total_members = 30)
  (h2 : club.badminton_players = 17)
  (h3 : club.neither_players = 2)
  (h4 : club.both_players = 8) :
  tennis_players club = 19 := by
  sorry

#eval tennis_players ⟨30, 17, 2, 8⟩

end NUMINAMATH_CALUDE_tennis_players_count_l2287_228766


namespace NUMINAMATH_CALUDE_max_y_value_l2287_228759

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : 
  ∀ (z : ℤ), z * x + 3 * x + 2 * z ≠ -4 ∨ z ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l2287_228759


namespace NUMINAMATH_CALUDE_chandler_skateboard_savings_l2287_228718

/-- Calculates the minimum number of full weeks required to save for a skateboard -/
def min_weeks_to_save (skateboard_cost : ℕ) (gift_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  ((skateboard_cost - gift_money + weekly_earnings - 1) / weekly_earnings : ℕ)

theorem chandler_skateboard_savings :
  min_weeks_to_save 550 130 18 = 24 := by
  sorry

end NUMINAMATH_CALUDE_chandler_skateboard_savings_l2287_228718


namespace NUMINAMATH_CALUDE_subway_scenarios_l2287_228701

/-- Represents the fare structure for the subway -/
def fare (x : ℕ) : ℕ :=
  if x ≤ 4 then 2
  else if x ≤ 9 then 4
  else if x ≤ 15 then 6
  else 0

/-- The maximum number of stations -/
def max_stations : ℕ := 15

/-- Calculates the number of scenarios where two passengers pay a total fare -/
def scenarios_for_total_fare (total_fare : ℕ) : ℕ := sorry

/-- Calculates the number of scenarios where passenger A gets off before passenger B -/
def scenarios_a_before_b (total_fare : ℕ) : ℕ := sorry

theorem subway_scenarios :
  (scenarios_for_total_fare 6 = 40) ∧
  (scenarios_a_before_b 8 = 34) := by sorry

end NUMINAMATH_CALUDE_subway_scenarios_l2287_228701


namespace NUMINAMATH_CALUDE_parabola_parameter_value_l2287_228705

/-- Proves that for a parabola y^2 = 2px (p > 0) with axis of symmetry at distance 4 from the point (3, 0), the value of p is 2. -/
theorem parabola_parameter_value (p : ℝ) (h1 : p > 0) : 
  (∃ (x y : ℝ), y^2 = 2*p*x) →  -- Parabola equation
  (∃ (a : ℝ), ∀ (x y : ℝ), y^2 = 2*p*x → x = a) →  -- Axis of symmetry exists
  (|3 - (- p/2)| = 4) →  -- Distance from (3, 0) to axis of symmetry is 4
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_parameter_value_l2287_228705


namespace NUMINAMATH_CALUDE_problem_statement_l2287_228751

theorem problem_statement (a b : ℤ) (h1 : a = -5) (h2 : b = 3) : 
  -a - b^4 + a*b = -91 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2287_228751


namespace NUMINAMATH_CALUDE_congruence_from_power_difference_l2287_228754

theorem congruence_from_power_difference (a b : ℕ+) (h : a^b.val - b^a.val = 1008) :
  a ≡ b [ZMOD 1008] := by
  sorry

end NUMINAMATH_CALUDE_congruence_from_power_difference_l2287_228754


namespace NUMINAMATH_CALUDE_male_female_ratio_l2287_228743

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  male_members : ℕ
  female_members : ℕ
  total_tickets : ℕ
  male_tickets : ℕ
  female_tickets : ℕ

/-- The conditions given in the problem -/
def association_conditions (a : Association) : Prop :=
  (a.total_tickets : ℚ) / (a.male_members + a.female_members : ℚ) = 66 ∧
  (a.female_tickets : ℚ) / (a.female_members : ℚ) = 70 ∧
  (a.male_tickets : ℚ) / (a.male_members : ℚ) = 58 ∧
  a.total_tickets = a.male_tickets + a.female_tickets

/-- The theorem stating that under the given conditions, the male to female ratio is 1:2 -/
theorem male_female_ratio (a : Association) (h : association_conditions a) :
  (a.male_members : ℚ) / (a.female_members : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_male_female_ratio_l2287_228743


namespace NUMINAMATH_CALUDE_alex_candles_used_l2287_228738

/-- The number of candles Alex used -/
def candles_used (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem stating that Alex used 32 candles -/
theorem alex_candles_used :
  let initial : ℕ := 44
  let remaining : ℕ := 12
  candles_used initial remaining = 32 := by
  sorry

end NUMINAMATH_CALUDE_alex_candles_used_l2287_228738


namespace NUMINAMATH_CALUDE_jim_distance_driven_l2287_228793

/-- The distance Jim has driven so far in his journey -/
def distance_driven (total_journey : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_journey - remaining_distance

/-- Theorem stating that Jim has driven 215 miles -/
theorem jim_distance_driven :
  distance_driven 1200 985 = 215 := by
  sorry

end NUMINAMATH_CALUDE_jim_distance_driven_l2287_228793


namespace NUMINAMATH_CALUDE_median_and_mode_of_S_l2287_228710

/-- The set of data --/
def S : Finset ℕ := {6, 7, 4, 7, 5, 2}

/-- Definition of median for a finite set of natural numbers --/
def median (s : Finset ℕ) : ℚ := sorry

/-- Definition of mode for a finite set of natural numbers --/
def mode (s : Finset ℕ) : ℕ := sorry

theorem median_and_mode_of_S :
  median S = 5.5 ∧ mode S = 7 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_of_S_l2287_228710


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_four_l2287_228749

/-- Given a triangle ABC with circumradius R, if 2R(sin²A - sin²B) = (√2a - c)sinC, 
    then the measure of angle B is π/4. -/
theorem angle_B_is_pi_over_four 
  (A B C : ℝ) 
  (a b c R : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) 
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h6 : 0 < R) 
  (h7 : a = 2 * R * Real.sin A) 
  (h8 : b = 2 * R * Real.sin B) 
  (h9 : c = 2 * R * Real.sin C) 
  (h10 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_B_is_pi_over_four_l2287_228749


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2287_228728

theorem quadratic_equation_roots (x : ℝ) :
  x^2 - 4*x - 2 = 0 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2287_228728


namespace NUMINAMATH_CALUDE_additional_cans_needed_l2287_228785

def martha_cans : ℕ := 90
def diego_cans : ℕ := martha_cans / 2 + 10
def leah_cans : ℕ := martha_cans / 3 - 5

def martha_aluminum : ℕ := (martha_cans * 70) / 100
def diego_aluminum : ℕ := (diego_cans * 50) / 100
def leah_aluminum : ℕ := (leah_cans * 80) / 100

def total_needed : ℕ := 200

theorem additional_cans_needed :
  total_needed - (martha_aluminum + diego_aluminum + leah_aluminum) = 90 :=
by sorry

end NUMINAMATH_CALUDE_additional_cans_needed_l2287_228785


namespace NUMINAMATH_CALUDE_f_at_seven_l2287_228712

def f (x : ℝ) : ℝ := 7*x^5 + 12*x^4 - 5*x^3 - 6*x^2 + 3*x - 5

theorem f_at_seven : f 7 = 144468 := by
  sorry

end NUMINAMATH_CALUDE_f_at_seven_l2287_228712


namespace NUMINAMATH_CALUDE_income_ratio_proof_l2287_228758

/-- Proves that the ratio of A's monthly income to B's monthly income is 2.5:1 -/
theorem income_ratio_proof (c_monthly_income b_monthly_income a_annual_income : ℝ) 
  (h1 : c_monthly_income = 14000)
  (h2 : b_monthly_income = c_monthly_income * 1.12)
  (h3 : a_annual_income = 470400) : 
  (a_annual_income / 12) / b_monthly_income = 2.5 := by
  sorry

#check income_ratio_proof

end NUMINAMATH_CALUDE_income_ratio_proof_l2287_228758


namespace NUMINAMATH_CALUDE_anna_candy_per_house_proof_l2287_228762

/-- The number of candy pieces Anna gets per house -/
def anna_candy_per_house : ℕ := 14

/-- The number of candy pieces Billy gets per house -/
def billy_candy_per_house : ℕ := 11

/-- The number of houses Anna visits -/
def anna_houses : ℕ := 60

/-- The number of houses Billy visits -/
def billy_houses : ℕ := 75

/-- The difference in total candy pieces between Anna and Billy -/
def candy_difference : ℕ := 15

theorem anna_candy_per_house_proof :
  anna_candy_per_house * anna_houses = billy_candy_per_house * billy_houses + candy_difference :=
by
  sorry

#eval anna_candy_per_house

end NUMINAMATH_CALUDE_anna_candy_per_house_proof_l2287_228762


namespace NUMINAMATH_CALUDE_route2_faster_l2287_228784

-- Define the probabilities and delay times for each route
def prob_green_A : ℚ := 1/2
def prob_green_B : ℚ := 2/3
def delay_A : ℕ := 2
def delay_B : ℕ := 3
def time_green_AB : ℕ := 20

def prob_green_a : ℚ := 3/4
def prob_green_b : ℚ := 2/5
def delay_a : ℕ := 8
def delay_b : ℕ := 5
def time_green_ab : ℕ := 15

-- Define the expected delay for each route
def expected_delay_route1 : ℚ := 
  (1 - prob_green_A) * delay_A + (1 - prob_green_B) * delay_B

def expected_delay_route2 : ℚ := 
  (1 - prob_green_a) * delay_a + (1 - prob_green_b) * delay_b

-- Define the expected travel time for each route
def expected_time_route1 : ℚ := time_green_AB + expected_delay_route1
def expected_time_route2 : ℚ := time_green_ab + expected_delay_route2

-- Theorem statement
theorem route2_faster : expected_time_route2 < expected_time_route1 :=
  sorry

end NUMINAMATH_CALUDE_route2_faster_l2287_228784


namespace NUMINAMATH_CALUDE_student_count_l2287_228765

theorem student_count (average_student_age : ℝ) (teacher_age : ℝ) (new_average_age : ℝ) :
  average_student_age = 15 →
  teacher_age = 26 →
  new_average_age = 16 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * average_student_age + teacher_age = (n + 1 : ℝ) * new_average_age ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2287_228765


namespace NUMINAMATH_CALUDE_envelope_touches_all_C_a_l2287_228719

/-- The curve C_a is defined by the equation (y - a^2)^2 = x^2(a^2 - x^2) for a > 0 -/
def C_a (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ (y - a^2)^2 = x^2 * (a^2 - x^2)

/-- The envelope curve -/
def envelope_curve (x y : ℝ) : Prop :=
  y = (3 * x^2) / 4

/-- Theorem stating that the envelope curve touches all C_a curves -/
theorem envelope_touches_all_C_a :
  ∀ (a x y : ℝ), C_a a x y → ∃ (x₀ y₀ : ℝ), 
    envelope_curve x₀ y₀ ∧ 
    C_a a x₀ y₀ ∧
    (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
      ∀ (x' y' : ℝ), 
        ((x' - x₀)^2 + (y' - y₀)^2 < δ^2) →
        (envelope_curve x' y' → ¬C_a a x' y') ∧
        (C_a a x' y' → ¬envelope_curve x' y')) :=
by sorry

end NUMINAMATH_CALUDE_envelope_touches_all_C_a_l2287_228719


namespace NUMINAMATH_CALUDE_unmeasurable_weights_theorem_l2287_228717

def available_weights : List Nat := [1, 2, 3, 8, 16, 32]

def is_measurable (n : Nat) (weights : List Nat) : Prop :=
  ∃ (subset : List Nat), subset.Sublist weights ∧ subset.sum = n

def unmeasurable_weights : Set Nat :=
  {n | n ≤ 60 ∧ ¬(is_measurable n available_weights)}

theorem unmeasurable_weights_theorem :
  unmeasurable_weights = {7, 15, 23, 31, 39, 47, 55} := by
  sorry

end NUMINAMATH_CALUDE_unmeasurable_weights_theorem_l2287_228717


namespace NUMINAMATH_CALUDE_max_daily_profit_l2287_228788

/-- Represents the daily profit function for a store selling a product -/
def daily_profit (x : ℝ) : ℝ :=
  (2 + 0.5 * x) * (200 - 10 * x)

/-- Theorem stating the maximum daily profit and the corresponding selling price -/
theorem max_daily_profit :
  ∃ (x : ℝ), daily_profit x = 720 ∧ 
  (∀ (y : ℝ), daily_profit y ≤ daily_profit x) ∧
  x = 8 :=
sorry

end NUMINAMATH_CALUDE_max_daily_profit_l2287_228788


namespace NUMINAMATH_CALUDE_subtraction_problem_l2287_228755

theorem subtraction_problem : 
  2000000000000 - 1111111111111 - 222222222222 = 666666666667 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2287_228755


namespace NUMINAMATH_CALUDE_group_size_l2287_228756

/-- 
Given a group of people with men, women, and children, where:
- The number of men is twice the number of women
- The number of women is 3 times the number of children
- The number of children is 30

Prove that the total number of people in the group is 300.
-/
theorem group_size (children women men : ℕ) 
  (h1 : men = 2 * women) 
  (h2 : women = 3 * children) 
  (h3 : children = 30) : 
  children + women + men = 300 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2287_228756


namespace NUMINAMATH_CALUDE_cup_sales_problem_l2287_228752

/-- Proves that the number of additional days is 11, given the conditions of the cup sales problem -/
theorem cup_sales_problem (first_day_sales : ℕ) (daily_sales : ℕ) (average_sales : ℚ) : 
  first_day_sales = 86 →
  daily_sales = 50 →
  average_sales = 53 →
  ∃ d : ℕ, 
    (first_day_sales + d * daily_sales : ℚ) / (d + 1 : ℚ) = average_sales ∧
    d = 11 := by
  sorry


end NUMINAMATH_CALUDE_cup_sales_problem_l2287_228752


namespace NUMINAMATH_CALUDE_original_number_is_192_l2287_228745

theorem original_number_is_192 (N : ℚ) : 
  (((N / 8 + 8) - 30) * 6) = 12 → N = 192 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_192_l2287_228745


namespace NUMINAMATH_CALUDE_expression_simplification_l2287_228767

theorem expression_simplification (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2287_228767


namespace NUMINAMATH_CALUDE_cistern_fill_time_l2287_228795

def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

theorem cistern_fill_time :
  let rate1 : ℚ := 1 / 10
  let rate2 : ℚ := 1 / 12
  let rate3 : ℚ := -1 / 25
  fill_time rate1 rate2 rate3 = 300 / 43 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l2287_228795


namespace NUMINAMATH_CALUDE_grass_cutting_cost_l2287_228774

/-- The cost of cutting grass once, given specific growth and cost conditions --/
theorem grass_cutting_cost
  (initial_height : ℝ)
  (growth_rate : ℝ)
  (cut_threshold : ℝ)
  (annual_cost : ℝ)
  (h1 : initial_height = 2)
  (h2 : growth_rate = 0.5)
  (h3 : cut_threshold = 4)
  (h4 : annual_cost = 300)
  : (annual_cost / (12 / ((cut_threshold - initial_height) / growth_rate))) = 100 := by
  sorry

end NUMINAMATH_CALUDE_grass_cutting_cost_l2287_228774


namespace NUMINAMATH_CALUDE_remove_six_maximizes_probability_l2287_228753

def original_list : List Int := List.range 15 |>.map (λ x => x - 2)

def remove_number (list : List Int) (n : Int) : List Int :=
  list.filter (λ x => x ≠ n)

def count_pairs_sum_11 (list : List Int) : Nat :=
  list.filterMap (λ x => 
    if x < 11 ∧ list.contains (11 - x) ∧ x ≠ 11 - x
    then some (x, 11 - x)
    else none
  ) |>.length

theorem remove_six_maximizes_probability :
  ∀ n ∈ original_list, n ≠ 6 →
    count_pairs_sum_11 (remove_number original_list 6) ≥ 
    count_pairs_sum_11 (remove_number original_list n) :=
by sorry

end NUMINAMATH_CALUDE_remove_six_maximizes_probability_l2287_228753


namespace NUMINAMATH_CALUDE_water_flow_solution_l2287_228709

/-- Represents the water flow problem --/
def water_flow_problem (t : ℝ) : Prop :=
  let initial_rate : ℝ := 2 / 10  -- 2 cups per 10 minutes
  let final_rate : ℝ := 4 / 10    -- 4 cups per 10 minutes
  let initial_duration : ℝ := 2 * t  -- flows for t minutes twice
  let final_duration : ℝ := 60    -- flows for 60 minutes at final rate
  let total_water : ℝ := initial_rate * initial_duration + final_rate * final_duration
  let remaining_water : ℝ := total_water / 2
  remaining_water = 18 ∧ t = 30

/-- Theorem stating the solution to the water flow problem --/
theorem water_flow_solution :
  ∃ t : ℝ, water_flow_problem t :=
sorry

end NUMINAMATH_CALUDE_water_flow_solution_l2287_228709


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2287_228791

/-- The constant term in the binomial expansion of (3x^2 - 2/x^3)^5 is 1080 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (3 * x^2 - 2 / x^3)^5
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = 1080 :=
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2287_228791


namespace NUMINAMATH_CALUDE_fly_ceiling_distance_l2287_228764

theorem fly_ceiling_distance (z : ℝ) :
  (3 : ℝ)^2 + 2^2 + z^2 = 6^2 → z = Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_fly_ceiling_distance_l2287_228764


namespace NUMINAMATH_CALUDE_product_neg_seventeen_sum_l2287_228760

theorem product_neg_seventeen_sum (a b c : ℤ) : 
  a * b * c = -17 → (a + b + c = -17 ∨ a + b + c = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_product_neg_seventeen_sum_l2287_228760


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2287_228790

theorem inequality_equivalence (x : ℝ) : 3 * x^2 + x < 8 ↔ -2 < x ∧ x < 4/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2287_228790


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l2287_228721

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 1) :
  1 / a + 27 / b ≥ 48 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 3 * a + b = 1 ∧ 1 / a + 27 / b < 48 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l2287_228721


namespace NUMINAMATH_CALUDE_unique_n_reaches_16_l2287_228706

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else (n / 2)^2

theorem unique_n_reaches_16 :
  ∃! n : ℕ, n ∈ Finset.range 100 ∧
  ∃ k : ℕ, (k.iterate g n) = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_n_reaches_16_l2287_228706


namespace NUMINAMATH_CALUDE_emily_bought_seven_songs_l2287_228733

/-- The number of songs Emily bought later -/
def songs_bought_later (initial_songs total_songs : ℕ) : ℕ :=
  total_songs - initial_songs

/-- Proof that Emily bought 7 songs later -/
theorem emily_bought_seven_songs :
  let initial_songs := 6
  let total_songs := 13
  songs_bought_later initial_songs total_songs = 7 := by
  sorry

end NUMINAMATH_CALUDE_emily_bought_seven_songs_l2287_228733


namespace NUMINAMATH_CALUDE_linear_regression_passes_through_mean_l2287_228778

variables {x y : ℝ} (x_bar y_bar a_hat b_hat : ℝ)

/-- The linear regression equation -/
def linear_regression (x : ℝ) : ℝ := b_hat * x + a_hat

/-- The intercept of the linear regression equation -/
def intercept : ℝ := y_bar - b_hat * x_bar

theorem linear_regression_passes_through_mean :
  a_hat = intercept x_bar y_bar b_hat →
  linear_regression x_bar a_hat b_hat = y_bar :=
sorry

end NUMINAMATH_CALUDE_linear_regression_passes_through_mean_l2287_228778


namespace NUMINAMATH_CALUDE_tickets_to_buy_l2287_228777

/-- The number of additional tickets Zach needs to buy for three rides -/
theorem tickets_to_buy (ferris_wheel_cost roller_coaster_cost log_ride_cost current_tickets : ℕ) 
  (h1 : ferris_wheel_cost = 2)
  (h2 : roller_coaster_cost = 7)
  (h3 : log_ride_cost = 1)
  (h4 : current_tickets = 1) :
  ferris_wheel_cost + roller_coaster_cost + log_ride_cost - current_tickets = 9 := by
  sorry

end NUMINAMATH_CALUDE_tickets_to_buy_l2287_228777


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2287_228739

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 2, 3]
  Matrix.det A = 2 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2287_228739


namespace NUMINAMATH_CALUDE_wedge_volume_l2287_228734

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (α : ℝ) : 
  d = 10 → α = 60 → (π * (d / 2)^2 * (d / 2 * Real.cos (α * π / 180))) = 125 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l2287_228734


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l2287_228772

theorem overlapping_sectors_area (r : ℝ) (h : r = 12) :
  let sector_angle : ℝ := 60
  let sector_area := (sector_angle / 360) * Real.pi * r^2
  let triangle_area := (Real.sqrt 3 / 4) * r^2
  let shaded_area := 2 * (sector_area - triangle_area)
  shaded_area = 48 * Real.pi - 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l2287_228772


namespace NUMINAMATH_CALUDE_darry_total_steps_l2287_228707

/-- Represents the number of steps climbed on a ladder -/
structure LadderClimb where
  steps : Nat
  times : Nat

/-- Calculates the total number of steps climbed on a ladder -/
def totalStepsOnLadder (climb : LadderClimb) : Nat :=
  climb.steps * climb.times

/-- Represents Darry's ladder climbs for the day -/
structure DarryClimbs where
  largest : LadderClimb
  medium : LadderClimb
  smaller : LadderClimb
  smallest : LadderClimb

/-- Darry's actual climbs for the day -/
def darryActualClimbs : DarryClimbs :=
  { largest := { steps := 20, times := 12 }
  , medium := { steps := 15, times := 8 }
  , smaller := { steps := 10, times := 10 }
  , smallest := { steps := 5, times := 15 }
  }

/-- Calculates the total number of steps Darry climbed -/
def totalStepsClimbed (climbs : DarryClimbs) : Nat :=
  totalStepsOnLadder climbs.largest +
  totalStepsOnLadder climbs.medium +
  totalStepsOnLadder climbs.smaller +
  totalStepsOnLadder climbs.smallest

/-- Theorem stating that Darry climbed 535 steps in total -/
theorem darry_total_steps :
  totalStepsClimbed darryActualClimbs = 535 := by
  sorry

end NUMINAMATH_CALUDE_darry_total_steps_l2287_228707


namespace NUMINAMATH_CALUDE_average_work_difference_l2287_228773

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

def days_in_week : ℕ := 7

theorem average_work_difference :
  (daily_differences.sum : ℚ) / days_in_week = 0.857 := by
  sorry

end NUMINAMATH_CALUDE_average_work_difference_l2287_228773


namespace NUMINAMATH_CALUDE_color_paint_can_size_is_one_gallon_l2287_228776

/-- Represents the paint job for a house --/
structure PaintJob where
  bedrooms : Nat
  otherRooms : Nat
  gallonsPerRoom : Nat
  whitePaintCanSize : Nat
  totalCans : Nat

/-- Calculates the size of each can of color paint --/
def colorPaintCanSize (job : PaintJob) : Rat :=
  let totalRooms := job.bedrooms + job.otherRooms
  let totalPaint := totalRooms * job.gallonsPerRoom
  let whitePaint := job.otherRooms * job.gallonsPerRoom
  let whiteCans := whitePaint / job.whitePaintCanSize
  let colorCans := job.totalCans - whiteCans
  let colorPaint := job.bedrooms * job.gallonsPerRoom
  colorPaint / colorCans

/-- Theorem stating that the size of each can of color paint is 1 gallon --/
theorem color_paint_can_size_is_one_gallon (job : PaintJob)
  (h1 : job.bedrooms = 3)
  (h2 : job.otherRooms = 2 * job.bedrooms)
  (h3 : job.gallonsPerRoom = 2)
  (h4 : job.whitePaintCanSize = 3)
  (h5 : job.totalCans = 10) :
  colorPaintCanSize job = 1 := by
  sorry

#eval colorPaintCanSize { bedrooms := 3, otherRooms := 6, gallonsPerRoom := 2, whitePaintCanSize := 3, totalCans := 10 }

end NUMINAMATH_CALUDE_color_paint_can_size_is_one_gallon_l2287_228776


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2287_228730

def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 3 = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 - q*x - p = 0}

theorem intersection_implies_sum (p q : ℝ) : A p ∩ B p q = {-1} → 2*p + q = -7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2287_228730


namespace NUMINAMATH_CALUDE_balloon_sum_equals_total_l2287_228720

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- Theorem stating that the sum of individual balloon counts equals the total -/
theorem balloon_sum_equals_total :
  fred_balloons + sam_balloons + mary_balloons = total_balloons :=
by sorry

end NUMINAMATH_CALUDE_balloon_sum_equals_total_l2287_228720


namespace NUMINAMATH_CALUDE_skew_lines_and_planes_l2287_228716

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the given conditions
variable (a b : Line)
variable (α : Plane)

-- Theorem statement
theorem skew_lines_and_planes 
  (h_skew : skew a b)
  (h_parallel : parallel a α) :
  (∃ β : Plane, parallel b β) ∧ 
  (∃ γ : Plane, subset b γ) ∧
  (∃ δ : Set Plane, Set.Infinite δ ∧ ∀ π ∈ δ, intersect b π) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_and_planes_l2287_228716


namespace NUMINAMATH_CALUDE_B_equals_zero_one_l2287_228761

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_zero_one : B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_B_equals_zero_one_l2287_228761


namespace NUMINAMATH_CALUDE_parabola_translation_l2287_228713

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 3 * x^2

/-- The translated parabola function -/
def translated_parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

/-- Theorem stating that the translated_parabola is the result of
    translating the original_parabola 1 unit right and 4 units down -/
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x - 1) - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2287_228713


namespace NUMINAMATH_CALUDE_student_number_problem_l2287_228782

theorem student_number_problem : ∃ x : ℤ, 2 * x - 148 = 110 ∧ x = 129 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2287_228782


namespace NUMINAMATH_CALUDE_ball_count_l2287_228729

theorem ball_count (num_red : ℕ) (prob_red : ℚ) (total : ℕ) : 
  num_red = 4 → prob_red = 1/3 → total = num_red / prob_red → total = 12 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_l2287_228729


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2287_228711

/-- A quadratic function with vertex (-3, 4) passing through (1, 2) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) :
  (∀ x, f a b c x = a * x^2 + b * x + c) →
  f a b c (-3) = 4 →
  (∀ h : ℝ, f a b c (-3 + h) = f a b c (-3 - h)) →
  f a b c 1 = 2 →
  a + b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2287_228711


namespace NUMINAMATH_CALUDE_ningan_properties_l2287_228742

-- Define a "Ning'an number"
def is_ningan (a : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ (a / 10 ≠ a % 10) ∧ (a / 10 ≠ 0) ∧ (a % 10 ≠ 0)

-- Define the function f
def f (a : ℕ) : ℕ :=
  (a + (10 * (a % 10) + (a / 10))) / 11

-- Theorem statement
theorem ningan_properties :
  (is_ningan 58 ∧ is_ningan 31 ∧ ¬is_ningan 60 ∧ ¬is_ningan 88) ∧
  (f 42 = 6) ∧
  (∀ n : ℕ, is_ningan (10 * n + (2 * n + 1)) → f (10 * n + (2 * n + 1)) = 13 → 10 * n + (2 * n + 1) = 49) ∧
  (∀ x : ℕ, is_ningan x → (x - 5 * f x > 30) ↔ (x = 71 ∨ x = 81 ∨ x = 82 ∨ x = 91 ∨ x = 92 ∨ x = 93)) :=
by sorry

end NUMINAMATH_CALUDE_ningan_properties_l2287_228742


namespace NUMINAMATH_CALUDE_farmer_loss_proof_l2287_228771

def expected_revenue : ℝ := 100000.00
def possible_loss : ℝ := 21987.53

theorem farmer_loss_proof :
  possible_loss ≥ (1/5) * expected_revenue ∧
  possible_loss ≤ (1/4) * expected_revenue :=
by sorry

end NUMINAMATH_CALUDE_farmer_loss_proof_l2287_228771


namespace NUMINAMATH_CALUDE_shaded_area_of_grid_square_l2287_228735

theorem shaded_area_of_grid_square (d : ℝ) (h1 : d = 10) : 
  let s := d / Real.sqrt 2
  let small_square_side := s / 5
  let small_square_area := small_square_side ^ 2
  let total_area := 25 * small_square_area
  total_area = 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_grid_square_l2287_228735


namespace NUMINAMATH_CALUDE_temperature_difference_is_eight_l2287_228732

-- Define the temperatures
def highest_temp : ℤ := 5
def lowest_temp : ℤ := -3

-- Define the temperature difference
def temp_difference : ℤ := highest_temp - lowest_temp

-- Theorem to prove
theorem temperature_difference_is_eight :
  temp_difference = 8 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_is_eight_l2287_228732


namespace NUMINAMATH_CALUDE_replaced_person_age_l2287_228723

/-- Represents a group of people with their ages -/
structure AgeGroup where
  size : ℕ
  totalAge : ℕ

/-- Theorem: Given the conditions, the replaced person's age was 46 years -/
theorem replaced_person_age
  (group : AgeGroup)
  (h_size : group.size = 10)
  (h_new_age : ℕ)
  (h_new_age_val : h_new_age = 16)
  (h_avg_decrease : ℕ)
  (h_avg_decrease_val : h_avg_decrease = 3)
  (h_after : AgeGroup)
  (h_after_size : h_after.size = group.size)
  (h_after_total_age : h_after.totalAge = group.totalAge - group.size * h_avg_decrease + h_new_age - (group.totalAge / group.size)) :
  group.totalAge / group.size - (group.totalAge - group.size * h_avg_decrease) / group.size = 46 - h_new_age :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_age_l2287_228723


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l2287_228775

def fair_coin_probability : ℚ := 1 / 2
def fair_die_probability : ℚ := 1 / 6

theorem coin_and_die_probability :
  let p_tails := fair_coin_probability
  let p_one_or_two := 2 * fair_die_probability
  p_tails * p_one_or_two = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l2287_228775


namespace NUMINAMATH_CALUDE_find_t_l2287_228727

/-- The number of hours I worked -/
def my_hours (t : ℝ) : ℝ := t - 4

/-- My hourly rate in dollars -/
def my_rate (t : ℝ) : ℝ := 3*t - 7

/-- The number of hours Bob worked -/
def bob_hours (t : ℝ) : ℝ := 3*t - 12

/-- Bob's hourly rate in dollars -/
def bob_rate (t : ℝ) : ℝ := t - 6

/-- Our total earnings were the same -/
def equal_earnings (t : ℝ) : Prop :=
  my_hours t * my_rate t = bob_hours t * bob_rate t

theorem find_t : ∃ t : ℝ, equal_earnings t ∧ t = 44 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l2287_228727


namespace NUMINAMATH_CALUDE_masha_result_non_negative_l2287_228768

theorem masha_result_non_negative (a b c d : ℝ) 
  (sum_eq_prod : a + b = c * d) 
  (prod_eq_sum : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_masha_result_non_negative_l2287_228768


namespace NUMINAMATH_CALUDE_kara_water_consumption_l2287_228724

/-- The amount of water Kara drinks with each dose of medication -/
def water_per_dose : ℕ := 4

/-- The number of doses Kara takes per day -/
def doses_per_day : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Kara followed her medication routine -/
def total_weeks : ℕ := 2

/-- The number of doses Kara forgot in the second week -/
def forgotten_doses : ℕ := 2

/-- Calculates the total amount of water Kara drank with her medication over two weeks -/
def total_water_consumption : ℕ :=
  water_per_dose * doses_per_day * days_per_week * total_weeks - water_per_dose * forgotten_doses

theorem kara_water_consumption :
  total_water_consumption = 160 := by
  sorry

end NUMINAMATH_CALUDE_kara_water_consumption_l2287_228724


namespace NUMINAMATH_CALUDE_tan_70_cos_10_sqrt_3_tan_20_minus_1_l2287_228750

theorem tan_70_cos_10_sqrt_3_tan_20_minus_1 : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_sqrt_3_tan_20_minus_1_l2287_228750


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2287_228798

/-- Given a hyperbola with semi-major axis a and semi-minor axis b, 
    a point P on its right branch, F as its right focus, 
    and M on the line x = -a²/c, where c is the focal distance,
    prove that if OP = OF + OM and OP ⋅ FM = 0, then the eccentricity is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P F M O : ℝ × ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →  -- P is on the right branch of the hyperbola
  P.1 > 0 →  -- P is on the right branch
  F = (c, 0) →  -- F is the right focus
  M.1 = -a^2 / c →  -- M is on the line x = -a²/c
  P.1 - O.1 = F.1 - O.1 + M.1 - O.1 →  -- OP = OF + OM (x-component)
  P.2 - O.2 = F.2 - O.2 + M.2 - O.2 →  -- OP = OF + OM (y-component)
  (P.1 - F.1) * (M.1 - F.1) + (P.2 - F.2) * (M.2 - F.2) = 0 →  -- OP ⋅ FM = 0
  c / a = 2 :=  -- eccentricity is 2
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2287_228798


namespace NUMINAMATH_CALUDE_standard_deviation_of_numbers_l2287_228708

def numbers : List ℝ := [9.8, 9.8, 9.9, 9.9, 10.0, 10.0, 10.1, 10.5]

theorem standard_deviation_of_numbers :
  let mean : ℝ := 10
  let count_within_one_std : ℕ := 7
  let n : ℕ := numbers.length
  ∀ σ : ℝ,
    (mean = (numbers.sum / n)) →
    (count_within_one_std = (numbers.filter (λ x => |x - mean| ≤ σ)).length) →
    (count_within_one_std = (n * 875 / 1000)) →
    σ = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_numbers_l2287_228708
