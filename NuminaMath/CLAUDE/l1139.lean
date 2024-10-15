import Mathlib

namespace NUMINAMATH_CALUDE_total_monthly_earnings_l1139_113972

/-- Represents an apartment floor --/
structure Floor :=
  (rooms : ℕ)
  (rent : ℝ)
  (occupancy : ℝ)

/-- Calculates the monthly earnings for a floor --/
def floorEarnings (f : Floor) : ℝ :=
  f.rooms * f.rent * f.occupancy

/-- Represents an apartment building --/
structure Building :=
  (floors : List Floor)

/-- Calculates the total monthly earnings for a building --/
def buildingEarnings (b : Building) : ℝ :=
  (b.floors.map floorEarnings).sum

/-- The first building --/
def building1 : Building :=
  { floors := [
    { rooms := 5, rent := 15, occupancy := 0.8 },
    { rooms := 6, rent := 25, occupancy := 0.75 },
    { rooms := 9, rent := 30, occupancy := 0.5 },
    { rooms := 4, rent := 60, occupancy := 0.85 }
  ] }

/-- The second building --/
def building2 : Building :=
  { floors := [
    { rooms := 7, rent := 20, occupancy := 0.9 },
    { rooms := 8, rent := 42.5, occupancy := 0.7 }, -- Average rent for the second floor
    { rooms := 6, rent := 60, occupancy := 0.6 }
  ] }

/-- The main theorem --/
theorem total_monthly_earnings :
  buildingEarnings building1 + buildingEarnings building2 = 1091.5 := by
  sorry

end NUMINAMATH_CALUDE_total_monthly_earnings_l1139_113972


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1139_113936

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∃ θ : ℝ, θ = Real.pi/4 ∧ 
    (∀ t : ℝ, ∃ x y : ℝ, x = t ∧ y = (b/a)*t ∨ x = t ∧ y = -(b/a)*t) ∧
    θ = Real.arctan ((2*(b/a))/(1 - (b/a)^2))) →
  a/b = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1139_113936


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1139_113989

theorem arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) (aₙ : ℕ) :
  a₁ = 1 →
  d = 2 →
  n > 0 →
  aₙ = 21 →
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ)) / 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1139_113989


namespace NUMINAMATH_CALUDE_f_max_value_when_a_2_f_no_min_value_when_a_2_f_decreasing_when_a_leq_neg_quarter_f_decreasing_when_neg_quarter_lt_a_leq_zero_f_monotonicity_when_a_gt_zero_l1139_113965

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x - (1/2) * x^2

-- Theorem for the maximum value when a = 2
theorem f_max_value_when_a_2 :
  ∃ (x : ℝ), x > 0 ∧ f 2 x = -(3/2) ∧ ∀ (y : ℝ), y > 0 → f 2 y ≤ f 2 x :=
sorry

-- Theorem for no minimum value when a = 2
theorem f_no_min_value_when_a_2 :
  ∀ (M : ℝ), ∃ (x : ℝ), x > 0 ∧ f 2 x < M :=
sorry

-- Theorem for monotonicity when a ≤ -1/4
theorem f_decreasing_when_a_leq_neg_quarter (a : ℝ) (h : a ≤ -(1/4)) :
  ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y :=
sorry

-- Theorem for monotonicity when -1/4 < a ≤ 0
theorem f_decreasing_when_neg_quarter_lt_a_leq_zero (a : ℝ) (h1 : -(1/4) < a) (h2 : a ≤ 0) :
  ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y :=
sorry

-- Theorem for monotonicity when a > 0
theorem f_monotonicity_when_a_gt_zero (a : ℝ) (h : a > 0) :
  let x0 := (-1 + Real.sqrt (1 + 4*a)) / 2
  ∀ (x y : ℝ), 0 < x → x < y → y < x0 → f a x < f a y ∧
  ∀ (x y : ℝ), x0 < x → x < y → f a x > f a y :=
sorry

end

end NUMINAMATH_CALUDE_f_max_value_when_a_2_f_no_min_value_when_a_2_f_decreasing_when_a_leq_neg_quarter_f_decreasing_when_neg_quarter_lt_a_leq_zero_f_monotonicity_when_a_gt_zero_l1139_113965


namespace NUMINAMATH_CALUDE_fraction_simplification_l1139_113906

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1139_113906


namespace NUMINAMATH_CALUDE_sin_cos_equation_l1139_113954

theorem sin_cos_equation (x : Real) (p q : Nat) 
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9/4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p - Real.sqrt q)
  (h3 : 0 < p) (h4 : 0 < q) : p + q = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equation_l1139_113954


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l1139_113939

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighth_term_of_sequence (a₁ d : ℝ) :
  arithmeticSequence a₁ d 4 = 22 →
  arithmeticSequence a₁ d 6 = 46 →
  arithmeticSequence a₁ d 8 = 70 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l1139_113939


namespace NUMINAMATH_CALUDE_min_value_expression_l1139_113904

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x^2 + y^2 + 1/x^2 + 2*y/x ≥ m) ∧
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x^2 + y^2 + 1/x^2 + 2*y/x = m) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1139_113904


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1139_113952

def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {2, 5, 8}

theorem intersection_of_M_and_N : M ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1139_113952


namespace NUMINAMATH_CALUDE_prob_A_value_l1139_113953

/-- The probability of producing a grade B product -/
def prob_B : ℝ := 0.05

/-- The probability of producing a grade C product -/
def prob_C : ℝ := 0.03

/-- The probability of a randomly inspected product being grade A (non-defective) -/
def prob_A : ℝ := 1 - prob_B - prob_C

theorem prob_A_value : prob_A = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_value_l1139_113953


namespace NUMINAMATH_CALUDE_friendly_point_sum_l1139_113971

/-- Friendly point transformation in 2D plane -/
def friendly_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 - 1, -p.1 - 1)

/-- Sequence of friendly points -/
def friendly_sequence (start : ℝ × ℝ) : ℕ → ℝ × ℝ
| 0 => start
| n + 1 => friendly_point (friendly_sequence start n)

theorem friendly_point_sum (x y : ℝ) :
  friendly_sequence (x, y) 2022 = (-3, -2) →
  x + y = 3 :=
by sorry

end NUMINAMATH_CALUDE_friendly_point_sum_l1139_113971


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l1139_113955

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = 18593/32768 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l1139_113955


namespace NUMINAMATH_CALUDE_odd_gon_symmetry_axis_through_vertex_l1139_113919

/-- A (2k+1)-gon is a polygon with 2k+1 vertices, where k is a positive integer. -/
structure OddGon where
  k : ℕ+
  vertices : Fin (2 * k + 1) → ℝ × ℝ

/-- An axis of symmetry for a polygon -/
structure SymmetryAxis (P : OddGon) where
  line : ℝ × ℝ → Prop

/-- A vertex lies on a line -/
def vertex_on_line (P : OddGon) (axis : SymmetryAxis P) (v : Fin (2 * P.k + 1)) : Prop :=
  axis.line (P.vertices v)

/-- The theorem stating that the axis of symmetry of a (2k+1)-gon passes through one of its vertices -/
theorem odd_gon_symmetry_axis_through_vertex (P : OddGon) (axis : SymmetryAxis P) :
  ∃ v : Fin (2 * P.k + 1), vertex_on_line P axis v := by
  sorry

end NUMINAMATH_CALUDE_odd_gon_symmetry_axis_through_vertex_l1139_113919


namespace NUMINAMATH_CALUDE_mona_unique_players_l1139_113921

/-- The number of groups Mona joined -/
def num_groups : ℕ := 9

/-- The number of other players in each group -/
def players_per_group : ℕ := 4

/-- The number of repeat players in the first group with repeats -/
def repeat_players_group1 : ℕ := 2

/-- The number of repeat players in the second group with repeats -/
def repeat_players_group2 : ℕ := 1

/-- The total number of unique players Mona grouped with -/
def unique_players : ℕ := num_groups * players_per_group - (repeat_players_group1 + repeat_players_group2)

theorem mona_unique_players : unique_players = 33 := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l1139_113921


namespace NUMINAMATH_CALUDE_ellipse_equation_l1139_113935

/-- Given a circle and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (A B : ℝ × ℝ),
    -- Point (1, 1/2) is on the circle x^2 + y^2 = 1
    1^2 + (1/2)^2 = 1 ∧
    -- A and B are points on the circle
    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧
    -- Line AB passes through (1, 0) (focus) and (0, 2) (upper vertex)
    ∃ (m c : ℝ), (m * 1 + c = 0) ∧ (m * 0 + c = 2) ∧
    (m * A.1 + c = A.2) ∧ (m * B.1 + c = B.2) ∧
    -- Ellipse equation
    ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) →
  a^2 = 5 ∧ b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1139_113935


namespace NUMINAMATH_CALUDE_investment_period_l1139_113958

/-- Proves that given a sum of 7000 invested at 15% p.a. and 12% p.a., 
    if the difference in interest received is 420, then the investment period is 2 years. -/
theorem investment_period (principal : ℝ) (rate_high : ℝ) (rate_low : ℝ) (interest_diff : ℝ) :
  principal = 7000 →
  rate_high = 0.15 →
  rate_low = 0.12 →
  interest_diff = 420 →
  ∃ (years : ℝ), principal * rate_high * years - principal * rate_low * years = interest_diff ∧ years = 2 :=
by sorry

end NUMINAMATH_CALUDE_investment_period_l1139_113958


namespace NUMINAMATH_CALUDE_james_injury_timeline_l1139_113933

/-- The number of days it took for James's pain to subside -/
def pain_subsided_days : ℕ := 3

/-- The total number of days until James can lift heavy again -/
def total_days : ℕ := 39

/-- The number of additional days James waits after the injury is fully healed -/
def additional_waiting_days : ℕ := 3

/-- The number of days (3 weeks) James waits before lifting heavy -/
def heavy_lifting_wait_days : ℕ := 21

theorem james_injury_timeline : 
  pain_subsided_days * 5 + pain_subsided_days + additional_waiting_days + heavy_lifting_wait_days = total_days :=
by sorry

end NUMINAMATH_CALUDE_james_injury_timeline_l1139_113933


namespace NUMINAMATH_CALUDE_vector_problem_l1139_113992

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m - 2)

def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem vector_problem :
  (∀ m : ℝ, parallel (vector_a m) (vector_b m) → m = 3 ∨ m = -1) ∧
  (∀ m : ℝ, perpendicular (vector_a m) (vector_b m) →
    let a := vector_a m
    let b := vector_b m
    dot_product (a.1 + 2 * b.1, a.2 + 2 * b.2) (2 * a.1 - b.1, 2 * a.2 - b.2) = 20) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l1139_113992


namespace NUMINAMATH_CALUDE_infinitely_many_lcm_greater_than_ck_l1139_113988

theorem infinitely_many_lcm_greater_than_ck
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_positive : ∀ n, a n > 0)
  (c : ℝ)
  (h_c_bounds : 0 < c ∧ c < 3/2) :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ Nat.lcm (a k) (a (k + 1)) > ↑k * c :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_lcm_greater_than_ck_l1139_113988


namespace NUMINAMATH_CALUDE_part_I_part_II_part_III_l1139_113985

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^2 + (1 - 3*m) * x + 2*m - 1

-- Part I
theorem part_I (a : ℝ) :
  (a > 0 ∧ {x : ℝ | f 2 x ≤ 0} ⊆ Set.Ioo a (2*a + 1)) ↔ (1/4 ≤ a ∧ a < 1) :=
sorry

-- Part II
def solution_set (m : ℝ) : Set ℝ :=
  if m < 0 then Set.Iic 1 ∪ Set.Ici (2 - 1/m)
  else if m = 0 then Set.Iic 1
  else if 0 < m ∧ m < 1 then Set.Icc (2 - 1/m) 1
  else if m = 1 then {1}
  else Set.Icc 1 (2 - 1/m)

theorem part_II (m : ℝ) :
  {x : ℝ | f m x ≤ 0} = solution_set m :=
sorry

-- Part III
theorem part_III (m : ℝ) :
  (∃ x > 0, f m x > -3*m*x + m - 1) ↔ m > -1/2 :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_part_III_l1139_113985


namespace NUMINAMATH_CALUDE_honda_production_l1139_113981

/-- Honda car production problem -/
theorem honda_production (day_shift second_shift total : ℕ) : 
  day_shift = 4 * second_shift → 
  second_shift = 1100 → 
  total = day_shift + second_shift → 
  total = 5500 := by
  sorry

end NUMINAMATH_CALUDE_honda_production_l1139_113981


namespace NUMINAMATH_CALUDE_arrangement_count_l1139_113977

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of people in the Jia-Bing-Yi group
def group_size : ℕ := 3

-- Define the number of units (group + other individuals)
def num_units : ℕ := total_people - group_size + 1

-- Theorem statement
theorem arrangement_count : 
  (num_units.factorial * group_size.factorial * 2) - (num_units.factorial * 2) = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1139_113977


namespace NUMINAMATH_CALUDE_marble_probability_l1139_113964

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 150) (h2 : blue = 24) (h3 : red = 37) :
  let white := total - blue - red
  (red + white : ℚ) / total = 21 / 25 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l1139_113964


namespace NUMINAMATH_CALUDE_intersection_values_l1139_113978

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the condition for intersection at a single point
def single_intersection (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ p.2 = line k p.1

-- Theorem statement
theorem intersection_values :
  {k : ℝ | single_intersection k} = {-1, 1, -Real.sqrt 5 / 2, Real.sqrt 5 / 2} :=
sorry

end NUMINAMATH_CALUDE_intersection_values_l1139_113978


namespace NUMINAMATH_CALUDE_exhibition_spacing_l1139_113931

theorem exhibition_spacing (wall_width : ℕ) (painting_width : ℕ) (num_paintings : ℕ) :
  wall_width = 320 ∧ painting_width = 30 ∧ num_paintings = 6 →
  (wall_width - num_paintings * painting_width) / (num_paintings + 1) = 20 :=
by sorry

end NUMINAMATH_CALUDE_exhibition_spacing_l1139_113931


namespace NUMINAMATH_CALUDE_jeff_shelter_cats_l1139_113975

/-- The number of cats in Jeff's shelter after a week of changes --/
def final_cat_count (initial : ℕ) (monday_added : ℕ) (tuesday_added : ℕ) (people_adopting : ℕ) (cats_per_adoption : ℕ) : ℕ :=
  initial + monday_added + tuesday_added - people_adopting * cats_per_adoption

/-- Theorem stating that Jeff's shelter has 17 cats after the week's changes --/
theorem jeff_shelter_cats : 
  final_cat_count 20 2 1 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jeff_shelter_cats_l1139_113975


namespace NUMINAMATH_CALUDE_electric_sharpener_time_l1139_113966

/-- Proves that an electric pencil sharpener takes 20 seconds to sharpen one pencil -/
theorem electric_sharpener_time : ∀ (hand_crank_time electric_time : ℕ),
  hand_crank_time = 45 →
  (360 / hand_crank_time : ℚ) + 10 = 360 / electric_time →
  electric_time = 20 :=
by sorry

end NUMINAMATH_CALUDE_electric_sharpener_time_l1139_113966


namespace NUMINAMATH_CALUDE_house_c_to_a_ratio_l1139_113907

/-- Represents the real estate problem with Nigella's sales --/
structure RealEstateProblem where
  base_salary : ℝ
  commission_rate : ℝ
  houses_sold : ℕ
  total_earnings : ℝ
  house_a_cost : ℝ
  house_b_cost : ℝ
  house_c_cost : ℝ

/-- Theorem stating the ratio of House C's cost to House A's cost before subtracting $110,000 --/
theorem house_c_to_a_ratio (problem : RealEstateProblem)
  (h1 : problem.base_salary = 3000)
  (h2 : problem.commission_rate = 0.02)
  (h3 : problem.houses_sold = 3)
  (h4 : problem.total_earnings = 8000)
  (h5 : problem.house_b_cost = 3 * problem.house_a_cost)
  (h6 : problem.house_c_cost = problem.house_a_cost * 2 - 110000)
  (h7 : problem.house_a_cost = 60000) :
  (problem.house_c_cost + 110000) / problem.house_a_cost = 2 := by
  sorry


end NUMINAMATH_CALUDE_house_c_to_a_ratio_l1139_113907


namespace NUMINAMATH_CALUDE_noodles_given_to_william_l1139_113924

/-- The number of noodles Daniel initially had -/
def initial_noodles : ℕ := 66

/-- The number of noodles Daniel has now -/
def remaining_noodles : ℕ := 54

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℕ := initial_noodles - remaining_noodles

theorem noodles_given_to_william : noodles_given = 12 := by
  sorry

end NUMINAMATH_CALUDE_noodles_given_to_william_l1139_113924


namespace NUMINAMATH_CALUDE_power_function_value_l1139_113969

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 / 2 → f 9 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1139_113969


namespace NUMINAMATH_CALUDE_minimize_sample_variance_l1139_113980

/-- Given a sample of size 5 with specific conditions, prove that the sample variance is minimized when a₄ = a₅ = 2.5 -/
theorem minimize_sample_variance (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ = 2.5 → a₂ = 3.5 → a₃ = 4 → a₄ + a₅ = 5 →
  let sample_variance := (1 / 5 : ℝ) * ((a₁ - 3)^2 + (a₂ - 3)^2 + (a₃ - 3)^2 + (a₄ - 3)^2 + (a₅ - 3)^2)
  ∀ b₄ b₅ : ℝ, b₄ + b₅ = 5 → 
  let alt_variance := (1 / 5 : ℝ) * ((a₁ - 3)^2 + (a₂ - 3)^2 + (a₃ - 3)^2 + (b₄ - 3)^2 + (b₅ - 3)^2)
  sample_variance ≤ alt_variance → a₄ = 2.5 ∧ a₅ = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_minimize_sample_variance_l1139_113980


namespace NUMINAMATH_CALUDE_value_of_A_l1139_113997

-- Define the letter values as variables
variable (F L A G E : ℤ)

-- Define the given conditions
axiom G_value : G = 15
axiom FLAG_value : F + L + A + G = 50
axiom LEAF_value : L + E + A + F = 65
axiom FEEL_value : F + E + E + L = 58

-- Theorem to prove
theorem value_of_A : A = 37 := by
  sorry

end NUMINAMATH_CALUDE_value_of_A_l1139_113997


namespace NUMINAMATH_CALUDE_atheris_population_2080_l1139_113983

def population_growth (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * (4 ^ (years / 30))

theorem atheris_population_2080 :
  population_growth 250 80 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_atheris_population_2080_l1139_113983


namespace NUMINAMATH_CALUDE_orange_juice_division_l1139_113922

theorem orange_juice_division (total_pints : ℚ) (num_glasses : ℕ) 
  (h1 : total_pints = 153)
  (h2 : num_glasses = 5) :
  total_pints / num_glasses = 30.6 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_division_l1139_113922


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1139_113996

theorem quadratic_root_range (a b : ℝ) (h1 : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2 ∧ x^2 + a*x + 2*b - 2 = 0 ∧ y^2 + a*y + 2*b - 2 = 0) :
  1/2 < (b - 4) / (a - 1) ∧ (b - 4) / (a - 1) < 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1139_113996


namespace NUMINAMATH_CALUDE_distinct_colorings_l1139_113930

/-- The number of disks in the circle -/
def n : ℕ := 7

/-- The number of blue disks -/
def blue : ℕ := 3

/-- The number of red disks -/
def red : ℕ := 3

/-- The number of green disks -/
def green : ℕ := 1

/-- The total number of colorings without considering symmetries -/
def total_colorings : ℕ := (n.choose blue) * ((n - blue).choose red)

/-- The number of rotational symmetries of the circle -/
def symmetries : ℕ := n

/-- The theorem stating the number of distinct colorings -/
theorem distinct_colorings : 
  (total_colorings / symmetries : ℚ) = 20 := by sorry

end NUMINAMATH_CALUDE_distinct_colorings_l1139_113930


namespace NUMINAMATH_CALUDE_dart_board_probability_l1139_113987

/-- The probability of a dart landing within the center hexagon of a dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let center_area := (3 * Real.sqrt 3 / 2) * s^2
  let total_area := (3 * Real.sqrt 3 / 2) * (2*s)^2
  center_area / total_area = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_dart_board_probability_l1139_113987


namespace NUMINAMATH_CALUDE_exponent_division_l1139_113902

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^2 / x^8 = 1 / x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1139_113902


namespace NUMINAMATH_CALUDE_uncle_bruce_chocolate_cookies_l1139_113950

theorem uncle_bruce_chocolate_cookies (total_dough : ℝ) (chocolate_percentage : ℝ) (leftover_chocolate : ℝ) :
  total_dough = 36 ∧ 
  chocolate_percentage = 0.20 ∧ 
  leftover_chocolate = 4 →
  ∃ initial_chocolate : ℝ,
    initial_chocolate = 13 ∧
    chocolate_percentage * (total_dough + initial_chocolate - leftover_chocolate) = initial_chocolate - leftover_chocolate :=
by sorry

end NUMINAMATH_CALUDE_uncle_bruce_chocolate_cookies_l1139_113950


namespace NUMINAMATH_CALUDE_euler_line_parallel_iff_condition_l1139_113914

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- A line parallel to the side BC of the triangle -/
def ParallelToBC (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The condition for Euler line parallelism -/
def EulerLineParallelCondition (t : Triangle) : Prop :=
  2 * t.a^4 = (t.b^2 - t.c^2)^2 + (t.b^2 + t.c^2) * t.a^2

/-- Theorem: The Euler line is parallel to side BC if and only if the condition holds -/
theorem euler_line_parallel_iff_condition (t : Triangle) :
  EulerLine t = ParallelToBC t ↔ EulerLineParallelCondition t := by sorry

end NUMINAMATH_CALUDE_euler_line_parallel_iff_condition_l1139_113914


namespace NUMINAMATH_CALUDE_tan_11_25_degrees_l1139_113940

theorem tan_11_25_degrees :
  ∃ (a b c d : ℕ+), 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (Real.tan (11.25 * π / 180) = Real.sqrt (a : ℝ) - Real.sqrt (b : ℝ) + Real.sqrt (c : ℝ) - (d : ℝ)) ∧
    (a = 2 + 2) ∧ (b = 2) ∧ (c = 1) ∧ (d = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_11_25_degrees_l1139_113940


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1139_113923

theorem expand_and_simplify (x : ℝ) : 6 * (x - 3) * (x + 10) = 6 * x^2 + 42 * x - 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1139_113923


namespace NUMINAMATH_CALUDE_fayes_coloring_books_l1139_113915

theorem fayes_coloring_books : 
  ∀ (initial_books : ℕ), 
  (initial_books - 3 + 48 = 79) → initial_books = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_fayes_coloring_books_l1139_113915


namespace NUMINAMATH_CALUDE_f_behavior_at_infinity_l1139_113993

def f (x : ℝ) := -3 * x^4 + 4 * x^2 + 5

theorem f_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → f x < M) :=
sorry

end NUMINAMATH_CALUDE_f_behavior_at_infinity_l1139_113993


namespace NUMINAMATH_CALUDE_keith_digimon_pack_price_l1139_113946

/-- The price of each pack of Digimon cards -/
def digimon_pack_price (total_spent : ℚ) (baseball_deck_price : ℚ) (num_digimon_packs : ℕ) : ℚ :=
  (total_spent - baseball_deck_price) / num_digimon_packs

theorem keith_digimon_pack_price :
  digimon_pack_price 23.86 6.06 4 = 4.45 := by
  sorry

end NUMINAMATH_CALUDE_keith_digimon_pack_price_l1139_113946


namespace NUMINAMATH_CALUDE_strawberry_pies_l1139_113960

/-- The number of pies that can be made from strawberries picked by Christine and Rachel -/
def number_of_pies (christine_picked : ℕ) (rachel_factor : ℕ) (pounds_per_pie : ℕ) : ℕ :=
  (christine_picked + christine_picked * rachel_factor) / pounds_per_pie

/-- Theorem stating that Christine and Rachel can make 10 pies -/
theorem strawberry_pies :
  number_of_pies 10 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_pies_l1139_113960


namespace NUMINAMATH_CALUDE_luke_game_rounds_l1139_113948

theorem luke_game_rounds (points_per_round : ℕ) (total_points : ℕ) (h1 : points_per_round = 146) (h2 : total_points = 22922) :
  total_points / points_per_round = 157 := by
  sorry

end NUMINAMATH_CALUDE_luke_game_rounds_l1139_113948


namespace NUMINAMATH_CALUDE_fraction_equality_l1139_113909

theorem fraction_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1139_113909


namespace NUMINAMATH_CALUDE_leap_year_statistics_l1139_113979

def leap_year_data : List ℕ := sorry

def median_of_modes (data : List ℕ) : ℚ := sorry

def median (data : List ℕ) : ℚ := sorry

def mean (data : List ℕ) : ℚ := sorry

theorem leap_year_statistics :
  let d := median_of_modes leap_year_data
  let M := median leap_year_data
  let μ := mean leap_year_data
  d < M ∧ M < μ := by sorry

end NUMINAMATH_CALUDE_leap_year_statistics_l1139_113979


namespace NUMINAMATH_CALUDE_count_green_curlers_l1139_113961

/-- Given a total number of curlers and relationships between different types,
    prove the number of large green curlers. -/
theorem count_green_curlers (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ)
  (h1 : total = 16)
  (h2 : pink = total / 4)
  (h3 : blue = 2 * pink)
  (h4 : green = total - pink - blue) :
  green = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_green_curlers_l1139_113961


namespace NUMINAMATH_CALUDE_correct_answers_is_120_l1139_113925

/-- Represents an exam scoring system -/
structure ExamScoring where
  totalScore : Int
  totalQuestions : Nat
  correctScore : Int
  wrongPenalty : Int

/-- Calculates the number of correct answers in an exam -/
def calculateCorrectAnswers (exam : ExamScoring) : Int :=
  (exam.totalScore + 2 * exam.totalQuestions) / (exam.correctScore - exam.wrongPenalty)

/-- Theorem: Given the exam conditions, the number of correct answers is 120 -/
theorem correct_answers_is_120 (exam : ExamScoring) 
  (h1 : exam.totalScore = 420)
  (h2 : exam.totalQuestions = 150)
  (h3 : exam.correctScore = 4)
  (h4 : exam.wrongPenalty = 2) :
  calculateCorrectAnswers exam = 120 := by
  sorry

#eval calculateCorrectAnswers { totalScore := 420, totalQuestions := 150, correctScore := 4, wrongPenalty := 2 }

end NUMINAMATH_CALUDE_correct_answers_is_120_l1139_113925


namespace NUMINAMATH_CALUDE_remaining_red_balloons_l1139_113947

/-- The number of red balloons remaining after destruction --/
def remaining_balloons (fred_balloons sam_balloons destroyed_balloons : ℝ) : ℝ :=
  fred_balloons + sam_balloons - destroyed_balloons

/-- Theorem stating the number of remaining red balloons --/
theorem remaining_red_balloons :
  remaining_balloons 10.0 46.0 16.0 = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_remaining_red_balloons_l1139_113947


namespace NUMINAMATH_CALUDE_hundredth_digit_of_13_over_90_l1139_113942

theorem hundredth_digit_of_13_over_90 : 
  ∃ (d : ℕ), d = 4 ∧ 
  (∃ (a b : ℕ), (13 : ℚ) / 90 = a + (d : ℚ) / 10^100 + b / 10^101 ∧ 
                0 ≤ b ∧ b < 10) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_13_over_90_l1139_113942


namespace NUMINAMATH_CALUDE_fraction_sum_equals_cube_sum_l1139_113913

theorem fraction_sum_equals_cube_sum (x : ℝ) : 
  ((x - 1) * (x + 1)) / (x * (x - 1) + 1) + (2 * (0.5 - x)) / (x * (1 - x) - 1) = 
  ((x - 1) * (x + 1) / (x * (x - 1) + 1))^3 + (2 * (0.5 - x) / (x * (1 - x) - 1))^3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_cube_sum_l1139_113913


namespace NUMINAMATH_CALUDE_percentage_calculation_l1139_113918

theorem percentage_calculation (x : ℝ) (h : 0.25 * x = 1200) : 0.35 * x = 1680 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1139_113918


namespace NUMINAMATH_CALUDE_power_of_power_l1139_113912

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1139_113912


namespace NUMINAMATH_CALUDE_no_intersection_l1139_113917

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- Define what it means for two functions to intersect at a point
def intersect_at (f g : ℝ → ℝ) (x : ℝ) : Prop := f x = g x

-- Theorem statement
theorem no_intersection :
  ¬ ∃ x : ℝ, intersect_at f g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l1139_113917


namespace NUMINAMATH_CALUDE_joan_balloon_count_l1139_113944

/-- Given an initial count of balloons and a number of lost balloons, 
    calculate the final count of balloons. -/
def final_balloon_count (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that with 8 initial balloons and 2 lost balloons, 
    the final count is 6. -/
theorem joan_balloon_count : final_balloon_count 8 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloon_count_l1139_113944


namespace NUMINAMATH_CALUDE_basketball_games_count_l1139_113984

theorem basketball_games_count : ∃ (x : ℕ), 
  x > 0 ∧ 
  x ∣ 60 ∧ 
  (3 * x / 5 : ℚ) = ⌊(3 * x / 5 : ℚ)⌋ ∧
  (7 * (x + 10) / 12 : ℚ) = ⌊(7 * (x + 10) / 12 : ℚ)⌋ ∧
  (7 * (x + 10) / 12 : ℕ) = (3 * x / 5 : ℕ) + 5 ∧
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_count_l1139_113984


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1139_113920

theorem trigonometric_simplification (x z : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + z) ^ 2 - 2 * Real.sin x * Real.sin z * Real.sin (x + z) = Real.sin z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1139_113920


namespace NUMINAMATH_CALUDE_a_minus_b_and_c_linearly_dependent_l1139_113937

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

/-- e₁ and e₂ are not collinear -/
axiom not_collinear : ¬ ∃ (r : ℝ), e₁ = r • e₂

/-- Definition of vector a -/
def a : V := 2 • e₁ - e₂

/-- Definition of vector b -/
def b : V := e₁ + 2 • e₂

/-- Definition of vector c -/
def c : V := (1/2) • e₁ - (3/2) • e₂

/-- Theorem stating that (a - b) and c are linearly dependent -/
theorem a_minus_b_and_c_linearly_dependent :
  ∃ (r s : ℝ) (hs : s ≠ 0), r • (a e₁ e₂ - b e₁ e₂) + s • c e₁ e₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_and_c_linearly_dependent_l1139_113937


namespace NUMINAMATH_CALUDE_megan_finished_problems_l1139_113932

theorem megan_finished_problems (total_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) 
  (h1 : total_problems = 40)
  (h2 : remaining_pages = 2)
  (h3 : problems_per_page = 7) :
  total_problems - (remaining_pages * problems_per_page) = 26 := by
sorry

end NUMINAMATH_CALUDE_megan_finished_problems_l1139_113932


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_n_199999_satisfies_condition_n_199999_is_largest_l1139_113938

theorem largest_n_divisible_by_seven (n : ℕ) : 
  (n < 200000 ∧ 
   (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0) →
  n ≤ 199999 :=
by sorry

theorem n_199999_satisfies_condition : 
  (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 :=
by sorry

theorem n_199999_is_largest : 
  ∀ m : ℕ, m < 200000 ∧ 
  (8 * (m - 3)^5 - 2 * m^2 + 18 * m - 36) % 7 = 0 →
  m ≤ 199999 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_n_199999_satisfies_condition_n_199999_is_largest_l1139_113938


namespace NUMINAMATH_CALUDE_system_solution_proof_l1139_113974

theorem system_solution_proof :
  ∃ (x y z : ℝ),
    (1/x + 1/(y+z) = 6/5) ∧
    (1/y + 1/(x+z) = 3/4) ∧
    (1/z + 1/(x+y) = 2/3) ∧
    (x = 2) ∧ (y = 3) ∧ (z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l1139_113974


namespace NUMINAMATH_CALUDE_scalar_for_coplanarity_l1139_113916

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points as vectors
variable (O A B C D : V)

-- Define the scalar k
variable (k : ℝ)

-- Define the equation
def equation (O A B C D : V) (k : ℝ) : Prop :=
  2 • (A - O) - 3 • (B - O) + 7 • (C - O) + k • (D - O) = 0

-- Define coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

-- Theorem statement
theorem scalar_for_coplanarity (O A B C D : V) :
  ∃ (k : ℝ), equation O A B C D k ∧ coplanar A B C D ∧ k = -6 := by sorry

end NUMINAMATH_CALUDE_scalar_for_coplanarity_l1139_113916


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l1139_113956

theorem recurring_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / (b.val : ℚ) = 36 / 99 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 15 := by
sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l1139_113956


namespace NUMINAMATH_CALUDE_revolver_game_theorem_l1139_113967

/-- The probability that player A fires the bullet in the revolver game -/
def revolver_game_prob : ℚ :=
  let p : ℚ := 1/6  -- probability of firing on a single shot
  6/11

/-- The revolver game theorem -/
theorem revolver_game_theorem :
  let p : ℚ := 1/6  -- probability of firing on a single shot
  let q : ℚ := 1 - p  -- probability of not firing on a single shot
  revolver_game_prob = p / (1 - q^2) :=
by sorry

#eval revolver_game_prob

end NUMINAMATH_CALUDE_revolver_game_theorem_l1139_113967


namespace NUMINAMATH_CALUDE_eighteen_men_handshakes_l1139_113910

/-- The maximum number of handshakes among n men without cyclic handshakes -/
def maxHandshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The maximum number of handshakes among 18 men without cyclic handshakes is 153 -/
theorem eighteen_men_handshakes :
  maxHandshakes 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_men_handshakes_l1139_113910


namespace NUMINAMATH_CALUDE_value_of_expression_l1139_113926

theorem value_of_expression (x y : ℝ) 
  (h1 : x - 2*y = -5) 
  (h2 : x*y = -2) : 
  2*x^2*y - 4*x*y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l1139_113926


namespace NUMINAMATH_CALUDE_double_scientific_notation_doubling_2_4_times_10_to_8_l1139_113901

theorem double_scientific_notation (x : Real) (n : Nat) :
  2 * (x * (10 ^ n)) = (2 * x) * (10 ^ n) := by sorry

theorem doubling_2_4_times_10_to_8 :
  2 * (2.4 * (10 ^ 8)) = 4.8 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_double_scientific_notation_doubling_2_4_times_10_to_8_l1139_113901


namespace NUMINAMATH_CALUDE_average_students_is_fifty_l1139_113994

/-- Represents a teacher's teaching data over multiple years -/
structure TeacherData where
  total_years : Nat
  first_year_students : Nat
  total_students : Nat

/-- Calculates the average number of students taught per year, excluding the first year -/
def averageStudentsPerYear (data : TeacherData) : Nat :=
  (data.total_students - data.first_year_students) / (data.total_years - 1)

/-- Theorem stating that for the given conditions, the average number of students per year (excluding the first year) is 50 -/
theorem average_students_is_fifty :
  let data : TeacherData := {
    total_years := 10,
    first_year_students := 40,
    total_students := 490
  }
  averageStudentsPerYear data = 50 := by
  sorry

#eval averageStudentsPerYear {
  total_years := 10,
  first_year_students := 40,
  total_students := 490
}

end NUMINAMATH_CALUDE_average_students_is_fifty_l1139_113994


namespace NUMINAMATH_CALUDE_acme_vowel_soup_words_l1139_113963

/-- The number of different letters available -/
def num_letters : ℕ := 5

/-- The number of times each letter appears -/
def letter_count : ℕ := 5

/-- The length of words to be formed -/
def word_length : ℕ := 5

/-- The total number of words that can be formed -/
def total_words : ℕ := num_letters ^ word_length

theorem acme_vowel_soup_words : total_words = 3125 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_words_l1139_113963


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1139_113962

theorem fly_distance_from_ceiling (x y z : ℝ) : 
  x = 3 → y = 4 → (x^2 + y^2 + z^2 = 5^2) → z = 0 := by sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1139_113962


namespace NUMINAMATH_CALUDE_class_average_l1139_113990

theorem class_average (total_students : ℕ) (top_scorers : ℕ) (zero_scorers : ℕ) (top_score : ℕ) (rest_average : ℕ) 
  (h1 : total_students = 20)
  (h2 : top_scorers = 2)
  (h3 : zero_scorers = 3)
  (h4 : top_score = 100)
  (h5 : rest_average = 40) :
  (top_scorers * top_score + zero_scorers * 0 + (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 40 := by
  sorry

#check class_average

end NUMINAMATH_CALUDE_class_average_l1139_113990


namespace NUMINAMATH_CALUDE_nineteen_in_base_three_l1139_113943

theorem nineteen_in_base_three : 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_in_base_three_l1139_113943


namespace NUMINAMATH_CALUDE_routes_from_p_to_q_l1139_113970

/-- Represents a directed graph with vertices P, R, S, T, Q -/
structure Network where
  vertices : Finset Char
  edges : Finset (Char × Char)

/-- Counts the number of paths between two vertices in the network -/
def count_paths (n : Network) (start finish : Char) : ℕ :=
  sorry

/-- The specific network described in the problem -/
def problem_network : Network :=
  { vertices := {'P', 'R', 'S', 'T', 'Q'},
    edges := {('P', 'R'), ('P', 'S'), ('P', 'T'), ('R', 'T'), ('R', 'Q'), ('S', 'R'), ('S', 'T'), ('S', 'Q'), ('T', 'R'), ('T', 'S'), ('T', 'Q')} }

theorem routes_from_p_to_q (n : Network := problem_network) :
  count_paths n 'P' 'Q' = 16 :=
sorry

end NUMINAMATH_CALUDE_routes_from_p_to_q_l1139_113970


namespace NUMINAMATH_CALUDE_jack_king_queen_probability_l1139_113911

theorem jack_king_queen_probability : 
  let deck_size : ℕ := 52
  let jack_count : ℕ := 4
  let king_count : ℕ := 4
  let queen_count : ℕ := 4
  let prob_jack : ℚ := jack_count / deck_size
  let prob_king : ℚ := king_count / (deck_size - 1)
  let prob_queen : ℚ := queen_count / (deck_size - 2)
  prob_jack * prob_king * prob_queen = 8 / 16575 :=
by sorry

end NUMINAMATH_CALUDE_jack_king_queen_probability_l1139_113911


namespace NUMINAMATH_CALUDE_logistics_problem_l1139_113959

/-- Represents the freight rates and charges for a logistics company. -/
structure FreightData where
  rateA : ℝ  -- Freight rate for goods A
  rateB : ℝ  -- Freight rate for goods B
  totalCharge : ℝ  -- Total freight charge

/-- Calculates the quantities of goods A and B transported given freight data for two months. -/
def calculateQuantities (march : FreightData) (april : FreightData) : ℝ × ℝ :=
  sorry

/-- Theorem stating that given the specific freight data for March and April,
    the quantities of goods A and B transported are 100 tons and 140 tons respectively. -/
theorem logistics_problem (march : FreightData) (april : FreightData) 
  (h1 : march.rateA = 50)
  (h2 : march.rateB = 30)
  (h3 : march.totalCharge = 9500)
  (h4 : april.rateA = 70)  -- 50 * 1.4 = 70
  (h5 : april.rateB = 40)
  (h6 : april.totalCharge = 13000) :
  calculateQuantities march april = (100, 140) :=
sorry

end NUMINAMATH_CALUDE_logistics_problem_l1139_113959


namespace NUMINAMATH_CALUDE_constant_value_c_l1139_113941

theorem constant_value_c (b c : ℚ) :
  (∀ x : ℚ, (x + 3) * (x + b) = x^2 + c*x + 8) →
  c = 17/3 := by
sorry

end NUMINAMATH_CALUDE_constant_value_c_l1139_113941


namespace NUMINAMATH_CALUDE_school_dinner_theatre_attendance_l1139_113986

theorem school_dinner_theatre_attendance
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (child_ticket_price : ℕ)
  (adult_ticket_price : ℕ)
  (h1 : total_tickets = 225)
  (h2 : total_revenue = 1875)
  (h3 : child_ticket_price = 6)
  (h4 : adult_ticket_price = 9) :
  ∃ (children_tickets : ℕ) (adult_tickets : ℕ),
    children_tickets + adult_tickets = total_tickets ∧
    child_ticket_price * children_tickets + adult_ticket_price * adult_tickets = total_revenue ∧
    children_tickets = 50 := by
  sorry


end NUMINAMATH_CALUDE_school_dinner_theatre_attendance_l1139_113986


namespace NUMINAMATH_CALUDE_wire_cutting_l1139_113982

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 80 →
  ratio = 3 / 5 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 50 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1139_113982


namespace NUMINAMATH_CALUDE_probability_red_or_green_is_13_22_l1139_113934

/-- Represents the count of jelly beans for each color -/
structure JellyBeanCounts where
  orange : ℕ
  purple : ℕ
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting either a red or green jelly bean -/
def probability_red_or_green (counts : JellyBeanCounts) : ℚ :=
  (counts.red + counts.green : ℚ) / (counts.orange + counts.purple + counts.red + counts.green)

/-- Theorem stating the probability of selecting a red or green jelly bean -/
theorem probability_red_or_green_is_13_22 :
  let counts : JellyBeanCounts := ⟨4, 5, 6, 7⟩
  probability_red_or_green counts = 13 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_green_is_13_22_l1139_113934


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1139_113903

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal interest time : ℚ) 
  (h1 : principal = 800)
  (h2 : interest = 128)
  (h3 : time = 4)
  (h4 : simple_interest principal (4 : ℚ) time = interest) : 
  ∃ (rate : ℚ), rate = 4 ∧ simple_interest principal rate time = interest := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1139_113903


namespace NUMINAMATH_CALUDE_rectangle_area_l1139_113999

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1139_113999


namespace NUMINAMATH_CALUDE_S_infinite_l1139_113998

/-- The number of positive divisors of a natural number -/
def d (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which n/d(n) is an integer -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, n = k * d n}

/-- Theorem: The set S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l1139_113998


namespace NUMINAMATH_CALUDE_birds_on_fence_l1139_113951

/-- The number of birds that fly away -/
def birds_flown : ℝ := 8.0

/-- The number of birds left on the fence -/
def birds_left : ℕ := 4

/-- The initial number of birds on the fence -/
def initial_birds : ℝ := birds_flown + birds_left

theorem birds_on_fence : initial_birds = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1139_113951


namespace NUMINAMATH_CALUDE_mixture_alcohol_percentage_l1139_113945

/-- The percentage of alcohol in solution X -/
def alcohol_percent_X : ℝ := 15

/-- The percentage of alcohol in solution Y -/
def alcohol_percent_Y : ℝ := 45

/-- The initial volume of solution X in milliliters -/
def initial_volume_X : ℝ := 300

/-- The volume of solution Y to be added in milliliters -/
def volume_Y : ℝ := 150

/-- The desired percentage of alcohol in the final solution -/
def target_alcohol_percent : ℝ := 25

/-- Theorem stating that adding 150 mL of solution Y to 300 mL of solution X
    results in a solution with 25% alcohol by volume -/
theorem mixture_alcohol_percentage :
  let total_volume := initial_volume_X + volume_Y
  let total_alcohol := (alcohol_percent_X / 100) * initial_volume_X + (alcohol_percent_Y / 100) * volume_Y
  (total_alcohol / total_volume) * 100 = target_alcohol_percent := by
  sorry

end NUMINAMATH_CALUDE_mixture_alcohol_percentage_l1139_113945


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1139_113957

theorem stratified_sampling_theorem (teachers male_students female_students : ℕ) 
  (female_sample : ℕ) (n : ℕ) : 
  teachers = 160 → 
  male_students = 960 → 
  female_students = 800 → 
  female_sample = 80 → 
  (female_students : ℚ) / (teachers + male_students + female_students : ℚ) = 
    (female_sample : ℚ) / (n : ℚ) → 
  n = 192 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1139_113957


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1139_113900

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1139_113900


namespace NUMINAMATH_CALUDE_quadratic_roots_negative_reciprocals_l1139_113928

theorem quadratic_roots_negative_reciprocals (k : ℝ) : 
  (∃ α : ℝ, α ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 10*x + k = 0 ↔ (x = α ∨ x = -1/α))) →
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_negative_reciprocals_l1139_113928


namespace NUMINAMATH_CALUDE_f_satisfies_condition_l1139_113976

-- Define the function f
def f (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem f_satisfies_condition : ∀ x : ℝ, f x + f (2 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_condition_l1139_113976


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1139_113968

/-- The y-coordinate of the vertex of the parabola y = 2x^2 + 16x + 35 is 3 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := 2 * x^2 + 16 * x + 35
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1139_113968


namespace NUMINAMATH_CALUDE_trig_expression_equals_zero_l1139_113973

theorem trig_expression_equals_zero :
  Real.cos (π / 3) - Real.tan (π / 4) + (3 / 4) * (Real.tan (π / 6))^2 - Real.sin (π / 6) + (Real.cos (π / 6))^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_zero_l1139_113973


namespace NUMINAMATH_CALUDE_triangle_theorem_l1139_113929

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C) :
  t.C = π / 3 ∧ 
  (t.a = 5 → t.b = 8 → t.c = 7) := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_triangle_theorem_l1139_113929


namespace NUMINAMATH_CALUDE_paper_length_equals_days_until_due_l1139_113908

/-- The number of pages in Stacy's history paper -/
def paper_length : ℕ := sorry

/-- The number of days until the paper is due -/
def days_until_due : ℕ := 12

/-- The number of pages Stacy needs to write per day to finish on time -/
def pages_per_day : ℕ := 1

/-- Theorem stating that the paper length is equal to the number of days until due -/
theorem paper_length_equals_days_until_due : 
  paper_length = days_until_due := by sorry

end NUMINAMATH_CALUDE_paper_length_equals_days_until_due_l1139_113908


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1139_113905

-- Define the constraint set
def ConstraintSet : Set (ℝ × ℝ) :=
  {(x, y) | 8 * x - y ≤ 4 ∧ x + y ≥ -1 ∧ y ≤ 4 * x}

-- Define the objective function
def ObjectiveFunction (a b : ℝ) (p : ℝ × ℝ) : ℝ :=
  a * p.1 + b * p.2

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (p : ℝ × ℝ), p ∈ ConstraintSet ∧ 
   ∀ (q : ℝ × ℝ), q ∈ ConstraintSet → ObjectiveFunction a b q ≤ ObjectiveFunction a b p) →
  (∀ (p : ℝ × ℝ), p ∈ ConstraintSet → ObjectiveFunction a b p ≤ 2) →
  1/a + 1/b ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1139_113905


namespace NUMINAMATH_CALUDE_root_product_identity_l1139_113995

theorem root_product_identity (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_identity_l1139_113995


namespace NUMINAMATH_CALUDE_apples_bought_l1139_113991

theorem apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) (bought : ℕ) : 
  initial ≥ used →
  bought = final - (initial - used) := by
  sorry

end NUMINAMATH_CALUDE_apples_bought_l1139_113991


namespace NUMINAMATH_CALUDE_fiftieth_student_age_l1139_113949

theorem fiftieth_student_age
  (total_students : Nat)
  (average_age : ℝ)
  (group1_count : Nat)
  (group1_avg : ℝ)
  (group2_count : Nat)
  (group2_avg : ℝ)
  (group3_count : Nat)
  (group3_avg : ℝ)
  (group4_count : Nat)
  (group4_avg : ℝ)
  (h1 : total_students = 50)
  (h2 : average_age = 20)
  (h3 : group1_count = 15)
  (h4 : group1_avg = 18)
  (h5 : group2_count = 15)
  (h6 : group2_avg = 22)
  (h7 : group3_count = 10)
  (h8 : group3_avg = 25)
  (h9 : group4_count = 9)
  (h10 : group4_avg = 24)
  (h11 : group1_count + group2_count + group3_count + group4_count = total_students - 1) :
  (total_students : ℝ) * average_age - 
  (group1_count : ℝ) * group1_avg - 
  (group2_count : ℝ) * group2_avg - 
  (group3_count : ℝ) * group3_avg - 
  (group4_count : ℝ) * group4_avg = 66 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_student_age_l1139_113949


namespace NUMINAMATH_CALUDE_lcm_18_20_l1139_113927

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_20_l1139_113927
