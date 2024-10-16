import Mathlib

namespace NUMINAMATH_CALUDE_mod_eight_equivalence_l1907_190730

theorem mod_eight_equivalence : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -3737 [ZMOD 8] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_equivalence_l1907_190730


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1907_190756

theorem inequality_solution_set : 
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1907_190756


namespace NUMINAMATH_CALUDE_angle_decomposition_negative_495_decomposition_l1907_190732

theorem angle_decomposition (angle : ℤ) : ∃ (k : ℤ) (θ : ℤ), 
  angle = k * 360 + θ ∧ -180 < θ ∧ θ ≤ 180 :=
by sorry

theorem negative_495_decomposition : 
  ∃ (k : ℤ), -495 = k * 360 + (-135) ∧ -180 < -135 ∧ -135 ≤ 180 :=
by sorry

end NUMINAMATH_CALUDE_angle_decomposition_negative_495_decomposition_l1907_190732


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1907_190711

open Real

theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x ≥ 0, 2 * (exp x) - 2 * a * x - a^2 + 3 - x^2 ≥ 0) →
  a ∈ Set.Icc (-Real.sqrt 5) (3 - Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1907_190711


namespace NUMINAMATH_CALUDE_frog_walk_probability_l1907_190740

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the 6x6 grid -/
def Grid := {p : Point // p.x ≤ 6 ∧ p.y ≤ 6}

/-- Defines the possible jump directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines a random walk on the grid -/
def RandomWalk := List Direction

/-- Checks if a point is on the boundary of the grid -/
def isBoundary (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 6 ∨ p.y = 0 ∨ p.y = 6

/-- Checks if a point is on the top or bottom horizontal side of the grid -/
def isHorizontalSide (p : Point) : Bool :=
  p.y = 0 ∨ p.y = 6

/-- Calculates the probability of ending on a horizontal side -/
noncomputable def probabilityHorizontalSide (start : Point) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem frog_walk_probability :
  probabilityHorizontalSide ⟨2, 3⟩ = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_frog_walk_probability_l1907_190740


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1907_190729

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + 2

theorem tangent_line_equation :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := (Real.cos x₀ + Real.exp x₀)
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = 2 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1907_190729


namespace NUMINAMATH_CALUDE_lucy_speed_calculation_l1907_190750

-- Define the cycling speeds
def eugene_speed : ℚ := 5
def carlos_relative_speed : ℚ := 4/5
def lucy_relative_speed : ℚ := 6/7

-- Theorem to prove
theorem lucy_speed_calculation :
  let carlos_speed := eugene_speed * carlos_relative_speed
  let lucy_speed := carlos_speed * lucy_relative_speed
  lucy_speed = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_lucy_speed_calculation_l1907_190750


namespace NUMINAMATH_CALUDE_line_relations_l1907_190788

-- Define the concept of a line in 3D space
variable (Line : Type)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem line_relations (a b c : Line) :
  parallel a b → perpendicular a c → perpendicular b c := by
  sorry

end NUMINAMATH_CALUDE_line_relations_l1907_190788


namespace NUMINAMATH_CALUDE_line_equation_problem_l1907_190751

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem
theorem line_equation_problem (l : Line) (P : Point) :
  (P.x = 2 ∧ P.y = 3) →
  (
    (l.slope = -Real.sqrt 3) ∨
    (l.slope = -2) ∨
    (l.slope = 3/2 ∧ l.y_intercept = 0) ∨
    (l.slope = 1 ∧ l.y_intercept = -1)
  ) →
  (
    (Real.sqrt 3 * P.x + P.y - 3 - 2 * Real.sqrt 3 = 0) ∨
    (2 * P.x + P.y - 7 = 0) ∨
    (3 * P.x - 2 * P.y = 0) ∨
    (P.x - P.y + 1 = 0)
  ) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_problem_l1907_190751


namespace NUMINAMATH_CALUDE_missing_number_l1907_190738

theorem missing_number (x z : ℕ) 
  (h1 : x * 2 = 8)
  (h2 : 2 * z = 16)
  (h3 : 8 * 7 = 56)
  (h4 : 16 * 7 = 112) :
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_l1907_190738


namespace NUMINAMATH_CALUDE_count_pairs_eq_27_l1907_190719

open Set

def S : Finset Char := {'a', 'b', 'c'}

/-- The number of ordered pairs (A, B) of subsets of S such that A ∪ B = S and A ≠ B -/
def count_pairs : ℕ :=
  (Finset.powerset S).card * (Finset.powerset S).card -
  (Finset.powerset S).card

theorem count_pairs_eq_27 : count_pairs = 27 := by sorry

end NUMINAMATH_CALUDE_count_pairs_eq_27_l1907_190719


namespace NUMINAMATH_CALUDE_divisibility_condition_l1907_190795

theorem divisibility_condition (a b : ℕ+) :
  (a.val * b.val^2 + b.val + 7) ∣ (a.val^2 * b.val + a.val + b.val) ↔
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k.val^2 ∧ b = 7 * k.val) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1907_190795


namespace NUMINAMATH_CALUDE_unique_sequence_existence_l1907_190742

theorem unique_sequence_existence :
  ∃! (a : ℕ → ℕ), 
    a 1 = 1 ∧
    a 2 > 1 ∧
    ∀ n : ℕ, n ≥ 1 → 
      (a (n + 1) * (a (n + 1) - 1) : ℚ) = 
        (a n * a (n + 2) : ℚ) / ((a n * a (n + 2) - 1 : ℚ) ^ (1/3) + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_existence_l1907_190742


namespace NUMINAMATH_CALUDE_A_div_B_between_zero_and_one_l1907_190761

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem A_div_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end NUMINAMATH_CALUDE_A_div_B_between_zero_and_one_l1907_190761


namespace NUMINAMATH_CALUDE_linear_equation_properties_l1907_190722

/-- Given a linear equation x + 2y = -6, this theorem proves:
    1. y can be expressed as y = -3 - x/2
    2. y is a negative number greater than -2 if and only if -6 < x < -2
-/
theorem linear_equation_properties (x y : ℝ) (h : x + 2 * y = -6) :
  (y = -3 - x / 2) ∧
  (y < 0 ∧ y > -2 ↔ -6 < x ∧ x < -2) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_properties_l1907_190722


namespace NUMINAMATH_CALUDE_complement_of_A_l1907_190701

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1907_190701


namespace NUMINAMATH_CALUDE_function_properties_l1907_190715

def is_additive (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) = f x + f y

theorem function_properties (f : ℝ → ℝ) 
  (h_additive : is_additive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_f1 : f 1 = -2) :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (f (-3) = 6 ∧ f 3 = -6) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1907_190715


namespace NUMINAMATH_CALUDE_yellow_crane_tower_visitor_l1907_190790

structure Person :=
  (name : String)
  (visited : Bool)
  (statement : Bool)

def A : Person := { name := "A", visited := false, statement := false }
def B : Person := { name := "B", visited := false, statement := false }
def C : Person := { name := "C", visited := false, statement := false }

def people : List Person := [A, B, C]

theorem yellow_crane_tower_visitor :
  (∃! p : Person, p.visited = true) →
  (∃! p : Person, p.statement = false) →
  (A.statement = (¬C.visited)) →
  (B.statement = B.visited) →
  (C.statement = A.statement) →
  A.visited = true :=
by sorry

end NUMINAMATH_CALUDE_yellow_crane_tower_visitor_l1907_190790


namespace NUMINAMATH_CALUDE_bike_license_count_l1907_190764

/-- The number of possible letters for a bike license -/
def num_letters : ℕ := 3

/-- The number of digits in a bike license -/
def num_digits : ℕ := 4

/-- The number of possible digits for each position (0-9) -/
def digits_per_position : ℕ := 10

/-- The total number of possible bike licenses -/
def total_licenses : ℕ := num_letters * (digits_per_position ^ num_digits)

theorem bike_license_count : total_licenses = 30000 := by
  sorry

end NUMINAMATH_CALUDE_bike_license_count_l1907_190764


namespace NUMINAMATH_CALUDE_fraction_simplification_l1907_190721

theorem fraction_simplification (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) :
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1907_190721


namespace NUMINAMATH_CALUDE_lcm_18_24_l1907_190767

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1907_190767


namespace NUMINAMATH_CALUDE_bullet_evaluation_l1907_190726

-- Define the bullet operation
def bullet (a b : ℤ) : ℤ := 10 * a - b

-- State the theorem
theorem bullet_evaluation :
  bullet (bullet (bullet 2 0) 1) 3 = 1987 := by
  sorry

end NUMINAMATH_CALUDE_bullet_evaluation_l1907_190726


namespace NUMINAMATH_CALUDE_flute_cost_l1907_190731

/-- The cost of a flute given the total spent and costs of other items --/
theorem flute_cost (total_spent music_stand_cost song_book_cost : ℚ) :
  total_spent = 158.35 →
  music_stand_cost = 8.89 →
  song_book_cost = 7 →
  total_spent - (music_stand_cost + song_book_cost) = 142.46 := by
  sorry

end NUMINAMATH_CALUDE_flute_cost_l1907_190731


namespace NUMINAMATH_CALUDE_james_sticker_collection_l1907_190782

theorem james_sticker_collection (initial : ℕ) (gift : ℕ) (given_away : ℕ) 
  (h1 : initial = 478) 
  (h2 : gift = 182) 
  (h3 : given_away = 276) : 
  initial + gift - given_away = 384 := by
  sorry

end NUMINAMATH_CALUDE_james_sticker_collection_l1907_190782


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l1907_190780

theorem divisibility_of_sum_of_powers (a b : ℤ) (n : ℕ) :
  (a + b) ∣ (a^(2*n + 1) + b^(2*n + 1)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l1907_190780


namespace NUMINAMATH_CALUDE_total_distance_is_151_l1907_190739

/-- Calculates the total distance Amy biked in a week -/
def total_distance_biked : ℝ :=
  let monday_distance : ℝ := 12
  let tuesday_distance : ℝ := 2 * monday_distance - 3
  let wednesday_distance : ℝ := 2 * 11
  let thursday_distance : ℝ := wednesday_distance + 2
  let friday_distance : ℝ := thursday_distance + 2
  let saturday_distance : ℝ := friday_distance + 2
  let sunday_distance : ℝ := 3 * 6
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + 
  friday_distance + saturday_distance + sunday_distance

theorem total_distance_is_151 : total_distance_biked = 151 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_151_l1907_190739


namespace NUMINAMATH_CALUDE_earnings_left_is_90_3_percent_l1907_190777

/-- Represents the percentage of earnings spent on rent -/
def rent_percentage : ℝ := 0.40

/-- Represents the percentage of rent spent on dishwasher -/
def dishwasher_percentage : ℝ := 0.70

/-- Represents the percentage of rent spent on groceries -/
def groceries_percentage : ℝ := 1.15

/-- Represents the annual interest rate -/
def interest_rate : ℝ := 0.05

/-- Calculates the percentage of earnings left after one year given the spending pattern and interest rate -/
def earnings_left_after_one_year : ℝ :=
  let total_spent_percentage := rent_percentage + 
                                (dishwasher_percentage * rent_percentage) + 
                                (groceries_percentage * rent_percentage)
  let savings_percentage := 1 - total_spent_percentage
  savings_percentage * (1 + interest_rate)

/-- Theorem stating that the percentage of earnings left after one year is 90.3% -/
theorem earnings_left_is_90_3_percent :
  earnings_left_after_one_year = 0.903 := by sorry

end NUMINAMATH_CALUDE_earnings_left_is_90_3_percent_l1907_190777


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1907_190727

theorem fraction_equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (1 / (x - 2) = 3 / x) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1907_190727


namespace NUMINAMATH_CALUDE_eighth_root_of_3906250000000001_l1907_190759

theorem eighth_root_of_3906250000000001 :
  let n : ℕ := 3906250000000001
  ∃ (m : ℕ), m ^ 8 = n ∧ m = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_3906250000000001_l1907_190759


namespace NUMINAMATH_CALUDE_new_boarders_count_l1907_190776

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day_scholars : ℕ) 
  (final_ratio_boarders : ℕ) (final_ratio_day_scholars : ℕ) :
  initial_boarders = 560 →
  initial_ratio_boarders = 7 →
  initial_ratio_day_scholars = 16 →
  final_ratio_boarders = 1 →
  final_ratio_day_scholars = 2 →
  ∃ (new_boarders : ℕ), 
    new_boarders = 80 ∧
    (initial_boarders + new_boarders) * final_ratio_day_scholars = 
      (initial_boarders * initial_ratio_day_scholars / initial_ratio_boarders) * final_ratio_boarders :=
by
  sorry

end NUMINAMATH_CALUDE_new_boarders_count_l1907_190776


namespace NUMINAMATH_CALUDE_shortest_side_is_thirteen_l1907_190717

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The length of the first segment of the divided side -/
  segment1 : ℝ
  /-- The length of the second segment of the divided side -/
  segment2 : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Condition: radius is positive -/
  radius_pos : radius > 0
  /-- Condition: segments are positive -/
  segment1_pos : segment1 > 0
  segment2_pos : segment2 > 0
  /-- Condition: shortest side is positive -/
  shortest_side_pos : shortest_side > 0

/-- Theorem: The shortest side of the triangle is 13 units -/
theorem shortest_side_is_thirteen (t : TriangleWithInscribedCircle) 
    (h1 : t.radius = 4)
    (h2 : t.segment1 = 6)
    (h3 : t.segment2 = 8) :
    t.shortest_side = 13 :=
  sorry


end NUMINAMATH_CALUDE_shortest_side_is_thirteen_l1907_190717


namespace NUMINAMATH_CALUDE_lines_can_coincide_by_rotation_l1907_190712

/-- Given two lines l₁ and l₂ in the xy-plane, prove that they can coincide
    by rotating l₂ around a point on l₁. -/
theorem lines_can_coincide_by_rotation (α c : ℝ) :
  ∃ (x₀ y₀ θ : ℝ), 
    (y₀ = x₀ * Real.sin α) ∧  -- Point (x₀, y₀) is on l₁
    (∀ x y : ℝ,
      y = 2*x + c →  -- Original equation of l₂
      ∃ x' y' : ℝ,
        x' = (x - x₀) * Real.cos θ - (y - y₀) * Real.sin θ + x₀ ∧
        y' = (x - x₀) * Real.sin θ + (y - y₀) * Real.cos θ + y₀ ∧
        y' = x' * Real.sin α) -- Rotated l₂ coincides with l₁
  := by sorry

end NUMINAMATH_CALUDE_lines_can_coincide_by_rotation_l1907_190712


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1907_190708

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 16/3 ∧ D = 5/3 ∧
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 →
    (7*x - 4) / (x^2 - 9*x - 36) = C / (x - 12) + D / (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1907_190708


namespace NUMINAMATH_CALUDE_second_interest_rate_is_ten_percent_l1907_190769

/-- Proves that given specific investment conditions, the second interest rate is 10% -/
theorem second_interest_rate_is_ten_percent 
  (total_investment : ℝ)
  (first_investment : ℝ)
  (first_rate : ℝ)
  (h_total : total_investment = 5400)
  (h_first : first_investment = 3000)
  (h_first_rate : first_rate = 0.08)
  (h_equal_interest : first_investment * first_rate = 
    (total_investment - first_investment) * (10 / 100)) :
  (10 : ℝ) / 100 = (first_investment * first_rate) / (total_investment - first_investment) :=
sorry

end NUMINAMATH_CALUDE_second_interest_rate_is_ten_percent_l1907_190769


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1907_190705

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1907_190705


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l1907_190724

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (second_speed : ℝ)
  (average_speed : ℝ)
  (h1 : initial_distance = 18)
  (h2 : initial_speed = 36)
  (h3 : second_speed = 60)
  (h4 : average_speed = 45)
  : ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = average_speed
    ∧ additional_distance = 18 := by
  sorry


end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l1907_190724


namespace NUMINAMATH_CALUDE_min_value_of_f_l1907_190723

open Real

/-- The minimum value of f(x) = (e^x - a)^2 + (e^{-x} - a)^2 for 0 < a < 2 is 2(a - 1)^2 -/
theorem min_value_of_f (a : ℝ) (ha : 0 < a) (ha' : a < 2) :
  (∀ x : ℝ, (exp x - a)^2 + (exp (-x) - a)^2 ≥ 2 * (a - 1)^2) ∧
  (∃ x : ℝ, (exp x - a)^2 + (exp (-x) - a)^2 = 2 * (a - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1907_190723


namespace NUMINAMATH_CALUDE_sequence_sum_100_l1907_190773

/-- Sequence sum type -/
def SequenceSum (a : ℕ+ → ℝ) : ℕ+ → ℝ 
  | n => (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Main theorem -/
theorem sequence_sum_100 (a : ℕ+ → ℝ) (t : ℝ) : 
  (∀ n : ℕ+, a n > 0) → 
  a 1 = 1 → 
  (∀ n : ℕ+, 2 * SequenceSum a n = a n * (a n + t)) → 
  SequenceSum a 100 = 5050 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_100_l1907_190773


namespace NUMINAMATH_CALUDE_sequence_inequality_l1907_190741

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

/-- The main theorem -/
theorem sequence_inequality (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (heq : a 11 = b 10) : 
  a 13 + a 9 ≤ b 14 + b 6 :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1907_190741


namespace NUMINAMATH_CALUDE_parabola_points_product_l1907_190771

/-- Two distinct points on a parabola with opposite slopes to a fixed point -/
structure ParabolaPoints where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  distinct : (x₁, y₁) ≠ (x₂, y₂)
  on_parabola₁ : y₁^2 = x₁
  on_parabola₂ : y₂^2 = x₂
  same_side : y₁ * y₂ > 0
  opposite_slopes : (y₁ / (x₁ - 1)) = -(y₂ / (x₂ - 1))

/-- The product of y-coordinates equals 1 -/
theorem parabola_points_product (p : ParabolaPoints) : p.y₁ * p.y₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_product_l1907_190771


namespace NUMINAMATH_CALUDE_circle_area_difference_l1907_190799

theorem circle_area_difference (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 21) (h₂ : r₂ = 31) :
  (r₃ ^ 2 = r₂ ^ 2 - r₁ ^ 2) → r₃ = 2 * Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1907_190799


namespace NUMINAMATH_CALUDE_complex_square_equals_negative_100_minus_64i_l1907_190744

theorem complex_square_equals_negative_100_minus_64i :
  ∀ z : ℂ, z^2 = -100 - 64*I ↔ z = 4 - 8*I ∨ z = -4 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equals_negative_100_minus_64i_l1907_190744


namespace NUMINAMATH_CALUDE_cookies_per_pack_l1907_190792

theorem cookies_per_pack (num_trays : ℕ) (cookies_per_tray : ℕ) (num_packs : ℕ) :
  num_trays = 4 →
  cookies_per_tray = 24 →
  num_packs = 8 →
  (num_trays * cookies_per_tray) / num_packs = 12 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_pack_l1907_190792


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l1907_190798

/-- The number of members in the water polo club -/
def total_members : ℕ := 18

/-- The number of players in the starting team -/
def team_size : ℕ := 8

/-- The number of field players (excluding captain and goalie) -/
def field_players : ℕ := 6

/-- Calculates the number of ways to choose the starting team -/
def choose_team : ℕ := total_members * (total_members - 1) * (Nat.choose (total_members - 2) field_players)

theorem water_polo_team_selection :
  choose_team = 2459528 :=
sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l1907_190798


namespace NUMINAMATH_CALUDE_min_value_product_l1907_190796

theorem min_value_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 8) :
  (2 * a + 3 * b) * (2 * b + 3 * c) * (2 * c + 3 * a) ≥ 288 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l1907_190796


namespace NUMINAMATH_CALUDE_probability_x_squared_gt_one_l1907_190765

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Define the event (x^2 > 1)
def event (x : ℝ) : Prop := x^2 > 1

-- Define the measure of the interval
def intervalMeasure : ℝ := 4

-- Define the measure of the event within the interval
def eventMeasure : ℝ := 2

-- State the theorem
theorem probability_x_squared_gt_one :
  (eventMeasure / intervalMeasure : ℝ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_x_squared_gt_one_l1907_190765


namespace NUMINAMATH_CALUDE_total_fish_caught_l1907_190709

-- Define the types of fish
inductive FishType
| Trout
| Salmon
| Tuna

-- Define a function to calculate the pounds of fish caught for each type
def poundsCaught (fishType : FishType) : ℕ :=
  match fishType with
  | .Trout => 200
  | .Salmon => 200 + 200 / 2
  | .Tuna => 2 * (200 + 200 / 2)

-- Theorem statement
theorem total_fish_caught :
  (poundsCaught FishType.Trout) +
  (poundsCaught FishType.Salmon) +
  (poundsCaught FishType.Tuna) = 1100 := by
  sorry


end NUMINAMATH_CALUDE_total_fish_caught_l1907_190709


namespace NUMINAMATH_CALUDE_age_difference_l1907_190703

/-- Proves that the age difference between a man and his son is 26 years, given the specified conditions. -/
theorem age_difference (son_age man_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
  sorry

#check age_difference

end NUMINAMATH_CALUDE_age_difference_l1907_190703


namespace NUMINAMATH_CALUDE_rectangle_area_l1907_190713

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1907_190713


namespace NUMINAMATH_CALUDE_range_of_quadratic_l1907_190737

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The domain of the function -/
def domain : Set ℝ := Set.Ioc 1 4

/-- The range of the function on the given domain -/
def range : Set ℝ := f '' domain

theorem range_of_quadratic : range = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_range_of_quadratic_l1907_190737


namespace NUMINAMATH_CALUDE_not_perfect_square_zero_six_l1907_190785

/-- A number composed only of digits 0 and 6 -/
def DigitsZeroSix (m : ℕ) : Prop :=
  ∀ d, d ∈ m.digits 10 → d = 0 ∨ d = 6

/-- The sum of digits of a natural number -/
def DigitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem not_perfect_square_zero_six (m : ℕ) (h : DigitsZeroSix m) : 
  ¬∃ k : ℕ, m = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_zero_six_l1907_190785


namespace NUMINAMATH_CALUDE_injury_healing_ratio_l1907_190702

/-- The number of days it takes for the pain to subside -/
def pain_subsided : ℕ := 3

/-- The number of days James waits after full healing before working out -/
def wait_before_workout : ℕ := 3

/-- The number of days James waits before lifting heavy -/
def wait_before_heavy : ℕ := 21

/-- The total number of days until James can lift heavy again -/
def total_days : ℕ := 39

/-- The number of days it takes for the injury to fully heal -/
def healing_time : ℕ := total_days - pain_subsided - wait_before_workout - wait_before_heavy

/-- The ratio of healing time to pain subsided time -/
def healing_ratio : ℚ := healing_time / pain_subsided

theorem injury_healing_ratio : healing_ratio = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_injury_healing_ratio_l1907_190702


namespace NUMINAMATH_CALUDE_f_strictly_increasing_and_symmetric_l1907_190736

def f (x : ℝ) : ℝ := x^(1/3)

theorem f_strictly_increasing_and_symmetric :
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_and_symmetric_l1907_190736


namespace NUMINAMATH_CALUDE_unique_vector_b_l1907_190762

def a : ℝ × ℝ := (-4, 3)
def c : ℝ × ℝ := (1, 1)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def acute_angle (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 > 0

theorem unique_vector_b :
  ∃! b : ℝ × ℝ,
    collinear b a ∧
    ‖b‖ = 10 ∧
    acute_angle b c ∧
    b = (8, -6) := by sorry

end NUMINAMATH_CALUDE_unique_vector_b_l1907_190762


namespace NUMINAMATH_CALUDE_twenty_four_game_4888_l1907_190783

/-- The "24 points" game with cards 4, 8, 8, 8 -/
theorem twenty_four_game_4888 :
  let a : ℕ := 4
  let b : ℕ := 8
  let c : ℕ := 8
  let d : ℕ := 8
  (a - (c / d)) * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_twenty_four_game_4888_l1907_190783


namespace NUMINAMATH_CALUDE_discount_effect_l1907_190725

theorem discount_effect (P N : ℝ) (h_pos_P : P > 0) (h_pos_N : N > 0) :
  let D : ℝ := 10
  let new_price : ℝ := (1 - D / 100) * P
  let new_quantity : ℝ := 1.25 * N
  let old_income : ℝ := P * N
  let new_income : ℝ := new_price * new_quantity
  (new_quantity / N = 1.25) ∧ (new_income / old_income = 1.125) :=
sorry

end NUMINAMATH_CALUDE_discount_effect_l1907_190725


namespace NUMINAMATH_CALUDE_sqrt_221_between_14_and_15_l1907_190775

theorem sqrt_221_between_14_and_15 : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_221_between_14_and_15_l1907_190775


namespace NUMINAMATH_CALUDE_division_problem_l1907_190786

theorem division_problem (divisor quotient remainder number : ℕ) : 
  divisor = 30 → 
  quotient = 9 → 
  remainder = 1 → 
  number = divisor * quotient + remainder → 
  number = 271 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1907_190786


namespace NUMINAMATH_CALUDE_limit_a_minus_log_n_eq_zero_l1907_190755

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => a n + Real.exp (-a n)

theorem limit_a_minus_log_n_eq_zero :
  ∃ L : ℝ, L = 0 ∧ ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - Real.log n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_a_minus_log_n_eq_zero_l1907_190755


namespace NUMINAMATH_CALUDE_cost_36_roses_l1907_190706

-- Define the proportionality factor
def prop_factor : ℚ := 30 / 18

-- Define the discount rate
def discount_rate : ℚ := 1 / 10

-- Define the discount threshold
def discount_threshold : ℕ := 30

-- Function to calculate the cost of a bouquet before discount
def cost_before_discount (roses : ℕ) : ℚ := prop_factor * roses

-- Function to apply discount if applicable
def apply_discount (cost : ℚ) (roses : ℕ) : ℚ :=
  if roses > discount_threshold then
    cost * (1 - discount_rate)
  else
    cost

-- Theorem stating the cost of 36 roses after discount
theorem cost_36_roses : apply_discount (cost_before_discount 36) 36 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cost_36_roses_l1907_190706


namespace NUMINAMATH_CALUDE_range_of_function_l1907_190733

theorem range_of_function (x : ℝ) (h : -π/2 ≤ x ∧ x ≤ π/2) :
  ∃ y, -Real.sqrt 3 ≤ y ∧ y ≤ 2 ∧ y = Real.sqrt 3 * Real.sin x + Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l1907_190733


namespace NUMINAMATH_CALUDE_volume_ratio_theorem_l1907_190758

/-- A right rectangular prism with edge lengths -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The set of points within distance r from any point in the prism -/
def S (B : RectangularPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The volume of S(r) -/
def volume_S (B : RectangularPrism) (r : ℝ) : ℝ :=
  sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

theorem volume_ratio_theorem (B : RectangularPrism) (coeff : VolumeCoefficients) :
    B.length = 2 ∧ B.width = 4 ∧ B.height = 6 →
    (∀ r : ℝ, volume_S B r = coeff.a * r^3 + coeff.b * r^2 + coeff.c * r + coeff.d) →
    coeff.b * coeff.c / (coeff.a * coeff.d) = 66 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_theorem_l1907_190758


namespace NUMINAMATH_CALUDE_carlas_order_cost_l1907_190772

/-- The original cost of Carla's order at McDonald's -/
def original_cost : ℝ := 7.50

/-- The coupon value -/
def coupon_value : ℝ := 2.50

/-- The senior discount percentage -/
def senior_discount : ℝ := 0.20

/-- The final amount Carla pays -/
def final_payment : ℝ := 4.00

/-- Theorem stating that the original cost is correct given the conditions -/
theorem carlas_order_cost :
  (original_cost - coupon_value) * (1 - senior_discount) = final_payment :=
by sorry

end NUMINAMATH_CALUDE_carlas_order_cost_l1907_190772


namespace NUMINAMATH_CALUDE_boat_transport_two_days_l1907_190781

/-- The number of people a boat can transport in multiple days -/
def boat_transport (capacity : ℕ) (trips_per_day : ℕ) (days : ℕ) : ℕ :=
  capacity * trips_per_day * days

/-- Theorem: A boat with capacity 12 making 4 trips per day can transport 96 people in 2 days -/
theorem boat_transport_two_days :
  boat_transport 12 4 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_boat_transport_two_days_l1907_190781


namespace NUMINAMATH_CALUDE_intersection_of_E_l1907_190766

def E (k : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ |p.1|^k ∧ |p.1| ≥ 1}

theorem intersection_of_E :
  (⋂ k ∈ Finset.range 1991, E (k + 1)) = {p : ℝ × ℝ | p.2 ≤ |p.1| ∧ |p.1| ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_E_l1907_190766


namespace NUMINAMATH_CALUDE_lassi_production_l1907_190710

theorem lassi_production (mangoes : ℕ) (lassis : ℕ) : 
  (3 * lassis = 13 * mangoes) → (15 * lassis = 65 * mangoes) :=
by sorry

end NUMINAMATH_CALUDE_lassi_production_l1907_190710


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1907_190760

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1907_190760


namespace NUMINAMATH_CALUDE_village_survival_time_l1907_190704

/-- The number of people a vampire drains per week -/
def vampire_drain_rate : ℕ := 3

/-- The number of people a werewolf eats per week -/
def werewolf_eat_rate : ℕ := 5

/-- The total number of people in the village -/
def village_population : ℕ := 72

/-- The number of weeks the village will last -/
def village_survival_weeks : ℕ := 9

/-- Theorem stating how long the village will last -/
theorem village_survival_time :
  village_population / (vampire_drain_rate + werewolf_eat_rate) = village_survival_weeks :=
by sorry

end NUMINAMATH_CALUDE_village_survival_time_l1907_190704


namespace NUMINAMATH_CALUDE_fraction_power_product_l1907_190757

theorem fraction_power_product : (3/4)^4 * (1/5) = 81/1280 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1907_190757


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_problem_l1907_190700

theorem absolute_value_sqrt_problem : |-2 * Real.sqrt 2| - Real.sqrt 4 * Real.sqrt 2 + (π - 5)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_problem_l1907_190700


namespace NUMINAMATH_CALUDE_max_cuboid_path_length_l1907_190743

noncomputable def max_path_length (a b c : ℝ) : ℝ :=
  4 * Real.sqrt (a^2 + b^2 + c^2) + 
  4 * Real.sqrt (max a b * max b c) + 
  min a (min b c) + 
  max a (max b c)

theorem max_cuboid_path_length :
  max_path_length 2 2 1 = 12 + 8 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_max_cuboid_path_length_l1907_190743


namespace NUMINAMATH_CALUDE_exact_three_blue_marbles_probability_l1907_190749

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_trials : ℕ := 6
def num_blue_selections : ℕ := 3

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exact_three_blue_marbles_probability :
  Nat.choose num_trials num_blue_selections *
  (prob_blue ^ num_blue_selections) *
  (prob_red ^ (num_trials - num_blue_selections)) =
  3512320 / 11390625 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_blue_marbles_probability_l1907_190749


namespace NUMINAMATH_CALUDE_treat_cost_theorem_l1907_190797

/-- Represents the cost of treats -/
structure TreatCost where
  chocolate : ℚ
  popsicle : ℚ
  lollipop : ℚ

/-- The cost relationships between treats -/
def cost_relationship (c : TreatCost) : Prop :=
  3 * c.chocolate = 2 * c.popsicle ∧ 2 * c.lollipop = 5 * c.chocolate

/-- The number of popsicles that can be bought with the money for 3 lollipops -/
def popsicles_for_lollipops (c : TreatCost) : ℚ :=
  (3 * c.lollipop) / c.popsicle

/-- The number of chocolates that can be bought with the money for 3 chocolates, 2 popsicles, and 2 lollipops -/
def chocolates_for_combination (c : TreatCost) : ℚ :=
  (3 * c.chocolate + 2 * c.popsicle + 2 * c.lollipop) / c.chocolate

theorem treat_cost_theorem (c : TreatCost) :
  cost_relationship c →
  popsicles_for_lollipops c = 5 ∧
  chocolates_for_combination c = 11 := by
  sorry

end NUMINAMATH_CALUDE_treat_cost_theorem_l1907_190797


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1907_190770

theorem pages_left_to_read (total_pages read_pages : ℕ) 
  (h1 : total_pages = 563)
  (h2 : read_pages = 147) :
  total_pages - read_pages = 416 := by
sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l1907_190770


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1907_190774

/-- Proves that given the conditions of a jogger and a train, the train's speed is 36 km/hr -/
theorem train_speed_calculation (jogger_speed : ℝ) (distance_ahead : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  distance_ahead = 240 →
  train_length = 130 →
  passing_time = 37 →
  ∃ (train_speed : ℝ), train_speed = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l1907_190774


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l1907_190791

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x₀ > 0, p x₀) ↔ ∀ x > 0, ¬(p x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x₀ > 0, 2^x₀ ≥ 3) ↔ ∀ x > 0, 2^x < 3 := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l1907_190791


namespace NUMINAMATH_CALUDE_inequality_theorem_l1907_190787

theorem inequality_theorem (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : x + a * b * y ≤ a * (y + z))
  (h2 : y + b * c * z ≤ b * (z + x))
  (h3 : z + c * a * x ≤ c * (x + y)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1907_190787


namespace NUMINAMATH_CALUDE_commute_time_is_120_minutes_l1907_190763

def minutes_in_hour : ℕ := 60

def rise_time : ℕ := 6 * 60  -- 6:00 a.m. in minutes
def leave_time : ℕ := 7 * 60  -- 7:00 a.m. in minutes
def return_time : ℕ := 17 * 60 + 30  -- 5:30 p.m. in minutes

def num_lectures : ℕ := 8
def lecture_duration : ℕ := 45
def lunch_duration : ℕ := 60
def library_duration : ℕ := 90

def total_time_away : ℕ := return_time - leave_time

def total_college_time : ℕ := num_lectures * lecture_duration + lunch_duration + library_duration

theorem commute_time_is_120_minutes :
  total_time_away - total_college_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_is_120_minutes_l1907_190763


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l1907_190714

/-- Given an ellipse with equation x²/16 + y²/9 = 1, 
    the length of the focal distance is 2√7 -/
theorem ellipse_focal_distance : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2/16 + y^2/9 = 1}
  ∃ c : ℝ, c = 2 * Real.sqrt 7 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ ellipse → 
      c = Real.sqrt ((x^2 + y^2) - 4 * Real.sqrt (x^2 * y^2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l1907_190714


namespace NUMINAMATH_CALUDE_hay_from_grass_l1907_190728

/-- The amount of hay obtained from freshly cut grass -/
theorem hay_from_grass (initial_mass : ℝ) (grass_moisture : ℝ) (hay_moisture : ℝ) : 
  initial_mass = 1000 →
  grass_moisture = 0.6 →
  hay_moisture = 0.15 →
  (initial_mass * (1 - grass_moisture)) / (1 - hay_moisture) = 470^10 / 17 := by
  sorry

#eval (470^10 : ℚ) / 17

end NUMINAMATH_CALUDE_hay_from_grass_l1907_190728


namespace NUMINAMATH_CALUDE_ratio_sum_to_base_l1907_190748

theorem ratio_sum_to_base (a b : ℚ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_base_l1907_190748


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1907_190778

/-- The number of yellow marbles in George's collection --/
def yellow_marbles : ℕ := 12

theorem yellow_marbles_count :
  let total_marbles : ℕ := 50
  let white_marbles : ℕ := total_marbles / 2
  let red_marbles : ℕ := 7
  let yellow_and_green : ℕ := total_marbles - white_marbles - red_marbles
  let green_marbles : ℕ := yellow_marbles / 2
  yellow_marbles + green_marbles = yellow_and_green ∧
  yellow_marbles > 0 ∧
  yellow_marbles = 12 :=
by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1907_190778


namespace NUMINAMATH_CALUDE_equation_represents_point_l1907_190734

theorem equation_represents_point (x y a b : ℝ) : 
  (x - a)^2 + (y + b)^2 = 0 ↔ x = a ∧ y = -b := by
sorry

end NUMINAMATH_CALUDE_equation_represents_point_l1907_190734


namespace NUMINAMATH_CALUDE_inequality_proof_l1907_190746

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : c > 1) : 
  a * b^c > b * a^c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1907_190746


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l1907_190779

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l1907_190779


namespace NUMINAMATH_CALUDE_scientific_notation_of_70819_l1907_190793

theorem scientific_notation_of_70819 : 
  70819 = 7.0819 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_70819_l1907_190793


namespace NUMINAMATH_CALUDE_hostel_provision_days_l1907_190754

/-- Calculates the initial number of days provisions were planned for in a hostel. -/
def initial_provision_days (initial_men : ℕ) (men_left : ℕ) (days_after_leaving : ℕ) : ℕ :=
  ((initial_men - men_left) * days_after_leaving) / initial_men

/-- Theorem stating that given the conditions, the initial provision days is 32. -/
theorem hostel_provision_days :
  initial_provision_days 250 50 40 = 32 := by
  sorry

#eval initial_provision_days 250 50 40

end NUMINAMATH_CALUDE_hostel_provision_days_l1907_190754


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l1907_190752

/-- An ellipse with equation 5x^2 - ky^2 = 5 and one focus at (0, 2) has k = -1 -/
theorem ellipse_focus_k_value (k : ℝ) :
  (∀ x y : ℝ, 5 * x^2 - k * y^2 = 5) →  -- Ellipse equation
  (∃ x : ℝ, (x, 2) ∈ {p : ℝ × ℝ | 5 * p.1^2 - k * p.2^2 = 5}) →  -- Focus at (0, 2)
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l1907_190752


namespace NUMINAMATH_CALUDE_lagrange_mvt_example_l1907_190745

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 6*x + 1

-- State the theorem
theorem lagrange_mvt_example :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 3,
    (f 3 - f (-1)) / (3 - (-1)) = 2*c + 6 :=
by
  sorry


end NUMINAMATH_CALUDE_lagrange_mvt_example_l1907_190745


namespace NUMINAMATH_CALUDE_probability_no_same_color_boxes_l1907_190784

/-- Represents a person with 4 colored blocks -/
structure Person :=
  (blocks : Fin 4 → Color)

/-- The four possible colors of blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | Black

/-- Represents a placement of blocks in boxes -/
def Placement := Fin 4 → Fin 4

/-- The probability space of all possible placements -/
def PlacementSpace := Person → Placement

/-- Checks if a box has blocks of all the same color -/
def hasSameColorBlocks (p : PlacementSpace) (box : Fin 4) : Prop :=
  ∃ c : Color, ∀ person : Person, (person.blocks ((p person) box)) = c

/-- The event where no box has blocks of all the same color -/
def NoSameColorBoxes (p : PlacementSpace) : Prop :=
  ∀ box : Fin 4, ¬(hasSameColorBlocks p box)

/-- The probability measure on the placement space -/
noncomputable def P : (PlacementSpace → Prop) → ℝ :=
  sorry

theorem probability_no_same_color_boxes :
  P NoSameColorBoxes = 14811 / 65536 :=
sorry

end NUMINAMATH_CALUDE_probability_no_same_color_boxes_l1907_190784


namespace NUMINAMATH_CALUDE_slower_painter_time_l1907_190707

-- Define the start time of the slower painter (2:00 PM)
def slower_start : ℝ := 14

-- Define the finish time (0.6 past midnight, which is 24.6)
def finish_time : ℝ := 24.6

-- Theorem to prove
theorem slower_painter_time :
  finish_time - slower_start = 10.6 := by
  sorry

end NUMINAMATH_CALUDE_slower_painter_time_l1907_190707


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l1907_190718

theorem modulus_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (3 - i) * (1 + 3*i)
  Complex.abs z = 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l1907_190718


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l1907_190720

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 6.2)
  (h2 : (c + d) / 2 = 6.1)
  (h3 : (e + f) / 2 = 6.9) :
  (a + b + c + d + e + f) / 6 = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l1907_190720


namespace NUMINAMATH_CALUDE_fraction_sum_l1907_190753

theorem fraction_sum : (3 : ℚ) / 9 + (5 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1907_190753


namespace NUMINAMATH_CALUDE_sum_of_medians_is_63_l1907_190768

/-- Represents the scores of a basketball player -/
def Scores := List ℕ

/-- Calculates the median of a list of scores -/
def median (scores : Scores) : ℚ :=
  sorry

/-- Player A's scores -/
def scoresA : Scores :=
  sorry

/-- Player B's scores -/
def scoresB : Scores :=
  sorry

/-- The sum of median scores of players A and B is 63 -/
theorem sum_of_medians_is_63 :
  median scoresA + median scoresB = 63 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_medians_is_63_l1907_190768


namespace NUMINAMATH_CALUDE_inequalities_proof_l1907_190747

theorem inequalities_proof (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : a + b > 0) :
  (a^2 > b^2) ∧ (1/a > 1/b) ∧ (a^2*b < b^3) ∧ ¬(∀ a b, a > 0 → 0 > b → a + b > 0 → a^3 < a*b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1907_190747


namespace NUMINAMATH_CALUDE_projectile_collision_time_l1907_190716

-- Define the parameters
def initial_distance : ℝ := 1386 -- km
def speed1 : ℝ := 445 -- km/h
def speed2 : ℝ := 545 -- km/h

-- Define the theorem
theorem projectile_collision_time :
  let relative_speed : ℝ := speed1 + speed2
  let time_hours : ℝ := initial_distance / relative_speed
  let time_minutes : ℝ := time_hours * 60
  ∃ ε > 0, |time_minutes - 84| < ε :=
sorry

end NUMINAMATH_CALUDE_projectile_collision_time_l1907_190716


namespace NUMINAMATH_CALUDE_definite_integrals_l1907_190789

theorem definite_integrals : 
  (∫ (x : ℝ) in (0)..(1), 2*x + 3) = 4 ∧ 
  (∫ (x : ℝ) in (Real.exp 1)..(Real.exp 3), 1/x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_definite_integrals_l1907_190789


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1907_190735

/-- 
Given a binomial expansion $(ax^m + bx^n)^{12}$ with specific conditions,
this theorem proves properties about the constant term and the range of $\frac{a}{b}$.
-/
theorem binomial_expansion_properties 
  (a b : ℝ) (m n : ℤ) 
  (ha : a > 0) (hb : b > 0) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : 2*m + n = 0) :
  (∃ (r : ℕ), r = 4 ∧ m*(12 - r) + n*r = 0) ∧ 
  (8/5 ≤ a/b ∧ a/b ≤ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1907_190735


namespace NUMINAMATH_CALUDE_simplify_fraction_l1907_190794

theorem simplify_fraction : (66 : ℚ) / 4356 = 1 / 66 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1907_190794
