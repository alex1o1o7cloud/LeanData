import Mathlib

namespace NUMINAMATH_CALUDE_fruit_box_arrangement_l594_59487

-- Define the fruits
inductive Fruit
  | Apple
  | Pear
  | Orange
  | Banana

-- Define a type for box numbers
inductive BoxNumber
  | One
  | Two
  | Three
  | Four

-- Define a function type for box labels
def BoxLabel := BoxNumber → Fruit

-- Define a function type for the actual content of boxes
def BoxContent := BoxNumber → Fruit

-- Define the property that all labels are incorrect
def AllLabelsIncorrect (label : BoxLabel) (content : BoxContent) : Prop :=
  ∀ b : BoxNumber, label b ≠ content b

-- Define the specific labels for each box
def SpecificLabels (label : BoxLabel) : Prop :=
  label BoxNumber.One = Fruit.Orange ∧
  label BoxNumber.Two = Fruit.Pear ∧
  (label BoxNumber.Three = Fruit.Apple ∨ label BoxNumber.Three = Fruit.Pear) ∧
  label BoxNumber.Four = Fruit.Apple

-- Define the conditional statement for Box 3
def Box3Condition (content : BoxContent) : Prop :=
  content BoxNumber.One = Fruit.Banana →
  (content BoxNumber.Three = Fruit.Apple ∨ content BoxNumber.Three = Fruit.Pear)

-- The main theorem
theorem fruit_box_arrangement :
  ∀ (label : BoxLabel) (content : BoxContent),
    AllLabelsIncorrect label content →
    SpecificLabels label →
    ¬Box3Condition content →
    content BoxNumber.One = Fruit.Banana ∧
    content BoxNumber.Two = Fruit.Apple ∧
    content BoxNumber.Three = Fruit.Orange ∧
    content BoxNumber.Four = Fruit.Pear :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_box_arrangement_l594_59487


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l594_59433

def total_figures : ℕ := 10
def triangle_count : ℕ := 3
def circle_count : ℕ := 3

theorem probability_triangle_or_circle :
  (triangle_count + circle_count : ℚ) / total_figures = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l594_59433


namespace NUMINAMATH_CALUDE_unique_integer_solution_l594_59400

theorem unique_integer_solution :
  ∃! (a b c : ℤ), a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l594_59400


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l594_59468

/-- The ratio of a man's age to his son's age in two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  let man_age := son_age + age_difference
  (man_age + 2) / (son_age + 2)

/-- Theorem: The ratio of the man's age to his son's age in two years is 2:1 -/
theorem age_ratio_is_two_to_one (son_age : ℕ) (age_difference : ℕ)
  (h1 : son_age = 23)
  (h2 : age_difference = 25) :
  age_ratio son_age age_difference = 2 := by
  sorry

#eval age_ratio 23 25

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l594_59468


namespace NUMINAMATH_CALUDE_rectangle_probability_in_n_gon_l594_59451

theorem rectangle_probability_in_n_gon (n : ℕ) (h1 : Even n) (h2 : n > 4) :
  let P := (3 : ℚ) / ((n - 1) * (n - 3))
  P = (Nat.choose (n / 2) 2 : ℚ) / (Nat.choose n 4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_probability_in_n_gon_l594_59451


namespace NUMINAMATH_CALUDE_ralph_peanuts_l594_59476

-- Define the initial number of peanuts
def initial_peanuts : ℕ := 74

-- Define the number of peanuts lost
def peanuts_lost : ℕ := 59

-- Theorem to prove
theorem ralph_peanuts : initial_peanuts - peanuts_lost = 15 := by
  sorry

end NUMINAMATH_CALUDE_ralph_peanuts_l594_59476


namespace NUMINAMATH_CALUDE_orchestra_members_count_l594_59437

theorem orchestra_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 4 ∧ 
  n % 9 = 6 ∧
  n = 212 := by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l594_59437


namespace NUMINAMATH_CALUDE_range_of_cos_2x_plus_2sin_x_l594_59439

open Real

theorem range_of_cos_2x_plus_2sin_x :
  ∀ x ∈ Set.Ioo 0 π,
  ∃ y ∈ Set.Icc 1 (3/2),
  y = cos (2*x) + 2 * sin x ∧
  ∀ z, z = cos (2*x) + 2 * sin x → z ∈ Set.Icc 1 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_cos_2x_plus_2sin_x_l594_59439


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l594_59446

/-- Alice's walking speed in miles per minute -/
def alice_speed : ℚ := 1 / 20

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Duration of travel in minutes -/
def duration : ℕ := 120

/-- The distance between Alice and Bob after the given duration -/
def distance_between (alice_speed bob_speed : ℚ) (duration : ℕ) : ℚ :=
  (alice_speed * duration) + (bob_speed * duration)

theorem distance_after_two_hours :
  distance_between alice_speed bob_speed duration = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l594_59446


namespace NUMINAMATH_CALUDE_sum_of_digits_11_pow_2003_l594_59484

theorem sum_of_digits_11_pow_2003 : ∃ n : ℕ, 
  11^2003 = 100 * n + 31 ∧ 3 + 1 = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_11_pow_2003_l594_59484


namespace NUMINAMATH_CALUDE_fraction_equality_l594_59470

theorem fraction_equality : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l594_59470


namespace NUMINAMATH_CALUDE_domain_of_f_sin_l594_59430

def is_in_domain (f : ℝ → ℝ) (x : ℝ) : Prop :=
  0 < x ∧ x ≤ 1

theorem domain_of_f_sin (f : ℝ → ℝ) :
  (∀ x, is_in_domain f x ↔ 0 < x ∧ x ≤ 1) →
  ∀ x, is_in_domain f (Real.sin x) ↔ ∃ k : ℤ, 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_sin_l594_59430


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l594_59418

theorem square_area_perimeter_ratio :
  ∀ s₁ s₂ : ℝ,
  s₁ > 0 → s₂ > 0 →
  s₁^2 / s₂^2 = 16 / 81 →
  (4 * s₁) / (4 * s₂) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l594_59418


namespace NUMINAMATH_CALUDE_cosine_sine_sum_l594_59488

theorem cosine_sine_sum (α : ℝ) : 
  (Real.cos (2 * α)) / (Real.sin (α - π/4)) = -Real.sqrt 2 / 2 → 
  Real.cos α + Real.sin α = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_l594_59488


namespace NUMINAMATH_CALUDE_stepped_design_reduces_blind_spots_l594_59441

/-- Represents a hall design --/
structure HallDesign where
  shape : String
  is_stepped : Bool

/-- Represents the visibility in a hall --/
structure Visibility where
  blind_spots : ℕ

/-- A function that calculates visibility based on hall design --/
def calculate_visibility (design : HallDesign) : Visibility :=
  sorry

/-- The theorem stating that a stepped design reduces blind spots --/
theorem stepped_design_reduces_blind_spots 
  (flat_design stepped_design : HallDesign)
  (h1 : flat_design.shape = "flat")
  (h2 : flat_design.is_stepped = false)
  (h3 : stepped_design.shape = "stepped")
  (h4 : stepped_design.is_stepped = true) :
  (calculate_visibility stepped_design).blind_spots < (calculate_visibility flat_design).blind_spots :=
sorry

end NUMINAMATH_CALUDE_stepped_design_reduces_blind_spots_l594_59441


namespace NUMINAMATH_CALUDE_octal_sum_units_digit_l594_59477

/-- The units digit of the sum of two octal numbers -/
def octal_units_digit_sum (a b : ℕ) : ℕ :=
  (a % 8 + b % 8) % 8

/-- Theorem: The units digit of 53₈ + 64₈ in base 8 is 7 -/
theorem octal_sum_units_digit :
  octal_units_digit_sum 53 64 = 7 := by
  sorry

end NUMINAMATH_CALUDE_octal_sum_units_digit_l594_59477


namespace NUMINAMATH_CALUDE_impossible_to_turn_all_off_l594_59461

/-- Represents the state of a lightning bug (on or off) -/
inductive BugState
| On
| Off

/-- Represents a 6x6 grid of lightning bugs -/
def Grid := Fin 6 → Fin 6 → BugState

/-- Represents a move on the grid -/
inductive Move
| Horizontal (row : Fin 6) (start_col : Fin 6)
| Vertical (col : Fin 6) (start_row : Fin 6)

/-- Applies a move to a grid -/
def applyMove (grid : Grid) (move : Move) : Grid :=
  sorry

/-- Checks if all bugs in the grid are off -/
def allOff (grid : Grid) : Prop :=
  ∀ (row col : Fin 6), grid row col = BugState.Off

/-- Initial grid configuration with one bug on -/
def initialGrid : Grid :=
  sorry

/-- Theorem stating the impossibility of turning all bugs off -/
theorem impossible_to_turn_all_off :
  ¬∃ (moves : List Move), allOff (moves.foldl applyMove initialGrid) :=
  sorry

end NUMINAMATH_CALUDE_impossible_to_turn_all_off_l594_59461


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l594_59492

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l594_59492


namespace NUMINAMATH_CALUDE_derivative_sin_plus_exp_cos_l594_59436

theorem derivative_sin_plus_exp_cos (x : ℝ) :
  let y : ℝ → ℝ := λ x => Real.sin x + Real.exp x * Real.cos x
  deriv y x = (1 + Real.exp x) * Real.cos x - Real.exp x * Real.sin x :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_plus_exp_cos_l594_59436


namespace NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_real_number_implication_l594_59480

theorem negation_of_implication_for_all (P : ℝ → ℝ → Prop) :
  (¬ ∀ a b : ℝ, P a b) ↔ ∃ a b : ℝ, ¬(P a b) :=
sorry

theorem negation_of_real_number_implication :
  (¬ ∀ a b : ℝ, a = 0 → a * b = 0) ↔ ∃ a b : ℝ, a = 0 ∧ a * b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_real_number_implication_l594_59480


namespace NUMINAMATH_CALUDE_power_of_power_l594_59419

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l594_59419


namespace NUMINAMATH_CALUDE_marcos_dads_strawberries_strawberry_problem_l594_59404

theorem marcos_dads_strawberries (initial_total : ℕ) (dads_extra : ℕ) (marcos_final : ℕ) : ℕ :=
  let dads_initial := initial_total - (marcos_final - dads_extra)
  dads_initial + dads_extra

theorem strawberry_problem : 
  marcos_dads_strawberries 22 30 36 = 46 := by
  sorry

end NUMINAMATH_CALUDE_marcos_dads_strawberries_strawberry_problem_l594_59404


namespace NUMINAMATH_CALUDE_tea_bags_in_box_l594_59416

theorem tea_bags_in_box (cups_per_bag_min cups_per_bag_max : ℕ) 
                        (natasha_cups inna_cups : ℕ) : 
  cups_per_bag_min = 2 →
  cups_per_bag_max = 3 →
  natasha_cups = 41 →
  inna_cups = 58 →
  ∃ n : ℕ, 
    n * cups_per_bag_min ≤ natasha_cups ∧ 
    natasha_cups ≤ n * cups_per_bag_max ∧
    n * cups_per_bag_min ≤ inna_cups ∧ 
    inna_cups ≤ n * cups_per_bag_max ∧
    n = 20 := by
  sorry

end NUMINAMATH_CALUDE_tea_bags_in_box_l594_59416


namespace NUMINAMATH_CALUDE_frank_bakes_two_trays_per_day_l594_59482

/-- The number of days Frank bakes cookies -/
def days : ℕ := 6

/-- The number of cookies Frank eats per day -/
def frankEatsPerDay : ℕ := 1

/-- The number of cookies Ted eats on the sixth day -/
def tedEats : ℕ := 4

/-- The number of cookies each tray makes -/
def cookiesPerTray : ℕ := 12

/-- The number of cookies left when Ted leaves -/
def cookiesLeft : ℕ := 134

/-- The number of trays Frank bakes per day -/
def traysPerDay : ℕ := 2

theorem frank_bakes_two_trays_per_day :
  traysPerDay * cookiesPerTray * days - 
  (frankEatsPerDay * days + tedEats) = cookiesLeft := by
  sorry

end NUMINAMATH_CALUDE_frank_bakes_two_trays_per_day_l594_59482


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l594_59440

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular_planes α β)
  (h2 : intersection α β = m)
  (h3 : subset n α) :
  perpendicular_line_plane n β ↔ perpendicular_lines n m :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l594_59440


namespace NUMINAMATH_CALUDE_largest_power_of_five_in_sum_of_factorials_l594_59452

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 98 + factorial 99 + factorial 100

theorem largest_power_of_five_in_sum_of_factorials :
  (∃ k : ℕ, sum_of_factorials = 5^26 * k ∧ ¬∃ m : ℕ, sum_of_factorials = 5^27 * m) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_five_in_sum_of_factorials_l594_59452


namespace NUMINAMATH_CALUDE_min_sum_fraction_min_sum_fraction_achievable_l594_59458

theorem min_sum_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 1 / (2 * Real.rpow 2 (1/3)) :=
sorry

theorem min_sum_fraction_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 1 / (2 * Real.rpow 2 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_fraction_min_sum_fraction_achievable_l594_59458


namespace NUMINAMATH_CALUDE_vikas_questions_l594_59497

/-- Prove that given a total of 24 questions submitted in the ratio 7 : 3 : 2,
    the number of questions submitted by the person corresponding to the second part of the ratio is 6. -/
theorem vikas_questions (total : ℕ) (r v a : ℕ) : 
  total = 24 →
  r + v + a = total →
  r = 7 * (total / (7 + 3 + 2)) →
  v = 3 * (total / (7 + 3 + 2)) →
  a = 2 * (total / (7 + 3 + 2)) →
  v = 6 :=
by sorry

end NUMINAMATH_CALUDE_vikas_questions_l594_59497


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l594_59401

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 3 and a_2 + a_3 = 6, prove that a_7 = 64 -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 2 + a 3 = 6)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) :
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l594_59401


namespace NUMINAMATH_CALUDE_dropped_students_score_l594_59460

theorem dropped_students_score (initial_students : ℕ) (remaining_students : ℕ) 
  (initial_average : ℚ) (remaining_average : ℚ) 
  (h1 : initial_students = 30) 
  (h2 : remaining_students = 26) 
  (h3 : initial_average = 60.25) 
  (h4 : remaining_average = 63.75) :
  (initial_students : ℚ) * initial_average - 
  (remaining_students : ℚ) * remaining_average = 150 := by
  sorry


end NUMINAMATH_CALUDE_dropped_students_score_l594_59460


namespace NUMINAMATH_CALUDE_division_mistake_remainder_l594_59444

theorem division_mistake_remainder (d q r : ℕ) (h1 : d > 0) (h2 : 472 = d * q + r) (h3 : 427 = d * (q - 5) + r) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_mistake_remainder_l594_59444


namespace NUMINAMATH_CALUDE_grisha_has_winning_strategy_l594_59431

/-- Represents the state of the game board -/
def GameBoard := List Nat

/-- Represents a player's move -/
inductive Move
| Square : Nat → Move  -- Square the number at a given index
| Increment : Nat → Move  -- Increment the number at a given index

/-- Represents a player -/
inductive Player
| Grisha
| Gleb

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move with
  | Move.Square i => sorry
  | Move.Increment i => sorry

/-- Checks if any number on the board is divisible by 2023 -/
def hasDivisibleBy2023 (board : GameBoard) : Bool :=
  sorry

/-- Represents a game strategy -/
def Strategy := GameBoard → Move

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that Grisha has a winning strategy -/
theorem grisha_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy Player.Grisha strategy :=
sorry

end NUMINAMATH_CALUDE_grisha_has_winning_strategy_l594_59431


namespace NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l594_59490

theorem infinitely_many_primes_3_mod_4 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 4 = 3} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l594_59490


namespace NUMINAMATH_CALUDE_buying_combinations_l594_59427

theorem buying_combinations (n : ℕ) (items : ℕ) : 
  n = 4 → 
  items = 2 → 
  (items ^ n) - 1 = 15 :=
by sorry

end NUMINAMATH_CALUDE_buying_combinations_l594_59427


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l594_59493

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - 3*x > 0) ↔ (∃ x : ℝ, x^3 - 3*x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l594_59493


namespace NUMINAMATH_CALUDE_inequality_proof_l594_59445

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b - c) * (b + 1/c - a) + (b + 1/c - a) * (c + 1/a - b) + (c + 1/a - b) * (a + 1/b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l594_59445


namespace NUMINAMATH_CALUDE_tangent_slopes_reciprocal_implies_a_between_one_and_two_l594_59494

open Real

theorem tangent_slopes_reciprocal_implies_a_between_one_and_two 
  (f : ℝ → ℝ) (a : ℝ) (l₁ l₂ : ℝ → ℝ) :
  a ≠ 0 →
  (∀ x, f x = log x - a * (x - 1)) →
  (∃ x₁ y₁, l₁ 0 = 0 ∧ l₁ x₁ = y₁ ∧ y₁ = f x₁) →
  (∃ x₂ y₂, l₂ 0 = 0 ∧ l₂ x₂ = y₂ ∧ y₂ = exp x₂) →
  (∃ k₁ k₂, (∀ x, l₁ x = k₁ * x) ∧ (∀ x, l₂ x = k₂ * x) ∧ k₁ * k₂ = 1) →
  1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slopes_reciprocal_implies_a_between_one_and_two_l594_59494


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l594_59421

theorem coin_and_die_probability : 
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads
  let d : ℕ := 6   -- number of sides on the die
  let p_coin : ℚ := 1/2  -- probability of heads on a fair coin
  let p_die : ℚ := 1/d  -- probability of rolling a 6 on a fair die
  (Nat.choose n k * p_coin^k * (1 - p_coin)^(n - k)) * p_die = 55/6144 :=
by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l594_59421


namespace NUMINAMATH_CALUDE_not_equivalent_polar_points_l594_59456

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Check if two polar points are equivalent -/
def equivalentPolarPoints (p1 p2 : PolarPoint) : Prop :=
  p1.r = p2.r ∧ ∃ k : ℤ, p1.θ = p2.θ + 2 * k * Real.pi

theorem not_equivalent_polar_points :
  ¬ equivalentPolarPoints ⟨2, 11 * Real.pi / 6⟩ ⟨2, Real.pi / 6⟩ := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_polar_points_l594_59456


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l594_59402

theorem sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b, a > 0 → b > 0 → a * b < 3 → 1 / a + 4 / b > 2) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 1 / a + 4 / b > 2 ∧ a * b ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l594_59402


namespace NUMINAMATH_CALUDE_BC_time_is_3_hours_l594_59496

-- Define work rates for A, B, and C
def A_rate : ℚ := 1 / 4
def B_rate : ℚ := 1 / 12
def C_rate : ℚ := 1 / 4

-- Define the combined rate of B and C
def BC_rate : ℚ := B_rate + C_rate

-- Theorem statement
theorem BC_time_is_3_hours :
  (A_rate = 1 / 4) →
  (B_rate = 1 / 12) →
  (A_rate + C_rate = 1 / 2) →
  (1 / BC_rate = 3) := by
sorry

end NUMINAMATH_CALUDE_BC_time_is_3_hours_l594_59496


namespace NUMINAMATH_CALUDE_terms_before_five_l594_59417

/-- Given an arithmetic sequence starting with 75 and having a common difference of -5,
    this theorem proves that the number of terms that appear before 5 is 14. -/
theorem terms_before_five (a : ℕ → ℤ) :
  a 0 = 75 ∧ 
  (∀ n : ℕ, a (n + 1) - a n = -5) →
  (∃ k : ℕ, a k = 5 ∧ k = 15) ∧ 
  (∀ m : ℕ, m < 15 → a m > 5) :=
by sorry

end NUMINAMATH_CALUDE_terms_before_five_l594_59417


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l594_59447

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 9

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 6

/-- The current number of roses in the vase -/
def current_roses : ℕ := 3

/-- The current number of orchids in the vase -/
def current_orchids : ℕ := 13

/-- Theorem stating the difference between the current number of orchids and roses -/
theorem orchid_rose_difference :
  current_orchids - current_roses = 10 := by sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l594_59447


namespace NUMINAMATH_CALUDE_ball_distribution_l594_59438

/-- The number of ways to distribute n indistinguishable balls into k boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of ways to distribute balls into three boxes with minimum requirements -/
def distributeWithMinimum (total : ℕ) (min1 min2 min3 : ℕ) : ℕ :=
  distribute (total - min1 - min2 - min3) 3

theorem ball_distribution :
  distributeWithMinimum 20 1 2 3 = 120 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_l594_59438


namespace NUMINAMATH_CALUDE_product_of_powers_l594_59432

theorem product_of_powers (n : ℕ) (hn : n > 1) :
  (n + 1) * (n^2 + 1) * (n^4 + 1) * (n^8 + 1) * (n^16 + 1) = 
    if n = 2 then
      n^32 - 1
    else
      (n^32 - 1) / (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_product_of_powers_l594_59432


namespace NUMINAMATH_CALUDE_tank_filling_time_l594_59455

theorem tank_filling_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a = 60 → (15 / b + 15 * (1 / 60 + 1 / b) = 1) → b = 40 := by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l594_59455


namespace NUMINAMATH_CALUDE_definitely_rain_next_tuesday_is_false_l594_59498

-- Define a proposition representing the statement "It will definitely rain next Tuesday"
def definitely_rain_next_tuesday : Prop := True

-- Define a proposition representing the uncertainty of future events
def future_events_are_uncertain : Prop := True

-- Theorem stating that the original statement is false
theorem definitely_rain_next_tuesday_is_false : 
  future_events_are_uncertain → ¬definitely_rain_next_tuesday := by
  sorry

end NUMINAMATH_CALUDE_definitely_rain_next_tuesday_is_false_l594_59498


namespace NUMINAMATH_CALUDE_three_digit_sum_reverse_l594_59405

theorem three_digit_sum_reverse : ∃ (a b c : ℕ), 
  (a ≥ 1 ∧ a ≤ 9) ∧ 
  (b ≥ 0 ∧ b ≤ 9) ∧ 
  (c ≥ 0 ∧ c ≤ 9) ∧
  (100 * a + 10 * b + c) + (100 * c + 10 * b + a) = 1777 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_reverse_l594_59405


namespace NUMINAMATH_CALUDE_box_makers_l594_59495

/-- Represents the possible makers of the boxes -/
inductive Maker
| Cellini
| CelliniSon
| Bellini
| BelliniSon

/-- Represents a box with its inscription and actual maker -/
structure Box where
  color : String
  inscription : Prop
  maker : Maker

/-- The setup of the problem with two boxes -/
def boxSetup (goldBox silverBox : Box) : Prop :=
  (goldBox.color = "gold" ∧ silverBox.color = "silver") ∧
  (goldBox.inscription = (goldBox.maker = Maker.Cellini ∨ goldBox.maker = Maker.CelliniSon) ∧
                         (silverBox.maker = Maker.Cellini ∨ silverBox.maker = Maker.CelliniSon)) ∧
  (silverBox.inscription = (goldBox.maker ≠ Maker.CelliniSon ∧ goldBox.maker ≠ Maker.BelliniSon) ∧
                           (silverBox.maker ≠ Maker.CelliniSon ∧ silverBox.maker ≠ Maker.BelliniSon)) ∧
  (goldBox.inscription ≠ silverBox.inscription)

theorem box_makers (goldBox silverBox : Box) :
  boxSetup goldBox silverBox →
  (goldBox.maker = Maker.Cellini ∧ silverBox.maker = Maker.Bellini) :=
by
  sorry

end NUMINAMATH_CALUDE_box_makers_l594_59495


namespace NUMINAMATH_CALUDE_range_of_a_given_proposition_l594_59473

theorem range_of_a_given_proposition (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) → a ≥ -8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_proposition_l594_59473


namespace NUMINAMATH_CALUDE_triangle_side_values_l594_59459

theorem triangle_side_values (a b c : ℝ) (A B C : ℝ) :
  -- Define triangle ABC
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  -- Area condition
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) →
  -- Given conditions
  (c = 2) →
  (A = π/3) →
  -- Conclusion
  (a = Real.sqrt 3 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_values_l594_59459


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l594_59443

theorem sock_pair_combinations (black green red : ℕ) 
  (h_black : black = 5) 
  (h_green : green = 3) 
  (h_red : red = 4) : 
  black * green + black * red + green * red = 47 := by
sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l594_59443


namespace NUMINAMATH_CALUDE_grain_distance_equation_l594_59425

/-- The distance between the two towers in feet -/
def tower_distance : ℝ := 400

/-- The height of the church tower in feet -/
def church_tower_height : ℝ := 180

/-- The height of the cathedral tower in feet -/
def cathedral_tower_height : ℝ := 240

/-- The speed of the bird from the church tower in ft/s -/
def church_bird_speed : ℝ := 20

/-- The speed of the bird from the cathedral tower in ft/s -/
def cathedral_bird_speed : ℝ := 25

/-- The theorem stating the equation for the distance of the grain from the church tower -/
theorem grain_distance_equation (x : ℝ) :
  x ≥ 0 ∧ x ≤ tower_distance →
  25 * x = 20 * (tower_distance - x) :=
sorry

end NUMINAMATH_CALUDE_grain_distance_equation_l594_59425


namespace NUMINAMATH_CALUDE_thief_reasoning_flaw_l594_59465

/-- Represents the components of the thief's argument --/
inductive ArgumentComponent
  | MajorPremise
  | MinorPremise
  | Conclusion

/-- Represents the thief's ability to open a video recorder --/
def can_open (x : Prop) : Prop := x

/-- Represents the ownership of the video recorder --/
def is_mine (x : Prop) : Prop := x

/-- The thief's argument structure --/
def thief_argument (recorder : Prop) : Prop :=
  (is_mine recorder → can_open recorder) ∧
  (can_open recorder) ∧
  (is_mine recorder)

/-- The flaw in the thief's reasoning --/
def flaw_in_reasoning (component : ArgumentComponent) : Prop :=
  component = ArgumentComponent.MajorPremise

/-- Theorem stating that the flaw in the thief's reasoning is in the major premise --/
theorem thief_reasoning_flaw (recorder : Prop) :
  thief_argument recorder → flaw_in_reasoning ArgumentComponent.MajorPremise :=
by sorry

end NUMINAMATH_CALUDE_thief_reasoning_flaw_l594_59465


namespace NUMINAMATH_CALUDE_total_donation_to_orphanages_l594_59448

theorem total_donation_to_orphanages (donation1 donation2 donation3 : ℝ) 
  (h1 : donation1 = 175)
  (h2 : donation2 = 225)
  (h3 : donation3 = 250) :
  donation1 + donation2 + donation3 = 650 :=
by sorry

end NUMINAMATH_CALUDE_total_donation_to_orphanages_l594_59448


namespace NUMINAMATH_CALUDE_triangle_problem_l594_59472

/-- 
Given an acute triangle ABC where a, b, c are sides opposite to angles A, B, C respectively,
if √3a = 2c sin A, a = 2, and the area of triangle ABC is 3√3/2,
then the measure of angle C is π/3 and c = √7.
-/
theorem triangle_problem (a b c A B C : Real) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 → -- acute triangle
  Real.sqrt 3 * a = 2 * c * Real.sin A →
  a = 2 →
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 →
  C = π/3 ∧ c = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l594_59472


namespace NUMINAMATH_CALUDE_problem_solution_l594_59491

theorem problem_solution :
  (∀ x : ℝ, (x + 1) * (x - 3) > (x + 2) * (x - 4)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * (x + y) = 36 →
    x * y ≤ 81 ∧
    (x * y = 81 ↔ x = 9 ∧ y = 9)) :=
by sorry


end NUMINAMATH_CALUDE_problem_solution_l594_59491


namespace NUMINAMATH_CALUDE_four_diamonds_balance_four_bullets_l594_59489

/-- Represents the balance of symbols in a weighing system -/
structure SymbolBalance where
  delta : ℚ      -- Represents Δ
  diamond : ℚ    -- Represents ♢
  bullet : ℚ     -- Represents •

/-- The balance equations given in the problem -/
def balance_equations (sb : SymbolBalance) : Prop :=
  (2 * sb.delta + 3 * sb.diamond = 12 * sb.bullet) ∧
  (sb.delta = 3 * sb.diamond + 2 * sb.bullet)

/-- The theorem to be proved -/
theorem four_diamonds_balance_four_bullets (sb : SymbolBalance) :
  balance_equations sb → 4 * sb.diamond = 4 * sb.bullet :=
by sorry

end NUMINAMATH_CALUDE_four_diamonds_balance_four_bullets_l594_59489


namespace NUMINAMATH_CALUDE_weight_distribution_l594_59499

theorem weight_distribution :
  ∃! (x y z : ℕ), x + y + z = 11 ∧ 3 * x + 7 * y + 14 * z = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_distribution_l594_59499


namespace NUMINAMATH_CALUDE_complement_of_P_l594_59485

def U : Set ℝ := Set.univ

def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

theorem complement_of_P : (Set.univ \ P) = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l594_59485


namespace NUMINAMATH_CALUDE_max_value_theorem_l594_59449

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : 0 < c ∧ c < 3) :
  ∃ (M : ℝ), ∀ (x : ℝ), x ≥ 0 →
    3 * (a - x) * (x + Real.sqrt (x^2 + b^2)) + c * x ≤ M ∧
    M = (3 - c) / 2 * b^2 + 9 * a^2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l594_59449


namespace NUMINAMATH_CALUDE_condition_relations_l594_59429

theorem condition_relations (A B C : Prop) 
  (h1 : B → A)  -- A is necessary for B
  (h2 : C → B)  -- C is sufficient for B
  (h3 : ¬(B → C))  -- C is not necessary for B
  : (C → A) ∧ ¬(A → C) := by sorry

end NUMINAMATH_CALUDE_condition_relations_l594_59429


namespace NUMINAMATH_CALUDE_odd_function_representation_l594_59478

def f (x : ℝ) : ℝ := x * (abs x - 2)

theorem odd_function_representation (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x ≥ 0, f x = x^2 - 2*x) →  -- definition for x ≥ 0
  (∀ x, f x = x * (abs x - 2)) :=  -- claim to prove
by
  sorry

end NUMINAMATH_CALUDE_odd_function_representation_l594_59478


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l594_59409

/-- The solution set of x^2 - 5x + 4 < 0 is a subset of x^2 - (a+5)x + 5a < 0 -/
def subset_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 5*x + 4 < 0 → x^2 - (a+5)*x + 5*a < 0

/-- The range of values for a -/
def a_range (a : ℝ) : Prop := a ≤ 1

/-- Theorem stating the relationship between the subset condition and the range of a -/
theorem subset_implies_a_range :
  ∀ a : ℝ, subset_condition a → a_range a :=
sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l594_59409


namespace NUMINAMATH_CALUDE_museum_visitors_survey_l594_59466

theorem museum_visitors_survey (V : ℕ) : 
  (∃ E : ℕ, 
    V = E + 140 ∧ 
    3 * V = 4 * E) →
  V = 560 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_visitors_survey_l594_59466


namespace NUMINAMATH_CALUDE_find_number_l594_59457

theorem find_number : ∃ x : ℤ, 4 * x + 100 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l594_59457


namespace NUMINAMATH_CALUDE_impossibleToTileModifiedChessboard_l594_59412

/-- Represents a square on the chessboard -/
inductive Square
| Black
| White

/-- Represents the chessboard -/
def Chessboard := Array (Array Square)

/-- Creates a standard 8x8 chessboard -/
def createStandardChessboard : Chessboard :=
  sorry

/-- Removes the top-left and bottom-right squares from the chessboard -/
def removeCornerSquares (board : Chessboard) : Chessboard :=
  sorry

/-- Counts the number of black and white squares on the chessboard -/
def countSquares (board : Chessboard) : (Nat × Nat) :=
  sorry

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement where
  position1 : Nat × Nat
  position2 : Nat × Nat

/-- Checks if a domino placement is valid on the given chessboard -/
def isValidPlacement (board : Chessboard) (placement : DominoPlacement) : Bool :=
  sorry

/-- Main theorem: It's impossible to tile the modified chessboard with dominos -/
theorem impossibleToTileModifiedChessboard :
  ∀ (placements : List DominoPlacement),
    let board := removeCornerSquares createStandardChessboard
    let (blackCount, whiteCount) := countSquares board
    (blackCount ≠ whiteCount) ∧
    (∀ p ∈ placements, isValidPlacement board p) →
    placements.length < 31 :=
  sorry

end NUMINAMATH_CALUDE_impossibleToTileModifiedChessboard_l594_59412


namespace NUMINAMATH_CALUDE_xyz_value_l594_59475

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3) :
  x * y * z = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l594_59475


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l594_59408

theorem grape_rate_calculation (grape_weight : ℕ) (mango_weight : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grape_weight = 8 →
  mango_weight = 9 →
  mango_rate = 55 →
  total_paid = 1055 →
  ∃ (grape_rate : ℕ), grape_rate * grape_weight + mango_rate * mango_weight = total_paid ∧ grape_rate = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l594_59408


namespace NUMINAMATH_CALUDE_boat_current_speed_l594_59413

/-- Given a boat traveling downstream at 15 km/h, and the distance traveled downstream
    in 4 hours equals the distance traveled upstream in 5 hours, prove that the speed
    of the water current is 1.5 km/h. -/
theorem boat_current_speed (v_d : ℝ) (t_d t_u : ℝ) (h1 : v_d = 15)
    (h2 : t_d = 4) (h3 : t_u = 5) (h4 : v_d * t_d = (2 * v_d - 15) * t_u / 2) :
    ∃ v_c : ℝ, v_c = 1.5 ∧ v_d = v_c + (2 * v_d - 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_boat_current_speed_l594_59413


namespace NUMINAMATH_CALUDE_roberts_gre_preparation_time_l594_59453

/-- Represents the preparation time for each subject in the GRE examination -/
structure GREPreparation where
  vocabulary : Nat
  writing : Nat
  quantitative : Nat

/-- Calculates the total preparation time for the GRE examination -/
def totalPreparationTime (prep : GREPreparation) : Nat :=
  prep.vocabulary + prep.writing + prep.quantitative

/-- Theorem: The total preparation time for Robert's GRE examination is 8 months -/
theorem roberts_gre_preparation_time :
  let robert_prep : GREPreparation := ⟨3, 2, 3⟩
  totalPreparationTime robert_prep = 8 := by
  sorry

#check roberts_gre_preparation_time

end NUMINAMATH_CALUDE_roberts_gre_preparation_time_l594_59453


namespace NUMINAMATH_CALUDE_work_done_cyclic_process_work_done_equals_665J_l594_59407

/-- Represents a point in the P-T diagram -/
structure Point where
  pressure : ℝ
  temperature : ℝ

/-- Represents the cyclic process abca -/
structure CyclicProcess where
  a : Point
  b : Point
  c : Point

/-- The gas constant -/
def R : ℝ := 8.314

/-- Theorem: Work done in the cyclic process -/
theorem work_done_cyclic_process (process : CyclicProcess) : ℝ :=
  let T₀ : ℝ := 320
  have h1 : process.a.temperature = T₀ := by sorry
  have h2 : process.c.temperature = T₀ := by sorry
  have h3 : process.a.pressure = process.c.pressure / 2 := by sorry
  have h4 : process.b.pressure = process.a.pressure := by sorry
  have h5 : (process.b.temperature - process.a.temperature) * process.a.pressure > 0 := by sorry
  (1/2) * R * T₀

/-- Main theorem: The work done is equal to 665 J -/
theorem work_done_equals_665J (process : CyclicProcess) : 
  work_done_cyclic_process process = 665 := by sorry

end NUMINAMATH_CALUDE_work_done_cyclic_process_work_done_equals_665J_l594_59407


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l594_59423

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ - m = 0 ∧ x₂^2 - 4*x₂ - m = 0) ↔ m > -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l594_59423


namespace NUMINAMATH_CALUDE_quadratic_positivity_condition_l594_59414

theorem quadratic_positivity_condition (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 0 ∧ 
  ∃ m₀ : ℝ, m₀ > 0 ∧ ¬(∀ x : ℝ, x^2 + 2*x + m₀ > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_positivity_condition_l594_59414


namespace NUMINAMATH_CALUDE_range_of_a_l594_59403

noncomputable def f (x : ℝ) : ℝ := 2 * x + (Real.exp x)⁻¹ - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) :
  a ≤ -1 ∨ a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l594_59403


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l594_59467

theorem unique_solution_quadratic (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l594_59467


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l594_59415

theorem discount_profit_calculation (discount : ℝ) (no_discount_profit : ℝ) (with_discount_profit : ℝ) :
  discount = 0.04 →
  no_discount_profit = 0.4375 →
  with_discount_profit = (1 + no_discount_profit) * (1 - discount) - 1 →
  with_discount_profit = 0.38 := by
sorry

end NUMINAMATH_CALUDE_discount_profit_calculation_l594_59415


namespace NUMINAMATH_CALUDE_sphere_surface_volume_relation_l594_59471

theorem sphere_surface_volume_relation : 
  ∀ (r : ℝ) (S V S' V' : ℝ), 
    r > 0 →
    S = 4 * Real.pi * r^2 →
    V = (4/3) * Real.pi * r^3 →
    S' = 4 * S →
    V' = (4/3) * Real.pi * (2*r)^3 →
    V' = 8 * V := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_volume_relation_l594_59471


namespace NUMINAMATH_CALUDE_parabola_with_directrix_neg_two_l594_59474

/-- A parabola is defined by its directrix and focus. -/
structure Parabola where
  directrix : ℝ  -- y-coordinate of the directrix
  focus : ℝ × ℝ  -- (x, y) coordinates of the focus

/-- The standard equation of a parabola with a vertical axis of symmetry. -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 4 * p.focus.2 * y) ↔ (y - p.focus.2) ^ 2 = (x - p.focus.1) ^ 2 + (y - p.directrix) ^ 2

theorem parabola_with_directrix_neg_two (p : Parabola) 
  (h : p.directrix = -2) : 
  standardEquation p ↔ ∀ x y : ℝ, x ^ 2 = 8 * y :=
sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_neg_two_l594_59474


namespace NUMINAMATH_CALUDE_a2_range_l594_59450

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem a2_range (a : ℕ → ℝ) 
  (h_mono : is_monotonically_increasing a)
  (h_a1 : a 1 = 2)
  (h_ineq : ∀ n : ℕ+, (n + 1 : ℝ) * a n ≥ n * a (2 * n)) :
  2 < a 2 ∧ a 2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_a2_range_l594_59450


namespace NUMINAMATH_CALUDE_specific_triangle_count_is_32_l594_59454

/-- Represents the count of triangles at different levels in a structure --/
structure TriangleCount where
  smallest : Nat
  intermediate : Nat
  larger : Nat
  even_larger : Nat
  whole_structure : Nat

/-- Calculates the total number of triangles in the structure --/
def total_triangles (count : TriangleCount) : Nat :=
  count.smallest + count.intermediate + count.larger + count.even_larger + count.whole_structure

/-- Theorem stating that for a specific triangle count, the total number of triangles is 32 --/
theorem specific_triangle_count_is_32 :
  ∃ (count : TriangleCount),
    count.smallest = 2 ∧
    count.intermediate = 6 ∧
    count.larger = 6 ∧
    count.even_larger = 6 ∧
    count.whole_structure = 12 ∧
    total_triangles count = 32 := by
  sorry

#eval total_triangles { smallest := 2, intermediate := 6, larger := 6, even_larger := 6, whole_structure := 12 }

end NUMINAMATH_CALUDE_specific_triangle_count_is_32_l594_59454


namespace NUMINAMATH_CALUDE_fraction_equality_l594_59462

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l594_59462


namespace NUMINAMATH_CALUDE_inequality_proof_l594_59464

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b * c + 1) + b / (a * c + 1) + c / (a * b + 1) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l594_59464


namespace NUMINAMATH_CALUDE_intersection_with_complement_l594_59463

def U : Finset ℕ := {0,1,2,3,4,5,6}
def A : Finset ℕ := {0,1,3,5}
def B : Finset ℕ := {1,2,4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0,3,5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l594_59463


namespace NUMINAMATH_CALUDE_alfred_maize_storage_l594_59424

/-- Calculates the total amount of maize Alfred has after 2 years of storage, theft, and donation -/
theorem alfred_maize_storage (
  monthly_storage : ℕ)  -- Amount of maize stored each month
  (storage_period : ℕ)   -- Storage period in years
  (stolen : ℕ)           -- Amount of maize stolen
  (donation : ℕ)         -- Amount of maize donated
  (h1 : monthly_storage = 1)
  (h2 : storage_period = 2)
  (h3 : stolen = 5)
  (h4 : donation = 8) :
  monthly_storage * (storage_period * 12) - stolen + donation = 27 :=
by
  sorry


end NUMINAMATH_CALUDE_alfred_maize_storage_l594_59424


namespace NUMINAMATH_CALUDE_job_application_age_range_l594_59481

theorem job_application_age_range 
  (average_age : ℝ) 
  (standard_deviation : ℝ) 
  (max_different_ages : ℕ) 
  (h1 : average_age = 31) 
  (h2 : standard_deviation = 9) 
  (h3 : max_different_ages = 19) :
  (max_different_ages : ℝ) / standard_deviation = 19 / 18 := by
sorry

end NUMINAMATH_CALUDE_job_application_age_range_l594_59481


namespace NUMINAMATH_CALUDE_square_sum_value_l594_59434

theorem square_sum_value (x y : ℝ) (h : 5 * x^2 + y^2 - 4*x*y + 24 ≤ 10*x - 1) : x^2 + y^2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l594_59434


namespace NUMINAMATH_CALUDE_permutation_inequality_l594_59442

theorem permutation_inequality (n : ℕ) : 
  (Nat.factorial (n + 1)).choose n ≠ Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequality_l594_59442


namespace NUMINAMATH_CALUDE_colored_copies_correct_l594_59410

/-- The number of colored copies Sandy made, given that:
  * Colored copies cost 10 cents each
  * White copies cost 5 cents each
  * Sandy made 400 copies in total
  * The total bill was $22.50 -/
def colored_copies : ℕ :=
  let colored_cost : ℚ := 10 / 100  -- 10 cents in dollars
  let white_cost : ℚ := 5 / 100     -- 5 cents in dollars
  let total_copies : ℕ := 400
  let total_bill : ℚ := 45 / 2      -- $22.50 as a rational number
  50  -- The actual value to be proven

theorem colored_copies_correct :
  let colored_cost : ℚ := 10 / 100
  let white_cost : ℚ := 5 / 100
  let total_copies : ℕ := 400
  let total_bill : ℚ := 45 / 2
  ∃ (white_copies : ℕ),
    colored_copies + white_copies = total_copies ∧
    colored_cost * colored_copies + white_cost * white_copies = total_bill :=
by sorry

end NUMINAMATH_CALUDE_colored_copies_correct_l594_59410


namespace NUMINAMATH_CALUDE_sin_neg_ten_pi_thirds_l594_59479

theorem sin_neg_ten_pi_thirds : Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_ten_pi_thirds_l594_59479


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l594_59411

theorem fraction_to_decimal : (5 : ℚ) / 40 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l594_59411


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l594_59486

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 5^n + 1
  let r : ℕ := 3^s - 3*s
  r = 3^126 - 378 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l594_59486


namespace NUMINAMATH_CALUDE_brick_height_is_6cm_l594_59469

/-- Proves that the height of each brick is 6 cm given the wall dimensions,
    brick base dimensions, and the number of bricks needed. -/
theorem brick_height_is_6cm 
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) :
  wall_length = 750 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 6000 →
  ∃ (brick_height : ℝ), 
    wall_length * wall_width * wall_height = 
    (brick_length * brick_width * brick_height) * num_bricks ∧
    brick_height = 6 :=
by sorry

end NUMINAMATH_CALUDE_brick_height_is_6cm_l594_59469


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l594_59420

theorem fraction_ratio_equality : ∃ x : ℚ, (3 / 7) / (6 / 5) = x / (2 / 5) ∧ x = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l594_59420


namespace NUMINAMATH_CALUDE_min_crossing_time_is_21_l594_59422

/-- Represents a person with their crossing time -/
structure Person where
  name : String
  time : ℕ

/-- Represents the tunnel crossing problem -/
structure TunnelProblem where
  people : List Person
  flashlight : ℕ := 1
  capacity : ℕ := 2

def minCrossingTime (problem : TunnelProblem) : ℕ :=
  sorry

/-- The specific problem instance -/
def ourProblem : TunnelProblem :=
  { people := [
      { name := "A", time := 3 },
      { name := "B", time := 4 },
      { name := "C", time := 5 },
      { name := "D", time := 6 }
    ]
  }

theorem min_crossing_time_is_21 :
  minCrossingTime ourProblem = 21 :=
by sorry

end NUMINAMATH_CALUDE_min_crossing_time_is_21_l594_59422


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l594_59435

theorem different_color_chips_probability :
  let total_chips : ℕ := 7 + 6 + 5
  let purple_chips : ℕ := 7
  let green_chips : ℕ := 6
  let orange_chips : ℕ := 5
  let prob_purple : ℚ := purple_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_orange : ℚ := orange_chips / total_chips
  let prob_not_purple : ℚ := (green_chips + orange_chips) / total_chips
  let prob_not_green : ℚ := (purple_chips + orange_chips) / total_chips
  let prob_not_orange : ℚ := (purple_chips + green_chips) / total_chips
  (prob_purple * prob_not_purple + prob_green * prob_not_green + prob_orange * prob_not_orange) = 107 / 162 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l594_59435


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l594_59406

/-- Given a geometric sequence {aₙ} with a₁ > 0 and a₂a₄ + 2a₃a₅ + a₄a₆ = 36, prove that a₃ + a₅ = 6 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
    (h_pos : a 1 > 0) (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
    a 3 + a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l594_59406


namespace NUMINAMATH_CALUDE_circle_passes_through_points_unique_circle_l594_59428

/-- A circle passing through three points -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Check if a point lies on a circle -/
def lies_on (c : Circle) (x y : ℝ) : Prop :=
  c.equation x y

/-- The specific circle we're interested in -/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*y = 0 }

theorem circle_passes_through_points :
  (lies_on our_circle (-1) 1) ∧
  (lies_on our_circle 1 1) ∧
  (lies_on our_circle 0 0) := by
  sorry

/-- Uniqueness of the circle -/
theorem unique_circle (c : Circle) :
  (lies_on c (-1) 1) ∧
  (lies_on c 1 1) ∧
  (lies_on c 0 0) →
  ∀ x y, c.equation x y ↔ our_circle.equation x y := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_unique_circle_l594_59428


namespace NUMINAMATH_CALUDE_foreign_language_speakers_l594_59426

theorem foreign_language_speakers (M F : ℕ) : 
  M = F →  -- number of male students equals number of female students
  (3 : ℚ) / 5 * M + (2 : ℚ) / 3 * F = (19 : ℚ) / 30 * (M + F) := by
  sorry

end NUMINAMATH_CALUDE_foreign_language_speakers_l594_59426


namespace NUMINAMATH_CALUDE_quadratic_inequality_implications_l594_59483

theorem quadratic_inequality_implications 
  (a b c t : ℝ) 
  (h1 : t > 1) 
  (h2 : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < t) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + (a-b)*x₁ - c = 0 ∧ a*x₂^2 + (a-b)*x₂ - c = 0) ∧
  (∀ x₁ x₂ : ℝ, a*x₁^2 + (a-b)*x₁ - c = 0 → a*x₂^2 + (a-b)*x₂ - c = 0 → |x₂ - x₁| > Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implications_l594_59483
