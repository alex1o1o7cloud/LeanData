import Mathlib

namespace NUMINAMATH_CALUDE_hamburger_combinations_count_l3804_380435

/-- The number of condiment choices available. -/
def num_condiments : ℕ := 9

/-- The number of options for meat patties. -/
def patty_options : ℕ := 3

/-- Calculates the number of different hamburger combinations. -/
def hamburger_combinations : ℕ := patty_options * 2^num_condiments

/-- Theorem stating that the number of different hamburger combinations is 1536. -/
theorem hamburger_combinations_count : hamburger_combinations = 1536 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_count_l3804_380435


namespace NUMINAMATH_CALUDE_hockey_league_teams_l3804_380422

/-- The number of teams in a hockey league -/
def num_teams : ℕ := 15

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := 1050

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) / 2) * games_per_pair = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l3804_380422


namespace NUMINAMATH_CALUDE_simplify_linear_expression_l3804_380454

theorem simplify_linear_expression (x : ℝ) : 5*x + 2*x + 7*x = 14*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_linear_expression_l3804_380454


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3804_380438

theorem complex_number_quadrant (z : ℂ) (m : ℝ) :
  z * Complex.I = Complex.I + m →
  z.im = 1 →
  z.re > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3804_380438


namespace NUMINAMATH_CALUDE_negation_equivalence_l3804_380430

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ > 0) ↔ (∀ x : ℝ, x^2 + 2*x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3804_380430


namespace NUMINAMATH_CALUDE_snow_probability_l3804_380467

theorem snow_probability (p : ℝ) (h : p = 2/3) :
  3 * p^2 * (1 - p) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l3804_380467


namespace NUMINAMATH_CALUDE_sufficient_condition_for_existence_l3804_380441

theorem sufficient_condition_for_existence (a : ℝ) :
  (a ≥ 2) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ x₀^2 - a ≤ 0) ∧
  ¬((∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ x₀^2 - a ≤ 0) → (a ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_existence_l3804_380441


namespace NUMINAMATH_CALUDE_special_multiplication_pattern_l3804_380472

theorem special_multiplication_pattern (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) := by
  sorry

end NUMINAMATH_CALUDE_special_multiplication_pattern_l3804_380472


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_value_l3804_380427

theorem tan_alpha_two_implies_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_value_l3804_380427


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3804_380457

def A : Set ℝ := {-1, 0, (1/2), 3}
def B : Set ℝ := {x : ℝ | x^2 ≥ 1}

theorem intersection_of_A_and_B :
  A ∩ B = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3804_380457


namespace NUMINAMATH_CALUDE_garden_length_is_50_l3804_380431

def garden_length (width : ℝ) : ℝ := 2 * width

def garden_perimeter (width : ℝ) : ℝ := 2 * garden_length width + 2 * width

theorem garden_length_is_50 :
  ∃ (width : ℝ), garden_perimeter width = 150 ∧ garden_length width = 50 :=
by sorry

end NUMINAMATH_CALUDE_garden_length_is_50_l3804_380431


namespace NUMINAMATH_CALUDE_rhombus_and_rectangle_diagonals_bisect_l3804_380452

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect_each_other (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem rhombus_and_rectangle_diagonals_bisect :
  ∀ q : Quadrilateral, 
    (is_rhombus q ∨ is_rectangle q) → diagonals_bisect_each_other q :=
by sorry

end NUMINAMATH_CALUDE_rhombus_and_rectangle_diagonals_bisect_l3804_380452


namespace NUMINAMATH_CALUDE_robin_gum_count_l3804_380492

theorem robin_gum_count (packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : 
  packages = 9 → pieces_per_package = 15 → total_pieces = packages * pieces_per_package → 
  total_pieces = 135 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l3804_380492


namespace NUMINAMATH_CALUDE_alpha_beta_ratio_l3804_380405

-- Define the angles
variable (α β x y : ℝ)

-- Define the angle relationships
axiom angle_relation_1 : y = x + β
axiom angle_relation_2 : 2 * y = 2 * x + α

-- Theorem to prove
theorem alpha_beta_ratio : α / β = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_ratio_l3804_380405


namespace NUMINAMATH_CALUDE_largest_negative_integer_l3804_380475

theorem largest_negative_integer :
  ∃! n : ℤ, n < 0 ∧ ∀ m : ℤ, m < 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l3804_380475


namespace NUMINAMATH_CALUDE_sum_bound_l3804_380458

theorem sum_bound (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : |a - b| + |b - c| + |c - a| = 1) : 
  a + b + c ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l3804_380458


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3804_380473

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

/-- The shifted parabola function -/
def shifted_parabola (n : ℝ) (x : ℝ) : ℝ := original_parabola (x - n)

/-- Theorem stating the conditions and the result to be proved -/
theorem parabola_shift_theorem (n : ℝ) (y1 y2 : ℝ) 
  (h1 : n > 0)
  (h2 : shifted_parabola n 2 = y1)
  (h3 : shifted_parabola n 4 = y2)
  (h4 : y1 > y2) :
  n = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3804_380473


namespace NUMINAMATH_CALUDE_bird_migration_difference_l3804_380494

/-- The number of bird families that flew to Asia is greater than the number
    of bird families that flew to Africa by 47. -/
theorem bird_migration_difference :
  let mountain_families : ℕ := 38
  let africa_families : ℕ := 47
  let asia_families : ℕ := 94
  asia_families - africa_families = 47 := by sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l3804_380494


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3804_380432

-- Define the condition "m < 1/4"
def condition (m : ℝ) : Prop := m < (1/4 : ℝ)

-- Define when a quadratic equation has real solutions
def has_real_solutions (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

-- State the theorem
theorem condition_sufficient_not_necessary :
  (∀ m : ℝ, condition m → has_real_solutions 1 1 m) ∧
  (∃ m : ℝ, ¬(condition m) ∧ has_real_solutions 1 1 m) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3804_380432


namespace NUMINAMATH_CALUDE_parabola_intersections_l3804_380465

-- Define the parabola
def W (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line x = 4
def L (x : ℝ) : Prop := x = 4

-- Define points A and B
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (4, -4)

-- Define point P
structure Point (x₀ y₀ : ℝ) : Prop :=
  (on_parabola : W x₀ y₀)
  (x_constraint : x₀ < 4)
  (y_constraint : y₀ ≥ 0)

-- Define the area of triangle PAB
def area_PAB (x₀ y₀ : ℝ) : ℝ := 4 * (4 - x₀)

-- Define the perpendicularity condition
def perp_condition (x₀ y₀ : ℝ) : Prop :=
  (4 - y₀^2/4)^2 = (4 - y₀) * (4 + y₀)

-- Define the area of triangle PMN
def area_PMN (y₀ : ℝ) : ℝ := y₀^2

theorem parabola_intersections 
  (x₀ y₀ : ℝ) (p : Point x₀ y₀) :
  (area_PAB x₀ y₀ = 4 → x₀ = 3 ∧ y₀ = 2 * Real.sqrt 3) ∧
  (perp_condition x₀ y₀ → Real.sqrt ((4 - x₀)^2 + (4 - y₀)^2) = 4 * Real.sqrt 2) ∧
  (area_PMN y₀ = area_PAB x₀ y₀ → area_PMN y₀ = 8) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersections_l3804_380465


namespace NUMINAMATH_CALUDE_unique_solution_m_value_l3804_380401

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := 16 * x^2 + m * x + 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := m^2 - 4 * 16 * 4

-- Theorem statement
theorem unique_solution_m_value :
  ∃! m : ℝ, m > 0 ∧ (∃! x : ℝ, quadratic_equation m x = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_m_value_l3804_380401


namespace NUMINAMATH_CALUDE_inequality_solutions_range_l3804_380418

theorem inequality_solutions_range (a : ℝ) : 
  (∃! (x₁ x₂ : ℕ), x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
   (∀ (x : ℕ), x > 0 → (3 * ↑x + a ≤ 2 ↔ (x = x₁ ∨ x = x₂)))) →
  -7 < a ∧ a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_range_l3804_380418


namespace NUMINAMATH_CALUDE_two_machines_total_copies_l3804_380414

/-- Represents a copy machine with a constant copying rate. -/
structure CopyMachine where
  rate : ℕ  -- Copies per minute

/-- Calculates the total number of copies made by a machine in a given time. -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Represents the problem setup with two copy machines. -/
structure TwoMachineProblem where
  machine1 : CopyMachine
  machine2 : CopyMachine
  duration : ℕ  -- Duration in minutes

/-- The main theorem stating the total number of copies made by both machines. -/
theorem two_machines_total_copies (problem : TwoMachineProblem)
    (h1 : problem.machine1.rate = 40)
    (h2 : problem.machine2.rate = 55)
    (h3 : problem.duration = 30) :
    copies_made problem.machine1 problem.duration +
    copies_made problem.machine2 problem.duration = 2850 := by
  sorry

end NUMINAMATH_CALUDE_two_machines_total_copies_l3804_380414


namespace NUMINAMATH_CALUDE_remaining_area_ratio_l3804_380408

/-- The ratio of remaining areas of two squares after cutting out smaller squares -/
theorem remaining_area_ratio (side_c side_d cut_side : ℕ) 
  (hc : side_c = 48) 
  (hd : side_d = 60) 
  (hcut : cut_side = 12) : 
  (side_c^2 - cut_side^2) / (side_d^2 - cut_side^2) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_ratio_l3804_380408


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3804_380493

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (h : ArithmeticSequence a) (h2 : a 2 + a 12 = 32) :
  a 3 + a 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3804_380493


namespace NUMINAMATH_CALUDE_difference_of_values_l3804_380437

theorem difference_of_values (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_of_values_l3804_380437


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_choose_2_l3804_380462

theorem binomial_coefficient_n_choose_2 (n : ℕ) (h : n ≥ 2) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_choose_2_l3804_380462


namespace NUMINAMATH_CALUDE_some_number_value_l3804_380416

theorem some_number_value (x : ℝ) : 60 + 5 * 12 / (180 / x) = 61 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3804_380416


namespace NUMINAMATH_CALUDE_extra_flowers_count_l3804_380415

def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4
def used_flowers : ℕ := 19

def total_picked : ℕ := tulips + roses + daisies + lilies

theorem extra_flowers_count : total_picked - used_flowers = 8 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_count_l3804_380415


namespace NUMINAMATH_CALUDE_unique_function_existence_l3804_380433

theorem unique_function_existence : 
  ∃! f : ℕ → ℕ, f 1 = 1 ∧ ∀ n : ℕ, f n * f (n + 2) = f (n + 1) ^ 2 + 1997 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l3804_380433


namespace NUMINAMATH_CALUDE_dave_apps_unchanged_l3804_380451

theorem dave_apps_unchanged (initial_files final_files deleted_files final_apps : ℕ) :
  initial_files = final_files + deleted_files →
  initial_files = 24 →
  final_files = 21 →
  deleted_files = 3 →
  final_apps = 17 →
  initial_apps = final_apps :=
by
  sorry

#check dave_apps_unchanged

end NUMINAMATH_CALUDE_dave_apps_unchanged_l3804_380451


namespace NUMINAMATH_CALUDE_characterize_valid_pairs_l3804_380480

def is_valid_pair (n p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ n.val ≤ 2 * p.val ∧ (p.val - 1)^n.val + 1 ∣ n.val^2

theorem characterize_valid_pairs :
  ∀ (n p : ℕ+), is_valid_pair n p ↔
    (n = 2 ∧ p = 2) ∨
    (n = 3 ∧ p = 3) ∨
    (n = 1 ∧ Nat.Prime p.val) :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_pairs_l3804_380480


namespace NUMINAMATH_CALUDE_olympic_numbers_l3804_380421

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def all_digits_different (x y : ℕ) : Prop :=
  ∀ d, is_valid_digit d → (d ∈ x.digits 10 ↔ d ∉ y.digits 10)

theorem olympic_numbers :
  ∀ x y : ℕ,
    x < 1000 ∧ x ≥ 100 ∧  -- x is a three-digit number
    y < 10000 ∧ y ≥ 1000 ∧  -- y is a four-digit number
    (∀ d, d ∈ x.digits 10 → is_valid_digit d) ∧
    (∀ d, d ∈ y.digits 10 → is_valid_digit d) ∧
    all_digits_different x y ∧
    1 ∉ x.digits 10 ∧
    9 ∉ x.digits 10 ∧
    x / y = 1 / 9  -- Rational division
    →
    x = 163 ∨ x = 318 ∨ x = 729 ∨ x = 1638 ∨ x = 1647 :=
by sorry

end NUMINAMATH_CALUDE_olympic_numbers_l3804_380421


namespace NUMINAMATH_CALUDE_left_handed_mouse_price_increase_l3804_380490

/-- Represents the store's weekly operation --/
structure StoreOperation where
  daysOpen : Nat
  miceSoldPerDay : Nat
  normalMousePrice : Nat
  weeklyRevenue : Nat

/-- Calculates the percentage increase in price --/
def percentageIncrease (normalPrice leftHandedPrice : Nat) : Nat :=
  ((leftHandedPrice - normalPrice) * 100) / normalPrice

/-- Theorem stating the percentage increase in left-handed mouse price --/
theorem left_handed_mouse_price_increase 
  (store : StoreOperation)
  (h1 : store.daysOpen = 4)
  (h2 : store.miceSoldPerDay = 25)
  (h3 : store.normalMousePrice = 120)
  (h4 : store.weeklyRevenue = 15600) :
  percentageIncrease store.normalMousePrice 
    ((store.weeklyRevenue / store.daysOpen) / store.miceSoldPerDay) = 30 := by
  sorry

#eval percentageIncrease 120 156

end NUMINAMATH_CALUDE_left_handed_mouse_price_increase_l3804_380490


namespace NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3804_380453

/-- A hyperbola with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  e_def : e = (a^2 + b^2).sqrt / a

theorem hyperbola_imaginary_axis_length 
  (h : Hyperbola) 
  (dist_foci : ∃ (p : ℝ × ℝ), p.1^2 / h.a^2 - p.2^2 / h.b^2 = 1 ∧ 
    ∃ (f₁ f₂ : ℝ × ℝ), (p.1 - f₁.1)^2 + (p.2 - f₁.2)^2 = 100 ∧ 
                        (p.1 - f₂.1)^2 + (p.2 - f₂.2)^2 = 16) 
  (h_e : h.e = 2) : 
  2 * h.b = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3804_380453


namespace NUMINAMATH_CALUDE_books_sum_l3804_380495

/-- The number of books Sam has -/
def sam_books : ℕ := 110

/-- The number of books Joan has -/
def joan_books : ℕ := 102

/-- The total number of books Sam and Joan have together -/
def total_books : ℕ := sam_books + joan_books

theorem books_sum :
  total_books = 212 :=
by sorry

end NUMINAMATH_CALUDE_books_sum_l3804_380495


namespace NUMINAMATH_CALUDE_alarm_system_probability_l3804_380485

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) :
  let prob_at_least_one := 1 - (1 - p) * (1 - p)
  prob_at_least_one = 0.64 := by
sorry

end NUMINAMATH_CALUDE_alarm_system_probability_l3804_380485


namespace NUMINAMATH_CALUDE_complex_division_result_l3804_380482

theorem complex_division_result : ∃ (i : ℂ), i * i = -1 ∧ (2 : ℂ) / (1 - i) = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l3804_380482


namespace NUMINAMATH_CALUDE_eighteen_player_tournament_l3804_380466

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A round-robin tournament with 18 players has 153 games -/
theorem eighteen_player_tournament : num_games 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_player_tournament_l3804_380466


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l3804_380499

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 10) : 
  x + y ≤ 93 / 44 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l3804_380499


namespace NUMINAMATH_CALUDE_bobby_blocks_l3804_380456

/-- Calculates the total number of blocks Bobby has after receiving blocks from his father. -/
def total_blocks (initial : ℕ) (multiplier : ℕ) : ℕ :=
  initial + multiplier * initial

/-- Proves that Bobby has 8 blocks in total given the initial conditions. -/
theorem bobby_blocks : total_blocks 2 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobby_blocks_l3804_380456


namespace NUMINAMATH_CALUDE_polygon_sides_l3804_380413

/-- A convex polygon with the sum of all angles except one equal to 2790° has 18 sides -/
theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  n ≥ 3 → -- convex polygon has at least 3 sides
  angle_sum = 2790 → -- sum of all angles except one is 2790°
  (n - 2) * 180 - angle_sum ≥ 0 → -- the missing angle is non-negative
  (n - 2) * 180 - angle_sum < 180 → -- the missing angle is less than 180°
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3804_380413


namespace NUMINAMATH_CALUDE_intersection_equals_B_intersection_with_complement_empty_l3804_380481

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 2 ≤ x ∧ x ≤ 2*a + 3}
def B : Set ℝ := {x : ℝ | x^2 - 6*x + 5 ≤ 0}

-- Theorem for the first question
theorem intersection_equals_B (a : ℝ) :
  (A a) ∩ B = B ↔ a ∈ Set.Icc 1 3 := by sorry

-- Theorem for the second question
theorem intersection_with_complement_empty (a : ℝ) :
  (A a) ∩ (Bᶜ) = ∅ ↔ a < -5 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_intersection_with_complement_empty_l3804_380481


namespace NUMINAMATH_CALUDE_no_given_factors_l3804_380471

def f (x : ℝ) : ℝ := x^5 + 3*x^3 - 4*x^2 + 12*x + 8

theorem no_given_factors :
  (∀ x, f x ≠ 0 → x + 1 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 + 1 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 - 2 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 + 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_given_factors_l3804_380471


namespace NUMINAMATH_CALUDE_binary_93_l3804_380470

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 93 is [true, false, true, true, true, false, true] -/
theorem binary_93 : toBinary 93 = [true, false, true, true, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_93_l3804_380470


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l3804_380455

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The specific problem setup -/
def problemSetup : ConeSphereProblem :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 2
  , sphereRadius := 0  -- To be maximized
  }

/-- The theorem stating the maximal value of r^2 -/
theorem max_sphere_radius_squared (setup : ConeSphereProblem) :
  setup.cone1 = setup.cone2 →
  setup.cone1.baseRadius = 4 →
  setup.cone1.height = 10 →
  setup.intersectionDistance = 2 →
  ∃ (r : ℝ), r^2 ≤ 144/29 ∧
    ∀ (s : ℝ), (∃ (config : ConeSphereProblem),
      config.cone1 = setup.cone1 ∧
      config.cone2 = setup.cone2 ∧
      config.intersectionDistance = setup.intersectionDistance ∧
      config.sphereRadius = s) →
    s^2 ≤ r^2 :=
by
  sorry


end NUMINAMATH_CALUDE_max_sphere_radius_squared_l3804_380455


namespace NUMINAMATH_CALUDE_darren_fergie_equal_debt_l3804_380488

/-- Represents the amount owed after t days with simple interest -/
def amountOwed (principal : ℝ) (rate : ℝ) (days : ℝ) : ℝ :=
  principal * (1 + rate * days)

/-- The problem statement -/
theorem darren_fergie_equal_debt : ∃ t : ℝ, 
  t = 20 ∧ 
  amountOwed 100 0.10 t = amountOwed 150 0.05 t :=
sorry

end NUMINAMATH_CALUDE_darren_fergie_equal_debt_l3804_380488


namespace NUMINAMATH_CALUDE_expression_evaluation_l3804_380468

theorem expression_evaluation : 
  11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3804_380468


namespace NUMINAMATH_CALUDE_total_spent_is_two_dollars_l3804_380426

/-- The price of a single pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants -/
def robert_pencils : ℕ := 5

/-- The number of pencils Melissa wants -/
def melissa_pencils : ℕ := 2

/-- Theorem: The total amount spent by the students is $2.00 -/
theorem total_spent_is_two_dollars :
  (tolu_pencils * pencil_price + robert_pencils * pencil_price + melissa_pencils * pencil_price) / 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_two_dollars_l3804_380426


namespace NUMINAMATH_CALUDE_inequality_property_l3804_380497

theorem inequality_property (a b : ℝ) : a > b → -5 * a < -5 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3804_380497


namespace NUMINAMATH_CALUDE_second_feeding_maggots_l3804_380486

/-- Given the total number of maggots served and the number of maggots in the first feeding,
    calculate the number of maggots in the second feeding. -/
def maggots_in_second_feeding (total_maggots : ℕ) (first_feeding : ℕ) : ℕ :=
  total_maggots - first_feeding

/-- Theorem stating that given 20 total maggots and 10 maggots in the first feeding,
    the number of maggots in the second feeding is 10. -/
theorem second_feeding_maggots :
  maggots_in_second_feeding 20 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_feeding_maggots_l3804_380486


namespace NUMINAMATH_CALUDE_infinitely_many_nth_powers_l3804_380412

/-- An infinite arithmetic progression of positive integers -/
structure ArithmeticProgression :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference

/-- Checks if a number is in the arithmetic progression -/
def ArithmeticProgression.contains (ap : ArithmeticProgression) (x : ℕ) : Prop :=
  ∃ k : ℕ, x = ap.a + k * ap.d

/-- Checks if a number is an nth power -/
def is_nth_power (x n : ℕ) : Prop :=
  ∃ m : ℕ, x = m^n

theorem infinitely_many_nth_powers
  (ap : ArithmeticProgression)
  (n : ℕ)
  (h : ∃ x : ℕ, ap.contains x ∧ is_nth_power x n) :
  ∀ N : ℕ, ∃ M : ℕ, M > N ∧ ap.contains M ∧ is_nth_power M n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_nth_powers_l3804_380412


namespace NUMINAMATH_CALUDE_seashell_collection_count_l3804_380478

theorem seashell_collection_count (initial_count additional_count : ℕ) :
  initial_count = 19 → additional_count = 6 →
  initial_count + additional_count = 25 :=
by sorry

end NUMINAMATH_CALUDE_seashell_collection_count_l3804_380478


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l3804_380447

def sum_of_digits (n : ℕ) : ℕ := sorry

def sequence_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_1_to_5000 : 
  sequence_sum 5000 = 194450 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l3804_380447


namespace NUMINAMATH_CALUDE_stella_profit_is_25_l3804_380403

/-- Represents the profit Stella makes from her antique shop sales -/
def stellas_profit (num_dolls num_clocks num_glasses : ℕ) 
                   (price_doll price_clock price_glass : ℚ) 
                   (cost : ℚ) : ℚ :=
  num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass - cost

/-- Theorem stating that Stella's profit is $25 given the specified conditions -/
theorem stella_profit_is_25 : 
  stellas_profit 3 2 5 5 15 4 40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_stella_profit_is_25_l3804_380403


namespace NUMINAMATH_CALUDE_dispatch_three_male_two_female_dispatch_at_least_two_male_l3804_380436

def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def team_size : ℕ := 5

/-- The number of ways to choose 3 male drivers and 2 female drivers -/
theorem dispatch_three_male_two_female : 
  (Nat.choose male_drivers 3) * (Nat.choose female_drivers 2) = 60 := by sorry

/-- The number of ways to dispatch with at least two male drivers -/
theorem dispatch_at_least_two_male : 
  (Nat.choose male_drivers 2) * (Nat.choose female_drivers 3) +
  (Nat.choose male_drivers 3) * (Nat.choose female_drivers 2) +
  (Nat.choose male_drivers 4) * (Nat.choose female_drivers 1) +
  (Nat.choose male_drivers 5) * (Nat.choose female_drivers 0) = 121 := by sorry

end NUMINAMATH_CALUDE_dispatch_three_male_two_female_dispatch_at_least_two_male_l3804_380436


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3804_380400

theorem polynomial_evaluation (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = (1 - 3*x) * (1 + x)^5) →
  (∀ x, f x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + (1/3)*a₁ + (1/3^2)*a₂ + (1/3^3)*a₃ + (1/3^4)*a₄ + (1/3^5)*a₅ + (1/3^6)*a₆ = 0 := by
sorry


end NUMINAMATH_CALUDE_polynomial_evaluation_l3804_380400


namespace NUMINAMATH_CALUDE_school_population_l3804_380423

theorem school_population (girls : ℕ) (boys : ℕ) (teachers : ℕ) (staff : ℕ)
  (h1 : girls = 542)
  (h2 : boys = 387)
  (h3 : teachers = 45)
  (h4 : staff = 27) :
  girls + boys + teachers + staff = 1001 := by
sorry

end NUMINAMATH_CALUDE_school_population_l3804_380423


namespace NUMINAMATH_CALUDE_unique_solution_l3804_380449

/-- Represents a three-digit number in the form ABA --/
def ABA (A B : ℕ) : ℕ := 100 * A + 10 * B + A

/-- Represents a four-digit number in the form CCDC --/
def CCDC (C D : ℕ) : ℕ := 1000 * C + 100 * C + 10 * D + C

theorem unique_solution :
  ∃! (A B C D : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧
    (ABA A B)^2 = CCDC C D ∧
    CCDC C D < 100000 ∧
    A = 2 ∧ B = 1 ∧ C = 4 ∧ D = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3804_380449


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3804_380440

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 8*x^3 + 12*x^2 + 20*x - 10
  let g : ℝ → ℝ := λ x => x - 2
  ∃ q : ℝ → ℝ, f x = g x * q x + 30 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3804_380440


namespace NUMINAMATH_CALUDE_percent_relation_l3804_380483

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : z = 1.2 * x) : 
  y = 0.75 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3804_380483


namespace NUMINAMATH_CALUDE_eleven_pow_2023_mod_50_l3804_380461

theorem eleven_pow_2023_mod_50 : 11^2023 % 50 = 31 := by
  sorry

end NUMINAMATH_CALUDE_eleven_pow_2023_mod_50_l3804_380461


namespace NUMINAMATH_CALUDE_iris_rose_ratio_l3804_380476

/-- Proves that given an initial ratio of irises to roses of 2:5, 
    with 25 roses initially and 20 roses added, 
    maintaining the same ratio results in a total of 18 irises. -/
theorem iris_rose_ratio (initial_roses : ℕ) (added_roses : ℕ) 
  (iris_ratio : ℕ) (rose_ratio : ℕ) : 
  initial_roses = 25 →
  added_roses = 20 →
  iris_ratio = 2 →
  rose_ratio = 5 →
  (iris_ratio : ℚ) / rose_ratio * (initial_roses + added_roses) = 18 := by
  sorry

#check iris_rose_ratio

end NUMINAMATH_CALUDE_iris_rose_ratio_l3804_380476


namespace NUMINAMATH_CALUDE_triangle_tangent_determinant_l3804_380464

/-- Given angles A, B, C of a non-right triangle, the determinant of the matrix
    | tan²A  1      1     |
    | 1      tan²B  1     |
    | 1      1      tan²C |
    is equal to 2. -/
theorem triangle_tangent_determinant (A B C : Real) 
  (h : A + B + C = π) 
  (h_non_right : A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2) : 
  let M : Matrix (Fin 3) (Fin 3) Real := 
    !![Real.tan A ^ 2, 1, 1; 
       1, Real.tan B ^ 2, 1; 
       1, 1, Real.tan C ^ 2]
  Matrix.det M = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_determinant_l3804_380464


namespace NUMINAMATH_CALUDE_car_payment_months_l3804_380489

def car_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem car_payment_months : 
  (car_price - initial_payment) / monthly_payment = 19 := by
  sorry

end NUMINAMATH_CALUDE_car_payment_months_l3804_380489


namespace NUMINAMATH_CALUDE_zeros_in_Q_l3804_380419

def R (k : ℕ) : ℚ := (10^k - 1) / 9

def Q : ℚ := R 25 / R 5

def count_zeros (q : ℚ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 16 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l3804_380419


namespace NUMINAMATH_CALUDE_i_in_first_quadrant_l3804_380444

/-- The complex number i corresponds to a point in the first quadrant of the complex plane. -/
theorem i_in_first_quadrant : Complex.I.re = 0 ∧ Complex.I.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_i_in_first_quadrant_l3804_380444


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3804_380479

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (stream_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  stream_speed = 5 →
  downstream_distance = 70 →
  downstream_time = 2 →
  ∃ (boat_speed : ℝ), boat_speed = 30 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3804_380479


namespace NUMINAMATH_CALUDE_inclination_angle_range_l3804_380477

theorem inclination_angle_range (θ : ℝ) :
  let k := Real.cos θ
  let α := Real.arctan k
  α ∈ Set.Icc 0 (π / 4) ∪ Set.Ico (3 * π / 4) π :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3804_380477


namespace NUMINAMATH_CALUDE_line_segment_parameterization_l3804_380409

/-- Given a line segment connecting points (1,3) and (4,9), parameterized by x = at + b and y = ct + d,
    where t = 0 corresponds to (1,3) and t = 1 corresponds to (4,9),
    prove that a^2 + b^2 + c^2 + d^2 = 55. -/
theorem line_segment_parameterization (a b c d : ℝ) : 
  (∀ t : ℝ, (a * t + b, c * t + d) = (1 - t, 3 - 3*t) + t • (4, 9)) →
  a^2 + b^2 + c^2 + d^2 = 55 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_l3804_380409


namespace NUMINAMATH_CALUDE_boat_travel_time_l3804_380450

theorem boat_travel_time (boat_speed : ℝ) (distance : ℝ) (return_time : ℝ) :
  boat_speed = 15.6 →
  distance = 96 →
  return_time = 5 →
  ∃ (current_speed : ℝ),
    current_speed > 0 ∧
    current_speed < boat_speed ∧
    distance = (boat_speed + current_speed) * return_time ∧
    distance / (boat_speed - current_speed) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boat_travel_time_l3804_380450


namespace NUMINAMATH_CALUDE_band_earnings_theorem_l3804_380404

/-- Represents a band with its earnings and gig information -/
structure Band where
  members : ℕ
  totalEarnings : ℕ
  gigs : ℕ

/-- Calculates the earnings per member per gig for a given band -/
def earningsPerMemberPerGig (b : Band) : ℚ :=
  (b.totalEarnings : ℚ) / (b.members : ℚ) / (b.gigs : ℚ)

/-- Theorem: For a band with 4 members that earned $400 after 5 gigs, 
    each member earns $20 per gig -/
theorem band_earnings_theorem (b : Band) 
    (h1 : b.members = 4) 
    (h2 : b.totalEarnings = 400) 
    (h3 : b.gigs = 5) : 
  earningsPerMemberPerGig b = 20 := by
  sorry


end NUMINAMATH_CALUDE_band_earnings_theorem_l3804_380404


namespace NUMINAMATH_CALUDE_expression_evaluation_l3804_380487

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 3 - 3
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3804_380487


namespace NUMINAMATH_CALUDE_sarah_copies_360_pages_l3804_380443

/-- The number of copies per person -/
def copies_per_person : ℕ := 2

/-- The number of people in the meeting -/
def number_of_people : ℕ := 9

/-- The number of pages in each contract -/
def pages_per_contract : ℕ := 20

/-- The total number of pages Sarah will copy -/
def total_pages : ℕ := copies_per_person * number_of_people * pages_per_contract

theorem sarah_copies_360_pages : total_pages = 360 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_360_pages_l3804_380443


namespace NUMINAMATH_CALUDE_complement_of_B_l3804_380442

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- Define the universal set U
def U (x : ℝ) : Set ℝ := A x ∪ B x

-- State the theorem
theorem complement_of_B (x : ℝ) :
  (B x ∪ (U x \ B x) = A x) →
  ((x = 0 → U x \ B x = {3}) ∧
   (x = Real.sqrt 3 → U x \ B x = {Real.sqrt 3}) ∧
   (x = -Real.sqrt 3 → U x \ B x = {-Real.sqrt 3})) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_B_l3804_380442


namespace NUMINAMATH_CALUDE_area_S_eq_four_sqrt_three_thirds_l3804_380484

/-- A rhombus with side length 4 and one angle of 150 degrees -/
structure Rhombus150 where
  side_length : ℝ
  angle_F : ℝ
  side_length_eq : side_length = 4
  angle_F_eq : angle_F = 150 * π / 180

/-- The region S inside the rhombus closer to vertex F than to other vertices -/
def region_S (r : Rhombus150) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region S -/
noncomputable def area_S (r : Rhombus150) : ℝ :=
  sorry

/-- Theorem stating that the area of region S is 4√3/3 -/
theorem area_S_eq_four_sqrt_three_thirds (r : Rhombus150) :
  area_S r = 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_area_S_eq_four_sqrt_three_thirds_l3804_380484


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_stewart_farm_sheep_count_proof_l3804_380459

theorem stewart_farm_sheep_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun sheep_count horse_count sheep_ratio horse_ratio =>
    sheep_count * horse_ratio = horse_count * sheep_ratio ∧
    horse_count * 230 = 12880 →
    sheep_count = 40

-- The proof is omitted
theorem stewart_farm_sheep_count_proof : stewart_farm_sheep_count 40 56 5 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_stewart_farm_sheep_count_proof_l3804_380459


namespace NUMINAMATH_CALUDE_line_point_distance_l3804_380463

/-- Given a line x = 3y + 5 passing through points (m, n) and (m + d, n + p),
    where p = 0.6666666666666666, prove that d = 2 -/
theorem line_point_distance (m n d p : ℝ) : 
  p = 0.6666666666666666 →
  m = 3 * n + 5 →
  (m + d) = 3 * (n + p) + 5 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_line_point_distance_l3804_380463


namespace NUMINAMATH_CALUDE_sum_after_removal_l3804_380496

theorem sum_after_removal : 
  let initial_sum : ℚ := 10 * 1.11 + 11 * 1.01
  let removed_sum : ℚ := 2 * 1.01
  let final_sum : ℚ := initial_sum - removed_sum
  final_sum = 20.19 := by sorry

end NUMINAMATH_CALUDE_sum_after_removal_l3804_380496


namespace NUMINAMATH_CALUDE_unit_circle_sector_angle_l3804_380428

/-- In a unit circle, a sector with area 1 has a central angle of 2 radians -/
theorem unit_circle_sector_angle (r : ℝ) (area : ℝ) (angle : ℝ) : 
  r = 1 → area = 1 → angle = 2 * area / r → angle = 2 := by sorry

end NUMINAMATH_CALUDE_unit_circle_sector_angle_l3804_380428


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3804_380417

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 512 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 60 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3804_380417


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3804_380410

def num_math_books : ℕ := 4
def num_history_books : ℕ := 7

def ways_to_arrange_books : ℕ :=
  -- Ways to choose math books for the ends
  (num_math_books * (num_math_books - 1)) *
  -- Ways to choose and arrange 2 history books from 7
  (num_history_books * (num_history_books - 1)) *
  -- Ways to choose the third book (math or history)
  (num_math_books + num_history_books - 3) *
  -- Ways to permute the first three books
  6 *
  -- Ways to arrange the remaining 6 books
  (6 * 5 * 4 * 3 * 2 * 1)

theorem book_arrangement_theorem :
  ways_to_arrange_books = 19571200 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3804_380410


namespace NUMINAMATH_CALUDE_power_product_equality_l3804_380407

theorem power_product_equality : 3^2 * 5 * 7^2 * 11 = 24255 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l3804_380407


namespace NUMINAMATH_CALUDE_omar_egg_rolls_l3804_380406

theorem omar_egg_rolls (karen_rolls : ℕ) (total_rolls : ℕ) (omar_rolls : ℕ) : 
  karen_rolls = 229 → total_rolls = 448 → omar_rolls = total_rolls - karen_rolls → omar_rolls = 219 := by
  sorry

end NUMINAMATH_CALUDE_omar_egg_rolls_l3804_380406


namespace NUMINAMATH_CALUDE_circles_have_another_common_tangent_l3804_380445

-- Define the basic geometric objects
structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the given conditions
def Semicircle (k : Circle) (A B : Point) : Prop :=
  k.center = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) ∧
  k.radius = (((B.x - A.x)^2 + (B.y - A.y)^2)^(1/2)) / 2

def OnCircle (P : Point) (k : Circle) : Prop :=
  (P.x - k.center.x)^2 + (P.y - k.center.y)^2 = k.radius^2

def Perpendicular (C D : Point) (A B : Point) : Prop :=
  (B.x - A.x) * (C.x - D.x) + (B.y - A.y) * (C.y - D.y) = 0

def Incircle (k : Circle) (A B C : Point) : Prop :=
  -- Definition of incircle omitted for brevity
  sorry

def TouchesSegmentAndCircle (k : Circle) (C D : Point) (semicircle : Circle) : Prop :=
  -- Definition of touching segment and circle omitted for brevity
  sorry

def CommonTangent (k1 k2 k3 : Circle) (A B : Point) : Prop :=
  -- Definition of common tangent omitted for brevity
  sorry

-- Main theorem
theorem circles_have_another_common_tangent
  (k semicircle : Circle) (A B C D : Point) (k1 k2 k3 : Circle) :
  Semicircle semicircle A B →
  OnCircle C semicircle →
  C ≠ A ∧ C ≠ B →
  Perpendicular C D A B →
  Incircle k1 A B C →
  TouchesSegmentAndCircle k2 C D semicircle →
  TouchesSegmentAndCircle k3 C D semicircle →
  CommonTangent k1 k2 k3 A B →
  ∃ (E F : Point), E ≠ F ∧ CommonTangent k1 k2 k3 E F ∧ (E ≠ A ∨ F ≠ B) :=
by
  sorry

end NUMINAMATH_CALUDE_circles_have_another_common_tangent_l3804_380445


namespace NUMINAMATH_CALUDE_identity_function_only_solution_l3804_380411

theorem identity_function_only_solution 
  (f : ℕ+ → ℕ+) 
  (h : ∀ a b : ℕ+, (a - f b) ∣ (a * f a - b * f b)) :
  ∀ x : ℕ+, f x = x :=
by sorry

end NUMINAMATH_CALUDE_identity_function_only_solution_l3804_380411


namespace NUMINAMATH_CALUDE_deaf_to_blind_ratio_l3804_380446

theorem deaf_to_blind_ratio (total_students blind_students : ℕ) 
  (h1 : total_students = 180)
  (h2 : blind_students = 45) :
  (total_students - blind_students) / blind_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_deaf_to_blind_ratio_l3804_380446


namespace NUMINAMATH_CALUDE_decimal_places_of_fraction_l3804_380434

theorem decimal_places_of_fraction : ∃ (n : ℕ), 
  (5^5 : ℚ) / (10^3 * 8) = n / 10 ∧ n % 10 ≠ 0 ∧ n < 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_places_of_fraction_l3804_380434


namespace NUMINAMATH_CALUDE_number_multiplication_l3804_380469

theorem number_multiplication (x : ℝ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l3804_380469


namespace NUMINAMATH_CALUDE_unique_a_value_l3804_380448

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem unique_a_value : ∃! a : ℝ, (9 ∈ (A a ∩ B a)) ∧ ({9} = A a ∩ B a) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3804_380448


namespace NUMINAMATH_CALUDE_reggie_shopping_spree_l3804_380460

def initial_amount : ℕ := 150
def num_books : ℕ := 5
def book_price : ℕ := 12
def game_price : ℕ := 45
def bottle_price : ℕ := 13
def snack_price : ℕ := 7

theorem reggie_shopping_spree :
  initial_amount - (num_books * book_price + game_price + bottle_price + snack_price) = 25 := by
  sorry

end NUMINAMATH_CALUDE_reggie_shopping_spree_l3804_380460


namespace NUMINAMATH_CALUDE_rectangle_subdivision_l3804_380439

/-- A rectangle can be subdivided into n pairwise noncongruent rectangles similar to the original -/
def can_subdivide (n : ℕ) : Prop :=
  ∃ (r : ℝ) (h : r > 1), ∃ (rectangles : Fin n → ℝ × ℝ),
    (∀ i j, i ≠ j → rectangles i ≠ rectangles j) ∧
    (∀ i, (rectangles i).1 / (rectangles i).2 = r)

theorem rectangle_subdivision (n : ℕ) (h : n > 1) :
  can_subdivide n ↔ n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_rectangle_subdivision_l3804_380439


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3804_380425

theorem polynomial_factorization (p q : ℝ) :
  ∃ (a b c d e f : ℝ), ∀ (x : ℝ),
    x^4 + p*x^2 + q = (a*x^2 + b*x + c) * (d*x^2 + e*x + f) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3804_380425


namespace NUMINAMATH_CALUDE_circle_area_increase_l3804_380474

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3804_380474


namespace NUMINAMATH_CALUDE_work_completion_l3804_380429

/-- Given that 36 men can complete a piece of work in 25 hours,
    prove that 10 men can complete the same work in 90 hours. -/
theorem work_completion (work : ℝ) : 
  work = 36 * 25 → work = 10 * 90 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l3804_380429


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l3804_380402

/-- Calculates the distance traveled against the stream in one hour -/
def distance_against_stream (boat_speed : ℝ) (downstream_distance : ℝ) : ℝ :=
  let stream_speed := downstream_distance - boat_speed
  boat_speed - stream_speed

/-- Theorem: Given a boat with speed 4 km/hr in still water that travels 6 km
    downstream in one hour, it will travel 2 km upstream in one hour -/
theorem boat_upstream_distance :
  distance_against_stream 4 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l3804_380402


namespace NUMINAMATH_CALUDE_complex_product_given_pure_imaginary_sum_l3804_380420

theorem complex_product_given_pure_imaginary_sum (a : ℝ) : 
  let z₁ : ℂ := a - 2*I
  let z₂ : ℂ := -1 + a*I
  (∃ (b : ℝ), z₁ + z₂ = b*I) → z₁ * z₂ = 1 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_given_pure_imaginary_sum_l3804_380420


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3804_380498

theorem quadratic_inequality_solution_set :
  {x : ℝ | -2 * x^2 - x + 6 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3/2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3804_380498


namespace NUMINAMATH_CALUDE_two_different_color_balls_probability_two_different_color_balls_probability_proof_l3804_380424

theorem two_different_color_balls_probability 
  (total_balls : ℕ) 
  (red_balls yellow_balls white_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : yellow_balls = 2)
  (h4 : white_balls = 1)
  : ℚ :=
4/5

theorem two_different_color_balls_probability_proof 
  (total_balls : ℕ) 
  (red_balls yellow_balls white_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : yellow_balls = 2)
  (h4 : white_balls = 1)
  : two_different_color_balls_probability total_balls red_balls yellow_balls white_balls h1 h2 h3 h4 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_balls_probability_two_different_color_balls_probability_proof_l3804_380424


namespace NUMINAMATH_CALUDE_women_workers_l3804_380491

/-- Represents a company with workers and retirement plans. -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  women_without_plan : ℕ
  men_with_plan : ℕ
  total_men : ℕ

/-- Conditions for the company structure -/
def company_conditions (c : Company) : Prop :=
  c.workers_without_plan = c.total_workers / 3 ∧
  c.women_without_plan = (2 * c.workers_without_plan) / 5 ∧
  c.men_with_plan = ((2 * c.total_workers) / 3) * 2 / 5 ∧
  c.total_men = 120

/-- The theorem to prove -/
theorem women_workers (c : Company) 
  (h : company_conditions c) : c.total_workers - c.total_men = 330 := by
  sorry

#check women_workers

end NUMINAMATH_CALUDE_women_workers_l3804_380491
