import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_at_two_l3355_335557

theorem expression_value_at_two :
  let x : ℚ := 2
  (x^2 - x - 6) / (x - 3) = 4 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_two_l3355_335557


namespace NUMINAMATH_CALUDE_smallest_square_arrangement_l3355_335514

theorem smallest_square_arrangement : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬ ∃ k : ℕ+, m * (1^2 + 2^2 + 3^2) = k^2) ∧
  (∃ k : ℕ+, n * (1^2 + 2^2 + 3^2) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_arrangement_l3355_335514


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3355_335579

theorem solution_satisfies_system :
  let x : ℝ := 1
  let y : ℝ := 1/2
  let w : ℝ := -1/2
  let z : ℝ := 1/3
  (Real.sqrt x - 1/y - 2*w + 3*z = 1) ∧
  (x + 1/(y^2) - 4*(w^2) - 9*(z^2) = 3) ∧
  (x * Real.sqrt x - 1/(y^3) - 8*(w^3) + 27*(z^3) = -5) ∧
  (x^2 + 1/(y^4) - 16*(w^4) - 81*(z^4) = 15) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3355_335579


namespace NUMINAMATH_CALUDE_inequality_proof_l3355_335521

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3355_335521


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l3355_335567

/-- The trajectory of the center of a moving circle externally tangent to a fixed circle and the y-axis -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    -- The moving circle is externally tangent to (x-2)^2 + y^2 = 1
    ((x - 2)^2 + y^2 = (r + 1)^2) ∧ 
    -- The moving circle is tangent to the y-axis
    (x = r)) →
  y^2 = 6*x - 3 := by
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l3355_335567


namespace NUMINAMATH_CALUDE_vector_problem_l3355_335543

def a : ℝ × ℝ := (1, 3)
def b (y : ℝ) : ℝ × ℝ := (2, y)

theorem vector_problem (y : ℝ) :
  (∀ y, (a.1 * (b y).1 + a.2 * (b y).2 = 5) → y = 1) ∧
  (∀ y, ((a.1 + (b y).1)^2 + (a.2 + (b y).2)^2 = (a.1 - (b y).1)^2 + (a.2 - (b y).2)^2) → y = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3355_335543


namespace NUMINAMATH_CALUDE_two_digit_squares_mod_15_l3355_335542

theorem two_digit_squares_mod_15 : ∃ (S : Finset Nat), (∀ a ∈ S, 10 ≤ a ∧ a < 100 ∧ a^2 % 15 = 1) ∧ S.card = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_squares_mod_15_l3355_335542


namespace NUMINAMATH_CALUDE_equation_solution_l3355_335577

theorem equation_solution : ∃ x : ℚ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3355_335577


namespace NUMINAMATH_CALUDE_complex_on_negative_y_axis_l3355_335556

theorem complex_on_negative_y_axis (a : ℝ) : 
  (∃ y : ℝ, y < 0 ∧ (a + Complex.I)^2 = Complex.I * y) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_negative_y_axis_l3355_335556


namespace NUMINAMATH_CALUDE_problem_statement_l3355_335524

/-- Given two real numbers a and b with average 110, and b and c with average 170,
    if a - c = 120, then c = -120 -/
theorem problem_statement (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110)
  (h2 : (b + c) / 2 = 170)
  (h3 : a - c = 120) :
  c = -120 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3355_335524


namespace NUMINAMATH_CALUDE_herrings_caught_l3355_335531

/-- Given the total number of fish caught and the number of pikes and sturgeons,
    calculate the number of herrings caught. -/
theorem herrings_caught (total : ℕ) (pikes : ℕ) (sturgeons : ℕ) 
  (h1 : total = 145) (h2 : pikes = 30) (h3 : sturgeons = 40) :
  total - pikes - sturgeons = 75 := by
  sorry

end NUMINAMATH_CALUDE_herrings_caught_l3355_335531


namespace NUMINAMATH_CALUDE_gcd_180_270_l3355_335539

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l3355_335539


namespace NUMINAMATH_CALUDE_equation_roots_existence_l3355_335578

theorem equation_roots_existence :
  (∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0) ∧ 
  (∃ k : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0 ∧ 
    z^2 - 2*|z| - (2*k + 1)^2 = 0) ∧ 
  (¬ ∃ k : ℝ, ∃! x : ℝ, x^2 - 2*|x| - (2*k + 1)^2 = 0) ∧
  (¬ ∃ k : ℝ, ∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    w^2 - 2*|w| - (2*k + 1)^2 = 0 ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0 ∧ 
    z^2 - 2*|z| - (2*k + 1)^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_existence_l3355_335578


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3355_335534

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 3 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ (x y : ℝ),
  hyperbola x y →
  (∃ (a b c : ℝ),
    a = 2 * Real.sqrt 3 ∧
    c = 3 ∧
    b^2 = a^2 - c^2 ∧
    ellipse x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3355_335534


namespace NUMINAMATH_CALUDE_representatives_selection_count_l3355_335565

def num_female : ℕ := 3
def num_male : ℕ := 4
def num_representatives : ℕ := 3

theorem representatives_selection_count :
  (Finset.sum (Finset.range (num_representatives - 1)) (λ k =>
    Nat.choose num_female (k + 1) * Nat.choose num_male (num_representatives - k - 1)))
  = 30 := by sorry

end NUMINAMATH_CALUDE_representatives_selection_count_l3355_335565


namespace NUMINAMATH_CALUDE_club_probability_theorem_l3355_335500

theorem club_probability_theorem (total_members : ℕ) (boys : ℕ) (girls : ℕ) :
  total_members = 15 →
  boys = 8 →
  girls = 7 →
  total_members = boys + girls →
  (Nat.choose total_members 2 - Nat.choose girls 2 : ℚ) / Nat.choose total_members 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_club_probability_theorem_l3355_335500


namespace NUMINAMATH_CALUDE_q_equals_sixteen_l3355_335513

/-- The polynomial with four distinct real roots in geometric progression -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ := x^4 + p*x^3 + q*x^2 - 18*x + 16

/-- The roots of the polynomial form a geometric progression -/
def roots_in_geometric_progression (p q : ℝ) : Prop :=
  ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 1 ∧
    (polynomial p q a = 0) ∧
    (polynomial p q (a*r) = 0) ∧
    (polynomial p q (a*r^2) = 0) ∧
    (polynomial p q (a*r^3) = 0)

/-- The theorem stating that q equals 16 for the given conditions -/
theorem q_equals_sixteen (p q : ℝ) :
  roots_in_geometric_progression p q → q = 16 := by
  sorry

end NUMINAMATH_CALUDE_q_equals_sixteen_l3355_335513


namespace NUMINAMATH_CALUDE_min_value_squared_distance_l3355_335507

theorem min_value_squared_distance (a b c d : ℝ) 
  (h : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  ∃ (min : ℝ), min = (9 : ℝ) / 2 ∧ 
  ∀ (x y : ℝ), (x - y)^2 + (b - d)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_squared_distance_l3355_335507


namespace NUMINAMATH_CALUDE_ferris_wheel_large_seats_undetermined_l3355_335570

structure FerrisWheel where
  smallSeats : Nat
  smallSeatCapacity : Nat
  largeSeatCapacity : Nat
  peopleOnSmallSeats : Nat

theorem ferris_wheel_large_seats_undetermined (fw : FerrisWheel)
  (h1 : fw.smallSeats = 2)
  (h2 : fw.smallSeatCapacity = 14)
  (h3 : fw.largeSeatCapacity = 54)
  (h4 : fw.peopleOnSmallSeats = 28) :
  ∀ n : Nat, ∃ m : Nat, m ≠ n ∧ 
    (∃ totalSeats totalCapacity : Nat,
      totalSeats = fw.smallSeats + n ∧
      totalCapacity = fw.smallSeats * fw.smallSeatCapacity + m * fw.largeSeatCapacity) :=
sorry

end NUMINAMATH_CALUDE_ferris_wheel_large_seats_undetermined_l3355_335570


namespace NUMINAMATH_CALUDE_cosecant_330_degrees_l3355_335538

theorem cosecant_330_degrees :
  let csc (θ : ℝ) := 1 / Real.sin θ
  let π : ℝ := Real.pi
  ∀ (θ : ℝ), Real.sin (2 * π - θ) = -Real.sin θ
  → Real.sin (π / 6) = 1 / 2
  → csc (11 * π / 6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_cosecant_330_degrees_l3355_335538


namespace NUMINAMATH_CALUDE_range_of_a_l3355_335598

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Icc 1 4 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3355_335598


namespace NUMINAMATH_CALUDE_import_value_calculation_l3355_335535

theorem import_value_calculation (tax_free_limit : ℝ) (tax_rate : ℝ) (tax_paid : ℝ) : 
  tax_free_limit = 500 →
  tax_rate = 0.08 →
  tax_paid = 18.40 →
  ∃ total_value : ℝ, total_value = 730 ∧ tax_paid = tax_rate * (total_value - tax_free_limit) :=
by sorry

end NUMINAMATH_CALUDE_import_value_calculation_l3355_335535


namespace NUMINAMATH_CALUDE_equivalence_condition_l3355_335559

theorem equivalence_condition (a b : ℝ) (h : a * b > 0) :
  a > b ↔ 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_equivalence_condition_l3355_335559


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3355_335541

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 6 = 3 →
  a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3355_335541


namespace NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l3355_335523

theorem lcm_of_numbers_with_given_hcf_and_product :
  ∀ a b : ℕ+,
  (Nat.gcd a.val b.val = 11) →
  (a * b = 1991) →
  (Nat.lcm a.val b.val = 181) :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l3355_335523


namespace NUMINAMATH_CALUDE_coin_worth_l3355_335593

def total_coins : ℕ := 20
def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def swap_difference : ℚ := 70 / 100

theorem coin_worth (n : ℕ) (h1 : n ≤ total_coins) :
  (n : ℚ) * nickel_value + (total_coins - n : ℚ) * dime_value + swap_difference = 
  (n : ℚ) * dime_value + (total_coins - n : ℚ) * nickel_value →
  (n : ℚ) * nickel_value + (total_coins - n : ℚ) * dime_value = 115 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_worth_l3355_335593


namespace NUMINAMATH_CALUDE_fourth_root_plus_cube_root_equation_solutions_l3355_335572

theorem fourth_root_plus_cube_root_equation_solutions :
  ∀ x : ℝ, (((3 - x) ^ (1/4) : ℝ) + ((x - 2) ^ (1/3) : ℝ) = 1) ↔ (x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_plus_cube_root_equation_solutions_l3355_335572


namespace NUMINAMATH_CALUDE_ratio_max_min_sequence_diff_l3355_335586

def geometric_sequence (n : ℕ) : ℚ :=
  (3/2) * (-1/2) ^ (n - 1)

def sum_n_terms (n : ℕ) : ℚ :=
  (3/2) * (1 - (-1/2)^n) / (1 + 1/2)

def sequence_diff (n : ℕ) : ℚ :=
  sum_n_terms n - 1 / sum_n_terms n

theorem ratio_max_min_sequence_diff :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
    sequence_diff m / sequence_diff n = -10/7 ∧
    ∀ (k : ℕ), k > 0 → 
      sequence_diff m ≥ sequence_diff k ∧
      sequence_diff k ≥ sequence_diff n) :=
sorry

end NUMINAMATH_CALUDE_ratio_max_min_sequence_diff_l3355_335586


namespace NUMINAMATH_CALUDE_rose_work_days_l3355_335544

/-- Given that Paul completes a work in 80 days and Paul and Rose together
    complete the same work in 48 days, prove that Rose completes the work
    alone in 120 days. -/
theorem rose_work_days (paul_days : ℕ) (together_days : ℕ) (rose_days : ℕ) : 
  paul_days = 80 → together_days = 48 → 
  1 / paul_days + 1 / rose_days = 1 / together_days →
  rose_days = 120 := by
  sorry

end NUMINAMATH_CALUDE_rose_work_days_l3355_335544


namespace NUMINAMATH_CALUDE_grapes_and_watermelon_cost_l3355_335595

/-- The cost of a pack of peanuts -/
def peanuts_cost : ℝ := sorry

/-- The cost of a cluster of grapes -/
def grapes_cost : ℝ := sorry

/-- The cost of a watermelon -/
def watermelon_cost : ℝ := sorry

/-- The cost of a box of figs -/
def figs_cost : ℝ := sorry

/-- The total cost of all items -/
def total_cost : ℝ := 30

/-- The statement of the problem -/
theorem grapes_and_watermelon_cost :
  (peanuts_cost + grapes_cost + watermelon_cost + figs_cost = total_cost) →
  (figs_cost = 2 * peanuts_cost) →
  (watermelon_cost = peanuts_cost - grapes_cost) →
  (grapes_cost + watermelon_cost = 7.5) :=
by sorry

end NUMINAMATH_CALUDE_grapes_and_watermelon_cost_l3355_335595


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3355_335530

/-- Given a cycle bought for Rs. 930 and sold with a gain of 30.107526881720432%,
    prove that the selling price is Rs. 1210. -/
theorem cycle_selling_price (cost_price : ℝ) (gain_percentage : ℝ) (selling_price : ℝ) :
  cost_price = 930 →
  gain_percentage = 30.107526881720432 →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  selling_price = 1210 :=
by sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3355_335530


namespace NUMINAMATH_CALUDE_expression_simplification_l3355_335549

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3355_335549


namespace NUMINAMATH_CALUDE_part1_part2_l3355_335589

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define the solution set of f(x) < 0
def solution_set (a b : ℝ) : Set ℝ := {x | 2 < x ∧ x < 3}

-- Define the condition for part 2
def condition_part2 (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 0, f a (3 - a) x ≥ 0

-- Statement for part 1
theorem part1 (a b : ℝ) : 
  (∀ x, f a b x < 0 ↔ x ∈ solution_set a b) → 
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Statement for part 2
theorem part2 (a : ℝ) :
  condition_part2 a → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3355_335589


namespace NUMINAMATH_CALUDE_expression_value_l3355_335501

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : (x + 1 / x) * (z - 1 / z) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3355_335501


namespace NUMINAMATH_CALUDE_system_one_solutions_system_two_solutions_l3355_335590

-- System 1
theorem system_one_solutions (x y : ℝ) :
  (x^2 - 2*x = 0 ∧ x^3 + y = 6) ↔ ((x = 0 ∧ y = 6) ∨ (x = 2 ∧ y = -2)) :=
sorry

-- System 2
theorem system_two_solutions (x y : ℝ) :
  (y^2 - 4*y + 3 = 0 ∧ 2*x + y = 9) ↔ ((x = 4 ∧ y = 1) ∨ (x = 3 ∧ y = 3)) :=
sorry

end NUMINAMATH_CALUDE_system_one_solutions_system_two_solutions_l3355_335590


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l3355_335564

/-- The number of X tiles -/
def num_x : ℕ := 5

/-- The number of O tiles -/
def num_o : ℕ := 4

/-- The total number of tiles -/
def total_tiles : ℕ := num_x + num_o

/-- The probability of the specific arrangement -/
def prob_specific_arrangement : ℚ := 1 / (total_tiles.choose num_x)

theorem specific_arrangement_probability :
  prob_specific_arrangement = 1 / 126 := by sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l3355_335564


namespace NUMINAMATH_CALUDE_census_suitable_for_class_spirit_awareness_only_class_spirit_awareness_census_suitable_l3355_335545

/-- Represents a survey scenario --/
inductive SurveyScenario
  | ShellLethalRadius
  | TVViewershipRating
  | YellowRiverFishSpecies
  | ClassSpiritAwareness

/-- Determines if a census method is suitable for a given survey scenario --/
def isCensusSuitable (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.ClassSpiritAwareness => True
  | _ => False

/-- Theorem: The survey to ascertain the awareness rate of the "Shanxi Spirit" 
    among the students of a certain class is suitable for using a census method --/
theorem census_suitable_for_class_spirit_awareness :
  isCensusSuitable SurveyScenario.ClassSpiritAwareness :=
by sorry

/-- Theorem: The survey to ascertain the awareness rate of the "Shanxi Spirit" 
    among the students of a certain class is the only one suitable for using a census method --/
theorem only_class_spirit_awareness_census_suitable :
  ∀ (scenario : SurveyScenario), 
    isCensusSuitable scenario ↔ scenario = SurveyScenario.ClassSpiritAwareness :=
by sorry

end NUMINAMATH_CALUDE_census_suitable_for_class_spirit_awareness_only_class_spirit_awareness_census_suitable_l3355_335545


namespace NUMINAMATH_CALUDE_unicorn_to_witch_ratio_l3355_335566

/-- Represents the number of votes for each cake type -/
structure CakeVotes where
  unicorn : ℕ
  witch : ℕ
  dragon : ℕ

/-- The conditions of the baking contest voting -/
def baking_contest (votes : CakeVotes) : Prop :=
  votes.dragon = votes.witch + 25 ∧
  votes.witch = 7 ∧
  votes.unicorn + votes.witch + votes.dragon = 60

theorem unicorn_to_witch_ratio (votes : CakeVotes) 
  (h : baking_contest votes) : 
  votes.unicorn / votes.witch = 3 :=
sorry

end NUMINAMATH_CALUDE_unicorn_to_witch_ratio_l3355_335566


namespace NUMINAMATH_CALUDE_village_population_l3355_335597

theorem village_population (P : ℝ) : 
  P > 0 → 
  (P * 0.9 * 0.8 = 3240) → 
  P = 4500 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3355_335597


namespace NUMINAMATH_CALUDE_function_characterization_l3355_335515

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f n) - f m = n

-- Theorem statement
theorem function_characterization :
  ∀ f : ℤ → ℤ, SatisfiesProperty f →
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3355_335515


namespace NUMINAMATH_CALUDE_roots_of_g_l3355_335580

theorem roots_of_g (a b : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + b = 0) ↔ (x = 2 ∨ x = 3)) →
  (∀ x : ℝ, (b*x^2 - a*x - 1 = 0) ↔ (x = 1 ∨ x = -1/6)) := by
sorry

end NUMINAMATH_CALUDE_roots_of_g_l3355_335580


namespace NUMINAMATH_CALUDE_expression_evaluation_l3355_335558

theorem expression_evaluation : (20 - 16) * (12 + 8) / 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3355_335558


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l3355_335551

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold. -/
def TicketSales.total (s : TicketSales) : ℕ :=
  s.orchestra + s.balcony

/-- Calculates the total revenue from ticket sales. -/
def TicketSales.revenue (s : TicketSales) : ℕ :=
  12 * s.orchestra + 8 * s.balcony

theorem theater_ticket_difference (s : TicketSales) :
  s.total = 355 → s.revenue = 3320 → s.balcony - s.orchestra = 115 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l3355_335551


namespace NUMINAMATH_CALUDE_triangle_shape_determination_l3355_335532

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of two sides and the included angle of a triangle -/
def ratio_two_sides_and_angle (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratios of the three angle bisectors of a triangle -/
def ratio_angle_bisectors (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratios of the three medians of a triangle -/
def ratio_medians (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratio of the circumradius to the inradius of a triangle -/
def ratio_circumradius_to_inradius (t : Triangle) : ℝ := sorry

/-- Two angles of a triangle -/
def two_angles (t : Triangle) : ℝ × ℝ := sorry

/-- Two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop := sorry

/-- The shape of a triangle is uniquely determined by a given property
    if any two triangles with the same property are similar -/
def uniquely_determines_shape (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → are_similar t1 t2

theorem triangle_shape_determination :
  uniquely_determines_shape ratio_two_sides_and_angle ∧
  uniquely_determines_shape ratio_angle_bisectors ∧
  uniquely_determines_shape ratio_medians ∧
  ¬ uniquely_determines_shape ratio_circumradius_to_inradius ∧
  uniquely_determines_shape two_angles := by sorry

end NUMINAMATH_CALUDE_triangle_shape_determination_l3355_335532


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3355_335502

theorem expression_value_at_three :
  ∀ x : ℝ, x ≠ 2 → x = 3 → (x^2 - 5*x + 6) / (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3355_335502


namespace NUMINAMATH_CALUDE_polynomial_infinite_solutions_l3355_335533

theorem polynomial_infinite_solutions (P : ℤ → ℤ) (d : ℤ) :
  (∃ (a b : ℤ), ∀ x, P x = a * x + b) ∨ (∀ x, P x = P 0) ↔
  (∃ (S : Set (ℤ × ℤ)), (∀ (x y : ℤ), (x, y) ∈ S → x ≠ y) ∧ 
                         Set.Infinite S ∧
                         (∀ (x y : ℤ), (x, y) ∈ S → P x - P y = d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_infinite_solutions_l3355_335533


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3355_335529

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 40 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3355_335529


namespace NUMINAMATH_CALUDE_range_of_a_l3355_335553

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*a*x + 9 ≥ 0) → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3355_335553


namespace NUMINAMATH_CALUDE_chord_intersection_lengths_l3355_335571

theorem chord_intersection_lengths (r : ℝ) (AK CH : ℝ) :
  r = 7 →
  AK = 3 →
  CH = 12 →
  let KB := 2 * r - AK
  ∃ (CK KH : ℝ),
    CK + KH = CH ∧
    AK * KB = CK * KH ∧
    AK = 3 ∧
    KB = 11 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_lengths_l3355_335571


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3355_335519

-- Define the general form of a hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the given hyperbola with known asymptotes
def given_hyperbola (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 = 1

-- Define the given ellipse with known foci
def given_ellipse (x y : ℝ) : Prop :=
  x^2 / 20 + y^2 / 2 = 1

-- Theorem stating the equation of the hyperbola
theorem hyperbola_equation :
  ∃ (a b : ℝ),
    (∀ (x y : ℝ), hyperbola a b x y ↔ given_hyperbola x y) ∧
    (∀ (x : ℝ), x^2 = 18 → hyperbola a b x 0) ∧
    a = 4 ∧ b = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3355_335519


namespace NUMINAMATH_CALUDE_blueberry_jelly_amount_l3355_335511

theorem blueberry_jelly_amount (total_jelly strawberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : strawberry_jelly = 1792) :
  total_jelly - strawberry_jelly = 4518 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_jelly_amount_l3355_335511


namespace NUMINAMATH_CALUDE_rounded_number_accuracy_l3355_335554

/-- 
Given an approximate number obtained by rounding, represented as 6.18 × 10^4,
prove that it is accurate to the hundred place.
-/
theorem rounded_number_accuracy : 
  let rounded_number : ℝ := 6.18 * 10^4
  ∃ (exact_number : ℝ), 
    (abs (exact_number - rounded_number) ≤ 50) ∧ 
    (∀ (place : ℕ), place > 2 → 
      ∃ (n : ℤ), rounded_number = (n * 10^place : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_rounded_number_accuracy_l3355_335554


namespace NUMINAMATH_CALUDE_water_container_problem_l3355_335596

theorem water_container_problem :
  ∀ (x : ℝ),
    x > 0 →
    x / 2 + (2 * x) / 3 + (4 * x) / 4 = 26 →
    x + 2 * x + 4 * x + 26 = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_water_container_problem_l3355_335596


namespace NUMINAMATH_CALUDE_power_of_three_decomposition_l3355_335552

theorem power_of_three_decomposition : 3^25 = 27^7 * 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_decomposition_l3355_335552


namespace NUMINAMATH_CALUDE_largest_quantity_l3355_335583

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l3355_335583


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3355_335537

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3355_335537


namespace NUMINAMATH_CALUDE_problem_solution_l3355_335584

theorem problem_solution :
  let x : ℝ := -39660 - 17280 * Real.sqrt 2
  (x + 720 * Real.sqrt 1152) / Real.rpow 15625 (1/3) = 7932 / (3^2 - Real.sqrt 196) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3355_335584


namespace NUMINAMATH_CALUDE_thomas_score_l3355_335527

def class_size : ℕ := 20
def initial_average : ℚ := 78
def final_average : ℚ := 79

theorem thomas_score :
  ∃ (score : ℚ),
    (class_size - 1) * initial_average + score = class_size * final_average ∧
    score = 98 := by
  sorry

end NUMINAMATH_CALUDE_thomas_score_l3355_335527


namespace NUMINAMATH_CALUDE_no_rational_solution_l3355_335575

theorem no_rational_solution (n : ℕ+) : ¬ ∃ (x y : ℚ), 0 < x ∧ 0 < y ∧ x + y + 1/x + 1/y = 3*n := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l3355_335575


namespace NUMINAMATH_CALUDE_camping_products_costs_l3355_335546

/-- The wholesale cost of a sleeping bag -/
def sleeping_bag_cost : ℚ := 560 / 23

/-- The wholesale cost of a tent -/
def tent_cost : ℚ := 200 / 3

/-- The selling price of a sleeping bag -/
def sleeping_bag_price : ℚ := 28

/-- The selling price of a tent -/
def tent_price : ℚ := 80

/-- The gross profit percentage for sleeping bags -/
def sleeping_bag_profit_percent : ℚ := 15 / 100

/-- The gross profit percentage for tents -/
def tent_profit_percent : ℚ := 20 / 100

theorem camping_products_costs :
  (sleeping_bag_cost * (1 + sleeping_bag_profit_percent) = sleeping_bag_price) ∧
  (tent_cost * (1 + tent_profit_percent) = tent_price) := by
  sorry

end NUMINAMATH_CALUDE_camping_products_costs_l3355_335546


namespace NUMINAMATH_CALUDE_max_y_rectangular_prism_l3355_335576

/-- The maximum value of y for a rectangular prism with volume 360 and integer dimensions x, y, z satisfying 1 < z < y < x -/
theorem max_y_rectangular_prism : 
  ∀ x y z : ℕ, 
  x * y * z = 360 → 
  1 < z → z < y → y < x → 
  y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_y_rectangular_prism_l3355_335576


namespace NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_100_l3355_335504

/-- Calculates the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (total_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : ℚ :=
  let cost_per_bar := (2 : ℚ) / 5
  let sell_per_bar := (1 : ℚ) / 2
  let total_cost := total_bars * cost_per_bar
  let total_revenue := total_bars * sell_per_bar
  total_revenue - total_cost

/-- Proves that the scout troop's profit is $100 -/
theorem scout_troop_profit_is_100 :
  scout_troop_profit 1000 ((2 : ℚ) / 5) ((1 : ℚ) / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_100_l3355_335504


namespace NUMINAMATH_CALUDE_max_y_coordinate_l3355_335587

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l3355_335587


namespace NUMINAMATH_CALUDE_count_integers_in_range_l3355_335599

theorem count_integers_in_range : ∃ (S : Finset ℤ), 
  (∀ n : ℤ, n ∈ S ↔ -12 * Real.sqrt Real.pi ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ 15 * Real.pi) ∧ 
  Finset.card S = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l3355_335599


namespace NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l3355_335509

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of suits
def num_suits : ℕ := 4

-- Define the number of cards drawn
def cards_drawn : ℕ := 5

-- Define the probability of drawing a card from a specific suit
def prob_suit : ℚ := 1 / 4

-- Theorem statement
theorem prob_all_suits_in_five_draws :
  let prob_sequence := (1 : ℚ) * (3 / 4) * (1 / 2) * (1 / 4)
  let num_sequences := 24
  (prob_sequence * num_sequences : ℚ) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l3355_335509


namespace NUMINAMATH_CALUDE_max_distance_with_tire_swap_l3355_335506

/-- Represents the maximum distance a car can travel with tire swapping -/
def maxDistanceWithTireSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + (rearTireLife - frontTireLife)

/-- Theorem: The maximum distance a car can travel with given tire lifespans -/
theorem max_distance_with_tire_swap :
  maxDistanceWithTireSwap 20000 30000 = 30000 := by
  sorry

#eval maxDistanceWithTireSwap 20000 30000

end NUMINAMATH_CALUDE_max_distance_with_tire_swap_l3355_335506


namespace NUMINAMATH_CALUDE_holiday_customers_l3355_335510

def normal_rate : ℕ := 175
def holiday_multiplier : ℕ := 2
def hours : ℕ := 8

theorem holiday_customers :
  normal_rate * holiday_multiplier * hours = 2800 :=
by sorry

end NUMINAMATH_CALUDE_holiday_customers_l3355_335510


namespace NUMINAMATH_CALUDE_partner_b_contribution_l3355_335555

/-- Represents the capital contribution of a business partner -/
structure Capital where
  amount : ℝ
  duration : ℕ

/-- Calculates the adjusted capital contribution -/
def adjustedCapital (c : Capital) : ℝ := c.amount * c.duration

/-- Represents the profit-sharing ratio between two partners -/
structure ProfitRatio where
  partner1 : ℕ
  partner2 : ℕ

theorem partner_b_contribution 
  (a : Capital) 
  (b : Capital)
  (ratio : ProfitRatio)
  (h1 : a.amount = 3500)
  (h2 : a.duration = 12)
  (h3 : b.duration = 7)
  (h4 : ratio.partner1 = 2)
  (h5 : ratio.partner2 = 3)
  (h6 : (adjustedCapital a) / (adjustedCapital b) = ratio.partner1 / ratio.partner2) :
  b.amount = 4500 := by
  sorry

end NUMINAMATH_CALUDE_partner_b_contribution_l3355_335555


namespace NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l3355_335592

def selling_price : ℚ := 1170
def cost_price : ℚ := 975

theorem profit_percentage_is_20_percent :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l3355_335592


namespace NUMINAMATH_CALUDE_zach_remaining_amount_l3355_335588

/-- Represents the financial situation for Zach's bike purchase --/
structure BikeSavings where
  bike_cost : ℕ
  weekly_allowance : ℕ
  lawn_mowing_pay : ℕ
  babysitting_rate : ℕ
  current_savings : ℕ
  babysitting_hours : ℕ

/-- Calculates the remaining amount needed to buy the bike --/
def remaining_amount (s : BikeSavings) : ℕ :=
  s.bike_cost - (s.current_savings + s.weekly_allowance + s.lawn_mowing_pay + s.babysitting_rate * s.babysitting_hours)

/-- Theorem stating the remaining amount Zach needs to earn --/
theorem zach_remaining_amount :
  let s : BikeSavings := {
    bike_cost := 100,
    weekly_allowance := 5,
    lawn_mowing_pay := 10,
    babysitting_rate := 7,
    current_savings := 65,
    babysitting_hours := 2
  }
  remaining_amount s = 6 := by sorry

end NUMINAMATH_CALUDE_zach_remaining_amount_l3355_335588


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_2_sqrt_3_l3355_335573

theorem abs_m_minus_n_equals_2_sqrt_3 (m n p : ℝ) 
  (h1 : m * n = 6)
  (h2 : m + n + p = 7)
  (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_2_sqrt_3_l3355_335573


namespace NUMINAMATH_CALUDE_soda_price_ratio_l3355_335561

theorem soda_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let brand_x_volume := 1.3 * v
  let brand_x_price := 0.85 * p
  (brand_x_price / brand_x_volume) / (p / v) = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l3355_335561


namespace NUMINAMATH_CALUDE_sequence_formula_l3355_335540

/-- For a sequence {a_n} where a_1 = 1 and a_{n+1} = 2^n * a_n for n ≥ 1,
    a_n = 2^(n*(n-1)/2) for all n ≥ 1 -/
theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2^n * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n*(n-1)/2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l3355_335540


namespace NUMINAMATH_CALUDE_lowest_possible_score_l3355_335568

-- Define the parameters of the problem
def mean : ℝ := 60
def std_dev : ℝ := 10
def z_score_95_percentile : ℝ := 1.645

-- Define the function to calculate the score from z-score
def score_from_z (z : ℝ) : ℝ := z * std_dev + mean

-- Define the conditions
def within_top_5_percent (score : ℝ) : Prop := score ≥ score_from_z z_score_95_percentile
def within_2_std_dev (score : ℝ) : Prop := score ≤ mean + 2 * std_dev

-- The theorem to prove
theorem lowest_possible_score :
  ∃ (score : ℕ), 
    (score : ℝ) = ⌈score_from_z z_score_95_percentile⌉ ∧
    within_top_5_percent score ∧
    within_2_std_dev score ∧
    ∀ (s : ℕ), s < score → ¬(within_top_5_percent s ∧ within_2_std_dev s) :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l3355_335568


namespace NUMINAMATH_CALUDE_square_sides_theorem_l3355_335548

theorem square_sides_theorem (total_length : ℝ) (area_difference : ℝ) 
  (h1 : total_length = 20)
  (h2 : area_difference = 120) :
  ∃ (x y : ℝ), x + y = total_length ∧ x^2 - y^2 = area_difference ∧ x = 13 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sides_theorem_l3355_335548


namespace NUMINAMATH_CALUDE_johns_piano_expenses_l3355_335591

/-- The total cost of John's piano learning expenses --/
def total_cost (piano_cost lesson_count lesson_price discount sheet_music maintenance : ℚ) : ℚ :=
  piano_cost + 
  (lesson_count * lesson_price * (1 - discount)) + 
  sheet_music + 
  maintenance

/-- Theorem stating that John's total piano learning expenses are $1275 --/
theorem johns_piano_expenses : 
  total_cost 500 20 40 (25/100) 75 100 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_johns_piano_expenses_l3355_335591


namespace NUMINAMATH_CALUDE_m_equals_2_sufficient_not_necessary_l3355_335547

def M (m : ℝ) : Set ℝ := {-1, m^2}
def N : Set ℝ := {2, 4}

theorem m_equals_2_sufficient_not_necessary :
  ∃ m : ℝ, (M m ∩ N = {4} ∧ m ≠ 2) ∧
  ∀ m : ℝ, m = 2 → M m ∩ N = {4} :=
sorry

end NUMINAMATH_CALUDE_m_equals_2_sufficient_not_necessary_l3355_335547


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3355_335562

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3355_335562


namespace NUMINAMATH_CALUDE_tim_needs_72_keys_l3355_335536

/-- The number of keys Tim needs to make for his rental properties -/
def total_keys (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_apartment : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_apartment

/-- Theorem stating that Tim needs 72 keys for his rental properties -/
theorem tim_needs_72_keys :
  total_keys 2 12 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tim_needs_72_keys_l3355_335536


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3355_335563

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 5 + a 10 = 12 → 3 * a 7 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3355_335563


namespace NUMINAMATH_CALUDE_cat_weight_sum_l3355_335520

/-- The combined weight of three cats -/
def combined_weight (w1 w2 w3 : ℕ) : ℕ := w1 + w2 + w3

/-- Theorem: The combined weight of three cats weighing 2, 7, and 4 pounds is 13 pounds -/
theorem cat_weight_sum : combined_weight 2 7 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_sum_l3355_335520


namespace NUMINAMATH_CALUDE_expression_equality_l3355_335550

theorem expression_equality : 
  Real.sqrt 12 - 3 * Real.tan (30 * π / 180) - (1 - π) ^ 0 + abs (-Real.sqrt 3) = 2 * Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3355_335550


namespace NUMINAMATH_CALUDE_seating_arrangement_exists_l3355_335522

-- Define a type for people
def Person : Type := Fin 5

-- Define a relation for acquaintance
def Acquainted : Person → Person → Prop := sorry

-- Define the condition that among any 3 people, 2 know each other and 2 don't
axiom acquaintance_condition : 
  ∀ (a b c : Person), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ((Acquainted a b ∧ Acquainted a c) ∨ 
     (Acquainted a b ∧ Acquainted b c) ∨ 
     (Acquainted a c ∧ Acquainted b c)) ∧
    ((¬Acquainted a b ∧ ¬Acquainted a c) ∨ 
     (¬Acquainted a b ∧ ¬Acquainted b c) ∨ 
     (¬Acquainted a c ∧ ¬Acquainted b c))

-- Define a circular arrangement
def CircularArrangement : Type := Fin 5 → Person

-- Define the property that each person is adjacent to two acquaintances
def ValidArrangement (arr : CircularArrangement) : Prop :=
  ∀ (i : Fin 5), 
    Acquainted (arr i) (arr ((i + 1) % 5)) ∧ 
    Acquainted (arr i) (arr ((i + 4) % 5))

-- The theorem to be proved
theorem seating_arrangement_exists : 
  ∃ (arr : CircularArrangement), ValidArrangement arr :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_exists_l3355_335522


namespace NUMINAMATH_CALUDE_segment_count_after_16_iterations_l3355_335505

/-- The number of segments after n iterations of the division process -/
def num_segments (n : ℕ) : ℕ := 2^n

/-- The length of each segment after n iterations of the division process -/
def segment_length (n : ℕ) : ℚ := (1 : ℚ) / 3^n

theorem segment_count_after_16_iterations :
  num_segments 16 = 2^16 := by sorry

end NUMINAMATH_CALUDE_segment_count_after_16_iterations_l3355_335505


namespace NUMINAMATH_CALUDE_negative_x_count_l3355_335518

theorem negative_x_count : 
  ∃ (S : Finset ℤ), 
    (∀ x ∈ S, x < 0 ∧ ∃ n : ℕ+, (x + 196 : ℝ) = n^2) ∧ 
    (∀ x : ℤ, x < 0 → (∃ n : ℕ+, (x + 196 : ℝ) = n^2) → x ∈ S) ∧
    Finset.card S = 13 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_count_l3355_335518


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l3355_335526

theorem quadratic_roots_expression (m n : ℝ) : 
  m ^ 2 + 2015 * m - 1 = 0 ∧ n ^ 2 + 2015 * n - 1 = 0 → m ^ 2 * n + m * n ^ 2 - m * n = 2016 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l3355_335526


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l3355_335525

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  1 - (Nat.choose men selected / Nat.choose total_people selected) = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l3355_335525


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3355_335516

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) → 
  (a < -2 ∨ a ≥ 6/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3355_335516


namespace NUMINAMATH_CALUDE_function_range_implies_a_value_l3355_335582

theorem function_range_implies_a_value (a : ℝ) (h1 : a > 0) : 
  (∀ x ∈ Set.Icc a (2 * a), (8 / x) ∈ Set.Icc (a / 4) 2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_range_implies_a_value_l3355_335582


namespace NUMINAMATH_CALUDE_average_waiting_time_is_nineteen_sixths_l3355_335508

/-- Represents a bus schedule with a given departure interval -/
structure BusSchedule where
  interval : ℕ

/-- Calculates the average waiting time for a set of bus schedules -/
def averageWaitingTime (schedules : List BusSchedule) : ℚ :=
  sorry

/-- Theorem stating that the average waiting time for the given bus schedules is 19/6 minutes -/
theorem average_waiting_time_is_nineteen_sixths :
  let schedules := [
    BusSchedule.mk 10,  -- Bus A
    BusSchedule.mk 12,  -- Bus B
    BusSchedule.mk 15   -- Bus C
  ]
  averageWaitingTime schedules = 19 / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_waiting_time_is_nineteen_sixths_l3355_335508


namespace NUMINAMATH_CALUDE_assembled_figure_surface_area_l3355_335574

/-- The surface area of a figure assembled from four identical bars -/
def figureSurfaceArea (barSurfaceArea : ℝ) (lostAreaPerJunction : ℝ) : ℝ :=
  4 * (barSurfaceArea - lostAreaPerJunction)

/-- Theorem: The surface area of the assembled figure is 64 cm² -/
theorem assembled_figure_surface_area :
  figureSurfaceArea 18 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_assembled_figure_surface_area_l3355_335574


namespace NUMINAMATH_CALUDE_probability_sum_five_l3355_335594

/-- The probability of the sum of four standard dice rolls equaling 5 -/
def prob_sum_five : ℚ := 1 / 324

/-- The number of faces on a standard die -/
def standard_die_faces : ℕ := 6

/-- The minimum value on a standard die -/
def min_die_value : ℕ := 1

/-- The maximum value on a standard die -/
def max_die_value : ℕ := 6

/-- A function representing a valid die roll -/
def valid_roll (n : ℕ) : Prop := min_die_value ≤ n ∧ n ≤ max_die_value

/-- The sum we're looking for -/
def target_sum : ℕ := 5

/-- The number of dice rolled -/
def num_dice : ℕ := 4

theorem probability_sum_five :
  ∀ (a b c d : ℕ), valid_roll a → valid_roll b → valid_roll c → valid_roll d →
  (a + b + c + d = target_sum) →
  (prob_sum_five = (↑(Nat.choose num_dice 1) / ↑(standard_die_faces ^ num_dice) : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l3355_335594


namespace NUMINAMATH_CALUDE_estimated_y_value_l3355_335512

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 0.50 * x - 0.81

theorem estimated_y_value (x : ℝ) (h : x = 25) : linear_regression x = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_value_l3355_335512


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3355_335517

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle --/
def isScalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem --/
theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    a ≥ 5 → b ≥ 5 → c ≥ 5 →
    isPrime a → isPrime b → isPrime c →
    isScalene a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3355_335517


namespace NUMINAMATH_CALUDE_bugs_meeting_point_l3355_335581

/-- A quadrilateral with sides of length 5, 7, 8, and 6 -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (ab_length : dist A B = 5)
  (bc_length : dist B C = 7)
  (cd_length : dist C D = 8)
  (da_length : dist D A = 6)

/-- The point where two bugs meet when starting from A and moving in opposite directions -/
def meeting_point (q : Quadrilateral) : ℝ × ℝ := sorry

/-- The distance between point B and the meeting point E -/
def BE (q : Quadrilateral) : ℝ := dist q.B (meeting_point q)

theorem bugs_meeting_point (q : Quadrilateral) : BE q = 6 := by sorry

end NUMINAMATH_CALUDE_bugs_meeting_point_l3355_335581


namespace NUMINAMATH_CALUDE_fraction_zero_l3355_335569

theorem fraction_zero (x : ℝ) (h : x ≠ -3) : (x^2 - 9) / (x + 3) = 0 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l3355_335569


namespace NUMINAMATH_CALUDE_football_group_stage_teams_l3355_335503

/-- The number of participating teams in the football group stage -/
def num_teams : ℕ := 16

/-- The number of stadiums used -/
def num_stadiums : ℕ := 6

/-- The number of games scheduled at each stadium per day -/
def games_per_stadium_per_day : ℕ := 4

/-- The number of consecutive days to complete all group stage matches -/
def num_days : ℕ := 10

theorem football_group_stage_teams :
  num_teams * (num_teams - 1) = num_stadiums * games_per_stadium_per_day * num_days :=
by sorry

end NUMINAMATH_CALUDE_football_group_stage_teams_l3355_335503


namespace NUMINAMATH_CALUDE_percentage_of_fair_haired_women_l3355_335585

theorem percentage_of_fair_haired_women (total : ℝ) 
  (h1 : total > 0) 
  (fair_haired_ratio : ℝ) 
  (h2 : fair_haired_ratio = 0.75)
  (women_ratio_among_fair_haired : ℝ) 
  (h3 : women_ratio_among_fair_haired = 0.40) : 
  (fair_haired_ratio * women_ratio_among_fair_haired) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_fair_haired_women_l3355_335585


namespace NUMINAMATH_CALUDE_sum_of_angles_satisfying_equation_l3355_335528

theorem sum_of_angles_satisfying_equation (x : Real) : 
  (0 ≤ x ∧ x ≤ 2 * Real.pi) →
  (Real.sin x ^ 3 + Real.cos x ^ 3 = 1 / Real.cos x + 1 / Real.sin x) →
  ∃ (y : Real), (0 ≤ y ∧ y ≤ 2 * Real.pi) ∧
    (Real.sin y ^ 3 + Real.cos y ^ 3 = 1 / Real.cos y + 1 / Real.sin y) ∧
    (x + y = 3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_angles_satisfying_equation_l3355_335528


namespace NUMINAMATH_CALUDE_complex_power_pure_integer_l3355_335560

def i : ℂ := Complex.I

theorem complex_power_pure_integer :
  ∃ (n : ℤ), ∃ (m : ℤ), (3 * n + 2 * i) ^ 6 = m := by
  sorry

end NUMINAMATH_CALUDE_complex_power_pure_integer_l3355_335560
