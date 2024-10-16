import Mathlib

namespace NUMINAMATH_CALUDE_min_distinct_values_l1146_114616

theorem min_distinct_values (list_size : ℕ) (mode_count : ℕ) (min_distinct : ℕ) : 
  list_size = 3045 →
  mode_count = 15 →
  min_distinct = 218 →
  (∀ n : ℕ, n < min_distinct → 
    n * (mode_count - 1) + mode_count < list_size) ∧
  min_distinct * (mode_count - 1) + mode_count ≥ list_size :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l1146_114616


namespace NUMINAMATH_CALUDE_vacation_cost_sharing_l1146_114610

theorem vacation_cost_sharing (john_paid mary_paid lisa_paid : ℝ) (j m : ℝ) : 
  john_paid = 150 →
  mary_paid = 90 →
  lisa_paid = 210 →
  j = 150 - john_paid →
  m = 150 - mary_paid →
  j - m = -60 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_sharing_l1146_114610


namespace NUMINAMATH_CALUDE_bracket_6_times_3_l1146_114675

-- Define the custom bracket operation
def bracket (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

-- Theorem statement
theorem bracket_6_times_3 : bracket 6 * bracket 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_bracket_6_times_3_l1146_114675


namespace NUMINAMATH_CALUDE_coloring_books_per_shelf_l1146_114688

theorem coloring_books_per_shelf
  (initial_stock : ℕ)
  (books_sold : ℕ)
  (num_shelves : ℕ)
  (h1 : initial_stock = 86)
  (h2 : books_sold = 37)
  (h3 : num_shelves = 7)
  : (initial_stock - books_sold) / num_shelves = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_books_per_shelf_l1146_114688


namespace NUMINAMATH_CALUDE_rectangle_similarity_symmetry_l1146_114664

/-- A rectangle with dimensions width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Congruence relation between rectangles -/
def congruent (r1 r2 : Rectangle) : Prop :=
  r1.width = r2.width ∧ r1.height = r2.height

/-- Similarity relation between rectangles -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- A composite rectangle formed from multiple congruent rectangles -/
structure CompositeRectangle where
  base : Rectangle
  width_multiplier : ℕ
  height_multiplier : ℕ

/-- The resulting rectangle from a composite rectangle -/
def resultingRectangle (cr : CompositeRectangle) : Rectangle :=
  { width := cr.base.width * cr.width_multiplier,
    height := cr.base.height * cr.height_multiplier }

/-- Main theorem: If rectangles congruent to A can form a rectangle similar to B,
    then rectangles congruent to B can form a rectangle similar to A -/
theorem rectangle_similarity_symmetry (A B : Rectangle)
  (h : ∃ (cr : CompositeRectangle), congruent cr.base A ∧ similar (resultingRectangle cr) B) :
  ∃ (cr : CompositeRectangle), congruent cr.base B ∧ similar (resultingRectangle cr) A :=
sorry

end NUMINAMATH_CALUDE_rectangle_similarity_symmetry_l1146_114664


namespace NUMINAMATH_CALUDE_francis_fruit_cups_l1146_114648

/-- The cost of a breakfast given the number of muffins and fruit cups -/
def breakfast_cost (muffins fruit_cups : ℕ) : ℕ := 2 * muffins + 3 * fruit_cups

/-- The problem statement -/
theorem francis_fruit_cups : ∃ f : ℕ, 
  breakfast_cost 2 f + breakfast_cost 2 1 = 17 ∧ f = 2 := by sorry

end NUMINAMATH_CALUDE_francis_fruit_cups_l1146_114648


namespace NUMINAMATH_CALUDE_work_completion_time_l1146_114646

/-- Given workers A, B, and C, where:
    - A can complete the work in 6 days
    - C can complete the work in 7.5 days
    - A, B, and C together complete the work in 2 days
    Prove that B can complete the work alone in 5 days -/
theorem work_completion_time (A B C : ℝ) 
  (hA : A = 1 / 6)  -- A's work rate per day
  (hC : C = 1 / 7.5)  -- C's work rate per day
  (hABC : A + B + C = 1 / 2)  -- Combined work rate of A, B, and C
  : B = 1 / 5 := by  -- B's work rate per day
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1146_114646


namespace NUMINAMATH_CALUDE_dog_tail_length_l1146_114662

/-- Represents the length of a dog's body parts in inches -/
structure DogMeasurements where
  overall : ℝ
  body : ℝ
  head : ℝ
  tail : ℝ

/-- Theorem stating the length of a dog's tail given specific proportions -/
theorem dog_tail_length (d : DogMeasurements) 
  (h1 : d.overall = 30)
  (h2 : d.tail = d.body / 2)
  (h3 : d.head = d.body / 6)
  (h4 : d.overall = d.head + d.body + d.tail) : 
  d.tail = 6 := by
  sorry

#check dog_tail_length

end NUMINAMATH_CALUDE_dog_tail_length_l1146_114662


namespace NUMINAMATH_CALUDE_shooter_probability_l1146_114605

theorem shooter_probability (p_10 p_9 p_8 : Real) 
  (h1 : p_10 = 0.20)
  (h2 : p_9 = 0.30)
  (h3 : p_8 = 0.10) :
  1 - (p_10 + p_9 + p_8) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l1146_114605


namespace NUMINAMATH_CALUDE_game_probability_l1146_114640

/-- A game with 8 rounds where one person wins each round -/
structure Game where
  rounds : Nat
  alex_prob : ℚ
  mel_prob : ℚ
  chelsea_prob : ℚ

/-- The probability of a specific outcome in the game -/
def outcome_probability (g : Game) (alex_wins mel_wins chelsea_wins : Nat) : ℚ :=
  (g.alex_prob ^ alex_wins) * (g.mel_prob ^ mel_wins) * (g.chelsea_prob ^ chelsea_wins) *
  (Nat.choose g.rounds alex_wins).choose mel_wins

/-- The theorem to be proved -/
theorem game_probability (g : Game) :
  g.rounds = 8 →
  g.alex_prob = 1/2 →
  g.mel_prob = g.chelsea_prob →
  g.alex_prob + g.mel_prob + g.chelsea_prob = 1 →
  outcome_probability g 4 3 1 = 35/512 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l1146_114640


namespace NUMINAMATH_CALUDE_election_votes_l1146_114625

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) : 
  total_votes = 9000 →
  invalid_percent = 30 / 100 →
  winner_percent = 60 / 100 →
  (total_votes : ℚ) * (1 - invalid_percent) * (1 - winner_percent) = 2520 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l1146_114625


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1146_114622

theorem set_intersection_problem (a : ℝ) : 
  let A : Set ℝ := {-4, 2*a-1, a^2}
  let B : Set ℝ := {a-5, 1-a, 9}
  9 ∈ (A ∩ B) → a = 5 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1146_114622


namespace NUMINAMATH_CALUDE_sheepdog_catch_time_l1146_114660

theorem sheepdog_catch_time (sheep_speed dog_speed initial_distance : ℝ) 
  (h1 : sheep_speed = 16)
  (h2 : dog_speed = 28)
  (h3 : initial_distance = 240) : 
  initial_distance / (dog_speed - sheep_speed) = 20 := by
  sorry

#check sheepdog_catch_time

end NUMINAMATH_CALUDE_sheepdog_catch_time_l1146_114660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1146_114642

/-- An arithmetic sequence with the given property -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_prop : ∀ n : ℕ+, a (n + 1) + a (n + 2) = 3 * (n : ℚ) + 5) :
  a 1 = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1146_114642


namespace NUMINAMATH_CALUDE_basketball_spectators_l1146_114611

theorem basketball_spectators 
  (total_spectators : ℕ) 
  (men : ℕ) 
  (ratio_children : ℕ) 
  (ratio_women : ℕ) : 
  total_spectators = 25000 →
  men = 15320 →
  ratio_children = 7 →
  ratio_women = 3 →
  (total_spectators - men) * ratio_children / (ratio_children + ratio_women) = 6776 :=
by sorry

end NUMINAMATH_CALUDE_basketball_spectators_l1146_114611


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1146_114613

/-- Given that x is inversely proportional to y, this theorem proves that
    if x₁/x₂ = 3/5, then y₁/y₂ = 5/3, where y₁ and y₂ are the corresponding
    y values for x₁ and x₂. -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, x * y = k) →  -- x is inversely proportional to y
  x₁ / x₂ = 3 / 5 →
  y₁ / y₂ = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1146_114613


namespace NUMINAMATH_CALUDE_sheets_count_l1146_114676

/-- The number of sheets in a set of writing materials -/
def S : ℕ := sorry

/-- The number of envelopes in a set of writing materials -/
def E : ℕ := sorry

/-- John's equation: sheets minus envelopes equals 80 -/
axiom john_equation : S - E = 80

/-- Mary's equation: sheets equals 4 times envelopes -/
axiom mary_equation : S = 4 * E

/-- Theorem: The number of sheets in each set is 320 -/
theorem sheets_count : S = 320 := by sorry

end NUMINAMATH_CALUDE_sheets_count_l1146_114676


namespace NUMINAMATH_CALUDE_equation_solution_l1146_114691

theorem equation_solution :
  ∃! x : ℚ, (x ≠ 3 ∧ x ≠ -2) ∧ (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 :=
by
  use (-7/6)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1146_114691


namespace NUMINAMATH_CALUDE_plane_parallel_from_skew_lines_l1146_114651

-- Define the types for planes and lines
variable (α β : Plane) (L m : Line)

-- Define the parallel relation between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the parallel relation between planes
def parallel_plane (p1 p2 : Plane) : Prop := sorry

-- Define skew lines
def skew_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem plane_parallel_from_skew_lines 
  (h_skew : skew_lines L m) 
  (h_L_alpha : parallel_line_plane L α)
  (h_m_alpha : parallel_line_plane m α)
  (h_L_beta : parallel_line_plane L β)
  (h_m_beta : parallel_line_plane m β) :
  parallel_plane α β := sorry

end NUMINAMATH_CALUDE_plane_parallel_from_skew_lines_l1146_114651


namespace NUMINAMATH_CALUDE_total_wheels_four_wheelers_l1146_114629

theorem total_wheels_four_wheelers (num_four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) :
  num_four_wheelers = 11 →
  wheels_per_four_wheeler = 4 →
  num_four_wheelers * wheels_per_four_wheeler = 44 :=
by sorry

end NUMINAMATH_CALUDE_total_wheels_four_wheelers_l1146_114629


namespace NUMINAMATH_CALUDE_square_differences_l1146_114627

theorem square_differences (n : ℕ) : 
  (n + 1)^2 = n^2 + (2*n + 1) ∧ (n - 1)^2 = n^2 - (2*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_differences_l1146_114627


namespace NUMINAMATH_CALUDE_min_c_value_l1146_114615

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem min_c_value (a b c d e : ℕ) : 
  (a + 1 = b) → 
  (b + 1 = c) → 
  (c + 1 = d) → 
  (d + 1 = e) → 
  (is_perfect_square (b + c + d)) →
  (is_perfect_cube (a + b + c + d + e)) →
  (c ≥ 675 ∧ ∀ x < c, ¬(is_perfect_square (x - 1 + x + x + 1) ∧ 
                        is_perfect_cube (x - 2 + x - 1 + x + x + 1 + x + 2))) :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1146_114615


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l1146_114655

/-- Given that the total marks in physics, chemistry, and mathematics is 180 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 90. -/
theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- P: marks in physics, C: marks in chemistry, M: marks in mathematics
  (h : P + C + M = P + 180) -- Given condition
  : (C + M) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l1146_114655


namespace NUMINAMATH_CALUDE_expression_evaluation_l1146_114659

theorem expression_evaluation (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (3 * x + y / 3 + 3 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (3 * z)⁻¹) = (9 * x * y * z)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1146_114659


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l1146_114652

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.somu = 14 ∧
  ages.somu - 7 = (ages.father - 7) / 5

/-- The theorem to prove -/
theorem somu_father_age_ratio (ages : Ages) :
  problem_conditions ages →
  (ages.somu : ℚ) / ages.father = 1 / 3 := by
  sorry

#check somu_father_age_ratio

end NUMINAMATH_CALUDE_somu_father_age_ratio_l1146_114652


namespace NUMINAMATH_CALUDE_scaled_determinant_l1146_114619

theorem scaled_determinant (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 12 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 108 := by
  sorry

end NUMINAMATH_CALUDE_scaled_determinant_l1146_114619


namespace NUMINAMATH_CALUDE_fundraising_amount_proof_l1146_114639

/-- Calculates the amount each person needs to raise in a fundraising event -/
def amount_per_person (total_goal : ℕ) (initial_donation : ℕ) (num_participants : ℕ) : ℕ :=
  (total_goal - initial_donation) / num_participants

/-- Proves that given the specified conditions, each person needs to raise $225 -/
theorem fundraising_amount_proof (total_goal : ℕ) (initial_donation : ℕ) (num_participants : ℕ) 
  (h1 : total_goal = 2000)
  (h2 : initial_donation = 200)
  (h3 : num_participants = 8) :
  amount_per_person total_goal initial_donation num_participants = 225 := by
  sorry

#eval amount_per_person 2000 200 8

end NUMINAMATH_CALUDE_fundraising_amount_proof_l1146_114639


namespace NUMINAMATH_CALUDE_taco_truck_revenue_is_66_l1146_114696

/-- Calculates the total revenue of a taco truck during lunch rush -/
def taco_truck_revenue (soft_taco_price hard_taco_price : ℕ)
  (family_soft_tacos family_hard_tacos : ℕ)
  (additional_customers : ℕ) : ℕ :=
  let total_soft_tacos := family_soft_tacos + 2 * additional_customers
  let soft_taco_revenue := soft_taco_price * total_soft_tacos
  let hard_taco_revenue := hard_taco_price * family_hard_tacos
  soft_taco_revenue + hard_taco_revenue

/-- The total revenue of the taco truck during lunch rush is $66 -/
theorem taco_truck_revenue_is_66 :
  taco_truck_revenue 2 5 3 4 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_taco_truck_revenue_is_66_l1146_114696


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l1146_114600

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3 ∧ 
     b = -12 ∧ 
     c = 24) ∧
    (Complex.I : ℂ)^2 = -1 ∧
    (a * (2 + 2 * Complex.I)^2 + b * (2 + 2 * Complex.I) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l1146_114600


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1146_114609

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_n_satisfying_conditions : 
  (∀ n : ℕ, n > 0 ∧ n < 2000 → ¬(is_perfect_fourth_power (5*n) ∧ is_perfect_cube (4*n))) ∧
  (is_perfect_fourth_power (5*2000) ∧ is_perfect_cube (4*2000)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1146_114609


namespace NUMINAMATH_CALUDE_two_in_A_implies_a_eq_two_A_eq_B_implies_a_eq_one_A_intersect_B_eq_A_implies_a_eq_one_or_four_l1146_114663

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 4*a = (a + 4)*x}

-- Define set B
def B : Set ℝ := {x | x^2 + 4 = 5*x}

-- Theorem 1
theorem two_in_A_implies_a_eq_two (a : ℝ) : 2 ∈ A a → a = 2 := by
  sorry

-- Theorem 2
theorem A_eq_B_implies_a_eq_one (a : ℝ) : A a = B → a = 1 := by
  sorry

-- Theorem 3
theorem A_intersect_B_eq_A_implies_a_eq_one_or_four (a : ℝ) : A a ∩ B = A a → a = 1 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_in_A_implies_a_eq_two_A_eq_B_implies_a_eq_one_A_intersect_B_eq_A_implies_a_eq_one_or_four_l1146_114663


namespace NUMINAMATH_CALUDE_expression_value_l1146_114668

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x + 5 - 4 * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1146_114668


namespace NUMINAMATH_CALUDE_range_of_a_for_two_roots_l1146_114694

theorem range_of_a_for_two_roots (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∃ x y : ℝ, x ≠ y ∧ a^x = x ∧ a^y = y) → 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_roots_l1146_114694


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l1146_114623

/-- The area of the shaded region formed by semicircles in a foot-long pattern -/
theorem semicircle_pattern_area :
  let diameter : ℝ := 3  -- diameter of each semicircle in inches
  let pattern_length : ℝ := 12  -- length of the pattern in inches (1 foot)
  let num_semicircles : ℝ := pattern_length / diameter  -- number of semicircles in the pattern
  let semicircle_area : ℝ → ℝ := λ r => (π * r^2) / 2  -- area of a semicircle
  let total_area : ℝ := num_semicircles * semicircle_area (diameter / 2)
  total_area = (9/2) * π
  := by sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l1146_114623


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1146_114607

/-- An arithmetic sequence with a common difference of 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y / x = z / y

/-- The main theorem -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1146_114607


namespace NUMINAMATH_CALUDE_dice_stack_top_bottom_sum_l1146_114661

/-- Represents a standard die -/
structure StandardDie :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ (f : Fin 6), faces f + faces (5 - f) = 7)

/-- Represents a stack of two standard dice -/
structure DiceStack :=
  (top : StandardDie)
  (bottom : StandardDie)
  (touching_sum : ∃ (f1 f2 : Fin 6), top.faces f1 + bottom.faces f2 = 5)

/-- Theorem: The sum of pips on the top and bottom faces of a dice stack is 9 -/
theorem dice_stack_top_bottom_sum (stack : DiceStack) : 
  ∃ (f1 f2 : Fin 6), stack.top.faces f1 + stack.bottom.faces f2 = 9 :=
sorry

end NUMINAMATH_CALUDE_dice_stack_top_bottom_sum_l1146_114661


namespace NUMINAMATH_CALUDE_coronavirus_cases_in_new_york_l1146_114677

theorem coronavirus_cases_in_new_york :
  ∀ (new_york california texas : ℕ),
    california = new_york / 2 →
    california = texas + 400 →
    new_york + california + texas = 3600 →
    new_york = 800 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_cases_in_new_york_l1146_114677


namespace NUMINAMATH_CALUDE_trig_identity_l1146_114698

theorem trig_identity (α : Real) (h : Real.sin (π/3 - α) = 1/3) :
  Real.cos (π/3 + 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1146_114698


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1146_114670

/-- Given a bowl of marbles with the following properties:
  - Total number of marbles is 19
  - Marbles are split into yellow, blue, and red
  - Ratio of blue to red marbles is 3:4
  - There are 3 more red marbles than yellow marbles
  Prove that the number of yellow marbles is 5. -/
theorem yellow_marbles_count :
  ∀ (yellow blue red : ℕ),
  yellow + blue + red = 19 →
  4 * blue = 3 * red →
  red = yellow + 3 →
  yellow = 5 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1146_114670


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l1146_114602

theorem four_digit_number_problem : ∃! (a b c d : ℕ),
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (0 ≤ d ∧ d ≤ 9) ∧
  (a + b = c + d) ∧
  (a + d = c) ∧
  (b + d = 2 * (a + c)) ∧
  (1000 * a + 100 * b + 10 * c + d = 1854) :=
by sorry

#check four_digit_number_problem

end NUMINAMATH_CALUDE_four_digit_number_problem_l1146_114602


namespace NUMINAMATH_CALUDE_subset_relation_l1146_114667

def A : Set ℝ := {x : ℝ | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem subset_relation :
  (¬(B (1/5) ⊆ A)) ∧
  (∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5)) := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l1146_114667


namespace NUMINAMATH_CALUDE_subset_of_sqrt_eleven_l1146_114636

theorem subset_of_sqrt_eleven (h : Real.sqrt 11 < 2 * Real.sqrt 3) :
  {Real.sqrt 11} ⊆ {x : ℝ | |x| ≤ 2 * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_subset_of_sqrt_eleven_l1146_114636


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1146_114606

-- Define the sets
def set1 : Set ℝ := {x | x * (x - 3) < 0}
def set2 : Set ℝ := {x | |x - 1| < 2}

-- State the theorem
theorem sufficient_but_not_necessary : set1 ⊆ set2 ∧ set1 ≠ set2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1146_114606


namespace NUMINAMATH_CALUDE_quotient_problem_l1146_114641

theorem quotient_problem (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l1146_114641


namespace NUMINAMATH_CALUDE_expression_bounds_l1146_114673

theorem expression_bounds (x y : ℝ) 
  (hx : -3 ≤ x ∧ x ≤ 3) (hy : -2 ≤ y ∧ y ≤ 2) : 
  -6 ≤ x * Real.sqrt (4 - y^2) + y * Real.sqrt (9 - x^2) ∧ 
  x * Real.sqrt (4 - y^2) + y * Real.sqrt (9 - x^2) ≤ 6 := by
  sorry

#check expression_bounds

end NUMINAMATH_CALUDE_expression_bounds_l1146_114673


namespace NUMINAMATH_CALUDE_base8_sum_3_to_100_l1146_114690

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 8 -/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (first last n : ℕ) : ℕ :=
  n * (first + last) / 2

theorem base8_sum_3_to_100 :
  let first := 3
  let last := base8ToBase10 100
  let n := last - first + 1
  base10ToBase8 (arithmeticSequenceSum first last n) = 4035 := by
  sorry

end NUMINAMATH_CALUDE_base8_sum_3_to_100_l1146_114690


namespace NUMINAMATH_CALUDE_student_council_selections_l1146_114686

-- Define the number of students
def n : ℕ := 6

-- Define the number of ways to select a two-person team
def two_person_selections : ℕ := 15

-- Define the number of ways to select a three-person team
def three_person_selections : ℕ := 20

-- Theorem statement
theorem student_council_selections :
  (Nat.choose n 2 = two_person_selections) →
  (Nat.choose n 3 = three_person_selections) :=
by sorry

end NUMINAMATH_CALUDE_student_council_selections_l1146_114686


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l1146_114612

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l1146_114612


namespace NUMINAMATH_CALUDE_fraction_inequality_l1146_114626

theorem fraction_inequality (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 →
  (4 * x + 3 > 2 * (8 - 3 * x) ↔ 13 / 10 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1146_114626


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l1146_114658

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_three_digit_product (n x y : ℕ) :
  (100 ≤ n ∧ n < 1000) →
  (x < 10 ∧ y < 10) →
  isPrime x →
  isPrime (10 * x + y) →
  n = x * (10 * x + y) →
  n ≤ 553 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l1146_114658


namespace NUMINAMATH_CALUDE_equation_solution_l1146_114608

theorem equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ - 3) * (x₁ + 1) = 5 ∧ 
  (x₂ - 3) * (x₂ + 1) = 5 ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1146_114608


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1146_114631

-- Define the quadratic function
def f (x : ℝ) := x^2 + x - 2

-- Define the solution set
def solution_set := {x : ℝ | x < -2 ∨ x > 1}

-- Theorem stating that the solution set is correct
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1146_114631


namespace NUMINAMATH_CALUDE_all_propositions_true_l1146_114649

def S (m n : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ n}

theorem all_propositions_true 
  (m n : ℝ) 
  (h_nonempty : (S m n).Nonempty) 
  (h_closure : ∀ x ∈ S m n, x^2 ∈ S m n) :
  (m = 1 → S m n = {1}) ∧ 
  (m = -1/2 → 1/4 ≤ n ∧ n ≤ 1) ∧ 
  (n = 1/2 → -Real.sqrt 2/2 ≤ m ∧ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_true_l1146_114649


namespace NUMINAMATH_CALUDE_football_field_theorem_l1146_114674

/-- Represents a rectangular football field -/
structure FootballField where
  length : ℝ  -- length in centimeters
  width : ℝ   -- width in meters
  perimeter_condition : 2 * (length / 100 + width) > 350
  area_condition : (length / 100) * width < 7560

/-- Checks if a field meets international match requirements -/
def is_international_match_compliant (field : FootballField) : Prop :=
  100 ≤ field.length / 100 ∧ field.length / 100 ≤ 110 ∧
  64 ≤ field.width ∧ field.width ≤ 75

theorem football_field_theorem (field : FootballField) 
  (h_width : field.width = 70) :
  (10.5 < field.length / 100 ∧ field.length / 100 < 108) ∧
  is_international_match_compliant field := by
  sorry

end NUMINAMATH_CALUDE_football_field_theorem_l1146_114674


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l1146_114693

def A (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; x, y]

theorem matrix_is_own_inverse (x y : ℝ) :
  A x y * A x y = 1 ↔ x = 15/2 ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l1146_114693


namespace NUMINAMATH_CALUDE_tylers_age_l1146_114650

/-- Given the ages of Tyler (T), his brother (B), and their sister (S),
    prove that Tyler's age is 5 years old. -/
theorem tylers_age (T B S : ℕ) : 
  T = B - 3 → 
  S = B + 2 → 
  S = 2 * T → 
  T + B + S = 30 → 
  T = 5 := by
  sorry

end NUMINAMATH_CALUDE_tylers_age_l1146_114650


namespace NUMINAMATH_CALUDE_williams_hot_dogs_left_l1146_114695

/-- Calculates the number of hot dogs left after selling in two periods -/
def hot_dogs_left (initial : ℕ) (sold_first : ℕ) (sold_second : ℕ) : ℕ :=
  initial - (sold_first + sold_second)

/-- Theorem stating that for William's hot dog sales, 45 hot dogs were left -/
theorem williams_hot_dogs_left : hot_dogs_left 91 19 27 = 45 := by
  sorry

end NUMINAMATH_CALUDE_williams_hot_dogs_left_l1146_114695


namespace NUMINAMATH_CALUDE_expression_equality_l1146_114604

theorem expression_equality (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10) 
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) = 
  6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1146_114604


namespace NUMINAMATH_CALUDE_triangle_inequality_l1146_114634

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(a + b > c ∧ b + c > a ∧ c + a > b) ↔ min a (min b c) + (a + b + c - max a (max b c) - min a (min b c)) ≤ max a (max b c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1146_114634


namespace NUMINAMATH_CALUDE_first_part_speed_l1146_114637

def trip_length : ℝ := 12
def part_time : ℝ := 0.25  -- 15 minutes in hours
def second_part_speed : ℝ := 12
def third_part_speed : ℝ := 20

theorem first_part_speed :
  ∃ (v : ℝ), v * part_time + second_part_speed * part_time + third_part_speed * part_time = trip_length ∧ v = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_part_speed_l1146_114637


namespace NUMINAMATH_CALUDE_event_attendees_l1146_114657

theorem event_attendees (num_children : ℕ) (num_adults : ℕ) : 
  num_children = 28 → 
  num_children = 2 * num_adults → 
  num_children + num_adults = 42 := by
  sorry

end NUMINAMATH_CALUDE_event_attendees_l1146_114657


namespace NUMINAMATH_CALUDE_impossible_tiling_l1146_114654

/-- Represents an L-tromino -/
structure LTromino :=
  (cells : Fin 3 → (Fin 5 × Fin 7))

/-- Represents a tiling of a 5x7 rectangle with L-trominos -/
structure Tiling :=
  (trominos : List LTromino)
  (coverage : Fin 5 → Fin 7 → ℕ)

/-- Theorem stating the impossibility of tiling a 5x7 rectangle with L-trominos 
    such that each cell is covered by the same number of trominos -/
theorem impossible_tiling : 
  ∀ (t : Tiling), ¬(∀ (i : Fin 5) (j : Fin 7), ∃ (k : ℕ), t.coverage i j = k) :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l1146_114654


namespace NUMINAMATH_CALUDE_pencil_cost_l1146_114643

-- Define the cost of a pen and a pencil as real numbers
variable (x y : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := 5 * x + 4 * y = 320
def condition2 : Prop := 3 * x + 6 * y = 246

-- State the theorem to be proved
theorem pencil_cost (h1 : condition1 x y) (h2 : condition2 x y) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l1146_114643


namespace NUMINAMATH_CALUDE_bin_problem_l1146_114672

def bin_prices (x y : ℝ) : Prop :=
  14 * x + 8 * y = 1600 ∧ 3 * x = 4 * y

def purchase_constraint (m : ℕ) : Prop :=
  29 ≤ m ∧ m ≤ 50 ∧ 80 * m + 60 * (50 - m) ≤ 3600

theorem bin_problem :
  (∃ x y : ℝ, bin_prices x y ∧ x = 80 ∧ y = 60) ∧
  (∀ m : ℕ, purchase_constraint m ↔ (m = 29 ∨ m = 30)) :=
sorry

end NUMINAMATH_CALUDE_bin_problem_l1146_114672


namespace NUMINAMATH_CALUDE_typhoon_fallen_trees_l1146_114697

/-- Represents the number of trees that fell during a typhoon --/
structure FallenTrees where
  narra : ℕ
  mahogany : ℕ

/-- Represents the initial and final state of trees on the farm --/
structure FarmState where
  initialNarra : ℕ
  initialMahogany : ℕ
  finalTotal : ℕ

def replantedTrees (fallen : FallenTrees) : ℕ :=
  2 * fallen.narra + 3 * fallen.mahogany

theorem typhoon_fallen_trees (farm : FarmState) 
  (h1 : farm.initialNarra = 30)
  (h2 : farm.initialMahogany = 50)
  (h3 : farm.finalTotal = 88) :
  ∃ (fallen : FallenTrees),
    fallen.mahogany = fallen.narra + 1 ∧
    farm.finalTotal = 
      farm.initialNarra + farm.initialMahogany - 
      (fallen.narra + fallen.mahogany) + 
      replantedTrees fallen ∧
    fallen.narra + fallen.mahogany = 5 :=
  sorry


end NUMINAMATH_CALUDE_typhoon_fallen_trees_l1146_114697


namespace NUMINAMATH_CALUDE_sin_cos_bound_l1146_114624

theorem sin_cos_bound (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_bound_l1146_114624


namespace NUMINAMATH_CALUDE_inequality_proof_l1146_114692

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1146_114692


namespace NUMINAMATH_CALUDE_smallest_seating_l1146_114647

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  chairs : Nat
  seated : Nat

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone already seated. -/
def satisfiesCondition (table : CircularTable) : Prop :=
  ∀ (new_seat : Nat), new_seat < table.chairs → 
    ∃ (adjacent_seat : Nat), adjacent_seat < table.chairs ∧ 
      (adjacent_seat = (new_seat + 1) % table.chairs ∨ 
       adjacent_seat = (new_seat + table.chairs - 1) % table.chairs)

/-- Theorem stating the smallest number of people that can be seated to satisfy the condition. -/
theorem smallest_seating (table : CircularTable) : 
  table.chairs = 90 → 
  (∀ n < 23, ¬(satisfiesCondition ⟨90, n⟩)) ∧ 
  satisfiesCondition ⟨90, 23⟩ := by
  sorry

end NUMINAMATH_CALUDE_smallest_seating_l1146_114647


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l1146_114680

theorem circle_equation_through_points : 
  let circle_eq := (fun (x y : ℝ) => x^2 + y^2 - 4*x - 6*y)
  (circle_eq 0 0 = 0) ∧ 
  (circle_eq 4 0 = 0) ∧ 
  (circle_eq (-1) 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l1146_114680


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1146_114644

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def four_consecutive_terms (b : ℕ → ℝ) (S : Set ℝ) : Prop :=
  ∃ k, (b k ∈ S) ∧ (b (k + 1) ∈ S) ∧ (b (k + 2) ∈ S) ∧ (b (k + 3) ∈ S)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℝ) :
  is_geometric_sequence a q →
  (∀ n, b n = a n + 1) →
  |q| > 1 →
  four_consecutive_terms b {-53, -23, 19, 37, 82} →
  q = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1146_114644


namespace NUMINAMATH_CALUDE_line_direction_vector_l1146_114620

/-- The direction vector of a line y = (2x - 6)/5 parameterized as [x, y] = [4, 0] + t * d,
    where t is the distance between [x, y] and [4, 0] for x ≥ 4. -/
theorem line_direction_vector :
  ∃ (d : ℝ × ℝ),
    (∀ (x y t : ℝ), x ≥ 4 →
      y = (2 * x - 6) / 5 →
      (x, y) = (4, 0) + t • d →
      t = Real.sqrt ((x - 4)^2 + y^2)) →
    d = (5 / Real.sqrt 29, 10 / Real.sqrt 29) := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1146_114620


namespace NUMINAMATH_CALUDE_selection_schemes_count_l1146_114621

/-- The number of boys in the selection pool -/
def num_boys : ℕ := 4

/-- The number of girls in the selection pool -/
def num_girls : ℕ := 3

/-- The total number of volunteers to be selected -/
def num_volunteers : ℕ := 4

/-- Function to calculate the number of ways to select volunteers -/
def selection_schemes : ℕ := sorry

/-- Theorem stating that the number of selection schemes is 25 -/
theorem selection_schemes_count : selection_schemes = 25 := by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l1146_114621


namespace NUMINAMATH_CALUDE_pizza_problem_solution_l1146_114679

/-- Represents the number of slices in a pizza --/
structure PizzaSlices where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased --/
structure PizzasPurchased where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person --/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  bill : ℕ
  fred : ℕ
  mark : ℕ

def pizza_problem (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) (leftover : ℕ) : Prop :=
  slices.small = 4 ∧
  slices.large = 8 ∧
  purchased.large = 2 ∧
  eaten.george = 3 ∧
  eaten.bob = eaten.george + 1 ∧
  eaten.susie = eaten.bob / 2 ∧
  eaten.bill = 3 ∧
  eaten.fred = 3 ∧
  eaten.mark = 3 ∧
  leftover = 10 ∧
  purchased.small * slices.small + purchased.large * slices.large =
    eaten.george + eaten.bob + eaten.susie + eaten.bill + eaten.fred + eaten.mark + leftover

theorem pizza_problem_solution 
  (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) (leftover : ℕ) :
  pizza_problem slices purchased eaten leftover → purchased.small = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_solution_l1146_114679


namespace NUMINAMATH_CALUDE_last_student_number_l1146_114683

def skip_pattern (n : ℕ) : ℕ := 3 * n - 1

def student_number (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => skip_pattern (student_number n)

theorem last_student_number :
  ∃ (k : ℕ), student_number k = 242 ∧ 
  ∀ (m : ℕ), m > k → student_number m > 500 := by
sorry

end NUMINAMATH_CALUDE_last_student_number_l1146_114683


namespace NUMINAMATH_CALUDE_no_valid_N_l1146_114601

theorem no_valid_N : ¬ ∃ (N P R : ℕ), 
  N < 40 ∧ 
  P + R = N ∧ 
  (71 * P + 56 * R : ℚ) / N = 66 ∧
  (76 * P : ℚ) / P = 75 ∧
  (61 * R : ℚ) / R = 59 ∧
  P = 2 * R :=
by sorry

end NUMINAMATH_CALUDE_no_valid_N_l1146_114601


namespace NUMINAMATH_CALUDE_range_of_a_l1146_114618

/-- The function f(x) = a^x - x - a has two zeros -/
def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  has_two_zeros (fun x => a^x - x - a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1146_114618


namespace NUMINAMATH_CALUDE_average_growth_rate_satisfies_equation_average_growth_rate_is_twenty_percent_l1146_114682

/-- The average monthly growth rate from March to May for a shopping mall's sales volume. -/
def average_growth_rate : ℝ := 0.2

/-- The sales volume in February in yuan. -/
def february_sales : ℝ := 4000000

/-- The sales volume increase rate from February to March. -/
def march_increase_rate : ℝ := 0.1

/-- The sales volume in May in yuan. -/
def may_sales : ℝ := 6336000

/-- Theorem stating that the calculated average growth rate satisfies the sales volume equation. -/
theorem average_growth_rate_satisfies_equation :
  february_sales * (1 + march_increase_rate) * (1 + average_growth_rate)^2 = may_sales := by sorry

/-- Theorem stating that the average growth rate is indeed 20%. -/
theorem average_growth_rate_is_twenty_percent :
  average_growth_rate = 0.2 := by sorry

end NUMINAMATH_CALUDE_average_growth_rate_satisfies_equation_average_growth_rate_is_twenty_percent_l1146_114682


namespace NUMINAMATH_CALUDE_right_triangle_area_l1146_114630

theorem right_triangle_area (a b c : ℝ) (ha : a^2 = 64) (hb : b^2 = 36) (hc : c^2 = 121) :
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1146_114630


namespace NUMINAMATH_CALUDE_solve_equation_l1146_114666

theorem solve_equation (x : ℝ) : 
  ((2*x + 8) + (7*x + 3) + (3*x + 9)) / 3 = 5*x^2 - 8*x + 2 → 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l1146_114666


namespace NUMINAMATH_CALUDE_avery_theorem_l1146_114678

/-- A shape with a certain number of 90-degree angles -/
structure Shape :=
  (angles : ℕ)

/-- A rectangular park is a shape with 4 90-degree angles -/
def rectangular_park : Shape :=
  ⟨4⟩

/-- Avery's visit to two places -/
structure AverysVisit :=
  (place1 : Shape)
  (place2 : Shape)
  (total_angles : ℕ)

/-- A rectangle or square is a shape with 4 90-degree angles -/
def rectangle_or_square (s : Shape) : Prop :=
  s.angles = 4

/-- The theorem to be proved -/
theorem avery_theorem (visit : AverysVisit) :
  visit.place1 = rectangular_park →
  visit.total_angles = 8 →
  rectangle_or_square visit.place2 :=
sorry

end NUMINAMATH_CALUDE_avery_theorem_l1146_114678


namespace NUMINAMATH_CALUDE_tiffany_lives_l1146_114681

theorem tiffany_lives (x : ℕ) : 
  (x - 14 + 27 = 56) → x = 43 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_l1146_114681


namespace NUMINAMATH_CALUDE_race_distance_l1146_114614

theorem race_distance (d : ℝ) (a b c : ℝ) : 
  (d > 0) →
  (d / a = (d - 30) / b) →
  (d / b = (d - 15) / c) →
  (d / a = (d - 40) / c) →
  d = 90 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l1146_114614


namespace NUMINAMATH_CALUDE_geometric_sum_mod_500_l1146_114632

theorem geometric_sum_mod_500 : (Finset.sum (Finset.range 1001) (fun i => 3^i)) % 500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_mod_500_l1146_114632


namespace NUMINAMATH_CALUDE_square_root_value_l1146_114633

theorem square_root_value (x : ℝ) (h : x = 5) : Real.sqrt (x - 3) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_value_l1146_114633


namespace NUMINAMATH_CALUDE_bond_value_after_eight_years_l1146_114669

/-- Represents the simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.08333333333333332

theorem bond_value_after_eight_years :
  ∀ initial_investment : ℝ,
  simple_interest initial_investment interest_rate 3 = 300 →
  simple_interest initial_investment interest_rate 8 = 400 :=
by
  sorry


end NUMINAMATH_CALUDE_bond_value_after_eight_years_l1146_114669


namespace NUMINAMATH_CALUDE_triangle_properties_l1146_114687

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = 2 * Real.sqrt 3 ∧
  a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0 →
  a = 3 ∧
  (b + c = Real.sqrt 11 →
    1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1146_114687


namespace NUMINAMATH_CALUDE_max_true_statements_l1146_114684

theorem max_true_statements : ∃ x : ℝ, 
  (-1 < x ∧ x < 1) ∧ 
  (-1 < x^3 ∧ x^3 < 1) ∧ 
  (0 < x ∧ x < 1) ∧ 
  (0 < x^2 ∧ x^2 < 1) ∧ 
  (0 < x^3 - x^2 ∧ x^3 - x^2 < 1) := by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l1146_114684


namespace NUMINAMATH_CALUDE_sum_of_arguments_l1146_114617

def complex_pow_eq (z : ℂ) : Prop := z^5 = -32 * Complex.I

theorem sum_of_arguments (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : complex_pow_eq z₁) (h₂ : complex_pow_eq z₂) (h₃ : complex_pow_eq z₃) 
  (h₄ : complex_pow_eq z₄) (h₅ : complex_pow_eq z₅) 
  (distinct : z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ 
              z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ 
              z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ 
              z₄ ≠ z₅) :
  Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + Complex.arg z₄ + Complex.arg z₅ = 
  990 * (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_arguments_l1146_114617


namespace NUMINAMATH_CALUDE_consecutive_squares_remainder_l1146_114603

theorem consecutive_squares_remainder (n : ℕ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 ≡ 2 [MOD 3] :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_remainder_l1146_114603


namespace NUMINAMATH_CALUDE_area_of_triangle_APB_l1146_114656

-- Define the square and point P
def Square (s : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (8, 8)
def D : ℝ × ℝ := (0, 8)
def F : ℝ × ℝ := (4, 8)

-- Define the conditions
def PointInSquare (P : ℝ × ℝ) : Prop := P ∈ Square 8

def EqualSegments (P : ℝ × ℝ) : Prop :=
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2

def PerpendicularPC_FD (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) * (F.1 - D.1) + (P.2 - C.2) * (F.2 - D.2) = 0

def PerpendicularPB_AC (P : ℝ × ℝ) : Prop :=
  (P.1 - B.1) * (A.1 - C.1) + (P.2 - B.2) * (A.2 - C.2) = 0

-- Theorem statement
theorem area_of_triangle_APB (P : ℝ × ℝ) 
  (h1 : PointInSquare P) 
  (h2 : EqualSegments P) 
  (h3 : PerpendicularPC_FD P) 
  (h4 : PerpendicularPB_AC P) : 
  ∃ (area : ℝ), area = 32/5 ∧ 
  area = (1/2) * ((P.1 - A.1)^2 + (P.2 - A.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_APB_l1146_114656


namespace NUMINAMATH_CALUDE_det_A_eq_11_l1146_114689

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -2; 3, 1]

theorem det_A_eq_11 : A.det = 11 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_11_l1146_114689


namespace NUMINAMATH_CALUDE_strict_manager_proposal_l1146_114671

/-- Represents the total monthly salary before changes -/
def initial_total_salary : ℕ := 10000

/-- Represents the total monthly salary after the kind manager's proposal -/
def kind_manager_total_salary : ℕ := 24000

/-- Represents the salary threshold -/
def salary_threshold : ℕ := 500

/-- Represents the number of employees -/
def total_employees : ℕ := 14

theorem strict_manager_proposal (x y : ℕ) 
  (h1 : x + y = total_employees)
  (h2 : 500 * x + y * salary_threshold ≤ initial_total_salary)
  (h3 : 3 * 500 * x + (initial_total_salary - 500 * x) + 1000 * y = kind_manager_total_salary) :
  500 * (x + y) = 7000 := by
  sorry

end NUMINAMATH_CALUDE_strict_manager_proposal_l1146_114671


namespace NUMINAMATH_CALUDE_harry_seed_cost_l1146_114699

/-- The cost of seeds for Harry's garden --/
def seedCost (pumpkinPrice tomatoPrice pepperPrice : ℚ)
             (pumpkinQty tomatoQty pepperQty : ℕ) : ℚ :=
  pumpkinPrice * pumpkinQty + tomatoPrice * tomatoQty + pepperPrice * pepperQty

/-- Theorem stating the total cost of seeds for Harry --/
theorem harry_seed_cost :
  seedCost (5/2) (3/2) (9/10) 3 4 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_harry_seed_cost_l1146_114699


namespace NUMINAMATH_CALUDE_function_existence_l1146_114638

theorem function_existence : ∃ f : ℤ → ℤ, ∀ a b : ℤ, f (a + b) + f (a * b - 1) = f a * f b + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_existence_l1146_114638


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1146_114628

theorem quadratic_inequality (x : ℝ) : x^2 - x - 30 < 0 ↔ -5 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1146_114628


namespace NUMINAMATH_CALUDE_mortgage_payback_time_l1146_114665

theorem mortgage_payback_time (a : ℝ) (r : ℝ) (S_n : ℝ) (n : ℕ) :
  a = 100 →
  r = 3 →
  S_n = 12100 →
  S_n = a * (1 - r^n) / (1 - r) →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_mortgage_payback_time_l1146_114665


namespace NUMINAMATH_CALUDE_amount_increase_l1146_114635

theorem amount_increase (initial_amount : ℚ) : 
  (initial_amount * (9/8) * (9/8) = 4050) → initial_amount = 3200 := by
  sorry

end NUMINAMATH_CALUDE_amount_increase_l1146_114635


namespace NUMINAMATH_CALUDE_sin_of_arcsin_plus_arctan_l1146_114685

theorem sin_of_arcsin_plus_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan 1) = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_of_arcsin_plus_arctan_l1146_114685


namespace NUMINAMATH_CALUDE_tom_found_15_seashells_l1146_114645

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The difference between Fred's and Tom's seashell counts -/
def difference : ℕ := 28

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := fred_seashells - difference

theorem tom_found_15_seashells : tom_seashells = 15 := by
  sorry

end NUMINAMATH_CALUDE_tom_found_15_seashells_l1146_114645


namespace NUMINAMATH_CALUDE_set_relationships_l1146_114653

-- Define the sets M, N, and P
def M : Set ℚ := {x | ∃ m : ℤ, x = m + 1/6}
def N : Set ℚ := {x | ∃ n : ℤ, x = n/2 - 1/3}
def P : Set ℚ := {x | ∃ p : ℤ, x = p/2 + 1/6}

-- State the theorem
theorem set_relationships : M ⊆ N ∧ N = P := by sorry

end NUMINAMATH_CALUDE_set_relationships_l1146_114653
