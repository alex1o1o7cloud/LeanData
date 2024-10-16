import Mathlib

namespace NUMINAMATH_CALUDE_aaron_remaining_erasers_l2147_214735

def initial_erasers : ℕ := 225
def given_to_doris : ℕ := 75
def given_to_ethan : ℕ := 40
def given_to_fiona : ℕ := 50

theorem aaron_remaining_erasers :
  initial_erasers - (given_to_doris + given_to_ethan + given_to_fiona) = 60 := by
  sorry

end NUMINAMATH_CALUDE_aaron_remaining_erasers_l2147_214735


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2147_214708

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2147_214708


namespace NUMINAMATH_CALUDE_river_flow_rate_l2147_214723

/-- Given a river with specified dimensions and flow rate, calculate its velocity --/
theorem river_flow_rate (depth : ℝ) (width : ℝ) (flow_rate : ℝ) (velocity : ℝ) : 
  depth = 8 →
  width = 25 →
  flow_rate = 26666.666666666668 →
  velocity = flow_rate / (depth * width) →
  velocity = 133.33333333333334 := by
  sorry

#check river_flow_rate

end NUMINAMATH_CALUDE_river_flow_rate_l2147_214723


namespace NUMINAMATH_CALUDE_odd_product_over_sum_equals_fifteen_fourths_l2147_214756

theorem odd_product_over_sum_equals_fifteen_fourths : 
  (1 * 3 * 5 * 7) / (1 + 2 + 3 + 4 + 5 + 6 + 7) = 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_product_over_sum_equals_fifteen_fourths_l2147_214756


namespace NUMINAMATH_CALUDE_satisfaction_theorem_l2147_214753

/-- Represents the setup of people around a round table -/
structure TableSetup :=
  (num_men : ℕ)
  (num_women : ℕ)

/-- Defines what it means for a man to be satisfied -/
def is_satisfied (setup : TableSetup) (p : ℝ) : Prop :=
  p = 1 - (setup.num_men - 1) / (setup.num_men + setup.num_women - 1) *
    (setup.num_men - 2) / (setup.num_men + setup.num_women - 2)

/-- The main theorem about the probability of satisfaction and expected number of satisfied men -/
theorem satisfaction_theorem (setup : TableSetup) 
  (h1 : setup.num_men = 50) (h2 : setup.num_women = 50) :
  ∃ (p : ℝ), 
    is_satisfied setup p ∧ 
    p = 25 / 33 ∧
    setup.num_men * p = 1250 / 33 := by
  sorry


end NUMINAMATH_CALUDE_satisfaction_theorem_l2147_214753


namespace NUMINAMATH_CALUDE_parabola_equation_l2147_214747

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  focus : ℝ × ℝ
  opens_left : Bool

/-- The standard form equation of a parabola. -/
def standard_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = -4 * p.focus.1 * x

theorem parabola_equation (p : Parabola) 
  (h1 : p.focus = (-3, 0)) 
  (h2 : p.opens_left = true) : 
  standard_equation p = fun x y => y^2 = -12 * x := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2147_214747


namespace NUMINAMATH_CALUDE_product_sum_relation_l2147_214757

theorem product_sum_relation (a b : ℤ) : 
  (a * b = 2 * (a + b) + 11) → (b = 7) → (b - a = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2147_214757


namespace NUMINAMATH_CALUDE_power_division_rule_l2147_214704

theorem power_division_rule (a : ℝ) : a^4 / a^3 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2147_214704


namespace NUMINAMATH_CALUDE_power_four_mod_nine_l2147_214770

theorem power_four_mod_nine : 4^3023 % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_power_four_mod_nine_l2147_214770


namespace NUMINAMATH_CALUDE_correct_time_allocation_l2147_214750

/-- Represents the time allocation for different tasks -/
structure TimeAllocation where
  clientCalls : ℕ
  accounting : ℕ
  reports : ℕ
  meetings : ℕ

/-- Calculates the time allocation based on a given ratio and total time -/
def calculateTimeAllocation (ratio : List ℚ) (totalTime : ℕ) : TimeAllocation :=
  sorry

/-- Checks if the calculated time allocation is correct -/
def isCorrectAllocation (allocation : TimeAllocation) : Prop :=
  allocation.clientCalls = 383 ∧
  allocation.accounting = 575 ∧
  allocation.reports = 767 ∧
  allocation.meetings = 255

/-- Theorem stating that the calculated time allocation for the given ratio and total time is correct -/
theorem correct_time_allocation :
  let ratio := [3, 4.5, 6, 2]
  let totalTime := 1980
  let allocation := calculateTimeAllocation ratio totalTime
  isCorrectAllocation allocation ∧ 
  allocation.clientCalls + allocation.accounting + allocation.reports + allocation.meetings = totalTime :=
by sorry

end NUMINAMATH_CALUDE_correct_time_allocation_l2147_214750


namespace NUMINAMATH_CALUDE_remainder_problem_l2147_214724

theorem remainder_problem (k : ℕ) (r : ℕ) (h1 : k > 0) (h2 : k < 38) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) (h5 : k % 7 = r) (h6 : r < 7) : k % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2147_214724


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2147_214774

def f (x : ℝ) := x^4 - 8*x^2 + 3

theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-2) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = min) ∧
    max = 3 ∧ min = -13 := by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2147_214774


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l2147_214784

theorem cubic_equation_sum (a b c : ℝ) : 
  a^3 - 7*a^2 + 10*a = 12 →
  b^3 - 7*b^2 + 10*b = 12 →
  c^3 - 7*c^2 + 10*c = 12 →
  (a*b)/c + (b*c)/a + (c*a)/b = -17/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l2147_214784


namespace NUMINAMATH_CALUDE_cipher_decoding_probabilities_l2147_214715

-- Define the probabilities of success for each person
def p_A : ℝ := 0.4
def p_B : ℝ := 0.35
def p_C : ℝ := 0.3

-- Define the probability of exactly two successes
def prob_two_successes : ℝ :=
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C

-- Define the probability of at least one success
def prob_at_least_one_success : ℝ :=
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C)

-- Theorem statement
theorem cipher_decoding_probabilities :
  prob_two_successes = 0.239 ∧ prob_at_least_one_success = 0.727 := by
  sorry

end NUMINAMATH_CALUDE_cipher_decoding_probabilities_l2147_214715


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2147_214781

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 36,
    diagonal_length := 48,
    longer_base := 60
  }
  area t = 1105.92 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2147_214781


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2147_214794

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 - 36*x + 323
  let solution_set := {x : ℝ | f x ≤ 5}
  let lower_bound := 18 - Real.sqrt 6
  let upper_bound := 18 + Real.sqrt 6
  solution_set = Set.Icc lower_bound upper_bound := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2147_214794


namespace NUMINAMATH_CALUDE_arrangements_count_l2147_214763

def num_tour_groups : ℕ := 4
def num_scenic_spots : ℕ := 4

/-- The number of arrangements for tour groups choosing scenic spots -/
def num_arrangements : ℕ :=
  (num_tour_groups.choose 2) * (num_scenic_spots * (num_scenic_spots - 1) * (num_scenic_spots - 2))

theorem arrangements_count :
  num_arrangements = 144 := by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2147_214763


namespace NUMINAMATH_CALUDE_total_celestial_bodies_count_l2147_214751

/-- A galaxy with specific ratios of celestial bodies -/
structure Galaxy where
  planets : ℕ
  solar_systems : ℕ
  stars : ℕ
  solar_system_planet_ratio : solar_systems = 8 * planets
  star_solar_system_ratio : stars = 4 * solar_systems
  planet_count : planets = 20

/-- The total number of celestial bodies in the galaxy -/
def total_celestial_bodies (g : Galaxy) : ℕ :=
  g.planets + g.solar_systems + g.stars

/-- Theorem stating that the total number of celestial bodies is 820 -/
theorem total_celestial_bodies_count (g : Galaxy) :
  total_celestial_bodies g = 820 := by
  sorry

end NUMINAMATH_CALUDE_total_celestial_bodies_count_l2147_214751


namespace NUMINAMATH_CALUDE_tom_gathering_plates_l2147_214789

/-- The number of plates used during a multi-day stay with multiple meals per day -/
def plates_used (people : ℕ) (days : ℕ) (meals_per_day : ℕ) (courses_per_meal : ℕ) (plates_per_course : ℕ) : ℕ :=
  people * days * meals_per_day * courses_per_meal * plates_per_course

/-- Theorem: Given the conditions from Tom's gathering, the total number of plates used is 1728 -/
theorem tom_gathering_plates :
  plates_used 12 6 4 3 2 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_tom_gathering_plates_l2147_214789


namespace NUMINAMATH_CALUDE_min_value_and_zeros_theorem_l2147_214767

-- Define the functions f, g, and F
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - a)
noncomputable def F (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := x * f a x - m

-- State the theorem
theorem min_value_and_zeros_theorem (a : ℝ) (m : ℝ) (x₁ x₂ : ℝ) :
  (∃ (y : ℝ), ∀ (x : ℝ), f a x ≥ y ∧ g a x ≥ y) →  -- f and g have the same minimum value
  (m < 0) →  -- m is negative
  (F a m x₁ = 0) →  -- x₁ is a zero of F
  (F a m x₂ = 0) →  -- x₂ is a zero of F
  (x₁ ≠ x₂) →  -- x₁ and x₂ are distinct
  (a = 1 ∧ x₁ * x₂ > -m^2 - m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_zeros_theorem_l2147_214767


namespace NUMINAMATH_CALUDE_painting_area_is_1836_l2147_214782

/-- The area of a rectangular painting within a frame -/
def painting_area (frame_width outer_length outer_width : ℝ) : ℝ :=
  (outer_length - 2 * frame_width) * (outer_width - 2 * frame_width)

/-- Theorem: The area of the painting is 1836 cm² -/
theorem painting_area_is_1836 :
  painting_area 8 70 50 = 1836 := by
  sorry

end NUMINAMATH_CALUDE_painting_area_is_1836_l2147_214782


namespace NUMINAMATH_CALUDE_equation_solution_l2147_214754

theorem equation_solution : ∃ x : ℤ, 45 - (5 * 3) = x + 7 ∧ x = 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2147_214754


namespace NUMINAMATH_CALUDE_final_white_pieces_l2147_214769

/-- Recursively calculates the number of remaining white pieces after each round of removal -/
def remainingWhitePieces : ℕ → ℕ
| 0 => 1990  -- Initial number of white pieces
| (n + 1) =>
  let previous := remainingWhitePieces n
  if previous % 2 = 0 then
    previous / 2
  else
    (previous + 1) / 2

/-- Theorem stating that after the removal process, 124 white pieces remain -/
theorem final_white_pieces :
  ∃ n : ℕ, remainingWhitePieces n = 124 ∧ ∀ m > n, remainingWhitePieces m = 124 :=
sorry

end NUMINAMATH_CALUDE_final_white_pieces_l2147_214769


namespace NUMINAMATH_CALUDE_odd_function_property_l2147_214790

-- Define an odd function f: ℝ → ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h_odd : isOddFunction f) (h_f_neg_three : f (-3) = 2) :
  f 3 + f 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2147_214790


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l2147_214797

theorem apple_pear_equivalence (apple_value pear_value : ℚ) :
  (3/4 : ℚ) * 12 * apple_value = 10 * pear_value →
  (2/3 : ℚ) * 9 * apple_value = (20/3 : ℚ) * pear_value :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l2147_214797


namespace NUMINAMATH_CALUDE_proposition_b_is_false_l2147_214729

theorem proposition_b_is_false : ¬(∀ x : ℝ, (¬(1/x < 1) → (-1 ≤ x ∧ x ≤ 1)) ∧
  ¬(∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → ¬(1/x < 1))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_is_false_l2147_214729


namespace NUMINAMATH_CALUDE_triple_lcm_equation_solution_l2147_214799

theorem triple_lcm_equation_solution (a b c n : ℕ+) 
  (h1 : a^2 + b^2 = n * Nat.lcm a b + n^2)
  (h2 : b^2 + c^2 = n * Nat.lcm b c + n^2)
  (h3 : c^2 + a^2 = n * Nat.lcm c a + n^2) :
  a = n ∧ b = n ∧ c = n := by
sorry

end NUMINAMATH_CALUDE_triple_lcm_equation_solution_l2147_214799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2147_214773

/-- Definition of arithmetic sequence sum -/
def arithmetic_sequence_sum (n : ℕ) : ℝ := sorry

/-- Theorem: For an arithmetic sequence with sum S_n, if S_3 = 15 and S_9 = 153, then S_6 = 66 -/
theorem arithmetic_sequence_sum_property :
  (arithmetic_sequence_sum 3 = 15) →
  (arithmetic_sequence_sum 9 = 153) →
  (arithmetic_sequence_sum 6 = 66) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2147_214773


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2147_214730

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 29) →
  (a 3 + a 6 + a 9 = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2147_214730


namespace NUMINAMATH_CALUDE_square_stack_sums_l2147_214727

theorem square_stack_sums : 
  (¬ ∃ n : ℕ+, 10 * n = 8016) ∧ 
  (∃ n : ℕ+, 10 * n = 8020) := by
  sorry

end NUMINAMATH_CALUDE_square_stack_sums_l2147_214727


namespace NUMINAMATH_CALUDE_area_of_triangle_DEF_l2147_214718

-- Define the square PQRS
def PQRS_area : ℝ := 36

-- Define the side length of smaller squares
def small_square_side : ℝ := 2

-- Define the triangle DEF
structure Triangle_DEF where
  DE : ℝ
  DF : ℝ
  EF : ℝ

-- Define the folding property
def folds_to_center (t : Triangle_DEF) (s : ℝ) : Prop :=
  t.DE = t.DF ∧ t.DE = s / 2 + 2 * small_square_side

-- Main theorem
theorem area_of_triangle_DEF (t : Triangle_DEF) (s : ℝ) :
  s^2 = PQRS_area →
  folds_to_center t s →
  t.EF = s - 2 * small_square_side →
  (1/2) * t.EF * t.DE = 10 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_DEF_l2147_214718


namespace NUMINAMATH_CALUDE_puzzle_solution_l2147_214772

/-- Represents the pieces of the puzzle -/
inductive Piece
| Two
| One
| Zero
| Minus

/-- Represents the arrangement of pieces -/
def Arrangement := List Piece

/-- Checks if an arrangement forms a valid subtraction equation -/
def isValidArrangement (arr : Arrangement) : Prop := sorry

/-- Calculates the result of a valid arrangement -/
def calculateResult (arr : Arrangement) : Int := sorry

/-- The main theorem: The correct arrangement results in -100 -/
theorem puzzle_solution :
  ∃ (arr : Arrangement),
    isValidArrangement arr ∧ calculateResult arr = -100 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2147_214772


namespace NUMINAMATH_CALUDE_point_distance_equality_l2147_214780

/-- Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
    if |BO| = |BA|, then b = 5. -/
theorem point_distance_equality (b : ℝ) : 
  let A : ℝ × ℝ := (4, 2)
  let B : ℝ × ℝ := (0, b)
  let O : ℝ × ℝ := (0, 0)
  (‖B - O‖ = ‖B - A‖) → b = 5 :=
by sorry

end NUMINAMATH_CALUDE_point_distance_equality_l2147_214780


namespace NUMINAMATH_CALUDE_four_digit_multiple_of_19_l2147_214710

theorem four_digit_multiple_of_19 (a : ℕ) : 
  (2000 + 100 * a + 17) % 19 = 0 → a = 7 := by
sorry

end NUMINAMATH_CALUDE_four_digit_multiple_of_19_l2147_214710


namespace NUMINAMATH_CALUDE_range_of_difference_l2147_214761

theorem range_of_difference (x y : ℝ) (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) :
  27 < x - y ∧ x - y < 56 := by
  sorry

end NUMINAMATH_CALUDE_range_of_difference_l2147_214761


namespace NUMINAMATH_CALUDE_all_points_above_x_axis_l2147_214744

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Checks if a point is above or on the x-axis -/
def isAboveOrOnXAxis (p : Point) : Prop :=
  p.y ≥ 0

/-- Checks if a point is inside or on the boundary of a parallelogram -/
def isInsideOrOnParallelogram (para : Parallelogram) (p : Point) : Prop :=
  sorry  -- Definition of this function is omitted for brevity

/-- The main theorem to be proved -/
theorem all_points_above_x_axis (para : Parallelogram) 
    (h1 : para.P = ⟨-4, 4⟩) 
    (h2 : para.Q = ⟨4, 2⟩)
    (h3 : para.R = ⟨2, -2⟩)
    (h4 : para.S = ⟨-6, -4⟩) :
    ∀ p : Point, isInsideOrOnParallelogram para p → isAboveOrOnXAxis p :=
  sorry

#check all_points_above_x_axis

end NUMINAMATH_CALUDE_all_points_above_x_axis_l2147_214744


namespace NUMINAMATH_CALUDE_S_13_equals_3510_l2147_214712

/-- The sequence S defined for natural numbers -/
def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

/-- Theorem stating that S(13) equals 3510 -/
theorem S_13_equals_3510 : S 13 = 3510 := by
  sorry

end NUMINAMATH_CALUDE_S_13_equals_3510_l2147_214712


namespace NUMINAMATH_CALUDE_square_of_1033_l2147_214785

theorem square_of_1033 : (1033 : ℕ)^2 = 1067089 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1033_l2147_214785


namespace NUMINAMATH_CALUDE_burger_cost_theorem_l2147_214707

/-- The cost of a single burger given the total spent, total burgers, double burger cost, and number of double burgers --/
def single_burger_cost (total_spent : ℚ) (total_burgers : ℕ) (double_burger_cost : ℚ) (double_burgers : ℕ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let double_burgers_cost := double_burger_cost * double_burgers
  let single_burgers_total_cost := total_spent - double_burgers_cost
  single_burgers_total_cost / single_burgers

theorem burger_cost_theorem :
  single_burger_cost 68.50 50 1.50 37 = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_theorem_l2147_214707


namespace NUMINAMATH_CALUDE_function_comparison_l2147_214752

theorem function_comparison (x₁ x₂ : ℝ) (h1 : x₁ < x₂) (h2 : x₁ + x₂ = 0) :
  let f := fun x => x^2 + 2*x + 4
  f x₁ < f x₂ := by
sorry

end NUMINAMATH_CALUDE_function_comparison_l2147_214752


namespace NUMINAMATH_CALUDE_point_relationship_l2147_214791

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem point_relationship (m n : ℝ) : 
  let l : Line := { slope := -2, intercept := 1 }
  let A : Point := { x := -1, y := m }
  let B : Point := { x := 3, y := n }
  A.liesOn l ∧ B.liesOn l → m > n := by
  sorry

end NUMINAMATH_CALUDE_point_relationship_l2147_214791


namespace NUMINAMATH_CALUDE_red_balls_count_l2147_214734

theorem red_balls_count (total_balls : ℕ) (red_prob : ℚ) (red_balls : ℕ) : 
  total_balls = 15 → red_prob = 1/3 → red_balls = (red_prob * total_balls).num → red_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2147_214734


namespace NUMINAMATH_CALUDE_inverse_count_mod_eleven_l2147_214746

theorem inverse_count_mod_eleven : 
  ∃ (S : Finset ℕ), 
    S.card = 10 ∧ 
    (∀ a ∈ S, a ≤ 10) ∧
    (∀ a ∈ S, ∃ b : ℕ, (a * b) % 11 = 1) ∧
    (∀ a : ℕ, a ≤ 10 → (∃ b : ℕ, (a * b) % 11 = 1) → a ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_inverse_count_mod_eleven_l2147_214746


namespace NUMINAMATH_CALUDE_parabola_properties_l2147_214758

/-- A parabola with focus on a given line -/
structure Parabola where
  p : ℝ
  focus_on_line : (p / 2) + (0 : ℝ) - 2 = 0

/-- The directrix of a parabola -/
def directrix (C : Parabola) : ℝ → Prop :=
  λ x => x = -(C.p / 2)

theorem parabola_properties (C : Parabola) :
  C.p = 4 ∧ directrix C = λ x => x = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2147_214758


namespace NUMINAMATH_CALUDE_counterexample_exists_l2147_214711

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2147_214711


namespace NUMINAMATH_CALUDE_adult_ticket_price_is_60_l2147_214725

/-- Represents the ticket prices and attendance for a football game -/
structure FootballGame where
  adultTicketPrice : ℕ
  childTicketPrice : ℕ
  totalAttendance : ℕ
  adultAttendance : ℕ
  totalRevenue : ℕ

/-- Theorem stating that the adult ticket price is 60 cents -/
theorem adult_ticket_price_is_60 (game : FootballGame) :
  game.childTicketPrice = 25 ∧
  game.totalAttendance = 280 ∧
  game.totalRevenue = 14000 ∧
  game.adultAttendance = 200 →
  game.adultTicketPrice = 60 := by
  sorry

#check adult_ticket_price_is_60

end NUMINAMATH_CALUDE_adult_ticket_price_is_60_l2147_214725


namespace NUMINAMATH_CALUDE_line_through_points_l2147_214700

theorem line_through_points (a n : ℝ) :
  (∀ x y, x = 3 * y + 5 → 
    ((x = a ∧ y = n) ∨ (x = a + 2 ∧ y = n + 2/3))) →
  a = 3 * n + 5 :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l2147_214700


namespace NUMINAMATH_CALUDE_g_of_5_equals_22_l2147_214741

/-- Given that g(x) = 4x + 2 for all x, prove that g(5) = 22 -/
theorem g_of_5_equals_22 (g : ℝ → ℝ) (h : ∀ x, g x = 4 * x + 2) : g 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_22_l2147_214741


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_half_l2147_214796

theorem tangent_slope_at_pi_half :
  let f (x : ℝ) := Real.tan (x / 2)
  (deriv f) (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_half_l2147_214796


namespace NUMINAMATH_CALUDE_limit_cube_minus_one_over_x_minus_one_l2147_214764

theorem limit_cube_minus_one_over_x_minus_one : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |(x^3 - 1) / (x - 1) - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_cube_minus_one_over_x_minus_one_l2147_214764


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2147_214798

/-- Given a hyperbola with equation (x^2 / 144) - (y^2 / 81) = 1 and asymptotes y = ±mx, prove that m = 3/4 -/
theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  ((x^2 / 144) - (y^2 / 81) = 1) → 
  (∃ (k : ℝ), y = k * m * x ∨ y = -k * m * x) → 
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2147_214798


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l2147_214787

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - Real.log x

theorem f_monotonicity_and_extrema :
  (∀ x > 0, ∀ a : ℝ,
    (a = 1/2 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
      (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
      (f a 1 = 1/2) ∧
      (∀ x ≠ 1, f a x > 1/2)) ∧
    (a ≤ 0 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
    (a > 0 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/(2*a) → f a x₁ > f a x₂) ∧
      (∀ x₁ x₂, 1/(2*a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l2147_214787


namespace NUMINAMATH_CALUDE_sqrt_comparison_l2147_214726

theorem sqrt_comparison : Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l2147_214726


namespace NUMINAMATH_CALUDE_fraction_solution_l2147_214701

theorem fraction_solution : ∃ x : ℝ, (x - 4) / (x^2) = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_solution_l2147_214701


namespace NUMINAMATH_CALUDE_cut_into_three_similar_rectangles_l2147_214703

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if two rectangles are similar -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- The original rectangle -/
def originalRect : Rectangle :=
  { width := 10, height := 9 }

/-- Theorem stating that the original rectangle can be cut into three unequal but similar rectangles -/
theorem cut_into_three_similar_rectangles :
  ∃ (r1 r2 r3 : Rectangle),
    r1.width + r2.width + r3.width = originalRect.width ∧
    r1.height + r2.height + r3.height = originalRect.height ∧
    r1.width ≠ r2.width ∧ r2.width ≠ r3.width ∧ r1.width ≠ r3.width ∧
    similar r1 r2 ∧ similar r2 r3 ∧ similar r1 r3 ∧
    similar r1 originalRect ∧ similar r2 originalRect ∧ similar r3 originalRect :=
  by sorry

end NUMINAMATH_CALUDE_cut_into_three_similar_rectangles_l2147_214703


namespace NUMINAMATH_CALUDE_intersection_M_N_l2147_214795

def M : Set ℝ := {x | 3 * x - 6 ≥ 0}
def N : Set ℝ := {x | x^2 < 16}

theorem intersection_M_N : M ∩ N = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2147_214795


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l2147_214749

theorem roots_sum_reciprocal_cubes (r s : ℂ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l2147_214749


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2147_214768

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l2147_214768


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2147_214709

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = √2 and b = 2 sin B + cos B = √2, then angle A measures π/6 radians. -/
theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  (a = Real.sqrt 2) →
  (b = 2 * Real.sin B + Real.cos B) →
  (b = Real.sqrt 2) →
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin B / b = Real.sin C / c) →
  (A = π / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2147_214709


namespace NUMINAMATH_CALUDE_all_circles_pass_through_point_l2147_214748

-- Define the parabola
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2 + 2)^2 = 4 * (P.1 - 1)

-- Define a circle with center P tangent to y-axis
def circle_tangent_y_axis (P : ℝ × ℝ) (r : ℝ) : Prop :=
  r = P.1

-- Theorem statement
theorem all_circles_pass_through_point :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    is_on_parabola P →
    circle_tangent_y_axis P r →
    (P.1 - 2)^2 + (P.2 + 2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_all_circles_pass_through_point_l2147_214748


namespace NUMINAMATH_CALUDE_operations_to_zero_l2147_214792

/-- The initial value before operations begin -/
def initial_value : ℕ := 2100

/-- The amount subtracted in each operation -/
def subtract_amount : ℕ := 50

/-- The amount added in each operation -/
def add_amount : ℕ := 20

/-- The effective change per operation -/
def effective_change : ℤ := (subtract_amount : ℤ) - (add_amount : ℤ)

/-- The number of operations needed to reach 0 -/
def num_operations : ℕ := initial_value / (effective_change.natAbs)

theorem operations_to_zero : num_operations = 70 := by
  sorry

end NUMINAMATH_CALUDE_operations_to_zero_l2147_214792


namespace NUMINAMATH_CALUDE_kids_wearing_socks_l2147_214731

theorem kids_wearing_socks (total : ℕ) (wearing_shoes : ℕ) (wearing_both : ℕ) (barefoot : ℕ) :
  total = 22 →
  wearing_shoes = 8 →
  wearing_both = 6 →
  barefoot = 8 →
  total - barefoot - (wearing_shoes - wearing_both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_kids_wearing_socks_l2147_214731


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2147_214793

/-- Theorem: Solution of quadratic inequality ax^2 + bx + c < 0 -/
theorem quadratic_inequality_solution 
  (a b c : ℝ) (h : a ≠ 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a > 0 → {x : ℝ | x1 < x ∧ x < x2} = {x : ℝ | a*x^2 + b*x + c < 0}) ∧
  (a < 0 → {x : ℝ | x < x1 ∨ x2 < x} = {x : ℝ | a*x^2 + b*x + c < 0}) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2147_214793


namespace NUMINAMATH_CALUDE_construct_axes_l2147_214737

/-- A parabola in a 2D plane -/
structure Parabola where
  f : ℝ → ℝ
  is_parabola : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  is_line : a ≠ 0 ∨ b ≠ 0

/-- Compass and straightedge construction operations -/
inductive Construction
  | point : Point → Construction
  | line : Point → Point → Construction
  | circle : Point → Point → Construction
  | intersect_lines : Line → Line → Construction
  | intersect_line_circle : Line → Point → Point → Construction
  | intersect_circles : Point → Point → Point → Point → Construction

/-- The theorem stating that coordinate axes can be constructed given a parabola -/
theorem construct_axes (p : Parabola) : 
  ∃ (origin : Point) (x_axis y_axis : Line) (constructions : List Construction),
    (∀ x : ℝ, p.f x = x^2) →
    (origin.x = 0 ∧ origin.y = 0) ∧
    (∀ x : ℝ, x_axis.a * x + x_axis.b * 0 + x_axis.c = 0) ∧
    (∀ y : ℝ, y_axis.a * 0 + y_axis.b * y + y_axis.c = 0) :=
sorry

end NUMINAMATH_CALUDE_construct_axes_l2147_214737


namespace NUMINAMATH_CALUDE_violet_distance_in_race_l2147_214720

/-- The distance Violet has covered in a race -/
def violet_distance (race_length : ℕ) (aubrey_finish : ℕ) (violet_remaining : ℕ) : ℕ :=
  aubrey_finish - violet_remaining

/-- Theorem: In a 1 km race, if Aubrey finishes when Violet is 279 meters from the finish line,
    then Violet has covered 721 meters -/
theorem violet_distance_in_race : 
  violet_distance 1000 1000 279 = 721 := by
  sorry

end NUMINAMATH_CALUDE_violet_distance_in_race_l2147_214720


namespace NUMINAMATH_CALUDE_technician_count_l2147_214742

theorem technician_count (total_workers : ℕ) (avg_salary : ℝ) (avg_tech_salary : ℝ) (avg_rest_salary : ℝ) :
  total_workers = 12 ∧ 
  avg_salary = 9000 ∧ 
  avg_tech_salary = 12000 ∧ 
  avg_rest_salary = 6000 →
  ∃ (tech_count : ℕ),
    tech_count = 6 ∧
    tech_count + (total_workers - tech_count) = total_workers ∧
    (avg_tech_salary * tech_count + avg_rest_salary * (total_workers - tech_count)) / total_workers = avg_salary :=
by
  sorry

end NUMINAMATH_CALUDE_technician_count_l2147_214742


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l2147_214775

/-- A seven-digit number in the form 945k317 is divisible by 11 if and only if k = 8 -/
theorem seven_digit_divisible_by_11 (k : ℕ) : k < 10 → (945000 + k * 1000 + 317) % 11 = 0 ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l2147_214775


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2147_214713

/-- A curve in the xy-plane parameterized by t -/
structure Curve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific curve defined in the problem -/
def problemCurve : Curve where
  x := fun t => t^2 - 3
  y := fun t => t^3 - 6*t + 4

/-- A point where the curve intersects itself -/
def selfIntersectionPoint (c : Curve) : ℝ × ℝ → Prop :=
  fun p => ∃ a b, a ≠ b ∧ 
    c.x a = c.x b ∧ 
    c.y a = c.y b ∧
    (c.x a, c.y a) = p

/-- The theorem stating that the curve intersects itself at (3, 4) -/
theorem curve_self_intersection :
  selfIntersectionPoint problemCurve (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2147_214713


namespace NUMINAMATH_CALUDE_replaced_person_age_l2147_214733

theorem replaced_person_age 
  (n : ℕ) 
  (original_avg : ℝ) 
  (new_avg : ℝ) 
  (new_person_age : ℝ) 
  (h1 : n = 10)
  (h2 : original_avg = new_avg + 3)
  (h3 : new_person_age = 12) : 
  n * original_avg - (n * new_avg + new_person_age) = 18 := by
sorry

end NUMINAMATH_CALUDE_replaced_person_age_l2147_214733


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2147_214736

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (a b : Line) (α β : Plane)
  (h1 : perp_line_plane a α)
  (h2 : perp_line_plane b β)
  (h3 : perp_plane α β) :
  perp_line a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2147_214736


namespace NUMINAMATH_CALUDE_complex_number_properties_l2147_214762

theorem complex_number_properties (z : ℂ) (h : z = -1/2 + Complex.I * (Real.sqrt 3 / 2)) : 
  z^3 = 1 ∧ z^2 + z + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2147_214762


namespace NUMINAMATH_CALUDE_simplify_expression_l2147_214765

theorem simplify_expression (x : ℝ) : (x + 2)^2 + x*(x - 4) = 2*x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2147_214765


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l2147_214778

theorem irrational_sqrt_three_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (22 / 7 : ℝ)) ∧ 
  (¬ Irrational (0 : ℝ)) ∧ 
  (¬ Irrational (3.14 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l2147_214778


namespace NUMINAMATH_CALUDE_price_increase_consumption_reduction_l2147_214766

/-- Theorem: If the price of a commodity increases by 25%, a person must reduce their consumption by 20% to maintain the same expenditure. -/
theorem price_increase_consumption_reduction (P C : ℝ) (h : P > 0) (h' : C > 0) :
  let new_price := P * 1.25
  let new_consumption := C * 0.8
  new_price * new_consumption = P * C := by
  sorry

end NUMINAMATH_CALUDE_price_increase_consumption_reduction_l2147_214766


namespace NUMINAMATH_CALUDE_range_of_m_l2147_214759

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (B m ≠ ∅) →
  (A ∪ B m = A) →
  (2 < m ∧ m ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2147_214759


namespace NUMINAMATH_CALUDE_curve_is_ellipse_l2147_214783

/-- Given real numbers a and b where ab ≠ 0, the curve bx² + ay² = ab represents an ellipse. -/
theorem curve_is_ellipse (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
  ∀ (x y : ℝ), b * x^2 + a * y^2 = a * b ↔ x^2 / A^2 + y^2 / B^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_l2147_214783


namespace NUMINAMATH_CALUDE_one_pair_probability_l2147_214771

/-- The number of colors of socks -/
def num_colors : ℕ := 5

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The total number of socks -/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn -/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks with the same color -/
def prob_one_pair : ℚ := 20 / 31.5

theorem one_pair_probability : 
  (num_colors.choose 1 * socks_per_color.choose 2 * (num_colors - 1).choose 3 * (socks_per_color.choose 1)^3) / 
  (total_socks.choose socks_drawn) = prob_one_pair :=
sorry

end NUMINAMATH_CALUDE_one_pair_probability_l2147_214771


namespace NUMINAMATH_CALUDE_number_manipulation_l2147_214779

theorem number_manipulation (x : ℝ) : 
  (x - 34) / 10 = 2 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l2147_214779


namespace NUMINAMATH_CALUDE_gcd_98_63_l2147_214714

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l2147_214714


namespace NUMINAMATH_CALUDE_savings_calculation_l2147_214745

theorem savings_calculation (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income = 19000 → 
  income_ratio = 10 → 
  expenditure_ratio = 4 → 
  income - (income * expenditure_ratio / income_ratio) = 11400 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l2147_214745


namespace NUMINAMATH_CALUDE_point_three_units_from_negative_four_l2147_214732

theorem point_three_units_from_negative_four (x : ℝ) : 
  (x = -4 - 3 ∨ x = -4 + 3) ↔ |x - (-4)| = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_three_units_from_negative_four_l2147_214732


namespace NUMINAMATH_CALUDE_fraction_equality_l2147_214755

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (2 * a - b) / (a + 4 * b) = 3) : 
  (a - 4 * b) / (2 * a + b) = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2147_214755


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2147_214706

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) ∧
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) = 7) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2147_214706


namespace NUMINAMATH_CALUDE_f_properties_l2147_214743

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem f_properties :
  (∃ a b : ℝ, a = -2 ∧ b = 0 ∧ ∀ x ∈ Set.Ioo a b, StrictMonoOn f (Set.Ioo a b)) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = x₁ - 2012 ∧ f x₂ = x₂ - 2012) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2147_214743


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2147_214788

theorem trigonometric_identity : 
  Real.cos (12 * π / 180) * Real.sin (42 * π / 180) - 
  Real.sin (12 * π / 180) * Real.cos (42 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2147_214788


namespace NUMINAMATH_CALUDE_price_A_base_correct_min_A_bundles_correct_l2147_214721

-- Define the price of type A seedlings at the base
def price_A_base : ℝ := 20

-- Define the price of type B seedlings at the base
def price_B_base : ℝ := 30

-- Define the total number of bundles to purchase
def total_bundles : ℕ := 100

-- Define the maximum spending limit
def max_spending : ℝ := 2400

-- Theorem for the price of type A seedlings at the base
theorem price_A_base_correct :
  ∃ (x : ℝ), x > 0 ∧ 300 / x - 300 / (1.5 * x) = 5 ∧ x = price_A_base :=
sorry

-- Theorem for the minimum number of type A seedlings to purchase
theorem min_A_bundles_correct :
  ∃ (m : ℕ), m ≥ 60 ∧
    ∀ (n : ℕ), n < m →
      price_A_base * n + price_B_base * (total_bundles - n) > max_spending :=
sorry

end NUMINAMATH_CALUDE_price_A_base_correct_min_A_bundles_correct_l2147_214721


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2147_214702

theorem imaginary_part_of_z (z : ℂ) : z = (2 * Complex.I^2 + 4) / (Complex.I + 1) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2147_214702


namespace NUMINAMATH_CALUDE_men_in_club_l2147_214705

theorem men_in_club (total : ℕ) (attendees : ℕ) (h_total : total = 30) (h_attendees : attendees = 18) :
  ∃ (men women : ℕ),
    men + women = total ∧
    men + (women / 3) = attendees ∧
    men = 12 := by
  sorry

end NUMINAMATH_CALUDE_men_in_club_l2147_214705


namespace NUMINAMATH_CALUDE_necessary_condition_implies_m_range_l2147_214722

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
def B (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ [-1, 1] ∧ y = 1/3 * x + m}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

-- State the theorem
theorem necessary_condition_implies_m_range :
  ∀ m : ℝ, (∀ x : ℝ, q m x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q m x) →
  m > 1/3 ∧ m < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_implies_m_range_l2147_214722


namespace NUMINAMATH_CALUDE_survival_probabilities_correct_l2147_214776

/-- Mortality table data -/
structure MortalityData :=
  (reach28 : ℕ)
  (reach35 : ℕ)
  (reach48 : ℕ)
  (reach55 : ℕ)
  (total : ℕ)

/-- Survival probabilities after 20 years -/
structure SurvivalProbabilities :=
  (bothAlive : ℚ)
  (husbandDead : ℚ)
  (wifeDead : ℚ)
  (bothDead : ℚ)
  (husbandDeadWifeAlive : ℚ)
  (husbandAliveWifeDead : ℚ)

/-- Calculate survival probabilities based on mortality data -/
def calculateSurvivalProbabilities (data : MortalityData) : SurvivalProbabilities :=
  sorry

/-- Theorem stating the correct survival probabilities -/
theorem survival_probabilities_correct (data : MortalityData) 
  (h1 : data.reach28 = 675)
  (h2 : data.reach35 = 630)
  (h3 : data.reach48 = 540)
  (h4 : data.reach55 = 486)
  (h5 : data.total = 1000) :
  let probs := calculateSurvivalProbabilities data
  probs.bothAlive = 108 / 175 ∧
  probs.husbandDead = 8 / 35 ∧
  probs.wifeDead = 1 / 5 ∧
  probs.bothDead = 8 / 175 ∧
  probs.husbandDeadWifeAlive = 32 / 175 ∧
  probs.husbandAliveWifeDead = 27 / 175 :=
by
  sorry

#check survival_probabilities_correct

end NUMINAMATH_CALUDE_survival_probabilities_correct_l2147_214776


namespace NUMINAMATH_CALUDE_ant_ratio_is_two_to_one_l2147_214739

/-- The number of ants Abe finds -/
def abe_ants : ℕ := 4

/-- The number of ants Beth sees -/
def beth_ants : ℕ := (3 * abe_ants) / 2

/-- The number of ants Duke discovers -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := 20

/-- The number of ants CeCe watches -/
def cece_ants : ℕ := total_ants - (abe_ants + beth_ants + duke_ants)

/-- The ratio of ants CeCe watches to ants Abe finds -/
def ant_ratio : ℚ := cece_ants / abe_ants

theorem ant_ratio_is_two_to_one : ant_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_ant_ratio_is_two_to_one_l2147_214739


namespace NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l2147_214760

theorem night_day_crew_loading_ratio 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (total_boxes : ℝ) 
  (h1 : night_crew = (4 : ℝ) / 9 * day_crew) 
  (h2 : (3 : ℝ) / 4 * total_boxes = day_crew_boxes)
  (h3 : day_crew_boxes + night_crew_boxes = total_boxes) : 
  (night_crew_boxes / night_crew) / (day_crew_boxes / day_crew) = (3 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l2147_214760


namespace NUMINAMATH_CALUDE_total_cupcakes_eq_768_l2147_214738

/-- The number of cupcakes ordered for each event -/
def cupcakes_per_event : ℝ := 96.0

/-- The number of different children's events -/
def number_of_events : ℝ := 8.0

/-- The total number of cupcakes needed -/
def total_cupcakes : ℝ := cupcakes_per_event * number_of_events

/-- Theorem stating that the total number of cupcakes is 768.0 -/
theorem total_cupcakes_eq_768 : total_cupcakes = 768.0 := by
  sorry

end NUMINAMATH_CALUDE_total_cupcakes_eq_768_l2147_214738


namespace NUMINAMATH_CALUDE_polyhedron_edge_sum_greater_than_3d_l2147_214777

-- Define a polyhedron type
structure Polyhedron where
  vertices : Set (ℝ × ℝ × ℝ)
  edges : Set ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ))
  -- Add conditions to ensure it's a valid polyhedron

-- Define the distance between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define the length of an edge
def edgeLength (e : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the sum of all edge lengths
def sumEdgeLengths (p : Polyhedron) : ℝ := sorry

-- Define the maximum distance between any two vertices
def maxVertexDistance (p : Polyhedron) : ℝ := sorry

-- The theorem to prove
theorem polyhedron_edge_sum_greater_than_3d (p : Polyhedron) : 
  sumEdgeLengths p > 3 * maxVertexDistance p := by sorry

end NUMINAMATH_CALUDE_polyhedron_edge_sum_greater_than_3d_l2147_214777


namespace NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_attained_l2147_214719

theorem minimum_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) ≥ 4 :=
by sorry

theorem minimum_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ p q r : ℝ, p > 0 ∧ q > 0 ∧ r > 0 ∧
    (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_attained_l2147_214719


namespace NUMINAMATH_CALUDE_euler_function_gcd_l2147_214717

open Nat

theorem euler_function_gcd (m n : ℕ) (h : φ (5^m - 1) = 5^n - 1) : (m.gcd n) > 1 := by
  sorry

end NUMINAMATH_CALUDE_euler_function_gcd_l2147_214717


namespace NUMINAMATH_CALUDE_f_is_linear_l2147_214786

/-- A function f: ℝ → ℝ is linear if there exist constants m and b such that 
    f(x) = mx + b for all x, where m ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), m ≠ 0 ∧ ∀ x, f x = m * x + b

/-- The function f(x) = -8x -/
def f (x : ℝ) : ℝ := -8 * x

/-- Theorem: f(x) = -8x is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  sorry


end NUMINAMATH_CALUDE_f_is_linear_l2147_214786


namespace NUMINAMATH_CALUDE_fitness_center_ratio_l2147_214728

theorem fitness_center_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) : 
  (55 * f + 80 * m) / (f + m) = 70 → f / m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fitness_center_ratio_l2147_214728


namespace NUMINAMATH_CALUDE_arithmetic_mean_change_l2147_214716

theorem arithmetic_mean_change (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 10 →
  b + c + d = 33 →
  a + c + d = 36 →
  a + b + d = 39 →
  (a + b + c) / 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_change_l2147_214716


namespace NUMINAMATH_CALUDE_total_distance_in_land_miles_l2147_214740

/-- Represents the speed of the sailboat in knots -/
structure SailboatSpeed where
  oneSail : ℝ
  twoSails : ℝ

/-- Represents the travel time in hours -/
structure TravelTime where
  oneSail : ℝ
  twoSails : ℝ

/-- Conversion factors -/
def knotToNauticalMile : ℝ := 1
def nauticalMileToLandMile : ℝ := 1.15

theorem total_distance_in_land_miles 
  (speed : SailboatSpeed) 
  (time : TravelTime) 
  (h1 : speed.oneSail = 25)
  (h2 : speed.twoSails = 50)
  (h3 : time.oneSail = 4)
  (h4 : time.twoSails = 4) :
  (speed.oneSail * time.oneSail + speed.twoSails * time.twoSails) * 
  knotToNauticalMile * nauticalMileToLandMile = 345 := by
  sorry

#check total_distance_in_land_miles

end NUMINAMATH_CALUDE_total_distance_in_land_miles_l2147_214740
