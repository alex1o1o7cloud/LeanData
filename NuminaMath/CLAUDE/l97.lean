import Mathlib

namespace NUMINAMATH_CALUDE_train_capacity_l97_9763

/-- Proves that given a train with 4 carriages, each initially having 25 seats
    and can accommodate 10 more passengers, the total number of passengers
    that would fill up 3 such trains is 420. -/
theorem train_capacity (initial_seats : Nat) (additional_seats : Nat) 
  (carriages_per_train : Nat) (number_of_trains : Nat) :
  initial_seats = 25 →
  additional_seats = 10 →
  carriages_per_train = 4 →
  number_of_trains = 3 →
  (initial_seats + additional_seats) * carriages_per_train * number_of_trains = 420 := by
  sorry

#eval (25 + 10) * 4 * 3  -- Should output 420

end NUMINAMATH_CALUDE_train_capacity_l97_9763


namespace NUMINAMATH_CALUDE_ratio_of_45_to_9_l97_9788

theorem ratio_of_45_to_9 (certain_number : ℕ) (h : certain_number = 45) : 
  certain_number / 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_45_to_9_l97_9788


namespace NUMINAMATH_CALUDE_voyage_year_difference_l97_9713

def zheng_he_voyage_year : ℕ := 2005 - 600
def columbus_voyage_year : ℕ := 1492

theorem voyage_year_difference : columbus_voyage_year - zheng_he_voyage_year = 87 := by
  sorry

end NUMINAMATH_CALUDE_voyage_year_difference_l97_9713


namespace NUMINAMATH_CALUDE_int_part_one_plus_sqrt_seven_l97_9719

theorem int_part_one_plus_sqrt_seven : ⌊1 + Real.sqrt 7⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_int_part_one_plus_sqrt_seven_l97_9719


namespace NUMINAMATH_CALUDE_problem_statement_l97_9798

theorem problem_statement (x y : ℝ) : 
  let a := x^3 * y
  let b := x^2 * y^2
  let c := x * y^3
  (a * c + b^2 - 2 * x^4 * y^4 = 0) ∧ 
  (a * y^2 + c * x^2 = 2 * x * y * b) ∧ 
  ¬(∀ x y : ℝ, a * b * c + b^3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l97_9798


namespace NUMINAMATH_CALUDE_point_movement_l97_9765

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Moves a point on the number line by a given distance -/
def movePoint (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_movement :
  let A : Point := ⟨-4⟩
  let B : Point := movePoint A 6
  B.value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l97_9765


namespace NUMINAMATH_CALUDE_bart_mixtape_second_side_l97_9750

def mixtape (first_side_songs : ℕ) (song_length : ℕ) (total_length : ℕ) : ℕ :=
  (total_length - first_side_songs * song_length) / song_length

theorem bart_mixtape_second_side :
  mixtape 6 4 40 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bart_mixtape_second_side_l97_9750


namespace NUMINAMATH_CALUDE_xy_squared_equals_one_l97_9786

theorem xy_squared_equals_one 
  (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 6) : 
  x^2 * y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_equals_one_l97_9786


namespace NUMINAMATH_CALUDE_max_visible_cubes_11_l97_9733

/-- Represents a cube made of unit cubes --/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point --/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  3 * cube.size^2 - 3 * (cube.size - 1) + 1

/-- Theorem stating that for an 11x11x11 cube, the maximum number of visible unit cubes is 331 --/
theorem max_visible_cubes_11 :
  max_visible_cubes ⟨11⟩ = 331 := by
  sorry

#eval max_visible_cubes ⟨11⟩

end NUMINAMATH_CALUDE_max_visible_cubes_11_l97_9733


namespace NUMINAMATH_CALUDE_complex_simplification_l97_9741

theorem complex_simplification :
  3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - Complex.I) + 2 * (1 + 2 * Complex.I) = 10 - 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l97_9741


namespace NUMINAMATH_CALUDE_max_chocolates_bob_l97_9782

/-- Given that Bob and Carol share 36 chocolates, and Carol eats a positive multiple
    of Bob's chocolates, prove that the maximum number of chocolates Bob could have eaten is 18. -/
theorem max_chocolates_bob (total : ℕ) (bob carol : ℕ) (k : ℕ) : 
  total = 36 →
  bob + carol = total →
  carol = k * bob →
  k > 0 →
  bob ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_max_chocolates_bob_l97_9782


namespace NUMINAMATH_CALUDE_function_decomposition_symmetry_l97_9753

theorem function_decomposition_symmetry (f : ℝ → ℝ) :
  ∃ (f₁ f₂ : ℝ → ℝ) (a : ℝ), a > 0 ∧
    (∀ x, f x = f₁ x + f₂ x) ∧
    (∀ x, f₁ (-x) = f₁ x) ∧
    (∀ x, f₂ (2 * a - x) = f₂ x) :=
by sorry

end NUMINAMATH_CALUDE_function_decomposition_symmetry_l97_9753


namespace NUMINAMATH_CALUDE_range_of_c_l97_9796

-- Define propositions p and q
def p (c : ℝ) : Prop := 2 < 3 * c
def q (c : ℝ) : Prop := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

-- Theorem statement
theorem range_of_c (c : ℝ) 
  (h : (p c ∨ q c) ∨ (p c ∧ q c)) : 
  2/3 < c ∧ c < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l97_9796


namespace NUMINAMATH_CALUDE_four_solutions_l97_9745

/-- The system of equations has exactly 4 distinct real solutions -/
theorem four_solutions (x y z w : ℝ) : 
  (x = z - w + x * z ∧
   y = w - x + y * w ∧
   z = x - y + x * z ∧
   w = y - z + y * w) →
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), s.card = 4 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔
    (a = c - d + a * c ∧
     b = d - a + b * d ∧
     c = a - b + a * c ∧
     d = b - c + b * d) :=
by sorry

end NUMINAMATH_CALUDE_four_solutions_l97_9745


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l97_9790

/-- Given three collinear points A(-1, 1), B(2, -4), and C(x, -9), prove that x = 5 -/
theorem collinear_points_x_value : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -4)
  let C : ℝ × ℝ := (x, -9)
  (∀ t : ℝ, (1 - t) * A.1 + t * B.1 = C.1 ∧ (1 - t) * A.2 + t * B.2 = C.2) →
  x = 5 := by
sorry


end NUMINAMATH_CALUDE_collinear_points_x_value_l97_9790


namespace NUMINAMATH_CALUDE_used_car_selection_l97_9792

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 10 →
  num_clients = 15 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 2 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selection_l97_9792


namespace NUMINAMATH_CALUDE_sin_double_alpha_l97_9759

theorem sin_double_alpha (α : Real) : 
  Real.sin (45 * π / 180 + α) = Real.sqrt 5 / 5 → Real.sin (2 * α) = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l97_9759


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l97_9717

/-- Circle O₁ with center (a, b) and radius √(b² + 1) -/
def circle_O₁ (a b x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = b^2 + 1

/-- Circle O₂ with center (c, d) and radius √(d² + 1) -/
def circle_O₂ (c d x y : ℝ) : Prop :=
  (x - c)^2 + (y - d)^2 = d^2 + 1

/-- Line l: 3x - 4y - 25 = 0 -/
def line_l (x y : ℝ) : Prop :=
  3*x - 4*y - 25 = 0

/-- The minimum distance between a point on the intersection of two circles and a line -/
theorem min_distance_point_to_line
  (a b c d : ℝ)
  (h1 : a * c = 8)
  (h2 : a / b = c / d)
  : ∃ (P : ℝ × ℝ),
    (circle_O₁ a b P.1 P.2 ∧ circle_O₂ c d P.1 P.2) →
    (∀ (M : ℝ × ℝ), line_l M.1 M.2 →
      ∃ (dist : ℝ),
        dist = Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ∧
        dist ≥ 2 ∧
        (∃ (M₀ : ℝ × ℝ), line_l M₀.1 M₀.2 ∧
          Real.sqrt ((P.1 - M₀.1)^2 + (P.2 - M₀.2)^2) = 2)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l97_9717


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l97_9749

theorem rectangle_area_problem (a b : ℝ) :
  (∀ (a b : ℝ), 
    ((a + 3) * b - a * b = 12) ∧
    ((a + 3) * (b + 3) - (a + 3) * b = 24)) →
  a * b = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l97_9749


namespace NUMINAMATH_CALUDE_vincent_earnings_l97_9775

/-- Represents Vincent's bookstore earnings over a period of days -/
def bookstore_earnings (fantasy_price : ℕ) (fantasy_sold : ℕ) (literature_sold : ℕ) (days : ℕ) : ℕ :=
  let literature_price := fantasy_price / 2
  let daily_earnings := fantasy_price * fantasy_sold + literature_price * literature_sold
  daily_earnings * days

/-- Theorem stating that Vincent's earnings after 5 days will be $180 -/
theorem vincent_earnings : bookstore_earnings 4 5 8 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_vincent_earnings_l97_9775


namespace NUMINAMATH_CALUDE_present_age_of_b_l97_9757

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → 
  (a = b + 11) → 
  b = 41 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_b_l97_9757


namespace NUMINAMATH_CALUDE_instruction_set_exists_l97_9722

/-- Represents a box that may contain a ball or be empty. -/
inductive Box
| withBall : Box
| empty : Box

/-- Represents an instruction to swap the contents of two boxes. -/
structure SwapInstruction where
  i : Nat
  j : Nat

/-- Represents a configuration of N boxes. -/
def BoxConfiguration (N : Nat) := Fin N → Box

/-- Represents an instruction set. -/
def InstructionSet := List SwapInstruction

/-- Checks if a configuration is sorted (balls to the left of empty boxes). -/
def isSorted (config : BoxConfiguration N) : Prop :=
  ∀ i j, i < j → config i = Box.empty → config j = Box.empty

/-- Applies an instruction set to a configuration. -/
def applyInstructions (config : BoxConfiguration N) (instructions : InstructionSet) : BoxConfiguration N :=
  sorry

/-- The main theorem to be proved. -/
theorem instruction_set_exists (N : Nat) :
  ∃ (instructions : InstructionSet),
    instructions.length ≤ 100 * N ∧
    ∀ (config : BoxConfiguration N),
      ∃ (subset : InstructionSet),
        subset.length ≤ instructions.length ∧
        isSorted (applyInstructions config subset) :=
  sorry

end NUMINAMATH_CALUDE_instruction_set_exists_l97_9722


namespace NUMINAMATH_CALUDE_min_cards_for_even_product_l97_9709

/-- Represents a card with an integer value -/
structure Card where
  value : Int
  even : Bool

/-- The set of cards in the box -/
def cards : Finset Card :=
  sorry

/-- A valid sequence of drawn cards according to the rules -/
def ValidSequence : List Card → Prop :=
  sorry

/-- The product of the values of a list of cards -/
def product : List Card → Int :=
  sorry

/-- Theorem: The minimum number of cards to ensure an even product is 3 -/
theorem min_cards_for_even_product :
  ∀ (s : List Card), ValidSequence s → product s % 2 = 0 → s.length ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_cards_for_even_product_l97_9709


namespace NUMINAMATH_CALUDE_parabola_c_value_l97_9742

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_c_value (p : Parabola) :
  p.y_at 1 = 3 →  -- vertex at (1, 3)
  p.y_at 0 = 2 →  -- passes through (0, 2)
  p.c = 2 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l97_9742


namespace NUMINAMATH_CALUDE_equation_solutions_l97_9787

/-- The integer part of a real number -/
noncomputable def intPart (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part of a real number -/
noncomputable def fracPart (x : ℝ) : ℝ :=
  x - intPart x

/-- The solutions to the equation [x] · {x} = 1991x -/
theorem equation_solutions :
  ∀ x : ℝ, intPart x * fracPart x = 1991 * x ↔ x = 0 ∨ x = -1 / 1992 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l97_9787


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_range_l97_9771

theorem quadratic_inequality_empty_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + a < 0) ↔ a < -3/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_range_l97_9771


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l97_9724

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a → a 6 = 6 → a 9 = 9 → a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l97_9724


namespace NUMINAMATH_CALUDE_sum_of_digits_n_l97_9785

/-- The least 6-digit number that leaves a remainder of 2 when divided by 4, 610, and 15 -/
def n : ℕ := 102482

/-- Condition: n is at least 100000 (6-digit number) -/
axiom n_six_digits : n ≥ 100000

/-- Condition: n leaves remainder 2 when divided by 4 -/
axiom n_mod_4 : n % 4 = 2

/-- Condition: n leaves remainder 2 when divided by 610 -/
axiom n_mod_610 : n % 610 = 2

/-- Condition: n leaves remainder 2 when divided by 15 -/
axiom n_mod_15 : n % 15 = 2

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m = 0 then 0 else m % 10 + sum_of_digits (m / 10)

/-- Theorem: The sum of digits of n is 17 -/
theorem sum_of_digits_n : sum_of_digits n = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_l97_9785


namespace NUMINAMATH_CALUDE_f_bounded_implies_k_eq_three_l97_9770

/-- The function f(x) = -4x³ + kx --/
def f (k : ℝ) (x : ℝ) : ℝ := -4 * x^3 + k * x

/-- The theorem stating that if f(x) ≤ 1 for all x in [-1, 1], then k = 3 --/
theorem f_bounded_implies_k_eq_three (k : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f k x ≤ 1) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_bounded_implies_k_eq_three_l97_9770


namespace NUMINAMATH_CALUDE_xy_value_l97_9784

theorem xy_value (x y : ℝ) (h : |x + 2*y| + (y - 3)^2 = 0) : x^y = -216 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l97_9784


namespace NUMINAMATH_CALUDE_xiao_ping_weighing_l97_9736

/-- Represents the weights of 8 items -/
structure EightWeights where
  weights : Fin 8 → ℕ
  distinct : ∀ i j, i ≠ j → weights i ≠ weights j
  bounded : ∀ i, 1 ≤ weights i ∧ weights i ≤ 15

/-- The weighing inequalities -/
def weighing_inequalities (w : EightWeights) : Prop :=
  w.weights 0 + w.weights 4 + w.weights 5 + w.weights 6 >
    w.weights 1 + w.weights 2 + w.weights 3 + w.weights 7 ∧
  w.weights 4 + w.weights 5 > w.weights 0 + w.weights 6 ∧
  w.weights 4 > w.weights 5

theorem xiao_ping_weighing (w : EightWeights) :
  weighing_inequalities w →
  (∀ i, i ≠ 4 → w.weights 4 ≤ w.weights i) →
  (∀ i, i ≠ 3 → w.weights i ≤ w.weights 3) →
  w.weights 4 = 11 ∧ w.weights 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ping_weighing_l97_9736


namespace NUMINAMATH_CALUDE_triangle_vector_division_l97_9756

/-- Given a triangle ABC with point M on side BC such that BM:MC = 2:5,
    and vectors AB = a and AC = b, prove that AM = (2/7)a + (5/7)b. -/
theorem triangle_vector_division (A B C M : EuclideanSpace ℝ (Fin 3))
  (a b : EuclideanSpace ℝ (Fin 3)) (h : B ≠ C) :
  (B - M) = (5 / 7 : ℝ) • (C - B) →
  (A - B) = a →
  (A - C) = -b →
  (A - M) = (2 / 7 : ℝ) • a + (5 / 7 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_division_l97_9756


namespace NUMINAMATH_CALUDE_julia_tag_players_l97_9738

theorem julia_tag_players (monday_kids : ℕ) (difference : ℕ) (tuesday_kids : ℕ) 
  (h1 : monday_kids = 18)
  (h2 : difference = 8)
  (h3 : monday_kids = tuesday_kids + difference) :
  tuesday_kids = 10 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_players_l97_9738


namespace NUMINAMATH_CALUDE_sum_of_possible_sums_l97_9779

theorem sum_of_possible_sums (n : ℕ) (h : n = 9) : 
  (n * (n * (n + 1) / 2) - (n * (n + 1) / 2)) = 360 := by
  sorry

#check sum_of_possible_sums

end NUMINAMATH_CALUDE_sum_of_possible_sums_l97_9779


namespace NUMINAMATH_CALUDE_jack_final_position_l97_9711

/-- Represents the number of steps in each flight of stairs -/
def steps_per_flight : ℕ := 12

/-- Represents the height of each step in inches -/
def step_height : ℕ := 8

/-- Represents the number of flights Jack goes up -/
def flights_up : ℕ := 3

/-- Represents the number of flights Jack goes down -/
def flights_down : ℕ := 6

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Theorem stating that Jack ends up 24 feet further down than his starting point -/
theorem jack_final_position : 
  (flights_down - flights_up) * steps_per_flight * step_height / inches_per_foot = 24 := by
  sorry


end NUMINAMATH_CALUDE_jack_final_position_l97_9711


namespace NUMINAMATH_CALUDE_economics_test_absentees_l97_9769

theorem economics_test_absentees (total_students : Nat) 
  (correct_q1 : Nat) (correct_q2 : Nat) (correct_both : Nat) :
  total_students = 30 →
  correct_q1 = 25 →
  correct_q2 = 22 →
  correct_both = 22 →
  correct_both = correct_q2 →
  total_students - correct_q2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_economics_test_absentees_l97_9769


namespace NUMINAMATH_CALUDE_min_pool_cost_l97_9732

/-- Represents the construction cost of a rectangular pool -/
def pool_cost (length width depth : ℝ) (wall_price : ℝ) : ℝ :=
  (2 * (length + width) * depth * wall_price) + (length * width * 1.5 * wall_price)

/-- Theorem stating the minimum cost for the pool construction -/
theorem min_pool_cost (a : ℝ) (h_a : a > 0) :
  let volume := 4800
  let depth := 3
  ∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    length * width * depth = volume ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → l * w * depth = volume →
      pool_cost length width depth a ≤ pool_cost l w depth a ∧
      pool_cost length width depth a = 2880 * a :=
sorry

end NUMINAMATH_CALUDE_min_pool_cost_l97_9732


namespace NUMINAMATH_CALUDE_system_solution_transformation_l97_9791

theorem system_solution_transformation (x y : ℝ) : 
  (2 * x + 3 * y = 19 ∧ 3 * x + 4 * y = 26) → 
  (2 * (2 * x + 4) + 3 * (y + 3) = 19 ∧ 3 * (2 * x + 4) + 4 * (y + 3) = 26) → 
  (x = 2 ∧ y = 5) → 
  (x = -1 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_transformation_l97_9791


namespace NUMINAMATH_CALUDE_sum_equation_l97_9701

theorem sum_equation (x y z : ℝ) (h1 : x + y = 4) (h2 : x * y = z^2 + 4) : 
  x + 2*y + 3*z = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_equation_l97_9701


namespace NUMINAMATH_CALUDE_equation_satisfied_l97_9776

theorem equation_satisfied (a b c : ℤ) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 3 ↔ (a = c + 1 ∧ b - 1 = a) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_l97_9776


namespace NUMINAMATH_CALUDE_strawberries_eaten_l97_9747

theorem strawberries_eaten (initial : Float) (remaining : Nat) (eaten : Nat) : 
  initial = 78.0 → remaining = 36 → eaten = 42 → initial - remaining.toFloat = eaten.toFloat := by
  sorry

end NUMINAMATH_CALUDE_strawberries_eaten_l97_9747


namespace NUMINAMATH_CALUDE_max_value_S_l97_9799

theorem max_value_S (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 5) :
  ∃ (max : ℝ), max = 18 ∧ ∀ (a' b' c' : ℝ), a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → a' + b' + c' = 5 →
    2*a' + 2*a'*b' + a'*b'*c' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_S_l97_9799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l97_9728

/-- An arithmetic sequence with first term 1 and sum of first n terms S_n -/
structure ArithSeq where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithSeq) 
  (h : seq.S 19 / 19 - seq.S 17 / 17 = 6) : 
  seq.S 10 = 280 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l97_9728


namespace NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l97_9783

theorem real_part_of_i_times_one_plus_i (i : ℂ) :
  i * i = -1 →
  Complex.re (i * (1 + i)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l97_9783


namespace NUMINAMATH_CALUDE_floor_plus_self_equation_l97_9772

theorem floor_plus_self_equation (r : ℝ) : ⌊r⌋ + r = 15.4 ↔ r = 7.4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equation_l97_9772


namespace NUMINAMATH_CALUDE_university_applications_l97_9754

theorem university_applications (n m k : ℕ) (h1 : n = 7) (h2 : m = 2) (h3 : k = 4) : 
  (Nat.choose (n - m + 1) k) + (Nat.choose m 1 * Nat.choose (n - m) (k - 1)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_university_applications_l97_9754


namespace NUMINAMATH_CALUDE_total_pennies_thrown_l97_9720

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
def pennies_thrown (rachelle gretchen rocky : ℕ) : ℕ := rachelle + gretchen + rocky

/-- Theorem: The total number of pennies thrown is 300 -/
theorem total_pennies_thrown : 
  ∀ (rachelle gretchen rocky : ℕ),
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rocky = gretchen / 3 →
  pennies_thrown rachelle gretchen rocky = 300 := by
sorry

end NUMINAMATH_CALUDE_total_pennies_thrown_l97_9720


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l97_9794

theorem walking_speed_calculation (total_distance : ℝ) (running_speed : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 0.75)
  (h4 : ∃ (walking_time running_time : ℝ), 
    walking_time + running_time = total_time ∧ 
    walking_time * walking_speed = running_time * running_speed ∧
    walking_time * walking_speed = total_distance / 2) :
  walking_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l97_9794


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l97_9797

/-- Proves that if the difference between compound interest and simple interest
    on a sum at 10% per annum for 2 years is Rs. 65, then the sum is Rs. 6500. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 65 → P = 6500 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l97_9797


namespace NUMINAMATH_CALUDE_probability_three_correct_out_of_seven_l97_9731

/-- The number of derangements of n elements -/
def derangement (n : ℕ) : ℕ := sorry

/-- The probability of exactly k people receiving their correct letter when n letters are randomly distributed to n people -/
def probability_correct_letters (n k : ℕ) : ℚ :=
  (Nat.choose n k * derangement (n - k)) / n.factorial

theorem probability_three_correct_out_of_seven :
  probability_correct_letters 7 3 = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_three_correct_out_of_seven_l97_9731


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l97_9734

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l97_9734


namespace NUMINAMATH_CALUDE_loss_60_l97_9718

/-- Represents the financial recording of a transaction amount in dollars -/
def record_transaction (amount : Int) : Int := amount

/-- Records a profit of $370 as +370 dollars -/
axiom profit_370 : record_transaction 370 = 370

/-- Proves that a loss of $60 is recorded as -60 dollars -/
theorem loss_60 : record_transaction (-60) = -60 := by sorry

end NUMINAMATH_CALUDE_loss_60_l97_9718


namespace NUMINAMATH_CALUDE_factors_of_1320_l97_9767

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def count_factors (factorization : List (ℕ × ℕ)) : ℕ := sorry

theorem factors_of_1320 :
  let factorization := prime_factorization 1320
  count_factors factorization = 24 := by sorry

end NUMINAMATH_CALUDE_factors_of_1320_l97_9767


namespace NUMINAMATH_CALUDE_ellipse_conditions_l97_9725

/-- A curve defined by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate for a curve being an ellipse -/
def is_ellipse (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0 ∧ c.a ≠ c.b

/-- The conditions a > 0 and b > 0 -/
def positive_conditions (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0

theorem ellipse_conditions (c : Curve) :
  (positive_conditions c → is_ellipse c) ∧
  ¬(is_ellipse c → positive_conditions c) :=
sorry

end NUMINAMATH_CALUDE_ellipse_conditions_l97_9725


namespace NUMINAMATH_CALUDE_expression_value_l97_9751

theorem expression_value : 
  (2 * 6) / (12 * 14) * (3 * 12 * 14) / (2 * 6 * 3) * 2 = 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_l97_9751


namespace NUMINAMATH_CALUDE_factor_expression_l97_9746

theorem factor_expression (x : ℝ) : 5*x*(x-4) + 6*(x-4) = (x-4)*(5*x+6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l97_9746


namespace NUMINAMATH_CALUDE_triangle_inequality_l97_9707

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l97_9707


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l97_9766

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def total_jelly_beans : ℕ := 4000
def red_jelly_beans : ℕ := (3 * total_jelly_beans) / 4
def coconut_flavored_jelly_beans : ℕ := 750

theorem jelly_bean_ratio :
  Ratio.mk coconut_flavored_jelly_beans red_jelly_beans = Ratio.mk 1 4 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l97_9766


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l97_9760

theorem geometric_sequence_value (a : ℝ) : 
  (∃ (r : ℝ), 1 * r = a ∧ a * r = (1/16 : ℝ)) → 
  (a = (1/4 : ℝ) ∨ a = -(1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l97_9760


namespace NUMINAMATH_CALUDE_tennis_preference_theorem_l97_9723

/-- The percentage of students preferring tennis when combining two schools -/
def combined_tennis_preference 
  (central_students : ℕ) 
  (central_tennis_percentage : ℚ)
  (north_students : ℕ)
  (north_tennis_percentage : ℚ) : ℚ :=
  ((central_students : ℚ) * central_tennis_percentage + 
   (north_students : ℚ) * north_tennis_percentage) / 
  ((central_students + north_students) : ℚ)

theorem tennis_preference_theorem : 
  combined_tennis_preference 1800 (25/100) 3000 (35/100) = 31/100 := by
  sorry

end NUMINAMATH_CALUDE_tennis_preference_theorem_l97_9723


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l97_9795

theorem tan_sum_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l97_9795


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l97_9704

theorem laptop_sticker_price (sticker_price : ℝ) : 
  (0.8 * sticker_price - 50 = 0.7 * sticker_price + 30) → 
  sticker_price = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l97_9704


namespace NUMINAMATH_CALUDE_product_equality_l97_9764

theorem product_equality (a b : ℝ) (h1 : 4 * a = 30) (h2 : 5 * b = 30) : 40 * a * b = 1800 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l97_9764


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l97_9774

theorem smallest_prime_dividing_sum : ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  p ∣ (2^11 + 7^13) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (2^11 + 7^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l97_9774


namespace NUMINAMATH_CALUDE_chessboard_3x1_rectangles_impossible_l97_9755

theorem chessboard_3x1_rectangles_impossible : ¬ ∃ n : ℕ, 3 * n = 64 := by sorry

end NUMINAMATH_CALUDE_chessboard_3x1_rectangles_impossible_l97_9755


namespace NUMINAMATH_CALUDE_pattern_steps_l97_9735

/-- The number of sticks used in the kth step of the pattern -/
def sticks_in_step (k : ℕ) : ℕ := 2 * k + 1

/-- The total number of sticks used in a pattern with n steps -/
def total_sticks (n : ℕ) : ℕ := n^2

/-- Theorem: If a stair-like pattern is constructed where the kth step uses 2k + 1 sticks,
    and the total number of sticks used is 169, then the number of steps in the pattern is 13 -/
theorem pattern_steps :
  ∀ n : ℕ, (∀ k : ℕ, k ≤ n → sticks_in_step k = 2 * k + 1) →
  total_sticks n = 169 → n = 13 := by sorry

end NUMINAMATH_CALUDE_pattern_steps_l97_9735


namespace NUMINAMATH_CALUDE_win_sectors_area_l97_9748

theorem win_sectors_area (r : ℝ) (p : ℝ) : 
  r = 15 → p = 3/7 → (p * π * r^2) = 675*π/7 := by sorry

end NUMINAMATH_CALUDE_win_sectors_area_l97_9748


namespace NUMINAMATH_CALUDE_inequality_proof_binomial_inequality_l97_9715

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a / Real.sqrt b + b / Real.sqrt a > Real.sqrt a + Real.sqrt b :=
by sorry

theorem binomial_inequality (x : ℝ) (m : ℕ) (hx : x > -1) (hm : m > 0) :
  (1 + x)^m ≥ 1 + m * x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_binomial_inequality_l97_9715


namespace NUMINAMATH_CALUDE_problem_solution_l97_9762

theorem problem_solution (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * a - (1/3 : ℝ)
  let B := -a^2 + (1/2 : ℝ) * a * b + (2/3 : ℝ)
  (a + 1)^2 + |b + 2| = 0 → 4 * A - (3 * A - 2 * B) = 11 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l97_9762


namespace NUMINAMATH_CALUDE_sum_of_cubes_l97_9716

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : 
  a^3 + b^3 = 26 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l97_9716


namespace NUMINAMATH_CALUDE_product_evaluation_l97_9705

theorem product_evaluation : (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7 - 1) * 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l97_9705


namespace NUMINAMATH_CALUDE_max_cherries_proof_l97_9780

/-- Represents the number of fruits Alice can buy -/
structure FruitPurchase where
  apples : ℕ
  bananas : ℕ
  cherries : ℕ

/-- Checks if a purchase satisfies all conditions -/
def isValidPurchase (p : FruitPurchase) : Prop :=
  p.apples ≥ 1 ∧ p.bananas ≥ 1 ∧ p.cherries ≥ 1 ∧
  2 * p.apples + 5 * p.bananas + 10 * p.cherries = 100

/-- The maximum number of cherries Alice can purchase -/
def maxCherries : ℕ := 8

theorem max_cherries_proof :
  (∃ p : FruitPurchase, isValidPurchase p ∧ p.cherries = maxCherries) ∧
  (∀ p : FruitPurchase, isValidPurchase p → p.cherries ≤ maxCherries) :=
sorry

end NUMINAMATH_CALUDE_max_cherries_proof_l97_9780


namespace NUMINAMATH_CALUDE_markers_final_count_l97_9721

def markers_problem (initial : ℕ) (robert_gave : ℕ) (sarah_took : ℕ) (teacher_multiplier : ℕ) : ℕ :=
  let after_robert := initial + robert_gave
  let after_sarah := after_robert - sarah_took
  let after_teacher := after_sarah + teacher_multiplier * after_sarah
  (after_teacher) / 2

theorem markers_final_count : 
  markers_problem 217 109 35 3 = 582 := by sorry

end NUMINAMATH_CALUDE_markers_final_count_l97_9721


namespace NUMINAMATH_CALUDE_weight_moved_is_540_l97_9714

/-- Calculates the total weight moved in three triples given the initial back squat and increase -/
def weightMovedInThreeTriples (initialBackSquat : ℝ) (backSquatIncrease : ℝ) : ℝ :=
  let newBackSquat := initialBackSquat + backSquatIncrease
  let frontSquat := 0.8 * newBackSquat
  let tripleWeight := 0.9 * frontSquat
  3 * tripleWeight

/-- Theorem stating that given John's initial back squat of 200 kg and an increase of 50 kg,
    the total weight moved in three triples is 540 kg -/
theorem weight_moved_is_540 :
  weightMovedInThreeTriples 200 50 = 540 := by
  sorry

#eval weightMovedInThreeTriples 200 50

end NUMINAMATH_CALUDE_weight_moved_is_540_l97_9714


namespace NUMINAMATH_CALUDE_cookie_box_count_l97_9730

/-- The number of cookies in a bag -/
def cookies_per_bag : ℕ := 7

/-- The number of cookies in a box -/
def cookies_per_box : ℕ := 12

/-- The number of bags used for comparison -/
def num_bags : ℕ := 9

/-- The additional number of cookies in boxes compared to bags -/
def extra_cookies : ℕ := 33

/-- The number of boxes -/
def num_boxes : ℕ := 8

theorem cookie_box_count :
  num_boxes * cookies_per_box = num_bags * cookies_per_bag + extra_cookies :=
by sorry

end NUMINAMATH_CALUDE_cookie_box_count_l97_9730


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l97_9781

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l97_9781


namespace NUMINAMATH_CALUDE_fraction_equality_l97_9744

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 2 * b) (h2 : b ≠ 0) :
  let x := a / b
  (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l97_9744


namespace NUMINAMATH_CALUDE_inequality_solution_range_l97_9737

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - m * x + (m - 1) ≥ 0) → m ≥ 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l97_9737


namespace NUMINAMATH_CALUDE_last_locker_opened_l97_9768

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the process of toggling lockers -/
def toggle_lockers (n : ℕ) (k : ℕ) : List LockerState → List LockerState :=
  sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Finds the largest perfect square less than or equal to a given number -/
def largest_perfect_square_le (n : ℕ) : ℕ :=
  sorry

theorem last_locker_opened (num_lockers : ℕ) (num_lockers_eq : num_lockers = 500) :
  largest_perfect_square_le num_lockers = 484 :=
sorry

end NUMINAMATH_CALUDE_last_locker_opened_l97_9768


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l97_9710

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) 
  (h1 : q → p)  -- p is necessary for q
  (h2 : ¬(p → q))  -- p is not sufficient for q
  : (¬p → ¬q) ∧ ¬(¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l97_9710


namespace NUMINAMATH_CALUDE_kite_only_always_perpendicular_diagonals_l97_9743

-- Define the types of quadrilaterals
inductive Quadrilateral
  | Rhombus
  | Rectangle
  | Square
  | Kite
  | IsoscelesTrapezoid

-- Define a property for perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Kite => true
  | _ => false

-- Theorem statement
theorem kite_only_always_perpendicular_diagonals :
  ∀ q : Quadrilateral, has_perpendicular_diagonals q ↔ q = Quadrilateral.Kite :=
by sorry

end NUMINAMATH_CALUDE_kite_only_always_perpendicular_diagonals_l97_9743


namespace NUMINAMATH_CALUDE_custom_op_difference_l97_9712

/-- Custom operation @ defined as x@y = xy - 3x -/
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

/-- Theorem stating that (6@2)-(2@6) = -12 -/
theorem custom_op_difference : at_op 6 2 - at_op 2 6 = -12 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_difference_l97_9712


namespace NUMINAMATH_CALUDE_crocodile_count_l97_9793

theorem crocodile_count (total : ℕ) (alligators : ℕ) (vipers : ℕ) 
  (h1 : total = 50)
  (h2 : alligators = 23)
  (h3 : vipers = 5)
  (h4 : ∃ crocodiles : ℕ, total = crocodiles + alligators + vipers) :
  ∃ crocodiles : ℕ, crocodiles = 22 ∧ total = crocodiles + alligators + vipers :=
by sorry

end NUMINAMATH_CALUDE_crocodile_count_l97_9793


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_sum_of_three_numbers_proof_l97_9726

theorem sum_of_three_numbers : ℕ → ℕ → ℕ → Prop :=
  fun second first third =>
    first = 2 * second ∧
    third = first / 3 ∧
    second = 60 →
    first + second + third = 220

-- The proof is omitted
theorem sum_of_three_numbers_proof : sum_of_three_numbers 60 120 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_sum_of_three_numbers_proof_l97_9726


namespace NUMINAMATH_CALUDE_mike_toys_count_l97_9727

/-- Proves that Mike has 6 toys given the conditions of the problem -/
theorem mike_toys_count :
  ∀ (mike annie tom : ℕ),
  annie = 3 * mike →
  tom = annie + 2 →
  mike + annie + tom = 56 →
  mike = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mike_toys_count_l97_9727


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l97_9778

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1) + n * (n - 1)

-- Define b_n
def b (n : ℕ) : ℚ := (-1)^(n-1) * (4 * n) / (a n * a (n+1))

-- Define T_n (sum of first n terms of b_n)
def T (n : ℕ) : ℚ :=
  if n % 2 = 0 then (2 * n) / (2 * n + 1)
  else (2 * n + 2) / (2 * n + 1)

-- Theorem statement
theorem arithmetic_sequence_proof :
  (∀ n : ℕ, S (n+1) - S n = 2) ∧  -- Common difference is 2
  (S 2)^2 = S 1 * S 4 ∧           -- S_1, S_2, S_4 form a geometric sequence
  (∀ n : ℕ, a n = 2 * n - 1) ∧    -- General formula for a_n
  (∀ n : ℕ, T n = if n % 2 = 0 then (2 * n) / (2 * n + 1) else (2 * n + 2) / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l97_9778


namespace NUMINAMATH_CALUDE_jonas_bookshelves_l97_9740

/-- Calculates the maximum number of bookshelves that can fit in a room -/
def max_bookshelves (total_space desk_space shelf_space : ℕ) : ℕ :=
  (total_space - desk_space) / shelf_space

/-- Proves that the maximum number of bookshelves in Jonas' room is 3 -/
theorem jonas_bookshelves :
  max_bookshelves 400 160 80 = 3 := by
  sorry

#eval max_bookshelves 400 160 80

end NUMINAMATH_CALUDE_jonas_bookshelves_l97_9740


namespace NUMINAMATH_CALUDE_dependent_variable_influence_l97_9700

/-- Linear regression model -/
structure LinearRegressionModel where
  y : ℝ → ℝ  -- Dependent variable
  x : ℝ      -- Independent variable
  b : ℝ      -- Slope
  a : ℝ      -- Intercept
  e : ℝ → ℝ  -- Random error term

/-- The dependent variable is influenced by both the independent variable and other factors -/
theorem dependent_variable_influence (model : LinearRegressionModel) :
  ∃ (x₁ x₂ : ℝ), model.y x₁ ≠ model.y x₂ ∧ model.x = model.x :=
by sorry

end NUMINAMATH_CALUDE_dependent_variable_influence_l97_9700


namespace NUMINAMATH_CALUDE_rectangle_area_l97_9703

theorem rectangle_area : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧
  x * y = (x + 3) * (y - 1) ∧
  x * y = (x - 4) * (y + 1.5) ∧
  x * y = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l97_9703


namespace NUMINAMATH_CALUDE_sum_of_distinct_words_l97_9752

/-- Calculates the number of distinct permutations of a word with repeated letters -/
def distinctPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "САМСА" has 5 total letters and 2 letters that repeat twice each -/
def samsa : ℕ := distinctPermutations 5 [2, 2]

/-- The word "ПАСТА" has 5 total letters and 1 letter that repeats twice -/
def pasta : ℕ := distinctPermutations 5 [2]

theorem sum_of_distinct_words : samsa + pasta = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_words_l97_9752


namespace NUMINAMATH_CALUDE_f_value_at_2_l97_9777

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 4

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 2 → f a b 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l97_9777


namespace NUMINAMATH_CALUDE_golden_ratio_properties_l97_9789

theorem golden_ratio_properties : ∃ a : ℝ, 
  (a = (Real.sqrt 5 - 1) / 2) ∧ 
  (a^2 + a - 1 = 0) ∧ 
  (a^3 - 2*a + 2015 = 2014) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_properties_l97_9789


namespace NUMINAMATH_CALUDE_uv_value_l97_9702

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -2/3 * x + 6

-- Define points P and Q
def P : ℝ × ℝ := (9, 0)
def Q : ℝ × ℝ := (0, 6)

-- Define point R
def R (u v : ℝ) : ℝ × ℝ := (u, v)

-- Define that R is on the line segment PQ
def R_on_PQ (u v : ℝ) : Prop :=
  line_equation u v ∧ 0 ≤ u ∧ u ≤ 9

-- Define the area ratio condition
def area_condition (u v : ℝ) : Prop :=
  (1/2 * 9 * 6) = 2 * (1/2 * 9 * v)

-- Theorem statement
theorem uv_value (u v : ℝ) 
  (h1 : R_on_PQ u v) 
  (h2 : area_condition u v) : 
  u * v = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_uv_value_l97_9702


namespace NUMINAMATH_CALUDE_monomial_exponent_equality_l97_9773

/-- Two monomials are of the same type if they have the same exponents for each variable. -/
def same_type_monomial (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

/-- The exponents of a monomial of the form x^a * y^b. -/
def monomial_exponents (a b : ℕ) : ℕ → ℕ
| 0 => a  -- exponent of x
| 1 => b  -- exponent of y
| _ => 0  -- all other variables have exponent 0

theorem monomial_exponent_equality (m : ℕ) :
  same_type_monomial (monomial_exponents (2 * m) 3) (monomial_exponents 6 3) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_equality_l97_9773


namespace NUMINAMATH_CALUDE_power_product_equality_l97_9708

theorem power_product_equality : (15 : ℕ)^2 * 8^3 * 256 = 29491200 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l97_9708


namespace NUMINAMATH_CALUDE_simplify_expression_l97_9739

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l97_9739


namespace NUMINAMATH_CALUDE_min_height_is_eleven_l97_9706

/-- Represents the dimensions of a rectangular box with square bases -/
structure BoxDimensions where
  base : ℝ
  height : ℝ

/-- Calculates the surface area of the box -/
def surfaceArea (d : BoxDimensions) : ℝ :=
  2 * d.base^2 + 4 * d.base * d.height

/-- Checks if the box dimensions satisfy the given conditions -/
def isValidBox (d : BoxDimensions) : Prop :=
  d.height = d.base + 6 ∧ surfaceArea d ≥ 150 ∧ d.base > 0

theorem min_height_is_eleven :
  ∀ d : BoxDimensions, isValidBox d → d.height ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_height_is_eleven_l97_9706


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l97_9758

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_condition (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : a 1 > 0) :
  (∀ h : a 3 < a 6, a 1 < a 3) ∧
  ¬(∀ h : a 1 < a 3, a 3 < a 6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l97_9758


namespace NUMINAMATH_CALUDE_tan_squared_sum_l97_9729

theorem tan_squared_sum (a b : ℝ) 
  (h1 : (Real.sin a)^2 / (Real.cos b)^2 + (Real.sin b)^2 / (Real.cos a)^2 = 2)
  (h2 : (Real.cos a)^3 / (Real.sin b)^3 + (Real.cos b)^3 / (Real.sin a)^3 = 4) :
  (Real.tan a)^2 / (Real.tan b)^2 + (Real.tan b)^2 / (Real.tan a)^2 = 30/13 := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_sum_l97_9729


namespace NUMINAMATH_CALUDE_total_books_is_80_l97_9761

/-- Calculates the total number of books bought given the conditions -/
def total_books (total_price : ℕ) (math_book_price : ℕ) (history_book_price : ℕ) (math_books_bought : ℕ) : ℕ :=
  let history_books_bought := (total_price - math_book_price * math_books_bought) / history_book_price
  math_books_bought + history_books_bought

/-- Proves that the total number of books bought is 80 under the given conditions -/
theorem total_books_is_80 :
  total_books 390 4 5 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_80_l97_9761
