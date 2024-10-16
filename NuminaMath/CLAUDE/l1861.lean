import Mathlib

namespace NUMINAMATH_CALUDE_freds_allowance_l1861_186167

theorem freds_allowance (allowance : ℝ) : 
  allowance / 2 + 6 = 14 → allowance = 16 := by sorry

end NUMINAMATH_CALUDE_freds_allowance_l1861_186167


namespace NUMINAMATH_CALUDE_store_inventory_problem_l1861_186174

/-- Represents the inventory of a store selling pomelos and watermelons -/
structure StoreInventory where
  pomelos : ℕ
  watermelons : ℕ

/-- Represents the daily sales of pomelos and watermelons -/
structure DailySales where
  pomelos : ℕ
  watermelons : ℕ

/-- The theorem statement for the store inventory problem -/
theorem store_inventory_problem 
  (initial : StoreInventory)
  (sales : DailySales)
  (days : ℕ) :
  initial.watermelons = 3 * initial.pomelos →
  sales.pomelos = 20 →
  sales.watermelons = 30 →
  days = 3 →
  initial.watermelons - days * sales.watermelons = 
    4 * (initial.pomelos - days * sales.pomelos) - 26 →
  initial.pomelos = 176 := by
  sorry


end NUMINAMATH_CALUDE_store_inventory_problem_l1861_186174


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l1861_186116

-- Part 1
theorem simplify_expression (a : ℝ) : -2*a^2 + 3 - (3*a^2 - 6*a + 1) + 3 = -5*a^2 + 6*a + 5 := by
  sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (hx : x = -2) (hy : y = -3) :
  1/2*x - 2*(x - 1/3*y^2) + (-3/2*x + 1/3*y^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l1861_186116


namespace NUMINAMATH_CALUDE_angle_relation_in_3x2_right_triangle_l1861_186179

theorem angle_relation_in_3x2_right_triangle (α β : Real) :
  (α + β = Real.pi / 2) →  -- Sum of angles in right triangle
  (2 * α + β = Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_relation_in_3x2_right_triangle_l1861_186179


namespace NUMINAMATH_CALUDE_max_value_constraint_l1861_186137

theorem max_value_constraint (x y : ℝ) : 
  x^2 + y^2 = 18*x + 8*y + 10 → (∀ a b : ℝ, a^2 + b^2 = 18*a + 8*b + 10 → 4*x + 3*y ≥ 4*a + 3*b) → 4*x + 3*y = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1861_186137


namespace NUMINAMATH_CALUDE_chessboard_3x1_rectangles_impossible_l1861_186125

theorem chessboard_3x1_rectangles_impossible : ¬ ∃ n : ℕ, 3 * n = 64 := by sorry

end NUMINAMATH_CALUDE_chessboard_3x1_rectangles_impossible_l1861_186125


namespace NUMINAMATH_CALUDE_total_regular_games_count_l1861_186188

def num_teams : ℕ := 15
def top_teams : ℕ := 5
def mid_teams : ℕ := 5
def bottom_teams : ℕ := 5

def top_vs_top_games : ℕ := 12
def top_vs_others_games : ℕ := 8
def mid_vs_mid_games : ℕ := 10
def mid_vs_top_games : ℕ := 6
def bottom_vs_bottom_games : ℕ := 8

def combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem total_regular_games_count : 
  (combinations top_teams 2 * top_vs_top_games + 
   top_teams * (num_teams - top_teams) * top_vs_others_games +
   combinations mid_teams 2 * mid_vs_mid_games + 
   mid_teams * top_teams * mid_vs_top_games +
   combinations bottom_teams 2 * bottom_vs_bottom_games) = 850 := by
  sorry

end NUMINAMATH_CALUDE_total_regular_games_count_l1861_186188


namespace NUMINAMATH_CALUDE_degree_of_minus_x_cubed_y_is_four_degree_of_minus_x_cubed_y_is_not_three_l1861_186128

/-- Represents a monomial in variables x and y -/
structure Monomial :=
  (coeff : ℤ)
  (x_power : ℕ)
  (y_power : ℕ)

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.x_power + m.y_power

/-- The monomial -x³y -/
def mono : Monomial :=
  { coeff := -1, x_power := 3, y_power := 1 }

/-- Theorem stating that the degree of -x³y is 4 -/
theorem degree_of_minus_x_cubed_y_is_four :
  degree mono = 4 :=
sorry

/-- Theorem stating that the degree of -x³y is not 3 -/
theorem degree_of_minus_x_cubed_y_is_not_three :
  degree mono ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_degree_of_minus_x_cubed_y_is_four_degree_of_minus_x_cubed_y_is_not_three_l1861_186128


namespace NUMINAMATH_CALUDE_total_shells_is_195_l1861_186104

/-- The total number of conch shells owned by David, Mia, Ava, and Alice -/
def total_shells (david_shells : ℕ) : ℕ :=
  let mia_shells := 4 * david_shells
  let ava_shells := mia_shells + 20
  let alice_shells := ava_shells / 2
  david_shells + mia_shells + ava_shells + alice_shells

/-- Theorem stating that the total number of shells is 195 when David has 15 shells -/
theorem total_shells_is_195 : total_shells 15 = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_195_l1861_186104


namespace NUMINAMATH_CALUDE_jorges_gifts_count_l1861_186123

/-- The number of gifts Jorge gave at Rosalina's wedding --/
def jorges_gifts (total_gifts emilios_gifts pedros_gifts : ℕ) : ℕ :=
  total_gifts - (emilios_gifts + pedros_gifts)

theorem jorges_gifts_count :
  jorges_gifts 21 11 4 = 6 :=
by sorry

end NUMINAMATH_CALUDE_jorges_gifts_count_l1861_186123


namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1861_186148

theorem cubic_equation_unique_solution :
  ∃! (x : ℤ), x^3 + (x+1)^3 + (x+2)^3 = (x+3)^3 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1861_186148


namespace NUMINAMATH_CALUDE_fraction_stayed_home_l1861_186199

theorem fraction_stayed_home (total : ℚ) (fun_fraction : ℚ) (youth_fraction : ℚ)
  (h1 : fun_fraction = 5 / 13)
  (h2 : youth_fraction = 4 / 13)
  (h3 : total = 1) :
  total - (fun_fraction + youth_fraction) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_stayed_home_l1861_186199


namespace NUMINAMATH_CALUDE_apples_in_basket_after_removal_l1861_186163

/-- Given a total number of apples and baskets, and a number of apples removed from each basket,
    calculate the number of apples remaining in each basket. -/
def applesPerBasket (totalApples : ℕ) (numBaskets : ℕ) (applesRemoved : ℕ) : ℕ :=
  (totalApples / numBaskets) - applesRemoved

/-- Theorem stating that for the given problem, each basket contains 9 apples after removal. -/
theorem apples_in_basket_after_removal :
  applesPerBasket 128 8 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_after_removal_l1861_186163


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1861_186147

theorem min_value_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2) ≥ 4 := by
  sorry

theorem min_value_attainable : ∃ x y : ℝ, (x*y - 2)^2 + (x^2 + y^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1861_186147


namespace NUMINAMATH_CALUDE_ellipse_condition_l1861_186157

/-- 
Given the equation x^2 + 9y^2 - 6x + 18y = k, 
this theorem states that it represents a non-degenerate ellipse 
if and only if k > -18.
-/
theorem ellipse_condition (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 9*y^2 - 6*x + 18*y = k) → 
  (∃ a b h1 h2 : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, (x - h1)^2 / a^2 + (y - h2)^2 / b^2 = 1) ↔ 
  k > -18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1861_186157


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1861_186165

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 2 -/
theorem perpendicular_vectors_m_value 
  (a b : ℝ × ℝ) 
  (h1 : a = (-2, 3)) 
  (h2 : b = (3, m)) 
  (h3 : a.fst * b.fst + a.snd * b.snd = 0) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1861_186165


namespace NUMINAMATH_CALUDE_proposition_relationship_l1861_186127

theorem proposition_relationship (x y : ℝ) :
  (∀ x y, x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 3) ∧ x + y = 5) := by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l1861_186127


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1861_186181

theorem inequality_solution_set :
  let S : Set ℝ := {x | (3 - x) / (2 * x - 4) < 1}
  S = {x | x < 2 ∨ x > 7/3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1861_186181


namespace NUMINAMATH_CALUDE_NaNO3_formed_l1861_186196

/-- Represents a chemical compound in a reaction -/
structure Compound where
  name : String
  moles : ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

def balancedReaction : Reaction :=
  { reactants := [
      { name := "NH4NO3", moles := 1 },
      { name := "NaOH", moles := 1 }
    ],
    products := [
      { name := "NaNO3", moles := 1 },
      { name := "NH3", moles := 1 },
      { name := "H2O", moles := 1 }
    ]
  }

def initialNH4NO3 : Compound :=
  { name := "NH4NO3", moles := 3 }

def initialNaOH : Compound :=
  { name := "NaOH", moles := 3 }

/-- Calculates the moles of a product formed in a reaction -/
def molesFormed (reaction : Reaction) (initialReactants : List Compound) (product : String) : ℝ :=
  sorry

theorem NaNO3_formed :
  molesFormed balancedReaction [initialNH4NO3, initialNaOH] "NaNO3" = 3 := by
  sorry

end NUMINAMATH_CALUDE_NaNO3_formed_l1861_186196


namespace NUMINAMATH_CALUDE_equation_solution_system_solution_l1861_186144

-- Define the equation
def equation (x : ℝ) : Prop := 64 * (x - 1)^3 + 27 = 0

-- Define the system of equations
def system (x y : ℝ) : Prop := x + y = 3 ∧ 2*x - 3*y = 6

-- Theorem for the equation solution
theorem equation_solution : ∃ x : ℝ, equation x ∧ x = 1/4 := by sorry

-- Theorem for the system of equations solution
theorem system_solution : ∃ x y : ℝ, system x y ∧ x = 3 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_equation_solution_system_solution_l1861_186144


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1861_186143

theorem z_in_first_quadrant (z : ℂ) (h : (1 : ℂ) + Complex.I = Complex.I / z) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1861_186143


namespace NUMINAMATH_CALUDE_parabola_shift_right_one_unit_l1861_186101

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

theorem parabola_shift_right_one_unit :
  let original := Parabola.mk (-1/2) 0 0
  let shifted := shift_parabola original 1
  shifted = Parabola.mk (-1/2) 1 (-1/2) := by sorry

end NUMINAMATH_CALUDE_parabola_shift_right_one_unit_l1861_186101


namespace NUMINAMATH_CALUDE_correct_miscopied_value_l1861_186149

/-- Given a set of values with an incorrect mean due to one miscopied value,
    calculate the correct value that should have been recorded. -/
theorem correct_miscopied_value
  (n : ℕ) -- Total number of values
  (initial_mean : ℚ) -- Initial (incorrect) mean
  (wrong_value : ℚ) -- Value that was incorrectly recorded
  (correct_mean : ℚ) -- Correct mean after fixing the error
  (h1 : n = 30) -- There are 30 values
  (h2 : initial_mean = 150) -- The initial mean was 150
  (h3 : wrong_value = 135) -- The value was incorrectly recorded as 135
  (h4 : correct_mean = 151) -- The correct mean is 151
  : ℚ := -- The theorem returns a rational number
by
  -- The proof goes here
  sorry

#check correct_miscopied_value

end NUMINAMATH_CALUDE_correct_miscopied_value_l1861_186149


namespace NUMINAMATH_CALUDE_replaced_student_weight_is_96_l1861_186130

/-- The weight of the replaced student given the conditions of the problem -/
def replaced_student_weight (initial_students : ℕ) (new_student_weight : ℝ) (average_decrease : ℝ) : ℝ :=
  let total_weight_decrease := initial_students * average_decrease
  let weight_difference := total_weight_decrease + new_student_weight
  weight_difference

/-- Theorem stating that under the given conditions, the replaced student's weight is 96 kg -/
theorem replaced_student_weight_is_96 :
  replaced_student_weight 4 64 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_is_96_l1861_186130


namespace NUMINAMATH_CALUDE_fishing_line_length_l1861_186194

/-- Given information about fishing line reels and sections, prove the length of each reel. -/
theorem fishing_line_length (num_reels : ℕ) (section_length : ℝ) (num_sections : ℕ) :
  num_reels = 3 →
  section_length = 10 →
  num_sections = 30 →
  (num_sections * section_length) / num_reels = 100 := by
  sorry

#check fishing_line_length

end NUMINAMATH_CALUDE_fishing_line_length_l1861_186194


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l1861_186190

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the property of being an ellipse
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (F₁ F₂ : ℝ × ℝ) (a : ℝ),
    a > Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) / 2 ∧
    ∀ (x y : ℝ), f x y ↔ 
      Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
      Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 2 * a

-- Theorem statement
theorem equation_describes_ellipse : is_ellipse equation := by
  sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l1861_186190


namespace NUMINAMATH_CALUDE_researchers_distribution_l1861_186162

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with at least one object in each box. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of researchers -/
def num_researchers : ℕ := 4

/-- The theorem stating that the number of ways to distribute 4 researchers
    to 3 schools, with at least one researcher in each school, is 36. -/
theorem researchers_distribution :
  distribute num_researchers num_schools = 36 := by sorry

end NUMINAMATH_CALUDE_researchers_distribution_l1861_186162


namespace NUMINAMATH_CALUDE_conference_arrangements_l1861_186192

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial / (2^k)

theorem conference_arrangements :
  number_of_arrangements 8 2 = 10080 := by
  sorry

end NUMINAMATH_CALUDE_conference_arrangements_l1861_186192


namespace NUMINAMATH_CALUDE_coffee_conference_theorem_l1861_186142

/-- Represents the number of participants who went for coffee -/
def coffee_goers (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ (remaining : ℕ), 
    remaining > 0 ∧ 
    remaining < n ∧ 
    remaining % 2 = 0 ∧ 
    k = n - remaining}

/-- The theorem stating the possible number of coffee goers -/
theorem coffee_conference_theorem :
  coffee_goers 14 = {6, 8, 10, 12} :=
sorry


end NUMINAMATH_CALUDE_coffee_conference_theorem_l1861_186142


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1861_186133

theorem no_integer_solutions :
  ∀ x : ℤ, x^5 - 31*x + 2015 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1861_186133


namespace NUMINAMATH_CALUDE_skee_ball_tickets_proof_l1861_186191

/-- The number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- The number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

/-- The number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := (remaining_tickets + spent_tickets) - whack_a_mole_tickets

theorem skee_ball_tickets_proof : skee_ball_tickets = 25 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_proof_l1861_186191


namespace NUMINAMATH_CALUDE_seconds_in_week_scientific_correct_l1861_186145

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of seconds in a week -/
def seconds_in_week : ℕ := 604800

/-- The scientific notation representation of the number of seconds in a week -/
def seconds_in_week_scientific : ScientificNotation :=
  { coefficient := 6.048
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem seconds_in_week_scientific_correct :
  (seconds_in_week_scientific.coefficient * (10 : ℝ) ^ seconds_in_week_scientific.exponent) = seconds_in_week := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_week_scientific_correct_l1861_186145


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l1861_186105

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  -- The number of smaller cubes painted on exactly 2 faces
  two_face_cubes : ℕ
  -- The total number of smaller cubes created
  total_cubes : ℕ

/-- Theorem stating that a cube cut into equal smaller cubes with 12 two-face cubes results in 27 total cubes -/
theorem cut_cube_theorem (c : CutCube) (h : c.two_face_cubes = 12) : c.total_cubes = 27 := by
  sorry


end NUMINAMATH_CALUDE_cut_cube_theorem_l1861_186105


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1861_186139

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

/-- The asymptote equation -/
def asymptote (m : ℝ) (x y : ℝ) : Prop := y = m * x ∨ y = -m * x

/-- Theorem: The positive slope of the asymptotes of the given hyperbola is 3/2 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), hyperbola x y → (∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > ε → asymptote m (x + δ) (y + δ))) ∧
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1861_186139


namespace NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l1861_186155

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 dots out of 25 -/
def totalWays : ℕ := Nat.choose totalDots chosenDots

/-- The number of horizontal lines in the grid -/
def horizontalLines : ℕ := gridSize

/-- The number of vertical lines in the grid -/
def verticalLines : ℕ := gridSize

/-- The number of major diagonals in the grid -/
def majorDiagonals : ℕ := 2

/-- The total number of collinear sets of 4 dots -/
def collinearSets : ℕ := horizontalLines + verticalLines + majorDiagonals

/-- The probability of selecting four collinear dots -/
def collinearProbability : ℚ := collinearSets / totalWays

theorem collinear_probability_in_5x5_grid :
  collinearProbability = 6 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l1861_186155


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1861_186177

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := (2 - i) / (1 + i)
  Complex.im z = -3/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1861_186177


namespace NUMINAMATH_CALUDE_banana_change_calculation_emily_banana_change_l1861_186146

/-- Calculates the change received when buying bananas with a discount --/
theorem banana_change_calculation (num_bananas : ℕ) (cost_per_banana : ℚ) 
  (discount_threshold : ℕ) (discount_rate : ℚ) (paid_amount : ℚ) : ℚ :=
  let total_cost := num_bananas * cost_per_banana
  let discounted_cost := if num_bananas > discount_threshold 
    then total_cost * (1 - discount_rate) 
    else total_cost
  paid_amount - discounted_cost

/-- Proves that Emily received $8.65 in change --/
theorem emily_banana_change : 
  banana_change_calculation 5 (30/100) 4 (10/100) 10 = 865/100 := by
  sorry

end NUMINAMATH_CALUDE_banana_change_calculation_emily_banana_change_l1861_186146


namespace NUMINAMATH_CALUDE_triangle_vector_division_l1861_186126

/-- Given a triangle ABC with point M on side BC such that BM:MC = 2:5,
    and vectors AB = a and AC = b, prove that AM = (2/7)a + (5/7)b. -/
theorem triangle_vector_division (A B C M : EuclideanSpace ℝ (Fin 3))
  (a b : EuclideanSpace ℝ (Fin 3)) (h : B ≠ C) :
  (B - M) = (5 / 7 : ℝ) • (C - B) →
  (A - B) = a →
  (A - C) = -b →
  (A - M) = (2 / 7 : ℝ) • a + (5 / 7 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_division_l1861_186126


namespace NUMINAMATH_CALUDE_final_lives_correct_tiffany_final_lives_l1861_186151

/-- Given an initial number of lives, lives lost, and a bonus multiplier,
    calculate the final number of lives after completing the bonus stage. -/
def finalLives (initialLives lostLives bonusMultiplier : ℕ) : ℕ :=
  let remainingLives := initialLives - lostLives
  remainingLives + bonusMultiplier * remainingLives

/-- Theorem: The final number of lives after the bonus stage is correct. -/
theorem final_lives_correct (initialLives lostLives bonusMultiplier : ℕ) 
    (h : lostLives ≤ initialLives) :
    finalLives initialLives lostLives bonusMultiplier = 
    (initialLives - lostLives) + bonusMultiplier * (initialLives - lostLives) := by
  sorry

/-- Corollary: For the specific case in the problem. -/
theorem tiffany_final_lives : 
    finalLives 250 58 3 = 768 := by
  sorry

end NUMINAMATH_CALUDE_final_lives_correct_tiffany_final_lives_l1861_186151


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1861_186158

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (a b : Line) (α β : Plane)
  (diff_lines : a ≠ b)
  (diff_planes : α ≠ β)
  (h1 : parallel_planes α β)
  (h2 : contained_in a α) :
  parallel_line_plane a β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1861_186158


namespace NUMINAMATH_CALUDE_triangle_sum_property_l1861_186118

theorem triangle_sum_property : ∃ (a b c d e f : ℤ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a = b + c + d ∧
  b = a + c + e ∧
  c = a + b + f :=
by sorry

end NUMINAMATH_CALUDE_triangle_sum_property_l1861_186118


namespace NUMINAMATH_CALUDE_remainder_46_pow_925_mod_21_l1861_186132

theorem remainder_46_pow_925_mod_21 : 46^925 % 21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_46_pow_925_mod_21_l1861_186132


namespace NUMINAMATH_CALUDE_no_solution_iff_v_eq_neg_one_l1861_186100

/-- The system of equations has no solution if and only if v = -1 -/
theorem no_solution_iff_v_eq_neg_one (v : ℝ) :
  (∀ x y z : ℝ, (x + y + z = v ∧ x + v*y + z = v ∧ x + y + v^2*z = v^2) → False) ↔ v = -1 :=
sorry

end NUMINAMATH_CALUDE_no_solution_iff_v_eq_neg_one_l1861_186100


namespace NUMINAMATH_CALUDE_magic_square_sum_l1861_186180

/-- Represents a 3x3 magic square with center 7 -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  x : ℤ
  y : ℤ := 7

/-- The magic sum of the square -/
def magicSum (s : MagicSquare) : ℤ := 22 + s.c

/-- Properties of the magic square -/
def isMagicSquare (s : MagicSquare) : Prop :=
  s.a + s.y + s.d = magicSum s ∧
  s.c + s.y + s.b = magicSum s ∧
  s.x + s.y + s.a = magicSum s ∧
  s.c + s.y + s.x = magicSum s

theorem magic_square_sum (s : MagicSquare) (h : isMagicSquare s) :
  s.x + s.y + s.a + s.b + s.c + s.d = 68 := by
  sorry

#check magic_square_sum

end NUMINAMATH_CALUDE_magic_square_sum_l1861_186180


namespace NUMINAMATH_CALUDE_xyz_sum_mod_9_l1861_186106

theorem xyz_sum_mod_9 (x y z : ℕ) : 
  0 < x ∧ x < 9 ∧
  0 < y ∧ y < 9 ∧
  0 < z ∧ z < 9 ∧
  (x * y * z) % 9 = 1 ∧
  (7 * z) % 9 = 4 ∧
  (8 * y) % 9 = (5 + y) % 9 →
  (x + y + z) % 9 = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_mod_9_l1861_186106


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l1861_186119

/-- Given the equation (x+y)^2 = x^2 + y^2 + 2x + 2y, prove it represents a hyperbola -/
theorem equation_represents_hyperbola (x y : ℝ) :
  (x + y)^2 = x^2 + y^2 + 2*x + 2*y ↔ (x - 1) * (y - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l1861_186119


namespace NUMINAMATH_CALUDE_distance_AC_l1861_186184

theorem distance_AC (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt 27
  let BC := 2
  let angle_ABC := 150 * Real.pi / 180
  let AC := Real.sqrt ((AB^2 + BC^2) - 2 * AB * BC * Real.cos angle_ABC)
  AC = 7 := by sorry

end NUMINAMATH_CALUDE_distance_AC_l1861_186184


namespace NUMINAMATH_CALUDE_solve_average_weight_problem_l1861_186102

def average_weight_problem (initial_average : ℝ) (new_man_weight : ℝ) (weight_increase : ℝ) (crew_size : ℕ) : Prop :=
  let replaced_weight := new_man_weight - (crew_size : ℝ) * weight_increase
  replaced_weight = initial_average * (crew_size : ℝ) + weight_increase * (crew_size : ℝ) - new_man_weight

theorem solve_average_weight_problem :
  average_weight_problem 0 71 1.8 10 = true :=
sorry

end NUMINAMATH_CALUDE_solve_average_weight_problem_l1861_186102


namespace NUMINAMATH_CALUDE_cube_of_complex_root_of_unity_l1861_186198

theorem cube_of_complex_root_of_unity (z : ℂ) : 
  z = Complex.cos (2 * Real.pi / 3) - Complex.I * Complex.sin (Real.pi / 3) → 
  z^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_complex_root_of_unity_l1861_186198


namespace NUMINAMATH_CALUDE_base4_to_decimal_conversion_l1861_186164

/-- Converts a base-4 digit to its decimal value -/
def base4ToDecimal (digit : Nat) : Nat :=
  match digit with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | _ => 0  -- Default case, should not occur in valid input

/-- Converts a list of base-4 digits to its decimal representation -/
def base4ListToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + (base4ToDecimal d) * (4 ^ (digits.length - 1 - i))) 0

theorem base4_to_decimal_conversion :
  base4ListToDecimal [0, 1, 3, 2, 0, 1, 3, 2] = 7710 := by
  sorry

#eval base4ListToDecimal [0, 1, 3, 2, 0, 1, 3, 2]

end NUMINAMATH_CALUDE_base4_to_decimal_conversion_l1861_186164


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l1861_186122

/-- Given a cubic polynomial with two equal integer roots, prove |ab| = 5832 -/
theorem cubic_polynomial_roots (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) ∧ 
   (r ≠ s)) → 
  |a * b| = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l1861_186122


namespace NUMINAMATH_CALUDE_log_identity_l1861_186186

theorem log_identity : Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 + 
  (Real.log 5 / Real.log 2) * (Real.log 8 / Real.log 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1861_186186


namespace NUMINAMATH_CALUDE_fast_food_order_l1861_186129

/-- A problem about friends ordering fast food --/
theorem fast_food_order (num_friends : ℕ) (hamburger_cost : ℚ) 
  (fries_sets : ℕ) (fries_cost : ℚ) (soda_cups : ℕ) (soda_cost : ℚ)
  (spaghetti_platters : ℕ) (spaghetti_cost : ℚ) (individual_payment : ℚ) :
  num_friends = 5 →
  hamburger_cost = 3 →
  fries_sets = 4 →
  fries_cost = 6/5 →
  soda_cups = 5 →
  soda_cost = 1/2 →
  spaghetti_platters = 1 →
  spaghetti_cost = 27/10 →
  individual_payment = 5 →
  ∃ (num_hamburgers : ℕ), 
    num_hamburgers * hamburger_cost + 
    fries_sets * fries_cost + 
    soda_cups * soda_cost + 
    spaghetti_platters * spaghetti_cost = 
    num_friends * individual_payment ∧
    num_hamburgers = 5 := by
  sorry


end NUMINAMATH_CALUDE_fast_food_order_l1861_186129


namespace NUMINAMATH_CALUDE_james_chore_time_l1861_186193

/-- The total time James spends on all chores -/
def total_chore_time (vacuum_time cleaning_time laundry_time organizing_time : ℝ) : ℝ :=
  vacuum_time + cleaning_time + laundry_time + organizing_time

/-- Theorem stating the total time James spends on chores -/
theorem james_chore_time :
  ∃ (vacuum_time cleaning_time laundry_time organizing_time : ℝ),
    vacuum_time = 3 ∧
    cleaning_time = 3 * vacuum_time ∧
    laundry_time = (1/2) * cleaning_time ∧
    organizing_time = 2 * (vacuum_time + cleaning_time + laundry_time) ∧
    total_chore_time vacuum_time cleaning_time laundry_time organizing_time = 49.5 :=
by
  sorry

end NUMINAMATH_CALUDE_james_chore_time_l1861_186193


namespace NUMINAMATH_CALUDE_triangle_problem_l1861_186170

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.cos B = 3 →
  b * Real.cos A = 1 →
  A - B = π / 6 →
  c = 4 ∧ B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1861_186170


namespace NUMINAMATH_CALUDE_ellipse_tangent_line_l1861_186131

/-- Given an ellipse x^2/a^2 + y^2/b^2 = 1, the tangent line at point P(x₀, y₀) 
    has the equation x₀x/a^2 + y₀y/b^2 = 1 -/
theorem ellipse_tangent_line (a b x₀ y₀ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y, (x₀ * x) / a^2 + (y₀ * y) / b^2 = 1 ↔ 
    (∃ t : ℝ, x = x₀ + t * (-2 * x₀ / a^2) ∧ y = y₀ + t * (-2 * y₀ / b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_line_l1861_186131


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1861_186138

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-6, 2)
  let b : ℝ × ℝ := (m, -3)
  parallel a b → m = 9 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1861_186138


namespace NUMINAMATH_CALUDE_rhombus_properties_l1861_186152

structure Rhombus where
  points : Fin 4 → ℝ × ℝ
  is_rhombus : ∀ i j : Fin 4, i ≠ j → dist (points i) (points j) = dist (points ((i+1) % 4)) (points ((j+1) % 4))

def diagonal1 (r : Rhombus) : ℝ × ℝ := r.points 0 - r.points 2
def diagonal2 (r : Rhombus) : ℝ × ℝ := r.points 1 - r.points 3

theorem rhombus_properties (r : Rhombus) :
  (∃ m : ℝ, diagonal1 r = m • (diagonal2 r)) ∧ 
  (diagonal1 r • diagonal2 r = 0) ∧
  (∀ i : Fin 4, dist (r.points i) (r.points ((i+1) % 4)) = dist (r.points ((i+1) % 4)) (r.points ((i+2) % 4))) ∧
  (¬ ∀ r : Rhombus, ‖diagonal1 r‖ = ‖diagonal2 r‖) := by
  sorry

#check rhombus_properties

end NUMINAMATH_CALUDE_rhombus_properties_l1861_186152


namespace NUMINAMATH_CALUDE_function_point_relation_l1861_186160

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is indeed the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Given condition: f^(-1)(-2) = 0
axiom condition : f_inv (-2) = 0

-- Theorem to prove
theorem function_point_relation :
  f (-5 + 5) = -2 :=
sorry

end NUMINAMATH_CALUDE_function_point_relation_l1861_186160


namespace NUMINAMATH_CALUDE_mary_score_l1861_186120

def AHSME_score (c w : ℕ) : ℕ := 30 + 4 * c - w

def unique_solution (s : ℕ) : Prop :=
  ∃! (c w : ℕ), AHSME_score c w = s ∧ c + w ≤ 30

def multiple_solutions (s : ℕ) : Prop :=
  ∃ (c₁ w₁ c₂ w₂ : ℕ), c₁ ≠ c₂ ∧ AHSME_score c₁ w₁ = s ∧ AHSME_score c₂ w₂ = s ∧ c₁ + w₁ ≤ 30 ∧ c₂ + w₂ ≤ 30

theorem mary_score :
  ∃ (s : ℕ),
    s = 119 ∧
    s > 80 ∧
    unique_solution s ∧
    ∀ s', 80 < s' ∧ s' < s → multiple_solutions s' :=
by sorry

end NUMINAMATH_CALUDE_mary_score_l1861_186120


namespace NUMINAMATH_CALUDE_sara_picked_six_pears_l1861_186189

/-- The number of pears picked by Tim -/
def tim_pears : ℕ := 5

/-- The total number of pears picked by Sara and Tim -/
def total_pears : ℕ := 11

/-- The number of pears picked by Sara -/
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_six_pears : sara_pears = 6 := by
  sorry

end NUMINAMATH_CALUDE_sara_picked_six_pears_l1861_186189


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1861_186172

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, where a = (3,1) and b = (x,-1), 
    if a is parallel to b, then x = -3 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -1)
  are_parallel a b → x = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1861_186172


namespace NUMINAMATH_CALUDE_nine_integer_chord_lengths_l1861_186150

/-- Represents a circle with a given radius and a point inside it -/
structure CircleWithPoint where
  radius : ℝ
  pointDistance : ℝ

/-- Counts the number of different integer chord lengths containing the given point -/
def countIntegerChordLengths (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle of radius 25 and a point 13 units from the center,
    there are exactly 9 different integer chord lengths -/
theorem nine_integer_chord_lengths :
  let c := CircleWithPoint.mk 25 13
  countIntegerChordLengths c = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_chord_lengths_l1861_186150


namespace NUMINAMATH_CALUDE_min_value_expression_l1861_186103

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c = 1) :
  (8 * a * c^2 + a) / (b * c) + 32 / (a + 1) ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1861_186103


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l1861_186107

def total_marbles : ℕ := 9
def marbles_per_color : ℕ := 3
def num_colors : ℕ := 3
def marbles_drawn_per_person : ℕ := 3

def total_ways_to_draw : ℕ := (total_marbles.choose marbles_drawn_per_person) * 
                               ((total_marbles - marbles_drawn_per_person).choose marbles_drawn_per_person) * 
                               ((total_marbles - 2 * marbles_drawn_per_person).choose marbles_drawn_per_person)

def favorable_outcomes : ℕ := num_colors * ((total_marbles - marbles_drawn_per_person).choose marbles_drawn_per_person)

theorem cheryl_same_color_probability : 
  (favorable_outcomes : ℚ) / total_ways_to_draw = 1 / 28 := by sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l1861_186107


namespace NUMINAMATH_CALUDE_green_hats_count_l1861_186173

/-- Proves that the number of green hats is 20 given the conditions of the problem -/
theorem green_hats_count (total_hats : ℕ) (blue_price green_price total_price : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 530) :
  ∃ (blue_hats green_hats : ℕ), 
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 20 := by
  sorry

#check green_hats_count

end NUMINAMATH_CALUDE_green_hats_count_l1861_186173


namespace NUMINAMATH_CALUDE_valid_numbers_l1861_186113

def is_valid_number (n : ℕ) : Prop :=
  523000 ≤ n ∧ n ≤ 523999 ∧ n % 7 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 523152 ∨ n = 523656 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1861_186113


namespace NUMINAMATH_CALUDE_modified_grid_perimeter_l1861_186111

/-- Represents a square grid with a hole and an additional row on top. -/
structure ModifiedGrid :=
  (side : ℕ)
  (hole_size : ℕ)
  (top_row : ℕ)

/-- Calculates the perimeter of the modified grid. -/
def perimeter (grid : ModifiedGrid) : ℕ :=
  2 * (grid.side + grid.top_row) + 2 * grid.side - 2 * grid.hole_size

/-- Theorem stating that the perimeter of the specific modified 3x3 grid is 9. -/
theorem modified_grid_perimeter :
  ∃ (grid : ModifiedGrid), grid.side = 3 ∧ grid.hole_size = 1 ∧ grid.top_row = 3 ∧ perimeter grid = 9 :=
sorry

end NUMINAMATH_CALUDE_modified_grid_perimeter_l1861_186111


namespace NUMINAMATH_CALUDE_total_students_correct_l1861_186108

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 40 / 100

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 528

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct :
  (1 - biology_enrollment_percentage) * total_students = non_biology_students :=
sorry

end NUMINAMATH_CALUDE_total_students_correct_l1861_186108


namespace NUMINAMATH_CALUDE_completing_square_l1861_186168

theorem completing_square (x : ℝ) : x^2 - 4*x + 2 = 0 ↔ (x - 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l1861_186168


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l1861_186195

theorem arctan_sum_special_case (a b : ℝ) : 
  a = 1/3 → (a + 1) * (b + 1) = 3 → Real.arctan a + Real.arctan b = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l1861_186195


namespace NUMINAMATH_CALUDE_smaller_number_value_l1861_186112

theorem smaller_number_value (s l : ℤ) : 
  (l - s = 28) → 
  (l + 13 = 2 * (s + 13)) → 
  s = 15 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_value_l1861_186112


namespace NUMINAMATH_CALUDE_evaluate_expression_l1861_186121

theorem evaluate_expression (x : ℝ) (h : x = -3) : 
  (5 + x*(5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1861_186121


namespace NUMINAMATH_CALUDE_cookie_difference_theorem_l1861_186154

def combined_difference (a b c : ℕ) : ℕ :=
  (a.max b - a.min b) + (a.max c - a.min c) + (b.max c - b.min c)

theorem cookie_difference_theorem :
  combined_difference 129 140 167 = 76 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_theorem_l1861_186154


namespace NUMINAMATH_CALUDE_right_triangle_area_l1861_186141

/-- A right triangle with one leg of length 15 and an inscribed circle of radius 3 has an area of 60. -/
theorem right_triangle_area (a b c r : ℝ) : 
  a = 15 → -- One leg is 15
  r = 3 → -- Radius of inscribed circle is 3
  a^2 + b^2 = c^2 → -- Right triangle (Pythagorean theorem)
  r * (a + b + c) / 2 = r * b → -- Area formula using semiperimeter and inradius
  a * b / 2 = 60 := by -- Area of the triangle is 60
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1861_186141


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l1861_186135

/-- A quadratic function f(x) = x^2 + 2mx + 10 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 10

/-- The function is increasing on [2, +∞) -/
def increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f m x < f m y

theorem quadratic_increasing_condition (m : ℝ) :
  increasing_on_interval m → m ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l1861_186135


namespace NUMINAMATH_CALUDE_rectangle_diagonal_intersections_l1861_186153

theorem rectangle_diagonal_intersections (ℓ b : ℕ) (hℓ : ℓ > 0) (hb : b > 0) : 
  let V := ℓ + b - Nat.gcd ℓ b
  ℓ = 6 → b = 4 → V = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_intersections_l1861_186153


namespace NUMINAMATH_CALUDE_power_of_power_equals_729_l1861_186161

theorem power_of_power_equals_729 : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_729_l1861_186161


namespace NUMINAMATH_CALUDE_three_digit_not_multiple_of_6_or_8_count_l1861_186183

/-- The count of three-digit numbers -/
def three_digit_count : ℕ := 900

/-- The count of three-digit numbers that are multiples of 6 -/
def multiples_of_6_count : ℕ := 150

/-- The count of three-digit numbers that are multiples of 8 -/
def multiples_of_8_count : ℕ := 112

/-- The count of three-digit numbers that are multiples of both 6 and 8 (i.e., multiples of 24) -/
def multiples_of_24_count : ℕ := 37

/-- Theorem: The count of three-digit numbers that are not multiples of 6 or 8 is 675 -/
theorem three_digit_not_multiple_of_6_or_8_count : 
  three_digit_count - (multiples_of_6_count + multiples_of_8_count - multiples_of_24_count) = 675 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_not_multiple_of_6_or_8_count_l1861_186183


namespace NUMINAMATH_CALUDE_min_value_theorem_l1861_186175

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The circle defined by the equation x^2 + y^2 + 4x + 2y + 1 = 0 -/
def given_circle : Circle :=
  { center := (-2, -1), radius := 2 }

/-- Predicate to check if a line bisects a circle -/
def bisects (l : Line) (c : Circle) : Prop :=
  l.a * c.center.1 + l.b * c.center.2 + l.c = 0

/-- The minimum value function we want to minimize -/
def min_value_function (a b : ℝ) : ℝ :=
  (a - 1)^2 + (b - 1)^2

/-- The main theorem -/
theorem min_value_theorem (l : Line) :
  bisects l given_circle →
  ∃ (min : ℝ), min = 4/5 ∧ 
    ∀ (a b : ℝ), l.a = a ∧ l.b = b → min_value_function a b ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1861_186175


namespace NUMINAMATH_CALUDE_canal_digging_time_l1861_186185

/-- Represents the time taken to dig a canal given the number of men, hours per day, and days worked. -/
def diggingTime (men : ℕ) (hoursPerDay : ℕ) (days : ℚ) : ℚ := men * hoursPerDay * days

/-- Theorem stating that 30 men working 8 hours a day will take 1.5 days to dig a canal
    that originally took 20 men working 6 hours a day for 3 days, assuming constant work rate. -/
theorem canal_digging_time :
  diggingTime 20 6 3 = diggingTime 30 8 (3/2 : ℚ) := by
  sorry

#check canal_digging_time

end NUMINAMATH_CALUDE_canal_digging_time_l1861_186185


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l1861_186109

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for a valid triangle here
  True

-- Define the angle C to be 60°
def angle_C_60 (A B C : ℝ × ℝ) : Prop :=
  -- Add condition for ∠C = 60° here
  True

-- Define D as the point where altitude from C meets AB
def altitude_C_D (A B C D : ℝ × ℝ) : Prop :=
  -- Add condition for D being on altitude from C here
  True

-- Define that the sides of triangle ABC are integers
def integer_sides (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for integer sides here
  True

-- Define BD = 17³
def BD_17_cubed (B D : ℝ × ℝ) : Prop :=
  -- Add condition for BD = 17³ here
  True

-- Define cos B = m/n where m and n are relatively prime positive integers
def cos_B_frac (B : ℝ × ℝ) (m n : ℕ) : Prop :=
  -- Add conditions for cos B = m/n and m, n coprime here
  True

theorem triangle_cosine_sum (A B C D : ℝ × ℝ) (m n : ℕ) :
  triangle_ABC A B C →
  angle_C_60 A B C →
  altitude_C_D A B C D →
  integer_sides A B C →
  BD_17_cubed B D →
  cos_B_frac B m n →
  m + n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l1861_186109


namespace NUMINAMATH_CALUDE_expression_simplification_l1861_186187

theorem expression_simplification :
  (4^2 * 7) / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11) = 44 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1861_186187


namespace NUMINAMATH_CALUDE_polynomial_properties_l1861_186182

def P (x : ℤ) : ℤ := x * (x + 1) * (x + 2)

theorem polynomial_properties :
  (∀ x : ℤ, ∃ k : ℤ, P x = 3 * k) ∧
  (∃ a b c d : ℤ, ∀ x : ℤ, P x = x^3 + a*x^2 + b*x + c) ∧
  (∃ a b c : ℤ, ∀ x : ℤ, P x = x^3 + a*x^2 + b*x + c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1861_186182


namespace NUMINAMATH_CALUDE_candy_distribution_l1861_186166

theorem candy_distribution (n : Nat) (f : Nat) (h1 : n = 30) (h2 : f = 4) :
  (∃ x : Nat, (n - x) % f = 0 ∧ ∀ y : Nat, y < x → (n - y) % f ≠ 0) →
  (∃ x : Nat, (n - x) % f = 0 ∧ ∀ y : Nat, y < x → (n - y) % f ≠ 0 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1861_186166


namespace NUMINAMATH_CALUDE_cylinder_volume_l1861_186114

/-- The volume of a cylinder given water displacement measurements -/
theorem cylinder_volume
  (initial_water_level : ℝ)
  (final_water_level : ℝ)
  (cylinder_min_marking : ℝ)
  (cylinder_max_marking : ℝ)
  (h1 : initial_water_level = 30)
  (h2 : final_water_level = 35)
  (h3 : cylinder_min_marking = 15)
  (h4 : cylinder_max_marking = 45) :
  let water_displaced := final_water_level - initial_water_level
  let cylinder_marking_range := cylinder_max_marking - cylinder_min_marking
  let submerged_proportion := (final_water_level - cylinder_min_marking) / cylinder_marking_range
  cylinder_marking_range / (final_water_level - cylinder_min_marking) * water_displaced = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1861_186114


namespace NUMINAMATH_CALUDE_chicken_price_per_pound_l1861_186159

/-- Given John's food order for a restaurant, prove the price per pound of chicken --/
theorem chicken_price_per_pound (beef_quantity : ℕ) (beef_price : ℚ) 
  (total_cost : ℚ) (chicken_quantity : ℕ) (chicken_price : ℚ) : chicken_price = 3 :=
by
  have h1 : beef_quantity = 1000 := by sorry
  have h2 : beef_price = 8 := by sorry
  have h3 : chicken_quantity = 2 * beef_quantity := by sorry
  have h4 : total_cost = 14000 := by sorry
  have h5 : total_cost = beef_quantity * beef_price + chicken_quantity * chicken_price := by sorry
  sorry

end NUMINAMATH_CALUDE_chicken_price_per_pound_l1861_186159


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l1861_186169

theorem smallest_distance_between_circles (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 7*I) = 4) :
  ∃ (z' w' : ℂ), 
    Complex.abs (z' + 2 + 4*I) = 2 ∧ 
    Complex.abs (w' - 6 - 7*I) = 4 ∧
    Complex.abs (z' - w') = Real.sqrt 185 - 6 ∧
    ∀ (z'' w'' : ℂ), 
      Complex.abs (z'' + 2 + 4*I) = 2 → 
      Complex.abs (w'' - 6 - 7*I) = 4 → 
      Complex.abs (z'' - w'') ≥ Real.sqrt 185 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l1861_186169


namespace NUMINAMATH_CALUDE_perfect_squares_and_multiple_of_40_l1861_186110

theorem perfect_squares_and_multiple_of_40 :
  ∃ n : ℤ, ∃ a b : ℤ,
    (2 * n + 1 = a^2) ∧
    (3 * n + 1 = b^2) ∧
    (∃ k : ℤ, n = 40 * k) :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_and_multiple_of_40_l1861_186110


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l1861_186136

/-- Two circles are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1_center c2_center : ℝ × ℝ) (r : ℝ) : Prop :=
  Real.sqrt ((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = 2 * r

theorem tangent_circles_radius (r : ℝ) (h : r > 0) :
  are_tangent (0, 0) (3, -1) r → r = Real.sqrt 10 / 2 := by
  sorry

#check tangent_circles_radius

end NUMINAMATH_CALUDE_tangent_circles_radius_l1861_186136


namespace NUMINAMATH_CALUDE_bcd_value_l1861_186178

/-- Represents the digits in the encoding system -/
inductive Digit
| A | B | C | D | E | F

/-- Maps the digits to their corresponding base-6 values -/
def digit_to_base6 : Digit → Nat
| Digit.A => 0
| Digit.B => 5
| Digit.C => 5
| Digit.D => 0
| Digit.E => 0
| Digit.F => 1

/-- Converts a three-digit code in the given encoding to its base-10 value -/
def code_to_base10 (d1 d2 d3 : Digit) : Nat :=
  (digit_to_base6 d1) * 36 + (digit_to_base6 d2) * 6 + (digit_to_base6 d3)

/-- The main theorem to prove -/
theorem bcd_value : 
  ∃ (n : Nat), 
    code_to_base10 Digit.A Digit.B Digit.C = n ∧
    code_to_base10 Digit.A Digit.B Digit.D = n + 1 ∧
    code_to_base10 Digit.A Digit.E Digit.F = n + 2 ∧
    code_to_base10 Digit.B Digit.C Digit.D = 181 :=
by sorry


end NUMINAMATH_CALUDE_bcd_value_l1861_186178


namespace NUMINAMATH_CALUDE_cans_in_sixth_bin_l1861_186171

theorem cans_in_sixth_bin (n : ℕ) (cans : ℕ → ℕ) : 
  (∀ k, cans k = k * (k + 1) / 2) → cans 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_cans_in_sixth_bin_l1861_186171


namespace NUMINAMATH_CALUDE_first_expedition_duration_l1861_186117

theorem first_expedition_duration (total_days : ℕ) 
  (h1 : total_days = 126) : ∃ (x : ℕ), 
  x * 7 + (x + 2) * 7 + 2 * (x + 2) * 7 = total_days ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_expedition_duration_l1861_186117


namespace NUMINAMATH_CALUDE_lcm_six_fifteen_l1861_186124

theorem lcm_six_fifteen : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_six_fifteen_l1861_186124


namespace NUMINAMATH_CALUDE_positive_intervals_l1861_186197

theorem positive_intervals (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_intervals_l1861_186197


namespace NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l1861_186176

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_x_axis 
  (center_C : ℝ × ℝ) 
  (center_D : ℝ × ℝ) 
  (radius : ℝ) : 
  center_C = (3, 5) →
  center_D = (13, 5) →
  radius = 5 →
  ∃ (area : ℝ), area = 50 - 25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l1861_186176


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1861_186156

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1861_186156


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1861_186140

def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence b →
  a 1 = 15 →
  b 1 = 35 →
  a 2 + b 2 = 60 →
  a 36 + b 36 = 400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1861_186140


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1861_186115

theorem perfect_square_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + k*x + 25 = y^2) → (k = 10 ∨ k = -10) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1861_186115


namespace NUMINAMATH_CALUDE_power_2014_of_abs_one_l1861_186134

theorem power_2014_of_abs_one (a : ℝ) : |a| = 1 → a^2014 = 1 := by sorry

end NUMINAMATH_CALUDE_power_2014_of_abs_one_l1861_186134
