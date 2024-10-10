import Mathlib

namespace ones_digit_of_large_power_l1772_177220

theorem ones_digit_of_large_power (n : ℕ) : 
  (35^(35*(17^17)) : ℕ) % 10 = 5 := by
  sorry

end ones_digit_of_large_power_l1772_177220


namespace sum_xyz_is_zero_l1772_177231

theorem sum_xyz_is_zero (x y z : ℝ) 
  (eq1 : x + y = 2*x + z)
  (eq2 : x - 2*y = 4*z)
  (eq3 : y = 6*z) : 
  x + y + z = 0 := by
sorry

end sum_xyz_is_zero_l1772_177231


namespace unique_point_on_curve_l1772_177227

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is on the curve if x^2 + 5x + 1 = 3y -/
def on_curve (x y : ℤ) : Prop := x^2 + 5*x + 1 = 3*y

theorem unique_point_on_curve : 
  ∀ x y : ℤ, second_quadrant x y → on_curve x y → (x = -7 ∧ y = 5) :=
by sorry

end unique_point_on_curve_l1772_177227


namespace equation_proof_l1772_177294

theorem equation_proof : ∃ (op1 op2 op3 op4 : ℕ → ℕ → ℕ), 
  (op1 = (·-·) ∧ op2 = (·*·) ∧ op3 = (·/·) ∧ op4 = (·+·)) ∨
  (op1 = (·-·) ∧ op2 = (·*·) ∧ op3 = (·+·) ∧ op4 = (·/·)) ∨
  (op1 = (·-·) ∧ op2 = (·+·) ∧ op3 = (·*·) ∧ op4 = (·/·)) ∨
  (op1 = (·-·) ∧ op2 = (·+·) ∧ op3 = (·/·) ∧ op4 = (·*·)) ∨
  (op1 = (·-·) ∧ op2 = (·/·) ∧ op3 = (·*·) ∧ op4 = (·+·)) ∨
  (op1 = (·-·) ∧ op2 = (·/·) ∧ op3 = (·+·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·-·) ∧ op3 = (·*·) ∧ op4 = (·/·)) ∨
  (op1 = (·+·) ∧ op2 = (·-·) ∧ op3 = (·/·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·*·) ∧ op3 = (·-·) ∧ op4 = (·/·)) ∨
  (op1 = (·+·) ∧ op2 = (·*·) ∧ op3 = (·/·) ∧ op4 = (·-·)) ∨
  (op1 = (·+·) ∧ op2 = (·/·) ∧ op3 = (·-·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·/·) ∧ op3 = (·*·) ∧ op4 = (·-·)) ∨
  (op1 = (·*·) ∧ op2 = (·-·) ∧ op3 = (·+·) ∧ op4 = (·/·)) ∨
  (op1 = (·*·) ∧ op2 = (·-·) ∧ op3 = (·/·) ∧ op4 = (·+·)) ∨
  (op1 = (·*·) ∧ op2 = (·+·) ∧ op3 = (·-·) ∧ op4 = (·/·)) ∨
  (op1 = (·*·) ∧ op2 = (·+·) ∧ op3 = (·/·) ∧ op4 = (·-·)) ∨
  (op1 = (·*·) ∧ op2 = (·/·) ∧ op3 = (·-·) ∧ op4 = (·+·)) ∨
  (op1 = (·*·) ∧ op2 = (·/·) ∧ op3 = (·+·) ∧ op4 = (·-·)) ∨
  (op1 = (·/·) ∧ op2 = (·-·) ∧ op3 = (·*·) ∧ op4 = (·+·)) ∨
  (op1 = (·/·) ∧ op2 = (·-·) ∧ op3 = (·+·) ∧ op4 = (·*·)) ∨
  (op1 = (·/·) ∧ op2 = (·+·) ∧ op3 = (·-·) ∧ op4 = (·*·)) ∨
  (op1 = (·/·) ∧ op2 = (·+·) ∧ op3 = (·*·) ∧ op4 = (·-·)) ∨
  (op1 = (·/·) ∧ op2 = (·*·) ∧ op3 = (·-·) ∧ op4 = (·+·)) ∨
  (op1 = (·/·) ∧ op2 = (·*·) ∧ op3 = (·+·) ∧ op4 = (·-·)) →
  (op3 (op1 132 (op2 7 6)) (op4 12 3)) = 6 := by
sorry

end equation_proof_l1772_177294


namespace coefficient_a3b2_in_expansion_l1772_177200

theorem coefficient_a3b2_in_expansion : ∃ (coeff : ℕ),
  coeff = (Nat.choose 5 3) * (Nat.choose 8 4) ∧
  coeff = 700 := by sorry

end coefficient_a3b2_in_expansion_l1772_177200


namespace base7_divisible_by_13_l1772_177207

/-- Converts a base-7 number of the form 3dd6₇ to base 10 --/
def base7ToBase10 (d : Nat) : Nat :=
  3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : Nat) : Prop :=
  n % 13 = 0

/-- A base-7 digit is between 0 and 6 inclusive --/
def isBase7Digit (d : Nat) : Prop :=
  d ≤ 6

theorem base7_divisible_by_13 :
  ∃ (d : Nat), isBase7Digit d ∧ isDivisibleBy13 (base7ToBase10 d) ∧ d = 5 := by
  sorry

end base7_divisible_by_13_l1772_177207


namespace smallest_positive_solution_l1772_177268

theorem smallest_positive_solution (x : ℕ) : x = 30 ↔ 
  (x > 0 ∧ 
   (51 * x + 15) % 35 = 5 ∧ 
   ∀ y : ℕ, y > 0 → (51 * y + 15) % 35 = 5 → x ≤ y) := by
  sorry

end smallest_positive_solution_l1772_177268


namespace income_calculation_l1772_177201

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 10 →  -- ratio of income to expenditure is 10:4
  savings = income - expenditure →  -- savings definition
  savings = 11400 →  -- given savings amount
  income = 19000 := by  -- prove that income is 19000
sorry

end income_calculation_l1772_177201


namespace jordan_rectangle_width_l1772_177211

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_width : 
  ∀ (carol_rect jordan_rect : Rectangle),
    carol_rect.length = 5 →
    carol_rect.width = 24 →
    jordan_rect.length = 12 →
    area carol_rect = area jordan_rect →
    jordan_rect.width = 10 := by
  sorry

end jordan_rectangle_width_l1772_177211


namespace herd_division_l1772_177242

theorem herd_division (total : ℕ) (fourth_son : ℕ) : 
  (1 : ℚ) / 3 + (1 : ℚ) / 6 + (1 : ℚ) / 9 + (fourth_son : ℚ) / total = 1 →
  fourth_son = 11 →
  total = 54 := by
sorry

end herd_division_l1772_177242


namespace opposite_of_negative_half_l1772_177205

theorem opposite_of_negative_half : -(-(1/2 : ℚ)) = 1/2 := by
  sorry

end opposite_of_negative_half_l1772_177205


namespace distribute_plumbers_count_l1772_177264

/-- The number of ways to distribute 4 plumbers to 3 residences -/
def distribute_plumbers : ℕ :=
  Nat.choose 4 2 * (3 * 2 * 1)

/-- The conditions of the problem -/
axiom plumbers : ℕ
axiom residences : ℕ
axiom plumbers_eq_four : plumbers = 4
axiom residences_eq_three : residences = 3
axiom all_plumbers_assigned : True
axiom one_residence_per_plumber : True
axiom all_residences_checked : True

/-- The theorem to be proved -/
theorem distribute_plumbers_count :
  distribute_plumbers = Nat.choose plumbers 2 * (residences * (residences - 1) * (residences - 2)) :=
sorry

end distribute_plumbers_count_l1772_177264


namespace soda_cans_purchased_l1772_177226

/-- Given that S cans of soda can be purchased for Q quarters, and 1 dollar is worth 5 quarters due to a fee,
    the number of cans of soda that can be purchased for D dollars is (5 * D * S) / Q. -/
theorem soda_cans_purchased (S Q D : ℚ) (hS : S > 0) (hQ : Q > 0) (hD : D ≥ 0) :
  (S / Q) * (5 * D) = (5 * D * S) / Q :=
by sorry

end soda_cans_purchased_l1772_177226


namespace distance_calculation_l1772_177249

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (5, 8, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_calculation :
  distance_to_line point line_point line_direction = Real.sqrt 10458 / 34 := by
  sorry

end distance_calculation_l1772_177249


namespace parabola_vertex_l1772_177239

/-- The parabola defined by y = -x^2 + 2x + 3 has its vertex at the point (1, 4). -/
theorem parabola_vertex (x y : ℝ) : 
  y = -x^2 + 2*x + 3 → (1, 4) = (x, y) ∨ ∃ t : ℝ, y < -t^2 + 2*t + 3 := by
  sorry

end parabola_vertex_l1772_177239


namespace inscribed_semicircle_radius_l1772_177255

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base length of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The area of the triangle equals the area of the semicircle -/
  area_equality : (1/2) * base * height = π * radius^2

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangleWithSemicircle)
    (h1 : t.base = 24) (h2 : t.height = 18) : t.radius = 18 / π := by
  sorry

end inscribed_semicircle_radius_l1772_177255


namespace trees_after_typhoon_l1772_177269

theorem trees_after_typhoon (initial_trees : ℕ) (dead_trees : ℕ) : 
  initial_trees = 20 → dead_trees = 16 → initial_trees - dead_trees = 4 := by
  sorry

end trees_after_typhoon_l1772_177269


namespace prob_even_sum_two_dice_l1772_177212

/-- Die with faces numbered 1 through 4 -/
def Die1 : Finset Nat := {1, 2, 3, 4}

/-- Die with faces numbered 1 through 8 -/
def Die2 : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8}

/-- The probability of getting an even sum when rolling two dice -/
def ProbEvenSum : ℚ := 1 / 2

/-- Theorem stating that the probability of getting an even sum when rolling
    two dice, one with faces 1-4 and another with faces 1-8, is equal to 1/2 -/
theorem prob_even_sum_two_dice :
  let outcomes := Die1.product Die2
  let even_sum := {p : Nat × Nat | (p.1 + p.2) % 2 = 0}
  (outcomes.filter (λ p => p ∈ even_sum)).card / outcomes.card = ProbEvenSum :=
sorry


end prob_even_sum_two_dice_l1772_177212


namespace midpoint_theorem_l1772_177228

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the potential midpoints
def midpoint1 : ℝ × ℝ := (1, 1)
def midpoint2 : ℝ × ℝ := (-1, 2)
def midpoint3 : ℝ × ℝ := (1, 3)
def midpoint4 : ℝ × ℝ := (-1, -4)

-- Define a function to check if a point is a valid midpoint
def is_valid_midpoint (m : ℝ × ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    hyperbola x1 y1 ∧ hyperbola x2 y2 ∧
    m.1 = (x1 + x2) / 2 ∧ m.2 = (y1 + y2) / 2

-- Theorem statement
theorem midpoint_theorem :
  ¬(is_valid_midpoint midpoint1) ∧
  ¬(is_valid_midpoint midpoint2) ∧
  ¬(is_valid_midpoint midpoint3) ∧
  is_valid_midpoint midpoint4 := by sorry

end midpoint_theorem_l1772_177228


namespace problem_solution_l1772_177262

theorem problem_solution (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬p) : 
  ¬p ∧ q := by
  sorry

end problem_solution_l1772_177262


namespace decimal_calculation_l1772_177236

theorem decimal_calculation : (0.25 * 0.8) - 0.12 = 0.08 := by
  sorry

end decimal_calculation_l1772_177236


namespace sum_due_from_discounts_l1772_177285

/-- The sum due (present value) given banker's discount and true discount -/
theorem sum_due_from_discounts (BD TD : ℝ) (h1 : BD = 42) (h2 : TD = 36) :
  ∃ PV : ℝ, PV = 216 ∧ BD = TD + TD^2 / PV :=
by sorry

end sum_due_from_discounts_l1772_177285


namespace batsman_average_increase_l1772_177278

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (prevInnings : ℕ) (prevTotalScore : ℕ) (newScore : ℕ) : ℚ :=
  let newAverage := (prevTotalScore + newScore) / (prevInnings + 1)
  let oldAverage := prevTotalScore / prevInnings
  newAverage - oldAverage

/-- Theorem stating the increase in average for the given batsman -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 19 →
    b.average = 64 →
    averageIncrease 18 (18 * (b.totalScore / 19)) 100 = 2 := by
  sorry

end batsman_average_increase_l1772_177278


namespace last_three_digits_of_9_pow_107_l1772_177281

theorem last_three_digits_of_9_pow_107 : 9^107 % 1000 = 969 := by
  sorry

end last_three_digits_of_9_pow_107_l1772_177281


namespace arithmetic_sequence_sum_l1772_177230

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Theorem: In an arithmetic sequence where a_3 + a_7 = 38, the sum a_2 + a_4 + a_6 + a_8 = 76 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 := by
  sorry


end arithmetic_sequence_sum_l1772_177230


namespace min_value_condition_l1772_177261

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |a * x + 1|

theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 3/2) ∧ (∃ x : ℝ, f a x = 3/2) ↔ a = -1/2 ∨ a = -2 := by
  sorry

end min_value_condition_l1772_177261


namespace table_tennis_play_time_l1772_177290

/-- Represents the table tennis playing scenario -/
structure TableTennis where
  total_students : ℕ
  playing_students : ℕ
  total_time : ℕ
  num_tables : ℕ
  play_time_per_student : ℕ

/-- The theorem statement -/
theorem table_tennis_play_time 
  (tt : TableTennis) 
  (h1 : tt.total_students = 6)
  (h2 : tt.playing_students = 4)
  (h3 : tt.total_time = 210)
  (h4 : tt.num_tables = 2)
  (h5 : tt.total_students % tt.playing_students = 0)
  (h6 : tt.play_time_per_student * tt.total_students = tt.total_time * tt.num_tables) :
  tt.play_time_per_student = 140 := by
  sorry


end table_tennis_play_time_l1772_177290


namespace crayons_left_l1772_177222

-- Define the initial number of crayons
def initial_crayons : ℕ := 62

-- Define the number of crayons eaten
def eaten_crayons : ℕ := 52

-- Theorem to prove
theorem crayons_left : initial_crayons - eaten_crayons = 10 := by
  sorry

end crayons_left_l1772_177222


namespace parabola_equation_l1772_177223

/-- Given a parabola C: y^2 = 2px and a circle x^2 + y^2 - 2x - 15 = 0,
    if the focus of the parabola coincides with the center of the circle,
    then the equation of the parabola C is y^2 = 4x. -/
theorem parabola_equation (p : ℝ) :
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 + y^2 - 2*x - 15 = 0 ∧
   (1, 0) = (x + p/2, 0)) →
  (∀ (x y : ℝ), y^2 = 2*p*x ↔ y^2 = 4*x) :=
by sorry

end parabola_equation_l1772_177223


namespace minimum_value_implies_a_l1772_177298

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / Real.log (1/2)
def g (x : ℝ) : ℝ := x^2 + 4*x - 2
def h (a : ℝ) (x : ℝ) : ℝ := if f a x ≥ g x then f a x else g x

-- State the theorem
theorem minimum_value_implies_a (a : ℝ) : 
  (∀ x, h a x ≥ -2) ∧ (∃ x, h a x = -2) → a = 4 := by
  sorry

end

end minimum_value_implies_a_l1772_177298


namespace translated_line_equation_translation_result_l1772_177252

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translate_line_vertical (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

theorem translated_line_equation (original : Line) (translation : ℝ) :
  let translated := translate_line_vertical original (-translation)
  translated.slope = original.slope ∧
  translated.intercept = original.intercept - translation :=
by sorry

/-- The original line y = -2x + 1 -/
def original_line : Line :=
  { slope := -2, intercept := 1 }

/-- The amount of downward translation -/
def translation_amount : ℝ := 4

theorem translation_result :
  let translated := translate_line_vertical original_line (-translation_amount)
  translated.slope = -2 ∧ translated.intercept = -3 :=
by sorry

end translated_line_equation_translation_result_l1772_177252


namespace problem_statement_l1772_177241

theorem problem_statement (x y : ℝ) 
  (hx : x * (Real.exp x + Real.log x + x) = 1)
  (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  0 < x ∧ x < 1 ∧ y - x > 1 ∧ y - x < 3/2 := by
  sorry

end problem_statement_l1772_177241


namespace prime_between_squares_l1772_177275

theorem prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p - 5 = n^2 ∧ p + 8 = (n + 1)^2 :=
by sorry

end prime_between_squares_l1772_177275


namespace complex_fraction_simplification_l1772_177224

theorem complex_fraction_simplification :
  let numerator := (10^4 + 500) * (25^4 + 500) * (40^4 + 500) * (55^4 + 500) * (70^4 + 500)
  let denominator := (5^4 + 500) * (20^4 + 500) * (35^4 + 500) * (50^4 + 500) * (65^4 + 500)
  ∀ x : ℕ, x^4 + 500 = (x^2 - 10*x + 50) * (x^2 + 10*x + 50) →
  (numerator / denominator : ℚ) = 240 := by
  sorry

end complex_fraction_simplification_l1772_177224


namespace sum_reciprocal_product_l1772_177202

open BigOperators

theorem sum_reciprocal_product : ∑ n in Finset.range 6, 1 / ((n + 3) * (n + 4)) = 2 / 9 := by
  sorry

end sum_reciprocal_product_l1772_177202


namespace problem_one_problem_two_l1772_177246

-- Problem 1
theorem problem_one : 
  (2 / 3 : ℝ) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3 : ℝ) * Real.sqrt 27 = -(4 / 3 : ℝ) * Real.sqrt 6 := by
  sorry

-- Problem 2
theorem problem_two : 
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 := by
  sorry

end problem_one_problem_two_l1772_177246


namespace exist_pouring_sequence_l1772_177289

/-- Represents the state of the three containers -/
structure ContainerState :=
  (a : ℕ) -- Volume in 10-liter container
  (b : ℕ) -- Volume in 7-liter container
  (c : ℕ) -- Volume in 4-liter container

/-- Represents a pouring action between containers -/
inductive PourAction
  | Pour10to7
  | Pour10to4
  | Pour7to10
  | Pour7to4
  | Pour4to10
  | Pour4to7

/-- Applies a pouring action to a container state -/
def applyAction (state : ContainerState) (action : PourAction) : ContainerState :=
  match action with
  | PourAction.Pour10to7 => sorry
  | PourAction.Pour10to4 => sorry
  | PourAction.Pour7to10 => sorry
  | PourAction.Pour7to4 => sorry
  | PourAction.Pour4to10 => sorry
  | PourAction.Pour4to7 => sorry

/-- Checks if a container state is valid -/
def isValidState (state : ContainerState) : Prop :=
  state.a ≤ 10 ∧ state.b ≤ 7 ∧ state.c ≤ 4 ∧ state.a + state.b + state.c = 10

/-- Theorem: There exists a sequence of pouring actions to reach the desired state -/
theorem exist_pouring_sequence :
  ∃ (actions : List PourAction),
    let finalState := actions.foldl applyAction ⟨10, 0, 0⟩
    isValidState finalState ∧ finalState = ⟨4, 2, 4⟩ :=
  sorry

end exist_pouring_sequence_l1772_177289


namespace sets_properties_l1772_177240

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_properties :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 9}) ∧
  (∀ a : ℝ, C a ⊆ B → (2 < a ∧ a < 8)) :=
by sorry

end sets_properties_l1772_177240


namespace japanese_study_fraction_l1772_177248

theorem japanese_study_fraction (j s : ℝ) (x : ℝ) : 
  s = 3 * j →                           -- Senior class is 3 times the junior class
  ((1/3) * s + x * j) / (s + j) = 0.4375 →  -- 0.4375 fraction of all students study Japanese
  x = 3/4 :=                             -- Fraction of juniors studying Japanese
by
  sorry

end japanese_study_fraction_l1772_177248


namespace range_of_a_l1772_177293

theorem range_of_a (a : ℝ) : 
  (|a - 1| + |a - 4| = 3) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end range_of_a_l1772_177293


namespace geometric_progression_floor_sum_l1772_177209

theorem geometric_progression_floor_sum (a b c k r : ℝ) : 
  a > 0 → b > 0 → c > 0 → k > 0 → r > 1 → 
  b = k * r → c = k * r^2 →
  ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 5 :=
by sorry

end geometric_progression_floor_sum_l1772_177209


namespace number_problem_l1772_177217

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 14) : 
  (40/100) * N = 168 := by
  sorry

end number_problem_l1772_177217


namespace indefinite_stick_shortening_l1772_177216

theorem indefinite_stick_shortening :
  ∃ t : ℝ, t > 1 ∧ ∀ n : ℕ, t^(3-n) > t^(2-n) + t^(1-n) := by
  sorry

end indefinite_stick_shortening_l1772_177216


namespace flower_percentage_l1772_177208

theorem flower_percentage (total_flowers : ℕ) (yellow_flowers : ℕ) (purple_increase : ℚ) :
  total_flowers = 35 →
  yellow_flowers = 10 →
  purple_increase = 80 / 100 →
  let purple_flowers := yellow_flowers + (purple_increase * yellow_flowers).floor
  let green_flowers := total_flowers - yellow_flowers - purple_flowers
  let yellow_and_purple := yellow_flowers + purple_flowers
  (green_flowers : ℚ) / yellow_and_purple * 100 = 25 := by
  sorry

end flower_percentage_l1772_177208


namespace total_fish_equation_l1772_177250

/-- The number of fish owned by four friends, given their relative quantities -/
def total_fish (x : ℝ) : ℝ :=
  let max_fish := x
  let sam_fish := 3.25 * max_fish
  let joe_fish := 9.5 * sam_fish
  let harry_fish := 5.5 * joe_fish
  max_fish + sam_fish + joe_fish + harry_fish

/-- Theorem stating that the total number of fish is 204.9375 times the number of fish Max has -/
theorem total_fish_equation (x : ℝ) : total_fish x = 204.9375 * x := by
  sorry

end total_fish_equation_l1772_177250


namespace units_digit_of_3542_to_876_l1772_177210

theorem units_digit_of_3542_to_876 : ∃ n : ℕ, 3542^876 ≡ 6 [ZMOD 10] :=
by sorry

end units_digit_of_3542_to_876_l1772_177210


namespace sqrt_expression_equality_l1772_177271

theorem sqrt_expression_equality : 
  Real.sqrt 8 - 2 * Real.sqrt (1/2) + (Real.sqrt 27 + 2 * Real.sqrt 6) / Real.sqrt 3 = 3 * Real.sqrt 2 + 3 := by
  sorry

end sqrt_expression_equality_l1772_177271


namespace consecutive_negative_integers_sum_l1772_177263

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -105 := by
  sorry

end consecutive_negative_integers_sum_l1772_177263


namespace walking_ring_width_l1772_177265

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * π * r₁ - 2 * π * r₂ = 20 * π) : 
  r₁ - r₂ = 10 := by
sorry

end walking_ring_width_l1772_177265


namespace rectangular_plot_width_l1772_177256

theorem rectangular_plot_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end rectangular_plot_width_l1772_177256


namespace notebooks_bought_is_four_l1772_177297

/-- The cost of one pencil -/
def pencil_cost : ℚ := sorry

/-- The cost of one notebook -/
def notebook_cost : ℚ := sorry

/-- The number of notebooks bought in the second case -/
def notebooks_bought : ℕ := sorry

/-- The cost of 8 dozen pencils and 2 dozen notebooks is 520 rupees -/
axiom eq1 : 96 * pencil_cost + 24 * notebook_cost = 520

/-- The cost of 3 pencils and some number of notebooks is 60 rupees -/
axiom eq2 : 3 * pencil_cost + notebooks_bought * notebook_cost = 60

/-- The sum of the cost of 1 pencil and 1 notebook is 15.512820512820513 rupees -/
axiom eq3 : pencil_cost + notebook_cost = 15.512820512820513

theorem notebooks_bought_is_four : notebooks_bought = 4 := by sorry

end notebooks_bought_is_four_l1772_177297


namespace impossible_all_multiples_of_10_l1772_177274

/-- Represents a grid operation (adding 1 to each cell in a subgrid) -/
structure GridOperation where
  startRow : Fin 8
  startCol : Fin 8
  size : Fin 2  -- 0 for 3x3, 1 for 4x4

/-- Represents the 8x8 grid of non-negative integers -/
def Grid := Fin 8 → Fin 8 → ℕ

/-- Applies a single grid operation to the given grid -/
def applyOperation (grid : Grid) (op : GridOperation) : Grid :=
  sorry

/-- Checks if all numbers in the grid are multiples of 10 -/
def allMultiplesOf10 (grid : Grid) : Prop :=
  ∀ i j, (grid i j) % 10 = 0

/-- Main theorem: It's impossible to make all numbers multiples of 10 -/
theorem impossible_all_multiples_of_10 (initialGrid : Grid) :
  ¬∃ (ops : List GridOperation), allMultiplesOf10 (ops.foldl applyOperation initialGrid) :=
sorry

end impossible_all_multiples_of_10_l1772_177274


namespace new_person_age_l1772_177266

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) :
  n = 8 ∧ initial_avg = 14 ∧ new_avg = 16 →
  ∃ new_age : ℝ,
    new_age = n * new_avg + new_avg - n * initial_avg ∧
    new_age = 32 := by
  sorry

end new_person_age_l1772_177266


namespace solution_set_a_eq_1_range_of_a_l1772_177272

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |x + a|

-- Theorem 1: Solution set when a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x < -2} = {x : ℝ | x > 3/2} := by sorry

-- Theorem 2: Range of 'a'
theorem range_of_a :
  {a : ℝ | ∀ x y : ℝ, -2 + f a y ≤ f a x ∧ f a x ≤ 2 + f a y} =
  {a : ℝ | -3 ≤ a ∧ a ≤ -1} := by sorry

end solution_set_a_eq_1_range_of_a_l1772_177272


namespace intersection_of_M_and_N_l1772_177286

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 ≥ 0}
def N : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -1 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l1772_177286


namespace y_relationship_l1772_177204

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 + 4

-- Define the points A, B, C
def A : ℝ × ℝ := (-3, f (-3))
def B : ℝ × ℝ := (0, f 0)
def C : ℝ × ℝ := (3, f 3)

-- Define y₁, y₂, y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem y_relationship : y₃ < y₂ ∧ y₂ < y₁ := by
  sorry

end y_relationship_l1772_177204


namespace equal_means_sum_l1772_177215

theorem equal_means_sum (group1 group2 : Finset ℕ) : 
  (Finset.card group1 = 10) →
  (Finset.card group2 = 207) →
  (group1 ∪ group2 = Finset.range 217) →
  (group1 ∩ group2 = ∅) →
  (Finset.sum group1 id / Finset.card group1 = Finset.sum group2 id / Finset.card group2) →
  Finset.sum group1 id = 1090 := by
sorry

end equal_means_sum_l1772_177215


namespace condition_necessary_not_sufficient_l1772_177232

theorem condition_necessary_not_sufficient : 
  (∃ x : ℝ, x^2 - 2*x = 0 ∧ x ≠ 0) ∧ 
  (∀ x : ℝ, x = 0 → x^2 - 2*x = 0) := by
  sorry

end condition_necessary_not_sufficient_l1772_177232


namespace circle_area_ratio_l1772_177283

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360) * (2 * Real.pi * r₁) = (48 / 360) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
  sorry

end circle_area_ratio_l1772_177283


namespace negation_existence_quadratic_inequality_l1772_177233

theorem negation_existence_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end negation_existence_quadratic_inequality_l1772_177233


namespace table_formula_proof_l1772_177235

def f (x : ℕ) : ℕ := x^2 + 3*x + 1

theorem table_formula_proof :
  (f 1 = 5) ∧ (f 2 = 11) ∧ (f 3 = 19) ∧ (f 4 = 29) ∧ (f 5 = 41) :=
by sorry

end table_formula_proof_l1772_177235


namespace island_puzzle_l1772_177287

-- Define the types of residents
inductive Resident
| TruthTeller
| Liar

-- Define the statement made by K
def kStatement (k m : Resident) : Prop :=
  k = Resident.Liar ∨ m = Resident.Liar

-- Theorem to prove
theorem island_puzzle :
  ∃ (k m : Resident),
    (k = Resident.TruthTeller ∧ 
     m = Resident.Liar ∧
     (k = Resident.TruthTeller → kStatement k m) ∧
     (k = Resident.Liar → ¬kStatement k m)) :=
sorry

end island_puzzle_l1772_177287


namespace circle_regions_l1772_177299

/-- Number of regions created by n circles -/
def P (n : ℕ) : ℕ := 2 + n * (n - 1)

/-- The problem statement -/
theorem circle_regions : P 2011 ≡ 2112 [ZMOD 10000] := by
  sorry

end circle_regions_l1772_177299


namespace parabola_shift_l1772_177295

/-- Given a parabola y = 5x², shifting it 2 units left and 3 units up results in y = 5(x + 2)² + 3 -/
theorem parabola_shift (x y : ℝ) :
  (y = 5 * x^2) →
  (∃ y_shifted : ℝ, y_shifted = 5 * (x + 2)^2 + 3 ∧
    y_shifted = y + 3 ∧
    ∀ x_orig : ℝ, y = 5 * x_orig^2 → x = x_orig - 2) :=
by sorry

end parabola_shift_l1772_177295


namespace union_covers_reals_l1772_177254

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem union_covers_reals (a : ℝ) : 
  (A a ∪ B a = Set.univ) ↔ a ∈ Set.Iic 2 :=
sorry

end union_covers_reals_l1772_177254


namespace proposition_equivalence_l1772_177291

theorem proposition_equivalence :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.sqrt x₀ ≤ x₀ + 1) ↔ ¬(∀ x : ℝ, x > 0 → Real.sqrt x > x + 1) :=
by sorry

end proposition_equivalence_l1772_177291


namespace complex_product_pure_imaginary_l1772_177238

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_product_pure_imaginary (b : ℝ) :
  isPureImaginary ((1 + b * Complex.I) * (2 - Complex.I)) → b = -2 := by
  sorry

end complex_product_pure_imaginary_l1772_177238


namespace polynomial_value_at_2_l1772_177234

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 8

-- Theorem statement
theorem polynomial_value_at_2 : f 2 = 6 := by
  sorry

end polynomial_value_at_2_l1772_177234


namespace optimal_tire_swap_distance_l1772_177292

/-- The lifespan of a front tire in kilometers -/
def front_tire_lifespan : ℕ := 5000

/-- The lifespan of a rear tire in kilometers -/
def rear_tire_lifespan : ℕ := 3000

/-- The total distance traveled before both tires wear out when swapped optimally -/
def total_distance : ℕ := 3750

/-- Theorem stating that given the lifespans of front and rear tires, 
    the total distance traveled before both tires wear out when swapped optimally is 3750 km -/
theorem optimal_tire_swap_distance :
  ∀ (front_lifespan rear_lifespan : ℕ),
    front_lifespan = front_tire_lifespan →
    rear_lifespan = rear_tire_lifespan →
    (∃ (swap_strategy : ℕ → Bool),
      (∀ n : ℕ, swap_strategy n = true → swap_strategy (n + 1) = false) →
      (∃ (wear_front wear_rear : ℕ → ℝ),
        (∀ n : ℕ, wear_front n + wear_rear n = n) ∧
        (∀ n : ℕ, wear_front n ≤ front_lifespan) ∧
        (∀ n : ℕ, wear_rear n ≤ rear_lifespan) ∧
        (∃ m : ℕ, wear_front m = front_lifespan ∧ wear_rear m = rear_lifespan) ∧
        m = total_distance)) :=
by sorry


end optimal_tire_swap_distance_l1772_177292


namespace disk_space_calculation_l1772_177258

/-- The total space on Mike's disk drive in GB. -/
def total_space : ℕ := 28

/-- The space taken by Mike's files in GB. -/
def file_space : ℕ := 26

/-- The space left over after backing up Mike's files in GB. -/
def space_left : ℕ := 2

/-- Theorem stating that the total space on Mike's disk drive is equal to
    the sum of the space taken by his files and the space left over. -/
theorem disk_space_calculation :
  total_space = file_space + space_left := by sorry

end disk_space_calculation_l1772_177258


namespace johns_umbrella_cost_l1772_177247

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * cost_per_umbrella

/-- Proof that John's total cost for umbrellas is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end johns_umbrella_cost_l1772_177247


namespace chord_length_circle_line_l1772_177260

/-- The length of the chord intercepted by a circle on a line -/
theorem chord_length_circle_line (t : ℝ → ℝ × ℝ) (c : ℝ × ℝ → Prop) :
  (∀ r, t r = (-2 + r, 1 - r)) →  -- Line definition
  (∀ p, c p ↔ (p.1 - 3)^2 + (p.2 + 1)^2 = 25) →  -- Circle definition
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ c (t t₁) ∧ c (t t₂) ∧ 
    Real.sqrt ((t t₁).1 - (t t₂).1)^2 + ((t t₁).2 - (t t₂).2)^2 = Real.sqrt 82 :=
by sorry

end chord_length_circle_line_l1772_177260


namespace average_of_remaining_results_l1772_177245

theorem average_of_remaining_results (average_40 : ℝ) (average_all : ℝ) :
  average_40 = 30 →
  average_all = 34.285714285714285 →
  (70 * average_all - 40 * average_40) / 30 = 40 := by
sorry

end average_of_remaining_results_l1772_177245


namespace cauchy_functional_equation_verify_solution_l1772_177267

/-- A function satisfying the additive Cauchy equation -/
def is_additive (f : ℕ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A function satisfying f(nk) = n f(k) for all n, k ∈ ℕ -/
def satisfies_property (f : ℕ → ℝ) : Prop :=
  ∀ n k, f (n * k) = n * f k

theorem cauchy_functional_equation (f : ℕ → ℝ) 
  (h_additive : is_additive f) (h_property : satisfies_property f) :
  ∃ a : ℝ, ∀ n : ℕ, f n = a * n := by sorry

theorem verify_solution (a : ℝ) :
  let f : ℕ → ℝ := λ n ↦ a * n
  is_additive f ∧ satisfies_property f := by sorry

end cauchy_functional_equation_verify_solution_l1772_177267


namespace xy_value_l1772_177244

theorem xy_value (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 + 6*y + 9 = 0) : x*y = 9 := by
  sorry

end xy_value_l1772_177244


namespace sqrt_sum_div_sqrt_eq_rational_l1772_177280

theorem sqrt_sum_div_sqrt_eq_rational : (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175 = 13 / 5 := by
  sorry

end sqrt_sum_div_sqrt_eq_rational_l1772_177280


namespace binary_101111011_equals_379_l1772_177225

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101111011₂ (least significant bit first) -/
def binary_101111011 : List Bool := [true, true, false, true, true, true, true, false, true]

theorem binary_101111011_equals_379 :
  binary_to_decimal binary_101111011 = 379 := by
  sorry

end binary_101111011_equals_379_l1772_177225


namespace no_integer_solution_l1772_177213

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end no_integer_solution_l1772_177213


namespace binomial_expansion_example_l1772_177219

theorem binomial_expansion_example : 97^3 + 3*(97^2) + 3*97 + 1 = 940792 := by
  sorry

end binomial_expansion_example_l1772_177219


namespace star_star_equation_l1772_177221

theorem star_star_equation : 
  ∀ (a b : ℕ), a * b = 34 → (a = 2 ∧ b = 17) ∨ (a = 1 ∧ b = 34) ∨ (a = 17 ∧ b = 2) ∨ (a = 34 ∧ b = 1) :=
by sorry

end star_star_equation_l1772_177221


namespace xy_value_l1772_177282

theorem xy_value (x y : ℕ+) (h1 : x + y = 36) (h2 : 3 * x * y + 15 * x = 4 * y + 396) : x * y = 260 := by
  sorry

end xy_value_l1772_177282


namespace inequality_equivalence_l1772_177206

theorem inequality_equivalence (x : ℝ) : 
  (x - 2) / (x - 4) ≤ 3 ↔ 4 < x ∧ x ≤ 5 :=
sorry

end inequality_equivalence_l1772_177206


namespace equal_circles_radius_l1772_177276

/-- The radius of two equal circles that satisfy the given conditions -/
def radius_of_equal_circles : ℝ := 16

/-- The radius of the third circle that touches the line -/
def radius_of_third_circle : ℝ := 4

/-- Theorem stating that the radius of the two equal circles is 16 -/
theorem equal_circles_radius :
  let r₁ := radius_of_equal_circles
  let r₂ := radius_of_third_circle
  (r₁ : ℝ) > 0 ∧ r₂ > 0 ∧
  r₁^2 + (r₁ - r₂)^2 = (r₁ + r₂)^2 →
  r₁ = 16 := by sorry


end equal_circles_radius_l1772_177276


namespace simplify_product_of_radicals_l1772_177270

theorem simplify_product_of_radicals (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt x :=
by sorry

end simplify_product_of_radicals_l1772_177270


namespace calculation_proof_l1772_177284

theorem calculation_proof : 2 * (75 * 1313 - 25 * 1313) = 131300 := by
  sorry

end calculation_proof_l1772_177284


namespace perpendicular_vectors_sum_magnitude_l1772_177253

theorem perpendicular_vectors_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x - 1, -x]
  (a 0 * b 0 + a 1 * b 1 = 0) → 
  Real.sqrt ((a 0 + b 0)^2 + (a 1 + b 1)^2) = Real.sqrt 10 := by
  sorry

end perpendicular_vectors_sum_magnitude_l1772_177253


namespace arccos_one_over_sqrt_two_l1772_177243

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by sorry

end arccos_one_over_sqrt_two_l1772_177243


namespace hiker_supply_per_mile_l1772_177259

/-- A hiker's supply calculation problem -/
theorem hiker_supply_per_mile
  (hiking_rate : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (first_pack_weight : ℝ)
  (resupply_percentage : ℝ)
  (h1 : hiking_rate = 2.5)
  (h2 : hours_per_day = 8)
  (h3 : days = 5)
  (h4 : first_pack_weight = 40)
  (h5 : resupply_percentage = 0.25)
  : (first_pack_weight + first_pack_weight * resupply_percentage) / (hiking_rate * hours_per_day * days) = 0.5 := by
  sorry

end hiker_supply_per_mile_l1772_177259


namespace added_number_after_doubling_l1772_177237

theorem added_number_after_doubling (x : ℕ) (y : ℕ) (h : x = 19) :
  3 * (2 * x + y) = 129 → y = 5 := by
  sorry

end added_number_after_doubling_l1772_177237


namespace hammer_wrench_problem_l1772_177257

theorem hammer_wrench_problem (H W : ℝ) (x : ℕ) 
  (h1 : 2 * H + 2 * W = (1 / 3) * (x * H + 5 * W))
  (h2 : W = 2 * H) :
  x = 8 := by
  sorry

end hammer_wrench_problem_l1772_177257


namespace line_parallel_to_x_axis_l1772_177229

/-- A line through two points (x₁, y₁) and (x₂, y₂) is parallel to the x-axis if and only if y₁ = y₂ -/
def parallel_to_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂

/-- The problem statement -/
theorem line_parallel_to_x_axis (k : ℝ) :
  parallel_to_x_axis 3 (2*k + 1) 8 (4*k - 5) ↔ k = 3 := by
  sorry

end line_parallel_to_x_axis_l1772_177229


namespace perfect_square_sum_implies_divisible_by_eight_l1772_177288

theorem perfect_square_sum_implies_divisible_by_eight (a n : ℕ) (h1 : a > 0) (h2 : Even a) 
  (h3 : ∃ k : ℕ, k^2 = (a^(n+1) - 1) / (a - 1)) : 8 ∣ a := by
  sorry

end perfect_square_sum_implies_divisible_by_eight_l1772_177288


namespace dartboard_sector_angle_l1772_177203

theorem dartboard_sector_angle (total_angle : ℝ) (sector_prob : ℝ) : 
  total_angle = 360 → 
  sector_prob = 1/4 → 
  sector_prob * total_angle = 90 :=
by sorry

end dartboard_sector_angle_l1772_177203


namespace quadratic_function_properties_l1772_177218

/-- A quadratic function symmetric about x = 1 and passing through the origin -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The function is symmetric about x = 1 -/
def symmetric_about_one (a b : ℝ) : Prop := ∀ x, f a b (1 + x) = f a b (1 - x)

/-- The function passes through the origin -/
def passes_through_origin (a b : ℝ) : Prop := f a b 0 = 0

theorem quadratic_function_properties (a b : ℝ) 
  (h1 : symmetric_about_one a b) (h2 : passes_through_origin a b) :
  (∀ x, f a b x = x^2 - 2*x) ∧ 
  Set.Icc (-1) 3 = Set.range (fun x => f a b x) ∩ Set.Ioo 0 3 := by
  sorry

end quadratic_function_properties_l1772_177218


namespace star_example_l1772_177296

/-- The star operation for fractions -/
def star (m n p q : ℚ) : ℚ := (m + 1) * (p - 1) * ((q + 1) / (n - 1))

/-- Theorem stating that 5/7 ★ 9/4 = 40 -/
theorem star_example : star 5 7 9 4 = 40 := by sorry

end star_example_l1772_177296


namespace max_points_in_tournament_l1772_177277

/-- Represents a tournament with the given conditions --/
structure Tournament :=
  (num_teams : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (points_for_loss : Nat)

/-- Calculates the total number of games in the tournament --/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1)) / 2 * 2

/-- Represents the maximum points achievable by top teams --/
def max_points_for_top_teams (t : Tournament) : Nat :=
  let games_with_other_top_teams := 4
  let games_with_lower_teams := 6
  games_with_other_top_teams * t.points_for_win / 2 +
  games_with_lower_teams * t.points_for_win

/-- The main theorem to be proved --/
theorem max_points_in_tournament (t : Tournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0) :
  max_points_for_top_teams t = 24 := by
  sorry

#eval max_points_for_top_teams ⟨6, 3, 1, 0⟩

end max_points_in_tournament_l1772_177277


namespace distributive_property_l1772_177279

theorem distributive_property (x : ℝ) : -2 * (x + 1) = -2 * x - 2 := by
  sorry

end distributive_property_l1772_177279


namespace fraction_inequality_l1772_177273

theorem fraction_inequality (x : ℝ) : 
  -4 ≤ (x^2 - 2*x - 3) / (2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3) / (2*x^2 + 2*x + 1) ≤ 1 := by
  sorry

end fraction_inequality_l1772_177273


namespace ellipse_k_range_l1772_177251

/-- The curve equation --/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - k) + y^2 / (1 + k) = 1

/-- The curve represents an ellipse --/
def is_ellipse (k : ℝ) : Prop :=
  1 - k > 0 ∧ 1 + k > 0 ∧ 1 - k ≠ 1 + k

/-- The range of k for which the curve represents an ellipse --/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ ((-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)) :=
by sorry

end ellipse_k_range_l1772_177251


namespace sqrt_equation_solution_l1772_177214

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 6) = 7 → x = 43 := by
  sorry

end sqrt_equation_solution_l1772_177214
