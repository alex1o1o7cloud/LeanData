import Mathlib

namespace subset_complement_of_intersection_eq_l856_85608

universe u

theorem subset_complement_of_intersection_eq {U : Type u} [TopologicalSpace U] (M N : Set U) 
  (h : M ∩ N = N) : (Mᶜ : Set U) ⊆ Nᶜ := by
  sorry

end subset_complement_of_intersection_eq_l856_85608


namespace tank_A_height_approx_5_l856_85630

/-- The circumference of Tank A in meters -/
def circumference_A : ℝ := 4

/-- The circumference of Tank B in meters -/
def circumference_B : ℝ := 10

/-- The height of Tank B in meters -/
def height_B : ℝ := 8

/-- The ratio of Tank A's capacity to Tank B's capacity -/
def capacity_ratio : ℝ := 0.10000000000000002

/-- The height of Tank A in meters -/
noncomputable def height_A : ℝ := 
  capacity_ratio * (circumference_B / circumference_A)^2 * height_B

theorem tank_A_height_approx_5 : 
  ∃ ε > 0, abs (height_A - 5) < ε := by sorry

end tank_A_height_approx_5_l856_85630


namespace dress_cost_theorem_l856_85663

/-- The total cost of dresses for Patty, Ida, Jean, and Pauline -/
def total_cost (patty ida jean pauline : ℕ) : ℕ := patty + ida + jean + pauline

/-- Theorem stating the total cost of dresses given the conditions -/
theorem dress_cost_theorem :
  ∀ (patty ida jean pauline : ℕ),
    patty = ida + 10 →
    ida = jean + 30 →
    jean = pauline - 10 →
    pauline = 30 →
    total_cost patty ida jean pauline = 160 := by
  sorry

end dress_cost_theorem_l856_85663


namespace tan_sum_given_sin_cos_sum_l856_85627

theorem tan_sum_given_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 5/13)
  (h2 : Real.cos x + Real.cos y = 12/13) : 
  Real.tan x + Real.tan y = 240/119 := by
  sorry

end tan_sum_given_sin_cos_sum_l856_85627


namespace speed_conversion_l856_85679

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

theorem speed_conversion :
  let speed_mps : ℝ := 5.0004
  mps_to_kmph speed_mps = 18.00144 := by sorry

end speed_conversion_l856_85679


namespace student_presentations_periods_class_presentation_periods_l856_85676

/-- Calculates the number of periods needed for all student presentations --/
theorem student_presentations_periods (total_students : ℕ) (period_length : ℕ) 
  (individual_presentation_time : ℕ) (individual_qa_time : ℕ) 
  (group_presentations : ℕ) (group_presentation_time : ℕ) : ℕ :=
  let individual_students := total_students - group_presentations
  let individual_time := individual_students * (individual_presentation_time + individual_qa_time)
  let group_time := group_presentations * group_presentation_time
  let total_time := individual_time + group_time
  (total_time + period_length - 1) / period_length

/-- The number of periods needed for the given class presentation scenario is 7 --/
theorem class_presentation_periods : 
  student_presentations_periods 32 40 5 3 4 12 = 7 := by
  sorry

end student_presentations_periods_class_presentation_periods_l856_85676


namespace string_folding_theorem_l856_85673

/-- The number of layers after folding a string n times -/
def layers (n : ℕ) : ℕ := 2^n

/-- The number of longer strings after folding and cutting -/
def longer_strings (total_layers : ℕ) : ℕ := total_layers - 1

/-- The number of shorter strings after folding and cutting -/
def shorter_strings (total_layers : ℕ) (num_cuts : ℕ) : ℕ :=
  (num_cuts - 2) * total_layers + 2

theorem string_folding_theorem (num_folds num_cuts : ℕ) 
  (h1 : num_folds = 10) (h2 : num_cuts = 10) :
  longer_strings (layers num_folds) = 1023 ∧
  shorter_strings (layers num_folds) num_cuts = 8194 := by
  sorry

#eval longer_strings (layers 10)
#eval shorter_strings (layers 10) 10

end string_folding_theorem_l856_85673


namespace rectangle_in_square_l856_85622

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the arrangement of rectangles in the square -/
def arrangement (r : Rectangle) : ℝ := 2 * r.length + 2 * r.width

/-- The theorem stating the properties of the rectangles in the square -/
theorem rectangle_in_square (r : Rectangle) : 
  arrangement r = 18 ∧ 3 * r.length = 18 → r.length = 6 ∧ r.width = 3 := by
  sorry

end rectangle_in_square_l856_85622


namespace fifteenth_triangular_less_than_square_l856_85605

-- Define the triangular number function
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement
theorem fifteenth_triangular_less_than_square :
  triangular_number 15 < 15^2 := by
  sorry

end fifteenth_triangular_less_than_square_l856_85605


namespace unique_perpendicular_line_l856_85616

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Predicate to check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Two lines are perpendicular if their slopes are negative reciprocals of each other -/
def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Main theorem: There exists exactly one perpendicular line through a point on a given line -/
theorem unique_perpendicular_line (l : Line) (p : Point) (h : p.liesOn l) :
  ∃! l_perp : Line, l_perp.isPerpendicular l ∧ p.liesOn l_perp :=
sorry

end unique_perpendicular_line_l856_85616


namespace inequality_solution_set_l856_85631

theorem inequality_solution_set : 
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = 
  Set.union (Set.Ioc (-2) 1) (Set.Icc 4 7) := by sorry

end inequality_solution_set_l856_85631


namespace sweater_price_theorem_l856_85691

def total_price_shirts : ℕ := 400
def num_shirts : ℕ := 25
def num_sweaters : ℕ := 75
def price_diff : ℕ := 4

theorem sweater_price_theorem :
  let avg_shirt_price := total_price_shirts / num_shirts
  let avg_sweater_price := avg_shirt_price + price_diff
  avg_sweater_price * num_sweaters = 1500 := by
  sorry

end sweater_price_theorem_l856_85691


namespace polygon_exterior_angle_l856_85629

theorem polygon_exterior_angle (n : ℕ) (h : n > 2) : 
  (360 : ℝ) / (n : ℝ) = 24 → n = 15 := by
  sorry

end polygon_exterior_angle_l856_85629


namespace g_1989_of_5_eq_5_l856_85674

def g (x : ℚ) : ℚ := (2 - x) / (1 + 2 * x)

def g_n : ℕ → (ℚ → ℚ)
| 0 => λ x => x
| n + 1 => λ x => g (g_n n x)

theorem g_1989_of_5_eq_5 : g_n 1989 5 = 5 := by sorry

end g_1989_of_5_eq_5_l856_85674


namespace jasons_books_l856_85648

theorem jasons_books (books_per_shelf : ℕ) (num_shelves : ℕ) (h1 : books_per_shelf = 45) (h2 : num_shelves = 7) :
  books_per_shelf * num_shelves = 315 := by
  sorry

end jasons_books_l856_85648


namespace combinations_to_arrangements_l856_85695

theorem combinations_to_arrangements (n : ℕ) (h1 : n ≥ 2) (h2 : Nat.choose n 2 = 15) :
  (n.factorial / (n - 2).factorial) = 30 := by
  sorry

end combinations_to_arrangements_l856_85695


namespace kiki_scarf_problem_l856_85668

/-- Kiki's scarf and hat buying problem -/
theorem kiki_scarf_problem (total_money : ℝ) (scarf_price : ℝ) :
  total_money = 90 →
  scarf_price = 2 →
  ∃ (num_scarves num_hats : ℕ) (hat_price : ℝ),
    num_hats = 2 * num_scarves ∧
    hat_price * num_hats = 0.6 * total_money ∧
    scarf_price * num_scarves = 0.4 * total_money ∧
    num_scarves = 18 := by
  sorry


end kiki_scarf_problem_l856_85668


namespace quadratic_roots_l856_85647

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ x₁^2 - 3 = 0 ∧ x₂^2 - 3 = 0 := by
  sorry

end quadratic_roots_l856_85647


namespace acidic_solution_concentration_l856_85664

/-- Represents the properties of an acidic solution -/
structure AcidicSolution where
  initialVolume : ℝ
  removedVolume : ℝ
  finalConcentration : ℝ
  initialConcentration : ℝ

/-- Theorem stating the relationship between initial and final concentrations -/
theorem acidic_solution_concentration 
  (solution : AcidicSolution)
  (h1 : solution.initialVolume = 27)
  (h2 : solution.removedVolume = 9)
  (h3 : solution.finalConcentration = 60)
  (h4 : solution.initialConcentration * solution.initialVolume = 
        solution.finalConcentration * (solution.initialVolume - solution.removedVolume)) :
  solution.initialConcentration = 40 := by
  sorry

#check acidic_solution_concentration

end acidic_solution_concentration_l856_85664


namespace jen_age_theorem_l856_85626

def jen_age_when_son_born (jen_present_age : ℕ) (son_present_age : ℕ) : ℕ :=
  jen_present_age - son_present_age

theorem jen_age_theorem (jen_present_age : ℕ) (son_present_age : ℕ) :
  son_present_age = 16 →
  jen_present_age = 3 * son_present_age - 7 →
  jen_age_when_son_born jen_present_age son_present_age = 25 := by
  sorry

end jen_age_theorem_l856_85626


namespace unique_function_satisfying_conditions_l856_85639

-- Define a 7-arithmetic fractional-linear function
def is_7_arithmetic_fractional_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x : ℝ, f x = (a * x + b) / (c * x + d)

-- State the theorem
theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, 
    is_7_arithmetic_fractional_linear f ∧ 
    f 0 = 0 ∧ 
    f 1 = 4 ∧ 
    f 4 = 2 ∧
    ∀ x : ℝ, f x = x / 2 := by
  sorry

end unique_function_satisfying_conditions_l856_85639


namespace paving_rate_per_square_meter_l856_85624

/-- Given a room with length 5.5 m and width 3.75 m, and a total paving cost of Rs. 16500,
    the rate of paving per square meter is Rs. 800. -/
theorem paving_rate_per_square_meter
  (length : ℝ)
  (width : ℝ)
  (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
sorry

end paving_rate_per_square_meter_l856_85624


namespace arithmetic_mean_of_scores_l856_85651

def scores : List ℝ := [93, 87, 90, 94, 88, 92]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 90.6667 := by
  sorry

end arithmetic_mean_of_scores_l856_85651


namespace roles_assignment_count_l856_85621

/-- The number of ways to assign n distinct roles to n different people. -/
def assignRoles (n : ℕ) : ℕ := Nat.factorial n

/-- There are four team members. -/
def numTeamMembers : ℕ := 4

/-- There are four different roles. -/
def numRoles : ℕ := 4

/-- Each person can only take one role. -/
axiom one_role_per_person : numTeamMembers = numRoles

theorem roles_assignment_count :
  assignRoles numTeamMembers = 24 :=
sorry

end roles_assignment_count_l856_85621


namespace compute_expression_l856_85644

theorem compute_expression : 7 + 4 * (5 - 9)^3 = -249 := by
  sorry

end compute_expression_l856_85644


namespace factor_implies_c_equals_three_l856_85610

theorem factor_implies_c_equals_three (c : ℝ) : 
  (∀ x : ℝ, (x + 7) ∣ (c * x^3 + 19 * x^2 - 4 * c * x + 20)) → c = 3 := by
  sorry

end factor_implies_c_equals_three_l856_85610


namespace cut_out_pieces_border_l856_85625

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a piece that can be cut out from the grid -/
inductive Piece
  | UnitSquare
  | LShape

/-- Represents the configuration of cut-out pieces -/
structure CutOutConfig :=
  (grid : Grid)
  (unitSquares : ℕ)
  (lShapes : ℕ)

/-- Predicate to check if two pieces border each other -/
def border (p1 p2 : Piece) : Prop := sorry

theorem cut_out_pieces_border
  (config : CutOutConfig)
  (h1 : config.grid.size = 55)
  (h2 : config.unitSquares = 500)
  (h3 : config.lShapes = 400) :
  ∃ (p1 p2 : Piece), p1 ≠ p2 ∧ border p1 p2 :=
sorry

end cut_out_pieces_border_l856_85625


namespace cost_price_calculation_cost_price_proof_l856_85612

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem cost_price_proof :
  cost_price_calculation 12000 0.1 0.08 = 10000 := by
  sorry

end cost_price_calculation_cost_price_proof_l856_85612


namespace johns_donation_l856_85666

theorem johns_donation (n : ℕ) (new_avg : ℚ) (increase_percent : ℚ) :
  n = 1 →
  new_avg = 75 →
  increase_percent = 50 / 100 →
  let old_avg := new_avg / (1 + increase_percent)
  let total_before := old_avg * n
  let total_after := new_avg * (n + 1)
  total_after - total_before = 100 := by
sorry

end johns_donation_l856_85666


namespace combinations_equal_twenty_l856_85606

/-- The number of available paint colors. -/
def num_colors : ℕ := 5

/-- The number of available painting methods. -/
def num_methods : ℕ := 4

/-- The total number of combinations of paint colors and painting methods. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20. -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end combinations_equal_twenty_l856_85606


namespace only_3_4_5_is_right_triangle_l856_85685

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 2), (3, 4, 5), (3, 4, 9), (4, 5, 7)]

/-- Theorem stating that only (3, 4, 5) forms a right-angled triangle --/
theorem only_3_4_5_is_right_triangle :
  ∃! (a b c : ℕ), (a, b, c) ∈ sets ∧ is_right_triangle a b c :=
by sorry

end only_3_4_5_is_right_triangle_l856_85685


namespace possible_values_of_a_l856_85615

theorem possible_values_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | a * x + 2 = 0} → 
  B = {-1, 2} → 
  A ⊆ B → 
  {a | ∃ (A : Set ℝ), A = {x : ℝ | a * x + 2 = 0} ∧ A ⊆ B} = {-1, 0, 2} := by
sorry

end possible_values_of_a_l856_85615


namespace salary_comparison_l856_85602

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  (b - a) / a * 100 = 25 := by
  sorry

end salary_comparison_l856_85602


namespace fraction_zero_implies_x_equals_three_l856_85633

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 → x = 3 := by
sorry

end fraction_zero_implies_x_equals_three_l856_85633


namespace parallelogram_area_theorem_l856_85672

/-- Represents the area of a parallelogram with a square removed -/
def parallelogram_area_with_square_removed (base : ℝ) (height : ℝ) (square_side : ℝ) : ℝ :=
  base * height - square_side * square_side

/-- Theorem stating that a parallelogram with base 20 and height 4, 
    after removing a 2x2 square, has an area of 76 square feet -/
theorem parallelogram_area_theorem :
  parallelogram_area_with_square_removed 20 4 2 = 76 := by
  sorry

#eval parallelogram_area_with_square_removed 20 4 2

end parallelogram_area_theorem_l856_85672


namespace deriv_zero_necessary_not_sufficient_l856_85646

-- Define a differentiable function f from ℝ to ℝ
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem deriv_zero_necessary_not_sufficient :
  (∀ x₀, IsExtremum f x₀ → deriv f x₀ = 0) ∧
  ¬(∀ x₀, deriv f x₀ = 0 → IsExtremum f x₀) :=
sorry

end deriv_zero_necessary_not_sufficient_l856_85646


namespace pizza_toppings_combinations_l856_85623

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l856_85623


namespace sum_of_powers_of_i_equals_zero_l856_85645

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^14560 + i^14561 + i^14562 + i^14563 = 0 := by
  sorry

end sum_of_powers_of_i_equals_zero_l856_85645


namespace square_triangle_equal_area_l856_85687

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 64 →
  triangle_height = 64 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * triangle_base →
  triangle_base = 8 := by
sorry

end square_triangle_equal_area_l856_85687


namespace final_state_is_green_l856_85611

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Function to model the color change when two chameleons of different colors meet -/
def colorChange (state : ChameleonState) : ChameleonState :=
  sorry

/-- Predicate to check if all chameleons have the same color -/
def allSameColor (state : ChameleonState) : Prop :=
  (state.yellow = totalChameleons) ∨ (state.red = totalChameleons) ∨ (state.green = totalChameleons)

/-- The main theorem to prove -/
theorem final_state_is_green :
  ∃ (finalState : ChameleonState),
    (allSameColor finalState) ∧ (finalState.green = totalChameleons) :=
  sorry

end final_state_is_green_l856_85611


namespace imaginary_part_of_z_l856_85642

theorem imaginary_part_of_z (z : ℂ) (h : (3 + 4*I)*z = 5) : 
  z.im = -4/5 := by sorry

end imaginary_part_of_z_l856_85642


namespace fraction_equality_l856_85669

theorem fraction_equality : (900 ^ 2 : ℝ) / (306 ^ 2 - 294 ^ 2) = 112.5 := by
  sorry

end fraction_equality_l856_85669


namespace polynomial_roots_theorem_l856_85653

theorem polynomial_roots_theorem (a b c : ℝ) : 
  (∃ (r s t : ℝ), 
    (∀ x : ℝ, x^4 - a*x^3 + b*x^2 - c*x + a = 0 ↔ x = 0 ∨ x = r ∨ x = s ∨ x = t) ∧
    (a > 0) ∧
    (∀ a' : ℝ, a' > 0 → a' ≥ a)) →
  a = 3 * Real.sqrt 3 ∧ b = 9 ∧ c = 3 * Real.sqrt 3 :=
by sorry

end polynomial_roots_theorem_l856_85653


namespace points_in_quadrants_I_and_II_l856_85650

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > -2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by sorry

end points_in_quadrants_I_and_II_l856_85650


namespace f_deriv_positive_at_midpoint_l856_85681

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + Real.log x

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := 2*x + 1/x

-- Theorem statement
theorem f_deriv_positive_at_midpoint 
  (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) 
  (h₂ : x₁ < x₂) 
  (h₃ : f x₁ = 0) 
  (h₄ : f x₂ = 0) :
  let x₀ := (x₁ + x₂) / 2
  f_deriv x₀ > 0 := by
sorry

end

end f_deriv_positive_at_midpoint_l856_85681


namespace number_of_hens_l856_85688

def number_of_goats : ℕ := 5
def total_cost : ℕ := 2500
def price_of_hen : ℕ := 50
def price_of_goat : ℕ := 400

theorem number_of_hens : 
  ∃ (h : ℕ), h * price_of_hen + number_of_goats * price_of_goat = total_cost ∧ h = 10 :=
by sorry

end number_of_hens_l856_85688


namespace prob_both_three_eq_one_forty_second_l856_85662

/-- A fair die with n sides -/
def FairDie (n : ℕ) := Fin n

/-- The probability of rolling a specific number on a fair die with n sides -/
def prob_specific_roll (n : ℕ) : ℚ := 1 / n

/-- The probability of rolling a 3 on a 6-sided die and a 7-sided die simultaneously -/
def prob_both_three : ℚ := (prob_specific_roll 6) * (prob_specific_roll 7)

theorem prob_both_three_eq_one_forty_second :
  prob_both_three = 1 / 42 := by sorry

end prob_both_three_eq_one_forty_second_l856_85662


namespace quadratic_root_sum_l856_85692

theorem quadratic_root_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2023*x₁ + 1 = 0 → 
  x₂^2 - 2023*x₂ + 1 = 0 → 
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = 2023 := by
sorry

end quadratic_root_sum_l856_85692


namespace variance_of_transformed_data_l856_85689

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (data : Fin n → ℝ) : ℝ := sorry

def transform (data : Fin n → ℝ) : Fin n → ℝ := 
  fun i => 3 * data i + 1

theorem variance_of_transformed_data 
  (h : variance x = 2) : 
  variance (transform x) = 18 := by sorry

end variance_of_transformed_data_l856_85689


namespace divisibility_of_product_difference_l856_85699

theorem divisibility_of_product_difference (a₁ a₂ b₁ b₂ c₁ c₂ d : ℤ) 
  (h1 : d ∣ (a₁ - a₂)) 
  (h2 : d ∣ (b₁ - b₂)) 
  (h3 : d ∣ (c₁ - c₂)) : 
  d ∣ (a₁ * b₁ * c₁ - a₂ * b₂ * c₂) := by
  sorry

end divisibility_of_product_difference_l856_85699


namespace fruit_stand_problem_l856_85609

/-- Represents the number of fruits Mary selects -/
structure FruitSelection where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculates the total cost of fruits in cents -/
def totalCost (s : FruitSelection) : ℕ :=
  40 * s.apples + 60 * s.oranges + 80 * s.bananas

/-- Calculates the average cost of fruits in cents -/
def averageCost (s : FruitSelection) : ℚ :=
  (totalCost s : ℚ) / (s.apples + s.oranges + s.bananas : ℚ)

theorem fruit_stand_problem (s : FruitSelection) 
  (total_fruits : s.apples + s.oranges + s.bananas = 12)
  (initial_avg : averageCost s = 55) :
  let new_selection := FruitSelection.mk s.apples (s.oranges - 6) s.bananas
  averageCost new_selection = 50 := by
  sorry

end fruit_stand_problem_l856_85609


namespace inequality_solution_l856_85656

theorem inequality_solution (x : ℝ) : 
  (10 * x^2 + 20 * x - 60) / ((3 * x - 5) * (x + 6)) < 4 ↔ 
  (x > -6 ∧ x < 5/3) ∨ x > 2 :=
by sorry

end inequality_solution_l856_85656


namespace greatest_n_value_l856_85613

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 ∧ ∃ m : ℤ, m = 7 ∧ 101 * m^2 ≤ 6400 :=
sorry

end greatest_n_value_l856_85613


namespace stock_transaction_profit_l856_85604

theorem stock_transaction_profit
  (initial_price : ℝ)
  (profit_percentage : ℝ)
  (loss_percentage : ℝ)
  (final_sale_percentage : ℝ)
  (h1 : initial_price = 1000)
  (h2 : profit_percentage = 0.1)
  (h3 : loss_percentage = 0.1)
  (h4 : final_sale_percentage = 0.9) :
  let first_sale_price := initial_price * (1 + profit_percentage)
  let second_sale_price := first_sale_price * (1 - loss_percentage)
  let final_sale_price := second_sale_price * final_sale_percentage
  final_sale_price - initial_price = 1 :=
by sorry

end stock_transaction_profit_l856_85604


namespace no_hexagon_for_19_and_20_l856_85658

theorem no_hexagon_for_19_and_20 : 
  (¬ ∃ (ℓ : ℤ), 19 = 2 * ℓ^2 + ℓ) ∧ (¬ ∃ (ℓ : ℤ), 20 = 2 * ℓ^2 + ℓ) := by
  sorry

end no_hexagon_for_19_and_20_l856_85658


namespace partial_fraction_decomposition_l856_85683

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (x^2 + 5*x - 6) / (x^3 - x) = 6 / x + (-5*x + 5) / (x^2 - 1) := by
  sorry

end partial_fraction_decomposition_l856_85683


namespace mean_of_six_numbers_with_sum_one_third_l856_85677

theorem mean_of_six_numbers_with_sum_one_third :
  ∀ (a b c d e f : ℚ),
  a + b + c + d + e + f = 1/3 →
  (a + b + c + d + e + f) / 6 = 1/18 := by
sorry

end mean_of_six_numbers_with_sum_one_third_l856_85677


namespace intersection_condition_l856_85635

/-- The set M in ℝ² -/
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}

/-- The set N in ℝ² parameterized by a -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- The necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end intersection_condition_l856_85635


namespace nurses_count_l856_85628

theorem nurses_count (total_staff : ℕ) (doctor_ratio nurse_ratio : ℕ) : 
  total_staff = 200 → 
  doctor_ratio = 4 → 
  nurse_ratio = 6 → 
  (nurse_ratio : ℚ) / (doctor_ratio + nurse_ratio : ℚ) * total_staff = 120 := by
sorry

end nurses_count_l856_85628


namespace three_numbers_sum_l856_85680

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 45 := by
  sorry

end three_numbers_sum_l856_85680


namespace constant_term_binomial_expansion_l856_85667

theorem constant_term_binomial_expansion :
  ∃ (n : ℕ), n = 11 ∧ 
  (∀ (r : ℕ), (15 : ℝ) - (3 / 2 : ℝ) * r = 0 → r = 10) ∧
  (∀ (k : ℕ), k ≠ n - 1 → 
    ∃ (c : ℝ), c ≠ 0 ∧ 
    (Nat.choose 15 k * (6 : ℝ)^(15 - k) * (-1 : ℝ)^k) * (0 : ℝ)^(15 - (3 * k) / 2) = c) :=
by sorry

end constant_term_binomial_expansion_l856_85667


namespace rectangle_longer_side_l856_85675

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ)
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : circle_area = Real.pi * circle_radius ^ 2)
  (h4 : rectangle_area = 2 * circle_radius * longer_side) :
  longer_side = 9 * Real.pi := by
  sorry

end rectangle_longer_side_l856_85675


namespace arithmetic_sequence_sum_l856_85600

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 29) →
  (a 3 + a 6 + a 9 = 13) :=
by
  sorry

end arithmetic_sequence_sum_l856_85600


namespace u_v_sum_of_squares_l856_85617

theorem u_v_sum_of_squares (u v : ℝ) (hu : u > 1) (hv : v > 1)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^Real.sqrt 5 + 7^Real.sqrt 5 := by
  sorry

end u_v_sum_of_squares_l856_85617


namespace max_triangle_area_l856_85637

def parabola (x : ℝ) : ℝ := -x^2 + 6*x - 5

theorem max_triangle_area :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (4, 3)
  let C (p : ℝ) : ℝ × ℝ := (p, parabola p)
  let triangle_area (p : ℝ) : ℝ := 
    (1/2) * abs ((A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2) - 
                 (A.2 * B.1 + B.2 * (C p).1 + (C p).2 * A.1))
  ∀ p : ℝ, 1 ≤ p ∧ p ≤ 4 → triangle_area p ≤ 27/8 :=
by
  sorry

#check max_triangle_area

end max_triangle_area_l856_85637


namespace vegetable_ghee_mixture_l856_85670

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

/-- The ratio of brand 'a' to brand 'b' in the mixture by volume -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3440

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 370

theorem vegetable_ghee_mixture :
  weight_a * (ratio_a * total_volume / (ratio_a + ratio_b)) +
  weight_b * (ratio_b * total_volume / (ratio_a + ratio_b)) = total_weight :=
sorry

end vegetable_ghee_mixture_l856_85670


namespace min_points_all_but_one_hemisphere_l856_85684

/-- A point on the surface of a sphere -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ
  on_sphere : x^2 + y^2 + z^2 = 1

/-- A hemisphere of a sphere -/
def Hemisphere := Set Point3D

/-- The set of all possible hemispheres of a sphere -/
def AllHemispheres : Set Hemisphere := sorry

/-- A set of points is in all hemispheres except one if it intersects with all but one hemisphere -/
def InAllButOneHemisphere (points : Set Point3D) : Prop :=
  ∃ h : Hemisphere, h ∈ AllHemispheres ∧ 
    ∀ h' : Hemisphere, h' ∈ AllHemispheres → h' ≠ h → (points ∩ h').Nonempty

theorem min_points_all_but_one_hemisphere :
  ∃ (points : Set Point3D), points.ncard = 4 ∧ InAllButOneHemisphere points ∧
    ∀ (points' : Set Point3D), points'.ncard < 4 → ¬InAllButOneHemisphere points' :=
  sorry

end min_points_all_but_one_hemisphere_l856_85684


namespace ray_reflection_l856_85671

/-- Given a point A, a line l, and a point B, prove the equations of the incident and reflected rays --/
theorem ray_reflection (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  A = (2, 3) → 
  B = (1, 1) → 
  (∀ x y, l x y ↔ x + y + 1 = 0) →
  ∃ (incident reflected : ℝ → ℝ → Prop),
    (∀ x y, incident x y ↔ 9*x - 7*y + 3 = 0) ∧
    (∀ x y, reflected x y ↔ 7*x - 9*y + 2 = 0) :=
by sorry

end ray_reflection_l856_85671


namespace nested_custom_op_equals_two_l856_85619

/-- Custom operation [a, b, c] defined as (a + b) / c where c ≠ 0 -/
def customOp (a b c : ℚ) : ℚ := (a + b) / c

/-- Theorem stating that [[50,25,75],[6,3,9],[8,4,12]] = 2 -/
theorem nested_custom_op_equals_two :
  customOp (customOp 50 25 75) (customOp 6 3 9) (customOp 8 4 12) = 2 := by
  sorry


end nested_custom_op_equals_two_l856_85619


namespace sqrt_52_rational_l856_85649

theorem sqrt_52_rational : 
  (((52 : ℝ).sqrt + 5) ^ (1/3 : ℝ)) - (((52 : ℝ).sqrt - 5) ^ (1/3 : ℝ)) = 1 := by
  sorry

end sqrt_52_rational_l856_85649


namespace electric_water_ratio_l856_85618

def monthly_earnings : ℚ := 6000
def house_rental : ℚ := 640
def food_expense : ℚ := 380
def insurance_ratio : ℚ := 1 / 5
def remaining_money : ℚ := 2280

theorem electric_water_ratio :
  let insurance_cost := insurance_ratio * monthly_earnings
  let total_expenses := house_rental + food_expense + insurance_cost
  let electric_water_bill := monthly_earnings - total_expenses - remaining_money
  electric_water_bill / monthly_earnings = 1 / 4 := by sorry

end electric_water_ratio_l856_85618


namespace cannot_find_fourth_vertex_l856_85661

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Symmetric point operation -/
def symmetricPoint (a b : Point) : Point :=
  { x := 2 * b.x - a.x, y := 2 * b.y - a.y }

/-- Represents a square -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Checks if a point is a valid fourth vertex of a square -/
def isValidFourthVertex (s : Square) (p : Point) : Prop := sorry

theorem cannot_find_fourth_vertex (s : Square) :
  ¬ ∃ (p : Point), (∃ (a b : Point), p = symmetricPoint a b) ∧ isValidFourthVertex s p := by
  sorry

end cannot_find_fourth_vertex_l856_85661


namespace student_arrangement_l856_85640

/-- The number of ways to arrange students with specific conditions -/
def arrangement_count : ℕ := 120

/-- The number of male students -/
def male_students : ℕ := 3

/-- The number of female students -/
def female_students : ℕ := 4

/-- The number of students that must stand at the ends -/
def end_students : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := male_students + female_students

theorem student_arrangement :
  arrangement_count = 
    (end_students * (total_students - end_students).factorial) :=
sorry

end student_arrangement_l856_85640


namespace square_perimeter_relation_l856_85694

/-- Given two squares A and B, where A has a perimeter of 40 cm and B has an area
    equal to one-third the area of A, the perimeter of B is (40√3)/3 cm. -/
theorem square_perimeter_relation (A B : Real → Real → Prop) :
  (∃ s, A s s ∧ 4 * s = 40) →  -- Square A has perimeter 40 cm
  (∀ x y, B x y ↔ x = y ∧ x^2 = (1/3) * s^2) →  -- B's area is 1/3 of A's area
  (∃ p, ∀ x y, B x y → 4 * x = p ∧ p = (40 * Real.sqrt 3) / 3) :=
by sorry

end square_perimeter_relation_l856_85694


namespace coffee_packages_solution_l856_85697

/-- Represents the number of 10-ounce packages -/
def num_10oz : ℕ := 4

/-- Represents the number of 5-ounce packages -/
def num_5oz : ℕ := num_10oz + 2

/-- Total ounces of coffee -/
def total_ounces : ℕ := 115

/-- Cost of a 5-ounce package in cents -/
def cost_5oz : ℕ := 150

/-- Cost of a 10-ounce package in cents -/
def cost_10oz : ℕ := 250

/-- Maximum total cost in cents -/
def max_cost : ℕ := 2000

theorem coffee_packages_solution :
  (num_10oz * 10 + num_5oz * 5 = total_ounces) ∧
  (num_10oz * cost_10oz + num_5oz * cost_5oz ≤ max_cost) :=
by sorry

end coffee_packages_solution_l856_85697


namespace choir_meeting_interval_l856_85682

/-- The number of days between drama club meetings -/
def drama_interval : ℕ := 3

/-- The number of days until the next joint meeting -/
def next_joint_meeting : ℕ := 15

/-- The number of days between choir meetings -/
def choir_interval : ℕ := 5

theorem choir_meeting_interval :
  (next_joint_meeting % drama_interval = 0) ∧
  (next_joint_meeting % choir_interval = 0) ∧
  (∀ x : ℕ, x > 1 ∧ x < choir_interval →
    ¬(next_joint_meeting % x = 0 ∧ next_joint_meeting % drama_interval = 0)) :=
by sorry

end choir_meeting_interval_l856_85682


namespace second_number_calculation_l856_85634

theorem second_number_calculation (A B : ℝ) (h1 : A = 680) (h2 : 0.2 * A = 0.4 * B + 80) : B = 140 := by
  sorry

end second_number_calculation_l856_85634


namespace andrew_total_hours_l856_85690

/-- Andrew's work on his Science report -/
def andrew_work : ℝ → ℝ → ℝ := fun days hours_per_day => days * hours_per_day

/-- The theorem stating the total hours Andrew worked -/
theorem andrew_total_hours : andrew_work 3 2.5 = 7.5 := by
  sorry

end andrew_total_hours_l856_85690


namespace cos_2alpha_value_l856_85696

theorem cos_2alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = -Real.sqrt 5 / 3 := by
  sorry

end cos_2alpha_value_l856_85696


namespace largest_valid_n_l856_85693

def is_valid_n (n : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ), ∀ (k : ℕ), (2020 ∣ P^[k] 0) ↔ (n ∣ k)

theorem largest_valid_n : 
  (∃ (N : ℕ), N ∈ Finset.range 2020 ∧ is_valid_n N ∧ 
    ∀ (M : ℕ), M ∈ Finset.range 2020 → is_valid_n M → M ≤ N) ∧
  (∀ (N : ℕ), N ∈ Finset.range 2020 ∧ is_valid_n N ∧ 
    (∀ (M : ℕ), M ∈ Finset.range 2020 → is_valid_n M → M ≤ N) → N = 1980) :=
by sorry


end largest_valid_n_l856_85693


namespace stratified_sample_female_count_l856_85655

/-- Represents the number of female students in a stratified sample -/
def female_students_in_sample (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_female * sample_size) / (total_male + total_female)

/-- Theorem: In a stratified sampling by gender with 500 male students, 400 female students, 
    and a sample size of 45, the number of female students in the sample is 20 -/
theorem stratified_sample_female_count :
  female_students_in_sample 500 400 45 = 20 := by
  sorry

end stratified_sample_female_count_l856_85655


namespace factorization_problem_l856_85607

theorem factorization_problem (a b c : ℤ) : 
  (∀ x, x^2 + 7*x + 12 = (x + a) * (x + b)) →
  (∀ x, x^2 - 8*x - 20 = (x - b) * (x - c)) →
  a - b + c = -9 :=
by
  sorry

end factorization_problem_l856_85607


namespace parabola_vertex_l856_85659

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = -5 * (x + 2)^2 - 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, -6)

/-- Theorem: The vertex of the parabola y = -5(x+2)^2 - 6 is at the point (-2, -6) -/
theorem parabola_vertex :
  ∀ (x y : ℝ), parabola_equation x y → (x, y) = vertex :=
sorry

end parabola_vertex_l856_85659


namespace john_shirts_total_l856_85614

theorem john_shirts_total (initial_shirts : ℕ) (bought_shirts : ℕ) : 
  initial_shirts = 12 → bought_shirts = 4 → initial_shirts + bought_shirts = 16 :=
by sorry

end john_shirts_total_l856_85614


namespace unique_factorial_solution_l856_85603

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_factorial_solution :
  ∃! (n : ℕ), n > 0 ∧ factorial (n + 1) + factorial (n + 4) = factorial n * 3480 :=
sorry

end unique_factorial_solution_l856_85603


namespace keith_cd_player_cost_l856_85632

/-- The amount Keith spent on speakers -/
def speakers_cost : ℚ := 136.01

/-- The amount Keith spent on new tires -/
def tires_cost : ℚ := 112.46

/-- The total amount Keith spent -/
def total_cost : ℚ := 387.85

/-- The amount Keith spent on the CD player -/
def cd_player_cost : ℚ := total_cost - (speakers_cost + tires_cost)

theorem keith_cd_player_cost :
  cd_player_cost = 139.38 := by sorry

end keith_cd_player_cost_l856_85632


namespace diophantine_equation_solutions_l856_85641

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 7^x + 1 = 3^y + 5^z →
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end diophantine_equation_solutions_l856_85641


namespace range_of_m_for_sqrt_function_l856_85643

/-- Given a function f(x) = √(x² - 2x + 2m - 1) with domain ℝ, 
    prove that the range of m is [1, ∞) -/
theorem range_of_m_for_sqrt_function (m : ℝ) : 
  (∀ x, ∃ y, y = Real.sqrt (x^2 - 2*x + 2*m - 1)) → m ≥ 1 := by
sorry

end range_of_m_for_sqrt_function_l856_85643


namespace three_times_work_time_l856_85678

/-- Given a person can complete a piece of work in a certain number of days,
    this function calculates how many days it will take to complete a multiple of that work. -/
def time_for_multiple_work (days_for_single_work : ℕ) (work_multiple : ℕ) : ℕ :=
  days_for_single_work * work_multiple

/-- Theorem stating that if a person can complete a piece of work in 8 days,
    then they will complete three times the work in 24 days. -/
theorem three_times_work_time :
  time_for_multiple_work 8 3 = 24 := by
  sorry

end three_times_work_time_l856_85678


namespace digit_count_proof_l856_85698

/-- The number of valid digits for each position after the first -/
def valid_digits : ℕ := 4

/-- The number of valid digits for the first position -/
def valid_first_digits : ℕ := 3

/-- The total count of numbers with the given properties -/
def total_count : ℕ := 192

/-- The number of digits in the numbers -/
def n : ℕ := 4

theorem digit_count_proof :
  valid_first_digits * valid_digits^(n - 1) = total_count :=
sorry

end digit_count_proof_l856_85698


namespace baguettes_left_l856_85636

/-- The number of batches of baguettes made per day -/
def batches_per_day : ℕ := 3

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := 48

/-- The number of baguettes sold after the first batch -/
def sold_after_first : ℕ := 37

/-- The number of baguettes sold after the second batch -/
def sold_after_second : ℕ := 52

/-- The number of baguettes sold after the third batch -/
def sold_after_third : ℕ := 49

/-- Theorem stating that the number of baguettes left is 6 -/
theorem baguettes_left : 
  batches_per_day * baguettes_per_batch - (sold_after_first + sold_after_second + sold_after_third) = 6 := by
  sorry

end baguettes_left_l856_85636


namespace max_xyz_value_l856_85620

theorem max_xyz_value (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq_cond : (2*x * 2*y) + 3*z = (x + 2*z) * (y + 2*z))
  (sum_cond : x + y + z = 2) :
  x * y * z ≤ 8 / 27 := by
sorry

end max_xyz_value_l856_85620


namespace interchange_digits_sum_product_l856_85601

/-- Given a two-digit number n and a constant k, prove that if n is (k+1) times the sum of its digits,
    then the number formed by interchanging its digits is (10-k) times the sum of its digits. -/
theorem interchange_digits_sum_product (a b k : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : b ≤ 9) :
  (10 * a + b = (k + 1) * (a + b)) →
  (10 * b + a = (10 - k) * (a + b)) :=
by sorry

end interchange_digits_sum_product_l856_85601


namespace a_range_l856_85638

/-- The line passing through points (x, y) with parameter a -/
def line (x y a : ℝ) : ℝ := x + y - a

/-- Predicate for points being on opposite sides of the line -/
def opposite_sides (a : ℝ) : Prop :=
  (line 0 0 a) * (line 1 1 a) < 0

/-- Theorem stating the range of a given the conditions -/
theorem a_range : 
  (∀ a : ℝ, opposite_sides a ↔ 0 < a ∧ a < 2) :=
sorry

end a_range_l856_85638


namespace rectangular_field_perimeter_l856_85654

/-- Calculates the perimeter of a rectangular field given its width and length ratio. --/
def field_perimeter (width : ℝ) (length_ratio : ℝ) : ℝ :=
  2 * (width + length_ratio * width)

/-- Theorem: The perimeter of a rectangular field with width 50 meters and length 7/5 times its width is 240 meters. --/
theorem rectangular_field_perimeter :
  field_perimeter 50 (7/5) = 240 := by
  sorry

end rectangular_field_perimeter_l856_85654


namespace line_intersection_x_axis_l856_85660

/-- The line passing through points (2, 6) and (4, 10) intersects the x-axis at (-1, 0) -/
theorem line_intersection_x_axis :
  let p1 : ℝ × ℝ := (2, 6)
  let p2 : ℝ × ℝ := (4, 10)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  ∃ x : ℝ, line x = 0 ∧ x = -1 :=
sorry

end line_intersection_x_axis_l856_85660


namespace course_choice_related_probability_three_males_l856_85652

-- Define the total number of students
def total_students : ℕ := 200

-- Define the number of female students
def female_students : ℕ := 80

-- Define the number of female students majoring in the field
def female_major : ℕ := 70

-- Define the number of male students not majoring in the field
def male_non_major : ℕ := 40

-- Define the chi-square statistic threshold for 99.9% certainty
def chi_square_threshold : ℚ := 10828 / 1000

-- Define the function to calculate the chi-square statistic
def chi_square (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem for the relationship between course choice, major, and gender
theorem course_choice_related :
  let male_students : ℕ := total_students - female_students
  let male_major : ℕ := male_students - male_non_major
  let female_non_major : ℕ := female_students - female_major
  chi_square female_major male_major female_non_major male_non_major > chi_square_threshold := by sorry

-- Theorem for the probability of selecting 3 males out of 5 students
theorem probability_three_males :
  (Nat.choose 4 3 : ℚ) / (Nat.choose 5 3) = 2 / 5 := by sorry

end course_choice_related_probability_three_males_l856_85652


namespace batsman_average_increase_l856_85665

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (b : Batsman) (additionalRuns : ℕ) : ℚ :=
  (b.totalRuns + additionalRuns) / (b.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 110 runs in the 11th inning, 
    then his new average is 60 runs -/
theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 10) 
  (h2 : newAverage b 110 = b.average + 5) : 
  newAverage b 110 = 60 := by
  sorry

#check batsman_average_increase

end batsman_average_increase_l856_85665


namespace paint_theorem_l856_85686

def paint_problem (initial_amount : ℚ) : Prop :=
  let first_day_remaining := initial_amount - (1/2 * initial_amount)
  let second_day_remaining := first_day_remaining - (1/4 * first_day_remaining)
  let third_day_remaining := second_day_remaining - (1/3 * second_day_remaining)
  third_day_remaining = 1/4 * initial_amount

theorem paint_theorem : paint_problem 1 := by
  sorry

end paint_theorem_l856_85686


namespace negative_difference_l856_85657

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end negative_difference_l856_85657
