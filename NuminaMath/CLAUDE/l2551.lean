import Mathlib

namespace problem_statement_l2551_255113

theorem problem_statement (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end problem_statement_l2551_255113


namespace diplomats_speaking_both_languages_l2551_255106

theorem diplomats_speaking_both_languages (T F H : ℕ) (p : ℚ) : 
  T = 120 →
  F = 20 →
  T - H = 32 →
  p = 20 / 100 →
  (p * T : ℚ) = 24 →
  (F + H - (F + H - T : ℤ) : ℚ) / T * 100 = 10 :=
by sorry

end diplomats_speaking_both_languages_l2551_255106


namespace toms_restaurant_bill_l2551_255141

/-- The total bill for a group at Tom's Restaurant -/
def total_bill (adults children meal_cost : ℕ) : ℕ :=
  (adults + children) * meal_cost

/-- Theorem: The bill for 2 adults and 5 children with $8 meals is $56 -/
theorem toms_restaurant_bill : total_bill 2 5 8 = 56 := by
  sorry

end toms_restaurant_bill_l2551_255141


namespace ac_plus_bd_equals_negative_26_l2551_255154

theorem ac_plus_bd_equals_negative_26
  (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -26 := by
  sorry

end ac_plus_bd_equals_negative_26_l2551_255154


namespace complex_equality_sum_l2551_255110

theorem complex_equality_sum (a b : ℝ) : 
  (a + b * Complex.I : ℂ) = Complex.I ^ 2 → a + b = -1 := by
  sorry

end complex_equality_sum_l2551_255110


namespace triangular_plot_size_l2551_255180

/-- The size of a triangular plot of land in acres, given its dimensions on a map and conversion factors. -/
theorem triangular_plot_size (base height : ℝ) (scale_factor : ℝ) (acres_per_square_mile : ℝ) : 
  base = 8 → height = 12 → scale_factor = 1 → acres_per_square_mile = 320 →
  (1/2 * base * height) * scale_factor^2 * acres_per_square_mile = 15360 := by
  sorry

end triangular_plot_size_l2551_255180


namespace rolling_circle_traces_hypotrochoid_l2551_255123

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on a 2D plane -/
def Point := ℝ × ℝ

/-- Represents a hypotrochoid curve -/
def Hypotrochoid := Point → ℝ → Point

theorem rolling_circle_traces_hypotrochoid 
  (large_circle : Circle)
  (small_circle : Circle)
  (h1 : large_circle.radius = 2 * small_circle.radius)
  (h2 : small_circle.radius > 0)
  (point : Point) 
  (h3 : ∃ (θ : ℝ), point = 
    (small_circle.center.1 + small_circle.radius * Real.cos θ, 
     small_circle.center.2 + small_circle.radius * Real.sin θ))
  : ∃ (curve : Hypotrochoid), 
    ∀ (t : ℝ), curve point t = 
      ((large_circle.radius - small_circle.radius) * Real.cos t + small_circle.radius * Real.cos ((large_circle.radius / small_circle.radius - 1) * t),
       (large_circle.radius - small_circle.radius) * Real.sin t - small_circle.radius * Real.sin ((large_circle.radius / small_circle.radius - 1) * t)) :=
by sorry

end rolling_circle_traces_hypotrochoid_l2551_255123


namespace least_exponent_sum_for_1985_l2551_255183

def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sumOfDistinctPowersOfTwo (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.map (λ k => 2^k)).sum = n ∧ powers.Nodup

def exponentSum (powers : List ℕ) : ℕ := powers.sum

theorem least_exponent_sum_for_1985 :
  ∃ (powers : List ℕ),
    sumOfDistinctPowersOfTwo 1985 powers ∧
    ∀ (other_powers : List ℕ),
      sumOfDistinctPowersOfTwo 1985 other_powers →
      exponentSum powers ≤ exponentSum other_powers ∧
      exponentSum powers = 40 :=
sorry

end least_exponent_sum_for_1985_l2551_255183


namespace parabola_intersection_distance_ellipse_parabola_intersection_distance_l2551_255125

/-- The distance between intersection points of a parabola and a vertical line -/
theorem parabola_intersection_distance 
  (a : ℝ) -- Parameter of the parabola
  (x_intersect : ℝ) -- x-coordinate of the vertical line
  (h1 : a > 0) -- Ensure parabola opens to the right
  (h2 : x_intersect > 0) -- Ensure vertical line is to the right of y-axis
  : 
  let y1 := Real.sqrt (4 * a * x_intersect)
  let y2 := -Real.sqrt (4 * a * x_intersect)
  abs (y1 - y2) = 2 * Real.sqrt (4 * a * x_intersect) :=
by sorry

/-- The main theorem about the specific ellipse and parabola -/
theorem ellipse_parabola_intersection_distance :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / 16 = 1
  let parabola := fun (x y : ℝ) => y^2 = (100 / 3) * x
  let x_intersect := 25 / 3
  abs ((Real.sqrt ((100 / 3) * x_intersect)) - (-Real.sqrt ((100 / 3) * x_intersect))) = 100 / 3 :=
by sorry

end parabola_intersection_distance_ellipse_parabola_intersection_distance_l2551_255125


namespace quadratic_solution_sum_l2551_255167

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 4 * x + 15 = 0 ↔ x = Complex.mk a b ∨ x = Complex.mk a (-b)) → 
  a + b^2 = 162 / 50 := by
sorry

end quadratic_solution_sum_l2551_255167


namespace carlos_laundry_time_l2551_255102

/-- The time it takes for Carlos to do his laundry -/
def laundry_time (num_loads : ℕ) (wash_time_per_load : ℕ) (dry_time : ℕ) : ℕ :=
  num_loads * wash_time_per_load + dry_time

/-- Theorem: Carlos's laundry takes 165 minutes -/
theorem carlos_laundry_time :
  laundry_time 2 45 75 = 165 := by
  sorry

end carlos_laundry_time_l2551_255102


namespace corrected_mean_l2551_255116

/-- Given a set of observations, calculate the corrected mean after fixing an error in one observation -/
theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n > 0 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  (corrected_sum / n) = 36.14 →
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 30 :=
by sorry

end corrected_mean_l2551_255116


namespace max_guarding_value_l2551_255173

/-- Represents the four possible directions a guard can look --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a position on the 8x8 board --/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a guard on the board --/
structure Guard :=
  (pos : Position)
  (dir : Direction)

/-- The type of a valid board configuration --/
def BoardConfiguration := Fin 8 → Fin 8 → Guard

/-- Checks if a guard at position (row, col) is guarded by another guard --/
def isGuardedBy (board : BoardConfiguration) (row col : Fin 8) (otherRow otherCol : Fin 8) : Prop :=
  sorry

/-- Counts the number of guards watching a specific position --/
def countGuardingGuards (board : BoardConfiguration) (row col : Fin 8) : Nat :=
  sorry

/-- Checks if all guards are guarded by at least k other guards --/
def allGuardsGuardedByAtLeastK (board : BoardConfiguration) (k : Nat) : Prop :=
  ∀ row col, countGuardingGuards board row col ≥ k

/-- The main theorem stating that 5 is the maximum value of k --/
theorem max_guarding_value :
  (∃ board : BoardConfiguration, allGuardsGuardedByAtLeastK board 5) ∧
  (¬∃ board : BoardConfiguration, allGuardsGuardedByAtLeastK board 6) :=
sorry

end max_guarding_value_l2551_255173


namespace factorization_example_l2551_255168

theorem factorization_example (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end factorization_example_l2551_255168


namespace divisibility_by_eleven_l2551_255182

theorem divisibility_by_eleven (a b : ℤ) : 
  (11 ∣ a^2 + b^2) → (11 ∣ a) ∧ (11 ∣ b) := by
  sorry

end divisibility_by_eleven_l2551_255182


namespace current_rate_calculation_l2551_255111

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 15 →
  distance = 3.6 →
  time = 1/5 →
  ∃ (current_rate : ℝ), current_rate = 3 ∧ 
    distance = (boat_speed + current_rate) * time :=
by sorry

end current_rate_calculation_l2551_255111


namespace five_sixths_of_twelve_fifths_l2551_255181

theorem five_sixths_of_twelve_fifths : (5 / 6 : ℚ) * (12 / 5 : ℚ) = 2 := by
  sorry

end five_sixths_of_twelve_fifths_l2551_255181


namespace one_pattern_cannot_fold_to_pyramid_l2551_255130

/-- Represents a pattern of identical squares -/
structure Pattern :=
  (squares : ℕ)
  (foldable : Bool)

/-- Represents a pyramid with a square base -/
structure Pyramid :=
  (base : ℕ)
  (sides : ℕ)

/-- Function to check if a pattern can be folded into a pyramid -/
def can_fold_to_pyramid (p : Pattern) (pyr : Pyramid) : Prop :=
  p.squares = pyr.base + pyr.sides ∧ p.foldable

/-- Theorem stating that exactly one pattern cannot be folded into a pyramid -/
theorem one_pattern_cannot_fold_to_pyramid 
  (A B C D : Pattern) 
  (pyr : Pyramid) 
  (h_pyr : pyr.base = 1 ∧ pyr.sides = 4) 
  (h_ABC : can_fold_to_pyramid A pyr ∧ can_fold_to_pyramid B pyr ∧ can_fold_to_pyramid C pyr) 
  (h_D : ¬can_fold_to_pyramid D pyr) : 
  ∃! p : Pattern, ¬can_fold_to_pyramid p pyr :=
sorry

end one_pattern_cannot_fold_to_pyramid_l2551_255130


namespace x_plus_y_value_l2551_255108

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : x > y) :
  x + y = 5 ∨ x + y = 1 := by
sorry

end x_plus_y_value_l2551_255108


namespace complex_modulus_problem_l2551_255190

theorem complex_modulus_problem (z : ℂ) (h : z = (2 + Complex.I) / Complex.I + Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2551_255190


namespace plot_length_l2551_255122

/-- Proves that the length of a rectangular plot is 55 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  length = breadth + 10 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 55 := by
  sorry

end plot_length_l2551_255122


namespace matrix_not_invertible_iff_l2551_255184

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2 + 2*x, 5],
    ![4 - 2*x, 9]]

theorem matrix_not_invertible_iff (x : ℚ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 1/14 := by
  sorry

end matrix_not_invertible_iff_l2551_255184


namespace arithmetic_sequence_sum_l2551_255176

/-- The sum of an arithmetic sequence with first term 1, common difference 2, and last term 17 -/
def arithmetic_sum : ℕ := 81

/-- The first term of the sequence -/
def a₁ : ℕ := 1

/-- The common difference of the sequence -/
def d : ℕ := 2

/-- The last term of the sequence -/
def aₙ : ℕ := 17

/-- The number of terms in the sequence -/
def n : ℕ := (aₙ - a₁) / d + 1

theorem arithmetic_sequence_sum :
  (n : ℕ) * (a₁ + aₙ) / 2 = arithmetic_sum :=
sorry

end arithmetic_sequence_sum_l2551_255176


namespace erins_launderette_machines_l2551_255133

/-- Represents the number of coins in a machine --/
structure CoinCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value of coins in dollars --/
def coinValue (c : CoinCount) : Rat :=
  (c.quarters * 25 + c.dimes * 10 + c.nickels * 5 + c.pennies) / 100

/-- Represents the launderette problem --/
structure LaunderetteProblem where
  machineCoins : CoinCount
  totalCashed : Rat
  minMachines : Nat
  maxMachines : Nat

/-- The specific launderette problem instance --/
def erinsProblem : LaunderetteProblem :=
  { machineCoins := { quarters := 80, dimes := 100, nickels := 50, pennies := 120 }
    totalCashed := 165
    minMachines := 3
    maxMachines := 5 }

theorem erins_launderette_machines (p : LaunderetteProblem) (h : p = erinsProblem) :
    ∃ n : Nat, n ≥ p.minMachines ∧ n ≤ p.maxMachines ∧ 
    n * coinValue p.machineCoins = p.totalCashed := by sorry

end erins_launderette_machines_l2551_255133


namespace complex_multiplication_subtraction_l2551_255136

theorem complex_multiplication_subtraction : ∃ (i : ℂ), i^2 = -1 ∧ (4 - 3*i) * (2 + 5*i) - (6 - 2*i) = 17 + 16*i := by
  sorry

end complex_multiplication_subtraction_l2551_255136


namespace dartboard_angle_l2551_255115

theorem dartboard_angle (P : ℝ) (θ : ℝ) : 
  P = 1/8 → θ = P * 360 → θ = 45 := by
  sorry

end dartboard_angle_l2551_255115


namespace car_speed_problem_l2551_255196

theorem car_speed_problem (total_time : ℝ) (initial_time : ℝ) (initial_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_time = 24)
  (h2 : initial_time = 4)
  (h3 : initial_speed = 35)
  (h4 : average_speed = 50) :
  let remaining_time := total_time - initial_time
  let total_distance := average_speed * total_time
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 53 := by
sorry

end car_speed_problem_l2551_255196


namespace angles_on_x_axis_characterization_l2551_255121

/-- The set of angles with terminal sides on the x-axis -/
def AnglesOnXAxis : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi}

/-- Theorem: The set of angles with terminal sides on the x-axis is equal to {α | α = kπ, k ∈ ℤ} -/
theorem angles_on_x_axis_characterization :
  AnglesOnXAxis = {α : ℝ | ∃ k : ℤ, α = k * Real.pi} := by
  sorry

end angles_on_x_axis_characterization_l2551_255121


namespace cars_meet_time_l2551_255101

/-- Two cars meet on a highway -/
theorem cars_meet_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 500) 
  (h2 : speed1 = 40) (h3 : speed2 = 60) : 
  (highway_length / (speed1 + speed2) = 5) := by
sorry

end cars_meet_time_l2551_255101


namespace modular_inverse_11_mod_101_l2551_255128

theorem modular_inverse_11_mod_101 :
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 100 ∧ (11 * x) % 101 = 1 :=
by
  use 46
  sorry

end modular_inverse_11_mod_101_l2551_255128


namespace smallest_solution_congruence_system_l2551_255135

theorem smallest_solution_congruence_system :
  ∃ (x : ℕ), x > 0 ∧ 
    (6 * x) % 31 = 17 % 31 ∧
    x % 7 = 3 % 7 ∧
    (∀ (y : ℕ), y > 0 ∧ (6 * y) % 31 = 17 % 31 ∧ y % 7 = 3 % 7 → x ≤ y) ∧
    x = 24 := by
  sorry

end smallest_solution_congruence_system_l2551_255135


namespace fahrenheit_celsius_conversion_l2551_255153

theorem fahrenheit_celsius_conversion (F C : ℝ) : 
  C = (5 / 9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end fahrenheit_celsius_conversion_l2551_255153


namespace set_equality_l2551_255119

open Set

def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x ≤ 3}

theorem set_equality : (Mᶜ ∩ (M ∩ N)) = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end set_equality_l2551_255119


namespace car_speed_proof_l2551_255185

/-- The speed of a car in km/h -/
def car_speed : ℝ := 30

/-- The reference speed in km/h -/
def reference_speed : ℝ := 36

/-- The additional time taken in seconds -/
def additional_time : ℝ := 20

/-- The distance traveled in km -/
def distance : ℝ := 1

theorem car_speed_proof :
  car_speed = 30 ∧
  (distance / car_speed) * 3600 = (distance / reference_speed) * 3600 + additional_time :=
sorry

end car_speed_proof_l2551_255185


namespace jessa_cupcakes_l2551_255174

/-- The number of cupcakes Jessa needs to make -/
def total_cupcakes : ℕ := sorry

/-- The number of fourth-grade classes -/
def fourth_grade_classes : ℕ := 12

/-- The number of students in each fourth-grade class -/
def students_per_fourth_grade : ℕ := 45

/-- The number of P.E. classes -/
def pe_classes : ℕ := 2

/-- The number of students in each P.E. class -/
def students_per_pe : ℕ := 90

/-- The number of afterschool clubs -/
def afterschool_clubs : ℕ := 4

/-- The number of students in each afterschool club -/
def students_per_afterschool : ℕ := 60

/-- Theorem stating that the total number of cupcakes Jessa needs to make is 960 -/
theorem jessa_cupcakes : total_cupcakes = 960 := by sorry

end jessa_cupcakes_l2551_255174


namespace power_of_2_ending_probabilities_l2551_255178

/-- The probability that 2^n ends with the digit 2, where n is a randomly chosen positive integer -/
def prob_ends_with_2 : ℚ := 1 / 4

/-- The probability that 2^n ends with the digits 12, where n is a randomly chosen positive integer -/
def prob_ends_with_12 : ℚ := 1 / 20

/-- Theorem stating the probabilities for 2^n ending with 2 and 12 -/
theorem power_of_2_ending_probabilities :
  (prob_ends_with_2 = 1 / 4) ∧ (prob_ends_with_12 = 1 / 20) := by
  sorry

end power_of_2_ending_probabilities_l2551_255178


namespace circle_through_points_tangent_line_through_D_tangent_touches_circle_l2551_255191

-- Define the points
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 4)
def D : ℝ × ℝ := (-1, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 5

-- Define the tangent line equation
def tangent_line_equation (x y : ℝ) : Prop :=
  2*x + y = 0

-- Theorem for the circle equation
theorem circle_through_points :
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  circle_equation C.1 C.2 := by sorry

-- Theorem for the tangent line
theorem tangent_line_through_D :
  tangent_line_equation D.1 D.2 := by sorry

-- Theorem that the tangent line touches the circle at exactly one point
theorem tangent_touches_circle :
  ∃! (x y : ℝ), circle_equation x y ∧ tangent_line_equation x y := by sorry

end circle_through_points_tangent_line_through_D_tangent_touches_circle_l2551_255191


namespace fifth_term_of_specific_geometric_progression_l2551_255117

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fifth_term_of_specific_geometric_progression :
  let a := 2 ^ (1/4 : ℝ)
  let r := 2 ^ (1/4 : ℝ)
  geometric_progression a r 1 = 2 ^ (1/4 : ℝ) ∧
  geometric_progression a r 2 = 2 ^ (1/2 : ℝ) ∧
  geometric_progression a r 3 = 2 ^ (3/4 : ℝ) →
  geometric_progression a r 5 = 2 ^ (5/4 : ℝ) := by
  sorry

end fifth_term_of_specific_geometric_progression_l2551_255117


namespace jeff_donuts_days_l2551_255175

/-- The number of donuts Jeff makes per day -/
def donuts_per_day : ℕ := 10

/-- The number of donuts Jeff eats per day -/
def jeff_eats_per_day : ℕ := 1

/-- The total number of donuts Chris eats -/
def chris_eats_total : ℕ := 8

/-- The number of donuts that fit in each box -/
def donuts_per_box : ℕ := 10

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled : ℕ := 10

/-- The number of days Jeff makes donuts -/
def days_making_donuts : ℕ := 12

theorem jeff_donuts_days :
  days_making_donuts * (donuts_per_day - jeff_eats_per_day) - chris_eats_total =
  boxes_filled * donuts_per_box :=
by
  sorry

end jeff_donuts_days_l2551_255175


namespace equation_solutions_l2551_255124

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, (2*y₁^2 - 2*y₁ = 1 ∧ 2*y₂^2 - 2*y₂ = 1) ∧ 
    y₁ = (1 + Real.sqrt 3) / 2 ∧ y₂ = (1 - Real.sqrt 3) / 2) := by
  sorry

end equation_solutions_l2551_255124


namespace list_number_fraction_l2551_255132

theorem list_number_fraction (S : ℝ) (n : ℝ) :
  n = 7 * (S / 50) →
  n / (S + n) = 7 / 57 := by
  sorry

end list_number_fraction_l2551_255132


namespace john_bought_three_shirts_l2551_255103

/-- The number of dress shirts John bought -/
def num_shirts : ℕ := 3

/-- The cost of each dress shirt in dollars -/
def shirt_cost : ℚ := 20

/-- The tax rate as a percentage -/
def tax_rate : ℚ := 10

/-- The total amount John paid in dollars -/
def total_paid : ℚ := 66

/-- Theorem stating that the number of shirts John bought is correct -/
theorem john_bought_three_shirts :
  (shirt_cost * num_shirts) * (1 + tax_rate / 100) = total_paid := by
  sorry

end john_bought_three_shirts_l2551_255103


namespace group_size_l2551_255157

/-- The number of people in the group -/
def n : ℕ := sorry

/-- The total weight of the group before the change -/
def W : ℝ := sorry

/-- The weight increase when the new person joins -/
def weight_increase : ℝ := 2.5

/-- The weight of the person being replaced -/
def old_weight : ℝ := 55

/-- The weight of the new person -/
def new_weight : ℝ := 75

theorem group_size :
  (W + new_weight - old_weight) / n = W / n + weight_increase →
  n = 8 := by sorry

end group_size_l2551_255157


namespace total_water_poured_l2551_255171

/-- 
Given two bottles with capacities of 4 and 8 cups respectively, 
if they are filled to the same fraction of their capacity and 
5.333333333333333 cups of water are poured into the 8-cup bottle, 
then the total amount of water poured into both bottles is 8 cups.
-/
theorem total_water_poured (bottle1_capacity bottle2_capacity : ℝ) 
  (water_in_bottle2 : ℝ) : 
  bottle1_capacity = 4 →
  bottle2_capacity = 8 →
  water_in_bottle2 = 5.333333333333333 →
  (water_in_bottle2 / bottle2_capacity) * bottle1_capacity + water_in_bottle2 = 8 := by
  sorry

end total_water_poured_l2551_255171


namespace vertical_translation_equation_translated_line_equation_l2551_255156

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateVertically (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + d }

theorem vertical_translation_equation (l : Line) (d : ℝ) :
  (translateVertically l d).slope = l.slope ∧
  (translateVertically l d).intercept = l.intercept + d := by
  sorry

/-- The original line y = -2x + 1 -/
def originalLine : Line :=
  { slope := -2, intercept := 1 }

/-- The translation distance -/
def translationDistance : ℝ := 2

theorem translated_line_equation :
  translateVertically originalLine translationDistance =
  { slope := -2, intercept := 3 } := by
  sorry

end vertical_translation_equation_translated_line_equation_l2551_255156


namespace expand_expression_l2551_255160

theorem expand_expression (y : ℝ) : 5 * (y - 2) * (y + 7) = 5 * y^2 + 25 * y - 70 := by
  sorry

end expand_expression_l2551_255160


namespace sufficient_not_necessary_condition_l2551_255192

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x ≥ 1) ∧ (∃ x, x ≥ 1 ∧ ¬(x > 1)) :=
by sorry

end sufficient_not_necessary_condition_l2551_255192


namespace greatest_b_value_l2551_255146

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end greatest_b_value_l2551_255146


namespace alex_coin_distribution_l2551_255140

/-- The minimum number of additional coins needed for unique distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum additional coins needed for Alex's distribution -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 94) :
  min_additional_coins friends initial_coins = 26 := by
  sorry

#eval min_additional_coins 15 94

end alex_coin_distribution_l2551_255140


namespace six_balls_four_boxes_l2551_255138

/-- Number of ways to distribute indistinguishable balls among distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls among 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 84 := by sorry

end six_balls_four_boxes_l2551_255138


namespace necessary_but_not_sufficient_l2551_255142

-- Define the condition |a-1| + |a| ≤ 1
def condition (a : ℝ) : Prop := abs (a - 1) + abs a ≤ 1

-- Define the property that y = a^x is decreasing on ℝ
def is_decreasing (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x > a^y

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_decreasing a → condition a) ∧
  (∃ a : ℝ, condition a ∧ ¬is_decreasing a) :=
sorry

end necessary_but_not_sufficient_l2551_255142


namespace claire_cakes_l2551_255150

/-- The number of cakes Claire can make -/
def num_cakes (packages_per_cake : ℕ) (price_per_package : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / price_per_package) / packages_per_cake

theorem claire_cakes : num_cakes 2 3 12 = 2 := by
  sorry

end claire_cakes_l2551_255150


namespace angle_in_first_or_third_quadrant_l2551_255152

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

/-- An angle is in the third quadrant if it's between 180° and 270° -/
def is_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

/-- The main theorem: for any integer k, the angle k·180° + 45° is in either the first or third quadrant -/
theorem angle_in_first_or_third_quadrant (k : ℤ) :
  let α := k * 180 + 45
  is_first_quadrant (α % 360) ∨ is_third_quadrant (α % 360) :=
sorry

end angle_in_first_or_third_quadrant_l2551_255152


namespace wall_length_calculation_l2551_255198

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  width : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.width

theorem wall_length_calculation
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (num_bricks : ℕ)
  (h1 : brick.length = 80)
  (h2 : brick.width = 11.25)
  (h3 : brick.height = 6)
  (h4 : wall.height = 600)
  (h5 : wall.width = 22.5)
  (h6 : num_bricks = 2000)
  (h7 : num_bricks * brickVolume brick = wallVolume wall) :
  wall.length = 800 := by
  sorry

#check wall_length_calculation

end wall_length_calculation_l2551_255198


namespace cubic_equation_ratio_l2551_255143

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with roots 1, 2, and 3,
    prove that c/d = -11/6 -/
theorem cubic_equation_ratio (a b c d : ℝ) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) →
  c / d = -11 / 6 := by
  sorry

end cubic_equation_ratio_l2551_255143


namespace equation_equivalence_implies_uvw_product_l2551_255149

theorem equation_equivalence_implies_uvw_product (a b x y : ℝ) (u v w : ℤ) :
  (a^10 * x * y - a^9 * y - a^8 * x = a^6 * (b^5 - 1)) →
  ((a^u * x - a^v) * (a^w * y - a^3) = a^6 * b^5) →
  u * v * w = 48 := by
  sorry

end equation_equivalence_implies_uvw_product_l2551_255149


namespace horner_method_properties_l2551_255158

def horner_polynomial (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

def horner_v2 (x : ℝ) : ℝ := (4 * x + 2) * x + 3.5

theorem horner_method_properties :
  let x : ℝ := 5
  (∃ (max_multiplications : ℕ), max_multiplications = 5 ∧
    ∀ (other_multiplications : ℕ),
      other_multiplications ≤ max_multiplications) ∧
  horner_v2 x = 113.5 := by
  sorry

end horner_method_properties_l2551_255158


namespace point_b_value_l2551_255118

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_value (a b : Point) (h1 : a.value = -2) (h2 : distance a b = 4) : b.value = 2 := by
  sorry

end point_b_value_l2551_255118


namespace inequality_equivalence_l2551_255199

theorem inequality_equivalence (x : ℝ) : (x - 1) / (x - 3) ≥ 2 ↔ x ∈ Set.Ioo 3 5 ∪ {5} := by
  sorry

end inequality_equivalence_l2551_255199


namespace quadratic_roots_nature_l2551_255107

theorem quadratic_roots_nature (x : ℝ) : 
  (x^2 - 6*x + 9 = 0) → (∃ r : ℝ, x = r ∧ x^2 - 6*x + 9 = 0) ∧ 
  (∃! r : ℝ, x^2 - 6*x + 9 = 0) := by
  sorry

end quadratic_roots_nature_l2551_255107


namespace product_of_roots_l2551_255186

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = 50) := by
sorry

end product_of_roots_l2551_255186


namespace largest_b_value_l2551_255170

theorem largest_b_value (a b : ℤ) (h1 : 29 < a ∧ a < 41) (h2 : 39 < b) 
  (h3 : (a : ℚ) / b - (30 : ℚ) / b = 0.4) : b ≤ 75 :=
sorry

end largest_b_value_l2551_255170


namespace find_f_2_l2551_255193

-- Define the real number a
variable (a : ℝ)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem find_f_2 
  (h_a_pos : a > 0)
  (h_a_neq_1 : a ≠ 1)
  (h_f_odd : ∀ x, f (-x) = -f x)
  (h_g_even : ∀ x, g (-x) = g x)
  (h_sum : ∀ x, f x + g x = a^x - a^(-x) + 2)
  (h_g_2 : g 2 = a) :
  f 2 = 15/4 := by
sorry

end find_f_2_l2551_255193


namespace polynomial_has_three_distinct_integer_roots_l2551_255147

def polynomial (x : ℤ) : ℤ := x^5 + 3*x^4 - 4044118*x^3 - 12132362*x^2 - 12132363*x - 2011^2

theorem polynomial_has_three_distinct_integer_roots :
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, polynomial x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end polynomial_has_three_distinct_integer_roots_l2551_255147


namespace geometric_sequence_common_ratio_l2551_255197

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

-- Theorem statement
theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (r : ℝ) (h : ∀ n : ℕ, n > 0 → 
    geometric_sequence a₁ r n * geometric_sequence a₁ r (n + 1) = 16 ^ n) : 
  r = 4 := by
sorry

end geometric_sequence_common_ratio_l2551_255197


namespace paper_area_proof_l2551_255188

/-- The side length of each square piece of paper in centimeters -/
def side_length : ℝ := 8.5

/-- The number of pieces of square paper -/
def num_pieces : ℝ := 3.2

/-- The total area when gluing the pieces together without any gap -/
def total_area : ℝ := 231.2

/-- Theorem stating that the total area of the glued pieces is 231.2 cm² -/
theorem paper_area_proof : 
  side_length * side_length * num_pieces = total_area := by
  sorry

end paper_area_proof_l2551_255188


namespace remainder_seven_n_l2551_255109

theorem remainder_seven_n (n : ℤ) (h : n % 5 = 3) : (7 * n) % 5 = 1 := by
  sorry

end remainder_seven_n_l2551_255109


namespace linear_function_slope_condition_l2551_255166

/-- Given two points on a linear function, if x-coordinate increases while y-coordinate decreases, then the slope is less than 2 -/
theorem linear_function_slope_condition (a x₁ y₁ x₂ y₂ : ℝ) : 
  y₁ = (a - 2) * x₁ + 1 →   -- Point A lies on the graph
  y₂ = (a - 2) * x₂ + 1 →   -- Point B lies on the graph
  (x₁ > x₂ → y₁ < y₂) →     -- When x₁ > x₂, y₁ < y₂
  a < 2 := by
sorry

end linear_function_slope_condition_l2551_255166


namespace max_triangles_three_families_l2551_255134

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (f1 f2 f3 : LineFamily) : ℕ :=
  150

/-- Theorem stating that three families of 10 parallel lines form at most 150 triangles -/
theorem max_triangles_three_families :
  ∀ (f1 f2 f3 : LineFamily),
    f1.count = 10 → f2.count = 10 → f3.count = 10 →
    max_triangles f1 f2 f3 = 150 :=
by
  sorry

#check max_triangles_three_families

end max_triangles_three_families_l2551_255134


namespace equation_represents_point_l2551_255112

/-- The equation represents a point in the xy-plane -/
theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2*a*x + 2*b*y + a^2 + b^2 = 0 ↔ (x = -a ∧ y = -b) :=
sorry

end equation_represents_point_l2551_255112


namespace marshmallow_ratio_l2551_255172

theorem marshmallow_ratio (joe_marshmallows : ℕ) (dad_marshmallows : ℕ) : 
  dad_marshmallows = 21 →
  (joe_marshmallows / 2 + dad_marshmallows / 3 = 49) →
  (joe_marshmallows : ℚ) / dad_marshmallows = 4 := by
sorry

end marshmallow_ratio_l2551_255172


namespace area_two_sectors_l2551_255189

/-- The area of a figure formed by two 45° sectors of a circle with radius 15 -/
theorem area_two_sectors (r : ℝ) (θ : ℝ) (h1 : r = 15) (h2 : θ = 45 * π / 180) :
  2 * (θ / (2 * π)) * π * r^2 = 56.25 * π :=
sorry

end area_two_sectors_l2551_255189


namespace initial_capacity_proof_l2551_255129

/-- The initial capacity of a barrel in liters -/
def initial_capacity : ℝ := 220

/-- The percentage of contents remaining after the leak -/
def remaining_percentage : ℝ := 0.9

/-- The amount of liquid remaining in the barrel after the leak, in liters -/
def remaining_liquid : ℝ := 198

/-- Theorem stating that the initial capacity is correct given the conditions -/
theorem initial_capacity_proof : 
  initial_capacity * remaining_percentage = remaining_liquid :=
by sorry

end initial_capacity_proof_l2551_255129


namespace grid_coloring_probability_l2551_255187

/-- The number of squares in a row or column of the grid -/
def gridSize : ℕ := 4

/-- The total number of possible colorings for the grid -/
def totalColorings : ℕ := 2^(gridSize^2)

/-- The number of colorings with at least one 3-by-3 yellow square -/
def coloringsWithYellowSquare : ℕ := 510

/-- The probability of obtaining a grid without a 3-by-3 yellow square -/
def probabilityNoYellowSquare : ℚ := (totalColorings - coloringsWithYellowSquare) / totalColorings

theorem grid_coloring_probability :
  probabilityNoYellowSquare = 65026 / 65536 :=
sorry

end grid_coloring_probability_l2551_255187


namespace cos_20_cos_10_minus_sin_160_sin_10_l2551_255148

theorem cos_20_cos_10_minus_sin_160_sin_10 :
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_20_cos_10_minus_sin_160_sin_10_l2551_255148


namespace paint_mixture_weight_l2551_255161

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue_percent : ℝ
  red_percent : ℝ
  yellow_percent : ℝ
  weight : ℝ

/-- The problem setup for the paint mixture calculation -/
def paint_problem : Prop :=
  ∃ (sky_blue green brown : PaintMixture),
    -- Sky blue paint composition
    sky_blue.blue_percent = 0.1 ∧
    sky_blue.red_percent = 0.9 ∧
    sky_blue.yellow_percent = 0 ∧
    -- Green paint composition
    green.blue_percent = 0.7 ∧
    green.red_percent = 0 ∧
    green.yellow_percent = 0.3 ∧
    -- Brown paint composition
    brown.blue_percent = 0.4 ∧
    -- Red pigment weight in brown paint
    brown.red_percent * brown.weight = 4.5 ∧
    -- Total weight of brown paint
    brown.weight = sky_blue.weight + green.weight ∧
    -- Blue pigment balance
    sky_blue.blue_percent * sky_blue.weight + green.blue_percent * green.weight = 
      brown.blue_percent * brown.weight ∧
    -- Total weight of brown paint is 10 grams
    brown.weight = 10

/-- The main theorem stating that the paint problem implies a 10-gram brown paint -/
theorem paint_mixture_weight : paint_problem → ∃ (brown : PaintMixture), brown.weight = 10 := by
  sorry


end paint_mixture_weight_l2551_255161


namespace mr_a_loss_l2551_255137

/-- Calculates the total loss for Mr. A in a house transaction --/
def calculate_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale_price := initial_value * (1 - loss_percent)
  let second_sale_price := first_sale_price * (1 + gain_percent)
  second_sale_price - initial_value

/-- Theorem stating that Mr. A loses $2040 in the house transaction --/
theorem mr_a_loss :
  calculate_loss 12000 0.15 0.20 = 2040 := by sorry

end mr_a_loss_l2551_255137


namespace periodic_function_theorem_l2551_255151

/-- A function f is periodic with period b if f(x + b) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (x + b) = f x

/-- The functional equation property for f -/
def HasFunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)

theorem periodic_function_theorem (f : ℝ → ℝ) (a : ℝ) (ha : a > 0) 
    (h : HasFunctionalEquation f a) : 
    IsPeriodic f (2 * a) := by
  sorry

end periodic_function_theorem_l2551_255151


namespace total_mangoes_l2551_255179

theorem total_mangoes (alexis_mangoes : ℕ) (dilan_ashley_mangoes : ℕ) : 
  alexis_mangoes = 60 →
  alexis_mangoes = 4 * dilan_ashley_mangoes →
  alexis_mangoes + dilan_ashley_mangoes = 75 :=
by
  sorry

end total_mangoes_l2551_255179


namespace pauls_pencil_stock_l2551_255169

/-- Calculates the number of pencils in stock at the end of the week -/
def pencils_in_stock_end_of_week (
  daily_production : ℕ)
  (working_days : ℕ)
  (initial_stock : ℕ)
  (sold_pencils : ℕ) : ℕ :=
  daily_production * working_days + initial_stock - sold_pencils

/-- Proves that Paul has 230 pencils in stock at the end of the week -/
theorem pauls_pencil_stock : 
  pencils_in_stock_end_of_week 100 5 80 350 = 230 := by
  sorry

end pauls_pencil_stock_l2551_255169


namespace maya_has_largest_result_l2551_255194

def start_number : ℕ := 15

def sara_result : ℕ := (start_number ^ 2 - 3) + 4

def liam_result : ℕ := ((start_number - 2) ^ 2) + 4

def maya_result : ℕ := (start_number - 3 + 4) ^ 2

theorem maya_has_largest_result :
  maya_result > sara_result ∧ maya_result > liam_result := by
  sorry

end maya_has_largest_result_l2551_255194


namespace tim_interest_rate_l2551_255155

/-- Tim's investment amount -/
def tim_investment : ℝ := 500

/-- Lana's investment amount -/
def lana_investment : ℝ := 1000

/-- Lana's annual interest rate -/
def lana_rate : ℝ := 0.05

/-- Number of years -/
def years : ℕ := 2

/-- Interest difference between Tim and Lana after 2 years -/
def interest_difference : ℝ := 2.5

/-- Calculate the compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Tim's annual interest rate -/
def tim_rate : ℝ := 0.1

theorem tim_interest_rate :
  compound_interest tim_investment tim_rate years =
  compound_interest lana_investment lana_rate years + interest_difference := by
  sorry

#check tim_interest_rate

end tim_interest_rate_l2551_255155


namespace unique_solution_for_floor_equation_l2551_255164

theorem unique_solution_for_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 ∧ n = 5 := by
  sorry

end unique_solution_for_floor_equation_l2551_255164


namespace beth_sheep_count_l2551_255114

theorem beth_sheep_count (aaron_sheep : ℕ) (beth_sheep : ℕ) 
  (h1 : aaron_sheep = 7 * beth_sheep) 
  (h2 : aaron_sheep + beth_sheep = 608) : 
  beth_sheep = 76 := by
sorry

end beth_sheep_count_l2551_255114


namespace dubblefud_product_l2551_255139

/-- Represents the number of points for each chip color in the game of Dubblefud -/
structure ChipPoints where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- Represents the number of chips for each color in a selection -/
structure ChipSelection where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The theorem statement for the Dubblefud game problem -/
theorem dubblefud_product (points : ChipPoints) (selection : ChipSelection) :
  points.yellow = 2 →
  points.blue = 4 →
  points.green = 5 →
  selection.blue = selection.green →
  selection.yellow = 4 →
  (points.yellow * selection.yellow) *
  (points.blue * selection.blue) *
  (points.green * selection.green) =
  72 * selection.blue :=
by sorry

end dubblefud_product_l2551_255139


namespace a_range_theorem_l2551_255165

/-- Sequence a_n defined as n^2 - 2an for n ∈ ℕ+ -/
def a_n (a : ℝ) (n : ℕ+) : ℝ := n.val^2 - 2*a*n.val

/-- Proposition: Given a_n = n^2 - 2an for n ∈ ℕ+, and a_n > a_4 for all n ≠ 4,
    the range of values for a is (7/2, 9/2) -/
theorem a_range_theorem (a : ℝ) : 
  (∀ (n : ℕ+), n ≠ 4 → a_n a n > a_n a 4) ↔ 
  (7/2 < a ∧ a < 9/2) :=
sorry

end a_range_theorem_l2551_255165


namespace m_greater_than_n_l2551_255105

theorem m_greater_than_n (a b : ℝ) (h1 : 0 < a) (h2 : a < 1/b) : 
  (1/(1+a) + 1/(1+b)) > (a/(1+a) + b/(1+b)) := by
sorry

end m_greater_than_n_l2551_255105


namespace rectangular_plot_breadth_l2551_255159

/-- A rectangular plot with length thrice its breadth and area 363 sq m has a breadth of 11 m -/
theorem rectangular_plot_breadth : 
  ∀ (breadth : ℝ),
  breadth > 0 →
  3 * breadth * breadth = 363 →
  breadth = 11 := by
sorry

end rectangular_plot_breadth_l2551_255159


namespace salary_remaining_l2551_255145

def salary : ℕ := 180000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_money : ℕ := 18000

theorem salary_remaining :
  salary - (↑salary * (food_fraction + rent_fraction + clothes_fraction)).floor = remaining_money := by
  sorry

end salary_remaining_l2551_255145


namespace equation_solution_l2551_255131

theorem equation_solution (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end equation_solution_l2551_255131


namespace find_b_squared_l2551_255127

/-- A complex function satisfying certain properties -/
def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * Complex.I) * z

/-- The main theorem -/
theorem find_b_squared (a b : ℝ) :
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z)) →
  a = 2 →
  Complex.abs (a + b * Complex.I) = 10 →
  b^2 = 99 := by
  sorry

end find_b_squared_l2551_255127


namespace product_of_five_consecutive_not_square_l2551_255163

theorem product_of_five_consecutive_not_square (n : ℕ) : 
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ≠ k^2 := by
  sorry

end product_of_five_consecutive_not_square_l2551_255163


namespace smallest_m_for_integral_solutions_l2551_255120

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), 
  (∀ (k : ℕ), k < m → ¬∃ (x y : ℤ), 15 * x^2 - k * x + 315 = 0 ∧ 15 * y^2 - k * y + 315 = 0 ∧ x ≠ y) ∧
  (∃ (x y : ℤ), 15 * x^2 - m * x + 315 = 0 ∧ 15 * y^2 - m * y + 315 = 0 ∧ x ≠ y) ∧
  m = 150 := by
sorry

end smallest_m_for_integral_solutions_l2551_255120


namespace ellipse_condition_l2551_255177

/-- A non-degenerate ellipse is represented by the equation x^2 + 9y^2 - 6x + 27y = b
    if and only if b > -145/4 -/
theorem ellipse_condition (b : ℝ) :
  (∃ (x y : ℝ), x^2 + 9*y^2 - 6*x + 27*y = b) ∧
  (∀ (x y : ℝ), x^2 + 9*y^2 - 6*x + 27*y = b → (x, y) ≠ (0, 0)) ↔
  b > -145/4 :=
by sorry

end ellipse_condition_l2551_255177


namespace find_m_find_t_range_l2551_255126

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Theorem 1: Find the value of m
theorem find_m :
  (∃ m : ℝ, ∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-2) 2) →
  (∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-2) 2) :=
sorry

-- Theorem 2: Find the range of t
theorem find_t_range (m : ℝ) (h : m = 2) :
  (∀ x t : ℝ, f m x ≥ -|x + 6| - t^2 + t) →
  (∀ t : ℝ, t ∈ Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

end find_m_find_t_range_l2551_255126


namespace special_sequence_values_l2551_255144

/-- An increasing sequence of natural numbers satisfying a_{a_k} = 3k for any k. -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k, a (a k) = 3 * k)

theorem special_sequence_values (a : ℕ → ℕ) (h : SpecialSequence a) :
  a 100 = 181 ∧ a 1983 = 3762 := by
  sorry

end special_sequence_values_l2551_255144


namespace tetrahedron_distance_sum_l2551_255100

/-- Theorem about distances in a tetrahedron -/
theorem tetrahedron_distance_sum (V : ℝ) (S₁ S₂ S₃ S₄ : ℝ) (H₁ H₂ H₃ H₄ : ℝ) (k : ℝ) :
  V > 0 →
  S₁ > 0 → S₂ > 0 → S₃ > 0 → S₄ > 0 →
  H₁ > 0 → H₂ > 0 → H₃ > 0 → H₄ > 0 →
  S₁ = k → S₂ = 2*k → S₃ = 3*k → S₄ = 4*k →
  V = (1/3) * (S₁*H₁ + S₂*H₂ + S₃*H₃ + S₄*H₄) →
  H₁ + 2*H₂ + 3*H₃ + 4*H₄ = 3*V/k :=
by sorry

end tetrahedron_distance_sum_l2551_255100


namespace solution_set_for_a_eq_2_range_of_a_for_bounded_f_l2551_255195

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Theorem 1
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem range_of_a_for_bounded_f :
  ∀ a : ℝ, (∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) → (a = -1 ∨ a = 1) := by sorry

end solution_set_for_a_eq_2_range_of_a_for_bounded_f_l2551_255195


namespace sqrt_plus_inverse_geq_two_ab_plus_one_neq_a_plus_b_iff_min_value_of_expression_l2551_255162

-- Statement 1
theorem sqrt_plus_inverse_geq_two (x : ℝ) (hx : x > 0) :
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by sorry

-- Statement 2
theorem ab_plus_one_neq_a_plus_b_iff (a b : ℝ) :
  a * b + 1 ≠ a + b ↔ a ≠ 1 ∧ b ≠ 1 := by sorry

-- Statement 3
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → a / b + 1 / (a * b) ≥ m) ∧
  a / b + 1 / (a * b) = 2 * Real.sqrt 2 + 2 := by sorry

end sqrt_plus_inverse_geq_two_ab_plus_one_neq_a_plus_b_iff_min_value_of_expression_l2551_255162


namespace complete_square_equivalence_l2551_255104

theorem complete_square_equivalence : 
  ∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ (x - 3)^2 = 16 :=
by
  sorry

end complete_square_equivalence_l2551_255104
