import Mathlib

namespace NUMINAMATH_CALUDE_school_transfer_percentage_l2328_232862

theorem school_transfer_percentage :
  ∀ (total_students : ℝ) (school_A_percentage : ℝ) (school_B_percentage : ℝ)
    (transfer_A_to_C_percentage : ℝ) (transfer_B_to_C_percentage : ℝ),
  school_A_percentage = 60 →
  school_B_percentage = 100 - school_A_percentage →
  transfer_A_to_C_percentage = 30 →
  transfer_B_to_C_percentage = 40 →
  let students_A := total_students * (school_A_percentage / 100)
  let students_B := total_students * (school_B_percentage / 100)
  let students_C := (students_A * (transfer_A_to_C_percentage / 100)) +
                    (students_B * (transfer_B_to_C_percentage / 100))
  (students_C / total_students) * 100 = 34 :=
by sorry

end NUMINAMATH_CALUDE_school_transfer_percentage_l2328_232862


namespace NUMINAMATH_CALUDE_divisibility_and_modulo_l2328_232840

theorem divisibility_and_modulo (n : ℤ) (h : 11 ∣ (4 * n + 3)) : 
  n % 11 = 2 ∧ n^4 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_modulo_l2328_232840


namespace NUMINAMATH_CALUDE_value_of_y_l2328_232882

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2328_232882


namespace NUMINAMATH_CALUDE_max_value_problem_l2328_232885

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  ∃ (max : ℝ), max = 1 ∧ x + y^3 + z^2 ≤ max ∧ ∃ (x' y' z' : ℝ), x' + y'^3 + z'^2 = max :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l2328_232885


namespace NUMINAMATH_CALUDE_integral_sqrt_a_squared_minus_x_squared_l2328_232816

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) (ha : a > 0) :
  ∫ x in -a..a, Real.sqrt (a^2 - x^2) = (1/2) * π * a^2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_a_squared_minus_x_squared_l2328_232816


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_problem_solution_l2328_232817

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := λ n => a * r^n

theorem infinite_geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem problem_solution :
  ∑' n, geometric_series (1/4) (1/3) n = 3/8 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_problem_solution_l2328_232817


namespace NUMINAMATH_CALUDE_min_roots_sum_squared_l2328_232836

/-- Given a quadratic equation x^2 + 2(k+3)x + k^2 + 3 = 0 with real parameter k,
    this function returns the value of (α - 1)^2 + (β - 1)^2,
    where α and β are the roots of the equation. -/
def rootsSumSquared (k : ℝ) : ℝ :=
  2 * (k + 4)^2 - 12

/-- The minimum value of (α - 1)^2 + (β - 1)^2 where α and β are real roots of
    x^2 + 2(k+3)x + k^2 + 3 = 0, and k is a real parameter. -/
theorem min_roots_sum_squared :
  ∃ (m : ℝ), m = 6 ∧ ∀ (k : ℝ), (∀ (x : ℝ), x^2 + 2*(k+3)*x + k^2 + 3 ≥ 0) →
    rootsSumSquared k ≥ m :=
  sorry

end NUMINAMATH_CALUDE_min_roots_sum_squared_l2328_232836


namespace NUMINAMATH_CALUDE_simplify_expression_l2328_232877

theorem simplify_expression (x y : ℝ) :
  3 * (x + y)^2 - 7 * (x + y) + 8 * (x + y)^2 + 6 * (x + y) = 11 * (x + y)^2 - (x + y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2328_232877


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2328_232852

-- First expression
theorem simplify_expression_1 (x y : ℝ) :
  2 - x + 3*y + 8*x - 5*y - 6 = 7*x - 2*y - 4 := by sorry

-- Second expression
theorem simplify_expression_2 (a b : ℝ) :
  15*a^2*b - 12*a*b^2 + 12 - 4*a^2*b - 18 + 8*a*b^2 = 11*a^2*b - 4*a*b^2 - 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2328_232852


namespace NUMINAMATH_CALUDE_weeks_to_save_for_shirt_l2328_232843

/-- Calculate the number of weeks needed to save for a shirt -/
theorem weeks_to_save_for_shirt (total_cost saved_amount savings_rate : ℚ) : 
  total_cost = 3 →
  saved_amount = 3/2 →
  savings_rate = 1/2 →
  (total_cost - saved_amount) / savings_rate = 3 := by
  sorry

#check weeks_to_save_for_shirt

end NUMINAMATH_CALUDE_weeks_to_save_for_shirt_l2328_232843


namespace NUMINAMATH_CALUDE_rectangle_rotation_l2328_232804

theorem rectangle_rotation (w : ℝ) (a : ℝ) (l : ℝ) : 
  w = 6 →
  (1/4) * Real.pi * (l^2 + w^2) = a →
  a = 45 * Real.pi →
  l = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_rotation_l2328_232804


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2328_232845

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x, f x) ↔ ∀ x, ¬ f x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2328_232845


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l2328_232851

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_11 (seq : ArithmeticSequence) :
  sum_n seq 15 = 75 ∧ seq.a 3 + seq.a 4 + seq.a 5 = 12 → sum_n seq 11 = 99 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l2328_232851


namespace NUMINAMATH_CALUDE_variance_or_std_dev_measures_stability_l2328_232838

-- Define a type for exam scores
def ExamScore := ℝ

-- Define a type for a set of exam scores
def ExamScores := List ExamScore

-- Define a function to calculate variance
noncomputable def variance (scores : ExamScores) : ℝ := sorry

-- Define a function to calculate standard deviation
noncomputable def standardDeviation (scores : ExamScores) : ℝ := sorry

-- Define a measure of stability
noncomputable def stabilityMeasure (scores : ExamScores) : ℝ := sorry

-- Theorem stating that variance or standard deviation is the most appropriate measure of stability
theorem variance_or_std_dev_measures_stability (scores : ExamScores) :
  (stabilityMeasure scores = variance scores) ∨ (stabilityMeasure scores = standardDeviation scores) :=
sorry

end NUMINAMATH_CALUDE_variance_or_std_dev_measures_stability_l2328_232838


namespace NUMINAMATH_CALUDE_area_code_digits_l2328_232895

/-- Represents the set of allowed digits -/
def allowed_digits : Finset ℕ := {2, 3, 4}

/-- Calculates the number of valid area codes for a given number of digits -/
def valid_codes (n : ℕ) : ℕ := 3^n - 1

/-- The actual number of valid codes as per the problem statement -/
def actual_valid_codes : ℕ := 26

/-- The theorem stating that the number of digits in each area code is 3 -/
theorem area_code_digits :
  ∃ (n : ℕ), n > 0 ∧ valid_codes n = actual_valid_codes ∧ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_area_code_digits_l2328_232895


namespace NUMINAMATH_CALUDE_altered_difference_larger_l2328_232881

theorem altered_difference_larger (a b : ℤ) (h1 : a > b) (h2 : b > 0) :
  (1.03 : ℝ) * (a : ℝ) - 0.98 * (b : ℝ) > (a : ℝ) - (b : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_altered_difference_larger_l2328_232881


namespace NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l2328_232890

theorem quadratic_root_arithmetic_sequence (p q r : ℝ) : 
  p ≥ q → q ≥ r → r ≥ 0 →  -- Conditions on p, q, r
  (∃ d : ℝ, q = p - d ∧ r = p - 2*d) →  -- Arithmetic sequence condition
  (∃! x : ℝ, p*x^2 + q*x + r = 0) →  -- Exactly one root condition
  (∃ x : ℝ, p*x^2 + q*x + r = 0 ∧ x = -2 + Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l2328_232890


namespace NUMINAMATH_CALUDE_smallest_reducible_fraction_l2328_232806

theorem smallest_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ n = 144 ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    ¬(∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = m - 17 ∧ a * (7 * b) = 7 * m + 8)) ∧
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = n - 17 ∧ a * (7 * b) = 7 * n + 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_reducible_fraction_l2328_232806


namespace NUMINAMATH_CALUDE_hoseok_candy_count_l2328_232864

/-- The number of candies Hoseok has of type A -/
def candies_A : ℕ := 2

/-- The number of candies Hoseok has of type B -/
def candies_B : ℕ := 5

/-- The total number of candies Hoseok has -/
def total_candies : ℕ := candies_A + candies_B

theorem hoseok_candy_count : total_candies = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_candy_count_l2328_232864


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l2328_232850

theorem max_sum_of_roots (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ 3 * Real.sqrt 3 ∧
  (Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = 3 * Real.sqrt 3 ↔ 
    x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l2328_232850


namespace NUMINAMATH_CALUDE_sum_binomial_coefficients_l2328_232888

theorem sum_binomial_coefficients (n : ℕ) : 
  (Finset.range (n + 1)).sum (fun k => Nat.choose n k) = 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_binomial_coefficients_l2328_232888


namespace NUMINAMATH_CALUDE_total_commute_time_is_16_l2328_232874

-- Define the time it takes to walk and bike to work
def walk_time : ℕ := 2
def bike_time : ℕ := 1

-- Define the number of times Roque walks and bikes to work per week
def walk_trips : ℕ := 3
def bike_trips : ℕ := 2

-- Define the total commuting time
def total_commute_time : ℕ := 
  2 * (walk_time * walk_trips + bike_time * bike_trips)

-- Theorem statement
theorem total_commute_time_is_16 : total_commute_time = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_commute_time_is_16_l2328_232874


namespace NUMINAMATH_CALUDE_rowing_round_trip_time_l2328_232859

/-- Calculates the total time for a round trip rowing journey given the rowing speed, current speed, and distance. -/
theorem rowing_round_trip_time 
  (rowing_speed : ℝ) 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (h1 : rowing_speed = 10)
  (h2 : current_speed = 2)
  (h3 : distance = 24) : 
  (distance / (rowing_speed + current_speed)) + (distance / (rowing_speed - current_speed)) = 5 := by
  sorry

#check rowing_round_trip_time

end NUMINAMATH_CALUDE_rowing_round_trip_time_l2328_232859


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2328_232883

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±2√2x -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes : ∀ x, ∃ y, y = 2 * Real.sqrt 2 * x ∨ y = -2 * Real.sqrt 2 * x) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2328_232883


namespace NUMINAMATH_CALUDE_multiplication_property_l2328_232894

theorem multiplication_property : 
  (142857 * 5 = 714285) ∧ (142857 * 3 = 428571) := by sorry

end NUMINAMATH_CALUDE_multiplication_property_l2328_232894


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2328_232868

theorem quadratic_inequality_solution (x : ℝ) : 
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3/2) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2328_232868


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l2328_232832

theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1020)
  (h2 : loss_percentage = 15) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l2328_232832


namespace NUMINAMATH_CALUDE_product_98_102_l2328_232853

theorem product_98_102 : 98 * 102 = 9996 := by
  sorry

end NUMINAMATH_CALUDE_product_98_102_l2328_232853


namespace NUMINAMATH_CALUDE_compound_composition_l2328_232889

/-- Atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- Atomic weight of phosphorus in g/mol -/
def atomic_weight_P : ℝ := 30.97

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 122

/-- The number of oxygen atoms in the compound -/
def num_oxygen_atoms : ℕ := 4

theorem compound_composition :
  ∀ x : ℕ, 
    atomic_weight_Al + atomic_weight_P + x * atomic_weight_O = compound_weight 
    ↔ 
    x = num_oxygen_atoms :=
by sorry

end NUMINAMATH_CALUDE_compound_composition_l2328_232889


namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l2328_232899

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2 + 1)) (λ i => 2 * i)

theorem greatest_prime_factor_f_28 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ f 28 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ f 28 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l2328_232899


namespace NUMINAMATH_CALUDE_mixture_theorem_l2328_232879

/-- Represents a mixture of three liquids -/
structure Mixture where
  lemon : ℚ
  oil : ℚ
  vinegar : ℚ

/-- Mix A composition -/
def mixA : Mixture := ⟨1, 2, 3⟩

/-- Mix B composition -/
def mixB : Mixture := ⟨3, 4, 5⟩

/-- Checks if it's possible to create a target mixture from Mix A and Mix B -/
def canCreateMixture (target : Mixture) : Prop :=
  ∃ (x y : ℚ), x ≥ 0 ∧ y ≥ 0 ∧
    x * mixA.lemon + y * mixB.lemon = (x + y) * target.lemon ∧
    x * mixA.oil + y * mixB.oil = (x + y) * target.oil ∧
    x * mixA.vinegar + y * mixB.vinegar = (x + y) * target.vinegar

theorem mixture_theorem :
  canCreateMixture ⟨3, 5, 7⟩ ∧
  ¬canCreateMixture ⟨2, 5, 8⟩ ∧
  ¬canCreateMixture ⟨4, 5, 6⟩ ∧
  ¬canCreateMixture ⟨5, 6, 7⟩ := by
  sorry


end NUMINAMATH_CALUDE_mixture_theorem_l2328_232879


namespace NUMINAMATH_CALUDE_select_students_with_female_l2328_232814

/-- The number of male students -/
def num_male : ℕ := 5

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select students with at least one female -/
def num_ways_with_female : ℕ := Nat.choose (num_male + num_female) num_selected - Nat.choose num_male num_selected

theorem select_students_with_female :
  num_ways_with_female = 25 := by
  sorry

end NUMINAMATH_CALUDE_select_students_with_female_l2328_232814


namespace NUMINAMATH_CALUDE_piece_exits_at_A2_l2328_232828

/-- Represents the directions a piece can move on the grid -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a cell on the 4x4 grid -/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the state of the grid -/
structure GridState :=
  (currentCell : Cell)
  (arrows : Cell → Direction)

/-- Defines a single move on the grid -/
def move (state : GridState) : GridState :=
  sorry

/-- Checks if a cell is on the boundary of the grid -/
def isOnBoundary (cell : Cell) : Bool :=
  sorry

/-- Simulates the movement of the piece until it reaches the boundary -/
def simulateUntilExit (initialState : GridState) : Cell :=
  sorry

/-- The main theorem to prove -/
theorem piece_exits_at_A2 :
  let initialState : GridState := {
    currentCell := { row := 2, col := 1 },  -- C2 in 0-indexed
    arrows := sorry  -- Initial arrow configuration
  }
  let exitCell := simulateUntilExit initialState
  exitCell = { row := 0, col := 1 }  -- A2 in 0-indexed
  :=
sorry

end NUMINAMATH_CALUDE_piece_exits_at_A2_l2328_232828


namespace NUMINAMATH_CALUDE_tank_fill_time_l2328_232898

/-- Represents a machine that can fill or empty a tank -/
structure Machine where
  fillRate : ℚ  -- Rate at which the machine fills the tank (fraction per minute)
  emptyRate : ℚ -- Rate at which the machine empties the tank (fraction per minute)

/-- Calculates the net rate of a machine that alternates between filling and emptying -/
def alternatingRate (fillTime emptyTime cycleTime : ℚ) : ℚ :=
  (fillTime / cycleTime) * (1 / fillTime) + (emptyTime / cycleTime) * (-1 / emptyTime)

/-- The main theorem stating the time to fill the tank -/
theorem tank_fill_time :
  let machineA : Machine := ⟨1/25, 0⟩
  let machineB : Machine := ⟨0, 1/50⟩
  let machineC : Machine := ⟨alternatingRate 5 5 10, 0⟩
  let combinedRate := machineA.fillRate - machineB.emptyRate + machineC.fillRate
  let remainingVolume := 1/2
  ⌈remainingVolume / combinedRate⌉ = 20 := by
  sorry


end NUMINAMATH_CALUDE_tank_fill_time_l2328_232898


namespace NUMINAMATH_CALUDE_tabitha_money_to_mom_l2328_232869

/-- The amount of money Tabitha gave her mom -/
def money_given_to_mom (initial_amount : ℚ) (item_cost : ℚ) (num_items : ℕ) (final_amount : ℚ) : ℚ :=
  initial_amount - 2 * (final_amount + item_cost * num_items)

/-- Theorem stating the amount of money Tabitha gave her mom -/
theorem tabitha_money_to_mom :
  money_given_to_mom 25 0.5 5 6 = 8 := by
  sorry

#eval money_given_to_mom 25 0.5 5 6

end NUMINAMATH_CALUDE_tabitha_money_to_mom_l2328_232869


namespace NUMINAMATH_CALUDE_angle_properties_l2328_232875

theorem angle_properties (α β : Real) : 
  α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) →  -- α is in the fourth quadrant
  Real.sin (Real.pi + α) = 2 * Real.sqrt 5 / 5 →
  Real.tan (α + β) = 1 / 7 →
  Real.cos (Real.pi / 3 + α) = (Real.sqrt 5 + 2 * Real.sqrt 15) / 10 ∧ 
  Real.tan β = 3 := by
sorry

end NUMINAMATH_CALUDE_angle_properties_l2328_232875


namespace NUMINAMATH_CALUDE_a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b_l2328_232837

theorem a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b (a b : ℝ) :
  a^2 * b > a * b^2 ↔ 1/a < 1/b :=
by sorry

end NUMINAMATH_CALUDE_a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b_l2328_232837


namespace NUMINAMATH_CALUDE_exists_strictly_convex_function_with_constraints_l2328_232819

-- Define the function type
def StrictlyConvexFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    f x₁ + f x₂ < 2 * f ((x₁ + x₂) / 2)

-- Main theorem
theorem exists_strictly_convex_function_with_constraints : 
  ∃ f : ℝ → ℝ,
    (∀ x : ℝ, x > 0 → f x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 4 → ∃ x : ℝ, x > 0 ∧ f x = y) ∧
    StrictlyConvexFunction f :=
sorry

end NUMINAMATH_CALUDE_exists_strictly_convex_function_with_constraints_l2328_232819


namespace NUMINAMATH_CALUDE_circle_radius_in_square_with_semicircles_l2328_232872

/-- Given a square with side length 108 and semicircles constructed inward on two adjacent sides,
    the radius of a circle touching one side and both semicircles is 27. -/
theorem circle_radius_in_square_with_semicircles (square_side : ℝ) 
  (h_side : square_side = 108) : ∃ (r : ℝ), r = 27 ∧ 
  r + (square_side / 2) = square_side - r := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_square_with_semicircles_l2328_232872


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2328_232880

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by
  sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x - 2 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2328_232880


namespace NUMINAMATH_CALUDE_algae_growth_l2328_232826

/-- Represents the number of cells in an algae colony after a given number of days -/
def algaeCells (initialCells : ℕ) (divisionPeriod : ℕ) (totalDays : ℕ) : ℕ :=
  initialCells * (2 ^ (totalDays / divisionPeriod))

/-- Theorem stating that an algae colony starting with 5 cells, doubling every 3 days,
    will have 20 cells after 9 days -/
theorem algae_growth : algaeCells 5 3 9 = 20 := by
  sorry


end NUMINAMATH_CALUDE_algae_growth_l2328_232826


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2328_232810

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 1 ≥ -8 ∧ ∃ y : ℝ, y^2 + 6*y + 1 = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2328_232810


namespace NUMINAMATH_CALUDE_medium_apple_cost_l2328_232854

/-- Proves that the cost of a medium apple is $2 given the conditions in the problem -/
theorem medium_apple_cost (small_apple_cost big_apple_cost total_cost : ℝ)
  (small_medium_count big_count : ℕ) :
  small_apple_cost = 1.5 →
  big_apple_cost = 3 →
  small_medium_count = 6 →
  big_count = 8 →
  total_cost = 45 →
  ∃ (medium_apple_cost : ℝ),
    small_apple_cost * (small_medium_count / 2) +
    medium_apple_cost * (small_medium_count / 2) +
    big_apple_cost * big_count = total_cost ∧
    medium_apple_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_medium_apple_cost_l2328_232854


namespace NUMINAMATH_CALUDE_expand_expression_l2328_232857

theorem expand_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2328_232857


namespace NUMINAMATH_CALUDE_mn_value_l2328_232847

theorem mn_value (m n : ℤ) : 
  (∀ x : ℤ, x^2 + m*x - 15 = (x + 3)*(x + n)) → m*n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mn_value_l2328_232847


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l2328_232815

theorem least_positive_integer_to_multiple_of_five : 
  ∀ n : ℕ, n > 0 → (725 + n) % 5 = 0 → n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l2328_232815


namespace NUMINAMATH_CALUDE_alice_coins_value_l2328_232818

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The number of pennies Alice has -/
def num_pennies : ℕ := 2

/-- The number of nickels Alice has -/
def num_nickels : ℕ := 3

/-- The number of dimes Alice has -/
def num_dimes : ℕ := 4

/-- The number of half-dollars Alice has -/
def num_half_dollars : ℕ := 1

/-- The total value of Alice's coins in cents -/
def total_cents : ℕ :=
  num_pennies * penny_value +
  num_nickels * nickel_value +
  num_dimes * dime_value +
  num_half_dollars * half_dollar_value

/-- The value of one dollar in cents -/
def dollar_in_cents : ℕ := 100

theorem alice_coins_value :
  (total_cents : ℚ) / (dollar_in_cents : ℚ) = 107 / 100 := by
  sorry

end NUMINAMATH_CALUDE_alice_coins_value_l2328_232818


namespace NUMINAMATH_CALUDE_simplify_expression_l2328_232892

theorem simplify_expression (x y : ℝ) : (2*x + 25) + (150*x + 40) + (5*y + 10) = 152*x + 5*y + 75 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2328_232892


namespace NUMINAMATH_CALUDE_allocation_schemes_l2328_232867

def doctors : ℕ := 2
def nurses : ℕ := 4
def hospitals : ℕ := 2
def doctors_per_hospital : ℕ := 1
def nurses_per_hospital : ℕ := 2

theorem allocation_schemes :
  (Nat.choose doctors hospitals) * (Nat.choose nurses (nurses_per_hospital * hospitals)) = 12 :=
sorry

end NUMINAMATH_CALUDE_allocation_schemes_l2328_232867


namespace NUMINAMATH_CALUDE_order_of_values_l2328_232873

theorem order_of_values : ∃ (a b c : ℝ),
  a = Real.exp 0.2 - 1 ∧
  b = Real.log 1.2 ∧
  c = Real.tan 0.2 ∧
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_order_of_values_l2328_232873


namespace NUMINAMATH_CALUDE_coin_flip_difference_l2328_232863

/-- Given 211 total coin flips with 65 heads, the difference between the number of tails and heads is 81. -/
theorem coin_flip_difference (total_flips : ℕ) (heads : ℕ) 
    (h1 : total_flips = 211)
    (h2 : heads = 65) : 
  total_flips - heads - heads = 81 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_difference_l2328_232863


namespace NUMINAMATH_CALUDE_zero_only_universal_prime_multiple_l2328_232855

theorem zero_only_universal_prime_multiple : ∃! n : ℤ, ∀ p : ℕ, Prime p → ∃ k : ℤ, n * p = k * p :=
sorry

end NUMINAMATH_CALUDE_zero_only_universal_prime_multiple_l2328_232855


namespace NUMINAMATH_CALUDE_ten_markers_five_friends_l2328_232861

/-- The number of ways to distribute n identical markers among k friends,
    where each friend must have at least one marker -/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical markers among 5 friends,
    where each friend must have at least one marker, is 126 -/
theorem ten_markers_five_friends :
  distributionWays 10 5 = 126 := by sorry

end NUMINAMATH_CALUDE_ten_markers_five_friends_l2328_232861


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_eccentricity_l2328_232823

/-- The eccentricity of a hyperbola that has a common point with the line y = 2x --/
def eccentricity_range (a b : ℝ) : Prop :=
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ x y : ℝ, y = 2*x ∧ x^2/a^2 - y^2/b^2 = 1) →
  1 < e ∧ e ≤ Real.sqrt 5

theorem hyperbola_line_intersection_eccentricity :
  ∀ a b : ℝ, a > 0 ∧ b > 0 → eccentricity_range a b :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_eccentricity_l2328_232823


namespace NUMINAMATH_CALUDE_problem_solution_l2328_232856

theorem problem_solution :
  ∀ x y : ℕ,
    x > 0 → y > 0 →
    x < 15 → y < 15 →
    x + y + x * y = 49 →
    x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2328_232856


namespace NUMINAMATH_CALUDE_d_equals_square_cases_l2328_232829

/-- Function to move the last digit of a number to the first position -/
def moveLastToFirst (n : ℕ) : ℕ := sorry

/-- Function to move the first digit of a number to the last position -/
def moveFirstToLast (n : ℕ) : ℕ := sorry

/-- The d function as described in the problem -/
def d (a : ℕ) : ℕ := 
  let b := moveLastToFirst a
  let c := b * b
  moveFirstToLast c

/-- Theorem stating the possible forms of a when d(a) = a^2 -/
theorem d_equals_square_cases (a : ℕ) (h : 0 < a) : 
  d a = a * a → (a = 2 ∨ a = 3 ∨ ∃ x y : ℕ, a = 20000 + 100 * x + 10 * y + 21) := by
  sorry

end NUMINAMATH_CALUDE_d_equals_square_cases_l2328_232829


namespace NUMINAMATH_CALUDE_addition_proof_l2328_232808

theorem addition_proof : 72 + 15 = 87 := by
  sorry

end NUMINAMATH_CALUDE_addition_proof_l2328_232808


namespace NUMINAMATH_CALUDE_equation_solution_l2328_232897

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 78 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2328_232897


namespace NUMINAMATH_CALUDE_equation_and_inequality_system_l2328_232896

theorem equation_and_inequality_system :
  -- Part 1: Equation
  (let equation := fun x : ℝ => 2 * x * (x - 2) = 1
   let solution1 := (2 + Real.sqrt 6) / 2
   let solution2 := (2 - Real.sqrt 6) / 2
   equation solution1 ∧ equation solution2) ∧
  -- Part 2: Inequality system
  (let inequality1 := fun x : ℝ => 2 * x + 3 > 1
   let inequality2 := fun x : ℝ => x - 2 ≤ (1 / 2) * (x + 2)
   ∀ x : ℝ, (inequality1 x ∧ inequality2 x) ↔ (-1 < x ∧ x ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequality_system_l2328_232896


namespace NUMINAMATH_CALUDE_polynomial_maximum_l2328_232844

/-- The polynomial function we're analyzing -/
def f (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 12

/-- The maximum value of the polynomial -/
def max_value : ℝ := 15

/-- The x-value at which the maximum occurs -/
def max_point : ℝ := -1

theorem polynomial_maximum :
  (∀ x : ℝ, f x ≤ max_value) ∧ f max_point = max_value :=
sorry

end NUMINAMATH_CALUDE_polynomial_maximum_l2328_232844


namespace NUMINAMATH_CALUDE_vacation_cost_l2328_232886

theorem vacation_cost (hotel_cost_per_person_per_day : ℕ) 
                      (total_vacation_cost : ℕ) 
                      (num_days : ℕ) 
                      (num_people : ℕ) : 
  hotel_cost_per_person_per_day = 12 →
  total_vacation_cost = 120 →
  num_days = 3 →
  num_people = 2 →
  (total_vacation_cost - (hotel_cost_per_person_per_day * num_days * num_people)) / num_people = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l2328_232886


namespace NUMINAMATH_CALUDE_formula_correctness_l2328_232800

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 2

theorem formula_correctness : 
  (f 2 = 10) ∧ 
  (f 3 = 21) ∧ 
  (f 4 = 38) ∧ 
  (f 5 = 61) ∧ 
  (f 6 = 90) := by
  sorry

end NUMINAMATH_CALUDE_formula_correctness_l2328_232800


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2328_232866

/-- A geometric sequence with positive first term and increasing terms -/
def IncreasingGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 > 0 ∧ q > 1 ∧ ∀ n, a (n + 1) = q * a n

/-- The relation between consecutive terms in the sequence -/
def SequenceRelation (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_increasing : IncreasingGeometricSequence a q)
  (h_relation : SequenceRelation a) :
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2328_232866


namespace NUMINAMATH_CALUDE_mp3_price_reduction_l2328_232801

/-- Given an item with a sale price of 112 after a 20% reduction,
    prove that its price after a 30% reduction would be 98. -/
theorem mp3_price_reduction (original_price : ℝ) : 
  (original_price * 0.8 = 112) → (original_price * 0.7 = 98) := by
  sorry

end NUMINAMATH_CALUDE_mp3_price_reduction_l2328_232801


namespace NUMINAMATH_CALUDE_difference_of_two_greatest_values_l2328_232824

def is_three_digit_integer (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

def hundreds_digit (x : ℕ) : ℕ :=
  (x / 100) % 10

def tens_digit (x : ℕ) : ℕ :=
  (x / 10) % 10

def units_digit (x : ℕ) : ℕ :=
  x % 10

def satisfies_conditions (x : ℕ) : Prop :=
  let a := hundreds_digit x
  let b := tens_digit x
  let c := units_digit x
  is_three_digit_integer x ∧ 4 * a = 2 * b ∧ 2 * b = c ∧ a > 0

def two_greatest_values (x y : ℕ) : Prop :=
  satisfies_conditions x ∧ satisfies_conditions y ∧
  ∀ z, satisfies_conditions z → z ≤ x ∧ (z ≠ x → z ≤ y)

theorem difference_of_two_greatest_values :
  ∃ x y, two_greatest_values x y ∧ x - y = 124 :=
sorry

end NUMINAMATH_CALUDE_difference_of_two_greatest_values_l2328_232824


namespace NUMINAMATH_CALUDE_log_eight_negative_seven_fourths_l2328_232803

theorem log_eight_negative_seven_fourths (x : ℝ) : 
  Real.log x / Real.log 8 = -1.75 → x = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_negative_seven_fourths_l2328_232803


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l2328_232891

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the line equation
def line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Theorem for the first circle
theorem circle1_properties :
  (∃ x y : ℝ, line x y ∧ circle1 x y) ∧  -- Center lies on the line
  circle1 2 (-3) ∧                       -- Passes through (2,-3)
  circle1 (-2) (-5) :=                   -- Passes through (-2,-5)
sorry

-- Theorem for the second circle
theorem circle2_properties :
  circle2 1 0 ∧    -- Passes through (1,0)
  circle2 (-1) (-2) ∧  -- Passes through (-1,-2)
  circle2 3 (-2) :=    -- Passes through (3,-2)
sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l2328_232891


namespace NUMINAMATH_CALUDE_range_of_a_l2328_232825

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 - (a+1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) ∧ 
  ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1)) ∧
  (∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → 
  ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2328_232825


namespace NUMINAMATH_CALUDE_road_trip_cost_equalization_l2328_232860

/-- The amount Jamie must give Dana to equalize costs on a road trip -/
theorem road_trip_cost_equalization
  (X Y Z : ℝ)  -- Amounts paid by Alexi, Jamie, and Dana respectively
  (hXY : Y > X)  -- Jamie paid more than Alexi
  (hYZ : Z > Y)  -- Dana paid more than Jamie
  : (X + Z - 2*Y) / 3 = 
    ((X + Y + Z) / 3 - Y)  -- Amount Jamie should give Dana
    := by sorry

end NUMINAMATH_CALUDE_road_trip_cost_equalization_l2328_232860


namespace NUMINAMATH_CALUDE_a_greater_than_b_l2328_232834

theorem a_greater_than_b (m : ℝ) (h : m > 1) : 
  (Real.sqrt m - Real.sqrt (m - 1)) > (Real.sqrt (m + 1) - Real.sqrt m) := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l2328_232834


namespace NUMINAMATH_CALUDE_medium_birdhouse_price_l2328_232884

/-- The price of a large birdhouse -/
def large_price : ℕ := 22

/-- The price of a small birdhouse -/
def small_price : ℕ := 7

/-- The number of large birdhouses sold -/
def large_sold : ℕ := 2

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total amount made from all birdhouses -/
def total_amount : ℕ := 97

/-- The price of a medium birdhouse -/
def medium_price : ℕ := 16

theorem medium_birdhouse_price : 
  large_price * large_sold + medium_price * medium_sold + small_price * small_sold = total_amount :=
by sorry

end NUMINAMATH_CALUDE_medium_birdhouse_price_l2328_232884


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2328_232878

/-- The repeating decimal 0.37̄246 expressed as a rational number -/
def repeating_decimal : ℚ := 37246 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = 371874 / 99900 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2328_232878


namespace NUMINAMATH_CALUDE_fourth_power_inequality_l2328_232848

theorem fourth_power_inequality (a b c : ℝ) : a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_inequality_l2328_232848


namespace NUMINAMATH_CALUDE_ny_mets_fans_count_l2328_232842

theorem ny_mets_fans_count (total_fans : ℕ) (yankees_mets_ratio : ℚ) (mets_redsox_ratio : ℚ) :
  total_fans = 390 →
  yankees_mets_ratio = 3 / 2 →
  mets_redsox_ratio = 4 / 5 →
  ∃ (yankees mets redsox : ℕ),
    yankees + mets + redsox = total_fans ∧
    (yankees : ℚ) / mets = yankees_mets_ratio ∧
    (mets : ℚ) / redsox = mets_redsox_ratio ∧
    mets = 104 :=
by sorry

end NUMINAMATH_CALUDE_ny_mets_fans_count_l2328_232842


namespace NUMINAMATH_CALUDE_points_per_bag_l2328_232831

theorem points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) :
  total_bags = 17 →
  unrecycled_bags = 8 →
  total_points = 45 →
  (total_points / (total_bags - unrecycled_bags) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_bag_l2328_232831


namespace NUMINAMATH_CALUDE_probability_even_sum_l2328_232841

theorem probability_even_sum (wheel1_even : ℚ) (wheel1_odd : ℚ) 
  (wheel2_even : ℚ) (wheel2_odd : ℚ) : 
  wheel1_even = 1/4 →
  wheel1_odd = 3/4 →
  wheel2_even = 2/3 →
  wheel2_odd = 1/3 →
  wheel1_even + wheel1_odd = 1 →
  wheel2_even + wheel2_odd = 1 →
  wheel1_even * wheel2_even + wheel1_odd * wheel2_odd = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l2328_232841


namespace NUMINAMATH_CALUDE_planning_committee_selections_l2328_232830

def student_council_size : ℕ := 6

theorem planning_committee_selections :
  (Nat.choose student_council_size 3 = 20) →
  (Nat.choose student_council_size 3 = 20) := by
  sorry

end NUMINAMATH_CALUDE_planning_committee_selections_l2328_232830


namespace NUMINAMATH_CALUDE_fruits_remaining_proof_l2328_232821

def initial_apples : ℕ := 7
def initial_oranges : ℕ := 8
def initial_mangoes : ℕ := 15

def apples_taken : ℕ := 2
def oranges_taken : ℕ := 2 * apples_taken
def mangoes_taken : ℕ := (2 * initial_mangoes) / 3

def remaining_fruits : ℕ :=
  (initial_apples - apples_taken) +
  (initial_oranges - oranges_taken) +
  (initial_mangoes - mangoes_taken)

theorem fruits_remaining_proof :
  remaining_fruits = 14 := by sorry

end NUMINAMATH_CALUDE_fruits_remaining_proof_l2328_232821


namespace NUMINAMATH_CALUDE_unique_solution_is_sqrt_two_l2328_232822

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

-- State the theorem
theorem unique_solution_is_sqrt_two :
  ∃! x, x > 1 ∧ f x = 1/4 ∧ x = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_sqrt_two_l2328_232822


namespace NUMINAMATH_CALUDE_field_division_l2328_232870

theorem field_division (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 900 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 405 := by
sorry

end NUMINAMATH_CALUDE_field_division_l2328_232870


namespace NUMINAMATH_CALUDE_hexagon_area_l2328_232827

/-- Given a square with area 16 and a regular hexagon with perimeter 3/4 of the square's perimeter,
    the area of the hexagon is 32√3/27. -/
theorem hexagon_area (s : ℝ) (t : ℝ) : 
  s^2 = 16 → 
  4 * s = 18 * t → 
  (3 * t^2 * Real.sqrt 3) / 2 = (32 * Real.sqrt 3) / 27 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l2328_232827


namespace NUMINAMATH_CALUDE_elizabeth_lost_bottles_l2328_232833

/-- The number of water bottles Elizabeth lost at school -/
def bottles_lost_at_school : ℕ := 2

theorem elizabeth_lost_bottles (initial_bottles : ℕ) (stolen_bottle : ℕ) (stickers_per_bottle : ℕ) (total_stickers : ℕ) :
  initial_bottles = 10 →
  stolen_bottle = 1 →
  stickers_per_bottle = 3 →
  total_stickers = 21 →
  stickers_per_bottle * (initial_bottles - bottles_lost_at_school - stolen_bottle) = total_stickers →
  bottles_lost_at_school = 2 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_lost_bottles_l2328_232833


namespace NUMINAMATH_CALUDE_poppy_seed_count_l2328_232812

def total_slices : ℕ := 58

theorem poppy_seed_count (x : ℕ) 
  (h1 : x ≤ total_slices)
  (h2 : Nat.choose x 3 = Nat.choose (total_slices - x) 2 * x) :
  total_slices - x = 21 := by
  sorry

end NUMINAMATH_CALUDE_poppy_seed_count_l2328_232812


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l2328_232805

/-- Given two lines in R², prove their intersection point and a parallel line through that point. -/
theorem intersection_and_parallel_line 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ))
  (h₁ : l₁ = {p : ℝ × ℝ | p.1 + 8 * p.2 + 7 = 0})
  (h₂ : l₂ = {p : ℝ × ℝ | 2 * p.1 + p.2 - 1 = 0})
  (l₃ : Set (ℝ × ℝ))
  (h₃ : l₃ = {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}) :
  (∃! p : ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂ ∧ p = (1, -1)) ∧
  (∃ l : Set (ℝ × ℝ), l = {p : ℝ × ℝ | p.1 + p.2 = 0} ∧ 
    (1, -1) ∈ l ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l → p.1 - q.1 = q.2 - p.2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l2328_232805


namespace NUMINAMATH_CALUDE_apple_cost_l2328_232811

/-- Proves that the cost of each apple is 4 dollars given the conditions -/
theorem apple_cost (total_money : ℕ) (kids : ℕ) (apples_per_kid : ℕ) :
  total_money = 360 →
  kids = 18 →
  apples_per_kid = 5 →
  total_money / (kids * apples_per_kid) = 4 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_l2328_232811


namespace NUMINAMATH_CALUDE_students_remaining_after_three_stops_l2328_232876

def initial_students : ℕ := 60

def remaining_after_first_stop (initial : ℕ) : ℕ :=
  initial - initial / 3

def remaining_after_second_stop (after_first : ℕ) : ℕ :=
  after_first - after_first / 4

def remaining_after_third_stop (after_second : ℕ) : ℕ :=
  after_second - after_second / 5

theorem students_remaining_after_three_stops :
  remaining_after_third_stop (remaining_after_second_stop (remaining_after_first_stop initial_students)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_after_three_stops_l2328_232876


namespace NUMINAMATH_CALUDE_car_speed_problem_l2328_232802

theorem car_speed_problem (train_speed_ratio : ℝ) (distance : ℝ) (train_stop_time : ℝ) :
  train_speed_ratio = 1.5 →
  distance = 75 →
  train_stop_time = 12.5 / 60 →
  ∃ (car_speed : ℝ),
    car_speed = 80 ∧
    distance = car_speed * (distance / car_speed) ∧
    distance = (train_speed_ratio * car_speed) * (distance / car_speed - train_stop_time) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2328_232802


namespace NUMINAMATH_CALUDE_cube_preserves_order_l2328_232813

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l2328_232813


namespace NUMINAMATH_CALUDE_horner_method_for_f_l2328_232858

def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem horner_method_for_f :
  f 3 = 1452.4 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l2328_232858


namespace NUMINAMATH_CALUDE_popsicle_stick_ratio_l2328_232865

theorem popsicle_stick_ratio : 
  ∀ (steve sid sam : ℕ),
  steve = 12 →
  sid = 2 * steve →
  sam + sid + steve = 108 →
  sam / sid = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_ratio_l2328_232865


namespace NUMINAMATH_CALUDE_cylinder_volume_unchanged_l2328_232846

/-- Theorem: For a cylinder with radius 5 inches and height 4 inches, 
    the value of x that keeps the volume unchanged when the radius 
    is increased by x and the height is decreased by x is 5 - 2√10. -/
theorem cylinder_volume_unchanged (R H : ℝ) (x : ℝ) : 
  R = 5 → H = 4 → 
  π * R^2 * H = π * (R + x)^2 * (H - x) → 
  x = 5 - 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_unchanged_l2328_232846


namespace NUMINAMATH_CALUDE_beautiful_equations_proof_l2328_232820

/-- Two linear equations are "beautiful equations" if the sum of their solutions is 1 -/
def beautiful_equations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), eq1 x ∧ eq2 y ∧ x + y = 1

/-- The first pair of equations -/
def eq1 (x : ℝ) : Prop := 4*x - (x + 5) = 1
def eq2 (y : ℝ) : Prop := -2*y - y = 3

/-- The second pair of equations with parameter n -/
def eq3 (n : ℝ) (x : ℝ) : Prop := 2*x - n + 3 = 0
def eq4 (n : ℝ) (x : ℝ) : Prop := x + 5*n - 1 = 0

theorem beautiful_equations_proof :
  (beautiful_equations eq1 eq2) ∧
  (∃ (n : ℝ), n = -1/3 ∧ beautiful_equations (eq3 n) (eq4 n)) :=
by sorry

end NUMINAMATH_CALUDE_beautiful_equations_proof_l2328_232820


namespace NUMINAMATH_CALUDE_distribute_five_books_three_people_l2328_232849

/-- The number of ways to distribute books among people -/
def distribute_books (n_books : ℕ) (n_people : ℕ) (min_books : ℕ) (max_books : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to distribute 5 books among 3 people -/
theorem distribute_five_books_three_people : 
  distribute_books 5 3 1 2 = 90 := by sorry

end NUMINAMATH_CALUDE_distribute_five_books_three_people_l2328_232849


namespace NUMINAMATH_CALUDE_f_increasing_scaled_l2328_232887

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_increasing_scaled (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : x₁ * f x₁ < x₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_scaled_l2328_232887


namespace NUMINAMATH_CALUDE_investment_principal_calculation_l2328_232809

/-- Proves that given an investment with a monthly interest payment of $228 and a simple annual interest rate of 9%, the principal amount of the investment is $30,400. -/
theorem investment_principal_calculation (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 228 →
  annual_rate = 0.09 →
  ∃ principal : ℝ, principal = 30400 ∧ monthly_interest = principal * (annual_rate / 12) :=
by sorry

end NUMINAMATH_CALUDE_investment_principal_calculation_l2328_232809


namespace NUMINAMATH_CALUDE_quadratic_roots_min_max_values_l2328_232893

/-- Given a quadratic equation with specific conditions, prove the minimum and maximum values of a related expression -/
theorem quadratic_roots_min_max_values (a b : ℝ) :
  let f := fun x : ℝ => x^2 - (a^2 + b^2 - 6*b)*x + a^2 + b^2 + 2*a - 4*b + 1
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≤ 0 ∧ 0 ≤ x₂ ∧ x₂ ≤ 1) →
  (∀ x : ℝ, a^2 + b^2 + 4*a ≥ -7/2) ∧
  (∃ a₀ b₀ : ℝ, a₀^2 + b₀^2 + 4*a₀ = 5 + 4*Real.sqrt 5) ∧
  (∀ a₁ b₁ : ℝ, a₁^2 + b₁^2 + 4*a₁ ≤ 5 + 4*Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_min_max_values_l2328_232893


namespace NUMINAMATH_CALUDE_cube_root_implies_value_l2328_232835

theorem cube_root_implies_value (x : ℝ) : 
  (2 * x - 14) ^ (1/3 : ℝ) = -2 → 2 * x + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_implies_value_l2328_232835


namespace NUMINAMATH_CALUDE_angle_D_value_l2328_232839

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the theorem
theorem angle_D_value :
  A + B = 180 →  -- Condition 1
  C = D →        -- Condition 2
  C + 50 + 60 = 180 →  -- Condition 3
  D = 70 :=  -- Conclusion
by
  sorry  -- Proof is omitted as per instructions


end NUMINAMATH_CALUDE_angle_D_value_l2328_232839


namespace NUMINAMATH_CALUDE_equal_area_circle_l2328_232807

theorem equal_area_circle (π : ℝ) : 
  π > 0 → 
  π * (6 * Real.sqrt 10)^2 = π * 23^2 - π * 13^2 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_circle_l2328_232807


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l2328_232871

/-- Calculates the corrected mean when one observation in a dataset is incorrect -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean + (correct_value - incorrect_value)) / n

/-- Theorem stating that the corrected mean is 36.02 given the problem conditions -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let incorrect_value : ℚ := 47
  let correct_value : ℚ := 48
  corrected_mean n initial_mean incorrect_value correct_value = 3602/100 := by
  sorry

#eval corrected_mean 50 36 47 48

end NUMINAMATH_CALUDE_corrected_mean_problem_l2328_232871
