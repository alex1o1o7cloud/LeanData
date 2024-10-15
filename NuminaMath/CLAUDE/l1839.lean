import Mathlib

namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1839_183928

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1839_183928


namespace NUMINAMATH_CALUDE_emily_walks_farther_l1839_183941

def troy_base_distance : ℕ := 75
def emily_base_distance : ℕ := 98

def troy_daily_distances : List ℕ := [90, 95, 85, 85, 80]
def emily_daily_distances : List ℕ := [108, 123, 108, 123, 108]

def calculate_total_distance (daily_distances : List ℕ) : ℕ :=
  2 * (daily_distances.sum)

theorem emily_walks_farther :
  calculate_total_distance emily_daily_distances - calculate_total_distance troy_daily_distances = 270 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l1839_183941


namespace NUMINAMATH_CALUDE_workout_calculation_l1839_183964

-- Define the exercise parameters
def bicep_curls_weight : ℕ := 20
def bicep_curls_dumbbells : ℕ := 2
def bicep_curls_reps : ℕ := 10
def bicep_curls_sets : ℕ := 3

def shoulder_press_weight1 : ℕ := 30
def shoulder_press_weight2 : ℕ := 40
def shoulder_press_reps : ℕ := 8
def shoulder_press_sets : ℕ := 2

def lunges_weight : ℕ := 30
def lunges_dumbbells : ℕ := 2
def lunges_reps : ℕ := 12
def lunges_sets : ℕ := 4

def bench_press_weight : ℕ := 40
def bench_press_dumbbells : ℕ := 2
def bench_press_reps : ℕ := 6
def bench_press_sets : ℕ := 3

-- Define the theorem
theorem workout_calculation :
  -- Total weight calculation
  (bicep_curls_weight * bicep_curls_dumbbells * bicep_curls_reps * bicep_curls_sets) +
  ((shoulder_press_weight1 + shoulder_press_weight2) * shoulder_press_reps * shoulder_press_sets) +
  (lunges_weight * lunges_dumbbells * lunges_reps * lunges_sets) +
  (bench_press_weight * bench_press_dumbbells * bench_press_reps * bench_press_sets) = 6640 ∧
  -- Average weight per rep for each exercise
  bicep_curls_weight * bicep_curls_dumbbells = 40 ∧
  shoulder_press_weight1 + shoulder_press_weight2 = 70 ∧
  lunges_weight * lunges_dumbbells = 60 ∧
  bench_press_weight * bench_press_dumbbells = 80 := by
  sorry

end NUMINAMATH_CALUDE_workout_calculation_l1839_183964


namespace NUMINAMATH_CALUDE_max_product_with_constraints_l1839_183978

theorem max_product_with_constraints :
  ∀ a b : ℕ,
  a + b = 100 →
  a % 3 = 2 →
  b % 7 = 5 →
  ∀ x y : ℕ,
  x + y = 100 →
  x % 3 = 2 →
  y % 7 = 5 →
  a * b ≤ 2491 ∧ (∃ a b : ℕ, a + b = 100 ∧ a % 3 = 2 ∧ b % 7 = 5 ∧ a * b = 2491) :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_with_constraints_l1839_183978


namespace NUMINAMATH_CALUDE_train_crossing_time_l1839_183983

/-- Proves that a train 130 m long, moving at 144 km/hr, takes 3.25 seconds to cross an electric pole -/
theorem train_crossing_time : 
  let train_length : ℝ := 130 -- meters
  let train_speed_kmh : ℝ := 144 -- km/hr
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600 -- Convert km/hr to m/s
  let crossing_time : ℝ := train_length / train_speed_ms
  crossing_time = 3.25 := by
sorry


end NUMINAMATH_CALUDE_train_crossing_time_l1839_183983


namespace NUMINAMATH_CALUDE_factorial_10_mod_11_l1839_183921

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_10_mod_11 : factorial 10 % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_11_l1839_183921


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1839_183948

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1839_183948


namespace NUMINAMATH_CALUDE_max_value_of_g_l1839_183976

-- Define the interval [0,1]
def interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define the function y = ax
def f (a : ℝ) : ℝ → ℝ := λ x ↦ a * x

-- Define the function y = 3ax - 1
def g (a : ℝ) : ℝ → ℝ := λ x ↦ 3 * a * x - 1

-- State the theorem
theorem max_value_of_g (a : ℝ) :
  (∃ (max min : ℝ), (∀ x ∈ interval, f a x ≤ max) ∧
                    (∀ x ∈ interval, min ≤ f a x) ∧
                    max + min = 3) →
  (∃ max : ℝ, (∀ x ∈ interval, g a x ≤ max) ∧
              (∃ y ∈ interval, g a y = max) ∧
              max = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1839_183976


namespace NUMINAMATH_CALUDE_power_five_mod_six_l1839_183984

theorem power_five_mod_six : 5^2023 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_six_l1839_183984


namespace NUMINAMATH_CALUDE_valid_plates_count_l1839_183938

/-- The number of digits available (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters available (A-Z) -/
def num_letters : ℕ := 26

/-- A license plate is valid if it satisfies the given conditions -/
def is_valid_plate (plate : Fin 4 → Char) : Prop :=
  (plate 0).isDigit ∧
  (plate 1).isAlpha ∧
  (plate 2).isAlpha ∧
  (plate 3).isDigit ∧
  plate 0 = plate 3

/-- The number of valid license plates -/
def num_valid_plates : ℕ := num_digits * num_letters * num_letters

theorem valid_plates_count :
  num_valid_plates = 6760 :=
sorry

end NUMINAMATH_CALUDE_valid_plates_count_l1839_183938


namespace NUMINAMATH_CALUDE_y_never_perfect_square_l1839_183901

theorem y_never_perfect_square (x : ℕ) : ∃ (n : ℕ), (x^4 + 2*x^3 + 2*x^2 + 2*x + 1) ≠ n^2 := by
  sorry

end NUMINAMATH_CALUDE_y_never_perfect_square_l1839_183901


namespace NUMINAMATH_CALUDE_carl_reach_probability_l1839_183926

-- Define the lily pad setup
def num_pads : ℕ := 16
def predator_pads : List ℕ := [4, 7, 12]
def start_pad : ℕ := 0
def goal_pad : ℕ := 14

-- Define Carl's movement probabilities
def hop_prob : ℚ := 1/2
def leap_prob : ℚ := 1/2

-- Define a function to calculate the probability of reaching a specific pad
def reach_prob (pad : ℕ) : ℚ :=
  sorry

-- State the theorem
theorem carl_reach_probability :
  reach_prob goal_pad = 3/512 :=
sorry

end NUMINAMATH_CALUDE_carl_reach_probability_l1839_183926


namespace NUMINAMATH_CALUDE_extremum_condition_l1839_183946

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (fun x => Real.exp x + a * x) x > 0 ∧
   ∀ y : ℝ, (fun x => Real.exp x + a * x) y ≤ (fun x => Real.exp x + a * x) x) →
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_condition_l1839_183946


namespace NUMINAMATH_CALUDE_simplify_expression_l1839_183910

theorem simplify_expression (a b : ℝ) :
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1839_183910


namespace NUMINAMATH_CALUDE_power_difference_equality_l1839_183922

theorem power_difference_equality : 4^(2+4+6) - (4^2 + 4^4 + 4^6) = 16772848 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l1839_183922


namespace NUMINAMATH_CALUDE_min_value_a_l1839_183974

theorem min_value_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + 2*a*x + 1 ≥ 0) ↔ a ≥ -5/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1839_183974


namespace NUMINAMATH_CALUDE_irrational_sum_rational_irrational_l1839_183945

theorem irrational_sum_rational_irrational (π : ℝ) (h : Irrational π) : Irrational (5 + π) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sum_rational_irrational_l1839_183945


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1839_183918

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * i) / i = b + i → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1839_183918


namespace NUMINAMATH_CALUDE_cistern_emptied_l1839_183990

/-- Represents the emptying rate of a pipe in terms of fraction of cistern per minute -/
structure PipeRate where
  fraction : ℚ
  time : ℚ

/-- Calculates the rate at which a pipe empties a cistern -/
def emptyingRate (p : PipeRate) : ℚ :=
  p.fraction / p.time

/-- Calculates the total emptying rate of multiple pipes -/
def totalRate (pipes : List PipeRate) : ℚ :=
  pipes.map emptyingRate |> List.sum

/-- Theorem: Given the specified pipes and time, the entire cistern will be emptied -/
theorem cistern_emptied (pipeA pipeB pipeC : PipeRate) 
    (h1 : pipeA = { fraction := 3/4, time := 12 })
    (h2 : pipeB = { fraction := 1/2, time := 15 })
    (h3 : pipeC = { fraction := 1/3, time := 10 })
    (time : ℚ)
    (h4 : time = 8) :
    totalRate [pipeA, pipeB, pipeC] * time ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_cistern_emptied_l1839_183990


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1839_183999

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  let R := r₁ + r₂
  let A_large := π * R^2
  let A_small₁ := π * r₁^2
  let A_small₂ := π * r₂^2
  let A_shaded := A_large - A_small₁ - A_small₂
  A_shaded = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1839_183999


namespace NUMINAMATH_CALUDE_one_student_reviewed_l1839_183955

/-- Represents the students in the problem -/
inductive Student : Type
  | Zhang
  | Li
  | Wang
  | Zhao
  | Liu

/-- The statement made by each student about how many reviewed math -/
def statement (s : Student) : Nat :=
  match s with
  | Student.Zhang => 0
  | Student.Li => 1
  | Student.Wang => 2
  | Student.Zhao => 3
  | Student.Liu => 4

/-- Predicate to determine if a student reviewed math -/
def reviewed : Student → Prop := sorry

/-- The number of students who reviewed math -/
def num_reviewed : Nat := sorry

theorem one_student_reviewed :
  (∃ s : Student, reviewed s) ∧
  (∃ s : Student, ¬reviewed s) ∧
  (∀ s : Student, reviewed s ↔ statement s = num_reviewed) ∧
  (num_reviewed = 1) := by sorry

end NUMINAMATH_CALUDE_one_student_reviewed_l1839_183955


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1839_183912

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (Q a ⊂ P ∧ Q a ≠ P) ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1839_183912


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l1839_183904

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a - x^2) / Real.exp x

theorem f_monotonicity_and_range (a : ℝ) :
  (a ≤ -1/2 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > -1/2 → ∀ x y : ℝ,
    ((x < y ∧ y < 1 - Real.sqrt (2 * a + 1)) ∨
     (x > 1 + Real.sqrt (2 * a + 1) ∧ y > x)) →
    f a x < f a y) ∧
  (a > -1/2 → ∀ x y : ℝ,
    (x > 1 - Real.sqrt (2 * a + 1) ∧ y < 1 + Real.sqrt (2 * a + 1) ∧ x < y) →
    f a x > f a y) ∧
  ((∀ x : ℝ, x ≥ 1 → f a x > -1) → a > (1 - Real.exp 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l1839_183904


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l1839_183907

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let reciprocal (q : ℚ) : ℚ := 1 / q
  reciprocal x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l1839_183907


namespace NUMINAMATH_CALUDE_error_percentage_l1839_183985

theorem error_percentage (x : ℝ) (h : x > 0) :
  ∃ ε > 0, abs ((x^2 - x/8) / x^2 * 100 - 88) < ε :=
sorry

end NUMINAMATH_CALUDE_error_percentage_l1839_183985


namespace NUMINAMATH_CALUDE_annalise_purchase_l1839_183919

/-- Represents the purchase of tissue boxes -/
structure TissuePurchase where
  packs_per_box : ℕ
  tissues_per_pack : ℕ
  tissue_cost_cents : ℕ
  total_spent_dollars : ℕ

/-- Calculates the number of boxes bought given a TissuePurchase -/
def boxes_bought (purchase : TissuePurchase) : ℕ :=
  (purchase.total_spent_dollars * 100) /
  (purchase.packs_per_box * purchase.tissues_per_pack * purchase.tissue_cost_cents)

/-- Theorem stating that Annalise bought 10 boxes -/
theorem annalise_purchase : 
  let purchase := TissuePurchase.mk 20 100 5 1000
  boxes_bought purchase = 10 := by
  sorry

end NUMINAMATH_CALUDE_annalise_purchase_l1839_183919


namespace NUMINAMATH_CALUDE_tangent_line_at_2_and_through_A_l1839_183908

/-- The function f(x) = x³ - 4x² + 5x - 4 -/
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

theorem tangent_line_at_2_and_through_A :
  /- Tangent line at X=2 -/
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ x - y - 4 = 0 ∧ 
    m = f' 2 ∧ -2 = m*2 + b) ∧ 
  /- Tangent lines through A(2,-2) -/
  (∃ (a : ℝ), 
    (∀ x y, y = -2 ↔ f a = f' a * (x - a) + f a ∧ -2 = f' a * (2 - a) + f a) ∨
    (∀ x y, x - y - 4 = 0 ↔ f a = f' a * (x - a) + f a ∧ -2 = f' a * (2 - a) + f a)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_and_through_A_l1839_183908


namespace NUMINAMATH_CALUDE_five_a_value_l1839_183969

theorem five_a_value (a : ℝ) (h : 5 * (a - 3) = 25) : 5 * a = 40 := by
  sorry

end NUMINAMATH_CALUDE_five_a_value_l1839_183969


namespace NUMINAMATH_CALUDE_sequence_existence_l1839_183961

theorem sequence_existence (a b : ℕ) (h1 : b > a) (h2 : a > 1) (h3 : ¬(a ∣ b))
  (b_seq : ℕ → ℕ) (h4 : ∀ n, b_seq (n + 1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ,
    (∀ n, a_seq (n + 1) - a_seq n = a ∨ a_seq (n + 1) - a_seq n = b) ∧
    (∀ m l, a_seq m + a_seq l ∉ Set.range b_seq) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l1839_183961


namespace NUMINAMATH_CALUDE_sum_of_roots_of_quartic_l1839_183903

theorem sum_of_roots_of_quartic (x : ℝ) : 
  (∃ a b c d : ℝ, x^4 - 6*x^3 + 8*x - 3 = (x^2 + a*x + b)*(x^2 + c*x + d)) →
  (∃ r₁ r₂ r₃ r₄ : ℝ, x^4 - 6*x^3 + 8*x - 3 = (x - r₁)*(x - r₂)*(x - r₃)*(x - r₄) ∧ r₁ + r₂ + r₃ + r₄ = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_quartic_l1839_183903


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l1839_183995

/-- The circle and parabola intersect at exactly one point if and only if b = 1/4 -/
theorem circle_parabola_intersection (b : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 4*b^2 ∧ p.2 = p.1^2 - 2*b) ↔ b = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l1839_183995


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1839_183987

theorem arithmetic_mean_problem (x : ℝ) :
  (x + 3*x + 1000 + 3000) / 4 = 2018 ↔ x = 1018 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1839_183987


namespace NUMINAMATH_CALUDE_grunters_win_probability_l1839_183997

theorem grunters_win_probability (n : ℕ) (p : ℚ) (h : p = 3/5) :
  p^n = 243/3125 → n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l1839_183997


namespace NUMINAMATH_CALUDE_limit_proof_l1839_183970

/-- The limit of (3x^2 + 5x - 2) / (x + 2) as x approaches -2 is -7 -/
theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -2 → |x + 2| < δ →
    |(3*x^2 + 5*x - 2) / (x + 2) + 7| < ε :=
by
  use ε/3
  sorry

end NUMINAMATH_CALUDE_limit_proof_l1839_183970


namespace NUMINAMATH_CALUDE_twenty_is_forty_percent_l1839_183915

theorem twenty_is_forty_percent : ∃ x : ℝ, x = 55 ∧ 20 / (x - 5) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_forty_percent_l1839_183915


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1839_183965

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 400)
  (h2 : first_discount = 10)
  (h3 : final_price = 342) :
  let price_after_first_discount := original_price * (1 - first_discount / 100)
  let second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100
  second_discount = 5 := by
sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1839_183965


namespace NUMINAMATH_CALUDE_interval_bound_l1839_183951

theorem interval_bound (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_interval_bound_l1839_183951


namespace NUMINAMATH_CALUDE_polynomial_root_nature_l1839_183940

def P (x : ℝ) : ℝ := x^6 - 5*x^5 - 7*x^3 - 2*x + 9

theorem polynomial_root_nature :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) :=
sorry

end NUMINAMATH_CALUDE_polynomial_root_nature_l1839_183940


namespace NUMINAMATH_CALUDE_angle_at_5pm_l1839_183981

/-- The angle between the hour and minute hands of a clock at a given hour -/
def clockAngle (hour : ℝ) : ℝ := 30 * hour

/-- Proposition: The angle between the minute hand and hour hand is 150° at 5 pm -/
theorem angle_at_5pm : clockAngle 5 = 150 := by sorry

end NUMINAMATH_CALUDE_angle_at_5pm_l1839_183981


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1839_183920

/-- Given two points P and Q that are symmetric with respect to the origin,
    prove that the sum of their x-coordinates plus the difference of their y-coordinates is zero. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (m - 1, 5) ∧ Q = (3, 2 - n) ∧ P = (-Q.1, -Q.2)) →
  m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1839_183920


namespace NUMINAMATH_CALUDE_snowflake_puzzle_solution_l1839_183954

-- Define the grid as a 3x3 matrix
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define the valid numbers
def ValidNumbers : List Nat := [1, 2, 3, 4, 5, 6]

-- Define the function to check if a number is valid in a given position
def isValidPlacement (grid : Grid) (row col : Fin 3) (num : Nat) : Prop :=
  -- Check row
  (∀ j : Fin 3, j ≠ col → grid row j ≠ num) ∧
  -- Check column
  (∀ i : Fin 3, i ≠ row → grid i col ≠ num) ∧
  -- Check diagonal (if applicable)
  (row = col → ∀ i : Fin 3, i ≠ row → grid i i ≠ num) ∧
  (row + col = 2 → ∀ i : Fin 3, grid i (2 - i) ≠ num)

-- Define the partially filled grid (Figure 2)
def initialGrid : Grid := λ i j =>
  if i = 0 ∧ j = 0 then 3
  else if i = 2 ∧ j = 2 then 4
  else 0  -- 0 represents an empty cell

-- Define the positions of A, B, C, D
def posA : Fin 3 × Fin 3 := (0, 1)
def posB : Fin 3 × Fin 3 := (1, 0)
def posC : Fin 3 × Fin 3 := (1, 1)
def posD : Fin 3 × Fin 3 := (1, 2)

-- Theorem statement
theorem snowflake_puzzle_solution :
  ∀ (grid : Grid),
    (∀ i j, grid i j ∈ ValidNumbers ∪ {0}) →
    (∀ i j, initialGrid i j ≠ 0 → grid i j = initialGrid i j) →
    (∀ i j, grid i j ≠ 0 → isValidPlacement grid i j (grid i j)) →
    (grid posA.1 posA.2 = 2 ∧
     grid posB.1 posB.2 = 5 ∧
     grid posC.1 posC.2 = 1 ∧
     grid posD.1 posD.2 = 6) :=
  sorry

end NUMINAMATH_CALUDE_snowflake_puzzle_solution_l1839_183954


namespace NUMINAMATH_CALUDE_fixed_point_of_power_plus_one_l1839_183933

/-- The function f(x) = x^n + 1 has a fixed point at (1, 2) for any positive integer n. -/
theorem fixed_point_of_power_plus_one (n : ℕ+) :
  let f : ℝ → ℝ := fun x ↦ x^(n : ℕ) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_power_plus_one_l1839_183933


namespace NUMINAMATH_CALUDE_vector_decomposition_l1839_183906

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![-1, 7, 0]
def p : Fin 3 → ℝ := ![0, 3, 1]
def q : Fin 3 → ℝ := ![1, -1, 2]
def r : Fin 3 → ℝ := ![2, -1, 0]

/-- Theorem stating the decomposition of x in terms of p and q -/
theorem vector_decomposition : x = 2 • p - q := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1839_183906


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1839_183929

theorem book_arrangement_theorem :
  let total_books : ℕ := 8
  let advanced_geometry_copies : ℕ := 5
  let essential_number_theory_copies : ℕ := 3
  total_books = advanced_geometry_copies + essential_number_theory_copies →
  (Nat.choose total_books advanced_geometry_copies) = 56 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1839_183929


namespace NUMINAMATH_CALUDE_donut_combinations_l1839_183972

/-- The number of ways to choose k items from n types with repetition. -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of donut types available. -/
def num_donut_types : ℕ := 6

/-- The number of remaining donuts to be chosen. -/
def remaining_donuts : ℕ := 2

/-- The total number of donuts in the order. -/
def total_donuts : ℕ := 8

/-- The number of donuts already accounted for (2 each of 3 specific kinds). -/
def accounted_donuts : ℕ := 6

theorem donut_combinations :
  choose_with_repetition num_donut_types remaining_donuts = 21 ∧
  total_donuts = accounted_donuts + remaining_donuts :=
by sorry

end NUMINAMATH_CALUDE_donut_combinations_l1839_183972


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1839_183998

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 32127 → ¬(510 ∣ (m + 3) ∧ 4590 ∣ (m + 3) ∧ 105 ∣ (m + 3))) ∧
  (510 ∣ (32127 + 3) ∧ 4590 ∣ (32127 + 3) ∧ 105 ∣ (32127 + 3)) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1839_183998


namespace NUMINAMATH_CALUDE_dodecagon_ratio_l1839_183902

/-- Represents a dodecagon with specific properties -/
structure Dodecagon where
  /-- Total area of the dodecagon -/
  total_area : ℝ
  /-- Area below the bisecting line PQ -/
  area_below_pq : ℝ
  /-- Base of the triangle below PQ -/
  triangle_base : ℝ
  /-- Width of the dodecagon (XQ + QY) -/
  width : ℝ
  /-- Assertion that the dodecagon is made of 12 unit squares -/
  area_is_twelve : total_area = 12
  /-- Assertion that PQ bisects the area -/
  pq_bisects : area_below_pq = total_area / 2
  /-- Assertion about the composition below PQ -/
  below_pq_composition : area_below_pq = 2 + (triangle_base * triangle_base / 12)
  /-- Assertion about the width of the dodecagon -/
  width_is_six : width = 6

/-- Theorem stating that for a dodecagon with given properties, XQ/QY = 2 -/
theorem dodecagon_ratio (d : Dodecagon) : ∃ (xq qy : ℝ), xq / qy = 2 ∧ xq + qy = d.width := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_ratio_l1839_183902


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l1839_183977

theorem factorization_of_difference_of_squares (a b : ℝ) :
  36 * a^2 - 4 * b^2 = 4 * (3*a + b) * (3*a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l1839_183977


namespace NUMINAMATH_CALUDE_decimal_period_11_13_l1839_183916

/-- The length of the smallest repeating block in the decimal expansion of a rational number -/
def decimal_period (n d : ℕ) : ℕ :=
  sorry

/-- Theorem: The length of the smallest repeating block in the decimal expansion of 11/13 is 6 -/
theorem decimal_period_11_13 : decimal_period 11 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_decimal_period_11_13_l1839_183916


namespace NUMINAMATH_CALUDE_divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n_l1839_183932

theorem divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n (n : ℕ) :
  let m := ⌈(Real.sqrt 3 + 1)^(2*n)⌉
  ∃ k : ℕ, m = 2^(n+1) * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n_l1839_183932


namespace NUMINAMATH_CALUDE_marias_water_bottles_l1839_183935

theorem marias_water_bottles (initial bottles_drunk final : ℕ) 
  (h1 : initial = 14)
  (h2 : bottles_drunk = 8)
  (h3 : final = 51) :
  final - (initial - bottles_drunk) = 45 := by
  sorry

end NUMINAMATH_CALUDE_marias_water_bottles_l1839_183935


namespace NUMINAMATH_CALUDE_no_solution_exists_l1839_183992

theorem no_solution_exists (a c : ℝ) : ¬∃ x : ℝ, 
  ((a + x) / 2 = 110) ∧ 
  ((x + c) / 2 = 170) ∧ 
  (a - c = 120) := by
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1839_183992


namespace NUMINAMATH_CALUDE_distribute_negative_five_l1839_183956

theorem distribute_negative_five (x y : ℝ) : -5 * (x - y) = -5 * x + 5 * y := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_five_l1839_183956


namespace NUMINAMATH_CALUDE_sum_100th_group_value_l1839_183962

/-- The sum of the three numbers in the 100th group of the sequence (n, n^2, n^3) -/
def sum_100th_group : ℕ := 100 + 100^2 + 100^3

/-- Theorem stating that the sum of the 100th group is 1010100 -/
theorem sum_100th_group_value : sum_100th_group = 1010100 := by
  sorry

end NUMINAMATH_CALUDE_sum_100th_group_value_l1839_183962


namespace NUMINAMATH_CALUDE_domain_transformation_l1839_183982

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem domain_transformation (h : Set.Icc (-3 : ℝ) 3 = {x | ∃ y, f (2*y - 1) = x}) :
  {x | ∃ y, f y = x} = Set.Icc (-7 : ℝ) 5 := by
  sorry

end NUMINAMATH_CALUDE_domain_transformation_l1839_183982


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l1839_183917

theorem framed_painting_ratio :
  ∀ (x : ℝ),
  x > 0 →
  (20 + 2*x) * (30 + 6*x) = 1800 →
  (20 + 2*x) / (30 + 6*x) = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l1839_183917


namespace NUMINAMATH_CALUDE_inequality_proof_l1839_183949

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1/3) ∧ 
  (b^2 / a + c^2 / b + a^2 / c ≥ 1) := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l1839_183949


namespace NUMINAMATH_CALUDE_complement_M_equals_interval_l1839_183953

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x : ℝ | (2 - x) / (x + 3) < 0}

-- Define the complement of M in ℝ
def complement_M : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem complement_M_equals_interval : 
  (U \ M) = complement_M :=
sorry

end NUMINAMATH_CALUDE_complement_M_equals_interval_l1839_183953


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1839_183963

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - (k + 1) * x - 6 = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 - (k + 1) * y - 6 = 0 ∧ y = -3 ∧ k = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1839_183963


namespace NUMINAMATH_CALUDE_sine_symmetry_sum_l1839_183930

open Real

theorem sine_symmetry_sum (α β : ℝ) :
  0 ≤ α ∧ α < π ∧
  0 ≤ β ∧ β < π ∧
  α ≠ β ∧
  sin (2 * α + π / 3) = 1 / 2 ∧
  sin (2 * β + π / 3) = 1 / 2 →
  α + β = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_sine_symmetry_sum_l1839_183930


namespace NUMINAMATH_CALUDE_sequence_sum_l1839_183925

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d →
  b - a = c - b →
  c * c = b * d →
  d - a = 30 →
  a + b + c + d = 129 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1839_183925


namespace NUMINAMATH_CALUDE_men_in_room_l1839_183950

theorem men_in_room (x : ℕ) 
  (h1 : 2 * (5 * x - 3) = 24) -- Women doubled and final count is 24
  : 4 * x + 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_men_in_room_l1839_183950


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1839_183937

/-- The volume of a sphere with surface area 8π is equal to (8 * sqrt(2) * π) / 3 -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 8 * π → (4 / 3) * π * r^3 = (8 * Real.sqrt 2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1839_183937


namespace NUMINAMATH_CALUDE_product_first_three_terms_l1839_183911

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference between consecutive terms
  d : ℝ
  -- The seventh term is 20
  seventh_term : a + 6 * d = 20
  -- The common difference is 2
  common_diff : d = 2

/-- The product of the first three terms of the arithmetic sequence is 960 -/
theorem product_first_three_terms (seq : ArithmeticSequence) :
  seq.a * (seq.a + seq.d) * (seq.a + 2 * seq.d) = 960 := by
  sorry


end NUMINAMATH_CALUDE_product_first_three_terms_l1839_183911


namespace NUMINAMATH_CALUDE_harmonic_sum_inequality_l1839_183975

theorem harmonic_sum_inequality : 1 + 1/2 + 1/3 < 2 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_inequality_l1839_183975


namespace NUMINAMATH_CALUDE_half_red_probability_l1839_183909

def num_balls : ℕ := 8
def num_red : ℕ := 4

theorem half_red_probability :
  let p_red : ℚ := 1 / 2
  let p_event : ℚ := (num_balls.choose num_red : ℚ) * p_red ^ num_balls
  p_event = 35 / 128 := by sorry

end NUMINAMATH_CALUDE_half_red_probability_l1839_183909


namespace NUMINAMATH_CALUDE_train_crossing_platform_time_l1839_183968

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_platform_time 
  (train_length : ℝ) 
  (signal_pole_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 200) 
  (h2 : signal_pole_time = 42) 
  (h3 : platform_length = 38.0952380952381) : 
  (train_length + platform_length) / (train_length / signal_pole_time) = 50 := by
  sorry

#check train_crossing_platform_time

end NUMINAMATH_CALUDE_train_crossing_platform_time_l1839_183968


namespace NUMINAMATH_CALUDE_cost_price_is_60_l1839_183959

/-- The cost price of a single ball, given the selling price of multiple balls and the loss incurred. -/
def cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) : ℕ :=
  selling_price / (num_balls_sold - num_balls_loss)

/-- Theorem stating that the cost price of a ball is 60 under given conditions. -/
theorem cost_price_is_60 :
  cost_price_of_ball 720 17 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_60_l1839_183959


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l1839_183924

/-- The number of meters of cloth sold by a trader -/
def meters_of_cloth : ℕ := 85

/-- The total selling price in dollars -/
def total_selling_price : ℕ := 8925

/-- The profit per meter of cloth in dollars -/
def profit_per_meter : ℕ := 15

/-- The cost price per meter of cloth in dollars -/
def cost_price_per_meter : ℕ := 90

/-- Theorem stating that the number of meters of cloth sold is correct -/
theorem cloth_sale_calculation :
  meters_of_cloth * (cost_price_per_meter + profit_per_meter) = total_selling_price :=
by sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l1839_183924


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1839_183936

theorem square_perimeter_relation (C D : Real) : 
  (C = 16) → -- perimeter of square C is 16 cm
  (D^2 = (C/4)^2 / 3) → -- area of D is one-third the area of C
  (4 * D = 16 * Real.sqrt 3 / 3) -- perimeter of D is 16√3/3 cm
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1839_183936


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1839_183971

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : a 1 > 0) :
  (is_increasing_sequence a → a 1 ^ 2 < a 2 ^ 2) ∧
  ¬(a 1 ^ 2 < a 2 ^ 2 → is_increasing_sequence a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1839_183971


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l1839_183973

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 56 := by sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l1839_183973


namespace NUMINAMATH_CALUDE_spider_eats_all_flies_l1839_183957

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the spider's movement strategy -/
structure SpiderStrategy where
  initialPosition : Position
  moveSequence : List Position

/-- Represents the web with flies -/
structure Web where
  size : Nat
  flyPositions : List Position

/-- Theorem stating that the spider can eat all flies in at most 1980 moves -/
theorem spider_eats_all_flies (web : Web) (strategy : SpiderStrategy) : 
  web.size = 100 → 
  web.flyPositions.length = 100 → 
  strategy.initialPosition.x = 0 ∨ strategy.initialPosition.x = 99 → 
  strategy.initialPosition.y = 0 ∨ strategy.initialPosition.y = 99 → 
  ∃ (moves : List Position), 
    moves.length ≤ 1980 ∧ 
    (∀ fly ∈ web.flyPositions, fly ∈ moves) := by
  sorry

end NUMINAMATH_CALUDE_spider_eats_all_flies_l1839_183957


namespace NUMINAMATH_CALUDE_sum_reciprocals_S_l1839_183943

def S : Set ℕ+ := {n : ℕ+ | ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 2017}

theorem sum_reciprocals_S : ∑' (s : S), (1 : ℝ) / (s : ℝ) = 2017 / 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_S_l1839_183943


namespace NUMINAMATH_CALUDE_chip_rearrangement_l1839_183952

/-- Represents a color of a chip -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a position in the rectangle -/
structure Position where
  row : Fin 3
  col : Nat

/-- Represents the state of the rectangle -/
def Rectangle (n : Nat) := Position → Color

/-- Checks if a given rectangle arrangement is valid -/
def isValidArrangement (n : Nat) (rect : Rectangle n) : Prop :=
  ∀ c : Color, ∀ i : Fin 3, ∃ j : Fin n, rect ⟨i, j⟩ = c

/-- Checks if a given rectangle arrangement satisfies the condition -/
def satisfiesCondition (n : Nat) (rect : Rectangle n) : Prop :=
  ∀ j : Fin n, ∀ c : Color, ∃ i : Fin 3, rect ⟨i, j⟩ = c

/-- The main theorem to be proved -/
theorem chip_rearrangement (n : Nat) :
  ∃ (rect : Rectangle n), isValidArrangement n rect ∧ satisfiesCondition n rect := by
  sorry


end NUMINAMATH_CALUDE_chip_rearrangement_l1839_183952


namespace NUMINAMATH_CALUDE_f_ratio_calc_l1839_183988

axiom f : ℝ → ℝ

axiom f_property : ∀ (a b : ℝ), b^2 * f a = a^2 * f b

axiom f2_nonzero : f 2 ≠ 0

theorem f_ratio_calc : (f 6 - f 3) / f 2 = 27 / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_ratio_calc_l1839_183988


namespace NUMINAMATH_CALUDE_ten_point_square_impossibility_l1839_183958

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of ten points in a plane -/
def TenPoints := Fin 10 → Point

/-- Predicate to check if four points lie on the boundary of some square -/
def FourPointsOnSquare (p₁ p₂ p₃ p₄ : Point) : Prop := sorry

/-- Predicate to check if all points in a set lie on the boundary of some square -/
def AllPointsOnSquare (points : TenPoints) : Prop := sorry

/-- The main theorem -/
theorem ten_point_square_impossibility (points : TenPoints) 
  (h : ∀ (a b c d : Fin 10), a ≠ b → b ≠ c → c ≠ d → d ≠ a → 
    FourPointsOnSquare (points a) (points b) (points c) (points d)) :
  ¬ AllPointsOnSquare points :=
sorry

end NUMINAMATH_CALUDE_ten_point_square_impossibility_l1839_183958


namespace NUMINAMATH_CALUDE_toy_car_cost_l1839_183942

theorem toy_car_cost (initial_amount : ℕ) (num_cars : ℕ) (scarf_cost : ℕ) (beanie_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 53 →
  num_cars = 2 →
  scarf_cost = 10 →
  beanie_cost = 14 →
  remaining_amount = 7 →
  (initial_amount - remaining_amount - scarf_cost - beanie_cost) / num_cars = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_car_cost_l1839_183942


namespace NUMINAMATH_CALUDE_regular_triangular_prism_edge_length_l1839_183980

/-- A regular triangular prism with edge length a and volume 16√3 has a = 4 -/
theorem regular_triangular_prism_edge_length (a : ℝ) : 
  a > 0 →  -- Ensure a is positive
  (1/4 : ℝ) * a^3 * Real.sqrt 3 = 16 * Real.sqrt 3 → 
  a = 4 := by
  sorry

#check regular_triangular_prism_edge_length

end NUMINAMATH_CALUDE_regular_triangular_prism_edge_length_l1839_183980


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1839_183960

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1839_183960


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1839_183994

theorem decimal_multiplication : (0.8 : ℝ) * 0.12 = 0.096 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1839_183994


namespace NUMINAMATH_CALUDE_fifteen_clockwise_opposite_l1839_183905

/-- Represents a circle of equally spaced children -/
structure ChildrenCircle where
  num_children : ℕ
  standard_child : ℕ

/-- The child directly opposite another child in the circle -/
def opposite_child (circle : ChildrenCircle) (child : ℕ) : ℕ :=
  (child + circle.num_children / 2) % circle.num_children

theorem fifteen_clockwise_opposite (circle : ChildrenCircle) :
  opposite_child circle circle.standard_child = (circle.standard_child + 15) % circle.num_children →
  circle.num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_clockwise_opposite_l1839_183905


namespace NUMINAMATH_CALUDE_determinant_of_trigonometric_matrix_l1839_183993

theorem determinant_of_trigonometric_matrix (α β γ : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, -Real.sin (α + γ)],
    ![-Real.sin β, Real.cos β * Real.cos γ, Real.sin β * Real.sin γ],
    ![Real.sin (α + γ) * Real.cos β, Real.sin (α + γ) * Real.sin β, Real.cos (α + γ)]
  ]
  Matrix.det M = 1 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_trigonometric_matrix_l1839_183993


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1839_183947

theorem quadratic_root_problem (k : ℤ) (b c : ℤ) (h1 : k > 9) 
  (h2 : k^2 - b*k + c = 0) (h3 : b = 2*k + 1) 
  (h4 : (k-7)^2 - b*(k-7) + c = 0) : c = 3*k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1839_183947


namespace NUMINAMATH_CALUDE_deceased_member_income_l1839_183966

/-- Given a family with 4 earning members and an average monthly income,
    calculate the income of a deceased member when the average income changes. -/
theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average_income : ℚ)
  (final_members : ℕ)
  (final_average_income : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = initial_members - 1)
  (h3 : initial_average_income = 840)
  (h4 : final_average_income = 650) :
  (initial_members : ℚ) * initial_average_income - (final_members : ℚ) * final_average_income = 1410 :=
by sorry

end NUMINAMATH_CALUDE_deceased_member_income_l1839_183966


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1839_183986

theorem remainder_divisibility (N : ℤ) : 
  ∃ k : ℤ, N = 45 * k + 31 → ∃ m : ℤ, N = 15 * m + 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1839_183986


namespace NUMINAMATH_CALUDE_sock_ratio_proof_l1839_183923

theorem sock_ratio_proof (green_socks red_socks : ℕ) (price_red : ℚ) :
  green_socks = 6 →
  (6 * (3 * price_red) + red_socks * price_red + 15 : ℚ) * (9/5) = 
    red_socks * (3 * price_red) + 6 * price_red + 15 →
  (green_socks : ℚ) / red_socks = 6 / 23 :=
by
  sorry

end NUMINAMATH_CALUDE_sock_ratio_proof_l1839_183923


namespace NUMINAMATH_CALUDE_max_peak_consumption_for_savings_l1839_183927

/-- Proves that the maximum average monthly electricity consumption during peak hours
    that allows for at least 10% savings on the original electricity cost is ≤ 118 kWh --/
theorem max_peak_consumption_for_savings (
  original_price : ℝ) (peak_price : ℝ) (off_peak_price : ℝ) (total_consumption : ℝ) 
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : 0 < original_price ∧ 0 < peak_price ∧ 0 < off_peak_price)
  (h6 : total_consumption > 0) :
  let peak_consumption := 
    { x : ℝ | x ≥ 0 ∧ x ≤ total_consumption ∧ 
      (peak_price * x + off_peak_price * (total_consumption - x)) ≤ 
      0.9 * (original_price * total_consumption) }
  ∃ max_peak : ℝ, max_peak ∈ peak_consumption ∧ max_peak ≤ 118 ∧ 
    ∀ y ∈ peak_consumption, y ≤ max_peak := by
  sorry

#check max_peak_consumption_for_savings

end NUMINAMATH_CALUDE_max_peak_consumption_for_savings_l1839_183927


namespace NUMINAMATH_CALUDE_least_integer_x_l1839_183914

theorem least_integer_x : ∃ x : ℤ, (∀ z : ℤ, |3*z + 5 - 4| ≤ 25 → x ≤ z) ∧ |3*x + 5 - 4| ≤ 25 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_x_l1839_183914


namespace NUMINAMATH_CALUDE_three_digit_puzzle_l1839_183967

theorem three_digit_puzzle :
  ∀ (A B C : ℕ),
  (A ≥ 1 ∧ A ≤ 9) →
  (B ≥ 0 ∧ B ≤ 9) →
  (C ≥ 0 ∧ C ≤ 9) →
  (100 * A + 10 * B + B ≥ 100 ∧ 100 * A + 10 * B + B ≤ 999) →
  (A * B * B ≥ 10 ∧ A * B * B ≤ 99) →
  A * B * B = 10 * A + C →
  A * C = C →
  100 * A + 10 * B + B = 144 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_puzzle_l1839_183967


namespace NUMINAMATH_CALUDE_field_of_miracles_l1839_183939

/-- The Field of Miracles problem -/
theorem field_of_miracles
  (a b : ℝ)
  (ha : a = 6)
  (hb : b = 2.5)
  (v_malvina : ℝ)
  (hv_malvina : v_malvina = 4)
  (v_buratino : ℝ)
  (hv_buratino : v_buratino = 6)
  (v_artemon : ℝ)
  (hv_artemon : v_artemon = 12) :
  let d := Real.sqrt (a^2 + b^2)
  let t := d / (v_malvina + v_buratino)
  v_artemon * t = 7.8 :=
by sorry

end NUMINAMATH_CALUDE_field_of_miracles_l1839_183939


namespace NUMINAMATH_CALUDE_unit_conversions_l1839_183934

-- Define conversion rates
def kgToGrams : ℚ → ℚ := (· * 1000)
def meterToDecimeter : ℚ → ℚ := (· * 10)

-- Theorem statement
theorem unit_conversions :
  (kgToGrams 4 = 4000) ∧
  (meterToDecimeter 3 - 2 = 28) ∧
  (meterToDecimeter 8 = 80) ∧
  ((1600 : ℚ) - 600 = kgToGrams 1) :=
by sorry

end NUMINAMATH_CALUDE_unit_conversions_l1839_183934


namespace NUMINAMATH_CALUDE_soccer_attendance_difference_l1839_183979

theorem soccer_attendance_difference (seattle_estimate chicago_estimate : ℕ) 
  (seattle_actual chicago_actual : ℝ) : 
  seattle_estimate = 40000 →
  chicago_estimate = 50000 →
  seattle_actual ≥ 0.85 * seattle_estimate ∧ seattle_actual ≤ 1.15 * seattle_estimate →
  chicago_actual ≥ chicago_estimate / 1.15 ∧ chicago_actual ≤ chicago_estimate / 0.85 →
  ∃ (max_diff : ℕ), max_diff = 25000 ∧ 
    ∀ (diff : ℝ), diff = chicago_actual - seattle_actual → 
      diff ≤ max_diff ∧ 
      (max_diff - 500 < diff ∨ diff < max_diff + 500) :=
by sorry

end NUMINAMATH_CALUDE_soccer_attendance_difference_l1839_183979


namespace NUMINAMATH_CALUDE_last_k_digits_theorem_l1839_183944

theorem last_k_digits_theorem (k : ℕ) (h : k ≥ 2) :
  (∃ n : ℕ+, (10^(10^n.val) : ℤ) ≡ 9^(9^n.val) [ZMOD 10^k]) ↔ k ∈ ({2, 3, 4} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_last_k_digits_theorem_l1839_183944


namespace NUMINAMATH_CALUDE_semicircle_chord_length_l1839_183989

theorem semicircle_chord_length (R a b : ℝ) (h1 : R > 0) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b = R) (h5 : (π/2) * (R^2 - a^2 - b^2) = 10*π) : 
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_chord_length_l1839_183989


namespace NUMINAMATH_CALUDE_test_total_points_l1839_183900

/-- Given a test with the following properties:
  * Total number of questions is 30
  * Questions are either worth 5 or 10 points
  * There are 20 questions worth 5 points each
  Prove that the total point value of the test is 200 points -/
theorem test_total_points :
  ∀ (total_questions five_point_questions : ℕ)
    (point_values : Finset ℕ),
  total_questions = 30 →
  five_point_questions = 20 →
  point_values = {5, 10} →
  (total_questions - five_point_questions) * 10 + five_point_questions * 5 = 200 :=
by sorry

end NUMINAMATH_CALUDE_test_total_points_l1839_183900


namespace NUMINAMATH_CALUDE_problem_solution_l1839_183913

theorem problem_solution (x y : ℝ) 
  (h1 : x^2 + x*y = 3) 
  (h2 : x*y + y^2 = -2) : 
  2*x^2 - x*y - 3*y^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1839_183913


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1839_183931

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x > y - 1) ∧
  (∃ x y : ℝ, x > y - 1 ∧ ¬(x > y)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1839_183931


namespace NUMINAMATH_CALUDE_consecutive_integers_product_not_square_l1839_183991

theorem consecutive_integers_product_not_square (a : ℕ) : 
  let A := Finset.range 20
  let sum := A.sum (λ i => a + i)
  let prod := A.prod (λ i => a + i)
  (sum % 23 ≠ 0) → (prod % 23 ≠ 0) → ¬ ∃ (n : ℕ), prod = n^2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_not_square_l1839_183991


namespace NUMINAMATH_CALUDE_expenditure_recording_l1839_183996

/-- Given that income is recorded as positive and an income of 20 yuan is recorded as +20 yuan,
    prove that an expenditure of 75 yuan should be recorded as -75 yuan. -/
theorem expenditure_recording (income_recording : ℤ → ℤ) (h : income_recording 20 = 20) :
  income_recording (-75) = -75 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_recording_l1839_183996
