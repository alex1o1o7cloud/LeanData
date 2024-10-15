import Mathlib

namespace NUMINAMATH_CALUDE_vector_relations_l2917_291706

/-- Given vectors a, b, n, and c in ℝ², prove the values of k for perpendicularity and parallelism conditions. -/
theorem vector_relations (a b n c : ℝ × ℝ) (k : ℝ) : 
  a = (-3, 1) → 
  b = (1, -2) → 
  c = (1, -1) → 
  n = (a.1 + k * b.1, a.2 + k * b.2) → 
  (((n.1 * (2 * a.1 - b.1) + n.2 * (2 * a.2 - b.2) = 0) → k = 5/3) ∧ 
   ((n.1 * (c.1 + k * b.1) = n.2 * (c.2 + k * b.2)) → k = -1/3)) := by
  sorry


end NUMINAMATH_CALUDE_vector_relations_l2917_291706


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2917_291758

-- Define the conditions p and q
def p (x : ℝ) : Prop := (x - 1) * (x - 3) ≤ 0
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Define the set A satisfying condition p
def A : Set ℝ := {x | p x}

-- Define the set B satisfying condition q
def B : Set ℝ := {x | q x}

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2917_291758


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l2917_291762

theorem richmond_tigers_ticket_sales (total_tickets : ℕ) (first_half_tickets : ℕ) (second_half_tickets : ℕ) :
  total_tickets = 9570 →
  first_half_tickets = 3867 →
  second_half_tickets = total_tickets - first_half_tickets →
  second_half_tickets = 5703 :=
by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l2917_291762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2917_291774

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 1) →  -- arithmetic sequence with common difference 1
  (a 2 + a 4 + a 6 = 9) →       -- given condition
  (a 5 + a 7 + a 9 = 18) :=     -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2917_291774


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2917_291734

theorem factorial_divisibility (m n : ℕ) : 
  (Nat.factorial (2 * m) * Nat.factorial (2 * n)) % 
  (Nat.factorial m * Nat.factorial n * Nat.factorial (m + n)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2917_291734


namespace NUMINAMATH_CALUDE_eugene_pencils_l2917_291736

theorem eugene_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l2917_291736


namespace NUMINAMATH_CALUDE_suzy_book_count_l2917_291702

/-- Calculates the final number of books Suzy has after three days of transactions -/
def final_book_count (initial : ℕ) (wed_out : ℕ) (thu_in : ℕ) (thu_out : ℕ) (fri_in : ℕ) : ℕ :=
  initial - wed_out + thu_in - thu_out + fri_in

/-- Theorem stating that given the specific transactions, Suzy ends up with 80 books -/
theorem suzy_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

#eval final_book_count 98 43 23 5 7

end NUMINAMATH_CALUDE_suzy_book_count_l2917_291702


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l2917_291708

theorem ratio_of_percentages (P M N R : ℝ) : 
  P > 0 ∧ P = 0.3 * R ∧ M = 0.35 * R ∧ N = 0.55 * R → M / N = 7 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l2917_291708


namespace NUMINAMATH_CALUDE_unique_number_guess_l2917_291798

/-- Represents the color feedback for a digit guess -/
inductive Color
  | Green
  | Yellow
  | Gray

/-- Represents a single round of guessing -/
structure GuessRound where
  digits : Fin 5 → Nat
  colors : Fin 5 → Color

/-- The set of all possible digits (0-9) -/
def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The correct five-digit number we're trying to prove -/
def CorrectNumber : Fin 5 → Nat := ![7, 1, 2, 8, 4]

theorem unique_number_guess (round1 round2 round3 : GuessRound) : 
  (round1.digits = ![2, 6, 1, 3, 8] ∧ 
   round1.colors = ![Color.Yellow, Color.Gray, Color.Yellow, Color.Gray, Color.Yellow]) →
  (round2.digits = ![4, 1, 9, 6, 2] ∧
   round2.colors = ![Color.Yellow, Color.Green, Color.Gray, Color.Gray, Color.Yellow]) →
  (round3.digits = ![8, 1, 0, 2, 5] ∧
   round3.colors = ![Color.Yellow, Color.Green, Color.Gray, Color.Yellow, Color.Gray]) →
  (∀ n : Fin 5, CorrectNumber n ∈ Digits) →
  (∀ i j : Fin 5, i ≠ j → CorrectNumber i ≠ CorrectNumber j) →
  CorrectNumber = ![7, 1, 2, 8, 4] := by
  sorry


end NUMINAMATH_CALUDE_unique_number_guess_l2917_291798


namespace NUMINAMATH_CALUDE_product_when_c_is_one_l2917_291783

theorem product_when_c_is_one (a b c : ℕ+) (h1 : a * b * c = a * b ^ 3) (h2 : c = 1) :
  a * b * c = a := by
  sorry

end NUMINAMATH_CALUDE_product_when_c_is_one_l2917_291783


namespace NUMINAMATH_CALUDE_function_properties_l2917_291732

/-- The function f(x) = ax ln x - x^2 - 2x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x^2 - 2 * x

/-- The derivative of f(x) --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (1 + Real.log x) - 2 * x - 2

/-- The function g(x) = f(x) + 2x --/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 * x

theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) :
  (∃ (x : ℝ), f_deriv 4 x = 4 * Real.log 2 - 2 ∧
    ∀ (y : ℝ), f_deriv 4 y ≤ 4 * Real.log 2 - 2) ∧
  (g a x₁ = 0 ∧ g a x₂ = 0 ∧ x₂ / x₁ > Real.exp 1 →
    Real.log a + Real.log (x₁ * x₂) > 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2917_291732


namespace NUMINAMATH_CALUDE_max_area_triangle_line_circle_l2917_291713

/-- The maximum area of a triangle formed by the origin and two intersection points of a line and a unit circle --/
theorem max_area_triangle_line_circle : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line (k : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 - 1}
  let intersectionPoints (k : ℝ) := circle ∩ line k
  let triangleArea (A B : ℝ × ℝ) := (1/2) * abs (A.1 * B.2 - A.2 * B.1)
  ∀ k : ℝ, ∀ A B : ℝ × ℝ, A ∈ intersectionPoints k → B ∈ intersectionPoints k → A ≠ B →
    triangleArea A B ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_line_circle_l2917_291713


namespace NUMINAMATH_CALUDE_barbara_savings_weeks_l2917_291738

/-- Calculates the number of weeks needed to save for a wristwatch -/
def weeks_to_save (watch_cost : ℕ) (weekly_allowance : ℕ) (current_savings : ℕ) : ℕ :=
  ((watch_cost - current_savings) + weekly_allowance - 1) / weekly_allowance

/-- Proves that Barbara needs 16 more weeks to save for the watch -/
theorem barbara_savings_weeks :
  weeks_to_save 100 5 20 = 16 := by
sorry

end NUMINAMATH_CALUDE_barbara_savings_weeks_l2917_291738


namespace NUMINAMATH_CALUDE_factorization_problems_l2917_291756

theorem factorization_problems (m x y : ℝ) : 
  (m^3 - 2*m^2 - 4*m + 8 = (m-2)^2*(m+2)) ∧ 
  (x^2 - 2*x*y + y^2 - 9 = (x-y+3)*(x-y-3)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2917_291756


namespace NUMINAMATH_CALUDE_probability_cap_given_sunglasses_l2917_291742

/-- The number of people wearing sunglasses -/
def sunglasses_wearers : ℕ := 60

/-- The number of people wearing caps -/
def cap_wearers : ℕ := 40

/-- The number of people wearing both sunglasses and caps and hats -/
def triple_wearers : ℕ := 8

/-- The probability that a person wearing a cap is also wearing sunglasses -/
def prob_sunglasses_given_cap : ℚ := 1/2

theorem probability_cap_given_sunglasses :
  let both_wearers := cap_wearers * prob_sunglasses_given_cap
  (both_wearers : ℚ) / sunglasses_wearers = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_cap_given_sunglasses_l2917_291742


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2917_291740

-- Define the sets P and S
def P : Set ℝ := {x | x^2 + x - 6 = 0}
def S (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

-- Define the set of possible values for a
def A : Set ℝ := {0, 1/3, -1/2}

-- Theorem statement
theorem possible_values_of_a :
  ∀ a : ℝ, (S a ⊆ P) ↔ (a ∈ A) := by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2917_291740


namespace NUMINAMATH_CALUDE_factor_x6_minus_81_l2917_291773

theorem factor_x6_minus_81 (x : ℝ) : x^6 - 81 = (x^3 + 9) * (x - 3) * (x^2 + 3*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_81_l2917_291773


namespace NUMINAMATH_CALUDE_guanghua_community_households_l2917_291761

theorem guanghua_community_households (num_buildings : ℕ) (floors_per_building : ℕ) (households_per_floor : ℕ) 
  (h1 : num_buildings = 14)
  (h2 : floors_per_building = 7)
  (h3 : households_per_floor = 8) :
  num_buildings * floors_per_building * households_per_floor = 784 := by
  sorry

end NUMINAMATH_CALUDE_guanghua_community_households_l2917_291761


namespace NUMINAMATH_CALUDE_x_value_proof_l2917_291743

theorem x_value_proof (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((10 * x) / 3) = x) : x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2917_291743


namespace NUMINAMATH_CALUDE_circle_equations_valid_l2917_291750

-- Define the points
def M : ℝ × ℝ := (-1, 1)
def N : ℝ × ℝ := (0, 2)
def Q : ℝ × ℝ := (2, 0)

-- Define the equations of the circles
def circle_C1_eq (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1/2)^2 = 5/2
def circle_C2_eq (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 5/2)^2 = 5/2

-- Define the line MN
def line_MN_eq (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem circle_equations_valid :
  -- Circle C1 passes through M, N, and Q
  (circle_C1_eq M.1 M.2 ∧ circle_C1_eq N.1 N.2 ∧ circle_C1_eq Q.1 Q.2) ∧
  -- C2 is the reflection of C1 about line MN
  (∀ x y : ℝ, circle_C1_eq x y ↔ 
    ∃ x' y' : ℝ, circle_C2_eq x' y' ∧ 
    ((x + x')/2 - (y + y')/2 + 2 = 0) ∧
    (y' - y)/(x' - x) = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_valid_l2917_291750


namespace NUMINAMATH_CALUDE_max_plus_min_of_f_l2917_291799

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + 1)^2 / (Real.sin x^2 + 1)

theorem max_plus_min_of_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
                (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧ 
                (M + m = 2) :=
sorry

end NUMINAMATH_CALUDE_max_plus_min_of_f_l2917_291799


namespace NUMINAMATH_CALUDE_polygon_120_sides_diagonals_l2917_291744

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 120 sides has 7020 diagonals -/
theorem polygon_120_sides_diagonals :
  num_diagonals 120 = 7020 := by
  sorry

end NUMINAMATH_CALUDE_polygon_120_sides_diagonals_l2917_291744


namespace NUMINAMATH_CALUDE_carpet_transformation_l2917_291721

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a piece cut from a rectangle -/
structure CutPiece where
  width : ℕ
  height : ℕ

/-- Represents the original carpet -/
def original_carpet : Rectangle := { width := 9, height := 12 }

/-- Represents the piece cut off by the dragon -/
def dragon_cut : CutPiece := { width := 1, height := 8 }

/-- Represents the final square carpet -/
def final_carpet : Rectangle := { width := 10, height := 10 }

/-- Function to calculate the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Function to calculate the area of a cut piece -/
def cut_area (c : CutPiece) : ℕ := c.width * c.height

/-- Theorem stating that it's possible to transform the damaged carpet into a square -/
theorem carpet_transformation :
  ∃ (part1 part2 part3 : Rectangle),
    area original_carpet - cut_area dragon_cut =
    area part1 + area part2 + area part3 ∧
    area final_carpet = area part1 + area part2 + area part3 := by
  sorry

end NUMINAMATH_CALUDE_carpet_transformation_l2917_291721


namespace NUMINAMATH_CALUDE_map_to_actual_ratio_l2917_291725

-- Define the actual distance in kilometers
def actual_distance_km : ℝ := 6

-- Define the map distance in centimeters
def map_distance_cm : ℝ := 20

-- Define the conversion factor from kilometers to centimeters
def km_to_cm : ℝ := 100000

-- Theorem statement
theorem map_to_actual_ratio :
  (map_distance_cm / (actual_distance_km * km_to_cm)) = (1 / 30000) := by
  sorry

end NUMINAMATH_CALUDE_map_to_actual_ratio_l2917_291725


namespace NUMINAMATH_CALUDE_cubic_inequality_l2917_291768

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 44*x - 16 > 0 ↔ 
  (x > 4 ∧ x < 4 + 2*Real.sqrt 3) ∨ x > 4 + 2*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2917_291768


namespace NUMINAMATH_CALUDE_semicircle_area_theorem_l2917_291780

theorem semicircle_area_theorem (x y z : ℝ) : 
  x^2 + y^2 = z^2 →
  (1/8) * π * x^2 = 50 * π →
  (1/8) * π * y^2 = 288 * π →
  (1/8) * π * z^2 = 338 * π :=
sorry

end NUMINAMATH_CALUDE_semicircle_area_theorem_l2917_291780


namespace NUMINAMATH_CALUDE_count_valid_antibirthdays_l2917_291741

/-- Represents a date in day.month format -/
structure Date :=
  (day : ℕ)
  (month : ℕ)

/-- Checks if a date is valid -/
def is_valid_date (d : Date) : Prop :=
  1 ≤ d.month ∧ d.month ≤ 12 ∧ 1 ≤ d.day ∧ d.day ≤ 31

/-- Swaps the day and month of a date -/
def swap_date (d : Date) : Date :=
  ⟨d.month, d.day⟩

/-- Checks if a date has a valid anti-birthday -/
def has_valid_antibirthday (d : Date) : Prop :=
  is_valid_date d ∧ 
  is_valid_date (swap_date d) ∧ 
  d.day ≠ d.month

/-- The number of days in a year with valid anti-birthdays -/
def days_with_valid_antibirthdays : ℕ := 132

/-- Theorem stating the number of days with valid anti-birthdays -/
theorem count_valid_antibirthdays : 
  (∀ d : Date, has_valid_antibirthday d) → 
  days_with_valid_antibirthdays = 132 := by
  sorry

#check count_valid_antibirthdays

end NUMINAMATH_CALUDE_count_valid_antibirthdays_l2917_291741


namespace NUMINAMATH_CALUDE_carbon_weight_in_C4H8O2_l2917_291712

/-- The molecular weight of the carbon part in C4H8O2 -/
def carbon_weight (atomic_weight : ℝ) (num_atoms : ℕ) : ℝ :=
  atomic_weight * num_atoms

/-- Proof that the molecular weight of the carbon part in C4H8O2 is 48.04 g/mol -/
theorem carbon_weight_in_C4H8O2 :
  let compound_weight : ℝ := 88
  let carbon_atomic_weight : ℝ := 12.01
  let num_carbon_atoms : ℕ := 4
  carbon_weight carbon_atomic_weight num_carbon_atoms = 48.04 := by
  sorry

end NUMINAMATH_CALUDE_carbon_weight_in_C4H8O2_l2917_291712


namespace NUMINAMATH_CALUDE_table_height_is_36_l2917_291722

/-- The height of the table in inches -/
def table_height : ℝ := 36

/-- The length of each wooden block in inches -/
def block_length : ℝ := sorry

/-- The width of each wooden block in inches -/
def block_width : ℝ := sorry

/-- Two blocks stacked from one end to the other across the table measure 38 inches -/
axiom scenario1 : block_length + table_height - block_width = 38

/-- One block stacked on top of another with the third block beside them measure 34 inches -/
axiom scenario2 : block_width + table_height - block_length = 34

theorem table_height_is_36 : table_height = 36 := by sorry

end NUMINAMATH_CALUDE_table_height_is_36_l2917_291722


namespace NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l2917_291789

/-- Represents a cell on the board -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- Counts the number of paths from (0,0) to (x,y) that don't pass through marked cells -/
def countPaths (board : Board) (x y : Nat) : Nat :=
  sorry

theorem bottom_right_not_divisible_by_2011 (board : Board) :
  board.size = 2012 →
  (∀ c ∈ board.markedCells, c.x = c.y ∧ c.x ≠ 0 ∧ c.x ≠ 2011) →
  ¬ (countPaths board 2011 2011 % 2011 = 0) :=
by sorry

end NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l2917_291789


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l2917_291716

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -4/3 * x^3 + 6 * x^2 - 50/3 * x - 14/3
  (q 1 = -8) ∧ (q 2 = -12) ∧ (q 3 = -20) ∧ (q 4 = -40) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l2917_291716


namespace NUMINAMATH_CALUDE_remainder_problem_l2917_291797

theorem remainder_problem (divisor : ℕ) (a b : ℕ) (rem_a rem_sum : ℕ) 
  (h_divisor : divisor = 13)
  (h_a : a = 242)
  (h_b : b = 698)
  (h_rem_a : a % divisor = rem_a)
  (h_rem_a_val : rem_a = 8)
  (h_rem_sum : (a + b) % divisor = rem_sum)
  (h_rem_sum_val : rem_sum = 4) :
  b % divisor = 9 := by
  sorry


end NUMINAMATH_CALUDE_remainder_problem_l2917_291797


namespace NUMINAMATH_CALUDE_pizza_slices_left_is_three_l2917_291776

/-- Calculates the number of pizza slices left after John and Sam eat -/
def pizza_slices_left (total : ℕ) (john_ate : ℕ) (sam_ate_multiplier : ℕ) : ℕ :=
  total - (john_ate + sam_ate_multiplier * john_ate)

/-- Theorem: The number of pizza slices left is 3 -/
theorem pizza_slices_left_is_three :
  pizza_slices_left 12 3 2 = 3 := by
  sorry

#eval pizza_slices_left 12 3 2

end NUMINAMATH_CALUDE_pizza_slices_left_is_three_l2917_291776


namespace NUMINAMATH_CALUDE_pizza_distribution_l2917_291793

theorem pizza_distribution (treShawn Michael LaMar : ℚ) : 
  treShawn = 1/2 →
  Michael = 1/3 →
  treShawn + Michael + LaMar = 1 →
  LaMar = 1/6 := by
sorry

end NUMINAMATH_CALUDE_pizza_distribution_l2917_291793


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2917_291709

-- Define the sampling scenarios
structure SamplingScenario where
  total : ℕ
  sample_size : ℕ
  categories : Option (List ℕ)

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the function to determine the most appropriate sampling method
def most_appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

-- Theorem to prove the correct sampling methods for given scenarios
theorem correct_sampling_methods :
  let scenario1 := SamplingScenario.mk 10 2 none
  let scenario2 := SamplingScenario.mk 1920 32 none
  let scenario3 := SamplingScenario.mk 160 20 (some [120, 16, 24])
  (most_appropriate_sampling_method scenario1 = SamplingMethod.SimpleRandom) ∧
  (most_appropriate_sampling_method scenario2 = SamplingMethod.Systematic) ∧
  (most_appropriate_sampling_method scenario3 = SamplingMethod.Stratified) :=
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2917_291709


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l2917_291726

theorem sqrt_equality_condition (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  Real.sqrt (a - b + c) = Real.sqrt a - Real.sqrt b + Real.sqrt c ↔ a = b ∨ b = c :=
sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l2917_291726


namespace NUMINAMATH_CALUDE_pipe_b_rate_is_50_l2917_291785

/-- Represents the water tank system with three pipes -/
structure WaterTankSystem where
  tank_capacity : ℕ
  pipe_a_rate : ℕ
  pipe_b_rate : ℕ
  pipe_c_rate : ℕ
  cycle_time : ℕ
  total_time : ℕ

/-- Calculates the volume filled in one cycle -/
def volume_per_cycle (system : WaterTankSystem) : ℤ :=
  system.pipe_a_rate * 1 + system.pipe_b_rate * 2 - system.pipe_c_rate * 2

/-- Theorem stating that the rate of Pipe B must be 50 L/min -/
theorem pipe_b_rate_is_50 (system : WaterTankSystem) 
  (h1 : system.tank_capacity = 2000)
  (h2 : system.pipe_a_rate = 200)
  (h3 : system.pipe_c_rate = 25)
  (h4 : system.cycle_time = 5)
  (h5 : system.total_time = 40)
  (h6 : (system.total_time / system.cycle_time : ℤ) * volume_per_cycle system = system.tank_capacity) :
  system.pipe_b_rate = 50 := by
  sorry

end NUMINAMATH_CALUDE_pipe_b_rate_is_50_l2917_291785


namespace NUMINAMATH_CALUDE_unique_solution_system_l2917_291767

theorem unique_solution_system (a b c d : ℝ) : 
  (a * b + c + d = 3) ∧
  (b * c + d + a = 5) ∧
  (c * d + a + b = 2) ∧
  (d * a + b + c = 6) →
  (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2917_291767


namespace NUMINAMATH_CALUDE_smallest_cut_length_l2917_291787

theorem smallest_cut_length (z : ℕ) : z ≥ 9 →
  (∃ x y : ℕ, x = z / 2 ∧ y = z - 2) →
  (13 - z / 2 + 22 - z ≤ 25 - z) →
  (13 - z / 2 + 25 - z ≤ 22 - z) →
  (22 - z + 25 - z ≤ 13 - z / 2) →
  ∀ w : ℕ, w ≥ 9 → w < z →
    ¬((13 - w / 2 + 22 - w ≤ 25 - w) ∧
      (13 - w / 2 + 25 - w ≤ 22 - w) ∧
      (22 - w + 25 - w ≤ 13 - w / 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l2917_291787


namespace NUMINAMATH_CALUDE_coin_radius_l2917_291770

/-- Given a coin with diameter 14 millimeters, its radius is 7 millimeters. -/
theorem coin_radius (d : ℝ) (h : d = 14) : d / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_coin_radius_l2917_291770


namespace NUMINAMATH_CALUDE_a_can_ensure_segments_l2917_291766

/-- Represents a point on the circle -/
structure Point where
  has_piece : Bool

/-- Represents a segment between two points -/
structure Segment where
  point1 : Point
  point2 : Point

/-- Represents the state of the game -/
structure GameState where
  n : Nat
  points : List Point
  segments : List Segment

/-- Player A's strategy -/
def player_a_strategy (state : GameState) : GameState :=
  sorry

/-- Player B's strategy -/
def player_b_strategy (state : GameState) : GameState :=
  sorry

/-- Counts the number of segments connecting a point with a piece and a point without a piece -/
def count_valid_segments (state : GameState) : Nat :=
  sorry

/-- Main theorem -/
theorem a_can_ensure_segments (n : Nat) (h : n ≥ 2) :
  ∃ (initial_state : GameState),
    initial_state.n = n ∧
    initial_state.points.length = 3 * n ∧
    (∀ (b_strategy : GameState → GameState),
      let final_state := (player_a_strategy ∘ b_strategy)^[n] initial_state
      count_valid_segments final_state ≥ (n - 1) / 6) :=
  sorry

end NUMINAMATH_CALUDE_a_can_ensure_segments_l2917_291766


namespace NUMINAMATH_CALUDE_orthocenter_coordinates_l2917_291730

/-- The orthocenter of a triangle --/
structure Orthocenter (A B C : ℝ × ℝ) where
  point : ℝ × ℝ
  is_orthocenter : Bool

/-- Definition of triangle ABC --/
def A : ℝ × ℝ := (5, -1)
def B : ℝ × ℝ := (4, -8)
def C : ℝ × ℝ := (-4, -4)

/-- The orthocenter of triangle ABC --/
def triangle_orthocenter : Orthocenter A B C := {
  point := (3, -5),
  is_orthocenter := sorry
}

/-- Theorem: The orthocenter of triangle ABC is (3, -5) --/
theorem orthocenter_coordinates :
  triangle_orthocenter.point = (3, -5) := by sorry

end NUMINAMATH_CALUDE_orthocenter_coordinates_l2917_291730


namespace NUMINAMATH_CALUDE_line_through_point_l2917_291751

/-- Given a line equation 2bx + (b+2)y = b + 6 that passes through the point (-3, 4), prove that b = 2/3 -/
theorem line_through_point (b : ℝ) : 
  (2 * b * (-3) + (b + 2) * 4 = b + 6) → b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2917_291751


namespace NUMINAMATH_CALUDE_course_selection_methods_l2917_291704

theorem course_selection_methods (n : ℕ) (k : ℕ) : 
  n = 3 → k = 4 → n ^ k = 81 := by sorry

end NUMINAMATH_CALUDE_course_selection_methods_l2917_291704


namespace NUMINAMATH_CALUDE_city_population_ratio_l2917_291794

theorem city_population_ratio (X Y Z : ℕ) (hY : Y = 2 * Z) (hX : X = 16 * Z) :
  X / Y = 8 := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l2917_291794


namespace NUMINAMATH_CALUDE_greatest_k_for_root_difference_l2917_291791

theorem greatest_k_for_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 8 = 0 ∧ 
    x₂^2 + k*x₂ + 8 = 0 ∧ 
    |x₁ - x₂| = 2*Real.sqrt 15) →
  k ≤ Real.sqrt 92 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_root_difference_l2917_291791


namespace NUMINAMATH_CALUDE_max_player_salary_l2917_291781

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (max_total_salary : ℕ) :
  num_players = 18 →
  min_salary = 20000 →
  max_total_salary = 800000 →
  ∃ (max_single_salary : ℕ),
    max_single_salary = 460000 ∧
    max_single_salary + (num_players - 1) * min_salary ≤ max_total_salary ∧
    ∀ (salary : ℕ), salary + (num_players - 1) * min_salary ≤ max_total_salary → salary ≤ max_single_salary :=
by sorry

#check max_player_salary

end NUMINAMATH_CALUDE_max_player_salary_l2917_291781


namespace NUMINAMATH_CALUDE_price_per_cup_l2917_291779

/-- Represents the number of trees each sister has -/
def trees : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges_per_tree : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges_per_tree : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the total earnings from selling the juice -/
def total_earnings : ℕ := 220000

/-- Calculates the total number of oranges harvested by all sisters -/
def total_oranges : ℕ := 
  trees * gabriela_oranges_per_tree + 
  trees * alba_oranges_per_tree + 
  trees * maricela_oranges_per_tree

/-- Calculates the total number of cups of juice that can be made -/
def total_cups : ℕ := total_oranges / oranges_per_cup

/-- Theorem stating that the price per cup of juice is $4 -/
theorem price_per_cup : total_earnings / total_cups = 4 := by
  sorry

end NUMINAMATH_CALUDE_price_per_cup_l2917_291779


namespace NUMINAMATH_CALUDE_four_fold_f_of_two_plus_i_l2917_291745

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

-- State the theorem
theorem four_fold_f_of_two_plus_i :
  f (f (f (f (2 + Complex.I)))) = 164833 + 354192 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_four_fold_f_of_two_plus_i_l2917_291745


namespace NUMINAMATH_CALUDE_power_function_coefficient_l2917_291749

theorem power_function_coefficient (m : ℝ) : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = (m^2 + 2*m - 2) * x^4 ∧ ∃ (k : ℝ), ∀ x, y x = x^k) → 
  m = 1 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_power_function_coefficient_l2917_291749


namespace NUMINAMATH_CALUDE_cross_to_square_l2917_291795

/-- Represents a cross made of unit squares -/
structure Cross :=
  (num_squares : ℕ)
  (side_length : ℝ)

/-- Represents a square -/
structure Square :=
  (side_length : ℝ)

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side_length ^ 2

/-- The cross in the problem -/
def problem_cross : Cross := { num_squares := 5, side_length := 1 }

/-- The theorem to be proved -/
theorem cross_to_square (c : Cross) (s : Square) 
  (h1 : c = problem_cross) 
  (h2 : s.side_length = Real.sqrt 5) : 
  s.area = c.num_squares * c.side_length ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cross_to_square_l2917_291795


namespace NUMINAMATH_CALUDE_stock_value_after_fluctuations_l2917_291748

theorem stock_value_after_fluctuations (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let limit_up := 1.1
  let limit_down := 0.9
  let final_value := initial_value * (limit_up ^ 5) * (limit_down ^ 5)
  final_value < initial_value :=
by sorry

end NUMINAMATH_CALUDE_stock_value_after_fluctuations_l2917_291748


namespace NUMINAMATH_CALUDE_playground_children_count_l2917_291755

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 44) 
  (h2 : girls = 53) : 
  boys + girls = 97 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l2917_291755


namespace NUMINAMATH_CALUDE_probability_is_one_half_l2917_291707

/-- Represents the class of a bus -/
inductive BusClass
| Upper
| Middle
| Lower

/-- Represents a sequence of three buses -/
def BusSequence := (BusClass × BusClass × BusClass)

/-- All possible bus sequences -/
def allSequences : List BusSequence := sorry

/-- Determines if Mr. Li boards an upper-class bus given a sequence -/
def boardsUpperClass (seq : BusSequence) : Bool := sorry

/-- The probability of Mr. Li boarding an upper-class bus -/
def probabilityOfUpperClass : ℚ := sorry

/-- Theorem stating that the probability of boarding an upper-class bus is 1/2 -/
theorem probability_is_one_half : probabilityOfUpperClass = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_half_l2917_291707


namespace NUMINAMATH_CALUDE_smallest_non_negative_solution_l2917_291720

theorem smallest_non_negative_solution (x : ℕ) : x = 2 ↔ 
  (∀ y : ℕ, (42 * y + 10) % 15 = 5 → y ≥ x) ∧ (42 * x + 10) % 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_negative_solution_l2917_291720


namespace NUMINAMATH_CALUDE_solid_circles_in_2006_l2917_291728

def circle_sequence (n : ℕ) : ℕ := n + 1

def total_circles (n : ℕ) : ℕ := (n * (n + 3)) / 2

theorem solid_circles_in_2006 : 
  ∃ n : ℕ, total_circles n ≤ 2006 ∧ total_circles (n + 1) > 2006 ∧ n = 61 :=
sorry

end NUMINAMATH_CALUDE_solid_circles_in_2006_l2917_291728


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l2917_291746

/-- Given a line with equation y = -2x - 3, prove that its symmetric line
    with respect to the y-axis has the equation y = 2x - 3 -/
theorem symmetric_line_wrt_y_axis (x y : ℝ) :
  (y = -2*x - 3) → (∃ (x' y' : ℝ), y' = 2*x' - 3 ∧ x' = -x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l2917_291746


namespace NUMINAMATH_CALUDE_invalid_assignment_l2917_291763

-- Define what constitutes a valid assignment statement
def is_valid_assignment (lhs : String) (rhs : String) : Prop :=
  lhs.length = 1 ∧ lhs.all Char.isAlpha

-- Define the statement in question
def statement : String × String := ("x*y", "a")

-- Theorem to prove
theorem invalid_assignment :
  ¬(is_valid_assignment statement.1 statement.2) :=
sorry

end NUMINAMATH_CALUDE_invalid_assignment_l2917_291763


namespace NUMINAMATH_CALUDE_rectangle_area_is_464_l2917_291735

-- Define the side lengths of the squares
def E : ℝ := 7
def H : ℝ := 2
def D : ℝ := 8

-- Define the side lengths of other squares in terms of H and D
def F : ℝ := H + E
def B : ℝ := H + 2 * E
def I : ℝ := 2 * H + E
def G : ℝ := 3 * H + E
def C : ℝ := 3 * H + D + E
def A : ℝ := 3 * H + 2 * D + E

-- Define the dimensions of the rectangle
def rectangle_width : ℝ := A + B
def rectangle_height : ℝ := A + C

-- Theorem to prove
theorem rectangle_area_is_464 : 
  rectangle_width * rectangle_height = 464 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_464_l2917_291735


namespace NUMINAMATH_CALUDE_line_through_points_l2917_291771

theorem line_through_points (a b : ℚ) : 
  (7 : ℚ) = a * 3 + b ∧ (19 : ℚ) = a * 10 + b → a - b = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2917_291771


namespace NUMINAMATH_CALUDE_fencing_required_l2917_291727

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_required (area : ℝ) (side : ℝ) : 
  area = 810 ∧ side = 30 → 
  ∃ (other_side : ℝ), 
    area = side * other_side ∧ 
    side + other_side + side = 87 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l2917_291727


namespace NUMINAMATH_CALUDE_set_inclusion_l2917_291729

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem set_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_set_inclusion_l2917_291729


namespace NUMINAMATH_CALUDE_tournament_matches_divisible_by_seven_l2917_291754

/-- Represents a single elimination tournament --/
structure Tournament :=
  (total_players : ℕ)
  (bye_players : ℕ)

/-- Calculates the total number of matches in a tournament --/
def total_matches (t : Tournament) : ℕ := t.total_players - 1

/-- Theorem: In a tournament with 120 players and 32 byes, the total matches is divisible by 7 --/
theorem tournament_matches_divisible_by_seven :
  ∀ (t : Tournament), t.total_players = 120 → t.bye_players = 32 →
  ∃ (k : ℕ), total_matches t = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_divisible_by_seven_l2917_291754


namespace NUMINAMATH_CALUDE_prize_logic_l2917_291703

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answered_all_correctly : Student → Prop)
variable (got_prize : Student → Prop)

-- State the theorem
theorem prize_logic (h : ∀ s : Student, answered_all_correctly s → got_prize s) :
  ∀ s : Student, ¬(got_prize s) → ¬(answered_all_correctly s) :=
by
  sorry

end NUMINAMATH_CALUDE_prize_logic_l2917_291703


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l2917_291778

/-- The value of p for which the focus of the parabola y² = 2px coincides with 
    the right focus of the hyperbola x²/4 - y²/5 = 1 -/
theorem parabola_hyperbola_focus (p : ℝ) : 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2/4 - y^2/5 = 1 ∧ 
   x = (Real.sqrt (4 + 5 : ℝ)) ∧ y = 0) → 
  p = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l2917_291778


namespace NUMINAMATH_CALUDE_pencil_and_pen_choices_l2917_291769

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1_size : ℕ) (set2_size : ℕ) : ℕ :=
  set1_size * set2_size

/-- Theorem: Choosing one item from a set of 3 and one from a set of 5 results in 15 possibilities -/
theorem pencil_and_pen_choices :
  choose_one_from_each 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_and_pen_choices_l2917_291769


namespace NUMINAMATH_CALUDE_store_money_made_l2917_291719

/-- Represents the total money made from pencil sales -/
def total_money_made (eraser_price regular_price short_price : ℚ)
                     (eraser_sold regular_sold short_sold : ℕ) : ℚ :=
  eraser_price * eraser_sold + regular_price * regular_sold + short_price * short_sold

/-- Theorem stating that the store made $194 from the given pencil sales -/
theorem store_money_made :
  total_money_made 0.8 0.5 0.4 200 40 35 = 194 := by sorry

end NUMINAMATH_CALUDE_store_money_made_l2917_291719


namespace NUMINAMATH_CALUDE_kellys_supplies_l2917_291718

/-- Calculates the number of supplies left after Kelly's art supply shopping adventure. -/
theorem kellys_supplies (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) (additional_paper : ℕ) : 
  students = 8 →
  paper_per_student = 3 →
  glue_bottles = 6 →
  additional_paper = 5 →
  ((students * paper_per_student + glue_bottles) / 2 + additional_paper : ℕ) = 20 := by
sorry

end NUMINAMATH_CALUDE_kellys_supplies_l2917_291718


namespace NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l2917_291765

theorem consecutive_sum_iff_not_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ (a k : ℕ), k > 0 ∧ n = (k * (2 * a + k - 1)) / 2) ↔ ¬∃ (m : ℕ), n = 2^m :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l2917_291765


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2917_291757

/-- 
A rhombus is a quadrilateral with four equal sides.
The perimeter of a rhombus is the sum of the lengths of all four sides.
-/
structure Rhombus where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 4 * side_length

theorem rhombus_side_length (r : Rhombus) (h : r.perimeter = 4) : r.side_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2917_291757


namespace NUMINAMATH_CALUDE_shorter_segment_length_l2917_291764

-- Define the triangle
def triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the altitude and segments
def altitude_segment (a b c x h : ℝ) : Prop :=
  triangle a b c ∧
  x > 0 ∧ h > 0 ∧
  x + (c - x) = c ∧
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2

-- Theorem statement
theorem shorter_segment_length :
  ∀ (x h : ℝ),
  altitude_segment 40 50 90 x h →
  x = 40 :=
sorry

end NUMINAMATH_CALUDE_shorter_segment_length_l2917_291764


namespace NUMINAMATH_CALUDE_milk_cost_l2917_291701

/-- Proves that the cost of a gallon of milk is $3 given the total groceries cost and the costs of other items. -/
theorem milk_cost (total : ℝ) (cereal_price cereal_qty : ℝ) (banana_price banana_qty : ℝ) 
  (apple_price apple_qty : ℝ) (cookie_qty : ℝ) :
  total = 25 ∧ 
  cereal_price = 3.5 ∧ cereal_qty = 2 ∧
  banana_price = 0.25 ∧ banana_qty = 4 ∧
  apple_price = 0.5 ∧ apple_qty = 4 ∧
  cookie_qty = 2 →
  ∃ (milk_price : ℝ),
    milk_price = 3 ∧
    total = cereal_price * cereal_qty + banana_price * banana_qty + 
            apple_price * apple_qty + milk_price + 2 * milk_price * cookie_qty :=
by sorry

end NUMINAMATH_CALUDE_milk_cost_l2917_291701


namespace NUMINAMATH_CALUDE_rose_garden_problem_l2917_291700

/-- Rose garden problem -/
theorem rose_garden_problem (total_rows : ℕ) (roses_per_row : ℕ) (total_pink : ℕ) :
  total_rows = 10 →
  roses_per_row = 20 →
  total_pink = 40 →
  ∃ (red_fraction : ℚ),
    red_fraction = 1/2 ∧
    ∀ (row : ℕ),
      row ≤ total_rows →
      ∃ (red white pink : ℕ),
        red + white + pink = roses_per_row ∧
        white = (3/5 : ℚ) * (roses_per_row - red) ∧
        pink = roses_per_row - red - white ∧
        red = (red_fraction * roses_per_row : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_rose_garden_problem_l2917_291700


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2917_291733

theorem polynomial_identity_sum_of_squares (p q r s t u v : ℤ) :
  (∀ x : ℝ, 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2917_291733


namespace NUMINAMATH_CALUDE_ladybug_dots_total_l2917_291723

/-- The total number of dots on ladybugs caught over three days -/
theorem ladybug_dots_total : 
  let monday_ladybugs : ℕ := 8
  let monday_dots_per_ladybug : ℕ := 6
  let tuesday_ladybugs : ℕ := 5
  let tuesday_dots_per_ladybug : ℕ := 7
  let wednesday_ladybugs : ℕ := 4
  let wednesday_dots_per_ladybug : ℕ := 8
  monday_ladybugs * monday_dots_per_ladybug + 
  tuesday_ladybugs * tuesday_dots_per_ladybug + 
  wednesday_ladybugs * wednesday_dots_per_ladybug = 115 := by
sorry

end NUMINAMATH_CALUDE_ladybug_dots_total_l2917_291723


namespace NUMINAMATH_CALUDE_lowry_earnings_l2917_291792

/-- Calculates the total earnings from bonsai sales with discounts applied --/
def bonsai_earnings (small_price medium_price big_price : ℚ)
                    (small_discount medium_discount big_discount : ℚ)
                    (small_count medium_count big_count : ℕ)
                    (small_discount_threshold medium_discount_threshold big_discount_threshold : ℕ) : ℚ :=
  let small_total := small_price * small_count
  let medium_total := medium_price * medium_count
  let big_total := big_price * big_count
  let small_discounted := if small_count ≥ small_discount_threshold then small_total * (1 - small_discount) else small_total
  let medium_discounted := if medium_count ≥ medium_discount_threshold then medium_total * (1 - medium_discount) else medium_total
  let big_discounted := if big_count > big_discount_threshold then big_total * (1 - big_discount) else big_total
  small_discounted + medium_discounted + big_discounted

theorem lowry_earnings :
  bonsai_earnings 30 45 60 0.1 0.15 0.05 8 5 7 4 3 5 = 806.25 := by
  sorry

end NUMINAMATH_CALUDE_lowry_earnings_l2917_291792


namespace NUMINAMATH_CALUDE_number_with_75_halves_l2917_291724

theorem number_with_75_halves (n : ℚ) : (∃ k : ℕ, n = k * (1/2) ∧ k = 75) → n = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_with_75_halves_l2917_291724


namespace NUMINAMATH_CALUDE_inverse_g_84_l2917_291717

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_84_l2917_291717


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l2917_291753

/-- Represents the number of male students in the group -/
def num_male : ℕ := 3

/-- Represents the number of female students in the group -/
def num_female : ℕ := 2

/-- Represents the number of students selected for the competition -/
def num_selected : ℕ := 2

/-- Represents the event of selecting at least one female student -/
def at_least_one_female : Set (Fin num_male × Fin num_female) := sorry

/-- Represents the event of selecting all male students -/
def all_male : Set (Fin num_male × Fin num_female) := sorry

/-- Theorem stating that the events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  (at_least_one_female ∩ all_male = ∅) ∧
  (at_least_one_female ∪ all_male = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l2917_291753


namespace NUMINAMATH_CALUDE_smallest_product_l2917_291777

def digits : List Nat := [4, 5, 6, 7]

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, valid_arrangement a b c d →
    product a b c d ≥ 2622 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l2917_291777


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2917_291711

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 7 - a 10 = -1 →
  a 11 - a 4 = 21 →
  a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2917_291711


namespace NUMINAMATH_CALUDE_range_of_a_l2917_291705

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) → 
  a ∈ Set.Ici 4 ∪ Set.Iic (-6) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2917_291705


namespace NUMINAMATH_CALUDE_equal_color_squares_count_l2917_291786

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid with a specific pattern of black cells -/
def Grid : Matrix (Fin 5) (Fin 5) Cell := sorry

/-- Checks if a sub-square has an equal number of black and white cells -/
def has_equal_colors (top_left : Fin 5 × Fin 5) (size : Nat) : Bool :=
  sorry

/-- Counts the number of sub-squares with equal black and white cells -/
def count_equal_color_squares (g : Matrix (Fin 5) (Fin 5) Cell) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem equal_color_squares_count :
  count_equal_color_squares Grid = 16 :=
sorry

end NUMINAMATH_CALUDE_equal_color_squares_count_l2917_291786


namespace NUMINAMATH_CALUDE_expected_difference_coffee_tea_days_l2917_291775

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
| One
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight

/-- Represents the drink choice based on the die roll -/
inductive DrinkChoice
| Coffee
| Tea

/-- Function to determine the drink choice based on the die outcome -/
def choosedrink (outcome : DieOutcome) : DrinkChoice :=
  match outcome with
  | DieOutcome.Two | DieOutcome.Three | DieOutcome.Five | DieOutcome.Seven => DrinkChoice.Coffee
  | _ => DrinkChoice.Tea

/-- Number of days in a leap year -/
def leapYearDays : Nat := 366

/-- Probability of rolling a number that results in drinking coffee -/
def probCoffee : ℚ := 4 / 7

/-- Probability of rolling a number that results in drinking tea -/
def probTea : ℚ := 3 / 7

/-- Expected number of days drinking coffee in a leap year -/
def expectedCoffeeDays : ℚ := probCoffee * leapYearDays

/-- Expected number of days drinking tea in a leap year -/
def expectedTeaDays : ℚ := probTea * leapYearDays

/-- Theorem stating the expected difference between coffee and tea days -/
theorem expected_difference_coffee_tea_days :
  ⌊expectedCoffeeDays - expectedTeaDays⌋ = 52 := by sorry

end NUMINAMATH_CALUDE_expected_difference_coffee_tea_days_l2917_291775


namespace NUMINAMATH_CALUDE_sheep_count_l2917_291715

theorem sheep_count (total animals : ℕ) (cows goats : ℕ) 
  (h1 : total = 200)
  (h2 : cows = 40)
  (h3 : goats = 104)
  (h4 : animals = total - cows - goats) :
  animals = 56 := by
  sorry

end NUMINAMATH_CALUDE_sheep_count_l2917_291715


namespace NUMINAMATH_CALUDE_base_19_representation_of_1987_l2917_291796

theorem base_19_representation_of_1987 :
  ∃! (x y z b : ℕ), 
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x < b ∧ y < b ∧ z < b ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 := by
  sorry

end NUMINAMATH_CALUDE_base_19_representation_of_1987_l2917_291796


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2917_291714

/-- Given a polynomial x^4 + 5x^3 + 6x^2 + 5x + 1 with complex roots, 
    the sum of the cubes of its roots is -54 -/
theorem sum_of_cubes_of_roots : 
  ∀ (x₁ x₂ x₃ x₄ : ℂ), 
    (x₁^4 + 5*x₁^3 + 6*x₁^2 + 5*x₁ + 1 = 0) →
    (x₂^4 + 5*x₂^3 + 6*x₂^2 + 5*x₂ + 1 = 0) →
    (x₃^4 + 5*x₃^3 + 6*x₃^2 + 5*x₃ + 1 = 0) →
    (x₄^4 + 5*x₄^3 + 6*x₄^2 + 5*x₄ + 1 = 0) →
    x₁^3 + x₂^3 + x₃^3 + x₄^3 = -54 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2917_291714


namespace NUMINAMATH_CALUDE_min_value_expression_l2917_291739

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (4 * q) / (2 * p + 2 * r) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2917_291739


namespace NUMINAMATH_CALUDE_order_of_abc_l2917_291710

theorem order_of_abc (a b c : ℝ) (ha : a = 17/18) (hb : b = Real.cos (1/3)) (hc : c = 3 * Real.sin (1/3)) :
  c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l2917_291710


namespace NUMINAMATH_CALUDE_square_root_sum_equals_absolute_value_sum_l2917_291788

theorem square_root_sum_equals_absolute_value_sum (x : ℝ) :
  Real.sqrt (x^2 + 4*x + 4) + Real.sqrt (x^2 - 6*x + 9) = |x + 2| + |x - 3| := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_absolute_value_sum_l2917_291788


namespace NUMINAMATH_CALUDE_three_person_subcommittee_from_eight_l2917_291782

theorem three_person_subcommittee_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittee_from_eight_l2917_291782


namespace NUMINAMATH_CALUDE_power_eight_seven_thirds_l2917_291752

theorem power_eight_seven_thirds : (8 : ℝ) ^ (7/3) = 128 := by sorry

end NUMINAMATH_CALUDE_power_eight_seven_thirds_l2917_291752


namespace NUMINAMATH_CALUDE_quadratic_and_trig_problem_l2917_291790

theorem quadratic_and_trig_problem :
  -- Part 1: Quadratic equation
  (∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 2 ∧ x2 = 1 - Real.sqrt 2 ∧
    x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) ∧
  -- Part 2: Trigonometric expression
  (4 * (Real.sin (60 * π / 180))^2 - Real.tan (45 * π / 180) +
   Real.sqrt 2 * Real.cos (45 * π / 180) - Real.sin (30 * π / 180) = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_trig_problem_l2917_291790


namespace NUMINAMATH_CALUDE_min_value_inequality_l2917_291760

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + y = 2) :
  1/x^2 + 4/y^2 ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + y₀ = 2 ∧ 1/x₀^2 + 4/y₀^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2917_291760


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l2917_291737

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0 and f(1-x) = f(1+x),
    prove that f(3^x) > f(2^x) for all x > 0 -/
theorem quadratic_function_inequality (a b c : ℝ) (x : ℝ) 
  (h1 : a > 0) 
  (h2 : ∀ y, a*(1-y)^2 + b*(1-y) + c = a*(1+y)^2 + b*(1+y) + c) 
  (h3 : x > 0) : 
  a*(3^x)^2 + b*(3^x) + c > a*(2^x)^2 + b*(2^x) + c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l2917_291737


namespace NUMINAMATH_CALUDE_garrison_size_l2917_291731

theorem garrison_size (initial_days : ℕ) (reinforcement_size : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) :
  initial_days = 62 →
  reinforcement_size = 2700 →
  days_before_reinforcement = 15 →
  remaining_days = 20 →
  ∃ (initial_men : ℕ),
    initial_men * (initial_days - days_before_reinforcement) = 
    (initial_men + reinforcement_size) * remaining_days ∧
    initial_men = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_garrison_size_l2917_291731


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l2917_291784

theorem pet_shop_dogs (ratio_dogs : ℕ) (ratio_cats : ℕ) (ratio_bunnies : ℕ) 
  (total_dogs_bunnies : ℕ) : 
  ratio_dogs = 3 → 
  ratio_cats = 5 → 
  ratio_bunnies = 9 → 
  total_dogs_bunnies = 204 → 
  ∃ (x : ℕ), x * (ratio_dogs + ratio_bunnies) = total_dogs_bunnies ∧ 
             x * ratio_dogs = 51 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l2917_291784


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2917_291759

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2917_291759


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l2917_291772

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (∃ n : ℕ, abc - 1 = n * ((a - 1) * (b - 1) * (c - 1))) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by
  sorry

#check abc_divisibility_problem

end NUMINAMATH_CALUDE_abc_divisibility_problem_l2917_291772


namespace NUMINAMATH_CALUDE_jobber_pricing_jobber_pricing_example_l2917_291747

theorem jobber_pricing (original_price : ℝ) (purchase_discount : ℝ) (desired_gain : ℝ) (sale_discount : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - purchase_discount)
  let selling_price := purchase_price * (1 + desired_gain)
  let marked_price := selling_price / (1 - sale_discount)
  marked_price

theorem jobber_pricing_example : jobber_pricing 24 0.125 (1/3) 0.2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_jobber_pricing_jobber_pricing_example_l2917_291747
