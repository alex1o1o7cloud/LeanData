import Mathlib

namespace NUMINAMATH_CALUDE_rhombus_area_l2255_225526

/-- Represents a rhombus with diagonals 2a and 2b, and an acute angle θ. -/
structure Rhombus where
  a : ℕ+
  b : ℕ+
  θ : ℝ
  acute_angle : 0 < θ ∧ θ < π / 2

/-- The area of a rhombus is 2ab, where a and b are half the lengths of its diagonals. -/
theorem rhombus_area (r : Rhombus) : Real.sqrt ((2 * r.a) ^ 2 + (2 * r.b) ^ 2) / 2 * Real.sqrt ((2 * r.a) ^ 2 + (2 * r.b) ^ 2) * Real.sin r.θ / 2 = 2 * r.a * r.b := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2255_225526


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l2255_225560

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  area pan.panDimensions / area pan.pieceDimensions

/-- Theorem: A 30-inch by 24-inch pan can be divided into exactly 120 pieces of 3-inch by 2-inch brownies -/
theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 30, width := 24 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 120 := by
  sorry


end NUMINAMATH_CALUDE_brownie_pieces_count_l2255_225560


namespace NUMINAMATH_CALUDE_after_tax_dividend_amount_l2255_225508

def expected_earnings : ℝ := 0.80
def actual_earnings : ℝ := 1.10
def additional_dividend_rate : ℝ := 0.04
def additional_earnings_threshold : ℝ := 0.10
def tax_rate_threshold : ℝ := 1.00
def low_tax_rate : ℝ := 0.15
def high_tax_rate : ℝ := 0.20
def num_shares : ℕ := 300

def calculate_after_tax_dividend (
  expected_earnings : ℝ)
  (actual_earnings : ℝ)
  (additional_dividend_rate : ℝ)
  (additional_earnings_threshold : ℝ)
  (tax_rate_threshold : ℝ)
  (low_tax_rate : ℝ)
  (high_tax_rate : ℝ)
  (num_shares : ℕ) : ℝ :=
  sorry

theorem after_tax_dividend_amount :
  calculate_after_tax_dividend
    expected_earnings
    actual_earnings
    additional_dividend_rate
    additional_earnings_threshold
    tax_rate_threshold
    low_tax_rate
    high_tax_rate
    num_shares = 124.80 := by sorry

end NUMINAMATH_CALUDE_after_tax_dividend_amount_l2255_225508


namespace NUMINAMATH_CALUDE_g_evaluation_l2255_225542

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem g_evaluation : 5 * g 2 + 4 * g (-2) = 186 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l2255_225542


namespace NUMINAMATH_CALUDE_inequality_solution_l2255_225552

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (16 - x^2) + Real.sqrt (16*x - x^4) ≥ 16) : 
  x = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2255_225552


namespace NUMINAMATH_CALUDE_min_value_problem_l2255_225504

theorem min_value_problem (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2255_225504


namespace NUMINAMATH_CALUDE_class_age_difference_l2255_225556

theorem class_age_difference (n : ℕ) (T : ℕ) : 
  T = n * 40 →
  (T + 408) / (n + 12) = 36 →
  40 - (T + 408) / (n + 12) = 4 :=
by sorry

end NUMINAMATH_CALUDE_class_age_difference_l2255_225556


namespace NUMINAMATH_CALUDE_donnys_spending_l2255_225517

/-- Donny's spending on Thursday given his savings from Monday to Wednesday -/
theorem donnys_spending (monday_savings : ℕ) (tuesday_savings : ℕ) (wednesday_savings : ℕ)
  (h1 : monday_savings = 15)
  (h2 : tuesday_savings = 28)
  (h3 : wednesday_savings = 13) :
  (monday_savings + tuesday_savings + wednesday_savings) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_donnys_spending_l2255_225517


namespace NUMINAMATH_CALUDE_time_with_family_l2255_225565

/-- Given a 24-hour day, if a person spends 1/3 of the day sleeping, 
    1/6 of the day in school, 1/12 of the day making assignments, 
    then the remaining time spent with family is 10 hours. -/
theorem time_with_family (total_hours : ℝ) 
  (sleep_fraction : ℝ) (school_fraction : ℝ) (assignment_fraction : ℝ) :
  total_hours = 24 →
  sleep_fraction = 1/3 →
  school_fraction = 1/6 →
  assignment_fraction = 1/12 →
  total_hours - (sleep_fraction + school_fraction + assignment_fraction) * total_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_time_with_family_l2255_225565


namespace NUMINAMATH_CALUDE_trains_meet_time_l2255_225599

/-- Two trains moving towards each other on a straight track meet at 10 a.m. -/
theorem trains_meet_time :
  -- Define the distance between stations P and Q
  let distance_PQ : ℝ := 110

  -- Define the speed of the first train
  let speed_train1 : ℝ := 20

  -- Define the speed of the second train
  let speed_train2 : ℝ := 25

  -- Define the time difference between the starts of the two trains (in hours)
  let time_diff : ℝ := 1

  -- Define the start time of the second train
  let start_time_train2 : ℝ := 8

  -- The time when the trains meet (in hours after midnight)
  let meet_time : ℝ := start_time_train2 + 
    (distance_PQ - speed_train1 * time_diff) / (speed_train1 + speed_train2)

  -- Prove that the meet time is 10 a.m.
  meet_time = 10 := by sorry

end NUMINAMATH_CALUDE_trains_meet_time_l2255_225599


namespace NUMINAMATH_CALUDE_cupcake_distribution_l2255_225540

theorem cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) 
  (h2 : num_children = 8) 
  (h3 : total_cupcakes % num_children = 0) : 
  total_cupcakes / num_children = 12 := by
  sorry

#check cupcake_distribution

end NUMINAMATH_CALUDE_cupcake_distribution_l2255_225540


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_greater_than_4_power_17_l2255_225521

theorem smallest_k_for_64_power_greater_than_4_power_17 : 
  ∀ k : ℕ, (64 ^ k > 4 ^ 17 ∧ ∀ m : ℕ, m < k → 64 ^ m ≤ 4 ^ 17) ↔ k = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_greater_than_4_power_17_l2255_225521


namespace NUMINAMATH_CALUDE_mark_amy_age_difference_mark_amy_age_difference_proof_l2255_225538

theorem mark_amy_age_difference : ℕ → Prop :=
  fun age_difference =>
    ∃ (mark_current_age amy_current_age : ℕ),
      amy_current_age = 15 ∧
      mark_current_age + 5 = 27 ∧
      mark_current_age - amy_current_age = age_difference ∧
      age_difference = 7

-- The proof is omitted
theorem mark_amy_age_difference_proof : mark_amy_age_difference 7 := by
  sorry

end NUMINAMATH_CALUDE_mark_amy_age_difference_mark_amy_age_difference_proof_l2255_225538


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2255_225532

theorem divisibility_by_three (u v : ℤ) (h : (9 : ℤ) ∣ (u^2 + u*v + v^2)) : (3 : ℤ) ∣ u ∧ (3 : ℤ) ∣ v := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2255_225532


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2255_225531

theorem x_range_for_quadratic_inequality :
  (∀ m : ℝ, |m| ≤ 2 → ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0) →
  ∀ x : ℝ, (-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2255_225531


namespace NUMINAMATH_CALUDE_shares_distribution_l2255_225590

/-- Proves that if 120 rs are divided among three people (a, b, c) such that a's share is 20 rs more than b's and 20 rs less than c's, then b's share is 20 rs. -/
theorem shares_distribution (a b c : ℕ) : 
  (a + b + c = 120) →  -- Total amount is 120 rs
  (a = b + 20) →       -- a's share is 20 rs more than b's
  (c = a + 20) →       -- c's share is 20 rs more than a's
  b = 20 :=            -- b's share is 20 rs
by sorry


end NUMINAMATH_CALUDE_shares_distribution_l2255_225590


namespace NUMINAMATH_CALUDE_burn_time_3x5_grid_l2255_225528

/-- Represents a grid of toothpicks -/
structure ToothpickGrid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents the burning state of the grid -/
def BurningGrid := ToothpickGrid → Set (Nat × Nat)

/-- The time it takes for one toothpick to burn -/
def burn_time : Nat := 10

/-- Function to get adjacent positions in the grid -/
def adjacent_positions (grid : ToothpickGrid) (pos : Nat × Nat) : Set (Nat × Nat) :=
  sorry

/-- Function to simulate fire spread for one time step -/
def spread_fire (grid : ToothpickGrid) (burning : BurningGrid) : BurningGrid :=
  sorry

/-- Function to calculate the total burning time -/
def total_burn_time (grid : ToothpickGrid) (initial_burning : BurningGrid) : Nat :=
  sorry

/-- Theorem stating that a 3x5 grid of toothpicks burns in 65 seconds -/
theorem burn_time_3x5_grid :
  let grid : ToothpickGrid := ⟨3, 5⟩
  let initial_burning : BurningGrid := λ _ => {(0, 0), (0, 4)}
  total_burn_time grid initial_burning = 65 :=
sorry

end NUMINAMATH_CALUDE_burn_time_3x5_grid_l2255_225528


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l2255_225500

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (h : ℝ) (k : ℝ), (x - h)^2 + (y - k)^2 = 16 ∧ 2*h - k - 5 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x = 1 ∨ 3*x + 4*y - 23 = 0

-- Theorem statement
theorem circle_and_line_equations :
  ∀ (x y : ℝ),
    (circle_C x y ↔ (x - 3)^2 + (y - 1)^2 = 16) ∧
    (line_l x y ↔ (x = 1 ∨ 3*x + 4*y - 23 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_equations_l2255_225500


namespace NUMINAMATH_CALUDE_nina_shells_to_liam_l2255_225520

theorem nina_shells_to_liam (oliver liam nina : ℕ) 
  (h1 : liam = 3 * oliver) 
  (h2 : nina = 4 * liam) 
  (h3 : oliver > 0) : 
  (nina - (oliver + liam + nina) / 3) / nina = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_nina_shells_to_liam_l2255_225520


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2255_225522

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x^2 < 4 → -2 < x ∧ x < 2) ↔ 
  (∀ x : ℝ, x ≤ -2 ∨ x ≥ 2 → x^2 ≥ 4) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2255_225522


namespace NUMINAMATH_CALUDE_quartic_equation_solution_l2255_225530

theorem quartic_equation_solution :
  ∀ x : ℂ, x^4 - 16*x^2 + 256 = 0 ↔ x = 4 ∨ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_quartic_equation_solution_l2255_225530


namespace NUMINAMATH_CALUDE_sine_absolute_value_integral_l2255_225553

theorem sine_absolute_value_integral : ∫ x in (0)..(2 * Real.pi), |Real.sin x| = 4 := by
  sorry

end NUMINAMATH_CALUDE_sine_absolute_value_integral_l2255_225553


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l2255_225535

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 3^x

theorem f_composition_negative_two : f (f (-2)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l2255_225535


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2255_225541

theorem r_value_when_n_is_3 : 
  ∀ (n s r : ℕ), 
    s = 2^n - 1 → 
    r = 3^s - s → 
    n = 3 → 
    r = 2180 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2255_225541


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2255_225570

/-- Given a line with slope -3 passing through (2, 5), prove m + b = 8 --/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -3 → 
  5 = m * 2 + b → 
  m + b = 8 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2255_225570


namespace NUMINAMATH_CALUDE_camden_rico_dog_fraction_l2255_225595

/-- Proves that the fraction of dogs Camden bought compared to Rico is 3/4 -/
theorem camden_rico_dog_fraction :
  let justin_dogs : ℕ := 14
  let rico_dogs : ℕ := justin_dogs + 10
  let camden_dog_legs : ℕ := 72
  let legs_per_dog : ℕ := 4
  let camden_dogs : ℕ := camden_dog_legs / legs_per_dog
  (camden_dogs : ℚ) / rico_dogs = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_camden_rico_dog_fraction_l2255_225595


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l2255_225589

def pizza_problem (total_slices : ℕ) (plain_cost anchovy_cost onion_cost : ℚ)
  (anchovy_slices onion_slices : ℕ) (jerry_plain_slices : ℕ) : Prop :=
  let total_cost := plain_cost + anchovy_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jerry_slices := anchovy_slices + onion_slices + jerry_plain_slices
  let tom_slices := total_slices - jerry_slices
  let jerry_cost := cost_per_slice * jerry_slices
  let tom_cost := cost_per_slice * tom_slices
  jerry_cost - tom_cost = 11.36

theorem pizza_payment_difference :
  pizza_problem 12 12 3 2 4 4 2 := by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l2255_225589


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2255_225548

-- Define the constant k
def k : ℝ := 4^2 * (256 ^ (1/4))

-- State the theorem
theorem inverse_variation_problem (x y : ℝ) 
  (h1 : x^2 * y^(1/4) = k)  -- x² and ⁴√y are inversely proportional
  (h2 : x * y = 128)        -- xy = 128
  : y = 8 := by
  sorry

-- Note: The condition x = 4 when y = 256 is implicitly used in the definition of k

end NUMINAMATH_CALUDE_inverse_variation_problem_l2255_225548


namespace NUMINAMATH_CALUDE_original_ball_count_original_ball_count_is_960_l2255_225557

-- Define the initial ratio of red to white balls
def initial_ratio : Rat := 19 / 13

-- Define the ratio after adding red balls
def ratio_after_red : Rat := 5 / 3

-- Define the ratio after adding white balls
def ratio_after_white : Rat := 13 / 11

-- Define the difference in added balls
def added_difference : ℕ := 80

-- Theorem statement
theorem original_ball_count : ℕ :=
  let initial_red : ℕ := 57
  let initial_white : ℕ := 39
  let final_red : ℕ := 65
  let final_white : ℕ := 55
  let portion_size : ℕ := added_difference / (final_white - initial_white - (final_red - initial_red))
  (initial_red + initial_white) * portion_size

-- Proof
theorem original_ball_count_is_960 : original_ball_count = 960 := by
  sorry

end NUMINAMATH_CALUDE_original_ball_count_original_ball_count_is_960_l2255_225557


namespace NUMINAMATH_CALUDE_school_furniture_prices_l2255_225525

/-- The price of a table in yuan -/
def table_price : ℕ := 36

/-- The price of a chair in yuan -/
def chair_price : ℕ := 9

/-- The total cost of 2 tables and 3 chairs in yuan -/
def total_cost : ℕ := 99

theorem school_furniture_prices :
  (2 * table_price + 3 * chair_price = total_cost) ∧
  (table_price = 4 * chair_price) ∧
  (table_price = 36) ∧
  (chair_price = 9) := by
  sorry

end NUMINAMATH_CALUDE_school_furniture_prices_l2255_225525


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2255_225582

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35)
  (h2 : b + c = 55)
  (h3 : c + a = 62) :
  a + b + c = 76 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2255_225582


namespace NUMINAMATH_CALUDE_total_amount_after_two_years_l2255_225506

/-- Calculates the total amount after compound interest --/
def totalAmount (P : ℝ) (r : ℝ) (t : ℝ) (CI : ℝ) : ℝ :=
  P + CI

/-- Theorem: Given the conditions, the total amount after 2 years is 4326.40 --/
theorem total_amount_after_two_years :
  ∃ (P : ℝ), 
    let r : ℝ := 0.04
    let t : ℝ := 2
    let CI : ℝ := 326.40
    totalAmount P r t CI = 4326.40 :=
by
  sorry


end NUMINAMATH_CALUDE_total_amount_after_two_years_l2255_225506


namespace NUMINAMATH_CALUDE_min_value_inequality_l2255_225587

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^2 + y^2)/(3*(x + y)) + (x^2 + z^2)/(3*(x + z)) + (y^2 + z^2)/(3*(y + z)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2255_225587


namespace NUMINAMATH_CALUDE_project_hours_difference_l2255_225501

theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 144 →
  pat_hours = 2 * kate_hours →
  pat_hours * 3 = mark_hours →
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours - kate_hours = 80 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2255_225501


namespace NUMINAMATH_CALUDE_no_double_factorial_sum_l2255_225523

theorem no_double_factorial_sum : ¬∃ (z : ℤ) (x₁ y₁ x₂ y₂ : ℕ),
  x₁ ≤ y₁ ∧ x₂ ≤ y₂ ∧ 
  (z : ℤ) = x₁.factorial + y₁.factorial ∧
  (z : ℤ) = x₂.factorial + y₂.factorial ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
sorry

end NUMINAMATH_CALUDE_no_double_factorial_sum_l2255_225523


namespace NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l2255_225551

def z1 (m : ℝ) : ℂ := Complex.mk (m^2 + m + 1) (m^2 + m - 4)
def z2 : ℂ := Complex.mk 3 (-2)

theorem m_one_sufficient_not_necessary :
  (∃ m : ℝ, z1 m = z2 ∧ m ≠ 1) ∧ (z1 1 = z2) := by sorry

end NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l2255_225551


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2255_225581

/-- Represents the ratio of product types A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a stratified sample -/
structure StratifiedSample where
  ratio : ProductRatio
  type_a_count : ℕ
  total_size : ℕ

/-- Theorem: Given a stratified sample with product ratio 5:2:3 and 15 Type A products,
    the total sample size is 30 -/
theorem stratified_sample_size
  (sample : StratifiedSample)
  (h_ratio : sample.ratio = ⟨5, 2, 3⟩)
  (h_type_a : sample.type_a_count = 15) :
  sample.total_size = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2255_225581


namespace NUMINAMATH_CALUDE_part_one_part_two_l2255_225573

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x < 4 - 2*a ↔ -4 < x ∧ x < 4) → a = 0 := 
sorry

-- Part 2
theorem part_two : 
  (∀ m : ℝ, (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + m) ↔ 2 ≤ m) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2255_225573


namespace NUMINAMATH_CALUDE_job_completion_time_l2255_225546

/-- The time taken for three workers to complete a job together, given their individual completion times -/
theorem job_completion_time (time_A time_B time_C : ℝ) 
  (hA : time_A = 7) 
  (hB : time_B = 10) 
  (hC : time_C = 12) : 
  1 / (1 / time_A + 1 / time_B + 1 / time_C) = 420 / 137 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2255_225546


namespace NUMINAMATH_CALUDE_factorization_proof_l2255_225516

theorem factorization_proof (x : ℝ) : 180 * x^2 + 36 * x + 4 = 4 * (3 * x + 1) * (15 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2255_225516


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l2255_225543

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-4, 6)

-- Define line l₃
def l₃ (x y : ℝ) : Prop := x - 3*y - 1 = 0

-- Theorem for the first line
theorem line_through_P_and_origin :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  (∀ x y, a*x + b*y + c = 0 ↔ (x = P.1 ∧ y = P.2 ∨ x = 0 ∧ y = 0)) ∧
  a = 3 ∧ b = 2 ∧ c = 0 :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  (∀ x y, a*x + b*y + c = 0 ↔ (x = P.1 ∧ y = P.2)) ∧
  (a*(1 : ℝ) + b*(-3 : ℝ) = 0) ∧
  a = 3 ∧ b = 1 ∧ c = 6 :=
sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l2255_225543


namespace NUMINAMATH_CALUDE_canteen_distance_l2255_225561

theorem canteen_distance (road_distance : ℝ) (perpendicular_distance : ℝ) 
  (hypotenuse_distance : ℝ) (canteen_distance : ℝ) :
  road_distance = 400 ∧ 
  perpendicular_distance = 300 ∧ 
  hypotenuse_distance = 500 ∧
  canteen_distance^2 = perpendicular_distance^2 + (road_distance - canteen_distance)^2 →
  canteen_distance = 312.5 := by
sorry

end NUMINAMATH_CALUDE_canteen_distance_l2255_225561


namespace NUMINAMATH_CALUDE_inequality_proof_l2255_225594

open Real

noncomputable def f (a x : ℝ) : ℝ := (a/2) * x^2 - (a-2) * x - 2 * x * log x

theorem inequality_proof (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : 0 < a ∧ a < 2)
  (h_x : x₁ < x₂)
  (h_zeros : ∃ (x : ℝ), x = x₁ ∨ x = x₂ ∧ (deriv (f a)) x = 0) :
  x₂ - x₁ > 4/a - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2255_225594


namespace NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l2255_225576

theorem sin_five_pi_sixths_minus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l2255_225576


namespace NUMINAMATH_CALUDE_cubic_inequality_and_sum_inequality_l2255_225507

theorem cubic_inequality_and_sum_inequality :
  (∀ x : ℝ, x > 0 → x^3 - 3*x ≥ -2) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    x^2*y/z + y^2*z/x + z^2*x/y + 2*(y/(x*z) + z/(x*y) + x/(y*z)) ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_and_sum_inequality_l2255_225507


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2255_225513

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Blue

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing 2 white balls and 2 blue balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.White} + 2 • {BallColor.Blue}

/-- Event: At least one white ball is drawn -/
def atLeastOneWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∨ outcome.second = BallColor.White

/-- Event: All drawn balls are blue -/
def allBlue (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Blue ∧ outcome.second = BallColor.Blue

/-- The probability of an event occurring when drawing two balls from the bag -/
noncomputable def probability (event : DrawOutcome → Prop) : ℝ := sorry

/-- Theorem: "At least one white ball" and "All are blue balls" are mutually exclusive -/
theorem mutually_exclusive_events :
  probability (λ outcome => atLeastOneWhite outcome ∧ allBlue outcome) = 0 :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2255_225513


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2255_225585

/-- Given a geometric sequence with first term a₁ and common ratio r,
    the nth term is given by aₙ = a₁ * r^(n-1) -/
def geometric_sequence (a₁ : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^(n-1)

/-- The 12th term of a geometric sequence with first term 5 and common ratio -3 is -885735 -/
theorem twelfth_term_of_specific_sequence :
  geometric_sequence 5 (-3) 12 = -885735 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2255_225585


namespace NUMINAMATH_CALUDE_sum_smallest_largest_angle_l2255_225503

/-- A hexagon with angles in arithmetic progression -/
structure ArithmeticHexagon where
  /-- The smallest angle of the hexagon -/
  a : ℝ
  /-- The common difference between consecutive angles -/
  n : ℝ
  /-- The angles are non-negative -/
  a_nonneg : 0 ≤ a
  n_nonneg : 0 ≤ n
  /-- The sum of all angles in a hexagon is 720° -/
  sum_angles : a + (a + n) + (a + 2*n) + (a + 3*n) + (a + 4*n) + (a + 5*n) = 720

/-- The sum of the smallest and largest angles in an arithmetic hexagon is 240° -/
theorem sum_smallest_largest_angle (h : ArithmeticHexagon) : h.a + (h.a + 5*h.n) = 240 := by
  sorry


end NUMINAMATH_CALUDE_sum_smallest_largest_angle_l2255_225503


namespace NUMINAMATH_CALUDE_max_sin_A_in_triangle_l2255_225529

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

-- Define the theorem
theorem max_sin_A_in_triangle (t : Triangle) 
  (h : Real.tan t.A / Real.tan t.B + Real.tan t.A / Real.tan t.C = 3) :
  Real.sin t.A ≤ Real.sqrt 21 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sin_A_in_triangle_l2255_225529


namespace NUMINAMATH_CALUDE_cube_root_of_product_with_nested_roots_l2255_225596

theorem cube_root_of_product_with_nested_roots (N : ℝ) (h : N > 1) :
  (N * (N * N^(1/3))^(1/2))^(1/3) = N^(5/9) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_with_nested_roots_l2255_225596


namespace NUMINAMATH_CALUDE_min_distance_to_tangent_point_l2255_225536

/-- The minimum distance from a point on the line y = x + 1 to a tangent point 
    on the circle (x - 3)² + y² = 1 is √7. -/
theorem min_distance_to_tangent_point : 
  ∃ (P : ℝ × ℝ) (T : ℝ × ℝ),
    (P.2 = P.1 + 1) ∧ 
    ((T.1 - 3)^2 + T.2^2 = 1) ∧
    (∀ (Q : ℝ × ℝ), (Q.2 = Q.1 + 1) → (Q.1 - 3)^2 + Q.2^2 = 1 → 
      dist P T ≤ dist Q T) ∧
    dist P T = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_tangent_point_l2255_225536


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2255_225588

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2255_225588


namespace NUMINAMATH_CALUDE_mans_rate_l2255_225534

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 18)
  (h2 : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l2255_225534


namespace NUMINAMATH_CALUDE_optimal_rental_plan_l2255_225510

/-- Represents the capacity and cost of different car types -/
structure CarType where
  capacity : ℕ
  cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  type_a : ℕ
  type_b : ℕ

/-- Calculates the total capacity of a rental plan -/
def total_capacity (plan : RentalPlan) (a : CarType) (b : CarType) : ℕ :=
  plan.type_a * a.capacity + plan.type_b * b.capacity

/-- Calculates the total cost of a rental plan -/
def total_cost (plan : RentalPlan) (a : CarType) (b : CarType) : ℕ :=
  plan.type_a * a.cost + plan.type_b * b.cost

/-- Checks if a rental plan is valid for the given total goods -/
def is_valid_plan (plan : RentalPlan) (a : CarType) (b : CarType) (total_goods : ℕ) : Prop :=
  total_capacity plan a b = total_goods

/-- Theorem: The optimal rental plan for transporting 27 tons of goods is 1 type A car and 6 type B cars, with a total cost of 820 yuan -/
theorem optimal_rental_plan :
  ∃ (a b : CarType) (optimal_plan : RentalPlan),
    -- Given conditions
    (2 * a.capacity + 3 * b.capacity = 18) ∧
    (a.capacity + 2 * b.capacity = 11) ∧
    (a.cost = 100) ∧
    (b.cost = 120) ∧
    -- Optimal plan
    (optimal_plan.type_a = 1) ∧
    (optimal_plan.type_b = 6) ∧
    -- Plan is valid
    (is_valid_plan optimal_plan a b 27) ∧
    -- Plan is optimal (minimum cost)
    (∀ (plan : RentalPlan),
      is_valid_plan plan a b 27 →
      total_cost optimal_plan a b ≤ total_cost plan a b) ∧
    -- Total cost is 820 yuan
    (total_cost optimal_plan a b = 820) :=
  sorry

end NUMINAMATH_CALUDE_optimal_rental_plan_l2255_225510


namespace NUMINAMATH_CALUDE_nathan_writes_25_letters_per_hour_l2255_225511

/-- The number of letters Nathan can write in one hour -/
def nathan_letters_per_hour : ℕ := sorry

/-- The number of letters Jacob can write in one hour -/
def jacob_letters_per_hour : ℕ := sorry

/-- Jacob writes twice as fast as Nathan -/
axiom jacob_twice_as_fast : jacob_letters_per_hour = 2 * nathan_letters_per_hour

/-- Together, Jacob and Nathan can write 750 letters in 10 hours -/
axiom combined_output : 10 * (jacob_letters_per_hour + nathan_letters_per_hour) = 750

theorem nathan_writes_25_letters_per_hour : nathan_letters_per_hour = 25 := by
  sorry

end NUMINAMATH_CALUDE_nathan_writes_25_letters_per_hour_l2255_225511


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l2255_225597

theorem sqrt_x_plus_one_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l2255_225597


namespace NUMINAMATH_CALUDE_cos_equality_theorem_l2255_225593

theorem cos_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (310 * π / 180)) ↔ (n = 50 ∨ n = 310) :=
by sorry

end NUMINAMATH_CALUDE_cos_equality_theorem_l2255_225593


namespace NUMINAMATH_CALUDE_weighted_average_closer_to_larger_set_l2255_225567

theorem weighted_average_closer_to_larger_set 
  (set1 set2 : Finset ℝ) 
  (mean1 mean2 : ℝ) 
  (h_size : set1.card > set2.card) 
  (h_mean1 : mean1 = (set1.sum id) / set1.card) 
  (h_mean2 : mean2 = (set2.sum id) / set2.card) 
  (h_total_mean : (set1.sum id + set2.sum id) / (set1.card + set2.card) = 80) :
  |80 - mean1| < |80 - mean2| :=
sorry

end NUMINAMATH_CALUDE_weighted_average_closer_to_larger_set_l2255_225567


namespace NUMINAMATH_CALUDE_weight_of_b_l2255_225575

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 30)
  (h2 : (a + b) / 2 = 25)
  (h3 : (b + c) / 2 = 28) : 
  b = 16 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l2255_225575


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l2255_225502

theorem polynomial_factor_theorem (c : ℚ) : 
  (∀ x : ℚ, (x + 7) ∣ (c * x^3 + 19 * x^2 - c * x - 49)) → c = 21/8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l2255_225502


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2255_225554

theorem trigonometric_identity (α : ℝ) :
  3.404 * (8 * Real.cos α ^ 4 - 4 * Real.cos α ^ 3 - 8 * Real.cos α ^ 2 + 3 * Real.cos α + 1) /
  (8 * Real.cos α ^ 4 + 4 * Real.cos α ^ 3 - 8 * Real.cos α ^ 2 - 3 * Real.cos α + 1) =
  -Real.tan (7 * α / 2) * Real.tan (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2255_225554


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l2255_225579

theorem sum_of_xyz_equals_sqrt_13 (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + y^2 + x*y = 3)
  (h_eq2 : y^2 + z^2 + y*z = 4)
  (h_eq3 : z^2 + x^2 + z*x = 7) :
  x + y + z = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l2255_225579


namespace NUMINAMATH_CALUDE_square_area_relationship_l2255_225509

/-- Given a square with side length a+b, prove that the relationship between 
    the areas of three squares formed within it can be expressed as a^2 + b^2 = c^2. -/
theorem square_area_relationship (a b c : ℝ) : 
  (∃ (total_area : ℝ), total_area = (a + b)^2) → 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_relationship_l2255_225509


namespace NUMINAMATH_CALUDE_quadratic_sum_l2255_225569

/-- A quadratic function g(x) = dx^2 + ex + f -/
def g (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  g d e f 0 = 5 → g d e f 2 = 3 → d + e + 3 * f = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2255_225569


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2255_225559

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℕ
  side : ℕ

/-- The theorem statement -/
theorem similar_triangles_side_length 
  (small large : Triangle)
  (h_diff : large.area - small.area = 32)
  (h_ratio : ∃ k : ℕ, large.area = k^2 * small.area)
  (h_small_side : small.side = 4) :
  large.side = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l2255_225559


namespace NUMINAMATH_CALUDE_percentage_problem_l2255_225584

theorem percentage_problem (x : ℝ) : 80 = 16.666666666666668 / 100 * x → x = 480 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2255_225584


namespace NUMINAMATH_CALUDE_winner_received_62_percent_l2255_225562

/-- Represents an election with two candidates -/
structure Election where
  winner_votes : ℕ
  winning_margin : ℕ

/-- Calculates the percentage of votes received by the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / ((e.winner_votes + (e.winner_votes - e.winning_margin)) : ℚ) * 100

/-- Theorem stating that in the given election scenario, the winner received 62% of votes -/
theorem winner_received_62_percent :
  let e : Election := { winner_votes := 775, winning_margin := 300 }
  winner_percentage e = 62 := by sorry

end NUMINAMATH_CALUDE_winner_received_62_percent_l2255_225562


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2255_225550

theorem quadratic_rewrite (d e f : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (d * x + e)^2 + f) →
  d * e = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2255_225550


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l2255_225578

theorem cylinder_volume_increase (R H : ℝ) (hR : R = 8) (hH : H = 3) :
  ∃ x : ℝ, x > 0 ∧
  ∃ C : ℝ, C > 0 ∧
  (Real.pi * (R + x)^2 * (H + x) = Real.pi * R^2 * H + C) →
  x = 16/3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l2255_225578


namespace NUMINAMATH_CALUDE_function_identity_l2255_225572

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, 2 * f x - f (-x) = 3 * x) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2255_225572


namespace NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_planes_from_perpendicular_lines_parallel_line_plane_from_perpendicular_planes_l2255_225555

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Axioms for the relations
axiom parallel_planes_trans {a b c : Plane} : 
  parallel_planes a b → parallel_planes b c → parallel_planes a c

axiom perpendicular_planes_of_perpendicular_lines {a b : Plane} {m n : Line} :
  perpendicular_plane_line a m → perpendicular_plane_line b n → 
  perpendicular_lines m n → perpendicular_planes a b

axiom parallel_line_plane_of_perpendicular_planes {a b : Plane} {m : Line} :
  perpendicular_planes a b → perpendicular_plane_line b m → 
  ¬line_in_plane m a → parallel_line_plane m a

-- Theorems to prove
theorem parallel_planes_transitive {a b c : Plane} :
  parallel_planes a b → parallel_planes b c → parallel_planes a c :=
sorry

theorem perpendicular_planes_from_perpendicular_lines {a b : Plane} {m n : Line} :
  perpendicular_plane_line a m → perpendicular_plane_line b n → 
  perpendicular_lines m n → perpendicular_planes a b :=
sorry

theorem parallel_line_plane_from_perpendicular_planes {a b : Plane} {m : Line} :
  perpendicular_planes a b → perpendicular_plane_line b m → 
  ¬line_in_plane m a → parallel_line_plane m a :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_planes_from_perpendicular_lines_parallel_line_plane_from_perpendicular_planes_l2255_225555


namespace NUMINAMATH_CALUDE_water_weight_l2255_225547

/-- Proves that a gallon of water weighs 8 pounds given the conditions of the water tank problem -/
theorem water_weight (tank_capacity : ℝ) (empty_tank_weight : ℝ) (fill_percentage : ℝ) (current_weight : ℝ)
  (h1 : tank_capacity = 200)
  (h2 : empty_tank_weight = 80)
  (h3 : fill_percentage = 0.8)
  (h4 : current_weight = 1360) :
  (current_weight - empty_tank_weight) / (fill_percentage * tank_capacity) = 8 := by
  sorry

end NUMINAMATH_CALUDE_water_weight_l2255_225547


namespace NUMINAMATH_CALUDE_right_triangle_areas_l2255_225544

theorem right_triangle_areas (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b) / 2
  let s1 := (s * b^2) / c^2
  let s2 := s - s1
  (s1 = 15.36 ∧ s2 = 8.64) := by sorry

end NUMINAMATH_CALUDE_right_triangle_areas_l2255_225544


namespace NUMINAMATH_CALUDE_positive_values_of_f_l2255_225524

open Set

noncomputable def f (a : ℝ) : ℝ := a + (-1 + 9*a + 4*a^2) / (a^2 - 3*a - 10)

theorem positive_values_of_f :
  {a : ℝ | f a > 0} = Ioo (-2 : ℝ) (-1) ∪ Ioo (-1 : ℝ) 1 ∪ Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_positive_values_of_f_l2255_225524


namespace NUMINAMATH_CALUDE_cases_in_1995_l2255_225505

/-- Calculates the number of cases in a given year assuming a linear decrease --/
def casesInYear (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let targetYearsSinceStart := targetYear - initialYear
  let decrease := (targetYearsSinceStart * totalDecrease) / totalYears
  initialCases - decrease

/-- Theorem stating that the number of cases in 1995 is 263,125 --/
theorem cases_in_1995 : 
  casesInYear 1970 700000 2010 1000 1995 = 263125 := by
  sorry

#eval casesInYear 1970 700000 2010 1000 1995

end NUMINAMATH_CALUDE_cases_in_1995_l2255_225505


namespace NUMINAMATH_CALUDE_talia_drive_distance_l2255_225571

/-- Represents the total distance Talia drives in a day -/
def total_distance (house_to_park park_to_store house_to_store : ℝ) : ℝ :=
  house_to_park + park_to_store + house_to_store

/-- Theorem stating the total distance Talia drives -/
theorem talia_drive_distance :
  ∀ (house_to_park park_to_store house_to_store : ℝ),
    house_to_park = 5 →
    park_to_store = 3 →
    house_to_store = 8 →
    total_distance house_to_park park_to_store house_to_store = 16 := by
  sorry

end NUMINAMATH_CALUDE_talia_drive_distance_l2255_225571


namespace NUMINAMATH_CALUDE_travel_time_A_l2255_225533

/-- The time it takes for A to travel 60 miles given the conditions -/
theorem travel_time_A (y : ℝ) 
  (h1 : y > 0) -- B's speed is positive
  (h2 : (60 / y) - (60 / (y + 2)) = 3/4) -- Time difference equation
  : 60 / (y + 2) = 30/7 := by
sorry

end NUMINAMATH_CALUDE_travel_time_A_l2255_225533


namespace NUMINAMATH_CALUDE_identity_function_only_solution_l2255_225539

theorem identity_function_only_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) = x + f (f y)) → (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_only_solution_l2255_225539


namespace NUMINAMATH_CALUDE_jimmy_crackers_needed_l2255_225545

/-- Calculates the number of crackers needed to reach a target calorie count -/
def crackers_needed (cracker_calories cookie_calories cookies_eaten target_calories : ℕ) : ℕ :=
  (target_calories - cookie_calories * cookies_eaten) / cracker_calories

/-- Proves that Jimmy needs 10 crackers to reach 500 total calories -/
theorem jimmy_crackers_needed :
  crackers_needed 15 50 7 500 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_crackers_needed_l2255_225545


namespace NUMINAMATH_CALUDE_log_equation_holds_l2255_225598

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 3) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 3 := by
  sorry


end NUMINAMATH_CALUDE_log_equation_holds_l2255_225598


namespace NUMINAMATH_CALUDE_quadratic_form_b_value_l2255_225577

theorem quadratic_form_b_value (b : ℝ) (n : ℝ) : 
  b < 0 →
  (∀ x, x^2 + b*x + 50 = (x + n)^2 + 16) →
  b = -2 * Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_b_value_l2255_225577


namespace NUMINAMATH_CALUDE_min_value_of_function_l2255_225591

theorem min_value_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1/a^5 + a^5 - 2) * (1/b^5 + b^5 - 2) ≥ 31^4 / 32^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2255_225591


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_5_l2255_225563

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Given vectors a and b, if they are parallel, then x = 5 -/
theorem parallel_vectors_imply_x_equals_5 :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  are_parallel a b → x = 5 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_5_l2255_225563


namespace NUMINAMATH_CALUDE_gavins_blue_shirts_l2255_225580

theorem gavins_blue_shirts (total_shirts : ℕ) (green_shirts : ℕ) (blue_shirts : ℕ) :
  total_shirts = 23 →
  green_shirts = 17 →
  total_shirts = green_shirts + blue_shirts →
  blue_shirts = 6 := by
sorry

end NUMINAMATH_CALUDE_gavins_blue_shirts_l2255_225580


namespace NUMINAMATH_CALUDE_equation_system_solutions_equation_system_unique_solutions_l2255_225519

/-- The system of equations has four solutions: (1, 1, 1) and three cyclic permutations of another triple -/
theorem equation_system_solutions :
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
  (∀ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 - y = (z - 1)^2 ∧
    y^2 - z = (x - 1)^2 ∧
    z^2 - x = (y - 1)^2 →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = a ∧ y = b ∧ z = c) ∨
     (x = b ∧ y = c ∧ z = a) ∨
     (x = c ∧ y = a ∧ z = b))) :=
by sorry

/-- The system of equations has exactly four solutions -/
theorem equation_system_unique_solutions :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), s.card = 4 ∧
  (∀ (x y z : ℝ), (x, y, z) ∈ s ↔
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 - y = (z - 1)^2 ∧
    y^2 - z = (x - 1)^2 ∧
    z^2 - x = (y - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_equation_system_unique_solutions_l2255_225519


namespace NUMINAMATH_CALUDE_right_triangle_area_l2255_225527

theorem right_triangle_area (a b c : ℝ) (h1 : a = 48) (h2 : c = 50) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 336 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2255_225527


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2255_225568

/-- An isosceles triangle with sides of 4cm and 3cm has a perimeter of either 10cm or 11cm. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Triangle inequality
  (a = 4 ∧ b = 3) ∨ (a = 3 ∧ b = 4) →  -- Given side lengths
  ((a = b ∧ c = 3) ∨ (a = c ∧ b = 3)) →  -- Isosceles condition
  a + b + c = 10 ∨ a + b + c = 11 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2255_225568


namespace NUMINAMATH_CALUDE_max_value_a_l2255_225558

theorem max_value_a (x y : ℝ) 
  (h1 : x - y ≤ 0) 
  (h2 : x + y - 5 ≥ 0) 
  (h3 : y - 3 ≤ 0) : 
  (∃ (a : ℝ), a = 25/13 ∧ 
    (∀ (b : ℝ), (∀ (x y : ℝ), 
      x - y ≤ 0 → x + y - 5 ≥ 0 → y - 3 ≤ 0 → 
      b * (x^2 + y^2) ≤ (x + y)^2) → 
    b ≤ a)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l2255_225558


namespace NUMINAMATH_CALUDE_kanul_cash_theorem_l2255_225586

/-- The total amount of cash Kanul had -/
def total_cash : ℝ := 5714.29

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery -/
def machinery : ℝ := 1000

/-- The percentage of total cash spent -/
def percentage_spent : ℝ := 0.30

theorem kanul_cash_theorem :
  total_cash = raw_materials + machinery + percentage_spent * total_cash := by
  sorry

end NUMINAMATH_CALUDE_kanul_cash_theorem_l2255_225586


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2255_225583

theorem product_of_three_numbers (x y z m : ℚ) : 
  x + y + z = 120 → 
  5 * x = m → 
  y - 12 = m → 
  z + 12 = m → 
  x ≤ y ∧ x ≤ z → 
  y ≥ z → 
  x * y * z = 4095360 / 1331 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2255_225583


namespace NUMINAMATH_CALUDE_integral_circle_area_l2255_225549

theorem integral_circle_area (f : ℝ → ℝ) (a b r : ℝ) (h : ∀ x ∈ Set.Icc a b, f x = Real.sqrt (r^2 - x^2)) :
  (∫ x in a..b, f x) = (π * r^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_integral_circle_area_l2255_225549


namespace NUMINAMATH_CALUDE_cos_18_degrees_l2255_225537

theorem cos_18_degrees :
  Real.cos (18 * π / 180) = Real.sqrt ((5 + Real.sqrt 5) / 8) := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l2255_225537


namespace NUMINAMATH_CALUDE_miranda_pillows_l2255_225514

/-- Calculates the number of pillows Miranda can stuff given the following conditions:
  * Each pillow needs 2 pounds of feathers
  * 1 pound of goose feathers is approximately 300 feathers
  * Miranda's goose has approximately 3600 feathers
-/
def pillows_from_goose (feathers_per_pillow : ℕ) (feathers_per_pound : ℕ) (goose_feathers : ℕ) : ℕ :=
  (goose_feathers / feathers_per_pound) / feathers_per_pillow

/-- Proves that Miranda can stuff 6 pillows given the conditions -/
theorem miranda_pillows : 
  pillows_from_goose 2 300 3600 = 6 := by
  sorry

end NUMINAMATH_CALUDE_miranda_pillows_l2255_225514


namespace NUMINAMATH_CALUDE_toris_height_l2255_225564

theorem toris_height (initial_height growth : ℝ) 
  (h1 : initial_height = 4.4)
  (h2 : growth = 2.86) :
  initial_height + growth = 7.26 := by
sorry

end NUMINAMATH_CALUDE_toris_height_l2255_225564


namespace NUMINAMATH_CALUDE_tickets_left_l2255_225512

/-- The number of tickets Dave started with -/
def initial_tickets : ℕ := 98

/-- The number of tickets Dave spent on the stuffed tiger -/
def spent_tickets : ℕ := 43

/-- Theorem stating that Dave had 55 tickets left after spending on the stuffed tiger -/
theorem tickets_left : initial_tickets - spent_tickets = 55 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l2255_225512


namespace NUMINAMATH_CALUDE_weight_removed_l2255_225574

/-- Given weights of sugar and salt bags, and their combined weight after removal,
    prove the amount of weight removed. -/
theorem weight_removed (sugar_weight salt_weight new_combined_weight : ℕ)
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : new_combined_weight = 42) :
  sugar_weight + salt_weight - new_combined_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_weight_removed_l2255_225574


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2255_225515

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0 ↔ x = 0) ∧ 
  (k - 1 ≠ 0) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2255_225515


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_666_and_8_mixed_number_representation_l2255_225592

def repeating_decimal_666 : ℚ := 2/3

theorem product_of_repeating_decimal_666_and_8 :
  repeating_decimal_666 * 8 = 16/3 :=
sorry

theorem mixed_number_representation :
  16/3 = 5 + 1/3 :=
sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_666_and_8_mixed_number_representation_l2255_225592


namespace NUMINAMATH_CALUDE_system_solution_existence_l2255_225566

theorem system_solution_existence (a : ℝ) : 
  (∃ b x y : ℝ, x^2 + y^2 + 2*a*(a + y - x) = 49 ∧ 
                y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)) ↔ 
  -24 ≤ a ∧ a ≤ 24 := by
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l2255_225566


namespace NUMINAMATH_CALUDE_die_product_divisible_by_48_l2255_225518

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem die_product_divisible_by_48 (S : Finset ℕ) (h : S ⊆ die_numbers) (h_card : S.card = 7) :
  48 ∣ S.prod id :=
sorry

end NUMINAMATH_CALUDE_die_product_divisible_by_48_l2255_225518
