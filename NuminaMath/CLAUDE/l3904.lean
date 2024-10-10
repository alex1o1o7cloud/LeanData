import Mathlib

namespace box_dimensions_l3904_390474

theorem box_dimensions (x : ℝ) 
  (h1 : x > 0)
  (h2 : ∃ (bow_length : ℝ), 6 * x + bow_length = 156)
  (h3 : ∃ (bow_length : ℝ), 7 * x + bow_length = 178) :
  x = 22 := by
sorry

end box_dimensions_l3904_390474


namespace original_number_proof_l3904_390491

theorem original_number_proof (r : ℝ) : 
  (1.20 * r - r) + (1.35 * r - r) - (r - 0.50 * r) = 110 → r = 2200 := by
  sorry

end original_number_proof_l3904_390491


namespace two_digit_sum_product_l3904_390469

/-- A function that returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- A function that returns the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Theorem: If c is a 2-digit positive integer where the sum of its digits is 10
    and the product of its digits is 25, then c = 55 -/
theorem two_digit_sum_product (c : ℕ) : 
  10 ≤ c ∧ c ≤ 99 ∧ 
  tens_digit c + ones_digit c = 10 ∧
  tens_digit c * ones_digit c = 25 →
  c = 55 := by
  sorry


end two_digit_sum_product_l3904_390469


namespace expand_binomials_l3904_390468

theorem expand_binomials (x : ℝ) : (7 * x + 9) * (3 * x + 4) = 21 * x^2 + 55 * x + 36 := by
  sorry

end expand_binomials_l3904_390468


namespace henry_twice_jills_age_l3904_390480

theorem henry_twice_jills_age (henry_age jill_age : ℕ) (years_ago : ℕ) : 
  henry_age + jill_age = 43 →
  henry_age = 27 →
  jill_age = 16 →
  henry_age - years_ago = 2 * (jill_age - years_ago) →
  years_ago = 5 := by
  sorry

end henry_twice_jills_age_l3904_390480


namespace total_rent_calculation_l3904_390475

/-- Calculates the total rent collected in a year for a rental building --/
theorem total_rent_calculation (total_units : ℕ) (occupancy_rate : ℚ) (rent_per_unit : ℕ) : 
  total_units = 100 → 
  occupancy_rate = 3/4 →
  rent_per_unit = 400 →
  (total_units : ℚ) * occupancy_rate * rent_per_unit * 12 = 360000 := by
  sorry

end total_rent_calculation_l3904_390475


namespace yellow_peaches_to_add_result_l3904_390414

/-- The number of yellow peaches needed to be added to satisfy the condition -/
def yellow_peaches_to_add (red green yellow : ℕ) : ℕ :=
  2 * (red + green) - yellow

/-- Theorem stating the number of yellow peaches to be added -/
theorem yellow_peaches_to_add_result :
  yellow_peaches_to_add 7 8 15 = 15 := by
  sorry

end yellow_peaches_to_add_result_l3904_390414


namespace mycoplasma_pneumonia_relation_l3904_390422

-- Define the contingency table
def a : ℕ := 40  -- infected with mycoplasma pneumonia and with chronic disease
def b : ℕ := 20  -- infected with mycoplasma pneumonia and without chronic disease
def c : ℕ := 60  -- not infected with mycoplasma pneumonia and with chronic disease
def d : ℕ := 80  -- not infected with mycoplasma pneumonia and without chronic disease
def n : ℕ := a + b + c + d

-- Define the K^2 statistic
def K_squared : ℚ := (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.5% confidence level
def critical_value : ℚ := 7.879

-- Define the number of cases with exactly one person having chronic disease
def favorable_cases : ℕ := 8
def total_cases : ℕ := 15

theorem mycoplasma_pneumonia_relation :
  K_squared > critical_value ∧ (favorable_cases : ℚ) / total_cases = 8 / 15 := by
  sorry

end mycoplasma_pneumonia_relation_l3904_390422


namespace linear_regression_passes_through_mean_l3904_390421

variables {x y : ℝ} (x_bar y_bar a_hat b_hat : ℝ)

/-- The linear regression equation -/
def linear_regression (x : ℝ) : ℝ := b_hat * x + a_hat

/-- The intercept of the linear regression equation -/
def intercept : ℝ := y_bar - b_hat * x_bar

theorem linear_regression_passes_through_mean :
  a_hat = intercept x_bar y_bar b_hat →
  linear_regression x_bar a_hat b_hat = y_bar :=
sorry

end linear_regression_passes_through_mean_l3904_390421


namespace permutations_of_four_l3904_390473

theorem permutations_of_four (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end permutations_of_four_l3904_390473


namespace geometric_sequence_middle_term_l3904_390484

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : ∃ r : ℝ, b = a * r ∧ c = b * r) : 
  (a = 25 ∧ c = 1/4) → b = 5/2 := by
sorry

end geometric_sequence_middle_term_l3904_390484


namespace exam_score_standard_deviations_l3904_390407

/-- Given an exam with mean score 76, where 60 is 2 standard deviations below the mean,
    prove that 100 is 3 standard deviations above the mean. -/
theorem exam_score_standard_deviations 
  (mean : ℝ) 
  (std_dev : ℝ) 
  (h1 : mean = 76) 
  (h2 : mean - 2 * std_dev = 60) 
  (h3 : mean + 3 * std_dev = 100) : 
  100 = mean + 3 * std_dev := by
  sorry

end exam_score_standard_deviations_l3904_390407


namespace A_equals_B_l3904_390403

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4*a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4*b^2 + 4*b + 2}

theorem A_equals_B : A = B := by sorry

end A_equals_B_l3904_390403


namespace simplification_proof_l3904_390466

theorem simplification_proof (x a : ℝ) :
  (3 * x^2 - 1 - 2*x - 5 + 3*x - x^2 = 2 * x^2 + x - 6) ∧
  (4 * (2 * a^2 - 1 + 2*a) - 3 * (a - 1 + a^2) = 5 * a^2 + 5*a - 1) :=
by sorry

end simplification_proof_l3904_390466


namespace triangle_trig_identity_l3904_390494

theorem triangle_trig_identity (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) (h1 : a = 4) (h2 : b = 7) (h3 : c = 5) :
  let α := Real.arccos ((c^2 + a^2 - b^2) / (2 * c * a))
  (Real.sin (α/2))^6 + (Real.cos (α/2))^6 = 7/25 := by sorry

end triangle_trig_identity_l3904_390494


namespace arctg_arcctg_comparison_l3904_390465

theorem arctg_arcctg_comparison : (5 * Real.sqrt 7) / 4 > Real.arctan (2 + Real.sqrt 5) + Real.arctan (1 / (2 - Real.sqrt 5)) := by
  sorry

end arctg_arcctg_comparison_l3904_390465


namespace number_of_observations_l3904_390498

theorem number_of_observations (initial_mean : ℝ) (wrong_value : ℝ) (correct_value : ℝ) (new_mean : ℝ) : 
  initial_mean = 36 → 
  wrong_value = 23 → 
  correct_value = 46 → 
  new_mean = 36.5 → 
  ∃ n : ℕ, n * initial_mean + (correct_value - wrong_value) = n * new_mean ∧ n = 46 :=
by sorry

end number_of_observations_l3904_390498


namespace percent_y_of_x_l3904_390427

theorem percent_y_of_x (x y : ℝ) (h : 0.5 * (x - y) = 0.3 * (x + y)) : y / x = 0.25 := by
  sorry

end percent_y_of_x_l3904_390427


namespace area_of_N_region_l3904_390496

-- Define the plane region for point M
def plane_region_M (a b : ℝ) : Prop := sorry

-- Define the transformation from M to N
def transform_M_to_N (a b : ℝ) : ℝ × ℝ := (a + b, a - b)

-- Define the plane region for point N
def plane_region_N (x y : ℝ) : Prop := sorry

-- Theorem statement
theorem area_of_N_region : 
  ∀ (a b : ℝ), plane_region_M a b → 
  (∃ (S : Set (ℝ × ℝ)), (∀ (x y : ℝ), (x, y) ∈ S ↔ plane_region_N x y) ∧ 
                         MeasureTheory.volume S = 4) :=
sorry

end area_of_N_region_l3904_390496


namespace y1_less_than_y2_l3904_390457

/-- A linear function f(x) = x + 4 -/
def f (x : ℝ) : ℝ := x + 4

/-- The theorem states that for two points on the graph of f,
    if the x-coordinate of the first point is less than the x-coordinate of the second point,
    then the y-coordinate of the first point is less than the y-coordinate of the second point. -/
theorem y1_less_than_y2 (y1 y2 : ℝ) 
    (h1 : f (-1/2) = y1) 
    (h2 : f 1 = y2) : 
  y1 < y2 := by
  sorry

end y1_less_than_y2_l3904_390457


namespace problem_statement_l3904_390453

theorem problem_statement (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -1 := by
sorry

end problem_statement_l3904_390453


namespace x_value_proof_l3904_390434

theorem x_value_proof : 
  ∀ x : ℝ, x = 88 * (1 + 25 / 100) → x = 110 := by
  sorry

end x_value_proof_l3904_390434


namespace linear_function_proof_l3904_390456

/-- A linear function passing through two given points -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  point1_x : ℝ
  point1_y : ℝ
  point2_x : ℝ
  point2_y : ℝ
  eq_at_point1 : point1_y = k * point1_x + b
  eq_at_point2 : point2_y = k * point2_x + b

/-- The specific linear function passing through (2,1) and (-3,6) -/
def specificLinearFunction : LinearFunction := {
  k := -1
  b := 3
  point1_x := 2
  point1_y := 1
  point2_x := -3
  point2_y := 6
  eq_at_point1 := by sorry
  eq_at_point2 := by sorry
}

theorem linear_function_proof :
  (specificLinearFunction.k = -1 ∧ specificLinearFunction.b = 3) ∧
  ¬(5 = specificLinearFunction.k * (-1) + specificLinearFunction.b) := by
  sorry

end linear_function_proof_l3904_390456


namespace binary_multiplication_division_l3904_390400

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 11010₂ -/
def a : Nat := binary_to_nat [true, true, false, true, false]

/-- Represents the binary number 11100₂ -/
def b : Nat := binary_to_nat [true, true, true, false, false]

/-- Represents the binary number 100₂ -/
def c : Nat := binary_to_nat [true, false, false]

/-- Represents the binary number 10101101₂ -/
def result : Nat := binary_to_nat [true, false, true, false, true, true, false, true]

/-- Theorem stating that 11010₂ × 11100₂ ÷ 100₂ = 10101101₂ -/
theorem binary_multiplication_division :
  a * b / c = result := by sorry

end binary_multiplication_division_l3904_390400


namespace product_of_real_parts_l3904_390467

theorem product_of_real_parts : ∃ (z₁ z₂ : ℂ),
  (z₁^2 - 4*z₁ = 3*Complex.I) ∧
  (z₂^2 - 4*z₂ = 3*Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = -1/2) := by
  sorry

end product_of_real_parts_l3904_390467


namespace no_finite_algorithm_for_infinite_sum_l3904_390443

-- Define what an algorithm is
def Algorithm : Type := ℕ → ℕ

-- Define the property of finiteness for algorithms
def IsFinite (a : Algorithm) : Prop := ∃ n : ℕ, ∀ m : ℕ, m ≥ n → a m = a n

-- Define the infinite sum
def InfiniteSum : ℕ → ℕ
  | 0 => 0
  | n + 1 => InfiniteSum n + (n + 1)

-- Theorem: There is no finite algorithm that can calculate the infinite sum
theorem no_finite_algorithm_for_infinite_sum :
  ¬∃ (a : Algorithm), (IsFinite a) ∧ (∀ n : ℕ, a n = InfiniteSum n) :=
sorry

end no_finite_algorithm_for_infinite_sum_l3904_390443


namespace x_range_l3904_390433

def p (x : ℝ) : Prop := x^2 - 4*x + 3 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

theorem x_range (x : ℝ) :
  (∀ y : ℝ, ¬(p y ∧ q y)) ∧ (∃ y : ℝ, p y ∨ q y) →
  ((1 < x ∧ x ≤ 2) ∨ x = 3) :=
by sorry

end x_range_l3904_390433


namespace trigonometric_expression_value_l3904_390449

theorem trigonometric_expression_value (α : ℝ) 
  (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) :
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3 / 40 := by
sorry

end trigonometric_expression_value_l3904_390449


namespace simplify_expression_l3904_390471

theorem simplify_expression (x y : ℝ) : 8*x + 5*y + 3 - 2*x + 9*y + 15 = 6*x + 14*y + 18 := by
  sorry

end simplify_expression_l3904_390471


namespace antons_winning_strategy_l3904_390476

theorem antons_winning_strategy :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
  (∀ x : ℕ, 
    let n := f x
    ¬ ∃ m : ℕ, n = m * m ∧  -- n is not a perfect square
    ∃ k : ℕ, n + (n + 1) + (n + 2) = k * k) -- sum of three consecutive numbers starting from n is a perfect square
  := by sorry

end antons_winning_strategy_l3904_390476


namespace remainder_double_n_l3904_390459

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end remainder_double_n_l3904_390459


namespace prob_region_D_total_prob_is_one_l3904_390460

/-- Represents the regions on the wheel of fortune -/
inductive Region
| A
| B
| C
| D

/-- The probability function for the wheel of fortune -/
def P : Region → ℚ
| Region.A => 1/4
| Region.B => 1/3
| Region.C => 1/6
| Region.D => 1 - (1/4 + 1/3 + 1/6)

/-- The theorem stating that the probability of landing on region D is 1/4 -/
theorem prob_region_D : P Region.D = 1/4 := by
  sorry

/-- The sum of all probabilities is 1 -/
theorem total_prob_is_one : P Region.A + P Region.B + P Region.C + P Region.D = 1 := by
  sorry

end prob_region_D_total_prob_is_one_l3904_390460


namespace min_sum_of_product_2450_l3904_390483

theorem min_sum_of_product_2450 (a b c : ℕ+) (h : a * b * c = 2450) :
  (∀ x y z : ℕ+, x * y * z = 2450 → a + b + c ≤ x + y + z) ∧ a + b + c = 82 := by
  sorry

end min_sum_of_product_2450_l3904_390483


namespace triangle_area_l3904_390447

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the tangent line at (1, 1)
def tangent_line (x : ℝ) : ℝ := 3*x - 2

-- Define the x-axis (y = 0)
def x_axis (x : ℝ) : ℝ := 0

-- Define the vertical line x = 2
def vertical_line : ℝ := 2

-- Theorem statement
theorem triangle_area : 
  let x_intercept : ℝ := 2/3
  let height : ℝ := tangent_line vertical_line
  (1/2) * (vertical_line - x_intercept) * height = 8/3 := by sorry

end triangle_area_l3904_390447


namespace equation_one_solutions_equation_two_solution_equation_three_solutions_equation_four_solutions_l3904_390424

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  (x + 3)^2 = (1 - 2*x)^2 ↔ x = 4 ∨ x = -2/3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 1)^2 = 4*x ↔ x = 1 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) :
  2*x^2 - 5*x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) :
  (2*x - 1)^2 = x*(3*x + 2) - 7 ↔ x = 4 ∨ x = 2 := by sorry

end equation_one_solutions_equation_two_solution_equation_three_solutions_equation_four_solutions_l3904_390424


namespace problem_solution_l3904_390413

theorem problem_solution : ∀ (P Q Y : ℚ),
  P = 3012 / 4 →
  Q = P / 2 →
  Y = P - Q →
  Y = 376.5 := by
sorry

end problem_solution_l3904_390413


namespace find_number_l3904_390464

theorem find_number : ∃! x : ℝ, (8 * x + 5400) / 12 = 530 := by
  sorry

end find_number_l3904_390464


namespace expression_value_l3904_390455

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = 5) :
  (x^4 + 2*y^2) / 6 = 11 := by
  sorry

end expression_value_l3904_390455


namespace principal_calculation_l3904_390438

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, the principal is 10040.625 -/
theorem principal_calculation :
  let interest : ℚ := 4016.25
  let rate : ℚ := 8
  let time : ℚ := 5
  calculate_principal interest rate time = 10040.625 := by
  sorry

end principal_calculation_l3904_390438


namespace count_lines_4x4_grid_l3904_390461

/-- A point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- A line in a 2D grid -/
structure GridLine where
  points : Set GridPoint

/-- A 4-by-4 grid of lattice points -/
def Grid4x4 : Set GridPoint :=
  {p | p.x < 4 ∧ p.y < 4}

/-- A function that determines if a line passes through at least two points in the grid -/
def passesThrough2Points (l : GridLine) (grid : Set GridPoint) : Prop :=
  (l.points ∩ grid).ncard ≥ 2

/-- The set of all lines that pass through at least two points in the 4-by-4 grid -/
def validLines : Set GridLine :=
  {l | passesThrough2Points l Grid4x4}

theorem count_lines_4x4_grid :
  (validLines).ncard = 88 := by sorry

end count_lines_4x4_grid_l3904_390461


namespace marcus_pies_l3904_390463

def pies_left (batch_size : ℕ) (num_batches : ℕ) (dropped_pies : ℕ) : ℕ :=
  batch_size * num_batches - dropped_pies

theorem marcus_pies :
  pies_left 5 7 8 = 27 := by
  sorry

end marcus_pies_l3904_390463


namespace border_area_is_198_l3904_390412

-- Define the photograph dimensions
def photo_height : ℕ := 12
def photo_width : ℕ := 15

-- Define the frame border width
def border_width : ℕ := 3

-- Define the function to calculate the area of the border
def border_area (h w b : ℕ) : ℕ :=
  (h + 2*b) * (w + 2*b) - h * w

-- Theorem statement
theorem border_area_is_198 :
  border_area photo_height photo_width border_width = 198 := by
  sorry

end border_area_is_198_l3904_390412


namespace problem_statement_l3904_390419

theorem problem_statement (x y z : ℚ) (hx : x = 4/3) (hy : y = 3/4) (hz : z = 3/2) :
  (1/2) * x^6 * y^7 * z^4 = 243/128 := by
  sorry

end problem_statement_l3904_390419


namespace equation_solutions_l3904_390452

theorem equation_solutions (x : ℝ) :
  x ∈ Set.Ioo 0 π ∧ (Real.sin x + Real.cos x) * Real.tan x = 2 * Real.cos x ↔
  x = (1/2) * (Real.arctan 3 + Real.arcsin (Real.sqrt 10 / 10)) ∨
  x = (1/2) * (π - Real.arcsin (Real.sqrt 10 / 10) + Real.arctan 3) :=
by sorry

end equation_solutions_l3904_390452


namespace lcm_of_26_and_16_l3904_390454

theorem lcm_of_26_and_16 :
  let n : ℕ := 26
  let m : ℕ := 16
  let gcf : ℕ := 8
  Nat.lcm n m = 52 ∧ Nat.gcd n m = gcf :=
by sorry

end lcm_of_26_and_16_l3904_390454


namespace pencil_distribution_l3904_390477

/-- Given an initial number of pencils, number of containers, and additional pencils,
    calculate the number of pencils that can be evenly distributed per container. -/
def evenDistribution (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) : ℕ :=
  (initialPencils + additionalPencils) / containers

/-- Prove that given 150 initial pencils, 5 containers, and 30 additional pencils,
    the number of pencils that can be evenly distributed between the containers
    after receiving additional pencils is 36. -/
theorem pencil_distribution :
  evenDistribution 150 5 30 = 36 := by
  sorry

end pencil_distribution_l3904_390477


namespace red_markers_count_l3904_390426

def total_markers : ℕ := 105
def blue_markers : ℕ := 64

theorem red_markers_count : 
  ∃ (red_markers : ℕ), red_markers = total_markers - blue_markers ∧ red_markers = 41 := by
  sorry

end red_markers_count_l3904_390426


namespace triangle_smallest_side_l3904_390481

theorem triangle_smallest_side (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_inequality : a^2 + b^2 > 5 * c^2) : c < a ∧ c < b := by
  sorry

end triangle_smallest_side_l3904_390481


namespace max_candy_leftover_l3904_390441

theorem max_candy_leftover (x : ℕ+) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ 0 < r ∧ r ≤ 11 :=
by sorry

end max_candy_leftover_l3904_390441


namespace ellipse_set_is_ellipse_l3904_390439

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] :=
  (A B : α)

/-- The set of points P such that PA + PB = 2AB -/
def EllipseSet (α : Type*) [NormedAddCommGroup α] (points : FixedPoints α) : Set α :=
  {P : α | ‖P - points.A‖ + ‖P - points.B‖ = 2 * ‖points.A - points.B‖}

/-- Definition of an ellipse with given foci and major axis -/
def Ellipse (α : Type*) [NormedAddCommGroup α] (F₁ F₂ : α) (major_axis : ℝ) : Set α :=
  {P : α | ‖P - F₁‖ + ‖P - F₂‖ = major_axis}

/-- Theorem stating that the set of points P such that PA + PB = 2AB 
    forms an ellipse with A and B as foci and major axis 2AB -/
theorem ellipse_set_is_ellipse (α : Type*) [NormedAddCommGroup α] (points : FixedPoints α) :
  EllipseSet α points = Ellipse α points.A points.B (2 * ‖points.A - points.B‖) := by
  sorry

end ellipse_set_is_ellipse_l3904_390439


namespace remainder_after_addition_l3904_390444

theorem remainder_after_addition : Int.mod (3452179 + 50) 7 = 4 := by
  sorry

end remainder_after_addition_l3904_390444


namespace power_division_l3904_390470

theorem power_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end power_division_l3904_390470


namespace uniform_purchase_theorem_l3904_390448

/-- Represents the price per set based on the number of sets purchased -/
def price_per_set (n : ℕ) : ℕ :=
  if n ≤ 50 then 50
  else if n ≤ 90 then 40
  else 30

/-- The total number of students in both classes -/
def total_students : ℕ := 92

/-- The range of students in Class A -/
def class_a_range (n : ℕ) : Prop := 51 < n ∧ n < 55

/-- The total amount paid when classes purchase uniforms separately -/
def separate_purchase_total : ℕ := 4080

/-- Theorem stating the number of students in each class and the most cost-effective plan -/
theorem uniform_purchase_theorem (class_a class_b : ℕ) :
  class_a + class_b = total_students →
  class_a_range class_a →
  price_per_set class_a * class_a + price_per_set class_b * class_b = separate_purchase_total →
  (class_a = 52 ∧ class_b = 40) ∧
  price_per_set 91 * 91 = 2730 ∧
  ∀ n : ℕ, n ≤ total_students - 8 → price_per_set 91 * 91 ≤ price_per_set n * n :=
by sorry

end uniform_purchase_theorem_l3904_390448


namespace cylinder_lateral_surface_area_l3904_390488

/-- The lateral surface area of a cylinder with given base circumference and height -/
def lateral_surface_area (base_circumference : ℝ) (height : ℝ) : ℝ :=
  base_circumference * height

/-- Theorem: The lateral surface area of a cylinder with base circumference 5cm and height 2cm is 10 cm² -/
theorem cylinder_lateral_surface_area :
  lateral_surface_area 5 2 = 10 := by
sorry

end cylinder_lateral_surface_area_l3904_390488


namespace sine_matrix_det_zero_l3904_390404

/-- The determinant of a 3x3 matrix with sine entries is zero -/
theorem sine_matrix_det_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]
  Matrix.det A = 0 := by
  sorry

end sine_matrix_det_zero_l3904_390404


namespace round_trip_distance_l3904_390485

/-- Proves that the total distance of a round trip is 2 miles given the specified conditions -/
theorem round_trip_distance
  (outbound_time : ℝ) (return_time : ℝ) (average_speed : ℝ)
  (h1 : outbound_time = 10) -- outbound time in minutes
  (h2 : return_time = 20) -- return time in minutes
  (h3 : average_speed = 4) -- average speed in miles per hour
  : (outbound_time + return_time) / 60 * average_speed = 2 := by
  sorry

#check round_trip_distance

end round_trip_distance_l3904_390485


namespace students_playing_neither_l3904_390406

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 11 := by
sorry

end students_playing_neither_l3904_390406


namespace express_regular_train_speed_ratio_l3904_390490

/-- The ratio of speeds between an express train and a regular train -/
def speed_ratio : ℝ := 2.5

/-- The time taken by the regular train from Moscow to St. Petersburg -/
def regular_train_time : ℝ := 10

/-- The time difference in arrival between regular and express trains -/
def arrival_time_difference : ℝ := 3

/-- The waiting time for the express train -/
def express_train_wait_time : ℝ := 3

/-- The time after departure when both trains are at the same distance from Moscow -/
def equal_distance_time : ℝ := 2

theorem express_regular_train_speed_ratio :
  ∀ (v_regular v_express : ℝ),
    v_regular > 0 →
    v_express > 0 →
    express_train_wait_time > 2.5 →
    v_express * equal_distance_time = v_regular * (express_train_wait_time + equal_distance_time) →
    v_express * (regular_train_time - arrival_time_difference - express_train_wait_time) = v_regular * regular_train_time →
    v_express / v_regular = speed_ratio := by
  sorry

end express_regular_train_speed_ratio_l3904_390490


namespace finite_painted_blocks_l3904_390492

theorem finite_painted_blocks : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    ∀ m n r : ℕ, 
      m * n * r = 2 * (m - 2) * (n - 2) * (r - 2) → 
      (m, n, r) ∈ S := by
sorry

end finite_painted_blocks_l3904_390492


namespace rectangular_field_dimensions_l3904_390420

/-- Theorem: For a rectangular field with length twice its width and perimeter 600 meters,
    the width is 100 meters and the length is 200 meters. -/
theorem rectangular_field_dimensions :
  ∀ (width length : ℝ),
  length = 2 * width →
  2 * (length + width) = 600 →
  width = 100 ∧ length = 200 :=
by sorry

end rectangular_field_dimensions_l3904_390420


namespace opposite_teal_is_yellow_l3904_390405

-- Define the colors
inductive Color
| Blue | Orange | Yellow | Violet | Teal | Lime

-- Define the faces of a cube
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define a cube as a function from Face to Color
def Cube := Face → Color

-- Define the property of opposite faces
def opposite (f1 f2 : Face) : Prop :=
  (f1 = Face.Top ∧ f2 = Face.Bottom) ∨
  (f1 = Face.Bottom ∧ f2 = Face.Top) ∨
  (f1 = Face.Left ∧ f2 = Face.Right) ∨
  (f1 = Face.Right ∧ f2 = Face.Left) ∨
  (f1 = Face.Front ∧ f2 = Face.Back) ∨
  (f1 = Face.Back ∧ f2 = Face.Front)

-- Define the views of the cube
def view1 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Violet ∧ c Face.Right = Color.Yellow

def view2 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Orange ∧ c Face.Right = Color.Yellow

def view3 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Lime ∧ c Face.Right = Color.Yellow

-- Theorem statement
theorem opposite_teal_is_yellow (c : Cube) :
  (∃ f1 f2 : Face, c f1 = Color.Teal ∧ opposite f1 f2) →
  (∀ f : Face, c f ≠ Color.Teal → c f ≠ Color.Yellow) →
  view1 c → view2 c → view3 c →
  ∃ f : Face, c f = Color.Teal ∧ c (Face.Right) = Color.Yellow ∧ opposite f Face.Right :=
by sorry

end opposite_teal_is_yellow_l3904_390405


namespace fibonacci_gcd_property_l3904_390458

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_gcd_property :
  Nat.gcd (fib 2017) (fib 99 * fib 101 + 1) = 1 := by
  sorry

end fibonacci_gcd_property_l3904_390458


namespace notebook_redistribution_l3904_390425

theorem notebook_redistribution (total_notebooks : ℕ) (initial_boxes : ℕ) (new_notebooks_per_box : ℕ) :
  total_notebooks = 1200 →
  initial_boxes = 30 →
  new_notebooks_per_box = 35 →
  total_notebooks % new_notebooks_per_box = 10 :=
by
  sorry

end notebook_redistribution_l3904_390425


namespace binomial_expansion_terms_l3904_390402

theorem binomial_expansion_terms (x a : ℚ) (n : ℕ) :
  (Nat.choose n 3 : ℚ) * x^(n - 3) * a^3 = 330 ∧
  (Nat.choose n 4 : ℚ) * x^(n - 4) * a^4 = 792 ∧
  (Nat.choose n 5 : ℚ) * x^(n - 5) * a^5 = 1716 →
  n = 7 :=
by sorry

end binomial_expansion_terms_l3904_390402


namespace painting_problem_l3904_390462

/-- The fraction of a wall that can be painted by two people working together in a given time -/
def combined_painting_fraction (rate1 rate2 time : ℚ) : ℚ :=
  (rate1 + rate2) * time

theorem painting_problem :
  let heidi_rate : ℚ := 1 / 60
  let linda_rate : ℚ := 1 / 40
  let work_time : ℚ := 12
  combined_painting_fraction heidi_rate linda_rate work_time = 1 / 2 := by
sorry

end painting_problem_l3904_390462


namespace abby_coins_l3904_390486

theorem abby_coins (total_coins : ℕ) (total_value : ℚ) 
  (h_total_coins : total_coins = 23)
  (h_total_value : total_value = 455/100)
  : ∃ (quarters nickels : ℕ),
    quarters + nickels = total_coins ∧
    (25 * quarters + 5 * nickels : ℚ) / 100 = total_value ∧
    quarters = 17 := by
  sorry

end abby_coins_l3904_390486


namespace no_convex_polygon_from_regular_triangles_l3904_390495

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields

/-- A regular triangle -/
structure RegularTriangle where
  -- Add necessary fields

/-- Predicate to check if triangles are non-overlapping -/
def non_overlapping (T : List RegularTriangle) : Prop :=
  sorry

/-- Predicate to check if triangles are distinct -/
def distinct (T : List RegularTriangle) : Prop :=
  sorry

/-- Predicate to check if a polygon is composed of given triangles -/
def composed_of (P : ConvexPolygon) (T : List RegularTriangle) : Prop :=
  sorry

theorem no_convex_polygon_from_regular_triangles 
  (P : ConvexPolygon) (T : List RegularTriangle) :
  T.length ≥ 2 → non_overlapping T → distinct T → ¬(composed_of P T) :=
sorry

end no_convex_polygon_from_regular_triangles_l3904_390495


namespace max_metro_speed_l3904_390423

/-- Represents the metro system and the students' travel scenario -/
structure MetroSystem where
  v : ℕ  -- Speed of metro trains in km/h
  S : ℝ  -- Distance between two nearest metro stations
  R : ℝ  -- Distance from home to nearest station

/-- Conditions for the metro system -/
def validMetroSystem (m : MetroSystem) : Prop :=
  m.S > 0 ∧ m.R > 0 ∧ m.R < m.S / 2

/-- Yegor's travel condition -/
def yegorCondition (m : MetroSystem) : Prop :=
  m.S / 24 > m.R / m.v

/-- Nikita's travel condition -/
def nikitaCondition (m : MetroSystem) : Prop :=
  m.S / 12 < (m.R + m.S) / m.v

/-- The maximum speed theorem -/
theorem max_metro_speed :
  ∃ (m : MetroSystem),
    validMetroSystem m ∧
    yegorCondition m ∧
    nikitaCondition m ∧
    (∀ (m' : MetroSystem),
      validMetroSystem m' ∧ yegorCondition m' ∧ nikitaCondition m' →
      m'.v ≤ m.v) ∧
    m.v = 23 := by
  sorry

end max_metro_speed_l3904_390423


namespace zero_not_in_range_of_g_l3904_390450

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end zero_not_in_range_of_g_l3904_390450


namespace intersection_A_B_l3904_390416

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}

-- Define set B
def B : Set ℝ := {-4, 1, 3, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end intersection_A_B_l3904_390416


namespace january_savings_l3904_390440

def savings_sequence (initial : ℝ) (n : ℕ) : ℝ :=
  initial + 4 * (n - 1)

def total_savings (initial : ℝ) (months : ℕ) : ℝ :=
  (List.range months).map (savings_sequence initial) |>.sum

theorem january_savings (x : ℝ) : total_savings x 6 = 126 → x = 11 := by
  sorry

end january_savings_l3904_390440


namespace circle_tangent_properties_l3904_390401

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define a point on the x-axis
def on_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0

-- Define the tangent property
def are_tangents (Q A B : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (P₁ P₂ : ℝ × ℝ) : ℝ := sorry

-- Define a line passing through a point
def line_passes_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := l P

-- The main theorem
theorem circle_tangent_properties :
  ∀ (Q A B : ℝ × ℝ),
  circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
  on_x_axis Q ∧
  are_tangents Q A B →
  (distance A B = 4 * Real.sqrt 2 / 3 → distance (0, 2) Q = 3) ∧
  (∃ (l : ℝ × ℝ → Prop), ∀ (Q' : ℝ × ℝ), on_x_axis Q' ∧ are_tangents Q' A B → 
    line_passes_through (0, 3/2) l) :=
by sorry

end circle_tangent_properties_l3904_390401


namespace angle_A_is_135_l3904_390410

/-- A trapezoid with specific angle relationships -/
structure SpecialTrapezoid where
  /-- The measure of angle A in degrees -/
  A : ℝ
  /-- The measure of angle B in degrees -/
  B : ℝ
  /-- The measure of angle C in degrees -/
  C : ℝ
  /-- The measure of angle D in degrees -/
  D : ℝ
  /-- AB is parallel to CD -/
  parallel : A + D = 180
  /-- Angle A is three times angle D -/
  A_eq_3D : A = 3 * D
  /-- Angle C is four times angle B -/
  C_eq_4B : C = 4 * B

/-- The measure of angle A in a special trapezoid is 135 degrees -/
theorem angle_A_is_135 (t : SpecialTrapezoid) : t.A = 135 := by
  sorry

end angle_A_is_135_l3904_390410


namespace p_not_sufficient_p_not_necessary_p_neither_sufficient_nor_necessary_l3904_390442

/-- Proposition p: x ≠ 2 and y ≠ 3 -/
def p (x y : ℝ) : Prop := x ≠ 2 ∧ y ≠ 3

/-- Proposition q: x + y ≠ 5 -/
def q (x y : ℝ) : Prop := x + y ≠ 5

/-- p is not a sufficient condition for q -/
theorem p_not_sufficient : ¬∀ x y : ℝ, p x y → q x y :=
sorry

/-- p is not a necessary condition for q -/
theorem p_not_necessary : ¬∀ x y : ℝ, q x y → p x y :=
sorry

/-- p is neither a sufficient nor a necessary condition for q -/
theorem p_neither_sufficient_nor_necessary : (¬∀ x y : ℝ, p x y → q x y) ∧ (¬∀ x y : ℝ, q x y → p x y) :=
sorry

end p_not_sufficient_p_not_necessary_p_neither_sufficient_nor_necessary_l3904_390442


namespace trapezoid_area_theorem_l3904_390409

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  midpoint_segment : ℝ

/-- The area of a trapezoid with the given properties -/
def trapezoid_area (t : Trapezoid) : ℝ := 6

/-- Theorem: The area of a trapezoid with diagonals 3 and 5, and midpoint segment 2, is 6 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.diagonal1 = 3) 
  (h2 : t.diagonal2 = 5) 
  (h3 : t.midpoint_segment = 2) : 
  trapezoid_area t = 6 := by
  sorry

#check trapezoid_area_theorem

end trapezoid_area_theorem_l3904_390409


namespace condition_property_l3904_390482

theorem condition_property (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end condition_property_l3904_390482


namespace consecutive_integers_sqrt_19_l3904_390431

theorem consecutive_integers_sqrt_19 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 19) → (Real.sqrt 19 < b) → (a + b = 9) := by
  sorry

end consecutive_integers_sqrt_19_l3904_390431


namespace half_abs_diff_squares_25_23_l3904_390418

theorem half_abs_diff_squares_25_23 : (1 / 2 : ℝ) * |25^2 - 23^2| = 48 := by
  sorry

end half_abs_diff_squares_25_23_l3904_390418


namespace necessary_not_sufficient_condition_l3904_390479

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x y : ℝ, x < y → x < y + 1) ∧ 
  (∃ x y : ℝ, x < y + 1 ∧ ¬(x < y)) := by
  sorry

end necessary_not_sufficient_condition_l3904_390479


namespace polynomial_perfect_square_l3904_390415

theorem polynomial_perfect_square (k : ℚ) : 
  (∃ a : ℚ, ∀ x : ℚ, x^2 + 2*(k-9)*x + (k^2 + 3*k + 4) = (x + a)^2) ↔ k = 11/3 := by
sorry

end polynomial_perfect_square_l3904_390415


namespace geometric_progression_proof_l3904_390446

theorem geometric_progression_proof (b : ℕ → ℚ) 
  (h1 : b 4 - b 2 = -45/32) 
  (h2 : b 6 - b 4 = -45/512) 
  (h_geom : ∀ n : ℕ, b (n + 1) = b 1 * (b 2 / b 1) ^ n) :
  ((b 1 = -6 ∧ b 2 / b 1 = -1/4) ∨ (b 1 = 6 ∧ b 2 / b 1 = 1/4)) :=
sorry

end geometric_progression_proof_l3904_390446


namespace min_bottles_to_fill_l3904_390411

theorem min_bottles_to_fill (large_capacity : ℕ) (small_capacity1 small_capacity2 : ℕ) :
  large_capacity = 720 ∧ small_capacity1 = 40 ∧ small_capacity2 = 45 →
  ∃ (x y : ℕ), x * small_capacity1 + y * small_capacity2 = large_capacity ∧
                x + y = 16 ∧
                ∀ (a b : ℕ), a * small_capacity1 + b * small_capacity2 = large_capacity →
                              x + y ≤ a + b :=
by sorry

end min_bottles_to_fill_l3904_390411


namespace contrapositive_false_l3904_390451

theorem contrapositive_false : ¬(∀ x y : ℝ, (x ≤ 0 ∨ y ≤ 0) → x + y ≤ 0) := by
  sorry

end contrapositive_false_l3904_390451


namespace inheritance_tax_problem_l3904_390493

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x) + (0.15 * (x - 0.25 * x)) = 15000 → x = 41379 :=
by sorry

end inheritance_tax_problem_l3904_390493


namespace smallest_integer_solution_of_inequalities_l3904_390478

theorem smallest_integer_solution_of_inequalities :
  ∀ x : ℤ,
  (5 * x + 7 > 3 * (x + 1)) ∧
  (1 - (3/2) * x ≤ (1/2) * x - 1) →
  x ≥ 1 ∧
  ∀ y : ℤ, y < 1 →
    ¬((5 * y + 7 > 3 * (y + 1)) ∧
      (1 - (3/2) * y ≤ (1/2) * y - 1)) :=
by sorry

end smallest_integer_solution_of_inequalities_l3904_390478


namespace scale_division_l3904_390430

/-- Represents the length of an object in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Length to total inches -/
def Length.toInches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- Converts total inches to a Length -/
def inchesToLength (totalInches : ℕ) : Length :=
  { feet := totalInches / 12,
    inches := totalInches % 12,
    h := by
      apply Nat.mod_lt
      exact Nat.zero_lt_succ 11 }

theorem scale_division (scale : Length) (h : scale.feet = 6 ∧ scale.inches = 8) :
  let totalInches := scale.toInches
  let halfInches := totalInches / 2
  let halfLength := inchesToLength halfInches
  halfLength.feet = 3 ∧ halfLength.inches = 4 := by
  sorry


end scale_division_l3904_390430


namespace mean_score_approx_71_l3904_390497

/-- Calculates the mean score of all students given the mean scores of two classes and the ratio of students in those classes. -/
def meanScoreAllStudents (morningMean afternoon_mean : ℚ) (morningStudents afternoonStudents : ℕ) : ℚ :=
  let totalStudents := morningStudents + afternoonStudents
  let totalScore := morningMean * morningStudents + afternoon_mean * afternoonStudents
  totalScore / totalStudents

/-- Proves that the mean score of all students is approximately 71 given the specified conditions. -/
theorem mean_score_approx_71 :
  ∃ (m a : ℕ), m > 0 ∧ a > 0 ∧ m = (5 * a) / 7 ∧ 
  abs (meanScoreAllStudents 78 65 m a - 71) < 1 :=
sorry


end mean_score_approx_71_l3904_390497


namespace range_of_a_l3904_390432

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x_0 : ℝ, x_0^2 + 2*a*x_0 + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end range_of_a_l3904_390432


namespace f_at_one_eq_neg_7878_l3904_390487

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 120*x + c

/-- Theorem stating that f(1) = -7878 under given conditions -/
theorem f_at_one_eq_neg_7878 (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →  -- g has three distinct roots
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →  -- Each root of g is a root of f
  g a (-100) = 0 →  -- -100 is a root of g
  f b c 1 = -7878 :=
by sorry

end f_at_one_eq_neg_7878_l3904_390487


namespace smallest_eight_digit_four_fours_l3904_390436

def is_eight_digit (n : ℕ) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).filter (· = d) |>.length

theorem smallest_eight_digit_four_fours : 
  ∀ n : ℕ, is_eight_digit n → count_digit n 4 = 4 → 10004444 ≤ n :=
sorry

end smallest_eight_digit_four_fours_l3904_390436


namespace average_after_removing_two_l3904_390417

def initial_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,12]

def remove_element (list : List ℕ) (elem : ℕ) : List ℕ :=
  list.filter (λ x => x ≠ elem)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem average_after_removing_two :
  average (remove_element initial_list 2) = 13/2 := by
  sorry

end average_after_removing_two_l3904_390417


namespace coffee_x_ratio_l3904_390499

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a coffee mixture -/
structure CoffeeMixture where
  p : ℕ  -- amount of coffee p in lbs
  v : ℕ  -- amount of coffee v in lbs

def total_p : ℕ := 24
def total_v : ℕ := 25

def coffee_x : CoffeeMixture := { p := 20, v := 0 }
def coffee_y : CoffeeMixture := { p := 0, v := 0 }

def ratio_y : Ratio := { numerator := 1, denominator := 5 }

theorem coffee_x_ratio : 
  coffee_x.p * 1 = coffee_x.v * 4 := by sorry

end coffee_x_ratio_l3904_390499


namespace largest_angle_in_ratio_triangle_l3904_390472

theorem largest_angle_in_ratio_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ratio : b = 2 * a ∧ c = 3 * a) (h_sum : a + b + c = 180) :
  c = 90 := by
  sorry

end largest_angle_in_ratio_triangle_l3904_390472


namespace xyz_sum_eq_32_l3904_390428

theorem xyz_sum_eq_32 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 48)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 64) :
  x*y + y*z + x*z = 32 := by
sorry

end xyz_sum_eq_32_l3904_390428


namespace sin_beta_value_l3904_390429

theorem sin_beta_value (α β : ℝ) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5) : 
  Real.sin β = -(3/5) := by
  sorry

end sin_beta_value_l3904_390429


namespace last_five_shots_made_l3904_390437

/-- Represents the number of shots made in a series of basketball attempts -/
structure ShotsMade where
  total : ℕ
  made : ℕ

/-- Calculates the shooting percentage -/
def shootingPercentage (s : ShotsMade) : ℚ :=
  s.made / s.total

theorem last_five_shots_made
  (initial : ShotsMade)
  (second : ShotsMade)
  (final : ShotsMade)
  (h1 : initial.total = 30)
  (h2 : shootingPercentage initial = 2/5)
  (h3 : second.total = initial.total + 10)
  (h4 : shootingPercentage second = 9/20)
  (h5 : final.total = second.total + 5)
  (h6 : shootingPercentage final = 23/50)
  : final.made - second.made = 2 := by
  sorry

end last_five_shots_made_l3904_390437


namespace fourth_day_temperature_l3904_390489

def temperature_problem (t1 t2 t3 t4 : ℤ) : Prop :=
  let temps := [t1, t2, t3, t4]
  (t1 = -36) ∧ (t2 = 13) ∧ (t3 = -10) ∧ 
  (temps.sum / temps.length = -12) ∧
  (t4 = -15)

theorem fourth_day_temperature :
  ∃ t4 : ℤ, temperature_problem (-36) 13 (-10) t4 := by
  sorry

end fourth_day_temperature_l3904_390489


namespace pave_square_iff_integer_hypotenuse_l3904_390435

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- Length of side AB
  b : ℕ  -- Length of side AC
  h : a^2 + b^2 > 0  -- Ensures the triangle is non-degenerate

/-- Checks if a square can be completely paved with a given right triangle -/
def can_pave_square (t : RightTriangle) : Prop :=
  ∃ (n : ℕ), ∃ (m : ℕ), m * (t.a * t.b) = 2 * n^2 * (t.a^2 + t.b^2)

/-- The main theorem: A square can be paved if and only if the hypotenuse is an integer -/
theorem pave_square_iff_integer_hypotenuse (t : RightTriangle) :
  can_pave_square t ↔ ∃ (k : ℕ), k^2 = t.a^2 + t.b^2 :=
sorry

end pave_square_iff_integer_hypotenuse_l3904_390435


namespace octal_sum_example_l3904_390408

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Converts a natural number to its octal representation --/
def toOctal (n : Nat) : OctalNumber := sorry

/-- Adds two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber := sorry

/-- Theorem: The sum of 356₈, 672₈, and 145₈ is 1477₈ in base 8 --/
theorem octal_sum_example : 
  octalAdd (octalAdd (toOctal 356) (toOctal 672)) (toOctal 145) = toOctal 1477 := by sorry

end octal_sum_example_l3904_390408


namespace lowest_sample_number_48_8_48_l3904_390445

/-- Calculates the lowest number in a systematic sample. -/
def lowestSampleNumber (totalPopulation : ℕ) (sampleSize : ℕ) (highestNumber : ℕ) : ℕ :=
  highestNumber - (totalPopulation / sampleSize) * (sampleSize - 1)

/-- Theorem: In a systematic sampling of 8 students from 48, with highest number 48, the lowest is 6. -/
theorem lowest_sample_number_48_8_48 :
  lowestSampleNumber 48 8 48 = 6 := by
  sorry

#eval lowestSampleNumber 48 8 48

end lowest_sample_number_48_8_48_l3904_390445
