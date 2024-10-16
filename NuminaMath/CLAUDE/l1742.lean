import Mathlib

namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l1742_174266

theorem equilateral_triangle_side_length_squared 
  (α β γ : ℂ) (s t : ℂ) :
  (∀ z, z^3 + s*z + t = 0 ↔ z = α ∨ z = β ∨ z = γ) →
  Complex.abs α ^ 2 + Complex.abs β ^ 2 + Complex.abs γ ^ 2 = 360 →
  ∃ l : ℝ, l > 0 ∧ 
    Complex.abs (α - β) = l ∧
    Complex.abs (β - γ) = l ∧
    Complex.abs (γ - α) = l →
  Complex.abs (α - β) ^ 2 = 360 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l1742_174266


namespace NUMINAMATH_CALUDE_impossible_to_transform_to_fives_l1742_174219

/-- Represents the three magician tricks -/
inductive MagicTrick
  | subtract_one
  | divide_by_two
  | multiply_by_three

/-- Represents the state of the transformation process -/
structure TransformState where
  numbers : List ℕ
  trick_counts : List ℕ
  deriving Repr

/-- Checks if a number is within the allowed range -/
def is_valid_number (n : ℕ) : Bool :=
  n ≤ 10

/-- Applies a magic trick to a number -/
def apply_trick (trick : MagicTrick) (n : ℕ) : Option ℕ :=
  match trick with
  | MagicTrick.subtract_one => if n > 0 then some (n - 1) else none
  | MagicTrick.divide_by_two => if n % 2 = 0 then some (n / 2) else none
  | MagicTrick.multiply_by_three => if n * 3 ≤ 10 then some (n * 3) else none

/-- Checks if the transformation is complete (all numbers are 5) -/
def is_transformation_complete (state : TransformState) : Bool :=
  state.numbers.all (· = 5)

/-- Checks if the transformation process is still valid -/
def is_valid_state (state : TransformState) : Bool :=
  state.numbers.all is_valid_number ∧
  state.trick_counts.all (· ≤ 5)

/-- The main theorem statement -/
theorem impossible_to_transform_to_fives :
  ¬ ∃ (final_state : TransformState),
    is_transformation_complete final_state ∧
    is_valid_state final_state ∧
    (∃ (initial_state : TransformState),
      initial_state.numbers = [3, 8, 9, 2, 4] ∧
      initial_state.trick_counts = [0, 0, 0] ∧
      -- There exists a sequence of valid transformations from initial_state to final_state
      True) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_transform_to_fives_l1742_174219


namespace NUMINAMATH_CALUDE_four_x_plus_g_is_odd_l1742_174243

theorem four_x_plus_g_is_odd (x g : ℤ) (h : 2 * x - g = 11) : 
  ∃ k : ℤ, 4 * x + g = 2 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_four_x_plus_g_is_odd_l1742_174243


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1742_174218

theorem complex_power_modulus : Complex.abs ((1/3 : ℂ) + (2/3 : ℂ) * Complex.I) ^ 8 = 625/6561 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1742_174218


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l1742_174241

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The transformation of the random variable -/
def eta (X : BinomialRV) : ℝ → ℝ := fun x ↦ -2 * x + 1

theorem variance_of_transformed_binomial (X : BinomialRV) 
  (h_n : X.n = 6) (h_p : X.p = 0.4) : 
  variance X * 4 = 5.76 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l1742_174241


namespace NUMINAMATH_CALUDE_pria_distance_driven_l1742_174280

/-- Calculates the distance driven with a full tank of gas given the advertised mileage,
    tank capacity, and difference between advertised and actual mileage. -/
def distance_driven (advertised_mileage : ℝ) (tank_capacity : ℝ) (mileage_difference : ℝ) : ℝ :=
  (advertised_mileage - mileage_difference) * tank_capacity

/-- Proves that given the specified conditions, the distance driven is 372 miles. -/
theorem pria_distance_driven :
  distance_driven 35 12 4 = 372 := by
  sorry

end NUMINAMATH_CALUDE_pria_distance_driven_l1742_174280


namespace NUMINAMATH_CALUDE_root_product_plus_one_l1742_174271

theorem root_product_plus_one (r s t : ℝ) : 
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  s^3 - 15*s^2 + 25*s - 10 = 0 →
  t^3 - 15*t^2 + 25*t - 10 = 0 →
  (1+r)*(1+s)*(1+t) = 51 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l1742_174271


namespace NUMINAMATH_CALUDE_expectation_xi_prob_usable_team_b_l1742_174249

/-- Represents the three teams harvesting ice blocks -/
inductive Team
| A
| B
| C

/-- The proportion of ice blocks harvested by each team -/
def team_proportion : Team → ℝ
| Team.A => 0.3
| Team.B => 0.3
| Team.C => 0.4

/-- The utilization rate of ice blocks for each team -/
def utilization_rate : Team → ℝ
| Team.A => 0.8
| Team.B => 0.75
| Team.C => 0.6

/-- The number of trials in the random selection process -/
def num_trials : ℕ := 3

/-- The probability of selecting a block from Team C -/
def prob_team_c : ℝ := team_proportion Team.C

/-- Theorem stating the expectation of ξ (number of times Team C's blocks are selected) -/
theorem expectation_xi : num_trials * prob_team_c = 6/5 := by sorry

/-- Theorem stating the probability that a usable block was harvested by Team B -/
theorem prob_usable_team_b : 
  (team_proportion Team.B * utilization_rate Team.B) / 
  (team_proportion Team.A * utilization_rate Team.A + 
   team_proportion Team.B * utilization_rate Team.B + 
   team_proportion Team.C * utilization_rate Team.C) = 15/47 := by sorry

end NUMINAMATH_CALUDE_expectation_xi_prob_usable_team_b_l1742_174249


namespace NUMINAMATH_CALUDE_smallest_distance_to_target_l1742_174242

def jump_distance_1 : ℕ := 364
def jump_distance_2 : ℕ := 715
def target_point : ℕ := 2010

theorem smallest_distance_to_target : 
  ∃ (x y : ℤ), 
    (∀ (a b : ℤ), |target_point - (jump_distance_1 * a + jump_distance_2 * b)| ≥ 
                   |target_point - (jump_distance_1 * x + jump_distance_2 * y)|) ∧
    |target_point - (jump_distance_1 * x + jump_distance_2 * y)| = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_to_target_l1742_174242


namespace NUMINAMATH_CALUDE_expression_simplification_l1742_174217

theorem expression_simplification (a : ℝ) (h : a^2 + 4*a + 1 = 0) :
  ((a + 2) / (a^2 - 2*a) + 8 / (4 - a^2)) / ((a^2 - 4) / a) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1742_174217


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l1742_174223

theorem initial_amount_calculation (deposit : ℚ) (initial : ℚ) : 
  deposit = 750 → 
  deposit = initial * (20 / 100) * (25 / 100) * (30 / 100) → 
  initial = 50000 := by
sorry

end NUMINAMATH_CALUDE_initial_amount_calculation_l1742_174223


namespace NUMINAMATH_CALUDE_problem_solution_l1742_174222

theorem problem_solution : 2 * Real.sin (60 * π / 180) + |Real.sqrt 3 - 3| + (π - 1)^0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1742_174222


namespace NUMINAMATH_CALUDE_glass_volume_l1742_174253

/-- The volume of a glass given pessimist and optimist perspectives --/
theorem glass_volume (V : ℝ) 
  (h_pessimist : 0.4 * V = V - 0.6 * V) 
  (h_optimist : 0.6 * V = V - 0.4 * V) 
  (h_difference : 0.6 * V - 0.4 * V = 46) : 
  V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l1742_174253


namespace NUMINAMATH_CALUDE_housing_price_growth_l1742_174276

/-- Proves that the equation relating initial housing price, final housing price, 
    and annual growth rate over two years is correct. -/
theorem housing_price_growth (initial_price final_price : ℝ) (x : ℝ) 
  (h_initial : initial_price = 5500)
  (h_final : final_price = 7000) :
  initial_price * (1 + x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_housing_price_growth_l1742_174276


namespace NUMINAMATH_CALUDE_geometry_theorem_l1742_174206

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (distinct_lines : m ≠ n) 
  (distinct_planes : α ≠ β) :
  (perpendicular m n → perpendicularLP m α → ¬subset n α → parallel n α) ∧
  (perpendicularLP m β → perpendicularPP α β → (parallel m α ∨ subset m α)) ∧
  (perpendicular m n → perpendicularLP m α → perpendicularLP n β → perpendicularPP α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1742_174206


namespace NUMINAMATH_CALUDE_dairy_water_mixture_l1742_174201

theorem dairy_water_mixture (original_price selling_price : ℝ) 
  (h1 : selling_price = original_price * 1.25) : 
  (selling_price - original_price) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_dairy_water_mixture_l1742_174201


namespace NUMINAMATH_CALUDE_book_length_l1742_174200

theorem book_length (area : ℝ) (width : ℝ) (h1 : area = 50) (h2 : width = 10) :
  area / width = 5 := by
  sorry

end NUMINAMATH_CALUDE_book_length_l1742_174200


namespace NUMINAMATH_CALUDE_common_number_exists_common_number_unique_l1742_174248

/-- Paul's sequence -/
def paul_sequence (n : ℕ) : ℤ := 3 * n - 2

/-- Penny's sequence -/
def penny_sequence (m : ℕ) : ℤ := 2022 - 5 * m

/-- The common number in both sequences -/
def common_number : ℤ := 2017

theorem common_number_exists : ∃ (n m : ℕ), 
  paul_sequence n = penny_sequence m ∧ 
  paul_sequence n = common_number :=
by sorry

theorem common_number_unique : ∀ (n m n' m' : ℕ),
  paul_sequence n = penny_sequence m → 
  paul_sequence n' = penny_sequence m' → 
  paul_sequence n = paul_sequence n' :=
by sorry

end NUMINAMATH_CALUDE_common_number_exists_common_number_unique_l1742_174248


namespace NUMINAMATH_CALUDE_greater_number_problem_l1742_174247

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * (x - y) = 12) (h3 : x > y) : x = 22 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1742_174247


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_fraction_division_l1742_174279

-- Problem 1
theorem simplify_and_evaluate (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 0) (h3 : a ≠ 1) (h4 : a ≠ 3) :
  (a^2 + a) / (a^2 - 3*a) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2 / (a^2 - 1) :=
by sorry

-- Problem 2
theorem fraction_division (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ -1) :
  (x^2 - 1) / (x - 4) / ((x + 1) / (4 - x)) = 1 - x :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_fraction_division_l1742_174279


namespace NUMINAMATH_CALUDE_max_value_f_l1742_174262

def f (x : ℝ) := x * (1 - x)

theorem max_value_f :
  ∃ (m : ℝ), ∀ (x : ℝ), 0 < x ∧ x < 1 → f x ≤ m ∧ (∃ (y : ℝ), 0 < y ∧ y < 1 ∧ f y = m) ∧ m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_f_l1742_174262


namespace NUMINAMATH_CALUDE_philips_farm_l1742_174283

theorem philips_farm (cows ducks pigs : ℕ) : 
  ducks = (3 * cows) / 2 →
  pigs = (cows + ducks) / 5 →
  cows + ducks + pigs = 60 →
  cows = 20 := by
sorry

end NUMINAMATH_CALUDE_philips_farm_l1742_174283


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1742_174264

/-- 
Theorem: If a line y = kx + 2 is tangent to the ellipse x^2/2 + 2y^2 = 2, 
then k^2 = 3/4.
-/
theorem line_tangent_to_ellipse (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2 → x^2 / 2 + 2 * y^2 = 2) →
  (∃! p : ℝ × ℝ, p.1^2 / 2 + 2 * p.2^2 = 2 ∧ p.2 = k * p.1 + 2) →
  k^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1742_174264


namespace NUMINAMATH_CALUDE_average_salary_is_8800_l1742_174236

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def number_of_people : ℕ := 5

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E

theorem average_salary_is_8800 :
  total_salary / number_of_people = 8800 := by sorry

end NUMINAMATH_CALUDE_average_salary_is_8800_l1742_174236


namespace NUMINAMATH_CALUDE_only_C_nonlinear_l1742_174295

-- Define the structure for a system of two equations
structure SystemOfEquations where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ

-- Define the systems A, B, C, and D
def systemA : SystemOfEquations := ⟨λ x y => x - 2, λ x y => y - 3⟩
def systemB : SystemOfEquations := ⟨λ x y => x + y - 1, λ x y => x - y - 2⟩
def systemC : SystemOfEquations := ⟨λ x y => x + y - 5, λ x y => x * y - 1⟩
def systemD : SystemOfEquations := ⟨λ x y => y - x, λ x y => x - 2*y - 1⟩

-- Define what it means for an equation to be linear
def isLinear (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

-- Define what it means for a system to be linear
def isLinearSystem (s : SystemOfEquations) : Prop :=
  isLinear s.eq1 ∧ isLinear s.eq2

-- Theorem statement
theorem only_C_nonlinear :
  isLinearSystem systemA ∧
  isLinearSystem systemB ∧
  ¬isLinearSystem systemC ∧
  isLinearSystem systemD :=
sorry

end NUMINAMATH_CALUDE_only_C_nonlinear_l1742_174295


namespace NUMINAMATH_CALUDE_chebyshev_polynomial_3_and_root_sum_l1742_174286

-- Define Chebyshev polynomials
def is_chebyshev_polynomial (P : ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ x, P (Real.cos x) = Real.cos (n * x)

-- Define P₃
def P₃ (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem chebyshev_polynomial_3_and_root_sum :
  ∃ (a b c d : ℝ),
    (is_chebyshev_polynomial (P₃ a b c d) 3) ∧
    (a = 4 ∧ b = 0 ∧ c = -3 ∧ d = 0) ∧
    (∃ (x₁ x₂ x₃ : ℝ),
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      x₁ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      x₂ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      x₃ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      (4 * x₁^3 - 3 * x₁ = 1/2) ∧
      (4 * x₂^3 - 3 * x₂ = 1/2) ∧
      (4 * x₃^3 - 3 * x₃ = 1/2) ∧
      x₁ + x₂ + x₃ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_chebyshev_polynomial_3_and_root_sum_l1742_174286


namespace NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l1742_174296

theorem sin_arccos_eight_seventeenths : 
  Real.sin (Real.arccos (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l1742_174296


namespace NUMINAMATH_CALUDE_loan_payment_difference_l1742_174240

/-- Calculates the monthly payment for a loan -/
def monthly_payment (loan_amount : ℚ) (months : ℕ) : ℚ :=
  loan_amount / months

/-- Represents the loan details -/
structure LoanDetails where
  amount : ℚ
  short_term_months : ℕ
  long_term_months : ℕ

theorem loan_payment_difference (loan : LoanDetails) 
  (h1 : loan.amount = 6000)
  (h2 : loan.short_term_months = 24)
  (h3 : loan.long_term_months = 60) :
  monthly_payment loan.amount loan.short_term_months - 
  monthly_payment loan.amount loan.long_term_months = 150 := by
  sorry


end NUMINAMATH_CALUDE_loan_payment_difference_l1742_174240


namespace NUMINAMATH_CALUDE_brick_width_is_10_cm_l1742_174220

/-- Prove that the width of a brick is 10 cm given the specified conditions -/
theorem brick_width_is_10_cm
  (brick_length : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_width : ℝ)
  (wall_height : ℝ)
  (num_bricks : ℕ)
  (h1 : brick_length = 20)
  (h2 : brick_height = 7.5)
  (h3 : wall_length = 2600)
  (h4 : wall_width = 200)
  (h5 : wall_height = 75)
  (h6 : num_bricks = 26000)
  (h7 : wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_height * brick_width)) :
  brick_width = 10 := by
  sorry


end NUMINAMATH_CALUDE_brick_width_is_10_cm_l1742_174220


namespace NUMINAMATH_CALUDE_root_in_interval_l1742_174255

theorem root_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ Real.exp x + Real.log x = 0 := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l1742_174255


namespace NUMINAMATH_CALUDE_katies_sister_candy_l1742_174203

theorem katies_sister_candy (katie_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) :
  katie_candy = 8 →
  eaten_candy = 8 →
  remaining_candy = 23 →
  katie_candy + (remaining_candy + eaten_candy) - katie_candy = remaining_candy + eaten_candy :=
by
  sorry

end NUMINAMATH_CALUDE_katies_sister_candy_l1742_174203


namespace NUMINAMATH_CALUDE_total_pencils_l1742_174251

def mitchell_pencils : ℕ := 30

def antonio_pencils : ℕ := mitchell_pencils - mitchell_pencils * 20 / 100

theorem total_pencils : mitchell_pencils + antonio_pencils = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1742_174251


namespace NUMINAMATH_CALUDE_inequality_proof_l1742_174207

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_squares : a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5)) + (b / (b + 5)) + (c / (c + 5)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1742_174207


namespace NUMINAMATH_CALUDE_geometric_series_sum_and_comparison_l1742_174204

theorem geometric_series_sum_and_comparison :
  let a : ℝ := 2
  let r : ℝ := 1/4
  let S : ℝ := a / (1 - r)
  S = 8/3 ∧ S ≤ 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_and_comparison_l1742_174204


namespace NUMINAMATH_CALUDE_student_ratio_proof_l1742_174212

/-- Proves that the ratio of elementary school students to other students is 8/9 -/
theorem student_ratio_proof 
  (m n : ℕ) -- number of elementary and other students
  (a b : ℝ) -- average heights of elementary and other students
  (α β : ℝ) -- given constants
  (h1 : a = α * b) -- condition 1
  (h2 : α = 3/4) -- given value of α
  (h3 : a = β * ((a * m + b * n) / (m + n))) -- condition 2
  (h4 : β = 19/20) -- given value of β
  : m / n = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_student_ratio_proof_l1742_174212


namespace NUMINAMATH_CALUDE_circle_tangency_l1742_174263

/-- Circle C with equation x^2 + y^2 - 2x - 4y + m = 0 -/
def circle_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + m = 0}

/-- Circle D with equation (x+2)^2 + (y+2)^2 = 1 -/
def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 + 2)^2 = 1}

/-- The number of common tangents between two circles -/
def common_tangents (C D : Set (ℝ × ℝ)) : ℕ := sorry

theorem circle_tangency (m : ℝ) :
  common_tangents (circle_C m) circle_D = 3 → m = -11 := by sorry

end NUMINAMATH_CALUDE_circle_tangency_l1742_174263


namespace NUMINAMATH_CALUDE_max_value_z_minus_2i_l1742_174275

theorem max_value_z_minus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max : ℝ), max = 3 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w - 2*I) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_z_minus_2i_l1742_174275


namespace NUMINAMATH_CALUDE_fan_sales_analysis_fan_sales_analysis_application_l1742_174235

/-- Represents the sales data for a week -/
structure WeeklySales where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the fan models and their prices -/
structure FanModels where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceA : ℕ
  sellingPriceB : ℕ

/-- Represents the purchase constraints -/
structure PurchaseConstraints where
  totalUnits : ℕ
  maxBudget : ℕ

/-- Main theorem encompassing all parts of the problem -/
theorem fan_sales_analysis 
  (week1 : WeeklySales)
  (week2 : WeeklySales)
  (models : FanModels)
  (constraints : PurchaseConstraints) :
  (models.sellingPriceA = 200 ∧ models.sellingPriceB = 150) ∧
  (∀ m : ℕ, m ≤ 37 → m * models.purchasePriceA + (constraints.totalUnits - m) * models.purchasePriceB ≤ constraints.maxBudget) ∧
  (∃ m : ℕ, m ≤ 37 ∧ m * (models.sellingPriceA - models.purchasePriceA) + 
    (constraints.totalUnits - m) * (models.sellingPriceB - models.purchasePriceB) > 2850) :=
by
  sorry

/-- Given data for the problem -/
def problem_data : WeeklySales × WeeklySales × FanModels × PurchaseConstraints :=
  ({ modelA := 4, modelB := 3, revenue := 1250 },
   { modelA := 5, modelB := 5, revenue := 1750 },
   { purchasePriceA := 140, purchasePriceB := 100, sellingPriceA := 0, sellingPriceB := 0 },
   { totalUnits := 50, maxBudget := 6500 })

/-- Application of the main theorem to the given data -/
theorem fan_sales_analysis_application :
  let (week1, week2, models, constraints) := problem_data
  (models.sellingPriceA = 200 ∧ models.sellingPriceB = 150) ∧
  (∀ m : ℕ, m ≤ 37 → m * models.purchasePriceA + (constraints.totalUnits - m) * models.purchasePriceB ≤ constraints.maxBudget) ∧
  (∃ m : ℕ, m ≤ 37 ∧ m * (models.sellingPriceA - models.purchasePriceA) + 
    (constraints.totalUnits - m) * (models.sellingPriceB - models.purchasePriceB) > 2850) :=
by
  sorry

end NUMINAMATH_CALUDE_fan_sales_analysis_fan_sales_analysis_application_l1742_174235


namespace NUMINAMATH_CALUDE_geometric_squares_existence_and_uniqueness_l1742_174267

theorem geometric_squares_existence_and_uniqueness :
  ∃! k : ℤ,
    (∃ a b c : ℤ,
      (49 + k = a^2) ∧
      (441 + k = b^2) ∧
      (961 + k = c^2) ∧
      (∃ r : ℚ, b = r * a ∧ c = r * b)) ∧
    k = 1152 := by
  sorry

end NUMINAMATH_CALUDE_geometric_squares_existence_and_uniqueness_l1742_174267


namespace NUMINAMATH_CALUDE_vector_equation_l1742_174221

/-- Given non-collinear points A, B, C, and a point O satisfying
    16*OA - 12*OB - 3*OC = 0, prove that OA = 12*AB + 3*AC -/
theorem vector_equation (A B C O : EuclideanSpace ℝ (Fin 3)) 
  (h_not_collinear : ¬Collinear ℝ {A, B, C})
  (h_equation : 16 • (O - A) - 12 • (O - B) - 3 • (O - C) = 0) :
  O - A = 12 • (B - A) + 3 • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l1742_174221


namespace NUMINAMATH_CALUDE_call_center_efficiency_l1742_174227

-- Define the number of agents in each team
variable (A B : ℕ)

-- Define the fraction of calls processed by each team
variable (calls_A calls_B : ℚ)

-- Define the theorem
theorem call_center_efficiency
  (h1 : A = (5 : ℚ) / 8 * B)  -- Team A has 5/8 as many agents as team B
  (h2 : calls_B = 8 / 11)     -- Team B processed 8/11 of the total calls
  (h3 : calls_A + calls_B = 1) -- Total calls processed by both teams is 1
  : (calls_A / A) / (calls_B / B) = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_call_center_efficiency_l1742_174227


namespace NUMINAMATH_CALUDE_first_month_sale_l1742_174292

def last_four_months_sales : List Int := [5660, 6200, 6350, 6500]
def sixth_month_sale : Int := 8270
def average_sale : Int := 6400
def num_months : Int := 6

theorem first_month_sale :
  (num_months * average_sale) - (sixth_month_sale + last_four_months_sales.sum) = 5420 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l1742_174292


namespace NUMINAMATH_CALUDE_circle_c_equation_l1742_174287

/-- A circle C with center in the first quadrant, satisfying specific conditions -/
structure CircleC where
  a : ℝ
  b : ℝ
  r : ℝ
  center_in_first_quadrant : a > 0 ∧ b > 0
  y_axis_chord : 2 * (r^2 - a^2).sqrt = 2
  x_axis_chord : 2 * (r^2 - b^2).sqrt = 4
  arc_length_ratio : (3 : ℝ) / 4 * 2 * Real.pi * r = 3 * ((1 : ℝ) / 4 * 2 * Real.pi * r)

/-- The equation of circle C is (x-√7)² + (y-2)² = 8 -/
theorem circle_c_equation (c : CircleC) : 
  c.a = Real.sqrt 7 ∧ c.b = 2 ∧ c.r = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_c_equation_l1742_174287


namespace NUMINAMATH_CALUDE_earth_sun_distance_calculation_l1742_174260

/-- The speed of light in vacuum (in m/s) -/
def speed_of_light : ℝ := 3 * 10^8

/-- The time taken for sunlight to reach Earth (in s) -/
def time_to_earth : ℝ := 5 * 10^2

/-- The distance between the Earth and the Sun (in m) -/
def earth_sun_distance : ℝ := 1.5 * 10^11

/-- Theorem stating that the distance between the Earth and the Sun
    is equal to the product of the speed of light and the time taken
    for sunlight to reach Earth -/
theorem earth_sun_distance_calculation :
  earth_sun_distance = speed_of_light * time_to_earth := by
  sorry

end NUMINAMATH_CALUDE_earth_sun_distance_calculation_l1742_174260


namespace NUMINAMATH_CALUDE_factors_of_23232_l1742_174256

theorem factors_of_23232 : Nat.card (Nat.divisors 23232) = 42 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_23232_l1742_174256


namespace NUMINAMATH_CALUDE_jeff_truck_count_l1742_174273

/-- The number of trucks Jeff has -/
def num_trucks : ℕ := sorry

/-- The number of cars Jeff has -/
def num_cars : ℕ := sorry

/-- The total number of vehicles Jeff has -/
def total_vehicles : ℕ := 60

theorem jeff_truck_count :
  (num_cars = 2 * num_trucks) ∧
  (num_cars + num_trucks = total_vehicles) →
  num_trucks = 20 := by sorry

end NUMINAMATH_CALUDE_jeff_truck_count_l1742_174273


namespace NUMINAMATH_CALUDE_knockout_tournament_matches_l1742_174298

/-- The number of matches in a knockout tournament -/
def num_matches (n : ℕ) : ℕ := n - 1

/-- A knockout tournament with 64 players -/
def tournament_size : ℕ := 64

theorem knockout_tournament_matches :
  num_matches tournament_size = 63 := by
  sorry

end NUMINAMATH_CALUDE_knockout_tournament_matches_l1742_174298


namespace NUMINAMATH_CALUDE_sherman_driving_time_l1742_174202

/-- Calculates the total driving time for Sherman in a week -/
def shermanWeeklyDrivingTime (dailyCommuteTime weekendDailyDrivingTime : ℕ) : ℕ :=
  let weekdayDrivingTime := 5 * dailyCommuteTime
  let weekendDrivingTime := 2 * weekendDailyDrivingTime
  weekdayDrivingTime + weekendDrivingTime

/-- Theorem: Sherman's weekly driving time is 9 hours -/
theorem sherman_driving_time :
  shermanWeeklyDrivingTime 60 120 = 9 * 60 := by
  sorry

#eval shermanWeeklyDrivingTime 60 120

end NUMINAMATH_CALUDE_sherman_driving_time_l1742_174202


namespace NUMINAMATH_CALUDE_monogram_combinations_l1742_174290

theorem monogram_combinations : ∀ n k : ℕ, 
  n = 14 ∧ k = 2 → (n.choose k) = 91 :=
by
  sorry

#check monogram_combinations

end NUMINAMATH_CALUDE_monogram_combinations_l1742_174290


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1742_174210

/-- The lateral surface area of a cone with base radius 1 and height √3 is 2π. -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let A : ℝ := π * r * l
  A = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1742_174210


namespace NUMINAMATH_CALUDE_elizabeth_climb_time_l1742_174254

-- Define the climbing times
def tom_time : ℕ := 2 * 60  -- Tom's time in minutes
def elizabeth_time : ℕ := tom_time / 4  -- Elizabeth's time in minutes

-- State the theorem
theorem elizabeth_climb_time :
  (tom_time = 4 * elizabeth_time) →  -- Tom takes 4 times as long as Elizabeth
  (tom_time = 2 * 60) →  -- Tom takes 2 hours (120 minutes)
  elizabeth_time = 30 :=  -- Elizabeth takes 30 minutes
by
  sorry

end NUMINAMATH_CALUDE_elizabeth_climb_time_l1742_174254


namespace NUMINAMATH_CALUDE_percent_of_y_l1742_174274

theorem percent_of_y (y : ℝ) (h : y > 0) : ((8 * y) / 20 + (3 * y) / 10) / y = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l1742_174274


namespace NUMINAMATH_CALUDE_enclosed_area_is_four_l1742_174259

-- Define the functions
def f (x : ℝ) : ℝ := 4 * x
def g (x : ℝ) : ℝ := x^3

-- Define the region
def region := {x : ℝ | 0 ≤ x ∧ x ≤ 2 ∧ g x ≤ f x}

-- State the theorem
theorem enclosed_area_is_four : 
  ∫ x in region, (f x - g x) = 4 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_is_four_l1742_174259


namespace NUMINAMATH_CALUDE_max_value_theorem_l1742_174246

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  ∀ x y z, 0 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 2 → 0 ≤ z ∧ z ≤ 2 →
    Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤
    Real.sqrt (x^2 * y^2 * z^2) + Real.sqrt ((2 - x) * (2 - y) * (2 - z)) →
    Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1742_174246


namespace NUMINAMATH_CALUDE_afternoon_sales_problem_l1742_174239

/-- Calculates the number of cookies sold in the afternoon given the initial count,
    morning sales, lunch sales, and remaining cookies. -/
def afternoon_sales (initial : ℕ) (morning_dozens : ℕ) (lunch : ℕ) (remaining : ℕ) : ℕ :=
  initial - (morning_dozens * 12 + lunch) - remaining

theorem afternoon_sales_problem :
  afternoon_sales 120 3 57 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_sales_problem_l1742_174239


namespace NUMINAMATH_CALUDE_power_sum_simplification_l1742_174299

theorem power_sum_simplification :
  (-1)^2006 - (-1)^2007 + 1^2008 + 1^2009 - 1^2010 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_simplification_l1742_174299


namespace NUMINAMATH_CALUDE_f_properties_l1742_174231

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 1/2

-- Theorem statement
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x y, x < y → f x < f y)  -- f is increasing
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l1742_174231


namespace NUMINAMATH_CALUDE_triangle_inradius_l1742_174224

/-- Given a triangle with perimeter 32 cm and area 56 cm², its inradius is 3.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
    (h1 : p = 32) 
    (h2 : A = 56) 
    (h3 : A = r * p / 2) : r = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1742_174224


namespace NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l1742_174229

theorem condition_sufficiency_not_necessity :
  (∀ x : ℝ, x^2 - 4*x < 0 → 0 < x ∧ x < 5) ∧
  (∃ x : ℝ, 0 < x ∧ x < 5 ∧ x^2 - 4*x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l1742_174229


namespace NUMINAMATH_CALUDE_wood_cost_is_1_50_l1742_174205

/-- The cost of producing birdhouses and selling them to Danny -/
structure BirdhouseProduction where
  wood_per_birdhouse : ℕ
  profit_per_birdhouse : ℚ
  price_for_two : ℚ

/-- Calculate the cost of each piece of wood -/
def wood_cost (p : BirdhouseProduction) : ℚ :=
  (p.price_for_two - 2 * p.profit_per_birdhouse) / (2 * p.wood_per_birdhouse)

/-- Theorem: Given the conditions, the cost of each piece of wood is $1.50 -/
theorem wood_cost_is_1_50 (p : BirdhouseProduction) 
  (h1 : p.wood_per_birdhouse = 7)
  (h2 : p.profit_per_birdhouse = 11/2)
  (h3 : p.price_for_two = 32) : 
  wood_cost p = 3/2 := by
  sorry

#eval wood_cost ⟨7, 11/2, 32⟩

end NUMINAMATH_CALUDE_wood_cost_is_1_50_l1742_174205


namespace NUMINAMATH_CALUDE_biology_class_boys_l1742_174209

theorem biology_class_boys (girls_to_boys_ratio : ℚ) (physics_students : ℕ) (biology_to_physics_ratio : ℚ) :
  girls_to_boys_ratio = 3 →
  physics_students = 200 →
  biology_to_physics_ratio = 1/2 →
  (physics_students : ℚ) * biology_to_physics_ratio / (1 + girls_to_boys_ratio) = 25 :=
by sorry

end NUMINAMATH_CALUDE_biology_class_boys_l1742_174209


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1742_174214

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1742_174214


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1742_174272

theorem inequality_solution_set (x : ℝ) : 
  (((1 - 2*x) / ((x - 3) * (2*x + 1))) ≥ 0) ↔ 
  (x ∈ Set.Iio (-1/2) ∪ Set.Icc (1/2) 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1742_174272


namespace NUMINAMATH_CALUDE_exists_permutation_equals_sixteen_l1742_174284

-- Define the set of operations
inductive Operation
  | Div : Operation
  | Add : Operation
  | Mul : Operation

-- Define a function to apply an operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Div => a / b
  | Operation.Add => a + b
  | Operation.Mul => a * b

-- Define a function to evaluate the expression given a permutation of operations
def evaluate (ops : List Operation) : ℚ :=
  match ops with
  | [op1, op2, op3] => applyOp op3 (applyOp op2 (applyOp op1 8 2) 3) 4
  | _ => 0  -- Invalid permutation

-- Theorem statement
theorem exists_permutation_equals_sixteen :
  ∃ (ops : List Operation),
    (ops.length = 3) ∧
    (Operation.Div ∈ ops) ∧
    (Operation.Add ∈ ops) ∧
    (Operation.Mul ∈ ops) ∧
    (evaluate ops = 16) := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_equals_sixteen_l1742_174284


namespace NUMINAMATH_CALUDE_expression_equality_l1742_174270

theorem expression_equality (x y : ℝ) (h : x - 2*y + 2 = 5) : 2*x - 4*y - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1742_174270


namespace NUMINAMATH_CALUDE_yahs_to_bahs_1500_l1742_174250

/-- Conversion rates between bahs, rahs, and yahs -/
structure ConversionRates where
  bah_to_rah : ℚ
  rah_to_yah : ℚ

/-- Given conversion rates, calculate the number of bahs equivalent to a given number of yahs -/
def yahs_to_bahs (rates : ConversionRates) (yahs : ℚ) : ℚ :=
  yahs * rates.rah_to_yah⁻¹ * rates.bah_to_rah⁻¹

/-- Theorem stating the equivalence of 1500 yahs to 562.5 bahs given the specified conversion rates -/
theorem yahs_to_bahs_1500 :
  let rates : ConversionRates := ⟨16/10, 20/12⟩
  yahs_to_bahs rates 1500 = 562.5 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_1500_l1742_174250


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1742_174257

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle

/-- The y-intercept of a line -/
def yIntercept (line : TangentLine) : ℝ := sorry

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (8, 0), radius := 2 }
  let line : TangentLine := { circle1 := c1, circle2 := c2 }
  yIntercept line = 2 * Real.sqrt 82 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1742_174257


namespace NUMINAMATH_CALUDE_exists_closer_vertex_l1742_174216

-- Define a convex polygon
def ConvexPolygon (vertices : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point being inside a polygon
def InsidePolygon (p : ℝ × ℝ) (polygon : Set (ℝ × ℝ)) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem exists_closer_vertex 
  (vertices : Set (ℝ × ℝ)) 
  (P Q : ℝ × ℝ) 
  (h_convex : ConvexPolygon vertices)
  (h_P_inside : InsidePolygon P vertices)
  (h_Q_inside : InsidePolygon Q vertices) :
  ∃ V ∈ vertices, distance V Q < distance V P := by
  sorry

end NUMINAMATH_CALUDE_exists_closer_vertex_l1742_174216


namespace NUMINAMATH_CALUDE_range_of_a_l1742_174297

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 3 ≤ 0) ↔ -3 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1742_174297


namespace NUMINAMATH_CALUDE_max_distance_to_line_family_l1742_174282

/-- The maximum distance from a point to a family of lines --/
theorem max_distance_to_line_family (a : ℝ) : 
  let P : ℝ × ℝ := (1, -1)
  let line := {(x, y) : ℝ × ℝ | a * x + 3 * y + 2 * a - 6 = 0}
  ∃ (Q : ℝ × ℝ), Q ∈ line ∧ 
    ∀ (R : ℝ × ℝ), R ∈ line → 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  ∧ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_max_distance_to_line_family_l1742_174282


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1742_174293

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_mean : (a + b) / 2 = 1 / 2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + y) / 2 = 1 / 2 → 1 / x + 1 / y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1742_174293


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1742_174213

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, ∃ y : ℝ, 4 * x - 7 + c = d * x + 2 * y + 4) ↔ d ≠ 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1742_174213


namespace NUMINAMATH_CALUDE_particular_number_exists_l1742_174245

theorem particular_number_exists : ∃! x : ℝ, 2 * ((x / 23) - 67) = 102 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_exists_l1742_174245


namespace NUMINAMATH_CALUDE_last_four_digits_of_3_24000_l1742_174233

theorem last_four_digits_of_3_24000 (h : 3^800 ≡ 1 [ZMOD 2000]) :
  3^24000 ≡ 1 [ZMOD 2000] := by sorry

end NUMINAMATH_CALUDE_last_four_digits_of_3_24000_l1742_174233


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l1742_174232

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- A geometric sequence -/
structure GeometricSequence where
  b : ℕ → ℝ
  q : ℝ
  h_geometric : ∀ n, b (n + 1) = b n * q

/-- The main theorem -/
theorem arithmetic_geometric_sequence_relation
  (a : ArithmeticSequence)
  (b : GeometricSequence)
  (h_consecutive : b.b 1 = a.a 5 ∧ b.b 2 = a.a 8 ∧ b.b 3 = a.a 13)
  (h_b2 : b.b 2 = 5) :
  ∀ n, b.b n = 5 * (5/3)^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l1742_174232


namespace NUMINAMATH_CALUDE_coefficient_x4_in_polynomial_product_l1742_174252

theorem coefficient_x4_in_polynomial_product : 
  let p1 : Polynomial ℤ := X^5 - 4*X^4 + 3*X^3 - 2*X^2 + X - 1
  let p2 : Polynomial ℤ := 3*X^2 - X + 5
  (p1 * p2).coeff 4 = -13 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_polynomial_product_l1742_174252


namespace NUMINAMATH_CALUDE_operator_value_l1742_174238

/-- The operator definition -/
def operator (a : ℝ) (x : ℝ) : ℝ := x * (a - x)

/-- Theorem stating the value of 'a' in the operator definition -/
theorem operator_value :
  ∃ a : ℝ, (∀ p : ℝ, p = 1 → p + 1 = operator a (p + 1)) → a = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_operator_value_l1742_174238


namespace NUMINAMATH_CALUDE_square_ratio_sum_l1742_174288

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l1742_174288


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1742_174234

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| < 2
def q (x : ℝ) : Prop := x^2 < 2 - x

-- Define the relationship between ¬p and ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∃ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  (∃ x : ℝ, ¬(q x) ∧ p x) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1742_174234


namespace NUMINAMATH_CALUDE_power_calculation_l1742_174208

theorem power_calculation : (-2 : ℝ)^2023 * (1/2 : ℝ)^2022 = -2 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l1742_174208


namespace NUMINAMATH_CALUDE_tom_run_distance_l1742_174225

theorem tom_run_distance (total_distance : ℝ) (walk_speed : ℝ) (run_speed : ℝ) 
  (friend_time : ℝ) (max_total_time : ℝ) :
  total_distance = 2800 →
  walk_speed = 75 →
  run_speed = 225 →
  friend_time = 5 →
  max_total_time = 30 →
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_distance ∧
    (x / walk_speed + (total_distance - x) / run_speed + friend_time ≤ max_total_time) ∧
    (total_distance - x ≤ 1387.5) :=
by sorry

end NUMINAMATH_CALUDE_tom_run_distance_l1742_174225


namespace NUMINAMATH_CALUDE_inequality_proof_l1742_174285

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1742_174285


namespace NUMINAMATH_CALUDE_girls_attending_event_l1742_174261

theorem girls_attending_event (total_students : ℕ) (total_attending : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1800 →
  total_attending = 1110 →
  girls + boys = total_students →
  (3 * girls) / 4 + (2 * boys) / 3 = total_attending →
  (3 * girls) / 4 = 690 :=
by sorry

end NUMINAMATH_CALUDE_girls_attending_event_l1742_174261


namespace NUMINAMATH_CALUDE_xy_range_l1742_174226

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 3*y + 2/x + 4/y = 10) : 
  1 ≤ x*y ∧ x*y ≤ 8/3 := by
sorry

end NUMINAMATH_CALUDE_xy_range_l1742_174226


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1742_174278

/-- A quadratic function y = x^2 + 2x + c with two distinct real roots -/
structure QuadraticFunction where
  c : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h₁ : x₁ < x₂
  h₂ : x₁^2 + 2*x₁ + c = 0
  h₃ : x₂^2 + 2*x₂ + c = 0

/-- A point on the graph of a quadratic function -/
structure PointOnGraph (f : QuadraticFunction) where
  m : ℝ
  n : ℝ
  h : n = m^2 + 2*m + f.c

theorem quadratic_function_property (f : QuadraticFunction) (P : PointOnGraph f) :
  P.n < 0 → f.x₁ < P.m ∧ P.m < f.x₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1742_174278


namespace NUMINAMATH_CALUDE_total_count_formula_specific_case_l1742_174228

/-- Represents the structure of a plant with branches and small branches -/
structure Plant where
  branches : ℕ
  smallBranches : ℕ

/-- Calculates the total number of stems, branches, and small branches in a plant -/
def totalCount (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranches

/-- Theorem stating that the total count equals x^2 + x + 1 -/
theorem total_count_formula (x : ℕ) :
  totalCount { branches := x, smallBranches := x } = x^2 + x + 1 := by
  sorry

/-- The specific case where the total count is 73 -/
theorem specific_case : ∃ x : ℕ, totalCount { branches := x, smallBranches := x } = 73 := by
  sorry

end NUMINAMATH_CALUDE_total_count_formula_specific_case_l1742_174228


namespace NUMINAMATH_CALUDE_cost_reduction_percentage_l1742_174215

/-- Proves the percentage reduction in cost price given specific conditions --/
theorem cost_reduction_percentage
  (original_cost : ℝ)
  (original_profit_rate : ℝ)
  (price_reduction : ℝ)
  (new_profit_rate : ℝ)
  (h1 : original_cost = 40)
  (h2 : original_profit_rate = 0.25)
  (h3 : price_reduction = 8.40)
  (h4 : new_profit_rate = 0.30)
  : ∃ (reduction_rate : ℝ),
    reduction_rate = 0.20 ∧
    (1 + new_profit_rate) * (original_cost * (1 - reduction_rate)) =
    (1 + original_profit_rate) * original_cost - price_reduction :=
by sorry

end NUMINAMATH_CALUDE_cost_reduction_percentage_l1742_174215


namespace NUMINAMATH_CALUDE_root_transformation_l1742_174277

theorem root_transformation (a b c d : ℂ) : 
  (a^4 - 2*a - 6 = 0) ∧ 
  (b^4 - 2*b - 6 = 0) ∧ 
  (c^4 - 2*c - 6 = 0) ∧ 
  (d^4 - 2*d - 6 = 0) →
  ∃ (y₁ y₂ y₃ y₄ : ℂ), 
    y₁ = 2*(a + b + c)/d^3 ∧
    y₂ = 2*(a + b + d)/c^3 ∧
    y₃ = 2*(a + c + d)/b^3 ∧
    y₄ = 2*(b + c + d)/a^3 ∧
    (2*y₁^4 - 2*y₁ + 48 = 0) ∧
    (2*y₂^4 - 2*y₂ + 48 = 0) ∧
    (2*y₃^4 - 2*y₃ + 48 = 0) ∧
    (2*y₄^4 - 2*y₄ + 48 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l1742_174277


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1742_174258

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (2*x - y)^2 - 4*(x - y)*(x + 2*y) = -8*x*y + 9*y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b c : ℝ) :
  (a - 2*b - 3*c)*(a - 2*b + 3*c) = a^2 + 4*b^2 - 4*a*b - 9*c^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1742_174258


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l1742_174294

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial : trailingZeroes 500 = 124 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l1742_174294


namespace NUMINAMATH_CALUDE_not_divisible_by_59_l1742_174281

theorem not_divisible_by_59 (x y : ℕ) 
  (h1 : ¬ 59 ∣ x) 
  (h2 : ¬ 59 ∣ y) 
  (h3 : 59 ∣ (3 * x + 28 * y)) : 
  ¬ 59 ∣ (5 * x + 16 * y) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_59_l1742_174281


namespace NUMINAMATH_CALUDE_line_through_points_l1742_174237

/-- Given a line passing through points (-1, -4) and (x, k), where the slope
    of the line is equal to k and k = 1, prove that x = 4. -/
theorem line_through_points (x : ℝ) :
  let k : ℝ := 1
  let slope : ℝ := (k - (-4)) / (x - (-1))
  slope = k → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1742_174237


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1742_174230

/-- A random variable following a binomial distribution B(n,p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_unique_parameters
  (X : BinomialRV)
  (h_expectation : expectation X = 1.6)
  (h_variance : variance X = 1.28) :
  X.n = 8 ∧ X.p = 0.2 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1742_174230


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1742_174211

theorem consecutive_even_integers_sum (n : ℤ) : 
  (n % 2 = 0) ∧ (n * (n + 2) * (n + 4) = 480) → n + (n + 2) + (n + 4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1742_174211


namespace NUMINAMATH_CALUDE_expression_evaluation_l1742_174269

theorem expression_evaluation : 200 * (200 - 2^3) - (200^2 - 2^4) = -1584 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1742_174269


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l1742_174244

theorem inscribed_circle_theorem (r : ℝ) (a b c : ℝ) :
  r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 →
  r = 4 →
  a + b = 14 →
  (∃ (s : ℝ), s = (a + b + c) / 2 ∧ s * r = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (c = 13 ∧ b = 15) ∨ (c = 15 ∧ b = 13) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l1742_174244


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1742_174291

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 3 * x / (x - 2) + (3 * x^2 - 36) / x
  ∃ (y : ℝ), y = (2 - Real.sqrt 58) / 3 ∧ 
    f y = 13 ∧ 
    ∀ (z : ℝ), f z = 13 → y ≤ z := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1742_174291


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_to_plane_implies_parallel_l1742_174265

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: If m ⟂ α and m ⟂ β, then α ∥ β
theorem perpendicular_implies_parallel 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular m β) : 
  parallel α β :=
sorry

-- Theorem 2: If m ⟂ β and n ⟂ β, then m ∥ n
theorem perpendicular_to_plane_implies_parallel 
  (m n : Line) (β : Plane) 
  (h1 : perpendicular m β) (h2 : perpendicular n β) : 
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_to_plane_implies_parallel_l1742_174265


namespace NUMINAMATH_CALUDE_percentage_of_450_is_172_8_l1742_174268

theorem percentage_of_450_is_172_8 : 
  ∃ p : ℝ, (p / 100) * 450 = 172.8 ∧ p = 38.4 := by sorry

end NUMINAMATH_CALUDE_percentage_of_450_is_172_8_l1742_174268


namespace NUMINAMATH_CALUDE_allan_balloons_l1742_174289

/-- The number of balloons Allan initially brought to the park -/
def initial_balloons : ℕ := 5

/-- The number of balloons Allan bought at the park -/
def bought_balloons : ℕ := 3

/-- The total number of balloons Allan brought to the park -/
def total_balloons : ℕ := initial_balloons + bought_balloons

theorem allan_balloons : total_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_allan_balloons_l1742_174289
