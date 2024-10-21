import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_l414_41485

/-- A's work efficiency -/
def A : ℝ := sorry

/-- B's work efficiency -/
def B : ℝ := sorry

/-- A is half as good a workman as B -/
axiom A_half_B : A = (1 / 2) * B

/-- Together, A and B finish a job in 13 days -/
axiom job_together : A + B = 1 / 13

/-- B can finish the job alone in 19.5 days -/
axiom B_alone : B = 1 / 19.5

/-- The ratio of A's work efficiency to B's work efficiency is 1:2 -/
theorem efficiency_ratio : A / B = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_l414_41485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l414_41493

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin ((2/3) * x + (3 * Real.pi) / 2)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + 3 * Real.pi) = f x) :=
by
  constructor
  · intro x
    -- Proof for evenness
    sorry
  · intro x
    -- Proof for periodicity
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l414_41493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l414_41417

theorem cubic_equation_solutions : 
  ∀ x : ℝ, ((18 * x - 2) ^ (1/3 : ℝ) + (16 * x + 2) ^ (1/3 : ℝ) = 4 * (2 * x) ^ (1/3 : ℝ)) ↔ 
  (x = 0 ∨ x = 15 / 261 ∨ x = -19 / 261) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l414_41417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_smallest_increase_l414_41429

def game_show_values : List ℕ := [100, 300, 600, 800, 1500, 3000, 4500, 7000, 10000, 15000, 30000, 45000, 75000, 150000, 300000]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def smallest_increase (values : List ℕ) : Prop :=
  let increase_2_3 := percent_increase (values.get! 1) (values.get! 2)
  let increase_4_5 := percent_increase (values.get! 3) (values.get! 4)
  let increase_7_8 := percent_increase (values.get! 6) (values.get! 7)
  let increase_12_13 := percent_increase (values.get! 11) (values.get! 12)
  let increase_14_15 := percent_increase (values.get! 13) (values.get! 14)
  increase_7_8 < increase_2_3 ∧
  increase_7_8 < increase_4_5 ∧
  increase_7_8 < increase_12_13 ∧
  increase_7_8 < increase_14_15

theorem game_show_smallest_increase :
  smallest_increase game_show_values :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_smallest_increase_l414_41429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l414_41424

theorem simplify_and_rationalize :
  1 / (2 + 1 / (Real.sqrt 5 - 2)) = (4 - Real.sqrt 5) / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l414_41424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l414_41459

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h_geometric : (seq.a 2) * (seq.a 6) = (seq.a 3)^2) :
  seq.a 1 * seq.d < 0 ∧ seq.d * S seq 3 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l414_41459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_quantity_minimizes_cost_l414_41416

/-- The optimal purchase quantity that minimizes total cost -/
noncomputable def optimal_quantity : ℝ := 20

/-- The total annual purchase quantity in tons -/
def total_annual_purchase : ℝ := 400

/-- The freight cost per purchase in yuan -/
def freight_cost_per_purchase : ℝ := 40000

/-- The annual storage cost per ton in yuan -/
def storage_cost_per_ton : ℝ := 40000

/-- The total cost function -/
noncomputable def total_cost (x : ℝ) : ℝ :=
  (total_annual_purchase / x) * freight_cost_per_purchase + storage_cost_per_ton * x

/-- Theorem stating that the optimal quantity minimizes the total cost -/
theorem optimal_quantity_minimizes_cost :
  ∀ x > 0, total_cost optimal_quantity ≤ total_cost x := by
  sorry

#check optimal_quantity_minimizes_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_quantity_minimizes_cost_l414_41416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_pure_imaginary_ratio_l414_41494

theorem complex_pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : Complex.I * Complex.im ((3 - 4 * Complex.I) * (p + q * Complex.I)) = (3 - 4 * Complex.I) * (p + q * Complex.I)) :
  p / q = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_pure_imaginary_ratio_l414_41494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apron_cost_l414_41487

/-- The cost of cooking gear for children --/
structure CookingGearCost where
  hand_mitts : ℝ
  utensils : ℝ
  knife : ℝ
  apron : ℝ

/-- The properties of the cooking gear costs --/
def CookingGearProperties (c : CookingGearCost) : Prop :=
  c.hand_mitts = 14 ∧
  c.utensils = 10 ∧
  c.knife = 2 * c.utensils ∧
  (3 * (c.hand_mitts + c.utensils + c.knife + c.apron)) * 0.75 = 135

/-- Theorem: The apron costs $16.00 --/
theorem apron_cost (c : CookingGearCost) 
  (h : CookingGearProperties c) : c.apron = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apron_cost_l414_41487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l414_41481

theorem vector_equation_solution :
  ∃ (a b : ℚ),
    a = 9 / 23 ∧
    b = 2 / 69 ∧
    a • (![3, 4] : Fin 2 → ℚ) + b • (![(-6), 15] : Fin 2 → ℚ) = ![1, 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l414_41481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_value_l414_41457

-- Define the sets T and S
def T (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 7)^2 ≤ r^2}

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀ θ : ℝ, Real.cos (2 * θ) + p.1 * Real.cos θ + p.2 ≥ 0}

-- State the theorem
theorem max_r_value :
  ∃ r_max : ℝ, r_max > 0 ∧
  (∀ r : ℝ, r > 0 → T r ⊆ S → r ≤ r_max) ∧
  (T r_max ⊆ S) ∧
  r_max = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_value_l414_41457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l414_41482

/-- Represents a car with given specifications -/
structure Car where
  speed : ℚ  -- Speed in miles per hour
  fuel_efficiency : ℚ  -- Miles per gallon
  tank_capacity : ℚ  -- Gallons
  fuel_used_fraction : ℚ  -- Fraction of full tank used

/-- Calculates the travel time for a given car -/
def travel_time (c : Car) : ℚ :=
  (c.fuel_used_fraction * c.tank_capacity * c.fuel_efficiency) / c.speed

/-- Theorem stating that the travel time for the given car specifications is 5 hours -/
theorem car_travel_time :
  let c : Car := {
    speed := 50,
    fuel_efficiency := 30,
    tank_capacity := 15,
    fuel_used_fraction := 5555555555555556 / 10000000000000000
  }
  travel_time c = 5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l414_41482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l414_41465

noncomputable def f (x : ℝ) : ℝ := 1/x + Real.log (1 - 2*x)

def IsValidArg (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y, f x = y

theorem domain_of_f :
  {x : ℝ | IsValidArg f x} = {x : ℝ | x < (1/2) ∧ x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l414_41465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_average_l414_41451

/-- The average water consumption over three days given specific consumption patterns -/
theorem water_consumption_average (first_day : ℝ) : 
  first_day = 215 →
  let second_day := first_day + 76
  let last_day := second_day - 53
  let total := first_day + second_day + last_day
  let average := total / 3
  average = 248 := by
  intro h
  -- Proof steps would go here
  sorry

#check water_consumption_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_average_l414_41451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_selection_count_l414_41414

-- Define the number of male and female athletes
def num_male : ℕ := 5
def num_female : ℕ := 4

-- Define the total number of athletes
def total_athletes : ℕ := num_male + num_female

-- Define the number of athletes to be selected
def select_count : ℕ := 4

-- Theorem statement
theorem athlete_selection_count :
  (Nat.choose total_athletes select_count - Nat.choose num_male select_count - Nat.choose num_female select_count) -
  (Nat.choose (total_athletes - 2) select_count - Nat.choose (num_male - 1) select_count) = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_selection_count_l414_41414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l414_41423

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (4 - a) * x else a^x

theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  2 ≤ a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l414_41423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_sum_l414_41480

/-- A function f with parameters a and b -/
noncomputable def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1/3

/-- Theorem stating that if f has an extreme value of 0 at x=1, then a + b = -7/9 -/
theorem extreme_value_implies_sum (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b 1 ≤ f a b x) ∧ 
  (f a b 1 = 0) →
  a + b = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_sum_l414_41480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_for_increasing_f_l414_41408

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the theorem
theorem range_of_x_for_increasing_f :
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → f x < f y) →  -- f is increasing on [-1, 1]
  (∀ x, x ∈ Set.Icc (-1) 1 → f (x - 2) < f (1 - x)) →  -- given condition
  {x : ℝ | x ∈ Set.Icc (-1) 1 ∧ f (x - 2) < f (1 - x)} = {x : ℝ | 1 ≤ x ∧ x < 3/2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_for_increasing_f_l414_41408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_eq_point_zero_three_l414_41468

/-- Regression equation: ŷ = 2x + 1 -/
def regression_equation (x : ℝ) : ℝ := 2 * x + 1

/-- Data points -/
def data_points : List (ℝ × ℝ) := [(2, 4.9), (3, 7.1), (4, 9.1)]

/-- Calculate residual for a single point -/
def calc_residual (point : ℝ × ℝ) : ℝ :=
  point.2 - regression_equation point.1

/-- Calculate squared residual for a single point -/
def squared_residual (point : ℝ × ℝ) : ℝ :=
  (calc_residual point) ^ 2

/-- Sum of squared residuals -/
def sum_squared_residuals : ℝ :=
  List.sum (List.map squared_residual data_points)

/-- Theorem: The sum of squared residuals is equal to 0.03 -/
theorem sum_squared_residuals_eq_point_zero_three :
  sum_squared_residuals = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_eq_point_zero_three_l414_41468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_boy_saturday_girl_sunday_l414_41427

def num_boys : ℕ := 2
def num_girls : ℕ := 2
def total_people : ℕ := num_boys + num_girls
def days : ℕ := 2

def total_arrangements : ℕ := (total_people.choose 2) * days.factorial

def favorable_arrangements : ℕ := num_boys * num_girls

theorem probability_boy_saturday_girl_sunday :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_boy_saturday_girl_sunday_l414_41427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l414_41452

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = -f a b (-x)) →
  f a b (1/2) = 2/5 →
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → g x = x / (x^2 + 1)) ∧
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧
    (Set.Ioo 0 (1/3) = {x | g (2*x - 1) + g x < 0})) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l414_41452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_theorem_l414_41495

/-- The curve equation -/
noncomputable def f (x : ℝ) : ℝ := x^2 / 4 - Real.log x

/-- The derivative of the curve equation -/
noncomputable def f' (x : ℝ) : ℝ := x / 2 - 1 / x

theorem tangent_point_theorem :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 1 / 4
  x₀ > 0 ∧ f x₀ = y₀ ∧ f' x₀ = -1/2 := by
  sorry

#check tangent_point_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_theorem_l414_41495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_four_factors_l414_41420

theorem smallest_sum_of_four_factors (p q r s : ℕ+) : 
  p * q * r * s = Nat.factorial 12 → 
  (∀ a b c d : ℕ+, a * b * c * d = Nat.factorial 12 → p + q + r + s ≤ a + b + c + d) →
  p + q + r + s = 777 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_four_factors_l414_41420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l414_41422

/-- A pentagon in a 2D plane --/
structure Pentagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ

/-- Calculate the area of a trapezoid given its bases and height --/
noncomputable def trapezoidArea (base1 base2 height : ℝ) : ℝ :=
  (base1 + base2) * height / 2

/-- Calculate the area of a triangle given its base and height --/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  base * height / 2

/-- The specific pentagon from the problem --/
def specificPentagon : Pentagon :=
  { v1 := (0, 0)
    v2 := (20, 0)
    v3 := (50, 30)
    v4 := (20, 50)
    v5 := (0, 40) }

/-- Theorem stating that the area of the specific pentagon is 2150 square units --/
theorem specific_pentagon_area :
  trapezoidArea 20 50 40 + triangleArea 30 50 = 2150 := by
  sorry

#check specific_pentagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l414_41422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l414_41449

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, -2]

def point_A : Fin 2 → ℝ := ![1, 2]
def point_B : Fin 2 → ℝ := ![3, 3]
def point_C : Fin 2 → ℝ := ![2, 1]

def transform_point (p : Fin 2 → ℝ) : Fin 2 → ℝ :=
  transformation_matrix.mulVec p

def point_A' : Fin 2 → ℝ := transform_point point_A
def point_B' : Fin 2 → ℝ := transform_point point_B
def point_C' : Fin 2 → ℝ := transform_point point_C

noncomputable def area_of_triangle (a b c : Fin 2 → ℝ) : ℝ :=
  let v1 := b - a
  let v2 := c - a
  (1/2) * abs (v1 0 * v2 1 - v1 1 * v2 0)

theorem transformed_triangle_area :
  area_of_triangle point_A' point_B' point_C' = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l414_41449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l414_41412

noncomputable def f (x : ℝ) := (1 - (2:ℝ)^x) / ((2:ℝ)^(x+1) + 2)

theorem odd_function_inequality (k : ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) 3, f (k*x^2) + f (2*x - 1) > 0) ↔
  k ∈ Set.Ioi (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l414_41412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_perimeter_l414_41454

/-- A regular pentagon with side length 5 meters has a perimeter of 25 meters. -/
theorem regular_pentagon_perimeter : 
  ∀ (p : Real), p = 5 → 5 * p = 25 :=
by
  intro p hp
  rw [hp]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_perimeter_l414_41454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_max_value_l414_41453

/-- The maximum value of a cubic polynomial expression --/
theorem cubic_polynomial_max_value (a b c : ℝ) (lambda : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h_lambda_pos : lambda > 0)
  (h_roots : x₁ < x₂ ∧ x₂ < x₃)
  (h_f_roots : ∀ x, x^3 + a*x^2 + b*x + c = (x - x₁)*(x - x₂)*(x - x₃))
  (h_lambda_diff : x₂ - x₁ = lambda)
  (h_x₃_gt_mid : x₃ > (x₁ + x₂)/2) :
  ∃ M : ℝ, M = (3 * Real.sqrt 3) / 2 ∧ 
    ∀ a b c lambda x₁ x₂ x₃, 
      lambda > 0 → 
      x₁ < x₂ ∧ x₂ < x₃ → 
      (∀ x, x^3 + a*x^2 + b*x + c = (x - x₁)*(x - x₂)*(x - x₃)) → 
      x₂ - x₁ = lambda → 
      x₃ > (x₁ + x₂)/2 → 
      (2*a^3 + 27*c + 9*a*b) / lambda^3 ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_max_value_l414_41453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_range_l414_41439

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2

theorem f_upper_bound_implies_a_range (a : ℝ) :
  (∀ x > 0, f a x ≤ a - 1) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_range_l414_41439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l414_41403

/-- The current age of the man -/
def M : ℕ := sorry

/-- The current age of the father -/
def F : ℕ := sorry

/-- The man's current age is (2/5) of his father's age -/
axiom current_age_ratio : M = (2 * F) / 5

/-- After 8 years, the man will be (1/2) of his father's age -/
axiom future_age_ratio : M + 8 = (F + 8) / 2

/-- The father's current age is 40 years -/
theorem fathers_age : F = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l414_41403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_third_quadrant_f_specific_angle_l414_41460

noncomputable section

open Real

-- Define the function f
def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * tan (-α + 3 * π / 2) * tan (-α - π)) / sin (-π - α)

-- Theorem 1: Simplification of f(α)
theorem f_simplification (α : ℝ) : f α = sin α * tan α := by sorry

-- Theorem 2: Value of f(α) in the third quadrant
theorem f_third_quadrant (α : ℝ) 
  (h1 : π < α ∧ α < 3 * π / 2) -- α is in the third quadrant
  (h2 : cos (α - 3 * π / 2) = 1 / 5) : 
  f α = -sqrt 6 / 60 := by sorry

-- Theorem 3: Value of f(α) when α = -1860°
theorem f_specific_angle : f (-1860 * π / 180) = 3 / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_third_quadrant_f_specific_angle_l414_41460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_count_l414_41431

theorem exam_pass_count (total_boys : ℕ) (avg_all : ℚ) (avg_pass : ℚ) (avg_fail : ℚ) 
  (h1 : total_boys = 120)
  (h2 : avg_all = 35)
  (h3 : avg_pass = 39)
  (h4 : avg_fail = 15) :
  ∃ (passed_boys : ℕ),
    passed_boys = 100 ∧
    passed_boys ≤ total_boys ∧
    (passed_boys : ℚ) * avg_pass + (total_boys - passed_boys : ℚ) * avg_fail = (total_boys : ℚ) * avg_all := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_count_l414_41431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_in_special_set_l414_41404

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- A set of 10 different natural numbers with specific properties -/
def SpecialSet : Set ℕ :=
  {n : ℕ | ∃ (s : Finset ℕ), s.card = 10 ∧ n ∈ s ∧
    (∀ m ∈ s, m ≠ n → digitSum m = digitSum n) ∧
    (s.sum id) = 604}

/-- The theorem stating the largest number in the special set is 109 -/
theorem largest_in_special_set :
  ∀ n ∈ SpecialSet, n ≤ 109 := by
  sorry

#check largest_in_special_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_in_special_set_l414_41404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l414_41428

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| + 3 * x = 12 :=
by
  -- The unique solution is x = 4
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    simp [abs_of_nonneg (by norm_num : (4 : ℝ) - 4 ≥ 0)]
    norm_num
  · -- Prove uniqueness
    intro y hy
    -- Case analysis on |y - 4|
    cases le_or_gt y 4 with
    | inl h_le =>
      -- Case: y ≤ 4
      have h_abs : |y - 4| = -(y - 4) := abs_of_nonpos (sub_nonpos_of_le h_le)
      rw [h_abs] at hy
      linarith
    | inr h_gt =>
      -- Case: y > 4
      have h_abs : |y - 4| = y - 4 := abs_of_pos (sub_pos.2 h_gt)
      rw [h_abs] at hy
      linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l414_41428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_knight_l414_41444

-- Define the children
inductive Child : Type
| Anu : Child
| Banu : Child
| Vanu : Child
| Danu : Child

-- Define the homework status
def did_homework : Child → Prop := sorry

-- Define whether a child is a knight or liar
def is_knight : Child → Prop := sorry

-- Anu's statement
axiom Anu_statement : 
  is_knight Child.Anu → 
  (did_homework Child.Banu ∧ did_homework Child.Vanu ∧ did_homework Child.Danu)

-- Banu's statement
axiom Banu_statement : 
  is_knight Child.Banu → 
  (¬did_homework Child.Anu ∧ ¬did_homework Child.Vanu ∧ ¬did_homework Child.Danu)

-- Vanu's statement
axiom Vanu_statement : 
  is_knight Child.Vanu → 
  (¬is_knight Child.Anu ∧ ¬is_knight Child.Banu)

-- Danu's statement
axiom Danu_statement : 
  is_knight Child.Danu → 
  (is_knight Child.Anu ∧ is_knight Child.Banu ∧ is_knight Child.Vanu)

-- A child is either a knight or a liar
axiom knight_or_liar (c : Child) : 
  is_knight c ∨ ¬is_knight c

-- The main theorem
theorem one_knight : 
  ∃! (c : Child), is_knight c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_knight_l414_41444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruentSideLengthForSpecificTriangle_l414_41425

/-- An isosceles triangle with given base and area -/
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ
  isPositive : base > 0 ∧ area > 0

/-- Calculate the length of the congruent sides of the isosceles triangle -/
noncomputable def congruentSideLength (triangle : IsoscelesTriangle) : ℝ :=
  let height := 2 * triangle.area / triangle.base
  Real.sqrt ((triangle.base / 2) ^ 2 + height ^ 2)

/-- Theorem stating the length of congruent sides for the specific triangle -/
theorem congruentSideLengthForSpecificTriangle :
  let triangle : IsoscelesTriangle := ⟨30, 72, by norm_num⟩
  congruentSideLength triangle = 15.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruentSideLengthForSpecificTriangle_l414_41425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_bounds_l414_41483

-- Define the parameters
variable (a b c : ℝ)

-- Define the conditions
def are_roots : Prop := a^2 + a + c = 0 ∧ b^2 + b + c = 0
def c_in_range : Prop := 0 ≤ c ∧ c ≤ 1

-- Define the distance between the lines
noncomputable def distance : ℝ := |a - b| / Real.sqrt 2

-- State the theorem
theorem line_distance_bounds (h1 : are_roots a b c) (h2 : c_in_range c) :
  ∃ (d_max d_min : ℝ), d_max = 1 / Real.sqrt 2 ∧ 
  (∀ ε > 0, ∃ d : ℝ, d < ε ∧ distance a b = d) ∧
  (∀ d : ℝ, 0 ≤ d ∧ d ≤ distance a b → d ≤ d_max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_bounds_l414_41483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l414_41464

open Real

/-- The function f(x) defined as sin(x + π/5) + √3 cos(x + 8π/15) -/
noncomputable def f (x : ℝ) : ℝ := sin (x + π/5) + sqrt 3 * cos (x + 8*π/15)

/-- The maximum value of f(x) is 1 -/
theorem f_max_value : ∃ (M : ℝ), M = 1 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l414_41464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l414_41441

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 6 * x^2 - 5 * x + 23 = 0 ↔ x = Complex.ofReal a + Complex.I * b ∨ x = Complex.ofReal a - Complex.I * b) →
  a + b^2 = 587/144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l414_41441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l414_41446

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

theorem min_positive_period_of_f : ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (S : ℝ), S > 0 → (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧ T = Real.pi := by
  sorry

#check min_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l414_41446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_hash_fraction_l414_41442

-- Define the @ operation
def atOp (a b : ℚ) : ℚ := a * b - b^3

-- Define the # operation
def hashOp (a b : ℚ) : ℚ := a + 2*b - a*b^3

-- Theorem statement
theorem at_hash_fraction :
  (atOp 7 3) / (hashOp 7 3) = 3 / 88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_hash_fraction_l414_41442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_no_min_omega_range_l414_41472

open Real Set

theorem sin_max_no_min_omega_range 
  (f : ℝ → ℝ) 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_f_def : ∀ x, f x = sin (ω * x + π / 4)) 
  (h_domain : Set ℝ := Ioo (π / 12) (π / 3)) :
  (∃ (M : ℝ), ∀ x ∈ h_domain, f x ≤ M) ∧ 
  (∀ (m : ℝ), ∃ x ∈ h_domain, f x < m) ↔ 
  ω > 3/4 ∧ ω < 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_no_min_omega_range_l414_41472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l414_41461

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 7).sum (fun k ↦ (Nat.choose 6 k : ℝ) * x^(6 - k) * 2^k) = 
  240 * x^2 + (Finset.range 7).sum (fun k ↦ if k ≠ 4 then (Nat.choose 6 k : ℝ) * x^(6 - k) * 2^k else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l414_41461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l414_41418

theorem trig_identity (α : Real) 
  (h1 : Real.sin α = 1/2 + Real.cos α) 
  (h2 : α ∈ Set.Ioo 0 (Real.pi/2)) : 
  Real.cos (2*α) / Real.sin (α - Real.pi/4) = -Real.sqrt 14/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l414_41418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_comparison_l414_41447

/-- The initial loan amount in coins -/
def initial_loan : ℝ := 100

/-- The loan term in days -/
def loan_term : ℝ := 128

/-- The daily interest rate for Option 1 (compound interest) -/
def compound_rate : ℝ := 0.01

/-- The daily interest rate for Option 2 (simple interest) -/
def simple_rate : ℝ := 0.02

/-- Calculate the total amount to be repaid under Option 1 (compound interest) -/
noncomputable def compound_total : ℝ := initial_loan * (1 + compound_rate) ^ loan_term

/-- Calculate the total amount to be repaid under Option 2 (simple interest) -/
def simple_total : ℝ := initial_loan * (1 + simple_rate * loan_term)

/-- The difference between compound and simple interest totals, rounded to the nearest integer -/
noncomputable def interest_difference : ℤ := Int.floor (compound_total - simple_total + 0.5)

theorem loan_comparison :
  interest_difference = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_comparison_l414_41447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vending_machine_coin_equivalence_l414_41402

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Theorem stating that if x quarters, 15 dimes, and 30 nickels have the same total value
    as 25 quarters, 5 dimes, and n nickels, then n = 50 -/
theorem vending_machine_coin_equivalence (x n : ℕ) :
  quarter_value * x + dime_value * 15 + nickel_value * 30 =
  quarter_value * 25 + dime_value * 5 + nickel_value * n →
  n = 50 := by
  sorry

#check vending_machine_coin_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vending_machine_coin_equivalence_l414_41402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_polynomials_in_G_eq_528_count_polynomials_in_G_l414_41407

/-- A polynomial in the set G -/
structure PolynomialG where
  coeffs : List ℤ
  constant_term : ℤ
  constant_term_eq_50 : constant_term = 50
  roots : List (ℤ × ℤ)
  roots_distinct : roots.Nodup
  degree_eq_roots : coeffs.length = roots.length

/-- The set G of polynomials -/
def G : Set PolynomialG := {p | True}

/-- The number of polynomials in G -/
noncomputable def num_polynomials_in_G : ℕ := 528

theorem num_polynomials_in_G_eq_528 : num_polynomials_in_G = 528 := by
  rfl

/-- The main theorem stating that there are 528 polynomials in G -/
theorem count_polynomials_in_G : ∃ (n : ℕ), n = num_polynomials_in_G ∧ n = 528 := by
  use 528
  constructor
  · exact num_polynomials_in_G_eq_528
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_polynomials_in_G_eq_528_count_polynomials_in_G_l414_41407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l414_41400

noncomputable section

/-- The volume of a cylinder -/
def cylinderVolume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The volume of a cone -/
def coneVolume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The ratio of cone volume to cylinder volume -/
def volumeRatio (r : ℝ) (h_cyl : ℝ) (h_cone : ℝ) : ℝ :=
  (coneVolume r h_cone) / (cylinderVolume r h_cyl)

theorem cone_cylinder_volume_ratio :
  let r : ℝ := 5
  let h_cyl : ℝ := 16
  let h_cone : ℝ := (3/4) * h_cyl
  volumeRatio r h_cyl h_cone = 1/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l414_41400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_ugly_product_and_ugly_product_l414_41456

def IsBeautiful (n : ℕ) : Prop :=
  ∃ x y : ℕ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ n * (x + y) = x^2 + y^2

def IsUgly (n : ℕ) : Prop :=
  ¬ IsBeautiful n

theorem beautiful_ugly_product_and_ugly_product :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ IsBeautiful a ∧ IsUgly b ∧ a * b = 2014) ∧
  (∀ u v : ℕ, u > 0 → v > 0 → IsUgly u → IsUgly v → IsUgly (u * v)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_ugly_product_and_ugly_product_l414_41456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_product_is_square_l414_41458

theorem gcd_product_is_square (x y z : ℕ+) 
  (eq : (1 : ℚ) / x.val - (1 : ℚ) / y.val = (1 : ℚ) / z.val) :
  ∃ k : ℕ, (Nat.gcd x.val (Nat.gcd y.val z.val)) * x.val * y.val * z.val = k^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_product_is_square_l414_41458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l414_41499

/-- The hyperbola C with equation x²/a² - y²/b² = 1 -/
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The real axis length of the hyperbola -/
noncomputable def real_axis_length (a : ℝ) : ℝ := 2 * a

/-- The distance from the right focus F(c, 0) to the asymptote y = (b/a)x -/
noncomputable def focus_asymptote_distance (a b c : ℝ) : ℝ :=
  (b * c) / Real.sqrt (a^2 + b^2)

theorem hyperbola_equation (a b c : ℝ) :
  real_axis_length a = 2 * Real.sqrt 5 →
  focus_asymptote_distance a b c = Real.sqrt 5 →
  ∀ x y, hyperbola a b x y ↔ hyperbola (Real.sqrt 5) (Real.sqrt 5) x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l414_41499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_cos_thirty_equals_sin_plus_cos_l414_41496

/-- The angle φ in degrees that satisfies the equation --/
noncomputable def phi : ℝ := 17

/-- Converts degrees to radians --/
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

theorem sqrt_three_cos_thirty_equals_sin_plus_cos :
  ∃ φ : ℝ, 0 < φ ∧ φ < 90 ∧ 
  (Real.sqrt 3 * Real.cos (deg_to_rad 30) = Real.sin (deg_to_rad φ) + Real.cos (deg_to_rad φ)) ∧
  |φ - phi| < 1 := by
  sorry

#check sqrt_three_cos_thirty_equals_sin_plus_cos

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_cos_thirty_equals_sin_plus_cos_l414_41496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l414_41438

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) : 
  y^2 = 10 * w^2 → 3 * w = y * (3 / Real.sqrt 10) :=
by sorry

#check rectangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l414_41438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l414_41489

/-- The focal length of an ellipse -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse on the x-axis with equation x^2/a^2 + y^2/2 = 1 (a > 0) and focal length 2, a = √3 -/
theorem ellipse_focal_length (a : ℝ) (h1 : a > 0) (h2 : focal_length a (Real.sqrt 2) = 2) :
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l414_41489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l414_41497

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_zero_one :
  let p : ℝ × ℝ := (0, 1)
  let tangent_line (x y : ℝ) := x - y + 1 = 0
  tangent_line p.1 p.2 ∧
  ∀ x y : ℝ, tangent_line x y →
    (y - p.2) = (deriv f p.1) * (x - p.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l414_41497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l414_41469

-- Define the function
noncomputable def f (ω : ℕ+) (x : ℝ) : ℝ := Real.cos (ω * x - Real.pi / 3)

-- State the theorem
theorem min_omega :
  ∀ ω : ℕ+, (∀ x : ℝ, f ω (Real.pi / 6 + x) = f ω (Real.pi / 6 - x)) →
  ∃ ω_min : ℕ+, ω_min = 2 ∧ ∀ ω' : ℕ+, (∀ x : ℝ, f ω' (Real.pi / 6 + x) = f ω' (Real.pi / 6 - x)) → ω' ≥ ω_min :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l414_41469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l414_41401

theorem binomial_expansion_theorem (x a : ℝ) (n : ℕ) : 
  (∃ k₁ k₂ k₃ : ℝ, 
    k₁ * (Nat.choose n 1) * x^(n-1) * a^1 = 56 ∧ 
    k₂ * (Nat.choose n 2) * x^(n-2) * a^2 = 126 ∧ 
    k₃ * (Nat.choose n 3) * x^(n-3) * a^3 = 210) → 
  n = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l414_41401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_four_l414_41413

/-- A standard die with six faces numbered from 1 to 6 -/
structure Die :=
  (faces : Finset Nat)
  (face_count : faces.card = 6)
  (face_range : ∀ n, n ∈ faces ↔ 1 ≤ n ∧ n ≤ 6)

/-- The probability of an event in a finite sample space -/
def probability (sample_space : Finset α) (event : Finset α) : ℚ :=
  event.card / sample_space.card

/-- Theorem: The probability of getting a number greater than 4 in a single throw of a standard die is 1/3 -/
theorem prob_greater_than_four (d : Die) :
  probability d.faces (d.faces.filter (λ n => n > 4)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_four_l414_41413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ann_older_than_kristine_current_ages_total_kristine_age_difference_in_10_years_l414_41421

/-- Represents the ages of Ann and Kristine -/
structure Ages where
  ann : ℕ
  kristine : ℕ

/-- The current ages of Ann and Kristine -/
def current_ages : Ages :=
  { ann := 15, kristine := 10 }

/-- The condition that Ann is 5 years older than Kristine -/
theorem ann_older_than_kristine (ages : Ages) : ages.ann = ages.kristine + 5 := by
  sorry

/-- The condition that their current ages total 25 -/
theorem current_ages_total (ages : Ages) : ages.ann + ages.kristine = 25 := by
  sorry

/-- The theorem to be proved -/
theorem kristine_age_difference_in_10_years (ages : Ages) : 
  (ages.kristine + 10) - (ages.ann - ages.kristine) = 5 := by
  have h1 : ages.ann = ages.kristine + 5 := ann_older_than_kristine ages
  have h2 : ages.ann + ages.kristine = 25 := current_ages_total ages
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ann_older_than_kristine_current_ages_total_kristine_age_difference_in_10_years_l414_41421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_sequence_l414_41419

theorem triangle_special_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (a * c = b^2) →  -- geometric sequence condition
  (a + c = 2 * b) →  -- arithmetic sequence condition
  (a^2 + b^2 - c^2) / (2 * a * b) = Real.cos A →  -- law of cosines for angle A
  (a^2 + c^2 - b^2) / (2 * a * c) = Real.cos B →  -- law of cosines for angle B
  (b^2 + c^2 - a^2) / (2 * b * c) = Real.cos C →  -- law of cosines for angle C
  Real.cos B = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_sequence_l414_41419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_remaining_proof_l414_41479

/-- Calculates the number of eggs remaining after recovering the initial investment -/
def remaining_eggs (total_eggs : ℕ) (initial_investment : ℚ) (selling_price : ℚ) : ℕ :=
  total_eggs - Int.toNat ((initial_investment / selling_price).floor)

/-- Proves that given a crate of 30 eggs bought for $5 and sold at $0.20 each, 
    the number of eggs remaining after recovering the initial investment is 5 -/
theorem eggs_remaining_proof :
  remaining_eggs 30 5 (20/100) = 5 := by
  -- Unfold the definition of remaining_eggs
  unfold remaining_eggs
  -- Evaluate the expression
  norm_num
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_remaining_proof_l414_41479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_implies_a_equals_four_l414_41463

theorem intersection_equality_implies_a_equals_four (a : ℝ) : 
  ({a^2 - 1, 2} : Set ℝ) ∩ ({1, 2, 3, 2*a - 4} : Set ℝ) = {a - 2} → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_implies_a_equals_four_l414_41463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_l414_41478

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem derivative_reciprocal :
  ∀ x : ℝ, x ≠ 0 → deriv f x = -1 / (x^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_l414_41478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_l414_41455

theorem shopkeeper_profit_percentage 
  (theft_loss_percent : ℝ) 
  (overall_loss_percent : ℝ) 
  (theft_loss_percent_eq : theft_loss_percent = 40)
  (overall_loss_percent_eq : overall_loss_percent = 34) :
  (theft_loss_percent - overall_loss_percent) / (100 - theft_loss_percent) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_l414_41455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_meter_is_26_50_l414_41432

/-- Represents the properties of a rectangular plot and its fencing cost -/
structure PlotInfo where
  length : ℝ
  breadth_difference : ℝ
  total_cost : ℝ

/-- Calculates the cost per meter of fencing for a given plot -/
noncomputable def cost_per_meter (plot : PlotInfo) : ℝ :=
  plot.total_cost / (2 * (plot.length + (plot.length - plot.breadth_difference)))

/-- Theorem stating that the cost per meter is 26.50 for the given plot specifications -/
theorem cost_per_meter_is_26_50 (plot : PlotInfo) 
  (h1 : plot.length = 58)
  (h2 : plot.length = plot.breadth_difference + (plot.length - plot.breadth_difference))
  (h3 : plot.breadth_difference = 16)
  (h4 : plot.total_cost = 5300) : 
  cost_per_meter plot = 26.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_meter_is_26_50_l414_41432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_100_roots_l414_41486

/-- Sequence of polynomials P_n(x) -/
def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | 1 => λ x => x
  | (n + 2) => λ x => x * P (n + 1) x - P n x

/-- The 100th polynomial in the sequence -/
def P_100 : ℝ → ℝ := P 100

/-- The kth root of P_100 -/
noncomputable def root (k : ℕ) : ℝ := 2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / 202)

/-- Theorem stating the properties of the roots of P_100 -/
theorem P_100_roots :
  (∀ k, k < 100 → P_100 (root k) = 0) ∧
  (∀ k l, k < 100 → l < 100 → k ≠ l → root k ≠ root l) ∧
  (∀ k, k < 100 → -2 ≤ root k ∧ root k ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_100_roots_l414_41486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_relationship_l414_41462

/-- Represents the temperature in °C -/
def Temperature : Type := ℝ

/-- Represents the time in minutes -/
def Time : Type := ℝ

/-- The rate of temperature increase per minute -/
def rate : ℝ := 7

/-- The initial temperature at t = 0 -/
def initial_temp : ℝ := 30

/-- The linear relationship between temperature and time -/
def temp_time_relation (t : ℝ) : ℝ :=
  rate * t + initial_temp

theorem water_boiling_relationship (t : ℝ) (h : t < 10) :
  temp_time_relation t = rate * t + initial_temp :=
by
  -- Unfold the definition of temp_time_relation
  unfold temp_time_relation
  -- The equality now holds by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_relationship_l414_41462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_calculation_l414_41440

/-- Represents the present worth of a sum -/
def present_worth : ℕ → ℕ := sorry

/-- Represents the banker's gain -/
def bankers_gain : ℕ := 16

/-- Represents the true discount -/
def true_discount : ℕ := 96

/-- 
Given a banker's gain of 16 and a true discount of 96,
prove that the present worth of the sum is 80.
-/
theorem present_worth_calculation :
  present_worth bankers_gain = 80 := by
  sorry

#check present_worth_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_calculation_l414_41440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l414_41474

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 4)

theorem range_of_f :
  Set.range f = Set.Ioo 0 (1/4) ∪ {1/4} := by
  sorry

#eval "The range of f has been defined and the theorem has been stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l414_41474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_implies_commutative_l414_41470

/-- A binary operation on a type -/
def BinOp (α : Type) := α → α → α

/-- The property that (a ⋆ b) ⋆ (a ⋆ b) = a ⋆ b for all a and b -/
def MyIdempotent {α : Type} (star : BinOp α) :=
  ∀ a b : α, star (star a b) (star a b) = star a b

/-- Theorem: If a binary operation is idempotent, then it is commutative -/
theorem idempotent_implies_commutative {α : Type} (star : BinOp α) 
  (h : MyIdempotent star) : ∀ a b : α, star a b = star b a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_implies_commutative_l414_41470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l414_41476

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

theorem function_inequality (a b : ℝ) (h : f (2*a + b) + f (4 - 3*b) > 0) : b - a > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l414_41476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l414_41450

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  c = 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l414_41450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_shift_l414_41415

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem symmetry_implies_shift (φ : ℝ) 
  (h1 : |φ| < π/2) 
  (h2 : ∀ x, f (π/6 + x) φ = f (π/6 - x) φ) : 
  ∀ x, f x φ = Real.sin (2 * (x + π/12)) := by
  sorry

#check symmetry_implies_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_shift_l414_41415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_characterization_l414_41488

theorem constant_function_characterization (f : ℝ → ℝ) : 
  (∀ a b : ℝ, f (a + b) = f (a * b)) → 
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_characterization_l414_41488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_implies_range_of_a_l414_41490

/-- The function f(x) = e^x / x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

/-- The function g(x) = -(x-1)^2 + a^2 -/
def g (a x : ℝ) : ℝ := -(x - 1)^2 + a^2

/-- The theorem statement -/
theorem existence_implies_range_of_a :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₂ > 0 ∧ f x₂ ≤ g a x₁) →
  a ∈ Set.Iic (-Real.sqrt (Real.exp 1)) ∪ Set.Ici (Real.sqrt (Real.exp 1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_implies_range_of_a_l414_41490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_neg_two_one_l414_41410

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.pi + Real.arctan (y / x)
           else if x < 0 && y < 0 then -Real.pi + Real.arctan (y / x)
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ)

theorem polar_coordinates_of_neg_two_one :
  let (r, θ) := rectangular_to_polar (-2) 1
  r = Real.sqrt 5 ∧ θ = Real.pi - Real.arctan (1 / 2) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_neg_two_one_l414_41410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l414_41448

noncomputable def f (x : ℝ) : ℝ := |3/4 - 1/2*x| - |5/4 + 1/2*x|

theorem function_properties :
  (∀ x : ℝ, ∀ a : ℝ, f x ≥ a^2 - 3*a ↔ 1 ≤ a ∧ a ≤ 2) ∧
  (∀ m n : ℝ, f m + f n = 4 → m < n → m + n < -5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l414_41448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l414_41466

theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (2, -2) →
  c = (1, lambda) →
  ∃ (k : ℝ), k ≠ 0 ∧ c = k • (2 • a + b) →
  lambda = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l414_41466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l414_41434

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.log x + (1/2) * x^2 - 5*x

-- State the theorem
theorem f_properties :
  (∀ x > 0, f x ≤ -9/2) ∧ 
  (f 1 = -9/2) ∧
  (∀ x > 0, f x ≥ 8 * Real.log 2 - 12) ∧
  (f 4 = 8 * Real.log 2 - 12) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Ioo (2*m) (m+1), ∀ y ∈ Set.Ioo (2*m) (m+1), x < y → f x > f y) ↔ 
    m ≥ 1/2 ∧ m < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l414_41434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_correct_l414_41491

/-- The original line equation -/
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- The intersection point of the original line with the y-axis -/
def intersection_point : ℝ × ℝ :=
  (0, -2)

/-- The rotated line equation -/
def rotated_line (x y : ℝ) : Prop := -x + 2 * y + 4 = 0

/-- Rotation function -/
noncomputable def rotate_point (center : ℝ × ℝ) (point : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

/-- Theorem stating that the rotated line is correct -/
theorem rotated_line_correct :
  ∀ x y : ℝ,
  original_line x y →
  ∃ x' y' : ℝ,
  rotated_line x' y' ∧
  (x', y') = rotate_point intersection_point (x, y) (π / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_correct_l414_41491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l414_41477

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  (x + y + z)^2 ≥ 4 * (y * z * Real.sin A^2 + z * x * Real.sin B^2 + x * y * Real.sin C^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l414_41477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximilian_wealth_greater_than_national_l414_41405

-- Define the wealth of each oligarch at the end of each year
variable (alejandro_2011 : ℝ)
variable (alejandro_2012 : ℝ)
variable (maximilian_2011 : ℝ)
variable (maximilian_2012 : ℝ)

-- Define the national wealth
variable (national_wealth : ℝ)

-- State the given conditions
axiom condition1 : alejandro_2012 = 2 * maximilian_2011
axiom condition2 : maximilian_2012 < alejandro_2011

-- Define the national wealth calculation
axiom national_wealth_def : national_wealth = alejandro_2012 + maximilian_2012 - alejandro_2011 - maximilian_2011

-- State the theorem to be proved
theorem maximilian_wealth_greater_than_national :
  maximilian_2012 > national_wealth :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximilian_wealth_greater_than_national_l414_41405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_one_l414_41430

/-- A quadratic function passing through three given points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : ((-2 : ℝ)^2 * a + (-2 : ℝ) * b + c) = 9
  point2 : ((4 : ℝ)^2 * a + 4 * b + c) = 9
  point3 : ((3 : ℝ)^2 * a + 3 * b + c) = 6

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- Theorem stating that the x-coordinate of the vertex is 1 -/
theorem vertex_x_is_one (f : QuadraticFunction) : vertex_x f = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_one_l414_41430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_l414_41445

def g : ℝ → ℝ := sorry

axiom g_definition (x : ℝ) : g (x + 1) = 2 * x + 3

theorem g_value (x : ℝ) : g x = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_l414_41445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_point_B_l414_41409

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line segment
structure LineSegment where
  start : Point
  finish : Point

def isParallelToXAxis (l : LineSegment) : Prop :=
  l.start.y = l.finish.y

noncomputable def length (l : LineSegment) : ℝ :=
  Real.sqrt ((l.finish.x - l.start.x)^2 + (l.finish.y - l.start.y)^2)

theorem coordinates_of_point_B (ab : LineSegment) 
  (h1 : isParallelToXAxis ab)
  (h2 : ab.start = Point.mk (-2) 5)
  (h3 : length ab = 3) :
  ab.finish = Point.mk (-5) 5 ∨ ab.finish = Point.mk 1 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_point_B_l414_41409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_3_11_l414_41471

/-- The decimal representation of 3/11 -/
def decimal_rep_3_11 : ℚ := 3 / 11

/-- The length of the repeating sequence in the decimal representation of 3/11 -/
def repeat_length : ℕ := 6

/-- The position of the 150th decimal place within the repeating sequence -/
def position_in_sequence : Fin repeat_length := ⟨150 % repeat_length, by {
  apply Nat.mod_lt
  exact Nat.zero_lt_succ 5
}⟩

/-- The 150th decimal digit in the representation of 3/11 -/
def digit_150 : ℕ := 7

theorem digit_150_of_3_11 :
  ∃ (seq : Fin repeat_length → ℕ),
    (∀ n, seq n < 10) ∧
    decimal_rep_3_11 = (3 / 11 : ℚ) ∧
    seq position_in_sequence = digit_150 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_3_11_l414_41471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_approx_3_9583_l414_41475

/-- Represents a segment of the journey -/
structure JourneySegment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a journey segment -/
noncomputable def time_taken (segment : JourneySegment) : ℝ := segment.distance / segment.speed

/-- Represents the complete journey -/
structure Journey where
  car_ab : JourneySegment
  train_ab : JourneySegment
  bike_ab : JourneySegment
  car_ba : JourneySegment
  train_ba : JourneySegment
  bike_ba : JourneySegment

/-- Calculates the total time for the round trip -/
noncomputable def total_time (j : Journey) : ℝ :=
  time_taken j.car_ab + time_taken j.train_ab + time_taken j.bike_ab +
  time_taken j.car_ba + time_taken j.train_ba + time_taken j.bike_ba

/-- The main theorem: proves that the total time for the given journey is approximately 3.9583 hours -/
theorem round_trip_time_approx_3_9583 (j : Journey) 
  (h1 : j.car_ab = { distance := 40, speed := 80 })
  (h2 : j.train_ab = { distance := 60, speed := 120 })
  (h3 : j.bike_ab = { distance := 20, speed := 20 })
  (h4 : j.car_ba = { distance := 40, speed := 120 })
  (h5 : j.train_ba = { distance := 60, speed := 96 })
  (h6 : j.bike_ba = { distance := 20, speed := 20 }) :
  ∃ ε > 0, |total_time j - 3.9583| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_approx_3_9583_l414_41475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baguettes_sold_after_first_batch_l414_41435

/-- The number of baguettes sold after the first batch in a bakery -/
theorem baguettes_sold_after_first_batch : ℕ := by
  let batches_per_day : ℕ := 3
  let baguettes_per_batch : ℕ := 48
  let total_baguettes : ℕ := batches_per_day * baguettes_per_batch
  let sold_after_second_batch : ℕ := 52
  let sold_after_third_batch : ℕ := 49
  let baguettes_left : ℕ := 6
  let total_sold : ℕ := total_baguettes - baguettes_left
  let sold_after_first_batch : ℕ := total_sold - (sold_after_second_batch + sold_after_third_batch)
  have h : sold_after_first_batch = 37 := by sorry
  exact sold_after_first_batch

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baguettes_sold_after_first_batch_l414_41435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_lineup_probability_l414_41433

def total_buses : ℕ := 16
def red_buses : ℕ := 5
def blue_buses : ℕ := 6
def yellow_buses : ℕ := 5
def lineup_size : ℕ := 7

theorem bus_lineup_probability :
  let p := (red_buses : ℚ) / total_buses *
           (Nat.choose blue_buses 4 * Nat.choose yellow_buses 2 : ℚ) /
           (Nat.choose (total_buses - 1) (lineup_size - 1))
  p = 75 / 8080 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_lineup_probability_l414_41433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l414_41484

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - a/2)*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (8/3 ≤ a ∧ a < 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l414_41484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_exists_l414_41443

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a color (red or blue) -/
inductive Color where
  | Red
  | Blue

/-- Represents a regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → Point

/-- Represents the sequence of 11 pentagons -/
def PentagonSequence := Fin 11 → RegularPentagon

/-- Predicate to check if four points form a cyclic quadrilateral -/
def IsCyclicQuadrilateral (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Function to get the midpoint of two points -/
def Midpoint (p1 p2 : Point) : Point := sorry

/-- Predicate to check if a pentagon's vertices are midpoints of the previous pentagon's edges -/
def IsMidpointPentagon (prev next : RegularPentagon) : Prop := sorry

/-- Coloring of all 55 vertices -/
def Coloring := Fin 11 → Fin 5 → Color

/-- Main theorem -/
theorem cyclic_quadrilateral_exists 
  (pentagons : PentagonSequence)
  (coloring : Coloring)
  (h1 : ∀ n : Fin 10, IsMidpointPentagon (pentagons n) (pentagons (n.succ))) :
  ∃ (i j k l : Fin 11) (a b c d : Fin 5),
    let p1 := (pentagons i).vertices a
    let p2 := (pentagons j).vertices b
    let p3 := (pentagons k).vertices c
    let p4 := (pentagons l).vertices d
    IsCyclicQuadrilateral p1 p2 p3 p4 ∧
    coloring i a = coloring j b ∧
    coloring j b = coloring k c ∧
    coloring k c = coloring l d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_exists_l414_41443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_equals_prob_at_l414_41437

/-- A random walk on a line with a reflecting barrier at 0 -/
structure RandomWalk where
  n : ℕ  -- number of steps
  a : ℕ → ℝ  -- probability of being at point 0 after n steps
  b : ℕ → ℝ  -- probability of being at point 1 after n steps
  c : ℕ → ℝ  -- probability of being at point 2 after n steps
  d : ℕ → ℝ  -- probability of being at point 3 after n steps

/-- The recurrence relations for the random walk -/
def valid_recurrence (rw : RandomWalk) : Prop :=
  (∀ k, rw.a (k+1) = rw.b k * (1/2)) ∧
  (∀ k, rw.b (k+1) = rw.a k * 1 + rw.c k * (1/2)) ∧
  (∀ k, rw.c (k+1) = rw.b k * (1/2)) ∧
  (∀ k, rw.d (k+1) = rw.d k * 1 + rw.c k * (1/2))

/-- Initial conditions for the random walk starting at point 1 -/
def valid_initial_conditions (rw : RandomWalk) : Prop :=
  rw.a 0 = 0 ∧ rw.b 0 = 1 ∧ rw.c 0 = 0 ∧ rw.d 0 = 0

/-- The probability of reaching point 3 at least once in n steps -/
noncomputable def probability_reach_point_3_at_least_once (rw : RandomWalk) (n : ℕ) : ℝ :=
  sorry -- This definition would be implemented based on the problem's specifics

/-- The main theorem: probability of reaching point 3 at least once in n steps
    is equal to the probability of being at point 3 after n steps -/
theorem prob_reach_equals_prob_at (rw : RandomWalk) :
  valid_recurrence rw → valid_initial_conditions rw →
  ∀ n, rw.d n = probability_reach_point_3_at_least_once rw n := by
  sorry -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_equals_prob_at_l414_41437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_results_related_to_school_probability_two_from_school_B_l414_41498

-- Define the data from the contingency table
def school_A_pass : ℕ := 20
def school_A_fail : ℕ := 40
def school_B_pass : ℕ := 30
def school_B_fail : ℕ := 20

-- Define the chi-square statistic function
noncomputable def chi_square (a b c d : ℕ) : ℝ :=
  let n : ℕ := a + b + c + d
  (n : ℝ) * (a * d - b * c : ℝ)^2 / ((a + b : ℝ) * (c + d : ℝ) * (a + c : ℝ) * (b + d : ℝ))

-- State the theorem
theorem exam_results_related_to_school : 
  chi_square school_A_pass school_A_fail school_B_pass school_B_fail > 6.635 := by
  sorry

-- Define the probability calculation for the second part
def prob_two_from_school_B : ℚ := 3 / 10

-- State the theorem for the second part
theorem probability_two_from_school_B : 
  prob_two_from_school_B = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_results_related_to_school_probability_two_from_school_B_l414_41498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_isosceles_min_perimeter_isosceles_l414_41467

-- Define a structure for a triangle
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Define the area of a triangle using Heron's formula
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Theorem for part (a)
theorem max_area_isosceles (a P : ℝ) (ha : a > 0) (hP : P > a) :
  ∀ t : Triangle, t.a = a → perimeter t = P →
    area t ≤ area { a := a, b := (P - a) / 2, c := (P - a) / 2, 
                    ha := ha, 
                    hb := by sorry, 
                    hc := by sorry, 
                    triangle_inequality := by sorry } :=
by sorry

-- Theorem for part (b)
theorem min_perimeter_isosceles (a S : ℝ) (ha : a > 0) (hS : S > 0) :
  ∀ t : Triangle, t.a = a → area t = S →
    perimeter t ≥ perimeter { a := a, 
                              b := Real.sqrt (S * (4 * S + a * a)) / a, 
                              c := Real.sqrt (S * (4 * S + a * a)) / a,
                              ha := ha, 
                              hb := by sorry, 
                              hc := by sorry, 
                              triangle_inequality := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_isosceles_min_perimeter_isosceles_l414_41467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l414_41492

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℚ
  downstream : ℚ

/-- Calculates the speed in still water given upstream and downstream speeds -/
def speedInStillWater (s : RowingSpeed) : ℚ :=
  (s.upstream + s.downstream) / 2

/-- Theorem: Given a man who can row upstream at 25 kmph and downstream at 39 kmph,
    his speed in still water is 32 kmph -/
theorem mans_speed_in_still_water :
  let s : RowingSpeed := { upstream := 25, downstream := 39 }
  speedInStillWater s = 32 := by
  -- Unfold the definition of speedInStillWater
  unfold speedInStillWater
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l414_41492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l414_41436

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_third_term 
  (seq : ArithmeticSequence) 
  (h : sum_n seq 5 = 25) : 
  seq.a 3 = 5 := by
  sorry

#check arithmetic_sequence_third_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l414_41436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l414_41473

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  |c₂ / Real.sqrt (a₂^2 + b₂^2) - c₁ / Real.sqrt (a₁^2 + b₁^2)|

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

theorem distance_between_given_lines :
  let line1 : ℝ → ℝ → ℝ := λ x y => 2*x + y - 2
  let line2 : ℝ → ℝ → ℝ := λ x y => 4*x + 2*y + 6
  parallel_lines 2 1 4 2 →
  distance_between_parallel_lines 2 1 (-2) 4 2 6 = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l414_41473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_satisfying_equation_l414_41411

-- Define the θ operation
def theta (m v : ℕ) : ℕ := m % v

-- Theorem statement
theorem smallest_x_satisfying_equation : 
  ∃ (x : ℕ), ((theta (theta x 33) 17) - (theta 99 (theta 33 17)) = 4) ∧ 
  (∀ (y : ℕ), y < x → ((theta (theta y 33) 17) - (theta 99 (theta 33 17)) ≠ 4)) ∧
  x = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_satisfying_equation_l414_41411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_sqrt5_plus_sqrt3_fourth_power_l414_41426

theorem nearest_integer_to_sqrt5_plus_sqrt3_fourth_power :
  ∃ n : ℤ, n = 248 ∧ 
  ∀ m : ℤ, |↑m - (Real.sqrt 5 + Real.sqrt 3)^4| ≥ |↑n - (Real.sqrt 5 + Real.sqrt 3)^4| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_sqrt5_plus_sqrt3_fourth_power_l414_41426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_distance_to_focus_l414_41406

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The maximum distance from any point on the ellipse to its left focus -/
noncomputable def max_distance_to_focus (e : Ellipse) : ℝ := e.a + Real.sqrt (e.a^2 - e.b^2)

/-- Membership of a point on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The statement to be proved -/
theorem ellipse_max_distance_to_focus 
  (e : Ellipse) 
  (chord_length : ℝ)
  (p : Point)
  (h_chord : chord_length = 6 * Real.sqrt 2)
  (h_p : p.x = 2 ∧ p.y = 1)
  (h_slope : ∃ (a b : Point), on_ellipse a e ∧ on_ellipse b e ∧ 
             p.x = (a.x + b.x) / 2 ∧ p.y = (a.y + b.y) / 2 ∧ 
             (b.y - a.y) / (b.x - a.x) = -1)
  : max_distance_to_focus e = 6 * Real.sqrt 2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_distance_to_focus_l414_41406
