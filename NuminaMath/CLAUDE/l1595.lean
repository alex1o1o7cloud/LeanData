import Mathlib

namespace NUMINAMATH_CALUDE_traveler_water_consumption_l1595_159576

/-- The amount of water drunk by the traveler and camel -/
theorem traveler_water_consumption (traveler_ounces : ℝ) : 
  traveler_ounces > 0 →  -- Assume the traveler drinks a positive amount
  (∃ (camel_ounces : ℝ), 
    camel_ounces = 7 * traveler_ounces ∧  -- Camel drinks 7 times as much
    128 * 2 = traveler_ounces + camel_ounces) →  -- Total consumption is 2 gallons
  traveler_ounces = 32 := by
sorry

end NUMINAMATH_CALUDE_traveler_water_consumption_l1595_159576


namespace NUMINAMATH_CALUDE_inequality_proof_l1595_159589

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^4 - y^4 = x - y) :
  (x - y) / (x^6 - y^6) ≤ (4/3) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1595_159589


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1595_159501

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 7 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 4 ∧ B = 3 ∧ C = -1 ∧ D = -1 ∧ E = 42 ∧ F = 10 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1595_159501


namespace NUMINAMATH_CALUDE_sum_g_32_neg_32_l1595_159567

/-- A function g defined as a polynomial of even degree -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 5

/-- Theorem stating that the sum of g(32) and g(-32) equals 6 -/
theorem sum_g_32_neg_32 (a b c : ℝ) (h : g a b c 32 = 3) :
  g a b c 32 + g a b c (-32) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_32_neg_32_l1595_159567


namespace NUMINAMATH_CALUDE_three_W_five_l1595_159570

-- Define the W operation
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- Theorem statement
theorem three_W_five : W 3 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_three_W_five_l1595_159570


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l1595_159528

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 3 → ℝ := ![3, -2, 4]
  let v₂ : Fin 3 → ℝ := ![2, -1, 5]
  v₁ - 3 • v₂ = ![(-3 : ℝ), 1, -11] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l1595_159528


namespace NUMINAMATH_CALUDE_sum_of_digits_1197_l1595_159511

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem sum_of_digits_1197 : digit_sum 1197 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1197_l1595_159511


namespace NUMINAMATH_CALUDE_student_rabbit_difference_l1595_159577

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 4

/-- The number of students in each third-grade classroom -/
def students_per_classroom : ℕ := 18

/-- The number of pet rabbits in each third-grade classroom -/
def rabbits_per_classroom : ℕ := 2

/-- The difference between the total number of students and the total number of rabbits -/
theorem student_rabbit_difference : 
  num_classrooms * students_per_classroom - num_classrooms * rabbits_per_classroom = 64 := by
  sorry

end NUMINAMATH_CALUDE_student_rabbit_difference_l1595_159577


namespace NUMINAMATH_CALUDE_vacation_cost_l1595_159531

/-- The total cost of a vacation satisfying specific conditions -/
theorem vacation_cost : ∃ (C P : ℝ), 
  C = 5 * P ∧ 
  C = 7 * (P - 40) ∧ 
  C = 8 * (P - 60) ∧ 
  C = 700 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l1595_159531


namespace NUMINAMATH_CALUDE_tangent_slope_x_squared_at_one_l1595_159597

theorem tangent_slope_x_squared_at_one : 
  let f : ℝ → ℝ := fun x ↦ x^2
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_x_squared_at_one_l1595_159597


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l1595_159530

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the distances
def AC : ℝ := 10
def AB : ℝ := 17
def DC : ℝ := 6

-- Define coplanarity
def coplanar (A B C D : ℝ × ℝ) : Prop := sorry

-- Define right angle
def right_angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_ABC (h1 : coplanar A B C D) 
                             (h2 : right_angle A D C) :
  triangle_area A B C = 84 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l1595_159530


namespace NUMINAMATH_CALUDE_polyhedron_property_l1595_159559

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  t : ℕ  -- Number of triangular faces
  h : ℕ  -- Number of hexagonal faces
  T : ℕ  -- Number of triangular faces meeting at each vertex
  H : ℕ  -- Number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 30
  face_types : t + h = F
  edge_count : E = (3 * t + 6 * h) / 2
  vertex_count : V = 3 * t / T
  triangle_hex_relation : T = 1 ∧ H = 2

/-- Theorem stating the specific property of the polyhedron -/
theorem polyhedron_property (p : ConvexPolyhedron) : 100 * p.H + 10 * p.T + p.V = 270 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l1595_159559


namespace NUMINAMATH_CALUDE_trailing_zeros_mod_500_l1595_159583

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  (List.range 51).map factorial
    |> List.foldl (·*·) 1
    |> Nat.digits 10
    |> List.reverse
    |> List.takeWhile (·==0)
    |> List.length

theorem trailing_zeros_mod_500 :
  trailingZeros 50 % 500 = 12 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_mod_500_l1595_159583


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1595_159524

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the conditions
def lies_on_y_axis (P : Point) : Prop :=
  P.x = 0

def parallel_to_x_axis (P Q : Point) : Prop :=
  P.y = Q.y

def equal_distance_to_axes (P : Point) : Prop :=
  |P.x| = |P.y|

-- Main theorem
theorem point_P_coordinates (a : ℝ) :
  let P : Point := ⟨2*a - 2, a + 5⟩
  let Q : Point := ⟨2, 5⟩
  (lies_on_y_axis P ∨ parallel_to_x_axis P Q ∨ equal_distance_to_axes P) →
  (P = ⟨12, 12⟩ ∨ P = ⟨-12, -12⟩ ∨ P = ⟨-4, 4⟩ ∨ P = ⟨4, -4⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1595_159524


namespace NUMINAMATH_CALUDE_card_sum_perfect_square_l1595_159598

theorem card_sum_perfect_square (n : ℕ) (h : n ≥ 100) :
  ∃ a b c : ℕ, n ≤ a ∧ a < b ∧ b < c ∧ c ≤ 2*n ∧
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = y^2 ∧ c + a = z^2 :=
by sorry

end NUMINAMATH_CALUDE_card_sum_perfect_square_l1595_159598


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1595_159586

theorem min_value_quadratic_form :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1595_159586


namespace NUMINAMATH_CALUDE_noahs_closet_capacity_l1595_159508

theorem noahs_closet_capacity (ali_capacity : ℕ) (noah_total_capacity : ℕ) : 
  ali_capacity = 200 → noah_total_capacity = 100 → 
  (noah_total_capacity / 2 : ℚ) / ali_capacity = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_noahs_closet_capacity_l1595_159508


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l1595_159563

/-- Represents a sampling task with a population size and sample size -/
structure SamplingTask where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a stratified population with different group sizes -/
structure StratifiedPopulation where
  group_sizes : List ℕ

/-- Enumeration of sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Determines the most appropriate sampling method for a given task -/
def most_appropriate_sampling_method (task : SamplingTask) (stratified_info : Option StratifiedPopulation) : SamplingMethod :=
  sorry

/-- The three sampling tasks from the problem -/
def yogurt_task : SamplingTask := { population_size := 10, sample_size := 3 }
def attendees_task : SamplingTask := { population_size := 1280, sample_size := 32 }
def staff_task : SamplingTask := { population_size := 160, sample_size := 20 }

/-- The stratified population information for the staff task -/
def staff_stratified : StratifiedPopulation := { group_sizes := [120, 16, 24] }

theorem appropriate_sampling_methods :
  most_appropriate_sampling_method yogurt_task none = SamplingMethod.SimpleRandom ∧
  most_appropriate_sampling_method attendees_task none = SamplingMethod.Systematic ∧
  most_appropriate_sampling_method staff_task (some staff_stratified) = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l1595_159563


namespace NUMINAMATH_CALUDE_grape_rate_specific_grape_rate_l1595_159545

/-- The rate of grapes per kg given the following conditions:
  1. 8 kg of grapes were purchased at an unknown rate
  2. 9 kg of mangoes were purchased at 50 rupees per kg
  3. The total amount paid was 1010 rupees -/
theorem grape_rate : ℕ → ℕ → ℕ → ℕ → Prop :=
  λ grape_quantity mango_quantity mango_rate total_paid =>
    ∃ (G : ℕ),
      grape_quantity * G + mango_quantity * mango_rate = total_paid ∧
      G = 70

/-- The specific instance of the problem -/
theorem specific_grape_rate : grape_rate 8 9 50 1010 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_specific_grape_rate_l1595_159545


namespace NUMINAMATH_CALUDE_simplify_fraction_l1595_159582

theorem simplify_fraction : (123 : ℚ) / 999 * 27 = 123 / 37 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1595_159582


namespace NUMINAMATH_CALUDE_cookie_solution_l1595_159533

def cookie_problem (initial_cookies : ℕ) : Prop :=
  let andy_ate : ℕ := 3
  let brother_ate : ℕ := 5
  let team_size : ℕ := 8
  let team_sequence : List ℕ := List.range team_size |>.map (λ n => 2*n + 1)
  let team_ate : ℕ := team_sequence.sum
  initial_cookies = andy_ate + brother_ate + team_ate

theorem cookie_solution : 
  ∃ (initial_cookies : ℕ), cookie_problem initial_cookies ∧ initial_cookies = 72 := by
  sorry

end NUMINAMATH_CALUDE_cookie_solution_l1595_159533


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_square_sum_l1595_159578

theorem consecutive_even_numbers_square_sum (a b c d : ℕ) : 
  (∃ x : ℕ, a = 2*x ∧ b = 2*x + 2 ∧ c = 2*x + 4 ∧ d = 2*x + 6) →  -- Consecutive even numbers
  a + b + c + d = 36 →                                           -- Sum is 36
  a^2 + b^2 + c^2 + d^2 = 344 :=                                 -- Sum of squares is 344
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_square_sum_l1595_159578


namespace NUMINAMATH_CALUDE_class_average_problem_l1595_159535

/-- Given a class of 25 students where 10 students average 88% and the overall average is 79%,
    this theorem proves that the average percentage of the remaining 15 students is 73%. -/
theorem class_average_problem (total_students : Nat) (group_a_students : Nat) (group_b_students : Nat)
    (group_b_average : ℝ) (overall_average : ℝ) :
    total_students = 25 →
    group_a_students = 15 →
    group_b_students = 10 →
    group_b_average = 88 →
    overall_average = 79 →
    (group_a_students * x + group_b_students * group_b_average) / total_students = overall_average →
    x = 73 :=
  by sorry


end NUMINAMATH_CALUDE_class_average_problem_l1595_159535


namespace NUMINAMATH_CALUDE_intersection_A_B_subset_A_C_iff_a_in_0_2_l1595_159550

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def B : Set ℝ := {x | (x-1)/(x-3) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - (2*a+4)*x + a^2 + 4*a ≤ 0}

-- Define the interval (3, 4]
def interval_3_4 : Set ℝ := {x | 3 < x ∧ x ≤ 4}

-- Theorem statements
theorem intersection_A_B : A ∩ B = interval_3_4 := by sorry

theorem subset_A_C_iff_a_in_0_2 :
  ∀ a : ℝ, A ⊆ C a ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_subset_A_C_iff_a_in_0_2_l1595_159550


namespace NUMINAMATH_CALUDE_candle_burning_theorem_l1595_159518

/-- Represents the state of a burning candle -/
structure BurningCandle where
  burn_rate : ℝ
  remaining : ℝ

/-- Represents the state of three burning candles -/
structure ThreeCandles where
  candle1 : BurningCandle
  candle2 : BurningCandle
  candle3 : BurningCandle

/-- 
Given three candles burning at constant rates, if 2/5 of the second candle
and 3/7 of the third candle remain when the first candle burns out, then
1/21 of the third candle will remain when the second candle burns out.
-/
theorem candle_burning_theorem (candles : ThreeCandles) 
  (h1 : candles.candle1.burn_rate > 0)
  (h2 : candles.candle2.burn_rate > 0)
  (h3 : candles.candle3.burn_rate > 0)
  (h4 : candles.candle2.remaining = 2/5)
  (h5 : candles.candle3.remaining = 3/7) :
  candles.candle3.remaining - (candles.candle2.remaining / candles.candle2.burn_rate) * candles.candle3.burn_rate = 1/21 := by
  sorry

end NUMINAMATH_CALUDE_candle_burning_theorem_l1595_159518


namespace NUMINAMATH_CALUDE_incorrect_statement_l1595_159549

theorem incorrect_statement : ¬(∀ (p q : Prop), (p ∧ q → False) → (p → False) ∧ (q → False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1595_159549


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1595_159552

theorem inequality_equivalence (x : ℝ) :
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1595_159552


namespace NUMINAMATH_CALUDE_max_cables_is_150_l1595_159556

/-- Represents the maximum number of cables that can be installed between
    Brand A and Brand B computers under given conditions. -/
def max_cables (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) 
               (connectable_brand_b : ℕ) : ℕ :=
  brand_a_computers * connectable_brand_b

/-- Theorem stating that the maximum number of cables is 150 under the given conditions. -/
theorem max_cables_is_150 :
  max_cables 50 15 35 10 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_max_cables_is_150_l1595_159556


namespace NUMINAMATH_CALUDE_product_of_integers_l1595_159532

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 3 * (a * b) + 4 * a = 5 * b + 318 → a * b = 56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l1595_159532


namespace NUMINAMATH_CALUDE_triangle_dot_product_l1595_159515

/-- Given a triangle ABC with |AB| = 4, |AC| = 1, and area √3, prove AB · AC = ±2 -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 16) →  -- |AB| = 4
  (AC.1^2 + AC.2^2 = 1) →   -- |AC| = 1
  (abs (AB.1 * AC.2 - AB.2 * AC.1) = 2 * Real.sqrt 3) →  -- Area = √3
  (AB.1 * AC.1 + AB.2 * AC.2 = 2) ∨ (AB.1 * AC.1 + AB.2 * AC.2 = -2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l1595_159515


namespace NUMINAMATH_CALUDE_ceiling_times_self_216_l1595_159573

theorem ceiling_times_self_216 :
  ∃! x : ℝ, ⌈x⌉ * x = 216 ∧ x = 14.4 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_216_l1595_159573


namespace NUMINAMATH_CALUDE_soda_price_increase_l1595_159571

theorem soda_price_increase (original_price : ℝ) (new_price : ℝ) (increase_percentage : ℝ) : 
  new_price = 15 ∧ increase_percentage = 50 ∧ new_price = original_price * (1 + increase_percentage / 100) → 
  original_price = 10 := by
sorry

end NUMINAMATH_CALUDE_soda_price_increase_l1595_159571


namespace NUMINAMATH_CALUDE_max_non_empty_intersection_l1595_159551

-- Define the set A_n
def A (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- Define the intersection of sets A_1 to A_n
def intersection_up_to (n : ℕ) : Set ℝ := ⋂ i ∈ Finset.range n, A (i + 1)

-- State the theorem
theorem max_non_empty_intersection :
  (∃ (n : ℕ), intersection_up_to n ≠ ∅ ∧
    ∀ (m : ℕ), m > n → intersection_up_to m = ∅) ∧
  (∀ (n : ℕ), intersection_up_to n ≠ ∅ → n ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_non_empty_intersection_l1595_159551


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l1595_159513

theorem buffet_meal_combinations : 
  (Nat.choose 4 2) * (Nat.choose 5 3) * (Nat.choose 5 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_buffet_meal_combinations_l1595_159513


namespace NUMINAMATH_CALUDE_value_of_x_l1595_159507

theorem value_of_x : (2015^2 - 2015) / 2015 = 2014 := by sorry

end NUMINAMATH_CALUDE_value_of_x_l1595_159507


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l1595_159562

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_term_of_specific_arithmetic_sequence :
  let a₁ := 3
  let a₂ := 7
  let a₃ := 11
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 5 = 19 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l1595_159562


namespace NUMINAMATH_CALUDE_gemma_change_is_five_l1595_159539

-- Define the given conditions
def number_of_pizzas : ℕ := 4
def price_per_pizza : ℕ := 10
def tip_amount : ℕ := 5
def payment_amount : ℕ := 50

-- Define the function to calculate the change
def calculate_change (pizzas : ℕ) (price : ℕ) (tip : ℕ) (payment : ℕ) : ℕ :=
  payment - (pizzas * price + tip)

-- Theorem statement
theorem gemma_change_is_five :
  calculate_change number_of_pizzas price_per_pizza tip_amount payment_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_gemma_change_is_five_l1595_159539


namespace NUMINAMATH_CALUDE_f_value_at_8pi_3_l1595_159502

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_8pi_3 :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, f (x + π) = f x) →  -- f has period π
  (∀ x ∈ Set.Icc 0 (π/2), f x = Real.sqrt 3 * Real.tan x - 1) →  -- definition of f on [0, π/2)
  f (8*π/3) = 2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_8pi_3_l1595_159502


namespace NUMINAMATH_CALUDE_reciprocal_problem_l1595_159564

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 16) : 200 * (1 / x) = 100 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l1595_159564


namespace NUMINAMATH_CALUDE_bobs_age_l1595_159520

theorem bobs_age (alice : ℝ) (bob : ℝ) 
  (h1 : bob = 3 * alice - 20) 
  (h2 : bob + alice = 70) : 
  bob = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_bobs_age_l1595_159520


namespace NUMINAMATH_CALUDE_keychain_cost_decrease_l1595_159565

theorem keychain_cost_decrease (P : ℝ) : 
  P > 0 →                           -- Selling price is positive
  P - 50 = 0.5 * P →                -- New profit is 50% of selling price
  P - 0.75 * P = 0.25 * P →         -- Initial profit was 25% of selling price
  0.75 * P = 75 :=                  -- Initial cost was $75
by
  sorry

end NUMINAMATH_CALUDE_keychain_cost_decrease_l1595_159565


namespace NUMINAMATH_CALUDE_water_cost_for_family_of_six_l1595_159572

/-- The cost of fresh water for a family for one day -/
def water_cost (family_size : ℕ) (purification_cost : ℚ) (water_per_person : ℚ) : ℚ :=
  family_size * water_per_person * purification_cost

/-- Proof that the water cost for a family of 6 is $3 -/
theorem water_cost_for_family_of_six :
  water_cost 6 1 (1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_water_cost_for_family_of_six_l1595_159572


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l1595_159541

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a 10-player round-robin tournament, there are 45 matches -/
theorem ten_player_tournament_matches :
  num_matches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l1595_159541


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1595_159543

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := {1 - 1 / n, 1 + 1 / n^2} ∪ Finset.range (n - 2)
  (Finset.sum set id) / n = 1 - 1 / n^2 + 1 / n^3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1595_159543


namespace NUMINAMATH_CALUDE_taxicab_distance_properties_l1595_159519

/-- Taxicab distance between two points -/
def taxicab_distance (p q : ℝ × ℝ) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

/-- Check if a point is on a line segment -/
def on_segment (a b c : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ c = (a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2))

/-- The set of points equidistant from two given points -/
def equidistant_set (m n : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | taxicab_distance p m = taxicab_distance p n}

theorem taxicab_distance_properties :
  (∀ a b c : ℝ × ℝ, on_segment a b c → taxicab_distance a c + taxicab_distance c b = taxicab_distance a b) ∧
  ¬(∀ a b c : ℝ × ℝ, taxicab_distance a c + taxicab_distance c b > taxicab_distance a b) ∧
  equidistant_set (-1, 0) (1, 0) = {p : ℝ × ℝ | p.1 = 0} ∧
  (∀ p : ℝ × ℝ, p.1 + p.2 = 2 * Real.sqrt 5 → taxicab_distance (0, 0) p ≥ 2 * Real.sqrt 5) ∧
  (∃ p : ℝ × ℝ, p.1 + p.2 = 2 * Real.sqrt 5 ∧ taxicab_distance (0, 0) p = 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_taxicab_distance_properties_l1595_159519


namespace NUMINAMATH_CALUDE_water_flow_rates_verify_conditions_l1595_159500

/-- Represents the water flow model with introducing and removing pipes -/
structure WaterFlowModel where
  /-- Water flow rate of one introducing pipe in m³/h -/
  inlet_rate : ℝ
  /-- Water flow rate of one removing pipe in m³/h -/
  outlet_rate : ℝ

/-- Theorem stating the correct water flow rates given the problem conditions -/
theorem water_flow_rates (model : WaterFlowModel) : 
  (5 * (4 * model.inlet_rate - 3 * model.outlet_rate) = 1000) ∧ 
  (2 * (2 * model.inlet_rate - 2 * model.outlet_rate) = 180) →
  model.inlet_rate = 65 ∧ model.outlet_rate = 20 := by
  sorry

/-- Function to calculate the net water gain in a given time period -/
def net_water_gain (model : WaterFlowModel) (inlet_count outlet_count : ℕ) (hours : ℝ) : ℝ :=
  hours * (inlet_count * model.inlet_rate - outlet_count * model.outlet_rate)

/-- Verifies that the calculated rates satisfy the given conditions -/
theorem verify_conditions (model : WaterFlowModel) 
  (h1 : model.inlet_rate = 65) 
  (h2 : model.outlet_rate = 20) : 
  net_water_gain model 4 3 5 = 1000 ∧ 
  net_water_gain model 2 2 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_rates_verify_conditions_l1595_159500


namespace NUMINAMATH_CALUDE_log_properties_l1595_159526

-- Define the logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- State the properties to be proven
theorem log_properties :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) :=
by sorry

end NUMINAMATH_CALUDE_log_properties_l1595_159526


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l1595_159566

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (5, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 21 :=
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l1595_159566


namespace NUMINAMATH_CALUDE_burger_cost_is_110_l1595_159527

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 110

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

theorem burger_cost_is_110 :
  (∃ (s : ℕ), 4 * burger_cost + 3 * s = 440 ∧ 3 * burger_cost + 2 * s = 330) →
  burger_cost = 110 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_110_l1595_159527


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1595_159547

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n ≥ 3 → 
  exterior_angle = 45 →
  (360 : ℝ) / exterior_angle = n →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1595_159547


namespace NUMINAMATH_CALUDE_max_area_and_optimal_length_l1595_159546

/-- Represents the dimensions and cost of a rectangular house. -/
structure House where
  x : ℝ  -- Length of front wall
  y : ℝ  -- Length of side wall
  h : ℝ  -- Height of the house
  coloredSteelPrice : ℝ  -- Price per meter of colored steel
  compositeSteelPrice : ℝ  -- Price per meter of composite steel
  roofPrice : ℝ  -- Price per square meter of roof

/-- Calculates the material cost of the house. -/
def materialCost (h : House) : ℝ :=
  2 * h.x * h.coloredSteelPrice * h.h +
  2 * h.y * h.compositeSteelPrice * h.h +
  h.x * h.y * h.roofPrice

/-- Calculates the area of the house. -/
def area (h : House) : ℝ := h.x * h.y

/-- Theorem stating the maximum area and optimal front wall length. -/
theorem max_area_and_optimal_length (h : House)
    (height_constraint : h.h = 2.5)
    (colored_steel_price : h.coloredSteelPrice = 450)
    (composite_steel_price : h.compositeSteelPrice = 200)
    (roof_price : h.roofPrice = 200)
    (cost_constraint : materialCost h ≤ 32000) :
    (∃ (max_area : ℝ) (optimal_x : ℝ),
      max_area = 100 ∧
      optimal_x = 20 / 3 ∧
      area h ≤ max_area ∧
      (area h = max_area ↔ h.x = optimal_x)) := by
  sorry


end NUMINAMATH_CALUDE_max_area_and_optimal_length_l1595_159546


namespace NUMINAMATH_CALUDE_total_apples_is_36_l1595_159594

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The number of apples picked by Olivia -/
def olivia_apples : ℕ := 12

/-- The number of apples picked by Thomas -/
def thomas_apples : ℕ := 8

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples

theorem total_apples_is_36 : total_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_36_l1595_159594


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_derived_inequality_solutions_l1595_159537

def quadratic_inequality (x : ℝ) : Prop := x^2 - 3*x + 2 > 0

def solution_set (x : ℝ) : Prop := x < 1 ∨ x > 2

def derived_inequality (x m : ℝ) : Prop := x^2 - (m + 2)*x + 2*m < 0

theorem quadratic_inequality_solution :
  ∀ x, quadratic_inequality x ↔ solution_set x :=
sorry

theorem derived_inequality_solutions :
  (∀ x, ¬(derived_inequality x 2)) ∧
  (∀ m, m < 2 → ∀ x, derived_inequality x m ↔ m < x ∧ x < 2) ∧
  (∀ m, m > 2 → ∀ x, derived_inequality x m ↔ 2 < x ∧ x < m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_derived_inequality_solutions_l1595_159537


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1595_159596

theorem polynomial_remainder_theorem (x : ℝ) : 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    (∀ x, x^50 = (x^2 - 5*x + 6) * Q x + R x) ∧ 
    (∃ a b : ℝ, ∀ x, R x = a*x + b) ∧
    R x = (3^50 - 2^50)*x + (2^50 - 2*3^50 + 2*2^50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1595_159596


namespace NUMINAMATH_CALUDE_rectangles_in_grid_l1595_159542

def grid_size : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in an n x n grid -/
def num_rectangles (n : ℕ) : ℕ := (choose_two n) ^ 2

theorem rectangles_in_grid :
  num_rectangles grid_size = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangles_in_grid_l1595_159542


namespace NUMINAMATH_CALUDE_expression_equality_l1595_159557

theorem expression_equality : 
  (Real.sqrt 3 - Real.sqrt 2) * (-Real.sqrt 3 - Real.sqrt 2) + (3 + 2 * Real.sqrt 5)^2 = 28 + 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1595_159557


namespace NUMINAMATH_CALUDE_intersection_M_N_l1595_159544

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1595_159544


namespace NUMINAMATH_CALUDE_total_slices_equals_twelve_l1595_159561

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := 7

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- The total number of slices of pie served today -/
def total_slices : ℕ := lunch_slices + dinner_slices

theorem total_slices_equals_twelve : total_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_equals_twelve_l1595_159561


namespace NUMINAMATH_CALUDE_tournament_players_l1595_159516

-- Define the number of Asian players
variable (n : ℕ)

-- Define the number of European players as 2n
def european_players := 2 * n

-- Define the total number of matches
def total_matches := n * (n - 1) / 2 + (2 * n) * (2 * n - 1) / 2 + 2 * n^2

-- Define the number of matches won by Europeans
def european_wins (x : ℕ) := (2 * n) * (2 * n - 1) / 2 + x

-- Define the number of matches won by Asians
def asian_wins (x : ℕ) := n * (n - 1) / 2 + 2 * n^2 - x

-- State the theorem
theorem tournament_players :
  ∃ x : ℕ, european_wins n x = (5 / 7) * asian_wins n x ∧ n = 3 ∧ n + european_players n = 9 := by
  sorry


end NUMINAMATH_CALUDE_tournament_players_l1595_159516


namespace NUMINAMATH_CALUDE_decoration_price_increase_l1595_159593

def price_1990 : ℝ := 11500
def increase_1990_to_1996 : ℝ := 0.13
def increase_1996_to_2001 : ℝ := 0.20

def price_2001 : ℝ :=
  price_1990 * (1 + increase_1990_to_1996) * (1 + increase_1996_to_2001)

theorem decoration_price_increase : price_2001 = 15594 := by
  sorry

end NUMINAMATH_CALUDE_decoration_price_increase_l1595_159593


namespace NUMINAMATH_CALUDE_pairing_probability_l1595_159591

/-- The probability of one student being paired with another specific student
    in a class where some students are absent. -/
theorem pairing_probability
  (total_students : ℕ)
  (absent_students : ℕ)
  (h1 : total_students = 40)
  (h2 : absent_students = 5)
  (h3 : absent_students < total_students) :
  (1 : ℚ) / (total_students - absent_students - 1) = 1 / 34 :=
by sorry

end NUMINAMATH_CALUDE_pairing_probability_l1595_159591


namespace NUMINAMATH_CALUDE_total_books_is_54_l1595_159522

def darla_books : ℕ := 6

def katie_books : ℕ := darla_books / 2

def darla_katie_books : ℕ := darla_books + katie_books

def gary_books : ℕ := 5 * darla_katie_books

def total_books : ℕ := darla_books + katie_books + gary_books

theorem total_books_is_54 : total_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_54_l1595_159522


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1595_159503

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1595_159503


namespace NUMINAMATH_CALUDE_no_positive_real_roots_l1595_159512

theorem no_positive_real_roots (m : ℝ) : 
  (∀ x > 0, (x^2 + (5-2*m)*x + m-3) / (x-1) ≠ 2*x + m) ↔ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_real_roots_l1595_159512


namespace NUMINAMATH_CALUDE_x_values_l1595_159574

theorem x_values (x : ℝ) : 
  ({1, 2} ∪ {x + 1, x^2 - 4*x + 6} : Set ℝ) = {1, 2, 3} → x = 2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l1595_159574


namespace NUMINAMATH_CALUDE_probability_defective_smartphones_l1595_159553

/-- Represents the probability of selecting two defective smartphones --/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * ((defective - 1) : ℚ) / ((total - 1) : ℚ)

/-- Theorem stating the probability of selecting two defective smartphones --/
theorem probability_defective_smartphones :
  let total := 250
  let type_a_total := 100
  let type_a_defective := 30
  let type_b_total := 80
  let type_b_defective := 25
  let type_c_total := 70
  let type_c_defective := 21
  let total_defective := type_a_defective + type_b_defective + type_c_defective
  abs (probability_two_defective total total_defective - 0.0916) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_probability_defective_smartphones_l1595_159553


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1595_159525

theorem smallest_solution_of_equation :
  ∃ x : ℝ, x = -3 ∧ 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 12) ∧
    (∀ y : ℝ, (3 * y / (y - 3) + (3 * y^2 - 27) / y = 12) → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1595_159525


namespace NUMINAMATH_CALUDE_triangle_problem_l1595_159529

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  t.a = 7/2 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 3/2 * Real.sqrt 3) ∧
  (t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0) →
  -- Conclusions to prove
  (t.A = π/3) ∧ (t.b + t.c = 11/2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1595_159529


namespace NUMINAMATH_CALUDE_projectile_max_height_l1595_159558

def f (t : ℝ) : ℝ := -8 * t^2 + 64 * t + 36

theorem projectile_max_height :
  ∃ (max : ℝ), max = 164 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1595_159558


namespace NUMINAMATH_CALUDE_log_arithmetic_progression_implies_power_relation_l1595_159584

theorem log_arithmetic_progression_implies_power_relation
  (k m n x : ℝ)
  (hk : k > 0)
  (hm : m > 0)
  (hn : n > 0)
  (hx_pos : x > 0)
  (hx_neq_one : x ≠ 1)
  (h_arith_prog : 2 * (Real.log x / Real.log m) = 
                  (Real.log x / Real.log k) + (Real.log x / Real.log n)) :
  n^2 = (n*k)^(Real.log m / Real.log k) :=
by sorry

end NUMINAMATH_CALUDE_log_arithmetic_progression_implies_power_relation_l1595_159584


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l1595_159587

theorem root_sum_reciprocals (p q r s : ℂ) : 
  p^4 + 6*p^3 + 11*p^2 + 6*p + 3 = 0 →
  q^4 + 6*q^3 + 11*q^2 + 6*q + 3 = 0 →
  r^4 + 6*r^3 + 11*r^2 + 6*r + 3 = 0 →
  s^4 + 6*s^3 + 11*s^2 + 6*s + 3 = 0 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l1595_159587


namespace NUMINAMATH_CALUDE_certain_number_value_l1595_159521

theorem certain_number_value : ∃! x : ℝ,
  (28 + x + 42 + 78 + 104) / 5 = 90 ∧
  (128 + 255 + 511 + 1023 + x) / 5 = 423 ∧
  x = 198 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l1595_159521


namespace NUMINAMATH_CALUDE_factor_expression_l1595_159504

theorem factor_expression (a : ℝ) : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1595_159504


namespace NUMINAMATH_CALUDE_distance_product_theorem_l1595_159580

theorem distance_product_theorem (a b : ℝ) (θ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let point1 := (Real.sqrt (a^2 - b^2), 0)
  let point2 := (-Real.sqrt (a^2 - b^2), 0)
  let line := fun (x y : ℝ) ↦ x * Real.cos θ / a + y * Real.sin θ / b = 1
  let distance (p : ℝ × ℝ) := 
    abs (b * Real.cos θ * p.1 + a * Real.sin θ * p.2 - a * b) / 
    Real.sqrt ((b * Real.cos θ)^2 + (a * Real.sin θ)^2)
  (distance point1) * (distance point2) = b^2 := by
sorry

end NUMINAMATH_CALUDE_distance_product_theorem_l1595_159580


namespace NUMINAMATH_CALUDE_genevieve_cherries_l1595_159506

/-- The number of kilograms of cherries Genevieve bought -/
def cherries_bought : ℕ := 277

/-- The original price of cherries per kilogram in cents -/
def original_price : ℕ := 800

/-- The discount percentage on cherries -/
def discount_percentage : ℚ := 1 / 10

/-- The amount Genevieve was short in cents -/
def short_amount : ℕ := 40000

/-- The amount Genevieve had in cents -/
def genevieve_amount : ℕ := 160000

/-- Theorem stating that given the conditions, Genevieve bought 277 kilograms of cherries -/
theorem genevieve_cherries :
  let discounted_price : ℚ := original_price * (1 - discount_percentage)
  let total_price : ℕ := genevieve_amount + short_amount
  (total_price : ℚ) / discounted_price = cherries_bought := by sorry

end NUMINAMATH_CALUDE_genevieve_cherries_l1595_159506


namespace NUMINAMATH_CALUDE_polynomial_identity_l1595_159540

theorem polynomial_identity (x : ℝ) : 
  let P (x : ℝ) := (x - 1/2)^2001 + 1/2
  P x + P (1 - x) = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1595_159540


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l1595_159523

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem union_equality_iff_a_in_range (a : ℝ) : 
  A ∪ B a = A ↔ a ∈ Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l1595_159523


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l1595_159585

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (distance_per_interval : ℝ) (interval_duration : ℝ) 
  (initial_ride_duration : ℝ) (break_duration : ℝ) (final_ride_duration : ℝ) 
  (h1 : distance_per_interval = 1.5)
  (h2 : interval_duration = 7)
  (h3 : initial_ride_duration = 21)
  (h4 : break_duration = 5)
  (h5 : final_ride_duration = 14) :
  (initial_ride_duration / interval_duration) * distance_per_interval + 
  (final_ride_duration / interval_duration) * distance_per_interval = 7.5 := by
  sorry

#check suzanna_bike_ride

end NUMINAMATH_CALUDE_suzanna_bike_ride_l1595_159585


namespace NUMINAMATH_CALUDE_sequence_properties_l1595_159568

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- Definition of S_n as the sum of the first n terms of a_n -/
def S (n : ℕ) : ℝ := sorry

/-- Definition of the arithmetic sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- Definition of T_n as the sum of the first n terms of a_n * b_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties (n : ℕ) :
  (∀ k, S k + a k = 1) ∧
  (b 1 + b 2 = b 3) ∧
  (b 3 = 3) →
  (S n = 1 - (1/2)^n) ∧
  (T n = 2 - (n + 2) * (1/2)^n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1595_159568


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1595_159579

theorem sqrt_product_equality : 2 * Real.sqrt 3 * (5 * Real.sqrt 6) = 30 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1595_159579


namespace NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l1595_159581

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, a > 0 → (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime number has property P
theorem prime_has_property_P :
  ∀ p : ℕ, Prime p → has_property_P p :=
sorry

-- Define a set of composite numbers with property P
def composite_with_property_P : Set ℕ :=
  {n : ℕ | ¬Prime n ∧ has_property_P n}

-- Theorem 2: There are infinitely many composite numbers with property P
theorem infinitely_many_composite_with_property_P :
  Set.Infinite composite_with_property_P :=
sorry

end NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l1595_159581


namespace NUMINAMATH_CALUDE_radio_quiz_win_probability_l1595_159536

/-- Represents a quiz show with multiple-choice questions. -/
structure QuizShow where
  num_questions : ℕ
  num_options : ℕ
  min_correct : ℕ

/-- Calculates the probability of winning a quiz show by random guessing. -/
def win_probability (quiz : QuizShow) : ℚ :=
  sorry

/-- The specific quiz show described in the problem. -/
def radio_quiz : QuizShow :=
  { num_questions := 4
  , num_options := 4
  , min_correct := 2 }

/-- Theorem stating the probability of winning the radio quiz. -/
theorem radio_quiz_win_probability :
  win_probability radio_quiz = 121 / 256 :=
by sorry

end NUMINAMATH_CALUDE_radio_quiz_win_probability_l1595_159536


namespace NUMINAMATH_CALUDE_petrol_price_equation_l1595_159560

/-- The original price of petrol in dollars per gallon -/
def P : ℝ := sorry

/-- The equation representing the price reduction scenario -/
theorem petrol_price_equation : (300 / P + 7) * (0.85 * P) = 300 := by sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l1595_159560


namespace NUMINAMATH_CALUDE_complete_square_sum_l1595_159588

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (64 * x^2 + 96 * x - 81 = 0) ∧ 
  (a > 0) ∧
  ((a : ℝ) * x + b)^2 = c ∧
  a + b + c = 131 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1595_159588


namespace NUMINAMATH_CALUDE_infinitely_many_special_pairs_l1595_159517

theorem infinitely_many_special_pairs :
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ,
    let (a, b) := f n
    (a : ℤ) > 0 ∧ (b : ℤ) > 0 ∧
    (∃ k : ℤ, (a : ℤ) * b + 1 = k * ((a : ℤ) + b)) ∧
    (∃ m : ℤ, (a : ℤ) * b - 1 = m * ((a : ℤ) - b)) ∧
    (b : ℤ) > 1 ∧
    (a : ℤ) > (b : ℤ) * Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_special_pairs_l1595_159517


namespace NUMINAMATH_CALUDE_greatest_valid_number_l1595_159548

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / 10) * (n % 10) = 12 ∧
  (n / 10) < (n % 10)

theorem greatest_valid_number :
  ∀ m : ℕ, is_valid_number m → m ≤ 34 :=
by sorry

end NUMINAMATH_CALUDE_greatest_valid_number_l1595_159548


namespace NUMINAMATH_CALUDE_loan_split_l1595_159554

theorem loan_split (total : ℝ) (years1 rate1 years2 rate2 : ℝ) 
  (h1 : total = 2704)
  (h2 : years1 = 8)
  (h3 : rate1 = 0.03)
  (h4 : years2 = 3)
  (h5 : rate2 = 0.05)
  (h6 : ∃ x : ℝ, x * years1 * rate1 = (total - x) * years2 * rate2) :
  ∃ y : ℝ, y = total - 1664 ∧ y * years1 * rate1 = (total - y) * years2 * rate2 := by
  sorry

end NUMINAMATH_CALUDE_loan_split_l1595_159554


namespace NUMINAMATH_CALUDE_pet_store_inventory_l1595_159505

/-- Represents the number of birds of each type in a cage -/
structure BirdCage where
  parrots : ℕ
  parakeets : ℕ
  canaries : ℕ
  cockatiels : ℕ
  lovebirds : ℕ
  finches : ℕ

/-- The pet store inventory -/
def petStore : List BirdCage :=
  (List.replicate 7 ⟨3, 5, 4, 0, 0, 0⟩) ++
  (List.replicate 6 ⟨0, 0, 0, 2, 3, 1⟩) ++
  (List.replicate 2 ⟨0, 0, 0, 0, 0, 0⟩)

/-- Calculate the total number of birds of each type -/
def totalBirds (store : List BirdCage) : BirdCage :=
  store.foldl (fun acc cage =>
    ⟨acc.parrots + cage.parrots,
     acc.parakeets + cage.parakeets,
     acc.canaries + cage.canaries,
     acc.cockatiels + cage.cockatiels,
     acc.lovebirds + cage.lovebirds,
     acc.finches + cage.finches⟩)
    ⟨0, 0, 0, 0, 0, 0⟩

theorem pet_store_inventory :
  totalBirds petStore = ⟨21, 35, 28, 12, 18, 6⟩ := by
  sorry

end NUMINAMATH_CALUDE_pet_store_inventory_l1595_159505


namespace NUMINAMATH_CALUDE_bicycle_wheels_l1595_159595

theorem bicycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (total_wheels : ℕ) (tricycle_wheels : ℕ) :
  num_bicycles = 6 →
  num_tricycles = 15 →
  total_wheels = 57 →
  tricycle_wheels = 3 →
  ∃ (bicycle_wheels : ℕ), 
    bicycle_wheels = 2 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_l1595_159595


namespace NUMINAMATH_CALUDE_aquarium_length_l1595_159538

/-- The length of an aquarium given its volume, breadth, and water height -/
theorem aquarium_length (volume : ℝ) (breadth height : ℝ) (h1 : volume = 10000)
  (h2 : breadth = 20) (h3 : height = 10) : volume / (breadth * height) = 50 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_length_l1595_159538


namespace NUMINAMATH_CALUDE_area_of_EFGH_l1595_159599

/-- Represents a parallelogram with a base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- The specific parallelogram EFGH from the problem -/
def EFGH : Parallelogram := { base := 5, height := 3 }

/-- Theorem stating that the area of parallelogram EFGH is 15 square units -/
theorem area_of_EFGH : area EFGH = 15 := by sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l1595_159599


namespace NUMINAMATH_CALUDE_squares_ending_in_identical_digits_l1595_159534

def endsIn (n : ℤ) (d : ℤ) : Prop := n % (10 ^ (d.natAbs + 1)) = d

theorem squares_ending_in_identical_digits :
  (∀ n : ℤ, (endsIn n 12 ∨ endsIn n 38 ∨ endsIn n 62 ∨ endsIn n 88) → endsIn (n^2) 44) ∧
  (∀ m : ℤ, (endsIn m 038 ∨ endsIn m 462 ∨ endsIn m 538 ∨ endsIn m 962) → endsIn (m^2) 444) :=
by sorry

end NUMINAMATH_CALUDE_squares_ending_in_identical_digits_l1595_159534


namespace NUMINAMATH_CALUDE_sachins_age_l1595_159575

theorem sachins_age (sachin_age rahul_age : ℝ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age / rahul_age = 7 / 9) :
  sachin_age = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l1595_159575


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1595_159514

theorem degree_to_radian_conversion (angle_in_degrees : ℝ) :
  angle_in_degrees = 1440 →
  (angle_in_degrees * (π / 180)) = 8 * π :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1595_159514


namespace NUMINAMATH_CALUDE_all_numbers_multiple_of_three_l1595_159592

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def numbers_to_check : List ℕ := [123, 234, 345, 456, 567]

theorem all_numbers_multiple_of_three 
  (h : ∀ n : ℕ, is_multiple_of_three n ↔ is_multiple_of_three (sum_of_digits n)) :
  ∀ n ∈ numbers_to_check, is_multiple_of_three n :=
by sorry

end NUMINAMATH_CALUDE_all_numbers_multiple_of_three_l1595_159592


namespace NUMINAMATH_CALUDE_reach_50_from_49_l1595_159555

def double (n : ℕ) : ℕ := n * 2

def eraseLast (n : ℕ) : ℕ := n / 10

def canReach (start target : ℕ) : Prop :=
  ∃ (steps : ℕ), ∃ (moves : Fin steps → Bool),
    (start = target) ∨
    (∃ (intermediate : Fin (steps + 1) → ℕ),
      intermediate 0 = start ∧
      intermediate (Fin.last steps) = target ∧
      ∀ i : Fin steps,
        (moves i = true → intermediate (i.succ) = double (intermediate i)) ∧
        (moves i = false → intermediate (i.succ) = eraseLast (intermediate i)))

theorem reach_50_from_49 : canReach 49 50 := by
  sorry

end NUMINAMATH_CALUDE_reach_50_from_49_l1595_159555


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_is_eight_l1595_159569

def sixDigitNumber (x y : ℕ) : ℕ := 123400 + 10 * x + y

theorem sum_of_last_two_digits_is_eight 
  (x y : ℕ) 
  (h1 : x < 10 ∧ y < 10) 
  (h2 : sixDigitNumber x y % 8 = 0) 
  (h3 : sixDigitNumber x y % 9 = 0) :
  x + y = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_is_eight_l1595_159569


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1595_159590

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}

theorem complement_of_M_in_U : 
  (U \ M) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1595_159590


namespace NUMINAMATH_CALUDE_douglas_county_x_votes_l1595_159509

/-- Represents the percentage of votes Douglas won in county X -/
def douglas_x_percent : ℝ := 64

/-- Represents the percentage of votes Douglas won in county Y -/
def douglas_y_percent : ℝ := 46

/-- Represents the ratio of voters in county X to county Y -/
def county_ratio : ℝ := 2

/-- Represents the total percentage of votes Douglas won in both counties -/
def total_percent : ℝ := 58

theorem douglas_county_x_votes :
  douglas_x_percent * county_ratio + douglas_y_percent = total_percent * (county_ratio + 1) :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_x_votes_l1595_159509


namespace NUMINAMATH_CALUDE_no_periodic_sequence_for_factorial_digits_l1595_159510

/-- a_n is the first non-zero digit from the right in the decimal representation of n! -/
def first_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The theorem states that for all natural numbers N, the sequence of first non-zero digits
    from the right in the decimal representation of (N+k)! for k ≥ 1 is not periodic. -/
theorem no_periodic_sequence_for_factorial_digits :
  ∀ N : ℕ, ¬ ∃ T : ℕ+, ∀ k : ℕ, first_nonzero_digit (N + k + 1) = first_nonzero_digit (N + k + 1 + T) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sequence_for_factorial_digits_l1595_159510
