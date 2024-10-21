import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_properties_l987_98791

/-- Represents a triangular prism with specific properties -/
structure TriangularPrism where
  /-- All edges of the prism have length 2 -/
  edge_length : ℝ
  edge_length_eq : edge_length = 2
  /-- One lateral edge forms 60° angles with adjacent base sides -/
  lateral_angle : ℝ
  lateral_angle_eq : lateral_angle = Real.pi / 3

/-- The volume of the triangular prism -/
noncomputable def volume (prism : TriangularPrism) : ℝ := 2 * Real.sqrt 2

/-- The total surface area of the triangular prism -/
noncomputable def total_surface_area (prism : TriangularPrism) : ℝ := 4 + 6 * Real.sqrt 3

/-- Theorem stating the volume and surface area of the specific triangular prism -/
theorem triangular_prism_properties (prism : TriangularPrism) :
  volume prism = 2 * Real.sqrt 2 ∧ total_surface_area prism = 4 + 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_properties_l987_98791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l987_98742

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  apply Set.empty_subset


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l987_98742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l987_98721

/-- The focus of a parabola given by y = ax^2 + bx + c --/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  let f := 1 / (4 * a)
  (h, k + f)

/-- Theorem: The focus of the parabola y = 2x^2 + 4x - 1 is at (-1, -23/8) --/
theorem focus_of_specific_parabola :
  parabola_focus 2 4 (-1) = (-1, -23/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l987_98721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equality_l987_98748

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log n 10 + 1

/-- Theorem: For natural numbers n and k, if n^n has k digits and k^k has n digits,
    then n = k and n is in the set {1, 2, 3, 4, 5, 6, 7, 8, 9} -/
theorem digit_equality (n k : ℕ) (h1 : n ≠ 0) (h2 : k ≠ 0) :
  numDigits (n^n) = k ∧ numDigits (k^k) = n →
  n = k ∧ n ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) := by
  sorry

#check digit_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equality_l987_98748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_equal_g_l987_98747

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) * Real.sqrt (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)

-- Define the domains of f and g
def domain_f : Set ℝ := {x | x ≥ 1}
def domain_g : Set ℝ := {x | x ≥ 1 ∨ x ≤ -1}

-- Theorem stating that f and g are not the same function
theorem f_not_equal_g : ¬(f = g) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_equal_g_l987_98747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fourth_six_l987_98704

/-- Represents a six-sided die -/
structure Die where
  /-- Probability of rolling a six -/
  prob_six : ℝ
  /-- Probability of rolling any other number -/
  prob_other : ℝ
  /-- Sum of probabilities is 1 -/
  prob_sum : prob_six + 5 * prob_other = 1

/-- Fair die -/
noncomputable def fair_die : Die where
  prob_six := 1/6
  prob_other := 1/6
  prob_sum := by norm_num

/-- Biased die -/
noncomputable def biased_die : Die where
  prob_six := 3/4
  prob_other := 1/20
  prob_sum := by norm_num

theorem probability_fourth_six (p : ℝ) 
  (h_p : p = 774/1292) : 
  p = (1/2 * (fair_die.prob_six ^ 3 * fair_die.prob_six + 
      biased_die.prob_six ^ 3 * biased_die.prob_six)) / 
      (1/2 * (fair_die.prob_six ^ 3 + biased_die.prob_six ^ 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fourth_six_l987_98704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_F_l987_98717

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x^2

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def F (x : ℝ) : ℝ := if f x ≥ g x then f x else g x

theorem max_value_of_F :
  ∃ (M : ℝ), M = 7/9 ∧ ∀ (x : ℝ), F x ≤ M ∧ ∃ (y : ℝ), F y = M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_F_l987_98717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l987_98765

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (p : ℝ) (x : ℝ) : ℝ := p * x - p / x - 2 * Real.log x

/-- The function g(x) defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.exp 1 / x

/-- Part I of the problem -/
theorem part_one (p : ℝ) :
  (∀ x > 0, Monotone (f p)) → p ≥ 1 := by sorry

/-- Part II of the problem -/
theorem part_two (p : ℝ) :
  p > 0 → (∃ x ∈ Set.Icc 1 (Real.exp 1), f p x > g x) → p > 4 * (Real.exp 1) / ((Real.exp 1)^2 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l987_98765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_theorem_l987_98770

/-- The force of water pressure on a vertical semicircular wall -/
noncomputable def water_pressure_force (R : ℝ) (γ : ℝ) : ℝ :=
  (2 * γ * R^3) / 3

/-- Theorem: The force of water pressure on a vertical semicircular wall -/
theorem water_pressure_force_theorem (R : ℝ) (γ : ℝ) 
  (h_R_pos : R > 0) (h_γ_pos : γ > 0) :
  water_pressure_force R γ = (2 * γ * R^3) / 3 := by
  rfl

#check water_pressure_force_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_theorem_l987_98770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_after_wednesday_is_friday_l987_98763

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, DecidableEq

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day of the week that occurs 'n' days after a given day -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

theorem hundred_days_after_wednesday_is_friday :
  dayAfter DayOfWeek.Wednesday 100 = DayOfWeek.Friday := by
  sorry

#eval dayAfter DayOfWeek.Wednesday 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_after_wednesday_is_friday_l987_98763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l987_98736

/-- A sequence defined by the roots of a quadratic equation -/
def sequence_a (n : ℕ) : ℝ := sorry

/-- The b_n term in the quadratic equation -/
def b (n : ℕ) : ℝ := sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Main theorem encompassing all parts of the problem -/
theorem sequence_properties :
  (∃ (r : ℝ), ∀ (n : ℕ), sequence_a (n + 1) - (1/3) * 2^(n + 1) = r * (sequence_a n - (1/3) * 2^n)) ∧ 
  (∀ (n : ℕ), S n = (1/3) * (2^(n+1) - 2 - ((-1)^n - 1)/2)) ∧
  (∃ (lambda : ℝ), lambda < 1 ∧ ∀ (n : ℕ), n ≥ 1 → b n - lambda * S n > 0) :=
by
  sorry

/-- The sequence satisfies the given quadratic equation -/
axiom sequence_property (n : ℕ) (h : n ≥ 1) : 
  (sequence_a n)^2 - 2^n * sequence_a n + b n = 0 ∧
  (sequence_a (n+1))^2 - 2^n * sequence_a (n+1) + b n = 0

/-- The first term of the sequence is 1 -/
axiom sequence_start : sequence_a 1 = 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l987_98736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_predicted_y_value_l987_98749

-- Define the data types
structure DataPoint where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the given data set
noncomputable def data : List DataPoint := [
  ⟨1, Real.exp 1, 1⟩,
  ⟨2, Real.exp 3, 3⟩,
  ⟨3, Real.exp 4, 4⟩,
  ⟨4, Real.exp 6, 6⟩
]

-- Define the regression equation
noncomputable def regression_equation (b : ℝ) (x : ℝ) : ℝ := Real.exp (b * x - 0.5)

-- Define the relationship between y and z
noncomputable def z_equation (y : ℝ) : ℝ := Real.log y

-- Define the linear equation for z
def z_linear_equation (b : ℝ) (x : ℝ) : ℝ := b * x - 0.5

-- Theorem statement
theorem predicted_y_value (b : ℝ) (h1 : ∀ d ∈ data, d.y = regression_equation b d.x) 
  (h2 : ∀ d ∈ data, d.z = z_equation d.y) 
  (h3 : ∀ d ∈ data, d.z = z_linear_equation b d.x) 
  (h4 : b = 1.6) :
  regression_equation b 5 = Real.exp (15 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_predicted_y_value_l987_98749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_l987_98720

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x - 2)

noncomputable def g (x : ℝ) : ℝ := f (x + 3)

def b : ℝ := -4

theorem g_equals_inverse : 
  ∀ x : ℝ, g x = g⁻¹ x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_l987_98720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l987_98703

-- Define the sequence a_n
noncomputable def a (n : ℕ+) : ℝ := n + 1

-- Define the sum S_n
noncomputable def S (n : ℕ+) (t : ℝ) : ℝ := 
  (1 - t) / 2 * n.val^2 + (3 - t) / 2 * n.val

-- State the theorem
theorem range_of_t (t : ℝ) : 
  (∀ n : ℕ+, S n t ≤ S 10 t) ↔ 12/11 ≤ t ∧ t ≤ 11/10 := by
  sorry

#check range_of_t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l987_98703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l987_98780

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

-- Theorem statement
theorem solve_for_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l987_98780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_primes_in_product_l987_98764

theorem distinct_primes_in_product : 
  ∃ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) ∧ 
  (S.prod id) = 77 * 81 * 85 * 87 ∧ 
  S.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_primes_in_product_l987_98764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_implies_m_zero_z₂_fourth_quadrant_implies_m_range_l987_98775

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m * (m - 1) + (m - 1) * Complex.I
def z₂ (m : ℝ) : ℂ := (m + 1) + (m^2 - 1) * Complex.I

-- Theorem for the first part
theorem z₁_pure_imaginary_implies_m_zero :
  ∀ m : ℝ, (z₁ m).re = 0 ∧ (z₁ m).im ≠ 0 → m = 0 := by
  sorry

-- Theorem for the second part
theorem z₂_fourth_quadrant_implies_m_range :
  ∀ m : ℝ, (z₂ m).re > 0 ∧ (z₂ m).im < 0 → -1 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_implies_m_zero_z₂_fourth_quadrant_implies_m_range_l987_98775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l987_98788

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  sum_n seq 13 = 52 → seq.a 4 + seq.a 8 + seq.a 9 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l987_98788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l987_98789

/-- Represents a leg of the journey -/
structure JourneyLeg where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a journey leg -/
noncomputable def timeTaken (leg : JourneyLeg) : ℝ :=
  leg.distance / leg.speed

/-- Represents the entire journey -/
structure Journey where
  legs : List JourneyLeg
  returnTimeFactor : ℝ

/-- Calculates the total distance of a journey -/
noncomputable def totalDistance (journey : Journey) : ℝ :=
  (journey.legs.map (·.distance)).sum * 2

/-- Calculates the total time of a journey -/
noncomputable def totalTime (journey : Journey) : ℝ :=
  ((journey.legs.map timeTaken).sum) * (1 + journey.returnTimeFactor)

/-- Calculates the average speed of a journey -/
noncomputable def averageSpeed (journey : Journey) : ℝ :=
  totalDistance journey / totalTime journey

theorem round_trip_average_speed :
  let journey : Journey := {
    legs := [
      { distance := 120, speed := 60 },
      { distance := 80, speed := 40 },
      { distance := 100, speed := 50 }
    ],
    returnTimeFactor := 1
  }
  ∃ ε > 0, |averageSpeed journey - 100/3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l987_98789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_l987_98710

theorem stratified_sample_size (population_ratio_A population_ratio_B population_ratio_C : ℕ)
  (sample_size_A : ℕ) :
  population_ratio_A = 3 →
  population_ratio_B = 4 →
  population_ratio_C = 7 →
  sample_size_A = 15 →
  (population_ratio_A + population_ratio_B + population_ratio_C) * sample_size_A / population_ratio_A = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_l987_98710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l987_98715

theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ = Real.log x₀ ∧ k = 1 / x₀) → k = 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l987_98715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_function_value_l987_98705

/-- Given a function f(x) = ax² + 1, if the integral of f from 0 to 1 equals f(x₀)
    and x₀ is in the interval [0,1], then x₀ equals √3/3. -/
theorem integral_equals_function_value (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 1
  ∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 1 ∧
    (∫ x in Set.Icc 0 1, f x) = f x₀ →
    x₀ = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_function_value_l987_98705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l987_98734

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 2}

-- Part 1
theorem part_one (a : ℝ) (h : a = 1) :
  A ∪ B a = {x | -2 ≤ x ∧ x ≤ 6} ∧
  A ∩ (Set.univ \ B a) = {x | -2 ≤ x ∧ x ≤ 0} ∪ {x | 5 ≤ x ∧ x ≤ 6} := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  A ∩ B a = B a ↔ a ∈ Set.Iic (-3/2) ∪ Set.Icc (-1) (4/3) := by sorry

-- Here, Set.Iic is the set of all reals less than or equal to a given value,
-- and Set.Icc is the closed interval [a, b].

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l987_98734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_way_split_l987_98790

/-- Definition of similar sizes -/
def similar (a b : ℝ) : Prop := a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a

/-- Definition of a valid split -/
def validSplit (x y z : ℝ) : Prop := similar x y ∧ similar y z ∧ similar x z

/-- Theorem: It's impossible to split a pile into three similar parts -/
theorem no_three_way_split (total : ℝ) (h : total > 0) :
  ¬∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = total ∧ validSplit x y z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_way_split_l987_98790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_cost_in_cents_l987_98776

-- Define the cost of a pound of rice in dollars
noncomputable def rice_cost : ℚ := 36/100

-- Define the relationship between eggs and rice
noncomputable def dozen_eggs_cost : ℚ := rice_cost

-- Define the relationship between kerosene and eggs
noncomputable def half_liter_kerosene_cost : ℚ := (dozen_eggs_cost / 12) * 8

-- Define the conversion rate from dollars to cents
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

-- Theorem statement
theorem kerosene_cost_in_cents : 
  dollars_to_cents (2 * half_liter_kerosene_cost) = 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_cost_in_cents_l987_98776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_and_polygon_impossibility_l987_98702

noncomputable def equation (x : ℝ) : Prop := 42 - 3 * x = 18

noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

theorem equation_solution_and_polygon_impossibility :
  -- Part 1: The solution to the equation is 8
  ∃ (x : ℝ), equation x ∧ x = 8 ∧
  -- Part 2: 8 cannot be an interior angle of any regular polygon
  ∀ (n : ℕ), n ≥ 3 → interior_angle n ≠ 8 := by
  sorry

#check equation_solution_and_polygon_impossibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_and_polygon_impossibility_l987_98702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_is_correct_l987_98796

-- Define the points A and B
noncomputable def A : ℝ × ℝ := (-1/2, 0)
noncomputable def B : ℝ × ℝ := (0, 1)

-- Define the reflection of point A across the y-axis
noncomputable def A' : ℝ × ℝ := (1/2, 0)

-- Define the equation of the reflected light ray
def reflected_ray_equation (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement
theorem reflected_ray_equation_is_correct :
  reflected_ray_equation A'.1 A'.2 ∧ reflected_ray_equation B.1 B.2 :=
by
  sorry

#check reflected_ray_equation_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_is_correct_l987_98796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_period_with_multiplier_cos_3x_period_l987_98750

/-- The period of cosine function with a frequency multiplier -/
theorem cos_period_with_multiplier (multiplier : ℝ) (multiplier_pos : multiplier > 0) :
  let f := fun x => Real.cos (multiplier * x)
  ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  sorry

/-- The period of y = cos(3x) is 2π/3 -/
theorem cos_3x_period :
  let f := fun x => Real.cos (3 * x)
  ∃ T > 0, T = 2 * Real.pi / 3 ∧ ∀ x, f (x + T) = f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_period_with_multiplier_cos_3x_period_l987_98750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekend_rental_miles_l987_98722

/-- Calculates the number of miles driven for a truck rental --/
def miles_driven (base_fee : ℚ) (total_paid : ℚ) : ℕ :=
  let tax_rate : ℚ := 1 / 10
  let per_mile_fee : ℚ := 1 / 4
  let taxed_base_fee : ℚ := base_fee * (1 + tax_rate)
  let miles_fee : ℚ := total_paid - taxed_base_fee
  (miles_fee / per_mile_fee).floor.toNat

/-- Theorem stating the number of miles driven for a weekend rental --/
theorem weekend_rental_miles :
  miles_driven (2499 / 100) (9574 / 100) = 273 := by
  sorry

#eval miles_driven (2499 / 100) (9574 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekend_rental_miles_l987_98722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l987_98716

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 6 ∨ x > 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l987_98716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_expression_l987_98754

theorem undefined_expression (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 125 = 0) ↔ (x = 5) := by
  sorry

#check undefined_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_expression_l987_98754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l987_98741

-- Define the function f(x) = x^(-1)
noncomputable def f (x : ℝ) : ℝ := x⁻¹

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f y < f x) := by
  sorry

-- You can add more specific lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l987_98741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l987_98700

-- Define the ellipses
noncomputable def C1 (x y : ℝ) : Prop := x^2 + 4*y^2 = 1

noncomputable def C2 (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the eccentricity of C2
noncomputable def e : ℝ := Real.sqrt 3 / 2

-- Define the relation between T, M, and N
def T_relation (xT yT xM yM xN yN : ℝ) : Prop :=
  xT = xM - xN + 2*xM + xN ∧ yT = yM - yN + 2*yM + yN

-- Define the slope product condition
def slope_product (xM yM xN yN : ℝ) : Prop :=
  (yM * yN) / (xM * xN) = -1/4

-- Main theorem
theorem ellipse_problem :
  ∀ (xT yT xM yM xN yN : ℝ),
  C2 xM yM → C2 xN yN →
  T_relation xT yT xM yM xN yN →
  slope_product xM yM xN yN →
  (∃ (xA yA xB yB : ℝ),
    xT^2/20 + yT^2/5 = 1 ∧
    Real.sqrt ((xT - xA)^2 + (yT - yA)^2) + Real.sqrt ((xT - xB)^2 + (yT - yB)^2) = 4 * Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l987_98700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l987_98745

/-- The volume of a cone with given conditions -/
theorem cone_volume (r h l : ℝ) : 
  r = 1 →  -- base radius is 1
  l = 4 →  -- slant height is 4 (derived from 90° sector)
  h = Real.sqrt (l^2 - r^2) →  -- height calculation
  (1/3 : ℝ) * Real.pi * r^2 * h = (Real.sqrt 15 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l987_98745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l987_98711

/-- Predicate to define what it means to be a directrix of a parabola -/
def is_directrix_of (d : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), parabola x y → 
    ∃ (f : ℝ), (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
      (p.1 - 0)^2 + (p.2 - f)^2 = (p.2 - d)^2)

/-- The directrix of a parabola y = -1/4 * x^2 is y = 1/16 -/
theorem parabola_directrix : 
  ∃ (d : ℝ), d = 1/16 ∧ is_directrix_of d (fun x y => y = -1/4 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l987_98711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_distance_l987_98761

-- Define the complex number ζ = e^(πi/4)
noncomputable def ζ : ℂ := Complex.exp (Complex.I * Real.pi / 4)

-- Define the position of Q₈ as a complex number
noncomputable def Q₈ : ℂ := 1 + 2*ζ + 3*ζ^2 + 4*ζ^3 + 5*ζ^4 + 6*ζ^5 + 7*ζ^6 + 8*ζ^7

-- State the theorem
theorem bug_distance : Complex.abs Q₈ = 3.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_distance_l987_98761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_extrema_on_interval_l987_98718

noncomputable def F (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 8*x

theorem F_extrema_on_interval :
  let a := 1
  let b := 3
  (∀ x ∈ Set.Icc a b, F x ≤ F b) ∧
  (∀ x ∈ Set.Icc a b, F ((a + b) / 2) ≤ F x) ∧
  F b = -6 ∧
  F ((a + b) / 2) = -28/3 := by
  sorry

#check F_extrema_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_extrema_on_interval_l987_98718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_course_distance_l987_98731

/-- Represents a race course with different sections -/
structure Course where
  totalDistance : ℝ
  flatBeforeSlope : ℝ
  flatAfterSlope : ℝ
  slopeType : Bool  -- true for uphill, false for downhill

/-- Represents the car's speed characteristics -/
structure CarSpeed where
  initialSpeed : ℝ
  uphillFactor : ℝ
  downhillFactor : ℝ

/-- Calculates the time taken to complete a course -/
noncomputable def courseTime (c : Course) (s : CarSpeed) : ℝ :=
  let flatTime := (c.flatBeforeSlope + c.flatAfterSlope) / s.initialSpeed
  let slopeTime := (c.totalDistance - c.flatBeforeSlope - c.flatAfterSlope) / 
    (if c.slopeType then s.initialSpeed * s.uphillFactor else s.initialSpeed * s.downhillFactor)
  flatTime + slopeTime

theorem race_course_distance 
  (c1 c2 : Course) 
  (s : CarSpeed) :
  c1.totalDistance = c2.totalDistance ∧
  c1.flatBeforeSlope = 26 ∧
  c1.flatAfterSlope = 4 ∧
  c1.slopeType = true ∧
  c2.flatBeforeSlope = 4 ∧
  c2.flatAfterSlope = 26 ∧
  c2.slopeType = false ∧
  s.uphillFactor = 0.75 ∧
  s.downhillFactor = 1.25 ∧
  courseTime c1 s = courseTime c2 { s with initialSpeed := s.initialSpeed * (5/6) } →
  c1.totalDistance = 92 := by
  sorry

#eval "Race course distance theorem defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_course_distance_l987_98731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_eight_factors_l987_98740

/-- A function that returns the number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A function that checks if a number has exactly eight distinct positive factors -/
def has_eight_factors (n : ℕ) : Prop :=
  num_factors n = 8

theorem least_number_with_eight_factors :
  ∃ (n : ℕ), n > 0 ∧ has_eight_factors n ∧ ∀ (m : ℕ), 0 < m → m < n → ¬has_eight_factors m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_eight_factors_l987_98740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l987_98708

def t : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 2 => if Even (n + 2) then 1 + t ((n + 2) / 2) else 1 / t (n + 1)

theorem sequence_value (n : ℕ) (h : t n = 19 / 87) : n = 1905 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l987_98708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_no_parallel_line_l987_98743

/-- An ellipse centered at the origin, passing through (2,3), with right focus (2,0) -/
structure Ellipse where
  equation : ℝ → ℝ → Prop
  passes_through : equation 2 3
  right_focus : equation 2 0

/-- The line through two points -/
def line_through (x₁ y₁ x₂ y₂ : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- Distance between a point and a line -/
noncomputable def distance_point_line (x₀ y₀ : ℝ) (l : ℝ → ℝ → Prop) : ℝ := sorry

theorem ellipse_equation_and_no_parallel_line (C : Ellipse) :
  (C.equation = λ x y ↦ x^2/16 + y^2/12 = 1) ∧
  ¬ ∃ (a b : ℝ), 
    (∀ x y, line_through 0 0 2 3 x y ↔ y = (3/2) * x) ∧
    (∀ x y, line_through a b 0 0 x y ↔ y = (3/2) * x + (b - (3/2)*a)) ∧
    (∃ x y, C.equation x y ∧ line_through a b 0 0 x y) ∧
    (distance_point_line a b (line_through 0 0 2 3) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_no_parallel_line_l987_98743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_C_l987_98706

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Statement A: If A < B, then sin A < sin B --/
theorem statement_A (t : Triangle) (h : t.A < t.B) : Real.sin t.A < Real.sin t.B := by
  sorry

/-- Statement C: If a/cos A = b/sin B, then A = 45° --/
theorem statement_C (t : Triangle) (h : t.a / Real.cos t.A = t.b / Real.sin t.B) : t.A = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_C_l987_98706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_M_l987_98701

def M : Finset ℕ := {0, 1, 2}

theorem number_of_subsets_of_M : (Finset.powerset M).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_M_l987_98701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_vector_l987_98723

/-- The line equation y = (3x - 5)/4 -/
def line_equation (x y : ℝ) : Prop := y = (3 * x - 5) / 4

/-- The parameterization of the line -/
def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_parameterization_vector :
  ∃ (d : ℝ × ℝ),
    let v : ℝ × ℝ := (3, 1)
    ∀ (x y t : ℝ),
      x ≥ 3 →
      line_equation x y →
      parameterization v d t = (x, y) →
      distance (x, y) v = t →
      d = (4/5, 3/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_vector_l987_98723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l987_98712

/-- The plane equation x + y + z = 10 -/
def plane_equation (p : ℝ × ℝ × ℝ) : Prop :=
  p.1 + p.2.1 + p.2.2 = 10

/-- Point A -/
noncomputable def A : ℝ × ℝ × ℝ := (-3, 9, 11)

/-- Point C -/
noncomputable def C : ℝ × ℝ × ℝ := (4, 7, 10)

/-- Point B (to be proved) -/
noncomputable def B : ℝ × ℝ × ℝ := (-5/3, 16/3, 25/3)

/-- The normal vector of the plane -/
def normal_vector : ℝ × ℝ × ℝ := (1, 1, 1)

theorem light_reflection :
  plane_equation B ∧
  ∃ (t : ℝ), B = (A.1 + t * normal_vector.1, A.2.1 + t * normal_vector.2.1, A.2.2 + t * normal_vector.2.2) ∧
  ∃ (s : ℝ), C = (B.1 + s * (C.1 - B.1), B.2.1 + s * (C.2.1 - B.2.1), B.2.2 + s * (C.2.2 - B.2.2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l987_98712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_lengths_theorem_l987_98755

-- Define the areas of the shapes
def square1_area : ℝ := 25
def square2_area : ℝ := 64
noncomputable def circle_area : ℝ := 36 * Real.pi

-- Define the side lengths and radius
noncomputable def square1_side : ℝ := Real.sqrt square1_area
noncomputable def square2_side : ℝ := Real.sqrt square2_area
noncomputable def circle_radius : ℝ := Real.sqrt (circle_area / Real.pi)

-- State the theorem
theorem average_lengths_theorem :
  (square1_side + square2_side + circle_radius) / 3 = 19 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_lengths_theorem_l987_98755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossing_time_approx_l987_98777

/-- Represents the time it takes for a man to cross a railway intersection with three trains approaching. -/
noncomputable def crossing_time (man_speed : ℝ) (train_a_length train_a_speed : ℝ) (train_b_length train_b_speed : ℝ) (train_c_length train_c_speed : ℝ) (track_width : ℝ) (waiting_time : ℝ) : ℝ :=
  let man_speed_ms := man_speed * 1000 / 3600
  let train_a_speed_ms := train_a_speed * 1000 / 3600
  let train_b_speed_ms := train_b_speed * 1000 / 3600
  let train_c_speed_ms := train_c_speed * 1000 / 3600
  let train_a_pass_time := train_a_length / train_a_speed_ms
  let train_b_pass_time := train_b_length / train_b_speed_ms
  let train_c_pass_time := train_c_length / train_c_speed_ms
  let man_cross_time := track_width / man_speed_ms
  train_a_pass_time + waiting_time + man_cross_time +
  train_b_pass_time + waiting_time + man_cross_time +
  train_c_pass_time + man_cross_time

/-- The total time for a man to cross a railway intersection with three trains approaching is approximately 90.74 seconds. -/
theorem crossing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |crossing_time 3 250 63 350 80 150 40 2 20 - 90.74| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossing_time_approx_l987_98777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniqueFunction_l987_98729

open Real

noncomputable def f₁ (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)
noncomputable def f₂ (x : ℝ) : ℝ := 2 * sin (2 * x - π / 6)
noncomputable def f₃ (x : ℝ) : ℝ := 2 * sin (x / 2 + π / 3)
noncomputable def f₄ (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

-- Define the period of a function
def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Define symmetry about a vertical line
def isSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Main theorem
theorem uniqueFunction :
  (hasPeriod f₂ π ∧ isSymmetricAbout f₂ (π/3)) ∧
  (¬(hasPeriod f₁ π ∧ isSymmetricAbout f₁ (π/3))) ∧
  (¬(hasPeriod f₃ π ∧ isSymmetricAbout f₃ (π/3))) ∧
  (¬(hasPeriod f₄ π ∧ isSymmetricAbout f₄ (π/3))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniqueFunction_l987_98729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_count_l987_98713

/-- Represents a group of dogs and people -/
structure AnimalGroup where
  dogs : ℕ
  people : ℕ

/-- The total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 4 * g.dogs + 2 * g.people

/-- The total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.dogs + g.people

/-- Theorem stating that the number of dogs is 14 given the conditions -/
theorem dogs_count (g : AnimalGroup) : 
  totalLegs g = 2 * totalHeads g + 28 → g.dogs = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_count_l987_98713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_factors_to_remove_for_2_l987_98782

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def endsWith2 (n : ℕ) : Prop := n % 10 = 2

def factorsOf5 (n : ℕ) : ℕ := (n / 5) + (n / 25)

def factorsToRemove (n : ℕ) : ℕ := factorsOf5 n + 1

theorem min_factors_to_remove_for_2 :
  ∃ (S : Finset ℕ),
    S.card = factorsToRemove 99 ∧
    endsWith2 (factorial 99 / S.prod id) ∧
    ∀ (T : Finset ℕ), T.card < factorsToRemove 99 →
      ¬ endsWith2 (factorial 99 / T.prod id) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_factors_to_remove_for_2_l987_98782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l987_98730

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x - Real.log x

-- State the theorem
theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂)
  (h₄ : f a x₁ = -3) (h₅ : f a x₂ = -3) :
  ((-Real.exp 2 < a) ∧ (a < 0)) ∧
  (a = -2 → x₁ + x₂ > 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l987_98730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l987_98799

-- Define the ellipse
def ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 25) + (P.2^2 / 16) = 1

-- Define the first circle
def circle1 (M : ℝ × ℝ) : Prop :=
  (M.1 + 3)^2 + M.2^2 = 1

-- Define the second circle
def circle2 (N : ℝ × ℝ) : Prop :=
  (N.1 - 3)^2 + N.2^2 = 4

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem min_distance_sum (P M N : ℝ × ℝ) 
  (hP : ellipse P) (hM : circle1 M) (hN : circle2 N) :
  ∃ (min_val : ℝ), min_val = 7 ∧ 
  ∀ (M' N' : ℝ × ℝ), circle1 M' → circle2 N' → 
  distance P M' + distance P N' ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l987_98799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l987_98725

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x else 2 * x^2

theorem range_of_f :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ∈ Set.Icc 0 8) ∧
  (∀ y ∈ Set.Icc 0 8, ∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l987_98725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_prime_factors_of_x_eq_five_l987_98735

/-- The probability of getting heads in a single toss of an unbiased coin -/
noncomputable def p : ℝ := 1 / 2

/-- The number of consecutive heads already obtained -/
def n : ℕ := 28

/-- The target number of consecutive heads -/
def target : ℕ := 60

/-- The expected number of additional tosses to get 'target' consecutive heads,
    given that the last 'n' consecutive flips have all resulted in heads -/
noncomputable def x : ℝ :=
  2^(target - n) - 1 + (1 - 2^(-(target - n : ℤ))) * (2^(target - 1) + 1)

/-- The sum of all distinct prime factors of a natural number -/
def sum_of_distinct_prime_factors (m : ℕ) : ℕ :=
  (Nat.factors m).toFinset.sum id

theorem sum_of_prime_factors_of_x_eq_five :
  sum_of_distinct_prime_factors (Int.floor x).toNat = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_prime_factors_of_x_eq_five_l987_98735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_theorem_l987_98794

-- Define the basic structures
structure Angle where

structure Circle where

structure Line where

structure Point where

-- Define the setup
def inscribed_in_angle (c : Circle) (a : Angle) : Prop := sorry

def intersects_at (l : Line) (p : Point) : Prop := sorry

def order_on_line (l : Line) (p1 p2 p3 p4 p5 p6 : Point) : Prop := sorry

def segment_length (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem inscribed_circles_theorem 
  (ω Ω : Circle) (α : Angle) (l : Line) 
  (A B C D E F : Point) :
  inscribed_in_angle ω α →
  inscribed_in_angle Ω α →
  intersects_at l A →
  intersects_at l F →
  intersects_at l B →
  intersects_at l C →
  intersects_at l D →
  intersects_at l E →
  order_on_line l A B C D E F →
  segment_length B C = segment_length D E →
  segment_length A B = segment_length E F := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_theorem_l987_98794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l987_98772

-- Define the propositions
def p : Prop := False
def q : Prop := True

-- Define the function for the smallest positive period of |cos x|
noncomputable def f (x : ℝ) : ℝ := |Real.cos x|

-- Define the function for x³ + sin x
noncomputable def g (x : ℝ) : ℝ := x^3 + Real.sin x

-- Theorem statement
theorem correct_proposition :
  (p ∨ q) ∧
  (∃ T : ℝ, T > 0 ∧ T < 2*Real.pi ∧ ∀ x, f (x + T) = f x) ∧
  (∀ x, g (-x) = -g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l987_98772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l987_98714

noncomputable def f (x : ℝ) := (1/2) * Real.sin (2*x) + (Real.sin x) ^ 2

theorem range_of_f :
  Set.range f = Set.Icc (-(Real.sqrt 2)/2 + 1/2) ((Real.sqrt 2)/2 + 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l987_98714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l987_98738

def A (b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, 3, b; 0, 1, 5; 0, 0, 1]

theorem matrix_power_sum (b : ℝ) (m : ℕ) : 
  (A b) ^ m = !![1, 27, 3005; 0, 1, 45; 0, 0, 1] → b + m = 283 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l987_98738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l987_98727

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log10 (x + 1) / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x > -1 ∧ x ≠ 2}

-- Theorem stating that the domain of f is (-1, 2) ∪ (2, +∞)
theorem domain_of_f : domain_f = Set.Ioo (-1) 2 ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l987_98727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l987_98737

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define point M
def point_M (p : ℝ) : ℝ × ℝ := (p, 0)

-- Define a line passing through M
def line_through_M (p m : ℝ) (y : ℝ) : ℝ := m * y + p

-- Define the intersection points A and B
def intersection_points (p m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the vector condition
def vector_condition (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  (M.1 - A.1, M.2 - A.2) = (2 * (B.1 - M.1), 2 * (B.2 - M.2))

-- Main theorem
theorem parabola_intersection_ratio (p m : ℝ) :
  let F := focus p
  let M := point_M p
  let (A, B) := intersection_points p m
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧ vector_condition A B M →
  let AF := Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2)
  let BF := Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)
  AF / BF = 5/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l987_98737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_problem_l987_98707

theorem triangle_abc_problem (A B C : ℝ) (a b c : ℝ) :
  (Real.sin (A - π/6) - Real.cos (A + 5*π/3) = Real.sqrt 2/2) →
  (a = Real.sqrt 5) →
  (Real.sin B ^ 2 + Real.cos (2 * C) = 1) →
  (A = 3*π/4 ∧ b = Real.sqrt 2 ∧ c = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_problem_l987_98707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l987_98759

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := x^2 / 4 + 2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the function to be minimized
noncomputable def f (x : ℝ) : ℝ :=
  parabola x + distance x (parabola x) 4 0

-- Theorem statement
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 6 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l987_98759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_when_perimeter_minimized_l987_98767

/-- A spatial quadrilateral with a rectangular base -/
structure SpatialQuadrilateral where
  -- Base rectangle
  AB : ℝ
  AD : ℝ
  -- Height
  PA : ℝ
  -- Points on edges
  M : ℝ  -- Position on AB
  N : ℝ  -- Position on BC

/-- The dihedral angle of a spatial quadrilateral -/
noncomputable def dihedralAngle (q : SpatialQuadrilateral) : ℝ :=
  sorry

/-- The perimeter of the spatial quadrilateral PMND -/
noncomputable def perimeter (q : SpatialQuadrilateral) : ℝ :=
  sorry

theorem dihedral_angle_cosine_when_perimeter_minimized 
  (q : SpatialQuadrilateral) 
  (h1 : q.AB = 2)
  (h2 : q.AD = 6)
  (h3 : q.PA = 2)
  (h4 : q.M ∈ Set.Icc 0 q.AB)
  (h5 : q.N ∈ Set.Icc 0 q.AB)
  (h6 : ∀ q', perimeter q ≤ perimeter q') :
  Real.cos (dihedralAngle q) = Real.sqrt 6 / 6 := by
  sorry

#check dihedral_angle_cosine_when_perimeter_minimized

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_when_perimeter_minimized_l987_98767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l987_98784

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l987_98784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_ellipse_line_intersection_l987_98793

/-- The ellipse C in standard form -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The line l -/
def line (k m x y : ℝ) : Prop := y = k * x + m

/-- The perpendicular bisector condition -/
def perp_bisector_condition (k m : ℝ) : Prop :=
  2 * k^2 + 1 = 2 * m

/-- The area of triangle AOB -/
noncomputable def triangle_area (m : ℝ) : ℝ := 
  (1 / 2) * Real.sqrt (4 * m - 2 * m^2)

/-- The main theorem -/
theorem max_triangle_area_ellipse_line_intersection :
  ∃ (k m : ℝ),
    k ≠ 0 ∧
    perp_bisector_condition k m ∧
    (∀ (k' m' : ℝ), k' ≠ 0 → perp_bisector_condition k' m' →
      triangle_area m ≥ triangle_area m') ∧
    triangle_area m = Real.sqrt 2 / 2 ∧
    (line k m = line (Real.sqrt 2 / 2) 1 ∨ line k m = line (-Real.sqrt 2 / 2) 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_ellipse_line_intersection_l987_98793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_is_97_6_l987_98779

/-- A rectangular solid with volume 512 cm³, surface area 352 cm², and dimensions in geometric progression -/
structure GeometricSolid where
  a : ℝ
  r : ℝ
  volume : a^3 = 512
  surface_area : 2 * (a^2 / r + a^2 * r + a^2) = 352
  geometric_prog : (a / r) * r = a ∧ a * r = a * r

/-- The sum of the lengths of all edges of the geometric solid -/
noncomputable def sum_of_edges (solid : GeometricSolid) : ℝ :=
  4 * (solid.a / solid.r + solid.a + solid.a * solid.r)

/-- Theorem stating that the sum of edge lengths is 97.6 cm -/
theorem sum_of_edges_is_97_6 (solid : GeometricSolid) :
  sum_of_edges solid = 97.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_is_97_6_l987_98779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_l987_98733

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 5

-- Theorem statement
theorem intersection_point_on_circle :
  ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ my_circle x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_l987_98733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_incorrectness_l987_98774

-- Define the functions
noncomputable def f (x : ℝ) := x^(-2 : ℤ)
noncomputable def g (x : ℝ) := Real.cos x
noncomputable def h (x : ℝ) := x * Real.log x
noncomputable def k (x : ℝ) := (2 : ℝ)^x

-- State the theorem
theorem derivative_incorrectness :
  (∃ x : ℝ, deriv f x ≠ -2 * x^(-1 : ℤ)) ∧
  (∀ x : ℝ, deriv g x = -Real.sin x) ∧
  (∀ x : ℝ, x > 0 → deriv h x = 1 + Real.log x) ∧
  (∀ x : ℝ, deriv k x = (2 : ℝ)^x * Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_incorrectness_l987_98774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_vectors_l987_98732

open Real InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle (a b : V) : ℝ := Real.arccos (inner a b / (norm a * norm b))

theorem max_angle_between_vectors (a b : V) (k : ℝ) 
  (h1 : norm a = 1) 
  (h2 : norm b = 1) 
  (h3 : k > 0) 
  (h4 : norm (k • a + b) = Real.sqrt 3 * norm (a - k • b)) : 
  angle a b ≤ π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_vectors_l987_98732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_fraction_is_four_fifths_l987_98753

/-- Represents the shaded fraction of a square at each level of division -/
def shadedFraction : ℕ → ℚ
| 0 => 12 / 16
| n + 1 => 12 / 16 * (1 / 16) ^ n

/-- The sum of shaded fractions for all levels of division -/
noncomputable def totalShadedFraction : ℚ := ∑' n, shadedFraction n

/-- Theorem stating that the total shaded fraction is equal to 4/5 -/
theorem total_shaded_fraction_is_four_fifths : 
  totalShadedFraction = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_fraction_is_four_fifths_l987_98753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_for_60_yuan_bill_l987_98798

/-- Represents the tiered water pricing system --/
structure WaterPricing where
  tier1_limit : ℚ
  tier1_price : ℚ
  tier2_limit : ℚ
  tier2_price : ℚ
  tier3_price : ℚ

/-- Calculates the water bill based on usage and pricing system --/
def calculate_bill (usage : ℚ) (pricing : WaterPricing) : ℚ :=
  if usage ≤ pricing.tier1_limit then
    usage * pricing.tier1_price
  else if usage ≤ pricing.tier2_limit then
    pricing.tier1_limit * pricing.tier1_price + (usage - pricing.tier1_limit) * pricing.tier2_price
  else
    pricing.tier1_limit * pricing.tier1_price + 
    (pricing.tier2_limit - pricing.tier1_limit) * pricing.tier2_price + 
    (usage - pricing.tier2_limit) * pricing.tier3_price

/-- Theorem stating that given the specific pricing system and a bill of 60 yuan, the water usage is 16m³ --/
theorem water_usage_for_60_yuan_bill (pricing : WaterPricing) 
  (h1 : pricing.tier1_limit = 12)
  (h2 : pricing.tier1_price = 3)
  (h3 : pricing.tier2_limit = 18)
  (h4 : pricing.tier2_price = 6)
  (h5 : pricing.tier3_price = 9) :
  ∃ (usage : ℚ), calculate_bill usage pricing = 60 ∧ usage = 16 := by
  use 16
  constructor
  · simp [calculate_bill, h1, h2, h3, h4, h5]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_for_60_yuan_bill_l987_98798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpicked_cards_sum_l987_98778

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def count_divisors (n : ℕ) : ℕ := (Finset.filter (λ d ↦ d ∣ n) (Finset.range (n + 1))).card

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem unpicked_cards_sum :
  ∀ (picked : Finset ℕ) (a b c : ℕ),
    picked.card = 9 ∧
    (∀ n, n ∈ picked → 1 ≤ n ∧ n ≤ 13) ∧
    (∃ x y, x ∈ picked ∧ y ∈ picked ∧ count_divisors x ≠ count_divisors y) ∧
    a ∈ picked ∧
    b ∈ picked ∧
    c ∈ picked ∧
    c = b - 2 ∧
    c = a + 1 ∧
    (is_odd b ∨ ¬is_odd b) →
    (Finset.sum (Finset.filter (λ n ↦ n ∉ picked) (Finset.range 14)) id) = 28 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpicked_cards_sum_l987_98778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l987_98746

/-- Represents a quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The length of a side in a quadrilateral -/
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- The angle between three points -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_area (q : Quadrilateral) :
  (side_length q.A q.B = 3) →
  (side_length q.B q.C = 4) →
  (side_length q.C q.D = 12) →
  (side_length q.D q.A = 13) →
  (angle q.C q.B q.A = Real.pi / 2) →
  (area q = 36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l987_98746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_wire_configuration_l987_98751

/-- Beam configuration -/
structure BeamConfig where
  d : ℝ  -- beam length
  G : ℝ  -- beam weight
  G1 : ℝ -- load weight
  r : ℝ  -- wire radius
  E : ℝ  -- wire elastic modulus

/-- Theorem for optimal wire configuration -/
theorem optimal_wire_configuration (config : BeamConfig) :
  let h_optimal := config.d
  let lambda_min := ((config.G / 2 + config.G1) / (config.E * Real.pi * config.r^2)) * 2 * config.d
  let F := (config.G / 2 + config.G1) * Real.sqrt 2
  -- 1. Optimal height
  (∀ h, h > 0 → lambda_min ≤ ((config.G / 2 + config.G1) / (config.E * Real.pi * config.r^2)) * (h^2 + config.d^2) / h) ∧
  -- 2. Minimum elongation
  (lambda_min = ((config.G / 2 + config.G1) / (config.E * Real.pi * config.r^2)) * 2 * config.d) ∧
  -- 3. Corresponding tensile force
  (F = (config.G / 2 + config.G1) * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_wire_configuration_l987_98751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treehouse_paint_cost_l987_98709

/-- Calculate the total cost of paint for a treehouse project -/
theorem treehouse_paint_cost (white_oz green_oz brown_oz blue_oz : ℝ)
  (white_price green_price brown_price blue_price : ℝ)
  (loss_percentage : ℝ) (oz_per_liter : ℝ) :
  white_oz = 20 →
  green_oz = 15 →
  brown_oz = 34 →
  blue_oz = 12 →
  white_price = 8.5 →
  green_price = 7.2 →
  brown_price = 6.9 →
  blue_price = 9.0 →
  loss_percentage = 0.1 →
  oz_per_liter = 33.814 →
  (((white_oz * (1 + loss_percentage) / oz_per_liter) * white_price) +
   ((green_oz * (1 + loss_percentage) / oz_per_liter) * green_price) +
   ((brown_oz * (1 + loss_percentage) / oz_per_liter) * brown_price) +
   ((blue_oz * (1 + loss_percentage) / oz_per_liter) * blue_price)) = 20.23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_treehouse_paint_cost_l987_98709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rented_trucks_problem_l987_98783

/-- The maximum number of trucks that could have been rented out given the conditions -/
def max_rented_trucks (total_trucks : ℕ) (return_rate : ℚ) (min_remaining : ℕ) : ℕ :=
  Int.toNat <| Int.floor ((total_trucks - min_remaining : ℚ) / (1 - return_rate))

/-- Theorem stating the maximum number of trucks that could have been rented out -/
theorem max_rented_trucks_problem :
  max_rented_trucks 45 (2/5) 25 = 33 := by
  sorry

#eval max_rented_trucks 45 (2/5) 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rented_trucks_problem_l987_98783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l987_98758

-- Define the points
variable (P Q R S T U : ℝ × ℝ)

-- Define the necessary functions and predicates
def IsRectangle (P Q R S : ℝ × ℝ) : Prop := sorry
def AngleTrisector (R T U : ℝ × ℝ) : Prop := sorry
def OnLine (P Q R : ℝ × ℝ) : Prop := sorry
def SegmentLength (P Q : ℝ × ℝ) : ℝ := sorry
noncomputable def Area (P Q R S : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem rectangle_area (P Q R S T U : ℝ × ℝ) 
  (h1 : IsRectangle P Q R S)
  (h2 : AngleTrisector R T U)
  (h3 : OnLine U P Q)
  (h4 : OnLine T P S)
  (h5 : SegmentLength Q U = 8)
  (h6 : SegmentLength P T = 3) :
  Area P Q R S = 192 * Real.sqrt 3 - 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l987_98758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l987_98773

theorem count_integers_in_square_range : 
  (Finset.filter (fun n : ℕ => 300 < n^2 ∧ n^2 < 1200) (Finset.range 35)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l987_98773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_product_l987_98797

def has_exactly_four_factors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 4

theorem factors_of_product (a b c d : ℕ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  has_exactly_four_factors a →
  has_exactly_four_factors b →
  has_exactly_four_factors c →
  has_exactly_four_factors d →
  (Finset.filter (· ∣ a^2 * b^3 * c^4 * d^5) (Finset.range (a^2 * b^3 * c^4 * d^5 + 1))).card = 14560 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_product_l987_98797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_length_l987_98744

/-- A quadrilateral with diagonals intersecting at a point -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (bo : dist O B = 4)
  (od : dist O D = 6)
  (ao : dist O A = 8)
  (oc : dist O C = 3)
  (ab : dist A B = 6)

/-- The length of AD in the quadrilateral is √166 -/
theorem ad_length (q : Quadrilateral) : dist q.A q.D = Real.sqrt 166 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_length_l987_98744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l987_98739

theorem sum_of_coefficients : 
  let p : Polynomial ℤ := (4 : Polynomial ℤ) - X
  let q : Polynomial ℤ := X + 3 * ((4 : Polynomial ℤ) + X)
  let expanded := p * q
  (expanded.coeff 2) + (expanded.coeff 1) + (expanded.coeff 0) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l987_98739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l987_98792

noncomputable def f (k : ℤ) (a : ℝ) (x : ℝ) : ℝ := a^x + k * a^(-x)

theorem f_properties 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_1 : a ≠ 1) 
  (h_f_1_half : f 1 a (1/2) = 3) 
  (k : ℤ) 
  (h_f_odd : ∀ x, f k a x = -f k a (-x)) 
  (h_a_lt_1 : 0 < a ∧ a < 1) :
  (f 1 a 2 = 47) ∧ 
  (∀ m : ℝ, (∀ x ∈ Set.Icc 1 3, f k a (m*x^2 - m*x - 1) + f k a (m - 5) > 0) ↔ m < 6/7) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l987_98792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_running_proof_l987_98724

/-- The length of the cord tying the dog to the tree -/
def cord_length : ℝ := 10

/-- The approximate distance the dog ran -/
def dog_distance : ℝ := 30

/-- The circumference of the circle formed by the fully extended cord -/
noncomputable def circle_circumference : ℝ := 2 * Real.pi * cord_length

theorem dog_running_proof :
  dog_distance < circle_circumference / 2 ∧
  ¬∃ (start_direction : ℝ), True :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_running_proof_l987_98724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_value_difference_bound_l987_98728

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

-- Part 1
theorem max_value_on_interval (x : ℝ) (hx : x ∈ Set.Icc 1 (Real.exp 1)) :
  f (-4) x ≤ (Real.exp 1)^2 - 4 := by sorry

-- Part 2
theorem value_difference_bound (a : ℝ) (ha : a > 0) (x₁ x₂ : ℝ)
  (hx₁ : x₁ ∈ Set.Icc (1/(Real.exp 1)) (1/2)) (hx₂ : x₂ ∈ Set.Icc (1/(Real.exp 1)) (1/2)) :
  |f a x₁ - f a x₂| ≤ |1/x₁ - 1/x₂| → a ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_value_difference_bound_l987_98728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l987_98726

theorem tan_difference (α : ℝ) (h : Real.tan α = 2) : Real.tan (α - 3 * π / 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l987_98726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l987_98769

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a - b * (1 - 2 * Real.sin (C / 2) ^ 2) = c / 2 →
  B = Real.pi / 3 ∧
  (b = 6 → 12 < a + b + c ∧ a + b + c ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l987_98769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_203_is_gray_l987_98795

/-- Represents the color of a marble -/
inductive MarbleColor
  | Gray
  | White
  | Black
  | Red

/-- The sequence pattern of marble colors -/
def marblePattern : List MarbleColor :=
  List.replicate 7 MarbleColor.Gray ++
  List.replicate 5 MarbleColor.White ++
  List.replicate 4 MarbleColor.Black ++
  List.replicate 4 MarbleColor.Red

/-- Get the color of the nth marble in the repeating sequence -/
def nthMarbleColor (n : Nat) : MarbleColor :=
  let patternLength := marblePattern.length
  let index := (n - 1) % patternLength
  marblePattern[index]'(by {
    simp [marblePattern]
    apply Nat.mod_lt
    exact Nat.zero_lt_succ _
  })

theorem marble_203_is_gray :
  nthMarbleColor 203 = MarbleColor.Gray := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_203_is_gray_l987_98795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l987_98771

/-- The function g(x) defined as sin^6 x - sin x cos x + cos^6 x -/
noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6

/-- Theorem stating that the range of g(x) is [5/8, 17/8] -/
theorem range_of_g :
  ∀ x : ℝ, 5/8 ≤ g x ∧ g x ≤ 17/8 ∧
  ∃ x₁ x₂ : ℝ, g x₁ = 5/8 ∧ g x₂ = 17/8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l987_98771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_quarter_l987_98787

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := cos x * cos (x - π/3)

-- State the theorem
theorem f_less_than_quarter (x : ℝ) : 
  f x < 1/4 ↔ ∃ k : ℤ, x ∈ Set.Ioo ((k : ℝ) * π - 7*π/12) ((k : ℝ) * π - π/12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_quarter_l987_98787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_problem_l987_98766

theorem base_conversion_problem (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are base-10 digits
  (5 * 8^2 + 4 * 8^1 + 7 * 8^0 = 300 + 10 * c + d) →  -- 547₈ = 3cd₁₀
  (c * d : ℚ) / 15 = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_problem_l987_98766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_trip_percentage_l987_98752

-- Define the total number of students
noncomputable def total_students : ℝ := 100

-- Define the percentage of students who went to the camping trip and took more than $100
noncomputable def percent_more_than_100 : ℝ := 20

-- Define the percentage of students who went to the camping trip and did not take more than $100
noncomputable def percent_not_more_than_100 : ℝ := 75

-- Define the function to calculate the total percentage of students who went to the camping trip
noncomputable def total_percent_camping : ℝ := percent_more_than_100

-- Theorem statement
theorem camping_trip_percentage :
  total_percent_camping = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_trip_percentage_l987_98752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tommy_loan_percentage_l987_98785

-- Define the given values
def first_home_cost : ℚ := 100000
def first_home_value_increase_percent : ℚ := 25
def new_home_cost : ℚ := 500000

-- Define the function to calculate the loan percentage
noncomputable def loan_percentage (first_cost new_cost increase_percent : ℚ) : ℚ :=
  let first_home_value := first_cost * (1 + increase_percent / 100)
  let loan_amount := new_cost - first_home_value
  (loan_amount / new_cost) * 100

-- Theorem statement
theorem tommy_loan_percentage :
  loan_percentage first_home_cost new_home_cost first_home_value_increase_percent = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tommy_loan_percentage_l987_98785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comet_watching_time_l987_98719

/-- Represents the time spent on various activities in minutes -/
structure ActivityTime where
  shopping : ℝ
  setup : ℝ
  snacks : ℝ
  watching : ℝ

/-- Calculates the total time spent on all activities -/
def totalTime (t : ActivityTime) : ℝ :=
  t.shopping + t.setup + t.snacks + t.watching

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Main theorem: The comet-watching time rounded to the nearest minute is 21 minutes -/
theorem comet_watching_time (t : ActivityTime) : 
  t.shopping = 120 ∧ 
  t.setup = 30 ∧ 
  t.snacks = 3 * t.setup ∧ 
  t.watching / totalTime t = 0.08 →
  roundToNearest t.watching = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comet_watching_time_l987_98719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_proof_l987_98762

-- Define the painting rates
def abe_rate : ℚ := 1 / 15
def bea_rate : ℚ := abe_rate * (3 / 2)
def coe_rate : ℚ := abe_rate * 2

-- Define the work schedule
def abe_alone_time : ℚ := 3 / 2
def half_room : ℚ := 1 / 2

-- Theorem statement
theorem painting_time_proof :
  let time_abe_alone := abe_alone_time
  let work_abe_alone := abe_rate * time_abe_alone
  let work_left_for_two := half_room - work_abe_alone
  let rate_abe_bea := abe_rate + bea_rate
  let time_abe_bea := work_left_for_two / rate_abe_bea
  let rate_all := abe_rate + bea_rate + coe_rate
  let time_all := (1 - half_room) / rate_all
  let total_time := time_abe_alone + time_abe_bea + time_all
  ⌊(total_time * 60)⌋ = 334 := by
    -- The proof goes here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_proof_l987_98762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_theorem_volume_comparison_l987_98757

/-- The volume of a solid constructed from 8 equilateral triangles with unit side length and 2 squares sharing one edge -/
noncomputable def solid_volume : ℝ := (Real.sqrt 3 + Real.sqrt 2) / 4

/-- The number of equilateral triangles in the solid -/
def num_triangles : ℕ := 8

/-- The number of squares in the solid -/
def num_squares : ℕ := 2

/-- The side length of the equilateral triangles -/
def triangle_side_length : ℝ := 1

/-- Theorem stating that the volume of the solid is equal to (√3 + √2) / 4 -/
theorem solid_volume_theorem :
  solid_volume = (Real.sqrt 3 + Real.sqrt 2) / 4 ∧
  num_triangles = 8 ∧
  num_squares = 2 ∧
  triangle_side_length = 1 :=
by
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  · rfl

/-- The volume of the solid examined in problem 1464 -/
noncomputable def problem_1464_volume : ℝ := 0.956

/-- Theorem stating that the volume of the solid is smaller than the volume in problem 1464 -/
theorem volume_comparison :
  solid_volume < problem_1464_volume :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_theorem_volume_comparison_l987_98757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l987_98786

-- Define the trapezoid EFGH
structure Trapezoid where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

-- Define the properties of the trapezoid
def is_valid_trapezoid (t : Trapezoid) : Prop :=
  t.E = (2, -7) ∧
  t.F = (802, 193) ∧
  t.H.1 = 4 ∧
  (∃ y : ℤ, t.H.2 = y) ∧
  (t.E.1 - t.G.1) * (t.F.1 - t.H.1) = (t.E.2 - t.G.2) * (t.F.2 - t.H.2) ∧ -- EG ⊥ FH
  (t.E.1 - t.F.1) * (t.G.2 - t.H.2) = (t.E.2 - t.F.2) * (t.G.1 - t.H.1) ∧ -- EF ∥ GH
  (t.E.1 - t.G.1)^2 + (t.E.2 - t.G.2)^2 = (t.F.1 - t.H.1)^2 + (t.F.2 - t.H.2)^2 -- EG = FH

-- Define the area function
noncomputable def area (t : Trapezoid) : ℝ :=
  let EF := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let GH := Real.sqrt ((t.G.1 - t.H.1)^2 + (t.G.2 - t.H.2)^2)
  let height := |t.E.2 - t.G.2|
  (EF + GH) * height / 2

-- Theorem statement
theorem trapezoid_area (t : Trapezoid) :
  is_valid_trapezoid t → area t = 100 * (200 * Real.sqrt 34 + 2 * Real.sqrt 8650) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l987_98786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_degree_add_sub_l987_98760

-- Define the degree of a polynomial
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

-- Define addition of polynomials
def add_poly (p q : Polynomial ℝ) : Polynomial ℝ := sorry

-- Define subtraction of polynomials
def sub_poly (p q : Polynomial ℝ) : Polynomial ℝ := sorry

theorem poly_degree_add_sub 
  (M N : Polynomial ℝ) 
  (hM : degree M = 5) 
  (hN : degree N = 3) : 
  degree (add_poly M N) = 5 ∧ degree (sub_poly M N) = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_degree_add_sub_l987_98760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_sequence_count_l987_98768

/-- Represents the number of subjects for exams -/
def num_subjects : ℕ := 6

/-- Represents the number of subjects excluding Chinese, Math, and English -/
def num_other_subjects : ℕ := 3

/-- Represents the number of gaps available after arranging other subjects -/
def num_gaps : ℕ := 4

/-- Represents the number of subjects to be placed in gaps (Math and English) -/
def num_gap_subjects : ℕ := 2

/-- Calculates the number of possible exam sequences -/
def exam_sequences : ℕ := 
  1 * (Nat.factorial num_other_subjects) * (Nat.factorial num_gaps / Nat.factorial (num_gaps - num_gap_subjects))

theorem exam_sequence_count :
  exam_sequences = 72 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_sequence_count_l987_98768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surrounding_circles_radius_theorem_l987_98756

/-- The radius of surrounding circles in a configuration where a central circle
    of radius 2 is surrounded by 4 tangent circles. -/
noncomputable def surrounding_circle_radius : ℝ := (4 * Real.sqrt 2 + 2) / 7

/-- Theorem stating the correct radius of surrounding circles in the given configuration. -/
theorem surrounding_circles_radius_theorem (r : ℝ) :
  (∃ (centers : Fin 4 → ℝ × ℝ),
    -- Centers form a square
    (∀ i j : Fin 4, i ≠ j → ‖centers i - centers j‖ = 2 * r ∨ ‖centers i - centers j‖ = 2 * r * Real.sqrt 2) ∧
    -- Surrounding circles are tangent to the central circle
    (∀ i : Fin 4, ‖centers i‖ = 2 + r) ∧
    -- Surrounding circles are tangent to each other
    (∀ i j : Fin 4, i ≠ j → ‖centers i - centers j‖ = 2 * r)) →
  r = surrounding_circle_radius := by
  sorry

#check surrounding_circles_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surrounding_circles_radius_theorem_l987_98756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_monotonicity_l987_98781

-- Define the function f(x) as noncomputable due to dependency on Real.log
noncomputable def f (x : ℝ) : ℝ := Real.log (9 - x^2)

-- Theorem for the domain and interval of monotonic increase
theorem f_domain_and_monotonicity :
  (∀ x : ℝ, x ∈ Set.Ioo (-3 : ℝ) 3 → f x ∈ Set.range f) ∧
  (∀ x y : ℝ, x ∈ Set.Ioc (-3 : ℝ) 0 → y ∈ Set.Ioc (-3 : ℝ) 0 → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_monotonicity_l987_98781
