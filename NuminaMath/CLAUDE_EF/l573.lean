import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fourth_l573_57313

-- Define the circles
noncomputable def large_circle_radius : ℝ := 4
noncomputable def small_circle_radius : ℝ := 2

-- Define the probability function
noncomputable def probability_in_smaller_circle (r_large r_small : ℝ) : ℝ :=
  (r_small^2) / (r_large^2)

-- Theorem statement
theorem probability_is_one_fourth :
  probability_in_smaller_circle large_circle_radius small_circle_radius = 1/4 :=
by
  -- Unfold the definitions
  unfold probability_in_smaller_circle large_circle_radius small_circle_radius
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fourth_l573_57313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_pole_length_l573_57340

theorem bridge_pole_length (bridge_length : ℝ) (total_distance : ℝ) : 
  bridge_length = 20000 →
  total_distance = 20001 →
  ∃ (pole_length : ℝ), 
    2 * Real.sqrt ((bridge_length / 2)^2 + pole_length^2) = total_distance ∧ 
    99.5 ≤ pole_length ∧ pole_length < 100.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_pole_length_l573_57340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l573_57361

open Set

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l573_57361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equal_one_l573_57382

-- Define the theorem
theorem sum_reciprocals_equal_one (m n : ℝ) (h1 : (2 : ℝ)^m = 6) (h2 : (9 : ℝ)^n = 6) : 
  1/m + 1/(2*n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equal_one_l573_57382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l573_57350

open Real

theorem max_value_trig_expression :
  ∀ x y z : ℝ,
  (Real.sin x + Real.sin (2*y) + Real.sin (3*z)) * (Real.cos x + Real.cos (2*y) + Real.cos (3*z)) ≤ 4.5 ∧
  ∃ a b c : ℝ, (Real.sin a + Real.sin (2*b) + Real.sin (3*c)) * (Real.cos a + Real.cos (2*b) + Real.cos (3*c)) = 4.5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l573_57350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_shared_foci_eccentricity_l573_57332

/-- A structure representing a conic section (ellipse or hyperbola) -/
structure Conic where
  eccentricity : ℝ
  foci : Prod ℝ ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Angle between three points in degrees -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Theorem: Eccentricity of a hyperbola sharing foci with an ellipse -/
theorem hyperbola_ellipse_shared_foci_eccentricity 
  (h : Conic) (e : Conic) (P : Point) :
  h.eccentricity > 1 → -- Condition for hyperbola
  e.eccentricity = Real.sqrt 2 / 2 → -- Eccentricity of ellipse
  h.foci = e.foci → -- Shared foci
  (∃ (F₁ F₂ : Point), h.foci = (F₁.x, F₁.y) ∧ 
    angle F₁ P F₂ = 60) → -- Common point P with 60° angle
  h.eccentricity = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_shared_foci_eccentricity_l573_57332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_min_value_l573_57378

-- Define the function f(x) with parameter k
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (4^x + k) / (2^x)

-- Part 1: Prove that f(x) is an odd function when k = -1
theorem f_is_odd : ∀ x : ℝ, f (-1) (-x) = -(f (-1) x) := by
  sorry

-- Part 2: Prove that the minimum value of f(x) is 2 when k = 1
theorem f_min_value : ∀ x : ℝ, f 1 x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_min_value_l573_57378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l573_57381

/-- The function f(x) = ln x - a(x-1) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1)

/-- The theorem stating that if f(x) ≤ (ln x) / (x+1) for all x ≥ 1, then a ≥ 1/2 --/
theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≤ (Real.log x) / (x + 1)) →
  a ≥ 1/2 := by
  sorry

#check f_inequality_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l573_57381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_store_income_l573_57311

/-- Calculates the amount of money kept after returns in a book store -/
def money_kept_after_returns (total_customers : ℕ) (return_rate : ℚ) (price_per_book : ℕ) : ℚ :=
  let total_sales := (total_customers : ℚ) * price_per_book
  let returns := (return_rate * total_customers : ℚ).floor
  let return_amount := (returns : ℚ) * price_per_book
  total_sales - return_amount

/-- Theorem stating that given 1000 customers, 37% return rate, and $15 per book, 
    the amount kept after returns is $9,450 -/
theorem book_store_income : 
  money_kept_after_returns 1000 (37 / 100) 15 = 9450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_store_income_l573_57311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l573_57302

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + 0.5 * t, Real.sqrt 3 / 2 * t)

-- Define point P
def point_P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem intersection_distance_product :
  ∃ (t₁ t₂ : ℝ),
    (circle_C t₁).1^2 + (circle_C t₁).2^2 = 4 * (circle_C t₁).1 ∧
    (circle_C t₂).1^2 + (circle_C t₂).2^2 = 4 * (circle_C t₂).1 ∧
    circle_C t₁ = line_l t₁ ∧
    circle_C t₂ = line_l t₂ ∧
    (point_P.1 - (line_l t₁).1)^2 + (point_P.2 - (line_l t₁).2)^2 *
    ((point_P.1 - (line_l t₂).1)^2 + (point_P.2 - (line_l t₂).2)^2) = 3^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l573_57302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_decreasing_l573_57364

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * q^(n - 1)

/-- A sequence is decreasing if a_{n+1} < a_n for all n -/
def is_decreasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (n + 1) < f n

theorem geometric_sequence_decreasing (a₁ q : ℝ) 
  (h1 : a₁ * (q - 1) < 0) (h2 : q > 0) : 
  is_decreasing (geometric_sequence a₁ q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_decreasing_l573_57364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_distance_l573_57370

noncomputable def A : ℝ × ℝ := (2, 3)
noncomputable def B : ℝ × ℝ := (4, 5)
noncomputable def C : ℝ × ℝ := (6, 5)
noncomputable def D : ℝ × ℝ := (7, 4)
noncomputable def E : ℝ × ℝ := (8, 1)
noncomputable def P : ℝ × ℝ := (5, 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem different_distance :
  distance P D ≠ distance P A ∧
  distance P D ≠ distance P B ∧
  distance P D ≠ distance P C ∧
  distance P D ≠ distance P E :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_distance_l573_57370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_12_triangle_area_12_l573_57318

-- Define the point P
def P : ℝ × ℝ := (3, 2)

-- Define a line passing through P
def line_through_P (m n : ℝ) : Prop :=
  3 / m + 2 / n = 1

-- Statement for scenario 1
theorem intercept_sum_12 :
  ∀ m n : ℝ, m > 0 → n > 0 →
  line_through_P m n →
  m + n = 12 →
  (m = 4 ∧ n = 8) ∨ (m = 9 ∧ n = 3) :=
by
  sorry

-- Statement for scenario 2
theorem triangle_area_12 :
  ∀ m n : ℝ, m > 0 → n > 0 →
  line_through_P m n →
  1/2 * m * n = 12 →
  m = 6 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_12_triangle_area_12_l573_57318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_eq_binary_digits_l573_57305

/-- The smallest sum of floor(a_i/i) for a permutation of (1,...,n) -/
def min_sum (n : ℕ) : ℕ :=
  Nat.rec 0 (λ k ih ↦ min k (Nat.floor (n / k) + ih)) n

/-- The number of digits in the binary representation of n -/
def binary_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem min_sum_eq_binary_digits (n : ℕ) (hn : n ≥ 1) :
  min_sum n = binary_digits n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_eq_binary_digits_l573_57305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l573_57372

noncomputable section

open Real

def f (x : ℝ) : ℝ := sqrt 3 * sin (π * x + π / 3)
def g (x : ℝ) : ℝ := sin (π / 6 - π * x)

def M : ℝ × ℝ := (-(1/6), sqrt 3 / 2)
def N : ℝ × ℝ := (5/6, -sqrt 3 / 2)
def O : ℝ × ℝ := (0, 0)

theorem intersection_dot_product :
  let OM : ℝ × ℝ := (M.1 - O.1, M.2 - O.2)
  let ON : ℝ × ℝ := (N.1 - O.1, N.2 - O.2)
  (f M.1 = M.2 ∧ g M.1 = M.2 ∧ f N.1 = N.2 ∧ g N.1 = N.2) →
  OM.1 * ON.1 + OM.2 * ON.2 = -8/9 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l573_57372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_plus_min_equals_two_l573_57358

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + 1)^2 / (Real.sin x^2 + 1)

-- State the theorem
theorem max_plus_min_equals_two :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, m ≤ f x) ∧ (M + m = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_plus_min_equals_two_l573_57358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_x_axis_l573_57388

/-- A line passing through two given points intersects the x-axis at a specific point. -/
theorem line_intersection_x_axis (x₁ y₁ x₂ y₂ : ℝ) :
  let m : ℝ := (y₂ - y₁) / (x₂ - x₁)
  let b : ℝ := y₁ - m * x₁
  let line := fun x => m * x + b
  x₁ = 3 ∧ y₁ = 2 ∧ x₂ = 6 ∧ y₂ = 5 →
  ∃ x : ℝ, line x = 0 ∧ x = 1 :=
by
  sorry

#check line_intersection_x_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_x_axis_l573_57388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l573_57387

/-- The area of a figure formed by rotating a semicircle about one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (2 * Real.pi * R^2 * α) / (2 * Real.pi)

/-- Theorem: The area of a figure formed by rotating a semicircle of radius R 
    about one of its ends by an angle of 60° (π/3 radians) is equal to (2πR²)/3 -/
theorem rotated_semicircle_area_60_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (Real.pi/3) = (2 * Real.pi * R^2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l573_57387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_circles_ratio_l573_57329

theorem inscribed_squares_circles_ratio : 
  ∀ (r : ℝ), r > 0 →
  (2 * r / Real.sqrt 2)^2 / (π * r^2) = 2 / π :=
by
  intros r hr
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_circles_ratio_l573_57329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_eight_l573_57312

theorem probability_factor_less_than_eight (n : ℕ) (h : n = 84) :
  let factors := Finset.filter (λ d => d > 0 ∧ n % d = 0) (Finset.range (n + 1))
  let factors_less_than_eight := Finset.filter (λ d => d < 8) factors
  (factors_less_than_eight.card : ℚ) / (factors.card : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_eight_l573_57312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_current_speed_l573_57379

/-- Proves that given a boat with a speed of 26 km/hr in still water,
    traveling 10.67 km downstream in 20 minutes, the rate of the current is 6.01 km/hr. -/
theorem boat_current_speed : 
  let boat_speed : ℝ := 26 -- km/hr
  let downstream_distance : ℝ := 10.67 -- km
  let travel_time : ℝ := 20 / 60 -- hr (20 minutes converted to hours)
  let current_speed : ℝ := downstream_distance / travel_time - boat_speed
  current_speed = 6.01 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_current_speed_l573_57379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_problem_l573_57337

theorem complex_power_problem (x y : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (x - 1) * i - y = 2 + i) : (1 + i : ℂ) ^ (x - y : ℂ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_problem_l573_57337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_for_18_18_l573_57399

/-- The number of positive factors of a positive integer -/
def num_factors (m : ℕ) : ℕ := sorry

/-- The maximum number of positive factors for b^n given constraints -/
def max_factors (b_max n_max : ℕ) : ℕ :=
  Finset.sup (Finset.range (b_max + 1) ×ˢ Finset.range (n_max + 1)) (fun ⟨b, n⟩ ↦ num_factors ((b + 1) ^ (n + 1)))

theorem max_factors_for_18_18 :
  max_factors 18 18 = 703 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_for_18_18_l573_57399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_nine_pi_thirds_eq_zero_l573_57336

theorem tan_nine_pi_thirds_eq_zero :
  Real.tan (9 * Real.pi / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_nine_pi_thirds_eq_zero_l573_57336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_codys_payment_is_17_l573_57366

/-- Calculates Cody's share of the final price given the initial purchase amount,
    tax rate, discount, and number of people splitting the bill. -/
noncomputable def codys_share (initial_purchase : ℝ) (tax_rate : ℝ) (discount : ℝ) (num_people : ℝ) : ℝ :=
  (initial_purchase * (1 + tax_rate) - discount) / num_people

/-- Proves that Cody's share is $17 given the specific conditions of the problem. -/
theorem codys_payment_is_17 :
  codys_share 40 0.05 8 2 = 17 := by
  sorry

-- Use #eval with specific Real numbers instead of the function
#eval (40 * (1 + 0.05) - 8) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_codys_payment_is_17_l573_57366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_relationship_l573_57324

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the relationship between roots
def root_relationship (r s : ℝ) : Prop := s = (1/2) * r^3

-- Theorem statement
theorem quadratic_root_relationship (a b c : ℝ) (ha : a ≠ 0) :
  (∃ r s : ℝ, quadratic_equation a b c r ∧ quadratic_equation a b c s ∧ root_relationship r s) →
  b^2 = a^2 * (1 + 2 * Real.rpow 4 (1/3)) + a * Real.rpow 2 (4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_relationship_l573_57324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_450_l573_57317

theorem sum_odd_divisors_450 : 
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors 450)).sum id = 403 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_450_l573_57317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l573_57367

theorem factorial_ratio (N : ℕ) : 
  (Nat.factorial (N + 1)) / (Nat.factorial (N + 2)) = 1 / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l573_57367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l573_57373

/-- Given two trains leaving at the same time, prove that the speed of the first train is 50 MPH -/
theorem first_train_speed (distance1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) (avg_time : ℝ) : 
  distance1 = 200 →
  distance2 = 240 →
  speed2 = 80 →
  avg_time = 4 →
  (distance1 / 50 + distance2 / speed2) / 2 = avg_time →
  distance1 / avg_time = 50 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l573_57373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_l573_57390

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + (a + 2) * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x - y + 2 = 0

-- Define when two lines are parallel
def parallel (m₁ b₁ m₂ b₂ : ℝ) : Prop := m₁ = m₂ ∧ b₁ ≠ b₂

-- Theorem statement
theorem line_intersection (a : ℝ) :
  (∃ x y : ℝ, l₁ a x y ∧ l₂ a x y) →
  (¬∃ m₁ b₁ m₂ b₂ : ℝ, (∀ x y : ℝ, l₁ a x y ↔ y = m₁ * x + b₁) ∧
                       (∀ x y : ℝ, l₂ a x y ↔ y = m₂ * x + b₂) ∧
                       parallel m₁ b₁ m₂ b₂) →
  a = 0 ∨ a = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_l573_57390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l573_57346

/-- The eccentricity of a hyperbola with the given properties is √3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (A F₁ F₂ : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  c > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x, y) ∈ Set.range (λ t ↦ (t, Real.sqrt (b^2*(t^2/a^2 - 1))))) →
  F₁ = (-c, 0) →
  F₂ = (c, 0) →
  A.1 = -c →
  Real.sqrt 3 * dist A F₁ = dist F₁ F₂ →
  c^2 = a^2 + b^2 →
  Real.sqrt 3 = c / a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l573_57346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l573_57353

/-- Represents a parabola in the Cartesian plane -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * a * x

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the Cartesian plane -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = m * x + b

noncomputable def Parabola.focus (p : Parabola) : Point :=
  { x := p.a / 2, y := 0 }

noncomputable def Parabola.directrix (p : Parabola) : Line :=
  { m := 0, b := -p.a / 2, equation := fun x y => x = -p.a / 2 }

theorem parabola_theorem (C : Parabola) 
    (h1 : C.focus = { x := 1, y := 0 }) :
  C.a = 2 ∧
  (∀ (M N : Point) (y₁ y₂ : ℝ),
    M.x = -1 ∧ M.y = y₁ ∧
    N.x = -1 ∧ N.y = y₂ ∧
    y₁ * y₂ = -4 →
    ∃ (A B : Point),
      C.equation A.x A.y ∧
      C.equation B.x B.y ∧
      (∃ (L : Line),
        L.equation A.x A.y ∧
        L.equation B.x B.y ∧
        L.equation 1 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l573_57353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l573_57354

theorem expression_evaluation : (Real.pi - 3 : ℝ) ^ (0 : ℝ) + 3⁻¹ * (2 + 1/4) ^ (1/2 : ℝ) = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l573_57354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_colabel_with_2014_l573_57320

/-- Represents the labeling pattern on a circular arrangement of points. -/
def CircularLabeling (n : ℕ) (m : ℕ) : Type :=
  {f : ℕ → ℕ // ∀ k, f k = ((k * (k + 1)) / 2) % n + 1 ∧ f k ≤ m}

/-- The theorem stating the smallest number sharing a point with 2014 in the given arrangement. -/
theorem smallest_colabel_with_2014 :
  ∀ (labeling : CircularLabeling 70 2014),
  ∃ (k : ℕ), k ≤ 5 ∧ labeling.val k = labeling.val 2014 ∧
  (∀ (j : ℕ), j < k → labeling.val j ≠ labeling.val 2014) :=
by
  sorry

#check smallest_colabel_with_2014

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_colabel_with_2014_l573_57320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l573_57393

/-- Given a sequence {a_n} with first n terms sum S_n = 2n^2 + 3n -/
def S (n : ℕ) : ℚ := 2 * n^2 + 3 * n

/-- The general term of the sequence -/
def a (n : ℕ) : ℚ := 4 * n + 1

/-- The b_n sequence defined as 1 / (a_n * a_{n+1}) -/
noncomputable def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

/-- The sum of the first n terms of the b_n sequence -/
def T (n : ℕ) : ℚ := n / (5 * (4 * n + 5))

theorem sequence_properties :
  (∀ n : ℕ, S n = 2 * n^2 + 3 * n) →
  (∀ n : ℕ, a n = 4 * n + 1) ∧
  (∀ n : ℕ, T n = n / (5 * (4 * n + 5))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l573_57393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowboarder_count_l573_57343

theorem snowboarder_count : ∃! x : ℕ,
  (0 < x) ∧
  (∃ y : ℕ, 
    (y ≤ 10) ∧ 
    (((3 * x - y : ℚ) / (3 * x)) > (45 : ℚ) / 100) ∧
    (((3 * x - y : ℚ) / (3 * x)) < (1 : ℚ) / 2)) ∧
  x = 5 := by
  sorry

#check snowboarder_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowboarder_count_l573_57343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_assignment_count_l573_57365

/-- Represents a workshop assignment for 4 employees to 3 workshops -/
def WorkshopAssignment := Fin 3 → Fin 4 → Bool

/-- Checks if an assignment is valid (each workshop has at least one employee) -/
def is_valid_assignment (a : WorkshopAssignment) : Prop :=
  ∀ w : Fin 3, ∃ e : Fin 4, a w e

/-- Checks if two specific employees (0 and 1) are in the same workshop -/
def employees_0_1_together (a : WorkshopAssignment) : Prop :=
  ∃ w : Fin 3, a w 0 ∧ a w 1

/-- The set of all valid assignments -/
def valid_assignments : Set WorkshopAssignment :=
  {a | is_valid_assignment a ∧ employees_0_1_together a}

/-- Provide instances for WorkshopAssignment -/
instance : Fintype WorkshopAssignment := by sorry

instance : DecidablePred (λ a : WorkshopAssignment => a ∈ valid_assignments) := by sorry

theorem workshop_assignment_count : Finset.card (Finset.filter (λ a => a ∈ valid_assignments) (Finset.univ : Finset WorkshopAssignment)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_assignment_count_l573_57365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_no_upper_bound_for_a_l573_57356

/-- The function f(x) as defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := a * Real.exp (x - 1) - Real.log x + Real.log a

/-- The function g(x) as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := (1 - Real.exp 1) * x

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) : 
  (∀ x > 0, Real.exp 1 * f a x ≥ g x) ↔ a ≥ Real.exp (-1) := by
  sorry

/-- The theorem stating there's no upper bound for a -/
theorem no_upper_bound_for_a : 
  ∀ M : ℝ, ∃ a : ℝ, a > M ∧ (∀ x > 0, Real.exp 1 * f a x ≥ g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_no_upper_bound_for_a_l573_57356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l573_57300

-- Define the function f as noncomputable
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

-- State the theorem
theorem omega_value :
  ∀ ω : ℝ,
  ω > 0 →
  (f ω (π / 6) + f ω (π / 2) = 0) →
  (∀ x y : ℝ, π / 6 < x → x < y → y < π / 2 → f ω y < f ω x) →
  ω = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l573_57300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l573_57341

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

-- State the theorem
theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- f is decreasing on [kπ + π/6, kπ + 2π/3] for any integer k
  (∀ (k : ℤ), ∀ (x y : ℝ),
    x ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) →
    y ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) →
    x < y → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l573_57341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l573_57380

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

noncomputable def Hyperbola.eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

noncomputable def Hyperbola.focus (h : Hyperbola) : Point :=
  { x := 0, y := -Real.sqrt (h.a^2 + h.b^2) }

noncomputable def Hyperbola.asymptote (h : Hyperbola) : Line :=
  { slope := h.a / h.b, y_intercept := 0 }

noncomputable def line_through_points (p1 p2 : Point) : Line :=
  { slope := (p2.y - p1.y) / (p2.x - p1.x)
  , y_intercept := p1.y - (p2.y - p1.y) / (p2.x - p1.x) * p1.x }

noncomputable def intersection_point (l1 l2 : Line) : Point :=
  { x := (l2.y_intercept - l1.y_intercept) / (l1.slope - l2.slope)
  , y := l1.slope * (l2.y_intercept - l1.y_intercept) / (l1.slope - l2.slope) + l1.y_intercept }

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

theorem hyperbola_eccentricity (h : Hyperbola) 
  (l : Line)
  (hf : l.slope = Real.sqrt 3 / 3)
  (ha : Point)
  (hb : Point)
  (hl : l = line_through_points h.focus hb)
  (hoa_eq_ob : distance ⟨0, 0⟩ ha = distance ⟨0, 0⟩ hb)
  (ha_on_asymptote : ha = intersection_point l h.asymptote)
  (hb_on_x_axis : hb.y = 0 ∧ hb.x > 0) :
  h.eccentricity = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l573_57380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l573_57319

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the angle at a vertex of the triangle -/
noncomputable def Triangle.angle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Calculate the length of a side of the triangle -/
noncomputable def Triangle.side (t : Triangle) (v1 v2 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the perimeter of the triangle -/
noncomputable def Triangle.perimeter (t : Triangle) : ℝ := sorry

/-- Check if the triangle is equilateral -/
def Triangle.is_equilateral (t : Triangle) : Prop := sorry

/-- A triangle with one 60° angle and the opposite side equal to one-third of its perimeter is equilateral -/
theorem triangle_equilateral (t : Triangle) : 
  t.angle t.A = 60 * π / 180 →
  t.side t.B t.C = t.perimeter / 3 →
  t.is_equilateral := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l573_57319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_max_product_l573_57369

noncomputable def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / 4^2 = 1

noncomputable def segment_length (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def focus (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - 16), 0)

theorem ellipse_focus_max_product (a : ℝ) (h1 : a > 0) (h2 : a ≠ 2) :
  let F := focus a
  ∃ A B : ℝ × ℝ,
    ellipse a A.1 A.2 ∧
    ellipse a B.1 B.2 ∧
    segment_length A B = 3 ∧
    (∀ C D : ℝ × ℝ, ellipse a C.1 C.2 → ellipse a D.1 D.2 → segment_length C D = 3 →
      segment_length A F * segment_length B F ≥ segment_length C F * segment_length D F) →
    (a = 8/3 ∨ a = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_max_product_l573_57369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l573_57345

/-- Calculates the return speed given total distance, outbound speed, and average speed -/
noncomputable def return_speed (total_distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let outbound_time := half_distance / outbound_speed
  let total_time := total_distance / average_speed
  let return_time := total_time - outbound_time
  half_distance / return_time

theorem round_trip_speed 
  (total_distance : ℝ) 
  (outbound_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = 360) 
  (h2 : outbound_speed = 90) 
  (h3 : average_speed = 60) : 
  return_speed total_distance outbound_speed average_speed = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l573_57345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_distance_is_2_3_l573_57309

/-- The distance between flagpoles on a line -/
noncomputable def distance_between_flagpoles (total_length : ℝ) (num_flagpoles : ℕ) : ℝ :=
  total_length / (num_flagpoles - 1 : ℝ)

/-- Theorem: The distance between flagpoles is 2.3 meters -/
theorem flagpole_distance_is_2_3 :
  distance_between_flagpoles 11.5 6 = 2.3 := by
  -- Unfold the definition of distance_between_flagpoles
  unfold distance_between_flagpoles
  -- Simplify the arithmetic
  simp [Nat.cast_sub, Nat.cast_one]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_distance_is_2_3_l573_57309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_on_real_axis_l573_57306

theorem complex_on_real_axis (b : ℝ) :
  (Complex.mk 1 b).im = 0 → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_on_real_axis_l573_57306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_prob_fewer_than_9_l573_57304

/-- A coin-tossing game is represented as a list of booleans, where true represents heads and false represents tails. -/
def CoinGame : Type := List Bool

/-- The game is over when either heads or tails reaches 5. -/
def gameOver (game : CoinGame) : Prop :=
  (game.filter id).length = 5 ∨ (game.filter not).length = 5

/-- The game has fewer than 9 tosses. -/
def fewerThan9Tosses (game : CoinGame) : Prop :=
  game.length < 9

/-- The probability of the game ending in fewer than 9 tosses. -/
def probEndFewerThan9Tosses : ℚ :=
  93 / 128

/-- Theorem stating that the probability of the game ending in fewer than 9 tosses is 93/128. -/
theorem coin_game_prob_fewer_than_9 :
  ∀ (games : Finset CoinGame),
    (∀ game ∈ games, gameOver game ∧ fewerThan9Tosses game) →
    (games.card : ℚ) / (2^9 : ℚ) = probEndFewerThan9Tosses :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_prob_fewer_than_9_l573_57304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_beads_count_l573_57334

/-- Represents the pattern of beads -/
inductive BeadColor
| Green
| Red
| Yellow
deriving BEq, Repr

/-- Defines the repeating pattern of beads -/
def pattern : List BeadColor :=
  [BeadColor.Green, BeadColor.Green, BeadColor.Green,
   BeadColor.Red, BeadColor.Red, BeadColor.Red, BeadColor.Red,
   BeadColor.Yellow]

/-- Calculates the number of red beads in a given number of total beads -/
def countRedBeads (totalBeads : Nat) : Nat :=
  let completeGroups := totalBeads / pattern.length
  let remainingBeads := totalBeads % pattern.length
  let redInCompleteGroups := completeGroups * (pattern.filter (· == BeadColor.Red)).length
  let redInRemainder := (pattern.take remainingBeads).filter (· == BeadColor.Red) |>.length
  redInCompleteGroups + redInRemainder

/-- Theorem: There are 42 red beads in a string of 85 beads with the given pattern -/
theorem red_beads_count :
  countRedBeads 85 = 42 := by
  sorry

#eval countRedBeads 85  -- This will evaluate the function and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_beads_count_l573_57334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_in_terms_of_cos_double_angle_l573_57375

theorem sin_plus_cos_in_terms_of_cos_double_angle (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < π / 2) -- θ is acute
  (h2 : Real.cos (2 * θ) = b) : 
  Real.sin θ + Real.cos θ = Real.sqrt ((1 - b) / 2) + Real.sqrt ((1 + b) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_in_terms_of_cos_double_angle_l573_57375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l573_57359

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x)
noncomputable def g (x : ℝ) : ℝ := (2 : ℝ)^x + 1

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x > 1}

-- State the theorem
theorem domain_range_intersection :
  M ∩ N = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l573_57359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_by_percentage_l573_57371

theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) (result : ℕ) : 
  initial = 60 → percentage = 50 / 100 → result = initial + (initial * percentage).floor → result = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_by_percentage_l573_57371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_m_bound_l573_57398

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Define the function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (x - m) * (f 1 x) - Real.exp x + x^2 + x

-- State the theorem
theorem g_increasing_implies_m_bound (m : ℝ) :
  (∀ x > 2, Monotone (fun x => g m x)) →
  m ≤ (2 * Real.exp 2 + 1) / (Real.exp 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_m_bound_l573_57398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_u_l573_57352

-- Define the function u
noncomputable def u (x y : ℝ) : ℝ := 4 / (4 - x^2) + 9 / (9 - y^2)

-- State the theorem
theorem min_value_of_u :
  ∀ x y : ℝ, -2 < x ∧ x < 2 ∧ -2 < y ∧ y < 2 → x * y = -1 →
  (∀ a b : ℝ, -2 < a ∧ a < 2 ∧ -2 < b ∧ b < 2 ∧ a * b = -1 → u x y ≤ u a b) →
  u x y = 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_u_l573_57352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_squared_zeros_l573_57348

-- Define the function f(x) = cos(x^2)
noncomputable def f (x : ℝ) := Real.cos (x^2)

-- Define the open interval (-2π, 2π)
def interval : Set ℝ := Set.Ioo (-2 * Real.pi) (2 * Real.pi)

-- Theorem statement
theorem cos_x_squared_zeros :
  ∃ (S : Finset ℝ), S.card = 25 ∧ ∀ x ∈ S, x ∈ interval ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_squared_zeros_l573_57348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017th_term_l573_57333

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 / 3 ∧
  ∀ n : ℕ, n ≥ 1 →
    (a (n + 2) - a n ≤ 2^n) ∧
    (a (n + 4) - a n ≥ 5 * 2^n)

theorem sequence_2017th_term (a : ℕ → ℚ) (h : my_sequence a) :
  a 2017 = 2^2017 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017th_term_l573_57333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invention_patents_growth_l573_57338

/-- The annual growth rate of valid invention patents -/
def growth_rate : ℝ := 0.23

/-- The relationship between the number of valid invention patents in 2021 and 2023 -/
theorem invention_patents_growth (a b : ℝ) :
  b = (1 + growth_rate)^2 * a := by
  sorry

#check invention_patents_growth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invention_patents_growth_l573_57338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bing_has_highest_concentration_l573_57349

-- Define the initial sugar water composition
noncomputable def initial_sugar : ℝ := 25
noncomputable def initial_water : ℝ := 100

-- Define the operations for each student
noncomputable def jia_final_sugar : ℝ := initial_sugar + 10
noncomputable def jia_final_water : ℝ := initial_water + 50

noncomputable def yi_final_sugar : ℝ := initial_sugar + 20
noncomputable def yi_final_water : ℝ := initial_water + 50

noncomputable def bing_final_sugar : ℝ := initial_sugar + 40
noncomputable def bing_final_water : ℝ := initial_water + 100

-- Define the concentration calculation function
noncomputable def concentration (sugar : ℝ) (water : ℝ) : ℝ :=
  sugar / (sugar + water) * 100

-- Theorem statement
theorem bing_has_highest_concentration :
  concentration bing_final_sugar bing_final_water >
  max (concentration jia_final_sugar jia_final_water)
      (concentration yi_final_sugar yi_final_water) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bing_has_highest_concentration_l573_57349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_distance_function_l573_57389

-- Define the distance function as noncomputable
noncomputable def d (x y : ℝ) : ℝ := |x - y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

-- State the theorem
theorem d_is_distance_function :
  (∀ x y, d x y ≥ 0) ∧
  (∀ x y, d x y = d y x) ∧
  (∀ x y z, d x y + d y z ≥ d x z) ∧
  (∀ x, d x x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_distance_function_l573_57389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_120_l573_57374

/-- Represents a boat journey with upstream and downstream legs -/
structure BoatJourney where
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the downstream distance for a given boat journey -/
noncomputable def downstream_distance (j : BoatJourney) : ℝ :=
  let boat_speed := j.upstream_distance / j.upstream_time + j.stream_speed
  (boat_speed + j.stream_speed) * j.downstream_time

/-- Theorem stating that for the given conditions, the downstream distance is 120 km -/
theorem downstream_distance_is_120 (j : BoatJourney) 
    (h1 : j.upstream_distance = 60)
    (h2 : j.upstream_time = 2)
    (h3 : j.downstream_time = 2)
    (h4 : j.stream_speed = 15) : 
  downstream_distance j = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_120_l573_57374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_hexagon_probability_is_one_fourth_l573_57335

/-- Represents a regular hexagonal dart board -/
structure HexagonalDartBoard where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The area of a regular hexagon -/
noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * Real.sqrt 3 / 2 * s^2

/-- The probability of a dart landing in the central hexagon -/
noncomputable def central_hexagon_probability (board : HexagonalDartBoard) : ℝ :=
  hexagon_area (board.side_length / 2) / hexagon_area board.side_length

/-- Theorem: The probability of a dart landing in the central hexagon is 1/4 -/
theorem central_hexagon_probability_is_one_fourth (board : HexagonalDartBoard) :
  central_hexagon_probability board = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_hexagon_probability_is_one_fourth_l573_57335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_arrangement_l573_57315

/-- The radius of a circle surrounded by six congruent parabolas -/
noncomputable def circle_radius : ℝ := 3/4

/-- A parabola in the arrangement -/
def parabola (x y : ℝ) : Prop := y = x^2 + circle_radius

/-- The line forming a 60° angle with the horizontal -/
def tangent_line (x y : ℝ) : Prop := y = x * Real.sqrt 3

/-- Theorem stating the tangency condition between the parabola and the line -/
theorem parabola_circle_arrangement :
  ∃ (x : ℝ), (parabola x (x * Real.sqrt 3)) ∧
             (∀ ε > 0, ∃ δ > 0, ∀ x' ≠ x, |x' - x| < δ →
               ¬(parabola x' (x * Real.sqrt 3)) ∧
               ¬(parabola x' (x' * Real.sqrt 3))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_arrangement_l573_57315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_slope_and_distance_l573_57377

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Calculates the distance of a line from the origin --/
noncomputable def distanceFromOrigin (l : Line) : ℝ :=
  |l.yIntercept| / Real.sqrt (1 + l.slope^2)

/-- Theorem: A line with slope 1 and distance 2√2 from the origin has equation x - y ± 4 = 0 --/
theorem line_equation_with_slope_and_distance :
  ∀ (l : Line),
  l.slope = 1 →
  distanceFromOrigin l = 2 * Real.sqrt 2 →
  (∀ (x y : ℝ), x - y + l.yIntercept = 0 ↔ (x - y + 4 = 0 ∨ x - y - 4 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_slope_and_distance_l573_57377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flute_practice_time_l573_57385

/-- Represents the practice schedule for a month -/
structure PracticeSchedule where
  pianoDaysEven : Nat -- Number of even days in a month
  pianoDaysOdd : Nat -- Number of odd days in a month
  pianoTimeEven : Nat -- Piano practice time on even days (in minutes)
  pianoTimeOdd : Nat -- Piano practice time on odd days (in minutes)
  violinMultiplier : Nat -- Violin practice time as a multiple of piano time
  weekendDays : Nat -- Number of weekend days in a month
  fluteRatio : Rat -- Flute practice time as a ratio of violin time

/-- Calculates the total flute practice time in a month -/
def totalFluteTime (schedule : PracticeSchedule) : ℚ :=
  let totalPianoTime := schedule.pianoDaysEven * schedule.pianoTimeEven + 
                        schedule.pianoDaysOdd * schedule.pianoTimeOdd
  let avgDailyPianoTime := (totalPianoTime : ℚ) / (schedule.pianoDaysEven + schedule.pianoDaysOdd)
  let dailyViolinTime := avgDailyPianoTime * schedule.violinMultiplier
  let dailyFluteTime := dailyViolinTime * schedule.fluteRatio
  dailyFluteTime * schedule.weekendDays

/-- The main theorem stating that the total flute practice time is 330 minutes -/
theorem flute_practice_time (schedule : PracticeSchedule) 
  (h1 : schedule.pianoDaysEven = 15)
  (h2 : schedule.pianoDaysOdd = 15)
  (h3 : schedule.pianoTimeEven = 25)
  (h4 : schedule.pianoTimeOdd = 30)
  (h5 : schedule.violinMultiplier = 3)
  (h6 : schedule.weekendDays = 8)
  (h7 : schedule.fluteRatio = 1/2) :
  totalFluteTime schedule = 330 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flute_practice_time_l573_57385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_equivalence_l573_57394

/-- Represents a train journey with specific conditions -/
structure TrainJourney where
  totalDistance : ℝ
  initialSpeed : ℝ
  midpointStopTime : ℝ
  secondHalfSpeed : ℝ

/-- Calculates the total time for a train journey -/
noncomputable def totalTime (journey : TrainJourney) : ℝ :=
  (journey.totalDistance / 2) / journey.initialSpeed +
  journey.midpointStopTime +
  (journey.totalDistance / 2) / journey.secondHalfSpeed

/-- Theorem stating the equivalence of two train journeys -/
theorem train_journey_equivalence :
  ∀ (x : ℝ),
  x > 0 →
  let yesterday : TrainJourney :=
    { totalDistance := 120
      initialSpeed := x
      midpointStopTime := 5 / 60
      secondHalfSpeed := x + 10 }
  let today : TrainJourney :=
    { totalDistance := 120
      initialSpeed := x
      midpointStopTime := 9 / 60
      secondHalfSpeed := 100 }
  totalTime yesterday = 120 / x →
  totalTime today = 120 / x := by
  sorry

#eval "Train journey theorem defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_equivalence_l573_57394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_trapezoid_radius_l573_57322

/-- A trapezoid ABCD inscribed in a circle with BC parallel to AD -/
structure CyclicTrapezoid where
  /-- Length of side BC -/
  a : ℝ
  /-- Length of side AD -/
  b : ℝ
  /-- Angle CAD in radians -/
  α : ℝ
  /-- BC is parallel to AD -/
  parallel : True
  /-- ABCD is inscribed in a circle -/
  inscribed : True

/-- The radius of the circle circumscribing the trapezoid -/
noncomputable def circleRadius (t : CyclicTrapezoid) : ℝ :=
  (Real.sqrt ((t.b - t.a)^2 + (t.b + t.a)^2 * Real.tan t.α^2)) / (4 * Real.sin t.α)

/-- Theorem: The radius of the circle circumscribing the trapezoid is as given -/
theorem cyclic_trapezoid_radius (t : CyclicTrapezoid) :
  circleRadius t = (Real.sqrt ((t.b - t.a)^2 + (t.b + t.a)^2 * Real.tan t.α^2)) / (4 * Real.sin t.α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_trapezoid_radius_l573_57322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_and_range_l573_57368

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (1/3)^(2*x) - (1/3)^(x-2) + 8

theorem function_value_and_range :
  ∀ a : ℝ, a > 0 → a ≠ 1 → f a 3 = 1/9 →
  a = 1/3 ∧
  Set.Icc 4 53 = {y | ∃ x ∈ Set.Icc (-2) 1, g x = y} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_and_range_l573_57368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_length_l573_57301

/-- Calculates the distance traveled in uniformly accelerated motion -/
noncomputable def takeoffDistance (time : ℝ) (finalVelocity : ℝ) : ℝ :=
  let acceleration := finalVelocity / time
  (1 / 2) * acceleration * time^2

/-- Converts km/h to m/s -/
noncomputable def kmhToMs (v : ℝ) : ℝ :=
  v * 1000 / 3600

theorem takeoff_run_length :
  let time : ℝ := 15
  let liftOffSpeed : ℝ := 100
  let distance := takeoffDistance time (kmhToMs liftOffSpeed)
  ⌊distance⌋₊ = 208 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_length_l573_57301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_fill_cost_l573_57325

/-- The cost of filling a cubic sandbox with sand -/
theorem sandbox_fill_cost
  (side_length : ℝ)
  (sandbox_volume : ℝ)
  (bag_volume : ℝ)
  (bag_cost : ℝ)
  (h1 : side_length = 3)
  (h2 : sandbox_volume = side_length ^ 3)
  (h3 : bag_volume = 3)
  (h4 : bag_cost = 4) :
  (sandbox_volume / bag_volume) * bag_cost = 36 := by
  sorry

#check sandbox_fill_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_fill_cost_l573_57325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_interior_diagonals_l573_57326

/-- A dodecahedron is a 3-dimensional figure with 12 pentagonal faces and 20 vertices,
    where 3 faces meet at each vertex. -/
structure Dodecahedron where
  faces : ℕ
  vertices : ℕ
  faces_per_vertex : ℕ
  faces_are_pentagons : faces = 12
  vertex_count : vertices = 20
  three_faces_per_vertex : faces_per_vertex = 3

/-- An interior diagonal is a segment connecting two vertices which do not lie on a common face. -/
def interior_diagonal (d : Dodecahedron) : Finset (ℕ × ℕ) :=
  Finset.filter (fun pair => pair.1 < d.vertices ∧ pair.2 < d.vertices ∧ pair.1 ≠ pair.2)
    (Finset.product (Finset.range d.vertices) (Finset.range d.vertices))

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : ℕ := (interior_diagonal d).card

theorem dodecahedron_interior_diagonals (d : Dodecahedron) : num_interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_interior_diagonals_l573_57326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flip_nine_disproves_claim_l573_57384

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define the visible sides
def visibleSides : List CardSide := [
  CardSide.Letter 'R',
  CardSide.Letter 'S',
  CardSide.Number 2,
  CardSide.Number 9,
  CardSide.Number 7
]

-- Define what it means for a character to be a consonant
def isConsonant (c : Char) : Bool :=
  c.isAlpha && !c.isLower || (c ≠ 'A' && c ≠ 'E' && c ≠ 'I' && c ≠ 'O' && c ≠ 'U')

-- Define what it means for a number to be prime
def isPrime (n : Nat) : Bool :=
  n > 1 && (∀ m, 2 ≤ m → m < n → n % m ≠ 0)

-- Jane's claim
def janesClaim (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) =>
      isConsonant c → isPrime n
  | (CardSide.Number n, CardSide.Letter c) =>
      isConsonant c → isPrime n
  | _ => True

-- Theorem statement
theorem flip_nine_disproves_claim :
  ∀ (cards : List Card),
    cards.length = 5 →
    (∀ card ∈ cards, (card.1 ∈ visibleSides ∨ card.2 ∈ visibleSides)) →
    (∃ card ∈ cards, card.1 = CardSide.Number 9 ∨ card.2 = CardSide.Number 9) →
    (∀ card ∈ cards, janesClaim card) →
    ∃ card ∈ cards,
      (card.1 = CardSide.Number 9 ∨ card.2 = CardSide.Number 9) ∧
      ¬(janesClaim card) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flip_nine_disproves_claim_l573_57384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_specific_circles_l573_57314

/-- The radius of the incircle of a triangle formed by the external tangent points of three circles -/
noncomputable def incircle_radius (r₁ r₂ r₃ : ℝ) : ℝ :=
  (2 * r₁ * r₂ * r₃ * Real.sqrt (r₁ + r₂ + r₃)) /
  (Real.sqrt ((r₁ + r₂) * (r₁ + r₃) * (r₂ + r₃)) *
   (Real.sqrt r₁ * Real.sqrt (r₂ + r₃) + Real.sqrt r₂ * Real.sqrt (r₁ + r₃) + Real.sqrt r₃ * Real.sqrt (r₁ + r₂)))

theorem incircle_radius_specific_circles :
  incircle_radius 2 3 5 = (9 - Real.sqrt 21) / (2 * Real.sqrt 7) ∧
  incircle_radius 2 3 5 = (9 * Real.sqrt 7 - 7 * Real.sqrt 3) / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_specific_circles_l573_57314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_split_to_cone_ratio_l573_57303

-- Define the number of scoops for each ice cream order
def banana_split : ℕ := sorry
def waffle_bowl : ℕ := sorry
def single_cone : ℕ := sorry
def double_cone : ℕ := sorry

-- Define the conditions
axiom single_cone_scoop : single_cone = 1
axiom double_cone_scoop : double_cone = 2 * single_cone
axiom waffle_bowl_scoop : waffle_bowl = banana_split + 1
axiom total_scoops : banana_split + waffle_bowl + single_cone + double_cone = 10

-- Theorem to prove
theorem banana_split_to_cone_ratio :
  banana_split = 3 * single_cone := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_split_to_cone_ratio_l573_57303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_connectivity_l573_57357

/-- Represents the state of connections between islands at a given time. -/
structure IslandConnections (n : ℕ) where
  connections : Fin n → Fin n → Prop

/-- Represents the evolution of island connections over time. -/
def ConnectionEvolution (n : ℕ) :=
  ℕ → IslandConnections n

/-- Initial condition: islands cannot be split into two disconnected subsets. -/
def InitialCondition (n : ℕ) (c : IslandConnections n) : Prop :=
  ∀ (s : Set (Fin n)), s.Nonempty → (s.compl).Nonempty →
    ∃ (i : Fin n) (j : Fin n), i ∈ s ∧ j ∈ s.compl ∧ c.connections i j

/-- Yearly update rule for connections. -/
def YearlyUpdate (n : ℕ) (prev next : IslandConnections n) : Prop :=
  ∃ (x y : Fin n),
    (∀ (i j : Fin n), i ≠ x ∧ i ≠ y ∧ j ≠ x ∧ j ≠ y →
      (next.connections i j ↔ prev.connections i j)) ∧
    (∀ (i : Fin n), i ≠ x ∧ i ≠ y →
      (prev.connections i x ∨ prev.connections i y) →
      ∃ (j : Fin n), j ≠ x ∧ j ≠ y ∧ next.connections i j)

/-- The main theorem to be proved. -/
theorem island_connectivity (n : ℕ) (h : n ≥ 3)
  (evolution : ConnectionEvolution n)
  (initial : InitialCondition n (evolution 0))
  (update : ∀ t, YearlyUpdate n (evolution t) (evolution (t + 1))) :
  ∃ (t : ℕ) (i : Fin n), ∀ (j : Fin n), j ≠ i → (evolution t).connections i j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_connectivity_l573_57357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l573_57396

open Real

-- Define the constants
noncomputable def a : ℝ := sin (13 * π / 180) + cos (13 * π / 180)
noncomputable def b : ℝ := 2 * sqrt 2 * (cos (14 * π / 180))^2 - sqrt 2
noncomputable def c : ℝ := sqrt 6 / 2

-- State the theorem
theorem relationship_abc : a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l573_57396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_nonnegative_l573_57310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sin x + Real.tan x - a * x

theorem tangent_and_nonnegative (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a x ≥ 0 ∧ f a 0 = 0) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a x ≥ 0) →
  a = 3 ∧ a ∈ Set.Iic 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_nonnegative_l573_57310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_range_l573_57360

/-- The ellipse with equation x²/9 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The x-coordinate of the left focus -/
noncomputable def f1_x : ℝ := -Real.sqrt 5

/-- The x-coordinate of the right focus -/
noncomputable def f2_x : ℝ := Real.sqrt 5

/-- The dot product of vectors PF₁ and PF₂ is negative -/
def vectors_dot_product_negative (x y : ℝ) : Prop :=
  (x - f1_x) * (x - f2_x) + y * y < 0

/-- The range of x-coordinates for points on the ellipse satisfying the condition -/
noncomputable def x_range (x : ℝ) : Prop := 
  -3 * Real.sqrt 5 / 5 < x ∧ x < 3 * Real.sqrt 5 / 5

theorem ellipse_x_range (x y : ℝ) : 
  ellipse x y → vectors_dot_product_negative x y → x_range x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_range_l573_57360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_sum_pi_half_iff_trig_identity_l573_57392

theorem acute_angle_sum_pi_half_iff_trig_identity 
  (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π/2) 
  (h_acute_β : 0 < β ∧ β < π/2) : 
  α + β = π/2 ↔ Real.sin α^4 / Real.cos β^2 + Real.cos α^4 / Real.sin β^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_sum_pi_half_iff_trig_identity_l573_57392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_digit_sum_15_l573_57395

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 4

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_number_with_digit_sum_15 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 15 → n ≥ 3444 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_digit_sum_15_l573_57395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt34_l573_57330

/-- A particle moves in a 2D plane. Its position at time t is given by (3t + 8, 5t - 18). -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 8, 5 * t - 18)

/-- The speed of a particle is defined as the magnitude of its velocity vector. -/
noncomputable def particle_speed (pos : ℝ → ℝ × ℝ) : ℝ :=
  let velocity := fun t => (pos (t + 1) - pos t)
  Real.sqrt ((velocity 0).1^2 + (velocity 0).2^2)

/-- The speed of the particle is √34 units of distance per unit of time. -/
theorem particle_speed_is_sqrt34 :
  particle_speed particle_position = Real.sqrt 34 := by
  -- Proof goes here
  sorry

#eval particle_position 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt34_l573_57330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_xOz_correct_l573_57342

/-- Given a point in 3D space, return its symmetrical point with respect to the xOz plane -/
def symmetrical_xOz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

theorem symmetry_xOz_correct :
  let A : ℝ × ℝ × ℝ := (-3, 2, -4)
  symmetrical_xOz A = (-3, -2, -4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_xOz_correct_l573_57342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_subset_B_l573_57351

def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}
def B : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) > 1}

theorem A_subset_B : A ⊆ B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_subset_B_l573_57351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reciprocal_sum_of_zeros_l573_57391

-- Define the piecewise function
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if x > 1 then k * x + 1
  else 0

-- Theorem statement
theorem max_reciprocal_sum_of_zeros (k : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 →
  ∀ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ f k y₁ = 0 ∧ f k y₂ = 0 →
  1 / x₁ + 1 / x₂ ≤ 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reciprocal_sum_of_zeros_l573_57391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l573_57362

-- Define the function (marked as noncomputable due to Real.sqrt and Real.log)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (5*x - 2) / Real.log (1/2))

-- State the theorem
theorem domain_of_f :
  Set.Ioo (2/5 : ℝ) (3/5 : ℝ) ⊆ {x | f x ∈ Set.range f} ∧
  {x | f x ∈ Set.range f} ⊆ Set.Ioc (2/5 : ℝ) (3/5 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l573_57362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_proposition_false_l573_57316

-- Define the functions
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
def g (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + (m - 1) * x^2 + 48 * (m - 2) * x + n

-- Define the propositions
def proposition1 : Prop := ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → |x^3| ≤ |0^3|

def proposition2 : Prop := ∀ (a b c d : ℝ), 
  (∃ (x : ℝ), ∀ (h : ℝ), h ≠ 0 → (f a b c d (x + h) - f a b c d x) * (f a b c d (x - h) - f a b c d x) ≤ 0) 
  ↔ b^2 - 3*a*c > 0

def proposition3 (m n : ℝ) : Prop := 
  (∀ (x : ℝ), g m n (-x) = -g m n x) ∧ 
  (∀ (x y : ℝ), -4 < x ∧ x < y ∧ y < 4 → g m n x > g m n y)

-- The main theorem
theorem only_first_proposition_false : 
  ¬proposition1 ∧ proposition2 ∧ ∃ (m n : ℝ), proposition3 m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_proposition_false_l573_57316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_inequality_l573_57307

theorem no_function_satisfies_inequality :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_inequality_l573_57307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l573_57331

open Set Real

/-- The set of points (r, θ) where r = sin(θ) and 0 ≤ θ ≤ t -/
def sinCircle (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ, 0 ≤ θ ∧ θ ≤ t ∧ p = (Real.sin θ, θ)}

/-- A predicate that checks if a set of points forms a complete circle -/
def isCompleteCircle (s : Set (ℝ × ℝ)) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, r > 0 ∧ s = {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

theorem smallest_t_for_complete_circle :
  ∀ t : ℝ, (t > 0 ∧ isCompleteCircle (sinCircle t)) →
    (∀ s : ℝ, s > 0 ∧ isCompleteCircle (sinCircle s) → t ≤ s) →
    t = π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l573_57331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_cone_volume_l573_57347

-- Define the cone and cylinder
structure Cone where
  radius : ℝ
  height : ℝ

structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the volumes
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height
noncomputable def frustumVolume (bottomRadius topRadius height : ℝ) : ℝ :=
  (1/3) * Real.pi * (bottomRadius^2 + bottomRadius * topRadius + topRadius^2) * height
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

-- State the theorem
theorem inscribed_cylinder_cone_volume 
  (cone : Cone) (cylinder : Cylinder) 
  (h_inscribed : cylinder.radius < cone.radius ∧ cylinder.height < cone.height) 
  (h_cylinder_volume : cylinderVolume cylinder = 9) 
  (h_frustum_volume : frustumVolume cone.radius cylinder.radius cylinder.height = 63) :
  coneVolume cone = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_cone_volume_l573_57347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picks_theorem_l573_57363

/-- A polygon with vertices on integer grid points -/
structure IntegerPolygon where
  vertices : List (Int × Int)
  is_closed : vertices.head? = vertices.getLast?

/-- The number of integer grid points inside the polygon -/
def interior_points (p : IntegerPolygon) : ℕ := sorry

/-- The number of integer grid points on the boundary of the polygon -/
def boundary_points (p : IntegerPolygon) : ℕ := sorry

/-- The area of the polygon -/
noncomputable def area (p : IntegerPolygon) : ℝ := sorry

/-- Pick's Theorem -/
theorem picks_theorem (p : IntegerPolygon) :
  area p = interior_points p + (boundary_points p : ℝ) / 2 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picks_theorem_l573_57363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l573_57327

theorem trigonometric_problem :
  ∀ α β : ℝ,
    Real.sin α = 3/5 →
    Real.cos β = 4/5 →
    α ∈ Set.Ioo (π/2) π →
    β ∈ Set.Ioo 0 (π/2) →
    Real.cos (α + β) = -1 ∧
    (∀ α' β' : ℝ,
      Real.cos α' = 1/7 →
      Real.cos (α' - β') = 13/14 →
      0 < β' →
      β' < α' →
      α' < π/2 →
      β' = π/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l573_57327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_exponents_l573_57339

theorem ordering_of_exponents :
  let y₁ : ℝ := (4 : ℝ) ^ (0.9 : ℝ)
  let y₂ : ℝ := (8 : ℝ) ^ (0.48 : ℝ)
  let y₃ : ℝ := ((1/2) : ℝ) ^ (-(1.1 : ℝ))
  y₁ > y₂ ∧ y₂ > y₃ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_exponents_l573_57339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicularity_theorem_l573_57328

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the perpendicular relationship
def perpendicular : (Line ⊕ Plane) → (Line ⊕ Plane) → Prop := sorry

-- State the theorem
theorem perpendicularity_theorem 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perpendicular (Sum.inl m) (Sum.inl n)) 
  (h4 : perpendicular (Sum.inl m) (Sum.inr α)) 
  (h5 : perpendicular (Sum.inl n) (Sum.inr β)) : 
  perpendicular (Sum.inr α) (Sum.inr β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicularity_theorem_l573_57328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_symmetry_implies_a_greater_than_one_l573_57308

-- Define the function f(x) = e^x - a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

-- Define the property of having odd symmetry points
def has_odd_symmetry_points (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ ≠ 0 ∧ f x₀ = -f (-x₀)

-- Theorem statement
theorem odd_symmetry_implies_a_greater_than_one :
  ∀ a : ℝ, has_odd_symmetry_points (f a) → a > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_symmetry_implies_a_greater_than_one_l573_57308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l573_57397

theorem weight_loss_challenge (W : ℝ) (hW : W > 0) : 
  let weight_after_loss := W * 0.88
  let final_weigh_in := weight_after_loss * 1.02
  (W - final_weigh_in) / W * 100 = 10.24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l573_57397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_bounds_l573_57321

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4*x + 4

theorem extreme_values_and_bounds :
  ∃ (local_max local_min global_max global_min : ℝ),
    (f (-2) = local_max ∧ local_max = 28/3) ∧
    (f 2 = local_min ∧ local_min = -4/3) ∧
    (∀ x ∈ Set.Icc (-3) 4, f x ≤ global_max) ∧
    (global_max = 28/3) ∧
    (∀ x ∈ Set.Icc (-3) 4, f x ≥ global_min) ∧
    (global_min = -4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_bounds_l573_57321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_cosine_l573_57376

/-- Given a geometric sequence {a_n} where a_3 * a_7 = π²/9, cos(a_5) = ±1/2 -/
theorem geometric_sequence_cosine (a : ℕ → ℝ) (h : ∀ n m : ℕ, a (n + m) = a n * (a 1 / a 0) ^ m) 
  (h_prod : a 3 * a 7 = Real.pi^2 / 9) : 
  Real.cos (a 5) = 1/2 ∨ Real.cos (a 5) = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_cosine_l573_57376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_5X_plus_4_l573_57355

-- Define the random variable X
def X : ℝ → ℝ := sorry

-- Define the probability mass function for X
noncomputable def P (x : ℝ) : ℝ := 
  if x = 0 then 0.3
  else if x = 2 then 0.2
  else if x = 4 then 0.5
  else 0

-- Define the expected value of X
noncomputable def E_X : ℝ := 0 * P 0 + 2 * P 2 + 4 * P 4

-- Define the expected value of 5X + 4
noncomputable def E_5X_plus_4 : ℝ := 5 * E_X + 4

-- Theorem to prove
theorem expected_value_5X_plus_4 : E_5X_plus_4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_5X_plus_4_l573_57355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_prove_f_x_l573_57344

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem function_transformation (x : ℝ) : f (x - 1) = x^2 - 1 := by
  sorry

-- Prove that f(x) = x^2 + 2x
theorem prove_f_x (x : ℝ) : f x = x^2 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_prove_f_x_l573_57344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_divisibility_l573_57323

theorem smallest_x_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(11^2021 ∣ 5^(3*y) - 3^(4*y))) ∧ 
  (11^2021 ∣ 5^(3*x) - 3^(4*x)) → 
  x = 11^2020 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_divisibility_l573_57323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l573_57383

noncomputable def f (a : ℝ) (x : ℝ) := Real.cos x + Real.cos (a * x)

theorem function_properties :
  (∀ a : ℝ, ∀ x : ℝ, f a x = f a (-x)) ∧ 
  (∃ S : Set ℚ, Set.Infinite S ∧ ∀ a ∈ S, ∃ x : ℝ, f a x = 2 ∧ ∀ y : ℝ, f a y ≤ 2) ∧
  (∃ a : ℝ, Irrational a ∧ ¬∃ t : ℝ, t ≠ 0 ∧ ∀ x : ℝ, f a (x + t) = f a x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l573_57383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l573_57386

noncomputable def f (x : ℝ) := 2 * Real.sin (x - Real.pi / 6) * Real.cos x + 1 / 2

noncomputable def g (x : ℝ) := Real.sin (2 * x + 2 * Real.pi / 3)

theorem range_of_g_on_interval :
  ∀ y ∈ Set.Icc (-Real.sqrt 3 / 2) 1,
    ∃ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3),
      g x = y ∧
      ∀ x' ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3),
        g x' ∈ Set.Icc (-Real.sqrt 3 / 2) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l573_57386
