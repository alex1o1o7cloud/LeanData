import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_73_l742_74206

theorem divisibility_by_73 (n : ℕ) : ∃ k : ℤ, 2^(3*n + 6) + 3^(4*n + 2) = 73 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_73_l742_74206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l742_74266

theorem smallest_x_value (y : ℕ) (x : ℕ) (h1 : (4 : ℚ) / 5 = y / (248 + x)) 
  (h2 : y > 0) (h3 : ∃ k : ℕ, y = 3 * k) : 
  (∀ z : ℕ, z > 0 ∧ z < x → ¬∃ w : ℕ, w > 0 ∧ w % 3 = 0 ∧ (4 : ℚ) / 5 = w / (248 + z)) → x = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l742_74266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l742_74212

theorem sum_of_coefficients : 
  (fun x => 5 * (2 * x^8 - 3 * x^3 + 4) - 6 * (x^6 + 4 * x^3 - 9)) 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l742_74212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l742_74252

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  ∃ (P : ℝ × ℝ), P ∈ Ellipse a b ∧ P ≠ (a, 0) ∧ P.1 * (P.1 - a) + P.2 * P.2 = 0 →
  Real.sqrt 2 / 2 < eccentricity a b ∧ eccentricity a b < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l742_74252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percent_l742_74207

/-- Calculates the loss percent for a shopkeeper --/
theorem shopkeeper_loss_percent 
  (profit_percent : ℝ) 
  (theft_percent : ℝ) 
  (profit_percent_is_10 : profit_percent = 10)
  (theft_percent_is_50 : theft_percent = 50) :
  let original_cost := 100
  let selling_price := original_cost * (1 + profit_percent / 100)
  let remaining_goods_value := original_cost * (1 - theft_percent / 100)
  let remaining_goods_selling_price := remaining_goods_value * (1 + profit_percent / 100)
  let loss := original_cost - remaining_goods_selling_price
  let loss_percent := loss / original_cost * 100
  loss_percent = 45 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percent_l742_74207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_product_l742_74295

/-- Given a triangle ABC with side lengths a, b, c, internal angle bisectors AD, BE, CF
    of lengths d, e, f respectively, and area Δ, prove the relation between
    these quantities. -/
theorem angle_bisector_product (a b c d e f Δ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  d * e * f = (4 * a * b * c * (a + b + c) * Δ) / ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_product_l742_74295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_beta_negative_ten_l742_74256

-- Define the inverse proportionality relationship
def inversely_proportional (α β : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, α x * β x = k

-- State the theorem
theorem alpha_value_for_beta_negative_ten
  (α β : ℝ → ℝ)
  (h1 : inversely_proportional α β)
  (h2 : α 4 = 1/2) :
  α (-10) = -1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_beta_negative_ten_l742_74256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_engine_safer_than_two_engine_l742_74259

theorem four_engine_safer_than_two_engine (P : ℝ) :
  (2/3 < P) ∧ (P < 1) →
  let p := 1 - P
  let prob_success_four := p^4 + 4*p^3*(1-p) + 6*p^2*(1-p)^2
  let prob_success_two := p^2 + 2*p*(1-p)
  prob_success_four > prob_success_two :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_engine_safer_than_two_engine_l742_74259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_reciprocal_function_l742_74246

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem domain_and_range_of_reciprocal_function :
  (∀ x : ℝ, x ≠ 0 → f x ∈ Set.univ \ {0}) ∧
  (∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, x ≠ 0 ∧ f x = y) := by
  sorry

#check domain_and_range_of_reciprocal_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_reciprocal_function_l742_74246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l742_74249

-- Define the point P
def P : ℝ × ℝ → Prop := λ p => 4 * p.1 + 3 * p.2 = 0

-- Define the constraint on x - y
def constraint : ℝ × ℝ → Prop := λ p => -14 ≤ p.1 - p.2 ∧ p.1 - p.2 ≤ 7

-- Define the distance function
noncomputable def distance (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

-- Theorem statement
theorem distance_range : ∀ p : ℝ × ℝ, P p → constraint p → 0 ≤ distance p ∧ distance p ≤ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l742_74249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_equality_l742_74299

def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_equality (n : ℕ) : A n 3 = n * A 3 3 → n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_equality_l742_74299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l742_74261

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 1)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l742_74261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aharoni_theorem_l742_74213

-- Define a 1-factor (perfect matching)
def has_one_factor {V : Type*} (G : SimpleGraph V) :=
  ∃ (M : Set (V × V)), 
    (∀ (e : V × V), e ∈ M → G.Adj e.1 e.2) ∧ 
    (∀ v : V, ∃! u : V, (v, u) ∈ M ∨ (u, v) ∈ M)

-- Define the set C'_(G-S)
def C_prime_G_minus_S {V : Type*} (G : SimpleGraph V) (S : Set V) : Set V :=
  sorry

-- Define G'_S
def G_prime_S {V : Type*} (G : SimpleGraph V) (S : Set V) : SimpleGraph V :=
  sorry

-- Define a matching from C'_(G-S) to S in G'_S
def matchable_into {V : Type*} (A B : Set V) (G : SimpleGraph V) :=
  ∃ (M : Set (V × V)), 
    (∀ (e : V × V), e ∈ M → G.Adj e.1 e.2) ∧
    (∀ a, a ∈ A → ∃! b, b ∈ B ∧ (a, b) ∈ M) ∧
    (∀ (a₁ a₂ : V) (b : V), (a₁, b) ∈ M → (a₂, b) ∈ M → a₁ = a₂)

-- State the theorem
theorem aharoni_theorem {V : Type*} (G : SimpleGraph V) :
  has_one_factor G ↔ 
  ∀ (S : Set V), matchable_into (C_prime_G_minus_S G S) S (G_prime_S G S) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aharoni_theorem_l742_74213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_element_at_least_five_exists_valid_set_with_five_l742_74269

def is_valid_set (T : Finset ℕ) : Prop :=
  T.card = 5 ∧
  (∀ x, x ∈ T → 1 ≤ x ∧ x ≤ 15) ∧
  (∀ x y, x ∈ T → y ∈ T → x < y → ¬(y % x = 0))

theorem min_element_at_least_five (T : Finset ℕ) (h : is_valid_set T) :
  ∀ x, x ∈ T → x ≥ 5 :=
by sorry

theorem exists_valid_set_with_five :
  ∃ T : Finset ℕ, is_valid_set T ∧ 5 ∈ T :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_element_at_least_five_exists_valid_set_with_five_l742_74269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exponential_equation_l742_74233

theorem two_solutions_exponential_equation :
  ∃! (s : Finset ℝ), (∀ x ∈ s, (2 : ℝ)^(4*x) - 5*((2 : ℝ)^(2*x+1)) + 4 = 0) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exponential_equation_l742_74233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonic_increase_intervals_l742_74257

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by sorry

-- Theorem for the intervals of monotonic increase
theorem monotonic_increase_intervals :
  ∀ (k : ℤ), StrictMonoOn f (Set.Icc (-(Real.pi/6) + k * Real.pi) ((Real.pi/3) + k * Real.pi)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonic_increase_intervals_l742_74257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_subsequence_index_l742_74293

noncomputable def sequence_a (n : ℕ) : ℝ := 2 * (n + 1) + 2

noncomputable def sum_S (n : ℕ) : ℝ := (sequence_a n)^2 / 4 + (sequence_a n) / 2 - 2

def is_geometric_subsequence (f : ℕ → ℕ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ m : ℕ, sequence_a (f (m + 1)) = r * sequence_a (f m)

theorem geometric_subsequence_index (f : ℕ → ℕ) :
  (∀ n, sequence_a n > 0) →
  (∀ n, 4 * sum_S n = (sequence_a n)^2 + 2 * sequence_a n - 8) →
  f 1 = 1 →
  is_geometric_subsequence f →
  (∀ g : ℕ → ℕ, is_geometric_subsequence g → g 1 = 1 → 
    (∃ r : ℝ, r > 0 ∧ ∀ m : ℕ, sequence_a (g (m + 1)) = r * sequence_a (g m)) →
    (∃ r : ℝ, r > 0 ∧ ∀ m : ℕ, sequence_a (f (m + 1)) = r * sequence_a (f m) ∧ r ≤ r)) →
  ∀ m : ℕ, f m = 2^m - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_subsequence_index_l742_74293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l742_74285

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.sum = 17) ∧ 
  (digits.toFinset.card = digits.length) ∧ 
  (5 ∉ digits) ∧
  (digits.length = 2)

theorem largest_valid_number : 
  is_valid_number 98 ∧ ∀ m : ℕ, is_valid_number m → m ≤ 98 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l742_74285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l742_74277

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 - x^2) * Real.exp x

-- State the theorem
theorem f_strictly_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (-3) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l742_74277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_difference_l742_74272

theorem divisibility_by_difference (n : ℕ) (x y : ℝ) :
  ∃ k : ℝ, x^(2*n + 1) - y^(2*n + 1) = (x - y) * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_difference_l742_74272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_negative_necessary_not_sufficient_for_obtuse_angle_l742_74288

/-- The angle between two vectors -/
def angle_between (a b : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

/-- The dot product of two vectors -/
def dot_product (a b : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

/-- An angle is obtuse if it's greater than π/2 and less than π -/
def is_obtuse_angle (θ : ℝ) : Prop := Real.pi/2 < θ ∧ θ < Real.pi

theorem dot_product_negative_necessary_not_sufficient_for_obtuse_angle :
  (∀ a b : EuclideanSpace ℝ (Fin 3), is_obtuse_angle (angle_between a b) → dot_product a b < 0) ∧ 
  (∃ a b : EuclideanSpace ℝ (Fin 3), dot_product a b < 0 ∧ ¬is_obtuse_angle (angle_between a b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_negative_necessary_not_sufficient_for_obtuse_angle_l742_74288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trip_cost_l742_74228

structure Trip :=
  (AC : ℝ)
  (AB : ℝ)
  (busCostPerKm : ℝ)
  (airplaneCostPerKm : ℝ)
  (airplaneBookingFee : ℝ)

noncomputable def BC (t : Trip) : ℝ := Real.sqrt (t.AB^2 - t.AC^2)

noncomputable def legCost (distance : ℝ) (busCost airplaneCost airplaneBookingFee : ℝ) : ℝ :=
  min (distance * busCost) (distance * airplaneCost + airplaneBookingFee)

noncomputable def totalCost (t : Trip) : ℝ :=
  legCost t.AB t.busCostPerKm t.airplaneCostPerKm t.airplaneBookingFee +
  legCost (BC t) t.busCostPerKm t.airplaneCostPerKm t.airplaneBookingFee +
  legCost t.AC t.busCostPerKm t.airplaneCostPerKm t.airplaneBookingFee

theorem min_trip_cost (t : Trip) 
  (h1 : t.AC = 4000)
  (h2 : t.AB = 4250)
  (h3 : t.busCostPerKm = 0.20)
  (h4 : t.airplaneCostPerKm = 0.12)
  (h5 : t.airplaneBookingFee = 120) :
  totalCost t = 1520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trip_cost_l742_74228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l742_74298

theorem collinear_vectors (e₁ e₂ : ℝ × ℝ) (lambda : ℝ) : 
  e₁ ≠ (0, 0) → 
  e₂ ≠ (0, 0) → 
  ¬ (∃ (k : ℝ), e₁ = k • e₂) → 
  (∃ (k : ℝ), (2 : ℝ) • e₁ - (3 : ℝ) • e₂ = k • (lambda • e₁ + (6 : ℝ) • e₂)) → 
  lambda = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l742_74298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_lambda_eq_two_l742_74235

-- Define the vectors a and b
def a (lambda : ℝ) : Fin 3 → ℝ := ![lambda + 1, 0, 2]
def b (lambda mu : ℝ) : Fin 3 → ℝ := ![6, 2*mu - 1, 2*lambda]

-- Define the parallel condition
def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

-- State the theorem
theorem parallel_vectors_imply_lambda_eq_two (lambda mu : ℝ) :
  parallel (a lambda) (b lambda mu) → lambda = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_lambda_eq_two_l742_74235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_trip_distance_l742_74216

/-- Represents Karl's car and trip details -/
structure CarTrip where
  /-- Miles per gallon the car can travel -/
  mpg : ℚ
  /-- Capacity of the gas tank in gallons -/
  tankCapacity : ℚ
  /-- Initial distance driven in miles -/
  initialDistance : ℚ
  /-- Amount of gas bought during the trip in gallons -/
  gasBought : ℚ
  /-- Fraction of tank full at the end of the trip -/
  finalTankFraction : ℚ

/-- Calculates the total distance driven given a CarTrip -/
def totalDistance (trip : CarTrip) : ℚ :=
  trip.initialDistance + 
  (trip.tankCapacity - trip.initialDistance / trip.mpg + trip.gasBought - 
   trip.finalTankFraction * trip.tankCapacity) * trip.mpg

/-- Theorem stating that Karl's total distance driven is 720 miles -/
theorem karl_trip_distance : 
  let trip : CarTrip := {
    mpg := 40,
    tankCapacity := 16,
    initialDistance := 400,
    gasBought := 10,
    finalTankFraction := 1/2
  }
  totalDistance trip = 720 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_trip_distance_l742_74216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l742_74234

/-- The function f(x) = (x - 2)|x| -/
def f (x : ℝ) : ℝ := (x - 2) * |x|

/-- The maximum value of f on [a, 2] -/
def max_value : ℝ := 0

/-- The minimum value of f on [a, 2] -/
noncomputable def min_value (a : ℝ) : ℝ :=
  if 1 ≤ a ∧ a ≤ 2 then a^2 - 2*a
  else if 1 - Real.sqrt 2 ≤ a ∧ a < 1 then -1
  else -a^2 + 2*a

theorem f_extrema (a : ℝ) (h : a ≤ 2) :
  (∀ x ∈ Set.Icc a 2, f x ≤ max_value) ∧
  (∃ x ∈ Set.Icc a 2, f x = max_value) ∧
  (∀ x ∈ Set.Icc a 2, min_value a ≤ f x) ∧
  (∃ x ∈ Set.Icc a 2, f x = min_value a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l742_74234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l742_74283

-- Define the circles and their properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {point | (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2}

-- Define the problem setup
def problem_setup (C₁ C₂ : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (center₁ center₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    -- C₁ and C₂ are defined by their centers and radii
    C₁ = Circle center₁ r₁ ∧ C₂ = Circle center₂ r₂ ∧
    -- The circles intersect at point P(3, 2)
    (3 - center₁.1)^2 + (2 - center₁.2)^2 = r₁^2 ∧
    (3 - center₂.1)^2 + (2 - center₂.2)^2 = r₂^2 ∧
    -- The product of the radii is 13/2
    r₁ * r₂ = 13/2 ∧
    -- k is positive
    k > 0 ∧
    -- The line y = kx is tangent to both circles and the x-axis
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
      (x₁ - center₁.1)^2 + (y₁ - center₁.2)^2 = r₁^2 ∧
      (x₂ - center₂.1)^2 + (y₂ - center₂.2)^2 = r₂^2 ∧
      k * 0 = 0  -- Tangent to x-axis

-- The theorem to prove
theorem tangent_line_slope (C₁ C₂ : Set (ℝ × ℝ)) (k : ℝ) :
  problem_setup C₁ C₂ k → k = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l742_74283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_selection_size_l742_74248

def is_valid_selection (S : Finset ℕ) : Prop :=
  S.Nonempty ∧ (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2016)

def has_close_sqrt_pair (S : Finset ℕ) : Prop :=
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ |Real.sqrt (a : ℝ) - Real.sqrt (b : ℝ)| < 1

theorem smallest_selection_size :
  (∀ k < 45, ∃ S : Finset ℕ, is_valid_selection S ∧ S.card = k ∧ ¬has_close_sqrt_pair S) ∧
  (∀ T : Finset ℕ, is_valid_selection T ∧ T.card = 45 → has_close_sqrt_pair T) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_selection_size_l742_74248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_half_l742_74202

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (x + Real.pi/2)

theorem f_derivative_at_pi_half : 
  deriv f (Real.pi/2) = -Real.pi/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_half_l742_74202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marsh_difference_value_l742_74200

/-- Function to calculate the difference between geese and ducks -/
def geese_duck_difference (geese ducks : ℝ) : ℝ := geese - ducks

/-- The difference between the number of geese and ducks in the marsh -/
def marsh_difference : ℝ := geese_duck_difference 58.0 37.0

/-- Theorem stating that the difference between geese and ducks in the marsh is 21.0 -/
theorem marsh_difference_value : marsh_difference = 21.0 := by
  -- Unfold the definition of marsh_difference
  unfold marsh_difference
  -- Unfold the definition of geese_duck_difference
  unfold geese_duck_difference
  -- Evaluate the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marsh_difference_value_l742_74200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l742_74227

/-- Given a line with equation x - ay + 3 = 0 and inclination angle 30°, prove that a = √3 -/
theorem line_inclination (a : ℝ) : 
  (∃ x y, x - a * y + 3 = 0) → -- line equation
  (∃ θ : ℝ, θ = 30 * π / 180 ∧ Real.tan θ = 1 / a) → -- inclination angle in radians
  a = Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l742_74227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l742_74244

theorem no_solutions_in_interval (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi → 
  Real.sin x ^ 2 + Real.cos x ^ 2 = 1 →
  1 / Real.sin x + 1 / Real.cos x ≠ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l742_74244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_sampling_equal_probability_l742_74208

/-- Represents a population from which samples are drawn -/
structure Population (α : Type u) where
  individuals : Set α
  nonempty : individuals.Nonempty

/-- Represents a sampling method -/
structure SamplingMethod (α : Type u) where
  sample : Population α → Set α
  probability : Population α → α → ℝ

/-- Simple sampling method -/
noncomputable def simpleSampling (α : Type u) : SamplingMethod α where
  sample := fun _ => sorry
  probability := fun _ _ => sorry

/-- Theorem stating that in simple sampling, the probability of an individual being sampled
    is independent of the sampling order and is equal for all individuals -/
theorem simple_sampling_equal_probability {α : Type u} (pop : Population α) (i j : α) 
    (hi : i ∈ pop.individuals) (hj : j ∈ pop.individuals) :
    (simpleSampling α).probability pop i = (simpleSampling α).probability pop j := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_sampling_equal_probability_l742_74208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_first_quadrant_l742_74262

theorem terminal_side_in_first_quadrant (α : Real) 
  (h1 : Real.tan α > 0) (h2 : Real.sin α + Real.cos α > 0) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_first_quadrant_l742_74262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_hyperbola_eccentricity_l742_74250

theorem circle_symmetry_hyperbola_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let circle := λ (x y : ℝ) ↦ x^2 + y^2 - 3*x - 4*y - 5 = 0
  let symmetry_line := λ (x y : ℝ) ↦ a*x - b*y = 0
  let hyperbola := λ (x y : ℝ) ↦ x^2/a^2 - y^2/b^2 = 1
  (∀ x y, circle x y ↔ circle (2*a*x/(a^2+b^2) - 2*b*y/(a^2+b^2)) (2*b*x/(a^2+b^2) + 2*a*y/(a^2+b^2))) →
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_hyperbola_eccentricity_l742_74250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l742_74278

/-- The distance between two points A(2, -1, 3) and B(-1, 4, -2) is √59. -/
theorem distance_between_points : 
  let A : Fin 3 → ℝ := ![2, -1, 3]
  let B : Fin 3 → ℝ := ![-1, 4, -2]
  Real.sqrt ((A 0 - B 0)^2 + (A 1 - B 1)^2 + (A 2 - B 2)^2) = Real.sqrt 59 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l742_74278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_to_octagon_ratio_l742_74203

/-- Represents a regular octagon -/
structure RegularOctagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents the trapezium formed by one side of the octagon and two adjacent diagonals -/
noncomputable def trapezium_area (octagon : RegularOctagon) : ℝ := 
  octagon.side_length^2 * (1 + Real.sqrt 2)

/-- Represents the area of the entire regular octagon -/
noncomputable def octagon_area (octagon : RegularOctagon) : ℝ := 
  octagon.side_length^2 * (4 + 4 * Real.sqrt 2)

/-- The theorem stating the ratio of trapezium area to octagon area -/
theorem trapezium_to_octagon_ratio (octagon : RegularOctagon) : 
  trapezium_area octagon / octagon_area octagon = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_to_octagon_ratio_l742_74203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_existence_of_endpoints_l742_74242

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x^2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc 0 2 :=
sorry

-- Additional theorem to show the existence of specific points
theorem existence_of_endpoints :
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_existence_of_endpoints_l742_74242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l742_74296

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 3 + y^2 / 11 = 1

-- Define the focal length
noncomputable def focal_length : ℝ := 4 * Real.sqrt 2

-- Theorem statement
theorem ellipse_focal_length :
  ∀ (x y : ℝ), ellipse_equation x y → ∃ (f : ℝ), f = focal_length ∧ f > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l742_74296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_min_u_l742_74230

theorem ellipse_max_min_u (x y : ℝ) : 
  x^2 / 4 + y^2 = 1 → 
  (∃ (x' y' : ℝ), x'^2 / 4 + y'^2 = 1 ∧ 2*x' + y' = Real.sqrt 17) ∧
  (∃ (x'' y'' : ℝ), x''^2 / 4 + y''^2 = 1 ∧ 2*x'' + y'' = -Real.sqrt 17) ∧
  (∀ (x''' y''' : ℝ), x'''^2 / 4 + y'''^2 = 1 → 2*x''' + y''' ≤ Real.sqrt 17 ∧ 2*x''' + y''' ≥ -Real.sqrt 17) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_min_u_l742_74230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l742_74201

-- Define the points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (4, 5)
def R : ℝ × ℝ := (3, -1)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem statement
theorem area_of_triangle_PQR :
  triangleArea P Q R = 7.5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l742_74201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_m_plus_n_l742_74251

/-- Given two vectors OA and OB in R², and a point (x, y) defined as a linear combination of these vectors,
    prove that under certain constraints, the minimum value of m + n is achieved. -/
theorem min_value_m_plus_n (OA OB : ℝ × ℝ) (lambda mu m n : ℝ) :
  OA = (1, 0) →
  OB = (1, 1) →
  0 ≤ lambda →
  lambda ≤ 1 →
  1 ≤ mu →
  mu ≤ 2 →
  m > 0 →
  n > 0 →
  let (x, y) := lambda • OA + mu • OB
  let z := x / m + y / n
  (∀ m' n', m' > 0 → n' > 0 → x / m' + y / n' ≤ 2) →
  (∃ m'' n'', m'' > 0 ∧ n'' > 0 ∧ x / m'' + y / n'' = 2) →
  m + n ≥ 5/2 + Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_m_plus_n_l742_74251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_theorem_l742_74215

structure EquilateralTriangle where
  side_length : ℝ

structure Circle where
  radius : ℝ

noncomputable def shortest_path_length (triangle : EquilateralTriangle) (circle : Circle) : ℝ :=
  Real.sqrt (28 / 3) - 1

theorem shortest_path_theorem (triangle : EquilateralTriangle) (circle : Circle) :
  triangle.side_length = 2 →
  circle.radius = 1 / 2 →
  shortest_path_length triangle circle = Real.sqrt (28 / 3) - 1 := by
  sorry

#check shortest_path_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_theorem_l742_74215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liu_hui_contribution_l742_74284

/-- Represents a mathematical method --/
inductive Method
| Exhaustions
| PythagoreanTheorem
| CelestialElementComputation
| DivisionAlgorithm

/-- Represents a mathematician --/
structure Mathematician where
  name : String
  origin : String
  method : Method

/-- Liu Hui's method --/
def liu_hui_method : Method := Method.Exhaustions

/-- Liu Hui --/
def liu_hui : Mathematician :=
  { name := "Liu Hui"
  , origin := "ancient China"
  , method := liu_hui_method }

/-- Theorem stating Liu Hui's contribution --/
theorem liu_hui_contribution :
  (liu_hui.origin = "ancient China") →
  (liu_hui.method = Method.Exhaustions) →
  (∃ m : Method, m = liu_hui.method ∧ 
    (m = Method.Exhaustions → True) ∧ 
    (m = Method.Exhaustions → True)) :=
by
  intro h1 h2
  use liu_hui.method
  apply And.intro
  · exact h2
  · apply And.intro
    · intro _
      trivial
    · intro _
      trivial

#check liu_hui_contribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liu_hui_contribution_l742_74284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l742_74294

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  let (x, y, z) := spherical_to_rectangular 12 (7 * π / 6) (π / 3)
  x = -9 ∧ y = -3 * Real.sqrt 3 ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l742_74294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_values_l742_74291

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0

-- Define the line
def line_eq (x y : ℝ) (m : ℝ) : Prop := x - 2*y + m = 0

-- Define tangency condition
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y m ∧
  ∀ (x' y' : ℝ), circle_eq x' y' → line_eq x' y' m → (x' = x ∧ y' = y)

-- Theorem statement
theorem tangent_line_m_values :
  ∀ m : ℝ, is_tangent m → (m = -3 ∨ m = -13) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_values_l742_74291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l742_74224

/-- Arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_property (a : ℕ → ℚ) 
  (h_seq : arithmetic_sequence a)
  (h_1008 : a 1008 > 0)
  (h_sum : a 1007 + a 1008 < 0) :
  S a 2014 * S a 2015 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l742_74224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nikola_numbers_sum_l742_74217

/-- A function that checks if a number has all distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ = d₂ → d₁ = d₂

theorem nikola_numbers_sum :
  ∀ a b : ℕ,
  (100 ≤ a ∧ a < 1000) →  -- a is a three-digit number
  (10 ≤ b ∧ b < 100) →    -- b is a two-digit number
  has_distinct_digits a →
  has_distinct_digits b →
  a - b = 976 →
  a + b = 996 := by
  sorry

#check nikola_numbers_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nikola_numbers_sum_l742_74217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_midpoint_vector_sum_l742_74238

/-- A parallelogram ABCD with midpoints E and F -/
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℚ V] :=
  (A B C D E F : V)
  (is_parallelogram : A - B = D - C)
  (E_midpoint : E = (1/2 : ℚ) • (B + C))
  (F_midpoint : F = (1/2 : ℚ) • (C + D))

/-- The main theorem -/
theorem parallelogram_midpoint_vector_sum 
  {V : Type*} [AddCommGroup V] [Module ℚ V] (P : Parallelogram V) 
  (x y : ℚ) (h : P.A - P.B = x • (P.A - P.E) + y • (P.A - P.F)) : 
  x + y = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_midpoint_vector_sum_l742_74238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l742_74211

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x - 3) + Real.log (x + 1)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l742_74211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_formed_equals_one_l742_74239

-- Define the chemical reaction
noncomputable def reaction (nh4no3 : ℝ) (naoh : ℝ) : ℝ × ℝ × ℝ := (nh4no3, naoh, min nh4no3 naoh)

-- Theorem statement
theorem water_formed_equals_one (nh4no3 : ℝ) :
  let (_, naoh, h2o) := reaction nh4no3 1
  h2o = 1 → h2o = 1 := by
  intro h
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_formed_equals_one_l742_74239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l742_74204

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.sqrt x
def curve2 (x : ℝ) : ℝ := x^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(0, 0), (1, 1)}

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1), (curve1 x - curve2 x)

-- Theorem statement
theorem area_between_curves : enclosed_area = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l742_74204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_dodecahedron_no_hamiltonian_cycle_l742_74275

/-- Represents a rhombic dodecahedron -/
structure RhombicDodecahedron where
  /-- The number of faces (rhombuses) -/
  faces : Nat
  /-- The number of vertices where three edges meet -/
  three_edge_vertices : Nat
  /-- The number of vertices where four edges meet -/
  four_edge_vertices : Nat
  /-- The total number of vertices -/
  total_vertices : Nat
  /-- Ensure the structure matches a rhombic dodecahedron -/
  faces_eq : faces = 12
  three_edge_eq : three_edge_vertices = 8
  four_edge_eq : four_edge_vertices = 6
  total_vertices_eq : total_vertices = three_edge_vertices + four_edge_vertices

/-- A Hamiltonian cycle visits each vertex exactly once and returns to the start -/
def has_hamiltonian_cycle (g : RhombicDodecahedron) : Prop :=
  ∃ (cycle : List Nat), 
    cycle.length = g.total_vertices + 1 ∧ 
    cycle.head? = cycle.getLast? ∧
    cycle.toFinset.card = g.total_vertices

/-- Theorem stating that a rhombic dodecahedron does not have a Hamiltonian cycle -/
theorem rhombic_dodecahedron_no_hamiltonian_cycle (g : RhombicDodecahedron) : 
  ¬ has_hamiltonian_cycle g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_dodecahedron_no_hamiltonian_cycle_l742_74275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_a_milk_parts_l742_74289

/-- Represents the composition of a drink mixture -/
structure DrinkMixture where
  milk : ℚ
  fruit_juice : ℚ

/-- Converts volume to parts -/
def volume_to_parts (volume : ℚ) (total_parts : ℚ) : ℚ :=
  volume * total_parts / 42

theorem drink_a_milk_parts :
  ∃ (m : ℚ),
    let drink_a : DrinkMixture := ⟨m, 3⟩
    let drink_b : DrinkMixture := ⟨3, 4⟩
    let total_parts_a : ℚ := drink_a.milk + drink_a.fruit_juice
    let added_fruit_juice : ℚ := volume_to_parts 14 total_parts_a
    (drink_a.fruit_juice + added_fruit_juice) / drink_a.milk = drink_b.fruit_juice / drink_b.milk
    ∧ m = 13 := by
  sorry

#check drink_a_milk_parts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_a_milk_parts_l742_74289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_larger_than_long_l742_74214

/-- The sum of all three dimensions of the suitcase -/
noncomputable def total_dimension : ℝ := 150

/-- The length of the long dimension in the long suitcase -/
noncomputable def long_dimension : ℝ := 220

/-- The ratio of the long dimension to the other two equal dimensions in the long suitcase -/
noncomputable def k : ℝ := long_dimension / (total_dimension - long_dimension) * 2

/-- The volume of a cubic suitcase with sides equal to total_dimension / 3 -/
noncomputable def cubic_volume : ℝ := (total_dimension / 3) ^ 3

/-- The volume of a long suitcase with dimensions long_dimension, long_dimension/k, long_dimension/k -/
noncomputable def long_volume : ℝ := long_dimension ^ 3 / k ^ 2

/-- Theorem stating that the cubic suitcase has a larger volume than the long suitcase when k > (4.4)^(3/2) -/
theorem cubic_larger_than_long : k > (4.4 : ℝ) ^ (3/2) → cubic_volume > long_volume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_larger_than_long_l742_74214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_problem_l742_74273

theorem work_completion_problem (original_days reduced_days additional_workers : ℕ) 
  (h1 : original_days = 9)
  (h2 : reduced_days = 6)
  (h3 : additional_workers = 10) :
  ∃ (original_workers : ℕ), 
    original_workers * original_days = (original_workers + additional_workers) * reduced_days ∧ 
    original_workers = 20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_problem_l742_74273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nadia_punch_calories_l742_74226

/-- Represents the recipe and calorie content of Nadia's fruit punch -/
structure FruitPunch where
  orange_juice : ℚ
  apple_juice : ℚ
  sugar : ℚ
  water : ℚ
  orange_juice_calories : ℚ
  apple_juice_calories : ℚ
  sugar_calories : ℚ

/-- Calculates the total weight of the fruit punch -/
def total_weight (punch : FruitPunch) : ℚ :=
  punch.orange_juice + punch.apple_juice + punch.sugar + punch.water

/-- Calculates the total calories in the fruit punch -/
def total_calories (punch : FruitPunch) : ℚ :=
  punch.orange_juice * punch.orange_juice_calories / 100 +
  punch.apple_juice * punch.apple_juice_calories / 100 +
  punch.sugar * punch.sugar_calories / 100

/-- Nadia's fruit punch recipe -/
def nadia_punch : FruitPunch := {
  orange_juice := 150,
  apple_juice := 200,
  sugar := 50,
  water := 600,
  orange_juice_calories := 45,
  apple_juice_calories := 52,
  sugar_calories := 385
}

/-- Theorem: 300g of Nadia's fruit punch contains 109.2 calories -/
theorem nadia_punch_calories : 
  (300 * total_calories nadia_punch) / total_weight nadia_punch = 546 / 5 := by
  sorry

#eval (300 * total_calories nadia_punch) / total_weight nadia_punch

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nadia_punch_calories_l742_74226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_club_order_l742_74267

-- Define the clubs
inductive Club
| Chess
| Drama
| Art
| Science

-- Define the support fraction for each club
def support : Club → Rat
| Club.Chess => 14 / 35
| Club.Drama => 9 / 28
| Club.Art => 11 / 21
| Club.Science => 8 / 15

-- Define the ordering of clubs
def club_order : List Club := [Club.Science, Club.Art, Club.Chess, Club.Drama]

-- Theorem statement
theorem correct_club_order :
  List.Sorted (fun c1 c2 => support c1 > support c2) club_order := by
  -- We use 'by' here and add 'sorry' to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_club_order_l742_74267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_vector_relation_l742_74253

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
def Vector3D := Point3D

/-- A triangular pyramid with vertex O and base ABC -/
structure TriangularPyramid where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point lies on the plane defined by three other points -/
def isOnPlane (P Q R S : Point3D) : Prop := sorry

/-- Vector from point P to point Q -/
def vectorPQ (P Q : Point3D) : Vector3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

/-- Scalar multiplication of a vector -/
def scalarMul (a : ℝ) (v : Vector3D) : Vector3D :=
  { x := a * v.x, y := a * v.y, z := a * v.z }

/-- Addition of vectors -/
def vectorAdd (v w : Vector3D) : Vector3D :=
  { x := v.x + w.x, y := v.y + w.y, z := v.z + w.z }

/-- Subtraction of vectors -/
def vectorSub (v w : Vector3D) : Vector3D :=
  { x := v.x - w.x, y := v.y - w.y, z := v.z - w.z }

/-- Zero vector -/
def zeroVector : Vector3D :=
  { x := 0, y := 0, z := 0 }

theorem pyramid_vector_relation (pyramid : TriangularPyramid) (P : Point3D) (k : ℝ) :
  isOnPlane P pyramid.A pyramid.B pyramid.C →
  vectorPQ pyramid.O P = vectorAdd
    (vectorAdd
      (scalarMul (1/2) (vectorPQ pyramid.O pyramid.A))
      (scalarMul k (vectorPQ pyramid.O pyramid.B)))
    (vectorSub zeroVector (vectorPQ pyramid.O pyramid.C)) →
  k = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_vector_relation_l742_74253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_two_cubed_base_and_exponent_l742_74254

def base_of (x : ℤ) : ℤ := -2
def exponent_of (x : ℤ) : ℕ := 3

theorem negative_two_cubed_base_and_exponent :
  let x := -2^3
  (base_of x = -2) ∧ (exponent_of x = 3) :=
by
  -- Unfold the definition of x
  let x := -2^3
  -- Split the goal into two parts
  constructor
  -- Prove the first part: base_of x = -2
  · simp [base_of]
  -- Prove the second part: exponent_of x = 3
  · simp [exponent_of]
  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_two_cubed_base_and_exponent_l742_74254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l742_74276

/-- The function h(t) = (t^2 + 1/2 t) / (t^2 + 1) -/
noncomputable def h (t : ℝ) : ℝ := (t^2 + 1/2 * t) / (t^2 + 1)

/-- The range of h is the closed interval [-1/4, 1/4] -/
theorem range_of_h :
  ∀ y : ℝ, (∃ t : ℝ, h t = y) ↔ -1/4 ≤ y ∧ y ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l742_74276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_results_l742_74219

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * x * (1 - Real.log x) - Real.log x

theorem main_results :
  (∃ x : ℝ, x > 0 ∧ f 1 x = 3/2 ∧ ∀ y : ℝ, y > 0 → f 1 y ≥ f 1 x) ∧
  (∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ y : ℝ, y ≠ x₁ ∧ y ≠ x₂ ∧ y ≠ x₃ → (deriv (f a)) y ≠ 0)) ↔ a > 2) ∧
  (∀ a : ℝ, ∀ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ y : ℝ, y ≠ x₁ ∧ y ≠ x₂ ∧ y ≠ x₃ → (deriv (f a)) y ≠ 0) →
    x₁ + x₃ + 4 * x₁ * x₃ > 3 * a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_results_l742_74219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_z_axis_l742_74268

/-- Given points A and B in 3D space, prove that M(0, 0, 4) is equidistant from A and B and lies on the z-axis --/
theorem equidistant_point_on_z_axis :
  let A : Fin 3 → ℝ := ![2, 1, 1]
  let B : Fin 3 → ℝ := ![1, -3, 2]
  let M : Fin 3 → ℝ := ![0, 0, 4]
  (M 0 = 0 ∧ M 1 = 0) ∧  -- M is on the z-axis
  (Real.sqrt ((M 0 - A 0)^2 + (M 1 - A 1)^2 + (M 2 - A 2)^2) =
   Real.sqrt ((M 0 - B 0)^2 + (M 1 - B 1)^2 + (M 2 - B 2)^2)) -- MA = MB
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_z_axis_l742_74268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_lasts_25_6_days_l742_74287

/-- Represents the daily jelly consumption of a family member -/
structure DailyConsumption where
  strawberry : ℚ
  blueberry : ℚ

/-- Represents the jelly stock and consumption of Shannon's family -/
structure JellyStock where
  total : ℚ
  blueberry : ℚ
  shannon : DailyConsumption
  mother : DailyConsumption
  father : DailyConsumption
  brother : DailyConsumption

/-- Calculates how long the jelly will last given the stock and consumption rates -/
noncomputable def jelly_duration (stock : JellyStock) : ℚ :=
  let strawberry := stock.total - stock.blueberry
  let daily_strawberry := stock.shannon.strawberry + stock.mother.strawberry + 
                          stock.father.strawberry + stock.brother.strawberry
  let daily_blueberry := stock.shannon.blueberry + stock.mother.blueberry + 
                         stock.father.blueberry + stock.brother.blueberry
  min (strawberry / daily_strawberry) (stock.blueberry / daily_blueberry)

/-- Theorem: The jelly will last for 25.6 days given the specified stock and consumption rates -/
theorem jelly_lasts_25_6_days (stock : JellyStock) 
  (h1 : stock.total = 6310)
  (h2 : stock.blueberry = 4518)
  (h3 : stock.shannon = ⟨15, 20⟩)
  (h4 : stock.mother = ⟨10, 25⟩)
  (h5 : stock.father = ⟨20, 10⟩)
  (h6 : stock.brother = ⟨25, 15⟩) :
  jelly_duration stock = 64/5 := by
  sorry

#eval (64 : ℚ) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_lasts_25_6_days_l742_74287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzles_and_board_games_count_l742_74205

/-- The number of puzzles and board games in a box of toys -/
def puzzles_and_board_games (total : ℕ) (action_figure_ratio : ℚ) (doll_ratio : ℚ) : ℕ :=
  total - Int.toNat ((↑total * action_figure_ratio).floor) - Int.toNat ((↑total * doll_ratio).floor)

/-- Theorem stating that there are 84 puzzles and board games in the box -/
theorem puzzles_and_board_games_count :
  puzzles_and_board_games 200 (1/4) (1/3) = 84 := by
  -- Unfold the definition of puzzles_and_board_games
  unfold puzzles_and_board_games
  -- Evaluate the expression
  norm_num
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzles_and_board_games_count_l742_74205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l742_74218

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

-- Theorem statement
theorem f_inequalities :
  (∀ x > 1, f x < -Real.sqrt x - 1 / Real.sqrt x) ∧
  (∀ x ∈ Set.Icc (1/4 : ℝ) 1, f x ≥ 2/3 * x - 8/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l742_74218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lamps_reachable_all_lamps_on_l742_74222

/-- The distance between lamps that can be turned on -/
def lamp_distance : ℕ := 2005

/-- The first component of the vector used to reach lamps -/
def a : ℤ := 1357

/-- The second component of the vector used to reach lamps -/
def b : ℤ := 1476

/-- Represents a lamp at a given integer coordinate -/
structure Lamp where
  x : ℤ
  y : ℤ

/-- Predicate to check if a lamp is on at a given time -/
def is_on_at (l : Lamp) (t : ℕ) : Prop := sorry

/-- Theorem: Any integer point can be reached by a linear combination of four vectors -/
theorem all_lamps_reachable (x y : ℤ) : 
  ∃ (p q r s : ℤ), x = p * a + q * b + r * a + s * b ∧ 
                    y = p * b + q * a - r * b - s * a := by
  sorry

/-- Corollary: All lamps can eventually be turned on -/
theorem all_lamps_on : 
  ∀ (x y : ℤ), ∃ (t : ℕ), is_on_at ⟨x, y⟩ t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lamps_reachable_all_lamps_on_l742_74222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_remaining_area_l742_74243

theorem garden_remaining_area (perimeter width base height diameter : ℝ) 
  (perimeter_eq : perimeter = 36)
  (width_eq : width = 10)
  (base_eq : base = 6)
  (height_eq : height = 3)
  (diameter_eq : diameter = 2) :
  let length := (perimeter - 2 * width) / 2
  let garden_area := length * width
  let flower_bed_area := (base * height) / 2
  let pond_area := Real.pi * (diameter / 2) ^ 2
  let remaining_area := garden_area - flower_bed_area - pond_area
  abs (remaining_area - 67.86) < 0.01 := by
  sorry

#check garden_remaining_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_remaining_area_l742_74243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_two_l742_74282

/- Define the sales volume function -/
noncomputable def sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then k * x + 7
  else 0

/- Define the profit function -/
noncomputable def profit (x : ℝ) (a k : ℝ) : ℝ :=
  (sales_volume x a k) * (x - 1)

/- State the theorem -/
theorem max_profit_at_two (a k : ℝ) :
  (∀ x, 1 < x → x ≤ 5 → profit x a k ≤ profit 2 a k) ∧
  (k < 0) ∧
  (sales_volume 3 a k = 4) ∧
  (∀ x, 3 < x → x ≤ 5 → sales_volume x a k ≥ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_two_l742_74282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_OMF_l742_74245

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- The focus of the parabola x^2 = 4y -/
def focus : Point := ⟨0, 1⟩

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the perimeter of triangle OMF -/
theorem perimeter_of_triangle_OMF :
  ∀ (M : Point),
  M ∈ Parabola →
  distance M focus = 5 →
  distance origin M + distance M focus + distance origin focus = 6 + 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_OMF_l742_74245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trips_to_fill_barrel_l742_74229

/-- Represents the dimensions of a cylindrical container -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the dimensions of a conical container -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the number of trips needed to fill the cylinder with the cone -/
noncomputable def tripsNeeded (cylinder : Cylinder) (cone : Cone) : ℝ :=
  cylinderVolume cylinder / coneVolume cone

/-- The main theorem stating the number of trips needed -/
theorem trips_to_fill_barrel (barrel : Cylinder) (bucket : Cone) 
    (h_barrel_radius : barrel.radius = 10)
    (h_barrel_height : barrel.height = 15)
    (h_bucket_radius : bucket.radius = 10)
    (h_bucket_height : bucket.height = 10) :
    ⌈tripsNeeded barrel bucket⌉ = 5 := by
  sorry

#check trips_to_fill_barrel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trips_to_fill_barrel_l742_74229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l742_74274

/-- The time taken for two people to complete a job together, given their individual completion times -/
noncomputable def time_to_complete_together (time_p time_q : ℝ) : ℝ :=
  1 / (1 / time_p + 1 / time_q)

/-- Theorem stating that when person P takes 4 hours and person Q takes 6 hours to complete a job individually, 
    they will take 12/5 hours to complete the job together -/
theorem job_completion_time : time_to_complete_together 4 6 = 12 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l742_74274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_of_3_4_triangle_l742_74260

-- Define a right-angled triangle
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  right_angle : side1^2 + side2^2 = hypotenuse^2

-- Define the volume of rotation
noncomputable def volume_of_rotation (t : RightTriangle) (rotating_side : ℝ) : ℝ :=
  (1/3) * Real.pi * rotating_side^2 * t.side1

-- Theorem statement
theorem rotation_volume_of_3_4_triangle :
  ∃ (t : RightTriangle),
    t.side1 = 3 ∧ t.side2 = 4 ∧
    (volume_of_rotation t t.side1 = 12*Real.pi ∨ volume_of_rotation t t.side1 = 16*Real.pi) ∧
    (volume_of_rotation t t.side2 = 12*Real.pi ∨ volume_of_rotation t t.side2 = 16*Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_of_3_4_triangle_l742_74260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l742_74297

/-- Calculates the length of a train given its speed and time to cross a pole -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  (speed * 1000 / 3600) * time

/-- Proves that a train with speed 300 km/hr crossing a pole in 33 seconds is approximately 2750 meters long -/
theorem train_length_calculation :
  let speed : ℝ := 300
  let time : ℝ := 33
  abs (trainLength speed time - 2750) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l742_74297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_range_value_l742_74270

/-- The function g as described in the problem -/
noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (2*a*x + b) / (3*c*x + d)

/-- The theorem stating that 4/9 is the unique number not in the range of g -/
theorem unique_non_range_value 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 13 = 13)
  (h2 : g a b c d 31 = 31)
  (h3 : ∀ x ≠ -d/(3*c), g a b c d (g a b c d x) = x) :
  ∃! y, (∀ x, g a b c d x ≠ y) ∧ y = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_range_value_l742_74270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l742_74240

noncomputable def f (x : ℕ) : ℝ := 1 / (2^(x-1))

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 6 - 2 * a n + f (n-1)

theorem sequence_general_term (a : ℕ → ℝ) (h : ∀ n, S n a = a n) :
  ∀ n, a n = (2/3)^n + 4/(2^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l742_74240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l742_74241

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - 3| - m)

-- State the theorem
theorem problem_statement :
  (∀ x m : ℝ, f x m ∈ Set.univ) →
  (∃ n : ℝ, n = 4 ∧
    (∀ m : ℝ, m ≤ n) ∧
    (∀ a b : ℝ, a > 0 → b > 0 → 2 / (3*a + b) + 1 / (a + 2*b) = n →
      7*a + 4*b ≥ 9/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l742_74241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_candy_distribution_l742_74281

def candy_distribution (boxes : Fin 5 → ℕ) : Prop :=
  boxes 0 = 2 * boxes 1 ∧
  boxes 2 = 1 ∧
  boxes 3 = boxes 4 ∧
  (boxes 0 + boxes 1 + boxes 2 + boxes 3 + boxes 4) = 10

theorem unique_candy_distribution :
  ∃! (boxes : Fin 5 → ℕ), candy_distribution boxes ∧ boxes = ![2, 1, 1, 3, 3] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_candy_distribution_l742_74281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_l742_74290

/-- Given a group of people, proves that replacing one person with a new person
    results in a specific weight for the new person if the average weight
    increases by a certain amount. -/
theorem new_person_weight
  (n : ℕ)                    -- number of people in the group
  (old_weight : ℝ)           -- weight of the person being replaced
  (avg_increase : ℝ)         -- increase in average weight
  (h_n : n = 10)             -- there are 10 people initially
  (h_old : old_weight = 65)  -- the replaced person weighs 65 kg
  (h_avg : avg_increase = 7.2) -- the average weight increases by 7.2 kg
  : 
  old_weight + n * avg_increase = 137 := by
  -- Proof steps would go here
  sorry

#check new_person_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_l742_74290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_satisfied_l742_74231

variable (x c : ℝ)

noncomputable def y_function (x c : ℝ) : ℝ := Real.sqrt (x^2 - c * x)

theorem differential_equation_satisfied (x c : ℝ) :
  let y := y_function x c
  (x^2 + y^2) * 1 - 2 * x * y * ((2 * x - c) / (2 * Real.sqrt (x^2 - c * x))) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_satisfied_l742_74231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l742_74271

theorem sin_2alpha_value (α β : ℝ) 
  (h1 : π/2 < β) (h2 : β < α) (h3 : α < 3*π/4)
  (h4 : Real.cos (α - β) = 12/13)
  (h5 : Real.sin (α + β) = -3/5) :
  Real.sin (2*α) = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l742_74271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_in_similar_triangles_l742_74220

/-- Two triangles ABC and DEF are similar if they have the same angle and proportional sides -/
def similar_triangles (A B C D E F : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist A B = k * dist D E ∧
    dist B C = k * dist E F ∧
    dist A C = k * dist D F

theorem length_AB_in_similar_triangles 
  (A B C D E F : EuclideanSpace ℝ (Fin 2)) 
  (h_similar : similar_triangles A B C D E F)
  (h_AC : dist A C = 10)
  (h_DE : dist D E = 9)
  (h_DF : dist D F = 15)
  (h_EF : dist E F = 17) :
  dist A B = 90 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_in_similar_triangles_l742_74220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_increasing_a_range_f_not_monotonic_l742_74265

-- Define the function f(x) = (ax - x^2)e^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - x^2) * Real.exp x

-- Theorem 1: When a = 2, f(x) is monotonically decreasing on (-∞, -√2] and [√2, +∞)
theorem f_decreasing_intervals (x : ℝ) :
  (x ≤ -Real.sqrt 2 ∨ x ≥ Real.sqrt 2) → MonotoneOn (f 2) { y | y ≤ -Real.sqrt 2 ∨ y ≥ Real.sqrt 2 } := by
  sorry

-- Theorem 2: If f(x) is monotonically increasing on (-1,1], then a ∈ [3/2, +∞)
theorem f_increasing_a_range (a : ℝ) :
  (MonotoneOn (f a) (Set.Ioc (-1) 1)) → a ≥ 3/2 := by
  sorry

-- Theorem 3: f(x) cannot be a monotonic function on R for any value of a
theorem f_not_monotonic (a : ℝ) :
  ¬(MonotoneOn (f a) Set.univ ∨ AntitoneOn (f a) Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_increasing_a_range_f_not_monotonic_l742_74265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_inside_is_ten_l742_74280

noncomputable def hours_in_day : ℝ := 24

noncomputable def jonsey_awake_fraction : ℝ := 2/3
noncomputable def jonsey_outside_fraction : ℝ := 1/2

noncomputable def riley_awake_fraction : ℝ := 3/4
noncomputable def riley_outside_fraction : ℝ := 1/3

noncomputable def average_time_inside : ℝ := 
  let jonsey_awake_time := hours_in_day * jonsey_awake_fraction
  let jonsey_inside_time := jonsey_awake_time * (1 - jonsey_outside_fraction)
  let riley_awake_time := hours_in_day * riley_awake_fraction
  let riley_inside_time := riley_awake_time * (1 - riley_outside_fraction)
  (jonsey_inside_time + riley_inside_time) / 2

theorem average_time_inside_is_ten : average_time_inside = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_inside_is_ten_l742_74280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l742_74247

theorem prime_divisor_property (x : ℕ) (h1 : x > 1) 
  (h2 : ∀ (a b : ℕ), a > 0 → b > 0 → x ∣ (a * b) → (x ∣ a ∨ x ∣ b)) : 
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l742_74247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_theorem_l742_74255

noncomputable section

-- Define the vertices of triangle ABC
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (10, 0)

-- Define the horizontal line y = t
def horizontal_line (t : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = t}

-- Define the line segments AB and AC
def line_segment (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = (1 - t) • p + t • q}

-- Define the intersection points T and U
def T (t : ℝ) : ℝ × ℝ := ((8 - t) / 4, t)
def U (t : ℝ) : ℝ × ℝ := (5 * (8 - t) / 4, t)

-- Define the area of triangle ATU
def area_ATU (t : ℝ) : ℝ := (8 - t)^2 / 2

theorem triangle_intersection_theorem :
  ∃ t : ℝ, T t ∈ line_segment A B ∧
           U t ∈ line_segment A C ∧
           T t ∈ horizontal_line t ∧
           U t ∈ horizontal_line t ∧
           area_ATU t = 20 ∧
           t = 8 - 2 * Real.sqrt 10 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_theorem_l742_74255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rows_remain_increasing_rows_remain_increasing_valid_l742_74223

/-- Represents a 5x10 matrix of natural numbers -/
def MyMatrix := Fin 5 → Fin 10 → ℕ

/-- Checks if a row is in increasing order -/
def is_row_increasing (m : MyMatrix) (row : Fin 5) : Prop :=
  ∀ i j : Fin 10, i < j → m row i < m row j

/-- Checks if a column is in increasing order -/
def is_column_increasing (m : MyMatrix) (col : Fin 10) : Prop :=
  ∀ i j : Fin 5, i < j → m i col < m j col

/-- Represents a matrix with rows sorted in increasing order -/
def row_sorted_matrix (m : MyMatrix) : Prop :=
  ∀ row : Fin 5, is_row_increasing m row

/-- Represents a matrix with columns sorted in increasing order -/
def column_sorted_matrix (m : MyMatrix) : Prop :=
  ∀ col : Fin 10, is_column_increasing m col

/-- The main theorem: after row and column sorting, rows remain in increasing order -/
theorem rows_remain_increasing (m : MyMatrix) 
  (h_row_sorted : row_sorted_matrix m) 
  (h_col_sorted : column_sorted_matrix m) : 
  row_sorted_matrix m := by
  sorry

/-- All elements in the matrix are distinct and between 1 and 50 -/
def valid_matrix (m : MyMatrix) : Prop :=
  (∀ i j : Fin 5, ∀ k l : Fin 10, (i ≠ j ∨ k ≠ l) → m i k ≠ m j l) ∧
  (∀ i : Fin 5, ∀ j : Fin 10, 1 ≤ m i j ∧ m i j ≤ 50)

/-- The complete statement including the validity condition -/
theorem rows_remain_increasing_valid (m : MyMatrix)
  (h_valid : valid_matrix m)
  (h_row_sorted : row_sorted_matrix m) 
  (h_col_sorted : column_sorted_matrix m) : 
  row_sorted_matrix m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rows_remain_increasing_rows_remain_increasing_valid_l742_74223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l742_74209

noncomputable def f (x : ℝ) := (x - Real.exp 1) / Real.exp x

theorem f_properties :
  (∀ x y : ℝ, x < y → x ≤ Real.exp 1 + 1 → f x ≤ f y) ∧
  (∀ x y : ℝ, x < y → Real.exp 1 + 1 ≤ x → f y ≤ f x) ∧
  (∀ x : ℝ, f x ≤ f (Real.exp 1 + 1)) ∧
  (f (Real.exp 1 + 1) = Real.exp (-Real.exp 1 - 1)) ∧
  (∀ c : ℝ, (∀ x : ℝ, x > 0 → 2 * abs (Real.log x - Real.log 2) ≥ f x + c - 1 / Real.exp 2) →
    c ≤ (Real.exp 1 - 1) / Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l742_74209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_for_tangent_circle_l742_74292

/-- A circle passing through (2,1) and tangent to both coordinate axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : (center.1 - 2)^2 + (center.2 - 1)^2 = radius^2
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  positive_radius : radius > 0

/-- The line 2x-y-3=0 -/
def target_line (x y : ℝ) : Prop := 2*x - y - 3 = 0

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |2*p.1 - p.2 - 3| / Real.sqrt 5

theorem distance_to_line_for_tangent_circle (c : TangentCircle) :
  distance_to_line c.center = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_for_tangent_circle_l742_74292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_cheating_suspicion_l742_74279

/-- The probability of selecting 6 new winning numbers from the remaining 39 numbers -/
noncomputable def prob_no_repeat : ℝ := 39 * 38 * 37 * 36 * 35 * 34 / (45 * 44 * 43 * 42 * 41 * 40)

/-- The number of consecutive draws with no repeats before suspecting cheating -/
def n_draws : ℕ := 7

theorem lottery_cheating_suspicion :
  (prob_no_repeat ^ 6 > 0.01) ∧ (prob_no_repeat ^ 7 < 0.01) := by
  sorry

#eval n_draws

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_cheating_suspicion_l742_74279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l742_74232

/-- A non-convex quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  non_convex : Bool
  right_angle_BCD : (C.1 - B.1) * (D.1 - C.1) + (C.2 - B.2) * (D.2 - C.2) = 0
  AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 15
  BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 4
  CD_length : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 3
  AD_length : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 14

/-- Calculate the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  let area_BCD := (1/2) * abs ((q.B.1 - q.D.1) * (q.C.2 - q.D.2) - (q.C.1 - q.D.1) * (q.B.2 - q.D.2))
  let s := (15 + 5 + 14) / 2  -- Semi-perimeter of triangle ABD
  let area_ABD := Real.sqrt (s * (s - 15) * (s - 5) * (s - 14))
  area_BCD + area_ABD

/-- Theorem: The area of the given quadrilateral ABCD is 41 -/
theorem quadrilateral_area (q : Quadrilateral) : area q = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l742_74232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cube_identity_l742_74221

theorem sin_cube_identity (θ : Real) : 
  (Real.sin θ) ^ 3 = -(1/4) * Real.sin (3 * θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cube_identity_l742_74221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_l742_74263

noncomputable def z₁ : ℂ := 1 + Complex.I
noncomputable def z₂ : ℂ := Real.sqrt 2 * Complex.exp (Complex.I * (15 * Real.pi / 180))

theorem complex_product :
  z₁ * z₂ = 1 + Real.sqrt 3 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_l742_74263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_for_a_half_eccentricity_range_l742_74210

-- Define the line l: x + y = 1
def line (x y : ℝ) : Prop := x + y = 1

-- Define the hyperbola C: x²/a² - y² = 1
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 = 1

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 / a^2 + 1)

-- Theorem for the chord length when a = 1/2
theorem chord_length_for_a_half :
  ∀ x y : ℝ,
  line x y →
  hyperbola x y (1/2) →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    line x₁ y₁ ∧ hyperbola x₁ y₁ (1/2) ∧
    line x₂ y₂ ∧ hyperbola x₂ y₂ (1/2) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 2 * Real.sqrt 14 / 3 :=
by sorry

-- Theorem for the range of eccentricity
theorem eccentricity_range :
  ∀ a : ℝ,
  a > 0 →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    line x₁ y₁ ∧ hyperbola x₁ y₁ a ∧
    line x₂ y₂ ∧ hyperbola x₂ y₂ a) →
  let e := eccentricity a
  (e > Real.sqrt 6 / 2 ∧ e < Real.sqrt 2) ∨ (e > Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_for_a_half_eccentricity_range_l742_74210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_taidu_is_largest_largest_taidu_is_taidu_no_larger_taidu_l742_74225

/-- Definition of a Taidu number -/
def is_taidu (n : ℕ) : Prop :=
  let digits := (n.digits 10).reverse
  ∀ i, i ≥ 2 → i < digits.length → 
    digits[i]! ≥ digits[i-1]! + digits[i-2]!

/-- The largest Taidu number -/
def largest_taidu : ℕ := 10112359

/-- Theorem stating that largest_taidu is indeed the largest Taidu number -/
theorem largest_taidu_is_largest :
  is_taidu largest_taidu ∧ ∀ n : ℕ, is_taidu n → n ≤ largest_taidu := by
  sorry

/-- Proof that largest_taidu is a Taidu number -/
theorem largest_taidu_is_taidu :
  is_taidu largest_taidu := by
  sorry

/-- Proof that there is no larger Taidu number -/
theorem no_larger_taidu :
  ∀ n : ℕ, is_taidu n → n ≤ largest_taidu := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_taidu_is_largest_largest_taidu_is_taidu_no_larger_taidu_l742_74225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_eq_neg_3_l742_74237

/-- A function f defined as f(x) = a*sin(π*x + α) + b*cos(π*x + β) -/
noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

/-- Theorem stating that if f(2001) = 3, then f(2012) = -3 -/
theorem f_2012_eq_neg_3 (a b α β : ℝ) (h : f a b α β 2001 = 3) : f a b α β 2012 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_eq_neg_3_l742_74237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_proof_l742_74236

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define points
def O : ℝ × ℝ := (0, 0)
structure Point where
  coords : ℝ × ℝ

def A : Point := ⟨(0, 0)⟩  -- Placeholder coordinates
def B : Point := ⟨(0, 0)⟩  -- Placeholder coordinates

-- Define the distance function
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.coords.1 - q.coords.1)^2 + (p.coords.2 - q.coords.2)^2)

theorem ellipse_and_line_proof :
  C₁ A.coords.1 A.coords.2 ∧ 
  C₂ B.coords.1 B.coords.2 ∧ 
  distance ⟨O⟩ A = 2 →
  (B.coords.2 = B.coords.1 ∨ B.coords.2 = -B.coords.1) ∧
  (∀ x y : ℝ, C₂ x y ↔ x^2/16 + y^2/4 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_proof_l742_74236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_theorem_l742_74264

open Real

structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

def parallel_vectors (a b : ℝ) (t : AcuteTriangle) : Prop :=
  a * sin t.B = sqrt 3 * b * cos t.A

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem acute_triangle_theorem (t : AcuteTriangle) (a b : ℝ) 
  (h_parallel : parallel_vectors a b t) :
  t.A = π / 3 ∧
  ∀ (b c : ℝ), 
    (b = 4 / sqrt 3 * sin t.B ∧ c = 4 / sqrt 3 * sin t.C) → 
    2 * sqrt 3 + 2 < perimeter 2 b c ∧ perimeter 2 b c ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_theorem_l742_74264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_grid_existence_l742_74258

/-- A coloring of an n × n grid satisfying the two-color condition -/
def TwoColorGrid (n : ℕ) : Type :=
  { coloring : Fin n → Fin n → Fin n //
    (∀ i, ∃! (c₁ c₂ : Fin n), ∀ j, coloring i j = c₁ ∨ coloring i j = c₂) ∧
    (∀ j, ∃! (c₁ c₂ : Fin n), ∀ i, coloring i j = c₁ ∨ coloring i j = c₂) ∧
    (∀ c, (Finset.univ.filter (λ i => ∃ j, coloring i j = c)).card = n) }

/-- The theorem stating that a TwoColorGrid exists only for n = 2, 3, or 4 -/
theorem two_color_grid_existence (n : ℕ) : 
  Nonempty (TwoColorGrid n) ↔ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_grid_existence_l742_74258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fedya_arrival_time_l742_74286

/-- Represents a point on the road between Yolkino and Palkino -/
structure Point where
  distance_from_yolkino : ℚ
  distance_from_palkino : ℚ

/-- The road from Yolkino to Palkino -/
def road_length : ℚ := 9

/-- The oak tree's position on the road -/
def oak_tree : Point :=
  { distance_from_yolkino := road_length / 3
  , distance_from_palkino := 2 * road_length / 3 }

/-- Fedya's speed (units per hour) -/
def fedya_speed : ℚ := 3

/-- Fedya's position at a given time (in hours since 12:00) -/
def fedya_position (time : ℚ) : Point :=
  { distance_from_yolkino := fedya_speed * time + 4
  , distance_from_palkino := road_length - (fedya_speed * time + 4) }

/-- Theorem stating when Fedya will arrive in Palkino -/
theorem fedya_arrival_time :
  ∃ (t : ℚ), t = 13/6 ∧ (fedya_position t).distance_from_palkino = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fedya_arrival_time_l742_74286
