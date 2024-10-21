import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l351_35131

theorem complex_fraction_simplification :
  (5 * (1 + Complex.I^3)) / ((2 + Complex.I) * (2 - Complex.I)) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l351_35131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_x0_1_convergence_x0_0_l351_35186

/-- Newton's method for f(x) = x^2 - x - 1 -/
noncomputable def newtonMethod (x : ℝ) : ℝ := (x^2 + 1) / (2*x - 1)

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Convergence of Newton's method for x₀ = 1 -/
theorem convergence_x0_1 (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n ≥ N, |newtonMethod^[n] 1 - φ| < ε := by
  sorry

/-- Convergence of Newton's method for x₀ = 0 -/
theorem convergence_x0_0 (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n ≥ N, |newtonMethod^[n] 0 - (-φ⁻¹)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_x0_1_convergence_x0_0_l351_35186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l351_35149

theorem tan_double_angle_special_case (θ : Real) 
  (h1 : Real.tan (π / 2 - θ) = 4 * Real.cos (2 * π - θ)) 
  (h2 : |θ| < π / 2) : 
  Real.tan (2 * θ) = Real.sqrt 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l351_35149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l351_35103

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 / Real.sqrt (x - 2)

-- State the theorem
theorem range_of_x (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ x > 2 := by
  sorry -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l351_35103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_diff_identity_l351_35195

theorem cos_diff_identity (a b : ℝ) : 
  Real.cos (a + b) - Real.cos (a - b) = -2 * Real.sin a * Real.sin b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_diff_identity_l351_35195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_intersection_and_inequality_l351_35187

-- Define sets A and B
def A : Set ℝ := {x | ∃ y, y = (1 : ℝ) / Real.sqrt (2^x - 1)}
def B : Set ℝ := {x | ∃ y, y = Real.log (x^2 - x - 6)}

-- State the theorem
theorem sets_intersection_and_inequality :
  (A ∩ B = Set.Ioi 3) ∧
  ({x : ℝ | x^2 + 2*x > 0} = A ∪ B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_intersection_and_inequality_l351_35187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_six_factors_l351_35145

def has_exactly_six_factors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 6

theorem least_integer_with_six_factors :
  ∃ (n : ℕ), has_exactly_six_factors n ∧ ∀ m < n, ¬has_exactly_six_factors m :=
by
  use 18
  constructor
  · -- Prove that 18 has exactly six factors
    simp [has_exactly_six_factors]
    sorry
  · -- Prove that no smaller number has exactly six factors
    intro m hm
    simp [has_exactly_six_factors]
    sorry

#eval (Finset.filter (· ∣ 18) (Finset.range 19)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_six_factors_l351_35145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l351_35198

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    prove that its eccentricity is 2 under specific conditions. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (x y c : ℝ),
  -- Equation of the hyperbola
  x^2 / a^2 - y^2 / b^2 = 1 ∧
  -- c is half the distance between foci
  c > 0 ∧
  -- Point P(x, y) is on the left branch of the hyperbola
  x < 0 ∧
  -- |PF₁| = |F₁F₂|
  (x + c)^2 + y^2 = (2*c)^2 ∧
  -- Distance from F₁ to line PF₂ is √7a
  (2*c*Real.sqrt 7*a)^2 = ((x + c)*(2*c) + y^2)^2 / ((x + c)^2 + y^2) →
  -- Eccentricity is 2
  c / a = 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l351_35198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l351_35184

theorem intersection_points_count : 
  ∃! (points : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ points ↔ (x^2 + y^2 = 1 ∧ x^2 + 9*y^2 = 9)) ∧ 
    (∃ (l : List (ℝ × ℝ)), points = l.toFinset ∧ l.length = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l351_35184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l351_35193

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define set A
def A : Set ℕ := {x ∈ PositiveIntegers | x ≤ 3}

-- Define set B
def B : Set ℕ := {x : ℕ | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l351_35193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_specific_l351_35112

/-- The area of a quadrilateral with a given diagonal and two offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1/2) * diagonal * offset1 + (1/2) * diagonal * offset2

/-- Theorem: The area of a quadrilateral with diagonal 28 cm and offsets 8 cm and 2 cm is 140 cm² -/
theorem quadrilateral_area_specific : quadrilateralArea 28 8 2 = 140 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp [mul_add, mul_comm, mul_assoc]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_specific_l351_35112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l351_35117

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * sin (2 * x) - cos (2 * x)

-- Define the shifted function
noncomputable def shifted_f (m : ℝ) (x : ℝ) : ℝ := f (x + abs m)

-- Theorem statement
theorem min_m_value (m : ℝ) (h1 : m > -π/2) 
  (h2 : ∀ x, shifted_f m x = shifted_f m (π/3 - x)) : 
  (∀ m' > -π/2, shifted_f m' (π/6) = shifted_f m' (π/6 - (π/6 - x)) → m ≤ m') → 
  m = -π/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l351_35117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l351_35183

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 12
def b : ℝ := 2

-- Define the focal length
noncomputable def focal_length : ℝ := 4 * Real.sqrt 2

-- Define the point P and the slope of the line
def P : ℝ × ℝ := (-2, 1)
def m : ℝ := 1  -- Changed 'slope' to 'm' to avoid naming conflict

-- Theorem statement
theorem ellipse_chord_length :
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (x : ℝ) => m * (x - P.1) + P.2
  let chord_length := Real.sqrt 42 / 2
  (∀ x y, ellipse x y → x^2 / 12 + y^2 / 4 = 1) ∧
  (a > b) ∧
  (b > 0) ∧
  (focal_length = 4 * Real.sqrt 2) ∧
  (b = 2) →
  ∃ x1 x2,
    ellipse x1 (line x1) ∧
    ellipse x2 (line x2) ∧
    Real.sqrt ((x2 - x1)^2 + (line x2 - line x1)^2) = chord_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l351_35183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_cosine_angle_l351_35148

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

/-- Helper function to calculate the cosine of an angle given three points -/
noncomputable def cos_angle (P Q R : ℝ × ℝ) : ℝ := sorry

/-- Given an ellipse with foci F₁(-c,0) and F₂(c,0), and equation x²/a² + y²/b² = 1,
    where a > b > 0, and a point P on the ellipse such that the area of triangle PF₁F₂
    is √2/2 * b², prove that cos∠F₁PF₂ = 1/3 -/
theorem ellipse_cosine_angle (a b c : ℝ) (P : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  (area_triangle P (-c, 0) (c, 0) = Real.sqrt 2 / 2 * b^2) →
  cos_angle P (-c, 0) (c, 0) = 1/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_cosine_angle_l351_35148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alison_money_calculation_l351_35124

-- Define exchange rates
def gbp_to_usd : ℝ := 1.35
def gbp_to_eur : ℝ := 1.15
def gbp_to_aud : ℝ := 2
def gbp_to_jpy : ℝ := 150

-- Define initial amounts
def kent_initial_gbp : ℝ := 1000

-- Define relationships
def charlie_borrowed_percentage : ℝ := 0.2
def daniella_borrowed_percentage : ℝ := 0.5
def daniella_to_elle_ratio : ℝ := 3

-- Define money relationships
def brooke_to_kent_ratio : ℝ := 2
def brittany_to_brooke_ratio : ℝ := 4
def alison_to_brittany_ratio : ℝ := 0.5

theorem alison_money_calculation :
  ∃ (kent_remaining charlie_money brooke_money brittany_money alison_money : ℝ),
    kent_remaining = kent_initial_gbp * (1 - charlie_borrowed_percentage) ∧
    brooke_money = kent_remaining * brooke_to_kent_ratio * (gbp_to_eur / gbp_to_usd) ∧
    brittany_money = brooke_money * brittany_to_brooke_ratio ∧
    alison_money = brittany_money * alison_to_brittany_ratio ∧
    abs (alison_money - 2370.37) < 0.01 := by
  sorry

#eval 1000 * (1 - 0.2) * 2 * (1.15 / 1.35) * 4 * 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alison_money_calculation_l351_35124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_matching_numbers_l351_35160

/-- Row-wise numbering function -/
def f (i j : ℕ) : ℕ := 19 * (i - 1) + j + 2

/-- Column-wise numbering function -/
def g (i j : ℕ) : ℕ := 15 * (j - 1) + i + 2

/-- The set of pairs (i, j) where the numbers match in both systems -/
def matching_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 ≤ 15 ∧ p.2 ≤ 19 ∧ f p.1 p.2 = g p.1 p.2) (Finset.product (Finset.range 16) (Finset.range 20))

/-- Theorem stating the sum of matching numbers is 440 -/
theorem sum_of_matching_numbers :
  (matching_pairs.sum fun p => f p.1 p.2) = 440 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_matching_numbers_l351_35160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_form_l351_35128

noncomputable def g (x : ℝ) : ℤ := ⌊3*x⌋ + ⌊5*x⌋ + ⌊7*x⌋ + ⌊9*x⌋

theorem count_integers_in_form (n : ℕ) :
  n = 1500 →
  (∃ (s : Finset ℕ), 
    (∀ m ∈ s, m ≤ n ∧ (∃ x : ℝ, 0 < x ∧ x ≤ 2 ∧ (g x).toNat = m)) ∧
    (∀ m : ℕ, m ≤ n → (∃ x : ℝ, 0 < x ∧ x ≤ 2 ∧ (g x).toNat = m) → m ∈ s) ∧
    s.card = 48) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_form_l351_35128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weight_of_solution_y_l351_35175

/-- Proves that the initial weight of solution y is 6 kg -/
theorem initial_weight_of_solution_y : ∀ (W : ℝ),
  -- Solution y is 30% liquid x and 70% water
  (0.3 * W + 0.7 * W = W) →
  -- 2 kg of water evaporate from solution y
  (let remaining_water := 0.7 * W - 2;
   -- 2 kg of solution y are added to the remaining 4 kg of liquid
   let new_liquid_x := 0.3 * W + 0.6;
   let new_water := remaining_water + 1.4;
   -- The new solution is 40% liquid x
   new_liquid_x / (new_liquid_x + new_water) = 0.4) →
  -- The initial weight of solution y is 6 kg
  W = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weight_of_solution_y_l351_35175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l351_35150

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1)

theorem f_monotonicity_and_inequality (a : ℝ) :
  (∀ x > 0, ∀ y > 0, x < y → (a ≤ 0 → f a x < f a y)) ∧
  (a > 0 → ∀ x y, 0 < x ∧ x < y ∧ y < 1/a → f a x < f a y) ∧
  (a > 0 → ∀ x y, 1/a < x ∧ x < y → f a x > f a y) ∧
  (∀ x ≥ 1, f a x ≤ Real.log x / (x + 1) ↔ a ≥ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l351_35150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l351_35110

theorem order_of_abc (a b c : ℝ) 
  (ha : a / (12 * Real.exp 1) = Real.log 4 / 4)
  (hb : b / (12 * Real.exp 1) = Real.log 3 / 3)
  (hc : c = 12) : 
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l351_35110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l351_35188

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (4 - x^2)

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- Theorem statement
theorem f_properties :
  (∀ x, x ∈ domain → f (-x) = -f x) ∧  -- f is odd
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f x < f y) ∧  -- f is monotonically increasing
  {m : ℝ | f (1 + m) + f (1 - m^2) < 0} = {m : ℝ | -Real.sqrt 3 < m ∧ m < -1} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l351_35188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l351_35100

-- Define complex number addition
def add (z w : ℂ) : ℂ := z + w

-- Define complex number multiplication
def mul (z w : ℂ) : ℂ := z * w

-- Define scalar multiplication for complex numbers
def smul (r : ℝ) (z : ℂ) : ℂ := r • z

-- State the theorem
theorem complex_simplification :
  add (smul 3 (add 4 (-2 * Complex.I))) (mul (smul 2 Complex.I) (add 3 (smul 2 Complex.I))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l351_35100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_five_value_l351_35194

/-- Given a real number x such that x + 1/x = 3, Sₘ is defined as xᵐ + 1/xᵐ. -/
noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1 / (x^m)

/-- Theorem: If x is a real number such that x + 1/x = 3, then S₅ = 123. -/
theorem S_five_value (x : ℝ) (h : x + 1/x = 3) : S x 5 = 123 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_five_value_l351_35194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l351_35115

theorem trig_identity (α : ℝ) (h : Real.cos α + Real.sin α = 2/3) :
  (Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l351_35115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_l351_35141

theorem power_of_eight (y : ℝ) (h : (8 : ℝ)^(3*y) = 512) : (8 : ℝ)^(3*y - 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_l351_35141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_prism_lateral_surface_area_l351_35119

/-- The lateral surface area of a regular octagonal prism -/
noncomputable def lateral_surface_area (volume : ℝ) (height : ℝ) : ℝ :=
  16 * Real.sqrt (2.2 * (Real.sqrt 2 - 1))

/-- Theorem: The lateral surface area of a regular octagonal prism with volume 8 m³ and height 2.2 m -/
theorem octagonal_prism_lateral_surface_area :
  lateral_surface_area 8 2.2 = 16 * Real.sqrt (2.2 * (Real.sqrt 2 - 1)) :=
by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- The left-hand side is now equal to the right-hand side
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_prism_lateral_surface_area_l351_35119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_meter_is_eleven_l351_35151

/-- Represents a rectangular farm with fencing on one long side, one short side, and the diagonal -/
structure RectangularFarm where
  area : ℝ
  shortSide : ℝ
  totalCost : ℝ

/-- Calculates the cost per meter of fencing for a given rectangular farm -/
noncomputable def costPerMeter (farm : RectangularFarm) : ℝ :=
  let longSide := farm.area / farm.shortSide
  let diagonal := Real.sqrt (longSide^2 + farm.shortSide^2)
  let totalLength := longSide + farm.shortSide + diagonal
  farm.totalCost / totalLength

/-- Theorem stating that for a farm with given specifications, the cost per meter is 11 -/
theorem cost_per_meter_is_eleven (farm : RectangularFarm) 
    (h1 : farm.area = 1200)
    (h2 : farm.shortSide = 30)
    (h3 : farm.totalCost = 1320) :
  costPerMeter farm = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_meter_is_eleven_l351_35151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l351_35140

noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

theorem expression_values :
  (log4 (Real.sqrt 8) + lg 50 + lg 2 + 5^(log5 3) + (-9.8)^0 = 27/4) ∧
  ((27/64)^(2/3) - (25/4)^(1/2) + (0.008)^(-2/3) * 2/5 = 129/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l351_35140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_size_is_600_l351_35138

/-- The rental cost per square foot in dollars -/
noncomputable def rental_cost_per_sqft : ℚ := 120 / 100

/-- Jillian's monthly budget for rent in dollars -/
def monthly_budget : ℚ := 720

/-- The largest apartment size Jillian should consider in square feet -/
noncomputable def largest_apartment_size : ℚ := monthly_budget / rental_cost_per_sqft

theorem largest_apartment_size_is_600 :
  largest_apartment_size = 600 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_size_is_600_l351_35138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l351_35118

theorem expression_simplification (k : ℤ) :
  (2 : ℝ)^(3-(2*k+1)) - (2 : ℝ)^(3-(2*k-1)) + (2 : ℝ)^(3-2*k) = -(2 : ℝ)^(2-(2*k+1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l351_35118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_120_meters_inverse_percentage_of_24_l351_35192

-- Define the percentage calculation function
noncomputable def percentage (x : ℝ) (p : ℝ) : ℝ := x * (p / 100)

-- Define the inverse percentage calculation function
noncomputable def inverse_percentage (y : ℝ) (p : ℝ) : ℝ := y / (p / 100)

-- Theorem for the first part of the problem
theorem percentage_of_120_meters : percentage 120 40 = 48 := by sorry

-- Theorem for the second part of the problem
theorem inverse_percentage_of_24 : inverse_percentage 24 40 = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_120_meters_inverse_percentage_of_24_l351_35192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pears_thrown_away_percentage_l351_35181

/-- Calculates the percentage of pears thrown away given the selling and discarding rates -/
theorem pears_thrown_away_percentage 
  (initial_pears : ℕ) 
  (sell_rate_day1 : ℚ) 
  (discard_rate_day1 : ℚ) 
  (sell_rate_day2 : ℚ) 
  (h1 : sell_rate_day1 = 4/5)
  (h2 : discard_rate_day1 = 1/2)
  (h3 : sell_rate_day2 = 4/5)
  : (initial_pears : ℚ) ≠ 0 → 
    let remaining_after_sell_day1 := initial_pears * (1 - sell_rate_day1)
    let remaining_after_discard_day1 := remaining_after_sell_day1 * (1 - discard_rate_day1)
    let remaining_after_sell_day2 := remaining_after_discard_day1 * (1 - sell_rate_day2)
    let total_discarded := (remaining_after_sell_day1 * discard_rate_day1) + remaining_after_sell_day2
    (total_discarded / initial_pears) * 100 = 12 := by
  intro h_nonzero
  sorry

#check pears_thrown_away_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pears_thrown_away_percentage_l351_35181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_sum_l351_35170

theorem nearest_integer_to_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : |a| + b = 5) (h2 : |a| * b + a^3 = -8) :
  ∃ (n : ℤ), n = 3 ∧ ∀ (m : ℤ), |↑m - (a + b)| ≥ |↑n - (a + b)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_sum_l351_35170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l351_35129

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then Real.log x else -Real.log (-x)

theorem sequence_properties :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
  (∃ (d : ℝ), d ≠ 0 ∧ x₂ = x₁ + d ∧ x₃ = x₂ + d ∧ x₄ = x₃ + d ∧
    ∃ (e : ℝ), e ≠ 0 ∧ f x₂ = f x₁ + e ∧ f x₃ = f x₂ + e ∧ f x₄ = f x₃ + e) ∧
  (∃ (q : ℝ), q > 0 ∧ q ≠ 1 ∧ x₂ = x₁ * q ∧ x₃ = x₂ * q ∧ x₄ = x₃ * q ∧
    ∃ (e : ℝ), e ≠ 0 ∧ f x₂ = f x₁ + e ∧ f x₃ = f x₂ + e ∧ f x₄ = f x₃ + e) ∧
  (∃ (d : ℝ), d ≠ 0 ∧ x₂ = x₁ + d ∧ x₃ = x₂ + d ∧ x₄ = x₃ + d ∧
    ∃ (r : ℝ), r > 0 ∧ r ≠ 1 ∧ f x₂ = f x₁ * r ∧ f x₃ = f x₂ * r ∧ f x₄ = f x₃ * r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l351_35129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequential_arrangement_satisfies_property_l351_35182

/-- A regular nonagon with numbers assigned to its vertices -/
structure NumberedNonagon where
  vertices : Fin 9 → ℕ

/-- Check if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  2 * b = a + c

/-- Vertices of a regular nonagon that form an equilateral triangle -/
def equilateralTriangleVertices (i : Fin 9) : Fin 3 → Fin 9
  | 0 => i
  | 1 => i + 3
  | 2 => i + 6

/-- The property we want to prove -/
def satisfiesArithmeticMeanProperty (n : NumberedNonagon) : Prop :=
  ∀ i : Fin 9, ∃ σ : Equiv.Perm (Fin 3), 
    let v := equilateralTriangleVertices i
    isArithmeticSequence (n.vertices (v (σ 0))) (n.vertices (v (σ 1))) (n.vertices (v (σ 2)))

/-- The sequential arrangement of numbers -/
def sequentialArrangement : NumberedNonagon where
  vertices i := 2016 + i.val

/-- The theorem to prove -/
theorem sequential_arrangement_satisfies_property :
  satisfiesArithmeticMeanProperty sequentialArrangement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequential_arrangement_satisfies_property_l351_35182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_with_913_l351_35113

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  ({a, b, c, d, e} : Finset ℕ) = {1, 3, 5, 8, 9} ∧
  ((b = 1 ∧ e = 1) ∨ (c = 1 ∧ d = 1))

def product (a b c d e : ℕ) : ℕ := (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product_with_913 :
  ∀ a b c d e : ℕ,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 9 1 3 8 5 :=
by
  sorry

#eval product 9 1 3 8 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_with_913_l351_35113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_equality_iff_45_degree_l351_35107

/-- A parallelogram with side lengths a and b, and diagonal lengths m and n. -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  m : ℝ
  n : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < m ∧ 0 < n

/-- The acute angle of a parallelogram. -/
noncomputable def acute_angle (p : Parallelogram) : ℝ := 
  Real.arccos ((p.m^2 + p.n^2 - 2 * p.a^2 - 2 * p.b^2) / (4 * p.a * p.b))

/-- Theorem: For a parallelogram, a^4 + b^4 = m^2 n^2 if and only if its acute angle is 45°. -/
theorem parallelogram_equality_iff_45_degree (p : Parallelogram) :
  p.a^4 + p.b^4 = p.m^2 * p.n^2 ↔ acute_angle p = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_equality_iff_45_degree_l351_35107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipsoid_volumes_l351_35144

/-- Represents an ellipse with major axis 2a and minor axis 2b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b

/-- Volume of an ellipsoid of revolution -/
noncomputable def ellipsoid_volume (a b : ℝ) : ℝ := (4 / 3) * Real.pi * a * b^2

/-- Volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem ellipsoid_volumes (e : Ellipse) :
  (ellipsoid_volume e.a e.b = (4 / 3) * Real.pi * e.a * e.b^2) ∧
  (ellipsoid_volume e.b e.a = (4 / 3) * Real.pi * e.a^2 * e.b) ∧
  (∀ r : ℝ, sphere_volume r = (4 / 3) * Real.pi * r^3) := by
  sorry

#check ellipsoid_volumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipsoid_volumes_l351_35144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_downstream_time_ratio_l351_35130

/-- Represents the ratio of upstream time to downstream time for a boat -/
noncomputable def upstreamToDownstreamTimeRatio (boatSpeed streamSpeed : ℝ) : ℝ :=
  let upstreamSpeed := boatSpeed - streamSpeed
  let downstreamSpeed := boatSpeed + streamSpeed
  downstreamSpeed / upstreamSpeed

/-- Theorem stating that for a boat with speed 24 kmph in still water and a stream with speed 8 kmph,
    the ratio of time taken to row upstream to time taken to row downstream is 2:1 -/
theorem upstream_downstream_time_ratio :
  upstreamToDownstreamTimeRatio 24 8 = 2 := by
  unfold upstreamToDownstreamTimeRatio
  simp
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_downstream_time_ratio_l351_35130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_perimeter_l351_35153

-- Define the triangle DEF
structure Triangle :=
  (DE : ℝ)
  (DF : ℝ)
  (EF : ℝ)

-- Define the properties of the right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.EF^2 = t.DE^2 + t.DF^2

-- Define the area function
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * t.DE * t.DF

-- Define the perimeter function
noncomputable def perimeter (t : Triangle) : ℝ :=
  t.DE + t.DF + t.EF

-- Theorem statement
theorem right_triangle_area_and_perimeter :
  ∀ (t : Triangle),
  is_right_triangle t →
  t.DE = 15 →
  t.DF = 10 →
  (area t = 75 ∧ perimeter t = 25 + 5 * Real.sqrt 13) := by
  sorry

#check right_triangle_area_and_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_perimeter_l351_35153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_half_km_l351_35139

/-- The distance A can run in 2 minutes -/
noncomputable def distance_A : ℝ := 2

/-- The time it takes B to run 2 km in minutes -/
noncomputable def time_B : ℝ := 2 + 40 / 60

/-- The speed of B in km/minute -/
noncomputable def speed_B : ℝ := distance_A / time_B

/-- The distance B can run in 2 minutes -/
noncomputable def distance_B : ℝ := speed_B * 2

theorem a_beats_b_by_half_km : distance_A - distance_B = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_half_km_l351_35139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_seven_count_l351_35190

theorem remainder_seven_count : 
  (Finset.filter (fun n : ℕ => n > 7 ∧ 59 % n = 7) (Finset.range 60)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_seven_count_l351_35190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_C_not_right_triangle_l351_35179

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions
def conditionA (t : Triangle) : Prop := t.angleA = t.angleB - t.angleC

def conditionB (t : Triangle) : Prop := ∃ (k : ℝ), t.a = 5 * k ∧ t.b = 12 * k ∧ t.c = 13 * k

def conditionC (t : Triangle) : Prop := ∃ (k : ℝ), t.angleA = 3 * k ∧ t.angleB = 4 * k ∧ t.angleC = 5 * k

def conditionD (t : Triangle) : Prop := t.a^2 = (t.b + t.c) * (t.b - t.c)

-- Define a right triangle
def isRightTriangle (t : Triangle) : Prop := t.angleA = 90 ∨ t.angleB = 90 ∨ t.angleC = 90

-- Theorem statement
theorem condition_C_not_right_triangle :
  ∃ (t : Triangle), conditionC t ∧ ¬(isRightTriangle t) ∧
  (∀ (t : Triangle), conditionA t → isRightTriangle t) ∧
  (∀ (t : Triangle), conditionB t → isRightTriangle t) ∧
  (∀ (t : Triangle), conditionD t → isRightTriangle t) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_C_not_right_triangle_l351_35179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_berries_cheapest_l351_35176

/-- Represents the cost of making jam using different methods -/
structure JamCost where
  train_ticket : ℚ
  berries_collected : ℚ
  market_berry_price : ℚ
  sugar_price : ℚ
  ready_made_jam_price : ℚ

/-- Calculates the cost of making 1.5 kg of jam by picking berries -/
def cost_picking_berries (jc : JamCost) : ℚ :=
  (jc.train_ticket / jc.berries_collected + jc.sugar_price) * (3/2)

/-- Calculates the cost of making 1.5 kg of jam by buying berries -/
def cost_buying_berries (jc : JamCost) : ℚ :=
  jc.market_berry_price + jc.sugar_price

/-- Calculates the cost of buying 1.5 kg of ready-made jam -/
def cost_ready_made_jam (jc : JamCost) : ℚ :=
  jc.ready_made_jam_price * (3/2)

/-- Theorem stating that picking berries is the cheapest method -/
theorem picking_berries_cheapest (jc : JamCost) 
  (h1 : jc.train_ticket = 200)
  (h2 : jc.berries_collected = 5)
  (h3 : jc.market_berry_price = 150)
  (h4 : jc.sugar_price = 54)
  (h5 : jc.ready_made_jam_price = 220) :
  cost_picking_berries jc < cost_buying_berries jc ∧ 
  cost_picking_berries jc < cost_ready_made_jam jc := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_berries_cheapest_l351_35176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lasagna_packages_theorem_l351_35159

noncomputable def beef_amount : ℚ := 10
noncomputable def existing_noodles : ℚ := 4
noncomputable def package_size : ℚ := 2

noncomputable def noodles_needed (beef : ℚ) : ℚ := 2 * beef

noncomputable def additional_noodles_needed (beef : ℚ) (existing : ℚ) : ℚ :=
  noodles_needed beef - existing

noncomputable def packages_to_buy (beef : ℚ) (existing : ℚ) (package_size : ℚ) : ℚ :=
  (additional_noodles_needed beef existing) / package_size

theorem lasagna_packages_theorem :
  packages_to_buy beef_amount existing_noodles package_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lasagna_packages_theorem_l351_35159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_l351_35146

/-- The volume of ice cream in a cone, cylinder, and hemisphere configuration -/
theorem ice_cream_volume (h_cone : ℝ) (r : ℝ) (h_cylinder : ℝ) 
  (cone_height : h_cone = 12)
  (radius : r = 3)
  (cylinder_height : h_cylinder = 2) :
  (1/3 * π * r^2 * h_cone) + (π * r^2 * h_cylinder) + (2/3 * π * r^3) = 72 * π := by
  -- Replace all occurrences of the variables with their given values
  rw [cone_height, radius, cylinder_height]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

#check ice_cream_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_l351_35146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_bound_l351_35158

def s (n : ℕ) (lambda : ℝ) : ℝ := 3^n * (lambda - n) - 6

def a (n : ℕ) (lambda : ℝ) : ℝ := s n lambda - s (n-1) lambda

theorem lambda_bound (lambda : ℝ) :
  (∀ n : ℕ, n > 1 → a n lambda > a (n+1) lambda) →
  lambda < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_bound_l351_35158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_sqrt_two_l351_35125

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then x^2 - 4
  else if x > 2 then 2*x
  else 0  -- undefined for x < 0, but we need to cover all cases

-- State the theorem
theorem unique_solution_is_sqrt_two :
  ∃! x₀ : ℝ, (0 ≤ x₀ ∧ x₀ ≤ 2 ∨ x₀ > 2) ∧ f x₀ = -2 ∧ x₀ = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_sqrt_two_l351_35125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_earnings_increase_l351_35126

/-- Represents Elaine's annual earnings from last year -/
def E : ℝ := 100  -- Arbitrary value for demonstration

/-- Represents the percentage increase in Elaine's earnings this year -/
def P : ℝ := 20   -- The value we're trying to prove

/-- The amount spent on rent last year -/
noncomputable def rent_last_year : ℝ := 0.20 * E

/-- The amount spent on rent this year -/
noncomputable def rent_this_year : ℝ := 0.30 * (E * (1 + P / 100))

/-- Theorem stating that given the conditions, the percentage increase in Elaine's earnings is 20% -/
theorem elaine_earnings_increase :
  rent_this_year = 1.80 * rent_last_year →
  P = 20 := by
  intro h
  -- The proof goes here
  sorry

#check elaine_earnings_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_earnings_increase_l351_35126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heating_process_efficiency_l351_35156

/-- Represents a monatomic ideal gas -/
structure MonatomicIdealGas where
  pressure : ℝ
  volume : ℝ
  temperature : ℝ
  amount : ℝ

/-- The universal gas constant -/
noncomputable def R : ℝ := sorry

/-- Represents a heating process of a monatomic ideal gas -/
structure HeatingProcess where
  initial : MonatomicIdealGas
  final : MonatomicIdealGas
  pressureTemp : ∀ (g : MonatomicIdealGas), g.pressure^2 = g.temperature

/-- The efficiency of a heating process -/
noncomputable def efficiency (process : HeatingProcess) : ℝ :=
  let workDone := (3 / 2) * process.initial.pressure * process.initial.volume
  let internalEnergyChange := (9 / 2) * process.initial.pressure * process.initial.volume
  workDone / (workDone + internalEnergyChange)

/-- Theorem: The efficiency of the heating process is 1/4 -/
theorem heating_process_efficiency (process : HeatingProcess) :
  efficiency process = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heating_process_efficiency_l351_35156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l351_35127

theorem complex_modulus_example : Complex.abs (-3 - (4/5) * Complex.I) = Real.sqrt 241 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l351_35127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_value_l351_35132

-- Define N as per the problem statement
noncomputable def N : ℝ := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)

-- Theorem statement
theorem N_value : N = 1 - (Real.sqrt 6 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_value_l351_35132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l351_35185

-- Define the complex plane
variable (z : ℂ)

-- Define the equation |z - 2| = 3|z + 2|
def equation (z : ℂ) : Prop :=
  Complex.abs (z - 2) = 3 * Complex.abs (z + 2)

-- Define the set of points satisfying the equation
def S : Set ℂ :=
  {z : ℂ | equation z}

-- Define the function that gives the distance from the origin to a point on S
noncomputable def f (z : ℂ) : ℝ :=
  Complex.abs z

-- Define the property of k intersecting S at exactly one point
def intersects_once (k : ℝ) : Prop :=
  ∃! (z : ℂ), z ∈ S ∧ f z = k

-- State the theorem
theorem intersection_values :
  ∀ k : ℝ, intersects_once k ↔ k = 2 ∨ k = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l351_35185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_g_is_five_l351_35199

/-- The degree of a polynomial -/
def polyDegree (p : ℝ → ℝ) : ℕ := sorry

/-- Given polynomials f and g, prove that the degree of g is 5 -/
theorem degree_of_g_is_five (f g : ℝ → ℝ) :
  (∀ x, f x = -3 * x^5 + 4 * x^3 + 2 * x^2 - 6) →
  polyDegree (λ x ↦ f x + g x) = 2 →
  polyDegree g = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_g_is_five_l351_35199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l351_35147

noncomputable def IsArithmeticSeq (s : List ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Nat, i < s.length - 1 → s[i+1]! - s[i]! = d

noncomputable def IsGeometricSeq (s : List ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Nat, i < s.length - 1 → s[i+1]! / s[i]! = r

theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (IsArithmeticSeq [-2, a₁, a₂, -8] ∧ 
   IsGeometricSeq [-2, b₁, b₂, b₃, -8]) → 
  (a₂ - a₁) / b₂ = 1/2 := by
  sorry

#check sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l351_35147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l351_35169

-- Define the ellipse G
def ellipse_G (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the unit circle (renamed to avoid conflict)
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the tangent line l passing through (m, 0)
def tangent_line (m x y : ℝ) : Prop := ∃ k, y = k * (x - m)

-- Define the condition |m| ≥ 1
def m_condition (m : ℝ) : Prop := |m| ≥ 1

-- Define the length of AB as a function of m
noncomputable def AB_length (m : ℝ) : ℝ := (4 * Real.sqrt 3 * |m|) / (m^2 + 3)

theorem ellipse_properties (m : ℝ) (h_m : m_condition m) :
  -- 1. The foci of G are at (-√3, 0) and (√3, 0)
  (∃ x y, ellipse_G x y ∧ (x = -Real.sqrt 3 ∧ y = 0 ∨ x = Real.sqrt 3 ∧ y = 0)) ∧
  -- 2. |AB| = (4√3|m|)/(m^2 + 3) for m ∈ (-∞, -1] ∪ [1, +∞)
  (∀ x y, tangent_line m x y → unit_circle x y → 
    ∃ x1 y1 x2 y2, ellipse_G x1 y1 ∧ ellipse_G x2 y2 ∧ 
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = AB_length m) ∧
  -- 3. The maximum value of |AB| is 2
  (∀ m', m_condition m' → AB_length m' ≤ 2) ∧ (∃ m', m_condition m' ∧ AB_length m' = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l351_35169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l351_35165

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + a = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop :=
  x = 1 ∧ y = -2

-- Define the radius of the circle
noncomputable def circle_radius (a : ℝ) : ℝ :=
  Real.sqrt (5 - a)

theorem circle_properties :
  (∀ a : ℝ, circle_radius a = 1 → a = 4) ∧
  (circle_radius 0 = Real.sqrt 5) ∧
  (∀ a : ℝ, a < 5 → ∃ x y : ℝ, circle_equation x y a ∧ circle_center x y) :=
by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l351_35165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pal_number_l351_35123

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d * d) |>.sum

def is_pal (n : ℕ) : Prop :=
  0 ∉ n.digits 10 ∧ is_perfect_square (sum_of_squares_of_digits n)

theorem exists_pal_number (n : ℕ) (h : n > 1) : 
  ∃ m : ℕ, (m.digits 10).length = n ∧ is_pal m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pal_number_l351_35123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l351_35108

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (4 * x^4 - 3 * x^2 - 10 * x + 26) - Real.sqrt (4 * x^4 - 19 * x^2 - 6 * x + 34)

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), IsMaxOn f Set.univ x ∧ (x = (-1 + Real.sqrt 23) / 2 ∨ x = (-1 - Real.sqrt 23) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l351_35108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_two_l351_35114

def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + b

theorem tangent_line_at_negative_two (b : ℝ) :
  b = -6 →
  let x₀ := -2
  let y₀ := f b x₀
  let m := (3 * x₀^2 - 12)  -- f'(x) = 3x^2 - 12
  (λ x ↦ m * (x - x₀) + y₀) = (λ _ ↦ 10) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_two_l351_35114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stopping_distance_formula_l351_35106

/-- The distance traveled by a car after its engine is turned off -/
noncomputable def stopping_distance (v₀ m P : ℝ) : ℝ :=
  (m * v₀^3) / P

/-- Resistance force as a function of velocity -/
noncomputable def resistance_force (α : ℝ) (v : ℝ) : ℝ := α * v

/-- Theorem stating the relationship between stopping distance, initial speed, mass, and power -/
theorem stopping_distance_formula 
  (v₀ m P α : ℝ) 
  (h₁ : v₀ > 0) 
  (h₂ : m > 0) 
  (h₃ : P > 0)
  (h₄ : α > 0) :
  ∃ s : ℝ, s = stopping_distance v₀ m P ∧ s > 0 := by
  use stopping_distance v₀ m P
  constructor
  · rfl
  · sorry  -- The proof of positivity is omitted for brevity

#check stopping_distance_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stopping_distance_formula_l351_35106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_inequality_l351_35152

theorem infinite_solutions_inequality :
  Set.Infinite {p : ℝ × ℝ | (32 : ℝ)^(p.1^2 + p.2) + (32 : ℝ)^(p.1 + p.2^2) ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_inequality_l351_35152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_equals_one_ninth_l351_35136

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x - 4 else (3 : ℝ)^x

-- Theorem statement
theorem f_f_2_equals_one_ninth : f (f 2) = 1/9 := by
  -- Evaluate f(2)
  have h1 : f 2 = -2 := by
    simp [f]
    norm_num
  
  -- Evaluate f(-2)
  have h2 : f (-2) = 1/9 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f 2) = f (-2) := by rw [h1]
    _       = 1/9    := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_equals_one_ninth_l351_35136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weight_of_solution_Y_l351_35162

/-- The initial weight of solution Y in kilograms -/
def W : ℝ := sorry

/-- The percentage of liquid X in solution Y -/
def liquid_X_percentage : ℝ := 0.30

/-- The amount of water that evaporates from solution Y in kilograms -/
def evaporated_water : ℝ := 3

/-- The amount of solution Y added after evaporation in kilograms -/
def added_solution : ℝ := 3

/-- The new percentage of liquid X after evaporation and addition -/
def new_liquid_X_percentage : ℝ := 0.4125

theorem initial_weight_of_solution_Y : 
  liquid_X_percentage * W + liquid_X_percentage * added_solution = 
  new_liquid_X_percentage * W → W = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weight_of_solution_Y_l351_35162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_brother_ducks_l351_35177

theorem eldest_brother_ducks (n : ℕ) (total_ducks : ℕ) 
  (h1 : n = 7)
  (h2 : total_ducks = 29)
  (h3 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ (di : ℕ), di ≥ 1)
  (h4 : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → ∃ (di dj : ℕ), di < dj)
  (h5 : (Finset.range n).sum (λ i => i + 1) = total_ducks - 1) :
  ∃ (d : ℕ), d = 8 ∧ (∀ i : ℕ, 1 ≤ i ∧ i < n → ∃ (di : ℕ), di < d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_brother_ducks_l351_35177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_gain_percent_l351_35111

/-- Given that the cost price of 121 chocolates equals the selling price of 77 chocolates,
    prove that the gain percent is 4/7 * 100. -/
theorem chocolate_gain_percent :
  ∀ (C S : ℝ), -- C: cost price per chocolate, S: selling price per chocolate
  C > 0 → S > 0 →
  121 * C = 77 * S →
  (S - C) / C * 100 = 4 / 7 * 100 := by
  sorry

#check chocolate_gain_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_gain_percent_l351_35111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_definition_l351_35164

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane -/
def Point := ℝ × ℝ

/-- Distance between two points in a plane -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Definition of a circle as a set of points -/
def isOnCircle (c : Circle) (p : Point) : Prop :=
  distance p c.center = c.radius

theorem circle_definition (c : Circle) :
  ∀ p : Point, isOnCircle c p ↔ distance p c.center = c.radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_definition_l351_35164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l351_35180

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^(x-1) - 1) / (2^(x+1) + a)

theorem odd_function_properties (a : ℝ) (h1 : a > 0) 
(h2 : ∀ x, f a x = -f a (-x)) :
  (a = 2) ∧ 
  (∀ x y, x < y → f a x < f a y) ∧ 
  (∀ m, (∀ x, f a (2*m - m*Real.sin x) + f a (Real.cos x)^2 ≥ 0) → m ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l351_35180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revs_theorem_l351_35171

/-- A bicycle with wheels of different sizes -/
structure Bicycle where
  front_wheel_radius : ℝ
  back_wheel_radius : ℝ

/-- The number of revolutions made by each wheel -/
structure Revolutions where
  front : ℝ
  back : ℝ

/-- Convert inches to feet -/
noncomputable def inches_to_feet (inches : ℝ) : ℝ := inches / 12

/-- Calculate the number of revolutions made by the back wheel -/
noncomputable def back_wheel_revolutions (b : Bicycle) (front_revs : ℝ) : ℝ :=
  (b.front_wheel_radius * front_revs) / b.back_wheel_radius

/-- Theorem stating the relationship between front and back wheel revolutions -/
theorem back_wheel_revs_theorem (b : Bicycle) (revs : Revolutions) :
  b.front_wheel_radius = 1.5 ∧
  b.back_wheel_radius = inches_to_feet 6 ∧
  revs.front = 150 →
  revs.back = back_wheel_revolutions b revs.front ∧
  revs.back = 450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revs_theorem_l351_35171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l351_35109

-- Define the complex numbers representing the known vertices
def z₁ : ℂ := Complex.mk 1 2
def z₂ : ℂ := Complex.mk (-2) 1
def z₃ : ℂ := Complex.mk (-1) (-2)

-- Define the fourth vertex we want to prove
def z₄ : ℂ := Complex.mk 2 (-1)

-- Theorem statement
theorem square_fourth_vertex : 
  ∃ (center : ℂ), 
    (z₁ + z₂ + z₃ + z₄) / 4 = center ∧
    Complex.abs (z₁ - center) = Complex.abs (z₂ - center) ∧
    Complex.abs (z₁ - center) = Complex.abs (z₃ - center) ∧
    Complex.abs (z₁ - center) = Complex.abs (z₄ - center) ∧
    (z₁ - center).re * (z₂ - center).re + (z₁ - center).im * (z₂ - center).im = 0 ∧
    (z₂ - center).re * (z₃ - center).re + (z₂ - center).im * (z₃ - center).im = 0 ∧
    (z₃ - center).re * (z₄ - center).re + (z₃ - center).im * (z₄ - center).im = 0 ∧
    (z₄ - center).re * (z₁ - center).re + (z₄ - center).im * (z₁ - center).im = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l351_35109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_properties_l351_35167

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the configuration for the ellipse and points
structure EllipseConfig (a b : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_a_gt_b : a > b
  h_b_pos : b > 0
  h_on_major_axis : A.2 = 0 ∧ B.2 = 0
  h_one_inside : (A.1^2 / a^2) + (A.2^2 / b^2) < 1 ∧ A ≠ (0, 0)
  h_one_outside : (B.1^2 / a^2) + (B.2^2 / b^2) > 1
  h_x_product : A.1 * B.1 = a^2

-- Define angle function (this is a placeholder, as Lean doesn't have a built-in angle function)
noncomputable def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem ellipse_angle_properties {a b : ℝ} (cfg : EllipseConfig a b) :
  (∀ P Q : ℝ × ℝ, P ∈ Ellipse a b → Q ∈ Ellipse a b → 
    (∃ t : ℝ, P = cfg.A + t • (Q - cfg.A)) → 
    angle P cfg.B cfg.A = angle Q cfg.B cfg.A) ∧
  (∀ P Q : ℝ × ℝ, P ∈ Ellipse a b → Q ∈ Ellipse a b → 
    (∃ t : ℝ, P = cfg.B + t • (Q - cfg.B)) → 
    angle P cfg.A cfg.B + angle Q cfg.A cfg.B = π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_properties_l351_35167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l351_35173

/-- A function satisfying the given conditions -/
def F (f : ℝ → ℝ) : Prop :=
  (∀ x, (deriv (deriv (deriv f))) x > 0) ∧
  (∀ x, f (f x - Real.exp x) = 1)

/-- Theorem stating the range of a -/
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h1 : F f) (h2 : ∀ x, f x ≥ a * x + a) :
  0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l351_35173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_north_to_south_southeast_angle_l351_35161

/-- Represents the number of rays in the circular pattern -/
def num_rays : ℕ := 12

/-- Represents the angle between adjacent rays in degrees -/
noncomputable def angle_between_rays : ℝ := 360 / num_rays

/-- Represents the number of ray segments between North and South-Southeast -/
def segments_between : ℕ := 5

/-- Theorem: The smaller angle between North and South-Southeast rays is 150 degrees -/
theorem north_to_south_southeast_angle : 
  angle_between_rays * segments_between = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_north_to_south_southeast_angle_l351_35161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l351_35178

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (Real.sqrt 3 * b * Real.cos A = a * Real.cos B) →
  (a = Real.sqrt 2) →
  (c / a = Real.sin A / Real.sin B) →
  -- Conclusions
  (A = π / 3) ∧
  (a + b + c = 3 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l351_35178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l351_35143

/-- Given vectors a and b, and a real number t, if ‖a + t • b‖ = 3 and a • b = 2, 
    then the maximum value of t is 9/8 -/
theorem max_t_value {n : Type*} [NormedAddCommGroup ℝ] [InnerProductSpace ℝ ℝ] 
    (a b : ℝ) (t : ℝ) 
    (h1 : ‖a + t * b‖ = 3) 
    (h2 : a * b = 2) : 
    ∃ (t_max : ℝ), t ≤ t_max ∧ t_max = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l351_35143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_coprime_numbers_sum_l351_35189

theorem five_coprime_numbers_sum (a b c d e : ℕ+) : 
  Nat.Coprime a.val b.val ∧ 
  Nat.Coprime a.val c.val ∧ 
  Nat.Coprime a.val d.val ∧ 
  Nat.Coprime a.val e.val ∧ 
  Nat.Coprime b.val c.val ∧ 
  Nat.Coprime b.val d.val ∧ 
  Nat.Coprime b.val e.val ∧ 
  Nat.Coprime c.val d.val ∧ 
  Nat.Coprime c.val e.val ∧ 
  Nat.Coprime d.val e.val →
  a * b = 2381 →
  b * c = 7293 →
  c * d = 19606 →
  d * e = 74572 →
  a + b + c + d + e = 34121 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_coprime_numbers_sum_l351_35189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_paths_count_l351_35142

/-- Represents a regular octagonal pyramid -/
structure OctagonalPyramid where
  base : Fin 8
  triangles : Fin 8

/-- Represents a path from apex to base -/
inductive PathType
  | Direct : OctagonalPyramid → PathType
  | OneTriangle : OctagonalPyramid → Fin 8 → PathType
  | TwoTriangles : OctagonalPyramid → Fin 8 → Fin 8 → PathType

/-- Checks if two triangles are adjacent -/
def adjacent (t1 t2 : Fin 8) : Prop :=
  (t1.val + 1) % 8 = t2.val ∨ (t2.val + 1) % 8 = t1.val

/-- Checks if a path is valid according to the rules -/
def isValidPath (p : PathType) : Prop :=
  match p with
  | PathType.Direct _ => true
  | PathType.OneTriangle _ _ => true
  | PathType.TwoTriangles _ t1 t2 => adjacent t1 t2

/-- Counts the number of valid paths -/
def countValidPaths (pyramid : OctagonalPyramid) : ℕ :=
  sorry

/-- Main theorem: There are 32 distinct valid paths -/
theorem distinct_paths_count (pyramid : OctagonalPyramid) : 
  countValidPaths pyramid = 32 := by
  sorry

#check distinct_paths_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_paths_count_l351_35142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l351_35155

-- Define the points A and B
def A : ℝ × ℝ := (-3, 5)
def B : ℝ × ℝ := (2, 15)

-- Define the line l: 3x - 4y + 4 = 0
def l (x y : ℝ) : Prop := 3 * x - 4 * y + 4 = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement of the theorem
theorem min_distance_sum :
  ∃ (P : ℝ × ℝ), l P.1 P.2 ∧
    (∀ (Q : ℝ × ℝ), l Q.1 Q.2 →
      distance P A + distance P B ≤ distance Q A + distance Q B) ∧
    distance P A + distance P B = 5 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l351_35155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_proof_l351_35116

/-- Calculates the speed given distance and time -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Converts minutes to hours -/
noncomputable def minutesToHours (minutes : ℝ) : ℝ := minutes / 60

theorem speed_difference_proof (distance : ℝ) (nora_time : ℝ) (mia_time : ℝ)
  (h1 : distance = 8)
  (h2 : nora_time = 15)
  (h3 : mia_time = 40) :
  speed distance (minutesToHours nora_time) - speed distance (minutesToHours mia_time) = 20 := by
  sorry

#check speed_difference_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_proof_l351_35116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_35_l351_35101

/-- Represents a clock with 12 hours --/
structure Clock :=
  (hours : Nat)
  (minutes : Nat)

/-- The angle represented by each hour on the clock face --/
def hourAngle : ℚ := 30

/-- Calculates the angle of the hour hand from 12 o'clock --/
def hourHandAngle (c : Clock) : ℚ :=
  (c.hours % 12 : ℚ) * hourAngle + (c.minutes : ℚ) / 60 * hourAngle

/-- Calculates the angle of the minute hand from 12 o'clock --/
def minuteHandAngle (c : Clock) : ℚ :=
  (c.minutes : ℚ) / 60 * 360

/-- Calculates the absolute difference between two angles --/
def angleDifference (a b : ℚ) : ℚ :=
  abs (a - b)

/-- Theorem stating that at 7:35, the angle between the clock hands is 17.5° --/
theorem clock_angle_at_7_35 :
  let c : Clock := ⟨7, 35⟩
  angleDifference (hourHandAngle c) (minuteHandAngle c) = 35/2 := by
  sorry

#eval angleDifference (hourHandAngle ⟨7, 35⟩) (minuteHandAngle ⟨7, 35⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_35_l351_35101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treadmill_time_difference_l351_35166

/-- Represents a day's treadmill usage --/
structure DayUsage where
  distance : ℚ
  speed : ℚ

/-- Calculates the time spent on the treadmill for a given day --/
def timeSpent (day : DayUsage) : ℚ :=
  day.distance / day.speed

/-- Calculates the total time spent on the treadmill for multiple days --/
def totalTimeSpent (days : List DayUsage) : ℚ :=
  days.map timeSpent |>.sum

/-- Calculates the total distance covered over multiple days --/
def totalDistance (days : List DayUsage) : ℚ :=
  days.map (λ d => d.distance) |>.sum

/-- The actual treadmill usage over 4 days --/
def actualUsage : List DayUsage := [
  ⟨3, 6⟩, ⟨2, 5⟩, ⟨4, 4⟩, ⟨1, 2⟩
]

/-- The constant speed for the hypothetical scenario --/
def constantSpeed : ℚ := 5

theorem treadmill_time_difference : 
  (totalTimeSpent actualUsage - totalDistance actualUsage / constantSpeed) * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_treadmill_time_difference_l351_35166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_triangle_side_length_l351_35121

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x + π / 6) + cos (2 * x + π / 6)

theorem f_monotone_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) :=
sorry

theorem triangle_side_length (A B C a b c : ℝ) :
  f A = sqrt 3 →
  sin C = 1 / 3 →
  a = 3 →
  b = sqrt 3 + 2 * sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_triangle_side_length_l351_35121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_choice_and_essay_l351_35137

/-- The probability of selecting a multiple-choice question and an essay question
    when choosing 3 questions from a set of 12 multiple-choice, 4 fill-in-the-blank,
    and 6 essay questions. -/
theorem probability_multiple_choice_and_essay :
  let total_questions := 12 + 4 + 6
  let multiple_choice := 12
  let fill_in_blank := 4
  let essay := 6
  let selected := 3
  (Nat.choose multiple_choice 1 * (Nat.choose essay 1 * Nat.choose fill_in_blank 1 + Nat.choose essay 2) +
   Nat.choose multiple_choice 2 * Nat.choose essay 1) /
  (Nat.choose total_questions selected - Nat.choose (fill_in_blank + essay) selected) = 
  (Nat.choose multiple_choice 1 * (Nat.choose essay 1 * Nat.choose fill_in_blank 1 + Nat.choose essay 2) +
   Nat.choose multiple_choice 2 * Nat.choose essay 1) /
  (Nat.choose total_questions selected - Nat.choose (fill_in_blank + essay) selected) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_choice_and_essay_l351_35137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_probability_l351_35135

theorem hiring_probability (n k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  (Nat.choose n k - Nat.choose (n - 2) k : ℚ) / Nat.choose n k = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_probability_l351_35135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l351_35157

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : IsArithmeticSequence a) :
  a 3 + a 7 = 15 → a 2 + a 8 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l351_35157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l351_35133

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1
  else -2^(-x) + 2*x + 1

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2*x + 9

-- State the theorem
theorem odd_function_and_inequality (m : ℝ) : 
  (∀ x, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x₁ ∈ Set.Ioc (-1) 3, ∃ x₂, f x₁ > g m x₂) →
  (∀ x < 0, f x = -2^(-x) + 2*x + 1) ∧
  m ≤ 1/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l351_35133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l351_35122

-- Define the points A, B, C
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (2, 1)

-- Define vectors AB and AC
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define AP in terms of lambda and mu
def AP (lambda mu : ℝ) : ℝ × ℝ := (lambda * AB.1 + mu * AC.1, lambda * AB.2 + mu * AC.2)

theorem vector_problem (lambda mu : ℝ) 
  (h1 : dot_product (AP lambda mu) AB = 0)
  (h2 : dot_product (AP lambda mu) AC = 3) :
  dot_product AB AC = 4 ∧ lambda + mu = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l351_35122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_calculation_l351_35102

/-- Calculates the total time for a round trip given rowing speed, current speed, and distance -/
noncomputable def total_rowing_time (rowing_speed : ℝ) (current_speed : ℝ) (distance : ℝ) : ℝ :=
  distance / (rowing_speed + current_speed) + distance / (rowing_speed - current_speed)

/-- Theorem: Given the specified conditions, the total rowing time is 10 hours -/
theorem rowing_time_calculation :
  let rowing_speed : ℝ := 10
  let current_speed : ℝ := 2
  let distance : ℝ := 48
  total_rowing_time rowing_speed current_speed distance = 10 := by
  -- Unfold the definition and simplify
  unfold total_rowing_time
  -- Perform algebraic manipulations
  simp [add_div, sub_div]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_calculation_l351_35102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l351_35154

noncomputable section

open Real

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (sin A) = b / (sin B) ∧
  a / (sin A) = c / (sin C)

-- State the theorem
theorem triangle_properties
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : triangle A B C a b c)
  (h_eq : b * sin A = sqrt 3 * a * cos B)
  (h_b : b = 3)
  (h_sin : sin C = 2 * sin A) :
  B = Real.pi / 3 ∧ a = sqrt 3 ∧ c = 2 * sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l351_35154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_bread_count_l351_35196

/-- Represents the number of loaves baked on each day of the week -/
def BreadSequence : List ℕ → Prop :=
  fun s => s.length = 6 ∧
           s.get? 1 = some 7 ∧ s.get? 2 = some 10 ∧ s.get? 3 = some 14 ∧ 
           s.get? 4 = some 19 ∧ s.get? 5 = some 25 ∧
           ∀ i : Fin 4, 
             (s.get? i.val).bind (fun x => 
             (s.get? (i.val + 1)).bind (fun y => 
             (s.get? (i.val + 2)).map (fun z => 
               y - x = z - y - 1))) = some true

theorem wednesday_bread_count (s : List ℕ) (h : BreadSequence s) : s.get? 0 = some 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_bread_count_l351_35196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_perimeter_l351_35134

-- Define a sequence of squares
def SquareSequence (n : ℕ) (area : ℝ) : Prop :=
  ∀ i, i < n → 
    (i = 0 ∧ area = 1) ∨ 
    (∃ prev_area, SquareSequence i prev_area ∧ area = 2 * prev_area)

-- Theorem statement
theorem largest_square_perimeter :
  ∀ area : ℝ, SquareSequence 5 area → 
    ∃ side : ℝ, side ^ 2 = area ∧ 4 * side = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_perimeter_l351_35134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_seven_one_to_2017_l351_35120

/-- Count of digit 7 appearances in a natural number -/
def countSevenInNumber (n : Nat) : Nat :=
  sorry

/-- Count of digit 7 appearances in a range of natural numbers -/
def countSevenInRange (start finish : Nat) : Nat :=
  sorry

/-- The main theorem stating that the count of digit 7 appearances
    in the sequence of natural numbers from 1 to 2017 inclusive is 602 -/
theorem count_seven_one_to_2017 :
  countSevenInRange 1 2017 = 602 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_seven_one_to_2017_l351_35120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l351_35168

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (total_length : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let bridge_length := total_length - train_length
  (train_length + bridge_length) / train_speed_ms

theorem train_crossing_bridge_time :
  train_crossing_time 150 45 225 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l351_35168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l351_35104

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a*x + Real.sin (x + Real.pi/2)

theorem even_function_condition (a : ℝ) :
  (∀ x, y x a = y (-x) a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l351_35104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cost_effective_years_l351_35172

/-- Represents the linear regression equation for a machine's profit over time. -/
def estimated_profit (x : ℕ) : ℝ := 10.47 - 1.3 * (x : ℝ)

/-- Represents the condition for the machine to be cost-effective. -/
def is_cost_effective (x : ℕ) : Prop := estimated_profit x ≥ 0

/-- Theorem stating that 8 is the maximum number of cost-effective years. -/
theorem max_cost_effective_years :
  (∀ x : ℕ, x ≤ 8 → is_cost_effective x) ∧
  (∀ x : ℕ, x > 8 → ¬is_cost_effective x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cost_effective_years_l351_35172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_unit_interval_l351_35174

-- Define the function f(x) = e^x - x
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

-- State the theorem
theorem max_value_of_f_on_unit_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 1 → f x ≤ f c ∧ f c = Real.exp 1 - 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_unit_interval_l351_35174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l351_35105

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 6 + Real.cos x ^ 4

-- State the theorem about the range of g
theorem g_range :
  ∃ (min_val : ℝ), 
    (∀ (x : ℝ), g x ≥ min_val) ∧ 
    (∀ (x : ℝ), g x ≤ 1) ∧
    (g ((Real.sqrt 7 - 1) / 3) = min_val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l351_35105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_probability_is_one_sixth_l351_35163

/-- Represents a hexagonal dart board with an equilateral triangle at its center -/
structure HexagonalDartBoard where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Calculates the area of an equilateral triangle -/
noncomputable def triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- Calculates the area of a regular hexagon -/
noncomputable def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

/-- Calculates the probability of a dart landing in the center triangle -/
noncomputable def center_triangle_probability (board : HexagonalDartBoard) : ℝ :=
  triangle_area board.side_length / hexagon_area board.side_length

/-- Theorem: The probability of a dart landing in the center triangle is 1/6 -/
theorem center_triangle_probability_is_one_sixth (board : HexagonalDartBoard) :
  center_triangle_probability board = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_probability_is_one_sixth_l351_35163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_solution_l351_35197

/-- A quadratic polynomial with real coefficients -/
def quadratic_polynomial (a b c : ℝ) : ℂ → ℂ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_polynomial_solution :
  ∃ (a b c : ℝ),
    (quadratic_polynomial a b c) (3 - 4*I) = 0 ∧
    b = 10 ∧
    quadratic_polynomial a b c = fun x ↦ -5/3 * x^2 + 10 * x - 125/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_solution_l351_35197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_profit_percentage_l351_35191

/-- Calculates the profit percentage for a retailer selling pens -/
theorem pen_profit_percentage 
  (num_pens : ℕ) 
  (price_pens : ℕ) 
  (discount_percent : ℚ) : 
  num_pens = 140 → 
  price_pens = 36 → 
  discount_percent = 1/100 → 
  (((num_pens : ℚ) * (1 - discount_percent) - price_pens) / price_pens) * 100 = 285 := by
  sorry

#check pen_profit_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_profit_percentage_l351_35191
