import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_specific_trapezoid_l269_26928

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- Length of side PS -/
  ps : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side SR -/
  sr : ℝ
  /-- PS equals QR -/
  ps_eq_qr : ps = qr
  /-- PS is positive -/
  ps_pos : ps > 0
  /-- PQ is positive -/
  pq_pos : pq > 0
  /-- SR is positive -/
  sr_pos : sr > 0
  /-- SR is greater than PQ -/
  sr_gt_pq : sr > pq

/-- The diameter of the inscribed circle in the trapezoid -/
noncomputable def inscribedCircleDiameter (t : InscribedCircleTrapezoid) : ℝ :=
  2 * Real.sqrt (t.ps ^ 2 - ((t.sr - t.pq) / 2) ^ 2)

/-- Theorem stating the diameter of the inscribed circle for specific trapezoid measurements -/
theorem inscribed_circle_diameter_specific_trapezoid :
    let t : InscribedCircleTrapezoid := {
      ps := 25,
      qr := 25,
      pq := 18,
      sr := 32,
      ps_eq_qr := by rfl,
      ps_pos := by norm_num,
      pq_pos := by norm_num,
      sr_pos := by norm_num,
      sr_gt_pq := by norm_num
    }
    inscribedCircleDiameter t = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_specific_trapezoid_l269_26928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_over_n_l269_26930

open Real

-- Define the functions f and h
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m
noncomputable def h (n : ℝ) (x : ℝ) : ℝ := 6*n^2 * log x - 4*n*x

-- State the theorem
theorem max_m_over_n (n : ℝ) (m : ℝ) (x₀ : ℝ) :
  n > 0 →
  x₀ > 0 →
  f m x₀ = h n x₀ →
  (deriv (f m)) x₀ = (deriv (h n)) x₀ →
  ∃ (max_val : ℝ), max_val = 3 * exp (-1/6) ∧ m / n ≤ max_val :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_over_n_l269_26930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_profit_share_is_50_l269_26909

/-- Represents an investor in the business --/
structure Investor where
  investment : ℚ
  months : ℚ

/-- Calculates the weighted investment of an investor --/
def weightedInvestment (i : Investor) : ℚ :=
  i.investment * i.months

/-- Calculates the total weighted investment of all investors --/
def totalWeightedInvestment (investors : List Investor) : ℚ :=
  investors.map weightedInvestment |>.sum

/-- Calculates an investor's share of the profit --/
def profitShare (i : Investor) (investors : List Investor) (totalProfit : ℚ) : ℚ :=
  (weightedInvestment i / totalWeightedInvestment investors) * totalProfit

theorem a_profit_share_is_50 (a b : Investor) (totalProfit : ℚ) :
  a.investment = 100 ∧ a.months = 12 ∧
  b.investment = 200 ∧ b.months = 6 ∧
  totalProfit = 100 →
  profitShare a [a, b] totalProfit = 50 := by
  sorry

#eval profitShare ⟨100, 12⟩ [⟨100, 12⟩, ⟨200, 6⟩] 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_profit_share_is_50_l269_26909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_reduction_l269_26968

/-- Given an initial price P and a discount amount D, 
    the percent reduction is equal to (D/P) * 100%. -/
theorem percent_reduction (P D : ℝ) (h₁ : P > 0) (h₂ : D ≥ 0) (h₃ : D ≤ P) :
  (D / P) * 100 = (D / P) * 100 :=
by
  -- The proof is trivial as we're stating equality to itself
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_reduction_l269_26968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l269_26976

theorem infinite_geometric_series_sum :
  let a : ℚ := 3/2
  let r : ℚ := -4/9
  (∑' n, a * r^n) = 27/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l269_26976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_length_ratio_l269_26994

theorem shadow_length_ratio (α β : Real) (h1 : Real.tan (α - β) = 1 / 3) (h2 : Real.tan β = 1) : Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_length_ratio_l269_26994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_complex_l269_26933

/-- Predicate to check if two sets of three complex points form similar triangles. -/
def IsSimilar (t1 t2 : Set ℂ) : Prop :=
  ∃ (z w : ℂ), z ≠ 0 ∧ ∀ p ∈ t1, ∃ q ∈ t2, q = z * p + w

/-- Two triangles in the complex plane are similar if and only if their vertices satisfy a specific equation. -/
theorem triangle_similarity_complex (a b c a' b' c' : ℂ) :
  IsSimilar {a, b, c} {a', b', c'} ↔ a' * (b - c) + b' * (c - a) + c' * (a - b) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_complex_l269_26933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_a_l269_26979

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for the solution set of f(x) ≥ 5
theorem solution_set_f_geq_5 :
  {x : ℝ | f x ≥ 5} = Set.Ici 2 ∪ Set.Iic (-3) :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x > a^2 - 2*a - 5} = Set.Ioo (-2) 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_a_l269_26979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_ellipse_l269_26942

/-- An ellipse with foci at (-1,0) and (1,0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a^2 - b^2 = 1

/-- The line x-y+3=0 -/
def line (x y : ℝ) : Prop := x - y + 3 = 0

/-- The ellipse intersects the line -/
def intersects (e : Ellipse) : Prop :=
  ∃ x y : ℝ, x^2 / (e.b^2 + 1) + y^2 / e.b^2 = 1 ∧ line x y

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 / (e.b^2 + 1))

/-- The theorem to be proved -/
theorem max_eccentricity_ellipse :
  ∀ e : Ellipse, intersects e →
  eccentricity e ≤ eccentricity ⟨Real.sqrt 5, 2, by norm_num⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_ellipse_l269_26942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l269_26931

theorem triangle_problem (a b c A B C : ℝ) : 
  b = 3 → 
  c = 1 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 2 →
  (Real.cos A = 1/3 ∧ a = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l269_26931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x14_in_quotient_l269_26923

theorem coefficient_of_x14_in_quotient : ∃ (q r : Polynomial ℤ),
  X^1951 - 1 = (X^4 + X^3 + 2*X^2 + X + 1) * q + r ∧
  r.degree < 4 ∧
  q.coeff 14 = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x14_in_quotient_l269_26923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_profit_is_1000_l269_26967

/-- Represents the partnership between Mary and Mike -/
structure Partnership where
  mary_investment : ℚ
  mike_investment : ℚ
  total_profit : ℚ
  mary_extra : ℚ

/-- Calculates the portion of profit divided equally -/
noncomputable def equal_profit_portion (p : Partnership) : ℚ :=
  let remaining_profit := p.total_profit - (p.mary_investment / (p.mary_investment + p.mike_investment) * 
    (4/5 * p.total_profit) + p.mike_investment / (p.mary_investment + p.mike_investment) * (1/5 * p.total_profit))
  2 * (p.mary_extra / 3)

/-- Theorem stating that the equal profit portion is $1000 given the conditions -/
theorem equal_profit_is_1000 (p : Partnership) 
  (h1 : p.mary_investment = 800)
  (h2 : p.mike_investment = 200)
  (h3 : p.total_profit = 3000)
  (h4 : p.mary_extra = 1200) :
  equal_profit_portion p = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_profit_is_1000_l269_26967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_ellipse_l269_26997

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt ((a^2 - b^2) / a^2)

/-- The eccentricity of the ellipse x²/9 + y²/4 = 1 is √5/3 -/
theorem eccentricity_of_specific_ellipse :
  eccentricity 3 2 = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_ellipse_l269_26997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_properties_l269_26978

open Real

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Define an obtuse triangle
def ObtuseTriangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  Triangle a b c A B C ∧ (A > Real.pi/2 ∨ B > Real.pi/2 ∨ C > Real.pi/2)

-- Theorem statement
theorem obtuse_triangle_properties :
  ∀ (a b c : ℝ) (A B C : ℝ),
  ObtuseTriangle a b c A B C →
  a = 7 →
  b = 3 →
  Real.cos C = 11/14 →
  c = 5 ∧
  A = 2*Real.pi/3 ∧
  Real.sin (2*C - Real.pi/6) = 71/98 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_properties_l269_26978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l269_26901

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  2 * (Real.log a / Real.log b / (a + b) + Real.log b / Real.log c / (b + c) + Real.log c / Real.log a / (c + a)) ≥ 9 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l269_26901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_tai_chi_function_l269_26990

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define a Tai Chi function
def is_tai_chi_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the specific function
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

-- Theorem statement
theorem not_tai_chi_function : ¬(is_tai_chi_function f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_tai_chi_function_l269_26990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trip_satisfies_conditions_l269_26957

/-- Represents a bus trip with an accident -/
structure BusTrip where
  initialSpeed : ℚ
  totalDistance : ℚ
  accidentDelay : ℚ
  speedReductionFactor : ℚ
  lateness : ℚ
  alternateAccidentDistance : ℚ
  alternateLatenessDifference : ℚ

/-- The specific bus trip described in the problem -/
def specificTrip : BusTrip where
  initialSpeed := 30
  totalDistance := 180
  accidentDelay := 1
  speedReductionFactor := 2/3
  lateness := 5
  alternateAccidentDistance := 30
  alternateLatenessDifference := 1

/-- Calculates the total travel time for a given trip -/
def totalTravelTime (trip : BusTrip) : ℚ :=
  2 + trip.accidentDelay + (trip.totalDistance - 2 * trip.initialSpeed) / (trip.speedReductionFactor * trip.initialSpeed)

/-- Calculates the total travel time for the alternate scenario -/
def alternateTravelTime (trip : BusTrip) : ℚ :=
  2 + trip.alternateAccidentDistance / trip.initialSpeed + trip.accidentDelay +
  (trip.totalDistance - 2 * trip.initialSpeed - trip.alternateAccidentDistance) / (trip.speedReductionFactor * trip.initialSpeed)

/-- Theorem stating that the specific trip satisfies all conditions -/
theorem specific_trip_satisfies_conditions :
  let trip := specificTrip
  totalTravelTime trip - (totalTravelTime trip - trip.lateness) = trip.lateness ∧
  alternateTravelTime trip - (alternateTravelTime trip - (trip.lateness - trip.alternateLatenessDifference)) = 
    trip.lateness - trip.alternateLatenessDifference ∧
  trip.totalDistance = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trip_satisfies_conditions_l269_26957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_root_with_small_fraction_l269_26902

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  (∀ k < n, ∀ s : ℝ, 0 < s → s < 1/500 → ¬(∃ t : ℕ, ((k : ℝ) + s)^3 = t)) →
  0 < r → r < 1/500 → 
  (∃ t : ℕ, ((n : ℝ) + r)^3 = t) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_root_with_small_fraction_l269_26902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l269_26912

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def sides_form_geometric_progression (t : Triangle) : Prop :=
  ∃ r : Real, r > 0 ∧ t.b = t.a * r ∧ t.c = t.b * r

def sine_relation (t : Triangle) : Prop :=
  Real.sin t.C = 2 * Real.sin t.A

theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : sides_form_geometric_progression t)
  (h3 : sine_relation t) :
  Real.cos t.B = 3/4 ∧ 
  t.B ≤ Real.pi/3 ∧
  (t.B = Real.pi/3 → t.a = t.b ∧ t.b = t.c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l269_26912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_eq_twelve_l269_26937

/-- Given positive integers a and b satisfying ab - 5a + 6b = 522, 
    the minimal possible value of |a - b| is 12. -/
theorem min_abs_diff_eq_twelve (a b : ℕ+) (h : a.val * b.val - 5 * a.val + 6 * b.val = 522) :
  (∀ (c d : ℕ+), c.val * d.val - 5 * c.val + 6 * d.val = 522 → |Int.ofNat a.val - Int.ofNat b.val| ≤ |Int.ofNat c.val - Int.ofNat d.val|) ∧
  |Int.ofNat a.val - Int.ofNat b.val| = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_eq_twelve_l269_26937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_self_descriptive_are_valid_l269_26982

/-- A sequence satisfies the self-descriptive property if each element counts its own occurrences. -/
def IsSelfDescriptive (s : List ℕ) : Prop :=
  ∀ j, j < s.length → s.get? j = some (s.filter (· = j)).length

/-- The set of all valid self-descriptive sequences. -/
def ValidSequences : Set (List ℕ) :=
  { s | IsSelfDescriptive s ∧
    (s = [1, 2, 1, 0] ∨
     s = [2, 0, 2, 0] ∨
     s = [2, 1, 2, 0, 0] ∨
     ∃ n ≥ 6, s = List.cons (n - 3) (2 :: 1 :: List.replicate (n - 5) 0 ++ [1, 0, 0, 0])) }

/-- All self-descriptive sequences are in the set of valid sequences. -/
theorem all_self_descriptive_are_valid (s : List ℕ) :
  IsSelfDescriptive s → s ∈ ValidSequences := by
  sorry

#check all_self_descriptive_are_valid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_self_descriptive_are_valid_l269_26982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_impact_x_coordinate_l269_26950

/-- The x-coordinate of the point of impact for a ball in free fall -/
theorem ball_impact_x_coordinate (g R : ℝ) (α : ℝ) (h : 0 < g) (h' : 0 < R) :
  let V := Real.sqrt (2 * g * R * Real.cos α)
  let x := λ t ↦ R * Real.sin α + V * Real.cos α * t
  let y := λ t ↦ R * (1 - Real.cos α) + V * Real.sin α * t - g * t^2 / 2
  let T := Real.sqrt (2 * R / g) * (Real.sin α * Real.sqrt (Real.cos α) + Real.sqrt (1 - Real.cos α^3))
  x T = R * (Real.sin α + Real.sin (2 * α) + Real.sqrt (Real.cos α * (1 - Real.cos α^3))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_impact_x_coordinate_l269_26950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l269_26955

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Represents a chord of an ellipse -/
structure Chord where
  start : Point
  finish : Point  -- Changed 'end' to 'finish' to avoid keyword conflict

/-- The theorem to be proved -/
theorem ellipse_chord_theorem (e : Ellipse) 
  (A B C : Point) (M N : Point) (F₁ : Point) 
  (AB MN : Chord) :
  -- Conditions
  (A.x = -e.a ∧ A.y = 0) →  -- A is the left vertex
  (C.x = 0) →  -- C is on y-axis
  (F₁.x = -e.a * e.b / e.a) →  -- F₁ is the left focus
  (AB.start = A ∧ AB.finish = B) →
  (MN.start = M ∧ MN.finish = N) →
  (∃ (k : ℝ), (B.y - A.y) / (B.x - A.x) = (N.y - M.y) / (N.x - M.x)) →  -- MN parallel to AB
  -- Conclusion
  e.a * ((M.x - N.x)^2 + (M.y - N.y)^2).sqrt = 
    ((A.x - B.x)^2 + (A.y - B.y)^2).sqrt * ((A.x - C.x)^2 + (A.y - C.y)^2).sqrt :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l269_26955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l269_26938

open Real Set

theorem constant_function_proof (f : ℝ → ℝ) 
  (h1 : ∀ x, x ∈ Ioo 0 1 → f x > 0)
  (h2 : ∀ x y, x ∈ Ioo 0 1 → y ∈ Ioo 0 1 → f x / f y + f (1 - x) / f (1 - y) ≤ 2) :
  ∀ x y, x ∈ Ioo 0 1 → y ∈ Ioo 0 1 → f x = f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l269_26938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_august_2023_prediction_l269_26963

/-- Represents the month codes -/
def month_codes : List ℚ := [1, 2, 3, 4, 5]

/-- Represents the lithium carbonate prices in thousands of yuan -/
def prices : List ℚ := [1/2, 7/10, 1, 6/5, 8/5]

/-- The y-intercept of the regression equation -/
def y_intercept : ℚ := 19/100

/-- Calculates the mean of a list of rational numbers -/
def mean (list : List ℚ) : ℚ :=
  (list.sum) / (list.length : ℚ)

/-- Calculates the slope of the regression line -/
def calculate_slope (x_mean y_mean : ℚ) : ℚ :=
  (y_mean - y_intercept) / x_mean

/-- Predicts the price for a given month code -/
def predict_price (slope : ℚ) (x : ℚ) : ℚ :=
  slope * x + y_intercept

/-- Theorem stating that the predicted price for August 2023 is 2.35 -/
theorem august_2023_prediction :
  let x_mean := mean month_codes
  let y_mean := mean prices
  let slope := calculate_slope x_mean y_mean
  predict_price slope 8 = 47/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_august_2023_prediction_l269_26963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_smallest_solutions_l269_26945

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def equation (x : ℝ) : Prop := x - (floor x : ℝ) = 2 / (floor x : ℝ)

def is_solution (x : ℝ) : Prop := equation x

theorem product_of_smallest_solutions :
  ∃ (s₁ s₂ s₃ : ℝ),
    (∀ x, is_solution x → x ≥ s₁) ∧
    (∀ x, is_solution x ∧ x > s₁ → x ≥ s₂) ∧
    (∀ x, is_solution x ∧ x > s₂ → x ≥ s₃) ∧
    is_solution s₁ ∧ is_solution s₂ ∧ is_solution s₃ ∧
    s₁^2 * s₂^2 * s₃^2 = 9801 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_smallest_solutions_l269_26945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l269_26905

open Real

/-- The function f(x) = e^x / x - ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x / x - a * x

/-- The theorem statement -/
theorem inequality_holds_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ / x₂ - f a x₂ / x₁ < 0) ↔
  a ≤ exp 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l269_26905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l269_26913

theorem trigonometric_identities (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin α = Real.sqrt 5 / 5) : 
  Real.sin (π/4 + α) = -(Real.sqrt 10)/10 ∧ 
  Real.cos (5*π/6 - 2*α) = (-4-3*Real.sqrt 3)/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l269_26913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l269_26972

/-- Triangle ABC with centroid M -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  centroid : M = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Length of a vector -/
noncomputable def length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (length v * length w))

/-- Area of a triangle given two sides and the included angle -/
noncomputable def area (a b θ : ℝ) : ℝ := (1/2) * a * b * Real.sin θ

theorem triangle_property (t : Triangle) (a b c : ℝ) :
  let MA := vector t.M t.A
  let MB := vector t.M t.B
  let MC := vector t.M t.C
  (a * MA.1 + b * MB.1 + (Real.sqrt 3 / 3) * c * MC.1 = 0 ∧
   a * MA.2 + b * MB.2 + (Real.sqrt 3 / 3) * c * MC.2 = 0) →
  (angle (vector t.B t.A) (vector t.C t.A) = π / 6 ∧
   (a = 3 → area a a (π / 6) = 9 * Real.sqrt 3 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l269_26972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_one_l269_26911

/-- The curve defined by parametric equations x = t^2 and y = 2t -/
def curve (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

/-- The point P -/
def P : ℝ × ℝ := (1, 0)

/-- The distance between P and a point on the curve -/
noncomputable def distance (t : ℝ) : ℝ :=
  ((curve t).1 - P.1)^2 + ((curve t).2 - P.2)^2

theorem shortest_distance_is_one :
  ∃ (t : ℝ), ∀ (s : ℝ), distance t ≤ distance s ∧ Real.sqrt (distance t) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_one_l269_26911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l269_26904

noncomputable def f (x : ℝ) : ℝ := (x^2 + 7) / x

def has_solutions (a : ℝ) : Prop := ∃ x : ℝ, 2 < x ∧ x < 7 ∧ a ≤ f x

theorem range_of_a :
  ∀ a : ℝ, has_solutions a ↔ a < 8 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l269_26904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_l269_26984

/-- The perimeter of a triangle -/
def PerimeterOfTriangle (triangle : Set Point) : ℝ := sorry

/-- Predicate stating that two triangles are similar -/
def IsSimilarTriangle (triangle1 triangle2 : Set Point) : Prop := sorry

/-- The length of the shortest side of a triangle -/
def MinSide (triangle : Set Point) : ℝ := sorry

/-- Given two similar triangles ABC and DEF, where ABC has sides 3, 5, and 6,
    and the shortest side of DEF is 9, prove that the perimeter of DEF is 42. -/
theorem similar_triangles_perimeter
  (ABC DEF : Set Point)
  (h1 : IsSimilarTriangle ABC DEF)
  (h2 : PerimeterOfTriangle ABC = 14)
  (h3 : MinSide ABC = 3)
  (h4 : MinSide DEF = 9) :
  PerimeterOfTriangle DEF = 42 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_l269_26984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l269_26970

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of the altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- One of the angles at the base in radians -/
  baseAngle : ℝ

/-- The area of an isosceles triangle with given properties -/
noncomputable def triangleArea (t : IsoscelesTriangle) : ℝ :=
  200 * Real.sqrt 2 - 200

/-- Theorem stating the area of the specific isosceles triangle -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle,
    t.altitude = 10 ∧
    t.perimeter = 40 ∧
    t.baseAngle = π / 4 →
    triangleArea t = 200 * Real.sqrt 2 - 200 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l269_26970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_and_nine_point_center_l269_26947

open Complex

/-- Represents a triangle in the complex plane -/
structure ComplexTriangle where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ
  h₁ : abs z₁ = 1
  h₂ : abs z₂ = 1
  h₃ : abs z₃ = 1

/-- The orthocenter of a triangle represented by complex numbers -/
noncomputable def orthocenter (t : ComplexTriangle) : ℂ := t.z₁ + t.z₂ + t.z₃

/-- The center of the nine-point circle of a triangle represented by complex numbers -/
noncomputable def ninePointCenter (t : ComplexTriangle) : ℂ := (1/2) * (t.z₁ + t.z₂ + t.z₃)

theorem orthocenter_and_nine_point_center (t : ComplexTriangle) :
  orthocenter t = t.z₁ + t.z₂ + t.z₃ ∧
  ninePointCenter t = (1/2) * (t.z₁ + t.z₂ + t.z₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_and_nine_point_center_l269_26947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_divides_factorial_iff_mersenne_l269_26944

theorem power_two_divides_factorial_iff_mersenne (n : ℕ) (hn : 0 < n) :
  (2^(n - 1) ∣ n!) ↔ ∃ k : ℕ, k > 0 ∧ n = 2^k - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_divides_factorial_iff_mersenne_l269_26944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l269_26920

open Real

-- Define the equation
def equation (x a : ℝ) : Prop :=
  2 * (sin (π - π * x^2 / 12)) * (cos (π / 6 * Real.sqrt (9 - x^2))) + 1 = 
  a + 2 * (sin (π / 6 * Real.sqrt (9 - x^2))) * (cos (π * x^2 / 12))

-- State the theorem
theorem smallest_a_value : 
  (∃ a : ℝ, ∀ a' : ℝ, (∃ x : ℝ, equation x a') → a ≤ a') ∧ 
  (∃ x : ℝ, equation x 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l269_26920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_completed_in_two_hours_l269_26932

/-- Represents the time taken to build a wall for each worker -/
structure WallBuildingTime where
  avery : ℚ
  tom : ℚ
  catherine : ℚ
  derek : ℚ

/-- Represents the state of workers at a given time -/
structure WorkerState where
  avery : Bool
  tom : Bool
  catherine : Bool
  derek : Bool

/-- Calculates the portion of wall built in a given time -/
noncomputable def wallBuilt (times : WallBuildingTime) (state : WorkerState) (time : ℚ) : ℚ :=
  (if state.avery then time / times.avery else 0) +
  (if state.tom then time / times.tom else 0) +
  (if state.catherine then time / times.catherine else 0) +
  (if state.derek then time / times.derek else 0)

theorem wall_completed_in_two_hours (times : WallBuildingTime)
  (h1 : times.avery = 3)
  (h2 : times.tom = 5/2)
  (h3 : times.catherine = 4)
  (h4 : times.derek = 5) :
  let state1 := { avery := true, tom := true, catherine := true, derek := false }
  let state2 := { avery := false, tom := true, catherine := true, derek := false }
  wallBuilt times state1 1 + wallBuilt times state2 1 ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_completed_in_two_hours_l269_26932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l269_26986

-- Define the domain M
def M : Set ℝ := {x | x < 1 ∨ x > 3}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + 2 - 3*(2^x)^2

-- Theorem statement
theorem function_properties :
  (∀ x, x ∈ M ↔ (3 - 4*x + x^2 > 0)) ∧
  (∃ max_val : ℝ, max_val = -8 ∧ ∀ x ∈ M, f x ≤ max_val) ∧
  (¬∃ min_val : ℝ, ∀ x ∈ M, f x ≥ min_val) := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma domain_characterization (x : ℝ) : x ∈ M ↔ 3 - 4*x + x^2 > 0 := by
  sorry

lemma max_value : ∃ max_val : ℝ, max_val = -8 ∧ ∀ x ∈ M, f x ≤ max_val := by
  sorry

lemma no_min_value : ¬∃ min_val : ℝ, ∀ x ∈ M, f x ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l269_26986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l269_26973

open Real MeasureTheory

theorem tan_cot_equation_solutions :
  ∃ (S : Finset ℝ), S.card = 16 ∧
  (∀ θ ∈ S, 0 < θ ∧ θ < 2 * π ∧ tan (3 * π * cos θ) = 1 / tan (4 * π * sin θ)) ∧
  (∀ θ, 0 < θ ∧ θ < 2 * π ∧ tan (3 * π * cos θ) = 1 / tan (4 * π * sin θ) → θ ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l269_26973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l269_26971

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of line l1: ax + 2y - 1 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / 2

/-- The slope of line l2: x + (a+1)y + 4 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -1 / (a + 1)

theorem parallel_lines_condition (a : ℝ) :
  (a = 1 → parallel (slope_l1 a) (slope_l2 a)) ∧
  ¬(parallel (slope_l1 a) (slope_l2 a) → a = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l269_26971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l269_26943

-- Define a triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi

-- State the theorem
theorem triangle_problem (A B C : ℝ) 
  (h_triangle : Triangle A B C)
  (h_equation : Real.cos C + (Real.cos A - Real.sqrt 3 * Real.sin A) * Real.cos B = 0) :
  B = Real.pi / 3 ∧ 
  (Real.sin (A - Real.pi / 3) = 3 / 5 → Real.sin (2 * C) = (24 + 7 * Real.sqrt 3) / 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l269_26943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_integer_l269_26989

/-- Definition of d(n, m) -/
def d : ℕ → ℕ → ℚ
  | n, 0 => 1
  | n, m => if n = m then 1 else
      if m < n then
        (m * d (n-1) m + (2*n - m) * d (n-1) (m-1)) / m
      else 0

/-- Theorem: d(n, m) are integers for all m, n ∈ ℕ -/
theorem d_is_integer (n m : ℕ) : ∃ (k : ℤ), d n m = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_integer_l269_26989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_a_l269_26962

/-- Triangle ABC with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The area of the right region of the triangle when divided by a vertical line x = a -/
noncomputable def rightRegionArea (t : Triangle) (a : ℝ) : ℝ :=
  let base := t.C.1 - a
  let height := t.A.2 - (a / t.C.1) * t.A.2
  triangleArea base height

/-- The area of the left region of the triangle when divided by a vertical line x = a -/
noncomputable def leftRegionArea (t : Triangle) (a : ℝ) : ℝ :=
  triangleArea t.C.1 t.A.2 - rightRegionArea t a

/-- The theorem stating the existence and uniqueness of a -/
theorem exists_unique_a (t : Triangle) : 
  t.A = (0, 2) → t.B = (0, 0) → t.C = (10, 0) → 
  ∃! a : ℝ, 0 < a ∧ a < 10 ∧ leftRegionArea t a = 2 * rightRegionArea t a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_a_l269_26962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l269_26966

theorem problem_solution (y : ℝ) (h1 : y > 0) (h2 : 2 * y * ⌊y⌋ = 162) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l269_26966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_2pi_3_l269_26948

theorem cos_alpha_plus_2pi_3 (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.cos (α - π/2) = -4*Real.sqrt 3/5)
  (h2 : -π/2 < α ∧ α < 0) :
  Real.cos (α + 2*π/3) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_2pi_3_l269_26948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_equals_two_l269_26975

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := ∑' n, (x / 2) ^ n

-- State the theorem
theorem integral_sqrt_equals_two :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  Real.sqrt (∫ y in Set.Ioc (Real.exp 1) 1, f y) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_equals_two_l269_26975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_bag_mass_difference_l269_26939

/-- The maximum difference in mass between any two bags of rice -/
theorem rice_bag_mass_difference (bag_mass : Set ℝ) : 
  (∀ m, m ∈ bag_mass → 19.7 ≤ m ∧ m ≤ 20.3) → 
  ∃ a b, a ∈ bag_mass ∧ b ∈ bag_mass ∧ 
    (∀ x y, x ∈ bag_mass → y ∈ bag_mass → |x - y| ≤ |a - b|) ∧ 
    |a - b| = 0.6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_bag_mass_difference_l269_26939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_good_number_l269_26910

/-- Represents the state of numbers on the board -/
def BoardState := List ℚ

/-- The initial state of the board with 1000 ones -/
def initialState : BoardState := List.replicate 1000 1

/-- The operation of replacing a number with three copies of its third -/
def replaceOperation (a : ℚ) : List ℚ := [a/3, a/3, a/3]

/-- Applies the replace operation to a single element of the board -/
def applyOperation (state : BoardState) (index : ℕ) : BoardState :=
  match List.splitAt index state with
  | (before, a::after) => before ++ replaceOperation a ++ after
  | _ => state  -- Return the original state if index is out of bounds

/-- Checks if a number is "good" according to the problem definition -/
def isGoodNumber (m : ℕ) (state : BoardState) : Prop :=
  ∃ (x : ℚ), state.count x ≥ m

/-- The main theorem stating that 667 is the largest good number -/
theorem largest_good_number :
  (∀ (operations : List (BoardState → BoardState)),
   ∀ (state : BoardState),
   state = List.foldl (λ s f => f s) initialState operations →
   isGoodNumber 667 state) ∧
  (¬ ∀ (operations : List (BoardState → BoardState)),
   ∀ (state : BoardState),
   state = List.foldl (λ s f => f s) initialState operations →
   isGoodNumber 668 state) := by
  sorry

#check largest_good_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_good_number_l269_26910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_C_eq_D_l269_26983

noncomputable def C (n : ℕ) : ℝ := 2048 * (1 - 1 / 2^n)

noncomputable def D (n : ℕ) : ℝ := (6144 / 3) * (1 - 1 / (-2)^n)

theorem no_solution_for_C_eq_D :
  ¬ ∃ (n : ℕ), n ≥ 1 ∧ C n = D n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_C_eq_D_l269_26983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_power_exceeding_100_l269_26934

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def isGeometricProgression (a b c : ℝ) : Prop :=
  b * b = a * c

theorem least_power_exceeding_100 (x : ℝ) (h1 : x > 0)
  (h2 : isGeometricProgression (frac x) ⌊x⌋ x) :
  (∃ n : ℕ, n > 0 ∧ x^n > 100 ∧ ∀ m : ℕ, m > 0 → m < n → x^m ≤ 100) →
  (∃ n : ℕ, n > 0 ∧ x^n > 100 ∧ ∀ m : ℕ, m > 0 → m < n → x^m ≤ 100) ∧
  (∀ n : ℕ, n > 0 ∧ x^n > 100 ∧ (∀ m : ℕ, m > 0 → m < n → x^m ≤ 100) → n = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_power_exceeding_100_l269_26934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_black_cells_for_25_ops_l269_26959

/-- Represents a cell in the table -/
inductive Cell
| Black
| White
deriving DecidableEq

/-- Represents the table -/
def Table := Fin 8 → Fin 12 → Cell

/-- Represents a three-cell corner operation -/
def CornerOperation := Fin 8 → Fin 12 → Table → Table

/-- The number of operations required to make the table completely white -/
noncomputable def OperationsRequired (t : Table) : ℕ := sorry

/-- The theorem stating the smallest number of black cells requiring at least 25 operations -/
theorem smallest_black_cells_for_25_ops :
  ∃ (N : ℕ) (t : Table),
    (∀ (i : Fin 8) (j : Fin 12), t i j = Cell.Black → 
      N = (Finset.filter (λ (p : Fin 8 × Fin 12) => t p.1 p.2 = Cell.Black) 
        (Finset.product (Finset.univ : Finset (Fin 8)) (Finset.univ : Finset (Fin 12)))).card) ∧
    OperationsRequired t ≥ 25 ∧
    (∀ (M : ℕ) (s : Table), M < N →
      (∀ (i : Fin 8) (j : Fin 12), s i j = Cell.Black → 
        M = (Finset.filter (λ (p : Fin 8 × Fin 12) => s p.1 p.2 = Cell.Black) 
          (Finset.product (Finset.univ : Finset (Fin 8)) (Finset.univ : Finset (Fin 12)))).card) →
      OperationsRequired s < 25) ∧
    N = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_black_cells_for_25_ops_l269_26959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooms_needed_l269_26906

/-- Represents a group of fans supporting a team and of a specific gender -/
structure FanGroup where
  count : Nat

/-- Represents the hotel's room allocation problem -/
structure HotelProblem where
  totalFans : Nat
  groups : Fin 6 → FanGroup
  roomCapacity : Nat

/-- Calculates the number of rooms needed for a given fan group -/
def roomsNeededForGroup (g : FanGroup) (capacity : Nat) : Nat :=
  (g.count + capacity - 1) / capacity

/-- Calculates the total number of rooms needed for all groups -/
def totalRoomsNeeded (p : HotelProblem) : Nat :=
  Finset.sum Finset.univ fun i => roomsNeededForGroup (p.groups i) p.roomCapacity

/-- The main theorem stating the maximum number of rooms needed -/
theorem max_rooms_needed (p : HotelProblem) 
    (h1 : p.totalFans = 100)
    (h2 : p.roomCapacity = 3)
    (h3 : Finset.sum Finset.univ (fun i => (p.groups i).count) = p.totalFans) :
    totalRoomsNeeded p ≤ 37 := by
  sorry

#check max_rooms_needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooms_needed_l269_26906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_asymptote_distance_l269_26995

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, 
    the distance from its foci to its asymptotes is 2√3 -/
theorem hyperbola_foci_asymptote_distance :
  ∃ (f : ℝ × ℝ) (a : ℝ → ℝ),
  (∀ t : ℝ, a t = t * Real.sqrt 3 ∨ a t = -t * Real.sqrt 3) ∧
  (∀ t : ℝ, dist f (t, a t) = 2 * Real.sqrt 3) :=
by
  -- We'll prove this later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_asymptote_distance_l269_26995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_ln_x_l269_26924

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) := x * Real.log x

/-- The theorem stating that the minimum value of f(x) = x ln x for x > 0 is -1/e -/
theorem min_value_x_ln_x :
  ∃ (m : ℝ), m = -1 / Real.exp 1 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_ln_x_l269_26924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_logarithmic_function_l269_26992

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((2 / (1 - x)) + a)

-- Define the property of being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- Theorem statement
theorem no_odd_logarithmic_function :
  ¬ ∃ a : ℝ, is_odd_function (f a) := by
  sorry

#check no_odd_logarithmic_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_logarithmic_function_l269_26992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_A_min_cost_team_B_successful_bid_range_l269_26961

-- Define the cost function for team A
noncomputable def cost_A (x : ℝ) : ℝ := 900 * (16 / x + x) + 7200

-- Define the bid function for team B
noncomputable def bid_B (a x : ℝ) : ℝ := (900 * a * (1 + x)) / x

-- Theorem for the minimum cost of team A
theorem team_A_min_cost :
  ∀ x : ℝ, 2 ≤ x → x ≤ 6 → cost_A x ≥ cost_A 4 := by sorry

-- Theorem for the range of a that allows team B to always bid successfully
theorem team_B_successful_bid_range :
  ∀ a : ℝ, (∀ x : ℝ, 2 ≤ x → x ≤ 6 → bid_B a x < cost_A x) ↔ 0 < a ∧ a < 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_A_min_cost_team_B_successful_bid_range_l269_26961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_equation_l269_26951

/-- Represents the time saved by taking Route B compared to Route A -/
noncomputable def time_saved : ℝ := 1/4

/-- The length of Route A in kilometers -/
noncomputable def route_a_length : ℝ := 25

/-- The additional length of Route B compared to Route A in kilometers -/
noncomputable def route_b_additional_length : ℝ := 7

/-- The percentage increase in speed for Route B compared to Route A -/
noncomputable def speed_increase_percentage : ℝ := 60/100

/-- Theorem stating the equation for the time difference between routes -/
theorem route_time_equation (x : ℝ) (hx : x > 0) :
  route_a_length / x - (route_a_length + route_b_additional_length) / (x * (1 + speed_increase_percentage)) = time_saved := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_equation_l269_26951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l269_26998

theorem right_triangle_hypotenuse (P Q R : ℝ × ℝ) (tanP : ℝ) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (R.1 - Q.1)^2 + (R.2 - Q.2)^2 →  -- Right angle at Q
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = 12^2 →  -- QR = 12
  tanP = 3/4 →  -- tan P = 3/4
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 15^2 :=  -- PR = 15
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l269_26998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_symmetry_l269_26999

/-- The circle C: x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circleC (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line L: 2ax - by + 2 = 0 -/
def lineL (a b x y : ℝ) : Prop :=
  2*a*x - b*y + 2 = 0

/-- The circle is symmetric with respect to the line -/
def symmetric (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), circleC x y ∧ lineL a b x y

theorem circle_line_symmetry (a b : ℝ) (h : symmetric a b) :
  a * b ≤ 1/4 ∧ ∀ ε > 0, ∃ (a' b' : ℝ), symmetric a' b' ∧ a' * b' > -1/ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_symmetry_l269_26999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_extremum_l269_26941

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - a * x - 3

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (1 - x) / x

noncomputable def g (m : ℝ) (a : ℝ) (x : ℝ) : ℝ := x^3 + x^2 * (m/2 + f_deriv a x)

noncomputable def g_deriv (m : ℝ) (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + (m + 2*a)*x - a

theorem f_monotonicity_and_g_extremum :
  (∀ x > 0, x < 1 → f_deriv 1 x > 0) ∧
  (∀ x > 1, f_deriv 1 x < 0) ∧
  (f_deriv (-2) 2 = 1 →
    ∀ m, -37/3 < m ∧ m < -9 ↔
      ∀ t ∈ Set.Icc 1 2, ∃ y ∈ Set.Ioo t 3, g_deriv m (-2) y = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_extremum_l269_26941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_for_factorial_product_l269_26964

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_difference_for_factorial_product (p q r : ℕ+) : 
  p.val * q.val * r.val = factorial 9 → p < q → q < r → 
  ∀ (p' q' r' : ℕ+), p'.val * q'.val * r'.val = factorial 9 → p' < q' → q' < r' → r'.val - p'.val ≥ r.val - p.val →
  r.val - p.val = 312 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_for_factorial_product_l269_26964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detach_bogie_crossing_time_l269_26903

/-- Represents a train with a given number of bogies --/
structure Train where
  numBogies : ℕ
  bogieLength : ℝ
  crossingTime : ℝ

/-- Calculates the time for a train to cross a telegraph post after detaching one bogie --/
noncomputable def crossingTimeAfterDetach (t : Train) : ℝ :=
  let initialLength := t.numBogies * t.bogieLength
  let speed := initialLength / t.crossingTime
  let newLength := (t.numBogies - 1) * t.bogieLength
  newLength / speed

/-- Theorem stating that detaching one bogie from a 12-bogie train reduces crossing time from 9 to 8.25 seconds --/
theorem detach_bogie_crossing_time :
  let initialTrain : Train := { numBogies := 12, bogieLength := 15, crossingTime := 9 }
  crossingTimeAfterDetach initialTrain = 8.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_detach_bogie_crossing_time_l269_26903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_for_four_lines_l269_26991

/-- Represents a point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a 2D coordinate plane -/
structure Line where
  -- We don't need to define the specifics of a line for this statement

/-- The number of lines at a given distance from a point -/
def linesAtDistance (p : Point) (d : ℝ) : Finset Line := sorry

theorem distance_range_for_four_lines (d : ℝ) : 
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨5, 5⟩
  (Finset.card (linesAtDistance A 1) + Finset.card (linesAtDistance B d) = 4) →
  4 < d ∧ d < 6 := by
  sorry

#check distance_range_for_four_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_for_four_lines_l269_26991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_angle_bound_l269_26969

-- Define a triangle with sides a, b, c and angles α, β, γ
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the cotangent function
noncomputable def ctg (θ : ℝ) : ℝ := 1 / Real.tan θ

-- Define lambda (using 'lambda' instead of 'λ' to avoid syntax issues)
noncomputable def lambda (t : Triangle) : ℝ := (ctg t.α + ctg t.β) / ctg t.γ

-- Theorem 1
theorem triangle_equality (t : Triangle) (h : ctg t.γ ≠ 0) :
  t.a^2 + t.b^2 = (1 + 2 / lambda t) * t.c^2 := by sorry

-- Theorem 2
theorem angle_bound (t : Triangle) (h : ctg t.γ ≠ 0) (h2 : lambda t = 2) :
  t.γ ≤ Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_angle_bound_l269_26969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_one_l269_26960

/-- Line l parameterized by t -/
noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + t/2, Real.sqrt 3 * t/2)

/-- Curve C -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Intersection points of line l and curve C -/
def intersection (t : ℝ) : Prop :=
  let (x, y) := line t
  curve x y

/-- The distance between intersection points is 1 -/
theorem intersection_distance_is_one :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ intersection t₁ ∧ intersection t₂ ∧ |t₁ - t₂| = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_one_l269_26960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_k_l269_26900

/-- The equation of an ellipse with semi-major axis length a and semi-minor axis length b -/
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The distance from the center to a focus of an ellipse -/
noncomputable def focus_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_focus_k (k : ℝ) :
  (∃ x y : ℝ, is_ellipse (Real.sqrt k) (Real.sqrt 5) x y) →
  focus_distance (Real.sqrt k) (Real.sqrt 5) = 2 →
  k = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_k_l269_26900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_range_of_a_l269_26993

-- Define the function f using the built-in abs function
def f (x : ℝ) : ℝ := |x + 1|

-- Statement for part 1
theorem solution_set_inequality (x : ℝ) :
  x * f x > f (x - 2) ↔ x > Real.sqrt 2 - 1 :=
sorry

-- Statement for part 2
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y = Real.log (f (x - 3) + f x + a)) ↔ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_range_of_a_l269_26993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_6_is_simplest_radical_form_l269_26980

-- Define what it means for a square root to be in simplest radical form
def is_simplest_radical_form (x : ℝ) : Prop :=
  x > 0 ∧ 
  ∀ y z : ℕ, (y * y * z = Int.natAbs (Int.floor x)) → y = 1

-- State the theorem
theorem sqrt_6_is_simplest_radical_form : 
  is_simplest_radical_form 6 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_6_is_simplest_radical_form_l269_26980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l269_26925

noncomputable def f (ω : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (ω * x - Real.pi / 6) + b

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2 * x - Real.pi / 3) - 1 / 2

theorem range_of_m (ω : ℝ) (b : ℝ) :
  (ω > 0) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f ω b x ≤ 1) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f ω b x ≥ -2) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 3), g x - 3 ≤ g x + 3) →
  (∀ m : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 3), g x - 3 ≤ m ∧ m ≤ g x + 3) → m ∈ Set.Icc (-2) 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l269_26925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_l269_26956

/-- Given a solution with alcohol and water, adding pure water reduces the alcohol percentage -/
theorem alcohol_dilution (original_volume : ℝ) (original_percentage : ℝ) (added_water : ℝ) 
  (h1 : original_volume = 11) 
  (h2 : original_percentage = 42) 
  (h3 : added_water = 3) : 
  (original_volume * (original_percentage / 100) / (original_volume + added_water)) * 100 = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_l269_26956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_l269_26981

/-- The number of students in the class -/
def n : ℕ := sorry

/-- The number of students older than Petya -/
def petya_older : ℕ := sorry

/-- The number of students younger than Petya -/
def petya_younger : ℕ := sorry

/-- The number of students older than Katya -/
def katya_older : ℕ := sorry

/-- The number of students younger than Katya -/
def katya_younger : ℕ := sorry

theorem class_size :
  (20 < n ∧ n < 30) ∧
  (petya_older = 2 * petya_younger) ∧
  (petya_older + petya_younger + 1 = n) ∧
  (katya_older * 3 = katya_younger) ∧
  (katya_older + katya_younger + 1 = n) →
  n = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_l269_26981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l269_26974

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) :=
  ∃ (a b c : ℝ),
    -- a is the length of BC
    a = Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) ∧
    -- b is the length of AC
    b = Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) ∧
    -- c is the length of AB
    c = Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) ∧
    -- a = 2
    a = 2 ∧
    -- D is the midpoint of AB
    let D := ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2);
    -- CD = √2
    Real.sqrt ((t.C.1 - D.1)^2 + (t.C.2 - D.2)^2) = Real.sqrt 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : TriangleProperties t) :
  ∃ (b : ℝ),
    -- Part 1: c = √2b
    Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = Real.sqrt 2 * b ∧
    -- Part 2: If ∠ACB = π/4, the area of triangle ABC is √3 - 1
    (Real.arccos ((t.A.1 - t.C.1) * (t.B.1 - t.C.1) + (t.A.2 - t.C.2) * (t.B.2 - t.C.2)) /
      (Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) *
       Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)) = π / 4 →
     1/2 * 2 * b * Real.sin (π / 4) = Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l269_26974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statistical_statements_correctness_l269_26946

/-- Represents the fitting effect of a regression model -/
def regression_fit_effect : ℝ → ℝ := sorry

/-- Represents a normal distribution with mean μ and standard deviation σ -/
def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

/-- Represents the probability of a random variable being greater than a given value -/
def probability_greater_than (dist : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- Represents the area of a rectangle in a frequency distribution histogram -/
def histogram_rectangle_area : ℝ := sorry

/-- Represents the frequency of a group in a frequency distribution -/
def group_frequency : ℝ := sorry

/-- Represents the observed value of the K² statistic for categorical variables X and Y -/
def k_squared_statistic : ℝ := sorry

/-- Represents the strength of relationship between categorical variables X and Y -/
def relationship_strength : ℝ → ℝ := sorry

theorem statistical_statements_correctness :
  (∀ x y : ℝ, x > y → regression_fit_effect x > regression_fit_effect y) ∧
  (probability_greater_than (normal_distribution 4 2) 4 = 1/2) ∧
  (histogram_rectangle_area ≠ group_frequency) ∧
  (∀ x y : ℝ, x > y → relationship_strength x > relationship_strength y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statistical_statements_correctness_l269_26946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_yearly_return_percentage_l269_26988

/-- Calculates the combined yearly return percentage of two investments -/
theorem combined_yearly_return_percentage 
  (investment1 : ℝ) 
  (investment2 : ℝ) 
  (return1 : ℝ) 
  (return2 : ℝ) 
  (h1 : investment1 = 500) 
  (h2 : investment2 = 1500) 
  (h3 : return1 = 0.07) 
  (h4 : return2 = 0.09) : 
  (investment1 * return1 + investment2 * return2) / (investment1 + investment2) = 0.085 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_yearly_return_percentage_l269_26988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l269_26919

noncomputable def f (x a : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) + a + 1

theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, f (x + Real.pi) a = f x a) ∧
  (∀ k : ℤ, StrictMonoOn (fun x => -f x a) (Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3))) ∧
  (IsMinOn (f · a) (Set.Icc 0 (Real.pi / 2)) 2 → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l269_26919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_three_l269_26935

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : a * (-2)^2 + b * (-2) + c = 3
  point2 : a * 8^2 + b * 8 + c = 3
  point3 : a * 10^2 + b * 10 + c = 9

/-- The x-coordinate of the vertex of the quadratic function -/
noncomputable def vertexX (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- Theorem stating that the x-coordinate of the vertex is 3 -/
theorem vertex_x_is_three (f : QuadraticFunction) : vertexX f = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_three_l269_26935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_properties_l269_26949

/-- A 3D line representation -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters

/-- A 3D plane representation -/
structure Plane3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters

/-- A 2D line representation -/
structure Line2D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters

/-- Two lines in 3D space are skew if they are neither parallel nor intersecting -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A line is parallel to a plane if it does not intersect the plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- A line is perpendicular to a plane if it forms a right angle with the plane -/
def line_perpendicular_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- The projection of a line onto a plane -/
noncomputable def line_projection (l : Line3D) (p : Plane3D) : Line2D := sorry

/-- Two 2D lines are parallel -/
def are_parallel_2D (l1 l2 : Line2D) : Prop := sorry

theorem skew_lines_properties :
  ∃ (l1 l2 : Line3D) (p : Plane3D),
    are_skew l1 l2 ∧
    (line_parallel_to_plane l1 p ∧ line_parallel_to_plane l2 p) ∧
    (line_perpendicular_to_plane l1 p → ¬ line_perpendicular_to_plane l2 p) ∧
    ∃ (proj_plane : Plane3D), are_parallel_2D (line_projection l1 proj_plane) (line_projection l2 proj_plane) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_properties_l269_26949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_time_l269_26940

noncomputable def jay_speed : ℝ := 0.75 / 15 -- miles per minute
noncomputable def kim_speed : ℝ := 3 / 45 -- miles per minute
noncomputable def time : ℝ := 2.25 * 60 -- minutes

noncomputable def jay_distance : ℝ := jay_speed * time
noncomputable def kim_distance : ℝ := kim_speed * time

theorem distance_after_time (jay_speed kim_speed time : ℝ) 
  (h1 : jay_speed = 0.75 / 15)
  (h2 : kim_speed = 3 / 45)
  (h3 : time = 2.25 * 60) :
  Real.sqrt ((jay_speed * time)^2 + (kim_speed * time)^2) = Real.sqrt 126.5625 := by
  sorry

#check distance_after_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_time_l269_26940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_playhouse_siding_cost_l269_26965

/-- Calculates the cost of siding needed for a playhouse -/
theorem playhouse_siding_cost
  (side_wall_length : ℕ)
  (side_wall_height : ℕ)
  (roof_face_length : ℕ)
  (roof_face_width : ℕ)
  (siding_section_length : ℕ)
  (siding_section_width : ℕ)
  (siding_section_cost : ℕ) :
  side_wall_length = 8 →
  side_wall_height = 6 →
  roof_face_length = 8 →
  roof_face_width = 5 →
  siding_section_length = 10 →
  siding_section_width = 15 →
  siding_section_cost = 35 →
  70 = (⌈(2 * side_wall_length * side_wall_height + 2 * roof_face_length * roof_face_width : ℝ) /
       (siding_section_length * siding_section_width : ℝ)⌉ * siding_section_cost) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_playhouse_siding_cost_l269_26965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_l269_26908

def is_not_divisible_by_10 (k : ℕ) : Prop := ¬(k % 10 = 0)

theorem smallest_t (p q r k t : ℕ) : 
  p > 0 → q > 0 → r > 0 →
  p + q + r = 2510 →
  Nat.factorial p * Nat.factorial q * Nat.factorial r = k * (10 : ℕ)^t →
  is_not_divisible_by_10 k →
  (∀ t' : ℕ, t' ≥ 0 → 
    (∃ k' : ℕ, Nat.factorial p * Nat.factorial q * Nat.factorial r = k' * (10 : ℕ)^t' ∧ 
    is_not_divisible_by_10 k') → t' ≥ t) →
  t = 626 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_l269_26908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_addition_example_l269_26952

/-- Convert a natural number to its octal representation -/
def to_octal (n : ℕ) : ℕ := sorry

/-- Convert an octal representation to a natural number -/
def from_octal (n : ℕ) : ℕ := sorry

/-- Addition of octal numbers -/
def octal_add (a b : ℕ) : ℕ :=
  to_octal (from_octal a + from_octal b)

/-- Theorem: 234₈ + 157₈ = 413₈ in octal arithmetic -/
theorem octal_addition_example : octal_add 234 157 = 413 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_addition_example_l269_26952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_B_Binv_l269_26907

open Matrix

variable {n : ℕ} [FiniteDimensional ℝ (Fin n → ℝ)]
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem det_A_B_Binv (hA : det A = -5) (hB : det B = 8) :
  det (A * B * B⁻¹) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_B_Binv_l269_26907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l269_26927

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (x - 1)

def A : ℝ × ℝ := (1, 0)

def O : ℝ × ℝ := (0, 0)

-- Define P and Q as variables
variable (P Q : ℝ × ℝ)

-- Assume P and Q are on the graph of f
axiom P_on_graph : (P.1, f P.1) = P
axiom Q_on_graph : (Q.1, f Q.1) = Q

-- Assume P, Q, and A are collinear
axiom PQA_collinear : ∃ (t : ℝ), P.1 - A.1 = t * (Q.1 - A.1) ∧ P.2 - A.2 = t * (Q.2 - A.2)

-- The main theorem
theorem dot_product_theorem : (P.1 + Q.1 - 2 * O.1, P.2 + Q.2 - 2 * O.2) • (A.1 - O.1, A.2 - O.2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l269_26927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_induced_formula_problem_l269_26936

theorem induced_formula_problem (α : ℝ) 
  (h1 : Real.cos (α + π) = 3/5)
  (h2 : π ≤ α)
  (h3 : α < 2*π) :
  Real.sin (-α - 2*π) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_induced_formula_problem_l269_26936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_function_l269_26958

theorem existence_of_special_function : ∃ f : ℕ → ℕ,
  (∀ i, i ∈ Finset.range 100 → f (i + 1) ≤ 25000) ∧
  (∀ i j, i ∈ Finset.range 100 → j ∈ Finset.range 100 → i ≠ j → f (i + 1) ≠ f (j + 1)) ∧
  (∀ a b c d, a ∈ Finset.range 100 → b ∈ Finset.range 100 → c ∈ Finset.range 100 → d ∈ Finset.range 100 →
    f (a + 1) + f (b + 1) = f (c + 1) + f (d + 1) →
    ({a, b} : Finset ℕ) = {c, d}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_function_l269_26958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_length_l269_26916

/-- A trapezoid with specific properties -/
structure Trapezoid where
  -- Points of the trapezoid
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  -- EF and GH are parallel
  parallel_EF_GH : (F.1 - E.1) * (H.2 - G.2) = (F.2 - E.2) * (H.1 - G.1)
  -- Lengths of sides
  EF_length : Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 40
  GH_length : Real.sqrt ((H.1 - G.1)^2 + (H.2 - G.2)^2) = 28
  side1_length : Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) = 17
  side2_length : Real.sqrt ((H.1 - E.1)^2 + (H.2 - E.2)^2) = 15
  -- Angle E is a right angle
  right_angle_E : (F.1 - E.1) * (H.1 - E.1) + (F.2 - E.2) * (H.2 - E.2) = 0

/-- The shorter diagonal of the trapezoid has length 15 + 2√2 -/
theorem shorter_diagonal_length (t : Trapezoid) :
  min (Real.sqrt ((t.G.1 - t.E.1)^2 + (t.G.2 - t.E.2)^2))
      (Real.sqrt ((t.F.1 - t.H.1)^2 + (t.F.2 - t.H.2)^2)) = 15 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_length_l269_26916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l269_26996

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the length of a side
noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the angle between two sides
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  Real.arccos ((side_length p1 p2)^2 + (side_length p1 p3)^2 - (side_length p2 p3)^2) /
    (2 * side_length p1 p2 * side_length p1 p3)

theorem triangle_side_length (t : Triangle) :
  angle t.B t.A t.C = 3 * angle t.C t.A t.B →
  side_length t.A t.B = 6 →
  side_length t.A t.C = 18 →
  side_length t.B t.C = 24 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l269_26996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_and_equation_roots_l269_26987

open Real

theorem function_minimum_and_equation_roots (m : ℝ) : 
  (∃ (a₁ a₂ a₃ : ℝ), a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ 
   a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃ ∧
   (∀ i ∈ ({a₁, a₂, a₃} : Set ℝ), 1 - log i + i - 2 / (9 * i) = m)) →
  1/3 - log 2 + log 3 < m ∧ m < -1/3 + log 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_and_equation_roots_l269_26987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l269_26929

/-- The original function f(x) = 2sin(2x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

/-- The translated function g(x) = f(x + π/12) -/
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 12)

/-- Predicate to check if a given x is on the axis of symmetry -/
def isAxisOfSymmetry (x : ℝ) : Prop :=
  ∀ y, g (x - y) = g (x + y)

/-- Theorem stating the axis of symmetry for the translated function -/
theorem axis_of_symmetry :
  ∀ x : ℝ, isAxisOfSymmetry x ↔ ∃ k : ℤ, x = k * Real.pi / 2 + Real.pi / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l269_26929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt3_over_2_l269_26915

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt3_over_2_l269_26915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l269_26953

/-- Represents Jack's walking journey with varying speeds -/
structure WalkingJourney where
  totalDistance : ℝ
  totalTime : ℝ
  speed1 : ℝ
  time1 : ℝ
  speed2 : ℝ
  time2 : ℝ
  speed3 : ℝ

/-- Calculates the average speed of the entire walk -/
noncomputable def averageSpeed (journey : WalkingJourney) : ℝ :=
  journey.totalDistance / journey.totalTime

/-- Theorem stating that the average speed calculation is correct -/
theorem average_speed_calculation (journey : WalkingJourney) 
  (h1 : journey.totalDistance = 8)
  (h2 : journey.totalTime = 1.25)
  (h3 : journey.speed1 = 4)
  (h4 : journey.time1 = 0.5)
  (h5 : journey.speed2 = 6)
  (h6 : journey.time2 = 1/3)
  (h7 : journey.speed3 = 3)
  (h8 : journey.time1 + journey.time2 + (journey.totalDistance - journey.speed1 * journey.time1 - journey.speed2 * journey.time2) / journey.speed3 = journey.totalTime) :
  ∃ ε > 0, |averageSpeed journey - 3.692| < ε := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l269_26953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l269_26921

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

-- State the theorem
theorem f_properties :
  -- Smallest positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ T, T > 0 ∧ (∀ x, f (x + T) = f x) → T ≥ π) ∧
  -- Maximum value
  (∀ x, f x ≤ 1 + Real.sqrt 3 / 2) ∧
  (∃ x, f x = 1 + Real.sqrt 3 / 2) ∧
  -- Minimum value
  (∀ x, f x ≥ -1 + Real.sqrt 3 / 2) ∧
  (∃ x, f x = -1 + Real.sqrt 3 / 2) ∧
  -- Intervals of monotonic increase
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-π/12 + k*π) (5*π/12 + k*π))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l269_26921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_for_doubling_l269_26977

/-- Given an original price and a price increase percentage, 
    calculate the required price decrease percentage to return to the original price. -/
noncomputable def price_decrease_percentage (original_price : ℝ) (increase_percentage : ℝ) : ℝ :=
  1 - (original_price / (original_price * (1 + increase_percentage)))

/-- Theorem stating that for an original price of 50 and a 100% increase, 
    a 50% decrease is required to return to the original price. -/
theorem price_decrease_for_doubling :
  price_decrease_percentage 50 1 = 0.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_for_doubling_l269_26977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l269_26926

def list_median (l : List ℕ) : ℕ := sorry
def list_mean (l : List ℕ) : ℚ := sorry

theorem max_element_of_list (l : List ℕ) : 
  l.length = 7 ∧ 
  list_median l = 4 ∧ 
  list_mean l = 13 →
  l.maximum ≤ some 82 :=
by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l269_26926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l269_26954

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt (2 * x - 1)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l269_26954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_product_l269_26917

theorem power_sum_product (ε : ℝ) (hε : ε > 0) : ∃ (x y : ℝ) (n : ℕ), 
  x > 0 ∧ y > 0 ∧ n > 0 ∧
  x^n + y^n = 91 ∧ 
  |x * y - 12| < ε ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_product_l269_26917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_root_interval_l269_26985

open Real

theorem equation_root_interval (k : ℤ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (k : ℝ) ((k : ℝ) + 1) ∧ log x = 8 - 2*x) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_root_interval_l269_26985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l269_26922

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (|x^3 - x^2 - 2*x|)/3 - |x + 1|

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < -1 then
    -1/3 * (x^3 - x^2 - 2*x) + (x + 1)
  else if x ≤ 0 then
    1/3 * (x^3 - x^2 - 2*x) - (x + 1)
  else if x < 2 then
    -1/3 * (x^3 - x^2 - 2*x) - (x + 1)
  else
    1/3 * (x^3 - x^2 - 2*x) - (x + 1)

-- Theorem stating that f and g are equal for all real x
theorem f_eq_g : ∀ x : ℝ, f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l269_26922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OA_OB_l269_26914

noncomputable def theta : ℝ := 2 * Real.arcsin (-4/5)

theorem min_distance_OA_OB (B : ℝ × ℝ) :
  Real.sin (theta/2) = -4/5 →
  Real.cos (theta/2) = 3/5 →
  (∃ t : ℝ, t > 0 ∧ B = (t * Real.cos theta, t * Real.sin theta)) →
  (∀ C : ℝ × ℝ, (∃ s : ℝ, s > 0 ∧ C = (s * Real.cos theta, s * Real.sin theta)) →
    ‖(-1, 0) - B‖ ≤ ‖(-1, 0) - C‖) →
  ‖(-1, 0) - B‖ = 24/25 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OA_OB_l269_26914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l269_26918

-- Part 1
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def F (a b c : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b c x else -f a b c x

theorem part1 (a b : ℝ) (h1 : a > 0) (h2 : f a b 1 (-1) = 0) 
  (h3 : ∀ x, f a b 1 x ≥ f a b 1 (-1)) :
  F a b 1 2 + F a b 1 (-2) = 8 := by sorry

-- Part 2
def g (b x : ℝ) : ℝ := x^2 + b * x

theorem part2 (b : ℝ) (h : ∀ x ∈ Set.Ioo 0 1, |g b x| ≤ 1) :
  -2 ≤ b ∧ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l269_26918
