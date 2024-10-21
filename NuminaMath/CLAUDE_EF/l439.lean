import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l439_43929

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 8^x

-- State the theorem
theorem inverse_function_problem (a : ℝ) (h : f (1/3) = a) : 
  f⁻¹ (a + 2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l439_43929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_divides_factorial_iff_power_two_l439_43909

theorem power_two_divides_factorial_iff_power_two (n k : ℕ) :
  2^(n - 1) ∣ n.factorial ↔ n = 2^(k - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_divides_factorial_iff_power_two_l439_43909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l439_43971

open Real Set

noncomputable def f (x : ℝ) : ℝ := 2 * x / exp x

theorem f_extrema_on_interval :
  let a := (1/2 : ℝ)
  let b := (2 : ℝ)
  ∃ (x_max x_min : ℝ),
    x_max ∈ Icc a b ∧
    x_min ∈ Icc a b ∧
    (∀ x ∈ Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Icc a b, f x_min ≤ f x) ∧
    x_max = 1 ∧
    x_min = 2 ∧
    f x_max = 2 / exp 1 ∧
    f x_min = 4 / exp 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l439_43971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_l439_43994

theorem find_y (x y : ℕ) (h1 : (x : ℝ) = 96 * y + 11.52) (h2 : (x : ℝ) / y = 96.12) : y = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_l439_43994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l439_43991

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≤ -1}

theorem intersection_complement_theorem :
  A ∩ (Bᶜ) = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l439_43991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_cubic_product_of_roots_specific_cubic_l439_43953

theorem product_of_roots_cubic (a b c d : ℝ) (h : a ≠ 0) :
  let p := -d / a
  let roots := {x : ℝ | a * x^3 + b * x^2 + c * x + d = 0}
  (∀ x, x ∈ roots → ∃ y z, y ∈ roots ∧ z ∈ roots ∧ x * y * z = p) →
  (∃ x y z : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ∧
               a * y^3 + b * y^2 + c * y + d = 0 ∧
               a * z^3 + b * z^2 + c * z + d = 0 ∧
               x * y * z = p) :=
by sorry

theorem product_of_roots_specific_cubic :
  let roots := {x : ℝ | x^3 - 15 * x^2 + 75 * x - 50 = 0}
  (∀ x, x ∈ roots → ∃ y z, y ∈ roots ∧ z ∈ roots ∧ x * y * z = 50) ∧
  (∃ x y z : ℝ, x^3 - 15 * x^2 + 75 * x - 50 = 0 ∧
               y^3 - 15 * y^2 + 75 * y - 50 = 0 ∧
               z^3 - 15 * z^2 + 75 * z - 50 = 0 ∧
               x * y * z = 50) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_cubic_product_of_roots_specific_cubic_l439_43953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_given_to_robert_l439_43989

/-- The number of cats Susan gave to Robert -/
def cats_given : ℕ := sorry

/-- Susan's initial number of cats -/
def susan_initial : ℕ := 21

/-- Bob's initial number of cats -/
def bob_initial : ℕ := 3

/-- The difference in cats between Susan and Bob after the exchange -/
def final_difference : ℕ := 14

theorem cats_given_to_robert :
  susan_initial - cats_given = (bob_initial + cats_given) + final_difference →
  cats_given = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_given_to_robert_l439_43989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terms_before_negative_eight_l439_43978

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1 : ℤ)

theorem terms_before_negative_eight :
  let a₁ := 100
  let d := -4
  let target := -8
  let n := ((a₁ - target) / (-d) : ℤ).toNat + 1
  (∀ k : ℕ, k < n → arithmetic_sequence a₁ d k > target) ∧
  arithmetic_sequence a₁ d n = target ∧
  n - 1 = 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terms_before_negative_eight_l439_43978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_percentage_bounds_l439_43949

/-- Represents the share packages in the bidding lot -/
structure SharePackage where
  razneft : ℕ
  dvaneft : ℕ
  trineft : ℕ

/-- Represents the prices of shares for each company -/
structure SharePrices where
  razneft : ℚ
  dvaneft : ℚ
  trineft : ℚ

/-- The conditions of the bidding lot -/
def BiddingLotConditions (package : SharePackage) (prices : SharePrices) : Prop :=
  package.razneft + package.dvaneft = package.trineft ∧
  prices.dvaneft * package.dvaneft = (1/4) * prices.razneft * package.razneft ∧
  prices.dvaneft * package.dvaneft + prices.razneft * package.razneft = prices.trineft * package.trineft ∧
  16000 ≤ prices.razneft - prices.dvaneft ∧ prices.razneft - prices.dvaneft ≤ 20000 ∧
  42000 ≤ prices.trineft ∧ prices.trineft ≤ 60000

/-- The percentage of Dvaneft shares in the total lot -/
noncomputable def DvaneftPercentage (package : SharePackage) : ℚ :=
  (package.dvaneft : ℚ) / ((package.razneft + package.dvaneft + package.trineft : ℕ) : ℚ) * 100

/-- Theorem stating the bounds of the Dvaneft share percentage -/
theorem dvaneft_percentage_bounds (package : SharePackage) (prices : SharePrices) :
  BiddingLotConditions package prices →
  (25/2 : ℚ) ≤ DvaneftPercentage package ∧ DvaneftPercentage package ≤ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_percentage_bounds_l439_43949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l439_43954

/-- The function f(x) = x^3 - (1/2)^(x-2) --/
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

/-- Theorem: f(x) has a zero point in the interval (1, 2) --/
theorem f_has_zero_in_interval : ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l439_43954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l439_43997

/-- An ellipse with the given properties has a major axis of length 4 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)),
  (∃ (x y : ℝ), (x, 0) ∈ E ∧ (0, y) ∈ E) →  -- tangent to x-axis and y-axis
  ((1 + Real.sqrt 3, 2) ∈ E ∧ (1 - Real.sqrt 3, 2) ∈ E) →  -- foci locations
  (∃ (a b : ℝ), ∀ (x y : ℝ), (x, y) ∈ E ↔ 
    (x - 1)^2 / a^2 + (y - 2)^2 / b^2 = 1) →  -- standard form of ellipse equation
  (∃ (p q : ℝ), p ≠ q ∧ (p, 2) ∈ E ∧ (q, 2) ∈ E ∧ |p - q| = 4) :=  -- major axis length
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l439_43997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inequality_l439_43943

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the structure for a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the structure for a square
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

-- Function to construct a square on a side of a triangle
noncomputable def constructSquare (p1 p2 : Point) : Square := sorry

-- Function to calculate the area of a triangle
noncomputable def areaTriangle (t : Triangle) : ℝ := sorry

-- Function to construct triangle XYZ
noncomputable def constructXYZ (squareAB squareBC squareCA : Square) : Triangle := sorry

-- Main theorem
theorem area_inequality (ABC : Triangle) :
  let squareAB := constructSquare ABC.A ABC.B
  let squareBC := constructSquare ABC.B ABC.C
  let squareCA := constructSquare ABC.C ABC.A
  let XYZ := constructXYZ squareAB squareBC squareCA
  areaTriangle XYZ ≤ (4 - 2 * Real.sqrt 3) * areaTriangle ABC := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inequality_l439_43943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_bound_l439_43906

theorem max_difference_bound (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_bound_l439_43906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l439_43983

/-- Given two externally tangent circles with radii 2 and 6,
    prove that a third circle tangent to both of these circles
    and their common external tangent has a radius of √3/4 -/
theorem third_circle_radius (P Q : ℝ × ℝ) : 
  let dist_PQ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  dist_PQ = 8 →
  ∃ (S : ℝ × ℝ),
    (Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) = 2 + Real.sqrt 3 / 4) ∧
    (Real.sqrt ((S.1 - Q.1)^2 + (S.2 - Q.2)^2) = 6 + Real.sqrt 3 / 4) ∧
    (∃ (T : ℝ × ℝ), 
      Real.sqrt ((T.1 - P.1)^2 + (T.2 - P.2)^2) = 2 ∧
      Real.sqrt ((T.1 - Q.1)^2 + (T.2 - Q.2)^2) = 6 ∧
      Real.sqrt ((T.1 - S.1)^2 + (T.2 - S.2)^2) = Real.sqrt 3 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l439_43983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_equation_solution_l439_43941

theorem rational_equation_solution (x : ℝ) : 
  x ≠ 2/3 →
  ((6 * x + 2) / (3 * x^2 + 6 * x - 4) = 3 * x / (3 * x - 2)) ↔ 
  (x = Real.sqrt 6 / 3 ∨ x = -Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_equation_solution_l439_43941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l439_43908

/-- Given vectors a, b, and c in ℝ², prove that lambda = -2 when c is parallel to (2a + b) -/
theorem parallel_vector_lambda (a b c : ℝ × ℝ) (lambda : ℝ) : 
  a = (1, 2) → 
  b = (2, lambda) → 
  c = (2, 1) → 
  (∃ (k : ℝ), k ≠ 0 ∧ c.1 = k * (2 * a.1 + b.1) ∧ c.2 = k * (2 * a.2 + b.2)) →
  lambda = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l439_43908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_closed_on_interval_g_closed_on_interval_h_closed_on_interval_l439_43907

-- Definition of a closed function on an interval
def is_closed_on (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x, x ∈ D → f x ∈ D

-- Statement 1
theorem f_not_closed_on_interval :
  ¬ is_closed_on (λ x ↦ x - 1) (Set.Icc (-2) 1) := by
  sorry

-- Statement 2
theorem g_closed_on_interval (a : ℝ) :
  is_closed_on (λ x ↦ (3*x + a) / (x + 1)) (Set.Icc 3 10) ↔ 3 ≤ a ∧ a ≤ 31 := by
  sorry

-- Statement 3
theorem h_closed_on_interval (a : ℝ) :
  is_closed_on (λ x ↦ x^2 + 2*x) (Set.Icc a (a + 1)) ↔ (-1 - Real.sqrt 5) / 2 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_closed_on_interval_g_closed_on_interval_h_closed_on_interval_l439_43907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_3x3_l439_43951

/-- A coloring of a 3x3 square. -/
def Coloring := Fin 3 → Fin 3 → ℕ

/-- Check if all elements in a list are distinct. -/
def allDistinct (l : List ℕ) : Prop := l.Nodup

/-- Check if a coloring satisfies the conditions for rows. -/
def validRows (c : Coloring) : Prop :=
  ∀ i : Fin 3, allDistinct [c i 0, c i 1, c i 2]

/-- Check if a coloring satisfies the conditions for columns. -/
def validColumns (c : Coloring) : Prop :=
  ∀ j : Fin 3, allDistinct [c 0 j, c 1 j, c 2 j]

/-- Check if a coloring satisfies the conditions for diagonals. -/
def validDiagonals (c : Coloring) : Prop :=
  allDistinct [c 0 0, c 1 1, c 2 2] ∧
  allDistinct [c 0 2, c 1 1, c 2 0]

/-- A valid coloring satisfies all conditions. -/
def validColoring (c : Coloring) : Prop :=
  validRows c ∧ validColumns c ∧ validDiagonals c

/-- The number of colors used in a coloring. -/
def numColors (c : Coloring) : ℕ :=
  (Finset.univ.image (λ (p : Fin 3 × Fin 3) => c p.1 p.2)).card

/-- The main theorem: The minimum number of colors for a valid 3x3 coloring is 5. -/
theorem min_colors_3x3 :
  (∃ c : Coloring, validColoring c ∧ numColors c = 5) ∧
  (∀ c : Coloring, validColoring c → numColors c ≥ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_3x3_l439_43951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_prayer_ratio_l439_43927

/-- Represents the number of times a pastor prays --/
structure PrayerCount where
  count : ℕ

/-- Represents the number of days in a week --/
def DaysInWeek : ℕ := 7

/-- Represents the number of non-Sunday days in a week --/
def NonSundayDays : ℕ := 6

/-- Pastor Paul's daily prayer count on regular days --/
def PaulRegularDay : PrayerCount := ⟨20⟩

/-- Pastor Paul's weekly prayer count --/
def PaulWeeklyCount (paulSunday : PrayerCount) : ℕ :=
  NonSundayDays * PaulRegularDay.count + paulSunday.count

/-- Pastor Bruce's weekly prayer count --/
def BruceWeeklyCount (paulSunday : PrayerCount) : ℕ :=
  NonSundayDays * (PaulRegularDay.count / 2) + 2 * paulSunday.count

/-- The difference between Paul's and Bruce's weekly prayer counts --/
def WeeklyDifference : ℕ := 20

/-- Theorem stating that the ratio of Pastor Paul's Sunday prayers to his regular day prayers is 2:1 --/
theorem paul_prayer_ratio :
  ∃ (paulSunday : PrayerCount),
    PaulWeeklyCount paulSunday = BruceWeeklyCount paulSunday + WeeklyDifference ∧
    paulSunday.count = 2 * PaulRegularDay.count :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_prayer_ratio_l439_43927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l439_43946

/-- Given an article sold for $1110 with a 20% profit, prove that its cost price was $925 -/
theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) 
  (h1 : selling_price = 1110)
  (h2 : profit_percentage = 20/100) :
  selling_price / (1 + profit_percentage) = 925 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l439_43946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_is_64_l439_43979

/-- Represents the geometry of a flagpole supported by a wire -/
structure FlagpoleSetup where
  -- Distance from flagpole base to wire ground point
  baseToWireGround : ℝ
  -- Distance Ana walks from flagpole base
  anaWalkDistance : ℝ
  -- Ana's height
  anaHeight : ℝ

/-- Calculates the height of the flagpole given the setup -/
noncomputable def flagpoleHeight (setup : FlagpoleSetup) : ℝ :=
  (setup.baseToWireGround * setup.anaHeight) / (setup.baseToWireGround - setup.anaWalkDistance)

/-- Theorem stating that given the problem conditions, the flagpole height is 6.4 meters -/
theorem flagpole_height_is_64 (setup : FlagpoleSetup)
    (h1 : setup.baseToWireGround = 4)
    (h2 : setup.anaWalkDistance = 3)
    (h3 : setup.anaHeight = 1.6) :
  flagpoleHeight setup = 6.4 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_is_64_l439_43979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_efficient_sorting_strategy_l439_43981

/-- Represents a coin with a unique weight -/
structure Coin where
  weight : ℝ

/-- Represents the result of weighing two coins -/
inductive WeighingResult
  | LighterThan
  | HeavierThan
  | EqualTo

/-- A strategy for sorting coins -/
def SortingStrategy := List Coin → List (Coin × Coin)

/-- The expected number of weighings for a given strategy -/
noncomputable def expectedWeighings (strategy : SortingStrategy) : ℝ := sorry

/-- Theorem stating that there exists a strategy with expected weighings less than 4.8 -/
theorem exists_efficient_sorting_strategy :
  ∃ (strategy : SortingStrategy),
    (∀ coins : List Coin, coins.length = 4 → 
      (∀ i j, i < coins.length → j < coins.length → i ≠ j → 
        (coins.get ⟨i, by sorry⟩).weight ≠ (coins.get ⟨j, by sorry⟩).weight)) →
    expectedWeighings strategy < 4.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_efficient_sorting_strategy_l439_43981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_college_students_count_prove_college_students_count_l439_43945

theorem college_students_count 
  (boys_to_girls_ratio : ℚ) 
  (num_girls : ℕ) 
  (absence_rate : ℚ) 
  (total_students : ℕ) : Prop :=
  boys_to_girls_ratio = 8 / 5 ∧
  num_girls = 160 ∧
  absence_rate = 15 / 100 ∧
  total_students = 478

theorem prove_college_students_count : 
  ∃ (boys_to_girls_ratio : ℚ) (num_girls : ℕ) (absence_rate : ℚ) (total_students : ℕ),
  college_students_count boys_to_girls_ratio num_girls absence_rate total_students :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_college_students_count_prove_college_students_count_l439_43945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l439_43952

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 1 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l439_43952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l439_43939

theorem factorial_equality : 2^6 * 3^3 * 2100 = Nat.factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l439_43939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_stoppage_time_is_10_minutes_l439_43969

/-- Represents a bus with its speeds and stoppage time -/
structure Bus where
  speed_without_stoppage : ℝ
  speed_with_stoppage : ℝ
  stoppage_time : ℝ

/-- Calculates the stoppage time for a bus given its speeds -/
noncomputable def calculate_stoppage_time (speed_without_stoppage speed_with_stoppage : ℝ) : ℝ :=
  (speed_without_stoppage - speed_with_stoppage) / speed_without_stoppage * 60

/-- Theorem: The average stoppage time for the three buses is 10 minutes -/
theorem average_stoppage_time_is_10_minutes 
  (bus_a bus_b bus_c : Bus)
  (ha : bus_a.speed_without_stoppage = 54)
  (hb : bus_b.speed_without_stoppage = 60)
  (hc : bus_c.speed_without_stoppage = 72)
  (ha' : bus_a.speed_with_stoppage = 45)
  (hb' : bus_b.speed_with_stoppage = 50)
  (hc' : bus_c.speed_with_stoppage = 60)
  (ha'' : bus_a.stoppage_time = calculate_stoppage_time bus_a.speed_without_stoppage bus_a.speed_with_stoppage)
  (hb'' : bus_b.stoppage_time = calculate_stoppage_time bus_b.speed_without_stoppage bus_b.speed_with_stoppage)
  (hc'' : bus_c.stoppage_time = calculate_stoppage_time bus_c.speed_without_stoppage bus_c.speed_with_stoppage) :
  (bus_a.stoppage_time + bus_b.stoppage_time + bus_c.stoppage_time) / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_stoppage_time_is_10_minutes_l439_43969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_volume_difference_l439_43926

/-- The volume of a cylindrical tube formed by a rectangular sheet of paper -/
noncomputable def cylinderVolume (width : ℝ) (height : ℝ) : ℝ :=
  (width^2 * height) / (4 * Real.pi)

/-- The difference in volumes of two cylindrical tubes multiplied by π -/
noncomputable def volumeDifference (width : ℝ) (height : ℝ) : ℝ :=
  Real.pi * |cylinderVolume width height - cylinderVolume height width|

theorem paper_roll_volume_difference :
  volumeDifference 9 12 = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_volume_difference_l439_43926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_achieve_target_l439_43931

/-- Represents a vessel containing a salt solution -/
structure Vessel where
  capacity : ℝ
  volume : ℝ
  salt_concentration : ℝ

/-- Represents the state of the two vessels -/
structure VesselState where
  vessel_a : Vessel
  vessel_b : Vessel

/-- Represents a liquid transfer operation between vessels -/
inductive Transfer where
  | a_to_b : ℝ → Transfer
  | b_to_a : ℝ → Transfer

/-- Initial state of the vessels -/
def initial_state : VesselState :=
  { vessel_a := { capacity := 3, volume := 1, salt_concentration := 0 }
  , vessel_b := { capacity := 3, volume := 1, salt_concentration := 0.02 } }

/-- Applies a transfer operation to the current state -/
def apply_transfer (state : VesselState) (transfer : Transfer) : VesselState :=
  sorry

/-- Checks if the given state has achieved the target concentration in vessel A -/
def target_achieved (state : VesselState) : Prop :=
  state.vessel_a.salt_concentration = 0.015

/-- Theorem stating that it's impossible to achieve the target concentration -/
theorem impossible_to_achieve_target : 
  ∀ (transfers : List Transfer), ¬(target_achieved (transfers.foldl apply_transfer initial_state)) :=
  sorry

#check impossible_to_achieve_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_achieve_target_l439_43931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_acceleration_bound_l439_43903

/-- Represents the state of a car at a given time -/
structure CarState where
  time : ℝ
  position : ℝ
  velocity : ℝ

/-- Represents the journey of a car -/
def CarJourney := ℝ → CarState

def is_valid_journey (j : CarJourney) : Prop :=
  ∃ t_end : ℝ,
    t_end > 0 ∧
    j 0 = ⟨0, 0, 0⟩ ∧  -- Starts at rest
    j t_end = ⟨t_end, 5280, 0⟩ ∧  -- Ends at rest after 1 mile (5280 feet)
    (∀ t, 0 ≤ t ∧ t ≤ t_end → (j t).time = t) ∧
    (∀ t, 0 ≤ t ∧ t ≤ t_end → (j t).velocity ≤ 132)  -- Max speed 90 mph = 132 ft/s

theorem car_acceleration_bound (j : CarJourney) (h : is_valid_journey j) :
  ∃ t a, 0 ≤ t ∧ t ≤ 60 ∧ |a| ≥ 6.6 ∧
    ∀ ε > 0, ∃ δ > 0, ∀ t', |t' - t| < δ →
      |(j t').velocity - (j t).velocity| ≤ (|a| + ε) * |t' - t| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_acceleration_bound_l439_43903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_troy_straws_l439_43955

theorem troy_straws (initial_straws : ℚ) : 
  (3 / 5 : ℚ) * initial_straws = (3 / 5 : ℚ) * initial_straws → -- Adult pigs ate 3/5 of straws
  (3 / 5 : ℚ) * initial_straws = 20 * 6 → -- Piglets ate equal amount as adult pigs, 20 piglets ate 6 each
  initial_straws = 200 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_troy_straws_l439_43955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l439_43992

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + c) / (x^2 - x - 12)

theorem exactly_one_vertical_asymptote (c : ℝ) :
  (∃! x, ¬∃ y, g c x = y) ↔ c = -24 ∨ c = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l439_43992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_equality_l439_43934

theorem xyz_equality (x y z : ℕ+) (a b c d : ℝ) 
  (h1 : x ≤ y ∧ y ≤ z)
  (h2 : (x : ℝ) ^ a = (y : ℝ) ^ b ∧ (y : ℝ) ^ b = (z : ℝ) ^ c ∧ (z : ℝ) ^ c = 70 ^ d)
  (h3 : 1 / a + 1 / b + 1 / c = 1 / d) :
  x + y = z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_equality_l439_43934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_year_afforestation_l439_43967

/-- Represents the afforestation plan of a forestry farm -/
structure AfforestationPlan where
  initial_area : ℝ  -- Initial area afforested in mu
  annual_increase : ℝ  -- Annual increase rate

/-- Calculates the area afforested in a given year -/
def area_afforested (plan : AfforestationPlan) (year : ℕ) : ℝ :=
  plan.initial_area * (1 + plan.annual_increase) ^ (year - 1)

/-- The main theorem about the afforestation area in the third year -/
theorem third_year_afforestation (plan : AfforestationPlan) 
  (h1 : plan.initial_area = 10000)
  (h2 : plan.annual_increase = 0.1) : 
  area_afforested plan 3 = 12100 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_year_afforestation_l439_43967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_roots_of_polynomial_l439_43998

-- Define the polynomial
def p (A B x : ℝ) : ℝ := x^4 + A*x^3 + 29.25*x^2 - 40.28*x + B

-- State the theorem
theorem other_roots_of_polynomial (A B : ℝ) :
  (p A B 1.1 = 0) →
  (p A B 2.4 = 0) →
  ∃ (x3 x4 : ℝ), x3 = 2.3 ∧ x4 = 3.2 ∧ 
    (p A B x3 = 0) ∧ (p A B x4 = 0) ∧
    (∀ x : ℝ, p A B x = 0 ↔ x ∈ ({1.1, 2.4, x3, x4} : Set ℝ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_roots_of_polynomial_l439_43998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_iff_k_eq_plus_minus_one_l439_43905

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - Real.exp (x * Real.log 2)) / (1 + k * Real.exp (x * Real.log 2))

theorem odd_function_iff_k_eq_plus_minus_one :
  ∀ k : ℝ, (∀ x : ℝ, f k (-x) = -(f k x)) ↔ (k = 1 ∨ k = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_iff_k_eq_plus_minus_one_l439_43905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_n_with_all_digits_gt5_and_square_lt5_l439_43938

/-- Given a positive integer n, returns true if all its digits are greater than 5 -/
def allDigitsGreaterThan5 (n : ℕ) : Prop :=
  ∀ d, d ∈ Nat.digits 10 n → d > 5

/-- Given a positive integer n, returns true if all its digits are less than 5 -/
def allDigitsLessThan5 (n : ℕ) : Prop :=
  ∀ d, d ∈ Nat.digits 10 n → d < 5

/-- The main theorem stating that no positive integer exists with all digits greater than 5
    and its square having all digits less than 5 -/
theorem no_n_with_all_digits_gt5_and_square_lt5 :
  ¬ ∃ n : ℕ, n > 0 ∧ allDigitsGreaterThan5 n ∧ allDigitsLessThan5 (n^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_n_with_all_digits_gt5_and_square_lt5_l439_43938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l439_43925

noncomputable section

/-- The hyperbola with equation x²/9 - y²/4 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) - (p.2^2 / 4) = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-Real.sqrt 13, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (Real.sqrt 13, 0)

/-- Vector from a point to F₁ -/
def PF₁ (P : ℝ × ℝ) : ℝ × ℝ := (F₁.1 - P.1, F₁.2 - P.2)

/-- Vector from a point to F₂ -/
def PF₂ (P : ℝ × ℝ) : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude of a 2D vector -/
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem hyperbola_property (P : ℝ × ℝ) 
  (h₁ : P ∈ Hyperbola) 
  (h₂ : dot_product (PF₁ P) (PF₂ P) = 0) : 
  magnitude (PF₁ P + PF₂ P) = 2 * Real.sqrt 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l439_43925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l439_43913

/-- The number of hours on a clock. -/
def hours_on_clock : ℕ := 12

/-- The angle between each hour mark on a clock. -/
noncomputable def angle_between_hours : ℝ := 360 / hours_on_clock

/-- The position of the minute hand at 3:15 p.m. in terms of hour marks. -/
def minute_hand_position : ℝ := 3

/-- The position of the hour hand at 3:15 p.m. in terms of hour marks. -/
def hour_hand_position : ℝ := 3.25

/-- The smaller angle formed by the hour and minute hands at 3:15 p.m. -/
def smaller_angle : ℝ := 7.5

/-- Theorem stating that the smaller angle between the hour and minute hands at 3:15 p.m. is 7.5 degrees. -/
theorem clock_angle_at_3_15 :
  let diff := |minute_hand_position - hour_hand_position|
  smaller_angle = min (diff * angle_between_hours) ((1 - diff) * angle_between_hours) := by
  sorry

#eval smaller_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l439_43913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_internal_tangent_specific_circles_l439_43987

/-- The length of the common internal tangent between two circles -/
noncomputable def common_internal_tangent_length (center_distance : ℝ) (radius1 : ℝ) (radius2 : ℝ) : ℝ :=
  Real.sqrt (center_distance^2 - (radius1 + radius2)^2)

/-- Theorem stating the length of the common internal tangent for the given circles -/
theorem common_internal_tangent_specific_circles :
  common_internal_tangent_length 50 7 10 = Real.sqrt 2211 := by
  sorry

#check common_internal_tangent_specific_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_internal_tangent_specific_circles_l439_43987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_lambda_exists_l439_43999

/-- A sequence of positive real numbers satisfying a_{n+1}^2 = a_n * a_{n+2} + k -/
def SpecialSequence (a : ℕ+ → ℝ) (k : ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0) ∧ 
  (∀ n : ℕ+, (a (n + 1))^2 = (a n) * (a (n + 2)) + k)

/-- The theorem stating the existence of λ for the special sequence -/
theorem special_sequence_lambda_exists 
  (a : ℕ+ → ℝ) (k a₁ a₂ : ℝ) 
  (h : SpecialSequence a k) 
  (h₁ : a 1 = a₁) 
  (h₂ : a 2 = a₂) :
  ∃ l : ℝ, l = (a₁^2 + a₂^2 - k) / (a₁ * a₂) ∧
    ∀ n : ℕ+, a n + a (n + 2) = l * a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_lambda_exists_l439_43999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divisibility_l439_43917

theorem odd_prime_divisibility (q : ℕ) (h_prime : Nat.Prime q) (h_odd : Odd q) :
  ¬ (∀ n : ℕ, n ∈ ({q - 4, q, q + 6, q + 3} : Set ℕ) → (q + 2)^(q - 3) + 1 ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divisibility_l439_43917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_value_l439_43974

-- Define the circle C in polar coordinates
noncomputable def circle_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

-- Define the line l in polar coordinates
noncomputable def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 4

-- Define the ratio |OP| / |OQ|
noncomputable def ratio (α : ℝ) : ℝ :=
  (2 * Real.sqrt 2 * Real.sin (α + Real.pi / 4)) / (4 / Real.sin (α + Real.pi / 4))

-- State the theorem
theorem max_ratio_value :
  ∀ α, 0 < α ∧ α < Real.pi / 2 →
  ratio α ≤ Real.sqrt 2 / 2 ∧
  ∃ α₀, 0 < α₀ ∧ α₀ < Real.pi / 2 ∧ ratio α₀ = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_value_l439_43974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_no_line_l439_43973

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focal distance of an ellipse -/
noncomputable def focalDistance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

theorem ellipse_equation_and_no_line (e : Ellipse) (A : Point) :
  isOnEllipse e A ∧ A.x = 1 ∧ A.y = Real.sqrt 2 / 2 →
  (e.a = Real.sqrt 2 ∧ e.b = 1) ∧
  ¬∃ (t : ℝ), ∃ (M N : Point), ∃ (P : Point), ∃ (Q : Point),
    (M.y = 2 * M.x + t) ∧
    (N.y = 2 * N.x + t) ∧
    isOnEllipse e M ∧
    isOnEllipse e N ∧
    M ≠ N ∧
    P.y = 5/3 ∧
    isOnEllipse e Q ∧
    (P.x - M.x, P.y - M.y) = (N.x - Q.x, N.y - Q.y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_no_line_l439_43973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_b_value_l439_43982

-- Define the parabola
def parabola (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the conditions
structure ParabolaConditions (b c : ℝ) :=
  (x₁ : ℝ)
  (x₂ : ℝ)
  (pos_x : 0 < x₁ ∧ 0 < x₂)
  (intersect_x : parabola x₁ b c = 0 ∧ parabola x₂ b c = 0)
  (length_AB : x₂ - x₁ = 1)
  (area_ABC : (1/2) * (x₂ - x₁) * |c| = 1)

-- State the theorem
theorem parabola_b_value (b c : ℝ) (h : ParabolaConditions b c) : b = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_b_value_l439_43982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_addition_theorem_l439_43985

/-- Represents the composition of an alloy --/
structure Alloy where
  total_mass : ℝ
  copper_mass : ℝ

/-- Calculates the percentage of copper in an alloy --/
noncomputable def copper_percentage (a : Alloy) : ℝ :=
  a.copper_mass / a.total_mass * 100

/-- The initial alloy --/
noncomputable def initial_alloy : Alloy :=
  { total_mass := 36
  , copper_mass := 36 * 0.45 }

/-- The amount of copper to be added --/
def copper_to_add : ℝ := 13.5

/-- The resulting alloy after adding copper --/
noncomputable def final_alloy : Alloy :=
  { total_mass := initial_alloy.total_mass + copper_to_add
  , copper_mass := initial_alloy.copper_mass + copper_to_add }

/-- Theorem stating that adding 13.5 kg of copper results in 60% copper content --/
theorem copper_addition_theorem :
  copper_percentage final_alloy = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_addition_theorem_l439_43985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_probability_l439_43977

def weekday_prob : ℚ := 1/2
def weekend_prob : ℚ := 3/4
def total_days : ℕ := 7
def weekdays : ℕ := 5
def weekends : ℕ := 2
def target_days : ℕ := 5

theorem chocolate_milk_probability :
  (Nat.choose weekdays 3 * Nat.choose weekends 2 * weekday_prob^3 * (1 - weekday_prob)^2 * weekend_prob^2 * (1 - weekend_prob)^0 +
   Nat.choose weekdays 4 * Nat.choose weekends 1 * weekday_prob^4 * (1 - weekday_prob)^1 * weekend_prob^1 * (1 - weekend_prob)^1 +
   Nat.choose weekdays 5 * Nat.choose weekends 0 * weekday_prob^5 * (1 - weekday_prob)^0 * weekend_prob^0 * (1 - weekend_prob)^2) = 781/1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_probability_l439_43977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_expressions_find_x_l439_43915

-- Part 1
theorem power_expressions (m n : ℤ) (a b : ℝ) 
  (h1 : (4 : ℝ)^m = a) (h2 : (8 : ℝ)^n = b) : 
  (2 : ℝ)^(2*m + 3*n) = a * b ∧ (2 : ℝ)^(4*m - 6*n) = a^2 / b^2 := by
  sorry

-- Part 2
theorem find_x (x : ℝ) 
  (h : 2 * (8 : ℝ)^x * 16 = (2 : ℝ)^23) : 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_expressions_find_x_l439_43915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_problem_l439_43970

/-- Calculate the total amount withdrawn after n years of annual deposits -/
noncomputable def total_amount (a r : ℝ) (n : ℕ) : ℝ :=
  a * ((1 + r)^(n + 1) - (1 + r)) / r

/-- The problem statement -/
theorem deposit_problem (a r : ℝ) (hr : r ≠ 0) :
  total_amount a r 7 = a * ((1 + r)^8 - (1 + r)) / r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_problem_l439_43970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l439_43984

-- Define the structure for a triangle
structure Triangle where
  angles : Fin 3 → Real
  sum_angles : (angles 0) + (angles 1) + (angles 2) = Real.pi

-- Define what it means for a triangle to be acute or obtuse
def is_acute (t : Triangle) : Prop :=
  ∀ i : Fin 3, t.angles i < Real.pi / 2

def is_obtuse (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i > Real.pi / 2

-- Define the relationship between the angles of triangles S and T
def angle_relationship (S T : Triangle) : Prop :=
  ∀ i : Fin 3, Real.cos (S.angles i) = Real.sin (T.angles i)

-- State the theorem
theorem triangle_classification (S T : Triangle) 
  (h : angle_relationship S T) : 
  is_acute S ∧ is_obtuse T :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l439_43984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_problem_representation_l439_43933

/-- Represents the number of sweet fruits -/
def x : ℕ := sorry

/-- Represents the number of bitter fruits -/
def y : ℕ := sorry

/-- The total number of fruits -/
def total_fruits : ℕ := 1000

/-- The total cost in wen -/
def total_cost : ℕ := 999

/-- The cost of 9 sweet fruits in wen -/
def sweet_fruit_cost : ℕ := 11

/-- The cost of 7 bitter fruits in wen -/
def bitter_fruit_cost : ℕ := 4

/-- Theorem stating that the system of linear equations correctly represents the problem -/
theorem fruit_problem_representation :
  (x + y = total_fruits) ∧
  ((sweet_fruit_cost : ℚ) / 9 * x + (bitter_fruit_cost : ℚ) / 7 * y = total_cost) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_problem_representation_l439_43933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_projects_to_ellipse_l439_43959

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Represents a parallelogram -/
structure Parallelogram where
  A' : Point2D
  B' : Point2D
  C' : Point2D
  D' : Point2D

/-- Represents a circle -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point2D
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Parallel projection function -/
noncomputable def parallelProjection (p : Point2D) : Point2D := sorry

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point2D) (c : Circle) : Prop := sorry

/-- Function to check if a point is on an ellipse -/
def isOnEllipse (p : Point2D) (e : Ellipse) : Prop := sorry

/-- Theorem stating that the image of an inscribed circle under parallel projection is an ellipse -/
theorem inscribed_circle_projects_to_ellipse 
  (s : Square) (p : Parallelogram) (c : Circle) 
  (h1 : c.center = Point2D.mk ((s.A.x + s.C.x) / 2) ((s.A.y + s.C.y) / 2))
  (h2 : c.radius = (s.B.x - s.A.x) / 2)
  (h3 : p.A' = parallelProjection s.A)
  (h4 : p.B' = parallelProjection s.B)
  (h5 : p.C' = parallelProjection s.C)
  (h6 : p.D' = parallelProjection s.D) :
  ∃ (e : Ellipse), 
    (∀ (p : Point2D), isOnEllipse p e ↔ 
      ∃ (q : Point2D), isOnCircle q c ∧ p = parallelProjection q) ∧
    (∃ (p1 p2 p3 p4 p5 p6 p7 p8 : Point2D),
      isOnEllipse p1 e ∧ isOnEllipse p2 e ∧ isOnEllipse p3 e ∧ isOnEllipse p4 e ∧
      isOnEllipse p5 e ∧ isOnEllipse p6 e ∧ isOnEllipse p7 e ∧ isOnEllipse p8 e ∧
      (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p1 ≠ p6 ∧ p1 ≠ p7 ∧ p1 ≠ p8) ∧
      (∀ (e' : Ellipse), 
        isOnEllipse p1 e' ∧ isOnEllipse p2 e' ∧ isOnEllipse p3 e' ∧ isOnEllipse p4 e' ∧
        isOnEllipse p5 e' ∧ isOnEllipse p6 e' ∧ isOnEllipse p7 e' ∧ isOnEllipse p8 e' →
        e = e')) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_projects_to_ellipse_l439_43959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logan_airport_passengers_approx_l439_43940

/-- The number of airline passengers (in millions) traveling to or from the United States in 1979 -/
noncomputable def total_passengers : ℝ := 38.3

/-- The fraction of total passengers that used Kennedy Airport -/
noncomputable def kennedy_fraction : ℝ := 1 / 3

/-- The ratio of Miami Airport passengers to Kennedy Airport passengers -/
noncomputable def miami_to_kennedy_ratio : ℝ := 1 / 2

/-- The ratio of Miami Airport passengers to Logan Airport passengers -/
noncomputable def miami_to_logan_ratio : ℝ := 4

/-- The number of passengers (in millions) who used Logan Airport -/
noncomputable def logan_passengers : ℝ := 
  (total_passengers * kennedy_fraction * miami_to_kennedy_ratio) / miami_to_logan_ratio

theorem logan_airport_passengers_approx :
  ∃ ε > 0, |logan_passengers - 1.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logan_airport_passengers_approx_l439_43940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l439_43961

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- The foot of the perpendicular from the origin to the plane -/
def foot : ℝ × ℝ × ℝ := (10, -2, 5)

/-- Checks if a point lies on a plane -/
def lies_on_plane (p : Plane) (point : ℝ × ℝ × ℝ) : Prop :=
  (p.A : ℝ) * point.fst + (p.B : ℝ) * point.snd.fst + (p.C : ℝ) * point.snd.snd + (p.D : ℝ) = 0

/-- Checks if a vector is perpendicular to a plane -/
def is_perpendicular (p : Plane) (v : ℝ × ℝ × ℝ) : Prop :=
  (p.A : ℝ) * v.fst + (p.B : ℝ) * v.snd.fst + (p.C : ℝ) * v.snd.snd = 0

theorem plane_equation (p : Plane) : 
  p.A = 10 ∧ p.B = -2 ∧ p.C = 5 ∧ p.D = -129 ↔ 
  lies_on_plane p foot ∧ is_perpendicular p foot := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l439_43961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l439_43958

/-- Represents the allocation of a research and development budget --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ
  basic_astrophysics : ℝ

/-- The theorem states that given the budget allocation percentages,
    the number of degrees representing basic astrophysics in a circle graph is 90 --/
theorem basic_astrophysics_degrees (b : BudgetAllocation) :
  b.microphotonics = 14 ∧
  b.home_electronics = 19 ∧
  b.food_additives = 10 ∧
  b.genetically_modified_microorganisms = 24 ∧
  b.industrial_lubricants = 8 ∧
  b.basic_astrophysics = 100 - (b.microphotonics + b.home_electronics + b.food_additives +
                                b.genetically_modified_microorganisms + b.industrial_lubricants) →
  b.basic_astrophysics * 3.6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l439_43958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_y_coordinate_l439_43935

/-- The y-coordinate of the point on the y-axis equidistant from C(-3, -1) and D(4, 7) -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℚ, y = 55 / 16 ∧ 
  ((-3 : ℚ) - 0)^2 + ((-1 : ℚ) - y)^2 = ((4 : ℚ) - 0)^2 + ((7 : ℚ) - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_y_coordinate_l439_43935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_curve_intersection_theorem_l439_43950

theorem circle_curve_intersection_theorem (R : ℝ) :
  R > 0 →
  (∃ (n : ℕ) (vertices : Fin n → ℝ × ℝ),
    (∀ (i : Fin n), (vertices i).1^2 + (vertices i).2^2 = R^2) ∧
    (∀ (i : Fin n), abs (abs (vertices i).1 - abs (vertices i).2) = 1) ∧
    (∀ (i j : Fin n), i ≠ j → vertices i ≠ vertices j) ∧
    (∀ (i j : Fin n), ∃ (k : Fin n), 
      (vertices k).1^2 + (vertices k).2^2 = R^2 ∧
      abs (abs (vertices k).1 - abs (vertices k).2) = 1 ∧
      dist (vertices i) (vertices k) = dist (vertices j) (vertices k))) →
  R = 1 ∨ R = Real.sqrt (2 + Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_curve_intersection_theorem_l439_43950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_b_value_monotone_increasing_implies_a_range_l439_43947

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 2 * x + b / x + Real.log x

-- Define the function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * x + Real.log x - a / x

-- Theorem 1: If x=1 is an extremum point of f(x), then b = 3
theorem extremum_point_implies_b_value (b : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f b 1 ≤ f b x) →
  b = 3 := by sorry

-- Theorem 2: If g(x) is monotonically increasing on [1,2], then a ≥ -3
theorem monotone_increasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → g a x ≤ g a y) →
  a ≥ -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_b_value_monotone_increasing_implies_a_range_l439_43947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_inequality_condition_l439_43922

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |x - 1| + 2 * |x - 2|

-- Theorem for the range of f
theorem f_range : Set.range f = Set.Ici 1 := by sorry

-- Theorem for the inequality condition
theorem inequality_condition (m : ℝ) : 
  (∃ x, f x - m < 0) → 3*m + 2/(m-1) > 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_inequality_condition_l439_43922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_ADB_l439_43976

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def triangle_area (y : ℝ) : ℝ := 1/2 * 1 * 2*y

noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ := Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

theorem constant_angle_ADB (a b : ℝ) (A B D : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  eccentricity a b = 1/2 ∧
  (∃ y, ellipse a b 1 y ∧ triangle_area y = 3/2) ∧
  (∃ m : ℝ, ellipse a b A.1 A.2 ∧ ellipse a b B.1 B.2 ∧ A.2 = m * (A.1 - 2/7) ∧ B.2 = m * (B.1 - 2/7)) ∧
  D = (a, 0) →
  angle (A.1 - D.1, A.2 - D.2) (B.1 - D.1, B.2 - D.2) = π/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_ADB_l439_43976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l439_43980

/-- An arithmetic sequence with the property that a_6 + a_10 = 20 -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 6 + a 10 = 20

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  sum_n seq 15 = 150 ∧
  seq.a 8 = 10 ∧
  seq.a 4 + seq.a 12 = 20 ∧
  seq.a 16 ≠ 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l439_43980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_time_sum_l439_43996

/-- The time Jerry spent in the pool, in minutes -/
noncomputable def jerry_time : ℝ := 3

/-- The time Elaine spent in the pool, in minutes -/
noncomputable def elaine_time : ℝ := 2 * jerry_time

/-- The time George spent in the pool, in minutes -/
noncomputable def george_time : ℝ := (1/3) * elaine_time

/-- The time Kramer spent in the pool, in minutes -/
noncomputable def kramer_time : ℝ := 0

/-- The total time spent in the pool by all friends -/
noncomputable def total_time : ℝ := jerry_time + elaine_time + george_time + kramer_time

theorem pool_time_sum : total_time = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_time_sum_l439_43996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l439_43995

/-- The time taken for a diver to reach a certain depth -/
noncomputable def time_to_reach_depth (depth : ℝ) (descent_rate : ℝ) : ℝ :=
  depth / descent_rate

/-- Theorem: The time taken for a diver to reach a depth of 3600 feet
    while descending at a rate of 60 feet per minute is 60 minutes -/
theorem diver_descent_time :
  time_to_reach_depth 3600 60 = 60 := by
  -- Unfold the definition of time_to_reach_depth
  unfold time_to_reach_depth
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l439_43995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_difference_l439_43972

/-- Represents the speed difference between two cyclists -/
noncomputable def speed_difference (time : ℝ) (distance : ℝ) (base_speed : ℝ) : ℝ :=
  2 * (distance / time - 2 * base_speed)

/-- Theorem stating the speed difference between two cyclists -/
theorem cyclist_speed_difference :
  let time : ℝ := 0.5
  let distance : ℝ := 14
  let base_speed : ℝ := 13
  speed_difference time distance base_speed = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_difference_l439_43972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_problem_l439_43900

/-- Represents the map scale in inches per mile -/
noncomputable def map_scale (map_distance : ℝ) (actual_distance : ℝ) : ℝ :=
  map_distance / actual_distance

/-- Represents the actual distance traveled in miles -/
noncomputable def actual_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem map_scale_problem (map_distance : ℝ) (travel_time : ℝ) (speed : ℝ) 
  (h1 : map_distance = 5)
  (h2 : travel_time = 5)
  (h3 : speed = 60) :
  map_scale map_distance (actual_distance speed travel_time) = 1 / 60 := by
  sorry

#check map_scale_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_problem_l439_43900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_prime_power_of_three_l439_43965

def K : ℕ → ℤ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 8
  | (n + 3) => 3 * K (n + 2) - K (n + 1) + 5 * (-1 : ℤ) ^ (n + 3)

theorem K_prime_power_of_three (n : ℕ) (h : Nat.Prime (Int.natAbs (K n))) : 
  ∃ k : ℕ, n = 3 ^ k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_prime_power_of_three_l439_43965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_intersection_length_l439_43932

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos (Real.pi / 4), t * Real.sin (Real.pi / 4))

-- Define the curve C in polar form
noncomputable def curve_C_polar (θ : ℝ) : ℝ :=
  4 * Real.cos θ / (Real.sin θ)^2

-- State the theorem
theorem curve_C_and_intersection_length :
  -- Part I: Cartesian equation of curve C
  (∀ x y : ℝ, (∃ θ : ℝ, x = curve_C_polar θ * Real.cos θ ∧ 
                         y = curve_C_polar θ * Real.sin θ) ↔ 
                         y^2 = 4*x) ∧
  -- Part II: Length of AB
  (∃ A B : ℝ × ℝ, ∃ t₁ t₂ : ℝ,
    A = line_l t₁ ∧ 
    B = line_l t₂ ∧
    (A.1)^2 = 4*(A.2) ∧
    (B.1)^2 = 4*(B.2) ∧
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_intersection_length_l439_43932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_intersection_segments_l439_43963

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℚ × ℚ

/-- The length between two points -/
noncomputable def length (p q : ℚ × ℚ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A diagonal of a polygon -/
structure Diagonal (n : ℕ) (p : Polygon n) where
  start : Fin n
  end_ : Fin n
  ne : start ≠ end_

/-- An intersection point of two diagonals -/
structure Intersection (n : ℕ) (p : Polygon n) where
  d1 : Diagonal n p
  d2 : Diagonal n p
  point : ℚ × ℚ

/-- A theorem stating that in a polygon with rational side and diagonal lengths,
    any segment formed by diagonal intersections is also rational -/
theorem rational_intersection_segments
  (n : ℕ)
  (p : Polygon n)
  (sides_rational : ∀ i j, ∃ q : ℚ, (length (p.vertices i) (p.vertices j) : ℝ) = q)
  (diagonals_rational : ∀ d : Diagonal n p, ∃ q : ℚ, (length (p.vertices d.start) (p.vertices d.end_) : ℝ) = q)
  (i1 i2 : Intersection n p)
  : ∃ q : ℚ, (length i1.point i2.point : ℝ) = q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_intersection_segments_l439_43963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_240km_40kmph_l439_43936

/-- Calculates the time taken to travel a given distance at a given speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem: The time taken to travel 240 km at a speed of 40 km/hr is 6 hours -/
theorem travel_time_240km_40kmph :
  travel_time 240 40 = 6 := by
  -- Unfold the definition of travel_time
  unfold travel_time
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_240km_40kmph_l439_43936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l439_43993

open Real

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * sin x * (sin x + Real.sqrt 3 * cos x)

noncomputable def g (x : ℝ) : ℝ := f (x + π / 3)

theorem graph_translation (x : ℝ) : g x = 2 * sin (2 * x - π / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l439_43993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_theorem_l439_43904

/-- Represents the configuration of three overlapping strips -/
structure StripConfiguration where
  width : ℝ
  angle : ℝ

/-- Calculates the total overlapping area of three strips -/
noncomputable def totalOverlappingArea (config : StripConfiguration) : ℝ :=
  (4 * config.width^2) / Real.sin config.angle

/-- Theorem stating the total overlapping area of three strips -/
theorem overlapping_area_theorem (config : StripConfiguration) 
    (h1 : config.width = 2) 
    (h2 : 0 < config.angle ∧ config.angle < Real.pi / 2) : 
  totalOverlappingArea config = 4 / Real.sin config.angle := by
  sorry

#check overlapping_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_theorem_l439_43904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_dot_product_l439_43988

/-- Given a quadrilateral ABCD in a plane, prove that AC · BD = 16 under specific conditions. -/
theorem quadrilateral_dot_product (A B C D E F : ℝ × ℝ) : 
  E = (A + D) / 2 →  -- E is midpoint of AD
  F = (B + C) / 2 →  -- F is midpoint of BC
  ‖B - A‖ = 1 →     -- AB = 1
  ‖F - E‖ = Real.sqrt 2 →    -- EF = √2
  ‖D - C‖ = 3 →     -- CD = 3
  (D - A) • (C - B) = 15 →  -- AD · BC = 15
  (C - A) • (D - B) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_dot_product_l439_43988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_eleventh_l439_43914

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) + sequence_a n

noncomputable def series_sum : ℝ := ∑' n, sequence_a n / 4^(n + 1)

theorem series_sum_equals_one_eleventh : series_sum = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_eleventh_l439_43914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_from_subset_sums_l439_43960

noncomputable def subset_sums (A : Finset ℝ) : Finset ℝ :=
  (A.powerset.filter (fun s => s.card = 3)).image (fun s => s.sum id)

theorem unique_set_from_subset_sums :
  ∃! (A : Finset ℝ),
    A.card = 5 ∧
    (∃ (a b c d e : ℝ), A = {a, b, c, d, e} ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e) ∧
    subset_sums A = {0, 3, 4, 8, 9, 10, 11, 12, 14, 19} ∧
    A = {-3, -1, 4, 7, 8} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_from_subset_sums_l439_43960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michelle_bill_l439_43916

/-- Represents a cell phone plan with its pricing structure -/
structure CellPhonePlan where
  baseCost : ℚ
  textCost : ℚ
  extraMinuteCost : ℚ
  extraDataCost : ℚ
  freeMinutes : ℚ
  freeData : ℚ

/-- Represents a customer's usage -/
structure Usage where
  texts : ℚ
  minutes : ℚ
  data : ℚ

/-- Calculates the total bill for a given plan and usage -/
def calculateBill (plan : CellPhonePlan) (usage : Usage) : ℚ :=
  plan.baseCost +
  plan.textCost * usage.texts +
  plan.extraMinuteCost * max (usage.minutes - plan.freeMinutes) 0 +
  plan.extraDataCost * max (usage.data - plan.freeData) 0

/-- The theorem stating that Michelle's bill is $39.50 -/
theorem michelle_bill :
  let plan := CellPhonePlan.mk 20 (3/100) (1/10) (15/100) (30 * 60) 500
  let usage := Usage.mk 200 (31 * 60) 550
  calculateBill plan usage = 39.5 := by
  sorry

#eval let plan := CellPhonePlan.mk 20 (3/100) (1/10) (15/100) (30 * 60) 500
      let usage := Usage.mk 200 (31 * 60) 550
      calculateBill plan usage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michelle_bill_l439_43916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l439_43948

/-- The unoccupied volume in a cube-shaped tank -/
theorem unoccupied_volume (tank_side : ℝ) (water_fraction : ℝ) (num_marbles : ℕ) (marble_diameter : ℝ) : 
  tank_side = 12 →
  water_fraction = 1/3 →
  num_marbles = 15 →
  marble_diameter = 1 →
  let tank_volume := tank_side^3
  let water_volume := water_fraction * tank_volume
  let marble_volume := (4/3) * Real.pi * (marble_diameter/2)^3
  let total_marble_volume := (num_marbles : ℝ) * marble_volume
  tank_volume - (water_volume + total_marble_volume) = 1728 - (576 + 15 * Real.pi / 6) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l439_43948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlotte_journey_time_l439_43964

/-- Calculate the time taken for a journey given distance and speed -/
noncomputable def journey_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Theorem: Charlotte's journey time is 6 hours -/
theorem charlotte_journey_time :
  let distance : ℝ := 60
  let speed : ℝ := 10
  journey_time distance speed = 6 := by
  -- Unfold the definition of journey_time
  unfold journey_time
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlotte_journey_time_l439_43964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_f_max_min_on_interval_l439_43911

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem f_property (x : ℝ) (h : x ≠ 1) : f ((x - 1) / (x + 1)) = -x - 1 := by sorry

theorem f_max_min_on_interval :
  let a := (2 : ℝ)
  let b := (6 : ℝ)
  (∀ x ∈ Set.Icc a b, f x ≤ f a) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ f b) ∧
  f a = 2 ∧
  f b = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_f_max_min_on_interval_l439_43911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_to_longest_side_l439_43957

/-- Triangle with sides a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Semi-perimeter of a triangle --/
noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- Area of a triangle using Heron's formula --/
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Altitude to side a --/
noncomputable def altitude_to_a (t : Triangle) : ℝ := 2 * area t / t.a

/-- Altitude to side b --/
noncomputable def altitude_to_b (t : Triangle) : ℝ := 2 * area t / t.b

/-- Altitude to side c --/
noncomputable def altitude_to_c (t : Triangle) : ℝ := 2 * area t / t.c

/-- The triangle with sides 13, 14, and 15 --/
def triangle_13_14_15 : Triangle where
  a := 13
  b := 14
  c := 15
  pos_a := by norm_num
  pos_b := by norm_num
  pos_c := by norm_num
  triangle_ineq := by norm_num

theorem shortest_altitude_to_longest_side :
  altitude_to_c triangle_13_14_15 < altitude_to_a triangle_13_14_15 ∧
  altitude_to_c triangle_13_14_15 < altitude_to_b triangle_13_14_15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_to_longest_side_l439_43957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_on_line_l439_43986

/-- The projection of vector u onto vector v -/
noncomputable def proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let v_norm_squared := v.1^2 + v.2^2
  (dot_product / v_norm_squared * v.1, dot_product / v_norm_squared * v.2)

/-- The theorem stating that vectors satisfying the projection condition lie on the line y = 3x - 15 -/
theorem vectors_on_line (u : ℝ × ℝ) :
  proj (3, -1) u = (9/2, -3/2) → u.2 = 3 * u.1 - 15 := by
  sorry

#check vectors_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_on_line_l439_43986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_moles_l439_43910

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  reactant1 : Moles
  reactant2 : Moles
  product1 : Moles
  product2 : Moles

/-- The balanced chemical equation for the reaction -/
def balancedEquation : ChemicalReaction → Prop :=
  λ reaction => reaction.reactant1 = reaction.reactant2 ∧
                reaction.reactant1 = reaction.product1 ∧
                reaction.reactant1 = reaction.product2

theorem hydrochloric_acid_moles (silver_nitrate : Moles) (silver_chloride : Moles) (nitric_acid : Moles)
    (h1 : silver_nitrate = (2 : ℝ))
    (h2 : silver_chloride = (2 : ℝ))
    (h3 : nitric_acid = (2 : ℝ))
    (reaction : ChemicalReaction)
    (h4 : balancedEquation reaction)
    (h5 : reaction.reactant1 = silver_nitrate)
    (h6 : reaction.product1 = silver_chloride)
    (h7 : reaction.product2 = nitric_acid) :
    reaction.reactant2 = (2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_moles_l439_43910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abs_power_sum_linear_function_through_points_l439_43990

-- Problem 1
theorem cube_root_abs_power_sum : (8 : ℝ) ^ (1/3) + |(-5 : ℝ)| + (-1 : ℝ) ^ 2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_points :
  ∃ k b : ℝ,
  (∀ x y : ℝ, y = k * x + b ↔ (x = 0 ∧ y = 1) ∨ (x = 2 ∧ y = 5)) ∧
  (∀ x : ℝ, k * x + b = 2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abs_power_sum_linear_function_through_points_l439_43990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_chord_length_l439_43923

-- Define the line l: y = x
def line (x y : ℝ) : Prop := y = x

-- Define the circle C: (x-2)^2 + (y-4)^2 = 10
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 10

-- Theorem for intersection points
theorem intersection_points :
  ∃ (x1 y1 x2 y2 : ℝ),
    line x1 y1 ∧ circle_eq x1 y1 ∧
    line x2 y2 ∧ circle_eq x2 y2 ∧
    ((x1 = 5 ∧ y1 = 5) ∨ (x1 = 1 ∧ y1 = 1)) ∧
    ((x2 = 5 ∧ y2 = 5) ∧ (x2 = 1 ∧ y2 = 1)) ∧
    x1 ≠ x2 := by
  sorry

-- Theorem for chord length
theorem chord_length :
  ∃ (x1 y1 x2 y2 : ℝ),
    line x1 y1 ∧ circle_eq x1 y1 ∧
    line x2 y2 ∧ circle_eq x2 y2 ∧
    x1 ≠ x2 ∧
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_chord_length_l439_43923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_constants_l439_43901

theorem polynomial_identity_constants : ∃! (a b : ℝ),
  ∀ (x : ℝ), ∃ (p q : ℝ → ℝ), p (x^2) * q (x+1) - p (x+1) * q (x^2) = x^2 + a*x + b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_constants_l439_43901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_inverse_eq_three_l439_43942

theorem x_plus_inverse_eq_three (x : ℝ) (h : x + x⁻¹ = 3) :
  x^(1/2 : ℝ) + x^(-1/2 : ℝ) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_inverse_eq_three_l439_43942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l439_43912

/-- The standard equation of an ellipse given specific conditions -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) 
  (h_circle : C = {(x, y) | x^2 + y^2 - 2*x - 15 = 0})
  (h_foci : ∃ c : ℝ, c > 0 ∧ ∀ x y, (x, y) ∈ C → x ∈ Set.Icc (-c) c)
  (h_eccentricity : (1/2 : ℝ) = 1/2)
  (h_major_axis : ∃ r : ℝ, (∀ x y, (x, y) ∈ C → x^2 + y^2 ≤ r^2) ∧ 
                       (∃ x y, (x, y) ∈ C ∧ x^2 + y^2 = r^2) ∧
                       (∃ a : ℝ, 2*a = r)) :
  ∃ E : Set (ℝ × ℝ), E = {(x, y) | x^2/4 + y^2/3 = 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l439_43912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_directrix_distance_is_sqrt5_plus_2_l439_43966

/-- Represents an ellipse that always passes through the point (1,2) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_fixed_point : 1 / a^2 + 4 / b^2 = 1

/-- The minimum value of a^2/c for the given ellipse -/
noncomputable def min_directrix_distance (e : Ellipse) : ℝ := Real.sqrt 5 + 2

/-- Theorem stating that the minimum distance from the center of the ellipse to a directrix is √5 + 2 -/
theorem min_directrix_distance_is_sqrt5_plus_2 (e : Ellipse) :
  ∃ c : ℝ, c > 0 ∧ c^2 = e.a^2 - e.b^2 ∧ 
  ∀ c' : ℝ, c' > 0 → c'^2 = e.a^2 - e.b^2 → e.a^2 / c' ≥ min_directrix_distance e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_directrix_distance_is_sqrt5_plus_2_l439_43966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_domain_l439_43919

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - Real.log (x - 1))

-- State the theorem
theorem domain_of_f : 
  {x : ℝ | x ∈ Set.Ioo 1 11} = {x : ℝ | x - 1 > 0 ∧ 1 - Real.log (x - 1) ≥ 0} := by
  sorry

-- Define the domain explicitly
def domain_f : Set ℝ := Set.Ioo 1 11

-- State that this is indeed the domain of f
theorem f_domain : 
  domain_f = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_domain_l439_43919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l439_43944

open Real

/- Define an acute triangle ABC -/
def AcuteTriangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/- Define vectors m and n -/
noncomputable def m (A : ℝ) : ℝ × ℝ := (cos (A + π/3), sin (A + π/3))
noncomputable def n (B : ℝ) : ℝ × ℝ := (cos B, sin B)

/- Define perpendicularity of vectors -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem triangle_properties (A B C : ℝ) (h_acute : AcuteTriangle A B C)
    (h_perp : perpendicular (m A) (n B)) :
  /- Part 1 -/
  A - B = π/6 ∧
  /- Part 2 -/
  (cos B = 3/5 → ∀ (AC : ℝ), AC = 8 →
    ∃ (BC : ℝ), BC = 4 * sqrt 3 + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l439_43944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_with_roots_as_coefficients_l439_43975

-- Define a quadratic equation
def quadratic_equation (a b : ℝ) (x : ℝ) : Prop :=
  x^2 + a*x + b = 0

-- Define the property that roots are equal to coefficients
def roots_equal_coefficients (a b : ℝ) : Prop :=
  ∃ x y : ℝ, quadratic_equation a b x ∧ quadratic_equation a b y ∧ 
  ((x = a ∧ y = b) ∨ (x = b ∧ y = a))

-- Theorem statement
theorem unique_quadratic_with_roots_as_coefficients :
  ∀ a b : ℝ, roots_equal_coefficients a b ↔ (a = 1 ∧ b = -2) :=
by sorry

#check unique_quadratic_with_roots_as_coefficients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_with_roots_as_coefficients_l439_43975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l439_43962

-- Define the power function as noncomputable
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x^α

-- Define the theorem
theorem power_function_theorem (f : ℝ → ℝ) (α : ℝ) 
  (h1 : f = powerFunction α)
  (h2 : f (1/8) = 2) :
  f (-1/8) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l439_43962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animal_ratio_l439_43924

theorem farm_animal_ratio (cows sheep pigs : ℕ) : 
  cows = 12 →
  pigs = 3 * sheep →
  cows + sheep + pigs = 108 →
  sheep / cows = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animal_ratio_l439_43924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_is_48pi_l439_43920

/-- The volume of ice cream in a cone with a hemispherical top -/
noncomputable def ice_cream_volume (cone_height : ℝ) (cone_radius : ℝ) : ℝ :=
  (1 / 3 * Real.pi * cone_radius ^ 2 * cone_height) + (2 / 3 * Real.pi * cone_radius ^ 3)

/-- Theorem: The total volume of ice cream in the given cone and hemisphere is 48π cubic inches -/
theorem ice_cream_volume_is_48pi :
  ice_cream_volume 10 3 = 48 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_is_48pi_l439_43920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_sum_l439_43930

noncomputable section

def F₁ : ℝ × ℝ := (0, 2)
def F₂ : ℝ × ℝ := (6, 2)

def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def is_on_ellipse (p : ℝ × ℝ) : Prop :=
  distance p F₁ + distance p F₂ = 10

def center : ℝ × ℝ :=
  ((F₁.1 + F₂.1) / 2, (F₁.2 + F₂.2) / 2)

def h : ℝ := center.1
def k : ℝ := center.2

def c : ℝ := distance F₁ center

def a : ℝ := 5  -- Half of the constant sum (10/2)
def b : ℝ := Real.sqrt (a^2 - c^2)

theorem ellipse_parameter_sum :
  h + k + a + b = 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_sum_l439_43930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l439_43902

theorem expansion_properties (x : ℝ) (x_pos : x > 0) :
  let expansion := (2 * Real.sqrt x - 1 / x) ^ 7
  ∃ (coeffs : List ℝ),
    (coeffs.length = 8) ∧
    (coeffs.sum = 1) ∧
    ((List.range 8).map (λ i => Nat.choose 7 i)).sum = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l439_43902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_in_standard_basis_l439_43968

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the standard basis vectors
variable (i j k : V)

-- Define the given basis vectors
def a (i j : V) : V := i + j
def b (j k : V) : V := j + k
def c (k i : V) : V := k + i

-- Define the vector m in the given basis
def m (i j k : V) : V := 8 • (a i j) + 6 • (b j k) + 4 • (c k i)

-- Theorem statement
theorem coordinates_in_standard_basis (i j k : V) :
  ∃ (x y z : ℝ), m i j k = x • i + y • j + z • k ∧ x = 12 ∧ y = 14 ∧ z = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_in_standard_basis_l439_43968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_10cm_sheet_l439_43928

/-- The height of a cone formed by rolling one sector of a circular sheet cut into four congruent sectors -/
noncomputable def cone_height (sheet_radius : ℝ) : ℝ :=
  let sector_arc_length := 2 * Real.pi * sheet_radius / 4
  let cone_base_radius := sector_arc_length / (2 * Real.pi)
  let slant_height := sheet_radius
  Real.sqrt (slant_height^2 - cone_base_radius^2)

/-- Theorem stating the height of the cone formed from a circular sheet with radius 10 cm -/
theorem cone_height_from_10cm_sheet :
  cone_height 10 = (5 * Real.sqrt 15) / 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cone_height 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_10cm_sheet_l439_43928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_game_condition_l439_43921

/-- Represents the game with 36 players and a 36-card deck -/
structure Game where
  num_players : ℕ
  num_cards : ℕ
  ace_of_diamonds : ℕ

/-- Represents a player in the game -/
structure Player where
  position : ℕ
  stake : ℝ

/-- Calculates the probability of winning for a player at a given position -/
noncomputable def win_probability (g : Game) (p : Player) : ℝ :=
  (35 / 36 : ℝ) ^ (p.position - 1) * (1 / 36 : ℝ)

/-- Calculates the expected value for a player -/
noncomputable def expected_value (g : Game) (p : Player) (total_pot : ℝ) : ℝ :=
  (total_pot - p.stake) * win_probability g p - p.stake * (1 - win_probability g p)

/-- Theorem stating the fair game condition -/
theorem fair_game_condition (g : Game) (p1 p2 : Player) (total_pot : ℝ) :
  g.num_players = 36 → g.num_cards = 36 →
  p2.position = p1.position + 1 →
  p2.stake = (35 / 36 : ℝ) * p1.stake →
  expected_value g p1 total_pot = expected_value g p2 total_pot := by
  sorry

#check fair_game_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_game_condition_l439_43921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_product_l439_43956

theorem symmetric_complex_product :
  ∀ (z₁ z₂ : ℂ),
  (z₁ = Complex.mk 2 (-1)) →
  (z₂ = -z₁) →
  (z₁ * z₂ = Complex.mk (-3) 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_product_l439_43956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l439_43918

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 2*x - 3) / Real.log 0.5

theorem monotonic_increasing_interval :
  {x : ℝ | x < -3} = {x : ℝ | ∀ y, f x < f y ↔ x < y ∧ x < -3 ∧ y < -3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l439_43918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_digit_product_l439_43937

theorem unique_square_digit_product : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (∃ m : ℕ, n = m^2) ∧
  (let d1 := n / 100;
   let d2 := (n / 10) % 10;
   let d3 := n % 10;
   d1 * d2 * d3 = Nat.sqrt n - 1) ∧
  n = 361 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_digit_product_l439_43937
