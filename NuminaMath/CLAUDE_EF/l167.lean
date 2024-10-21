import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_run_rate_l167_16781

def first_run_distance : ℝ := 5
def first_run_rate : ℝ := 10
def second_run_distance : ℝ := 4
def total_time : ℝ := 88

theorem second_run_rate : 
  (total_time - first_run_distance * first_run_rate) / second_run_distance = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_run_rate_l167_16781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoxiao_dosage_calculation_l167_16718

/-- Represents the medication dosage information -/
structure MedicationInfo where
  mg_per_tablet : ℕ
  min_mg_per_kg : ℕ
  max_mg_per_kg : ℕ

/-- Calculates the daily dosage range for a given weight -/
def calculate_daily_dosage (info : MedicationInfo) (weight_kg : ℕ) : ℕ × ℕ :=
  (info.min_mg_per_kg * weight_kg, info.max_mg_per_kg * weight_kg)

/-- Calculates the number of tablets per dose, given the daily dosage and number of doses -/
def calculate_tablets_per_dose (info : MedicationInfo) (daily_dosage : ℕ) (num_doses : ℕ) : ℕ :=
  ((daily_dosage / info.mg_per_tablet + num_doses - 1) / num_doses)

/-- Theorem stating the correct dosage calculations for Xiaoxiao -/
theorem xiaoxiao_dosage_calculation (info : MedicationInfo) 
  (h_mg_per_tablet : info.mg_per_tablet = 200)
  (h_min_mg_per_kg : info.min_mg_per_kg = 25)
  (h_max_mg_per_kg : info.max_mg_per_kg = 40)
  (weight_kg : ℕ) (h_weight : weight_kg = 45)
  (num_doses : ℕ) (h_doses : num_doses = 3) :
  let (min_daily, max_daily) := calculate_daily_dosage info weight_kg
  let min_tablets := calculate_tablets_per_dose info min_daily num_doses
  let max_tablets := calculate_tablets_per_dose info max_daily num_doses
  min_daily = 1125 ∧ max_daily = 1800 ∧ min_tablets = 2 ∧ max_tablets = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoxiao_dosage_calculation_l167_16718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x9_is_21_16_l167_16717

/-- The coefficient of x^9 in the expansion of (x^2 - 1/(2x))^9 -/
def coefficient_x9 : ℚ :=
  (-1)^6 * (1 / 2^6) * (Nat.choose 9 6)

/-- The expansion of (x^2 - 1/(2x))^9 -/
noncomputable def expansion (x : ℝ) : ℝ :=
  (x^2 - 1/(2*x))^9

theorem coefficient_x9_is_21_16 :
  coefficient_x9 = 21/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x9_is_21_16_l167_16717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_always_even_l167_16738

theorem expression_always_even (a b c : ℕ) (ha : Even a) (hb : Odd b) :
  Even (2^a * (b+1)^2 * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_always_even_l167_16738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_circle_l167_16765

/-- Given a circle with polar equation ρ = 2cos(θ), 
    its Cartesian coordinate equation is (x-1)^2 + y^2 = 1 -/
theorem polar_to_cartesian_circle :
  ∀ (x y : ℝ → ℝ) (θ : ℝ),
  let ρ : ℝ → ℝ := λ θ => 2 * Real.cos θ
  (∀ θ, x θ = ρ θ * Real.cos θ ∧ y θ = ρ θ * Real.sin θ) →
  (x θ - 1)^2 + (y θ)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_circle_l167_16765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_when_a_zero_f_unique_zero_iff_a_positive_l167_16782

/-- The function f(x) = ax - 1/x - (a+1)ln(x) where x > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

/-- The maximum value of f(x) when a = 0 is -1 -/
theorem f_max_value_when_a_zero :
  ∃ (M : ℝ), M = -1 ∧ ∀ x, x > 0 → f 0 x ≤ M := by
  sorry

/-- f(x) has exactly one zero if and only if a ∈ (0, +∞) -/
theorem f_unique_zero_iff_a_positive :
  ∀ a : ℝ, (∃! x, x > 0 ∧ f a x = 0) ↔ a > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_when_a_zero_f_unique_zero_iff_a_positive_l167_16782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l167_16784

-- Define the givens
noncomputable def train_speed_kmh : ℝ := 54
noncomputable def time_passing_platform : ℝ := 22
noncomputable def time_passing_man : ℝ := 20

-- Convert speed to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

-- Define the theorem
theorem platform_length :
  let train_length := train_speed_ms * time_passing_man
  let platform_length := train_speed_ms * time_passing_platform - train_length
  platform_length = 30 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l167_16784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_play_seating_theorem_l167_16761

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 : ℕ) (d : ℤ) (n : ℕ) : ℕ :=
  ((n : ℤ) * (2 * (a1 : ℤ) + ((n : ℤ) - 1) * d) / 2).toNat

/-- Represents the seating arrangement for the school play -/
structure SeatingArrangement where
  sectionA_rows : ℕ
  sectionA_chairs_per_row : ℕ
  sectionB_rows : ℕ
  sectionB_first_row : ℕ
  sectionB_increment : ℕ
  sectionC_rows : ℕ
  sectionC_first_row : ℕ
  sectionC_decrement : ℕ

/-- Calculates the total number of chairs in the seating arrangement -/
def totalChairs (s : SeatingArrangement) : ℕ :=
  s.sectionA_rows * s.sectionA_chairs_per_row +
  arithmeticSum s.sectionB_first_row s.sectionB_increment s.sectionB_rows +
  arithmeticSum s.sectionC_first_row (-s.sectionC_decrement) s.sectionC_rows

/-- The specific seating arrangement for the school play -/
def schoolPlaySeating : SeatingArrangement where
  sectionA_rows := 25
  sectionA_chairs_per_row := 17
  sectionB_rows := 30
  sectionB_first_row := 20
  sectionB_increment := 2
  sectionC_rows := 29
  sectionC_first_row := 16
  sectionC_decrement := 1

theorem school_play_seating_theorem :
  totalChairs schoolPlaySeating = 1953 := by
  sorry

#eval totalChairs schoolPlaySeating

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_play_seating_theorem_l167_16761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l167_16705

noncomputable def T (b : ℝ) := 15 / (1 - b)

theorem geometric_series_sum (b : ℝ) (h1 : -2 < b) (h2 : b < 2) (h3 : T b * T (-b) = 3600) :
  T b + T (-b) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l167_16705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_z_plus_i_l167_16733

noncomputable def complex_magnitude (z : ℂ) : ℝ := Real.sqrt (z.re * z.re + z.im * z.im)

theorem magnitude_z_plus_i (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  complex_magnitude (z + Complex.I) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_z_plus_i_l167_16733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l167_16793

def has_144_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 144

def has_10_consecutive_divisors (n : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ, i < 10 → (k + i) ∣ n

theorem smallest_n_with_conditions :
  ∃ n : ℕ, n = 110880 ∧ has_144_divisors n ∧ has_10_consecutive_divisors n ∧
  ∀ m : ℕ, m < n → ¬(has_144_divisors m ∧ has_10_consecutive_divisors m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l167_16793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l167_16795

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (1 + 3 * Real.cos θ, 3 + 3 * Real.sin θ)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + 0.5 * t, 3 + (Real.sqrt 3 / 2) * t)

-- Define point P
def point_P : ℝ × ℝ := (3, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement of the theorem
theorem intersection_distance_sum :
  ∃ (θ₁ θ₂ t₁ t₂ : ℝ),
    curve_C θ₁ = line_l t₁ ∧
    curve_C θ₂ = line_l t₂ ∧
    θ₁ ≠ θ₂ ∧
    distance point_P (curve_C θ₁) + distance point_P (curve_C θ₂) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l167_16795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_plus_minus_one_l167_16725

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log (Real.sqrt (x^2 + 1) - a*x)

theorem even_function_implies_a_plus_minus_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_plus_minus_one_l167_16725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_covers_larger_portion_in_first_triangle_l167_16721

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithIncircle where
  /-- The ratio of the distance from the base to the incircle touch point, to the length of the leg -/
  touch_ratio : ℝ
  /-- Assumes 0 < touch_ratio < 1 -/
  touch_ratio_pos : 0 < touch_ratio
  touch_ratio_lt_one : touch_ratio < 1

/-- Calculates the ratio of incircle area to triangle area for a given isosceles triangle with incircle -/
noncomputable def incircle_area_ratio (t : IsoscelesTriangleWithIncircle) : ℝ :=
  (2 * Real.pi) / (t.touch_ratio * (4 - t.touch_ratio))

/-- The first triangle with incircle touching at 1/3 from the base -/
noncomputable def first_triangle : IsoscelesTriangleWithIncircle :=
  { touch_ratio := 1/3
    touch_ratio_pos := by norm_num
    touch_ratio_lt_one := by norm_num }

/-- The second triangle with incircle touching at 2/3 from the base -/
noncomputable def second_triangle : IsoscelesTriangleWithIncircle :=
  { touch_ratio := 2/3
    touch_ratio_pos := by norm_num
    touch_ratio_lt_one := by norm_num }

/-- Theorem stating that the incircle covers a larger portion of the first triangle -/
theorem incircle_covers_larger_portion_in_first_triangle :
  incircle_area_ratio first_triangle > incircle_area_ratio second_triangle :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_covers_larger_portion_in_first_triangle_l167_16721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_inequality_l167_16714

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| - |2*x - 1|

-- Theorem 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 0} = Set.Ioo 0 2 := by sorry

-- Theorem 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, a > 0 →
  (∀ x : ℝ, f a x < 1) ↔ (0 < a ∧ a < 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_inequality_l167_16714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_intersection_over_union_l167_16709

def A : Finset ℕ := Finset.filter (fun n => ∃ k ∈ Finset.range 6, n = 6 * (k + 1) - 4) (Finset.range 33)
def B : Finset ℕ := Finset.filter (fun n => ∃ k ∈ Finset.range 6, n = 2^k) (Finset.range 33)

theorem probability_intersection_over_union : 
  (Finset.card (A ∩ B) : ℚ) / (Finset.card (A ∪ B) : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_intersection_over_union_l167_16709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_p_q_l167_16759

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 4 * Real.log x + 1 else 2 * x - 1

theorem min_sum_p_q (p q : ℝ) (h1 : p ≠ q) (h2 : f p + f q = 2) :
  ∃ (m : ℝ), m = 3 - 2 * Real.log 2 ∧ p + q ≥ m ∧ ∀ (p' q' : ℝ), p' ≠ q' → f p' + f q' = 2 → p' + q' ≥ m := by
  sorry

#check min_sum_p_q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_p_q_l167_16759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_needed_l167_16785

def total_files : ℕ := 40
def disk_capacity : ℚ := 2

def file_sizes : List ℚ := 
  (List.replicate 8 0.9) ++ (List.replicate 20 0.6) ++ (List.replicate 12 0.5)

def valid_disk_arrangement (arrangement : List (List ℚ)) : Prop :=
  (arrangement.length ≤ total_files) ∧ 
  (∀ disk ∈ arrangement, disk.sum ≤ disk_capacity) ∧
  (arrangement.join.toFinset = file_sizes.toFinset)

theorem min_disks_needed : 
  ∀ arrangement : List (List ℚ), 
    valid_disk_arrangement arrangement → 
    arrangement.length ≥ 15 :=
by
  intro arrangement h
  sorry

#check min_disks_needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_needed_l167_16785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_9_40_l167_16734

/-- Represents a rectangular yard with flower beds -/
structure YardWithFlowerBeds where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- The fraction of the yard occupied by flower beds -/
noncomputable def flower_bed_fraction (yard : YardWithFlowerBeds) : ℝ :=
  let triangle_leg := (yard.trapezoid_long_side - yard.trapezoid_short_side) / 2
  let flower_bed_area := 2 * (triangle_leg^2 / 2)
  let total_area := yard.length * yard.width
  flower_bed_area / total_area

/-- Theorem stating the fraction of the yard occupied by flower beds -/
theorem flower_bed_fraction_is_9_40 (yard : YardWithFlowerBeds) 
  (h1 : yard.length = 30)
  (h2 : yard.width = 8)
  (h3 : yard.trapezoid_short_side = 20)
  (h4 : yard.trapezoid_long_side = 35) :
  flower_bed_fraction yard = 9/40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_9_40_l167_16734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_is_68_l167_16752

/-- The list price of an article after two successive discounts -/
def discounted_price (original_price : ℝ) : ℝ :=
  original_price * (1 - 0.1) * (1 - 0.08235294117647069)

/-- Theorem stating that the original price is approximately 68 given the conditions -/
theorem original_price_is_68 :
  ∃ (price : ℝ), abs (discounted_price price - 56.16) < 0.01 ∧ abs (price - 68) < 0.01 := by
  sorry

#check original_price_is_68

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_is_68_l167_16752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_fourNinths_l167_16787

/-- The sum of the infinite series 1/(4^1) + 2/(4^2) + 3/(4^3) + ... + k/(4^k) + ... -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (4 : ℝ) ^ k

theorem infiniteSeries_eq_fourNinths : infiniteSeries = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_fourNinths_l167_16787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_11π_12_increasing_on_interval_not_right_shift_l167_16743

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x - π / 3)

-- Theorem for symmetry about x = 11π/12
theorem symmetry_about_11π_12 : ∀ x : ℝ, f (11 * π / 12 + x) = f (11 * π / 12 - x) := by
  sorry

-- Theorem for increasing on [-π/12, 5π/12]
theorem increasing_on_interval : ∀ x y : ℝ, -π/12 ≤ x → x < y → y ≤ 5*π/12 → f x < f y := by
  sorry

-- Theorem that f is not a right shift of 3sin(2x) by π/3
theorem not_right_shift : ∃ x : ℝ, f x ≠ 3 * sin (2 * (x - π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_11π_12_increasing_on_interval_not_right_shift_l167_16743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_angle_l167_16751

/-- The circle's equation: x^2 + y^2 - 4x + 3 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 3 = 0

/-- A line passing through the origin -/
def line_through_origin (k : ℝ) (x y : ℝ) : Prop :=
  k*x = y

/-- The line is tangent to the circle -/
def is_tangent (k : ℝ) : Prop :=
  ∃ x y : ℝ, circle_equation x y ∧ line_through_origin k x y

/-- The slope angle of a line with slope k -/
noncomputable def slope_angle (k : ℝ) : ℝ :=
  Real.arctan k

theorem tangent_line_slope_angle :
  ∀ k : ℝ, is_tangent k →
    (slope_angle k = π/6 ∨ slope_angle k = 5*π/6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_angle_l167_16751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_right_angle_ATC_l167_16753

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a line in 2D space -/
structure Line where
  a : Real
  b : Real
  c : Real

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle extends Triangle

/-- Represents the angle between three points -/
def angle (p1 p2 p3 : Point) : Real := sorry

/-- Checks if a triangle is acute -/
def IsAcuteTriangle (t : Triangle) : Prop := 
  angle t.A t.B t.C < Real.pi / 2 ∧
  angle t.B t.C t.A < Real.pi / 2 ∧
  angle t.C t.A t.B < Real.pi / 2

/-- Checks if two triangles are similar -/
def IsSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Represents the point of intersection of two lines -/
noncomputable def IntersectionPoint (l1 l2 : Line) : Point := sorry

/-- Checks if all points are distinct -/
def AllPointsDistinct (t1 t2 t3 : Triangle) (p : Point) : Prop := sorry

/-- Checks if an angle is a right angle -/
def IsRightAngle (θ : Real) : Prop := θ = Real.pi / 2

theorem not_right_angle_ATC (ABC : Triangle) 
  (h_acute : IsAcuteTriangle ABC)
  (h_angle_order : angle ABC.A ABC.B ABC.C > angle ABC.B ABC.C ABC.A ∧ 
                   angle ABC.B ABC.C ABC.A > angle ABC.C ABC.A ABC.B)
  (AC₁B : IsoscelesTriangle)
  (CB₁A : IsoscelesTriangle)
  (h_similar : IsSimilar AC₁B.toTriangle CB₁A.toTriangle)
  (T : Point)
  (h_intersection : T = IntersectionPoint 
                        (Line.mk 1 1 1) -- placeholder for BB₁
                        (Line.mk 1 1 1)) -- placeholder for CC₁
  (h_distinct : AllPointsDistinct ABC AC₁B.toTriangle CB₁A.toTriangle T) :
  ¬ IsRightAngle (angle ABC.A T ABC.C) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_right_angle_ATC_l167_16753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_chemistry_count_l167_16796

theorem basketball_team_chemistry_count (total_players biology_players both_sciences : ℕ) :
  total_players = 25 →
  biology_players = 12 →
  both_sciences = 6 →
  (∀ p, p ∈ (Finset.range total_players) → 
    p ∈ (Finset.range biology_players) ∪ (Finset.range (total_players - biology_players + both_sciences))) →
  ∃ chemistry_players : ℕ, chemistry_players = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_chemistry_count_l167_16796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l167_16723

/-- Represents a complex number --/
structure MyComplex where
  re : ℝ
  im : ℝ

/-- The rotation factor for each move --/
noncomputable def ω : MyComplex := { re := Real.sqrt 3 / 2, im := 1 / 2 }

/-- The translation distance for each move --/
def translation : ℝ := 6

/-- The number of moves --/
def moves : ℕ := 72

/-- The initial position of the particle --/
def initial_position : MyComplex := { re := 3, im := 0 }

/-- Performs one move on a given position --/
noncomputable def move (z : MyComplex) : MyComplex :=
  { re := ω.re * z.re - ω.im * z.im + translation,
    im := ω.im * z.re + ω.re * z.im }

/-- The final position after 'moves' number of moves --/
noncomputable def final_position : MyComplex := sorry

/-- Theorem stating that the final position is the same as the initial position --/
theorem particle_returns_to_start :
  final_position = initial_position := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l167_16723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l167_16755

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Part 1: Tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ x - y - 1 = 0 :=
by
  -- Proof steps would go here
  sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x, x > 1 → f a x > 0) ↔ a ≤ 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l167_16755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_given_line_l167_16744

-- Define the function f(x) = e^x * sin(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := Real.exp x * Real.sin x + Real.exp x * Real.cos x

theorem tangent_line_parallel_to_given_line :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let slope : ℝ := f_derivative x₀
  ∀ m l : ℝ,
    (slope = 1) →
    (∀ x y : ℝ, y - y₀ = slope * (x - x₀) ↔ y = x) →
    (∀ x y : ℝ, x + m * y + l = 0 ↔ y = -1/m * x - l/m) →
    m = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_given_line_l167_16744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_shifted_l167_16726

/-- Two monic polynomials of degree 2014 that are never equal -/
structure DistinctMonicPolynomials where
  P : Polynomial ℝ
  Q : Polynomial ℝ
  monic_P : P.Monic
  monic_Q : Q.Monic
  degree_P : P.degree = 2014
  degree_Q : Q.degree = 2014
  never_equal : ∀ x : ℝ, P.eval x ≠ Q.eval x

/-- The main theorem -/
theorem exists_equal_shifted (dmp : DistinctMonicPolynomials) :
  ∃ x : ℝ, dmp.P.eval (x - 1) = dmp.Q.eval (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_shifted_l167_16726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_violet_marbles_l167_16750

/-- The number of violet marbles Dan had initially -/
def initial_violet : ℕ := 64

/-- The number of red marbles Mary gave Dan -/
def red_marbles : ℕ := 14

/-- The total number of marbles Dan has now -/
def total_marbles : ℕ := 78

/-- Theorem stating that the number of violet marbles Dan had initially is 64 -/
theorem dans_violet_marbles : initial_violet = total_marbles - red_marbles := by
  rfl

#eval initial_violet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_violet_marbles_l167_16750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dispatch_plans_equal_48_l167_16772

/-- Represents the number of volunteers --/
def num_volunteers : ℕ := 5

/-- Represents the number of jobs --/
def num_jobs : ℕ := 4

/-- Represents the number of volunteers who can only do translation and tour guiding --/
def num_limited_volunteers : ℕ := 2

/-- Represents the number of volunteers who can do all jobs --/
def num_flexible_volunteers : ℕ := 3

/-- Represents the number of jobs that limited volunteers can do --/
def num_limited_jobs : ℕ := 2

/-- Calculates the total number of different dispatch plans --/
def total_dispatch_plans : ℕ :=
  (Nat.choose num_limited_jobs 1) * (Nat.factorial num_flexible_volunteers) +
  (Nat.choose num_flexible_volunteers 1) * (Nat.choose num_limited_jobs 1) * (Nat.factorial 2) * (Nat.factorial 2) +
  (Nat.factorial 2) * (Nat.choose num_flexible_volunteers 2) * (Nat.factorial 2)

/-- Theorem stating that the total number of dispatch plans is 48 --/
theorem dispatch_plans_equal_48 : total_dispatch_plans = 48 := by
  rw [total_dispatch_plans]
  norm_num
  rfl

#eval total_dispatch_plans

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dispatch_plans_equal_48_l167_16772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l167_16704

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^2 * Real.sin (2*x) + (a-2) * Real.cos (2*x)

/-- The theorem stating the maximum value of f(x) -/
theorem max_value_of_f (a : ℝ) (h1 : a < 0) 
  (h2 : ∀ x : ℝ, f a x = f a (-x - π/4)) : 
  ∃ M : ℝ, M = 4 * Real.sqrt 2 ∧ ∀ x : ℝ, f a x ≤ M := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l167_16704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_travel_theorem_l167_16706

/-- The speed required for Mr. Bird to arrive at work exactly on time -/
noncomputable def exact_speed : ℝ := 48

/-- The distance from Mr. Bird's home to his workplace -/
noncomputable def distance : ℝ := 12

/-- The time it takes to arrive exactly on time (in hours) -/
noncomputable def exact_time : ℝ := 1/4

/-- The speed at which Mr. Bird arrives 3 minutes late -/
noncomputable def late_speed : ℝ := 40

/-- The speed at which Mr. Bird arrives 3 minutes early -/
noncomputable def early_speed : ℝ := 60

/-- The time difference (in hours) when arriving late or early -/
noncomputable def time_diff : ℝ := 3/60

/-- Theorem stating the relationships between distance, time, and speeds -/
theorem bird_travel_theorem :
  (distance = late_speed * (exact_time + time_diff)) ∧
  (distance = early_speed * (exact_time - time_diff)) ∧
  (exact_speed * exact_time = distance) := by
  sorry

#check bird_travel_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_travel_theorem_l167_16706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_heads_probability_l167_16767

/-- Represents a coin with a given probability of heads -/
structure Coin where
  prob_heads : ℚ
  prob_heads_nonneg : 0 ≤ prob_heads
  prob_heads_le_one : prob_heads ≤ 1

/-- A fair coin has a 1/2 probability of heads -/
def fair_coin : Coin where
  prob_heads := 1/2
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- A biased coin with 2/5 probability of heads -/
def biased_coin : Coin where
  prob_heads := 2/5
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- The set of coins each person flips -/
def coin_set : List Coin := [fair_coin, fair_coin, biased_coin]

/-- The probability of getting the same number of heads -/
def prob_same_heads : ℚ := 63/200

theorem same_heads_probability :
  prob_same_heads = 63/200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_heads_probability_l167_16767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_theorem_l167_16766

noncomputable def initial_height : ℝ := 500
noncomputable def bounce_ratio : ℝ := 2/3
noncomputable def target_height : ℝ := 2

noncomputable def height_after_bounces (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

theorem ball_bounce_theorem :
  ∃ k : ℕ, (∀ n < k, height_after_bounces n ≥ target_height) ∧
           (height_after_bounces k < target_height) ∧
           (k = 14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_theorem_l167_16766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l167_16769

theorem diophantine_equation_solutions :
  ∀ a b c x : ℕ+,
  a + b + c = x * a * b * c →
  ((a = 1 ∧ b = 1 ∧ c = 2 ∧ x = 2) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ x = 3) ∨
   (a = 1 ∧ b = 2 ∧ c = 3 ∧ x = 1)) ∨
  (∃ (p : Equiv.Perm (Fin 3)), 
    let v := [a, b, c]
    (v[p 0]! = 1 ∧ v[p 1]! = 1 ∧ v[p 2]! = 2 ∧ x = 2) ∨
    (v[p 0]! = 1 ∧ v[p 1]! = 2 ∧ v[p 2]! = 3 ∧ x = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l167_16769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_approx_l167_16700

-- Define the movements
def south_distance : ℚ := 50
def west_distance : ℚ := 70
def north_distance : ℚ := 30
def east_distance : ℚ := 40

-- Calculate net distances
def net_south : ℚ := south_distance - north_distance
def net_west : ℚ := west_distance - east_distance

-- Define the distance between A and C
noncomputable def distance_AC : ℝ := Real.sqrt ((net_south : ℝ) ^ 2 + (net_west : ℝ) ^ 2)

-- Theorem statement
theorem distance_AC_approx :
  ∃ ε > 0, |distance_AC - 36.06| < ε := by
  sorry

#eval net_south
#eval net_west

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_approx_l167_16700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l167_16745

/-- Circle C₁ with center (0, 0) and radius 2 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Circle C₂ with center (1, 3) and radius 2 -/
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 3)^2 = 4}

/-- The function to be minimized -/
def f (p : ℝ × ℝ) : ℝ := p.1^2 + p.2^2 - 6*p.1 - 4*p.2 + 13

/-- Condition that P(a, b) forms equal length tangents to C₁ and C₂ -/
def equalTangents (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℝ × ℝ), m ∈ C₁ ∧ n ∈ C₂ ∧
    (p.1 - m.1)^2 + (p.2 - m.2)^2 = (p.1 - n.1)^2 + (p.2 - n.2)^2

theorem min_value_of_f :
  ∃ (p : ℝ × ℝ), equalTangents p ∧
    (∀ (q : ℝ × ℝ), equalTangents q → f q ≥ f p) ∧
    f p = 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l167_16745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plain_cookies_sold_l167_16764

/-- Represents the number of boxes of chocolate chip cookies sold -/
def C : ℕ := sorry

/-- Represents the number of boxes of plain cookies sold -/
def P : ℕ := sorry

/-- The price of a box of chocolate chip cookies in cents -/
def chocolate_price : ℕ := 125

/-- The price of a box of plain cookies in cents -/
def plain_price : ℕ := 75

/-- The total number of boxes sold -/
def total_boxes : ℕ := 1585

/-- The total value of sales in cents -/
def total_value : ℕ := 158675

theorem plain_cookies_sold :
  C + P = total_boxes ∧
  chocolate_price * C + plain_price * P = total_value →
  P = 789 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plain_cookies_sold_l167_16764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l167_16711

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (x - 1)

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) + 1

-- Theorem statement
theorem inverse_function_proof :
  (∀ x > 1, f x ∈ Set.univ) ∧
  (∀ y ∈ Set.univ, g y > 1) ∧
  (∀ x > 1, g (f x) = x) ∧
  (∀ y ∈ Set.univ, f (g y) = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l167_16711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_time_opposite_approx_11_01_l167_16778

-- Define the speeds of the trains in km/h
noncomputable def speed1 : ℝ := 60
noncomputable def speed2 : ℝ := 40

-- Define the time to cross when running in the same direction in seconds
noncomputable def time_same_direction : ℝ := 55

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 5 / 18

-- Define the function to calculate the time to cross in opposite directions
noncomputable def time_opposite_directions (s1 s2 t_same : ℝ) : ℝ :=
  let relative_speed_same := (s1 - s2) * kmh_to_ms
  let train_length := relative_speed_same * t_same / 2
  let relative_speed_opposite := (s1 + s2) * kmh_to_ms
  2 * train_length / relative_speed_opposite

-- State the theorem
theorem cross_time_opposite_approx_11_01 :
  ∃ ε > 0, |time_opposite_directions speed1 speed2 time_same_direction - 11.01| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_time_opposite_approx_11_01_l167_16778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l167_16791

theorem smallest_number : 
  let a : ℝ := -3
  let b : ℝ := 3⁻¹
  let c : ℝ := -|(-(1/3))|
  let d : ℝ := 0
  min a (min b (min c d)) = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l167_16791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_rounded_l167_16735

def berry_price : ℚ := 3.66
def apple_price : ℚ := 1.89
def peach_price : ℚ := 2.45

def berry_quantity : ℚ := 3
def apple_quantity : ℚ := 6.5
def peach_quantity : ℚ := 4

def total_spent : ℚ := 
  berry_price * berry_quantity + 
  apple_price * apple_quantity + 
  peach_price * peach_quantity

def round_to_cents (x : ℚ) : ℚ := 
  (x * 100).floor / 100

theorem total_spent_rounded : 
  round_to_cents total_spent = 33.07 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_rounded_l167_16735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l167_16798

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 4]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2)

-- Theorem statement
theorem vector_properties :
  (dot_product a b = 5) ∧
  (dot_product a b / (magnitude a * magnitude b) = Real.sqrt 5 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l167_16798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_for_quadrants_not_sufficient_condition_for_quadrants_l167_16730

noncomputable section

-- Define the line equation
def line_equation (m n x : ℝ) : ℝ := (m / n) * x - (1 / n)

-- Define the condition for passing through first, second, and fourth quadrants
def passes_through_quadrants (m n : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₄ y₄ : ℝ),
    x₁ > 0 ∧ y₁ > 0 ∧ line_equation m n x₁ = y₁ ∧
    x₂ < 0 ∧ y₂ > 0 ∧ line_equation m n x₂ = y₂ ∧
    x₄ > 0 ∧ y₄ < 0 ∧ line_equation m n x₄ = y₄

-- The theorem to prove
theorem necessary_condition_for_quadrants (m n : ℝ) :
  passes_through_quadrants m n → m * n < 0 :=
by sorry

-- The theorem to prove that the condition is not sufficient
theorem not_sufficient_condition_for_quadrants :
  ∃ (m n : ℝ), m * n < 0 ∧ ¬(passes_through_quadrants m n) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_for_quadrants_not_sufficient_condition_for_quadrants_l167_16730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l167_16789

/-- Triangle ABC with interior angles A, B, C opposite sides a, b, c respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := t.a^2 / (3 * Real.sin t.A)

/-- The perimeter of the triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem about sin B sin C and perimeter of triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h_area : area t = t.a^2 / (3 * Real.sin t.A))
  (h_cos : 6 * Real.cos t.B * Real.cos t.C = 1)
  (h_a : t.a = 3) :
  Real.sin t.B * Real.sin t.C = 2/3 ∧ 
  perimeter t = 3 + Real.sqrt 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l167_16789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_and_sum_l167_16702

-- Define the sequences a_n and b_n
def a : ℕ → ℝ := sorry

def b : ℕ → ℝ := sorry

-- Define the sum of the first n terms of b_n
def S : ℕ → ℝ := sorry

-- Axioms based on the given conditions
axiom a_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
axiom a_1 : a 1 = 3
axiom a_4 : a 4 = 12
axiom b_1 : b 1 = 4
axiom b_4 : b 4 = 20
axiom b_minus_a_geometric : ∀ n : ℕ, (b (n + 2) - a (n + 2)) / (b (n + 1) - a (n + 1)) = (b (n + 1) - a (n + 1)) / (b n - a n)

-- Theorem to prove
theorem sequences_and_sum :
  (∀ n : ℕ, n ≥ 1 → a n = 3 * n) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2^(n-1) + 3 * n) ∧
  (∀ n : ℕ, n ≥ 1 → S n = 2^n - 1 + (3 * n^2 + 3 * n) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_and_sum_l167_16702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l167_16768

noncomputable def ω : ℝ := 1/4

noncomputable def f (x : ℝ) : ℝ := Real.sin (ω * x) * (Real.sin (ω * x) + Real.cos (ω * x)) - 1/2

noncomputable def symmetry_distance : ℝ := 2 * Real.pi

noncomputable def α : ℝ := Real.pi / 3
noncomputable def β : ℝ := Real.pi / 6

theorem function_properties :
  ω > 0 ∧
  symmetry_distance = 2 * Real.pi ∧
  α + 2 * β = 2 * Real.pi / 3 ∧
  0 < α ∧ α < Real.pi / 2 ∧
  0 < β ∧ β < Real.pi / 2 ∧
  ω = 1 / 4 ∧
  (∀ x, f x = Real.sqrt 2 / 2 * Real.sin (x / 2 - Real.pi / 4)) ∧
  f (α + Real.pi / 2) * f (2 * β + 3 * Real.pi / 2) = Real.sqrt 3 / 8 ∧
  α = Real.pi / 3 ∧
  β = Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l167_16768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l167_16780

/-- The expansion of (√x + 1/(2·4x))^8 -/
noncomputable def expansion (x : ℝ) := (Real.sqrt x + 1 / (2 * 4 * x)) ^ 8

/-- The r-th term in the expansion -/
noncomputable def term (r : ℕ) (x : ℝ) : ℝ := 
  (1/2)^r * (Nat.choose 8 r) * x^((16 - 3*r) / 4)

/-- Proposition about the rational terms and largest coefficient terms in the expansion -/
theorem expansion_properties :
  ∀ x > 0,
  -- The rational terms
  {r : ℕ | ∃ q : ℚ, term r x = ↑q} = {0, 4, 8} ∧
  term 0 x = x^4 ∧
  term 4 x = 35/8 * x ∧
  term 8 x = 1/(256 * x^2) ∧
  -- The terms with largest coefficient
  (∀ r, 0 ≤ r ∧ r ≤ 8 → term 2 x ≥ term r x) ∧
  (∀ r, 0 ≤ r ∧ r ≤ 8 → term 3 x ≥ term r x) ∧
  term 2 x = 7 * x^(5/2) ∧
  term 3 x = 7 * x^(7/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l167_16780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lieutenant_age_is_24_l167_16740

/-- Represents the number of rows in the initial formation -/
def initial_rows : ℕ → Prop := sorry

/-- Represents the age of the lieutenant -/
def lieutenant_age : ℕ → Prop := sorry

/-- The number of soldiers in each row of the initial formation -/
def initial_soldiers_per_row (n : ℕ) : ℕ := n + 5

/-- The number of soldiers in each row of the second formation -/
def second_soldiers_per_row (n : ℕ) : ℕ := n + 9

/-- The total number of soldiers in the formation -/
def total_soldiers (n : ℕ) : ℕ := n * (n + 5)

theorem lieutenant_age_is_24 :
  ∀ n : ℕ, initial_rows n →
    (∃ x : ℕ, lieutenant_age x ∧
      n * (n + 5) = x * (n + 9) ∧
      x = 24) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lieutenant_age_is_24_l167_16740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_5_l167_16737

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := 2 * i - 5 / (2 - i)

theorem abs_z_equals_sqrt_5 : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_5_l167_16737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_identity_l167_16728

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a * x) / (3 * x + 4)

theorem f_composition_identity (a : ℝ) :
  (∀ x : ℝ, x ≠ -4/3 → f a (f a x) = x) ↔ a = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_identity_l167_16728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_b_eq_one_f_monotone_increasing_l167_16763

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x) / Real.log 2

-- Theorem 1: f is an odd function iff b = 1
theorem f_odd_iff_b_eq_one (b : ℝ) :
  (∀ x, f b x = -f b (-x)) ↔ b = 1 := by sorry

-- Theorem 2: f is monotonically increasing for any real b
theorem f_monotone_increasing (b : ℝ) :
  Monotone (f b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_b_eq_one_f_monotone_increasing_l167_16763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l167_16776

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := -1/3 * x^2 + 2*x - 4

-- State the theorem
theorem quadratic_function_proof :
  (∀ x, f x = -1/3 * x^2 + 2*x - 4) ∧
  f 3 = -1 ∧
  f 0 = -4 := by
  constructor
  · intro x
    rfl
  · constructor
    · sorry
    · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l167_16776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_area_l167_16736

-- Define the function f(x)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x - b / x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 7 * x - 4 * y - 12 = 0

-- Theorem statement
theorem function_and_triangle_area 
  (a b : ℝ) -- Parameters of the function
  (h1 : tangent_line 2 (f a b 2)) -- Condition that (2, f(2)) satisfies the tangent line equation
  : 
  (∀ x, f a b x = x - 3 / x) ∧ -- Part 1: Analytical expression of f(x)
  (∀ x₀, x₀ > 0 → -- Part 2: Constant area of the triangle
    let y₀ := f a b x₀
    let m := (f a b x₀ - f a b 2) / (x₀ - 2) -- Slope of the tangent line
    let x_int := 2 * x₀ -- x-intercept of the tangent line with y = x
    let y_int := y₀ - m * x₀ -- y-intercept of the tangent line
    1/2 * x_int * y_int = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_area_l167_16736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l167_16716

/-- Represents the contents of the urn -/
structure UrnContents where
  red : ℕ
  blue : ℕ

/-- Represents the possible actions: drawing a red or blue ball -/
inductive BallAction
  | Red
  | Blue

/-- Performs one iteration of George's operation -/
def performOperation (urn : UrnContents) (action : BallAction) : UrnContents :=
  match action with
  | BallAction.Red => UrnContents.mk (urn.red + 2) urn.blue
  | BallAction.Blue => UrnContents.mk urn.red (urn.blue + 2)

/-- Calculates the probability of a specific sequence of actions -/
noncomputable def sequenceProbability (actions : List BallAction) : ℚ :=
  sorry

/-- Calculates the total probability of all valid sequences -/
noncomputable def totalProbability (validSequences : List (List BallAction)) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability : 
  let initialUrn : UrnContents := UrnContents.mk 2 1
  let finalUrnTotal : ℕ := 12
  let desiredRed : ℕ := 6
  let desiredBlue : ℕ := 6
  let validSequences : List (List BallAction) := sorry
  totalProbability validSequences = 16 / 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l167_16716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_to_cos_double_angle_l167_16731

theorem tan_to_cos_double_angle (α : ℝ) :
  Real.tan (α - π/4) = 2 → Real.cos (2*α) = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_to_cos_double_angle_l167_16731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l167_16754

open Real

-- Define the function f
noncomputable def f (x : ℝ) := 4 * cos x * sin (x + π / 6) - 1

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The maximum value on [-π/6, π/4] is 2
  (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 → f x ≤ 2) ∧
  (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = 2) ∧
  -- The minimum value on [-π/6, π/4] is -1
  (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 → f x ≥ -1) ∧
  (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l167_16754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l167_16799

theorem marble_distribution (n : Nat) (hn : n = 720) :
  (Finset.filter (fun d => d > 1 ∧ d < n ∧ n % d = 0) (Finset.range (n + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l167_16799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_w_l167_16779

theorem smallest_positive_w (y w : ℝ) : 
  Real.sin y = 0 → 
  Real.cos (y + w) = -1/2 → 
  (∀ w' : ℝ, w' > 0 → Real.cos (y + w') = -1/2 → w ≤ w') → 
  w = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_w_l167_16779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_inverse_proportion_range_l167_16790

-- Define the function f(x) = a/x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x

-- State the theorem
theorem increasing_inverse_proportion_range (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x < y → f a x < f a y) → a < 0 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_inverse_proportion_range_l167_16790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_width_calculation_l167_16741

/-- The surface area of a rectangular prism -/
noncomputable def surface_area (l w h : ℝ) : ℝ := 2*l*w + 2*l*h + 2*w*h

/-- The width of a brick given its length, height, and surface area -/
noncomputable def brick_width (l h sa : ℝ) : ℝ := (sa - 2*l*h) / (2*l + 2*h)

theorem brick_width_calculation (l h sa : ℝ) (hl : l = 8) (hh : h = 2) (hsa : sa = 152) :
  brick_width l h sa = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_width_calculation_l167_16741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_center_of_symmetry_range_on_interval_l167_16757

noncomputable def f (x : ℝ) := 4 * Real.sin (x) ^ 2 + 4 * Real.sin (x) ^ 2 - 3

-- The smallest positive period is π
theorem smallest_positive_period : ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = π := by
  sorry

-- The center of symmetry is of the form (kπ/2, 1), where k ∈ ℤ
theorem center_of_symmetry : ∃ (k : ℤ), ∀ (x : ℝ), 
  f (k * π / 2 + x) = f (k * π / 2 - x) ∧ f (k * π / 2) = 1 := by
  sorry

-- The range of f(x) on the interval [-π/4, π/4] is [3, 5]
theorem range_on_interval : Set.Icc 3 5 = {y | ∃ x ∈ Set.Icc (-π/4) (π/4), f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_center_of_symmetry_range_on_interval_l167_16757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l167_16746

theorem power_equation (y : ℝ) (h : (8 : ℝ)^y - (8 : ℝ)^(y-1) = 56) : (3*y)^y = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l167_16746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_sum_l167_16749

-- Define the circles and their properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the problem parameters
def C₁ : Set (ℝ × ℝ) := sorry
def C₂ : Set (ℝ × ℝ) := sorry
def intersection_point : ℝ × ℝ := (7, 4)
def radii_product : ℝ := 85

-- Define the tangent lines
def x_axis_tangent (C : Set (ℝ × ℝ)) : Prop := sorry
def y_eq_nx_tangent (C : Set (ℝ × ℝ)) (n : ℝ) : Prop := sorry

-- Define the properties of n
noncomputable def n_expression (p q r : ℕ) : ℝ := (p : ℝ) * Real.sqrt q / r

-- Theorem statement
theorem circles_intersection_sum (p q r : ℕ) :
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  intersection_point ∈ C₁ ∧ intersection_point ∈ C₂ ∧
  (∃ (r₁ r₂ : ℝ), r₁ * r₂ = radii_product) ∧
  x_axis_tangent C₁ ∧ x_axis_tangent C₂ ∧
  (∃ (n : ℝ), n > 0 ∧ y_eq_nx_tangent C₁ n ∧ y_eq_nx_tangent C₂ n ∧ n = n_expression p q r) ∧
  (∀ (prime : ℕ), Nat.Prime prime → ¬(prime^2 ∣ q)) ∧
  Nat.Coprime p r →
  p + q + r = 272 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_sum_l167_16749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l167_16747

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the geometric progression property
def angles_in_geometric_progression (t : Triangle) : Prop :=
  ∃ q : ℝ, q > 0 ∧ t.A * q = t.B ∧ t.B * q = t.C

-- Define the given equation
def satisfies_equation (t : Triangle) : Prop :=
  t.b^2 - t.a^2 = t.a * t.c

-- Main theorem
theorem angle_B_measure (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : angles_in_geometric_progression t)
  (h3 : satisfies_equation t) :
  t.B = 2 * Real.pi / 7 := by
  sorry

#check angle_B_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l167_16747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossiblePartition_l167_16722

/-- Represents a piece in the square cutting problem -/
structure Piece where
  area : ℕ

/-- Represents a partition of the square -/
structure Partition where
  pieces : List Piece
  valid : (pieces.map (λ p => p.area)).sum = 14400

/-- The difference in area between any two pieces is divisible by 3 -/
def validPieceDifference (partition : Partition) : Prop :=
  ∀ p1 p2, p1 ∈ partition.pieces → p2 ∈ partition.pieces → (p1.area - p2.area) % 3 = 0

theorem impossiblePartition (n : ℕ) :
  ¬∃ (partition : Partition), partition.pieces.length = n + 5 ∧ validPieceDifference partition :=
by
  sorry

#check impossiblePartition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossiblePartition_l167_16722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_parallelism_l167_16703

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define points A, B, C, X, and Y
variable (A B C X Y : ℝ × ℝ)

-- Define that AB is a diameter of the circle
def is_diameter (circle : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ circle ∧ B ∈ circle ∧ ∀ P ∈ circle, dist A P + dist P B = dist A B

-- Define that C is on the diameter AB
def on_diameter (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B

-- Define that X and Y are on the circle
def on_circle (circle : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  P ∈ circle

-- Define symmetry of X and Y with respect to AB
def symmetric_wrt_line (X Y A B : ℝ × ℝ) : Prop :=
  let midpoint := (A + B) / 2
  dist X midpoint = dist Y midpoint ∧ 
  (X - midpoint) • (B - A) = (Y - midpoint) • (B - A)

-- Define perpendicularity of YC to XA
def perpendicular (L1 L2 : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let (x1, y1, x2, y2) := L1
  let (x3, y3, x4, y4) := L2
  (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3) = 0

-- Define parallelism of XB to YC
def parallel (L1 L2 : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let (x1, y1, x2, y2) := L1
  let (x3, y3, x4, y4) := L2
  (x2 - x1) * (y4 - y3) = (y2 - y1) * (x4 - x3)

-- State the theorem
theorem circle_symmetry_parallelism 
  (h1 : is_diameter circle A B)
  (h2 : on_diameter A B C)
  (h3 : on_circle circle X)
  (h4 : on_circle circle Y)
  (h5 : symmetric_wrt_line X Y A B)
  (h6 : perpendicular (Y.1, Y.2, C.1, C.2) (X.1, X.2, A.1, A.2)) :
  parallel (X.1, X.2, B.1, B.2) (Y.1, Y.2, C.1, C.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_parallelism_l167_16703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_addition_l167_16720

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Converts a Digit12 to its decimal value --/
def digit12ToNat (d : Digit12) : Nat :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11

/-- Represents a number in base 12 --/
def Base12 := List Digit12

/-- Converts a Base12 number to its decimal value --/
def toDecimal (n : Base12) : Nat :=
  n.foldr (fun d acc => digit12ToNat d + 12 * acc) 0

/-- The first number: A85 in base 12 --/
def num1 : Base12 := [Digit12.A, Digit12.D8, Digit12.D5]

/-- The second number: 2B4 in base 12 --/
def num2 : Base12 := [Digit12.D2, Digit12.B, Digit12.D4]

/-- The expected sum: 1179 in base 12 --/
def expectedSum : Base12 := [Digit12.D1, Digit12.D1, Digit12.D7, Digit12.D9]

theorem base12_addition : toDecimal num1 + toDecimal num2 = toDecimal expectedSum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_addition_l167_16720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_when_dot_product_is_one_f_range_in_triangle_l167_16788

-- Define the vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 4), 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 4), (Real.cos (x / 4))^2)

-- Define the dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := dot_product (m x) (n x)

-- Theorem 1
theorem cos_value_when_dot_product_is_one (x : ℝ) :
  dot_product (m x) (n x) = 1 → Real.cos (2 * Real.pi / 3 - x) = -1 / 2 := by sorry

-- Theorem 2
theorem f_range_in_triangle (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < 2 * Real.pi / 3) →
  (0 < B ∧ B < Real.pi) →
  (0 < C ∧ C < Real.pi) →
  (A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  ((2 * a - c) * Real.cos B = b * Real.cos C) →
  (∃ (y : ℝ), 1 < y ∧ y < 3 / 2 ∧ f A = y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_when_dot_product_is_one_f_range_in_triangle_l167_16788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_rays_acute_angle_l167_16710

-- Define a ray in ℝ³
def Ray : Type := ℝ × ℝ × ℝ

-- Define the angle between two rays
noncomputable def angle (r1 r2 : Ray) : ℝ := sorry

-- Define an acute angle
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Statement of the theorem
theorem least_rays_acute_angle :
  ∀ (n : ℕ), n ≥ 7 →
  ∀ (rays : Fin n → Ray),
  ∃ (i j : Fin n), i ≠ j ∧ is_acute (angle (rays i) (rays j)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_rays_acute_angle_l167_16710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_l167_16715

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem three_tangent_lines : 
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x₀ ∈ s, x₀ * Real.exp x₀ * (x₀^2 - x₀ - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_l167_16715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_xlny_l167_16701

/-- Given that x and y satisfy the equation log_x(y) + log_y(x) = 5/2 and log_x(y) > 1,
    the minimum value of x * ln(y) is -2/e -/
theorem min_value_xlny (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x ≠ 1) (h4 : y ≠ 1)
  (h5 : (Real.log y) / (Real.log x) + (Real.log x) / (Real.log y) = 5/2) 
  (h6 : (Real.log y) / (Real.log x) > 1) :
  ∃ (min_val : ℝ), min_val = -2/Real.exp 1 ∧
    ∀ (z w : ℝ), z > 0 → w > 0 → z ≠ 1 → w ≠ 1 →
      (Real.log w) / (Real.log z) + (Real.log z) / (Real.log w) = 5/2 → 
      (Real.log w) / (Real.log z) > 1 →
      z * Real.log w ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_xlny_l167_16701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_abc_fraction_l167_16773

theorem max_value_abc_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ (1 : ℝ) / 4 ∧
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_abc_fraction_l167_16773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l167_16713

theorem negation_of_proposition :
  (∃ x : ℝ, x ≤ 0 ∧ (x + 1) * Real.exp x ≤ 1) ↔ 
  ¬(∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l167_16713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_still_water_speed_l167_16771

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  upstream : ℚ
  downstream : ℚ

/-- Calculates the speed of a rower in still water -/
def stillWaterSpeed (s : RowerSpeed) : ℚ :=
  (s.upstream + s.downstream) / 2

/-- Theorem: Given upstream speed of 7 kmph and downstream speed of 33 kmph, 
    the speed in still water is 20 kmph -/
theorem rower_still_water_speed :
  let s : RowerSpeed := { upstream := 7, downstream := 33 }
  stillWaterSpeed s = 20 := by
  -- Unfold the definition of stillWaterSpeed
  unfold stillWaterSpeed
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_still_water_speed_l167_16771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_condition_l167_16748

theorem two_zeros_condition (ω : ℝ) : ω > 0 →
  (∃! (z₁ z₂ : ℝ), 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧
    Real.sin (ω * z₁) = Real.cos (ω * z₁) ∧ Real.sin (ω * z₂) = Real.cos (ω * z₂) ∧
    ∀ x, 0 < x ∧ x < π → (Real.sin (ω * x) = Real.cos (ω * x) → x = z₁ ∨ x = z₂)) ↔
  5/4 < ω ∧ ω ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_condition_l167_16748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l167_16732

/-- A convex hexagon with exactly two distinct side lengths -/
structure ConvexHexagon where
  side_length1 : ℝ
  side_length2 : ℝ
  side_count1 : ℕ
  side_count2 : ℕ
  distinct_lengths : side_length1 ≠ side_length2
  total_sides : side_count1 + side_count2 = 6
  perimeter : side_length1 * (side_count1 : ℝ) + side_length2 * (side_count2 : ℝ) = 30

/-- Theorem about a specific convex hexagon -/
theorem hexagon_side_count (h : ConvexHexagon) 
  (h_side1 : h.side_length1 = 7 ∨ h.side_length2 = 7)
  (h_side2 : h.side_length1 = 4 ∨ h.side_length2 = 4) :
  (h.side_length1 = 4 ∧ h.side_count1 = 4) ∨ (h.side_length2 = 4 ∧ h.side_count2 = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l167_16732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_arithmetic_sequence_sum_l167_16729

/-- Geometric sequence with common ratio q -/
noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ := q^(n-1)

/-- Sum of first n terms of geometric sequence -/
noncomputable def geometric_sum (q : ℝ) (n : ℕ) : ℝ := (1 - q^n) / (1 - q)

/-- Sequence b_n -/
noncomputable def b_sequence (n : ℕ) : ℝ := (2*n - 1) - 2^(n-1)

/-- Sum of first n terms of b_sequence -/
noncomputable def b_sum (n : ℕ) : ℝ := n^2 - 2^n + 1

theorem geometric_and_arithmetic_sequence_sum :
  ∀ q : ℝ,
  q > 0 →
  geometric_sequence q 1 = 1 →
  geometric_sum q 3 = 7 →
  b_sequence 1 = 0 →
  b_sequence 3 = 1 →
  (∀ n : ℕ, ∃ d : ℝ, geometric_sequence q n + b_sequence n = d * n + (geometric_sequence q 1 + b_sequence 1)) →
  ∀ n : ℕ, b_sum n = n^2 - 2^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_arithmetic_sequence_sum_l167_16729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_necessary_not_sufficient_for_differentiability_l167_16774

open Function Real

/-- A function f is continuous at x₀ -/
def IsContinuousAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ContinuousAt f x₀

/-- A function f is differentiable at x₀ -/
def IsDifferentiableAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀

/-- Continuity is necessary but not sufficient for differentiability -/
theorem continuity_necessary_not_sufficient_for_differentiability :
  ∃ f : ℝ → ℝ, ∃ x₀ : ℝ,
    (IsContinuousAt f x₀ ∧ ¬IsDifferentiableAt f x₀) ∧
    (∀ g : ℝ → ℝ, IsDifferentiableAt g x₀ → IsContinuousAt g x₀) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_necessary_not_sufficient_for_differentiability_l167_16774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_l167_16777

/-- The volume of a regular quadrilateral pyramid -/
noncomputable def pyramid_volume (h Q : ℝ) : ℝ :=
  (2/3) * h * (Real.sqrt (h^2 + 4*Q^2) - h^2)

/-- Theorem: The volume of a regular quadrilateral pyramid with height h 
    and lateral face area Q is (2/3) * h * (√(h² + 4Q²) - h²) -/
theorem regular_quadrilateral_pyramid_volume 
  (h Q : ℝ) (h_pos : h > 0) (Q_pos : Q > 0) : 
  ∃ V : ℝ, V = pyramid_volume h Q ∧ 
  V = (2/3) * h * (Real.sqrt (h^2 + 4*Q^2) - h^2) := by
  use pyramid_volume h Q
  constructor
  · rfl
  · rfl

#check regular_quadrilateral_pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_l167_16777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_volume_l167_16797

-- Define the edge length of the cube
noncomputable def cube_edge_length : ℝ := Real.sqrt 2

-- Define the convex polyhedron formed by the centers of the cube's faces
def convex_polyhedron (edge_length : ℝ) : Set (Fin 3 → ℝ) :=
  sorry

-- Define the volume function for the convex polyhedron
def volume (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem convex_polyhedron_volume :
  volume (convex_polyhedron cube_edge_length) = Real.sqrt 2 / 3 :=
by
  sorry

#check convex_polyhedron_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_volume_l167_16797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l167_16742

/-- Define the nested radical function recursively -/
noncomputable def nestedRadical : ℕ → ℝ
  | 0 => Real.sqrt (1 + 2018 * 2020)
  | n + 1 => Real.sqrt (1 + (2017 - n) * nestedRadical n)

/-- The main theorem stating that the nested radical equals 3 -/
theorem nested_radical_equals_three : nestedRadical 2017 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l167_16742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018_l167_16762

def our_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a n + a (n + 1) + a (n + 2) = 15) ∧
  a 4 = 1 ∧
  a 12 = 5

theorem sequence_2018 (a : ℕ → ℤ) (h : our_sequence a) : a 2018 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018_l167_16762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l167_16783

noncomputable def expansion (x : ℝ) := (2 * x - x^(-(1/3 : ℝ)))^12

theorem constant_term_of_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → expansion x = c + x * (expansion x - c) ∧ c = -1760 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l167_16783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l167_16739

open Real

/-- The surface area of a cone with base radius r and height h -/
noncomputable def coneSurfaceArea (r h : ℝ) : ℝ :=
  π * r * r + π * r * Real.sqrt (r^2 + h^2)

/-- Theorem: The surface area of a cone with base radius 4 cm and height 2√5 cm is 40π cm² -/
theorem cone_surface_area_specific : coneSurfaceArea 4 (2 * Real.sqrt 5) = 40 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l167_16739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_relation_l167_16794

/-- Represents a geometric sequence with a common ratio -/
structure GeometricSequence where
  b : ℕ+ → ℝ  -- The sequence terms indexed by positive natural numbers
  q : ℝ        -- The common ratio
  h : ∀ n : ℕ+, b (n + 1) = q * b n  -- The defining property of a geometric sequence

/-- 
Theorem: In a geometric sequence, the relation between any two terms b_n and b_m 
is given by b_n = b_m · q^(n-m), where m and n are positive integers.
-/
theorem geometric_sequence_relation (seq : GeometricSequence) (m n : ℕ+) :
  seq.b n = seq.b m * (seq.q : ℝ) ^ (n.val - m.val) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_relation_l167_16794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_sum_of_distances_l167_16770

-- Helper definitions (not to be proved, just for context)
noncomputable def is_tangent_point (P : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop := sorry
noncomputable def is_circumcenter (O A B C : ℝ × ℝ) : Prop := sorry
noncomputable def distance (P : ℝ × ℝ) (L : Set (ℝ × ℝ)) : ℝ := sorry
noncomputable def line_of (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem tangent_circles_sum_of_distances (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 5) (h₃ : r₃ = 7) :
  ∃ A B C O : ℝ × ℝ, 
    (is_tangent_point A r₁ r₂) ∧ 
    (is_tangent_point B r₁ r₃) ∧ 
    (is_tangent_point C r₂ r₃) ∧
    (is_circumcenter O A B C) →
    (distance O (line_of A B) + distance O (line_of B C) + distance O (line_of C A) = 
     7/4 + 7/(3 * Real.sqrt 6) + 7/Real.sqrt 30) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_sum_of_distances_l167_16770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_sequence_is_arithmetic_l167_16756

/-- Given an arithmetic sequence with common difference d, prove that the sequence
    formed by summing terms 3 apart is arithmetic with common difference 2d -/
theorem new_sequence_is_arithmetic (a : ℕ → ℝ) (d : ℝ) 
  (h : ∀ n : ℕ, a (n + 1) = a n + d) :
  let b := λ n ↦ a n + a (n + 3)
  ∀ n : ℕ, b (n + 1) - b n = 2 * d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_sequence_is_arithmetic_l167_16756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l167_16786

theorem trigonometric_identity : 
  (Real.sin (92 * π / 180) - Real.sin (32 * π / 180) * Real.cos (60 * π / 180)) / Real.cos (32 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l167_16786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_steps_to_meet_hannah_l167_16719

theorem lucas_steps_to_meet_hannah (lucas_speed : ℝ) (hannah_speed : ℝ) 
  (distance_miles : ℝ) (lucas_step_length : ℝ) 
  (h1 : lucas_speed > 0)
  (h2 : hannah_speed = 3 * lucas_speed)
  (h3 : distance_miles = 3)
  (h4 : lucas_step_length = 3 / 5280) : 
  (distance_miles * 5280) / (2 * lucas_step_length) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_steps_to_meet_hannah_l167_16719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l167_16707

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem absolute_difference_of_solution (x y : ℝ) :
  (floor x : ℝ) + frac y = 3.7 →
  frac x + (floor y : ℝ) = 4.2 →
  |x - y| = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l167_16707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_gave_away_eight_bracelets_l167_16727

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  total_bracelets : ℕ
  material_cost : ℚ
  selling_price : ℚ
  profit : ℚ

/-- Calculates the number of bracelets given away -/
def bracelets_given_away (sale : BraceletSale) : ℕ :=
  sale.total_bracelets - (((sale.profit + sale.material_cost) / sale.selling_price).floor.toNat)

/-- Theorem stating that Alice gave away 8 bracelets -/
theorem alice_gave_away_eight_bracelets :
  let sale : BraceletSale := {
    total_bracelets := 52,
    material_cost := 3,
    selling_price := 1/4,
    profit := 8
  }
  bracelets_given_away sale = 8 := by
  sorry

#eval bracelets_given_away {
  total_bracelets := 52,
  material_cost := 3,
  selling_price := 1/4,
  profit := 8
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_gave_away_eight_bracelets_l167_16727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_time_is_six_l167_16712

/-- Represents the travel time of a boat in a river with current --/
noncomputable def upstream_travel_time (downstream_distance : ℝ) (downstream_time : ℝ) (current_speed : ℝ) : ℝ :=
  let boat_speed := downstream_distance / downstream_time - current_speed
  downstream_distance / (boat_speed - current_speed)

/-- Theorem stating that under given conditions, upstream travel time is 6 hours --/
theorem upstream_time_is_six :
  upstream_travel_time 24 4 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_time_is_six_l167_16712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_of_square_l167_16758

theorem simplify_sqrt_of_square : Real.sqrt ((1 - Real.sqrt 3)^2) = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_of_square_l167_16758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_modulus_vector_l167_16775

noncomputable def vector_modulus (v : ℝ × ℝ) : ℝ :=
  Real.sqrt ((v.1)^2 + (v.2)^2)

def arithmetic_vector_sequence (a : ℕ → ℝ × ℝ) : Prop :=
  ∃ d : ℝ × ℝ, ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = d

theorem smallest_modulus_vector 
  (a : ℕ → ℝ × ℝ) 
  (h1 : arithmetic_vector_sequence a) 
  (h2 : a 1 = (-20, 13)) 
  (h3 : a 3 = (-18, 15)) : 
  (∃ n : ℕ, n = 4 ∨ n = 5) ∧ 
  (∀ m : ℕ, vector_modulus (a m) ≥ vector_modulus (a 4)) ∧
  (∀ m : ℕ, vector_modulus (a m) ≥ vector_modulus (a 5)) :=
by
  sorry

#check smallest_modulus_vector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_modulus_vector_l167_16775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log21_not_expressible_l167_16792

-- Define the given logarithms
noncomputable def log3 : ℝ := Real.log 3
noncomputable def log5 : ℝ := Real.log 5

-- Define a function that checks if a logarithm can be expressed using log3 and log5
def expressible (x : ℝ) : Prop :=
  ∃ (a b : ℚ), x = a * log3 + b * log5

-- State the theorem
theorem log21_not_expressible :
  ¬(expressible (Real.log 21)) ∧
  expressible (Real.log 28) ∧
  expressible (Real.log (36/25)) ∧
  expressible (Real.log 750) ∧
  expressible (Real.log 0.75) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log21_not_expressible_l167_16792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_is_odd_and_decreasing_l167_16760

-- Define the interval (0,1)
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x < 1}

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f y < f x

-- Define the given functions
noncomputable def f₁ : ℝ → ℝ := λ x => -1/x
def f₂ : ℝ → ℝ := λ x => x
noncomputable def f₃ : ℝ → ℝ := λ x => Real.log |x - 1| / Real.log 2
noncomputable def f₄ : ℝ → ℝ := λ x => -Real.sin x

-- State the theorem
theorem sin_is_odd_and_decreasing :
  (is_odd f₄ ∧ is_decreasing_on f₄ open_unit_interval) ∧
  ¬(is_odd f₁ ∧ is_decreasing_on f₁ open_unit_interval) ∧
  ¬(is_odd f₂ ∧ is_decreasing_on f₂ open_unit_interval) ∧
  ¬(is_odd f₃ ∧ is_decreasing_on f₃ open_unit_interval) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_is_odd_and_decreasing_l167_16760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_max_sum_l167_16724

theorem min_value_of_max_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_non_neg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 100) : 
  let M := max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))
  ∃ m, m = (100 : ℝ) / 3 ∧ ∀ M', (∀ x₁' x₂' x₃' x₄' x₅', 
    (x₁' ≥ 0 ∧ x₂' ≥ 0 ∧ x₃' ≥ 0 ∧ x₄' ≥ 0 ∧ x₅' ≥ 0) → 
    x₁' + x₂' + x₃' + x₄' + x₅' = 100 → 
    M' ≥ max (x₁' + x₂') (max (x₂' + x₃') (max (x₃' + x₄') (x₄' + x₅')))) → M' ≥ m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_max_sum_l167_16724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_two_l167_16708

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 7 else 4 * x - 1

theorem f_neg_three_eq_neg_two : f (-3) = -2 := by
  -- Unfold the definition of f
  unfold f
  -- Evaluate the if-then-else expression
  simp [if_pos]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_two_l167_16708
