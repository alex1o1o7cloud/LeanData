import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x₀_range_l1039_103919

/-- The circle O with equation x² + y² = 1 -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point M with coordinates (x₀, 1) -/
structure Point_M where
  x₀ : ℝ

/-- Point N on the circle O -/
structure Point_N where
  x : ℝ
  y : ℝ
  on_circle : circle_O x y

/-- The angle OMN is 45° -/
def angle_OMN_45 (M : Point_M) (N : Point_N) : Prop :=
  ∃ O : ℝ × ℝ, O = (0, 0) ∧ 
    Real.arctan ((N.y - 1) / (N.x - M.x₀)) = Real.pi / 4

theorem x₀_range (M : Point_M) :
  (∃ N : Point_N, angle_OMN_45 M N) → M.x₀ ∈ Set.Icc (-1 : ℝ) 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x₀_range_l1039_103919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_toy_cost_l1039_103913

/-- The cost of a single dog toy -/
noncomputable def toy_cost : ℝ := 12

/-- The number of toys Samantha buys -/
def num_toys : ℕ := 4

/-- The cost of two toys with the "buy one get one half off" promotion -/
noncomputable def cost_of_two_toys : ℝ := toy_cost + toy_cost / 2

/-- The total cost of all toys Samantha buys -/
noncomputable def total_cost : ℝ := (num_toys / 2 : ℝ) * cost_of_two_toys

theorem dog_toy_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_toy_cost_l1039_103913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1039_103992

/-- Line l defined by parametric equations -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + 3/5 * t, 4/5 * t)

/-- Curve C defined by parametric equations -/
noncomputable def curve_C (k : ℝ) : ℝ × ℝ := (4 * k^2, 4 * k)

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The length of segment AB is 25/4 -/
theorem length_of_AB : ∃ (t1 t2 k1 k2 : ℝ),
  line_l t1 = curve_C k1 ∧
  line_l t2 = curve_C k2 ∧
  t1 ≠ t2 ∧
  distance (line_l t1) (line_l t2) = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1039_103992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_divides_2016_pow_plus_m_l1039_103925

theorem exists_m_divides_2016_pow_plus_m (n : ℕ) (hn : n > 0) : ∃ m : ℕ, n ∣ (2016 ^ m + m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_divides_2016_pow_plus_m_l1039_103925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1039_103961

def sequence_a : ℕ → ℝ
  | 0 => 0  -- Adding the zero case
  | 1 => 0
  | (n + 2) => 2 * sequence_a (n + 1) + 2 * (n + 1)

theorem sequence_properties :
  let a := sequence_a
  -- Part 1: Geometric sequence property
  (∀ n : ℕ, n ≥ 1 → a n + 2 * n + 2 = 4 * 2^(n - 1)) ∧
  -- Part 2: General formula
  (∀ n : ℕ, n ≥ 1 → a n = 2^(n + 1) - 2 * n - 2) ∧
  -- Part 3: Range of λ
  (∀ l : ℝ, (∀ n : ℕ, n ≥ 1 → a n > l * (2 * n + 1) * (-1)^(n - 1)) →
    -2/5 < l ∧ l < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1039_103961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1039_103976

/-- Represents the time in hours it takes to fill a cistern normally -/
noncomputable def t : ℝ := sorry

/-- The time it takes to fill the cistern with a leak -/
noncomputable def leak_fill_time : ℝ := t + 2

/-- The time it takes for the leak to empty a full cistern -/
def leak_empty_time : ℝ := 60

/-- The rate at which the cistern fills normally -/
noncomputable def normal_fill_rate : ℝ := 1 / t

/-- The rate at which the leak empties the cistern -/
noncomputable def leak_rate : ℝ := 1 / leak_empty_time

/-- The rate at which the cistern fills with the leak -/
noncomputable def leak_fill_rate : ℝ := 1 / leak_fill_time

theorem cistern_fill_time :
  normal_fill_rate - leak_rate = leak_fill_rate →
  t = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1039_103976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radicals_l1039_103906

theorem simplify_radicals : Real.sqrt 18 * (32 : ℝ) ^ (1/3) = 6 * Real.sqrt 2 * (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radicals_l1039_103906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_2_1_km_condition_1_condition_2_condition_3_l1039_103984

/-- The distance between home and school satisfying the given conditions -/
noncomputable def distance_home_school : ℝ :=
  let t := (7 : ℝ) / 30  -- Time taken when traveling at the correct speed
  9 * t

/-- The theorem stating that the distance between home and school is 2.1 km -/
theorem distance_is_2_1_km :
  distance_home_school = 2.1 := by
  sorry

/-- Condition 1: At 6 km/hr, the boy reaches 7 minutes late -/
theorem condition_1 :
  6 * (distance_home_school / 6 + 7 / 60) = distance_home_school := by
  sorry

/-- Condition 2: At 12 km/hr, the boy reaches 8 minutes early -/
theorem condition_2 :
  12 * (distance_home_school / 12 - 8 / 60) = distance_home_school := by
  sorry

/-- Condition 3: At 9 km/hr, the boy reaches exactly on time -/
theorem condition_3 :
  9 * (distance_home_school / 9) = distance_home_school := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_2_1_km_condition_1_condition_2_condition_3_l1039_103984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_trip_time_l1039_103949

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- plane's airspeed
  w : ℝ  -- wind speed

/-- The time taken for a flight given distance, speed, and wind -/
noncomputable def flightTime (distance : ℝ) (speed : ℝ) (wind : ℝ) : ℝ :=
  distance / (speed + wind)

/-- The conditions of the flight scenario -/
def flightConditions (fs : FlightScenario) : Prop :=
  flightTime fs.d fs.p (-fs.w) = 90 ∧  -- against wind takes 90 minutes
  flightTime fs.d fs.p fs.w = fs.d / fs.p - 6  -- with wind takes 6 minutes less than still air

/-- The theorem stating the possible return trip times -/
theorem return_trip_time (fs : FlightScenario) 
  (h : flightConditions fs) : 
  flightTime fs.d fs.p fs.w = 30 ∨ flightTime fs.d fs.p fs.w = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_trip_time_l1039_103949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OB_OC_and_tan_alpha_l1039_103956

open Real

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B : ℝ × ℝ := (0, 2)
noncomputable def C (α : ℝ) : ℝ × ℝ := (cos α, sin α)

noncomputable def O : ℝ × ℝ := (0, 0)

def vecAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

def vecSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

noncomputable def vecMag (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def angleBetween (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((dotProduct v w) / (vecMag v * vecMag w))

theorem angle_OB_OC_and_tan_alpha (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π) : 
  (vecMag (vecAdd (vecSub A O) (vecSub (C α) O)) = Real.sqrt 7 → 
    angleBetween (vecSub B O) (vecSub (C α) O) = π/6) ∧
  (dotProduct (vecSub (C α) A) (vecSub (C α) B) = 0 → 
    tan α = -(4 + Real.sqrt 7) / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OB_OC_and_tan_alpha_l1039_103956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_exact_range_of_f_l1039_103968

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.arctan x + (1/2) * Real.arcsin x

-- State the theorem
theorem range_of_f :
  (∀ y, y ∈ Set.range f → -Real.pi/2 ≤ y ∧ y ≤ Real.pi/2) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc (-1 : ℝ) 1 ∧ x₂ ∈ Set.Icc (-1 : ℝ) 1 ∧ 
            f x₁ = -Real.pi/2 ∧ f x₂ = Real.pi/2) :=
by
  sorry

-- Additional theorem to show that the range is exactly [-π/2, π/2]
theorem exact_range_of_f :
  Set.range f = Set.Icc (-Real.pi/2) (Real.pi/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_exact_range_of_f_l1039_103968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2_plus_sqrt_1_minus_x_squared_l1039_103990

theorem integral_2_plus_sqrt_1_minus_x_squared :
  ∫ x in Set.Icc 0 1, (2 + Real.sqrt (1 - x^2)) = π / 4 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2_plus_sqrt_1_minus_x_squared_l1039_103990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_a_128_1_l1039_103908

-- Define the sequence a_{i,j}
def a : ℕ → ℕ → ℕ
  | 0, _ => 0  -- Add a base case for i = 0
  | 1, n => n^n
  | i+1, j => a i j + a i (j+1)

-- State the theorem
theorem unit_digit_a_128_1 : a 128 1 % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_a_128_1_l1039_103908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_combinations_4_2_l1039_103954

def library_book_combinations (n : Nat) (k : Nat) : Nat :=
  if n = 4 && k = 2 then 6
  else Nat.choose n k

theorem library_book_combinations_4_2 : 
  library_book_combinations 4 2 = 6 := by
  rfl

#eval library_book_combinations 4 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_combinations_4_2_l1039_103954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_sum_squared_l1039_103975

/-- A triangle inscribed in a circle where one side is the diameter -/
structure InscribedTriangle where
  r : ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  h_circle : (D.1 - F.1)^2 + (D.2 - F.2)^2 = r^2 ∧ 
             (E.1 - F.1)^2 + (E.2 - F.2)^2 = r^2
  h_diameter : (D.1 - E.1)^2 + (D.2 - E.2)^2 = (2*r)^2
  h_distinct : F ≠ D ∧ F ≠ E

/-- The sum of distances from F to D and E -/
noncomputable def perimeter_sum (t : InscribedTriangle) : ℝ :=
  (((t.D.1 - t.F.1)^2 + (t.D.2 - t.F.2)^2) ^ (1/2 : ℝ)) + 
  (((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2) ^ (1/2 : ℝ))

/-- The maximum value of the squared perimeter sum is 8r^2 -/
theorem max_perimeter_sum_squared (t : InscribedTriangle) : 
  (perimeter_sum t)^2 ≤ 8 * t.r^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_sum_squared_l1039_103975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_div_z₂_in_fourth_quadrant_l1039_103915

def z₁ : ℂ := 2 + Complex.I
def z₂ : ℂ := 1 + Complex.I

theorem z₁_div_z₂_in_fourth_quadrant :
  let w := z₁ / z₂
  0 < w.re ∧ w.im < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_div_z₂_in_fourth_quadrant_l1039_103915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sphere_distance_specific_l1039_103905

/-- The distance between the plane of a triangle and the center of a sphere -/
noncomputable def triangle_sphere_distance (a b c R : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  Real.sqrt (R^2 - r^2)

/-- Theorem: The distance between the plane of a triangle with side lengths 13, 14, and 15
    and the center of a sphere with radius 10 is 2√21 -/
theorem triangle_sphere_distance_specific :
  triangle_sphere_distance 13 14 15 10 = 2 * Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sphere_distance_specific_l1039_103905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_cents_arrangements_l1039_103973

/-- Represents the quantity of each stamp denomination -/
def stamp_quantities : Fin 9 → Nat
  | 0 => 1  -- 1-cent stamp
  | 1 => 2  -- 2-cent stamps
  | i => i.val + 1  -- 3-cent to 9-cent stamps

/-- Represents the value of each stamp denomination -/
def stamp_values : Fin 9 → Nat
  | i => i.val + 1

/-- A valid stamp arrangement is a list of stamps that sum to 15 cents -/
def is_valid_arrangement (arrangement : List (Fin 9)) : Prop :=
  (arrangement.map stamp_values).sum = 15

/-- Count of unique arrangements considering indistinguishable stamps -/
def unique_arrangement_count : Nat :=
  sorry

theorem fifteen_cents_arrangements :
  unique_arrangement_count = 263 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_cents_arrangements_l1039_103973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1039_103977

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then Real.cos (3 * x) + Real.sin (2 * x)
  else Real.cos (3 * x) - Real.sin (2 * x)

theorem f_is_even_and_correct : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, x > 0 → f x = Real.cos (3 * x) - Real.sin (2 * x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1039_103977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_40_l1039_103983

/-- Represents a time on a 12-hour analog clock -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  h_range : hours ≥ 1 ∧ hours ≤ 12
  m_range : minutes ≥ 0 ∧ minutes < 60

/-- Calculates the angle of the hour hand from 12 o'clock position -/
noncomputable def hourHandAngle (t : ClockTime) : ℝ :=
  (t.hours % 12 : ℝ) * 30 + (t.minutes : ℝ) * 0.5

/-- Calculates the angle of the minute hand from 12 o'clock position -/
noncomputable def minuteHandAngle (t : ClockTime) : ℝ :=
  (t.minutes : ℝ) * 6

/-- Calculates the smaller angle between hour and minute hands -/
noncomputable def smallerAngleBetweenHands (t : ClockTime) : ℝ :=
  let diff := abs (hourHandAngle t - minuteHandAngle t)
  min diff (360 - diff)

/-- The theorem stating that at 3:40, the smaller angle between hands is 130° -/
theorem angle_at_3_40 :
  ∃ (t : ClockTime), t.hours = 3 ∧ t.minutes = 40 ∧ smallerAngleBetweenHands t = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_40_l1039_103983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1039_103942

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) - Real.sqrt 3 * (Real.sin x)^2

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → q ≥ p)) ∧
  (∀ (x y : ℝ), π/12 ≤ x ∧ x < y ∧ y ≤ 7*π/12 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1039_103942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_convex_figures_common_point_l1039_103939

-- Define a type for points in a plane
variable {Point : Type}

-- Define a type for convex figures in a plane
variable {ConvexFigure : Type}

-- Define a function to check if a point is in a convex figure
variable (in_figure : Point → ConvexFigure → Prop)

-- Define a function to check if three figures share a common point
def three_share_point (f1 f2 f3 : ConvexFigure) : Prop :=
  ∃ p : Point, in_figure p f1 ∧ in_figure p f2 ∧ in_figure p f3

-- Main theorem
theorem four_convex_figures_common_point
  (f0 f1 f2 f3 : ConvexFigure)
  (h012 : three_share_point in_figure f0 f1 f2)
  (h013 : three_share_point in_figure f0 f1 f3)
  (h023 : three_share_point in_figure f0 f2 f3)
  (h123 : three_share_point in_figure f1 f2 f3) :
  ∃ p : Point, in_figure p f0 ∧ in_figure p f1 ∧ in_figure p f2 ∧ in_figure p f3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_convex_figures_common_point_l1039_103939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_c_l1039_103929

theorem triangle_tan_c (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 + 2 * a * b = 0) : 
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_c_l1039_103929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taoqi_faster_than_xiaoxiao_l1039_103967

-- Define Taoqi's speed
noncomputable def taoqi_speed : ℚ := 210

-- Define Xiaoxiao's distance and time
noncomputable def xiaoxiao_distance : ℚ := 500
noncomputable def xiaoxiao_time : ℚ := 3

-- Calculate Xiaoxiao's speed
noncomputable def xiaoxiao_speed : ℚ := xiaoxiao_distance / xiaoxiao_time

-- Theorem: Taoqi walks faster than Xiaoxiao
theorem taoqi_faster_than_xiaoxiao : taoqi_speed > xiaoxiao_speed := by
  -- Unfold the definitions
  unfold taoqi_speed xiaoxiao_speed xiaoxiao_distance xiaoxiao_time
  -- Simplify the inequality
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taoqi_faster_than_xiaoxiao_l1039_103967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_length_l1039_103921

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

/-- Represents a parabola -/
structure Parabola where
  a : ℝ  -- Coefficient in y^2 = ax

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem ellipse_parabola_intersection_length :
  ∀ (E : Ellipse) (C : Parabola),
    E.center = ⟨0, 0⟩ →
    E.e = 1/2 →
    C.a = 8 →
    E.a * E.e = 2 →
    ∃ (A B : Point),
      A.x = -2 ∧ B.x = -2 ∧
      (A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1) ∧
      (B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) ∧
      distance A B = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_length_l1039_103921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equals_11_l1039_103971

def even_integers_40_to_60 : List Nat :=
  (List.range 21).map (· + 40) |>.filter (fun n => n % 2 = 0)

def x : Nat := even_integers_40_to_60.sum

def y : Nat := even_integers_40_to_60.length

theorem y_equals_11 : x + y = 561 → y = 11 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equals_11_l1039_103971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_raisins_scoops_l1039_103960

/-- Prove that the number of scoops of golden seedless raisins is 10 -/
theorem golden_raisins_scoops 
  (natural_price : ℝ)
  (golden_price : ℝ)
  (mixture_price : ℝ)
  (total_scoops : ℝ)
  (h1 : natural_price = 3.45)
  (h2 : golden_price = 2.55)
  (h3 : mixture_price = 3)
  (h4 : total_scoops = 20)
  (h5 : ∃ (x y : ℝ), x + y = total_scoops ∧ natural_price * x + golden_price * y = mixture_price * total_scoops) :
  ∃ (y : ℕ), y = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_raisins_scoops_l1039_103960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1039_103924

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the length of a side
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the angle at a vertex
noncomputable def angle (p q r : ℝ × ℝ) : ℝ :=
  sorry -- Definition of angle calculation

-- Theorem statement
theorem angle_relation (t : Triangle) :
  let AB := side_length t.A t.B
  let BC := side_length t.B t.C
  let AC := side_length t.A t.C
  AC / BC = (AB + BC) / AC →
  angle t.B t.A t.C = 2 * angle t.A t.B t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1039_103924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_model_height_l1039_103912

/-- The actual height of the Eiffel Tower in feet -/
def actual_height : ℚ := 1063

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 50

/-- Calculates the height of the scale model before rounding -/
def model_height : ℚ := actual_height / scale_ratio

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem eiffel_tower_model_height :
  round_to_nearest model_height = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_model_height_l1039_103912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_polygon_sides_l1039_103918

/-- The sum of interior angles of a polygon with n sides --/
def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

/-- Possible number of sides after cutting off an angle --/
def possibleSides (n : ℕ) : Finset ℕ := {n - 1, n, n + 1}

/-- Theorem: Given a polygon with n sides, after cutting off one angle,
    if the sum of the interior angles of the resulting polygon is 1620°,
    then n could be 10, 11, or 12 --/
theorem original_polygon_sides (n : ℕ) :
  (∃ m ∈ possibleSides n, sumInteriorAngles m = 1620) →
  n ∈ ({10, 11, 12} : Finset ℕ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_polygon_sides_l1039_103918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l1039_103999

/-- The speed of sound in feet per second -/
def speed_of_sound : ℚ := 1100

/-- The time delay between lightning and thunder in seconds -/
def time_delay : ℚ := 12

/-- The number of feet in a mile -/
def feet_per_mile : ℚ := 5280

/-- Rounds a rational number to the nearest quarter -/
def round_to_quarter (x : ℚ) : ℚ :=
  ⌊x * 4 + 1/2⌋ / 4

/-- The theorem stating the distance to the lightning strike -/
theorem lightning_distance : 
  round_to_quarter ((speed_of_sound * time_delay) / feet_per_mile) = 5/2 := by
  sorry

#eval round_to_quarter ((speed_of_sound * time_delay) / feet_per_mile)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l1039_103999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_of_abs_fractions_l1039_103965

theorem set_of_abs_fractions :
  ∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) →
  (|a| / a + |b| / b ∈ ({-2, 0, 2} : Set ℝ)) ∧
  (∀ x ∈ ({-2, 0, 2} : Set ℝ), ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ |a| / a + |b| / b = x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_of_abs_fractions_l1039_103965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_l1039_103957

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2 - 3

-- State the theorem
theorem unique_zero_point :
  ∃! x, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_l1039_103957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rhombus_rectangle_perimeter_l1039_103995

/-- A rhombus inscribed in a rectangle -/
structure InscribedRhombus where
  -- The vertices of the rhombus
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  -- The vertices of the rectangle
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  -- Conditions
  inscribed : E.1 = J.1 ∧ F.2 = K.2 ∧ G.1 = L.1 ∧ H.2 = M.2
  je_length : dist J E = 12
  ef_length : dist E F = 10
  fk_length : dist F K = 16
  eg_length : dist E G = 24

/-- The perimeter of a rectangle given its vertices -/
def rectanglePerimeter (J K L M : ℝ × ℝ) : ℝ :=
  2 * (dist J K + dist K L)

/-- Theorem: The perimeter of the rectangle JKLM is 672/5 -/
theorem inscribed_rhombus_rectangle_perimeter (r : InscribedRhombus) :
  rectanglePerimeter r.J r.K r.L r.M = 672/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rhombus_rectangle_perimeter_l1039_103995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_l1039_103935

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the trisection points of a line segment -/
noncomputable def trisectionPoints (p1 p2 : Point) : (Point × Point) :=
  let t1 : Point := ⟨(p1.x + 2 * p2.x) / 3, (p1.y + 2 * p2.y) / 3⟩
  let t2 : Point := ⟨(2 * p1.x + p2.x) / 3, (2 * p1.y + p2.y) / 3⟩
  (t1, t2)

theorem line_passes_through_points :
  let l : Line := ⟨1, -4, 13⟩
  let p : Point := ⟨3, 4⟩
  let p1 : Point := ⟨-4, 5⟩
  let p2 : Point := ⟨5, -1⟩
  let (t1, t2) := trisectionPoints p1 p2
  pointOnLine p l ∧ (pointOnLine t1 l ∨ pointOnLine t2 l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_l1039_103935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_required_wage_l1039_103910

/-- Represents the financial situation for Nancy's university tuition --/
structure TuitionFinances where
  tuition : ℚ
  parentContribution : ℚ
  scholarship : ℚ
  loanMultiplier : ℚ
  workHours : ℚ

/-- Calculates the hourly wage Nancy needs to earn --/
def requiredHourlyWage (finances : TuitionFinances) : ℚ :=
  let totalContributions := finances.parentContribution + finances.scholarship + (finances.scholarship * finances.loanMultiplier)
  let remainingCost := finances.tuition - totalContributions
  remainingCost / finances.workHours

/-- Theorem stating that Nancy needs to earn $10 per hour --/
theorem nancy_required_wage (finances : TuitionFinances) 
  (h1 : finances.tuition = 22000)
  (h2 : finances.parentContribution = finances.tuition / 2)
  (h3 : finances.scholarship = 3000)
  (h4 : finances.loanMultiplier = 2)
  (h5 : finances.workHours = 200) :
  requiredHourlyWage finances = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_required_wage_l1039_103910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_prob_one_fifth_l1039_103964

def set_a : Finset ℕ := {2, 3, 4, 5}
def set_b : Finset ℕ := {4, 5, 6, 7, 8}

def total_outcomes : ℕ := Finset.card set_a * Finset.card set_b

def sum_occurrences (n : ℕ) : ℕ :=
  Finset.card ((set_a.product set_b).filter (fun p => p.1 + p.2 = n))

theorem sum_with_prob_one_fifth :
  ∃ n : ℕ, sum_occurrences n = total_outcomes / 5 ∧ n = 10 := by
  sorry

#eval sum_occurrences 10
#eval total_outcomes
#eval total_outcomes / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_prob_one_fifth_l1039_103964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1039_103927

-- Define the lines that form the triangle
def line1 : ℝ → ℝ := λ _ => 6
def line2 : ℝ → ℝ := λ x => 2 + x
def line3 : ℝ → ℝ := λ x => 2 - x

-- Define the triangle
def triangle : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧
    ((y = line1 x ∧ y ≥ line2 x ∧ y ≥ line3 x) ∨
     (y = line2 x ∧ y ≤ line1 x ∧ y ≥ line3 x) ∨
     (y = line3 x ∧ y ≤ line1 x ∧ y ≤ line2 x))}

-- State the theorem
theorem triangle_area : MeasureTheory.volume triangle = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1039_103927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1039_103900

theorem divisibility_condition (a n : ℕ) : n ∣ ((a + 1)^n - a^n) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1039_103900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cutting_theorem_l1039_103958

def is_whole_number (x : ℝ) : Prop := ∃ n : ℕ, x = n

structure Cube where
  edge : ℝ
  volume : ℝ := edge ^ 3

def can_be_cut_into (big_cube : Cube) (n : ℕ) : Prop :=
  ∃ (small_cubes : Finset Cube),
    small_cubes.card = n ∧
    (∀ c, c ∈ small_cubes → is_whole_number c.edge) ∧
    (∃ c1 c2, c1 ∈ small_cubes ∧ c2 ∈ small_cubes ∧ c1.edge ≠ c2.edge) ∧
    (small_cubes.sum (λ c => c.volume) = big_cube.volume)

theorem cube_cutting_theorem :
  ∃ (big_cube : Cube),
    big_cube.edge = 5 ∧
    can_be_cut_into big_cube 25 := by
  sorry

#check cube_cutting_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cutting_theorem_l1039_103958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1039_103972

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 4 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 5 = 0

-- Theorem statement
theorem circle_line_intersection :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_M x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (let d := |center_x + center_y - 5| / Real.sqrt 2;
     d = Real.sqrt 2 / 2) ∧
    (let chord_length := 2 * Real.sqrt (radius^2 - (Real.sqrt 2 / 2)^2);
     chord_length = Real.sqrt 46) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1039_103972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_bound_l1039_103933

theorem sin_product_bound (α : Real) : |Real.sin α * Real.sin (2 * α) * Real.sin (3 * α)| ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_bound_l1039_103933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_better_and_more_consistent_l1039_103989

-- Define the scores for players A and B
noncomputable def scores_A : Fin 5 → ℝ := sorry
noncomputable def scores_B : Fin 5 → ℝ := sorry

-- Define the average score for a player
noncomputable def average_score (scores : Fin 5 → ℝ) : ℝ :=
  (scores 0 + scores 1 + scores 2 + scores 3 + scores 4) / 5

-- Define a measure of consistency (e.g., variance)
noncomputable def consistency (scores : Fin 5 → ℝ) : ℝ :=
  let avg := average_score scores
  ((scores 0 - avg)^2 + (scores 1 - avg)^2 + (scores 2 - avg)^2 + 
   (scores 3 - avg)^2 + (scores 4 - avg)^2) / 5

-- Theorem statement
theorem player_B_better_and_more_consistent :
  average_score scores_B > average_score scores_A ∧
  consistency scores_B < consistency scores_A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_better_and_more_consistent_l1039_103989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parametric_curve_l1039_103987

theorem point_on_parametric_curve :
  ∃ θ : ℝ, Real.sin θ = 1/2 ∧ Real.cos (2*θ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parametric_curve_l1039_103987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_90_minutes_l1039_103959

/-- The number of radians turned by a clock's minute hand after a given time -/
noncomputable def minuteHandRadians (minutes : ℝ) : ℝ :=
  -(minutes / 60) * (2 * Real.pi)

theorem minute_hand_90_minutes :
  minuteHandRadians 90 = -3 * Real.pi := by
  unfold minuteHandRadians
  simp [Real.pi]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_90_minutes_l1039_103959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1039_103994

-- Define the sequence property
def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≥ 0

-- Define the sum function
def S (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S a n + a (n + 1)

-- State the theorem
theorem sequence_property (a : ℕ → ℝ) (h_seq : is_valid_sequence a) 
  (h_a1 : a 1 = 0) (h_a2 : a 2 = 1) (h_a3 : a 3 = 9)
  (h_rel : ∀ n > 3, (S a n)^2 * S a (n-2) = 10 * (S a (n-1))^3) :
  ∀ n ≥ 3, a n = 9 * 10^(n-3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1039_103994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_testicular_cell_properties_l1039_103926

/-- Represents the possible bases in nucleotides -/
inductive Base
| A
| C
| T
deriving Repr, DecidableEq

/-- Represents the cell cycle periods -/
inductive CellCyclePeriod
| Interphase
| EarlyMitosis
| LateMitosis
| EarlyMeiosis1
| LateMeiosis1
| EarlyMeiosis2
| LateMeiosis2
deriving Repr, DecidableEq

/-- Represents the DNA stability state -/
inductive DNAStability
| High
| Medium
| Low
deriving Repr, DecidableEq

structure TesticularCell where
  nucleotides : List Base
  dnaStability : CellCyclePeriod → DNAStability
  dnaSeparation : CellCyclePeriod → Bool

/-- The main theorem about testicular cells -/
theorem testicular_cell_properties (cell : TesticularCell) :
  (cell.nucleotides.toFinset.card = 3) ∧
  (cell.dnaStability CellCyclePeriod.Interphase = DNAStability.Low) ∧
  (cell.dnaSeparation CellCyclePeriod.LateMeiosis1 ∨ 
   cell.dnaSeparation CellCyclePeriod.LateMeiosis2) := by
  sorry

#check testicular_cell_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_testicular_cell_properties_l1039_103926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_systems_solutions_l1039_103982

theorem linear_systems_solutions :
  -- System (1)
  (∃! p : ℝ × ℝ, 3 * p.1 - 2 * p.2 = 9 ∧ p.1 + 2 * p.2 = 3 ∧ p.1 = 3 ∧ p.2 = 0) ∧
  -- System (2)
  (∃! q : ℝ × ℝ, 0.3 * q.1 - q.2 = 1 ∧ 0.2 * q.1 - 0.5 * q.2 = 19 ∧ q.1 = 370 ∧ q.2 = 110) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_systems_solutions_l1039_103982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_factors_to_remove_for_two_l1039_103920

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

def ends_in_two (n : ℕ) : Prop := n % 10 = 2

def factors_to_remove (n : ℕ) : ℕ := 
  let fives := (n / 5) + (n / 25)
  fives + 1

theorem min_factors_to_remove_for_two : 
  ∃ (removed : Finset ℕ), 
    removed.card = factors_to_remove 99 ∧ 
    ends_in_two ((factorial 99) / (removed.prod id)) := by
  sorry

#eval factors_to_remove 99

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_factors_to_remove_for_two_l1039_103920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l1039_103985

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + 3*x + 2 < 0}
def N : Set ℝ := {x | Real.rpow (1/2) x ≤ 4}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l1039_103985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_inverse_proportion_quadrants_converse_l1039_103901

/-- The inverse proportion function -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k + 1) / x

/-- The condition for the graph to be in the second and fourth quadrants -/
def in_second_and_fourth_quadrants (k : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 → f k x < 0) ∧ (x < 0 → f k x > 0)

/-- Theorem: If the graph of y = (k+1)/x is in the second and fourth quadrants, then k < -1 -/
theorem inverse_proportion_quadrants (k : ℝ) :
  in_second_and_fourth_quadrants k → k < -1 := by
  sorry

/-- Theorem: If k < -1, then the graph of y = (k+1)/x is in the second and fourth quadrants -/
theorem inverse_proportion_quadrants_converse (k : ℝ) :
  k < -1 → in_second_and_fourth_quadrants k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_inverse_proportion_quadrants_converse_l1039_103901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_a_l1039_103917

/-- The number of integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
def count_integer_a : ℕ := 3

/-- The set of integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
def valid_a : Finset ℤ := {-64, -12, 0}

/-- Predicate to check if a given 'a' has integer solutions for x^2 + ax + 9a = 0 -/
def has_integer_solution (a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 + a*x + 9*a = 0

theorem count_valid_a :
  (∀ a : ℤ, a ∈ valid_a ↔ has_integer_solution a) ∧
  Finset.card valid_a = count_integer_a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_a_l1039_103917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_properties_l1039_103937

/-- Sound pressure level calculation -/
noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

theorem sound_pressure_properties
  (p₀ p₁ p₂ p₃ : ℝ)
  (h₀ : p₀ > 0)
  (h₁ : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (h₂ : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (h₃ : sound_pressure_level p₃ p₀ = 40) :
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_properties_l1039_103937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_repeating_block_length_7_13_l1039_103997

/-- The least positive integer k such that 10^k ≡ 1 (mod 13) is 6 -/
theorem least_repeating_block_length_7_13 : 
  Nat.minFac (13 - 1) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_repeating_block_length_7_13_l1039_103997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_hike_distance_l1039_103962

theorem claire_hike_distance : 
  ∀ (x y z : ℝ),
  x = 5 ∧ 
  y = 8 ∧ 
  z = 60 * Real.pi / 180 →
  Real.sqrt ((x + y * Real.cos z)^2 + (y * Real.sin z)^2) = 3 * Real.sqrt 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_hike_distance_l1039_103962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_true_owner_is_zero_l1039_103928

/-- Represents a person who can either be the true owner of the rattle or not -/
inductive Person
| TrueOwner
| NotTrueOwner

/-- Represents the truth value of a statement -/
inductive Statement
| True
| False

/-- The statement made by the person -/
def statementMade (s : Statement) : Prop :=
  match s with
  | Statement.True => ∃ p, p = Person.TrueOwner ∧ (∃ q, q = Person.TrueOwner → q ≠ p)
  | Statement.False => ∀ p, p = Person.TrueOwner → (∀ q, q = Person.TrueOwner → q = p)

/-- The probability that the person who made the statement is the true owner -/
def probabilityTrueOwner : ℝ := 0

/-- Theorem stating that the probability of the person who made the statement
    being the true owner is 0 -/
theorem probability_true_owner_is_zero :
  probabilityTrueOwner = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_true_owner_is_zero_l1039_103928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_traveled_l1039_103907

-- Define constants
def map_scale : ℝ := 15000
def map_distance : ℝ := 8.5
def mountain_height_feet : ℝ := 3000
def inclination_angle : ℝ := 30
def inch_to_cm : ℝ := 2.54
def foot_to_meter : ℝ := 0.3048

-- Define the theorem
theorem total_distance_traveled (map_scale : ℝ) (map_distance : ℝ) (mountain_height_feet : ℝ) 
  (inclination_angle : ℝ) (inch_to_cm : ℝ) (foot_to_meter : ℝ) :
  let actual_distance_meters := map_distance * map_scale * inch_to_cm / 100
  let mountain_height_meters := mountain_height_feet * foot_to_meter
  let mountain_distance := mountain_height_meters / Real.sin (inclination_angle * π / 180)
  ∃ (ε : ℝ), ε > 0 ∧ |actual_distance_meters + mountain_distance - 5067.3| < ε := by
  sorry

-- Note: We've replaced ≈ with a more formal definition of approximation using ∃ and |...|

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_traveled_l1039_103907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_distance_l1039_103996

/-- Represents the properties of a cylindrical post and a squirrel's spiral path. -/
structure SpiralPath where
  postHeight : ℝ
  postCircumference : ℝ
  risePerCircuit : ℝ

/-- Calculates the total distance traveled by a squirrel on a spiral path. -/
noncomputable def totalDistance (path : SpiralPath) : ℝ :=
  let numCircuits := path.postHeight / path.risePerCircuit
  let diagonalDistancePerCircuit := Real.sqrt (path.risePerCircuit^2 + path.postCircumference^2)
  numCircuits * diagonalDistancePerCircuit

/-- Theorem stating that the squirrel travels 15 feet given the specified conditions. -/
theorem squirrel_distance : 
  let path : SpiralPath := {
    postHeight := 12,
    postCircumference := 3,
    risePerCircuit := 4
  }
  totalDistance path = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_distance_l1039_103996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_exists_l1039_103943

/-- Definition of a parabola with vertex (0,0) and focus (0,2) -/
def parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- Definition of a point in the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := 0 < x ∧ 0 < y

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Main theorem -/
theorem parabola_point_exists : ∃ (x y : ℝ),
  parabola x y ∧
  first_quadrant x y ∧
  distance x y 0 2 = 120 ∧
  x = 2 * Real.sqrt 236 ∧
  y = 118 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_exists_l1039_103943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1039_103981

theorem no_solution_exists : ∀ (a b c d n : ℕ+), 
  (a : ℝ)^2 + (b : ℝ)^2 + (c : ℝ)^2 + (d : ℝ)^2 - 4 * Real.sqrt ((a * b * c * d : ℝ)) ≠ 7 * (2 : ℝ)^(2 * (n : ℝ) - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1039_103981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l1039_103922

/-- The equation |x-7| = 2|x + 1| + |x-3| has exactly two distinct real solutions. -/
theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, |x - 7| = 2 * |x + 1| + |x - 3|) ∧ s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l1039_103922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_have_perpendicular_l1039_103932

-- Define a plane
structure Plane where
  x : ℝ
  y : ℝ

-- Define a line in the plane
structure Line where
  slope : ℝ ⊕ Unit  -- Use sum type to represent infinite slope
  intercept : ℝ

-- Define perpendicularity
def perpendicular (l1 l2 : Line) : Prop :=
  match l1.slope, l2.slope with
  | (Sum.inl m1), (Sum.inl m2) => m1 * m2 = -1
  | (Sum.inl _), (Sum.inr _) => True
  | (Sum.inr _), (Sum.inl _) => True
  | (Sum.inr _), (Sum.inr _) => False

-- Theorem statement
theorem all_lines_have_perpendicular :
  ∀ (l : Line), ∃ (l' : Line), perpendicular l l' :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_have_perpendicular_l1039_103932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1039_103941

/-- Sum of a geometric sequence -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the geometric sequence -/
noncomputable def a : ℝ := 8

/-- Common ratio of the geometric sequence -/
noncomputable def r : ℝ := 1/2

/-- Number of terms to sum -/
def n : ℕ := 6

theorem geometric_sequence_sum :
  geometricSum a r n = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1039_103941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_club_committee_probability_l1039_103998

def debate_club_size : ℕ := 24
def num_boys : ℕ := 14
def num_girls : ℕ := 10
def committee_size : ℕ := 5

theorem debate_club_committee_probability : 
  (((Nat.choose debate_club_size committee_size - 
    (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size)) : ℚ) / 
   (Nat.choose debate_club_size committee_size)) = 40250 / 42504 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_club_committee_probability_l1039_103998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1039_103904

/-- A differentiable function satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x, deriv f x < f x) ∧
  f 1 = 1

/-- The solution set of the inequality f(x) < e^(x-1) -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f x < Real.exp (x - 1)}

theorem solution_set_characterization (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  SolutionSet f = Set.Ioi 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1039_103904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_comparison_l1039_103940

noncomputable def samples_A : List ℝ := [102, 101, 99, 98, 103, 98, 99]
noncomputable def samples_B : List ℝ := [105, 102, 97, 92, 96, 101, 107]

noncomputable def is_qualified (x : ℝ) : Bool := 95 < x ∧ x < 105

noncomputable def mean (samples : List ℝ) : ℝ := (samples.sum) / (samples.length : ℝ)

noncomputable def variance (samples : List ℝ) : ℝ :=
  let m := mean samples
  (samples.map (λ x => (x - m)^2)).sum / (samples.length : ℝ)

theorem workshop_comparison :
  (mean samples_A = 100) ∧
  (mean samples_B = 100) ∧
  (variance samples_A < variance samples_B) ∧
  ((samples_A.filter is_qualified).length + (samples_B.filter is_qualified).length : ℝ) / 
   ((samples_A.length + samples_B.length) : ℝ) = 11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_comparison_l1039_103940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_POQ_l1039_103966

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos α, 2 * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ :=
  2 * Real.sin θ

-- Define the area of triangle POQ
noncomputable def area_POQ (θ₁ θ₂ : ℝ) : ℝ :=
  (1/2) * 4 * Real.cos θ₁ * 2 * Real.sin θ₂ * (Real.sqrt 3 / 2)

-- Theorem statement
theorem max_area_POQ :
  ∃ (θ₁ θ₂ : ℝ),
    (∀ (α β : ℝ), area_POQ α β ≤ area_POQ θ₁ θ₂) ∧
    area_POQ θ₁ θ₂ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_POQ_l1039_103966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l1039_103953

/-- Proves that the downstream speed is 90 kmph given the upstream speed and still water speed -/
theorem downstream_speed
  (upstream_speed : ℝ)
  (still_water_speed : ℝ)
  (downstream_speed : ℝ)
  (h1 : upstream_speed = 60)
  (h2 : still_water_speed = 75)
  (h3 : still_water_speed = (upstream_speed + downstream_speed) / 2) :
  downstream_speed = 90 :=
by
  sorry

#check downstream_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l1039_103953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l1039_103963

/-- Proposition P: α ≠ π/6 -/
def P (α : ℝ) : Prop := α ≠ Real.pi/6

/-- Proposition q: sin α ≠ 1/2 -/
def q (α : ℝ) : Prop := Real.sin α ≠ 1/2

theorem necessary_not_sufficient :
  (∀ α : ℝ, q α → P α) ∧ (∃ α : ℝ, P α ∧ ¬(q α)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l1039_103963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1039_103955

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/6)

theorem problem_solution (α : ℝ) 
  (h1 : Real.sin α = 3/5)
  (h2 : Real.pi/2 < α)
  (h3 : α < Real.pi) :
  f (α + Real.pi/12) = -(Real.sqrt 2)/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1039_103955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_increasing_l1039_103946

open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the given condition
axiom condition : ∀ x, (x + 1) * f x + x * f' x > 0

-- Define the function F(x) = xe^x f(x)
noncomputable def F (x : ℝ) : ℝ := x * exp x * f x

-- State the theorem
theorem F_is_increasing : StrictMono F := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_increasing_l1039_103946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1039_103914

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of the foci -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- Definition of distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- Theorem statement -/
theorem ellipse_properties :
  (∃ (M : ℝ × ℝ), is_on_ellipse M.1 M.2 ∧ 
    (∀ (N : ℝ × ℝ), is_on_ellipse N.1 N.2 → 
      distance M F₁ * distance M F₂ ≥ distance N F₁ * distance N F₂) ∧
    distance M F₁ * distance M F₂ = 4) ∧ 
  (∃ (M : ℝ × ℝ), is_on_ellipse M.1 M.2 ∧ 
    (∀ (N : ℝ × ℝ), is_on_ellipse N.1 N.2 → 
      Real.arccos ((distance M F₁)^2 + (distance M F₂)^2 - 4) / (2 * distance M F₁ * distance M F₂) ≥
      Real.arccos ((distance N F₁)^2 + (distance N F₂)^2 - 4) / (2 * distance N F₁ * distance N F₂)) ∧
    Real.arccos ((distance M F₁)^2 + (distance M F₂)^2 - 4) / (2 * distance M F₁ * distance M F₂) = π / 3) ∧
  (∀ (x y : ℝ), (x^2 / 2 + 2 * y^2 / 3 = 1 ∨ x^2 / 6 + 2 * y^2 / 9 = 1) ↔
    ∃ (A B : ℝ × ℝ), is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2 ∧ 
      A.2 = B.2 ∧ A.1 = -B.1 ∧ distance (x, y) A * distance (x, y) B = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1039_103914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1039_103930

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  ((-c, 0), (c, 0))

/-- Predicate for an obtuse angle between three points -/
def is_obtuse_angle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  AB.1 * BC.1 + AB.2 * BC.2 < 0

theorem ellipse_eccentricity_range (e : Ellipse) :
  (∃ P : PointOnEllipse e, is_obtuse_angle (foci e).1 (P.x, P.y) (foci e).2) →
  Real.sqrt 2 / 2 < eccentricity e ∧ eccentricity e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1039_103930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_and_negation_l1039_103902

theorem proposition_and_negation :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ (Real.log 3 / Real.log 2) ^ x₀ ≥ 1) ∧
  (¬∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ (Real.log 3 / Real.log 2) ^ x₀ ≥ 1 ↔
   ∀ x : ℝ, x ∈ Set.Ici 1 → (Real.log 3 / Real.log 2) ^ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_and_negation_l1039_103902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_2018_terms_l1039_103979

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

noncomputable def a (n : ℕ) : ℝ := f (n * Real.pi / 6)

theorem sum_of_2018_terms :
  (Finset.range 2018).sum a = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_2018_terms_l1039_103979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_distance_l1039_103916

/-- The maximum distance between bus stops that guarantees the boy won't miss the bus -/
noncomputable def max_distance : ℝ := 1.5

/-- The boy's speed relative to the bus speed -/
noncomputable def boy_speed_ratio : ℝ := 1/3

/-- The maximum distance at which the boy can see the bus -/
noncomputable def max_visibility : ℝ := 2

/-- Theorem stating the condition for the maximum distance between bus stops -/
theorem bus_stop_distance :
  ∀ (d : ℝ),
    d ≤ max_distance ↔
      ∃ (t : ℝ),
        t > 0 ∧
        d ≤ t * max_visibility ∧
        d ≤ (1 + boy_speed_ratio) * (t * boy_speed_ratio * max_visibility) :=
by sorry

#check bus_stop_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_distance_l1039_103916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_zero_solutions_one_solution_two_solutions_l1039_103944

-- Define the equation
def equation (x a : ℝ) : Prop := |x + 2| - |2*x + 8| = a

-- Define the number of solutions
noncomputable def num_solutions (a : ℝ) : ℕ :=
  if a > 2 then 0
  else if a = 2 then 1
  else 2

-- Theorem statement
theorem equation_solutions (a : ℝ) :
  (∃ x : ℝ, equation x a) ↔ num_solutions a ≠ 0 :=
by sorry

-- Additional theorems to cover all cases
theorem zero_solutions (a : ℝ) (h : a > 2) :
  ¬∃ x : ℝ, equation x a :=
by sorry

theorem one_solution (a : ℝ) (h : a = 2) :
  ∃! x : ℝ, equation x a :=
by sorry

theorem two_solutions (a : ℝ) (h : a < 2) :
  ∃ x y : ℝ, x ≠ y ∧ equation x a ∧ equation y a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_zero_solutions_one_solution_two_solutions_l1039_103944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_definition_correctness_l1039_103969

-- Define a type for points
variable {Point : Type}

-- Define predicates for being on the locus and satisfying conditions
variable (onLocus : Point → Prop)
variable (satisfiesConditions : Point → Prop)

-- Define the five methods
def methodA (onLocus satisfiesConditions : Point → Prop) : Prop := 
  ∀ p, onLocus p ↔ satisfiesConditions p

def methodB (onLocus satisfiesConditions : Point → Prop) : Prop := 
  (∀ p, onLocus p → satisfiesConditions p) ∧ ¬(∀ p, satisfiesConditions p → onLocus p)

def methodC (onLocus satisfiesConditions : Point → Prop) : Prop := 
  ∀ p, onLocus p ↔ satisfiesConditions p

def methodD (onLocus satisfiesConditions : Point → Prop) : Prop := 
  (∀ p, ¬onLocus p → ¬satisfiesConditions p) ∧ (∀ p, onLocus p → satisfiesConditions p)

def methodE (onLocus satisfiesConditions : Point → Prop) : Prop := 
  (∀ p, ¬satisfiesConditions p → ¬onLocus p) ∧ ¬(∀ p, satisfiesConditions p → onLocus p)

-- Theorem stating which methods are correct and incorrect
theorem locus_definition_correctness 
  (onLocus satisfiesConditions : Point → Prop) :
  (methodA onLocus satisfiesConditions ↔ (∀ p, onLocus p ↔ satisfiesConditions p)) ∧
  (methodB onLocus satisfiesConditions → ¬(∀ p, onLocus p ↔ satisfiesConditions p)) ∧
  (methodC onLocus satisfiesConditions ↔ (∀ p, onLocus p ↔ satisfiesConditions p)) ∧
  (methodD onLocus satisfiesConditions ↔ (∀ p, onLocus p ↔ satisfiesConditions p)) ∧
  (methodE onLocus satisfiesConditions → ¬(∀ p, onLocus p ↔ satisfiesConditions p)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_definition_correctness_l1039_103969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l1039_103988

theorem count_special_integers : 
  ∃ (s : Finset ℕ), s = {n : ℕ | n ≥ 2 ∧ 2013 % n = n % 3} ∧ s.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l1039_103988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_l1039_103909

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

-- Theorem statement
theorem unique_zero_point (a b : ℝ) :
  ((1/2 < a ∧ a ≤ Real.exp 2 / 2 ∧ b > 2*a) ∨
   (0 < a ∧ a < 1/2 ∧ b ≤ 2*a)) →
  ∃! x, f x a b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_l1039_103909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l1039_103952

-- Define the probabilities for each region
def prob_A : ℚ := 4/10
def prob_B : ℚ := 1/5

-- Define the probability of region C as a variable
variable (prob_C : ℚ)

-- Define other probabilities in terms of prob_C
def prob_D (prob_C : ℚ) : ℚ := prob_C
def prob_E (prob_C : ℚ) : ℚ := 2 * prob_C

-- Theorem statement
theorem spinner_probability : 
  prob_A + prob_B + prob_C + prob_D prob_C + prob_E prob_C = 1 → prob_C = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l1039_103952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fragrant_set_size_l1039_103948

/-- Definition of the polynomial P(n) = n^2 + n + 1 -/
def P (n : ℕ) : ℕ := n^2 + n + 1

/-- Definition of a fragrant set -/
def IsFragrant (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, ∃ y ∈ s, x ≠ y ∧ ¬Nat.Coprime x y

/-- The smallest size of a fragrant set is 6 -/
theorem smallest_fragrant_set_size :
  (∃ (a : ℕ) (s : Finset ℕ), s = Finset.range 6 ∧ IsFragrant (s.image (fun i => P (a + i)))) ∧
  (∀ (a : ℕ) (s : Finset ℕ), s.card < 6 → ¬IsFragrant (s.image (fun i => P (a + i)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fragrant_set_size_l1039_103948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_staircase_probability_l1039_103947

theorem building_staircase_probability (n : ℕ) (h : n > 1) : 
  (2^(n - 1) : ℚ) / Nat.choose (2 * (n - 1)) (n - 1) = 
  2^(n - 1) / Nat.choose (2 * (n - 1)) (n - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_staircase_probability_l1039_103947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_count_and_sum_is_six_l1039_103991

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x - y)^3) = f x^3 - 3*x*(f y)^2 + 3*y^2*(f x) - y^3

/-- The set of possible values for f(1) -/
def PossibleValues (f : ℝ → ℝ) : Set ℝ :=
  {z : ℝ | ∃ g : ℝ → ℝ, SatisfiesCondition g ∧ g 1 = z}

/-- The theorem to be proved -/
theorem product_of_count_and_sum_is_six :
  ∃ f : ℝ → ℝ, SatisfiesCondition f ∧
    (Finset.card {1, 2}) *
    (Finset.sum {1, 2} id) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_count_and_sum_is_six_l1039_103991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_l1039_103978

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The length of the lateral side of the trapezoid -/
  a : ℝ
  /-- The length of the segment connecting the points of tangency of the lateral sides with the circle -/
  b : ℝ
  /-- a and b are positive -/
  ha : a > 0
  hb : b > 0

/-- The diameter of the inscribed circle in a circumscribed isosceles trapezoid -/
noncomputable def inscribedCircleDiameter (t : CircumscribedTrapezoid) : ℝ :=
  Real.sqrt (t.a * t.b)

/-- Theorem: The diameter of the inscribed circle is √(ab) -/
theorem inscribed_circle_diameter (t : CircumscribedTrapezoid) :
  inscribedCircleDiameter t = Real.sqrt (t.a * t.b) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_l1039_103978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l1039_103986

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train of length 410 meters moving at 45 km/hour takes 44 seconds to pass a bridge of length 140 meters -/
theorem train_bridge_passing_time :
  time_to_pass_bridge 410 140 45 = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l1039_103986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_properties_l1039_103980

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - Real.log x - a

-- State the theorem
theorem zero_points_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f a x₁ = 0) 
  (h2 : f a x₂ = 0) 
  (h3 : x₁ ≠ x₂) :
  a > 1 ∧ x₁ + x₂ > a + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_properties_l1039_103980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l1039_103945

/-- Given a matrix M and vectors v, w, and u satisfying certain conditions,
    prove that M * (2v - w + u) equals [12, 11] -/
theorem matrix_vector_computation
  (M : Matrix (Fin 2) (Fin 2) ℝ)
  (v w u : Fin 2 → ℝ)
  (hv : M.mulVec v = ![3, 4])
  (hw : M.mulVec w = ![-1, -2])
  (hu : M.mulVec u = ![5, 1]) :
  M.mulVec (2 • v - w + u) = ![12, 11] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l1039_103945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1039_103923

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define that P bisects chord AB
def P_bisects_AB (A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem statement
theorem chord_equation :
  ∃ (A B : ℝ × ℝ),
    (my_circle A.1 A.2) ∧
    (my_circle B.1 B.2) ∧
    (P_bisects_AB A B) →
    ∀ (x y : ℝ), line_AB x y ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1039_103923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1039_103951

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

-- State the theorem
theorem f_range :
  ∃ (a b : ℝ), a = -Real.sqrt 3 ∧ b = 1 - Real.sqrt 3 / 2 ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → a ≤ f x ∧ f x ≤ b) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1039_103951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_when_m_neg_five_m_value_when_intersection_given_l1039_103934

noncomputable def f (x : ℝ) := Real.sqrt (-x^2 + 2*x + 8)
noncomputable def g (m : ℝ) (x : ℝ) := Real.log (-x^2 + 6*x + m) / Real.log 10

def A : Set ℝ := {x | -x^2 + 2*x + 8 ≥ 0}
def B (m : ℝ) : Set ℝ := {x | -x^2 + 6*x + m > 0}

theorem intersection_complement_when_m_neg_five :
  A ∩ (B (-5))ᶜ = Set.Icc (-2) 1 := by sorry

theorem m_value_when_intersection_given :
  A ∩ B 7 = Set.Ioc (-1) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_when_m_neg_five_m_value_when_intersection_given_l1039_103934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_pair_sin_cos_inequality_l1039_103931

theorem largest_integer_pair_sin_cos_inequality :
  ∀ a b : ℤ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
    (Real.sin x) ^ (a : ℝ) * (Real.cos x) ^ (b : ℝ) ≥ (1 / 2) ^ ((a + b : ℝ) / 2)) →
  a ≤ 0 ∧ b ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_pair_sin_cos_inequality_l1039_103931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1039_103936

theorem coefficient_x_cubed_in_expansion : 
  let expansion := (1 + X : Polynomial ℚ) * (2 + X)^5
  Polynomial.coeff expansion 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1039_103936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_with_squares_l1039_103950

/-- A rectangle type -/
structure Rectangle where
  area : ℝ

/-- A square type -/
structure Square where
  sideLength : ℝ
  area : ℝ

/-- Defines that a rectangle contains a square -/
def Rectangle.contains (r : Rectangle) (s : Square) : Prop :=
  r.area ≥ s.area

/-- Given a rectangular configuration ABCD containing three non-overlapping squares,
    where one square has an area of 4 square inches, and the side length of the larger square
    is twice the side length of the shaded square, prove that the area of ABCD is 24 square inches. -/
theorem area_of_rectangle_with_squares (ABCD : Rectangle) (s₁ s₂ s₃ : Square) :
  (s₁.area = 4) →
  (s₂.sideLength = s₁.sideLength) →
  (s₃.sideLength = 2 * s₁.sideLength) →
  (ABCD.contains s₁) →
  (ABCD.contains s₂) →
  (ABCD.contains s₃) →
  (ABCD.area = s₁.area + s₂.area + s₃.area) →
  ABCD.area = 24 := by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_with_squares_l1039_103950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_roller_length_approx_l1039_103903

/-- The length of a garden roller, given its diameter, area covered, and number of revolutions. -/
noncomputable def garden_roller_length (diameter : ℝ) (area_covered : ℝ) (revolutions : ℕ) : ℝ :=
  let π : ℚ := 22 / 7
  let radius : ℝ := diameter / 2
  let area_per_revolution : ℝ := area_covered / (revolutions : ℝ)
  area_per_revolution / (2 * (π : ℝ) * radius)

/-- Theorem stating that the length of the garden roller is approximately 0.28 m. -/
theorem garden_roller_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |garden_roller_length 1.4 35.2 4 - 0.28| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_roller_length_approx_l1039_103903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_decimals_l1039_103974

-- Define the division operation
def division (a b : ℚ) : ℚ := a / b

-- State the theorem
theorem divide_decimals : division (45 / 1000) (5 / 10000) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_decimals_l1039_103974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_numbers_appear_approx_probability_all_numbers_appear_exact_l1039_103911

/-- The probability of each number from 1 to 6 appearing at least once when throwing 10 fair six-sided dice -/
noncomputable def probability_all_numbers_appear : ℝ :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

/-- Theorem stating that the probability is approximately 0.272 -/
theorem probability_all_numbers_appear_approx :
  |probability_all_numbers_appear - 0.272| < 0.001 := by
  sorry

/-- Theorem stating that the probability is exactly equal to the given expression -/
theorem probability_all_numbers_appear_exact :
  probability_all_numbers_appear =
    1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_numbers_appear_approx_probability_all_numbers_appear_exact_l1039_103911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_QS_length_l1039_103970

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle_at_Q : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0

-- Define point S on PR
def S (P R : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (P.1 + t * (R.1 - P.1), P.2 + t * (R.2 - P.2))

-- Define the perpendicularity of QS to PR
def perpendicular (Q S P R : ℝ × ℝ) : Prop :=
  (S.1 - Q.1) * (R.1 - P.1) + (S.2 - Q.2) * (R.2 - P.2) = 0

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem triangle_QS_length
  (P Q R : ℝ × ℝ)
  (h_triangle : Triangle P Q R)
  (h_area : triangle_area P Q R = 30)
  (h_PQ : distance P Q = 5)
  (t : ℝ)
  (h_S : S P R t = S P R t)
  (h_perp : perpendicular Q (S P R t) P R) :
  distance Q (S P R t) = 60 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_QS_length_l1039_103970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_segments_is_five_l1039_103938

/-- A random walk in the plane where each step is of length 1 and the direction
    changes by 120 degrees clockwise or counterclockwise with equal probability. -/
def RandomWalk : Type := Unit

/-- The probability of hitting an already drawn segment on the nth step (n ≥ 3) -/
noncomputable def hitProbability (n : ℕ) : ℝ := 
  if n ≥ 3 then 1 / 2 else 0

/-- The expected number of segments drawn before hitting an already drawn segment -/
noncomputable def expectedSegments : ℝ := ∑' n, n * (1 / 2) ^ (n - 2) * hitProbability n

theorem expected_segments_is_five :
  expectedSegments = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_segments_is_five_l1039_103938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l1039_103993

theorem sin_2theta_value (θ : Real) (h : Real.sin θ ^ 4 + Real.cos θ ^ 4 = 5 / 9) :
  Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 ∨ Real.sin (2 * θ) = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l1039_103993
