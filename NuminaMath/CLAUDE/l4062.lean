import Mathlib

namespace NUMINAMATH_CALUDE_candy_mixture_cost_l4062_406278

/-- Given the conditions of mixing two types of candy, prove the cost of the first type --/
theorem candy_mixture_cost 
  (weight_first : ℝ) 
  (weight_second : ℝ) 
  (cost_second : ℝ) 
  (cost_mixture : ℝ) 
  (h1 : weight_first = 15) 
  (h2 : weight_second = 30) 
  (h3 : cost_second = 5) 
  (h4 : cost_mixture = 6) : 
  ∃ (cost_first : ℝ), cost_first = 8 ∧ 
    weight_first * cost_first + weight_second * cost_second = 
    (weight_first + weight_second) * cost_mixture :=
sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l4062_406278


namespace NUMINAMATH_CALUDE_largest_coin_distribution_l4062_406201

theorem largest_coin_distribution (n : ℕ) : n ≤ 111 ↔ 
  (∃ (k : ℕ), n = 12 * k + 3 ∧ n < 120) :=
sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_l4062_406201


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l4062_406264

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l4062_406264


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l4062_406294

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 8)
  (h_a5 : a 5 = 64) :
  a 3 = 16 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l4062_406294


namespace NUMINAMATH_CALUDE_decreasing_cubic_function_l4062_406206

-- Define the function f(x) = ax³ - 2x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x

-- State the theorem
theorem decreasing_cubic_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_function_l4062_406206


namespace NUMINAMATH_CALUDE_coefficient_expansion_l4062_406261

/-- The coefficient of x³ in the expansion of (x²-1)(x-2)⁷ -/
def coefficient_x_cubed : ℤ := -112

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_expansion :
  coefficient_x_cubed =
    binomial 7 6 * (-2)^6 - binomial 7 4 * (-2)^4 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l4062_406261


namespace NUMINAMATH_CALUDE_max_vector_sum_is_6_l4062_406298

-- Define the points in R²
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (1, 0)

-- Define the set of points D that satisfy |CD| = 1
def D : Set (ℝ × ℝ) := {d | ‖C - d‖ = 1}

-- Define the vector sum OA + OB + OD
def vectorSum (d : ℝ × ℝ) : ℝ × ℝ := A + B + d

-- Theorem statement
theorem max_vector_sum_is_6 :
  ∃ (m : ℝ), m = 6 ∧ ∀ (d : ℝ × ℝ), d ∈ D → ‖vectorSum d‖ ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_vector_sum_is_6_l4062_406298


namespace NUMINAMATH_CALUDE_mean_of_points_l4062_406288

def points : List ℝ := [81, 73, 83, 86, 73]

theorem mean_of_points : (points.sum / points.length : ℝ) = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_points_l4062_406288


namespace NUMINAMATH_CALUDE_frog_egg_hatching_fraction_l4062_406252

theorem frog_egg_hatching_fraction (total_eggs : ℕ) (dry_up_percent : ℚ) (eaten_percent : ℚ) (hatched_frogs : ℕ) :
  total_eggs = 800 →
  dry_up_percent = 1/10 →
  eaten_percent = 7/10 →
  hatched_frogs = 40 →
  (hatched_frogs : ℚ) / (total_eggs * (1 - dry_up_percent - eaten_percent)) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_frog_egg_hatching_fraction_l4062_406252


namespace NUMINAMATH_CALUDE_m_range_characterization_l4062_406232

/-- 
Given a real number m, this theorem states that m is in the open interval (2, 3)
if and only if both of the following conditions are satisfied:
1. The equation x^2 + mx + 1 = 0 has two distinct negative roots.
2. The equation 4x^2 + 4(m - 2)x + 1 = 0 has no real roots.
-/
theorem m_range_characterization (m : ℝ) : 
  (2 < m ∧ m < 3) ↔ 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∧
  (∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_m_range_characterization_l4062_406232


namespace NUMINAMATH_CALUDE_tom_completion_time_l4062_406287

/-- The time it takes Tom to complete a wall on his own after working with Avery for one hour -/
theorem tom_completion_time (avery_rate tom_rate : ℚ) : 
  avery_rate = 1/2 →  -- Avery's rate in walls per hour
  tom_rate = 1/4 →    -- Tom's rate in walls per hour
  (avery_rate + tom_rate) * 1 = 3/4 →  -- Combined work in first hour
  (1 - (avery_rate + tom_rate) * 1) / tom_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_completion_time_l4062_406287


namespace NUMINAMATH_CALUDE_proj_equals_v_l4062_406200

/-- Given two 2D vectors v and w, prove that the projection of v onto w is equal to v itself. -/
theorem proj_equals_v (v w : Fin 2 → ℝ) (hv : v = ![- 3, 2]) (hw : w = ![4, - 2]) :
  (v • w / (w • w)) • w = v := by sorry

end NUMINAMATH_CALUDE_proj_equals_v_l4062_406200


namespace NUMINAMATH_CALUDE_max_m_value_inequality_solution_l4062_406217

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for the maximum value of m
theorem max_m_value : 
  (∃ m : ℝ, ∀ x : ℝ, f x - m ≥ 0 ∧ ¬∃ m' : ℝ, m' > m ∧ ∀ x : ℝ, f x - m' ≥ 0) → 
  (∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, f x - m ≥ 0 ∧ ¬∃ m' : ℝ, m' > m ∧ ∀ x : ℝ, f x - m' ≥ 0) :=
sorry

-- Theorem for the solution of the inequality
theorem inequality_solution :
  {x : ℝ | |x - 3| - 2*x ≤ 4} = {x : ℝ | x ≥ -1/3} :=
sorry

end NUMINAMATH_CALUDE_max_m_value_inequality_solution_l4062_406217


namespace NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l4062_406290

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x in its domain -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- The product of an odd function and the absolute value of an even function is odd -/
theorem odd_times_abs_even_is_odd (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * |g x|) := by sorry

end NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l4062_406290


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l4062_406262

theorem shaded_area_between_circles (r : ℝ) : 
  r > 0 → 
  (2 * r = 6) → 
  (π * (3 * r)^2 - π * r^2 = 72 * π) := by
sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l4062_406262


namespace NUMINAMATH_CALUDE_fraction_problem_l4062_406275

theorem fraction_problem : ∃ x : ℝ, x * (5/9) * (1/2) = 0.11111111111111112 ∧ x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l4062_406275


namespace NUMINAMATH_CALUDE_common_difference_is_negative_three_l4062_406230

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ → ℤ
  first_seventh_sum : a 1 + a 7 = -8
  second_term : a 2 = 2
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The common difference of an arithmetic progression -/
def common_difference (ap : ArithmeticProgression) : ℤ :=
  ap.a 2 - ap.a 1

theorem common_difference_is_negative_three (ap : ArithmeticProgression) :
  common_difference ap = -3 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_negative_three_l4062_406230


namespace NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l4062_406297

/-- A figure in a plane --/
structure Figure where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Represents an axis of symmetry for a figure --/
structure AxisOfSymmetry where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Represents a center of symmetry for a figure --/
structure CenterOfSymmetry where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- A function to determine if a figure has exactly two axes of symmetry --/
def has_exactly_two_axes_of_symmetry (f : Figure) : Prop :=
  ∃ (a1 a2 : AxisOfSymmetry), a1 ≠ a2 ∧
    (∀ (a : AxisOfSymmetry), a = a1 ∨ a = a2)

/-- A function to determine if a figure has a center of symmetry --/
def has_center_of_symmetry (f : Figure) : Prop :=
  ∃ (c : CenterOfSymmetry), true  -- Placeholder, replace with actual condition

/-- Theorem: If a figure has exactly two axes of symmetry, it must have a center of symmetry --/
theorem two_axes_implies_center_symmetry (f : Figure) :
  has_exactly_two_axes_of_symmetry f → has_center_of_symmetry f :=
by sorry

end NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l4062_406297


namespace NUMINAMATH_CALUDE_min_value_of_function_l4062_406236

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  let y := 2*x + 4/(x-1) - 1
  ∀ z, z = 2*x + 4/(x-1) - 1 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l4062_406236


namespace NUMINAMATH_CALUDE_fraction_equality_l4062_406272

theorem fraction_equality : (1 / 4 - 1 / 6) / (1 / 3 + 1 / 2) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4062_406272


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_l4062_406229

theorem tangent_line_at_negative_one (x y : ℝ) :
  y = 2*x - x^3 → 
  let tangent_point := (-1, 2*(-1) - (-1)^3)
  let tangent_slope := -3*(-1)^2 + 2
  (x + y + 2 = 0) = 
    ((y - tangent_point.2) = tangent_slope * (x - tangent_point.1)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_l4062_406229


namespace NUMINAMATH_CALUDE_total_students_l4062_406249

theorem total_students (french : ℕ) (spanish : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 5)
  (h2 : spanish = 10)
  (h3 : both = 4)
  (h4 : neither = 13) :
  french + spanish + both + neither = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l4062_406249


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l4062_406257

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem f_monotone_decreasing :
  ∀ f : ℝ → ℝ, (∀ x, deriv f x = f' x) →
  ∀ x ∈ Set.Ioo 0 2, deriv f (x + 1) < 0 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l4062_406257


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l4062_406248

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis of the ellipse is 8 --/
theorem ellipse_major_axis_length :
  let cylinder_radius : ℝ := 2
  let major_minor_ratio : ℝ := 2
  major_axis_length cylinder_radius major_minor_ratio = 8 := by
  sorry

#check ellipse_major_axis_length

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l4062_406248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4062_406286

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  S : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- Given conditions for the arithmetic sequence -/
def given_conditions (seq : ArithmeticSequence) : Prop :=
  seq.a 5 + seq.a 9 = -2 ∧ seq.S 3 = 57

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : given_conditions seq) :
  (∀ n : ℕ, seq.a n = 27 - 4 * n) ∧
  (∃ m : ℕ, ∀ n : ℕ, seq.S n ≤ m ∧ seq.S n = m ↔ n = 6) ∧ 
  (∃ m : ℕ, m = 78 ∧ ∀ n : ℕ, seq.S n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4062_406286


namespace NUMINAMATH_CALUDE_circle_line_intersection_point_P_on_line_distance_range_l4062_406253

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}

-- Define the line l
def l (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (m + 2) * p.1 + (2 * m + 1) * p.2 = 7 * m + 8}

-- Define the point P when m = 1
def P : ℝ × ℝ := (0, 5)

theorem circle_line_intersection (m : ℝ) : (C ∩ l m).Nonempty := by sorry

theorem point_P_on_line : P ∈ l 1 := by sorry

theorem distance_range : 
  ∀ Q ∈ C, (2 * Real.sqrt 2 - 2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧ 
             Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ (2 * Real.sqrt 2 + 2) := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_point_P_on_line_distance_range_l4062_406253


namespace NUMINAMATH_CALUDE_min_team_size_is_six_l4062_406255

/-- Represents the job parameters and conditions -/
structure JobParameters where
  totalDays : ℕ
  initialDays : ℕ
  initialWorkCompleted : ℚ
  initialTeamSize : ℕ
  rateIncreaseDay : ℕ
  rateIncreaseFactor : ℚ

/-- Calculates the minimum team size required from the rate increase day -/
def minTeamSizeAfterRateIncrease (params : JobParameters) : ℕ :=
  sorry

/-- Theorem stating that the minimum team size after rate increase is 6 -/
theorem min_team_size_is_six (params : JobParameters)
  (h1 : params.totalDays = 40)
  (h2 : params.initialDays = 10)
  (h3 : params.initialWorkCompleted = 1/4)
  (h4 : params.initialTeamSize = 12)
  (h5 : params.rateIncreaseDay = 20)
  (h6 : params.rateIncreaseFactor = 2) :
  minTeamSizeAfterRateIncrease params = 6 :=
sorry

end NUMINAMATH_CALUDE_min_team_size_is_six_l4062_406255


namespace NUMINAMATH_CALUDE_weeks_of_papayas_l4062_406293

def jake_papayas_per_week : ℕ := 3
def brother_papayas_per_week : ℕ := 5
def father_papayas_per_week : ℕ := 4
def total_papayas_bought : ℕ := 48

theorem weeks_of_papayas : 
  (total_papayas_bought / (jake_papayas_per_week + brother_papayas_per_week + father_papayas_per_week) = 4) := by
  sorry

end NUMINAMATH_CALUDE_weeks_of_papayas_l4062_406293


namespace NUMINAMATH_CALUDE_brocard_angle_inequalities_l4062_406218

theorem brocard_angle_inequalities (α β γ φ : Real) 
  (triangle_angles : α + β + γ = π) 
  (brocard_angle : 0 < φ ∧ φ ≤ π/6)
  (brocard_identity : Real.sin (α - φ) * Real.sin (β - φ) * Real.sin (γ - φ) = Real.sin φ ^ 3) :
  φ ^ 3 ≤ (α - φ) * (β - φ) * (γ - φ) ∧ 8 * φ ^ 3 ≤ α * β * γ := by
  sorry

end NUMINAMATH_CALUDE_brocard_angle_inequalities_l4062_406218


namespace NUMINAMATH_CALUDE_hall_dark_tile_fraction_l4062_406285

/-- Represents a tiling pattern on a floor -/
structure TilingPattern :=
  (size : Nat)
  (dark_tiles_in_section : Nat)
  (section_size : Nat)

/-- The fraction of dark tiles in a tiling pattern -/
def dark_tile_fraction (pattern : TilingPattern) : Rat :=
  pattern.dark_tiles_in_section / (pattern.section_size * pattern.section_size)

/-- Theorem stating that for the given tiling pattern, the fraction of dark tiles is 5/8 -/
theorem hall_dark_tile_fraction :
  ∀ (pattern : TilingPattern),
    pattern.size = 8 ∧
    pattern.section_size = 4 ∧
    pattern.dark_tiles_in_section = 10 →
    dark_tile_fraction pattern = 5 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_hall_dark_tile_fraction_l4062_406285


namespace NUMINAMATH_CALUDE_sum_of_squares_is_integer_l4062_406212

theorem sum_of_squares_is_integer 
  (a b c : ℚ) 
  (h1 : ∃ k : ℤ, (a + b + c : ℚ) = k)
  (h2 : ∃ m : ℤ, (a * b + b * c + c * a) / (a + b + c) = m) :
  ∃ n : ℤ, (a^2 + b^2 + c^2) / (a + b + c) = n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_integer_l4062_406212


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l4062_406234

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 5 ≥ 0 ∧ x < 7) ↔ (5 ≤ x ∧ x < 7) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l4062_406234


namespace NUMINAMATH_CALUDE_coupon_problem_l4062_406260

/-- Calculates the total number of bottles that would be received if no coupons were lost -/
def total_bottles (bottles_per_coupon : ℕ) (lost_coupons : ℕ) (remaining_coupons : ℕ) : ℕ :=
  (remaining_coupons + lost_coupons) * bottles_per_coupon

/-- Proves that given the conditions, the total number of bottles would be 21 -/
theorem coupon_problem :
  let bottles_per_coupon : ℕ := 3
  let lost_coupons : ℕ := 3
  let remaining_coupons : ℕ := 4
  total_bottles bottles_per_coupon lost_coupons remaining_coupons = 21 := by
  sorry

end NUMINAMATH_CALUDE_coupon_problem_l4062_406260


namespace NUMINAMATH_CALUDE_impossible_digit_assignment_l4062_406244

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  vertices : ℕ
  sides_eq : sides = n
  vertices_eq : vertices = n

/-- Assignment of digits to vertices -/
def DigitAssignment (n : ℕ) := Fin n → Fin 10

/-- Predicate to check if an assignment satisfies the condition -/
def SatisfiesCondition (n : ℕ) (assignment : DigitAssignment n) : Prop :=
  ∀ (i j : Fin 10), i ≠ j → 
    ∃ (v w : Fin n), v.val + 1 = w.val ∨ (v.val = n - 1 ∧ w.val = 0) ∧ 
      assignment v = i ∧ assignment w = j

theorem impossible_digit_assignment :
  ¬ ∃ (assignment : DigitAssignment 45), SatisfiesCondition 45 assignment := by
  sorry

end NUMINAMATH_CALUDE_impossible_digit_assignment_l4062_406244


namespace NUMINAMATH_CALUDE_edge_sum_is_96_l4062_406208

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- Three dimensions in geometric progression
  a : ℝ
  r : ℝ
  -- Volume is 512 cm³
  volume_eq : a * (a * r) * (a * r * r) = 512
  -- Surface area is 384 cm²
  surface_area_eq : 2 * (a * (a * r) + a * (a * r * r) + (a * r) * (a * r * r)) = 384

/-- The sum of all edge lengths of the rectangular solid is 96 cm -/
theorem edge_sum_is_96 (solid : RectangularSolid) :
  4 * (solid.a + solid.a * solid.r + solid.a * solid.r * solid.r) = 96 := by
  sorry

end NUMINAMATH_CALUDE_edge_sum_is_96_l4062_406208


namespace NUMINAMATH_CALUDE_pond_volume_1400_l4062_406210

/-- The volume of a rectangular prism-shaped pond -/
def pond_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of a pond with dimensions 28 m x 10 m x 5 m is 1400 cubic meters -/
theorem pond_volume_1400 :
  pond_volume 28 10 5 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_1400_l4062_406210


namespace NUMINAMATH_CALUDE_gcd_840_1764_l4062_406269

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l4062_406269


namespace NUMINAMATH_CALUDE_car_average_speed_l4062_406207

/-- Given a car that travels 65 km in the first hour and 45 km in the second hour,
    prove that its average speed is 55 km/h. -/
theorem car_average_speed (distance1 : ℝ) (distance2 : ℝ) (time : ℝ) 
  (h1 : distance1 = 65)
  (h2 : distance2 = 45)
  (h3 : time = 2) :
  (distance1 + distance2) / time = 55 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l4062_406207


namespace NUMINAMATH_CALUDE_spade_combination_l4062_406291

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_combination : spade 5 (spade 2 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_combination_l4062_406291


namespace NUMINAMATH_CALUDE_gcd_360_210_l4062_406213

theorem gcd_360_210 : Nat.gcd 360 210 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_210_l4062_406213


namespace NUMINAMATH_CALUDE_quadratic_vertex_and_symmetry_l4062_406243

/-- Given a quadratic function f(x) = -x^2 - 4x + 2, 
    its vertex is at (-2, 6) and its axis of symmetry is x = -2 -/
theorem quadratic_vertex_and_symmetry :
  let f : ℝ → ℝ := λ x ↦ -x^2 - 4*x + 2
  ∃ (vertex : ℝ × ℝ) (axis : ℝ),
    vertex = (-2, 6) ∧
    axis = -2 ∧
    (∀ x, f x = f (2 * axis - x)) ∧
    (∀ x, f x ≤ f axis) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_and_symmetry_l4062_406243


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_parallel_lines_perpendicular_planes_l4062_406237

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)

-- Define the given conditions
variable (l₁ l₂ : Line) (α β : Plane)
variable (h1 : perpendicular l₁ α)
variable (h2 : contains β l₂)

-- Theorem to prove
theorem line_plane_perpendicular_parallel 
  (h3 : parallel α β) : perpendicularLines l₁ l₂ :=
sorry

theorem parallel_lines_perpendicular_planes 
  (h4 : perpendicularLines l₁ l₂) : perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_parallel_lines_perpendicular_planes_l4062_406237


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l4062_406274

/-- Represents the number of fruits in a salad -/
structure FruitSalad where
  blueberries : ℕ
  raspberries : ℕ
  grapes : ℕ
  cherries : ℕ

/-- Conditions for the fruit salad problem -/
def validFruitSalad (s : FruitSalad) : Prop :=
  s.blueberries + s.raspberries + s.grapes + s.cherries = 350 ∧
  s.raspberries = 3 * s.blueberries ∧
  s.grapes = 4 * s.cherries ∧
  s.cherries = 5 * s.raspberries

/-- Theorem stating that a valid fruit salad has 66 cherries -/
theorem fruit_salad_cherries (s : FruitSalad) (h : validFruitSalad s) : s.cherries = 66 := by
  sorry

#check fruit_salad_cherries

end NUMINAMATH_CALUDE_fruit_salad_cherries_l4062_406274


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l4062_406238

theorem rectangle_area_problem : ∃ (x y : ℝ), 
  (x + 3) * (y - 1) = x * y ∧ 
  (x - 3) * (y + 1.5) = x * y ∧ 
  x * y = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l4062_406238


namespace NUMINAMATH_CALUDE_binomial_coefficient_x_cubed_in_x_plus_one_to_sixth_l4062_406226

theorem binomial_coefficient_x_cubed_in_x_plus_one_to_sixth : 
  (Finset.range 7).sum (fun k => Nat.choose 6 k * X^k) = 
    X^6 + 6*X^5 + 15*X^4 + 20*X^3 + 15*X^2 + 6*X + 1 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x_cubed_in_x_plus_one_to_sixth_l4062_406226


namespace NUMINAMATH_CALUDE_six_pencil_sharpeners_affordable_remaining_money_buys_four_pencil_cases_remaining_money_after_ten_pencil_cases_l4062_406299

-- Define the given prices and budget
def total_budget : ℕ := 100
def pencil_sharpener_price : ℕ := 15
def notebooks_6_price : ℕ := 24
def pencil_case_price : ℕ := 5
def colored_pencils_2boxes_price : ℕ := 16

-- Theorem 1: 6 pencil sharpeners cost less than or equal to 100 yuan
theorem six_pencil_sharpeners_affordable :
  6 * pencil_sharpener_price ≤ total_budget :=
sorry

-- Theorem 2: After buying 20 notebooks, the remaining money can buy exactly 4 pencil cases
theorem remaining_money_buys_four_pencil_cases :
  (total_budget - (20 * (notebooks_6_price / 6))) / pencil_case_price = 4 :=
sorry

-- Theorem 3: After buying 10 pencil cases, the remaining money is 50 yuan
theorem remaining_money_after_ten_pencil_cases :
  total_budget - (10 * pencil_case_price) = 50 :=
sorry

end NUMINAMATH_CALUDE_six_pencil_sharpeners_affordable_remaining_money_buys_four_pencil_cases_remaining_money_after_ten_pencil_cases_l4062_406299


namespace NUMINAMATH_CALUDE_area_condition_implies_isosceles_right_l4062_406289

/-- A triangle with sides a and b, and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  S : ℝ

/-- Definition of an isosceles right triangle -/
def IsIsoscelesRight (t : Triangle) : Prop :=
  t.a = t.b ∧ t.S = (1/2) * t.a * t.b

/-- Theorem: If the area of a triangle is 1/4(a^2 + b^2), then it's an isosceles right triangle -/
theorem area_condition_implies_isosceles_right (t : Triangle) 
    (h : t.S = (1/4) * (t.a^2 + t.b^2)) : 
    IsIsoscelesRight t := by
  sorry


end NUMINAMATH_CALUDE_area_condition_implies_isosceles_right_l4062_406289


namespace NUMINAMATH_CALUDE_percent_relation_l4062_406209

theorem percent_relation (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l4062_406209


namespace NUMINAMATH_CALUDE_f_monotone_and_max_a_l4062_406271

noncomputable def f (a x : ℝ) : ℝ := (x - a - 1) * Real.exp (x - 1) - (1/2) * x^2 + a * x

theorem f_monotone_and_max_a :
  (∀ x y : ℝ, x < y → f 1 x < f 1 y) ∧
  (∃ a : ℝ, a = Real.exp 1 / 2 - 1 ∧
    (∀ b : ℝ, (∃ x : ℝ, x > 0 ∧ f b x = -1/2) →
      (∀ y : ℝ, y > 0 → f b y ≥ -1/2) →
      b ≤ a)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_max_a_l4062_406271


namespace NUMINAMATH_CALUDE_marbles_remaining_l4062_406239

/-- Calculates the number of marbles remaining in a store after sales. -/
theorem marbles_remaining 
  (initial_marbles : ℕ) 
  (num_customers : ℕ) 
  (marbles_per_customer : ℕ) 
  (h1 : initial_marbles = 400)
  (h2 : num_customers = 20)
  (h3 : marbles_per_customer = 15) : 
  initial_marbles - num_customers * marbles_per_customer = 100 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l4062_406239


namespace NUMINAMATH_CALUDE_same_solution_k_value_l4062_406224

theorem same_solution_k_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = 0 ↔ 5 * x + 3 * k = 21) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l4062_406224


namespace NUMINAMATH_CALUDE_map_scale_l4062_406295

theorem map_scale (map_length : ℝ) (actual_distance : ℝ) :
  (15 : ℝ) * actual_distance = 90 * map_length →
  (20 : ℝ) * actual_distance = 120 * map_length :=
by sorry

end NUMINAMATH_CALUDE_map_scale_l4062_406295


namespace NUMINAMATH_CALUDE_fathers_children_l4062_406228

theorem fathers_children (father_age : ℕ) (children_sum : ℕ) (n : ℕ) : 
  father_age = 75 →
  father_age = children_sum →
  children_sum + 15 * n = 2 * (father_age + 15) →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_fathers_children_l4062_406228


namespace NUMINAMATH_CALUDE_pi_half_irrational_l4062_406204

theorem pi_half_irrational : Irrational (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l4062_406204


namespace NUMINAMATH_CALUDE_no_integer_solutions_l4062_406254

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l4062_406254


namespace NUMINAMATH_CALUDE_orange_trees_remaining_fruit_l4062_406265

theorem orange_trees_remaining_fruit (num_trees : ℕ) (fruits_per_tree : ℕ) (fraction_picked : ℚ) : 
  num_trees = 8 → 
  fruits_per_tree = 200 → 
  fraction_picked = 2/5 → 
  (num_trees * fruits_per_tree) - (num_trees * fruits_per_tree * fraction_picked) = 960 := by
sorry

end NUMINAMATH_CALUDE_orange_trees_remaining_fruit_l4062_406265


namespace NUMINAMATH_CALUDE_classroom_benches_l4062_406223

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of benches needed given the number of students and students per bench -/
def benchesNeeded (students : ℕ) (studentsPerBench : ℕ) : ℕ := sorry

theorem classroom_benches :
  let studentsBase5 : ℕ := 312
  let studentsPerBench : ℕ := 3
  let studentsBase10 : ℕ := base5ToBase10 studentsBase5
  benchesNeeded studentsBase10 studentsPerBench = 28 := by sorry

end NUMINAMATH_CALUDE_classroom_benches_l4062_406223


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l4062_406270

theorem square_fence_perimeter 
  (total_posts : ℕ) 
  (post_width : ℚ) 
  (gap_width : ℕ) : 
  total_posts = 36 → 
  post_width = 1/3 → 
  gap_width = 6 → 
  (4 * ((total_posts / 4 + 1) * post_width + (total_posts / 4) * gap_width)) = 204 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l4062_406270


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l4062_406241

/-- Represents the number of emails Jack received at different times of the day. -/
structure EmailCount where
  morning : Nat
  total : Nat

/-- Calculates the number of emails Jack received in the afternoon. -/
def afternoon_emails (e : EmailCount) : Nat :=
  e.total - e.morning

/-- Theorem: Jack received 1 email in the afternoon. -/
theorem jack_afternoon_emails :
  let e : EmailCount := { morning := 4, total := 5 }
  afternoon_emails e = 1 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l4062_406241


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4062_406276

theorem right_triangle_hypotenuse (x : ℝ) :
  x > 0 ∧
  (1/2 * x * (2*x - 1) = 72) →
  Real.sqrt (x^2 + (2*x - 1)^2) = Real.sqrt 370 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4062_406276


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l4062_406203

theorem half_abs_diff_squares : (1/2 : ℝ) * |25^2 - 20^2| = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l4062_406203


namespace NUMINAMATH_CALUDE_function_value_problem_l4062_406266

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x + 3)
  (h2 : f (m - 1) = 6) : 
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_function_value_problem_l4062_406266


namespace NUMINAMATH_CALUDE_chocolate_division_l4062_406279

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_edward : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  piles_to_edward = 2 →
  piles_to_edward * (total_chocolate / num_piles) = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l4062_406279


namespace NUMINAMATH_CALUDE_candy_distribution_l4062_406292

theorem candy_distribution (num_clowns num_children initial_candies remaining_candies : ℕ) 
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : initial_candies = 700)
  (h4 : remaining_candies = 20)
  (h5 : ∃ (candies_per_person : ℕ), 
    (num_clowns + num_children) * candies_per_person = initial_candies - remaining_candies) :
  ∃ (candies_per_person : ℕ), candies_per_person = 20 ∧ 
    (num_clowns + num_children) * candies_per_person = initial_candies - remaining_candies :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l4062_406292


namespace NUMINAMATH_CALUDE_quadratic_function_value_l4062_406267

/-- Given a quadratic function f(x) = ax^2 + bx + 2 with f(1) = 4 and f(2) = 10, prove that f(3) = 20 -/
theorem quadratic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + 2)
  (h2 : f 1 = 4)
  (h3 : f 2 = 10) :
  f 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l4062_406267


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l4062_406202

theorem incorrect_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (correct_value : ℝ) 
  (new_mean : ℝ) 
  (h1 : n = 40)
  (h2 : initial_mean = 100)
  (h3 : correct_value = 50)
  (h4 : new_mean = 99.075) :
  (n : ℝ) * initial_mean - (n : ℝ) * new_mean + correct_value = 87 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l4062_406202


namespace NUMINAMATH_CALUDE_jumper_cost_l4062_406250

def initial_amount : ℕ := 26
def tshirt_cost : ℕ := 4
def heels_cost : ℕ := 5
def remaining_amount : ℕ := 8

theorem jumper_cost :
  initial_amount - tshirt_cost - heels_cost - remaining_amount = 9 :=
by sorry

end NUMINAMATH_CALUDE_jumper_cost_l4062_406250


namespace NUMINAMATH_CALUDE_binary_1100_is_12_l4062_406263

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1100_is_12 :
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_is_12_l4062_406263


namespace NUMINAMATH_CALUDE_remainder_theorem_l4062_406216

theorem remainder_theorem (x : ℂ) : 
  (x^2023 + 1) % (x^10 - x^8 + x^6 - x^4 + x^2 - 1) = -x^7 + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4062_406216


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4062_406220

theorem polynomial_division_remainder (k : ℚ) : 
  ∃! k, ∃ q : Polynomial ℚ, 
    3 * X^3 + k * X^2 - 8 * X + 52 = (3 * X + 4) * q + 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4062_406220


namespace NUMINAMATH_CALUDE_joan_remaining_books_l4062_406245

/-- The number of books Joan has after selling some -/
def books_remaining (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Theorem: Joan has 7 books remaining -/
theorem joan_remaining_books : books_remaining 33 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_books_l4062_406245


namespace NUMINAMATH_CALUDE_breakfast_cost_is_17_l4062_406259

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost (muffin_price fruit_cup_price : ℕ) 
  (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins + kiera_muffins) * muffin_price + 
  (francis_fruit_cups + kiera_fruit_cups) * fruit_cup_price

/-- Theorem stating that the total cost of breakfast for Francis and Kiera is $17 -/
theorem breakfast_cost_is_17 : 
  breakfast_cost 2 3 2 2 2 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_17_l4062_406259


namespace NUMINAMATH_CALUDE_initial_marbles_count_l4062_406221

/-- Represents the number of marbles Connie had initially -/
def initial_marbles : ℕ := sorry

/-- Represents the number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- Represents the number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- Theorem stating that the initial number of marbles is 143 -/
theorem initial_marbles_count : initial_marbles = 143 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_count_l4062_406221


namespace NUMINAMATH_CALUDE_magic_trick_always_succeeds_l4062_406277

/-- Represents a box in the magic trick setup -/
structure Box :=
  (index : Fin 13)

/-- Represents the state of the magic trick setup -/
structure MagicTrickSetup :=
  (boxes : Fin 13 → Box)
  (coin_boxes : Fin 2 → Box)
  (opened_box : Box)

/-- Represents the magician's strategy -/
structure MagicianStrategy :=
  (choose_boxes : MagicTrickSetup → Fin 4 → Box)

/-- Predicate to check if a strategy is successful -/
def is_successful_strategy (strategy : MagicianStrategy) : Prop :=
  ∀ (setup : MagicTrickSetup),
    ∃ (i j : Fin 4),
      strategy.choose_boxes setup i = setup.coin_boxes 0 ∧
      strategy.choose_boxes setup j = setup.coin_boxes 1

theorem magic_trick_always_succeeds :
  ∃ (strategy : MagicianStrategy), is_successful_strategy strategy := by
  sorry

end NUMINAMATH_CALUDE_magic_trick_always_succeeds_l4062_406277


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4062_406219

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I - 3) / (Complex.I + 1) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4062_406219


namespace NUMINAMATH_CALUDE_power_division_rule_l4062_406251

theorem power_division_rule (m : ℝ) : m^7 / m^3 = m^4 := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l4062_406251


namespace NUMINAMATH_CALUDE_solution_set_theorem_l4062_406227

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the domain of f
def domain : Set ℝ := { x | x > 0 }

-- State the theorem
theorem solution_set_theorem 
  (h_deriv : ∀ x ∈ domain, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x ∈ domain, f x < -x * f' x) :
  { x ∈ domain | f (x + 1) > (x - 1) * f (x^2 - 1) } = { x | x > 2 } := by
  sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l4062_406227


namespace NUMINAMATH_CALUDE_distance_from_origin_l4062_406231

theorem distance_from_origin (x : ℝ) : |x| = 2 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l4062_406231


namespace NUMINAMATH_CALUDE_square_of_number_doubled_exceeds_fifth_l4062_406240

theorem square_of_number_doubled_exceeds_fifth : ∃ x : ℝ, 2 * x = (1/5) * x + 9 ∧ x^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_doubled_exceeds_fifth_l4062_406240


namespace NUMINAMATH_CALUDE_three_hits_in_five_shots_l4062_406247

/-- The probability of hitting the target exactly k times in n independent shots,
    where p is the probability of hitting the target in each shot. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of hitting the target exactly 3 times in 5 shots,
    where the probability of hitting the target in each shot is 0.6,
    is equal to 0.3456. -/
theorem three_hits_in_five_shots :
  binomial_probability 5 3 0.6 = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_three_hits_in_five_shots_l4062_406247


namespace NUMINAMATH_CALUDE_range_of_m_l4062_406242

theorem range_of_m (α : ℝ) (m : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sqrt 3 * Real.sin α + Real.cos α = m) :
  m ∈ Set.Ioo 1 2 ∪ {2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l4062_406242


namespace NUMINAMATH_CALUDE_fred_cards_left_l4062_406258

/-- Represents the number of baseball cards Fred has left after Melanie's purchase. -/
def cards_left (initial : ℕ) (bought : ℕ) : ℕ := initial - bought

/-- Theorem stating that Fred has 2 baseball cards left after Melanie's purchase. -/
theorem fred_cards_left : cards_left 5 3 = 2 := by sorry

end NUMINAMATH_CALUDE_fred_cards_left_l4062_406258


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l4062_406211

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l4062_406211


namespace NUMINAMATH_CALUDE_third_dog_summer_avg_distance_proof_l4062_406256

/-- Represents the average daily distance walked by the third dog in summer -/
def third_dog_summer_avg_distance : ℝ := 2.2

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the number of weekend days in a month -/
def weekend_days : ℕ := 8

/-- Represents the distance walked by the third dog on a summer weekday -/
def third_dog_summer_distance : ℝ := 3

theorem third_dog_summer_avg_distance_proof :
  third_dog_summer_avg_distance = 
    (third_dog_summer_distance * (days_in_month - weekend_days)) / days_in_month :=
by sorry

end NUMINAMATH_CALUDE_third_dog_summer_avg_distance_proof_l4062_406256


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4062_406273

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3*x - 10 > 0} = {x : ℝ | x < -2 ∨ x > 5} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4062_406273


namespace NUMINAMATH_CALUDE_height_difference_l4062_406296

theorem height_difference (height_A : ℝ) (initial_ratio : ℝ) (growth : ℝ) : 
  height_A = 72 →
  initial_ratio = 2/3 →
  growth = 10 →
  height_A - (initial_ratio * height_A + growth) = 14 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l4062_406296


namespace NUMINAMATH_CALUDE_customer_ratio_l4062_406280

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

/-- The predicted number of customers for Saturday -/
def predicted_saturday_customers : ℕ := 574

/-- The total number of customers on Friday -/
def friday_customers : ℕ := breakfast_customers + lunch_customers + dinner_customers

/-- The theorem stating the ratio of predicted Saturday customers to Friday customers -/
theorem customer_ratio : 
  (predicted_saturday_customers : ℚ) / (friday_customers : ℚ) = 574 / 287 := by
  sorry


end NUMINAMATH_CALUDE_customer_ratio_l4062_406280


namespace NUMINAMATH_CALUDE_min_triangle_area_on_unit_grid_l4062_406281

/-- The area of a triangle given three points on a 2D grid -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℤ) : ℚ :=
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- The minimum area of a triangle on a unit grid -/
theorem min_triangle_area_on_unit_grid : 
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    triangleArea x1 y1 x2 y2 x3 y3 = (1 / 2 : ℚ) ∧ 
    (∀ (a1 b1 a2 b2 a3 b3 : ℤ), triangleArea a1 b1 a2 b2 a3 b3 ≥ (1 / 2 : ℚ)) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_on_unit_grid_l4062_406281


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_34_l4062_406215

theorem consecutive_integers_sum_34 :
  ∃! (a : ℕ), a > 0 ∧ (a + (a + 1) + (a + 2) + (a + 3) = 34) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_34_l4062_406215


namespace NUMINAMATH_CALUDE_solution_set_rational_inequality_l4062_406235

theorem solution_set_rational_inequality :
  ∀ x : ℝ, x ≠ 0 → ((x - 1) / x ≥ 2 ↔ -1 ≤ x ∧ x < 0) := by sorry

end NUMINAMATH_CALUDE_solution_set_rational_inequality_l4062_406235


namespace NUMINAMATH_CALUDE_evaluate_expression_l4062_406268

theorem evaluate_expression (y : ℝ) (Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi + y^2) = 4 * Q + 10 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4062_406268


namespace NUMINAMATH_CALUDE_extra_eyes_percentage_l4062_406282

def total_frogs : ℕ := 150
def extra_eyes : ℕ := 5

def percentage_with_extra_eyes : ℚ :=
  (extra_eyes : ℚ) / (total_frogs : ℚ) * 100

def rounded_percentage : ℕ := 
  (percentage_with_extra_eyes + 1/2).floor.toNat

theorem extra_eyes_percentage :
  rounded_percentage = 3 :=
sorry

end NUMINAMATH_CALUDE_extra_eyes_percentage_l4062_406282


namespace NUMINAMATH_CALUDE_min_boxes_fit_l4062_406284

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- The large box dimensions -/
def largeBox : BoxDimensions := ⟨12, 14, 16⟩

/-- The approximate dimensions of the small irregular boxes -/
def smallBox : BoxDimensions := ⟨3, 7, 2⟩

/-- Theorem stating that at least 64 small boxes can fit into the large box -/
theorem min_boxes_fit (irreg_shape : Prop) : ∃ n : ℕ, n ≥ 64 ∧ n * boxVolume smallBox ≤ boxVolume largeBox := by
  sorry

end NUMINAMATH_CALUDE_min_boxes_fit_l4062_406284


namespace NUMINAMATH_CALUDE_adam_dried_fruits_weight_l4062_406222

def nuts_weight : ℝ := 3
def nuts_price : ℝ := 12
def dried_fruits_price : ℝ := 8
def total_cost : ℝ := 56

theorem adam_dried_fruits_weight :
  ∃ (x : ℝ), x * dried_fruits_price + nuts_weight * nuts_price = total_cost ∧ x = 2.5 := by
sorry

end NUMINAMATH_CALUDE_adam_dried_fruits_weight_l4062_406222


namespace NUMINAMATH_CALUDE_exponential_function_point_l4062_406214

theorem exponential_function_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_point_l4062_406214


namespace NUMINAMATH_CALUDE_parabola_comparison_l4062_406283

theorem parabola_comparison : ∀ x : ℝ, x^2 - x + 3 < x^2 - x + 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_comparison_l4062_406283


namespace NUMINAMATH_CALUDE_horse_food_per_day_l4062_406205

/-- Given the ratio of sheep to horses, number of sheep, and total horse food,
    prove the amount of food each horse gets per day. -/
theorem horse_food_per_day
  (sheep_horse_ratio : ℚ) -- Ratio of sheep to horses
  (num_sheep : ℕ) -- Number of sheep
  (total_horse_food : ℕ) -- Total amount of horse food in ounces
  (h1 : sheep_horse_ratio = 2 / 7) -- The ratio of sheep to horses is 2:7
  (h2 : num_sheep = 16) -- There are 16 sheep on the farm
  (h3 : total_horse_food = 12880) -- The farm needs 12,880 ounces of horse food per day
  : ℕ :=
by
  sorry

#check horse_food_per_day

end NUMINAMATH_CALUDE_horse_food_per_day_l4062_406205


namespace NUMINAMATH_CALUDE_louisa_travel_l4062_406246

/-- Louisa's travel problem -/
theorem louisa_travel (average_speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  average_speed = 40 →
  second_day_distance = 280 →
  time_difference = 3 →
  let second_day_time := second_day_distance / average_speed
  let first_day_time := second_day_time - time_difference
  let first_day_distance := average_speed * first_day_time
  first_day_distance = 160 := by
  sorry

end NUMINAMATH_CALUDE_louisa_travel_l4062_406246


namespace NUMINAMATH_CALUDE_grapes_purchased_l4062_406233

/-- The problem of calculating the amount of grapes purchased -/
theorem grapes_purchased (grape_cost mango_cost total_paid : ℕ) (mango_amount : ℕ) : 
  grape_cost = 70 →
  mango_amount = 9 →
  mango_cost = 65 →
  total_paid = 1145 →
  ∃ (grape_amount : ℕ), grape_amount * grape_cost + mango_amount * mango_cost = total_paid ∧ grape_amount = 8 :=
by sorry

end NUMINAMATH_CALUDE_grapes_purchased_l4062_406233


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l4062_406225

/-- 
Proves that an arithmetic sequence with first term 2, common difference 3, 
and last term 2014 has 671 terms.
-/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ),
    a = 2 → d = 3 → last = 2014 → 
    last = a + (n - 1) * d → n = 671 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l4062_406225
