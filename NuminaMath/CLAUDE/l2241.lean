import Mathlib

namespace NUMINAMATH_CALUDE_one_pole_inside_l2241_224177

/-- Represents a non-convex polygon fence -/
structure Fence where
  is_non_convex : Bool

/-- Represents a power line with poles -/
structure PowerLine where
  total_poles : Nat

/-- Represents a spy walking around the fence -/
structure Spy where
  counted_poles : Nat

/-- Theorem stating that given the conditions, there is one pole inside the fence -/
theorem one_pole_inside (fence : Fence) (power_line : PowerLine) (spy : Spy) :
  fence.is_non_convex ∧
  power_line.total_poles = 36 ∧
  spy.counted_poles = 2015 →
  ∃ (poles_inside : Nat), poles_inside = 1 :=
sorry

end NUMINAMATH_CALUDE_one_pole_inside_l2241_224177


namespace NUMINAMATH_CALUDE_symmetric_line_l2241_224136

/-- Given a point (x, y) on the line y = -x + 2, prove that it is symmetric to a point on the line y = x about the line x = 1 -/
theorem symmetric_line (x y : ℝ) : 
  y = -x + 2 → 
  ∃ (x' y' : ℝ), 
    (y' = x') ∧  -- Point (x', y') is on the line y = x
    ((x + x') / 2 = 1) ∧  -- Midpoint of x and x' is on the line x = 1
    (y = y') -- y-coordinates are the same
    := by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2241_224136


namespace NUMINAMATH_CALUDE_mean_of_additional_numbers_l2241_224195

theorem mean_of_additional_numbers
  (original_count : Nat)
  (original_mean : ℝ)
  (new_count : Nat)
  (new_mean : ℝ)
  (h1 : original_count = 7)
  (h2 : original_mean = 72)
  (h3 : new_count = 9)
  (h4 : new_mean = 80) :
  let x_plus_y := new_count * new_mean - original_count * original_mean
  let mean_x_y := x_plus_y / 2
  mean_x_y = 108 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_additional_numbers_l2241_224195


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2241_224152

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x = 2 ∧ x^2 / a^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = 3/2) →
  a^2 = 4 ∧ b^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2241_224152


namespace NUMINAMATH_CALUDE_real_part_reciprocal_l2241_224182

theorem real_part_reciprocal (z : ℂ) (h1 : z ≠ (z.re : ℂ)) (h2 : Complex.abs z = 2) :
  ((2 - z)⁻¹).re = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_reciprocal_l2241_224182


namespace NUMINAMATH_CALUDE_largest_multiple_of_3_and_5_under_800_l2241_224159

theorem largest_multiple_of_3_and_5_under_800 : 
  ∀ n : ℕ, n < 800 ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ 795 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_3_and_5_under_800_l2241_224159


namespace NUMINAMATH_CALUDE_response_rate_percentage_l2241_224113

def responses_needed : ℕ := 300
def questionnaires_mailed : ℕ := 600

theorem response_rate_percentage : 
  (responses_needed : ℚ) / questionnaires_mailed * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l2241_224113


namespace NUMINAMATH_CALUDE_special_polynomial_form_l2241_224175

/-- A polynomial of two variables satisfying specific conditions -/
structure SpecialPolynomial where
  P : ℝ → ℝ → ℝ
  n : ℕ+
  homogeneous : ∀ (t x y : ℝ), P (t * x) (t * y) = t ^ n.val * P x y
  sum_condition : ∀ (a b c : ℝ), P (a + b) c + P (b + c) a + P (c + a) b = 0
  normalization : P 1 0 = 1

/-- The theorem stating the form of the special polynomial -/
theorem special_polynomial_form (sp : SpecialPolynomial) :
  ∃ (n : ℕ+), ∀ (x y : ℝ), sp.P x y = (x - 2 * y) * (x + y) ^ (n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l2241_224175


namespace NUMINAMATH_CALUDE_garden_radius_increase_l2241_224164

theorem garden_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 40)
  (h2 : final_circumference = 50) :
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_garden_radius_increase_l2241_224164


namespace NUMINAMATH_CALUDE_second_larger_perfect_square_l2241_224137

theorem second_larger_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m ^ 2) ∧
  (∀ y : ℕ, y > x ∧ (∃ l : ℕ, y = l ^ 2) → y ≥ n) ∧
  n = x + 4 * (x.sqrt) + 4 :=
sorry

end NUMINAMATH_CALUDE_second_larger_perfect_square_l2241_224137


namespace NUMINAMATH_CALUDE_sequence_properties_l2241_224188

def S (n : ℕ) : ℤ := n^2 - 9*n

def a (n : ℕ) : ℤ := 2*n - 10

theorem sequence_properties :
  (∀ n : ℕ, S (n+1) - S n = a (n+1)) ∧
  (∃! k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8 ∧ k = 8) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2241_224188


namespace NUMINAMATH_CALUDE_barking_ratio_is_one_fourth_l2241_224180

/-- Represents the state of dogs in a park -/
structure DogPark where
  total : ℕ
  running : ℕ
  playing : ℕ
  idle : ℕ

/-- The ratio of barking dogs to total dogs -/
def barkingRatio (park : DogPark) : Rat :=
  let barking := park.total - (park.running + park.playing + park.idle)
  barking / park.total

/-- Theorem stating the barking ratio in the given scenario -/
theorem barking_ratio_is_one_fourth :
  ∃ (park : DogPark),
    park.total = 88 ∧
    park.running = 12 ∧
    park.playing = 44 ∧
    park.idle = 10 ∧
    barkingRatio park = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_barking_ratio_is_one_fourth_l2241_224180


namespace NUMINAMATH_CALUDE_complex_arithmetic_l2241_224193

/-- Given complex numbers B, Q, R, and T, prove that 2(B - Q + R + T) = 18 + 10i -/
theorem complex_arithmetic (B Q R T : ℂ) 
  (hB : B = 3 + 2*I)
  (hQ : Q = -5)
  (hR : R = -2*I)
  (hT : T = 1 + 5*I) :
  2 * (B - Q + R + T) = 18 + 10*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l2241_224193


namespace NUMINAMATH_CALUDE_triangle_problem_l2241_224127

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  A = 2 * π / 3 →  -- 120° in radians
  c > b →
  a = Real.sqrt 21 →
  S = Real.sqrt 3 →
  (1/2) * b * c * Real.sin A = S →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (∃ (B C : ℝ), A + B + C = π ∧ 
    a / Real.sin A = b / Real.sin B ∧
    Real.cos C = (a^2 + b^2 - c^2) / (2*a*b)) →
  (b = 1 ∧ c = 4) ∧
  Real.sin B + Real.cos C = (Real.sqrt 7 + 2 * Real.sqrt 21) / 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2241_224127


namespace NUMINAMATH_CALUDE_x_range_l2241_224110

theorem x_range (x : ℝ) (h1 : (1 : ℝ) / x < 3) (h2 : (1 : ℝ) / x > -4) : 
  x > 1/3 ∨ x < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2241_224110


namespace NUMINAMATH_CALUDE_trapezoid_sides_l2241_224158

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithCircle where
  r : ℝ  -- radius of the inscribed circle
  a : ℝ  -- shorter base
  b : ℝ  -- longer base
  c : ℝ  -- left side
  d : ℝ  -- right side (hypotenuse)
  h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r > 0  -- all lengths are positive
  ha : a = 4*r/3  -- shorter base condition
  hsum : a + b = c + d  -- sum of bases equals sum of non-parallel sides
  hright : c^2 + a^2 = d^2  -- right angle condition

/-- The sides of the trapezoid are 2r, 4r/3, 10r/3, and 4r -/
theorem trapezoid_sides (t : RightTrapezoidWithCircle) :
  t.c = 2*t.r ∧ t.a = 4*t.r/3 ∧ t.b = 10*t.r/3 ∧ t.d = 4*t.r :=
sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l2241_224158


namespace NUMINAMATH_CALUDE_rupert_candles_l2241_224194

/-- Given that Peter has 10 candles on his cake and Rupert is 3.5 times older than Peter,
    prove that Rupert's cake will have 35 candles. -/
theorem rupert_candles (peter_candles : ℕ) (age_ratio : ℚ) : ℕ :=
  by
  -- Define Peter's candles
  have h1 : peter_candles = 10 := by sorry
  -- Define the age ratio between Rupert and Peter
  have h2 : age_ratio = 3.5 := by sorry
  -- Calculate Rupert's candles
  have h3 : ↑peter_candles * age_ratio = 35 := by sorry
  -- Prove that Rupert's candles equal 35
  exact 35

end NUMINAMATH_CALUDE_rupert_candles_l2241_224194


namespace NUMINAMATH_CALUDE_boys_camp_total_l2241_224145

theorem boys_camp_total (total : ℕ) : 
  (total * 20 / 100 : ℕ) > 0 →  -- Ensure there are boys from school A
  (((total * 20 / 100) * 70 / 100 : ℕ) = 35) →  -- 35 boys from school A not studying science
  total = 250 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l2241_224145


namespace NUMINAMATH_CALUDE_compound_interest_rate_l2241_224138

/-- Compound interest calculation -/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (I : ℝ) (r : ℝ) : 
  P = 6000 → t = 2 → I = 1260.000000000001 → 
  P * (1 + r)^t = P + I → r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l2241_224138


namespace NUMINAMATH_CALUDE_pebbles_count_l2241_224125

/-- The width of a splash made by a pebble in meters -/
def pebble_splash : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash : ℚ := 2

/-- The number of rocks tossed -/
def rocks_tossed : ℕ := 3

/-- The number of boulders tossed -/
def boulders_tossed : ℕ := 2

/-- The total width of all splashes in meters -/
def total_splash_width : ℚ := 7

/-- The number of pebbles tossed -/
def pebbles_tossed : ℕ := 6

theorem pebbles_count :
  pebbles_tossed * pebble_splash + 
  rocks_tossed * rock_splash + 
  boulders_tossed * boulder_splash = 
  total_splash_width := by sorry

end NUMINAMATH_CALUDE_pebbles_count_l2241_224125


namespace NUMINAMATH_CALUDE_quadratic_j_value_l2241_224181

-- Define the quadratic function
def quadratic (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem quadratic_j_value (p q r : ℝ) :
  (∃ m n : ℝ, ∀ x : ℝ, 4 * (quadratic p q r x) = m * (x - 5)^2 + n) →
  (∃ m n : ℝ, ∀ x : ℝ, quadratic p q r x = 3 * (x - 5)^2 + 15) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_j_value_l2241_224181


namespace NUMINAMATH_CALUDE_alternating_hexagon_area_l2241_224139

/-- A hexagon with alternating side lengths and specified corner triangles -/
structure AlternatingHexagon where
  short_side : ℝ
  long_side : ℝ
  corner_triangle_base : ℝ
  corner_triangle_altitude : ℝ

/-- The area of an alternating hexagon -/
def area (h : AlternatingHexagon) : ℝ := sorry

/-- Theorem: The area of the specified hexagon is 36 square units -/
theorem alternating_hexagon_area :
  let h : AlternatingHexagon := {
    short_side := 2,
    long_side := 4,
    corner_triangle_base := 2,
    corner_triangle_altitude := 3
  }
  area h = 36 := by sorry

end NUMINAMATH_CALUDE_alternating_hexagon_area_l2241_224139


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l2241_224173

-- Define the fixed circle
def fixed_circle (x y : ℝ) : Prop := (x - 5)^2 + (y + 7)^2 = 16

-- Define the moving circle with radius 1
def moving_circle (center_x center_y : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 1

-- Define the tangency condition
def is_tangent (center_x center_y : ℝ) : Prop :=
  ∃ x y : ℝ, fixed_circle x y ∧ moving_circle center_x center_y x y

-- Theorem statement
theorem trajectory_of_moving_circle_center :
  ∀ center_x center_y : ℝ,
    is_tangent center_x center_y →
    ((center_x - 5)^2 + (center_y + 7)^2 = 25 ∨
     (center_x - 5)^2 + (center_y + 7)^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l2241_224173


namespace NUMINAMATH_CALUDE_ray_AB_not_equal_ray_BA_l2241_224157

-- Define a point type
def Point := ℝ × ℝ

-- Define a ray type
structure Ray where
  start : Point
  direction : Point

-- Define an equality relation for rays
def ray_eq (r1 r2 : Ray) : Prop :=
  r1.start = r2.start ∧ r1.direction = r2.direction

-- Theorem statement
theorem ray_AB_not_equal_ray_BA (A B : Point) (h : A ≠ B) :
  ¬(ray_eq (Ray.mk A B) (Ray.mk B A)) := by
  sorry

end NUMINAMATH_CALUDE_ray_AB_not_equal_ray_BA_l2241_224157


namespace NUMINAMATH_CALUDE_tomato_suggestion_count_tomato_suggestion_count_proof_l2241_224128

theorem tomato_suggestion_count : ℕ → ℕ → ℕ → Prop :=
  fun bacon_count difference tomato_count =>
    (bacon_count = tomato_count + difference) →
    (bacon_count = 337 ∧ difference = 314) →
    tomato_count = 23

theorem tomato_suggestion_count_proof :
  ∃ (tomato_count : ℕ), tomato_suggestion_count 337 314 tomato_count :=
sorry

end NUMINAMATH_CALUDE_tomato_suggestion_count_tomato_suggestion_count_proof_l2241_224128


namespace NUMINAMATH_CALUDE_remainder_proof_l2241_224156

theorem remainder_proof : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2241_224156


namespace NUMINAMATH_CALUDE_room_problem_l2241_224197

theorem room_problem (boys girls : ℕ) : 
  boys = 3 * girls ∧ 
  (boys - 4) = 5 * (girls - 4) →
  boys + girls = 32 :=
by sorry

end NUMINAMATH_CALUDE_room_problem_l2241_224197


namespace NUMINAMATH_CALUDE_target_miss_probability_l2241_224196

theorem target_miss_probability (p_I p_II p_III : ℝ) 
  (h_I : p_I = 0.35) 
  (h_II : p_II = 0.30) 
  (h_III : p_III = 0.25) : 
  1 - (p_I + p_II + p_III) = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_target_miss_probability_l2241_224196


namespace NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l2241_224119

theorem negative_integer_sum_square_twelve (M : ℤ) : 
  M < 0 → M^2 + M = 12 → M = -4 := by sorry

end NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l2241_224119


namespace NUMINAMATH_CALUDE_no_existence_of_complex_numbers_l2241_224185

theorem no_existence_of_complex_numbers : ¬∃ (a b c : ℂ) (h : ℕ), 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  (∀ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) → 
    Complex.abs (1 + k • a + l • b + m • c) > 1 / h) := by
  sorry


end NUMINAMATH_CALUDE_no_existence_of_complex_numbers_l2241_224185


namespace NUMINAMATH_CALUDE_perimeter_of_non_shaded_region_l2241_224169

/-- A structure representing the geometrical figure described in the problem -/
structure Figure where
  outer_rectangle_length : ℝ
  outer_rectangle_width : ℝ
  small_rectangle_side : ℝ
  shaded_square_side : ℝ
  shaded_rectangle_length : ℝ
  shaded_rectangle_width : ℝ
  shaded_area : ℝ

/-- The theorem statement based on the problem -/
theorem perimeter_of_non_shaded_region
  (fig : Figure)
  (h1 : fig.outer_rectangle_length = 12)
  (h2 : fig.outer_rectangle_width = 9)
  (h3 : fig.small_rectangle_side = 3)
  (h4 : fig.shaded_square_side = 3)
  (h5 : fig.shaded_rectangle_length = 3)
  (h6 : fig.shaded_rectangle_width = 2)
  (h7 : fig.shaded_area = 65)
  : ∃ (p : ℝ), p = 30 ∧ p = 2 * (12 + 3) :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_non_shaded_region_l2241_224169


namespace NUMINAMATH_CALUDE_pizza_slices_left_l2241_224172

theorem pizza_slices_left (total_slices : ℕ) (fraction_eaten : ℚ) : 
  total_slices = 16 → fraction_eaten = 3/4 → total_slices - (total_slices * fraction_eaten).floor = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l2241_224172


namespace NUMINAMATH_CALUDE_expression_simplification_l2241_224167

theorem expression_simplification (a : ℝ) (h : a^2 + 3*a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2*a)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2241_224167


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2241_224141

theorem decimal_to_fraction : (2.25 : ℚ) = 9 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2241_224141


namespace NUMINAMATH_CALUDE_snow_volume_on_blocked_sidewalk_l2241_224183

/-- Calculates the volume of snow to shovel from a partially blocked rectangular sidewalk. -/
theorem snow_volume_on_blocked_sidewalk
  (total_length : ℝ)
  (width : ℝ)
  (blocked_length : ℝ)
  (snow_depth : ℝ)
  (h1 : total_length = 30)
  (h2 : width = 3)
  (h3 : blocked_length = 5)
  (h4 : snow_depth = 2/3)
  : (total_length - blocked_length) * width * snow_depth = 50 := by
  sorry

#check snow_volume_on_blocked_sidewalk

end NUMINAMATH_CALUDE_snow_volume_on_blocked_sidewalk_l2241_224183


namespace NUMINAMATH_CALUDE_mrs_heine_biscuits_l2241_224144

/-- Calculates the total number of biscuits needed for Mrs. Heine's pets -/
def total_biscuits (num_dogs : ℕ) (num_cats : ℕ) (num_birds : ℕ) 
                   (biscuits_per_dog : ℕ) (biscuits_per_cat : ℕ) (biscuits_per_bird : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog + num_cats * biscuits_per_cat + num_birds * biscuits_per_bird

/-- Theorem stating that Mrs. Heine needs to buy 11 biscuits in total -/
theorem mrs_heine_biscuits : 
  total_biscuits 2 1 3 3 2 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_biscuits_l2241_224144


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l2241_224191

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

-- Define the relationship between ¬p and ¬q
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x : ℝ, ¬(q x) → ¬(p x)) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l2241_224191


namespace NUMINAMATH_CALUDE_trail_length_proof_l2241_224186

/-- The total length of a trail where two friends walk from opposite ends, with one friend 20% faster than the other, and the faster friend walks 12 km when they meet. -/
def trail_length : ℝ := 22

/-- The distance walked by the faster friend when they meet. -/
def faster_friend_distance : ℝ := 12

/-- The ratio of the faster friend's speed to the slower friend's speed. -/
def speed_ratio : ℝ := 1.2

theorem trail_length_proof :
  ∃ (v : ℝ), v > 0 ∧
    trail_length = faster_friend_distance + v * (faster_friend_distance / (speed_ratio * v)) :=
by sorry

end NUMINAMATH_CALUDE_trail_length_proof_l2241_224186


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2241_224160

/-- Given a curve C with equation 4x^2 + 9y^2 = 36, 
    the maximum value of 3x + 4y for any point (x,y) on C is √145. -/
theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 145 ∧ 
  (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 36 → 3 * x + 4 * y ≤ M) ∧
  (∃ (x y : ℝ), 4 * x^2 + 9 * y^2 = 36 ∧ 3 * x + 4 * y = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2241_224160


namespace NUMINAMATH_CALUDE_f_one_geq_25_l2241_224176

/-- A function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x ≤ f m y

theorem f_one_geq_25 (m : ℝ) (h : is_increasing_on_interval m) : f m 1 ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_f_one_geq_25_l2241_224176


namespace NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l2241_224133

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

theorem arithmetic_seq_fifth_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_seq a)
  (h_eq : 2 * a 9 = a 12 + 6) :
  a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l2241_224133


namespace NUMINAMATH_CALUDE_probability_matching_letter_l2241_224100

def word1 : String := "MATHEMATICS"
def word2 : String := "CALCULUS"

def is_in_word2 (c : Char) : Bool :=
  word2.contains c

def count_matching_letters : Nat :=
  word1.toList.filter is_in_word2 |>.length

theorem probability_matching_letter :
  (count_matching_letters : ℚ) / word1.length = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_matching_letter_l2241_224100


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2241_224109

theorem quadratic_equation_solution (C : ℝ) (h : C = 3) :
  ∃ x : ℝ, 3 * x^2 - 6 * x + C = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2241_224109


namespace NUMINAMATH_CALUDE_specific_structure_surface_area_l2241_224179

/-- Represents a complex structure composed of unit cubes -/
structure CubeStructure where
  num_cubes : ℕ
  height : ℕ
  length : ℕ
  width : ℕ

/-- Calculates the surface area of a cube structure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.length * s.width + s.length * s.height + s.width * s.height)

/-- Theorem stating that a specific cube structure has a surface area of 84 square units -/
theorem specific_structure_surface_area :
  ∃ (s : CubeStructure), s.num_cubes = 15 ∧ s.height = 4 ∧ s.length = 5 ∧ s.width = 3 ∧
  surface_area s = 84 :=
sorry

end NUMINAMATH_CALUDE_specific_structure_surface_area_l2241_224179


namespace NUMINAMATH_CALUDE_rectangle_breadth_l2241_224178

/-- 
Given a rectangle where:
1. The area is 24 times its breadth
2. The difference between the length and the breadth is 10 meters
Prove that the breadth is 14 meters
-/
theorem rectangle_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 24 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l2241_224178


namespace NUMINAMATH_CALUDE_cross_section_distance_in_pyramid_l2241_224161

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Theorem about the distance of a cross section in a right hexagonal pyramid -/
theorem cross_section_distance_in_pyramid 
  (pyramid : RightHexagonalPyramid)
  (section1 section2 : CrossSection) :
  section1.area = 216 * Real.sqrt 3 →
  section2.area = 486 * Real.sqrt 3 →
  |section1.distance_from_apex - section2.distance_from_apex| = 8 →
  section2.distance_from_apex = 24 :=
by sorry

end NUMINAMATH_CALUDE_cross_section_distance_in_pyramid_l2241_224161


namespace NUMINAMATH_CALUDE_profit_difference_theorem_l2241_224115

/-- Represents the profit distribution for a business partnership --/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of a and c --/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.a_investment + pd.b_investment + pd.c_investment
  let total_profit := pd.b_profit * total_investment / pd.b_investment
  let a_profit := total_profit * pd.a_investment / total_investment
  let c_profit := total_profit * pd.c_investment / total_investment
  c_profit - a_profit

/-- Theorem stating the difference between profit shares of a and c --/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 3500) :
  profit_difference pd = 1400 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_theorem_l2241_224115


namespace NUMINAMATH_CALUDE_legos_given_to_sister_l2241_224149

theorem legos_given_to_sister (initial : ℕ) (lost : ℕ) (current : ℕ) : 
  initial = 380 → lost = 57 → current = 299 → initial - lost - current = 24 :=
by sorry

end NUMINAMATH_CALUDE_legos_given_to_sister_l2241_224149


namespace NUMINAMATH_CALUDE_fraction_simplification_l2241_224184

theorem fraction_simplification :
  (5 : ℝ) / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108) = (5 * Real.sqrt 3) / 54 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2241_224184


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l2241_224163

/-- Given a function f(x) = x², prove that the derivative of f at x = -1 is -2. -/
theorem derivative_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = x^2) :
  deriv f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l2241_224163


namespace NUMINAMATH_CALUDE_machine_chip_production_l2241_224147

/-- The number of computer chips produced by a machine in a day, given the number of
    video game consoles it can supply chips for and the number of chips per console. -/
def chips_per_day (consoles_per_day : ℕ) (chips_per_console : ℕ) : ℕ :=
  consoles_per_day * chips_per_console

/-- Theorem stating that a machine supplying chips for 93 consoles per day,
    with 5 chips per console, produces 465 chips per day. -/
theorem machine_chip_production :
  chips_per_day 93 5 = 465 := by
  sorry

end NUMINAMATH_CALUDE_machine_chip_production_l2241_224147


namespace NUMINAMATH_CALUDE_original_price_l2241_224132

theorem original_price (p q : ℝ) (h : p ≠ 0 ∧ q ≠ 0) :
  let x := (20000 : ℝ) / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  let final_price := x * (1 + p/100) * (1 + q/100) * (1 - q/100) * (1 - p/100)
  final_price = 2 := by sorry

end NUMINAMATH_CALUDE_original_price_l2241_224132


namespace NUMINAMATH_CALUDE_equation_solution_l2241_224140

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2241_224140


namespace NUMINAMATH_CALUDE_complex_number_value_l2241_224199

theorem complex_number_value (z : ℂ) (h1 : z^2 = 6*z - 27 + 12*I) (h2 : ∃ (n : ℕ), Complex.abs z = n) :
  z = 3 + (Real.sqrt 6 + Real.sqrt 6 * I) ∨ z = 3 - (Real.sqrt 6 + Real.sqrt 6 * I) :=
sorry

end NUMINAMATH_CALUDE_complex_number_value_l2241_224199


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2241_224143

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2241_224143


namespace NUMINAMATH_CALUDE_ones_and_seven_primality_l2241_224131

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def ones_and_seven (n : ℕ) : ℕ :=
  if n = 1 then 7
  else (10^(n-1) - 1) / 9 + 7 * 10^((n-1) / 2)

theorem ones_and_seven_primality (n : ℕ) :
  is_prime (ones_and_seven n) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_ones_and_seven_primality_l2241_224131


namespace NUMINAMATH_CALUDE_hash_difference_l2241_224187

/-- Custom operation # defined as x#y = xy + 2x -/
def hash (x y : ℤ) : ℤ := x * y + 2 * x

/-- Theorem stating that (5#3) - (3#5) = 4 -/
theorem hash_difference : hash 5 3 - hash 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l2241_224187


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l2241_224122

/-- A point on the unit circle reached by moving counterclockwise from (1,0) along an arc length of 2π/3 has coordinates (-1/2, √3/2). -/
theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Real.cos (2 * Real.pi / 3) = Q.1 ∧ Real.sin (2 * Real.pi / 3) = Q.2) →  -- Q is reached by moving 2π/3 radians counterclockwise from (1,0)
  (Q.1 = -1/2 ∧ Q.2 = Real.sqrt 3 / 2) :=  -- Q has coordinates (-1/2, √3/2)
by sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l2241_224122


namespace NUMINAMATH_CALUDE_highlighter_difference_l2241_224118

theorem highlighter_difference (total pink blue yellow : ℕ) : 
  total = 40 →
  yellow = 7 →
  blue = pink + 5 →
  total = yellow + pink + blue →
  pink - yellow = 7 := by
sorry

end NUMINAMATH_CALUDE_highlighter_difference_l2241_224118


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2241_224189

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 3)
  (h_sum2 : a 2 + a 3 = 6) :
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2241_224189


namespace NUMINAMATH_CALUDE_solve_equation_l2241_224135

theorem solve_equation : ∃ x : ℝ, 3 * x + 15 = (1/3) * (6 * x + 45) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2241_224135


namespace NUMINAMATH_CALUDE_log_ratio_problem_l2241_224155

theorem log_ratio_problem (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h_log : Real.log p / Real.log 16 = Real.log q / Real.log 20 ∧ 
           Real.log p / Real.log 16 = Real.log (p + q) / Real.log 25) : 
  p / q = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_problem_l2241_224155


namespace NUMINAMATH_CALUDE_soccer_team_bottles_l2241_224165

theorem soccer_team_bottles (total_bottles : ℕ) (football_players : ℕ) (football_bottles_per_player : ℕ)
  (lacrosse_extra_bottles : ℕ) (rugby_bottles : ℕ) :
  total_bottles = 254 →
  football_players = 11 →
  football_bottles_per_player = 6 →
  lacrosse_extra_bottles = 12 →
  rugby_bottles = 49 →
  total_bottles - (football_players * football_bottles_per_player + 
    (football_players * football_bottles_per_player + lacrosse_extra_bottles) + 
    rugby_bottles) = 61 := by
  sorry

#check soccer_team_bottles

end NUMINAMATH_CALUDE_soccer_team_bottles_l2241_224165


namespace NUMINAMATH_CALUDE_cycle_gains_and_overall_gain_l2241_224101

def cycle1_purchase : ℚ := 900
def cycle1_sale : ℚ := 1440
def cycle2_purchase : ℚ := 1200
def cycle2_sale : ℚ := 1680
def cycle3_purchase : ℚ := 1500
def cycle3_sale : ℚ := 1950

def gain_percentage (purchase : ℚ) (sale : ℚ) : ℚ :=
  ((sale - purchase) / purchase) * 100

def total_purchase : ℚ := cycle1_purchase + cycle2_purchase + cycle3_purchase
def total_sale : ℚ := cycle1_sale + cycle2_sale + cycle3_sale

theorem cycle_gains_and_overall_gain :
  (gain_percentage cycle1_purchase cycle1_sale = 60) ∧
  (gain_percentage cycle2_purchase cycle2_sale = 40) ∧
  (gain_percentage cycle3_purchase cycle3_sale = 30) ∧
  (gain_percentage total_purchase total_sale = 40 + 5/6) :=
sorry

end NUMINAMATH_CALUDE_cycle_gains_and_overall_gain_l2241_224101


namespace NUMINAMATH_CALUDE_girls_who_left_l2241_224104

theorem girls_who_left (initial_boys : ℕ) (initial_girls : ℕ) (final_students : ℕ) :
  initial_boys = 24 →
  initial_girls = 14 →
  final_students = 30 →
  ∃ (left_girls : ℕ),
    left_girls = initial_girls - (final_students - (initial_boys - left_girls)) ∧
    left_girls = 4 := by
  sorry

end NUMINAMATH_CALUDE_girls_who_left_l2241_224104


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2241_224171

theorem smallest_square_containing_circle (r : ℝ) (h : r = 4) :
  (2 * r) ^ 2 = 64 := by sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2241_224171


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2241_224168

theorem sphere_surface_area (r : ℝ) (h : r > 0) :
  let plane_distance : ℝ := 3
  let section_area : ℝ := 16 * Real.pi
  let section_radius : ℝ := (section_area / Real.pi).sqrt
  r * r = plane_distance * plane_distance + section_radius * section_radius →
  4 * Real.pi * r * r = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2241_224168


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2241_224190

theorem complex_equation_solution (a : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a^2 - a : ℂ) + (3*a - 1 : ℂ) * i = 2 + 5*i →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2241_224190


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l2241_224166

theorem root_sum_reciprocal_products (p q r s t : ℂ) : 
  p^5 - 4*p^4 + 7*p^3 - 3*p^2 + p - 1 = 0 →
  q^5 - 4*q^4 + 7*q^3 - 3*q^2 + q - 1 = 0 →
  r^5 - 4*r^4 + 7*r^3 - 3*r^2 + r - 1 = 0 →
  s^5 - 4*s^4 + 7*s^3 - 3*s^2 + s - 1 = 0 →
  t^5 - 4*t^4 + 7*t^3 - 3*t^2 + t - 1 = 0 →
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 → t ≠ 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 7 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l2241_224166


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l2241_224130

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -3457 [ZMOD 13] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l2241_224130


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l2241_224103

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 4 + 8 = 11) → initial_people = 7 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l2241_224103


namespace NUMINAMATH_CALUDE_sequence_general_term_l2241_224117

theorem sequence_general_term 
  (a : ℕ+ → ℝ) 
  (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = 3 * n.val ^ 2 - 2 * n.val) :
  ∀ n : ℕ+, a n = 6 * n.val - 5 :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2241_224117


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_f_geq_three_halves_l2241_224123

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |a*x + 1| + |x - a|
def g (x : ℝ) : ℝ := x^2 + x

-- State the theorems
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, g x ≥ f 1 x ↔ x ≤ -3 ∨ x ≥ 1 := by sorry

theorem range_of_a_for_f_geq_three_halves :
  (∀ x : ℝ, f a x ≥ 3/2) → a ≥ Real.sqrt 2 / 2 := by sorry

-- Note: We assume 'a' is positive as given in the original problem
variable (a : ℝ) (ha : a > 0)

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_f_geq_three_halves_l2241_224123


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l2241_224107

/-- The probability of selecting at least one female student when randomly choosing 2 students
    from a group of 3 males and 1 female is equal to 1/2. -/
theorem prob_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (team_size : ℕ) (h1 : total_students = male_students + female_students) 
  (h2 : male_students = 3) (h3 : female_students = 1) (h4 : team_size = 2) :
  1 - (Nat.choose male_students team_size : ℚ) / (Nat.choose total_students team_size : ℚ) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l2241_224107


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l2241_224153

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ
  fixed_cost : ℝ

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails) - c.fixed_cost

/-- Theorem: The expected worth of the specific unfair coin is -1/3 -/
theorem expected_worth_unfair_coin :
  let c : UnfairCoin := {
    prob_heads := 1/3,
    prob_tails := 2/3,
    gain_heads := 6,
    loss_tails := 2,
    fixed_cost := 1
  }
  expected_worth c = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l2241_224153


namespace NUMINAMATH_CALUDE_russian_doll_price_l2241_224150

theorem russian_doll_price (original_quantity : ℕ) (discounted_quantity : ℕ) (discounted_price : ℚ) :
  original_quantity = 15 →
  discounted_quantity = 20 →
  discounted_price = 3 →
  (discounted_quantity * discounted_price) / original_quantity = 4 := by
  sorry

end NUMINAMATH_CALUDE_russian_doll_price_l2241_224150


namespace NUMINAMATH_CALUDE_andy_cake_profit_l2241_224192

/-- Calculates the profit per cake given the total ingredient cost for two cakes,
    the packaging cost per cake, and the selling price per cake. -/
def profit_per_cake (ingredient_cost_for_two : ℚ) (packaging_cost : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price - (ingredient_cost_for_two / 2 + packaging_cost)

/-- Theorem stating that for Andy's cake business, given the specific costs and selling price,
    the profit per cake is $8. -/
theorem andy_cake_profit :
  profit_per_cake 12 1 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_andy_cake_profit_l2241_224192


namespace NUMINAMATH_CALUDE_log_difference_equals_two_l2241_224120

theorem log_difference_equals_two :
  (Real.log 80 / Real.log 2) / (Real.log 2 / Real.log 40) -
  (Real.log 160 / Real.log 2) / (Real.log 2 / Real.log 20) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_two_l2241_224120


namespace NUMINAMATH_CALUDE_chair_cost_l2241_224108

/-- Proves that the cost of one chair is $11 given the conditions of Nadine's garage sale purchase. -/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  ∃ (chair_cost : ℕ), 
    chair_cost * num_chairs = total_spent - table_cost ∧
    chair_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_chair_cost_l2241_224108


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2241_224102

/-- The expression to be simplified -/
def original_expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 5 * (x^3 - 2*x^2 + 4*x - 1)

/-- The fully simplified form of the expression -/
def simplified_expression (x : ℝ) : ℝ := -5*x^3 + 13*x^2 - 29*x + 14

/-- The coefficients of the simplified expression -/
def coefficients : List ℝ := [-5, 13, -29, 14]

/-- Theorem stating that the sum of squares of coefficients equals 1231 -/
theorem sum_of_squares_of_coefficients :
  (coefficients.map (λ c => c^2)).sum = 1231 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2241_224102


namespace NUMINAMATH_CALUDE_james_profit_l2241_224124

/-- Calculates the profit from selling toys --/
def calculate_profit (initial_quantity : ℕ) (buy_price sell_price : ℚ) (sell_percentage : ℚ) : ℚ :=
  let total_cost := initial_quantity * buy_price
  let sold_quantity := (initial_quantity : ℚ) * sell_percentage
  let total_revenue := sold_quantity * sell_price
  total_revenue - total_cost

/-- Proves that James' profit is $800 --/
theorem james_profit :
  calculate_profit 200 20 30 (4/5) = 800 := by
  sorry

end NUMINAMATH_CALUDE_james_profit_l2241_224124


namespace NUMINAMATH_CALUDE_correct_average_weight_l2241_224134

/-- Given a class of boys with an initially miscalculated average weight and a single misread weight, 
    calculate the correct average weight. -/
theorem correct_average_weight 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) 
  (h1 : n = 20) 
  (h2 : initial_avg = 58.4) 
  (h3 : misread_weight = 56) 
  (h4 : correct_weight = 61) : 
  (n * initial_avg + (correct_weight - misread_weight)) / n = 58.65 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l2241_224134


namespace NUMINAMATH_CALUDE_factor_expression_l2241_224111

theorem factor_expression (t : ℝ) : 4 * t^2 - 144 + 8 = 4 * (t^2 - 34) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2241_224111


namespace NUMINAMATH_CALUDE_point_inside_circle_l2241_224112

theorem point_inside_circle (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 ↔ -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2241_224112


namespace NUMINAMATH_CALUDE_right_trapezoid_bases_l2241_224129

/-- 
Given a right trapezoid with lateral sides c and d (c < d), if a line parallel to its bases 
splits it into two smaller trapezoids each with an inscribed circle, then the bases of the 
original trapezoid are (√(d+c) + √(d-c))² / 4 and (√(d+c) - √(d-c))² / 4.
-/
theorem right_trapezoid_bases (c d : ℝ) (h : c < d) : 
  ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    (∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ 
      y^2 = x * z ∧
      c + d = x + 2*y + z) →
    x = ((Real.sqrt (d+c) - Real.sqrt (d-c))^2) / 4 ∧
    z = ((Real.sqrt (d+c) + Real.sqrt (d-c))^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_bases_l2241_224129


namespace NUMINAMATH_CALUDE_solve_for_z_l2241_224146

theorem solve_for_z (x y z : ℝ) : 
  x^2 - 3*x + 6 = y - 10 → 
  y = 2*z → 
  x = -5 → 
  z = 28 := by
sorry

end NUMINAMATH_CALUDE_solve_for_z_l2241_224146


namespace NUMINAMATH_CALUDE_sum_of_powers_half_l2241_224126

theorem sum_of_powers_half : 
  (-1/2 : ℚ)^3 + (-1/2 : ℚ)^2 + (-1/2 : ℚ)^1 + (1/2 : ℚ)^1 + (1/2 : ℚ)^2 + (1/2 : ℚ)^3 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_half_l2241_224126


namespace NUMINAMATH_CALUDE_cupcake_packages_l2241_224174

/-- Given the initial number of cupcakes, the number of eaten cupcakes, and the number of cupcakes per package,
    calculate the number of complete packages that can be made. -/
def calculate_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

/-- Theorem: Given 20 initial cupcakes, 11 eaten cupcakes, and 3 cupcakes per package,
    the number of complete packages that can be made is 3. -/
theorem cupcake_packages : calculate_packages 20 11 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l2241_224174


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2241_224154

theorem sin_cos_identity : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) - 
  Real.sin (253 * π / 180) * Real.cos (43 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2241_224154


namespace NUMINAMATH_CALUDE_determinant_equality_l2241_224116

theorem determinant_equality (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 3 → 
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 3 := by
sorry

end NUMINAMATH_CALUDE_determinant_equality_l2241_224116


namespace NUMINAMATH_CALUDE_smallest_x_factorization_l2241_224142

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def is_perfect_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

theorem smallest_x_factorization :
  let x := 2^15 * 3^20 * 5^24
  ∀ y : ℕ, y > 0 →
    (is_perfect_square (2*y) ∧ 
     is_perfect_cube (3*y) ∧ 
     is_perfect_fifth_power (5*y)) →
    y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_factorization_l2241_224142


namespace NUMINAMATH_CALUDE_gain_percentage_is_20_percent_l2241_224162

def selling_price : ℝ := 90
def gain : ℝ := 15

theorem gain_percentage_is_20_percent :
  let cost_price := selling_price - gain
  (gain / cost_price) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_gain_percentage_is_20_percent_l2241_224162


namespace NUMINAMATH_CALUDE_gcf_90_150_l2241_224114

theorem gcf_90_150 : Nat.gcd 90 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_90_150_l2241_224114


namespace NUMINAMATH_CALUDE_mango_dishes_l2241_224148

theorem mango_dishes (total_dishes : ℕ) (mango_salsa_dishes : ℕ) (mango_jelly_dishes : ℕ) (oliver_willing_dishes : ℕ) :
  total_dishes = 36 →
  mango_salsa_dishes = 3 →
  mango_jelly_dishes = 1 →
  oliver_willing_dishes = 28 →
  let fresh_mango_dishes : ℕ := total_dishes / 6
  let pickable_fresh_mango_dishes : ℕ := total_dishes - (oliver_willing_dishes + mango_salsa_dishes + mango_jelly_dishes)
  pickable_fresh_mango_dishes = 4 := by
sorry

end NUMINAMATH_CALUDE_mango_dishes_l2241_224148


namespace NUMINAMATH_CALUDE_mobius_speed_theorem_l2241_224106

theorem mobius_speed_theorem (total_distance : ℝ) (loaded_speed : ℝ) (total_time : ℝ) (rest_time : ℝ) :
  total_distance = 286 →
  loaded_speed = 11 →
  total_time = 26 →
  rest_time = 2 →
  ∃ v : ℝ, v > 0 ∧ (total_distance / 2) / loaded_speed + (total_distance / 2) / v = total_time - rest_time ∧ v = 13 := by
  sorry

end NUMINAMATH_CALUDE_mobius_speed_theorem_l2241_224106


namespace NUMINAMATH_CALUDE_complement_A_union_B_when_a_3_A_intersect_B_equals_B_iff_l2241_224198

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 3*x + 10}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Statement 1
theorem complement_A_union_B_when_a_3 :
  (Set.univ \ A) ∪ (B 3) = {x | x ≤ -2 ∨ (4 ≤ x ∧ x ≤ 7)} := by sorry

-- Statement 2
theorem A_intersect_B_equals_B_iff (a : ℝ) :
  A ∩ (B a) = B a ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_when_a_3_A_intersect_B_equals_B_iff_l2241_224198


namespace NUMINAMATH_CALUDE_alcohol_percentage_problem_l2241_224121

theorem alcohol_percentage_problem (initial_volume : Real) 
  (added_alcohol : Real) (final_percentage : Real) :
  initial_volume = 6 →
  added_alcohol = 3.6 →
  final_percentage = 50 →
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := final_volume * (final_percentage / 100)
  let initial_alcohol := final_alcohol - added_alcohol
  initial_alcohol / initial_volume * 100 = 20 := by
sorry


end NUMINAMATH_CALUDE_alcohol_percentage_problem_l2241_224121


namespace NUMINAMATH_CALUDE_chess_group_players_l2241_224170

theorem chess_group_players (n : ℕ) : n > 0 →
  (n * (n - 1)) / 2 = 21 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_players_l2241_224170


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2241_224105

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2241_224105


namespace NUMINAMATH_CALUDE_line_is_integral_curve_no_inflection_points_l2241_224151

/-- Represents a function y(x) that satisfies the differential equation y' = 2x - y -/
def IntegralCurve (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y) x = 2 * x - y x

/-- The line y = 2x - 2 is an integral curve of the differential equation y' = 2x - y -/
theorem line_is_integral_curve :
  IntegralCurve (λ x ↦ 2 * x - 2) := by sorry

/-- For any integral curve of y' = 2x - y, its second derivative is never zero -/
theorem no_inflection_points (y : ℝ → ℝ) (h : IntegralCurve y) :
  ∀ x, (deriv (deriv y)) x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_line_is_integral_curve_no_inflection_points_l2241_224151
