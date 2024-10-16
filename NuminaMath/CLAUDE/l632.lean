import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_shift_l632_63256

def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x + 3

def g (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + 3

theorem quadratic_shift (b : ℝ) : 
  (∀ x, g b x = f b (x + 6)) → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l632_63256


namespace NUMINAMATH_CALUDE_plane_equation_l632_63207

/-- A plane in 3D space defined by a normal vector and a point it passes through. -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Checks if a given point lies on the plane. -/
def Plane.contains (π : Plane) (p : ℝ × ℝ × ℝ) : Prop :=
  let (nx, ny, nz) := π.normal
  let (ax, ay, az) := π.point
  let (x, y, z) := p
  nx * (x - ax) + ny * (y - ay) + nz * (z - az) = 0

/-- The main theorem stating the equation of the plane. -/
theorem plane_equation (π : Plane) (h : π.normal = (1, -1, 2) ∧ π.point = (0, 3, 1)) :
  ∀ p : ℝ × ℝ × ℝ, π.contains p ↔ p.1 - p.2.1 + 2 * p.2.2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l632_63207


namespace NUMINAMATH_CALUDE_new_quadratic_equation_l632_63251

theorem new_quadratic_equation (α β : ℝ) : 
  (3 * α^2 + 7 * α + 4 = 0) → 
  (3 * β^2 + 7 * β + 4 = 0) → 
  (21 * (α / (β - 1))^2 - 23 * (α / (β - 1)) + 6 = 0) ∧
  (21 * (β / (α - 1))^2 - 23 * (β / (α - 1)) + 6 = 0) := by
sorry

end NUMINAMATH_CALUDE_new_quadratic_equation_l632_63251


namespace NUMINAMATH_CALUDE_fish_size_difference_l632_63289

/-- The size difference between Seongjun's and Sungwoo's fish given the conditions -/
theorem fish_size_difference (S J W : ℝ) 
  (h1 : S = J + 21.52)
  (h2 : J = W - 12.64) :
  S - W = 8.88 := by
  sorry

end NUMINAMATH_CALUDE_fish_size_difference_l632_63289


namespace NUMINAMATH_CALUDE_roots_equation_value_l632_63209

theorem roots_equation_value (x₁ x₂ : ℝ) 
  (h₁ : 3 * x₁^2 - 2 * x₁ - 4 = 0)
  (h₂ : 3 * x₂^2 - 2 * x₂ - 4 = 0)
  (h₃ : x₁ ≠ x₂) :
  3 * x₁^2 + 2 * x₂ = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_value_l632_63209


namespace NUMINAMATH_CALUDE_min_value_of_squares_l632_63242

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_value_of_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_set : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S)
  (h_sum : p + q + r + s ≥ 5) :
  (∀ a b c d e f g h : Int,
    a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a + b + c + d ≥ 5 →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 26) ∧
  (∃ a b c d e f g h : Int,
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c + d ≥ 5 ∧
    (a + b + c + d)^2 + (e + f + g + h)^2 = 26) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l632_63242


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l632_63247

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 5*(x^5 - 2*x^4 + 3*x^2) - 8*(x^5 + x^3 - x) + 6*(3*x^5 - x^4 + 2)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (p : ℝ → ℝ) : ℝ := 
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p : leading_coefficient p = 15 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l632_63247


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_union_A_B_equals_A_iff_m_in_range_l632_63225

-- Define sets A and B
def A : Set ℝ := {x | -3 < 2*x + 1 ∧ 2*x + 1 < 11}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m + 1}

-- Theorem 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | -2 < x ∧ x < 2} := by sorry

-- Theorem 2
theorem union_A_B_equals_A_iff_m_in_range (m : ℝ) :
  A ∪ B m = A ↔ m < -2 ∨ (-1 < m ∧ m < 2) := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_union_A_B_equals_A_iff_m_in_range_l632_63225


namespace NUMINAMATH_CALUDE_max_triangle_area_l632_63279

theorem max_triangle_area (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (htri : a + b > c ∧ a + c > b ∧ b + c > a) :
  ∃ (area : ℝ), area ≤ 1 ∧ ∀ (other_area : ℝ), 
    (∃ (x y z : ℝ), 0 < x ∧ x ≤ 1 ∧ 1 ≤ y ∧ y ≤ 2 ∧ 2 ≤ z ∧ z ≤ 3 ∧ 
      x + y > z ∧ x + z > y ∧ y + z > x ∧
      other_area = (x + y + z) * (- x + y + z) * (x - y + z) * (x + y - z) / (4 * (x + y + z))) →
    other_area ≤ area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l632_63279


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l632_63230

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let original : Rectangle := { length := 9, width := 6 }
  let pieces : ℕ := 4
  ∃ (max_piece min_piece : Rectangle),
    (pieces * max_piece.length * max_piece.width = original.length * original.width) ∧
    (pieces * min_piece.length * min_piece.width = original.length * original.width) ∧
    (∀ piece : Rectangle, 
      (pieces * piece.length * piece.width = original.length * original.width) → 
      (perimeter piece ≤ perimeter max_piece ∧ perimeter piece ≥ perimeter min_piece)) ∧
    (perimeter max_piece - perimeter min_piece = 9) := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l632_63230


namespace NUMINAMATH_CALUDE_expected_value_specialized_coin_l632_63258

/-- A specialized coin with given probabilities and payoffs -/
structure Coin where
  prob_heads : ℚ
  prob_tails : ℚ
  payoff_heads : ℚ
  payoff_tails : ℚ

/-- The expected value of a single flip of the coin -/
def expected_value (c : Coin) : ℚ :=
  c.prob_heads * c.payoff_heads + c.prob_tails * c.payoff_tails

/-- The expected value of two flips of the coin -/
def expected_value_two_flips (c : Coin) : ℚ :=
  2 * expected_value c

theorem expected_value_specialized_coin :
  let c : Coin := {
    prob_heads := 1/4,
    prob_tails := 3/4,
    payoff_heads := 4,
    payoff_tails := -3
  }
  expected_value_two_flips c = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_specialized_coin_l632_63258


namespace NUMINAMATH_CALUDE_corner_sum_equality_l632_63214

/-- A matrix satisfying the given condition for any 2x2 sub-matrix -/
def SpecialMatrix (n : ℕ) := Matrix (Fin n) (Fin n) ℝ

/-- The condition that must hold for any 2x2 sub-matrix -/
def satisfies_condition (A : SpecialMatrix 2000) : Prop :=
  ∀ i j, i.val < 1999 → j.val < 1999 →
    A i j + A (Fin.succ i) (Fin.succ j) = A i (Fin.succ j) + A (Fin.succ i) j

/-- The theorem to be proved -/
theorem corner_sum_equality (A : SpecialMatrix 2000) (h : satisfies_condition A) :
  A 0 0 + A 1999 1999 = A 0 1999 + A 1999 0 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_equality_l632_63214


namespace NUMINAMATH_CALUDE_min_value_f_and_m_plus_2n_l632_63204

-- Define the function f
def f (x a : ℝ) : ℝ := x + |x - a|

-- State the theorem
theorem min_value_f_and_m_plus_2n :
  ∃ (a : ℝ),
    (∀ x, (f x a - 2)^4 ≥ 0 ∧ f x a ≤ 4) →
    (∃ x₀, ∀ x, f x a ≥ f x₀ a ∧ f x₀ a = 2) ∧
    (∀ m n : ℕ+, 1 / (m : ℝ) + 2 / (n : ℝ) = 2 →
      (m : ℝ) + 2 * (n : ℝ) ≥ 9/2) ∧
    (∃ m₀ n₀ : ℕ+, 1 / (m₀ : ℝ) + 2 / (n₀ : ℝ) = 2 ∧
      (m₀ : ℝ) + 2 * (n₀ : ℝ) = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_and_m_plus_2n_l632_63204


namespace NUMINAMATH_CALUDE_unique_xxyy_square_l632_63226

/-- Represents a four-digit number in the form xxyy --/
def xxyy_number (x y : Nat) : Nat :=
  1100 * x + 11 * y

/-- Predicate to check if a number is a perfect square --/
def is_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem unique_xxyy_square :
  ∀ x y : Nat, x < 10 → y < 10 →
    (is_perfect_square (xxyy_number x y) ↔ x = 7 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_xxyy_square_l632_63226


namespace NUMINAMATH_CALUDE_hoodie_price_l632_63240

/-- Proves that the price of the hoodie is $80 given the conditions of Celina's hiking equipment purchase. -/
theorem hoodie_price (total_spent : ℝ) (boots_original : ℝ) (boots_discount : ℝ) (flashlight_ratio : ℝ) 
  (h_total : total_spent = 195)
  (h_boots_original : boots_original = 110)
  (h_boots_discount : boots_discount = 0.1)
  (h_flashlight : flashlight_ratio = 0.2) : 
  ∃ (hoodie_price : ℝ), 
    hoodie_price = 80 ∧ 
    (boots_original * (1 - boots_discount) + flashlight_ratio * hoodie_price + hoodie_price = total_spent) :=
by
  sorry


end NUMINAMATH_CALUDE_hoodie_price_l632_63240


namespace NUMINAMATH_CALUDE_parabola_standard_form_l632_63294

/-- A parabola with axis of symmetry x = 1 -/
structure Parabola where
  axis_of_symmetry : ℝ
  h_axis : axis_of_symmetry = 1

/-- The standard form of a parabola equation y^2 = ax -/
def standard_form (a : ℝ) (x y : ℝ) : Prop :=
  y^2 = a * x

/-- Theorem stating that the standard form of the parabola with axis of symmetry x = 1 is y^2 = -4x -/
theorem parabola_standard_form (p : Parabola) :
  ∃ a : ℝ, (∀ x y : ℝ, standard_form a x y) ∧ a = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_form_l632_63294


namespace NUMINAMATH_CALUDE_movie_marathon_end_time_correct_l632_63268

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addDurationToTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes + d.hours * 60
  { hours := t.hours + totalMinutes / 60,
    minutes := totalMinutes % 60 }

def movie_marathon_end_time (start : Time) 
  (movie1 : Duration) (break1 : Duration) 
  (movie2 : Duration) (break2 : Duration) 
  (movie3 : Duration) : Time :=
  let t1 := addDurationToTime start movie1
  let t2 := addDurationToTime t1 break1
  let t3 := addDurationToTime t2 movie2
  let t4 := addDurationToTime t3 break2
  addDurationToTime t4 movie3

theorem movie_marathon_end_time_correct :
  let start := Time.mk 13 0  -- 1:00 p.m.
  let movie1 := Duration.mk 2 20
  let break1 := Duration.mk 0 20
  let movie2 := Duration.mk 1 45
  let break2 := Duration.mk 0 20
  let movie3 := Duration.mk 2 10
  movie_marathon_end_time start movie1 break1 movie2 break2 movie3 = Time.mk 19 55  -- 7:55 p.m.
  := by sorry

end NUMINAMATH_CALUDE_movie_marathon_end_time_correct_l632_63268


namespace NUMINAMATH_CALUDE_money_problem_l632_63245

theorem money_problem (a b : ℝ) 
  (h1 : 6 * a + b > 78)
  (h2 : 4 * a - b = 42)
  (h3 : a ≥ 0)  -- Assuming money can't be negative
  (h4 : b ≥ 0)  -- Assuming money can't be negative
  : a > 12 ∧ b > 6 :=
by
  sorry

end NUMINAMATH_CALUDE_money_problem_l632_63245


namespace NUMINAMATH_CALUDE_hexagon_angle_problem_l632_63249

/-- Given a hexagon with specific angle conditions, prove that the unknown angle is 25 degrees. -/
theorem hexagon_angle_problem (a b c d e x : ℝ) : 
  -- Sum of interior angles of a hexagon
  a + b + c + d + e + x = (6 - 2) * 180 →
  -- Sum of five known angles
  a + b + c + d + e = 100 →
  -- Two adjacent angles are 75° each
  75 + x + 75 = 360 →
  -- Conclusion: x is 25°
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_problem_l632_63249


namespace NUMINAMATH_CALUDE_framed_rectangle_dimensions_l632_63265

/-- A rectangle on a grid with a one-cell-wide frame around it. -/
structure FramedRectangle where
  length : ℕ
  width : ℕ

/-- The area of the inner rectangle. -/
def FramedRectangle.inner_area (r : FramedRectangle) : ℕ :=
  r.length * r.width

/-- The area of the frame around the rectangle. -/
def FramedRectangle.frame_area (r : FramedRectangle) : ℕ :=
  (r.length + 2) * (r.width + 2) - r.length * r.width

/-- The property that the inner area equals the frame area. -/
def FramedRectangle.area_equality (r : FramedRectangle) : Prop :=
  r.inner_area = r.frame_area

/-- The theorem stating that if the inner area equals the frame area,
    then the dimensions are either 3 × 10 or 4 × 6. -/
theorem framed_rectangle_dimensions (r : FramedRectangle) :
  r.area_equality →
  ((r.length = 3 ∧ r.width = 10) ∨ (r.length = 4 ∧ r.width = 6) ∨
   (r.length = 10 ∧ r.width = 3) ∨ (r.length = 6 ∧ r.width = 4)) :=
by sorry

end NUMINAMATH_CALUDE_framed_rectangle_dimensions_l632_63265


namespace NUMINAMATH_CALUDE_speed_calculation_l632_63220

/-- Proves that given a distance of 600 meters and a time of 5 minutes, the speed is 7.2 km/hour -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 5) :
  (distance / 1000) / (time / 60) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l632_63220


namespace NUMINAMATH_CALUDE_unbroken_seashells_l632_63244

def total_seashells : ℕ := 7
def broken_seashells : ℕ := 4

theorem unbroken_seashells :
  total_seashells - broken_seashells = 3 := by sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l632_63244


namespace NUMINAMATH_CALUDE_simplify_expression_l632_63205

theorem simplify_expression (x y : ℝ) : (x - y) * (x + y) + (x - y)^2 = 2*x^2 - 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l632_63205


namespace NUMINAMATH_CALUDE_money_split_ratio_l632_63276

theorem money_split_ratio (parker_share richie_share total : ℚ) : 
  parker_share / richie_share = 2 / 3 →
  parker_share = 50 →
  parker_share < richie_share →
  total = parker_share + richie_share →
  total = 125 := by
sorry

end NUMINAMATH_CALUDE_money_split_ratio_l632_63276


namespace NUMINAMATH_CALUDE_evaluate_expression_l632_63261

theorem evaluate_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l632_63261


namespace NUMINAMATH_CALUDE_sum_of_fractions_l632_63203

theorem sum_of_fractions : (3 / 20 : ℚ) + (5 / 50 : ℚ) + (7 / 2000 : ℚ) = 0.2535 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l632_63203


namespace NUMINAMATH_CALUDE_union_equals_real_when_m_is_one_sufficient_necessary_condition_l632_63219

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B (m : ℝ) : Set ℝ := {x | (x - m) * (x - m - 1) ≥ 0}

theorem union_equals_real_when_m_is_one :
  A ∪ B 1 = Set.univ := by sorry

theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A ↔ x ∈ B m) ↔ m ≤ -2 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_equals_real_when_m_is_one_sufficient_necessary_condition_l632_63219


namespace NUMINAMATH_CALUDE_class_size_problem_l632_63270

/-- Given classes A, B, and C with the following properties:
  * Class A is twice as big as Class B
  * Class A is a third the size of Class C
  * Class B has 20 people
  Prove that Class C has 120 people -/
theorem class_size_problem (class_A class_B class_C : ℕ) : 
  class_A = 2 * class_B →
  class_A = class_C / 3 →
  class_B = 20 →
  class_C = 120 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l632_63270


namespace NUMINAMATH_CALUDE_toms_beach_trip_l632_63202

/-- Tom's beach trip problem -/
theorem toms_beach_trip (daily_seashells : ℕ) (total_seashells : ℕ) (days : ℕ) :
  daily_seashells = 7 →
  total_seashells = 35 →
  total_seashells = daily_seashells * days →
  days = 5 := by
  sorry

end NUMINAMATH_CALUDE_toms_beach_trip_l632_63202


namespace NUMINAMATH_CALUDE_hikmet_seventh_l632_63236

/-- Represents the position of a racer in a 12-person race -/
def Position := Fin 12

/-- The race results -/
structure RaceResult where
  david : Position
  hikmet : Position
  jack : Position
  marta : Position
  rand : Position
  todd : Position

/-- Conditions of the race -/
def race_conditions (result : RaceResult) : Prop :=
  result.marta.val = result.jack.val + 3 ∧
  result.jack.val = result.todd.val + 1 ∧
  result.todd.val = result.rand.val + 3 ∧
  result.rand.val + 5 = result.hikmet.val ∧
  result.hikmet.val + 4 = result.david.val ∧
  result.marta.val = 9

/-- Theorem stating that Hikmet finished in 7th place -/
theorem hikmet_seventh (result : RaceResult) 
  (h : race_conditions result) : result.hikmet.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_hikmet_seventh_l632_63236


namespace NUMINAMATH_CALUDE_signal_count_is_324_l632_63264

/-- Represents the number of indicator lights in a row -/
def total_lights : Nat := 6

/-- Represents the number of lights displayed at a time -/
def displayed_lights : Nat := 3

/-- Represents the number of possible colors for each light -/
def color_options : Nat := 3

/-- Calculates the number of different signals that can be displayed -/
def signal_count : Nat :=
  let adjacent_pair_positions := total_lights - 1
  let non_adjacent_positions := total_lights - 2
  (adjacent_pair_positions * non_adjacent_positions) * color_options^displayed_lights

/-- Theorem stating that the number of different signals is 324 -/
theorem signal_count_is_324 : signal_count = 324 := by
  sorry

end NUMINAMATH_CALUDE_signal_count_is_324_l632_63264


namespace NUMINAMATH_CALUDE_revenue_unchanged_with_price_increase_l632_63222

theorem revenue_unchanged_with_price_increase
  (original_price original_demand : ℝ)
  (price_increase : ℝ)
  (demand_decrease : ℝ)
  (h1 : price_increase = 0.3)
  (h2 : demand_decrease = 0.2308) :
  original_price * original_demand ≤
  (original_price * (1 + price_increase)) * (original_demand * (1 - demand_decrease)) :=
by sorry

end NUMINAMATH_CALUDE_revenue_unchanged_with_price_increase_l632_63222


namespace NUMINAMATH_CALUDE_tangent_length_right_triangle_l632_63292

/-- Given a right triangle with legs a and b, and hypotenuse c,
    the length of the tangent to the circumcircle drawn parallel
    to the hypotenuse is equal to c(a + b)²/(2ab) -/
theorem tangent_length_right_triangle (a b c : ℝ) 
  (h_right : c^2 = a^2 + b^2) (h_pos : a > 0 ∧ b > 0) :
  let x := c * (a + b)^2 / (2 * a * b)
  ∃ (m : ℝ), m > 0 ∧ 
    (c / x = m / (m + c/2)) ∧
    (m * c = a * b) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_right_triangle_l632_63292


namespace NUMINAMATH_CALUDE_rectangle_diagonal_ratio_l632_63228

theorem rectangle_diagonal_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≤ b) :
  (a + b - Real.sqrt (a^2 + b^2) = b / 3) → (a / b = 5 / 12) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_ratio_l632_63228


namespace NUMINAMATH_CALUDE_rectangle_area_l632_63254

theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let square_side := 1
  let rectangle_perimeter := 2 * l + 2 * w
  let square_perimeter := 4 * square_side
  rectangle_perimeter = square_perimeter → l * w = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l632_63254


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l632_63238

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 → 
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l632_63238


namespace NUMINAMATH_CALUDE_f_extrema_l632_63223

def f (x : ℝ) := x^2 - 2*x - 1

theorem f_extrema :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 2, f x = min) ∧
    max = 14 ∧ min = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l632_63223


namespace NUMINAMATH_CALUDE_positive_integer_N_equals_121_l632_63200

theorem positive_integer_N_equals_121 :
  ∃ (N : ℕ), N > 0 ∧ 33^2 * 55^2 = 15^2 * N^2 ∧ N = 121 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_N_equals_121_l632_63200


namespace NUMINAMATH_CALUDE_final_apartments_can_be_less_l632_63263

/-- Represents the structure of an apartment building project -/
structure ApartmentProject where
  entrances : ℕ
  floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in a project -/
def total_apartments (p : ApartmentProject) : ℕ :=
  p.entrances * p.floors * p.apartments_per_floor

/-- Applies the architect's adjustments to a project -/
def adjust_project (p : ApartmentProject) (removed_entrances floors_added : ℕ) : ApartmentProject :=
  { entrances := p.entrances - removed_entrances,
    floors := p.floors + floors_added,
    apartments_per_floor := p.apartments_per_floor }

/-- The main theorem stating that the final number of apartments can be less than the initial number -/
theorem final_apartments_can_be_less :
  ∃ (initial : ApartmentProject)
    (removed_entrances1 floors_added1 removed_entrances2 floors_added2 : ℕ),
    initial.entrances = 5 ∧
    initial.floors = 2 ∧
    initial.apartments_per_floor = 1 ∧
    removed_entrances1 = 2 ∧
    floors_added1 = 3 ∧
    removed_entrances2 = 2 ∧
    floors_added2 = 3 ∧
    let first_adjustment := adjust_project initial removed_entrances1 floors_added1
    let final_project := adjust_project first_adjustment removed_entrances2 floors_added2
    total_apartments final_project < total_apartments initial :=
by
  sorry

end NUMINAMATH_CALUDE_final_apartments_can_be_less_l632_63263


namespace NUMINAMATH_CALUDE_fraction_simplification_l632_63215

theorem fraction_simplification : 
  (1 / 3 + 1 / 4) / (2 / 5 - 1 / 6) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l632_63215


namespace NUMINAMATH_CALUDE_f_properties_l632_63201

def f (x : ℝ) := -7 * x

theorem f_properties :
  (∀ x y : ℝ, (x > 0 ∧ f x < 0) ∨ (x < 0 ∧ f x > 0)) ∧
  f 1 = -7 ∧
  (∀ x y : ℝ, x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l632_63201


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l632_63231

theorem circle_area_from_circumference :
  ∀ (r : ℝ),
  (2 * π * r = 30 * π) →
  (π * r^2 = 225 * π) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l632_63231


namespace NUMINAMATH_CALUDE_cell_phone_bill_is_45_l632_63287

/-- Calculates the total cell phone bill based on given parameters --/
def calculate_bill (fixed_charge : ℚ) (daytime_rate : ℚ) (evening_rate : ℚ) 
                   (free_evening_minutes : ℕ) (daytime_minutes : ℕ) (evening_minutes : ℕ) : ℚ :=
  let daytime_cost := daytime_rate * daytime_minutes
  let chargeable_evening_minutes := max (evening_minutes - free_evening_minutes) 0
  let evening_cost := evening_rate * chargeable_evening_minutes
  fixed_charge + daytime_cost + evening_cost

/-- Theorem stating that the cell phone bill is $45 given the specified conditions --/
theorem cell_phone_bill_is_45 :
  calculate_bill 20 0.1 0.05 200 200 300 = 45 := by
  sorry


end NUMINAMATH_CALUDE_cell_phone_bill_is_45_l632_63287


namespace NUMINAMATH_CALUDE_new_regression_line_after_point_removal_l632_63224

/-- Represents a sample point -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Represents a regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the regression line from a list of sample points -/
def calculateRegressionLine (sample : List SamplePoint) : RegressionLine :=
  sorry

/-- Theorem stating the properties of the new regression line after removing two specific points -/
theorem new_regression_line_after_point_removal 
  (sample : List SamplePoint)
  (initial_line : RegressionLine)
  (mean_x : ℝ) :
  sample.length = 10 →
  initial_line = { slope := 2, intercept := -0.4 } →
  mean_x = 2 →
  let new_sample := sample.filter (λ p => ¬(p.x = -3 ∧ p.y = 1) ∧ ¬(p.x = 3 ∧ p.y = -1))
  let new_line := calculateRegressionLine new_sample
  new_line.slope = 3 →
  new_line = { slope := 3, intercept := -3 } :=
sorry

end NUMINAMATH_CALUDE_new_regression_line_after_point_removal_l632_63224


namespace NUMINAMATH_CALUDE_basketball_match_children_l632_63216

/-- Calculates the number of children at a basketball match given the total number of spectators,
    the number of men, and the ratio of children to women. -/
def number_of_children (total : ℕ) (men : ℕ) (child_to_woman_ratio : ℕ) : ℕ :=
  let non_men := total - men
  let women := non_men / (child_to_woman_ratio + 1)
  child_to_woman_ratio * women

/-- Theorem stating that given the specific conditions of the basketball match,
    the number of children is 2500. -/
theorem basketball_match_children :
  number_of_children 10000 7000 5 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_basketball_match_children_l632_63216


namespace NUMINAMATH_CALUDE_rectangle_area_change_l632_63253

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let L' := 1.3 * L
  let B' := 0.75 * B
  let A := L * B
  let A' := L' * B'
  A' / A = 0.975 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l632_63253


namespace NUMINAMATH_CALUDE_slag_transport_allocation_l632_63257

/-- Represents the daily rental income for slag transport vehicles --/
def daily_rental_income (x : ℕ) : ℕ := 80000 - 200 * x

/-- Theorem stating the properties of the slag transport vehicle allocation problem --/
theorem slag_transport_allocation :
  (∀ x : ℕ, x ≤ 20 → daily_rental_income x = 80000 - 200 * x) ∧
  (∀ x : ℕ, x ≤ 20 → (daily_rental_income x ≥ 79600 ↔ x ≤ 2)) ∧
  (∀ x : ℕ, x ≤ 20 → daily_rental_income x ≤ 80000) ∧
  (daily_rental_income 0 = 80000) := by
  sorry

#check slag_transport_allocation

end NUMINAMATH_CALUDE_slag_transport_allocation_l632_63257


namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l632_63217

/-- Represents the inventory of a bookstore -/
structure BookInventory where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Conditions for Joel's bookstore inventory -/
def joelsBookstore (inventory : BookInventory) : Prop :=
  inventory.historicalFiction = (30 * inventory.total) / 100 ∧
  inventory.historicalFictionNewReleases = (30 * inventory.historicalFiction) / 100 ∧
  inventory.otherNewReleases = (40 * (inventory.total - inventory.historicalFiction)) / 100

/-- Theorem: The fraction of all new releases that are historical fiction is 9/37 -/
theorem historical_fiction_new_releases_fraction 
  (inventory : BookInventory) (h : joelsBookstore inventory) :
  (inventory.historicalFictionNewReleases : ℚ) / 
  (inventory.historicalFictionNewReleases + inventory.otherNewReleases) = 9 / 37 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l632_63217


namespace NUMINAMATH_CALUDE_intersection_line_circle_l632_63299

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define a line with slope 1
def line_with_slope_1 (x y b : ℝ) : Prop := y = x + b

-- Define a point on both the circle and the line
def point_on_circle_and_line (x y b : ℝ) : Prop :=
  circle_C x y ∧ line_with_slope_1 x y b

-- Define that a circle passes through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = x₁^2 + y₁^2 + x₂^2 + y₂^2

theorem intersection_line_circle :
  ∃ b : ℝ, b = 1 ∨ b = -4 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    point_on_circle_and_line x₁ y₁ b ∧
    point_on_circle_and_line x₂ y₂ b ∧
    x₁ ≠ x₂ ∧
    circle_through_origin x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l632_63299


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l632_63298

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the condition for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  m < 5/4

-- Define the intersection condition
def intersects_at_mn (m : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ),
    circle_equation M.1 M.2 m ∧
    circle_equation N.1 N.2 m ∧
    line_equation M.1 M.2 ∧
    line_equation N.1 N.2 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (4/5 * Real.sqrt 5)^2

theorem circle_intersection_theorem :
  ∀ m : ℝ, is_circle m → intersects_at_mn m → m = 3.62 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l632_63298


namespace NUMINAMATH_CALUDE_closest_product_l632_63239

def options : List ℝ := [1600, 1800, 2000, 2200, 2400]

theorem closest_product : 
  let product := 0.000625 * 3142857
  ∀ x ∈ options, x ≠ 1800 → |product - 1800| < |product - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_product_l632_63239


namespace NUMINAMATH_CALUDE_rice_mixture_price_l632_63267

/-- Given two types of rice mixed together, prove the price of the first type --/
theorem rice_mixture_price (price2 : ℚ) (weight1 weight2 : ℚ) (mixture_price : ℚ) 
  (h1 : price2 = 960 / 100)  -- Rs. 9.60 converted to a rational number
  (h2 : weight1 = 49)
  (h3 : weight2 = 56)
  (h4 : mixture_price = 820 / 100)  -- Rs. 8.20 converted to a rational number
  : ∃ (price1 : ℚ), price1 = 660 / 100 ∧  -- Rs. 6.60 converted to a rational number
    (weight1 * price1 + weight2 * price2) / (weight1 + weight2) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l632_63267


namespace NUMINAMATH_CALUDE_work_completion_time_l632_63282

/-- Given:
  * A can do a work in 20 days
  * A works for 10 days and then leaves
  * B can finish the remaining work in 15 days
Prove that B can do the entire work in 30 days -/
theorem work_completion_time (a_time b_remaining_time : ℕ) 
  (h1 : a_time = 20)
  (h2 : b_remaining_time = 15) :
  let a_work_rate : ℚ := 1 / a_time
  let a_work_done : ℚ := a_work_rate * 10
  let remaining_work : ℚ := 1 - a_work_done
  let b_rate : ℚ := remaining_work / b_remaining_time
  b_rate⁻¹ = 30 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l632_63282


namespace NUMINAMATH_CALUDE_harkamal_payment_l632_63269

/-- The total amount Harkamal paid to the shopkeeper for grapes and mangoes. -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1125 to the shopkeeper. -/
theorem harkamal_payment : total_amount_paid 9 70 9 55 = 1125 := by
  sorry

#eval total_amount_paid 9 70 9 55

end NUMINAMATH_CALUDE_harkamal_payment_l632_63269


namespace NUMINAMATH_CALUDE_sector_max_area_l632_63232

/-- Given a sector with perimeter 20 cm, its area is maximized when the radius is 5 cm -/
theorem sector_max_area (r : ℝ) (l : ℝ) : 
  l + 2*r = 20 →
  (∀ r' : ℝ, r' > 0 → ∃ l' : ℝ, l' + 2*r' = 20 → r'*l'/2 ≤ r*l/2) →
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_l632_63232


namespace NUMINAMATH_CALUDE_only_constant_one_is_divisor_respecting_l632_63227

-- Define the number of positive divisors function
def d (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n)).card + 1

-- Define divisor-respecting property
def divisor_respecting (F : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, d (F (m * n)) = d (F m) * d (F n)) ∧
  (∀ n : ℕ, d (F n) ≤ d n)

-- Theorem statement
theorem only_constant_one_is_divisor_respecting :
  ∀ F : ℕ → ℕ, divisor_respecting F → ∀ x : ℕ, F x = 1 :=
by sorry

end NUMINAMATH_CALUDE_only_constant_one_is_divisor_respecting_l632_63227


namespace NUMINAMATH_CALUDE_curve_equation_l632_63255

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t - Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

theorem curve_equation :
  ∃ (a b c : ℝ), ∀ (t : ℝ),
    a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1 ∧
    a = 1/9 ∧ b = 2/45 ∧ c = 4/45 := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_l632_63255


namespace NUMINAMATH_CALUDE_star_sqrt_11_l632_63260

/-- Custom binary operation ¤ -/
def star (x y z : ℝ) : ℝ := (x + y)^2 - z^2

theorem star_sqrt_11 (z : ℝ) :
  star (Real.sqrt 11) (Real.sqrt 11) z = 44 → z = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_sqrt_11_l632_63260


namespace NUMINAMATH_CALUDE_max_profit_on_day_6_l632_63278

-- Define the sales price function
def p (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 44 + x
  else if 6 < x ∧ x ≤ 20 then 56 - x
  else 0

-- Define the sales volume function
def q (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 8 then 48 - x
  else if 8 < x ∧ x ≤ 20 then 32 + x
  else 0

-- Define the profit function
def profit (x : ℕ) : ℝ := (p x - 25) * q x

-- Theorem statement
theorem max_profit_on_day_6 :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 20 → profit x ≤ profit 6 ∧ profit 6 = 1050 :=
sorry

end NUMINAMATH_CALUDE_max_profit_on_day_6_l632_63278


namespace NUMINAMATH_CALUDE_rooster_stamps_count_l632_63297

theorem rooster_stamps_count (daffodil_stamps : ℕ) (rooster_stamps : ℕ) 
  (h1 : daffodil_stamps = 2) 
  (h2 : rooster_stamps - daffodil_stamps = 0) : 
  rooster_stamps = 2 := by
  sorry

end NUMINAMATH_CALUDE_rooster_stamps_count_l632_63297


namespace NUMINAMATH_CALUDE_max_value_d_l632_63243

theorem max_value_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_d_l632_63243


namespace NUMINAMATH_CALUDE_height_derivative_at_one_l632_63272

-- Define the height function
def h (t : ℝ) : ℝ := -4.9 * t^2 + 10 * t

-- State the theorem
theorem height_derivative_at_one :
  (deriv h) 1 = 0.2 := by sorry

end NUMINAMATH_CALUDE_height_derivative_at_one_l632_63272


namespace NUMINAMATH_CALUDE_perpendicular_construction_l632_63259

-- Define the basic geometric elements
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

-- Define the concept of a point being on a line
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define the construction process
def construct_perpendicular (l : Line) (A : Point) (m l1 l2 lm : Line) (M1 M2 B : Point) : Prop :=
  A.on_line l ∧
  A.on_line m ∧
  parallel l1 m ∧
  parallel l2 m ∧
  M1.on_line l ∧
  M1.on_line l1 ∧
  M2.on_line l ∧
  M2.on_line l2 ∧
  parallel lm (Line.mk (M1.x - A.x) (M1.y - A.y) 0) ∧
  B.on_line l2 ∧
  B.on_line lm

-- State the theorem
theorem perpendicular_construction (l : Line) (A : Point) :
  ∃ (m l1 l2 lm : Line) (M1 M2 B : Point),
    construct_perpendicular l A m l1 l2 lm M1 M2 B →
    perpendicular l (Line.mk (B.x - A.x) (B.y - A.y) 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_construction_l632_63259


namespace NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l632_63241

/-- An angle in the fourth quadrant -/
structure FourthQuadrantAngle where
  α : Real
  in_fourth_quadrant : α > -π/2 ∧ α < 0

/-- A point on the terminal side of an angle -/
structure TerminalPoint where
  x : Real
  y : Real

/-- Properties of the angle α -/
structure AngleProperties (α : FourthQuadrantAngle) where
  terminal_point : TerminalPoint
  x_coord : terminal_point.x = 4
  sin_value : Real.sin α.α = terminal_point.y / 5

theorem tan_value_fourth_quadrant (α : FourthQuadrantAngle) 
  (props : AngleProperties α) : Real.tan α.α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l632_63241


namespace NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l632_63262

theorem absolute_sum_zero_implies_sum (a b : ℝ) : 
  |a - 5| + |b + 8| = 0 → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l632_63262


namespace NUMINAMATH_CALUDE_line_y_intercept_l632_63252

/-- A straight line in the xy-plane with slope 2 and passing through (259, 520) has y-intercept 2 -/
theorem line_y_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∀ x y, f y = 2 * x + f 0) →  -- slope is 2
  f 520 = 2 * 259 + f 0 →      -- point (259, 520) lies on the line
  f 0 = 2 := by               -- y-intercept is 2
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l632_63252


namespace NUMINAMATH_CALUDE_largest_base_digit_sum_not_nine_l632_63266

/-- The sum of digits of a natural number in a given base -/
def sum_of_digits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Conversion of a number from base 10 to another base -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem largest_base_digit_sum_not_nine :
  ∀ b : ℕ, b > 8 →
    sum_of_digits (12^3) b = 3^2 ∧
    sum_of_digits (12^3) 8 ≠ 3^2 ∧
    (12^3 : ℕ) = 1728 := by sorry

end NUMINAMATH_CALUDE_largest_base_digit_sum_not_nine_l632_63266


namespace NUMINAMATH_CALUDE_near_square_quotient_l632_63234

/-- A natural number is a near-square if it is the product of two consecutive natural numbers. -/
def is_near_square (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

/-- Theorem stating that any near-square can be represented as the quotient of two near-squares. -/
theorem near_square_quotient (n : ℕ) : 
  is_near_square (n * (n + 1)) → 
  ∃ a b c : ℕ, 
    is_near_square a ∧ 
    is_near_square b ∧ 
    is_near_square c ∧ 
    n * (n + 1) = a / c ∧
    b = c * (n + 2) :=
sorry

end NUMINAMATH_CALUDE_near_square_quotient_l632_63234


namespace NUMINAMATH_CALUDE_expression_value_l632_63210

theorem expression_value (a b c : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a - c = Real.rpow 7 (1/3)) : 
  (c - b) * ((a - b)^2 + (a - b)*(a - c) + (a - c)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l632_63210


namespace NUMINAMATH_CALUDE_perpendicular_vector_l632_63250

theorem perpendicular_vector (a b : ℝ × ℝ) : 
  a = (Real.sqrt 3, Real.sqrt 5) →
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (b.1^2 + b.2^2 = 4) →
  (b = (-Real.sqrt (10) / 2, Real.sqrt 6 / 2) ∨ 
   b = (Real.sqrt (10) / 2, -Real.sqrt 6 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_l632_63250


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l632_63280

theorem contrapositive_real_roots :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔
  (∀ m : ℝ, (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l632_63280


namespace NUMINAMATH_CALUDE_employed_males_percentage_l632_63275

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 70)
  (h2 : employed_females_percentage = 70)
  (h3 : total_population > 0) :
  (employed_percentage / 100 * (1 - employed_females_percentage / 100) * 100) = 21 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l632_63275


namespace NUMINAMATH_CALUDE_xavier_success_probability_l632_63221

theorem xavier_success_probability 
  (p_yvonne : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_and_yvonne_not_zelda : ℝ) 
  (h1 : p_yvonne = 1/2) 
  (h2 : p_zelda = 5/8) 
  (h3 : p_xavier_and_yvonne_not_zelda = 0.0625) :
  ∃ p_xavier : ℝ, 
    p_xavier_and_yvonne_not_zelda = p_xavier * p_yvonne * (1 - p_zelda) ∧ 
    p_xavier = 1/3 :=
sorry

end NUMINAMATH_CALUDE_xavier_success_probability_l632_63221


namespace NUMINAMATH_CALUDE_appliance_cost_after_discount_l632_63229

/-- Calculates the total cost of a washing machine and dryer after applying a discount -/
theorem appliance_cost_after_discount
  (washing_machine_cost : ℝ)
  (dryer_cost_difference : ℝ)
  (discount_percentage : ℝ)
  (h1 : washing_machine_cost = 100)
  (h2 : dryer_cost_difference = 30)
  (h3 : discount_percentage = 0.1) :
  let dryer_cost := washing_machine_cost - dryer_cost_difference
  let total_cost := washing_machine_cost + dryer_cost
  let discount_amount := discount_percentage * total_cost
  washing_machine_cost + dryer_cost - discount_amount = 153 := by
sorry

end NUMINAMATH_CALUDE_appliance_cost_after_discount_l632_63229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l632_63295

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l632_63295


namespace NUMINAMATH_CALUDE_blue_whale_tongue_weight_l632_63288

-- Define the weight of one ton in pounds
def ton_in_pounds : ℕ := 2000

-- Define the weight of a blue whale's tongue in tons
def blue_whale_tongue_tons : ℕ := 3

-- Theorem: The weight of a blue whale's tongue in pounds
theorem blue_whale_tongue_weight :
  blue_whale_tongue_tons * ton_in_pounds = 6000 := by
  sorry

end NUMINAMATH_CALUDE_blue_whale_tongue_weight_l632_63288


namespace NUMINAMATH_CALUDE_projection_onto_plane_l632_63206

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ :=
  sorry

theorem projection_onto_plane (P : Plane) :
  project (2, 4, 7) P = (1, 3, 3) →
  project (6, -3, 8) P = (41/9, -40/9, 20/9) := by
  sorry

end NUMINAMATH_CALUDE_projection_onto_plane_l632_63206


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l632_63218

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l632_63218


namespace NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l632_63283

/-- 
Given a cylinder and a cone with the same radius, where the cone's height is one-third of the cylinder's height,
prove that the ratio of the cone's volume to the cylinder's volume is 1/9.
-/
theorem cone_to_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l632_63283


namespace NUMINAMATH_CALUDE_election_winner_votes_l632_63296

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  (winner_percentage = 3/5) →
  (vote_difference = 240) →
  (winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference) →
  (winner_percentage * total_votes = 720) := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l632_63296


namespace NUMINAMATH_CALUDE_barbara_shopping_cost_l632_63235

-- Define the quantities and prices
def tuna_packs : ℕ := 5
def tuna_price : ℚ := 2
def water_bottles : ℕ := 4
def water_price : ℚ := 3/2
def other_goods_cost : ℚ := 40

-- Define the total cost function
def total_cost : ℚ :=
  (tuna_packs * tuna_price) + (water_bottles * water_price) + other_goods_cost

-- Theorem statement
theorem barbara_shopping_cost :
  total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_barbara_shopping_cost_l632_63235


namespace NUMINAMATH_CALUDE_domain_of_f_l632_63237

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (2 * sin x - 1) + sqrt (1 - 2 * cos x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = ⋃ (k : ℤ), Ico (2 * k * π + π / 3) (2 * k * π + 5 * π / 6) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l632_63237


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_two_l632_63293

theorem binomial_coefficient_n_plus_one_choose_two (n : ℕ) : 
  Nat.choose (n + 1) 2 = (n + 1) * n / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_two_l632_63293


namespace NUMINAMATH_CALUDE_bella_steps_to_meet_l632_63286

/-- The number of steps Bella takes when meeting Ella -/
def steps_to_meet (distance : ℕ) (speed_ratio : ℕ) (step_length : ℕ) : ℕ :=
  (distance * 2) / ((speed_ratio + 1) * step_length)

/-- Theorem stating that Bella takes 1056 steps to meet Ella under given conditions -/
theorem bella_steps_to_meet :
  steps_to_meet 15840 4 3 = 1056 :=
by sorry

end NUMINAMATH_CALUDE_bella_steps_to_meet_l632_63286


namespace NUMINAMATH_CALUDE_f_expression_for_x_gt_1_l632_63290

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (-x + 1)

theorem f_expression_for_x_gt_1 (f : ℝ → ℝ) 
  (h1 : is_even_shifted f) 
  (h2 : ∀ x, x < 1 → f x = x^2 + 1) :
  ∀ x, x > 1 → f x = x^2 - 4*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_f_expression_for_x_gt_1_l632_63290


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_probability_l632_63277

/-- Regular tetrahedron with inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  r : ℝ  -- radius of inscribed sphere
  R : ℝ  -- radius of circumscribed sphere
  h : R = 3 * r  -- relationship between R and r

/-- External sphere tangent to a face of the tetrahedron and the circumscribed sphere -/
structure ExternalSphere (t : RegularTetrahedron) where
  radius : ℝ
  h : radius = 1.5 * t.r

/-- The probability theorem for the tetrahedron and spheres setup -/
theorem tetrahedron_sphere_probability (t : RegularTetrahedron) 
  (e : ExternalSphere t) (n : ℕ) (h_n : n = 4) :
  let v_external := n * (4 / 3 * Real.pi * e.radius ^ 3)
  let v_circumscribed := 4 / 3 * Real.pi * t.R ^ 3
  v_external ≤ v_circumscribed ∧ 
  v_external / v_circumscribed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_probability_l632_63277


namespace NUMINAMATH_CALUDE_gcf_lcm_product_l632_63291

def numbers : List Nat := [6, 18, 24]

theorem gcf_lcm_product (A B : Nat) 
  (h1 : A = Nat.gcd 6 (Nat.gcd 18 24))
  (h2 : B = Nat.lcm 6 (Nat.lcm 18 24)) :
  A * B = 432 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_product_l632_63291


namespace NUMINAMATH_CALUDE_b_plus_3c_positive_l632_63211

theorem b_plus_3c_positive (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_3c_positive_l632_63211


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_in_range_l632_63233

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^3 - 2*a*x + 1 else (a-1)^x - 7

theorem increasing_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_in_range_l632_63233


namespace NUMINAMATH_CALUDE_simple_interest_problem_l632_63274

theorem simple_interest_problem (P : ℝ) : 
  P * 0.08 * 3 = 0.5 * 4000 * ((1 + 0.10)^2 - 1) ↔ P = 1750 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l632_63274


namespace NUMINAMATH_CALUDE_lollipop_distribution_l632_63281

/-- The number of kids in the group -/
def num_kids : ℕ := 42

/-- The initial number of lollipops available -/
def initial_lollipops : ℕ := 650

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The additional lollipops needed -/
def additional_lollipops : ℕ := sum_first_n num_kids - initial_lollipops

theorem lollipop_distribution :
  additional_lollipops = 253 ∧
  ∀ k, k ≤ num_kids → k ≤ sum_first_n num_kids ∧
  sum_first_n num_kids = initial_lollipops + additional_lollipops :=
sorry

end NUMINAMATH_CALUDE_lollipop_distribution_l632_63281


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equation_solutions_l632_63212

theorem sum_of_reciprocal_equation_solutions : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ j k : ℕ, j > 0 ∧ k > 0 ∧ 1 / j + 1 / k = (1 : ℚ) / 4 ∧ n = j + k) ∧ 
  (∀ j k : ℕ, j > 0 → k > 0 → 1 / j + 1 / k = (1 : ℚ) / 4 → (j + k) ∈ S) ∧
  S.sum id = 59 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equation_solutions_l632_63212


namespace NUMINAMATH_CALUDE_line_equation_and_intersection_l632_63246

/-- The slope of the first line -/
def m : ℚ := 3 / 4

/-- The y-intercept of the first line -/
def b : ℚ := 3 / 2

/-- The slope of the second line -/
def m' : ℚ := -1

/-- The y-intercept of the second line -/
def b' : ℚ := 7

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := 11 / 7

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := 25 / 7

theorem line_equation_and_intersection :
  (∀ x y : ℚ, 3 * (x - 2) + (-4) * (y - 3) = 0 ↔ y = m * x + b) ∧
  (m * x_intersect + b = m' * x_intersect + b') ∧
  (y_intersect = m * x_intersect + b) ∧
  (y_intersect = m' * x_intersect + b') := by
  sorry

end NUMINAMATH_CALUDE_line_equation_and_intersection_l632_63246


namespace NUMINAMATH_CALUDE_factoring_transformation_l632_63208

theorem factoring_transformation (y : ℝ) : 4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factoring_transformation_l632_63208


namespace NUMINAMATH_CALUDE_jens_age_difference_l632_63273

/-- Proves that the difference between 3 times Jen's son's current age and Jen's current age is 7 years -/
theorem jens_age_difference (jen_age_at_birth : ℕ) (son_current_age : ℕ) (jen_current_age : ℕ) : 
  jen_age_at_birth = 25 →
  son_current_age = 16 →
  jen_current_age = 41 →
  3 * son_current_age - jen_current_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_jens_age_difference_l632_63273


namespace NUMINAMATH_CALUDE_harmonious_number_properties_l632_63285

/-- A harmonious number is a three-digit number where the tens digit 
    is equal to the sum of its units digit and hundreds digit. -/
def is_harmonious (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  (n / 10 % 10 = n % 10 + n / 100)

/-- The smallest harmonious number -/
def smallest_harmonious : ℕ := 110

/-- The largest harmonious number -/
def largest_harmonious : ℕ := 990

/-- Algebraic expression for a harmonious number -/
def harmonious_expression (a b : ℕ) : ℕ := 110 * b - 99 * a

theorem harmonious_number_properties :
  (∀ n : ℕ, is_harmonious n → smallest_harmonious ≤ n ∧ n ≤ largest_harmonious) ∧
  (∀ n : ℕ, is_harmonious n → 
    ∃ a b : ℕ, a ≥ 0 ∧ b ≥ 1 ∧ b > a ∧ 
    n = harmonious_expression a b) :=
sorry

end NUMINAMATH_CALUDE_harmonious_number_properties_l632_63285


namespace NUMINAMATH_CALUDE_coin_value_calculation_l632_63271

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of pennies -/
def num_pennies : ℕ := 9

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 3

/-- The number of quarters -/
def num_quarters : ℕ := 7

/-- The number of half-dollars -/
def num_half_dollars : ℕ := 5

/-- The total value of the coins in dollars -/
def total_value : ℚ :=
  num_pennies * penny_value +
  num_nickels * nickel_value +
  num_dimes * dime_value +
  num_quarters * quarter_value +
  num_half_dollars * half_dollar_value

theorem coin_value_calculation :
  total_value = 4.84 := by sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l632_63271


namespace NUMINAMATH_CALUDE_seashells_to_find_l632_63284

def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

theorem seashells_to_find : target_seashells - current_seashells = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashells_to_find_l632_63284


namespace NUMINAMATH_CALUDE_largest_power_of_2020_dividing_product_l632_63248

def pow (n : ℕ) : ℕ :=
  sorry

def largest_power_dividing_product (base : ℕ) (upper_bound : ℕ) : ℕ :=
  sorry

theorem largest_power_of_2020_dividing_product :
  largest_power_dividing_product 2020 7200 = 72 :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_2020_dividing_product_l632_63248


namespace NUMINAMATH_CALUDE_f_properties_l632_63213

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem f_properties : 
  (∀ x ≠ 0, f (-x) = -f x) ∧ 
  (∀ x, ∃ y, f x = y) ∧
  (∀ y, f ⁻¹' {y} ≠ ∅ → -1 < y ∧ y < 1) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l632_63213
