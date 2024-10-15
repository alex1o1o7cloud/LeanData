import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3773_377398

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h2 : a 3 = 4) 
  (h3 : a 6 = 1/2) : 
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3773_377398


namespace NUMINAMATH_CALUDE_division_remainder_l3773_377396

-- Define the dividend polynomial
def dividend (x : ℝ) : ℝ := 3*x^5 + 2*x^4 - 5*x^3 + 6*x - 8

-- Define the divisor polynomial
def divisor (x : ℝ) : ℝ := x^2 + 3*x + 2

-- Define the remainder polynomial
def remainder (x : ℝ) : ℝ := 34*x + 24

-- Theorem statement
theorem division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, dividend x = divisor x * q x + remainder x :=
sorry

end NUMINAMATH_CALUDE_division_remainder_l3773_377396


namespace NUMINAMATH_CALUDE_lines_can_coincide_by_rotation_l3773_377363

/-- Given two lines l₁ and l₂ in the xy-plane, prove that they can coincide
    by rotating l₂ around a point on l₁. -/
theorem lines_can_coincide_by_rotation (α c : ℝ) :
  ∃ (x₀ y₀ θ : ℝ), 
    (y₀ = x₀ * Real.sin α) ∧  -- Point (x₀, y₀) is on l₁
    (∀ x y : ℝ,
      y = 2*x + c →  -- Original equation of l₂
      ∃ x' y' : ℝ,
        x' = (x - x₀) * Real.cos θ - (y - y₀) * Real.sin θ + x₀ ∧
        y' = (x - x₀) * Real.sin θ + (y - y₀) * Real.cos θ + y₀ ∧
        y' = x' * Real.sin α) -- Rotated l₂ coincides with l₁
  := by sorry

end NUMINAMATH_CALUDE_lines_can_coincide_by_rotation_l3773_377363


namespace NUMINAMATH_CALUDE_constant_sum_perpendicular_distances_l3773_377302

/-- A regular pentagon with circumradius R -/
structure RegularPentagon where
  R : ℝ
  R_pos : R > 0

/-- A point inside a regular pentagon -/
structure InnerPoint (p : RegularPentagon) where
  x : ℝ
  y : ℝ
  inside : x^2 + y^2 < p.R^2

/-- The sum of perpendicular distances from a point to the sides of a regular pentagon -/
noncomputable def sum_perpendicular_distances (p : RegularPentagon) (k : InnerPoint p) : ℝ :=
  sorry

/-- Theorem stating that the sum of perpendicular distances is constant -/
theorem constant_sum_perpendicular_distances (p : RegularPentagon) :
  ∃ (c : ℝ), ∀ (k : InnerPoint p), sum_perpendicular_distances p k = c :=
sorry

end NUMINAMATH_CALUDE_constant_sum_perpendicular_distances_l3773_377302


namespace NUMINAMATH_CALUDE_prob_five_shots_expected_shots_l3773_377350

-- Define the probability of hitting a target
variable (p : ℝ) (hp : 0 < p) (hp1 : p < 1)

-- Define the number of targets
def num_targets : ℕ := 3

-- Theorem for part (a)
theorem prob_five_shots : 
  (6 : ℝ) * p^3 * (1 - p)^2 = 
  (num_targets.choose 2) * p^3 * (1 - p)^2 := by sorry

-- Theorem for part (b)
theorem expected_shots : 
  (3 : ℝ) / p = num_targets / p := by sorry

end NUMINAMATH_CALUDE_prob_five_shots_expected_shots_l3773_377350


namespace NUMINAMATH_CALUDE_min_people_like_both_tea_and_coffee_l3773_377378

theorem min_people_like_both_tea_and_coffee
  (total : ℕ)
  (tea_lovers : ℕ)
  (coffee_lovers : ℕ)
  (h1 : total = 150)
  (h2 : tea_lovers = 120)
  (h3 : coffee_lovers = 100) :
  (tea_lovers + coffee_lovers - total : ℤ) ≥ 70 :=
sorry

end NUMINAMATH_CALUDE_min_people_like_both_tea_and_coffee_l3773_377378


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l3773_377324

/-- 
Given a triangle with two sides of 27 units and 34 units, and the third side having an integral length,
the least possible perimeter is 69 units.
-/
theorem least_perimeter_triangle : 
  ∀ z : ℕ, 
  z > 0 → 
  z + 27 > 34 → 
  34 + 27 > z → 
  27 + z > 34 → 
  ∀ w : ℕ, 
  w > 0 → 
  w + 27 > 34 → 
  34 + 27 > w → 
  27 + w > 34 → 
  w ≥ z → 
  27 + 34 + w ≥ 69 :=
by sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l3773_377324


namespace NUMINAMATH_CALUDE_circle_center_l3773_377326

theorem circle_center (x y : ℝ) : 
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 → 
  ∃ (h k : ℝ), h = 2 ∧ k = -1 ∧ (x - h)^2 + (y - k)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3773_377326


namespace NUMINAMATH_CALUDE_zero_not_in_empty_set_l3773_377358

theorem zero_not_in_empty_set : 0 ∉ (∅ : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_empty_set_l3773_377358


namespace NUMINAMATH_CALUDE_solve_equation_l3773_377394

theorem solve_equation : ∃ x : ℝ, 0.3 * x + 0.1 * 0.5 = 0.29 ∧ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3773_377394


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_five_l3773_377330

theorem sqrt_equality_implies_one_five (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  (Real.sqrt (1 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 5) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_five_l3773_377330


namespace NUMINAMATH_CALUDE_new_ratio_after_removing_clothing_l3773_377323

/-- Represents the ratio of books to clothes to electronics -/
structure Ratio :=
  (books : ℕ)
  (clothes : ℕ)
  (electronics : ℕ)

/-- Calculates the new ratio of books to clothes after removing some clothing -/
def newRatio (initial : Ratio) (electronicsWeight : ℕ) (clothingRemoved : ℕ) : Ratio :=
  sorry

/-- Theorem stating the new ratio after removing clothing -/
theorem new_ratio_after_removing_clothing 
  (initial : Ratio)
  (electronicsWeight : ℕ)
  (clothingRemoved : ℕ)
  (h1 : initial = ⟨7, 4, 3⟩)
  (h2 : electronicsWeight = 9)
  (h3 : clothingRemoved = 6) :
  (newRatio initial electronicsWeight clothingRemoved).books = 7 ∧
  (newRatio initial electronicsWeight clothingRemoved).clothes = 2 :=
by sorry

end NUMINAMATH_CALUDE_new_ratio_after_removing_clothing_l3773_377323


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l3773_377362

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 150) (h2 : b = 180) :
  (Nat.gcd a b) * (Nat.lcm a b) = 54000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l3773_377362


namespace NUMINAMATH_CALUDE_unknown_number_in_set_l3773_377348

theorem unknown_number_in_set (x : ℝ) : 
  ((14 + 32 + 53) / 3 = (21 + x + 22) / 3 + 3) → x = 47 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_in_set_l3773_377348


namespace NUMINAMATH_CALUDE_quadratic_form_decomposition_l3773_377340

theorem quadratic_form_decomposition (x y z : ℝ) : 
  x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2 = 
  (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_decomposition_l3773_377340


namespace NUMINAMATH_CALUDE_cubic_roots_theorem_l3773_377388

theorem cubic_roots_theorem (a b c : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x - c = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a = 1 ∧ b = -2 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_theorem_l3773_377388


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l3773_377369

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom inequality_condition : ∀ x y : ℝ, f (x^2 - 2*x) ≤ -f (2*y - y^2)
axiom symmetry_condition : ∀ x : ℝ, f (x - 1) = f (1 - x)

-- Define the theorem
theorem range_of_y_over_x :
  (∀ x y : ℝ, 1 ≤ x → x ≤ 4 → f x = y → -1/2 ≤ y/x ∧ y/x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l3773_377369


namespace NUMINAMATH_CALUDE_marie_magazine_sales_l3773_377390

/-- Given that Marie sold a total of 425.0 magazines and newspapers,
    and 275.0 of them were newspapers, prove that she sold 150.0 magazines. -/
theorem marie_magazine_sales :
  let total_sales : ℝ := 425.0
  let newspaper_sales : ℝ := 275.0
  let magazine_sales : ℝ := total_sales - newspaper_sales
  magazine_sales = 150.0 := by sorry

end NUMINAMATH_CALUDE_marie_magazine_sales_l3773_377390


namespace NUMINAMATH_CALUDE_union_membership_intersection_membership_positive_product_l3773_377333

-- Statement 1
theorem union_membership (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B := by sorry

-- Statement 2
theorem intersection_membership (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B := by sorry

-- Statement 3
theorem positive_product (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 := by sorry

end NUMINAMATH_CALUDE_union_membership_intersection_membership_positive_product_l3773_377333


namespace NUMINAMATH_CALUDE_unique_value_2n_plus_m_l3773_377328

theorem unique_value_2n_plus_m : ∃! v : ℤ, ∀ n m : ℤ,
  (3 * n - m < 5) →
  (n + m > 26) →
  (3 * m - 2 * n < 46) →
  (2 * n + m = v) := by
  sorry

end NUMINAMATH_CALUDE_unique_value_2n_plus_m_l3773_377328


namespace NUMINAMATH_CALUDE_chandler_can_buy_bike_l3773_377338

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 640

/-- The total amount of birthday money Chandler received in dollars -/
def birthday_money : ℕ := 60 + 40 + 20

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks Chandler needs to save to buy the bike -/
def weeks_to_save : ℕ := 26

/-- Theorem stating that Chandler can buy the bike after saving for 26 weeks -/
theorem chandler_can_buy_bike : 
  birthday_money + weekly_earnings * weeks_to_save = bike_cost := by
  sorry

end NUMINAMATH_CALUDE_chandler_can_buy_bike_l3773_377338


namespace NUMINAMATH_CALUDE_no_valid_rope_net_with_2001_knots_l3773_377343

/-- A rope net is a structure where knots are connected by ropes. -/
structure RopeNet where
  knots : ℕ
  ropes_per_knot : ℕ

/-- A valid rope net has a positive number of knots and exactly 3 ropes per knot. -/
def is_valid_rope_net (net : RopeNet) : Prop :=
  net.knots > 0 ∧ net.ropes_per_knot = 3

/-- The total number of rope ends in a rope net. -/
def total_rope_ends (net : RopeNet) : ℕ :=
  net.knots * net.ropes_per_knot

/-- The number of distinct ropes in a rope net. -/
def distinct_ropes (net : RopeNet) : ℚ :=
  (total_rope_ends net : ℚ) / 2

/-- Theorem: It is impossible for a valid rope net to have exactly 2001 knots. -/
theorem no_valid_rope_net_with_2001_knots :
  ¬ ∃ (net : RopeNet), is_valid_rope_net net ∧ net.knots = 2001 :=
sorry

end NUMINAMATH_CALUDE_no_valid_rope_net_with_2001_knots_l3773_377343


namespace NUMINAMATH_CALUDE_ratio_is_two_l3773_377332

/-- Three integers a, b, and c where a < b < c and a = 0 -/
def IntegerTriple := {abc : ℤ × ℤ × ℤ // abc.1 < abc.2.1 ∧ abc.2.1 < abc.2.2 ∧ abc.1 = 0}

/-- Three integers p, q, r where p < q < r and r ≠ 0 -/
def GeometricTriple := {pqr : ℤ × ℤ × ℤ // pqr.1 < pqr.2.1 ∧ pqr.2.1 < pqr.2.2 ∧ pqr.2.2 ≠ 0}

/-- The mean of three integers is half the median -/
def MeanHalfMedian (abc : IntegerTriple) : Prop :=
  (abc.val.1 + abc.val.2.1 + abc.val.2.2) / 3 = abc.val.2.1 / 2

/-- The product of three integers is 0 -/
def ProductZero (abc : IntegerTriple) : Prop :=
  abc.val.1 * abc.val.2.1 * abc.val.2.2 = 0

/-- Three integers are in geometric progression -/
def GeometricProgression (pqr : GeometricTriple) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ r ≠ 1 ∧ pqr.val.2.1 = pqr.val.1 * r ∧ pqr.val.2.2 = pqr.val.2.1 * r

/-- Sum of squares equals square of sum -/
def SumSquaresEqualSquareSum (abc : IntegerTriple) (pqr : GeometricTriple) : Prop :=
  abc.val.1^2 + abc.val.2.1^2 + abc.val.2.2^2 = (pqr.val.1 + pqr.val.2.1 + pqr.val.2.2)^2

theorem ratio_is_two (abc : IntegerTriple) (pqr : GeometricTriple)
  (h1 : MeanHalfMedian abc)
  (h2 : ProductZero abc)
  (h3 : GeometricProgression pqr)
  (h4 : SumSquaresEqualSquareSum abc pqr) :
  abc.val.2.2 / abc.val.2.1 = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_is_two_l3773_377332


namespace NUMINAMATH_CALUDE_even_function_inequality_l3773_377361

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

theorem even_function_inequality (f : ℝ → ℝ) (m : ℝ) :
  is_even_function f →
  (∀ x, -2 ≤ x → x ≤ 2 → f x ∈ Set.range f) →
  monotone_decreasing_on f 0 2 →
  f (1 - m) < f m →
  -1 ≤ m ∧ m < 1/2 := by sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3773_377361


namespace NUMINAMATH_CALUDE_sin_plus_cos_alpha_l3773_377331

theorem sin_plus_cos_alpha (α : ℝ) 
  (h1 : Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10)
  (h2 : Real.cos (2 * α) = 7/25) : 
  Real.sin α + Real.cos α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_alpha_l3773_377331


namespace NUMINAMATH_CALUDE_equidistant_points_on_axes_l3773_377371

/-- Given points A(1, 5) and B(2, 4), this theorem states that (0, 3) and (-3, 0) are the only points
    on the coordinate axes that are equidistant from A and B. -/
theorem equidistant_points_on_axes (A B P : ℝ × ℝ) : 
  A = (1, 5) → B = (2, 4) → 
  (P.1 = 0 ∨ P.2 = 0) →  -- P is on a coordinate axis
  (dist A P = dist B P) →  -- P is equidistant from A and B
  (P = (0, 3) ∨ P = (-3, 0)) :=
by sorry

#check equidistant_points_on_axes

end NUMINAMATH_CALUDE_equidistant_points_on_axes_l3773_377371


namespace NUMINAMATH_CALUDE_digit_value_in_different_bases_l3773_377342

theorem digit_value_in_different_bases :
  ∃ (d : ℕ), d < 7 ∧ d * 7 + 4 = d * 8 + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_digit_value_in_different_bases_l3773_377342


namespace NUMINAMATH_CALUDE_original_room_width_l3773_377335

/-- Proves that the original width of the room is 18 feet given the problem conditions -/
theorem original_room_width (length : ℝ) (increased_size : ℝ) (total_area : ℝ) : 
  length = 13 →
  increased_size = 2 →
  total_area = 1800 →
  ∃ w : ℝ, 
    (4 * ((length + increased_size) * (w + increased_size)) + 
     2 * ((length + increased_size) * (w + increased_size))) = total_area ∧
    w = 18 := by
  sorry

end NUMINAMATH_CALUDE_original_room_width_l3773_377335


namespace NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l3773_377399

theorem cos_double_angle_from_series_sum (θ : ℝ) 
  (h : ∑' n, (Real.cos θ) ^ (2 * n) = 9) : 
  Real.cos (2 * θ) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l3773_377399


namespace NUMINAMATH_CALUDE_cubic_sum_from_system_l3773_377304

theorem cubic_sum_from_system (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^3 + y^3 = 416000 / 729 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_from_system_l3773_377304


namespace NUMINAMATH_CALUDE_farm_animals_product_l3773_377380

theorem farm_animals_product (pigs chickens : ℕ) : 
  chickens = pigs + 12 →
  chickens + pigs = 52 →
  pigs * chickens = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_product_l3773_377380


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3773_377365

def A : Set ℝ := {x : ℝ | x > 3}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 4) < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 3 4 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3773_377365


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l3773_377312

theorem ellipse_foci_on_y_axis (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (2*k - 1) = 1 → 
    (∃ c : ℝ, c > 0 ∧ 
      ∀ p : ℝ × ℝ, 
        (p.1 = 0 → (p.2 = c ∨ p.2 = -c)) ∧ 
        (p.2 = c ∨ p.2 = -c → p.1 = 0))) → 
  1 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l3773_377312


namespace NUMINAMATH_CALUDE_xiao_ming_brother_age_l3773_377359

/-- Check if a year has unique digits -/
def has_unique_digits (year : Nat) : Bool := sorry

/-- Find the latest year before 2013 that is a multiple of 19 and has unique digits -/
def find_birth_year : Nat := sorry

/-- Calculate age in 2013 given a birth year -/
def calculate_age (birth_year : Nat) : Nat := 2013 - birth_year

theorem xiao_ming_brother_age :
  (∀ y : Nat, y < 2013 → ¬(has_unique_digits y)) →
  has_unique_digits 2013 →
  find_birth_year % 19 = 0 →
  has_unique_digits find_birth_year →
  calculate_age find_birth_year = 18 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_brother_age_l3773_377359


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l3773_377393

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l3773_377393


namespace NUMINAMATH_CALUDE_remainder_preserved_l3773_377385

theorem remainder_preserved (n : ℤ) (h : n % 8 = 3) : (n + 5040) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_preserved_l3773_377385


namespace NUMINAMATH_CALUDE_combined_mean_of_sets_l3773_377397

theorem combined_mean_of_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) 
  (new_set1_count : ℕ) (new_set1_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 21 →
  new_set1_count = set1_count + 1 →
  new_set1_mean = 16 →
  let total_count := new_set1_count + set2_count
  let total_sum := new_set1_mean * new_set1_count + set2_mean * set2_count
  (total_sum / total_count : ℚ) = 37/2 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_sets_l3773_377397


namespace NUMINAMATH_CALUDE_cloth_square_theorem_l3773_377366

/-- Represents a rectangular piece of cloth -/
structure Cloth where
  length : ℕ
  width : ℕ

/-- Represents a square -/
structure Square where
  side : ℕ

/-- Calculates the maximum number of squares that can be cut from a cloth -/
def maxSquares (c : Cloth) (s : Square) : ℕ :=
  (c.length / s.side) * (c.width / s.side)

theorem cloth_square_theorem :
  let cloth : Cloth := { length := 40, width := 27 }
  let square : Square := { side := 2 }
  maxSquares cloth square = 260 := by
  sorry

#eval maxSquares { length := 40, width := 27 } { side := 2 }

end NUMINAMATH_CALUDE_cloth_square_theorem_l3773_377366


namespace NUMINAMATH_CALUDE_x_value_l3773_377329

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : x = 134 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3773_377329


namespace NUMINAMATH_CALUDE_max_value_trigonometric_function_l3773_377374

open Real

theorem max_value_trigonometric_function :
  ∃ (max : ℝ), max = 6 - 4 * Real.sqrt 2 ∧
  ∀ θ : ℝ, θ ∈ Set.Ioo 0 (π / 2) →
    (2 * sin θ * cos θ) / ((sin θ + 1) * (cos θ + 1)) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_function_l3773_377374


namespace NUMINAMATH_CALUDE_probability_is_one_third_l3773_377347

/-- Right triangle XYZ with XY = 10 and XZ = 6 -/
structure RightTriangle where
  xy : ℝ
  xz : ℝ
  xy_eq : xy = 10
  xz_eq : xz = 6

/-- Random point Q in the interior of triangle XYZ -/
def RandomPoint (t : RightTriangle) : Type := Unit

/-- Area of triangle QYZ -/
def AreaQYZ (t : RightTriangle) (q : RandomPoint t) : ℝ := sorry

/-- Area of triangle XYZ -/
def AreaXYZ (t : RightTriangle) : ℝ := sorry

/-- Probability that area of QYZ is less than one-third of area of XYZ -/
def Probability (t : RightTriangle) : ℝ := sorry

/-- Theorem: The probability is equal to 1/3 -/
theorem probability_is_one_third (t : RightTriangle) :
  Probability t = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l3773_377347


namespace NUMINAMATH_CALUDE_train_length_problem_l3773_377367

theorem train_length_problem (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 47) (h2 : slower_speed = 36) 
  (h3 : passing_time = 36) : ∃ (train_length : ℝ), train_length = 55 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l3773_377367


namespace NUMINAMATH_CALUDE_expression_equality_l3773_377389

theorem expression_equality : (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3773_377389


namespace NUMINAMATH_CALUDE_ln_inequality_solution_set_l3773_377318

theorem ln_inequality_solution_set :
  {x : ℝ | Real.log (2 * x - 1) < 0} = Set.Ioo (1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ln_inequality_solution_set_l3773_377318


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3773_377309

/-- The minimum distance from any point on the line x + y - 4 = 0 to the origin (0, 0) is 2√2 -/
theorem min_distance_to_origin : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∀ p ∈ line, Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2)) ≥ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3773_377309


namespace NUMINAMATH_CALUDE_mary_nickels_problem_l3773_377373

theorem mary_nickels_problem (initial : ℕ) (given : ℕ) (total : ℕ) : 
  given = 5 → total = 12 → initial + given = total → initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_problem_l3773_377373


namespace NUMINAMATH_CALUDE_tangent_slope_points_l3773_377387

theorem tangent_slope_points (x y : ℝ) : 
  y = x^3 ∧ (3 * x^2 = 3) ↔ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_points_l3773_377387


namespace NUMINAMATH_CALUDE_data_properties_l3773_377381

def data : List ℕ := [3, 3, 4, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ := sorry

def fractionLessThanMode (l : List ℕ) : ℚ := sorry

def firstQuartile (l : List ℕ) : ℕ := sorry

def medianWithinFirstQuartile (l : List ℕ) : ℚ := sorry

theorem data_properties :
  fractionLessThanMode data = 4/11 ∧
  medianWithinFirstQuartile data = 4 := by sorry

end NUMINAMATH_CALUDE_data_properties_l3773_377381


namespace NUMINAMATH_CALUDE_range_of_distance_from_origin_l3773_377337

theorem range_of_distance_from_origin : ∀ x y : ℝ,
  x + y = 10 →
  -5 ≤ x - y →
  x - y ≤ 5 →
  5 * Real.sqrt 2 ≤ Real.sqrt (x^2 + y^2) ∧
  Real.sqrt (x^2 + y^2) ≤ (5 * Real.sqrt 10) / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_distance_from_origin_l3773_377337


namespace NUMINAMATH_CALUDE_student_count_l3773_377322

theorem student_count : ∃ S : ℕ, 
  (S / 3 : ℚ) + 10 = S - 6 ∧ S = 24 := by sorry

end NUMINAMATH_CALUDE_student_count_l3773_377322


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l3773_377316

theorem magnitude_of_complex_fourth_power :
  Complex.abs ((5 : ℂ) + (2 * Complex.I * Real.sqrt 3)) ^ 4 = 1369 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l3773_377316


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_l3773_377355

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (2 * a - 2, a + 5)

-- Define the point Q
def Q : ℝ × ℝ := (4, 5)

-- Theorem for part 1
theorem P_on_x_axis (a : ℝ) : 
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Theorem for part 2
theorem P_parallel_to_y_axis (a : ℝ) :
  (P a).1 = Q.1 → P a = (4, 8) ∧ (P a).1 > 0 ∧ (P a).2 > 0 :=
sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_l3773_377355


namespace NUMINAMATH_CALUDE_weaving_problem_l3773_377392

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- The weaving problem -/
theorem weaving_problem : arithmeticSum 5 (16/29) 30 = 390 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l3773_377392


namespace NUMINAMATH_CALUDE_ascending_order_abc_l3773_377376

theorem ascending_order_abc : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l3773_377376


namespace NUMINAMATH_CALUDE_trajectory_of_point_B_l3773_377370

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a parallelogram ABCD -/
def is_parallelogram (A B C D : Point) : Prop := sorry

/-- Definition of a point lying on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem: Trajectory of point B in parallelogram ABCD -/
theorem trajectory_of_point_B 
  (A B C D : Point)
  (h_parallelogram : is_parallelogram A B C D)
  (h_A : A = ⟨3, -1⟩)
  (h_C : C = ⟨2, -3⟩)
  (l : Line)
  (h_l : l = ⟨3, -1, 1⟩)
  (h_D_on_l : point_on_line D l) :
  point_on_line B ⟨3, -1, -20⟩ := by
    sorry

end NUMINAMATH_CALUDE_trajectory_of_point_B_l3773_377370


namespace NUMINAMATH_CALUDE_rectangle_area_l3773_377364

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3773_377364


namespace NUMINAMATH_CALUDE_smallest_multiple_divisible_by_all_up_to_20_l3773_377341

/-- The smallest positive integer divisible by all numbers from 1 to 20 -/
def smallestMultiple : Nat := 232792560

/-- Checks if a number is divisible by all integers from 1 to 20 -/
def divisibleByAllUpTo20 (n : Nat) : Prop :=
  ∀ i : Nat, 1 ≤ i ∧ i ≤ 20 → n % i = 0

theorem smallest_multiple_divisible_by_all_up_to_20 :
  divisibleByAllUpTo20 smallestMultiple ∧
  ∀ n : Nat, n > 0 ∧ n < smallestMultiple → ¬(divisibleByAllUpTo20 n) := by
  sorry

#eval smallestMultiple

end NUMINAMATH_CALUDE_smallest_multiple_divisible_by_all_up_to_20_l3773_377341


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3773_377391

theorem sqrt_equation_solutions (x : ℝ) :
  Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 5 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3773_377391


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l3773_377346

theorem geometric_series_second_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/4) 
  (h2 : S = 16) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : second_term = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l3773_377346


namespace NUMINAMATH_CALUDE_magnitude_of_2_plus_i_l3773_377311

theorem magnitude_of_2_plus_i : Complex.abs (2 + Complex.I) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_2_plus_i_l3773_377311


namespace NUMINAMATH_CALUDE_complex_operations_l3773_377377

theorem complex_operations (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 2 + 3*I) (h₂ : z₂ = 5 - 7*I) : 
  (z₁ + z₂ = 7 - 4*I) ∧ 
  (z₁ - z₂ = -3 + 10*I) ∧ 
  (z₁ * z₂ = 31 + I) := by
  sorry

#check complex_operations

end NUMINAMATH_CALUDE_complex_operations_l3773_377377


namespace NUMINAMATH_CALUDE_parallelogram_height_l3773_377379

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 216) 
  (h_base : base = 12) 
  (h_formula : area = base * height) : 
  height = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3773_377379


namespace NUMINAMATH_CALUDE_line_intersection_canonical_equations_l3773_377375

/-- The canonical equations of the line of intersection of two planes -/
theorem line_intersection_canonical_equations
  (p₁ : Real → Real → Real → Real)
  (p₂ : Real → Real → Real → Real)
  (h₁ : ∀ x y z, p₁ x y z = 3*x + y - z - 6)
  (h₂ : ∀ x y z, p₂ x y z = 3*x - y + 2*z)
  : ∃ (t : Real), ∀ x y z,
    (p₁ x y z = 0 ∧ p₂ x y z = 0) ↔
    (x = 1 + t ∧ y = 3 - 9*t ∧ z = -6*t) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_equations_l3773_377375


namespace NUMINAMATH_CALUDE_onions_on_scale_l3773_377306

/-- The number of onions initially on the scale -/
def N : ℕ := sorry

/-- The total weight of onions in grams -/
def W : ℕ := 7680

/-- The average weight of remaining onions in grams -/
def avg_remaining : ℕ := 190

/-- The average weight of removed onions in grams -/
def avg_removed : ℕ := 206

/-- The number of removed onions -/
def removed : ℕ := 5

theorem onions_on_scale :
  W = (N - removed) * avg_remaining + removed * avg_removed ∧ N = 40 := by sorry

end NUMINAMATH_CALUDE_onions_on_scale_l3773_377306


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3773_377349

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 1) * (n + 3) * (n + 7) < 0)
    (Finset.Icc (-10 : ℤ) 12)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3773_377349


namespace NUMINAMATH_CALUDE_hex_to_decimal_conversion_l3773_377368

/-- Given that the hexadecimal number (3m502_(16)) is equal to 4934 in decimal,
    prove that m = 4. -/
theorem hex_to_decimal_conversion (m : ℕ) : 
  (3 * 16^4 + m * 16^3 + 5 * 16^2 + 0 * 16^1 + 2 * 16^0 = 4934) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_decimal_conversion_l3773_377368


namespace NUMINAMATH_CALUDE_two_distinct_prime_factors_l3773_377384

def append_threes (n : ℕ) : ℕ :=
  12320 * 4^(10*n + 1) + (4^(10*n + 1) - 1) / 3

theorem two_distinct_prime_factors (n : ℕ) : 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ 
   append_threes n = p * q) ↔ n = 0 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_prime_factors_l3773_377384


namespace NUMINAMATH_CALUDE_composite_sequence_existence_l3773_377315

theorem composite_sequence_existence (m : ℕ) (hm : m > 0) :
  ∃ n : ℕ, ∀ i : ℤ, -m ≤ i ∧ i ≤ m → 
    (2 : ℕ)^n + i > 0 ∧ ¬(Nat.Prime ((2 : ℕ)^n + i).toNat) := by
  sorry

end NUMINAMATH_CALUDE_composite_sequence_existence_l3773_377315


namespace NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l3773_377327

theorem remainder_3_1000_mod_7 : 3^1000 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l3773_377327


namespace NUMINAMATH_CALUDE_translation_of_A_l3773_377357

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (5, 2)
def C : ℝ × ℝ := (3, -1)

-- Define the translation function
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + (C.1 - B.1), p.2 + (C.2 - B.2))

-- Theorem statement
theorem translation_of_A :
  translate A = (0, 1) := by sorry

end NUMINAMATH_CALUDE_translation_of_A_l3773_377357


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3773_377305

def is_valid_number (N : ℕ) : Prop :=
  ∃ X : ℕ, 
    X > 0 ∧
    (N - 12) % 8 = 0 ∧
    (N - 12) % 12 = 0 ∧
    (N - 12) % 24 = 0 ∧
    (N - 12) % X = 0 ∧
    (N - 12) / Nat.lcm 24 X = 276

theorem smallest_valid_number : 
  is_valid_number 6636 ∧ ∀ n < 6636, ¬ is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3773_377305


namespace NUMINAMATH_CALUDE_larger_number_problem_l3773_377386

theorem larger_number_problem (x y : ℝ) : 
  x - y = 5 → x + y = 27 → max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3773_377386


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3773_377325

theorem arctan_equation_solution (y : ℝ) :
  2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 →
  y = -121/60 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3773_377325


namespace NUMINAMATH_CALUDE_sophomore_sample_count_l3773_377372

/-- Represents a stratified sampling scenario in a high school. -/
structure HighSchoolSampling where
  total_students : ℕ
  sophomore_count : ℕ
  sample_size : ℕ

/-- Calculates the number of sophomores in a stratified sample. -/
def sophomores_in_sample (h : HighSchoolSampling) : ℕ :=
  (h.sophomore_count * h.sample_size) / h.total_students

/-- Theorem: The number of sophomores in the sample is 93 given the specific scenario. -/
theorem sophomore_sample_count (h : HighSchoolSampling) 
  (h_total : h.total_students = 2800)
  (h_sophomores : h.sophomore_count = 930)
  (h_sample : h.sample_size = 280) : 
  sophomores_in_sample h = 93 := by
sorry

end NUMINAMATH_CALUDE_sophomore_sample_count_l3773_377372


namespace NUMINAMATH_CALUDE_random_walk_prob_4_in_3_to_9_l3773_377307

/-- A one-dimensional random walk on integers -/
def RandomWalk := ℕ → ℤ

/-- The probability of a random walk reaching a specific distance -/
def prob_reach_distance (w : RandomWalk) (d : ℕ) (steps : ℕ) : ℚ :=
  sorry

/-- The probability of a random walk reaching a specific distance at least once within a range of steps -/
def prob_reach_distance_in_range (w : RandomWalk) (d : ℕ) (min_steps max_steps : ℕ) : ℚ :=
  sorry

/-- The main theorem: probability of reaching distance 4 at least once in 3 to 9 steps is 47/224 -/
theorem random_walk_prob_4_in_3_to_9 (w : RandomWalk) :
  prob_reach_distance_in_range w 4 3 9 = 47 / 224 := by
  sorry

end NUMINAMATH_CALUDE_random_walk_prob_4_in_3_to_9_l3773_377307


namespace NUMINAMATH_CALUDE_smallest_b_value_l3773_377351

/-- The second smallest positive integer with exactly 3 factors -/
def a : ℕ := 9

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_b_value :
  ∃ b : ℕ,
    b > 0 ∧
    num_factors b = a ∧
    a ∣ b ∧
    ∀ c : ℕ, c > 0 → num_factors c = a → a ∣ c → b ≤ c ∧
    b = 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3773_377351


namespace NUMINAMATH_CALUDE_feet_quadrilateral_similar_l3773_377344

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The feet of perpendiculars from vertices to diagonals -/
def feet_of_perpendiculars (q : Quadrilateral) : Quadrilateral :=
  sorry

/-- Similarity relation between two quadrilaterals -/
def is_similar (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- Theorem: The quadrilateral formed by the feet of perpendiculars 
    is similar to the original quadrilateral -/
theorem feet_quadrilateral_similar (q : Quadrilateral) :
  is_similar q (feet_of_perpendiculars q) :=
sorry

end NUMINAMATH_CALUDE_feet_quadrilateral_similar_l3773_377344


namespace NUMINAMATH_CALUDE_symmetric_circle_l3773_377300

/-- Given a circle and a line of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle (x y : ℝ) : 
  -- Original circle
  ((x - 2)^2 + (y - 3)^2 = 1) →
  -- Line of symmetry
  (x + y - 1 = 0) →
  -- Symmetric circle
  ((x + 2)^2 + (y + 1)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_l3773_377300


namespace NUMINAMATH_CALUDE_right_triangle_to_square_l3773_377303

theorem right_triangle_to_square (a b : ℝ) : 
  b = 10 → -- longer leg is 10
  a * b / 2 = a^2 → -- area of triangle equals area of square
  b = 2 * a → -- longer leg is twice the shorter leg
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_to_square_l3773_377303


namespace NUMINAMATH_CALUDE_car_travel_distance_l3773_377353

/-- Proves that Car X travels 294 miles from when Car Y starts until both cars stop -/
theorem car_travel_distance (speed_x speed_y : ℝ) (head_start : ℝ) : 
  speed_x = 35 →
  speed_y = 40 →
  head_start = 1.2 →
  (speed_y * (head_start + (294 / speed_x))) = (speed_x * (294 / speed_x) + speed_x * head_start) →
  294 = speed_x * (294 / speed_x) :=
by
  sorry

#check car_travel_distance

end NUMINAMATH_CALUDE_car_travel_distance_l3773_377353


namespace NUMINAMATH_CALUDE_white_balls_added_l3773_377352

theorem white_balls_added (m : ℕ) : 
  (10 + m : ℚ) / (16 + m) = 4/5 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_added_l3773_377352


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3773_377395

theorem sum_of_three_numbers (a b c : ℤ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 52)
  (sum_ca : c + a = 61) :
  a + b + c = 74 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3773_377395


namespace NUMINAMATH_CALUDE_weather_probability_l3773_377317

theorem weather_probability (p_rain p_cloudy : ℝ) 
  (h_rain : p_rain = 0.45)
  (h_cloudy : p_cloudy = 0.20)
  (h_nonneg_rain : 0 ≤ p_rain)
  (h_nonneg_cloudy : 0 ≤ p_cloudy)
  (h_sum_le_one : p_rain + p_cloudy ≤ 1) :
  1 - p_rain - p_cloudy = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_weather_probability_l3773_377317


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3773_377320

theorem cube_sum_theorem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3773_377320


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3773_377310

/-- A geometric sequence {a_n} satisfying given conditions has the general term formula a_n = 1 / (2^(n-4)) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2^(n-4)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3773_377310


namespace NUMINAMATH_CALUDE_max_pencils_is_13_l3773_377313

def john_money : ℚ := 10
def regular_price : ℚ := 0.75
def discount_price : ℚ := 0.65
def discount_threshold : ℕ := 10

def cost (n : ℕ) : ℚ :=
  if n ≤ discount_threshold then
    n * regular_price
  else
    discount_threshold * regular_price + (n - discount_threshold) * discount_price

def can_afford (n : ℕ) : Prop :=
  cost n ≤ john_money

theorem max_pencils_is_13 :
  ∀ n : ℕ, can_afford n → n ≤ 13 ∧
  ∃ m : ℕ, m = 13 ∧ can_afford m :=
by sorry

end NUMINAMATH_CALUDE_max_pencils_is_13_l3773_377313


namespace NUMINAMATH_CALUDE_geometric_arithmetic_relation_l3773_377339

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

def arithmetic_sequence (b : ℕ → ℝ) := ∀ n, b (n + 1) - b n = b 2 - b 1

theorem geometric_arithmetic_relation (a : ℕ → ℝ) (b : ℕ → ℝ) :
  geometric_sequence a ∧ 
  a 1 = 2 ∧ 
  a 4 = 16 ∧
  arithmetic_sequence b ∧
  b 3 = a 3 ∧
  b 5 = a 5 →
  (∀ n, a n = 2^n) ∧ 
  b 45 = a 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_relation_l3773_377339


namespace NUMINAMATH_CALUDE_power_product_eq_four_l3773_377321

theorem power_product_eq_four (a b : ℕ+) (h : (3 ^ a.val) ^ b.val = 3 ^ 3) :
  3 ^ a.val * 3 ^ b.val = 3 ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eq_four_l3773_377321


namespace NUMINAMATH_CALUDE_infinite_solutions_l3773_377314

theorem infinite_solutions (k : ℝ) : 
  (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_l3773_377314


namespace NUMINAMATH_CALUDE_second_year_increase_is_25_percent_l3773_377354

/-- Calculates the percentage increase in the second year given the initial population,
    first year increase percentage, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_first_year := initial_population * (1 + first_year_increase)
  let second_year_factor := final_population / population_after_first_year
  (second_year_factor - 1) * 100

theorem second_year_increase_is_25_percent :
  second_year_increase 800 (22/100) 1220 = 25 := by
  sorry

#eval second_year_increase 800 (22/100) 1220

end NUMINAMATH_CALUDE_second_year_increase_is_25_percent_l3773_377354


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l3773_377301

/-- Proves that when the surface area of a sphere is increased to 4 times its original size,
    its volume is increased to 8 times the original. -/
theorem sphere_volume_increase (r : ℝ) (S V : ℝ → ℝ) 
    (hS : ∃ k : ℝ, ∀ x, S x = k * x^2)  -- Surface area is proportional to radius squared
    (hV : ∃ c : ℝ, ∀ x, V x = c * x^3)  -- Volume is proportional to radius cubed
    (hS_increase : S (2 * r) = 4 * S r) : -- Surface area is increased 4 times
  V (2 * r) = 8 * V r := by
sorry


end NUMINAMATH_CALUDE_sphere_volume_increase_l3773_377301


namespace NUMINAMATH_CALUDE_inequality_proof_l3773_377308

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt (a * b)) / (Real.sqrt a + Real.sqrt b) ≤ (a * b) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3773_377308


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3773_377360

theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : a + b + c = 39)
  (diagonal : a^2 + b^2 + c^2 = 625) :
  2 * (a * b + b * c + c * a) = 896 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3773_377360


namespace NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l3773_377345

theorem infinitely_many_a_for_perfect_cube (n : ℕ) :
  ∃ (f : ℕ → ℤ), Function.Injective f ∧ ∀ (k : ℕ), ∃ (m : ℕ), (n^6 + 3 * (f k) : ℤ) = m^3 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l3773_377345


namespace NUMINAMATH_CALUDE_solve_equation_l3773_377382

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3773_377382


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_20_l3773_377319

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers that sum to 20 -/
theorem unique_consecutive_sum_20 : 
  ∃! s : ConsecutiveSet, sum_consecutive s = 20 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_20_l3773_377319


namespace NUMINAMATH_CALUDE_no_rational_solution_l3773_377334

theorem no_rational_solution : ¬∃ (p q r : ℚ), p + q + r = 0 ∧ p * q * r = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l3773_377334


namespace NUMINAMATH_CALUDE_probability_of_event_A_l3773_377356

/-- A tetrahedron with faces numbered 0, 1, 2, and 3 -/
inductive TetrahedronFace
| Zero
| One
| Two
| Three

/-- The result of throwing the tetrahedron twice -/
structure ThrowResult where
  first : TetrahedronFace
  second : TetrahedronFace

/-- Convert TetrahedronFace to a natural number -/
def faceToNat (face : TetrahedronFace) : Nat :=
  match face with
  | TetrahedronFace.Zero => 0
  | TetrahedronFace.One => 1
  | TetrahedronFace.Two => 2
  | TetrahedronFace.Three => 3

/-- Event A: m^2 + n^2 ≤ 4 -/
def eventA (result : ThrowResult) : Prop :=
  let m := faceToNat result.first
  let n := faceToNat result.second
  m^2 + n^2 ≤ 4

/-- The probability of event A occurring -/
def probabilityOfEventA : ℚ := 3/8

theorem probability_of_event_A :
  probabilityOfEventA = 3/8 := by sorry

end NUMINAMATH_CALUDE_probability_of_event_A_l3773_377356


namespace NUMINAMATH_CALUDE_expression_simplification_l3773_377336

theorem expression_simplification (a : ℝ) (h : a = -2) :
  (1 - a / (a + 1)) / (1 / (1 - a^2)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3773_377336


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3773_377383

def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

theorem min_value_implies_a (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 5 ∧ ∀ x : ℝ, f a x ≥ 5) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3773_377383
