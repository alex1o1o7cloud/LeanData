import Mathlib

namespace NUMINAMATH_CALUDE_calculation_result_l4072_407260

theorem calculation_result : 12.05 * 5.4 + 0.6 = 65.67 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l4072_407260


namespace NUMINAMATH_CALUDE_book_price_increase_l4072_407229

theorem book_price_increase (original_price : ℝ) : 
  original_price > 0 →
  original_price * 1.5 = 450 →
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l4072_407229


namespace NUMINAMATH_CALUDE_square_root_of_nine_l4072_407224

theorem square_root_of_nine (x : ℝ) : x^2 = 9 → (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l4072_407224


namespace NUMINAMATH_CALUDE_min_sum_of_valid_set_l4072_407215

def is_valid_set (s : Finset ℕ) : Prop :=
  s.card = 10 ∧ 
  (∀ t ⊆ s, t.card = 5 → Even (t.prod id)) ∧
  Odd (s.sum id)

theorem min_sum_of_valid_set :
  ∃ (s : Finset ℕ), is_valid_set s ∧ 
  (∀ t : Finset ℕ, is_valid_set t → s.sum id ≤ t.sum id) ∧
  s.sum id = 65 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_valid_set_l4072_407215


namespace NUMINAMATH_CALUDE_cistern_fill_time_l4072_407228

-- Define the fill rates of pipes
def fill_rate_p : ℚ := 1 / 10
def fill_rate_q : ℚ := 1 / 15
def drain_rate_r : ℚ := 1 / 30

-- Define the time pipes p and q are open together
def initial_time : ℚ := 2

-- Define the function to calculate the remaining time to fill the cistern
def remaining_fill_time (fill_rate_p fill_rate_q drain_rate_r initial_time : ℚ) : ℚ :=
  let initial_fill := (fill_rate_p + fill_rate_q) * initial_time
  let remaining_volume := 1 - initial_fill
  let net_fill_rate := fill_rate_q - drain_rate_r
  remaining_volume / net_fill_rate

-- Theorem statement
theorem cistern_fill_time :
  remaining_fill_time fill_rate_p fill_rate_q drain_rate_r initial_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l4072_407228


namespace NUMINAMATH_CALUDE_smaller_square_side_length_l4072_407202

/-- A square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- An equilateral triangle with vertices P, T, U where T is on RS and U is on SQ of square PQRS -/
structure EquilateralTriangle (sq : Square) :=
  (P T U : ℝ × ℝ)
  (is_equilateral : sorry)
  (T_on_RS : sorry)
  (U_on_SQ : sorry)

/-- A smaller square with vertex R and a vertex on PT -/
structure SmallerSquare (sq : Square) (tri : EquilateralTriangle sq) :=
  (side : ℝ)
  (vertex_on_PT : sorry)
  (sides_parallel : sorry)

/-- The theorem stating the properties of the smaller square's side length -/
theorem smaller_square_side_length 
  (sq : Square) 
  (tri : EquilateralTriangle sq) 
  (small_sq : SmallerSquare sq tri) :
  ∃ (d e f : ℕ), 
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    ¬ (∃ (p : ℕ), Prime p ∧ p^2 ∣ e) ∧
    small_sq.side = (d - Real.sqrt e) / f ∧
    d = 4 ∧ e = 10 ∧ f = 3 ∧
    d + e + f = 17 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_side_length_l4072_407202


namespace NUMINAMATH_CALUDE_root_sum_gt_one_l4072_407232

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (x^2 - x) * f a x

theorem root_sum_gt_one (a m : ℝ) (x₁ x₂ : ℝ) :
  a < 0 →
  x₁ ≠ x₂ →
  h a x₁ = m →
  h a x₂ = m →
  x₁ + x₂ > 1 := by
sorry

end NUMINAMATH_CALUDE_root_sum_gt_one_l4072_407232


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l4072_407289

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  size : Nat
  h_size : vertices.length = size
  h_size_ge_3 : size ≥ 3

/-- A guard position -/
def Guard := ℝ × ℝ

/-- A point is visible from a guard if the line segment between them doesn't intersect any edge of the polygon -/
def isVisible (p : Polygon) (point guard : ℝ × ℝ) : Prop := sorry

/-- A set of guards covers a polygon if every point in the polygon is visible from at least one guard -/
def covers (p : Polygon) (guards : List Guard) : Prop :=
  ∀ point, ∃ guard ∈ guards, isVisible p point guard

/-- The main theorem: ⌊n/3⌋ guards are sufficient to cover any polygon with n sides -/
theorem art_gallery_theorem (p : Polygon) :
  ∃ guards : List Guard, guards.length ≤ p.size / 3 ∧ covers p guards := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l4072_407289


namespace NUMINAMATH_CALUDE_division_multiplication_identity_l4072_407299

theorem division_multiplication_identity (a : ℝ) (h : a ≠ 0) : 1 / a * a = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_identity_l4072_407299


namespace NUMINAMATH_CALUDE_existence_of_k_values_l4072_407241

/-- Represents a triple of numbers -/
structure Triple :=
  (a b c : ℤ)

/-- Checks if the sums of powers with exponents 1, 2, and 3 are equal for two triples -/
def sumPowersEqual (t1 t2 : Triple) : Prop :=
  ∀ m : ℕ, m ≤ 3 → t1.a^m + t1.b^m + t1.c^m = t2.a^m + t2.b^m + t2.c^m

/-- Represents the 6-member group formed from two triples -/
def sixMemberGroup (t1 t2 : Triple) (k : ℤ) : Finset ℤ :=
  {t1.a, t1.b, t1.c, t2.a + k, t2.b + k, t2.c + k}

/-- Checks if a 6-member group can be simplified to a 4-member group -/
def simplifiesToFour (t1 t2 : Triple) (k : ℤ) : Prop :=
  (sixMemberGroup t1 t2 k).card = 4

/-- Checks if a 6-member group can be simplified to a 5-member group but not further -/
def simplifiesToFiveOnly (t1 t2 : Triple) (k : ℤ) : Prop :=
  (sixMemberGroup t1 t2 k).card = 5

/-- The main theorem to be proved -/
theorem existence_of_k_values 
  (I II III IV : Triple)
  (h1 : sumPowersEqual I II)
  (h2 : sumPowersEqual III IV) :
  ∃ k : ℤ, 
    (simplifiesToFour I II k ∨ simplifiesToFour II I k) ∧
    (simplifiesToFiveOnly III IV k ∨ simplifiesToFiveOnly IV III k) :=
sorry

end NUMINAMATH_CALUDE_existence_of_k_values_l4072_407241


namespace NUMINAMATH_CALUDE_square_of_negative_x_plus_one_l4072_407255

theorem square_of_negative_x_plus_one (x : ℝ) : (-x - 1)^2 = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_x_plus_one_l4072_407255


namespace NUMINAMATH_CALUDE_clara_score_remainder_l4072_407282

theorem clara_score_remainder (a b c : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) →  -- 'a' represents the tens digit
  (0 ≤ b ∧ b ≤ 9) →  -- 'b' represents the ones digit
  (0 ≤ c ∧ c ≤ 9) →  -- 'c' represents the appended digit
  ∃ r : ℕ, r < 10 ∧ ((100 * a + 10 * b + c) - (10 * a + b)) % 9 = r :=
by sorry

end NUMINAMATH_CALUDE_clara_score_remainder_l4072_407282


namespace NUMINAMATH_CALUDE_gym_charges_twice_a_month_l4072_407295

/-- Represents a gym's monthly charging system -/
structure Gym where
  members : ℕ
  charge_per_payment : ℕ
  monthly_income : ℕ

/-- Calculates the number of times a gym charges its members per month -/
def charges_per_month (g : Gym) : ℕ :=
  g.monthly_income / (g.members * g.charge_per_payment)

/-- Theorem stating that for the given gym conditions, the number of charges per month is 2 -/
theorem gym_charges_twice_a_month :
  let g : Gym := { members := 300, charge_per_payment := 18, monthly_income := 10800 }
  charges_per_month g = 2 := by
  sorry

end NUMINAMATH_CALUDE_gym_charges_twice_a_month_l4072_407295


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l4072_407272

-- Define the parabola C: y^2 = 2px
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define point A
def point_A : ℝ × ℝ := (2, -4)

-- Define point B
def point_B : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem parabola_and_line_properties
  (p : ℝ)
  (h_p_pos : p > 0)
  (h_A_on_C : parabola p point_A.1 point_A.2) :
  -- Part 1: Equation of parabola and its directrix
  (∃ (x y : ℝ), parabola 4 x y ∧ y^2 = 8*x) ∧
  (∃ (x : ℝ), x = -2) ∧
  -- Part 2: Equations of line l
  (∃ (x y : ℝ),
    (x = 0 ∨ y = 2 ∨ x - y + 2 = 0) ∧
    (x = point_B.1 ∧ y = point_B.2) ∧
    (∃! (z : ℝ), parabola 4 x z ∧ z = y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l4072_407272


namespace NUMINAMATH_CALUDE_intersection_parallel_to_l_l4072_407209

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the relation for a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the relation for planes intersecting
variable (planes_intersect : Plane → Plane → Prop)

-- Define the relation for the line of intersection between two planes
variable (intersection_line : Plane → Plane → Line)

-- Define the parallel relation between lines
variable (parallel_line : Line → Line → Prop)

-- Define the skew relation between lines
variable (skew_lines : Line → Line → Prop)

theorem intersection_parallel_to_l 
  (m n l : Line) (α β : Plane) 
  (h1 : skew_lines m n)
  (h2 : perp_line_plane m α)
  (h3 : perp_line_plane n β)
  (h4 : perp_line l m)
  (h5 : perp_line l n)
  (h6 : ¬ line_in_plane l α)
  (h7 : ¬ line_in_plane l β) :
  planes_intersect α β ∧ parallel_line (intersection_line α β) l :=
sorry

end NUMINAMATH_CALUDE_intersection_parallel_to_l_l4072_407209


namespace NUMINAMATH_CALUDE_muffin_division_l4072_407256

theorem muffin_division (total_muffins : ℕ) (total_people : ℕ) (muffins_per_person : ℕ) : 
  total_muffins = 20 →
  total_people = 5 →
  total_muffins = total_people * muffins_per_person →
  muffins_per_person = 4 :=
by sorry

end NUMINAMATH_CALUDE_muffin_division_l4072_407256


namespace NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l4072_407267

/-- Represents the number of slices in the cake -/
def cake_slices : ℕ := 8

/-- Represents the number of calories in each cake slice -/
def calories_per_cake_slice : ℕ := 347

/-- Represents the number of brownies in a pan -/
def brownies_count : ℕ := 6

/-- Represents the number of calories in each brownie -/
def calories_per_brownie : ℕ := 375

/-- Theorem stating the difference in total calories between the cake and the brownies -/
theorem cake_brownie_calorie_difference :
  cake_slices * calories_per_cake_slice - brownies_count * calories_per_brownie = 526 := by
  sorry


end NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l4072_407267


namespace NUMINAMATH_CALUDE_smallest_exponent_divisibility_l4072_407258

theorem smallest_exponent_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3) :
  (∀ n : ℕ, n < 13 → ¬(x * y * z ∣ (x + y + z)^n)) ∧
  (x * y * z ∣ (x + y + z)^13) := by
sorry

end NUMINAMATH_CALUDE_smallest_exponent_divisibility_l4072_407258


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2926_l4072_407266

theorem smallest_prime_factor_of_2926 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2926 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2926 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2926_l4072_407266


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_unique_k_value_l4072_407291

theorem triangle_inequality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

theorem unique_k_value :
  ∀ k : ℕ, k > 0 →
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ b + c > a ∧ c + a > b) →
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_condition_unique_k_value_l4072_407291


namespace NUMINAMATH_CALUDE_stevens_peaches_l4072_407269

/-- Given that Jake has 7 peaches and 12 fewer peaches than Steven, prove that Steven has 19 peaches. -/
theorem stevens_peaches (jake_peaches : ℕ) (steven_jake_diff : ℕ) 
  (h1 : jake_peaches = 7)
  (h2 : steven_jake_diff = 12) :
  jake_peaches + steven_jake_diff = 19 := by
sorry

end NUMINAMATH_CALUDE_stevens_peaches_l4072_407269


namespace NUMINAMATH_CALUDE_base_comparison_l4072_407268

/-- Converts a number from given base to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- The theorem to be proved --/
theorem base_comparison : 
  let base_6_num := to_decimal [5, 4] 6
  let base_4_num := to_decimal [2, 3] 4
  let base_5_num := to_decimal [3, 2, 1] 5
  base_6_num + base_4_num > base_5_num := by
sorry

end NUMINAMATH_CALUDE_base_comparison_l4072_407268


namespace NUMINAMATH_CALUDE_smallest_number_in_sample_l4072_407247

/-- Systematic sampling function that returns the smallest number in the sample -/
def systematicSample (totalProducts : ℕ) (sampleSize : ℕ) (containsProduct : ℕ) : ℕ :=
  containsProduct % (totalProducts / sampleSize)

/-- Theorem: The smallest number in the systematic sample is 10 -/
theorem smallest_number_in_sample :
  systematicSample 80 5 42 = 10 := by
  sorry

#eval systematicSample 80 5 42

end NUMINAMATH_CALUDE_smallest_number_in_sample_l4072_407247


namespace NUMINAMATH_CALUDE_equal_circles_in_quadrilateral_l4072_407245

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a convex quadrilateral with four circles inside -/
structure ConvexQuadrilateral where
  circle_a : Circle
  circle_b : Circle
  circle_c : Circle
  circle_d : Circle
  is_convex : Bool
  circles_touch_sides : Bool
  circles_touch_each_other : Bool
  has_inscribed_circle : Bool

/-- 
Given a convex quadrilateral with four circles inside, each touching two adjacent sides 
and two other circles externally, and given that a circle can be inscribed in the quadrilateral, 
at least two of the four circles have equal radii.
-/
theorem equal_circles_in_quadrilateral (q : ConvexQuadrilateral) 
  (h1 : q.is_convex = true) 
  (h2 : q.circles_touch_sides = true)
  (h3 : q.circles_touch_each_other = true)
  (h4 : q.has_inscribed_circle = true) : 
  (q.circle_a.radius = q.circle_b.radius) ∨ 
  (q.circle_a.radius = q.circle_c.radius) ∨ 
  (q.circle_a.radius = q.circle_d.radius) ∨ 
  (q.circle_b.radius = q.circle_c.radius) ∨ 
  (q.circle_b.radius = q.circle_d.radius) ∨ 
  (q.circle_c.radius = q.circle_d.radius) :=
by
  sorry

end NUMINAMATH_CALUDE_equal_circles_in_quadrilateral_l4072_407245


namespace NUMINAMATH_CALUDE_square_plus_product_equals_square_l4072_407280

theorem square_plus_product_equals_square (x y : ℤ) :
  x^2 + x*y = y^2 ↔ x = 0 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_product_equals_square_l4072_407280


namespace NUMINAMATH_CALUDE_harry_iguanas_l4072_407235

/-- The number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- The number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- The number of snakes Harry owns -/
def num_snakes : ℕ := 4

/-- The cost to feed each snake per month -/
def snake_feed_cost : ℕ := 10

/-- The cost to feed each iguana per month -/
def iguana_feed_cost : ℕ := 5

/-- The cost to feed each gecko per month -/
def gecko_feed_cost : ℕ := 15

/-- The total yearly cost to feed all pets -/
def yearly_feed_cost : ℕ := 1140

theorem harry_iguanas :
  num_iguanas * iguana_feed_cost * 12 +
  num_geckos * gecko_feed_cost * 12 +
  num_snakes * snake_feed_cost * 12 = yearly_feed_cost :=
by sorry

end NUMINAMATH_CALUDE_harry_iguanas_l4072_407235


namespace NUMINAMATH_CALUDE_triangle_reconstruction_theorem_l4072_407244

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define the given points
variable (D E F : Point)

-- Define the properties of the given points
def is_altitude_median_intersection (D : Point) (T : Triangle) : Prop := sorry

def is_altitude_bisector_intersection (E : Point) (T : Triangle) : Prop := sorry

def is_median_bisector_intersection (F : Point) (T : Triangle) : Prop := sorry

-- State the theorem
theorem triangle_reconstruction_theorem 
  (hD : ∃ T : Triangle, is_altitude_median_intersection D T)
  (hE : ∃ T : Triangle, is_altitude_bisector_intersection E T)
  (hF : ∃ T : Triangle, is_median_bisector_intersection F T) :
  ∃! T : Triangle, 
    is_altitude_median_intersection D T ∧ 
    is_altitude_bisector_intersection E T ∧ 
    is_median_bisector_intersection F T :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_theorem_l4072_407244


namespace NUMINAMATH_CALUDE_two_distinct_negative_roots_l4072_407225

/-- The polynomial function for which we're finding roots -/
def f (p : ℝ) (x : ℝ) : ℝ := x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1

/-- A root of the polynomial is a real number x such that f p x = 0 -/
def is_root (p : ℝ) (x : ℝ) : Prop := f p x = 0

/-- A function to represent that a real number is negative -/
def is_negative (x : ℝ) : Prop := x < 0

/-- The main theorem stating that for p > 1, there are at least two distinct negative real roots -/
theorem two_distinct_negative_roots (p : ℝ) (hp : p > 1) :
  ∃ (x y : ℝ), x ≠ y ∧ is_negative x ∧ is_negative y ∧ is_root p x ∧ is_root p y :=
sorry

end NUMINAMATH_CALUDE_two_distinct_negative_roots_l4072_407225


namespace NUMINAMATH_CALUDE_work_completion_time_l4072_407205

/-- The time taken for all three workers (p, q, and r) to complete the work together -/
theorem work_completion_time 
  (efficiency_p : ℝ) 
  (efficiency_q : ℝ) 
  (efficiency_r : ℝ) 
  (time_p : ℝ) 
  (h1 : efficiency_p = 1.3 * efficiency_q) 
  (h2 : time_p = 23) 
  (h3 : efficiency_r = 1.5 * (efficiency_p + efficiency_q)) : 
  (time_p * efficiency_p) / (efficiency_p + efficiency_q + efficiency_r) = 5.2 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l4072_407205


namespace NUMINAMATH_CALUDE_women_science_majors_percentage_l4072_407219

theorem women_science_majors_percentage
  (non_science_percentage : Real)
  (men_percentage : Real)
  (men_science_percentage : Real)
  (h1 : non_science_percentage = 0.6)
  (h2 : men_percentage = 0.4)
  (h3 : men_science_percentage = 0.5500000000000001) :
  let women_percentage := 1 - men_percentage
  let total_science_percentage := 1 - non_science_percentage
  let men_science_total_percentage := men_percentage * men_science_percentage
  let women_science_total_percentage := total_science_percentage - men_science_total_percentage
  women_science_total_percentage / women_percentage = 0.29999999999999993 :=
by sorry

end NUMINAMATH_CALUDE_women_science_majors_percentage_l4072_407219


namespace NUMINAMATH_CALUDE_blue_ball_weight_is_6_l4072_407248

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 9.12 - 3.12

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := 3.12

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := 9.12

theorem blue_ball_weight_is_6 : blue_ball_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_weight_is_6_l4072_407248


namespace NUMINAMATH_CALUDE_dihedral_angle_of_inscribed_spheres_l4072_407264

theorem dihedral_angle_of_inscribed_spheres (r R : ℝ) (θ : ℝ) : 
  r > 0 → 
  R = 3 * r → 
  (R + r) * Real.cos θ = (R + r) * (1/2) → 
  Real.cos (θ) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_dihedral_angle_of_inscribed_spheres_l4072_407264


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l4072_407276

/-- A coloring function that satisfies the given conditions -/
def valid_coloring (n : ℕ) (S : Finset ℕ) (f : Finset ℕ → Fin 8) : Prop :=
  (S.card = 3 * n) ∧
  ∀ A B C : Finset ℕ,
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S →
    A.card = n ∧ B.card = n ∧ C.card = n →
    A ≠ B ∧ A ≠ C ∧ B ≠ C →
    (A ∩ B).card ≤ 1 ∧ (A ∩ C).card ≤ 1 ∧ (B ∩ C).card ≤ 1 →
    f A ≠ f B ∨ f A ≠ f C ∨ f B ≠ f C

/-- There exists a valid coloring for any set S with 3n elements -/
theorem exists_valid_coloring (n : ℕ) :
  ∀ S : Finset ℕ, S.card = 3 * n → ∃ f : Finset ℕ → Fin 8, valid_coloring n S f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l4072_407276


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l4072_407243

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- Define the non-overlapping property for planes
variable (non_overlapping : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) 
  (h3 : non_overlapping α β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l4072_407243


namespace NUMINAMATH_CALUDE_davids_remaining_money_l4072_407251

/-- The amount of money David has left after mowing lawns, buying shoes, and giving money to his mom. -/
def davidsRemainingMoney (hourlyRate : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℕ) : ℚ :=
  let totalEarned := hourlyRate * hoursPerDay * daysPerWeek
  let afterShoes := totalEarned / 2
  afterShoes / 2

theorem davids_remaining_money :
  davidsRemainingMoney 14 2 7 = 49 := by
  sorry

#eval davidsRemainingMoney 14 2 7

end NUMINAMATH_CALUDE_davids_remaining_money_l4072_407251


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l4072_407254

theorem zoo_animal_ratio :
  ∀ (birds non_birds : ℕ),
    birds = 450 →
    birds = non_birds + 360 →
    (birds : ℚ) / non_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l4072_407254


namespace NUMINAMATH_CALUDE_cosine_equality_l4072_407212

theorem cosine_equality (x y : ℝ) : 
  x = 2 * Real.cos (2 * Real.pi / 5) →
  y = 2 * Real.cos (4 * Real.pi / 5) →
  x + y + 1 = 0 →
  x = (-1 + Real.sqrt 5) / 2 ∧ y = (-1 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l4072_407212


namespace NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_eq_7_l4072_407227

theorem log_sqrt12_1728sqrt12_eq_7 : Real.log (1728 * Real.sqrt 12) / Real.log (Real.sqrt 12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_eq_7_l4072_407227


namespace NUMINAMATH_CALUDE_solution_to_equation_l4072_407226

theorem solution_to_equation (x : ℝ) : -200 * x = 1600 → x = -8 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l4072_407226


namespace NUMINAMATH_CALUDE_right_triangle_sets_l4072_407240

/-- Checks if three side lengths can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

theorem right_triangle_sets :
  ¬(is_right_triangle 5 7 10) ∧
  (is_right_triangle 3 4 5) ∧
  ¬(is_right_triangle 1 3 2) ∧
  (is_right_triangle 7 24 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l4072_407240


namespace NUMINAMATH_CALUDE_jack_and_jill_games_l4072_407287

/-- A game between Jack and Jill -/
structure Game where
  winner : Bool  -- true if Jack wins, false if Jill wins

/-- The score of a player in a single game -/
def score (g : Game) (isJack : Bool) : Nat :=
  if g.winner == isJack then 2 else 1

/-- The total score of a player across multiple games -/
def totalScore (games : List Game) (isJack : Bool) : Nat :=
  (games.map (fun g => score g isJack)).sum

theorem jack_and_jill_games 
  (games : List Game) 
  (h1 : games.length > 0)
  (h2 : (games.filter (fun g => g.winner)).length = 4)  -- Jack won 4 games
  (h3 : totalScore games false = 10)  -- Jill's final score is 10
  : games.length = 7 := by
  sorry


end NUMINAMATH_CALUDE_jack_and_jill_games_l4072_407287


namespace NUMINAMATH_CALUDE_second_class_size_l4072_407286

theorem second_class_size (students1 : ℕ) (avg1 : ℕ) (avg2 : ℕ) (avg_total : ℕ) :
  students1 = 12 →
  avg1 = 40 →
  avg2 = 60 →
  avg_total = 54 →
  ∃ students2 : ℕ, 
    students2 = 28 ∧
    (students1 * avg1 + students2 * avg2) = (students1 + students2) * avg_total :=
by sorry


end NUMINAMATH_CALUDE_second_class_size_l4072_407286


namespace NUMINAMATH_CALUDE_sweeties_remainder_l4072_407283

theorem sweeties_remainder (m : ℕ) (h1 : m > 0) (h2 : m % 7 = 6) : (4 * m) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sweeties_remainder_l4072_407283


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l4072_407230

theorem company_picnic_attendance 
  (total_employees : ℕ) 
  (men_percentage : ℝ) 
  (women_percentage : ℝ) 
  (men_attendance : ℝ) 
  (women_attendance : ℝ) 
  (h1 : men_percentage = 0.35) 
  (h2 : women_percentage = 1 - men_percentage) 
  (h3 : men_attendance = 0.20) 
  (h4 : women_attendance = 0.40) : 
  (men_percentage * men_attendance + women_percentage * women_attendance) * 100 = 33 :=
sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l4072_407230


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l4072_407201

/-- Proves that the price of each apple is $0.90 given the conditions of the fruit stand problem -/
theorem fruit_stand_problem (total_cost : ℝ) (total_fruits : ℕ) (banana_price : ℝ)
  (h_total_cost : total_cost = 6.50)
  (h_total_fruits : total_fruits = 9)
  (h_banana_price : banana_price = 0.70) :
  ∃ (apple_price : ℝ) (num_apples : ℕ),
    apple_price = 0.90 ∧
    num_apples + (total_fruits - num_apples) = total_fruits ∧
    apple_price * num_apples + banana_price * (total_fruits - num_apples) = total_cost :=
by
  sorry

#check fruit_stand_problem

end NUMINAMATH_CALUDE_fruit_stand_problem_l4072_407201


namespace NUMINAMATH_CALUDE_sum_of_altitudes_triangle_l4072_407208

/-- The sum of altitudes of a triangle formed by the line 8x + 3y = 48 and the coordinate axes -/
theorem sum_of_altitudes_triangle (x y : ℝ) (h : 8 * x + 3 * y = 48) :
  let x_intercept : ℝ := 48 / 8
  let y_intercept : ℝ := 48 / 3
  let hypotenuse : ℝ := Real.sqrt (x_intercept^2 + y_intercept^2)
  let altitude_to_hypotenuse : ℝ := 96 / hypotenuse
  x_intercept + y_intercept + altitude_to_hypotenuse = (22 * Real.sqrt 292 + 96) / Real.sqrt 292 := by
sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_triangle_l4072_407208


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4072_407207

theorem necessary_but_not_sufficient :
  let p : ℝ → Prop := λ x ↦ |x + 1| > 2
  let q : ℝ → Prop := λ x ↦ x > 2
  (∀ x, ¬(q x) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4072_407207


namespace NUMINAMATH_CALUDE_tire_repair_cost_l4072_407257

/-- Calculates the final cost of tire repairs -/
def final_cost (repair_cost : ℚ) (sales_tax : ℚ) (num_tires : ℕ) : ℚ :=
  (repair_cost + sales_tax) * num_tires

/-- Theorem: The final cost for repairing 4 tires is $30 -/
theorem tire_repair_cost : final_cost 7 0.5 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tire_repair_cost_l4072_407257


namespace NUMINAMATH_CALUDE_det_eq_ten_l4072_407275

/-- The matrix for which we need to calculate the determinant -/
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 3, 2*x]

/-- The theorem stating the condition for the determinant to be 10 -/
theorem det_eq_ten (x : ℝ) : 
  Matrix.det (A x) = 10 ↔ x = Real.sqrt (8/3) ∨ x = -Real.sqrt (8/3) := by
  sorry

end NUMINAMATH_CALUDE_det_eq_ten_l4072_407275


namespace NUMINAMATH_CALUDE_f_properties_l4072_407259

/-- The function f(x) = mx^2 + (1-3m)x - 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (1 - 3*m) * x - 4

theorem f_properties :
  -- Part I
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 1 x ≤ 4 ∧ f 1 x ≥ -5) ∧
  (∃ x₁ ∈ Set.Icc (-2 : ℝ) 2, f 1 x₁ = 4) ∧
  (∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, f 1 x₂ = -5) ∧

  -- Part II (simplified representation of the solution sets)
  (∀ m : ℝ, ∃ S : Set ℝ, ∀ x : ℝ, f m x > -1 ↔ x ∈ S) ∧

  -- Part III
  (∀ m < 0, (∃ x₀ > 1, f m x₀ > 0) → m < -1 ∨ (-1/9 < m ∧ m < 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4072_407259


namespace NUMINAMATH_CALUDE_sally_box_sales_l4072_407237

theorem sally_box_sales (saturday_sales : ℕ) : 
  (saturday_sales + (3 / 2 : ℚ) * saturday_sales = 150) → 
  saturday_sales = 60 := by
sorry

end NUMINAMATH_CALUDE_sally_box_sales_l4072_407237


namespace NUMINAMATH_CALUDE_intersection_line_equation_l4072_407233

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    A ≠ B →
    ∀ (P : ℝ × ℝ),
      (∃ t : ℝ, P = t • A + (1 - t) • B) ↔ line P.1 P.2 :=
by sorry


end NUMINAMATH_CALUDE_intersection_line_equation_l4072_407233


namespace NUMINAMATH_CALUDE_river_crossing_l4072_407250

/-- Represents a river with islands -/
structure River :=
  (width : ℝ)
  (islandsPerimeter : ℝ)

/-- Theorem stating that it's possible to cross the river in less than 3 meters -/
theorem river_crossing (r : River) 
  (h_width : r.width = 1)
  (h_perimeter : r.islandsPerimeter = 8) : 
  ∃ (path : ℝ), path < 3 ∧ path ≥ r.width :=
sorry

end NUMINAMATH_CALUDE_river_crossing_l4072_407250


namespace NUMINAMATH_CALUDE_two_valid_numbers_l4072_407222

def digits (n : ℕ) : Finset ℕ :=
  (n.digits 10).toFinset

def valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (digits n ∪ digits (n * n) = Finset.range 9)

theorem two_valid_numbers :
  {n : ℕ | valid_number n} = {567, 854} := by sorry

end NUMINAMATH_CALUDE_two_valid_numbers_l4072_407222


namespace NUMINAMATH_CALUDE_expression_equals_one_l4072_407284

theorem expression_equals_one (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) : 
  (a^2 * b^2) / ((a^2 - b*c) * (b^2 - a*c)) + 
  (a^2 * c^2) / ((a^2 - b*c) * (c^2 - a*b)) + 
  (b^2 * c^2) / ((b^2 - a*c) * (c^2 - a*b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l4072_407284


namespace NUMINAMATH_CALUDE_dog_weight_fraction_is_one_fourth_l4072_407214

/-- Represents the capacity and weight scenario of Penny's canoe --/
structure CanoeScenario where
  capacity : ℕ              -- Capacity without dog
  capacityWithDog : ℚ       -- Fraction of capacity with dog
  personWeight : ℕ          -- Weight of each person in pounds
  totalWeightWithDog : ℕ    -- Total weight in canoe with dog and people

/-- Calculates the dog's weight as a fraction of a person's weight --/
def dogWeightFraction (scenario : CanoeScenario) : ℚ :=
  let peopleWithDog := ⌊scenario.capacity * scenario.capacityWithDog⌋
  let peopleWeight := peopleWithDog * scenario.personWeight
  let dogWeight := scenario.totalWeightWithDog - peopleWeight
  dogWeight / scenario.personWeight

/-- Theorem stating that the dog's weight is 1/4 of a person's weight --/
theorem dog_weight_fraction_is_one_fourth (scenario : CanoeScenario) 
  (h1 : scenario.capacity = 6)
  (h2 : scenario.capacityWithDog = 2/3)
  (h3 : scenario.personWeight = 140)
  (h4 : scenario.totalWeightWithDog = 595) :
  dogWeightFraction scenario = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_dog_weight_fraction_is_one_fourth_l4072_407214


namespace NUMINAMATH_CALUDE_jesus_squares_count_l4072_407246

/-- The number of squares Pedro has -/
def pedro_squares : ℕ := 200

/-- The number of squares Linden has -/
def linden_squares : ℕ := 75

/-- The number of extra squares Pedro has compared to Jesus and Linden combined -/
def extra_squares : ℕ := 65

/-- The number of squares Jesus has -/
def jesus_squares : ℕ := pedro_squares - linden_squares - extra_squares

theorem jesus_squares_count : jesus_squares = 60 := by sorry

end NUMINAMATH_CALUDE_jesus_squares_count_l4072_407246


namespace NUMINAMATH_CALUDE_journey_time_equation_l4072_407218

theorem journey_time_equation (x : ℝ) (h1 : x > 0) : 
  (240 / x - 240 / (1.5 * x) = 1) ↔ 
  (240 / x = 240 / (1.5 * x) + 1) := by
sorry

end NUMINAMATH_CALUDE_journey_time_equation_l4072_407218


namespace NUMINAMATH_CALUDE_rectangle_perimeter_after_increase_l4072_407200

/-- Given a rectangle with width 10 meters and original area 150 square meters,
    if its length is increased such that the new area is 4/3 times the original area,
    then the new perimeter is 60 meters. -/
theorem rectangle_perimeter_after_increase (original_length : ℝ) (new_length : ℝ) : 
  original_length * 10 = 150 →
  new_length * 10 = 150 * (4/3) →
  2 * (new_length + 10) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_after_increase_l4072_407200


namespace NUMINAMATH_CALUDE_convex_polygon_properties_l4072_407213

/-- A convex n-gon -/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here
  n_ge_3 : n ≥ 3

/-- The sum of interior angles of a convex n-gon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The number of triangles formed by non-intersecting diagonals in a convex n-gon -/
def num_triangles (n : ℕ) : ℕ := n - 2

theorem convex_polygon_properties {n : ℕ} (p : ConvexPolygon n) :
  (sum_interior_angles n = (n - 2) * 180) ∧
  (num_triangles n = n - 2) :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_properties_l4072_407213


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l4072_407278

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem a_equals_one_sufficient_not_necessary (a : ℝ) :
  (a = 1 → is_purely_imaginary ((a - 1) * (a + 2) + (a + 3) * I)) ∧
  ¬(is_purely_imaginary ((a - 1) * (a + 2) + (a + 3) * I) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l4072_407278


namespace NUMINAMATH_CALUDE_grid_filling_ways_l4072_407273

/-- Represents a 6x6 grid with special cells -/
structure Grid :=
  (size : Nat)
  (specialCells : Nat)
  (valuesPerSpecialCell : Nat)

/-- Calculates the number of ways to fill the grid -/
def numberOfWays (g : Grid) : Nat :=
  (g.valuesPerSpecialCell ^ g.specialCells) ^ 4

/-- Theorem: The number of ways to fill the grid is 16 -/
theorem grid_filling_ways (g : Grid) 
  (h1 : g.size = 6)
  (h2 : g.specialCells = 4)
  (h3 : g.valuesPerSpecialCell = 2) :
  numberOfWays g = 16 := by
  sorry

#eval numberOfWays { size := 6, specialCells := 4, valuesPerSpecialCell := 2 }

end NUMINAMATH_CALUDE_grid_filling_ways_l4072_407273


namespace NUMINAMATH_CALUDE_logical_consequences_l4072_407292

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (passed : Student → Prop)
variable (scored_above_90_percent : Student → Prop)

-- Define the given condition
variable (h : ∀ s : Student, scored_above_90_percent s → passed s)

-- Theorem to prove
theorem logical_consequences :
  (∀ s : Student, ¬(passed s) → ¬(scored_above_90_percent s)) ∧
  (∀ s : Student, ¬(scored_above_90_percent s) → ¬(passed s)) ∧
  (∀ s : Student, passed s → scored_above_90_percent s) :=
by sorry

end NUMINAMATH_CALUDE_logical_consequences_l4072_407292


namespace NUMINAMATH_CALUDE_problem_solution_l4072_407277

theorem problem_solution (x y : ℝ) 
  (h1 : x * y + x + y = 17) 
  (h2 : x^2 * y + x * y^2 = 66) : 
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4072_407277


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l4072_407270

theorem no_solution_to_inequality_system : 
  ¬ ∃ x : ℝ, (x - 3 ≥ 0) ∧ (2*x - 5 < 1) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l4072_407270


namespace NUMINAMATH_CALUDE_ivans_initial_money_l4072_407203

theorem ivans_initial_money (initial_money : ℝ) : 
  (4/5 * initial_money - 5 = 3) → initial_money = 10 := by
  sorry

end NUMINAMATH_CALUDE_ivans_initial_money_l4072_407203


namespace NUMINAMATH_CALUDE_society_coleaders_selection_l4072_407242

theorem society_coleaders_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 2) :
  Nat.choose n k = 190 := by
  sorry

end NUMINAMATH_CALUDE_society_coleaders_selection_l4072_407242


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l4072_407238

theorem circle_ratio_after_increase (r : ℝ) (h : r > 0) : 
  (2 * π * (r + 2)) / (2 * (r + 2)) = π := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l4072_407238


namespace NUMINAMATH_CALUDE_point_position_l4072_407279

theorem point_position (a : ℝ) : 
  (a < 0) → -- A is on the negative side of the origin
  (2 > 0) → -- B is on the positive side of the origin
  (|a + 3| = 4) → -- CO = 2BO, where BO = 2
  a = -7 := by
sorry

end NUMINAMATH_CALUDE_point_position_l4072_407279


namespace NUMINAMATH_CALUDE_E_parity_l4072_407262

def E : ℕ → ℤ
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | n + 3 => E (n + 2) + 2 * E (n + 1) - E n

theorem E_parity : (E 10 % 2 = 1) ∧ (E 11 % 2 = 0) ∧ (E 12 % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_E_parity_l4072_407262


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l4072_407253

theorem min_distance_to_origin (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 4 = 0) :
  ∃ (min_dist : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 6*b + 4 = 0 → 
    min_dist ≤ Real.sqrt (a^2 + b^2)) ∧ min_dist = Real.sqrt 13 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l4072_407253


namespace NUMINAMATH_CALUDE_fraction_spent_l4072_407296

def borrowed_brother : ℕ := 20
def borrowed_father : ℕ := 40
def borrowed_mother : ℕ := 30
def gift_grandmother : ℕ := 70
def savings : ℕ := 100
def remaining : ℕ := 65

def total_amount : ℕ := borrowed_brother + borrowed_father + borrowed_mother + gift_grandmother + savings

theorem fraction_spent (h : total_amount - remaining = 195) :
  (total_amount - remaining : ℚ) / total_amount = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_spent_l4072_407296


namespace NUMINAMATH_CALUDE_jared_earnings_proof_l4072_407249

/-- The monthly salary of a diploma holder in dollars -/
def diploma_salary : ℕ := 4000

/-- The ratio of a degree holder's salary to a diploma holder's salary -/
def degree_to_diploma_ratio : ℕ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Jared's annual earnings after graduating with a degree -/
def jared_annual_earnings : ℕ := degree_to_diploma_ratio * diploma_salary * months_in_year

theorem jared_earnings_proof :
  jared_annual_earnings = 144000 :=
sorry

end NUMINAMATH_CALUDE_jared_earnings_proof_l4072_407249


namespace NUMINAMATH_CALUDE_scale_division_l4072_407217

/-- Given a scale of length 80 inches divided into equal parts of 20 inches each,
    prove that the number of equal parts is 4. -/
theorem scale_division (scale_length : ℕ) (part_length : ℕ) (h1 : scale_length = 80) (h2 : part_length = 20) :
  scale_length / part_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l4072_407217


namespace NUMINAMATH_CALUDE_fraction_calls_team_B_value_l4072_407252

/-- Represents the fraction of calls processed by team B in a call center scenario -/
def fraction_calls_team_B (num_agents_A num_agents_B : ℚ) 
  (calls_per_agent_A calls_per_agent_B : ℚ) : ℚ :=
  (num_agents_B * calls_per_agent_B) / 
  (num_agents_A * calls_per_agent_A + num_agents_B * calls_per_agent_B)

/-- Theorem stating the fraction of calls processed by team B -/
theorem fraction_calls_team_B_value 
  (num_agents_A num_agents_B : ℚ) 
  (calls_per_agent_A calls_per_agent_B : ℚ) 
  (h1 : num_agents_A = (5 / 8) * num_agents_B)
  (h2 : calls_per_agent_A = (6 / 5) * calls_per_agent_B) :
  fraction_calls_team_B num_agents_A num_agents_B calls_per_agent_A calls_per_agent_B = 4 / 7 := by
  sorry


end NUMINAMATH_CALUDE_fraction_calls_team_B_value_l4072_407252


namespace NUMINAMATH_CALUDE_jerry_bacon_calories_l4072_407293

/-- Represents Jerry's breakfast -/
structure Breakfast where
  pancakes : ℕ
  pancake_calories : ℕ
  bacon_strips : ℕ
  cereal_calories : ℕ
  total_calories : ℕ

/-- Calculates the calories per strip of bacon -/
def bacon_calories_per_strip (b : Breakfast) : ℕ :=
  (b.total_calories - (b.pancakes * b.pancake_calories + b.cereal_calories)) / b.bacon_strips

/-- Theorem stating that each strip of bacon in Jerry's breakfast has 100 calories -/
theorem jerry_bacon_calories :
  let jerry_breakfast : Breakfast := {
    pancakes := 6,
    pancake_calories := 120,
    bacon_strips := 2,
    cereal_calories := 200,
    total_calories := 1120
  }
  bacon_calories_per_strip jerry_breakfast = 100 := by
  sorry

end NUMINAMATH_CALUDE_jerry_bacon_calories_l4072_407293


namespace NUMINAMATH_CALUDE_remaining_artifacts_correct_l4072_407206

structure MarineArtifacts where
  clam_shells : ℕ
  conch_shells : ℕ
  oyster_shells : ℕ
  coral_pieces : ℕ
  sea_glass_shards : ℕ
  starfish : ℕ

def initial_artifacts : MarineArtifacts :=
  { clam_shells := 325
  , conch_shells := 210
  , oyster_shells := 144
  , coral_pieces := 96
  , sea_glass_shards := 180
  , starfish := 110 }

def given_away (a : MarineArtifacts) : MarineArtifacts :=
  { clam_shells := a.clam_shells / 4
  , conch_shells := 50
  , oyster_shells := a.oyster_shells / 3
  , coral_pieces := a.coral_pieces / 2
  , sea_glass_shards := a.sea_glass_shards / 5
  , starfish := 0 }

def remaining_artifacts (a : MarineArtifacts) : MarineArtifacts :=
  { clam_shells := a.clam_shells - (given_away a).clam_shells
  , conch_shells := a.conch_shells - (given_away a).conch_shells
  , oyster_shells := a.oyster_shells - (given_away a).oyster_shells
  , coral_pieces := a.coral_pieces - (given_away a).coral_pieces
  , sea_glass_shards := a.sea_glass_shards - (given_away a).sea_glass_shards
  , starfish := a.starfish - (given_away a).starfish }

theorem remaining_artifacts_correct :
  (remaining_artifacts initial_artifacts) =
    { clam_shells := 244
    , conch_shells := 160
    , oyster_shells := 96
    , coral_pieces := 48
    , sea_glass_shards := 144
    , starfish := 110 } := by
  sorry

end NUMINAMATH_CALUDE_remaining_artifacts_correct_l4072_407206


namespace NUMINAMATH_CALUDE_consecutive_pair_with_17_l4072_407265

theorem consecutive_pair_with_17 (a b : ℤ) : 
  (a = 17 ∨ b = 17) → 
  (abs (a - b) = 1) → 
  (a + b = 35) → 
  (35 % 5 = 0) → 
  ((a = 17 ∧ b = 18) ∨ (a = 18 ∧ b = 17)) := by sorry

end NUMINAMATH_CALUDE_consecutive_pair_with_17_l4072_407265


namespace NUMINAMATH_CALUDE_non_monotonic_derivative_range_l4072_407297

open Real

theorem non_monotonic_derivative_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, deriv f x = exp x + k^2 / exp x - 1 / k) →
  (¬ Monotone f) →
  0 < k ∧ k < sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_derivative_range_l4072_407297


namespace NUMINAMATH_CALUDE_two_complex_roots_iff_m_values_l4072_407223

/-- The equation (x / (x+2)) + (x / (x+3)) = mx has exactly two complex roots
    if and only if m is equal to 0, 2i, or -2i. -/
theorem two_complex_roots_iff_m_values (m : ℂ) : 
  (∃! (r₁ r₂ : ℂ), ∀ (x : ℂ), x ≠ -2 ∧ x ≠ -3 →
    (x / (x + 2) + x / (x + 3) = m * x) ↔ (x = r₁ ∨ x = r₂)) ↔
  (m = 0 ∨ m = 2*I ∨ m = -2*I) :=
sorry

end NUMINAMATH_CALUDE_two_complex_roots_iff_m_values_l4072_407223


namespace NUMINAMATH_CALUDE_intersection_of_lines_l4072_407231

theorem intersection_of_lines (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 ∧ y = k * x + 4 ∧ x = 1 ∧ y = 1) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l4072_407231


namespace NUMINAMATH_CALUDE_D_nec_not_suff_A_l4072_407288

-- Define propositions A, B, C, and D
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom C_nec_and_suff_B : (B ↔ C)
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem D_nec_not_suff_A : (A → D) ∧ ¬(D → A) := by
  sorry

end NUMINAMATH_CALUDE_D_nec_not_suff_A_l4072_407288


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4072_407221

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℤ, (x : ℝ) ≥ 2 ∧ (x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2) ∧ 
   ∀ y : ℤ, y < x → (y : ℝ) - m < 4 ∨ y - 4 > 3 * (y - 2)) →
  -3 < m ∧ m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4072_407221


namespace NUMINAMATH_CALUDE_abc_reciprocal_sum_l4072_407236

theorem abc_reciprocal_sum (a b c : ℝ) 
  (h1 : a + 1/b = 9)
  (h2 : b + 1/c = 10)
  (h3 : c + 1/a = 11) :
  a * b * c + 1 / (a * b * c) = 960 := by
  sorry

end NUMINAMATH_CALUDE_abc_reciprocal_sum_l4072_407236


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l4072_407298

/-- The discriminant of the quadratic equation 2x^2 + (2 + 1/2)x + 1/2 is 9/4 -/
theorem quadratic_discriminant : 
  let a : ℚ := 2
  let b : ℚ := 5/2
  let c : ℚ := 1/2
  let discriminant := b^2 - 4*a*c
  discriminant = 9/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l4072_407298


namespace NUMINAMATH_CALUDE_parabola_chord_constant_sum_l4072_407290

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = x^2 -/
def parabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Point C on the y-axis -/
def C : Point :=
  ⟨0, 2⟩

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- The theorem to be proved -/
theorem parabola_chord_constant_sum :
  ∀ A B : Point,
  parabola A → parabola B →
  (C.y - A.y) / (C.x - A.x) = (B.y - A.y) / (B.x - A.x) →
  (1 / distanceSquared A C + 1 / distanceSquared B C) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_constant_sum_l4072_407290


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l4072_407211

/-- Given a point P₀ and a plane, this theorem states that the line passing through P₀ 
    and perpendicular to the plane has a specific equation. -/
theorem line_perpendicular_to_plane 
  (P₀ : ℝ × ℝ × ℝ) 
  (plane_normal : ℝ × ℝ × ℝ) 
  (plane_constant : ℝ) :
  let (x₀, y₀, z₀) := P₀
  let (a, b, c) := plane_normal
  (P₀ = (3, 4, 2) ∧ 
   plane_normal = (8, -4, 5) ∧ 
   plane_constant = -4) →
  (∀ (x y z : ℝ), 
    ((x - x₀) / a = (y - y₀) / b ∧ (y - y₀) / b = (z - z₀) / c) ↔
    (x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = (x₀ + a*t, y₀ + b*t, z₀ + c*t)}) :=
by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l4072_407211


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l4072_407220

theorem geometric_arithmetic_geometric_sequence 
  (a b c : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric progression condition
  (h2 : b + 2 = (a + c) / 2)  -- arithmetic progression condition
  (h3 : (b + 2) ^ 2 = a * (c + 16))  -- second geometric progression condition
  : (a = 1 ∧ b = 3 ∧ c = 9) ∨ (a = 1/9 ∧ b = -5/9 ∧ c = 25/9) := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l4072_407220


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l4072_407210

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_pattern :
  ∃ (n : ℕ) (a : ℕ),
    n > 0 ∧
    a ≥ 1 ∧ a < 30 ∧
    consecutive_pair a (a + 1) ∧
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 30 ∧ i ≠ a ∧ i ≠ (a + 1) → is_divisible n i) ∧
    ¬(is_divisible n a) ∧
    ¬(is_divisible n (a + 1)) ∧
    (∀ m : ℕ, m < n →
      ¬(∃ (b : ℕ),
        b ≥ 1 ∧ b < 30 ∧
        consecutive_pair b (b + 1) ∧
        (∀ i : ℕ, 1 ≤ i ∧ i ≤ 30 ∧ i ≠ b ∧ i ≠ (b + 1) → is_divisible m i) ∧
        ¬(is_divisible m b) ∧
        ¬(is_divisible m (b + 1)))) ∧
    n = 12252240 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l4072_407210


namespace NUMINAMATH_CALUDE_factory_material_usage_extension_l4072_407281

/-- Given a factory with m tons of raw materials and an original plan to use a tons per day (a > 1),
    prove that if the factory reduces daily usage by 1 ton, it can use the materials for m / (a(a-1))
    additional days compared to the original plan. -/
theorem factory_material_usage_extension (m a : ℝ) (ha : a > 1) :
  let original_days := m / a
  let new_days := m / (a - 1)
  new_days - original_days = m / (a * (a - 1)) := by sorry

end NUMINAMATH_CALUDE_factory_material_usage_extension_l4072_407281


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_l4072_407285

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: A rectangle with length 5 and width 3 has area 15 and perimeter 16 -/
theorem rectangle_area_perimeter :
  let r : Rectangle := ⟨5, 3⟩
  area r = 15 ∧ perimeter r = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_l4072_407285


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l4072_407204

/-- 
Given two cyclists traveling in opposite directions for 2 hours,
with one traveling at 15 km/h and ending up 50 km apart,
prove that the speed of the other cyclist is 10 km/h.
-/
theorem cyclist_speed_problem (time : ℝ) (distance : ℝ) (speed_south : ℝ) (speed_north : ℝ) :
  time = 2 →
  distance = 50 →
  speed_south = 15 →
  (speed_north + speed_south) * time = distance →
  speed_north = 10 := by
sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l4072_407204


namespace NUMINAMATH_CALUDE_absoluteError_2175000_absoluteError_1730000_l4072_407294

/-- Calculates the absolute error of an approximate number -/
def absoluteError (x : ℕ) : ℕ :=
  if x % 10 ≠ 0 then 1
  else if x % 100 ≠ 0 then 10
  else if x % 1000 ≠ 0 then 100
  else if x % 10000 ≠ 0 then 1000
  else 10000

/-- The absolute error of 2175000 is 1 -/
theorem absoluteError_2175000 : absoluteError 2175000 = 1 := by sorry

/-- The absolute error of 1730000 (173 * 10^4) is 10000 -/
theorem absoluteError_1730000 : absoluteError 1730000 = 10000 := by sorry

end NUMINAMATH_CALUDE_absoluteError_2175000_absoluteError_1730000_l4072_407294


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l4072_407263

/-- Represents the number of people in each stratum -/
structure Strata :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)
  (remaining : ℕ)

/-- Represents the sample sizes for each stratum -/
structure Sample :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)
  (remaining : ℕ)

/-- Calculates the total population size -/
def totalPopulation (s : Strata) : ℕ :=
  s.senior + s.intermediate + s.junior + s.remaining

/-- Checks if the sample sizes are proportional to the strata sizes -/
def isProportionalSample (strata : Strata) (sample : Sample) (totalSampleSize : ℕ) : Prop :=
  let total := totalPopulation strata
  sample.senior * total = strata.senior * totalSampleSize ∧
  sample.intermediate * total = strata.intermediate * totalSampleSize ∧
  sample.junior * total = strata.junior * totalSampleSize ∧
  sample.remaining * total = strata.remaining * totalSampleSize

/-- Theorem: The given sample sizes are proportional for the given strata -/
theorem correct_stratified_sample :
  let strata : Strata := ⟨160, 320, 200, 120⟩
  let sample : Sample := ⟨8, 16, 10, 6⟩
  let totalSampleSize : ℕ := 40
  totalPopulation strata = 800 →
  isProportionalSample strata sample totalSampleSize :=
sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l4072_407263


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l4072_407234

theorem lcm_factor_problem (A B : ℕ) (X : ℕ+) :
  A = 400 →
  Nat.gcd A B = 25 →
  Nat.lcm A B = 25 * X * 16 →
  X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l4072_407234


namespace NUMINAMATH_CALUDE_arithmetic_sequence_68th_term_l4072_407261

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℕ
  term_21 : ℕ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℕ :=
  seq.first_term + (n - 1) * ((seq.term_21 - seq.first_term) / 20)

/-- Theorem stating that the 68th term of the given arithmetic sequence is 204 -/
theorem arithmetic_sequence_68th_term
  (seq : ArithmeticSequence)
  (h1 : seq.first_term = 3)
  (h2 : seq.term_21 = 63) :
  nth_term seq 68 = 204 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_68th_term_l4072_407261


namespace NUMINAMATH_CALUDE_total_people_in_tribes_l4072_407274

theorem total_people_in_tribes (cannoneers : ℕ) (women : ℕ) (men : ℕ) : 
  cannoneers = 63 → 
  women = 2 * cannoneers → 
  men = 2 * women → 
  cannoneers + women + men = 378 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_tribes_l4072_407274


namespace NUMINAMATH_CALUDE_escalator_length_l4072_407216

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time : ℝ),
    escalator_speed = 10 →
    person_speed = 4 →
    time = 8 →
    (escalator_speed + person_speed) * time = 112 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_l4072_407216


namespace NUMINAMATH_CALUDE_draw_balls_count_l4072_407271

/-- The number of ways to draw 3 balls in order from a bin of 12 balls, 
    where each ball remains outside the bin after it is drawn. -/
def draw_balls : ℕ :=
  12 * 11 * 10

/-- Theorem stating that the number of ways to draw 3 balls in order 
    from a bin of 12 balls, where each ball remains outside the bin 
    after it is drawn, is equal to 1320. -/
theorem draw_balls_count : draw_balls = 1320 := by
  sorry

end NUMINAMATH_CALUDE_draw_balls_count_l4072_407271


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l4072_407239

/-- Proves the length of a bridge given train specifications and crossing times -/
theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_time = 600 →
  let train_speed := train_length / signal_post_time
  let bridge_length := train_speed * bridge_time - train_length
  bridge_length = 8400 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l4072_407239
