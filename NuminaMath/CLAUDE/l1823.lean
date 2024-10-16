import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l1823_182390

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1823_182390


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_l1823_182369

theorem smallest_solution_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 →
  x ≥ -Real.sqrt 26 ∧
  ∃ y, y^4 - 50*y^2 + 576 = 0 ∧ y = -Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_l1823_182369


namespace NUMINAMATH_CALUDE_round_39_982_to_three_sig_figs_l1823_182385

/-- Rounds a number to a specified number of significant figures -/
def roundToSigFigs (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Checks if a real number has exactly n significant figures -/
def hasSigFigs (x : ℝ) (n : ℕ) : Prop := sorry

theorem round_39_982_to_three_sig_figs :
  let x := 39.982
  let result := roundToSigFigs x 3
  result = 40.0 ∧ hasSigFigs result 3 := by sorry

end NUMINAMATH_CALUDE_round_39_982_to_three_sig_figs_l1823_182385


namespace NUMINAMATH_CALUDE_smallest_sum_after_slice_l1823_182326

-- Define the structure of a die
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

-- Define the cube structure
structure Cube :=
  (dice : Fin 27 → Die)

-- Define the function to calculate the sum of visible faces
def sum_visible_faces (c : Cube) : Nat :=
  -- Implementation details omitted
  sorry

-- Main theorem
theorem smallest_sum_after_slice (c : Cube) : sum_visible_faces c ≥ 98 :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_after_slice_l1823_182326


namespace NUMINAMATH_CALUDE_max_right_angles_is_14_l1823_182310

/-- A triangular prism -/
structure TriangularPrism :=
  (faces : Nat)
  (angles : Nat)
  (h_faces : faces = 5)
  (h_angles : angles = 18)

/-- The maximum number of right angles in a triangular prism -/
def max_right_angles (prism : TriangularPrism) : Nat := 14

/-- Theorem: The maximum number of right angles in a triangular prism is 14 -/
theorem max_right_angles_is_14 (prism : TriangularPrism) :
  max_right_angles prism = 14 := by sorry

end NUMINAMATH_CALUDE_max_right_angles_is_14_l1823_182310


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1823_182315

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1823_182315


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l1823_182366

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 300) (h2 : cat_owners = 30) : 
  (cat_owners : ℚ) / total_students * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l1823_182366


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1823_182357

/-- Represents the base length of an isosceles triangle -/
def BaseLengthIsosceles (area : ℝ) (equalSide : ℝ) : Set ℝ :=
  {x | x > 0 ∧ (x * (equalSide ^ 2 - (x / 2) ^ 2).sqrt / 2 = area)}

/-- Theorem: The base length of an isosceles triangle with area 3 cm² and equal side 25 cm is either 14 cm or 48 cm -/
theorem isosceles_triangle_base_length :
  BaseLengthIsosceles 3 25 = {14, 48} := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1823_182357


namespace NUMINAMATH_CALUDE_ad_length_is_sqrt_397_l1823_182382

/-- A quadrilateral with intersecting diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (bo : dist B O = 5)
  (od : dist O D = 7)
  (ao : dist A O = 9)
  (oc : dist O C = 4)
  (ab : dist A B = 7)

/-- The length of AD in the quadrilateral -/
def ad_length (q : Quadrilateral) : ℝ := dist q.A q.D

/-- Theorem stating that AD length is √397 -/
theorem ad_length_is_sqrt_397 (q : Quadrilateral) : ad_length q = Real.sqrt 397 := by
  sorry


end NUMINAMATH_CALUDE_ad_length_is_sqrt_397_l1823_182382


namespace NUMINAMATH_CALUDE_collinear_vectors_n_equals_one_l1823_182399

def a (n : ℝ) : Fin 2 → ℝ := ![1, n]
def b (n : ℝ) : Fin 2 → ℝ := ![-1, n - 2]

def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem collinear_vectors_n_equals_one :
  ∀ n : ℝ, collinear (a n) (b n) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_n_equals_one_l1823_182399


namespace NUMINAMATH_CALUDE_aisha_driving_problem_l1823_182341

theorem aisha_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (average_speed : ℝ) :
  initial_distance = 18 →
  initial_speed = 36 →
  second_speed = 60 →
  average_speed = 48 →
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = average_speed ∧
    additional_distance = 30 :=
by sorry

end NUMINAMATH_CALUDE_aisha_driving_problem_l1823_182341


namespace NUMINAMATH_CALUDE_integer_solution_system_l1823_182370

theorem integer_solution_system :
  ∀ x y z : ℤ,
  (x * y + y * z + z * x = -4) →
  (x^2 + y^2 + z^2 = 24) →
  (x^3 + y^3 + z^3 + 3*x*y*z = 16) →
  ((x = 2 ∧ y = -2 ∧ z = 4) ∨
   (x = 2 ∧ y = 4 ∧ z = -2) ∨
   (x = -2 ∧ y = 2 ∧ z = 4) ∨
   (x = -2 ∧ y = 4 ∧ z = 2) ∨
   (x = 4 ∧ y = 2 ∧ z = -2) ∨
   (x = 4 ∧ y = -2 ∧ z = 2)) :=
by sorry


end NUMINAMATH_CALUDE_integer_solution_system_l1823_182370


namespace NUMINAMATH_CALUDE_equality_from_sum_squares_l1823_182378

theorem equality_from_sum_squares (x y z : ℝ) :
  x^2 + y^2 + z^2 = x*y + y*z + z*x → x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equality_from_sum_squares_l1823_182378


namespace NUMINAMATH_CALUDE_inequality_proof_l1823_182371

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b > 1) : a * b < a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1823_182371


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l1823_182327

/-- Given a configuration of two concentric circles and four identical circles
    tangent to each other and the concentric circles, if the radius of the smaller
    concentric circle is 1, then the radius of the larger concentric circle is 3 + 2√2. -/
theorem concentric_circles_radius (r : ℝ) : 
  r > 0 ∧ 
  r^2 - 2*r - 1 = 0 → 
  1 + 2*r = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l1823_182327


namespace NUMINAMATH_CALUDE_marble_jar_count_l1823_182391

theorem marble_jar_count : ∃ (total : ℕ), 
  (total / 2 : ℕ) + (total / 4 : ℕ) + 27 + 14 = total ∧ total = 164 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_count_l1823_182391


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1823_182351

theorem largest_prime_divisor_factorial_sum : ∃ p : ℕ, 
  Nat.Prime p ∧ 
  p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1823_182351


namespace NUMINAMATH_CALUDE_no_intersection_intersection_count_is_zero_l1823_182331

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 1|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
by
  sorry

-- Define the number of intersection points
def intersection_count : ℕ := 0

-- Theorem to prove the number of intersection points is 0
theorem intersection_count_is_zero :
  intersection_count = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_intersection_intersection_count_is_zero_l1823_182331


namespace NUMINAMATH_CALUDE_hyperbola_parabola_same_foci_l1823_182312

-- Define the hyperbola equation
def hyperbola (k : ℝ) (x y : ℝ) : Prop :=
  y^2 / 5 - x^2 / k = 1

-- Define the parabola equation
def parabola (x y : ℝ) : Prop :=
  x^2 = 12 * y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 3)

-- Define the property of having the same foci
def same_foci (k : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 5 + (-k) ∧ c = 3

-- Theorem statement
theorem hyperbola_parabola_same_foci (k : ℝ) :
  same_foci k → k = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_same_foci_l1823_182312


namespace NUMINAMATH_CALUDE_bus_departure_interval_l1823_182319

/-- The departure interval of the bus -/
def x : ℝ := sorry

/-- The speed of the bus -/
def bus_speed : ℝ := sorry

/-- The speed of Xiao Hong -/
def xiao_hong_speed : ℝ := sorry

/-- The time interval between buses passing Xiao Hong from behind -/
def overtake_interval : ℝ := 6

/-- The time interval between buses approaching Xiao Hong head-on -/
def approach_interval : ℝ := 3

theorem bus_departure_interval :
  (overtake_interval * (bus_speed - xiao_hong_speed) = x * bus_speed) ∧
  (approach_interval * (bus_speed + xiao_hong_speed) = x * bus_speed) →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_bus_departure_interval_l1823_182319


namespace NUMINAMATH_CALUDE_inscribed_prism_volume_l1823_182368

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A regular triangular prism inscribed in a regular tetrahedron -/
structure InscribedPrism (a : ℝ) extends RegularTetrahedron a where
  /-- One base of the prism has vertices on the lateral edges of the tetrahedron -/
  base_on_edges : Bool
  /-- The other base of the prism lies in the plane of the tetrahedron's base -/
  base_in_plane : Bool
  /-- All edges of the prism are equal -/
  equal_edges : Bool

/-- The volume of the inscribed prism -/
noncomputable def prism_volume (p : InscribedPrism a) : ℝ :=
  (a^3 * (27 * Real.sqrt 2 - 22 * Real.sqrt 3)) / 2

/-- Theorem: The volume of the inscribed prism is (a³(27√2 - 22√3))/2 -/
theorem inscribed_prism_volume (a : ℝ) (p : InscribedPrism a) :
  prism_volume p = (a^3 * (27 * Real.sqrt 2 - 22 * Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_prism_volume_l1823_182368


namespace NUMINAMATH_CALUDE_inequality_proof_l1823_182313

theorem inequality_proof (x : ℝ) (h : x ≠ 2) : x^2 / (x - 2)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1823_182313


namespace NUMINAMATH_CALUDE_bill_animals_l1823_182365

-- Define the number of rats
def num_rats : ℕ := 60

-- Define the relationship between rats and chihuahuas
def num_chihuahuas : ℕ := num_rats / 6

-- Define the total number of animals
def total_animals : ℕ := num_rats + num_chihuahuas

-- Theorem to prove
theorem bill_animals : total_animals = 70 := by
  sorry

end NUMINAMATH_CALUDE_bill_animals_l1823_182365


namespace NUMINAMATH_CALUDE_equation_solutions_l1823_182302

theorem equation_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → x + 1 ≠ 7 * (x - 1) - x^2) ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → (x + 1 = x * (10 - x) - 7 ↔ x = 8 ∨ x = 1)) ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → (x + 1 = x * (7 - x) + 1 ↔ x = 6)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1823_182302


namespace NUMINAMATH_CALUDE_gain_percent_example_l1823_182361

/-- Calculates the gain percent given the cost price and selling price -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the gain percent is 50% when an article is bought for $10 and sold for $15 -/
theorem gain_percent_example : gain_percent 10 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_example_l1823_182361


namespace NUMINAMATH_CALUDE_square_area_l1823_182362

/-- The area of a square with side length 11 cm is 121 cm². -/
theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length ^ 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l1823_182362


namespace NUMINAMATH_CALUDE_equal_pairs_l1823_182336

theorem equal_pairs : 
  (-2^4 ≠ (-2)^4) ∧ 
  (5^3 ≠ 3^5) ∧ 
  (-(-3) ≠ -|(-3)|) ∧ 
  ((-1)^2 = (-1)^2008) := by
  sorry

end NUMINAMATH_CALUDE_equal_pairs_l1823_182336


namespace NUMINAMATH_CALUDE_find_m_l1823_182364

theorem find_m : ∃ m : ℤ, 3^4 - 6 = 5^2 + m ∧ m = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1823_182364


namespace NUMINAMATH_CALUDE_average_weight_problem_l1823_182389

theorem average_weight_problem (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 37 →
  (b + c) / 2 = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1823_182389


namespace NUMINAMATH_CALUDE_inequality_solution_l1823_182345

theorem inequality_solution : 
  ∀ x y : ℤ, 
    (x - 3*y + 2 ≥ 1) → 
    (-x + 2*y + 1 ≥ 1) → 
    (x^2 / Real.sqrt (x - 3*y + 2 : ℝ) + y^2 / Real.sqrt (-x + 2*y + 1 : ℝ) ≥ y^2 + 2*x^2 - 2*x - 1) →
    ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 1)) := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_inequality_solution_l1823_182345


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_200_l1823_182340

theorem largest_multiple_of_11_below_negative_200 :
  ∀ n : ℤ, n * 11 < -200 → n * 11 ≤ -209 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_200_l1823_182340


namespace NUMINAMATH_CALUDE_range_of_f_l1823_182386

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1823_182386


namespace NUMINAMATH_CALUDE_square_area_ratio_l1823_182308

/-- Given three square regions A, B, and C, where the perimeter of A is 20 units and
    the perimeter of B is 40 units, and assuming the side length of C increases
    proportionally from B as B did from A, the ratio of the area of A to the area of C is 1/16. -/
theorem square_area_ratio (A B C : ℝ) : 
  (A * 4 = 20) →  -- Perimeter of A is 20 units
  (B * 4 = 40) →  -- Perimeter of B is 40 units
  (C = 2 * B) →   -- Side length of C increases proportionally
  (A^2 / C^2 = 1/16) :=
by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1823_182308


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1823_182375

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the interior angles of a pentagon is 540 degrees. -/
theorem sum_interior_angles_pentagon : 
  sum_interior_angles pentagon_sides = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1823_182375


namespace NUMINAMATH_CALUDE_forty_percent_changed_ratings_l1823_182324

/-- Represents the survey results for parents' ratings of online class experience -/
structure SurveyResults where
  total_parents : ℕ
  upgrade_percent : ℚ
  maintain_percent : ℚ
  downgrade_percent : ℚ

/-- Calculates the percentage of parents who changed their ratings -/
def changed_ratings_percentage (results : SurveyResults) : ℚ :=
  (results.upgrade_percent + results.downgrade_percent) * 100

/-- Theorem stating that given the survey conditions, 40% of parents changed their ratings -/
theorem forty_percent_changed_ratings (results : SurveyResults) 
  (h1 : results.total_parents = 120)
  (h2 : results.upgrade_percent = 30 / 100)
  (h3 : results.maintain_percent = 60 / 100)
  (h4 : results.downgrade_percent = 10 / 100)
  (h5 : results.upgrade_percent + results.maintain_percent + results.downgrade_percent = 1) :
  changed_ratings_percentage results = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_changed_ratings_l1823_182324


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l1823_182318

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 7 - 3 * y + 2 > 23 → y ≤ x) ↔ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l1823_182318


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l1823_182394

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  n < 1000 ∧ 
  ∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l1823_182394


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l1823_182353

/-- Proves that given a man rowing downstream with a current speed of 3 kmph,
    covering 80 meters in 15.99872010239181 seconds, his speed in still water is 15 kmph. -/
theorem mans_speed_in_still_water
  (current_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : current_speed = 3)
  (h2 : distance = 80)
  (h3 : time = 15.99872010239181)
  : ∃ (speed_still_water : ℝ), speed_still_water = 15 := by
  sorry

#check mans_speed_in_still_water

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l1823_182353


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1823_182347

theorem sum_of_roots_quadratic : ∃ (x₁ x₂ : ℤ),
  (x₁^2 = x₁ + 272) ∧ 
  (x₂^2 = x₂ + 272) ∧ 
  (∀ x : ℤ, x^2 = x + 272 → x = x₁ ∨ x = x₂) ∧
  (x₁ + x₂ = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1823_182347


namespace NUMINAMATH_CALUDE_negative_of_negative_two_equals_two_l1823_182352

theorem negative_of_negative_two_equals_two : -(-2) = 2 := by sorry

end NUMINAMATH_CALUDE_negative_of_negative_two_equals_two_l1823_182352


namespace NUMINAMATH_CALUDE_log_3897_between_consecutive_integers_l1823_182344

theorem log_3897_between_consecutive_integers : 
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 3897 / Real.log 10 ∧ Real.log 3897 / Real.log 10 < b ∧ a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_3897_between_consecutive_integers_l1823_182344


namespace NUMINAMATH_CALUDE_triangle_area_is_nine_l1823_182304

-- Define the slopes and intersection point
def slope1 : ℚ := 1/3
def slope2 : ℚ := 3
def intersection : ℚ × ℚ := (1, 1)

-- Define the lines
def line1 (x : ℚ) : ℚ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℚ) : ℚ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℚ) : Prop := x + y = 8

-- Define the triangle area function
def triangle_area (A B C : ℚ × ℚ) : ℚ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Theorem statement
theorem triangle_area_is_nine :
  ∃ A B C : ℚ × ℚ,
    A = intersection ∧
    line3 B.1 B.2 ∧
    line3 C.1 C.2 ∧
    B.2 = line1 B.1 ∧
    C.2 = line2 C.1 ∧
    triangle_area A B C = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_nine_l1823_182304


namespace NUMINAMATH_CALUDE_graduating_class_size_l1823_182332

theorem graduating_class_size 
  (geometry_count : ℕ) 
  (biology_count : ℕ) 
  (overlap_difference : ℕ) 
  (h1 : geometry_count = 144) 
  (h2 : biology_count = 119) 
  (h3 : overlap_difference = 88) :
  geometry_count + biology_count - (biology_count - overlap_difference) = 232 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_size_l1823_182332


namespace NUMINAMATH_CALUDE_line_of_symmetry_l1823_182305

/-- Definition of circle O -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

/-- Definition of line l -/
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

/-- Theorem stating that line l is the line of symmetry for circles O and C -/
theorem line_of_symmetry :
  ∀ (x y : ℝ), line_l x y → (∃ (x' y' : ℝ), circle_O x' y' ∧ circle_C x y ∧
    x' = 2*x - x ∧ y' = 2*y - y) :=
sorry

end NUMINAMATH_CALUDE_line_of_symmetry_l1823_182305


namespace NUMINAMATH_CALUDE_cards_distribution_l1823_182333

/-- Given a total number of cards and people, calculates how many people receive fewer than the ceiling of the average number of cards. -/
def people_with_fewer_cards (total_cards : ℕ) (total_people : ℕ) : ℕ :=
  let avg_cards := total_cards / total_people
  let remainder := total_cards % total_people
  let max_cards := avg_cards + 1
  total_people - remainder

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) 
  (h1 : total_cards = 60) (h2 : total_people = 9) :
  people_with_fewer_cards total_cards total_people = 3 := by
  sorry

#eval people_with_fewer_cards 60 9

end NUMINAMATH_CALUDE_cards_distribution_l1823_182333


namespace NUMINAMATH_CALUDE_profit_rate_change_is_three_percent_l1823_182396

/-- Represents the change in profit rate that causes A's income to increase by 300 --/
def profit_rate_change (a_share : ℚ) (a_capital : ℕ) (income_increase : ℕ) : ℚ :=
  (income_increase : ℚ) / a_capital / a_share * 100

/-- Theorem stating the change in profit rate given the problem conditions --/
theorem profit_rate_change_is_three_percent :
  profit_rate_change (2/3) 15000 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_rate_change_is_three_percent_l1823_182396


namespace NUMINAMATH_CALUDE_triangle_side_length_l1823_182372

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →  -- Positive side lengths
  (0 < A ∧ A < π) →  -- Valid angle measures
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →  -- Sum of angles in a triangle
  (c * Real.cos B = 12) →  -- Given condition
  (b * Real.sin C = 5) →  -- Given condition
  (a / Real.sin A = b / Real.sin B) →  -- Sine rule
  (b / Real.sin B = c / Real.sin C) →  -- Sine rule
  c = 13 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1823_182372


namespace NUMINAMATH_CALUDE_greatest_n_value_l1823_182307

theorem greatest_n_value (n : ℤ) (h : 102 * n^2 ≤ 8100) : n ≤ 8 ∧ ∃ (m : ℤ), m = 8 ∧ 102 * m^2 ≤ 8100 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l1823_182307


namespace NUMINAMATH_CALUDE_prob_coprime_42_l1823_182384

/-- The number of positive integers less than or equal to n that are relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

theorem prob_coprime_42 : (phi 42 : ℚ) / 42 = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_coprime_42_l1823_182384


namespace NUMINAMATH_CALUDE_unique_prime_perfect_power_l1823_182322

theorem unique_prime_perfect_power : 
  ∃! p : ℕ, p.Prime ∧ p ≤ 1000 ∧ ∃ m n : ℕ, n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_power_l1823_182322


namespace NUMINAMATH_CALUDE_square_root_equality_l1823_182397

theorem square_root_equality (m : ℝ) (x : ℝ) (h1 : m > 0) 
  (h2 : Real.sqrt m = x + 1) (h3 : Real.sqrt m = 5 + 2*x) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l1823_182397


namespace NUMINAMATH_CALUDE_sum_squares_and_products_ge_ten_l1823_182356

theorem sum_squares_and_products_ge_ten (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (prod_eq_one : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_and_products_ge_ten_l1823_182356


namespace NUMINAMATH_CALUDE_calculation_proof_l1823_182350

theorem calculation_proof :
  ((-56 * (-3/8)) / (-1 - 2/5) = -15) ∧
  ((-12) / (-4) * (1/4) = 3/4) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1823_182350


namespace NUMINAMATH_CALUDE_lcm_count_l1823_182317

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (9^9) (Nat.lcm (12^12) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (9^9) (Nat.lcm (12^12) k) ≠ 18^18)) :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_count_l1823_182317


namespace NUMINAMATH_CALUDE_largest_package_size_l1823_182363

theorem largest_package_size (a b c : ℕ) (ha : a = 60) (hb : b = 36) (hc : c = 48) :
  Nat.gcd a (Nat.gcd b c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1823_182363


namespace NUMINAMATH_CALUDE_number_of_students_is_five_l1823_182342

/-- The number of students who will receive stickers from Miss Walter -/
def number_of_students : ℕ :=
  let gold_stickers : ℕ := 50
  let silver_stickers : ℕ := 2 * gold_stickers
  let bronze_stickers : ℕ := silver_stickers - 20
  let total_stickers : ℕ := gold_stickers + silver_stickers + bronze_stickers
  let stickers_per_student : ℕ := 46
  total_stickers / stickers_per_student

/-- Theorem stating that the number of students who will receive stickers is 5 -/
theorem number_of_students_is_five : number_of_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_is_five_l1823_182342


namespace NUMINAMATH_CALUDE_max_min_product_l1823_182367

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 12) (h_prod_sum : x*y + y*z + z*x = 30) :
  ∃ (n : ℝ), n = min (x*y) (min (y*z) (z*x)) ∧ n ≤ 2 ∧ 
  ∀ (m : ℝ), (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 12 ∧ a*b + b*c + c*a = 30 ∧ 
    m = min (a*b) (min (b*c) (c*a))) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l1823_182367


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_27_factorial_l1823_182383

/-- The largest power of 3 that divides n! -/
def largest_power_of_three_dividing_factorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27)

/-- The ones digit of 3^n -/
def ones_digit_of_power_of_three (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case is unreachable, but needed for exhaustiveness

theorem ones_digit_of_largest_power_of_three_dividing_27_factorial :
  ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 27) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_27_factorial_l1823_182383


namespace NUMINAMATH_CALUDE_fraction_equality_l1823_182393

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) (h1 : a / b = 3 / 4) : a / (a + b) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1823_182393


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l1823_182388

-- Define the work day in minutes
def work_day_minutes : ℕ := 8 * 60

-- Define the durations of the meetings
def meeting1_duration : ℕ := 30
def meeting2_duration : ℕ := 60
def meeting3_duration : ℕ := meeting1_duration + meeting2_duration

-- Define the total meeting time
def total_meeting_time : ℕ := meeting1_duration + meeting2_duration + meeting3_duration

-- Theorem to prove
theorem meetings_percentage_of_workday :
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l1823_182388


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1823_182330

theorem simplify_sqrt_sum : 2 * Real.sqrt 8 + 3 * Real.sqrt 32 = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1823_182330


namespace NUMINAMATH_CALUDE_production_rates_correct_l1823_182380

/-- Represents the production data for a company in November and March --/
structure ProductionData where
  nov_production : ℕ
  mar_production : ℕ
  time_difference : ℕ
  efficiency_ratio : Rat

/-- Calculates the production rates given the production data --/
def calculate_production_rates (data : ProductionData) : ℚ × ℚ :=
  let nov_rate := 2 * data.efficiency_ratio
  let mar_rate := 3 * data.efficiency_ratio
  (nov_rate, mar_rate)

theorem production_rates_correct (data : ProductionData) 
  (h1 : data.nov_production = 1400)
  (h2 : data.mar_production = 2400)
  (h3 : data.time_difference = 50)
  (h4 : data.efficiency_ratio = 2/3) :
  calculate_production_rates data = (4, 6) := by
  sorry

#eval calculate_production_rates {
  nov_production := 1400,
  mar_production := 2400,
  time_difference := 50,
  efficiency_ratio := 2/3
}

end NUMINAMATH_CALUDE_production_rates_correct_l1823_182380


namespace NUMINAMATH_CALUDE_water_filling_solution_l1823_182320

/-- Represents the water filling problem -/
def WaterFillingProblem (canCapacity : ℝ) (initialCans : ℕ) (initialFillRatio : ℝ) (initialTime : ℝ) (targetCans : ℕ) : Prop :=
  let initialWaterFilled := canCapacity * initialFillRatio * initialCans
  let fillRate := initialWaterFilled / initialTime
  let targetWaterToFill := canCapacity * targetCans
  targetWaterToFill / fillRate = 5

/-- Theorem stating the solution to the water filling problem -/
theorem water_filling_solution :
  WaterFillingProblem 8 20 (3/4) 3 25 := by
  sorry

end NUMINAMATH_CALUDE_water_filling_solution_l1823_182320


namespace NUMINAMATH_CALUDE_happy_number_transformation_l1823_182373

def is_happy_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 + (n / 10 % 10) - n % 10 = 6)

def transform (m : ℕ) : ℕ :=
  let c := m % 10
  let a := m / 100
  let b := (m / 10) % 10
  2 * c * 100 + a * 10 + b

theorem happy_number_transformation :
  {m : ℕ | is_happy_number m ∧ is_happy_number (transform m)} = {532, 464} := by
  sorry

end NUMINAMATH_CALUDE_happy_number_transformation_l1823_182373


namespace NUMINAMATH_CALUDE_speed_against_stream_calculation_mans_speed_is_six_l1823_182337

/-- Calculate the speed against the stream given the rate in still water and speed with the stream -/
def speed_against_stream (rate_still : ℝ) (speed_with : ℝ) : ℝ :=
  |rate_still - 2 * (speed_with - rate_still)|

/-- Theorem: Given a man's rowing rate in still water and his speed with the stream,
    his speed against the stream is the absolute difference between his rate in still water
    and twice the difference of his speed with the stream and his rate in still water -/
theorem speed_against_stream_calculation (rate_still : ℝ) (speed_with : ℝ) :
  speed_against_stream rate_still speed_with = 
  |rate_still - 2 * (speed_with - rate_still)| := by
  sorry

/-- The man's speed against the stream given his rate in still water and speed with the stream -/
def mans_speed_against_stream : ℝ :=
  speed_against_stream 5 16

/-- Theorem: The man's speed against the stream is 6 km/h -/
theorem mans_speed_is_six :
  mans_speed_against_stream = 6 := by
  sorry

end NUMINAMATH_CALUDE_speed_against_stream_calculation_mans_speed_is_six_l1823_182337


namespace NUMINAMATH_CALUDE_equation_solution_l1823_182346

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (2 * x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1823_182346


namespace NUMINAMATH_CALUDE_average_hamburgers_per_day_l1823_182398

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7

theorem average_hamburgers_per_day :
  (total_hamburgers : ℚ) / (days_in_week : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_hamburgers_per_day_l1823_182398


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1823_182338

theorem triangle_angle_c (A B C : ℝ) (a b c : ℝ) : 
  A = 80 * π / 180 →
  a^2 = b * (b + c) →
  A + B + C = π →
  a = 2 * Real.sin (A / 2) →
  b = 2 * Real.sin (B / 2) →
  c = 2 * Real.sin (C / 2) →
  C = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1823_182338


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l1823_182343

theorem complex_expression_equals_negative_two :
  let z : ℂ := Complex.exp (3 * Real.pi * Complex.I / 8)
  (z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l1823_182343


namespace NUMINAMATH_CALUDE_function_property_l1823_182354

theorem function_property (f : ℝ → ℝ) (k : ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + k * f x * y) :
  ∃ (a b : ℝ), (a = 0 ∧ b = 4) ∧ 
    (f 2 = a ∨ f 2 = b) ∧
    (∀ c : ℝ, f 2 = c → (c = a ∨ c = b)) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1823_182354


namespace NUMINAMATH_CALUDE_lottery_is_simple_random_sampling_l1823_182358

/-- Represents a sampling method --/
inductive SamplingMethod
| PostcardLottery
| SystematicSampling
| StratifiedSampling
| LotteryMethod

/-- Defines the concept of simple random sampling --/
def isSimpleRandomSampling (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.LotteryMethod => true
  | _ => false

/-- Theorem stating that the lottery method is a simple random sampling method --/
theorem lottery_is_simple_random_sampling :
  isSimpleRandomSampling SamplingMethod.LotteryMethod := by
  sorry

#check lottery_is_simple_random_sampling

end NUMINAMATH_CALUDE_lottery_is_simple_random_sampling_l1823_182358


namespace NUMINAMATH_CALUDE_right_triangle_area_l1823_182377

/-- The area of a right triangle with legs of 30 inches and 45 inches is 675 square inches. -/
theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 45) : 
  (1/2) * a * b = 675 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1823_182377


namespace NUMINAMATH_CALUDE_equation_solution_l1823_182329

theorem equation_solution :
  ∀ x : ℝ, (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1823_182329


namespace NUMINAMATH_CALUDE_cruz_marbles_l1823_182311

/-- The number of marbles Atticus has -/
def atticus : ℕ := 4

/-- The number of marbles Jensen has -/
def jensen : ℕ := 2 * atticus

/-- The number of marbles Cruz has -/
def cruz : ℕ := 20 - (atticus + jensen)

/-- The total number of marbles -/
def total : ℕ := atticus + jensen + cruz

theorem cruz_marbles :
  (3 * total = 60) ∧ (atticus = 4) ∧ (jensen = 2 * atticus) → cruz = 8 := by
  sorry


end NUMINAMATH_CALUDE_cruz_marbles_l1823_182311


namespace NUMINAMATH_CALUDE_min_steps_equal_iff_path_without_leaves_l1823_182335

/-- Represents a tree of players and ropes -/
structure PlayerTree where
  players : ℕ
  ropes : ℕ
  is_tree : ropes = players - 1

/-- Minimum steps to form a path in unrestricted scenario -/
def min_steps_unrestricted (t : PlayerTree) : ℕ := sorry

/-- Minimum steps to form a path in neighbor-only scenario -/
def min_steps_neighbor_only (t : PlayerTree) : ℕ := sorry

/-- Checks if the tree without leaves is a path -/
def is_path_without_leaves (t : PlayerTree) : Prop := sorry

/-- Main theorem: equality of minimum steps iff tree without leaves is a path -/
theorem min_steps_equal_iff_path_without_leaves (t : PlayerTree) :
  min_steps_unrestricted t = min_steps_neighbor_only t ↔ is_path_without_leaves t := by
  sorry

end NUMINAMATH_CALUDE_min_steps_equal_iff_path_without_leaves_l1823_182335


namespace NUMINAMATH_CALUDE_smallest_divisible_by_million_l1823_182334

/-- Represents a geometric sequence with first term a and common ratio r -/
def GeometricSequence (a : ℚ) (r : ℚ) : ℕ → ℚ := λ n => a * r^(n - 1)

/-- The nth term of the specific geometric sequence in the problem -/
def SpecificSequence : ℕ → ℚ := GeometricSequence (1/2) 60

/-- Predicate to check if a rational number is divisible by one million -/
def DivisibleByMillion (q : ℚ) : Prop := ∃ (k : ℤ), q = (k : ℚ) * 1000000

theorem smallest_divisible_by_million :
  (∀ n < 7, ¬ DivisibleByMillion (SpecificSequence n)) ∧
  DivisibleByMillion (SpecificSequence 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_million_l1823_182334


namespace NUMINAMATH_CALUDE_expected_red_lights_value_l1823_182325

/-- The number of traffic posts -/
def n : ℕ := 3

/-- The probability of encountering a red light at each post -/
def p : ℝ := 0.4

/-- The expected number of red lights encountered -/
def expected_red_lights : ℝ := n * p

/-- Theorem: The expected number of red lights encountered is 1.2 -/
theorem expected_red_lights_value : expected_red_lights = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_lights_value_l1823_182325


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1823_182316

/-- An isosceles triangle with two sides of lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = b ∨ b = c ∨ a = c) →
  ((a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) ∨ (b = 2 ∧ c = 4) ∨ (b = 4 ∧ c = 2) ∨ (a = 2 ∧ c = 4) ∨ (a = 4 ∧ c = 2)) →
  a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1823_182316


namespace NUMINAMATH_CALUDE_sample_size_C_l1823_182323

def total_students : ℕ := 150 + 150 + 400 + 300
def students_in_C : ℕ := 400
def total_survey_size : ℕ := 40

theorem sample_size_C : 
  (students_in_C * total_survey_size) / total_students = 16 :=
sorry

end NUMINAMATH_CALUDE_sample_size_C_l1823_182323


namespace NUMINAMATH_CALUDE_binomial_60_3_l1823_182349

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1823_182349


namespace NUMINAMATH_CALUDE_minimize_sqrt_difference_l1823_182355

theorem minimize_sqrt_difference (p : ℕ) (h_p : Nat.Prime p) (h_odd : Odd p) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    (∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0 ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

#check minimize_sqrt_difference

end NUMINAMATH_CALUDE_minimize_sqrt_difference_l1823_182355


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1823_182301

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x ≤ 24 ∧ (1015 + x) % 25 = 0 ∧ ∀ y : ℕ, y < x → (1015 + y) % 25 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1823_182301


namespace NUMINAMATH_CALUDE_binomial_12_3_l1823_182314

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l1823_182314


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1823_182300

theorem perfect_square_sum (a b : ℕ) :
  (∃ k : ℕ, 2^(2*a) + 2^b + 5 = k^2) → (a + b = 4 ∨ a + b = 5) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1823_182300


namespace NUMINAMATH_CALUDE_hayley_meatballs_l1823_182309

/-- The number of meatballs Hayley has left after Kirsten stole some -/
def meatballs_left (initial : ℕ) (stolen : ℕ) : ℕ :=
  initial - stolen

/-- Theorem stating that Hayley has 11 meatballs left -/
theorem hayley_meatballs : meatballs_left 25 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hayley_meatballs_l1823_182309


namespace NUMINAMATH_CALUDE_exterior_angle_regular_nonagon_exterior_angle_regular_nonagon_proof_l1823_182303

/-- The measure of an exterior angle in a regular nonagon is 40 degrees. -/
theorem exterior_angle_regular_nonagon : ℝ :=
  40

/-- A regular nonagon has 9 sides. -/
def regular_nonagon_sides : ℕ := 9

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- An exterior angle and its corresponding interior angle sum to 180 degrees. -/
axiom exterior_interior_sum : ℝ → ℝ → Prop

/-- The measure of an exterior angle in a regular nonagon is 40 degrees. -/
theorem exterior_angle_regular_nonagon_proof :
  exterior_angle_regular_nonagon =
    180 - (sum_interior_angles regular_nonagon_sides / regular_nonagon_sides) :=
by
  sorry

#check exterior_angle_regular_nonagon_proof

end NUMINAMATH_CALUDE_exterior_angle_regular_nonagon_exterior_angle_regular_nonagon_proof_l1823_182303


namespace NUMINAMATH_CALUDE_largest_increase_2007_2008_l1823_182339

def students : Fin 6 → ℕ
  | 0 => 50  -- 2003
  | 1 => 58  -- 2004
  | 2 => 65  -- 2005
  | 3 => 75  -- 2006
  | 4 => 80  -- 2007
  | 5 => 100 -- 2008

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

theorem largest_increase_2007_2008 :
  ∀ i : Fin 5, percentageIncrease (students 4) (students 5) ≥ percentageIncrease (students i) (students (i + 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2007_2008_l1823_182339


namespace NUMINAMATH_CALUDE_problem_statement_l1823_182306

theorem problem_statement (x : ℚ) : 5 * x - 10 = 15 * x + 5 → 5 * (x + 3) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1823_182306


namespace NUMINAMATH_CALUDE_exactly_one_fail_probability_l1823_182395

/-- The probability that exactly one item fails the inspection when one item is taken from each of two types of products with pass rates of 0.90 and 0.95 respectively is 0.14. -/
theorem exactly_one_fail_probability (pass_rate1 pass_rate2 : ℝ) 
  (h1 : pass_rate1 = 0.90) (h2 : pass_rate2 = 0.95) : 
  pass_rate1 * (1 - pass_rate2) + (1 - pass_rate1) * pass_rate2 = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_fail_probability_l1823_182395


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l1823_182328

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) 
  (fishermen_with_400 : ℕ) (fish_per_fisherman : ℕ) :
  total_fishermen = 20 →
  total_fish = 10000 →
  fishermen_with_400 = 19 →
  fish_per_fisherman = 400 →
  total_fish - (fishermen_with_400 * fish_per_fisherman) = 2400 := by
sorry

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l1823_182328


namespace NUMINAMATH_CALUDE_original_cost_l1823_182392

theorem original_cost (final_cost : ℝ) : 
  final_cost = 72 → 
  ∃ (original_cost : ℝ), 
    original_cost * (1 + 0.2) * (1 - 0.2) = final_cost ∧ 
    original_cost = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_original_cost_l1823_182392


namespace NUMINAMATH_CALUDE_mike_current_salary_l1823_182360

def mike_salary_five_months_ago : ℕ := 10000
def fred_salary_five_months_ago : ℕ := 1000
def salary_increase_percentage : ℕ := 40

theorem mike_current_salary :
  let total_salary_five_months_ago := mike_salary_five_months_ago + fred_salary_five_months_ago
  let salary_increase := (salary_increase_percentage * total_salary_five_months_ago) / 100
  mike_salary_five_months_ago + salary_increase = 15400 := by
  sorry

end NUMINAMATH_CALUDE_mike_current_salary_l1823_182360


namespace NUMINAMATH_CALUDE_cabin_cost_l1823_182376

theorem cabin_cost (total_cost land_cost cabin_cost : ℕ) : 
  total_cost = 30000 →
  land_cost = 4 * cabin_cost →
  total_cost = land_cost + cabin_cost →
  cabin_cost = 6000 := by
  sorry

end NUMINAMATH_CALUDE_cabin_cost_l1823_182376


namespace NUMINAMATH_CALUDE_fraction_equality_l1823_182379

theorem fraction_equality (u v : ℝ) (h : (1/u + 1/v) / (1/u - 1/v) = 2024) :
  (u + v) / (u - v) = 2024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1823_182379


namespace NUMINAMATH_CALUDE_cyclic_inequality_l1823_182348

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l1823_182348


namespace NUMINAMATH_CALUDE_man_double_son_age_l1823_182321

/-- Calculates the number of years until a man's age is twice his son's age. -/
def yearsUntilDoubleAge (manAge sonAge : ℕ) : ℕ :=
  sorry

/-- Proves that the number of years until the man's age is twice his son's age is 2. -/
theorem man_double_son_age :
  let sonAge : ℕ := 14
  let manAge : ℕ := sonAge + 16
  yearsUntilDoubleAge manAge sonAge = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_double_son_age_l1823_182321


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l1823_182381

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_area := 6 * s^2
  let new_edge := 1.25 * s
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l1823_182381


namespace NUMINAMATH_CALUDE_ella_work_days_l1823_182374

/-- Represents the number of days Ella worked at each age --/
structure WorkDays where
  age10 : ℕ
  age11 : ℕ
  age12 : ℕ

/-- Calculates the total pay for the given work days --/
def totalPay (w : WorkDays) : ℕ :=
  4 * (10 * w.age10 + 11 * w.age11 + 12 * w.age12)

theorem ella_work_days :
  ∃ (w : WorkDays),
    w.age10 + w.age11 + w.age12 = 180 ∧
    totalPay w = 7920 ∧
    w.age11 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ella_work_days_l1823_182374


namespace NUMINAMATH_CALUDE_factorial_simplification_l1823_182359

theorem factorial_simplification : (12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l1823_182359


namespace NUMINAMATH_CALUDE_polynomial_always_positive_l1823_182387

theorem polynomial_always_positive (x : ℝ) : x^12 - x^9 + x^4 - x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_always_positive_l1823_182387
