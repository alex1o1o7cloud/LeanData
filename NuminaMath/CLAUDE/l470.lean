import Mathlib

namespace NUMINAMATH_CALUDE_clown_balloons_theorem_l470_47096

def balloons_problem (initial_dozens : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  initial_dozens * 12 - (boys + girls)

theorem clown_balloons_theorem :
  balloons_problem 3 3 12 = 21 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_theorem_l470_47096


namespace NUMINAMATH_CALUDE_cuboid_area_example_l470_47055

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 6 cm, and height 9 cm is 348 cm² -/
theorem cuboid_area_example : cuboid_surface_area 8 6 9 = 348 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l470_47055


namespace NUMINAMATH_CALUDE_baker_sold_cakes_l470_47090

/-- Given that Baker bought 31 cakes and sold 47 more cakes than he bought,
    prove that Baker sold 78 cakes. -/
theorem baker_sold_cakes : ℕ → Prop :=
  fun cakes_bought : ℕ =>
    cakes_bought = 31 →
    ∃ cakes_sold : ℕ,
      cakes_sold = cakes_bought + 47 ∧
      cakes_sold = 78

/-- Proof of the theorem -/
lemma prove_baker_sold_cakes : baker_sold_cakes 31 := by
  sorry

end NUMINAMATH_CALUDE_baker_sold_cakes_l470_47090


namespace NUMINAMATH_CALUDE_base_8_units_digit_l470_47037

theorem base_8_units_digit : ∃ n : ℕ, (356 * 78 + 49) % 8 = 1 ∧ (356 * 78 + 49) = 8 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_base_8_units_digit_l470_47037


namespace NUMINAMATH_CALUDE_otimes_three_four_l470_47047

-- Define the ⊗ operation
def otimes (m : ℤ) (a b : ℕ) : ℚ :=
  (m * a + b) / (2 * a * b)

-- Theorem statement
theorem otimes_three_four (m : ℤ) :
  (∀ (a b : ℕ), a ≠ 0 → b ≠ 0 → otimes m a b = otimes m b a) →
  otimes m 1 4 = otimes m 2 3 →
  otimes m 3 4 = 11 / 12 := by
  sorry


end NUMINAMATH_CALUDE_otimes_three_four_l470_47047


namespace NUMINAMATH_CALUDE_selection_theorem_l470_47017

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of days of service --/
def days : ℕ := 2

/-- The number of people selected each day --/
def selected_per_day : ℕ := 2

/-- The number of ways to select exactly one person to serve for both days --/
def selection_ways : ℕ := n * (n - 1) * (n - 2)

theorem selection_theorem : selection_ways = 60 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l470_47017


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l470_47010

/-- Given a rectangle with sides 9 cm and 12 cm inscribed in a circle,
    prove that the circumference of the circle is 15π cm. -/
theorem inscribed_rectangle_circle_circumference :
  ∀ (circle : Real → Real → Prop) (rectangle : Real → Real → Prop),
    (∃ (x y : Real), rectangle x y ∧ x = 9 ∧ y = 12) →
    (∀ (x y : Real), rectangle x y → ∃ (center : Real × Real) (r : Real),
      circle = λ a b => (a - center.1)^2 + (b - center.2)^2 = r^2) →
    (∃ (circumference : Real), circumference = 15 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l470_47010


namespace NUMINAMATH_CALUDE_collins_initial_flowers_l470_47088

/-- Proves that Collin's initial number of flowers is 25 given the problem conditions --/
theorem collins_initial_flowers :
  ∀ (collins_initial_flowers : ℕ) (ingrids_flowers : ℕ) (petals_per_flower : ℕ) (collins_total_petals : ℕ),
    ingrids_flowers = 33 →
    petals_per_flower = 4 →
    collins_total_petals = 144 →
    collins_total_petals = (collins_initial_flowers + ingrids_flowers / 3) * petals_per_flower →
    collins_initial_flowers = 25 :=
by sorry

end NUMINAMATH_CALUDE_collins_initial_flowers_l470_47088


namespace NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l470_47007

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (J K L M : Point)

/-- Checks if a point is on the extension of a line segment -/
def isOnExtension (A B P : Point) : Prop := sorry

/-- Checks if two line segments intersect at a point -/
def intersectsAt (A B C D Q : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ := sorry

theorem parallelogram_intersection_theorem (JKLM : Parallelogram) (P Q R : Point) :
  isOnExtension JKLM.L JKLM.M P →
  intersectsAt JKLM.K P JKLM.L JKLM.J Q →
  intersectsAt JKLM.K P JKLM.J JKLM.M R →
  distance Q R = 40 →
  distance R P = 30 →
  distance JKLM.K Q = 20 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l470_47007


namespace NUMINAMATH_CALUDE_f_inverse_g_l470_47002

noncomputable def f (x : ℝ) : ℝ := 3 - 7*x + x^2

noncomputable def g (x : ℝ) : ℝ := (7 + Real.sqrt (37 + 4*x)) / 2

theorem f_inverse_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_f_inverse_g_l470_47002


namespace NUMINAMATH_CALUDE_percent_relation_l470_47079

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := by sorry

end NUMINAMATH_CALUDE_percent_relation_l470_47079


namespace NUMINAMATH_CALUDE_portrait_in_silver_box_l470_47069

-- Define the possible box locations
inductive Box
| Gold
| Silver
| Lead

-- Define the propositions
def p (portrait_location : Box) : Prop := portrait_location = Box.Gold
def q (portrait_location : Box) : Prop := portrait_location ≠ Box.Silver
def r (portrait_location : Box) : Prop := portrait_location ≠ Box.Gold

-- Theorem statement
theorem portrait_in_silver_box :
  ∃! (portrait_location : Box),
    (p portrait_location ∨ q portrait_location ∨ r portrait_location) ∧
    (¬(p portrait_location ∧ q portrait_location) ∧
     ¬(p portrait_location ∧ r portrait_location) ∧
     ¬(q portrait_location ∧ r portrait_location)) →
  portrait_location = Box.Silver :=
by sorry

end NUMINAMATH_CALUDE_portrait_in_silver_box_l470_47069


namespace NUMINAMATH_CALUDE_pen_cost_ratio_l470_47018

theorem pen_cost_ratio (blue_pens : ℕ) (red_pens : ℕ) (blue_cost : ℚ) (total_cost : ℚ) : 
  blue_pens = 10 →
  red_pens = 15 →
  blue_cost = 1/10 →
  total_cost = 4 →
  (total_cost - blue_pens * blue_cost) / red_pens / blue_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_ratio_l470_47018


namespace NUMINAMATH_CALUDE_right_triangle_sin_z_l470_47064

theorem right_triangle_sin_z (X Y Z : ℝ) : 
  -- XYZ is a right triangle
  0 ≤ X ∧ X < π/2 ∧ 0 ≤ Y ∧ Y < π/2 ∧ 0 ≤ Z ∧ Z < π/2 ∧ X + Y + Z = π/2 →
  -- sin X = 3/5
  Real.sin X = 3/5 →
  -- cos Y = 0
  Real.cos Y = 0 →
  -- Then sin Z = 3/5
  Real.sin Z = 3/5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_z_l470_47064


namespace NUMINAMATH_CALUDE_count_special_numbers_l470_47000

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def leftmost_digit (n : ℕ) : ℕ := n / 1000

def second_digit (n : ℕ) : ℕ := (n / 100) % 10

def third_digit (n : ℕ) : ℕ := (n / 10) % 10

def last_digit (n : ℕ) : ℕ := n % 10

def all_digits_different (n : ℕ) : Prop :=
  leftmost_digit n ≠ second_digit n ∧
  leftmost_digit n ≠ third_digit n ∧
  leftmost_digit n ≠ last_digit n ∧
  second_digit n ≠ third_digit n ∧
  second_digit n ≠ last_digit n ∧
  third_digit n ≠ last_digit n

theorem count_special_numbers :
  ∃ (S : Finset ℕ),
    (∀ n ∈ S,
      is_four_digit n ∧
      leftmost_digit n % 2 = 1 ∧
      leftmost_digit n < 5 ∧
      second_digit n % 2 = 0 ∧
      second_digit n < 6 ∧
      all_digits_different n ∧
      n % 5 = 0) ∧
    S.card = 48 :=
by sorry

end NUMINAMATH_CALUDE_count_special_numbers_l470_47000


namespace NUMINAMATH_CALUDE_pentagon_area_l470_47013

/-- Represents a pentagon formed by removing a triangular section from a rectangle --/
structure Pentagon where
  sides : Finset ℕ
  area : ℕ

/-- The theorem stating the area of the specific pentagon --/
theorem pentagon_area : ∃ (p : Pentagon), 
  p.sides = {17, 23, 26, 28, 34} ∧ p.area = 832 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l470_47013


namespace NUMINAMATH_CALUDE_goldbach_126_max_diff_l470_47094

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem goldbach_126_max_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p ≠ q ∧ 
    p + q = 126 ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r ≠ s → r + s = 126 → 
      (max r s - min r s) ≤ (max p q - min p q) ∧
    (max p q - min p q) = 100 :=
sorry

end NUMINAMATH_CALUDE_goldbach_126_max_diff_l470_47094


namespace NUMINAMATH_CALUDE_square_of_nine_l470_47043

theorem square_of_nine (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nine_l470_47043


namespace NUMINAMATH_CALUDE_petrol_expense_percentage_l470_47073

/-- Represents the problem of calculating the percentage of income spent on petrol --/
theorem petrol_expense_percentage
  (total_income : ℝ)
  (petrol_expense : ℝ)
  (rent_expense : ℝ)
  (rent_percentage : ℝ)
  (h1 : petrol_expense = 300)
  (h2 : rent_expense = 210)
  (h3 : rent_percentage = 30)
  (h4 : rent_expense = (rent_percentage / 100) * (total_income - petrol_expense)) :
  (petrol_expense / total_income) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_petrol_expense_percentage_l470_47073


namespace NUMINAMATH_CALUDE_line_relationship_indeterminate_l470_47045

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Perpendicularity relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- States that two lines have an indeterminate relationship -/
def indeterminate_relationship (l1 l2 : Line3D) : Prop := sorry

theorem line_relationship_indeterminate 
  (l1 l2 l3 l4 : Line3D) 
  (h1 : perpendicular l1 l2)
  (h2 : parallel l2 l3)
  (h3 : perpendicular l3 l4)
  (h4 : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4) :
  indeterminate_relationship l1 l4 := by
  sorry

end NUMINAMATH_CALUDE_line_relationship_indeterminate_l470_47045


namespace NUMINAMATH_CALUDE_second_particle_catches_up_l470_47041

/-- The time (in minutes) when the second particle enters the pipe after the first -/
def time_difference : ℝ := 6.8

/-- The constant speed of the first particle in meters per minute -/
def speed_first : ℝ := 5

/-- The initial speed of the second particle in meters per minute -/
def initial_speed_second : ℝ := 3

/-- The acceleration of the second particle in meters per minute² -/
def acceleration_second : ℝ := 0.5

/-- The distance traveled by the first particle after time t -/
def distance_first (t : ℝ) : ℝ :=
  speed_first * (time_difference + t)

/-- The distance traveled by the second particle after time t -/
def distance_second (t : ℝ) : ℝ :=
  0.25 * t^2 + 2.75 * t

/-- The time when the second particle catches up with the first -/
def catch_up_time : ℝ := 17

theorem second_particle_catches_up :
  distance_first catch_up_time = distance_second catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_second_particle_catches_up_l470_47041


namespace NUMINAMATH_CALUDE_tan_double_angle_l470_47024

theorem tan_double_angle (α : ℝ) (h : (1 + Real.cos (2 * α)) / Real.sin (2 * α) = 1/2) :
  Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l470_47024


namespace NUMINAMATH_CALUDE_g_composition_of_3_l470_47034

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 2*n + 1 else 2*n + 4

theorem g_composition_of_3 : g (g (g 3)) = 76 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l470_47034


namespace NUMINAMATH_CALUDE_smallest_sector_angle_l470_47083

/-- Represents the properties of the circle division problem -/
structure CircleDivision where
  n : ℕ  -- number of sectors
  a₁ : ℕ  -- first term of the arithmetic sequence
  d : ℕ   -- common difference of the arithmetic sequence
  sum : ℕ -- sum of all angles

/-- The circle division satisfies the problem conditions -/
def validCircleDivision (cd : CircleDivision) : Prop :=
  cd.n = 15 ∧
  cd.sum = 360 ∧
  ∀ i : ℕ, i > 0 ∧ i ≤ cd.n → (cd.a₁ + (i - 1) * cd.d) > 0

/-- The theorem stating the smallest possible sector angle -/
theorem smallest_sector_angle (cd : CircleDivision) :
  validCircleDivision cd →
  (∃ cd' : CircleDivision, validCircleDivision cd' ∧ cd'.a₁ < cd.a₁) ∨ cd.a₁ = 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_sector_angle_l470_47083


namespace NUMINAMATH_CALUDE_no_valid_base_l470_47080

theorem no_valid_base : ¬ ∃ (b : ℕ), 0 < b ∧ b^6 ≤ 196 ∧ 196 < b^7 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_l470_47080


namespace NUMINAMATH_CALUDE_chocolate_game_student_count_l470_47032

def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ (120 - 1) % n = 0

theorem chocolate_game_student_count :
  {n : ℕ | is_valid_student_count n} = {7, 17} := by
  sorry

end NUMINAMATH_CALUDE_chocolate_game_student_count_l470_47032


namespace NUMINAMATH_CALUDE_value_of_y_l470_47050

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 14 ∧ y = 98 / 3 := by sorry

end NUMINAMATH_CALUDE_value_of_y_l470_47050


namespace NUMINAMATH_CALUDE_largest_equal_cost_integer_l470_47019

/-- Calculates the sum of digits for a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the cost of binary representation -/
def binaryCost (n : ℕ) : ℕ := sorry

/-- Theorem stating that 311 is the largest integer less than 500 with equal costs -/
theorem largest_equal_cost_integer :
  ∀ n : ℕ, n < 500 → n > 311 → sumOfDigits n ≠ binaryCost n ∧
  sumOfDigits 311 = binaryCost 311 := by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_integer_l470_47019


namespace NUMINAMATH_CALUDE_square_roots_problem_l470_47011

theorem square_roots_problem (n : ℝ) (h : n > 0) :
  (∃ a : ℝ, (a + 2)^2 = n ∧ (2*a - 11)^2 = n) → n = 225 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l470_47011


namespace NUMINAMATH_CALUDE_towel_shrinkage_l470_47026

theorem towel_shrinkage (original_length original_breadth : ℝ) 
  (h_positive : original_length > 0 ∧ original_breadth > 0) :
  let new_length := 0.7 * original_length
  let new_area := 0.525 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.75 * original_breadth ∧
    new_area = new_length * new_breadth :=
by sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l470_47026


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l470_47025

def isArithmeticSequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sumIsTerm (a : ℕ → ℕ) : Prop :=
  ∀ p q, ∃ k, a k = a p + a q

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℕ) (d : ℕ) 
  (h1 : isArithmeticSequence a d)
  (h2 : a 1 = 9)
  (h3 : sumIsTerm a) :
  d = 1 ∨ d = 3 ∨ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l470_47025


namespace NUMINAMATH_CALUDE_circle_area_equals_circumference_l470_47057

theorem circle_area_equals_circumference (r : ℝ) (h : r > 0) :
  π * r^2 = 2 * π * r → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equals_circumference_l470_47057


namespace NUMINAMATH_CALUDE_tangent_line_equation_l470_47063

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x^2 / E.a^2) + (y^2 / E.b^2) = 1

/-- The equation of a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation of the tangent line to an ellipse at a point on the ellipse -/
theorem tangent_line_equation (E : Ellipse) (P : PointOnEllipse E) :
  ∃ (L : Line), L.a = P.x / E.a^2 ∧ L.b = P.y / E.b^2 ∧ L.c = -1 ∧
  (∀ (x y : ℝ), (x^2 / E.a^2) + (y^2 / E.b^2) ≤ 1 → L.a * x + L.b * y + L.c ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l470_47063


namespace NUMINAMATH_CALUDE_expectation_linear_transformation_l470_47099

variable (X : Type) [MeasurableSpace X]
variable (μ : Measure X)
variable (f : X → ℝ)

noncomputable def expectation (f : X → ℝ) : ℝ := ∫ x, f x ∂μ

theorem expectation_linear_transformation 
  (h : expectation μ f = 6) : 
  expectation μ (fun x => 3 * (f x - 2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_expectation_linear_transformation_l470_47099


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l470_47060

theorem concentric_circles_ratio (r₁ r₂ r₃ : ℝ) (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : r₂ < r₃) :
  (r₂^2 - r₁^2 = 2 * (r₃^2 - r₂^2)) →
  (r₃^2 = 3 * (r₂^2 - r₁^2)) →
  ∃ (k : ℝ), r₃ = k ∧ r₂ = k * Real.sqrt (5/6) ∧ r₁ = k / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l470_47060


namespace NUMINAMATH_CALUDE_faster_pipe_rate_l470_47062

/-- Given two pipes with different filling rates, prove that the faster pipe is 4 times faster than the slower pipe. -/
theorem faster_pipe_rate (slow_rate fast_rate : ℝ) : 
  slow_rate > 0 →
  fast_rate > slow_rate →
  (1 : ℝ) / slow_rate = 180 →
  1 / (slow_rate + fast_rate) = 36 →
  fast_rate = 4 * slow_rate :=
by sorry

end NUMINAMATH_CALUDE_faster_pipe_rate_l470_47062


namespace NUMINAMATH_CALUDE_octahedral_die_red_faces_l470_47049

theorem octahedral_die_red_faces (n : ℕ) (k : ℕ) (opposite_pairs : ℕ) :
  n = 8 →
  k = 2 →
  opposite_pairs = 4 →
  Nat.choose n k - opposite_pairs = 24 :=
by sorry

end NUMINAMATH_CALUDE_octahedral_die_red_faces_l470_47049


namespace NUMINAMATH_CALUDE_six_customOp_three_l470_47042

/-- Definition of the custom operation " -/
def customOp (m n : ℕ) : ℕ := n ^ 2 - m

/-- Theorem stating that 6 " 3 = 3 -/
theorem six_customOp_three : customOp 6 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_six_customOp_three_l470_47042


namespace NUMINAMATH_CALUDE_max_discriminant_quadratic_l470_47071

theorem max_discriminant_quadratic (a b c u v w : ℤ) :
  u ≠ v ∧ u ≠ w ∧ v ≠ w →
  a * u^2 + b * u + c = 0 →
  a * v^2 + b * v + c = 0 →
  a * w^2 + b * w + c = 2 →
  ∃ (max : ℤ), max = 16 ∧ b^2 - 4*a*c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_discriminant_quadratic_l470_47071


namespace NUMINAMATH_CALUDE_absolute_value_sum_l470_47056

theorem absolute_value_sum : |(-8 : ℤ)| + |(-4 : ℤ)| = 12 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l470_47056


namespace NUMINAMATH_CALUDE_martinez_chiquita_height_difference_l470_47051

/-- The height difference between Mr. Martinez and Chiquita -/
theorem martinez_chiquita_height_difference :
  ∀ (martinez_height chiquita_height : ℝ),
  chiquita_height = 5 →
  martinez_height + chiquita_height = 12 →
  martinez_height - chiquita_height = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_martinez_chiquita_height_difference_l470_47051


namespace NUMINAMATH_CALUDE_problem_solution_l470_47027

-- Define the function f
def f (x : ℝ) : ℝ := |2 * |x| - 1|

-- Define the solution set A
def A : Set ℝ := {x | f x ≤ 1}

-- Theorem statement
theorem problem_solution :
  (A = {x : ℝ | -1 ≤ x ∧ x ≤ 1}) ∧
  (∀ m n : ℝ, m ∈ A → n ∈ A → |m + n| ≤ m * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l470_47027


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l470_47015

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.00000043 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 4.3 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l470_47015


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l470_47029

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  a 2 * a 5 = 32 → a 4 * a 7 = 512 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l470_47029


namespace NUMINAMATH_CALUDE_point_on_x_axis_l470_47031

/-- A point P(x, y) lies on the x-axis if and only if its y-coordinate is 0 -/
def lies_on_x_axis (x y : ℝ) : Prop := y = 0

/-- The theorem states that if the point P(a-4, a+3) lies on the x-axis, then a = -3 -/
theorem point_on_x_axis (a : ℝ) :
  lies_on_x_axis (a - 4) (a + 3) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l470_47031


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_odd_integers_product_less_500_l470_47008

theorem greatest_sum_consecutive_odd_integers_product_less_500 : 
  (∃ (n : ℤ), 
    Odd n ∧ 
    n * (n + 2) < 500 ∧ 
    n + (n + 2) = 44 ∧ 
    (∀ (m : ℤ), Odd m → m * (m + 2) < 500 → m + (m + 2) ≤ 44)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_odd_integers_product_less_500_l470_47008


namespace NUMINAMATH_CALUDE_function_equality_l470_47014

theorem function_equality (k : ℝ) (x : ℝ) (h1 : k > 0) (h2 : x ≠ Real.sqrt k) :
  (x^2 - k) / (x - Real.sqrt k) = 3 * x → x = Real.sqrt k / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l470_47014


namespace NUMINAMATH_CALUDE_corn_purchase_proof_l470_47052

/-- The cost of corn in dollars per pound -/
def corn_cost : ℝ := 0.99

/-- The cost of beans in dollars per pound -/
def bean_cost : ℝ := 0.75

/-- The total weight of corn and beans in pounds -/
def total_weight : ℝ := 20

/-- The total cost in dollars -/
def total_cost : ℝ := 16.80

/-- The number of pounds of corn purchased -/
def corn_weight : ℝ := 7.5

theorem corn_purchase_proof :
  ∃ (bean_weight : ℝ),
    bean_weight ≥ 0 ∧
    corn_weight ≥ 0 ∧
    bean_weight + corn_weight = total_weight ∧
    bean_cost * bean_weight + corn_cost * corn_weight = total_cost :=
by sorry

end NUMINAMATH_CALUDE_corn_purchase_proof_l470_47052


namespace NUMINAMATH_CALUDE_expand_and_simplify_1_simplify_division_2_l470_47033

-- Define variables
variable (a b : ℝ)

-- Theorem 1
theorem expand_and_simplify_1 : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b := by
  sorry

-- Theorem 2
theorem simplify_division_2 : (12 * a^3 - 6 * a^2 + 3 * a) / (3 * a) = 4 * a^2 - 2 * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_1_simplify_division_2_l470_47033


namespace NUMINAMATH_CALUDE_sum_in_base4_is_1022_l470_47068

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The sum of 321₄, 32₄, and 3₄ in base 4 --/
def sumInBase4 : List Nat :=
  let sum := base4ToBase10 [1, 2, 3] + base4ToBase10 [2, 3] + base4ToBase10 [3]
  base10ToBase4 sum

theorem sum_in_base4_is_1022 : sumInBase4 = [1, 0, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base4_is_1022_l470_47068


namespace NUMINAMATH_CALUDE_simplify_square_roots_l470_47078

theorem simplify_square_roots : Real.sqrt 12 * Real.sqrt 27 - 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l470_47078


namespace NUMINAMATH_CALUDE_square_side_length_l470_47093

theorem square_side_length (x : ℝ) : 
  x > 0 ∧ 
  x + (x + 17) + (x + 11) = 52 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l470_47093


namespace NUMINAMATH_CALUDE_highest_red_probability_l470_47035

/-- Represents the contents of a bag --/
structure Bag where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a bag --/
def redProbability (bag : Bag) : ℚ :=
  bag.red / (bag.red + bag.white)

/-- The average probability of drawing a red ball from two bags --/
def averageProbability (bag1 bag2 : Bag) : ℚ :=
  (redProbability bag1 + redProbability bag2) / 2

/-- The theorem stating the highest probability of drawing a red ball --/
theorem highest_red_probability :
  ∃ (bag1 bag2 : Bag),
    bag1.red + bag2.red = 5 ∧
    bag1.white + bag2.white = 12 ∧
    bag1.red + bag1.white > 0 ∧
    bag2.red + bag2.white > 0 ∧
    averageProbability bag1 bag2 = 5/8 ∧
    ∀ (other1 other2 : Bag),
      other1.red + other2.red = 5 →
      other1.white + other2.white = 12 →
      other1.red + other1.white > 0 →
      other2.red + other2.white > 0 →
      averageProbability other1 other2 ≤ 5/8 :=
by sorry

end NUMINAMATH_CALUDE_highest_red_probability_l470_47035


namespace NUMINAMATH_CALUDE_remaining_hard_hats_l470_47021

/-- Represents the number of hard hats in the truck -/
structure HardHats :=
  (pink : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of hard hats -/
def totalHardHats (hats : HardHats) : ℕ :=
  hats.pink + hats.green + hats.yellow

/-- Represents the actions of Carl and John -/
def removeHardHats (initial : HardHats) : HardHats :=
  let afterCarl := HardHats.mk (initial.pink - 4) initial.green initial.yellow
  let johnPinkRemoval := 6
  HardHats.mk 
    (afterCarl.pink - johnPinkRemoval)
    (afterCarl.green - 2 * johnPinkRemoval)
    afterCarl.yellow

/-- The main theorem to prove -/
theorem remaining_hard_hats (initial : HardHats) 
  (h1 : initial.pink = 26) 
  (h2 : initial.green = 15) 
  (h3 : initial.yellow = 24) :
  totalHardHats (removeHardHats initial) = 43 := by
  sorry

end NUMINAMATH_CALUDE_remaining_hard_hats_l470_47021


namespace NUMINAMATH_CALUDE_slope_angle_range_l470_47053

-- Define Circle C
def CircleC (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define Line l
def LineL (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the condition that O is inside circle with diameter AB
def OInsideAB (k : ℝ) : Prop := 4 * (k^2 + 1) > 4 * k^2 + 3

-- Main theorem
theorem slope_angle_range :
  ∀ k : ℝ,
  (∃ x y : ℝ, CircleC x y ∧ LineL k x y) →  -- Line l intersects Circle C
  OInsideAB k →                            -- O is inside circle with diameter AB
  Real.arctan (1/2) < Real.arctan k ∧ Real.arctan k < π - Real.arctan (1/2) :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_range_l470_47053


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l470_47022

/-- Given a polynomial p(x) satisfying specific conditions, 
    prove that its remainder when divided by (x-1)(x+1)(x-3) is -x^2 + 4x + 2 -/
theorem polynomial_remainder_theorem (p : ℝ → ℝ) 
  (h1 : p 1 = 5) (h2 : p 3 = 7) (h3 : p (-1) = 9) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 1) * (x + 1) * (x - 3) + (-x^2 + 4*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l470_47022


namespace NUMINAMATH_CALUDE_veg_eaters_count_l470_47089

/-- Represents the number of people in different dietary categories in a family. -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ

/-- Calculates the total number of people who eat veg in the family. -/
def total_veg_eaters (diet : FamilyDiet) : ℕ :=
  diet.only_veg + diet.both_veg_and_non_veg

/-- Theorem stating that the total number of people who eat veg in the family is 19. -/
theorem veg_eaters_count (diet : FamilyDiet)
  (h1 : diet.only_veg = 13)
  (h2 : diet.only_non_veg = 8)
  (h3 : diet.both_veg_and_non_veg = 6) :
  total_veg_eaters diet = 19 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l470_47089


namespace NUMINAMATH_CALUDE_greatest_average_speed_l470_47070

/-- Checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- Finds the greatest palindrome less than or equal to a given number -/
def greatest_palindrome_le (n : ℕ) : ℕ := sorry

theorem greatest_average_speed (initial_reading : ℕ) (trip_duration : ℕ) (max_speed : ℕ) :
  is_palindrome initial_reading →
  initial_reading = 13831 →
  trip_duration = 5 →
  max_speed = 80 →
  let max_distance := max_speed * trip_duration
  let max_final_reading := initial_reading + max_distance
  let actual_final_reading := greatest_palindrome_le max_final_reading
  let distance_traveled := actual_final_reading - initial_reading
  let average_speed := distance_traveled / trip_duration
  average_speed = 62 := by sorry

end NUMINAMATH_CALUDE_greatest_average_speed_l470_47070


namespace NUMINAMATH_CALUDE_remainder_problem_l470_47091

theorem remainder_problem (P D Q R Q' R' : ℕ) 
  (h1 : P = Q * D + 2 * R)
  (h2 : Q = 2 * D * Q' + R') :
  P % (2 * D^2) = D * R' + 2 * R := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l470_47091


namespace NUMINAMATH_CALUDE_quadratic_with_odd_coeff_no_rational_roots_l470_47075

theorem quadratic_with_odd_coeff_no_rational_roots (a b c : ℤ) :
  Odd a → Odd b → Odd c → ¬ IsSquare (b^2 - 4*a*c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_odd_coeff_no_rational_roots_l470_47075


namespace NUMINAMATH_CALUDE_magnitude_of_z_l470_47006

open Complex

theorem magnitude_of_z (z : ℂ) (h : i * (1 - z) = 1) : abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l470_47006


namespace NUMINAMATH_CALUDE_f_2_nonneg_necessary_not_sufficient_l470_47030

/-- A quadratic function f(x) = ax^2 + bx -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

/-- f(x) is monotonically increasing on (1, +∞) -/
def monotonically_increasing_on_interval (a b : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f a b x < f a b y

/-- f(2) ≥ 0 is a necessary but not sufficient condition for
    f(x) to be monotonically increasing on (1, +∞) -/
theorem f_2_nonneg_necessary_not_sufficient (a b : ℝ) :
  (∀ a b, monotonically_increasing_on_interval a b → f a b 2 ≥ 0) ∧
  ¬(∀ a b, f a b 2 ≥ 0 → monotonically_increasing_on_interval a b) :=
by sorry

end NUMINAMATH_CALUDE_f_2_nonneg_necessary_not_sufficient_l470_47030


namespace NUMINAMATH_CALUDE_storybook_pages_l470_47061

/-- The number of days between two dates (inclusive) -/
def daysBetween (startDate endDate : Nat) : Nat :=
  endDate - startDate + 1

theorem storybook_pages : 
  let startDate := 10  -- March 10th
  let endDate := 20    -- March 20th
  let pagesPerDay := 11
  let readingDays := daysBetween startDate endDate
  readingDays * pagesPerDay = 121 := by
  sorry

end NUMINAMATH_CALUDE_storybook_pages_l470_47061


namespace NUMINAMATH_CALUDE_xy_plus_y_squared_l470_47016

theorem xy_plus_y_squared (x y : ℝ) (h : x * (x + y) = x^2 + y + 12) : 
  x * y + y^2 = y^2 + y + 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_plus_y_squared_l470_47016


namespace NUMINAMATH_CALUDE_log_inequality_l470_47072

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + Real.sqrt x) < Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l470_47072


namespace NUMINAMATH_CALUDE_joan_gave_63_seashells_l470_47095

/-- The number of seashells Joan gave to Mike -/
def seashells_given_to_mike (initial_seashells : ℕ) (remaining_seashells : ℕ) : ℕ :=
  initial_seashells - remaining_seashells

/-- Theorem: Joan gave 63 seashells to Mike -/
theorem joan_gave_63_seashells :
  seashells_given_to_mike 79 16 = 63 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_63_seashells_l470_47095


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l470_47004

-- Define an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_seq_sum (a : ℕ → ℝ) :
  is_arithmetic_seq a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l470_47004


namespace NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l470_47077

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h2 : a 2 = 9) (h5 : a 5 = 33) : 
  ∃ d : ℝ, d = 8 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l470_47077


namespace NUMINAMATH_CALUDE_max_min_difference_is_five_l470_47058

/-- Given non-zero real numbers a and b satisfying a² + b² = 25,
    prove that the difference between the maximum and minimum values
    of the function y = (ax + b) / (x² + 1) is 5. -/
theorem max_min_difference_is_five (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 + b^2 = 25) :
  let f : ℝ → ℝ := λ x => (a * x + b) / (x^2 + 1)
  ∃ y₁ y₂ : ℝ, (∀ x, f x ≤ y₁) ∧ (∀ x, f x ≥ y₂) ∧ y₁ - y₂ = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_is_five_l470_47058


namespace NUMINAMATH_CALUDE_max_multicolored_sets_l470_47097

/-- A color distribution is a list of positive integers representing the number of points of each color. -/
def ColorDistribution := List Nat

/-- The number of multi-colored sets for a given color distribution. -/
def multiColoredSets (d : ColorDistribution) : Nat :=
  d.prod

/-- Predicate to check if a color distribution is valid for the problem. -/
def isValidDistribution (d : ColorDistribution) : Prop :=
  d.length > 0 ∧ 
  d.sum = 2012 ∧ 
  d.Nodup ∧
  d.all (· > 0)

/-- The theorem stating that 61 colors maximize the number of multi-colored sets. -/
theorem max_multicolored_sets : 
  ∃ (d : ColorDistribution), isValidDistribution d ∧ d.length = 61 ∧
  ∀ (d' : ColorDistribution), isValidDistribution d' → d'.length ≠ 61 → 
    multiColoredSets d ≥ multiColoredSets d' :=
  sorry

end NUMINAMATH_CALUDE_max_multicolored_sets_l470_47097


namespace NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_two_l470_47076

theorem inequality_holds_for_p_greater_than_two (p q : ℝ) 
  (hp : p > 2) (hq : q > 0) : 
  4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 4 * p * q) / (p + q) > 3 * p^3 * q := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_two_l470_47076


namespace NUMINAMATH_CALUDE_remainder_6_pow_23_mod_5_l470_47059

theorem remainder_6_pow_23_mod_5 : 6^23 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6_pow_23_mod_5_l470_47059


namespace NUMINAMATH_CALUDE_find_special_number_l470_47038

theorem find_special_number : 
  ∃ n : ℕ, 
    (∃ k : ℕ, 3 * n = 2 * k + 1) ∧ 
    (∃ m : ℕ, 3 * n = 9 * m) ∧ 
    (∀ x : ℕ, x < n → ¬((∃ k : ℕ, 3 * x = 2 * k + 1) ∧ (∃ m : ℕ, 3 * x = 9 * m))) :=
by sorry

end NUMINAMATH_CALUDE_find_special_number_l470_47038


namespace NUMINAMATH_CALUDE_total_dolls_l470_47098

theorem total_dolls (hannah_ratio : ℝ) (sister_dolls : ℝ) : 
  hannah_ratio = 5.5 →
  sister_dolls = 8.5 →
  hannah_ratio * sister_dolls + sister_dolls = 55.25 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l470_47098


namespace NUMINAMATH_CALUDE_teacher_group_arrangements_l470_47087

theorem teacher_group_arrangements : 
  let total_female : ℕ := 2
  let total_male : ℕ := 4
  let groups : ℕ := 2
  let female_per_group : ℕ := 1
  let male_per_group : ℕ := 2
  Nat.choose total_female female_per_group * Nat.choose total_male male_per_group = 12 :=
by sorry

end NUMINAMATH_CALUDE_teacher_group_arrangements_l470_47087


namespace NUMINAMATH_CALUDE_snow_cone_stand_problem_l470_47086

/-- Represents the snow-cone stand financial problem --/
theorem snow_cone_stand_problem 
  (borrowed : ℝ)  -- Amount borrowed from brother
  (repay : ℝ)     -- Amount to repay brother
  (ingredients : ℝ) -- Cost of ingredients
  (sold : ℕ)      -- Number of snow cones sold
  (price : ℝ)     -- Price per snow cone
  (remaining : ℝ) -- Amount remaining after repayment
  (h1 : repay = 110)
  (h2 : ingredients = 75)
  (h3 : sold = 200)
  (h4 : price = 0.75)
  (h5 : remaining = 65)
  (h6 : sold * price = borrowed + remaining - ingredients) :
  borrowed = 250 := by
  sorry

end NUMINAMATH_CALUDE_snow_cone_stand_problem_l470_47086


namespace NUMINAMATH_CALUDE_no_m_for_all_x_x_range_for_bounded_m_l470_47005

-- Define the inequality function
def f (m x : ℝ) : ℝ := m * x^2 - 2*x - m + 1

-- Statement 1
theorem no_m_for_all_x : ¬ ∃ m : ℝ, ∀ x : ℝ, f m x < 0 := by sorry

-- Statement 2
theorem x_range_for_bounded_m :
  ∀ m : ℝ, |m| ≤ 2 →
  ∀ x : ℝ, ((-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2) →
  f m x < 0 := by sorry

end NUMINAMATH_CALUDE_no_m_for_all_x_x_range_for_bounded_m_l470_47005


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l470_47009

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := -3
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l470_47009


namespace NUMINAMATH_CALUDE_trapezoid_area_l470_47054

/-- The area of a trapezoid with bases 3h and 5h, and height h, is equal to 4h² -/
theorem trapezoid_area (h : ℝ) : h * ((3 * h + 5 * h) / 2) = 4 * h^2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l470_47054


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l470_47020

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l470_47020


namespace NUMINAMATH_CALUDE_daylight_rice_yield_related_l470_47085

-- Define the concept of related variables
def are_related (x y : Type) : Prop := 
  ¬(∃ f : x → y, Function.Injective f ∧ Function.Surjective f) ∧ 
  ∃ (f : x → y), ∀ (a b : x), a ≠ b → f a ≠ f b

-- Define the variables
def edge_length : Type := Real
def cube_volume : Type := Real
def angle_radian : Type := Real
def sine_value : Type := Real
def daylight_duration : Type := Real
def rice_yield : Type := Real
def person_height : Type := Real
def eyesight : Type := Real

-- State the theorem
theorem daylight_rice_yield_related :
  (¬ are_related edge_length cube_volume) ∧
  (¬ are_related angle_radian sine_value) ∧
  (are_related daylight_duration rice_yield) ∧
  (¬ are_related person_height eyesight) :=
by sorry

end NUMINAMATH_CALUDE_daylight_rice_yield_related_l470_47085


namespace NUMINAMATH_CALUDE_max_k_value_l470_47066

theorem max_k_value (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a + b + c = a * b + b * c + c * a) :
  ∃ k : ℝ, k = 1 ∧ 
  ∀ k' : ℝ, 
    ((a + b + c) * ((1 / (a + b)) + (1 / (c + b)) + (1 / (a + c)) - k') ≥ k') → 
    k' ≤ k :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l470_47066


namespace NUMINAMATH_CALUDE_trapezoid_median_equals_12_l470_47067

/-- Given a triangle and a trapezoid with equal areas and altitudes, where the triangle's base is 24 inches and one base of the trapezoid is twice the other, prove the trapezoid's median is 12 inches. -/
theorem trapezoid_median_equals_12 (h : ℝ) (x : ℝ) : 
  h > 0 →  -- Altitude is positive
  (1/2) * 24 * h = ((x + 2*x) / 2) * h →  -- Equal areas
  (x + 2*x) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_median_equals_12_l470_47067


namespace NUMINAMATH_CALUDE_sum_of_roots_l470_47001

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x, x^2 - 12*a*x - 13*b = 0 ↔ x = c ∨ x = d) →
  (∀ x, x^2 - 12*c*x - 13*d = 0 ↔ x = a ∨ x = b) →
  a + b + c + d = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l470_47001


namespace NUMINAMATH_CALUDE_combined_exterior_angles_pentagon_hexagon_l470_47039

-- Define the sum of exterior angles for any convex polygon
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Define a pentagon
def pentagon : ℕ := 5

-- Define a hexagon
def hexagon : ℕ := 6

-- Theorem statement
theorem combined_exterior_angles_pentagon_hexagon :
  sum_exterior_angles pentagon + sum_exterior_angles hexagon = 720 := by
  sorry

end NUMINAMATH_CALUDE_combined_exterior_angles_pentagon_hexagon_l470_47039


namespace NUMINAMATH_CALUDE_executive_committee_selection_l470_47036

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem executive_committee_selection (total_members senior_members committee_size : ℕ) 
  (h1 : total_members = 30)
  (h2 : senior_members = 10)
  (h3 : committee_size = 5) :
  (choose senior_members 2 * choose (total_members - senior_members) 3 +
   choose senior_members 3 * choose (total_members - senior_members) 2 +
   choose senior_members 4 * choose (total_members - senior_members) 1 +
   choose senior_members 5) = 78552 := by
  sorry

end NUMINAMATH_CALUDE_executive_committee_selection_l470_47036


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l470_47012

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → 2 / a < 1) ∧
  (∃ a, 2 / a < 1 ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l470_47012


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l470_47046

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) 
  (h2 : n2 = 50) 
  (h3 : avg1 = 30) 
  (h4 : avg2 = 60) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 48.75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l470_47046


namespace NUMINAMATH_CALUDE_hyperbola_specific_equation_l470_47028

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The general equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The focus of a hyperbola -/
def focus (x y : ℝ) : Prop := x = 2 ∧ y = 0

/-- The asymptotes of a hyperbola -/
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The theorem stating the specific equation of the hyperbola given the conditions -/
theorem hyperbola_specific_equation (a b : ℝ) (h : Hyperbola a b) 
  (focus_cond : focus 2 0)
  (asymp_cond : ∀ x y, asymptotes x y ↔ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :
  ∀ x y, hyperbola_equation h x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_specific_equation_l470_47028


namespace NUMINAMATH_CALUDE_student_percentage_theorem_l470_47023

theorem student_percentage_theorem (total : ℝ) (third_year_percent : ℝ) (second_year_fraction : ℝ)
  (h1 : third_year_percent = 50)
  (h2 : second_year_fraction = 2/3)
  (h3 : total > 0) :
  let non_third_year := total - (third_year_percent / 100) * total
  let second_year := second_year_fraction * non_third_year
  (total - second_year) / total * 100 = 66.66666666666667 :=
sorry

end NUMINAMATH_CALUDE_student_percentage_theorem_l470_47023


namespace NUMINAMATH_CALUDE_total_nails_l470_47048

/-- The number of nails each person has -/
structure NailCount where
  violet : ℕ
  tickletoe : ℕ
  sillysocks : ℕ

/-- The conditions of the nail counting problem -/
def nail_conditions (n : NailCount) : Prop :=
  n.violet = 2 * n.tickletoe + 3 ∧
  n.sillysocks = 3 * n.tickletoe - 2 ∧
  3 * n.tickletoe = 2 * n.violet ∧
  4 * n.tickletoe = 3 * n.sillysocks ∧
  n.violet = 27

/-- The theorem stating the total number of nails -/
theorem total_nails (n : NailCount) (h : nail_conditions n) : 
  n.violet + n.tickletoe + n.sillysocks = 73 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_l470_47048


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplification_l470_47082

theorem sum_of_fractions_simplification 
  (p q r : ℝ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) 
  (h_sum : p + q + r = 1) :
  1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2*q*r) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplification_l470_47082


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l470_47092

/-- Represents the investment of a partner -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the capital-time product of an investment -/
def capitalTimeProduct (i : Investment) : ℕ :=
  i.amount * i.duration

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem profit_ratio_theorem (a b : Investment) 
    (h1 : a.amount = 36000) (h2 : a.duration = 12)
    (h3 : b.amount = 54000) (h4 : b.duration = 4) :
    Ratio.mk (capitalTimeProduct a) (capitalTimeProduct b) = Ratio.mk 2 1 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_theorem_l470_47092


namespace NUMINAMATH_CALUDE_first_term_exceeding_thousand_l470_47044

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Predicate to check if a term exceeds 1000 -/
def exceedsThousand (x : ℝ) : Prop :=
  x > 1000

theorem first_term_exceeding_thousand :
  let a₁ := 2
  let d := 3
  (∀ n < 334, ¬(exceedsThousand (arithmeticSequenceTerm a₁ d n))) ∧
  exceedsThousand (arithmeticSequenceTerm a₁ d 334) :=
by sorry

end NUMINAMATH_CALUDE_first_term_exceeding_thousand_l470_47044


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l470_47074

/-- The increase in radius and height of a cylinder that results in quadrupling its volume -/
theorem cylinder_volume_increase (x : ℝ) : x > 0 →
  π * (10 + x)^2 * (5 + x) = 4 * (π * 10^2 * 5) →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l470_47074


namespace NUMINAMATH_CALUDE_elizabeth_money_l470_47040

/-- The amount of money Elizabeth has, given the costs of pens and pencils and the relationship between the number of pens and pencils she can buy. -/
theorem elizabeth_money : 
  let pencil_cost : ℚ := 8/5  -- $1.60 expressed as a rational number
  let pen_cost : ℚ := 2       -- $2.00
  let pencil_count : ℕ := 5   -- Number of pencils
  let pen_count : ℕ := 6      -- Number of pens
  (pencil_cost * pencil_count + pen_cost * pen_count : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_money_l470_47040


namespace NUMINAMATH_CALUDE_distance_for_boy_problem_l470_47081

/-- Calculates the distance traveled given time in minutes and speed in meters per second -/
def distance_traveled (time_minutes : ℕ) (speed_meters_per_second : ℕ) : ℕ :=
  time_minutes * 60 * speed_meters_per_second

/-- Theorem: Given 36 minutes and a speed of 4 meters per second, the distance traveled is 8640 meters -/
theorem distance_for_boy_problem : distance_traveled 36 4 = 8640 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_boy_problem_l470_47081


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l470_47084

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a ∧ a < 2) (hb : 1 < b ∧ b < 4) :
  -19 < 3 * a - b ∧ 3 * a - b < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l470_47084


namespace NUMINAMATH_CALUDE_distance_between_tangent_circles_l470_47003

/-- The distance between the centers of two externally tangent circles -/
def distance_between_centers (r1 r2 : ℝ) : ℝ := r1 + r2

/-- Two circles are externally tangent -/
axiom externally_tangent (O O' : Set ℝ) : Prop

theorem distance_between_tangent_circles 
  (O O' : Set ℝ) (r1 r2 : ℝ) 
  (h1 : externally_tangent O O')
  (h2 : r1 = 8)
  (h3 : r2 = 3) : 
  distance_between_centers r1 r2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_tangent_circles_l470_47003


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l470_47065

theorem sufficient_not_necessary (a b : ℝ) :
  (b > a ∧ a > 0 → (a + 2) / (b + 2) > a / b) ∧
  ∃ a b : ℝ, (a + 2) / (b + 2) > a / b ∧ ¬(b > a ∧ a > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l470_47065
