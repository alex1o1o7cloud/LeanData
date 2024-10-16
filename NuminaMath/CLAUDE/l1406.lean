import Mathlib

namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l1406_140691

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, |2*y + 9| ≤ 20 → y ≥ -14) ∧ |2*(-14) + 9| ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l1406_140691


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1406_140620

/-- The distance between the vertices of a hyperbola -/
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 16 = 1

theorem hyperbola_vertices_distance :
  ∃ (a : ℝ), a^2 = 36 ∧ distance_between_vertices a = 12 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1406_140620


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l1406_140684

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 2.5) : x^2 + (1 / x^2) = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l1406_140684


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l1406_140631

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (a b : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : parallel a α) 
  (h4 : perpendicular a β) : 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l1406_140631


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l1406_140662

theorem no_function_satisfies_inequality :
  ¬∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f x + f y ≥ 2 * f ((x + y) / 2) + 2 * |x - y| := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l1406_140662


namespace NUMINAMATH_CALUDE_complex_number_problem_l1406_140688

theorem complex_number_problem (z : ℂ) 
  (h1 : z.re > 0) 
  (h2 : Complex.abs z = 2 * Real.sqrt 5) 
  (h3 : (Complex.I + 2) * z = Complex.I * (Complex.I * z).im) 
  (h4 : ∃ (m n : ℝ), z^2 + m*z + n = 0) : 
  z = 4 + 2*Complex.I ∧ 
  ∃ (m n : ℝ), z^2 + m*z + n = 0 ∧ m = -8 ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1406_140688


namespace NUMINAMATH_CALUDE_hundredth_row_sum_l1406_140676

def triangular_array_sum (n : ℕ) : ℕ :=
  2^(n+1) - 4

theorem hundredth_row_sum : 
  triangular_array_sum 100 = 2^101 - 4 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_row_sum_l1406_140676


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1406_140667

theorem x_minus_y_value (x y : ℝ) 
  (h1 : |x| = 3)
  (h2 : y^2 = 1/4)
  (h3 : x + y < 0) :
  x - y = -7/2 ∨ x - y = -5/2 :=
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1406_140667


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1406_140647

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1406_140647


namespace NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l1406_140615

/-- Given an angle α = 7π/5, prove that its terminal side is located in the third quadrant. -/
theorem terminal_side_in_third_quadrant (α : Real) (h : α = 7 * Real.pi / 5) :
  ∃ (x y : Real), x < 0 ∧ y < 0 ∧ (∃ (t : Real), x = t * Real.cos α ∧ y = t * Real.sin α) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l1406_140615


namespace NUMINAMATH_CALUDE_max_two_digit_decimals_l1406_140680

def digits : List Nat := [2, 0, 5]

def is_valid_two_digit_decimal (n : Nat) (d : Nat) : Bool :=
  n ∈ digits ∧ d ∈ digits ∧ (n ≠ 0 ∨ d ≠ 0)

def count_valid_decimals : Nat :=
  (List.filter (fun (pair : Nat × Nat) => is_valid_two_digit_decimal pair.1 pair.2)
    (List.product digits digits)).length

theorem max_two_digit_decimals : count_valid_decimals = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_two_digit_decimals_l1406_140680


namespace NUMINAMATH_CALUDE_integer_root_count_theorem_l1406_140656

/-- A polynomial of degree 5 with integer coefficients -/
structure IntPolynomial5 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The number of integer roots of a polynomial, counting multiplicity -/
def numIntegerRoots (p : IntPolynomial5) : ℕ := sorry

/-- The set of possible values for the number of integer roots -/
def possibleRootCounts : Set ℕ := {0, 1, 2, 3, 5}

/-- Theorem: The number of integer roots of a degree 5 polynomial 
    with integer coefficients is always in the set {0, 1, 2, 3, 5} -/
theorem integer_root_count_theorem (p : IntPolynomial5) : 
  numIntegerRoots p ∈ possibleRootCounts := by sorry

end NUMINAMATH_CALUDE_integer_root_count_theorem_l1406_140656


namespace NUMINAMATH_CALUDE_is_center_of_symmetry_l1406_140628

/-- The function f(x) = (x+2)³ - x + 1 -/
def f (x : ℝ) : ℝ := (x + 2)^3 - x + 1

/-- The center of symmetry for the function f -/
def center_of_symmetry : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the given point is the center of symmetry for f -/
theorem is_center_of_symmetry :
  ∀ x : ℝ, f (center_of_symmetry.1 + x) + f (center_of_symmetry.1 - x) = 2 * center_of_symmetry.2 :=
sorry

end NUMINAMATH_CALUDE_is_center_of_symmetry_l1406_140628


namespace NUMINAMATH_CALUDE_tangent_ball_prism_area_relation_l1406_140685

/-- A quadrangular prism with a small ball tangent to each face -/
structure TangentBallPrism where
  S₁ : ℝ  -- Area of the upper base
  S₂ : ℝ  -- Area of the lower base
  S : ℝ   -- Lateral surface area
  h₁ : 0 < S₁  -- S₁ is positive
  h₂ : 0 < S₂  -- S₂ is positive
  h₃ : 0 < S   -- S is positive

/-- The relationship between the lateral surface area and the base areas -/
theorem tangent_ball_prism_area_relation (p : TangentBallPrism) : 
  Real.sqrt p.S = Real.sqrt p.S₁ + Real.sqrt p.S₂ := by
  sorry

end NUMINAMATH_CALUDE_tangent_ball_prism_area_relation_l1406_140685


namespace NUMINAMATH_CALUDE_train_length_l1406_140674

/-- Calculates the length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 36 → 
  time_s = 9.99920006399488 → 
  (speed_kmh * 1000 / 3600) * time_s = 99.9920006399488 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1406_140674


namespace NUMINAMATH_CALUDE_lcm_18_30_l1406_140613

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l1406_140613


namespace NUMINAMATH_CALUDE_T_divisibility_l1406_140681

def T : Set ℤ := {t | ∃ n : ℤ, t = n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2}

theorem T_divisibility :
  (∀ t ∈ T, ¬(9 ∣ t)) ∧ (∃ t ∈ T, 5 ∣ t) := by
  sorry

end NUMINAMATH_CALUDE_T_divisibility_l1406_140681


namespace NUMINAMATH_CALUDE_badminton_medals_count_l1406_140632

theorem badminton_medals_count (total_medals : ℕ) (track_medals : ℕ) : 
  total_medals = 20 →
  track_medals = 5 →
  total_medals = track_medals + 2 * track_medals + (total_medals - track_medals - 2 * track_medals) →
  (total_medals - track_medals - 2 * track_medals) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_badminton_medals_count_l1406_140632


namespace NUMINAMATH_CALUDE_triangle_shape_l1406_140660

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition
def satisfiesCondition (t : Triangle) : Prop :=
  (Real.cos t.A) / (Real.cos t.B) = t.b / t.a

-- Define isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- Define right triangle
def isRight (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_shape (t : Triangle) :
  satisfiesCondition t → isIsosceles t ∨ isRight t :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_shape_l1406_140660


namespace NUMINAMATH_CALUDE_apples_to_eat_raw_l1406_140665

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 →
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - (wormy + bruised) = 42 := by
sorry

end NUMINAMATH_CALUDE_apples_to_eat_raw_l1406_140665


namespace NUMINAMATH_CALUDE_l_shape_area_is_58_l1406_140644

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the "L" shaped figure -/
structure LShape where
  outerRectangle : Rectangle
  innerRectangle : Rectangle

/-- Calculates the area of the "L" shaped figure -/
def lShapeArea (l : LShape) : ℝ :=
  rectangleArea l.outerRectangle - rectangleArea l.innerRectangle

/-- Theorem: The area of the "L" shaped figure is 58 square units -/
theorem l_shape_area_is_58 :
  let outer := Rectangle.mk 10 7
  let inner := Rectangle.mk 4 3
  let l := LShape.mk outer inner
  lShapeArea l = 58 := by sorry

end NUMINAMATH_CALUDE_l_shape_area_is_58_l1406_140644


namespace NUMINAMATH_CALUDE_new_average_weight_l1406_140643

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 19 → 
  initial_average = 15 → 
  new_student_weight = 3 → 
  (initial_students * initial_average + new_student_weight) / (initial_students + 1) = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l1406_140643


namespace NUMINAMATH_CALUDE_octahedron_triangle_count_l1406_140645

/-- The number of vertices in a regular octahedron -/
def octahedron_vertices : ℕ := 6

/-- The number of distinct triangles that can be formed by connecting three different vertices of a regular octahedron -/
def octahedron_triangles : ℕ := Nat.choose octahedron_vertices 3

theorem octahedron_triangle_count : octahedron_triangles = 20 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_triangle_count_l1406_140645


namespace NUMINAMATH_CALUDE_paper_products_pallets_l1406_140697

theorem paper_products_pallets : ∃ P : ℚ,
  P / 2 + P / 4 + P / 5 + 1 = P ∧ P = 20 := by
  sorry

end NUMINAMATH_CALUDE_paper_products_pallets_l1406_140697


namespace NUMINAMATH_CALUDE_problem_solution_l1406_140678

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 3

theorem problem_solution :
  (∀ x : ℝ, |g x| < 5 ↔ x ∈ Set.Ioo (-1) 3) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
    a ∈ Set.Iic (-6) ∪ Set.Ici 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1406_140678


namespace NUMINAMATH_CALUDE_three_digit_reversal_difference_l1406_140664

theorem three_digit_reversal_difference (A B C : ℕ) 
  (h1 : A ≠ C) 
  (h2 : A ≥ 1 ∧ A ≤ 9) 
  (h3 : B ≥ 0 ∧ B ≤ 9) 
  (h4 : C ≥ 0 ∧ C ≤ 9) : 
  ∃ k : ℤ, (100 * A + 10 * B + C) - (100 * C + 10 * B + A) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_reversal_difference_l1406_140664


namespace NUMINAMATH_CALUDE_apple_purchase_cost_l1406_140605

/-- The price of apples per kilogram before discount -/
def original_price : ℝ := 5

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.4

/-- The quantity of apples in kilograms -/
def quantity : ℝ := 10

/-- Calculates the discounted price per kilogram -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- Calculates the total cost for the given quantity of apples -/
def total_cost : ℝ := discounted_price * quantity

/-- Theorem stating that the total cost for 10 kilograms of apples with a 40% discount is $30 -/
theorem apple_purchase_cost : total_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_cost_l1406_140605


namespace NUMINAMATH_CALUDE_square_with_inscribed_semicircles_l1406_140607

theorem square_with_inscribed_semicircles (square_side : ℝ) (semicircle_count : ℕ) : 
  square_side = 4 → 
  semicircle_count = 4 → 
  (square_side^2 - semicircle_count * (π * (square_side/2)^2 / 2)) = 16 - 8*π := by
sorry

end NUMINAMATH_CALUDE_square_with_inscribed_semicircles_l1406_140607


namespace NUMINAMATH_CALUDE_max_distance_le_150cm_l1406_140635

/-- Represents the extended table with two semicircles and a rectangular section -/
structure ExtendedTable where
  semicircle_diameter : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The maximum distance between any two points on the extended table -/
def max_distance (table : ExtendedTable) : ℝ :=
  sorry

/-- Theorem stating that the maximum distance between any two points on the extended table
    is less than or equal to 150 cm -/
theorem max_distance_le_150cm (table : ExtendedTable)
  (h1 : table.semicircle_diameter = 1)
  (h2 : table.rectangle_length = 1)
  (h3 : table.rectangle_width = 0.5) :
  max_distance table ≤ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_le_150cm_l1406_140635


namespace NUMINAMATH_CALUDE_largest_special_number_last_digit_l1406_140626

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_three_digits (n : ℕ) : ℕ := n / 10

theorem largest_special_number_last_digit :
  ∃ (n : ℕ), is_four_digit n ∧ 
             n % 9 = 0 ∧ 
             (first_three_digits n) % 4 = 0 ∧
             ∀ (m : ℕ), (is_four_digit m ∧ 
                         m % 9 = 0 ∧ 
                         (first_three_digits m) % 4 = 0) → 
                         m ≤ n ∧
             n % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_special_number_last_digit_l1406_140626


namespace NUMINAMATH_CALUDE_floor_sum_difference_l1406_140659

theorem floor_sum_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ⌊a + b⌋ - (⌊a⌋ + ⌊b⌋) = 0 ∨ ⌊a + b⌋ - (⌊a⌋ + ⌊b⌋) = 1 :=
by sorry

end NUMINAMATH_CALUDE_floor_sum_difference_l1406_140659


namespace NUMINAMATH_CALUDE_original_number_proof_l1406_140670

theorem original_number_proof (increased_number : ℝ) (increase_percentage : ℝ) :
  increased_number = 480 ∧ increase_percentage = 0.2 →
  (1 + increase_percentage) * (increased_number / (1 + increase_percentage)) = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1406_140670


namespace NUMINAMATH_CALUDE_pear_sales_ratio_l1406_140682

/-- Given the total amount of pears sold in a day and the amount sold in the afternoon,
    prove that the ratio of pears sold in the afternoon to pears sold in the morning is 2:1. -/
theorem pear_sales_ratio (total : ℕ) (afternoon : ℕ) 
    (h1 : total = 510)
    (h2 : afternoon = 340) : 
  (afternoon : ℚ) / ((total - afternoon) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pear_sales_ratio_l1406_140682


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1406_140671

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 16*x - 4) → (∃ y : ℝ, y^2 = 16*y - 4 ∧ x + y = 16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1406_140671


namespace NUMINAMATH_CALUDE_system_solution_sum_l1406_140661

theorem system_solution_sum (a b : ℝ) : 
  (a * 1 + b * 2 = 4 ∧ b * 1 - a * 2 = 7) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_sum_l1406_140661


namespace NUMINAMATH_CALUDE_power_multiplication_l1406_140651

theorem power_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1406_140651


namespace NUMINAMATH_CALUDE_train_journey_time_l1406_140603

theorem train_journey_time 
  (D : ℝ) -- Distance in km
  (T : ℝ) -- Original time in hours
  (h1 : D = 48 * T) -- Distance equation for original journey
  (h2 : D = 60 * (40 / 60)) -- Distance equation for faster journey
  : T * 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l1406_140603


namespace NUMINAMATH_CALUDE_area_triangle_ADC_l1406_140609

/-- Given two triangles ABD and ADC sharing a common height,
    prove that the area of triangle ADC is 18 square centimeters. -/
theorem area_triangle_ADC (BD DC : ℝ) (area_ABD : ℝ) :
  BD / DC = 4 / 3 →
  area_ABD = 24 →
  ∃ (area_ADC : ℝ), area_ADC = 18 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_ADC_l1406_140609


namespace NUMINAMATH_CALUDE_baby_grasshoppers_count_l1406_140639

/-- The number of grasshoppers Jemma found on the African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := 31

/-- The number of baby grasshoppers under the plant -/
def baby_grasshoppers : ℕ := total_grasshoppers - grasshoppers_on_plant

theorem baby_grasshoppers_count : baby_grasshoppers = 24 := by
  sorry

end NUMINAMATH_CALUDE_baby_grasshoppers_count_l1406_140639


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l1406_140618

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1| + 2 * |x + 2|

-- Theorem for part (I)
theorem f_lower_bound : ∀ x : ℝ, f x ≥ 5 := by sorry

-- Theorem for part (II)
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, 15 - 2 * f x < a^2 + 9 / (a^2 + 1)) ↔ a ≠ Real.sqrt 2 ∧ a ≠ -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l1406_140618


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1406_140687

theorem trigonometric_equation_solution (t : ℝ) : 
  (Real.sin (2 * t))^6 + (Real.cos (2 * t))^6 = 
    3/2 * ((Real.sin (2 * t))^4 + (Real.cos (2 * t))^4) + 1/2 * (Real.sin t + Real.cos t) ↔ 
  (∃ k : ℤ, t = π * (2 * k + 1)) ∨ 
  (∃ n : ℤ, t = π/2 * (4 * n - 1)) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1406_140687


namespace NUMINAMATH_CALUDE_ophelia_age_proof_l1406_140696

/-- Represents the current year -/
def currentYear : ℕ := 2022

/-- Represents the future year when ages are compared -/
def futureYear : ℕ := 2030

/-- Represents the current age of Lennon -/
def lennonAge : ℝ := 15 - (futureYear - currentYear)

/-- Represents the current age of Mike -/
def mikeAge : ℝ := lennonAge + 5

/-- Represents the current age of Ophelia -/
def opheliaAge : ℝ := 20.5

theorem ophelia_age_proof :
  /- In 15 years, Ophelia will be 3.5 times as old as Lennon -/
  opheliaAge + 15 = 3.5 * (lennonAge + 15) ∧
  /- In 15 years, Mike will be twice as old as the age difference between Ophelia and Lennon -/
  mikeAge + 15 = 2 * (opheliaAge - lennonAge) ∧
  /- In 15 years, JB will be 0.75 times as old as the sum of Ophelia's and Lennon's age -/
  mikeAge + 15 = 0.75 * (opheliaAge + lennonAge + 30) :=
by sorry

end NUMINAMATH_CALUDE_ophelia_age_proof_l1406_140696


namespace NUMINAMATH_CALUDE_trajectory_of_point_M_l1406_140673

/-- The trajectory of point M satisfying the given distance conditions -/
theorem trajectory_of_point_M (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ - 0)^2 + (y₀ - 4)^2 = (abs (y₀ + 5) - 1)^2 → x₀^2 = 16 * y₀) →
  x^2 + (y - 4)^2 = (abs (y + 5) - 1)^2 →
  x^2 = 16 * y := by
  sorry


end NUMINAMATH_CALUDE_trajectory_of_point_M_l1406_140673


namespace NUMINAMATH_CALUDE_current_calculation_l1406_140657

-- Define the variables and their types
variable (Q I R t : ℝ)

-- Define the theorem
theorem current_calculation 
  (heat_equation : Q = I^2 * R * t)
  (resistance : R = 5)
  (heat_generated : Q = 30)
  (time : t = 1) :
  I = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_current_calculation_l1406_140657


namespace NUMINAMATH_CALUDE_platform_completion_time_l1406_140612

/-- Represents the number of days required to complete a portion of a project given a number of workers -/
def days_to_complete (workers : ℕ) (portion : ℚ) : ℚ :=
  sorry

theorem platform_completion_time :
  let initial_workers : ℕ := 90
  let initial_days : ℕ := 6
  let initial_portion : ℚ := 1/2
  let remaining_workers : ℕ := 60
  let remaining_portion : ℚ := 1/2
  days_to_complete initial_workers initial_portion = initial_days →
  days_to_complete remaining_workers remaining_portion = 9 :=
by sorry

end NUMINAMATH_CALUDE_platform_completion_time_l1406_140612


namespace NUMINAMATH_CALUDE_max_weighing_ways_exact_89_ways_weighing_theorem_l1406_140694

/-- Represents the set of weights with masses 1, 2, 4, ..., 512 grams -/
def WeightSet : Set ℕ := {n : ℕ | ∃ k : ℕ, k ≤ 9 ∧ n = 2^k}

/-- Number of ways to weigh a load P using weights up to 2^n -/
def K (n : ℕ) (P : ℤ) : ℕ := sorry

/-- Maximum number of ways to weigh any load using weights up to 2^n -/
def MaxK (n : ℕ) : ℕ := sorry

/-- Theorem stating that no load can be weighed in more than 89 ways -/
theorem max_weighing_ways :
  ∀ P : ℤ, K 9 P ≤ 89 :=
sorry

/-- Theorem stating that 171 grams can be weighed in exactly 89 ways -/
theorem exact_89_ways :
  K 9 171 = 89 :=
sorry

/-- Main theorem combining both parts of the problem -/
theorem weighing_theorem :
  (∀ P : ℤ, K 9 P ≤ 89) ∧ (K 9 171 = 89) :=
sorry

end NUMINAMATH_CALUDE_max_weighing_ways_exact_89_ways_weighing_theorem_l1406_140694


namespace NUMINAMATH_CALUDE_park_walk_distance_l1406_140627

theorem park_walk_distance (area : ℝ) (π_approx : ℝ) (extra_distance : ℝ) : 
  area = 616 →
  π_approx = 22 / 7 →
  extra_distance = 3 →
  2 * π_approx * (area / π_approx).sqrt + extra_distance = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_park_walk_distance_l1406_140627


namespace NUMINAMATH_CALUDE_combined_earnings_l1406_140648

def dwayne_earnings : ℕ := 1500
def brady_extra : ℕ := 450

theorem combined_earnings :
  dwayne_earnings + (dwayne_earnings + brady_extra) = 3450 :=
by sorry

end NUMINAMATH_CALUDE_combined_earnings_l1406_140648


namespace NUMINAMATH_CALUDE_line_condition_perpendicular_to_x_axis_equal_intercepts_l1406_140646

-- Define the equation coefficients as functions of m
def a (m : ℝ) := m^2 - 2*m - 3
def b (m : ℝ) := 2*m^2 + m - 1
def c (m : ℝ) := 5 - 2*m

-- Theorem 1: Condition for the equation to represent a line
theorem line_condition (m : ℝ) : 
  (a m = 0 ∧ b m = 0) ↔ m = -1 :=
sorry

-- Theorem 2: Condition for the line to be perpendicular to x-axis
theorem perpendicular_to_x_axis (m : ℝ) :
  (a m ≠ 0 ∧ b m = 0) ↔ (m^2 - 2*m - 3 ≠ 0 ∧ 2*m^2 + m - 1 = 0) :=
sorry

-- Theorem 3: Condition for equal intercepts on both axes
theorem equal_intercepts (m : ℝ) :
  (m ≠ 5/2 → (2*m - 5)/(m^2 - 2*m - 3) = (2*m - 5)/(2*m^2 + m - 1)) ↔ m = 5/2 :=
sorry

end NUMINAMATH_CALUDE_line_condition_perpendicular_to_x_axis_equal_intercepts_l1406_140646


namespace NUMINAMATH_CALUDE_number_equation_solution_l1406_140675

theorem number_equation_solution : 
  ∃ x : ℝ, 5 * x + 4 = 19 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1406_140675


namespace NUMINAMATH_CALUDE_third_equals_sixth_implies_seven_odd_terms_sum_128_implies_eight_and_max_term_l1406_140611

-- For the first part of the problem
theorem third_equals_sixth_implies_seven (n : ℕ) :
  (Nat.choose n 2 = Nat.choose n 5) → n = 7 := by sorry

-- For the second part of the problem
theorem odd_terms_sum_128_implies_eight_and_max_term (n : ℕ) (x : ℝ) :
  (2^(n-1) = 128) →
  n = 8 ∧
  (Nat.choose 8 4 * x^4 * x^(2/3) = 70 * x^4 * x^(2/3)) := by sorry

end NUMINAMATH_CALUDE_third_equals_sixth_implies_seven_odd_terms_sum_128_implies_eight_and_max_term_l1406_140611


namespace NUMINAMATH_CALUDE_coin_distribution_theorem_l1406_140623

/-- Represents the coin distribution between Pete and Paul -/
def coin_distribution (x : ℕ) : Prop :=
  -- Paul's final coin count
  let paul_coins := x
  -- Pete's coin count using the sum formula
  let pete_coins := x * (x + 1) / 2
  -- The condition that Pete has 5 times as many coins as Paul
  pete_coins = 5 * paul_coins

/-- The total number of coins distributed -/
def total_coins (x : ℕ) : ℕ := 6 * x

theorem coin_distribution_theorem :
  ∃ x : ℕ, coin_distribution x ∧ total_coins x = 54 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_theorem_l1406_140623


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1406_140629

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 0 ∧ b = 2 ∧ c = 1) ∨ 
     (a = 2 ∧ b = 1 ∧ c = 0) ∨ 
     (a = 1 ∧ b = 0 ∧ c = 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1406_140629


namespace NUMINAMATH_CALUDE_solution_l1406_140624

def complex_number_problem (z : ℂ) : Prop :=
  (∃ (r : ℝ), z - 3 * Complex.I = r) ∧
  (∃ (t : ℝ), (z - 5 * Complex.I) / (2 - Complex.I) = t * Complex.I)

theorem solution (z : ℂ) (h : complex_number_problem z) :
  z = -1 + 3 * Complex.I ∧ Complex.abs (z / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_l1406_140624


namespace NUMINAMATH_CALUDE_triangle_area_solutions_l1406_140677

theorem triangle_area_solutions : 
  let vertex_A : ℝ × ℝ := (-5, 0)
  let vertex_B : ℝ × ℝ := (5, 0)
  let vertex_C (θ : ℝ) : ℝ × ℝ := (5 * Real.cos θ, 5 * Real.sin θ)
  let triangle_area (θ : ℝ) : ℝ := 
    abs ((vertex_B.1 - vertex_A.1) * (vertex_C θ).2 - (vertex_B.2 - vertex_A.2) * (vertex_C θ).1 - 
         (vertex_A.1 * vertex_B.2 - vertex_A.2 * vertex_B.1)) / 2
  ∃! (solutions : Finset ℝ), 
    (∀ θ ∈ solutions, 0 ≤ θ ∧ θ < 2 * Real.pi) ∧ 
    (∀ θ ∈ solutions, triangle_area θ = 10) ∧ 
    solutions.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_solutions_l1406_140677


namespace NUMINAMATH_CALUDE_third_year_students_l1406_140634

theorem third_year_students (total_first_year : ℕ) (total_selected : ℕ) (second_year_selected : ℕ) :
  total_first_year = 720 →
  total_selected = 180 →
  second_year_selected = 40 →
  ∃ (first_year_selected third_year_selected : ℕ),
    first_year_selected = (second_year_selected + third_year_selected) / 2 ∧
    first_year_selected + second_year_selected + third_year_selected = total_selected ∧
    (total_first_year * third_year_selected : ℚ) / first_year_selected = 960 :=
by sorry

end NUMINAMATH_CALUDE_third_year_students_l1406_140634


namespace NUMINAMATH_CALUDE_softball_team_savings_l1406_140621

/-- Calculates the savings for a softball team when buying uniforms with a group discount -/
theorem softball_team_savings 
  (regular_shirt_price regular_pants_price regular_socks_price : ℚ)
  (discounted_shirt_price discounted_pants_price discounted_socks_price : ℚ)
  (team_size : ℕ)
  (h_regular_shirt : regular_shirt_price = 7.5)
  (h_regular_pants : regular_pants_price = 15)
  (h_regular_socks : regular_socks_price = 4.5)
  (h_discounted_shirt : discounted_shirt_price = 6.75)
  (h_discounted_pants : discounted_pants_price = 13.5)
  (h_discounted_socks : discounted_socks_price = 3.75)
  (h_team_size : team_size = 12) :
  let regular_uniform_cost := regular_shirt_price + regular_pants_price + regular_socks_price
  let discounted_uniform_cost := discounted_shirt_price + discounted_pants_price + discounted_socks_price
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by sorry


end NUMINAMATH_CALUDE_softball_team_savings_l1406_140621


namespace NUMINAMATH_CALUDE_initial_fish_count_l1406_140686

theorem initial_fish_count (x : ℕ) : 
  x - 50 - (x - 50) / 3 + 200 = 300 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_fish_count_l1406_140686


namespace NUMINAMATH_CALUDE_geometric_sequence_103rd_term_l1406_140679

/-- Given a geometric sequence with first term a and common ratio r,
    this function returns the nth term of the sequence. -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

theorem geometric_sequence_103rd_term :
  let a : ℝ := 4
  let r : ℝ := -3
  geometric_sequence a r 103 = 4 * 3^102 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_103rd_term_l1406_140679


namespace NUMINAMATH_CALUDE_lilac_paint_mixture_l1406_140689

/-- Given a paint mixture where 70% is blue, 20% is red, and the rest is white,
    if 140 ounces of blue paint is added, then 20 ounces of white paint is added. -/
theorem lilac_paint_mixture (blue_percent : ℝ) (red_percent : ℝ) (blue_amount : ℝ) : 
  blue_percent = 0.7 →
  red_percent = 0.2 →
  blue_amount = 140 →
  let total_amount := blue_amount / blue_percent
  let white_percent := 1 - blue_percent - red_percent
  let white_amount := total_amount * white_percent
  white_amount = 20 := by
  sorry

end NUMINAMATH_CALUDE_lilac_paint_mixture_l1406_140689


namespace NUMINAMATH_CALUDE_difference_of_squares_l1406_140606

theorem difference_of_squares (a b : ℝ) : (a - b) * (-b - a) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1406_140606


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1406_140641

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1406_140641


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1406_140669

theorem polynomial_simplification (q : ℝ) :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1406_140669


namespace NUMINAMATH_CALUDE_f_properties_l1406_140604

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

theorem f_properties :
  (is_smallest_positive_period f π) ∧
  (∀ x, 1/2 ≤ f x ∧ f x ≤ 5/2) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-π/3 + k*π) (π/6 + k*π))) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1406_140604


namespace NUMINAMATH_CALUDE_solution_set_l1406_140636

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (3 - m, 1)

-- Define the condition for P being in the second quadrant
def is_in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Define the inequality
def inequality (m x : ℝ) : Prop := (2 - m) * x + m > 2

theorem solution_set (m : ℝ) : 
  is_in_second_quadrant (P m) → 
  (∀ x : ℝ, inequality m x ↔ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l1406_140636


namespace NUMINAMATH_CALUDE_black_haired_girls_count_l1406_140608

/-- Represents the choir composition --/
structure Choir where
  initial_total : ℕ
  added_blonde : ℕ
  initial_blonde : ℕ

/-- Calculates the number of black-haired girls in the choir --/
def black_haired_count (c : Choir) : ℕ :=
  (c.initial_total + c.added_blonde) - (c.initial_blonde + c.added_blonde)

/-- Theorem stating the number of black-haired girls in the choir --/
theorem black_haired_girls_count (c : Choir) 
  (h1 : c.initial_total = 80) 
  (h2 : c.added_blonde = 10) 
  (h3 : c.initial_blonde = 30) : 
  black_haired_count c = 50 := by
  sorry

#eval black_haired_count ⟨80, 10, 30⟩

end NUMINAMATH_CALUDE_black_haired_girls_count_l1406_140608


namespace NUMINAMATH_CALUDE_wife_cookie_percentage_l1406_140614

theorem wife_cookie_percentage (total_cookies : ℕ) (daughter_cookies : ℕ) (uneaten_cookies : ℕ) :
  total_cookies = 200 →
  daughter_cookies = 40 →
  uneaten_cookies = 50 →
  ∃ (wife_percentage : ℚ),
    wife_percentage = 30 ∧
    (total_cookies - (wife_percentage / 100) * total_cookies - daughter_cookies) / 2 = uneaten_cookies :=
by sorry

end NUMINAMATH_CALUDE_wife_cookie_percentage_l1406_140614


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l1406_140683

theorem smallest_number_with_remainder_two : ∃ n : ℕ,
  n > 2 ∧
  n % 9 = 2 ∧
  n % 10 = 2 ∧
  n % 11 = 2 ∧
  (∀ m : ℕ, m > 2 ∧ m % 9 = 2 ∧ m % 10 = 2 ∧ m % 11 = 2 → m ≥ n) ∧
  n = 992 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l1406_140683


namespace NUMINAMATH_CALUDE_complex_cube_root_l1406_140600

theorem complex_cube_root : ∃ (z : ℂ), z^2 + 2 = 0 ∧ (z^3 = 2 * Real.sqrt 2 * I ∨ z^3 = -2 * Real.sqrt 2 * I) := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l1406_140600


namespace NUMINAMATH_CALUDE_product_of_base_nine_digits_7654_l1406_140695

/-- Represents a number in base 9 as a list of digits -/
def BaseNineRepresentation := List Nat

/-- Converts a base 10 number to its base 9 representation -/
def toBaseNine (n : Nat) : BaseNineRepresentation :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List Nat) : Nat :=
  sorry

theorem product_of_base_nine_digits_7654 :
  productOfList (toBaseNine 7654) = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base_nine_digits_7654_l1406_140695


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_twelve_l1406_140640

/-- The number of ways to arrange 4 people in a row of 4 seats, 
    where 2 specific people must sit next to each other. -/
def seating_arrangements : ℕ := 12

/-- Theorem stating that the number of seating arrangements is 12. -/
theorem seating_arrangements_eq_twelve : seating_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_twelve_l1406_140640


namespace NUMINAMATH_CALUDE_range_of_a_l1406_140625

/-- Proposition p: For all real x, x²-2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: There exists a real x₀ such that x₀²+2ax₀+2-a=0 -/
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

/-- The range of a given the conditions on p and q -/
theorem range_of_a : ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1406_140625


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l1406_140617

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l1406_140617


namespace NUMINAMATH_CALUDE_tom_gave_cars_to_five_nephews_l1406_140649

/-- The number of nephews Tom gave cars to -/
def number_of_nephews : ℕ := by sorry

theorem tom_gave_cars_to_five_nephews :
  let packages := 10
  let cars_per_package := 5
  let total_cars := packages * cars_per_package
  let cars_left := 30
  let cars_given_away := total_cars - cars_left
  let fraction_per_nephew := 1 / 5
  number_of_nephews = (cars_given_away : ℚ) / (fraction_per_nephew * cars_given_away) := by sorry

end NUMINAMATH_CALUDE_tom_gave_cars_to_five_nephews_l1406_140649


namespace NUMINAMATH_CALUDE_camel_cost_l1406_140630

/-- The cost of animals in a zoo -/
structure AnimalCosts where
  camel : ℕ
  horse : ℕ
  ox : ℕ
  elephant : ℕ
  giraffe : ℕ
  zebra : ℕ

/-- The conditions given in the problem -/
def zoo_conditions (c : AnimalCosts) : Prop :=
  10 * c.camel = 24 * c.horse ∧
  16 * c.horse = 4 * c.ox ∧
  6 * c.ox = 4 * c.elephant ∧
  3 * c.elephant = 15 * c.giraffe ∧
  8 * c.giraffe = 20 * c.zebra ∧
  12 * c.elephant = 180000

theorem camel_cost (c : AnimalCosts) :
  zoo_conditions c → c.camel = 6000 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l1406_140630


namespace NUMINAMATH_CALUDE_verandah_width_is_two_l1406_140642

/-- Represents the dimensions of a rectangular room with a surrounding verandah. -/
structure RoomWithVerandah where
  roomLength : ℝ
  roomWidth : ℝ
  verandahWidth : ℝ

/-- Calculates the area of the verandah given the room dimensions. -/
def verandahArea (r : RoomWithVerandah) : ℝ :=
  (r.roomLength + 2 * r.verandahWidth) * (r.roomWidth + 2 * r.verandahWidth) - r.roomLength * r.roomWidth

/-- Theorem stating that for a room of 15m x 12m with a verandah of area 124 sq m, the verandah width is 2m. -/
theorem verandah_width_is_two :
  ∃ (r : RoomWithVerandah), r.roomLength = 15 ∧ r.roomWidth = 12 ∧ verandahArea r = 124 ∧ r.verandahWidth = 2 :=
by sorry

end NUMINAMATH_CALUDE_verandah_width_is_two_l1406_140642


namespace NUMINAMATH_CALUDE_carpet_area_in_sq_yards_l1406_140692

def living_room_length : ℝ := 18
def living_room_width : ℝ := 9
def storage_room_side : ℝ := 3
def sq_feet_per_sq_yard : ℝ := 9

theorem carpet_area_in_sq_yards :
  (living_room_length * living_room_width + storage_room_side * storage_room_side) / sq_feet_per_sq_yard = 19 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_in_sq_yards_l1406_140692


namespace NUMINAMATH_CALUDE_intersection_M_N_l1406_140610

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1406_140610


namespace NUMINAMATH_CALUDE_hydrogen_weight_in_H2CrO4_l1406_140638

def atomic_weight_H : ℝ := 1.008
def molecular_weight_H2CrO4 : ℝ := 118

theorem hydrogen_weight_in_H2CrO4 :
  let hydrogen_count : ℕ := 2
  let hydrogen_weight : ℝ := atomic_weight_H * hydrogen_count
  hydrogen_weight = 2.016 := by sorry

end NUMINAMATH_CALUDE_hydrogen_weight_in_H2CrO4_l1406_140638


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l1406_140658

/-- Given two circles in the plane, this theorem states that the line passing through their
    intersection points has a specific equation. -/
theorem intersection_line_of_circles
  (circle1 : Set (ℝ × ℝ))
  (circle2 : Set (ℝ × ℝ))
  (h1 : circle1 = {(x, y) | x^2 + y^2 - 4*x + 6*y = 0})
  (h2 : circle2 = {(x, y) | x^2 + y^2 - 6*x = 0})
  (h3 : (circle1 ∩ circle2).Nonempty) :
  ∃ (A B : ℝ × ℝ),
    A ∈ circle1 ∧ A ∈ circle2 ∧
    B ∈ circle1 ∧ B ∈ circle2 ∧
    A ≠ B ∧
    (∀ (x y : ℝ), (x, y) ∈ Set.Icc A B → x + 3*y = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l1406_140658


namespace NUMINAMATH_CALUDE_third_place_winnings_l1406_140616

theorem third_place_winnings (num_people : ℕ) (contribution : ℝ) (first_place_percentage : ℝ) :
  num_people = 8 →
  contribution = 5 →
  first_place_percentage = 0.8 →
  let total_pot := num_people * contribution
  let first_place_amount := first_place_percentage * total_pot
  let remaining_amount := total_pot - first_place_amount
  remaining_amount / 2 = 4 := by sorry

end NUMINAMATH_CALUDE_third_place_winnings_l1406_140616


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1406_140699

theorem reciprocal_sum_theorem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) :
  1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1406_140699


namespace NUMINAMATH_CALUDE_congruence_problem_l1406_140663

theorem congruence_problem (x : ℤ) :
  x ≡ 3 [ZMOD 7] →
  x^2 ≡ 44 [ZMOD (7^2)] →
  x^3 ≡ 111 [ZMOD (7^3)] →
  x ≡ 17 [ZMOD 343] := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l1406_140663


namespace NUMINAMATH_CALUDE_percentage_problem_l1406_140622

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 1.2 * x = 600 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1406_140622


namespace NUMINAMATH_CALUDE_age_problem_l1406_140633

theorem age_problem :
  ∀ (a b c : ℕ),
  a + b + c = 29 →
  a = b →
  c = 11 →
  a = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1406_140633


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1406_140698

/-- A point on a 2D grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A set of black cells on the grid --/
def BlackCells := Set GridPoint

/-- A line on the grid (vertical, horizontal, or diagonal) --/
inductive GridLine
  | Vertical (x : ℤ)
  | Horizontal (y : ℤ)
  | Diagonal (m : ℤ) (b : ℤ)

/-- The number of black cells on a given line --/
def blackCellsOnLine (cells : BlackCells) (line : GridLine) : ℕ :=
  sorry

/-- The property that a set of black cells satisfies the k-cell condition --/
def satisfiesKCellCondition (cells : BlackCells) (k : ℕ) : Prop :=
  ∀ line : GridLine, blackCellsOnLine cells line = k ∨ blackCellsOnLine cells line = 0

theorem exists_valid_coloring (k : ℕ) : 
  ∃ (cells : BlackCells), cells.Nonempty ∧ Set.Finite cells ∧ satisfiesKCellCondition cells k :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1406_140698


namespace NUMINAMATH_CALUDE_spending_solution_l1406_140668

def spending_problem (n : ℚ) : Prop :=
  let after_hardware := (3/4) * n
  let after_cleaners := after_hardware - 9
  let after_grocery := (1/2) * after_cleaners
  after_grocery = 12

theorem spending_solution : 
  ∃ (n : ℚ), spending_problem n ∧ n = 44 :=
sorry

end NUMINAMATH_CALUDE_spending_solution_l1406_140668


namespace NUMINAMATH_CALUDE_personal_income_tax_example_l1406_140653

/-- Calculate the personal income tax for a citizen given their salary and prize information --/
def personal_income_tax (salary_jan_jun : ℕ) (salary_jul_dec : ℕ) (prize_value : ℕ) : ℕ :=
  let salary_tax_rate : ℚ := 13 / 100
  let prize_tax_rate : ℚ := 35 / 100
  let non_taxable_prize : ℕ := 4000
  let total_salary : ℕ := salary_jan_jun * 6 + salary_jul_dec * 6
  let salary_tax : ℕ := (total_salary * salary_tax_rate).floor.toNat
  let taxable_prize : ℕ := max (prize_value - non_taxable_prize) 0
  let prize_tax : ℕ := (taxable_prize * prize_tax_rate).floor.toNat
  salary_tax + prize_tax

/-- Theorem stating that the personal income tax for the given scenario is 39540 rubles --/
theorem personal_income_tax_example : 
  personal_income_tax 23000 25000 10000 = 39540 := by
  sorry

end NUMINAMATH_CALUDE_personal_income_tax_example_l1406_140653


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1406_140655

theorem complex_equation_solution (z : ℂ) (h : 10 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 - 1) + 40) :
  z + 9 / z = (9 + Real.sqrt 61) / 2 ∨ z + 9 / z = (9 - Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1406_140655


namespace NUMINAMATH_CALUDE_vacation_photos_count_l1406_140650

/-- The number of photos Alyssa took on vacation -/
def total_photos : ℕ := 100

/-- The total number of pages in the album -/
def total_pages : ℕ := 30

/-- The number of photos that can be placed on each of the first 10 pages -/
def photos_per_page_first : ℕ := 3

/-- The number of photos that can be placed on each of the next 10 pages -/
def photos_per_page_second : ℕ := 4

/-- The number of photos that can be placed on each of the remaining pages -/
def photos_per_page_last : ℕ := 3

/-- The number of pages in the first section -/
def pages_first_section : ℕ := 10

/-- The number of pages in the second section -/
def pages_second_section : ℕ := 10

theorem vacation_photos_count : 
  total_photos = 
    photos_per_page_first * pages_first_section + 
    photos_per_page_second * pages_second_section + 
    photos_per_page_last * (total_pages - pages_first_section - pages_second_section) :=
by sorry

end NUMINAMATH_CALUDE_vacation_photos_count_l1406_140650


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l1406_140672

/-- 
Given a geometric sequence where:
  a₅ = 5 (5th term is 5)
  a₈ = 40 (8th term is 40)
Prove that a₁₁ = 320 (11th term is 320)
-/
theorem geometric_sequence_11th_term 
  (a : ℕ → ℝ) -- The geometric sequence
  (h₁ : a 5 = 5) -- 5th term is 5
  (h₂ : a 8 = 40) -- 8th term is 40
  (h₃ : ∀ n m : ℕ, a (n + m) = a n * (a 6 / a 5) ^ m) -- Geometric sequence property
  : a 11 = 320 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l1406_140672


namespace NUMINAMATH_CALUDE_max_d_l1406_140690

/-- The sequence b_n defined as (7^n - 4) / 3 -/
def b (n : ℕ) : ℤ := (7^n - 4) / 3

/-- The greatest common divisor of b_n and b_{n+1} -/
def d' (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

/-- The maximum value of d'_n is 3 for all natural numbers n -/
theorem max_d'_is_3 : ∀ n : ℕ, d' n = 3 := by sorry

end NUMINAMATH_CALUDE_max_d_l1406_140690


namespace NUMINAMATH_CALUDE_truncated_cube_vertex_edge_count_l1406_140652

/-- A polyhedron with 8 triangular faces and 6 heptagonal faces -/
structure TruncatedCube where
  triangularFaces : ℕ
  heptagonalFaces : ℕ
  triangularFaces_eq : triangularFaces = 8
  heptagonalFaces_eq : heptagonalFaces = 6

/-- The number of vertices in a TruncatedCube -/
def vertexCount (cube : TruncatedCube) : ℕ := 21

/-- The number of edges in a TruncatedCube -/
def edgeCount (cube : TruncatedCube) : ℕ := 33

/-- Theorem stating that a TruncatedCube has 21 vertices and 33 edges -/
theorem truncated_cube_vertex_edge_count (cube : TruncatedCube) : 
  vertexCount cube = 21 ∧ edgeCount cube = 33 := by
  sorry


end NUMINAMATH_CALUDE_truncated_cube_vertex_edge_count_l1406_140652


namespace NUMINAMATH_CALUDE_intersection_M_N_l1406_140602

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 = x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1406_140602


namespace NUMINAMATH_CALUDE_equation_solution_l1406_140666

theorem equation_solution : ∃! x : ℝ, (x^2 + x)^2 + Real.sqrt (x^2 - 1) = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1406_140666


namespace NUMINAMATH_CALUDE_charlie_won_two_games_l1406_140619

/-- Represents a player in the tournament -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player

/-- The number of games won by a player -/
def games_won (p : Player) : ℕ :=
  match p with
  | Player.Alice => 2
  | Player.Bob => 1
  | Player.Charlie => sorry  -- To be proven

/-- The number of games lost by a player -/
def games_lost (p : Player) : ℕ :=
  match p with
  | Player.Alice => 1
  | Player.Bob => 2
  | Player.Charlie => 2

/-- The total number of games played in the tournament -/
def total_games : ℕ := 3

theorem charlie_won_two_games :
  games_won Player.Charlie = 2 := by sorry

end NUMINAMATH_CALUDE_charlie_won_two_games_l1406_140619


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l1406_140693

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℕ
  b : ℕ

/-- Represents a square with side length s -/
structure Square where
  s : ℕ

theorem rectangle_side_lengths 
  (rect : Rectangle)
  (sq1 sq2 : Square)
  (h1 : sq1.s = sq2.s)
  (h2 : rect.a + rect.b = 19)
  (h3 : 2 * (rect.a + rect.b) + 2 * sq1.s = 48)
  (h4 : 2 * (rect.a + rect.b) + 4 * sq1.s = 58)
  (h5 : rect.a > rect.b)
  (h6 : rect.a ≤ 13) :
  rect.a = 12 ∧ rect.b = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l1406_140693


namespace NUMINAMATH_CALUDE_smallest_prime_10_less_than_perfect_square_l1406_140637

/-- A number is a perfect square if it's the square of an integer -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- The smallest prime that is 10 less than a perfect square -/
theorem smallest_prime_10_less_than_perfect_square :
  (∃ a : ℕ, Nat.Prime a ∧
    (∃ b : ℕ, is_perfect_square b ∧ a = b - 10) ∧
    (∀ a' : ℕ, a' < a →
      ¬(Nat.Prime a' ∧ ∃ b' : ℕ, is_perfect_square b' ∧ a' = b' - 10))) →
  (∃ a : ℕ, a = 71 ∧ Nat.Prime a ∧
    (∃ b : ℕ, is_perfect_square b ∧ a = b - 10) ∧
    (∀ a' : ℕ, a' < a →
      ¬(Nat.Prime a' ∧ ∃ b' : ℕ, is_perfect_square b' ∧ a' = b' - 10))) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_10_less_than_perfect_square_l1406_140637


namespace NUMINAMATH_CALUDE_inequality_chain_l1406_140654

theorem inequality_chain (x : ℝ) 
  (h1 : 0 < x) (h2 : x < 1) 
  (a b c : ℝ) 
  (ha : a = x^2) 
  (hb : b = 1/x) 
  (hc : c = Real.sqrt x) : 
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_inequality_chain_l1406_140654


namespace NUMINAMATH_CALUDE_number_of_roots_l1406_140601

/-- The number of real roots of a quadratic equation (m-5)x^2 - 2(m+2)x + m = 0,
    given that mx^2 - 2(m+2)x + m + 5 = 0 has no real roots -/
theorem number_of_roots (m : ℝ) 
  (h : ∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) :
  (∃! x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m = 0) ∨ 
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2*(m+2)*x + m = 0 ∧ (m-5) * y^2 - 2*(m+2)*y + m = 0) :=
sorry

end NUMINAMATH_CALUDE_number_of_roots_l1406_140601
