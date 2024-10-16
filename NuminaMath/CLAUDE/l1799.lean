import Mathlib

namespace NUMINAMATH_CALUDE_angle_complement_half_supplement_l1799_179935

theorem angle_complement_half_supplement : 
  ∃ (x : ℝ), x > 0 ∧ x < 90 ∧ (90 - x) = (1/2) * (180 - x) ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_half_supplement_l1799_179935


namespace NUMINAMATH_CALUDE_f_properties_l1799_179963

def f (x : ℝ) := x^3 - x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x < -Real.sqrt 3 / 3 → ∀ y, x < y → f x < f y) ∧
  (∀ x, x > Real.sqrt 3 / 3 → ∀ y, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1799_179963


namespace NUMINAMATH_CALUDE_cricket_average_increase_l1799_179937

theorem cricket_average_increase (initial_average : ℝ) : 
  (16 * initial_average + 92) / 17 = initial_average + 4 → 
  (16 * initial_average + 92) / 17 = 28 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l1799_179937


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l1799_179932

/-- The area of a square with one side on y = 8 and endpoints on y = x^2 + 4x + 3 is 36 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 8) ∧
  (x₂^2 + 4*x₂ + 3 = 8) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 36) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l1799_179932


namespace NUMINAMATH_CALUDE_limit_implies_range_l1799_179987

theorem limit_implies_range (a : ℝ) : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) → 
  a ∈ Set.Ioo (-4 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_limit_implies_range_l1799_179987


namespace NUMINAMATH_CALUDE_planes_not_parallel_l1799_179966

/-- Represents a 3D vector --/
structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space --/
structure Plane where
  normal : Vec3

/-- Check if two planes are parallel --/
def are_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.normal = Vec3.mk (k * p2.normal.x) (k * p2.normal.y) (k * p2.normal.z)

theorem planes_not_parallel : ¬ (are_parallel 
  (Plane.mk (Vec3.mk 0 1 3)) 
  (Plane.mk (Vec3.mk 1 0 3))) := by
  sorry

#check planes_not_parallel

end NUMINAMATH_CALUDE_planes_not_parallel_l1799_179966


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1799_179950

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- The x-coordinate of the point the hyperbola passes through -/
  point_x : ℝ
  /-- The y-coordinate of the point the hyperbola passes through -/
  point_y : ℝ

/-- The equation of the hyperbola satisfies the given conditions -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 1/2)
  (h_point : h.point_x = 4 ∧ h.point_y = Real.sqrt 2) :
  ∃ (f : ℝ → ℝ → Prop), 
    (∀ x y, f x y ↔ x^2/8 - y^2/2 = 1) ∧ 
    (f h.point_x h.point_y) ∧
    (∀ x, f x (h.asymptote_slope * x) ∨ f x (-h.asymptote_slope * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1799_179950


namespace NUMINAMATH_CALUDE_disk_covering_radius_bound_l1799_179918

theorem disk_covering_radius_bound (R : ℝ) (r : ℝ) :
  R = 1 →
  (∃ (centers : Fin 7 → ℝ × ℝ),
    (∀ x y : ℝ × ℝ, (x.1 - y.1)^2 + (x.2 - y.2)^2 ≤ R^2 →
      ∃ i : Fin 7, (x.1 - (centers i).1)^2 + (x.2 - (centers i).2)^2 ≤ r^2)) →
  r ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_disk_covering_radius_bound_l1799_179918


namespace NUMINAMATH_CALUDE_difference_of_squares_l1799_179968

theorem difference_of_squares (x : ℝ) : x^2 - 16 = (x + 4) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1799_179968


namespace NUMINAMATH_CALUDE_gravitational_force_at_distance_l1799_179900

/-- Gravitational force calculation -/
theorem gravitational_force_at_distance 
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances from Earth's center
  (f₁ : ℝ) -- Force at distance d₁
  (h₁ : d₁ > 0)
  (h₂ : d₂ > 0)
  (h₃ : f₁ > 0)
  (h₄ : k = f₁ * d₁^2) -- Force-distance relation at d₁
  (h₅ : d₁ = 4000) -- Distance to Earth's surface in miles
  (h₆ : f₁ = 500) -- Force at Earth's surface in Newtons
  (h₇ : d₂ = 40000) -- Distance to space station in miles
  : f₁ * (d₂ / d₁)^2 = 5 := by
  sorry

#check gravitational_force_at_distance

end NUMINAMATH_CALUDE_gravitational_force_at_distance_l1799_179900


namespace NUMINAMATH_CALUDE_contrapositive_odd_product_l1799_179991

theorem contrapositive_odd_product (a b : ℤ) :
  (¬(Odd (a * b)) → ¬(Odd a ∧ Odd b)) ↔
  ((Odd a ∧ Odd b) → Odd (a * b)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_odd_product_l1799_179991


namespace NUMINAMATH_CALUDE_train_crossing_time_l1799_179983

/-- Calculates the time taken for two trains to cross each other. -/
theorem train_crossing_time (length1 length2 speed1 speed2 initial_distance : ℝ) 
  (h1 : length1 = 135.5)
  (h2 : length2 = 167.2)
  (h3 : speed1 = 55)
  (h4 : speed2 = 43)
  (h5 : initial_distance = 250) :
  ∃ (time : ℝ), (abs (time - 20.3) < 0.1) ∧ 
  (time = (length1 + length2 + initial_distance) / ((speed1 + speed2) * (5/18))) :=
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1799_179983


namespace NUMINAMATH_CALUDE_a_is_independent_variable_l1799_179944

-- Define the perimeter function for a rhombus
def rhombus_perimeter (a : ℝ) : ℝ := 4 * a

-- Statement to prove
theorem a_is_independent_variable :
  ∃ (C : ℝ → ℝ), C = rhombus_perimeter ∧ 
  (∀ (a : ℝ), C a = 4 * a) ∧
  (∀ (a₁ a₂ : ℝ), a₁ ≠ a₂ → C a₁ ≠ C a₂) :=
sorry

end NUMINAMATH_CALUDE_a_is_independent_variable_l1799_179944


namespace NUMINAMATH_CALUDE_only_cylinder_quadrilateral_l1799_179976

-- Define the types of geometric solids
inductive GeometricSolid
  | Cone
  | Sphere
  | Cylinder

-- Define the possible shapes of plane sections
inductive PlaneSection
  | Circle
  | Ellipse
  | Parabola
  | Triangle
  | Quadrilateral

-- Function to determine possible plane sections for each solid
def possibleSections (solid : GeometricSolid) : Set PlaneSection :=
  match solid with
  | GeometricSolid.Cone => {PlaneSection.Circle, PlaneSection.Ellipse, PlaneSection.Parabola, PlaneSection.Triangle}
  | GeometricSolid.Sphere => {PlaneSection.Circle}
  | GeometricSolid.Cylinder => {PlaneSection.Circle, PlaneSection.Ellipse, PlaneSection.Quadrilateral}

-- Theorem stating that only a cylinder can produce a quadrilateral section
theorem only_cylinder_quadrilateral :
  ∀ (solid : GeometricSolid),
    PlaneSection.Quadrilateral ∈ possibleSections solid ↔ solid = GeometricSolid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_only_cylinder_quadrilateral_l1799_179976


namespace NUMINAMATH_CALUDE_M_mod_1000_l1799_179994

/-- The number of distinguishable flagpoles -/
def num_flagpoles : ℕ := 2

/-- The total number of flags -/
def total_flags : ℕ := 21

/-- The number of blue flags -/
def blue_flags : ℕ := 12

/-- The number of green flags -/
def green_flags : ℕ := 9

/-- The minimum number of flags required on each flagpole -/
def min_flags_per_pole : ℕ := 3

/-- The function to calculate the number of distinguishable arrangements -/
def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_mod_1000 : M % 1000 = 596 := by sorry

end NUMINAMATH_CALUDE_M_mod_1000_l1799_179994


namespace NUMINAMATH_CALUDE_intersection_M_N_l1799_179919

def M : Set ℝ := {-1, 1, 2, 3, 4}
def N : Set ℝ := {x : ℝ | x^2 + 2*x > 3}

theorem intersection_M_N : M ∩ N = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1799_179919


namespace NUMINAMATH_CALUDE_unique_determination_by_digit_sums_l1799_179969

/-- Given a natural number, compute the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Given a natural number N, generate a sequence of digit sums for consecutive numbers starting from N+1 -/
def digit_sum_sequence (N : ℕ) (length : ℕ) : List ℕ := sorry

/-- Theorem: For any natural number N, there exists a finite sequence of digit sums that uniquely determines N -/
theorem unique_determination_by_digit_sums (N : ℕ) : 
  ∃ (length : ℕ), ∀ (M : ℕ), M ≠ N → 
    digit_sum_sequence N length ≠ digit_sum_sequence M length := by
  sorry

#check unique_determination_by_digit_sums

end NUMINAMATH_CALUDE_unique_determination_by_digit_sums_l1799_179969


namespace NUMINAMATH_CALUDE_middle_card_is_four_l1799_179907

/-- Represents the three cards with positive integers -/
structure Cards where
  left : ℕ+
  middle : ℕ+
  right : ℕ+
  different : left ≠ middle ∧ middle ≠ right ∧ left ≠ right
  increasing : left < middle ∧ middle < right
  sum_15 : left + middle + right = 15

/-- Casey cannot determine the other two numbers given the leftmost card -/
def casey_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.left = cards.left → 
    (other_cards.middle ≠ cards.middle ∨ other_cards.right ≠ cards.right)

/-- Tracy cannot determine the other two numbers given the rightmost card -/
def tracy_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.right = cards.right → 
    (other_cards.left ≠ cards.left ∨ other_cards.middle ≠ cards.middle)

/-- Stacy cannot determine the other two numbers given the middle card -/
def stacy_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.middle = cards.middle → 
    (other_cards.left ≠ cards.left ∨ other_cards.right ≠ cards.right)

/-- The main theorem stating that the middle card must be 4 -/
theorem middle_card_is_four (cards : Cards) 
  (h_casey : casey_statement cards)
  (h_tracy : tracy_statement cards)
  (h_stacy : stacy_statement cards) : 
  cards.middle = 4 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_four_l1799_179907


namespace NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l1799_179992

theorem divisibility_implies_gcd_greater_than_one
  (a b x y : ℕ)
  (h : (a^2 + b^2) ∣ (a*x + b*y)) :
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l1799_179992


namespace NUMINAMATH_CALUDE_strawberry_calculation_l1799_179955

theorem strawberry_calculation (initial : ℝ) (sold : ℝ) (given_away : ℝ) (eaten : ℝ) 
  (h1 : initial = 120.5)
  (h2 : sold = 8.25)
  (h3 : given_away = 33.5)
  (h4 : eaten = 4.3) :
  initial - sold - given_away - eaten = 74.45 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_calculation_l1799_179955


namespace NUMINAMATH_CALUDE_f_shifted_l1799_179922

/-- Given a function f(x) = 3x - 5, prove that f(x - 4) = 3x - 17 for any real number x -/
theorem f_shifted (x : ℝ) : (fun x => 3 * x - 5) (x - 4) = 3 * x - 17 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_l1799_179922


namespace NUMINAMATH_CALUDE_train_speed_l1799_179904

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 24) :
  (train_length + bridge_length) / crossing_time = 400 / 24 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l1799_179904


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l1799_179920

theorem complex_arithmetic_proof :
  let z₁ : ℂ := 5 + 6*I
  let z₂ : ℂ := -1 + 4*I
  let z₃ : ℂ := 3 - 2*I
  (z₁ + z₂) - z₃ = 1 + 12*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l1799_179920


namespace NUMINAMATH_CALUDE_polynomial_negative_roots_l1799_179977

def polynomial (q x : ℝ) : ℝ := x^4 + 2*q*x^3 + 2*x^2 + 2*q*x + 4

theorem polynomial_negative_roots (q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
   polynomial q x₁ = 0 ∧ polynomial q x₂ = 0) ↔ 
  q > (3 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_polynomial_negative_roots_l1799_179977


namespace NUMINAMATH_CALUDE_trapezoid_bases_l1799_179936

/-- An isosceles trapezoid with a circumscribed circle -/
structure IsoscelesTrapezoid where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The ratio of the lower part of the height to the total height -/
  heightRatio : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- The radius is positive -/
  radiusPos : radius > 0
  /-- The height ratio is between 0 and 1 -/
  heightRatioValid : 0 < heightRatio ∧ heightRatio < 1
  /-- The height is positive -/
  heightPos : height > 0
  /-- The bases are positive -/
  basesPos : shorterBase > 0 ∧ longerBase > 0
  /-- The longer base is longer than the shorter base -/
  basesOrder : shorterBase < longerBase
  /-- The center of the circle divides the height in the given ratio -/
  centerDivision : heightRatio = 4 / 7
  /-- The median is equal to the height -/
  medianEqualsHeight : (shorterBase + longerBase) / 2 = height

/-- The theorem stating the bases of the trapezoid given the conditions -/
theorem trapezoid_bases (t : IsoscelesTrapezoid) (h : t.radius = 10) :
  t.shorterBase = 12 ∧ t.longerBase = 16 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l1799_179936


namespace NUMINAMATH_CALUDE_article_cost_l1799_179930

/-- The cost of an article, given selling conditions -/
theorem article_cost : ∃ (cost : ℝ), 
  (580 - cost) = 1.08 * (520 - cost) ∧ 
  cost = 230 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1799_179930


namespace NUMINAMATH_CALUDE_sequence_problem_l1799_179953

theorem sequence_problem (a : Fin 100 → ℚ) :
  (∀ i : Fin 98, a (Fin.succ i) = a i * a (Fin.succ (Fin.succ i))) →
  (∀ i : Fin 100, a i ≠ 0) →
  a 0 = 2018 →
  a 99 = 1 / 2018 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1799_179953


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1799_179914

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1799_179914


namespace NUMINAMATH_CALUDE_equation_solution_l1799_179949

theorem equation_solution (x : ℝ) : 3 - 1 / (1 - x) = 2 * (1 / (1 - x)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1799_179949


namespace NUMINAMATH_CALUDE_building_height_from_shadows_l1799_179970

/-- Given a bamboo pole and a building with their respective shadows, 
    calculate the height of the building using similar triangles. -/
theorem building_height_from_shadows 
  (bamboo_height : ℝ) 
  (bamboo_shadow : ℝ) 
  (building_shadow : ℝ) 
  (h_bamboo_height : bamboo_height = 1.8)
  (h_bamboo_shadow : bamboo_shadow = 3)
  (h_building_shadow : building_shadow = 35)
  : (bamboo_height / bamboo_shadow) * building_shadow = 21 := by
  sorry


end NUMINAMATH_CALUDE_building_height_from_shadows_l1799_179970


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1799_179997

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a > r1) (hb : b > r2) :
  let d := Nat.gcd (a - r1) (b - r2)
  d = Nat.gcd a b ∧ 
  a % d = r1 ∧ 
  b % d = r2 ∧ 
  ∀ m : ℕ, m > d → (a % m = r1 ∧ b % m = r2) → False :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1799_179997


namespace NUMINAMATH_CALUDE_age_difference_l1799_179952

theorem age_difference (A B C : ℕ) (h1 : C = A - 17) : A + B - (B + C) = 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1799_179952


namespace NUMINAMATH_CALUDE_sector_area_l1799_179945

/-- The area of a sector with radius 10 cm and central angle 120° is (100π/3) cm² -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 10 → θ = 2 * π / 3 → (1/2) * r^2 * θ = (100 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1799_179945


namespace NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l1799_179909

theorem angle_measure_in_acute_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  b = 2 * a * Real.sin B →
  A = π/6 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l1799_179909


namespace NUMINAMATH_CALUDE_binary_1101_to_base5_l1799_179934

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base-5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem binary_1101_to_base5 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true]) = [2, 3] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_to_base5_l1799_179934


namespace NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l1799_179989

/-- Converts a base 9 number to base 10 --/
def base9_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 --/
def base10_to_base9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 9 number --/
def is_4digit_base9 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 8888

theorem greatest_4digit_base9_divisible_by_7 :
  ∀ n : ℕ, is_4digit_base9 n →
    (base9_to_base10 n) % 7 = 0 →
    n ≤ 8050 :=
by sorry

end NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l1799_179989


namespace NUMINAMATH_CALUDE_no_integer_root_pairs_l1799_179981

theorem no_integer_root_pairs (n : ℕ) : ¬ ∃ (a b : Fin 5 → ℤ),
  (∀ k : Fin 5, ∃ (x y : ℤ), x^2 + a k * x + b k = 0 ∧ y^2 + a k * y + b k = 0) ∧
  (∀ k : Fin 5, ∃ m : ℤ, a k = 2 * n + 2 * k + 2 ∨ a k = 2 * n + 2 * k + 4) ∧
  (∀ k : Fin 5, ∃ m : ℤ, b k = 2 * n + 2 * k + 2 ∨ b k = 2 * n + 2 * k + 4) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_root_pairs_l1799_179981


namespace NUMINAMATH_CALUDE_fraction_states_1800_1809_l1799_179984

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 5

/-- The total number of states considered (first 30 states) -/
def total_states : ℕ := 30

/-- The fraction of states that joined from 1800 to 1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_states_1800_1809 : fraction_1800_1809 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_states_1800_1809_l1799_179984


namespace NUMINAMATH_CALUDE_expression_simplification_l1799_179933

theorem expression_simplification (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  2 * (a^2 * b + a * b^2) - 3 * (a^2 * b + 1) - 2 * a * b^2 - 2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1799_179933


namespace NUMINAMATH_CALUDE_line_parallel_to_skew_line_l1799_179972

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relations between lines
variable (parallel skew intersecting : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_to_skew_line
  (l1 l2 l3 : Line)
  (h1 : skew l1 l2)
  (h2 : parallel l3 l1) :
  intersecting l3 l2 ∨ skew l3 l2 :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_skew_line_l1799_179972


namespace NUMINAMATH_CALUDE_solution_value_l1799_179912

theorem solution_value (x y a : ℝ) : 
  x = 1 ∧ y = 1 ∧ 2*x - a*y = 3 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1799_179912


namespace NUMINAMATH_CALUDE_new_person_weight_l1799_179971

/-- Given a group of 8 people where one person weighing 66 kg is replaced by a new person,
    if the average weight of the group increases by 2.5 kg,
    then the weight of the new person is 86 kg. -/
theorem new_person_weight (initial_group_size : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_group_size = 8 →
  weight_increase = 2.5 →
  replaced_weight = 66 →
  (initial_group_size : ℝ) * weight_increase + replaced_weight = 86 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1799_179971


namespace NUMINAMATH_CALUDE_least_cube_divisible_by_168_l1799_179927

theorem least_cube_divisible_by_168 :
  ∀ k : ℕ, k > 0 → k^3 % 168 = 0 → k ≥ 42 :=
by
  sorry

end NUMINAMATH_CALUDE_least_cube_divisible_by_168_l1799_179927


namespace NUMINAMATH_CALUDE_point_a_in_second_quadrant_l1799_179925

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the second quadrant -/
def in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The specific point we're considering -/
def point_a : Point :=
  { x := -1, y := 2 }

/-- Theorem stating that point_a is in the second quadrant -/
theorem point_a_in_second_quadrant : in_second_quadrant point_a := by
  sorry

end NUMINAMATH_CALUDE_point_a_in_second_quadrant_l1799_179925


namespace NUMINAMATH_CALUDE_g_zero_value_l1799_179964

def f (x : ℝ) : ℝ := 2 * x + 3

theorem g_zero_value (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = f x) : g 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_value_l1799_179964


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1799_179995

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1799_179995


namespace NUMINAMATH_CALUDE_odd_function_sum_l1799_179962

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_sum : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1799_179962


namespace NUMINAMATH_CALUDE_range_of_M_l1799_179947

theorem range_of_M (a θ : ℝ) (ha : a ≠ 0) :
  let M := (a^2 - a * Real.sin θ + 1) / (a^2 - a * Real.cos θ + 1)
  (4 - Real.sqrt 7) / 3 ≤ M ∧ M ≤ (4 + Real.sqrt 7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_M_l1799_179947


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1799_179965

theorem simplify_and_evaluate (a : ℚ) : 
  let b : ℚ := -1/3
  (a + b)^2 - a * (2*b + a) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1799_179965


namespace NUMINAMATH_CALUDE_equation_solution_l1799_179958

theorem equation_solution :
  let f : ℂ → ℂ := λ x => (x - 2)^4 + (x - 6)^4
  ∃ (a b c d : ℂ),
    (f a = 16 ∧ f b = 16 ∧ f c = 16 ∧ f d = 16) ∧
    (a = 4 + Complex.I * Real.sqrt (12 - 8 * Real.sqrt 2)) ∧
    (b = 4 - Complex.I * Real.sqrt (12 - 8 * Real.sqrt 2)) ∧
    (c = 4 + Complex.I * Real.sqrt (12 + 8 * Real.sqrt 2)) ∧
    (d = 4 - Complex.I * Real.sqrt (12 + 8 * Real.sqrt 2)) ∧
    ∀ (x : ℂ), f x = 16 → (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1799_179958


namespace NUMINAMATH_CALUDE_least_integer_with_2023_divisors_l1799_179902

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Check if n is divisible by d -/
def is_divisible_by (n d : ℕ) : Prop := sorry

theorem least_integer_with_2023_divisors :
  ∃ (m k : ℕ),
    (num_divisors (m * 6^k) = 2023) ∧
    (¬ is_divisible_by m 6) ∧
    (∀ n : ℕ, num_divisors n = 2023 → n ≥ m * 6^k) ∧
    m = 9216 ∧
    k = 6 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_2023_divisors_l1799_179902


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1799_179931

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of ways to choose 3 vertices from 10 vertices -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with one side being a side of the decagon -/
def one_side_triangles : ℕ := decagon_vertices * 5

/-- The number of triangles with two sides being sides of the decagon -/
def two_side_triangles : ℕ := decagon_vertices

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of selecting a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1799_179931


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l1799_179985

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (2 : ℚ) / 7 < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < (3 : ℚ) / 4 ∧ 
  ∀ (y : ℤ), ((2 : ℚ) / 7 < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < (3 : ℚ) / 4) → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l1799_179985


namespace NUMINAMATH_CALUDE_l₁_passes_through_neg_one_neg_one_perpendicular_condition_l1799_179998

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def l₂ (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

-- Theorem 1: l₁ passes through (-1, -1) for all a
theorem l₁_passes_through_neg_one_neg_one (a : ℝ) : l₁ a (-1) (-1) := by sorry

-- Theorem 2: If l₁ ⊥ l₂, then a = 0 or a = -4
theorem perpendicular_condition (a : ℝ) : 
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0) → 
  a = 0 ∨ a = -4 := by sorry

end NUMINAMATH_CALUDE_l₁_passes_through_neg_one_neg_one_perpendicular_condition_l1799_179998


namespace NUMINAMATH_CALUDE_lg_sum_equals_one_l1799_179939

theorem lg_sum_equals_one (a b : ℝ) 
  (ha : a + Real.log a = 10) 
  (hb : b + (10 : ℝ)^b = 10) : 
  Real.log (a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_sum_equals_one_l1799_179939


namespace NUMINAMATH_CALUDE_red_balls_in_stratified_sample_l1799_179942

/-- Calculates the number of red balls to be sampled in a stratified sampling by color -/
def stratifiedSampleRedBalls (totalPopulation : ℕ) (totalRedBalls : ℕ) (sampleSize : ℕ) : ℕ :=
  (totalRedBalls * sampleSize) / totalPopulation

/-- Theorem: The number of red balls in a stratified sample of 100 from 1000 balls with 50 red balls is 5 -/
theorem red_balls_in_stratified_sample :
  stratifiedSampleRedBalls 1000 50 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_in_stratified_sample_l1799_179942


namespace NUMINAMATH_CALUDE_digits_of_product_l1799_179990

theorem digits_of_product (n : ℕ) : n = 2^10 * 5^7 * 3^2 → (Nat.digits 10 n).length = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_digits_of_product_l1799_179990


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l1799_179993

-- Define the sequence (a_n)
def a : ℕ → ℤ
  | 0 => sorry  -- We don't know the initial value, so we use sorry
  | n + 1 => (a n)^3 + 1999

-- Define what it means for an integer to be a perfect square
def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k^2

-- State the theorem
theorem at_most_one_perfect_square :
  ∃! n : ℕ, is_perfect_square (a n) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l1799_179993


namespace NUMINAMATH_CALUDE_merchant_revenue_l1799_179903

/-- Calculates the total revenue for a set of vegetables --/
def total_revenue (quantities : List ℝ) (prices : List ℝ) (sold_percentages : List ℝ) : ℝ :=
  List.sum (List.zipWith3 (fun q p s => q * p * s) quantities prices sold_percentages)

/-- The total revenue generated by the merchant is $134.1 --/
theorem merchant_revenue : 
  let quantities : List ℝ := [20, 18, 12, 25, 10]
  let prices : List ℝ := [2, 3, 4, 1, 5]
  let sold_percentages : List ℝ := [0.6, 0.4, 0.75, 0.5, 0.8]
  total_revenue quantities prices sold_percentages = 134.1 := by
  sorry

end NUMINAMATH_CALUDE_merchant_revenue_l1799_179903


namespace NUMINAMATH_CALUDE_self_inverse_matrix_l1799_179996

theorem self_inverse_matrix (c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; c, d]
  A * A = 1 → c = 7.5 ∧ d = -4 := by
sorry

end NUMINAMATH_CALUDE_self_inverse_matrix_l1799_179996


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1799_179960

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1799_179960


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l1799_179924

noncomputable def f (x : ℝ) : ℝ := 16 * x^4 - 8 * x^3 + 9 * x^2 - 3 * x + 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = 0.5 ∧ f r = 0 ∧ ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l1799_179924


namespace NUMINAMATH_CALUDE_right_side_exponent_l1799_179938

theorem right_side_exponent (s : ℝ) : 
  (2^16 : ℝ) * (25^s) = 5 * (10^16) → 16 = 16 := by sorry

end NUMINAMATH_CALUDE_right_side_exponent_l1799_179938


namespace NUMINAMATH_CALUDE_train_crossing_time_l1799_179923

/-- Given a train and platform with specific dimensions and time to pass the platform,
    calculate the time it takes for the train to cross a point object (tree). -/
theorem train_crossing_time (train_length platform_length time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 500)
  (h3 : time_to_pass_platform = 170) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1799_179923


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l1799_179948

theorem ufo_convention_attendees :
  ∀ (total male female : ℕ),
    total = 120 →
    male = female + 4 →
    total = male + female →
    male = 62 := by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l1799_179948


namespace NUMINAMATH_CALUDE_machine_working_time_l1799_179913

theorem machine_working_time : ∃ y : ℝ, y > 0 ∧ 1 / (y + 4) + 1 / (y + 2) + 1 / y^2 = 1 / y ∧ y = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_l1799_179913


namespace NUMINAMATH_CALUDE_spinner_prime_probability_l1799_179940

def spinner : List Nat := [2, 7, 9, 11, 15, 17]

def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

def countPrimes (l : List Nat) : Nat :=
  (l.filter isPrime).length

theorem spinner_prime_probability :
  (countPrimes spinner : Rat) / (spinner.length : Rat) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_spinner_prime_probability_l1799_179940


namespace NUMINAMATH_CALUDE_sin_neg_five_pi_sixths_l1799_179916

theorem sin_neg_five_pi_sixths : Real.sin (-5 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_five_pi_sixths_l1799_179916


namespace NUMINAMATH_CALUDE_nelly_outbid_multiple_l1799_179908

def joes_bid : ℕ := 160000
def nellys_bid : ℕ := 482000
def additional_amount : ℕ := 2000

theorem nelly_outbid_multiple : 
  (nellys_bid - joes_bid - additional_amount) / joes_bid = 2 := by
  sorry

end NUMINAMATH_CALUDE_nelly_outbid_multiple_l1799_179908


namespace NUMINAMATH_CALUDE_inequality_proof_l1799_179901

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 
  1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1799_179901


namespace NUMINAMATH_CALUDE_opposite_numbers_l1799_179941

theorem opposite_numbers : ∀ x : ℚ, |x| = -x → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l1799_179941


namespace NUMINAMATH_CALUDE_triangle_interior_angle_ratio_l1799_179961

theorem triangle_interior_angle_ratio 
  (α β γ : ℝ) 
  (h1 : 2 * α + 3 * β = 4 * γ) 
  (h2 : α = 4 * β - γ) :
  ∃ (k : ℝ), k > 0 ∧ 
    2 * k = 180 - α ∧
    9 * k = 180 - β ∧
    4 * k = 180 - γ := by
sorry

end NUMINAMATH_CALUDE_triangle_interior_angle_ratio_l1799_179961


namespace NUMINAMATH_CALUDE_count_eights_theorem_l1799_179954

/-- Count of digit 8 appearances in integers from 1 to 800 -/
def count_eights : ℕ := 160

/-- The upper bound of the integer range -/
def upper_bound : ℕ := 800

/-- Counts the occurrences of a specific digit in a given range of integers -/
def count_digit_occurrences (digit : ℕ) (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

theorem count_eights_theorem :
  count_digit_occurrences 8 1 upper_bound = count_eights :=
sorry

end NUMINAMATH_CALUDE_count_eights_theorem_l1799_179954


namespace NUMINAMATH_CALUDE_lindsay_dolls_theorem_l1799_179959

theorem lindsay_dolls_theorem (blonde : ℕ) (brown : ℕ) (black : ℕ) : 
  blonde = 4 →
  brown = 4 * blonde →
  black = brown - 2 →
  brown + black - blonde = 26 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_dolls_theorem_l1799_179959


namespace NUMINAMATH_CALUDE_tiger_tree_trunk_time_l1799_179911

/-- The time taken for a tiger to run above a fallen tree trunk -/
theorem tiger_tree_trunk_time (tiger_length : ℝ) (tree_trunk_length : ℝ) (time_to_pass_point : ℝ) : 
  tiger_length = 5 →
  tree_trunk_length = 20 →
  time_to_pass_point = 1 →
  (tiger_length + tree_trunk_length) / (tiger_length / time_to_pass_point) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tiger_tree_trunk_time_l1799_179911


namespace NUMINAMATH_CALUDE_penguin_colony_ratio_l1799_179905

theorem penguin_colony_ratio :
  ∀ (initial_penguins end_first_year_penguins current_penguins : ℕ),
  end_first_year_penguins = 3 * initial_penguins →
  current_penguins = 3 * end_first_year_penguins + 129 →
  current_penguins = 1077 →
  end_first_year_penguins / initial_penguins = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_penguin_colony_ratio_l1799_179905


namespace NUMINAMATH_CALUDE_probability_multiple_2_or_3_30_l1799_179973

def is_multiple_of_2_or_3 (n : ℕ) : Bool :=
  n % 2 = 0 || n % 3 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_2_or_3 |>.length

theorem probability_multiple_2_or_3_30 :
  (count_multiples 30 : ℚ) / 30 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_2_or_3_30_l1799_179973


namespace NUMINAMATH_CALUDE_cube_difference_factorization_l1799_179915

theorem cube_difference_factorization (t : ℝ) : t^3 - 8 = (t - 2) * (t^2 + 2*t + 4) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_factorization_l1799_179915


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l1799_179910

theorem mean_equality_implies_y_equals_three :
  let mean1 := (3 + 7 + 11 + 15) / 4
  let mean2 := (10 + 14 + y) / 3
  mean1 = mean2 → y = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l1799_179910


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1799_179986

/-- Given a geometric sequence {aₙ}, prove that if a₃ · a₄ = 5, then a₁ · a₂ · a₅ · a₆ = 5 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) 
  (h_prod : a 3 * a 4 = 5) : a 1 * a 2 * a 5 * a 6 = 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1799_179986


namespace NUMINAMATH_CALUDE_compound_proposition_true_l1799_179943

theorem compound_proposition_true (a b : ℝ) :
  (a > 0 ∧ a + b < 0) → b < 0 := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_true_l1799_179943


namespace NUMINAMATH_CALUDE_largest_m_value_l1799_179982

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem largest_m_value :
  ∀ m x y : ℕ,
    m ≥ 1000 →
    m < 10000 →
    is_prime x →
    is_prime y →
    is_prime (10 * x + y) →
    x < 10 →
    y < 10 →
    x > y →
    m = x * y * (10 * x + y) →
    m ≤ 1533 :=
sorry

end NUMINAMATH_CALUDE_largest_m_value_l1799_179982


namespace NUMINAMATH_CALUDE_distance_theorem_l1799_179956

/-- The distance between Maxwell's and Brad's homes --/
def distance_between_homes (maxwell_speed : ℝ) (brad_speed : ℝ) (brad_delay : ℝ) (total_time : ℝ) : ℝ :=
  maxwell_speed * total_time + brad_speed * (total_time - brad_delay)

/-- Theorem stating the distance between Maxwell's and Brad's homes --/
theorem distance_theorem (maxwell_speed : ℝ) (brad_speed : ℝ) (brad_delay : ℝ) (total_time : ℝ)
  (h1 : maxwell_speed = 4)
  (h2 : brad_speed = 6)
  (h3 : brad_delay = 1)
  (h4 : total_time = 2) :
  distance_between_homes maxwell_speed brad_speed brad_delay total_time = 14 := by
  sorry

#check distance_theorem

end NUMINAMATH_CALUDE_distance_theorem_l1799_179956


namespace NUMINAMATH_CALUDE_triathlon_bike_speed_l1799_179926

def triathlon_speed (swim_distance : ℚ) (bike_distance : ℚ) (run_distance : ℚ)
                    (swim_speed : ℚ) (run_speed : ℚ) (total_time : ℚ) : ℚ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let bike_time := total_time - swim_time - run_time
  bike_distance / bike_time

theorem triathlon_bike_speed :
  triathlon_speed (1/2) 10 2 (3/2) 5 (3/2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bike_speed_l1799_179926


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1799_179980

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

#eval num_diagonals 30  -- This should output 405

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1799_179980


namespace NUMINAMATH_CALUDE_processing_box_function_is_assignment_and_calculation_l1799_179975

/-- Represents the possible functions of a processing box in an algorithm -/
inductive ProcessingBoxFunction
  | startIndicator
  | inputIndicator
  | assignmentAndCalculation
  | conditionJudgment

/-- The function of a processing box -/
def processingBoxFunction : ProcessingBoxFunction :=
  ProcessingBoxFunction.assignmentAndCalculation

/-- Theorem stating that the function of a processing box is assignment and calculation -/
theorem processing_box_function_is_assignment_and_calculation :
  processingBoxFunction = ProcessingBoxFunction.assignmentAndCalculation :=
by sorry

end NUMINAMATH_CALUDE_processing_box_function_is_assignment_and_calculation_l1799_179975


namespace NUMINAMATH_CALUDE_power_of_two_triples_l1799_179906

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triples :
  ∀ a b c : ℕ, valid_triple a b c ↔
    (a = 2 ∧ b = 2 ∧ c = 2) ∨
    (a = 2 ∧ b = 2 ∧ c = 3) ∨
    (a = 3 ∧ b = 5 ∧ c = 7) ∨
    (a = 2 ∧ b = 6 ∧ c = 11) ∨
    (a = 2 ∧ b = 3 ∧ c = 2) ∨
    (a = 2 ∧ b = 11 ∧ c = 6) ∨
    (a = 3 ∧ b = 7 ∧ c = 5) ∨
    (a = 5 ∧ b = 7 ∧ c = 3) ∨
    (a = 5 ∧ b = 3 ∧ c = 7) ∨
    (a = 6 ∧ b = 11 ∧ c = 2) ∨
    (a = 7 ∧ b = 3 ∧ c = 5) ∨
    (a = 7 ∧ b = 5 ∧ c = 3) ∨
    (a = 11 ∧ b = 2 ∧ c = 6) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_triples_l1799_179906


namespace NUMINAMATH_CALUDE_locus_of_symmetric_point_l1799_179979

/-- Given a parabola y = x^2 and a fixed point A(a, 0) where a ≠ 0, 
    the locus of point Q symmetric to A with respect to a point on the parabola 
    is described by the equation y = (1/2)(x + a)^2 -/
theorem locus_of_symmetric_point (a : ℝ) (ha : a ≠ 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ (x y : ℝ), (y = x^2) → 
      ∃ (qx qy : ℝ), 
        (qx + x = 2 * a ∧ qy + y = 0) → 
        f qx = qy ∧ f qx = (1/2) * (qx + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_symmetric_point_l1799_179979


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l1799_179957

/-- Polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- Polynomial g(x) -/
def g (x : ℝ) : ℝ := 3 - 2*x + x^2 - 6*x^3 + 11*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The theorem stating that -5/11 is the value of c that makes h(x) a polynomial of degree 3 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -5/11 ∧ 
  (∀ (x : ℝ), h c x = (1 + 3*c) + (-12 - 2*c)*x + (3 + c)*x^2 + (-4 - 6*c)*x^3) :=
sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l1799_179957


namespace NUMINAMATH_CALUDE_defect_rate_two_procedures_l1799_179999

/-- The defect rate of a product after two independent procedures -/
def overall_defect_rate (a b : ℝ) : ℝ := 1 - (1 - a) * (1 - b)

/-- Theorem: The overall defect rate of a product after two independent procedures
    with defect rates a and b is 1 - (1-a)(1-b) -/
theorem defect_rate_two_procedures
  (a b : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  : overall_defect_rate a b = 1 - (1 - a) * (1 - b) :=
by sorry

end NUMINAMATH_CALUDE_defect_rate_two_procedures_l1799_179999


namespace NUMINAMATH_CALUDE_geometric_progression_equation_l1799_179978

theorem geometric_progression_equation (x y z : ℝ) (r : ℝ) :
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∃ (a : ℝ), a ≠ 0 ∧
    x * (y - z) = a ∧
    y * (z - x) = a * r ∧
    z * (x - y) = a * r^3 →
  r^3 + r + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_equation_l1799_179978


namespace NUMINAMATH_CALUDE_coin_problem_l1799_179946

theorem coin_problem (x y : ℕ) : 
  x + y = 12 →
  5 * x + 10 * y = 90 →
  x = 6 ∧ y = 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l1799_179946


namespace NUMINAMATH_CALUDE_complete_square_m_values_l1799_179929

/-- A polynomial of the form x^2 + mx + 4 can be factored using the complete square formula -/
def is_complete_square (m : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 4 = (x + a)^2

/-- If a polynomial x^2 + mx + 4 can be factored using the complete square formula,
    then m = 4 or m = -4 -/
theorem complete_square_m_values (m : ℝ) :
  is_complete_square m → m = 4 ∨ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_m_values_l1799_179929


namespace NUMINAMATH_CALUDE_book_pages_count_l1799_179921

/-- Represents the number of pages read in a day period --/
structure ReadingPeriod where
  days : ℕ
  pagesPerDay : ℕ

/-- Calculates the total pages read in a period --/
def totalPages (period : ReadingPeriod) : ℕ :=
  period.days * period.pagesPerDay

/-- Represents Robert's reading schedule --/
def robertReading : List ReadingPeriod :=
  [{ days := 3, pagesPerDay := 28 },
   { days := 3, pagesPerDay := 35 },
   { days := 3, pagesPerDay := 42 }]

/-- The number of pages Robert read on the last day --/
def lastDayPages : ℕ := 15

/-- Theorem stating the total number of pages in the book --/
theorem book_pages_count :
  (robertReading.map totalPages).sum + lastDayPages = 330 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1799_179921


namespace NUMINAMATH_CALUDE_watermelon_weight_is_reasonable_l1799_179928

/-- The approximate weight of a typical watermelon in grams -/
def watermelon_weight : ℕ := 4000

/-- Predicate to determine if a given weight is a reasonable approximation for a watermelon -/
def is_reasonable_watermelon_weight (weight : ℕ) : Prop :=
  3500 ≤ weight ∧ weight ≤ 4500

/-- Theorem stating that the defined watermelon weight is a reasonable approximation -/
theorem watermelon_weight_is_reasonable : 
  is_reasonable_watermelon_weight watermelon_weight := by
  sorry

end NUMINAMATH_CALUDE_watermelon_weight_is_reasonable_l1799_179928


namespace NUMINAMATH_CALUDE_inverse_inequality_l1799_179917

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l1799_179917


namespace NUMINAMATH_CALUDE_motorboat_speed_calculation_l1799_179988

/-- The flood flow speed in kilometers per hour -/
def flood_speed : ℝ := 10

/-- The downstream distance in kilometers -/
def downstream_distance : ℝ := 2

/-- The upstream distance in kilometers -/
def upstream_distance : ℝ := 1.2

/-- The maximum speed of the motorboat in still water in kilometers per hour -/
def motorboat_speed : ℝ := 40

theorem motorboat_speed_calculation :
  (downstream_distance / (motorboat_speed + flood_speed) = 
   upstream_distance / (motorboat_speed - flood_speed)) ∧
  motorboat_speed = 40 := by sorry

end NUMINAMATH_CALUDE_motorboat_speed_calculation_l1799_179988


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_l1799_179967

theorem unique_solution_cube_difference (n m : ℤ) : 
  (n + 2)^4 - n^4 = m^3 ↔ n = -1 ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_l1799_179967


namespace NUMINAMATH_CALUDE_simplest_form_fraction_other_fractions_not_simplest_l1799_179974

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1. -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- The fraction (x^2 + y^2) / (x + y) is in simplest form. -/
theorem simplest_form_fraction (x y : ℤ) :
  IsSimplestForm (x^2 + y^2) (x + y) := by
  sorry

/-- Other fractions can be simplified further. -/
theorem other_fractions_not_simplest (x y : ℤ) :
  ¬IsSimplestForm (x * y) (x^2) ∧
  ¬IsSimplestForm (y^2 + y) (x * y) ∧
  ¬IsSimplestForm (x^2 - y^2) (x + y) := by
  sorry

end NUMINAMATH_CALUDE_simplest_form_fraction_other_fractions_not_simplest_l1799_179974


namespace NUMINAMATH_CALUDE_sum_of_three_digit_numbers_divisible_by_37_l1799_179951

/-- A function that generates all possible three-digit numbers from three digits -/
def generateThreeDigitNumbers (a b c : ℕ) : List ℕ :=
  [100*a + 10*b + c,
   100*a + 10*c + b,
   100*b + 10*a + c,
   100*b + 10*c + a,
   100*c + 10*a + b,
   100*c + 10*b + a]

/-- Theorem: The sum of all possible three-digit numbers formed from three distinct non-zero digits is divisible by 37 -/
theorem sum_of_three_digit_numbers_divisible_by_37 
  (a b c : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  37 ∣ (List.sum (generateThreeDigitNumbers a b c)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_digit_numbers_divisible_by_37_l1799_179951
