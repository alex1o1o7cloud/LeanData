import Mathlib

namespace NUMINAMATH_CALUDE_students_per_table_unchanged_l3910_391054

theorem students_per_table_unchanged (tables : ℝ) (initial_students_per_table : ℝ) 
  (h1 : tables = 34.0) 
  (h2 : initial_students_per_table = 6.0) : 
  (tables * initial_students_per_table) / tables = initial_students_per_table := by
  sorry

#check students_per_table_unchanged

end NUMINAMATH_CALUDE_students_per_table_unchanged_l3910_391054


namespace NUMINAMATH_CALUDE_team_x_games_l3910_391013

/-- Prove that Team X played 24 games given the conditions -/
theorem team_x_games (x : ℕ) 
  (h1 : (3 : ℚ) / 4 * x = x - (1 : ℚ) / 4 * x)  -- Team X wins 3/4 of its games
  (h2 : (2 : ℚ) / 3 * (x + 9) = (x + 9) - (1 : ℚ) / 3 * (x + 9))  -- Team Y wins 2/3 of its games
  (h3 : (2 : ℚ) / 3 * (x + 9) = (3 : ℚ) / 4 * x + 4)  -- Team Y won 4 more games than Team X
  : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_team_x_games_l3910_391013


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3910_391038

theorem negation_of_universal_proposition (P Q : Prop) :
  (P ↔ ∀ x : ℤ, x < 1) →
  (Q ↔ ∃ x : ℤ, x ≥ 1) →
  (¬P ↔ Q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3910_391038


namespace NUMINAMATH_CALUDE_inverse_proportion_constant_l3910_391069

/-- Given an inverse proportion function y = k/x passing through the point (-2, -3), prove that k = 6 -/
theorem inverse_proportion_constant (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x ≠ 0, f x = k / x) ∧ f (-2) = -3) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_constant_l3910_391069


namespace NUMINAMATH_CALUDE_least_cubes_from_cuboid_l3910_391025

/-- Given a cuboidal block with dimensions 6 cm x 9 cm x 12 cm,
    prove that the least possible number of equal cubes that can be cut from this block is 24. -/
theorem least_cubes_from_cuboid (length width height : ℕ) 
  (h_length : length = 6)
  (h_width : width = 9)
  (h_height : height = 12) :
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧
    length % cube_side = 0 ∧
    width % cube_side = 0 ∧
    height % cube_side = 0 ∧
    (length * width * height) / (cube_side ^ 3) = 24 ∧
    ∀ (other_side : ℕ), other_side > cube_side →
      ¬(length % other_side = 0 ∧
        width % other_side = 0 ∧
        height % other_side = 0)) :=
by sorry

end NUMINAMATH_CALUDE_least_cubes_from_cuboid_l3910_391025


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3910_391037

/-- The line y = mx + 2 is tangent to the ellipse x² + y²/4 = 1 if and only if m² = 0 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! p : ℝ × ℝ, p.1^2 + (p.2^2 / 4) = 1 ∧ p.2 = m * p.1 + 2) ↔ m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3910_391037


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_l3910_391074

theorem sqrt_sum_squares : Real.sqrt (3^2) + (Real.sqrt 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_l3910_391074


namespace NUMINAMATH_CALUDE_problem_statement_l3910_391004

theorem problem_statement (x y z t : ℝ) 
  (eq1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (eq2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (eq3 : x^2 - x * y + y^2 = t) :
  t ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3910_391004


namespace NUMINAMATH_CALUDE_max_value_of_min_expression_l3910_391036

theorem max_value_of_min_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (⨅ x ∈ ({1/a, 2/b, 4/c, (a*b*c)^(1/3)} : Set ℝ), x) ≤ Real.sqrt 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    (⨅ x ∈ ({1/a', 2/b', 4/c', (a'*b'*c')^(1/3)} : Set ℝ), x) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_min_expression_l3910_391036


namespace NUMINAMATH_CALUDE_T_is_four_sided_polygon_l3910_391044

-- Define the set T
def T (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    a ≤ x ∧ x ≤ 3*a ∧
    a ≤ y ∧ y ≤ 3*a ∧
    x + y ≥ 2*a ∧
    x + 2*a ≥ y ∧
    y + 2*a ≥ x ∧
    x + y ≤ 4*a}

-- Theorem statement
theorem T_is_four_sided_polygon (a : ℝ) (h : a > 0) :
  ∃ (v1 v2 v3 v4 : ℝ × ℝ),
    v1 ∈ T a ∧ v2 ∈ T a ∧ v3 ∈ T a ∧ v4 ∈ T a ∧
    (∀ p ∈ T a, p = v1 ∨ p = v2 ∨ p = v3 ∨ p = v4 ∨
      (∃ t : ℝ, 0 < t ∧ t < 1 ∧
        (p = (1 - t) • v1 + t • v2 ∨
         p = (1 - t) • v2 + t • v3 ∨
         p = (1 - t) • v3 + t • v4 ∨
         p = (1 - t) • v4 + t • v1))) :=
by sorry

end NUMINAMATH_CALUDE_T_is_four_sided_polygon_l3910_391044


namespace NUMINAMATH_CALUDE_swapped_divisible_by_37_l3910_391098

/-- Represents a nine-digit number split into two parts -/
structure SplitNumber where
  x : ℕ
  y : ℕ
  k : ℕ
  h1 : k > 0
  h2 : k < 10

/-- The original nine-digit number -/
def originalNumber (n : SplitNumber) : ℕ :=
  n.x * 10^(9 - n.k) + n.y

/-- The swapped nine-digit number -/
def swappedNumber (n : SplitNumber) : ℕ :=
  n.y * 10^n.k + n.x

/-- Theorem stating that if the original number is divisible by 37,
    then the swapped number is also divisible by 37 -/
theorem swapped_divisible_by_37 (n : SplitNumber) :
  37 ∣ originalNumber n → 37 ∣ swappedNumber n := by
  sorry


end NUMINAMATH_CALUDE_swapped_divisible_by_37_l3910_391098


namespace NUMINAMATH_CALUDE_astroid_length_l3910_391005

/-- The astroid curve -/
def astroid (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^(2/3) + p.2^(2/3) = a^(2/3)) ∧ a > 0}

/-- The length of a curve -/
noncomputable def curveLength (C : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The length of the astroid x^(2/3) + y^(2/3) = a^(2/3) is 6a -/
theorem astroid_length (a : ℝ) (h : a > 0) : 
  curveLength (astroid a) = 6 * a := by sorry

end NUMINAMATH_CALUDE_astroid_length_l3910_391005


namespace NUMINAMATH_CALUDE_line_divides_area_in_half_l3910_391057

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The L-shaped region defined by its vertices -/
def LShapedRegion : List Point2D := [
  ⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point2D) : ℝ :=
  sorry

/-- Calculate the area of a polygon formed by the origin and a line intersecting the L-shaped region -/
def areaAboveLine (slope : ℝ) : ℝ :=
  sorry

/-- The theorem stating that the line with slope 1/9 divides the L-shaped region in half -/
theorem line_divides_area_in_half :
  let totalArea := polygonArea LShapedRegion
  let slope := 1 / 9
  areaAboveLine slope = totalArea / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_divides_area_in_half_l3910_391057


namespace NUMINAMATH_CALUDE_scientific_notation_of_number_l3910_391034

def number : ℕ := 97070000000

theorem scientific_notation_of_number :
  (9.707 : ℝ) * (10 : ℝ) ^ (10 : ℕ) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_number_l3910_391034


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3910_391049

theorem right_triangle_hypotenuse (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : q < p) (hpq2 : p < q * Real.sqrt 1.8) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b = p ∧
    (1/3 : ℝ) * Real.sqrt (a^2 + 4*b^2) + (1/3 : ℝ) * Real.sqrt (4*a^2 + b^2) = q ∧
    c^2 = a^2 + b^2 ∧
    c^2 = (p^4 - 9*q^4) / (2*(p^2 - 5*q^2)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3910_391049


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l3910_391022

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 150

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The maximum number of team pairs that can play within the game limit -/
def max_team_pairs : ℕ := max_games / games_between_teams

/-- The function to calculate the number of unique pairs of teams -/
def team_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n > 0 ∧ team_pairs n ≤ max_team_pairs ∧ 
  ∀ (m : ℕ), m > n → team_pairs m > max_team_pairs :=
sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l3910_391022


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3910_391092

def cubic_equation (x : ℝ) : Prop := 60 * x^3 - 70 * x^2 + 24 * x - 2 = 0

theorem root_sum_theorem (p q r : ℝ) :
  cubic_equation p ∧ cubic_equation q ∧ cubic_equation r ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 0 < r ∧ r < 2 →
  1 / (2 - p) + 1 / (2 - q) + 1 / (2 - r) = 116 / 15 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3910_391092


namespace NUMINAMATH_CALUDE_quadratic_equal_real_roots_l3910_391090

theorem quadratic_equal_real_roots
  (a c : ℝ)
  (h_disc_zero : (4 * Real.sqrt 2) ^ 2 - 4 * a * c = 0)
  : ∃ x : ℝ, (a * x ^ 2 - 4 * x * Real.sqrt 2 + c = 0) ∧
    (∀ y : ℝ, a * y ^ 2 - 4 * y * Real.sqrt 2 + c = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_real_roots_l3910_391090


namespace NUMINAMATH_CALUDE_fish_count_l3910_391012

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 9

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 19 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3910_391012


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l3910_391078

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The solid CXYZ formed by slicing the prism -/
structure SolidCXYZ where
  prism : RightPrism
  C : Point3D
  X : Point3D
  Y : Point3D
  Z : Point3D

/-- Function to calculate the surface area of SolidCXYZ -/
def surface_area_CXYZ (solid : SolidCXYZ) : ℝ :=
  sorry

/-- Theorem statement -/
theorem surface_area_of_sliced_prism (solid : SolidCXYZ) 
  (h1 : solid.prism.base_side_length = 12)
  (h2 : solid.prism.height = 16)
  (h3 : solid.X.x = (solid.C.x + solid.prism.base_side_length / 2))
  (h4 : solid.Y.x = (solid.C.x + solid.prism.base_side_length))
  (h5 : solid.Z.z = (solid.C.z + solid.prism.height / 2)) :
  surface_area_CXYZ solid = 48 + 9 * Real.sqrt 3 + 3 * Real.sqrt 91 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l3910_391078


namespace NUMINAMATH_CALUDE_replacement_cost_100_movies_l3910_391047

/-- The cost to replace VHS movies with DVDs -/
def replacement_cost (num_movies : ℕ) (vhs_trade_value : ℚ) (dvd_cost : ℚ) : ℚ :=
  num_movies * dvd_cost - num_movies * vhs_trade_value

/-- Theorem: The cost to replace 100 VHS movies with DVDs is $800 -/
theorem replacement_cost_100_movies :
  replacement_cost 100 2 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_100_movies_l3910_391047


namespace NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l3910_391059

theorem sufficient_to_necessary_contrapositive (A B : Prop) 
  (h : A → B) : ¬B → ¬A := by sorry

end NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l3910_391059


namespace NUMINAMATH_CALUDE_sum_of_digits_511_base2_l3910_391023

/-- The sum of the digits in the base-2 representation of 511₁₀ is 9. -/
theorem sum_of_digits_511_base2 : 
  (List.range 9).sum = (Nat.digits 2 511).sum := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_511_base2_l3910_391023


namespace NUMINAMATH_CALUDE_sum_mod_nine_l3910_391008

theorem sum_mod_nine : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l3910_391008


namespace NUMINAMATH_CALUDE_min_boxes_for_cube_l3910_391000

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of boxes needed to form a cube -/
def boxesNeededForCube (box : BoxDimensions) : ℕ :=
  let lcm := Nat.lcm (Nat.lcm box.width box.length) box.height
  (lcm / box.width) * (lcm / box.length) * (lcm / box.height)

/-- The main theorem stating that 24 boxes are needed to form a cube -/
theorem min_boxes_for_cube :
  let box : BoxDimensions := ⟨18, 12, 9⟩
  boxesNeededForCube box = 24 := by
  sorry

#eval boxesNeededForCube ⟨18, 12, 9⟩

end NUMINAMATH_CALUDE_min_boxes_for_cube_l3910_391000


namespace NUMINAMATH_CALUDE_exponent_multiplication_calculate_expression_l3910_391087

theorem exponent_multiplication (a : ℕ) (m n : ℕ) : 
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem calculate_expression : 3000 * (3000 ^ 2500) = 3000 ^ 2501 :=
by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_calculate_expression_l3910_391087


namespace NUMINAMATH_CALUDE_exists_cycle_l3910_391020

structure Team :=
  (id : Nat)

structure Tournament :=
  (teams : Finset Team)
  (score : Team → Nat)
  (beats : Team → Team → Prop)
  (round_robin : ∀ t1 t2 : Team, t1 ≠ t2 → (beats t1 t2 ∨ beats t2 t1))

theorem exists_cycle (t : Tournament) 
  (h : ∃ t1 t2 : Team, t1 ∈ t.teams ∧ t2 ∈ t.teams ∧ t1 ≠ t2 ∧ t.score t1 = t.score t2) :
  ∃ A B C : Team, A ∈ t.teams ∧ B ∈ t.teams ∧ C ∈ t.teams ∧ 
    t.beats A B ∧ t.beats B C ∧ t.beats C A :=
sorry

end NUMINAMATH_CALUDE_exists_cycle_l3910_391020


namespace NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l3910_391018

/-- Given two cubic polynomials with specific root relationships, 
    this theorem states the possible values for the constant term of the first polynomial. -/
theorem cubic_polynomials_constant_term (c d : ℝ) (u v : ℝ) : 
  (∃ w : ℝ, u^3 + c*u + d = 0 ∧ v^3 + c*v + d = 0 ∧ w^3 + c*w + d = 0) →
  (∃ w : ℝ, (u+2)^3 + c*(u+2) + (d-120) = 0 ∧ 
            (v-5)^3 + c*(v-5) + (d-120) = 0 ∧ 
             w^3 + c*w + (d-120) = 0) →
  d = 396 ∨ d = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l3910_391018


namespace NUMINAMATH_CALUDE_satellite_sensor_upgrade_fraction_l3910_391084

theorem satellite_sensor_upgrade_fraction :
  ∀ (total_units : ℕ) (non_upgraded_per_unit : ℕ) (total_upgraded : ℕ),
    total_units = 24 →
    non_upgraded_per_unit * 4 = total_upgraded →
    (total_upgraded : ℚ) / (total_upgraded + total_units * non_upgraded_per_unit) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_satellite_sensor_upgrade_fraction_l3910_391084


namespace NUMINAMATH_CALUDE_problem_solution_l3910_391052

theorem problem_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (6 * x) * Real.sqrt (5 * x) * Real.sqrt (20 * x) = 20) : 
  x = (1 / 18) ^ (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3910_391052


namespace NUMINAMATH_CALUDE_winnie_keeps_six_l3910_391048

/-- The number of cherry lollipops Winnie has -/
def cherry : ℕ := 60

/-- The number of wintergreen lollipops Winnie has -/
def wintergreen : ℕ := 135

/-- The number of grape lollipops Winnie has -/
def grape : ℕ := 5

/-- The number of shrimp cocktail lollipops Winnie has -/
def shrimp : ℕ := 250

/-- The number of Winnie's friends -/
def friends : ℕ := 12

/-- The total number of lollipops Winnie has -/
def total : ℕ := cherry + wintergreen + grape + shrimp

/-- The number of lollipops Winnie keeps for herself -/
def kept : ℕ := total % friends

theorem winnie_keeps_six : kept = 6 := by
  sorry

end NUMINAMATH_CALUDE_winnie_keeps_six_l3910_391048


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l3910_391055

/-- The number of schools in the club -/
def num_schools : ℕ := 3

/-- The number of members from each school -/
def members_per_school : ℕ := 6

/-- The number of representatives sent by the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of possible ways to arrange the presidency meeting -/
def total_arrangements : ℕ := 2160

theorem presidency_meeting_arrangements :
  (num_schools * (members_per_school.choose host_representatives) *
   (members_per_school.choose non_host_representatives) *
   (members_per_school.choose non_host_representatives)) = total_arrangements :=
by sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l3910_391055


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3910_391053

theorem polynomial_simplification (x : ℝ) :
  (14 * x^12 + 8 * x^9 + 3 * x^8) + (2 * x^14 - x^12 + 2 * x^9 + 5 * x^5 + 7 * x^2 + 6) =
  2 * x^14 + 13 * x^12 + 10 * x^9 + 3 * x^8 + 5 * x^5 + 7 * x^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3910_391053


namespace NUMINAMATH_CALUDE_triangle_reflection_slope_l3910_391065

/-- Triangle DEF with vertices D(3,2), E(5,4), and F(2,6) reflected across y=2x -/
theorem triangle_reflection_slope (D E F D' E' F' : ℝ × ℝ) :
  D = (3, 2) →
  E = (5, 4) →
  F = (2, 6) →
  D' = (1, 3/2) →
  E' = (2, 5/2) →
  F' = (3, 1) →
  (D'.2 - D.2) / (D'.1 - D.1) ≠ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_reflection_slope_l3910_391065


namespace NUMINAMATH_CALUDE_log_equality_implies_product_l3910_391001

theorem log_equality_implies_product (x y : ℝ) :
  x > 1 →
  y > 1 →
  (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  x^2 * y^2 = 225^Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_product_l3910_391001


namespace NUMINAMATH_CALUDE_no_reverse_equal_base6_l3910_391085

/-- Function to reverse the digits of a natural number in base 10 --/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Function to convert a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Theorem stating that no natural number greater than 5 has its reversed decimal representation equal to its base 6 representation --/
theorem no_reverse_equal_base6 :
  ∀ n : ℕ, n > 5 → reverseDigits n ≠ toBase6 n :=
sorry

end NUMINAMATH_CALUDE_no_reverse_equal_base6_l3910_391085


namespace NUMINAMATH_CALUDE_triangle_area_is_twelve_l3910_391068

/-- The area of a triangle formed by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangle -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_twelve :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ > 0 ∧ y₂ > 0 ∧
    lineEquation x₁ 0 ∧
    lineEquation 0 y₂ ∧
    (1/2 : ℝ) * x₁ * y₂ = triangleArea :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_twelve_l3910_391068


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3910_391093

/-- Given a geometric sequence where the last four terms are a, b, 243, 729,
    prove that the first term of the sequence is 3. -/
theorem geometric_sequence_first_term
  (a b : ℝ)
  (h1 : ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ 243 = b * r ∧ 729 = 243 * r)
  : ∃ (n : ℕ), 3 * (a / 243) ^ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3910_391093


namespace NUMINAMATH_CALUDE_base_conversion_arithmetic_l3910_391042

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ :=
  (x + 1/2).floor

theorem base_conversion_arithmetic : 
  let base8_2468 := toBase10 [8, 6, 4, 2] 8
  let base4_110 := toBase10 [0, 1, 1] 4
  let base9_3571 := toBase10 [1, 7, 5, 3] 9
  let base10_1357 := 1357
  roundToNearest (base8_2468 / base4_110) - base9_3571 + base10_1357 = -1232 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_arithmetic_l3910_391042


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3910_391073

theorem rectangular_box_surface_area 
  (x y z : ℝ) 
  (h1 : 4 * x + 4 * y + 4 * z = 160) 
  (h2 : Real.sqrt (x^2 + y^2 + z^2) = 25) : 
  2 * (x * y + y * z + z * x) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3910_391073


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l3910_391058

/-- Given three numbers 2, m, 8 forming a geometric sequence, 
    the eccentricity of the conic section x^2/m + y^2/2 = 1 is either √2/2 or √3 -/
theorem conic_section_eccentricity (m : ℝ) :
  (2 * m = m * 8) →
  let e := if m > 0 then Real.sqrt (1 - 2 / m) else Real.sqrt (1 + m / 2)
  e = Real.sqrt 2 / 2 ∨ e = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l3910_391058


namespace NUMINAMATH_CALUDE_square_area_ratio_l3910_391096

theorem square_area_ratio (x : ℝ) (h : x > 0) :
  (x^2) / ((3*x)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3910_391096


namespace NUMINAMATH_CALUDE_old_edition_pages_l3910_391021

theorem old_edition_pages (new_edition : ℕ) (old_edition : ℕ) 
  (h1 : new_edition = 450) 
  (h2 : new_edition = 2 * old_edition - 230) : old_edition = 340 := by
  sorry

end NUMINAMATH_CALUDE_old_edition_pages_l3910_391021


namespace NUMINAMATH_CALUDE_moses_percentage_l3910_391039

theorem moses_percentage (total : ℝ) (moses_amount : ℝ) (esther_amount : ℝ) : 
  total = 50 ∧
  moses_amount = esther_amount + 5 ∧
  moses_amount + 2 * esther_amount = total →
  moses_amount / total = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_moses_percentage_l3910_391039


namespace NUMINAMATH_CALUDE_additional_blurays_is_six_l3910_391066

/-- Represents the movie collection and purchase scenario -/
structure MovieCollection where
  initialDVDRatio : Nat
  initialBluRayRatio : Nat
  newDVDRatio : Nat
  newBluRayRatio : Nat
  totalInitialMovies : Nat

/-- Calculates the number of additional Blu-ray movies purchased -/
def additionalBluRays (mc : MovieCollection) : Nat :=
  let initialX := mc.totalInitialMovies / (mc.initialDVDRatio + mc.initialBluRayRatio)
  let initialDVD := mc.initialDVDRatio * initialX
  let initialBluRay := mc.initialBluRayRatio * initialX
  ((initialDVD * mc.newBluRayRatio) - (initialBluRay * mc.newDVDRatio)) / mc.newDVDRatio

/-- Theorem stating that the number of additional Blu-ray movies purchased is 6 -/
theorem additional_blurays_is_six (mc : MovieCollection) 
  (h1 : mc.initialDVDRatio = 7)
  (h2 : mc.initialBluRayRatio = 2)
  (h3 : mc.newDVDRatio = 13)
  (h4 : mc.newBluRayRatio = 4)
  (h5 : mc.totalInitialMovies = 351) :
  additionalBluRays mc = 6 := by
  sorry

#eval additionalBluRays { initialDVDRatio := 7, initialBluRayRatio := 2, 
                          newDVDRatio := 13, newBluRayRatio := 4, 
                          totalInitialMovies := 351 }

end NUMINAMATH_CALUDE_additional_blurays_is_six_l3910_391066


namespace NUMINAMATH_CALUDE_orange_ribbons_l3910_391091

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 8 →
  silver = 45 →
  yellow + purple + orange + silver = total →
  orange = 19 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l3910_391091


namespace NUMINAMATH_CALUDE_jessie_weight_l3910_391002

/-- Jessie's weight problem -/
theorem jessie_weight (initial_weight lost_weight : ℕ) (h1 : initial_weight = 74) (h2 : lost_weight = 7) :
  initial_weight - lost_weight = 67 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_l3910_391002


namespace NUMINAMATH_CALUDE_unique_rational_root_l3910_391083

def f (x : ℚ) : ℚ := 6 * x^5 - 4 * x^4 - 16 * x^3 + 8 * x^2 + 4 * x - 3

theorem unique_rational_root :
  ∀ x : ℚ, f x = 0 ↔ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_unique_rational_root_l3910_391083


namespace NUMINAMATH_CALUDE_monotonic_cubic_implies_m_range_l3910_391077

/-- A function f : ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = x^3 + x^2 + mx + 1 is monotonic on ℝ, then m ≥ 1/3. -/
theorem monotonic_cubic_implies_m_range (m : ℝ) :
  Monotonic (fun x => x^3 + x^2 + m*x + 1) → m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_cubic_implies_m_range_l3910_391077


namespace NUMINAMATH_CALUDE_henry_tic_tac_toe_games_l3910_391009

theorem henry_tic_tac_toe_games (wins losses draws : ℕ) 
  (h_wins : wins = 2)
  (h_losses : losses = 2)
  (h_draws : draws = 10) :
  wins + losses + draws = 14 := by
  sorry

end NUMINAMATH_CALUDE_henry_tic_tac_toe_games_l3910_391009


namespace NUMINAMATH_CALUDE_fraction_equality_l3910_391016

theorem fraction_equality (x : ℝ) (h : x / (x^2 + x - 1) = 1 / 7) :
  x^2 / (x^4 - x^2 + 1) = 1 / 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3910_391016


namespace NUMINAMATH_CALUDE_meaningful_expression_l3910_391010

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3910_391010


namespace NUMINAMATH_CALUDE_total_erasers_l3910_391061

/-- Given an initial number of erasers and a number of erasers added, 
    the total number of erasers is equal to the sum of the initial number and the added number. -/
theorem total_erasers (initial_erasers added_erasers : ℕ) :
  initial_erasers + added_erasers = initial_erasers + added_erasers :=
by sorry

end NUMINAMATH_CALUDE_total_erasers_l3910_391061


namespace NUMINAMATH_CALUDE_card_area_reduction_l3910_391071

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The theorem to be proved --/
theorem card_area_reduction (initial : Rectangle) :
  initial.length = 5 ∧ initial.width = 8 →
  ∃ (reduced : Rectangle),
    (reduced.length = initial.length - 2 ∨ reduced.width = initial.width - 2) ∧
    area reduced = 21 →
  ∃ (other_reduced : Rectangle),
    (other_reduced.length = initial.length - 2 ∨ other_reduced.width = initial.width - 2) ∧
    other_reduced ≠ reduced ∧
    area other_reduced = 24 := by
  sorry

end NUMINAMATH_CALUDE_card_area_reduction_l3910_391071


namespace NUMINAMATH_CALUDE_min_reciprocal_81_l3910_391072

/-- The reciprocal function -/
def reciprocal (x : ℚ) : ℚ := 1 / x

/-- Apply the reciprocal function n times -/
def apply_reciprocal (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => reciprocal (apply_reciprocal x n)

/-- Theorem: The minimum number of times to apply the reciprocal function to 81 to return to 81 is 2 -/
theorem min_reciprocal_81 :
  (∃ n : ℕ, apply_reciprocal 81 n = 81 ∧ n > 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < 2 → apply_reciprocal 81 m ≠ 81) ∧
  apply_reciprocal 81 2 = 81 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_81_l3910_391072


namespace NUMINAMATH_CALUDE_f_properties_l3910_391043

noncomputable def f (x : ℝ) := (1 + Real.sqrt 3 * Real.tan x) * (Real.cos x)^2

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 → ∃ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) → f x ∈ Set.Ioc 0 (3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3910_391043


namespace NUMINAMATH_CALUDE_arc_length_120_degrees_l3910_391041

/-- Given a circle with circumference 90 meters and an arc subtended by a 120° central angle,
    prove that the length of the arc is 30 meters. -/
theorem arc_length_120_degrees (circle_circumference : ℝ) (central_angle : ℝ) (arc_length : ℝ) :
  circle_circumference = 90 →
  central_angle = 120 →
  arc_length = (central_angle / 360) * circle_circumference →
  arc_length = 30 := by
sorry

end NUMINAMATH_CALUDE_arc_length_120_degrees_l3910_391041


namespace NUMINAMATH_CALUDE_savings_amount_correct_l3910_391099

def calculate_savings (lightweight_price medium_price heavyweight_price : ℚ)
  (home_lightweight grandparents_medium_factor neighbor_heavyweight : ℕ)
  (dad_total dad_lightweight_percent dad_medium_percent dad_heavyweight_percent : ℚ) : ℚ :=
  let home_medium := home_lightweight * grandparents_medium_factor
  let dad_lightweight := dad_total * dad_lightweight_percent
  let dad_medium := dad_total * dad_medium_percent
  let dad_heavyweight := dad_total * dad_heavyweight_percent
  let total_amount := 
    lightweight_price * (home_lightweight + dad_lightweight) +
    medium_price * (home_medium + dad_medium) +
    heavyweight_price * (neighbor_heavyweight + dad_heavyweight)
  total_amount / 2

theorem savings_amount_correct :
  calculate_savings 0.15 0.25 0.35 12 3 46 250 0.5 0.3 0.2 = 41.45 :=
by sorry

end NUMINAMATH_CALUDE_savings_amount_correct_l3910_391099


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3910_391026

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3910_391026


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l3910_391019

/-- Proves that the cost of each movie ticket is $10.62 --/
theorem movie_ticket_cost (ticket_count : ℕ) (rental_cost movie_cost total_cost : ℚ) :
  ticket_count = 2 →
  rental_cost = 1.59 →
  movie_cost = 13.95 →
  total_cost = 36.78 →
  ∃ (ticket_price : ℚ), 
    ticket_price * ticket_count + rental_cost + movie_cost = total_cost ∧
    ticket_price = 10.62 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l3910_391019


namespace NUMINAMATH_CALUDE_correct_sales_growth_equation_l3910_391024

/-- Represents the growth of new energy vehicle sales over two months -/
def sales_growth (initial_sales : ℝ) (final_sales : ℝ) (growth_rate : ℝ) : Prop :=
  initial_sales * (1 + growth_rate)^2 = final_sales

/-- Theorem stating that the given equation correctly represents the sales growth -/
theorem correct_sales_growth_equation :
  ∃ x : ℝ, sales_growth 33.2 54.6 x :=
sorry

end NUMINAMATH_CALUDE_correct_sales_growth_equation_l3910_391024


namespace NUMINAMATH_CALUDE_train_length_problem_l3910_391089

/-- Given a bridge length, train speed, and time to pass over the bridge, 
    calculate the length of the train. -/
def train_length (bridge_length : ℝ) (train_speed : ℝ) (time_to_pass : ℝ) : ℝ :=
  train_speed * time_to_pass - bridge_length

/-- Theorem stating that under the given conditions, the train length is 400 meters. -/
theorem train_length_problem : 
  let bridge_length : ℝ := 2800
  let train_speed : ℝ := 800
  let time_to_pass : ℝ := 4
  train_length bridge_length train_speed time_to_pass = 400 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l3910_391089


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l3910_391045

/-- A quadratic function f(x) = ax^2 + bx + c with vertex (4, 10) and one x-intercept at (1, 0) -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem other_x_intercept_of_quadratic 
  (a b c : ℝ) 
  (h_vertex : QuadraticFunction a b c 4 = 10 ∧ (∀ x, QuadraticFunction a b c x ≥ 10 ∨ QuadraticFunction a b c x ≤ 10))
  (h_intercept : QuadraticFunction a b c 1 = 0) :
  ∃ x, x ≠ 1 ∧ QuadraticFunction a b c x = 0 ∧ x = 7 := by
sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l3910_391045


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3910_391064

/-- Given an incident ray along the line 2x - y + 2 = 0 reflected off the y-axis,
    the equation of the line containing the reflected ray is 2x + y - 2 = 0 -/
theorem reflected_ray_equation (x y : ℝ) :
  (2 * x - y + 2 = 0) →  -- incident ray equation
  (∃ (x' y' : ℝ), 2 * x' + y' - 2 = 0) -- reflected ray equation
  := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3910_391064


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3910_391080

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 8) - (2 * x^3 + x^2 + 3 * x - 15) = x^3 + 3 * x^2 - 8 * x + 23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3910_391080


namespace NUMINAMATH_CALUDE_area_maximized_at_m_pm1_l3910_391015

/-- Ellipse E with equation x²/6 + y²/2 = 1 -/
def ellipse_E (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

/-- Focus F₁ at (-2, 0) -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- Line l with equation x - my - 2 = 0 -/
def line_l (m x y : ℝ) : Prop := x - m * y - 2 = 0

/-- Intersection points of ellipse E and line l -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse_E p.1 p.2 ∧ line_l m p.1 p.2}

/-- Area of quadrilateral AF₁BC -/
noncomputable def area_AF₁BC (m : ℝ) : ℝ := sorry

/-- Theorem: Area of AF₁BC is maximized when m = ±1 -/
theorem area_maximized_at_m_pm1 :
  ∀ m : ℝ, area_AF₁BC m ≤ area_AF₁BC 1 ∧ area_AF₁BC m ≤ area_AF₁BC (-1) :=
sorry

end NUMINAMATH_CALUDE_area_maximized_at_m_pm1_l3910_391015


namespace NUMINAMATH_CALUDE_intercepted_line_with_midpoint_at_origin_l3910_391067

/-- Given two lines l₁ and l₂, prove that the line x + 6y = 0 is intercepted by both lines
    and has its midpoint at the origin. -/
theorem intercepted_line_with_midpoint_at_origin :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 4*x + y + 6 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 3*x - 5*y - 6 = 0
  let intercepted_line : ℝ → ℝ → Prop := λ x y ↦ x + 6*y = 0
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧
    intercepted_line x₁ y₁ ∧ intercepted_line x₂ y₂ ∧
    (x₁ + x₂) / 2 = 0 ∧ (y₁ + y₂) / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_intercepted_line_with_midpoint_at_origin_l3910_391067


namespace NUMINAMATH_CALUDE_usable_seats_in_section_C_l3910_391088

def x : ℝ := 60 + 3 * 80
def y : ℝ := 3 * x + 20
def z : ℝ := 2 * y - 30.5

theorem usable_seats_in_section_C : z = 1809.5 := by
  sorry

end NUMINAMATH_CALUDE_usable_seats_in_section_C_l3910_391088


namespace NUMINAMATH_CALUDE_exp_13pi_over_2_equals_i_l3910_391030

-- Define Euler's formula
axiom euler_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ

-- State the theorem
theorem exp_13pi_over_2_equals_i : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_exp_13pi_over_2_equals_i_l3910_391030


namespace NUMINAMATH_CALUDE_inequality_solution_l3910_391011

theorem inequality_solution (x : ℝ) :
  x + 1 > 0 →
  x + 1 - Real.sqrt (x + 1) ≠ 0 →
  (x^2 / ((x + 1 - Real.sqrt (x + 1))^2) < (x^2 + 3*x + 18) / (x + 1)^2) ↔ 
  (-1 < x ∧ x < 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3910_391011


namespace NUMINAMATH_CALUDE_not_decreasing_if_f0_lt_f4_l3910_391046

def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

theorem not_decreasing_if_f0_lt_f4 (f : ℝ → ℝ) (h : f 0 < f 4) : ¬ IsDecreasing f := by
  sorry

end NUMINAMATH_CALUDE_not_decreasing_if_f0_lt_f4_l3910_391046


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_140_l3910_391027

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_140 :
  rectangle_area 1225 10 = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_140_l3910_391027


namespace NUMINAMATH_CALUDE_grape_lollipops_count_l3910_391050

/-- Given a total number of lollipops and the number of flavors for non-cherry lollipops,
    calculate the number of lollipops of a specific non-cherry flavor. -/
def grape_lollipops (total : ℕ) (non_cherry_flavors : ℕ) : ℕ :=
  (total / 2) / non_cherry_flavors

/-- Theorem stating that with 42 total lollipops and 3 non-cherry flavors,
    the number of grape lollipops is 7. -/
theorem grape_lollipops_count :
  grape_lollipops 42 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_grape_lollipops_count_l3910_391050


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3910_391017

theorem fraction_equation_solution (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 2) (hy1 : y ≠ 0) (hy2 : y ≠ 3) :
  3 / x + 2 / y = 5 / 6 → x = 18 * y / (5 * y - 12) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3910_391017


namespace NUMINAMATH_CALUDE_games_played_calculation_l3910_391079

/-- Represents the number of games played by a baseball team --/
def GamesPlayed : ℕ → ℕ → ℕ → ℕ
  | games_won, games_left, games_to_win_more => games_won + games_left - games_to_win_more

/-- Represents the total number of games in a season --/
def TotalGames : ℕ → ℕ → ℕ
  | games_played, games_left => games_played + games_left

theorem games_played_calculation (games_won : ℕ) (games_left : ℕ) (games_to_win_more : ℕ) :
  games_won = 12 →
  games_left = 10 →
  games_to_win_more = 8 →
  (3 * (games_won + games_to_win_more) = 2 * TotalGames (GamesPlayed games_won games_left games_to_win_more) games_left) →
  GamesPlayed games_won games_left games_to_win_more = 20 := by
  sorry

end NUMINAMATH_CALUDE_games_played_calculation_l3910_391079


namespace NUMINAMATH_CALUDE_statue_increase_factor_l3910_391076

/-- The factor by which the number of statues increased in the second year --/
def increase_factor : ℝ := 4

/-- The initial number of statues --/
def initial_statues : ℕ := 4

/-- The number of statues added in the third year --/
def added_third_year : ℕ := 12

/-- The number of statues broken in the third year --/
def broken_third_year : ℕ := 3

/-- The final number of statues after four years --/
def final_statues : ℕ := 31

theorem statue_increase_factor : 
  (initial_statues : ℝ) * increase_factor + 
  (added_third_year : ℝ) - (broken_third_year : ℝ) + 
  2 * (broken_third_year : ℝ) = (final_statues : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_statue_increase_factor_l3910_391076


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_configuration_l3910_391029

/-- The area of the shaded region in a circle configuration --/
theorem shaded_area_in_circle_configuration (R : ℝ) (h : R = 9) :
  let r : ℝ := R / 2
  let larger_circle_area : ℝ := π * R^2
  let smaller_circle_area : ℝ := π * r^2
  let total_smaller_circles_area : ℝ := 3 * smaller_circle_area
  let shaded_area : ℝ := larger_circle_area - total_smaller_circles_area
  shaded_area = 20.25 * π :=
by sorry


end NUMINAMATH_CALUDE_shaded_area_in_circle_configuration_l3910_391029


namespace NUMINAMATH_CALUDE_x_value_is_five_l3910_391006

theorem x_value_is_five (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_is_five_l3910_391006


namespace NUMINAMATH_CALUDE_root_domain_implies_a_bound_l3910_391028

/-- The equation has real roots for m in this set -/
def A : Set ℝ := {m | ∃ x, (m + 1) * x^2 - m * x + m - 1 = 0}

/-- The domain of the function f(x) -/
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a > 0}

/-- The main theorem -/
theorem root_domain_implies_a_bound (a : ℝ) : A ⊆ B a → a > 2/3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_root_domain_implies_a_bound_l3910_391028


namespace NUMINAMATH_CALUDE_exists_g_for_f_l3910_391086

-- Define the function f: ℝ² → ℝ
variable (f : ℝ × ℝ → ℝ)

-- State the condition for f
axiom f_condition : ∀ (x y z : ℝ), f (x, y) + f (y, z) + f (z, x) = 0

-- Theorem statement
theorem exists_g_for_f : 
  ∃ (g : ℝ → ℝ), ∀ (x y : ℝ), f (x, y) = g x - g y := by sorry

end NUMINAMATH_CALUDE_exists_g_for_f_l3910_391086


namespace NUMINAMATH_CALUDE_boys_in_basketball_camp_l3910_391032

theorem boys_in_basketball_camp (total : ℕ) (boy_ratio girl_ratio : ℕ) (boys girls : ℕ) : 
  total = 48 →
  boy_ratio = 3 →
  girl_ratio = 5 →
  boys + girls = total →
  boy_ratio * girls = girl_ratio * boys →
  boys = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_in_basketball_camp_l3910_391032


namespace NUMINAMATH_CALUDE_improved_running_distance_l3910_391075

/-- Proves that if a person can run 40 yards in 5 seconds and improves their speed by 40%, 
    they can run 112 yards in 10 seconds. -/
theorem improved_running_distance 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (improvement_percentage : ℝ) 
  (new_time : ℝ) :
  initial_distance = 40 ∧ 
  initial_time = 5 ∧ 
  improvement_percentage = 40 ∧ 
  new_time = 10 →
  (initial_distance + initial_distance * (improvement_percentage / 100)) * (new_time / initial_time) = 112 :=
by sorry

end NUMINAMATH_CALUDE_improved_running_distance_l3910_391075


namespace NUMINAMATH_CALUDE_money_division_l3910_391081

/-- The problem of dividing money among three children -/
theorem money_division (anusha babu esha : ℕ) : 
  12 * anusha = 8 * babu ∧ 
  8 * babu = 6 * esha ∧ 
  anusha = 84 → 
  anusha + babu + esha = 378 := by
  sorry


end NUMINAMATH_CALUDE_money_division_l3910_391081


namespace NUMINAMATH_CALUDE_three_more_than_twice_x_l3910_391003

/-- The algebraic expression for a number that is 3 more than twice x is 2x + 3. -/
theorem three_more_than_twice_x (x : ℝ) : 2 * x + 3 = 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_three_more_than_twice_x_l3910_391003


namespace NUMINAMATH_CALUDE_elon_has_13_teslas_l3910_391014

/-- The number of teslas Chris has -/
def chris_teslas : ℕ := 6

/-- The number of teslas Sam has -/
def sam_teslas : ℕ := chris_teslas / 2

/-- The number of teslas Elon has -/
def elon_teslas : ℕ := sam_teslas + 10

/-- Theorem stating that Elon has 13 teslas -/
theorem elon_has_13_teslas : elon_teslas = 13 := by
  sorry

end NUMINAMATH_CALUDE_elon_has_13_teslas_l3910_391014


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l3910_391007

/-- Represents the number of wheels for each vehicle type -/
structure VehicleWheels where
  car : Nat
  bike : Nat
  truck : Nat
  bus : Nat

/-- Represents the count of each vehicle type in the parking lot -/
structure VehicleCount where
  cars : Nat
  bikes : Nat
  trucks : Nat
  buses : Nat

/-- Calculates the total number of wheels in the parking lot -/
def totalWheels (wheels : VehicleWheels) (count : VehicleCount) : Nat :=
  wheels.car * count.cars +
  wheels.bike * count.bikes +
  wheels.truck * count.trucks +
  wheels.bus * count.buses

/-- Theorem: The total number of wheels in the parking lot is 156 -/
theorem parking_lot_wheels :
  let wheels : VehicleWheels := ⟨4, 2, 8, 6⟩
  let count : VehicleCount := ⟨14, 10, 7, 4⟩
  totalWheels wheels count = 156 := by
  sorry

#check parking_lot_wheels

end NUMINAMATH_CALUDE_parking_lot_wheels_l3910_391007


namespace NUMINAMATH_CALUDE_flight_duration_sum_l3910_391062

-- Define the flight departure and arrival times in minutes since midnight
def departure_time : ℕ := 10 * 60 + 34
def arrival_time : ℕ := 13 * 60 + 18

-- Define the flight duration in hours and minutes
def flight_duration (h m : ℕ) : Prop :=
  h * 60 + m = arrival_time - departure_time ∧ 0 < m ∧ m < 60

-- Theorem statement
theorem flight_duration_sum :
  ∃ (h m : ℕ), flight_duration h m ∧ h + m = 46 :=
sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l3910_391062


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3910_391063

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of a hyperbola is parallel to a line -/
def asymptote_parallel (h : Hyperbola) (m : ℝ) : Prop :=
  ∃ (c : ℝ), ∀ (x y : ℝ), y = m * x + c → (x^2 - y^2 / h.b^2 = 1 → False)

theorem hyperbola_asymptote_slope (h : Hyperbola) 
  (parallel : asymptote_parallel h 2) : h.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3910_391063


namespace NUMINAMATH_CALUDE_range_of_a_l3910_391040

/-- The function f(x) = |x+a| + |x-2| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

/-- The solution set A for f(x) ≤ |x-4| -/
def A (a : ℝ) : Set ℝ := {x | f a x ≤ |x - 4|}

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x ∈ A a) → a ∈ Set.Icc (-3) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3910_391040


namespace NUMINAMATH_CALUDE_total_cans_stored_l3910_391033

/-- Represents a closet with its storage capacity -/
structure Closet where
  cansPerRow : Nat
  rowsPerShelf : Nat
  numShelves : Nat

/-- Calculates the total number of cans that can be stored in a closet -/
def cansInCloset (c : Closet) : Nat :=
  c.cansPerRow * c.rowsPerShelf * c.numShelves

/-- The first closet in Jack's emergency bunker -/
def closet1 : Closet := ⟨12, 4, 10⟩

/-- The second closet in Jack's emergency bunker -/
def closet2 : Closet := ⟨15, 5, 8⟩

/-- Theorem stating the total number of cans Jack can store -/
theorem total_cans_stored : cansInCloset closet1 + cansInCloset closet2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_stored_l3910_391033


namespace NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l3910_391060

theorem sum_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (fun i => 12 * (i + 1)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l3910_391060


namespace NUMINAMATH_CALUDE_special_hexagon_angle_sum_l3910_391056

/-- A hexagon with specific angle properties -/
structure SpecialHexagon where
  /-- Exterior angle measuring 40° -/
  ext_angle : ℝ
  /-- First interior angle measuring 45° -/
  int_angle1 : ℝ
  /-- Second interior angle measuring 80° -/
  int_angle2 : ℝ
  /-- Third interior angle -/
  int_angle3 : ℝ
  /-- Fourth interior angle -/
  int_angle4 : ℝ
  /-- Condition: Exterior angle is 40° -/
  h1 : ext_angle = 40
  /-- Condition: First interior angle is 45° -/
  h2 : int_angle1 = 45
  /-- Condition: Second interior angle is 80° -/
  h3 : int_angle2 = 80
  /-- Sum of interior angles of a hexagon is 720° -/
  h4 : int_angle1 + int_angle2 + int_angle3 + int_angle4 + (180 - ext_angle) + 90 = 720

/-- The sum of the third and fourth interior angles is 15° -/
theorem special_hexagon_angle_sum (h : SpecialHexagon) : h.int_angle3 + h.int_angle4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_angle_sum_l3910_391056


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l3910_391051

theorem sufficiency_not_necessity (p q : Prop) : 
  (¬p ∧ ¬q → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ ¬(¬p ∧ ¬q) := by
sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l3910_391051


namespace NUMINAMATH_CALUDE_min_value_system_min_value_exact_l3910_391035

open Real

theorem min_value_system (x y z : ℝ) 
  (eq1 : 2 * cos x = 1 / tan y)
  (eq2 : 2 * sin y = tan z)
  (eq3 : cos z = 1 / tan x) :
  ∀ (a b c : ℝ), 
    (2 * cos a = 1 / tan b) → 
    (2 * sin b = tan c) → 
    (cos c = 1 / tan a) → 
    sin x + cos z ≤ sin a + cos c :=
by sorry

theorem min_value_exact (x y z : ℝ) 
  (eq1 : 2 * cos x = 1 / tan y)
  (eq2 : 2 * sin y = tan z)
  (eq3 : cos z = 1 / tan x) :
  ∃ (a b c : ℝ), 
    (2 * cos a = 1 / tan b) ∧ 
    (2 * sin b = tan c) ∧ 
    (cos c = 1 / tan a) ∧ 
    sin a + cos c = -5 * Real.sqrt 3 / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_system_min_value_exact_l3910_391035


namespace NUMINAMATH_CALUDE_last_ball_is_white_l3910_391082

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the state of the box -/
structure BoxState where
  white : Nat
  black : Nat

/-- The process of drawing and replacing balls -/
def process (state : BoxState) : BoxState :=
  sorry

/-- Predicate to check if the process has terminated (only one ball left) -/
def isTerminated (state : BoxState) : Prop :=
  state.white + state.black = 1

/-- Theorem stating that the last ball is white given an odd initial number of white balls -/
theorem last_ball_is_white 
  (initial_white : Nat) 
  (initial_black : Nat) 
  (h_odd : Odd initial_white) :
  ∃ (final_state : BoxState), 
    (∃ (n : Nat), final_state = (process^[n] ⟨initial_white, initial_black⟩)) ∧ 
    isTerminated final_state ∧ 
    final_state.white = 1 :=
  sorry

end NUMINAMATH_CALUDE_last_ball_is_white_l3910_391082


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l3910_391095

theorem smaller_integer_problem (x y : ℤ) : 
  y = 2 * x → x + y = 96 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l3910_391095


namespace NUMINAMATH_CALUDE_difference_equals_1011_l3910_391097

/-- The sum of consecutive odd numbers from 1 to 2021 -/
def sum_odd : ℕ := (2021 + 1) / 2 ^ 2

/-- The sum of consecutive even numbers from 2 to 2020 -/
def sum_even : ℕ := (2020 / 2) * (2020 / 2 + 1)

/-- The difference between the sum of odd numbers and the sum of even numbers -/
def difference : ℕ := sum_odd - sum_even

theorem difference_equals_1011 : difference = 1011 := by
  sorry

end NUMINAMATH_CALUDE_difference_equals_1011_l3910_391097


namespace NUMINAMATH_CALUDE_estimate_greater_than_exact_l3910_391094

theorem estimate_greater_than_exact 
  (a b c a' b' c' : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≤ c) :
  (a' / b') - c' > (a / b) - c :=
sorry

end NUMINAMATH_CALUDE_estimate_greater_than_exact_l3910_391094


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3910_391031

theorem quadratic_equations_solutions :
  -- Equation 1
  (∃ x : ℝ, x^2 - 4 = 0) ∧
  (∀ x : ℝ, x^2 - 4 = 0 → x = 2 ∨ x = -2) ∧
  -- Equation 2
  (∃ x : ℝ, x^2 - 6*x + 9 = 0) ∧
  (∀ x : ℝ, x^2 - 6*x + 9 = 0 → x = 3) ∧
  -- Equation 3
  (∃ x : ℝ, x^2 - 7*x + 12 = 0) ∧
  (∀ x : ℝ, x^2 - 7*x + 12 = 0 → x = 3 ∨ x = 4) ∧
  -- Equation 4
  (∃ x : ℝ, 2*x^2 - 3*x = 5) ∧
  (∀ x : ℝ, 2*x^2 - 3*x = 5 → x = 5/2 ∨ x = -1) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3910_391031


namespace NUMINAMATH_CALUDE_cascade_properties_l3910_391070

/-- A cascade generated by a natural number r -/
def Cascade (r : ℕ) : Finset ℕ :=
  Finset.image (λ i => i * r) (Finset.range 12)

/-- The property that a pair of natural numbers belongs to exactly six cascades -/
def BelongsToSixCascades (a b : ℕ) : Prop :=
  ∃ (cascades : Finset ℕ), cascades.card = 6 ∧
    ∀ r ∈ cascades, a ∈ Cascade r ∧ b ∈ Cascade r

/-- A coloring function from natural numbers to 12 colors -/
def ColoringFunction := ℕ → Fin 12

/-- The property that a coloring function assigns different colors to all elements in any cascade -/
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ r : ℕ, ∀ i j : Fin 12, i ≠ j → f (r * (i.val + 1)) ≠ f (r * (j.val + 1))

theorem cascade_properties :
  (∃ a b : ℕ, BelongsToSixCascades a b) ∧
  (∃ f : ColoringFunction, ValidColoring f) := by sorry

end NUMINAMATH_CALUDE_cascade_properties_l3910_391070
