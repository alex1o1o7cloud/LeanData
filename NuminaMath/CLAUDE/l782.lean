import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_difference_45_15_l782_78258

theorem sin_cos_difference_45_15 :
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) -
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_45_15_l782_78258


namespace NUMINAMATH_CALUDE_largest_integer_l782_78250

theorem largest_integer (x y z w : ℤ) 
  (sum1 : x + y + z = 234)
  (sum2 : x + y + w = 255)
  (sum3 : x + z + w = 271)
  (sum4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_l782_78250


namespace NUMINAMATH_CALUDE_conference_theorem_l782_78241

def conference_problem (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_tea tea_soda coffee_soda : ℕ) (all_three : ℕ) : Prop :=
  let drank_at_least_one := coffee + tea + soda - coffee_tea - tea_soda - coffee_soda + all_three
  total - drank_at_least_one = 5

theorem conference_theorem : 
  conference_problem 30 15 13 9 7 4 3 2 := by sorry

end NUMINAMATH_CALUDE_conference_theorem_l782_78241


namespace NUMINAMATH_CALUDE_small_sphere_radius_l782_78271

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Checks if two spheres are externally tangent -/
def are_externally_tangent (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- The configuration of 5 spheres as described in the problem -/
structure SpheresConfiguration where
  s1 : Sphere
  s2 : Sphere
  s3 : Sphere
  s4 : Sphere
  small : Sphere
  h1 : s1.radius = 2
  h2 : s2.radius = 2
  h3 : s3.radius = 3
  h4 : s4.radius = 3
  h5 : are_externally_tangent s1 s2
  h6 : are_externally_tangent s1 s3
  h7 : are_externally_tangent s1 s4
  h8 : are_externally_tangent s2 s3
  h9 : are_externally_tangent s2 s4
  h10 : are_externally_tangent s3 s4
  h11 : are_externally_tangent s1 small
  h12 : are_externally_tangent s2 small
  h13 : are_externally_tangent s3 small
  h14 : are_externally_tangent s4 small

/-- The main theorem stating that the radius of the small sphere is 6/11 -/
theorem small_sphere_radius (config : SpheresConfiguration) : config.small.radius = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_small_sphere_radius_l782_78271


namespace NUMINAMATH_CALUDE_horner_rule_v2_value_l782_78221

/-- Horner's Rule evaluation function -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x⁵ + 4x⁴ + x² + 20x + 16 -/
def f : ℝ → ℝ := fun x => x^5 + 4*x^4 + x^2 + 20*x + 16

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 4, 0, 1, 20, 16]

theorem horner_rule_v2_value :
  let x := -2
  let v₂ := (horner_eval (f_coeffs.take 3) x)
  v₂ = -4 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v2_value_l782_78221


namespace NUMINAMATH_CALUDE_not_lucky_1982_1983_l782_78219

/-- Checks if a given year is a lucky year -/
def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat),
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem not_lucky_1982_1983 :
  ¬(isLuckyYear 1982) ∧ ¬(isLuckyYear 1983) :=
by sorry

end NUMINAMATH_CALUDE_not_lucky_1982_1983_l782_78219


namespace NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l782_78270

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  center : Point

/-- Returns true if the given point is a focus of the hyperbola -/
def isFocus (h : Hyperbola) (p : Point) : Prop :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  (p.x = h.center.x - c ∧ p.y = h.center.y) ∨
  (p.x = h.center.x + c ∧ p.y = h.center.y)

/-- Returns true if p1 has a smaller x-coordinate than p2 -/
def hasSmaller_x (p1 p2 : Point) : Prop :=
  p1.x < p2.x

theorem hyperbola_focus_smaller_x (h : Hyperbola) :
  h.a = 7 ∧ h.b = 3 ∧ h.center = { x := 1, y := -8 } →
  ∃ (f : Point), isFocus h f ∧ ∀ (f' : Point), isFocus h f' → hasSmaller_x f f' ∨ f = f' →
  f = { x := 1 - Real.sqrt 58, y := -8 } := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l782_78270


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l782_78251

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3 * y = 1 → 1 / a + 2 / b ≤ 1 / x + 2 / y) ∧
  1 / a + 2 / b = 3 * Real.sqrt 6 + 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l782_78251


namespace NUMINAMATH_CALUDE_A_power_50_l782_78232

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; -4, -1]

theorem A_power_50 : 
  A^50 = 50 * 8^49 • A - 399 * 8^49 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_A_power_50_l782_78232


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l782_78237

theorem sum_first_150_remainder (n : ℕ) (h : n = 150) : 
  (n * (n + 1) / 2) % 12000 = 11325 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l782_78237


namespace NUMINAMATH_CALUDE_probability_of_red_ball_in_bag_A_l782_78287

theorem probability_of_red_ball_in_bag_A 
  (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) :
  let total_A := m + n
  let total_B := 7
  let prob_red_A := m / total_A
  let prob_white_A := n / total_A
  let prob_red_B_after_red := (3 + 1) / (total_B + 1)
  let prob_red_B_after_white := 3 / (total_B + 1)
  let total_prob_red := prob_red_A * prob_red_B_after_red + prob_white_A * prob_red_B_after_white
  total_prob_red = 15/32 → prob_red_A = 3/4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_in_bag_A_l782_78287


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l782_78280

-- Problem 1
theorem simplify_expression_1 :
  Real.sqrt 8 + 2 * Real.sqrt 3 - (Real.sqrt 27 - Real.sqrt 2) = 3 * Real.sqrt 2 - Real.sqrt 3 :=
by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (ha : a > 0) :
  Real.sqrt (4 * a^2 * b^3) = 2 * a * b * Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l782_78280


namespace NUMINAMATH_CALUDE_total_marbles_l782_78262

/-- Proves that the total number of marbles is 4.51b given the conditions -/
theorem total_marbles (b : ℝ) (h1 : b > 0) : 
  let r := 1.3 * b
  let g := 1.7 * r
  b + r + g = 4.51 * b := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l782_78262


namespace NUMINAMATH_CALUDE_quadratic_discriminant_relationship_l782_78230

/-- The discriminant of a quadratic equation ax^2 + 2bx + c = 0 is 1 -/
def discriminant_is_one (a b c : ℝ) : Prop :=
  (2 * b)^2 - 4 * a * c = 1

/-- The relationship between a, b, and c -/
def relationship (a b c : ℝ) : Prop :=
  b^2 - a * c = 1/4

/-- Theorem: If the discriminant of ax^2 + 2bx + c = 0 is 1, 
    then b^2 - ac = 1/4 -/
theorem quadratic_discriminant_relationship 
  (a b c : ℝ) : discriminant_is_one a b c → relationship a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_relationship_l782_78230


namespace NUMINAMATH_CALUDE_prob_rain_all_days_l782_78269

def prob_rain_friday : ℚ := 40 / 100
def prob_rain_saturday : ℚ := 50 / 100
def prob_rain_sunday : ℚ := 30 / 100

def events_independent : Prop := True

theorem prob_rain_all_days : 
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday = 6 / 100 :=
by sorry

end NUMINAMATH_CALUDE_prob_rain_all_days_l782_78269


namespace NUMINAMATH_CALUDE_janes_mean_score_l782_78281

def janes_scores : List ℝ := [85, 90, 95, 80, 100]

theorem janes_mean_score : 
  (janes_scores.sum / janes_scores.length : ℝ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_janes_mean_score_l782_78281


namespace NUMINAMATH_CALUDE_dennis_marbles_l782_78274

theorem dennis_marbles (laurie kurt dennis : ℕ) 
  (h1 : laurie = kurt + 12)
  (h2 : kurt + 45 = dennis)
  (h3 : laurie = 37) : 
  dennis = 70 := by
sorry

end NUMINAMATH_CALUDE_dennis_marbles_l782_78274


namespace NUMINAMATH_CALUDE_pascal_triangle_sum_l782_78214

/-- The number of elements in the nth row of Pascal's Triangle -/
def elements_in_row (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def sum_elements (n : ℕ) : ℕ := 
  (n * (n + 1)) / 2

theorem pascal_triangle_sum : sum_elements 25 = 325 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_sum_l782_78214


namespace NUMINAMATH_CALUDE_simple_interest_principle_l782_78284

theorem simple_interest_principle (r t A : ℚ) (h1 : r = 5 / 100) (h2 : t = 12 / 5) (h3 : A = 896) :
  ∃ P : ℚ, P * (1 + r * t) = A ∧ P = 800 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principle_l782_78284


namespace NUMINAMATH_CALUDE_burattino_awake_journey_fraction_l782_78208

theorem burattino_awake_journey_fraction (x : ℝ) (h : x > 0) :
  let distance_before_sleep := x / 2
  let distance_slept := x / 3
  let distance_after_wake := x - (distance_before_sleep + distance_slept)
  distance_after_wake = distance_slept / 2 →
  (distance_before_sleep + distance_after_wake) / x = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_burattino_awake_journey_fraction_l782_78208


namespace NUMINAMATH_CALUDE_yanna_shirt_purchase_l782_78288

theorem yanna_shirt_purchase (shirt_price : ℕ) (sandals_cost : ℕ) (total_spent : ℕ) 
  (h1 : shirt_price = 5)
  (h2 : sandals_cost = 9)
  (h3 : total_spent = 59) :
  ∃ (num_shirts : ℕ), num_shirts * shirt_price + sandals_cost = total_spent ∧ num_shirts = 10 := by
  sorry

end NUMINAMATH_CALUDE_yanna_shirt_purchase_l782_78288


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l782_78236

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

-- Theorem statement
theorem diamond_equation_solution :
  ∃ A : ℝ, diamond A 3 = 85 ∧ A = 17.25 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l782_78236


namespace NUMINAMATH_CALUDE_factorization_problems_l782_78211

theorem factorization_problems :
  (∀ a : ℝ, 4 * a^2 - 9 = (2*a + 3) * (2*a - 3)) ∧
  (∀ x y : ℝ, 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l782_78211


namespace NUMINAMATH_CALUDE_students_history_not_statistics_l782_78294

/-- Given a group of students with the following properties:
  * There are 150 students in total
  * 58 students are taking history
  * 42 students are taking statistics
  * 95 students are taking history or statistics or both
  Then the number of students taking history but not statistics is 53. -/
theorem students_history_not_statistics 
  (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 150 →
  history = 58 →
  statistics = 42 →
  history_or_statistics = 95 →
  history - (history + statistics - history_or_statistics) = 53 := by
sorry

end NUMINAMATH_CALUDE_students_history_not_statistics_l782_78294


namespace NUMINAMATH_CALUDE_negation_of_all_nonnegative_l782_78259

theorem negation_of_all_nonnegative (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_nonnegative_l782_78259


namespace NUMINAMATH_CALUDE_radical_product_simplification_l782_78256

theorem radical_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (98 * x) * Real.sqrt (18 * x) * Real.sqrt (50 * x) = 210 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l782_78256


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l782_78249

theorem nested_square_root_equality : Real.sqrt (13 + Real.sqrt (7 + Real.sqrt 4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l782_78249


namespace NUMINAMATH_CALUDE_three_divisors_of_2469_minus_5_l782_78261

theorem three_divisors_of_2469_minus_5 : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (2469 : ℤ).natAbs % (m^2 - 5 : ℤ).natAbs = 0) ∧ 
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_divisors_of_2469_minus_5_l782_78261


namespace NUMINAMATH_CALUDE_min_rows_required_l782_78248

/-- The number of seats in each row -/
def seats_per_row : ℕ := 168

/-- The total number of students -/
def total_students : ℕ := 2016

/-- The maximum number of students from each school -/
def max_students_per_school : ℕ := 40

/-- Represents the seating arrangement in the arena -/
structure Arena where
  rows : ℕ
  students_seated : ℕ
  school_integrity : Bool  -- True if students from each school are in a single row

/-- A function to check if a seating arrangement is valid -/
def is_valid_arrangement (a : Arena) : Prop :=
  a.students_seated = total_students ∧
  a.school_integrity ∧
  a.rows * seats_per_row ≥ total_students

/-- The main theorem stating the minimum number of rows required -/
theorem min_rows_required : 
  ∀ a : Arena, is_valid_arrangement a → a.rows ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_min_rows_required_l782_78248


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cylinder_l782_78228

/-- Given a cylinder of volume V and a pyramid inscribed in it such that:
    - The base of the pyramid is an isosceles triangle with angle α between equal sides
    - The pyramid's base is inscribed in the base of the cylinder
    - The pyramid's apex coincides with the midpoint of one of the cylinder's generatrices
    Then the volume of the pyramid is (V / (6π)) * sin(α) * cos²(α/2) -/
theorem pyramid_volume_in_cylinder (V : ℝ) (α : ℝ) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cylinder_l782_78228


namespace NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l782_78218

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)

-- State the theorem
theorem line_perp_to_plane_and_line_para_to_plane_implies_lines_perp
  (m n : Line) (α : Plane)
  (h1 : perp m α)
  (h2 : para n α) :
  perpLine m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l782_78218


namespace NUMINAMATH_CALUDE_range_of_expression_l782_78257

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 4 * Real.arcsin x - Real.arccos y ∧ 
  -5 * π / 2 ≤ z ∧ z ≤ 3 * π / 2 ∧
  (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = 1 ∧ 4 * Real.arcsin x₁ - Real.arccos y₁ = -5 * π / 2) ∧
  (∃ (x₂ y₂ : ℝ), x₂^2 + y₂^2 = 1 ∧ 4 * Real.arcsin x₂ - Real.arccos y₂ = 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l782_78257


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l782_78255

theorem coefficient_x4_in_expansion (x : ℝ) : 
  (Finset.range 9).sum (λ k => (Nat.choose 8 k : ℝ) * x^k * (2 * Real.sqrt 3)^(8 - k)) = 
  10080 * x^4 + (Finset.range 9).sum (λ k => if k ≠ 4 then (Nat.choose 8 k : ℝ) * x^k * (2 * Real.sqrt 3)^(8 - k) else 0) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l782_78255


namespace NUMINAMATH_CALUDE_smallest_balloon_count_l782_78244

def balloon_count (color : String) : ℕ :=
  match color with
  | "red" => 10
  | "blue" => 8
  | "yellow" => 5
  | "green" => 6
  | _ => 0

def has_seven_or_more (color : String) : Bool :=
  balloon_count color ≥ 7

def colors_with_seven_or_more : List String :=
  ["red", "blue", "yellow", "green"].filter has_seven_or_more

theorem smallest_balloon_count : 
  (colors_with_seven_or_more.map balloon_count).minimum? = some 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_balloon_count_l782_78244


namespace NUMINAMATH_CALUDE_sara_quarters_sum_l782_78220

-- Define the initial number of quarters Sara had
def initial_quarters : ℝ := 783.0

-- Define the number of quarters Sara's dad gave her
def dad_quarters : ℝ := 271.0

-- Define the total number of quarters Sara has now
def total_quarters : ℝ := initial_quarters + dad_quarters

-- Theorem to prove
theorem sara_quarters_sum :
  total_quarters = 1054.0 := by sorry

end NUMINAMATH_CALUDE_sara_quarters_sum_l782_78220


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l782_78282

/-- Proves that a single discount of 40.5% is equivalent to two successive discounts of 15% and 30% on an item originally priced at $50. -/
theorem successive_discounts_equivalence :
  let original_price : ℝ := 50
  let first_discount : ℝ := 0.15
  let second_discount : ℝ := 0.30
  let equivalent_discount : ℝ := 0.405
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = original_price * (1 - equivalent_discount) := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l782_78282


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l782_78245

/-- A parabola y = ax^2 + 12 is tangent to the line y = 2x if and only if a = 1/12 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 12 = 2 * x) ↔ a = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l782_78245


namespace NUMINAMATH_CALUDE_circle_line_distance_l782_78260

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  equation : ℝ → ℝ → Prop

/-- Calculates the distance from a point to a line --/
def distancePointToLine (point : ℝ × ℝ) (line : Line) : ℝ := sorry

/-- The main theorem --/
theorem circle_line_distance (c : Circle) (l : Line) :
  c.equation = fun x y => x^2 + y^2 - 2*x - 8*y + 1 = 0 →
  l.equation = fun x y => l.a*x - y + 1 = 0 →
  c.center = (1, 4) →
  distancePointToLine c.center l = 1 →
  l.a = 4/3 := by sorry

end NUMINAMATH_CALUDE_circle_line_distance_l782_78260


namespace NUMINAMATH_CALUDE_jessica_flour_calculation_l782_78283

/-- Given a recipe that requires a total amount of flour and the amount of flour still needed,
    calculate the amount of flour already added. -/
def flour_already_added (total_flour : ℕ) (flour_needed : ℕ) : ℕ :=
  total_flour - flour_needed

theorem jessica_flour_calculation :
  let total_flour : ℕ := 8
  let flour_needed : ℕ := 4
  flour_already_added total_flour flour_needed = 4 := by
  sorry

end NUMINAMATH_CALUDE_jessica_flour_calculation_l782_78283


namespace NUMINAMATH_CALUDE_gretzky_street_length_proof_l782_78264

/-- The length of Gretzky Street in kilometers -/
def gretzky_street_length : ℝ := 5.95

/-- The number of numbered intersecting streets -/
def num_intersections : ℕ := 15

/-- The distance between each intersecting street in meters -/
def intersection_distance : ℝ := 350

/-- The number of additional segments at the beginning and end -/
def additional_segments : ℕ := 2

/-- Theorem stating that the length of Gretzky Street is 5.95 kilometers -/
theorem gretzky_street_length_proof :
  gretzky_street_length = 
    (intersection_distance * (num_intersections + additional_segments)) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_gretzky_street_length_proof_l782_78264


namespace NUMINAMATH_CALUDE_handshake_count_l782_78203

theorem handshake_count (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l782_78203


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l782_78212

/-- Represents a pentagonal prism -/
structure PentagonalPrism where
  /-- Number of faces of the pentagonal prism -/
  faces : Nat
  /-- Number of edges of the pentagonal prism -/
  edges : Nat
  /-- Number of vertices of the pentagonal prism -/
  vertices : Nat
  /-- The number of faces is 7 (2 pentagonal + 5 rectangular) -/
  faces_eq : faces = 7
  /-- The number of edges is 15 (5 + 5 + 5) -/
  edges_eq : edges = 15
  /-- The number of vertices is 10 (5 + 5) -/
  vertices_eq : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) :
  p.faces + p.edges + p.vertices = 32 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l782_78212


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l782_78227

theorem mean_of_three_numbers (n : ℤ) : 
  n = (17 + 23 + 2*n) / 3 → n = 40 := by
sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l782_78227


namespace NUMINAMATH_CALUDE_farmer_apples_l782_78253

def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88
def apples_left : ℕ := 39

theorem farmer_apples : initial_apples = apples_given_away + apples_left := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l782_78253


namespace NUMINAMATH_CALUDE_smallest_n_for_432n_perfect_square_l782_78267

theorem smallest_n_for_432n_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), 432 * n = k^2) ∧
  (∀ (m : ℕ), m > 0 → m < n → ¬∃ (j : ℕ), 432 * m = j^2) ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_432n_perfect_square_l782_78267


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l782_78213

theorem distance_to_origin_of_complex_number : ∃ (z : ℂ), 
  z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l782_78213


namespace NUMINAMATH_CALUDE_prism_volume_and_ak_length_l782_78243

/-- Regular triangular prism with inscribed sphere -/
structure RegularPrismWithSphere where
  -- Height of the prism
  h : ℝ
  -- Radius of the inscribed sphere
  r : ℝ
  -- Point K on edge AA₁
  k : ℝ
  -- Point L on edge BB₁
  l : ℝ
  -- Assumption that h = 12
  h_eq : h = 12
  -- Assumption that r = √(35/3)
  r_eq : r = Real.sqrt (35/3)
  -- Assumption that KL is parallel to AB
  kl_parallel_ab : True  -- We can't directly express this geometric condition
  -- Assumption that plane KBC touches the sphere
  kbc_touches_sphere : True  -- We can't directly express this geometric condition
  -- Assumption that plane LA₁C₁ touches the sphere
  la1c1_touches_sphere : True  -- We can't directly express this geometric condition

/-- Theorem about the volume and AK length of the regular triangular prism with inscribed sphere -/
theorem prism_volume_and_ak_length (p : RegularPrismWithSphere) :
  ∃ (v : ℝ) (ak : ℝ),
    v = 420 * Real.sqrt 3 ∧
    (ak = 8 ∨ ak = 4) :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_and_ak_length_l782_78243


namespace NUMINAMATH_CALUDE_parallelogram_bisecting_line_l782_78297

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

def parallelogram : Parallelogram := {
  v1 := ⟨12, 50⟩,
  v2 := ⟨12, 120⟩,
  v3 := ⟨30, 160⟩,
  v4 := ⟨30, 90⟩
}

/-- Function to check if a line cuts a parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (p : Parallelogram) (l : Line) : Prop := sorry

/-- Function to express a real number as a fraction of relatively prime positive integers -/
def asRelativelyPrimeFraction (x : ℝ) : (ℕ × ℕ) := sorry

theorem parallelogram_bisecting_line :
  ∃ (l : Line),
    cutsIntoCongruentPolygons parallelogram l ∧
    l.slope = 5 ∧
    let (m, n) := asRelativelyPrimeFraction l.slope
    m + n = 6 := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisecting_line_l782_78297


namespace NUMINAMATH_CALUDE_divisibility_of_binomial_difference_l782_78217

theorem divisibility_of_binomial_difference (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (a + b)^p - a^p - b^p = k * p :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_binomial_difference_l782_78217


namespace NUMINAMATH_CALUDE_min_value_expression_l782_78278

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a = 2 * b) (hac : a = 2 * c) :
  (a + b) / c + (a + c) / b + (b + c) / a = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l782_78278


namespace NUMINAMATH_CALUDE_ellipse_properties_l782_78277

/-- Ellipse C with given properties -/
structure Ellipse :=
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (2^2 / a^2) + (2^2 / b^2) = 1)
  (h4 : a^2 - b^2 = 2 * b^2)

/-- Line l passing through (-1, 0) -/
def Line := ℝ → ℝ

/-- Intersection points of line l and ellipse C -/
def IntersectionPoints (C : Ellipse) (l : Line) := ℝ × ℝ

/-- Foci of ellipse C -/
def Foci (C : Ellipse) := ℝ × ℝ

/-- Areas of triangles formed by foci and intersection points -/
def TriangleAreas (C : Ellipse) (l : Line) := ℝ × ℝ

/-- Main theorem -/
theorem ellipse_properties (C : Ellipse) (l : Line) :
  (∀ x y : ℝ, x^2 / 12 + y^2 / 6 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (∃ S : Set ℝ, S = {x | 0 ≤ x ∧ x ≤ Real.sqrt 3} ∧
    ∀ areas : TriangleAreas C l, |areas.1 - areas.2| ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l782_78277


namespace NUMINAMATH_CALUDE_valid_rod_pairs_l782_78207

def is_valid_polygon (a b c d e : ℕ) : Prop :=
  a + b + c + d > e ∧ a + b + c + e > d ∧ a + b + d + e > c ∧ 
  a + c + d + e > b ∧ b + c + d + e > a

def count_valid_pairs : ℕ → ℕ → ℕ → ℕ := sorry

theorem valid_rod_pairs : 
  let rod_lengths : List ℕ := List.range 50
  let selected_rods : List ℕ := [8, 12, 20]
  let remaining_rods : List ℕ := rod_lengths.filter (λ x => x ∉ selected_rods)
  count_valid_pairs 8 12 20 = 135 := by sorry

end NUMINAMATH_CALUDE_valid_rod_pairs_l782_78207


namespace NUMINAMATH_CALUDE_roller_coaster_cars_l782_78286

theorem roller_coaster_cars (people_in_line : ℕ) (people_per_car : ℕ) (num_runs : ℕ) 
  (h1 : people_in_line = 84)
  (h2 : people_per_car = 2)
  (h3 : num_runs = 6)
  (h4 : people_in_line = num_runs * (num_cars * people_per_car)) :
  num_cars = 7 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cars_l782_78286


namespace NUMINAMATH_CALUDE_triangle_segment_product_l782_78206

/-- Given a triangle ABC with an interior point P, this theorem proves that
    if the segments created by extending lines from vertices through P
    to opposite sides have lengths a, b, c, and d, where a + b + c = 43
    and d = 3, then the product abc equals 441. -/
theorem triangle_segment_product (a b c d : ℝ) (h1 : a + b + c = 43) (h2 : d = 3) :
  a * b * c = 441 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_product_l782_78206


namespace NUMINAMATH_CALUDE_highlight_film_average_time_l782_78209

def point_guard_time : ℕ := 130
def shooting_guard_time : ℕ := 145
def small_forward_time : ℕ := 85
def power_forward_time : ℕ := 60
def center_time : ℕ := 180
def number_of_players : ℕ := 5

def total_time : ℕ := point_guard_time + shooting_guard_time + small_forward_time + power_forward_time + center_time

theorem highlight_film_average_time :
  (total_time / number_of_players : ℚ) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_highlight_film_average_time_l782_78209


namespace NUMINAMATH_CALUDE_complex_magnitude_l782_78235

theorem complex_magnitude (z : ℂ) : z = Complex.I * (2 - Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l782_78235


namespace NUMINAMATH_CALUDE_jackson_gpa_probability_l782_78201

-- Define the point values for each grade
def pointA : ℚ := 5
def pointB : ℚ := 4
def pointC : ℚ := 2
def pointD : ℚ := 1

-- Define the probabilities for Literature grades
def litProbA : ℚ := 1/5
def litProbB : ℚ := 2/5
def litProbC : ℚ := 2/5

-- Define the probabilities for Sociology grades
def socProbA : ℚ := 1/3
def socProbB : ℚ := 1/2
def socProbC : ℚ := 1/6

-- Define the number of classes
def numClasses : ℕ := 5

-- Define the minimum GPA required
def minGPA : ℚ := 4

-- Define the function to calculate GPA
def calculateGPA (points : ℚ) : ℚ := points / numClasses

-- Theorem statement
theorem jackson_gpa_probability :
  let confirmedPoints : ℚ := pointA + pointA  -- Calculus and Physics
  let minRemainingPoints : ℚ := minGPA * numClasses - confirmedPoints
  let probTwoAs : ℚ := litProbA * socProbA
  let probALitBSoc : ℚ := litProbA * socProbB
  let probASocBLit : ℚ := socProbA * litProbB
  (probTwoAs + probALitBSoc + probASocBLit) = 2/5 := by sorry

end NUMINAMATH_CALUDE_jackson_gpa_probability_l782_78201


namespace NUMINAMATH_CALUDE_river_name_proof_l782_78216

theorem river_name_proof :
  ∃! (x y z : ℕ),
    x + y + z = 35 ∧
    x - y = y - (z + 1) ∧
    (x + 3) * z = y^2 ∧
    x = 5 ∧ y = 12 ∧ z = 18 := by
  sorry

end NUMINAMATH_CALUDE_river_name_proof_l782_78216


namespace NUMINAMATH_CALUDE_triangle_base_length_l782_78225

/-- Given a square with perimeter 40 and a triangle with height 40 that share a side and have equal areas, 
    the base of the triangle is 5. -/
theorem triangle_base_length : 
  ∀ (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ),
    square_perimeter = 40 →
    triangle_height = 40 →
    (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
    triangle_base = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l782_78225


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l782_78275

def complex_i : ℂ := Complex.I

theorem point_in_first_quadrant (z : ℂ) :
  z = complex_i * (2 - 3 * complex_i) →
  Complex.re z > 0 ∧ Complex.im z > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l782_78275


namespace NUMINAMATH_CALUDE_valid_numbers_count_l782_78242

/-- A function that generates all valid five-digit numbers satisfying the conditions -/
def validNumbers : List Nat := sorry

/-- A predicate that checks if a number satisfies all conditions -/
def isValid (n : Nat) : Bool := sorry

/-- The main theorem stating that there are exactly 20 valid numbers -/
theorem valid_numbers_count : (validNumbers.filter isValid).length = 20 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l782_78242


namespace NUMINAMATH_CALUDE_product_of_cubes_equality_l782_78299

theorem product_of_cubes_equality : 
  (8 / 9 : ℚ)^3 * (-1 / 3 : ℚ)^3 * (3 / 4 : ℚ)^3 = -8 / 729 := by sorry

end NUMINAMATH_CALUDE_product_of_cubes_equality_l782_78299


namespace NUMINAMATH_CALUDE_inequality_problem_l782_78265

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * y > x * z) ∧
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * (y - z) > 0) ∧
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * z * (z - x) < 0) ∧
  ¬(∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < z * y^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l782_78265


namespace NUMINAMATH_CALUDE_stating_children_count_l782_78254

/-- Represents the number of children in the problem -/
def num_children : ℕ := 6

/-- Represents the age of the youngest child -/
def youngest_age : ℕ := 7

/-- Represents the interval between children's ages -/
def age_interval : ℕ := 3

/-- Represents the sum of all children's ages -/
def total_age : ℕ := 65

/-- 
  Theorem stating that given the conditions of the problem,
  the number of children is 6
-/
theorem children_count : 
  (∃ (n : ℕ), 
    n * (2 * youngest_age + (n - 1) * age_interval) = 2 * total_age ∧
    n = num_children) :=
by sorry

end NUMINAMATH_CALUDE_stating_children_count_l782_78254


namespace NUMINAMATH_CALUDE_cos_54_degrees_l782_78210

theorem cos_54_degrees : Real.cos (54 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l782_78210


namespace NUMINAMATH_CALUDE_min_value_expression_l782_78289

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1/(a*b) + 1/(a*(a-b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l782_78289


namespace NUMINAMATH_CALUDE_guaranteed_scores_l782_78222

/-- Represents a player in the card game -/
inductive Player : Type
| One
| Two

/-- Represents a card in the game -/
def Card := Nat

/-- The deck of cards for Player One -/
def player_one_deck : List Card := List.range 1000 |>.map (fun n => 2 * n + 2)

/-- The deck of cards for Player Two -/
def player_two_deck : List Card := List.range 1001 |>.map (fun n => 2 * n + 1)

/-- The number of rounds in the game -/
def num_rounds : Nat := 1000

/-- A strategy for playing the game -/
def Strategy := List Card → List Card → Card

/-- The result of playing the game -/
structure GameResult where
  player_one_score : Nat
  player_two_score : Nat

/-- Play the game with given strategies -/
def play_game (strategy_one strategy_two : Strategy) : GameResult :=
  sorry

/-- Theorem stating the guaranteed minimum scores for both players -/
theorem guaranteed_scores :
  (∃ (strategy_one : Strategy),
    ∀ (strategy_two : Strategy),
      (play_game strategy_one strategy_two).player_one_score ≥ 499) ∧
  (∃ (strategy_two : Strategy),
    ∀ (strategy_one : Strategy),
      (play_game strategy_one strategy_two).player_two_score ≥ 501) :=
sorry

end NUMINAMATH_CALUDE_guaranteed_scores_l782_78222


namespace NUMINAMATH_CALUDE_coefficient_of_b_squared_l782_78293

theorem coefficient_of_b_squared (a : ℝ) : 
  (∃ b₁ b₂ : ℝ, b₁ + b₂ = 4.5 ∧ 
    (∀ b : ℝ, 4 * b^4 - a * b^2 + 100 = 0 → b ≤ b₁ ∧ b ≤ b₂) ∧
    (4 * b₁^4 - a * b₁^2 + 100 = 0) ∧ 
    (4 * b₂^4 - a * b₂^2 + 100 = 0)) →
  a = 4.5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_b_squared_l782_78293


namespace NUMINAMATH_CALUDE_geometric_sequence_equality_l782_78272

theorem geometric_sequence_equality (a b c d : ℝ) :
  (a / b = c / d) ↔ (a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_equality_l782_78272


namespace NUMINAMATH_CALUDE_fraction_simplification_l782_78246

theorem fraction_simplification :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l782_78246


namespace NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l782_78239

theorem negative_a_squared_times_a_fourth (a : ℝ) : (-a)^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l782_78239


namespace NUMINAMATH_CALUDE_father_current_age_l782_78273

/-- The father's current age -/
def father_age : ℕ := sorry

/-- The son's current age -/
def son_age : ℕ := sorry

/-- Six years ago, the father's age was five times the son's age -/
axiom past_age_relation : father_age - 6 = 5 * (son_age - 6)

/-- In six years, the sum of their ages will be 78 -/
axiom future_age_sum : father_age + 6 + son_age + 6 = 78

theorem father_current_age : father_age = 51 := by
  sorry

end NUMINAMATH_CALUDE_father_current_age_l782_78273


namespace NUMINAMATH_CALUDE_john_lap_time_improvement_l782_78268

theorem john_lap_time_improvement :
  let initial_laps : ℚ := 15
  let initial_time : ℚ := 40
  let current_laps : ℚ := 18
  let current_time : ℚ := 36
  let initial_lap_time := initial_time / initial_laps
  let current_lap_time := current_time / current_laps
  let improvement := initial_lap_time - current_lap_time
  improvement = 2/3
:= by sorry

end NUMINAMATH_CALUDE_john_lap_time_improvement_l782_78268


namespace NUMINAMATH_CALUDE_abigail_report_time_l782_78231

/-- The time it takes Abigail to finish her report -/
def report_completion_time (total_words : ℕ) (words_per_half_hour : ℕ) (words_written : ℕ) (proofreading_time : ℕ) : ℕ :=
  let words_left := total_words - words_written
  let half_hour_blocks := (words_left + words_per_half_hour - 1) / words_per_half_hour
  let writing_time := half_hour_blocks * 30
  writing_time + proofreading_time

/-- Theorem stating that Abigail will take 225 minutes to finish her report -/
theorem abigail_report_time :
  report_completion_time 1500 250 200 45 = 225 := by
  sorry

end NUMINAMATH_CALUDE_abigail_report_time_l782_78231


namespace NUMINAMATH_CALUDE_edwards_remaining_money_l782_78285

/-- Calculates the remaining money after a purchase with sales tax -/
def remaining_money (initial_amount purchase_amount tax_rate : ℚ) : ℚ :=
  let sales_tax := purchase_amount * tax_rate
  let total_cost := purchase_amount + sales_tax
  initial_amount - total_cost

/-- Theorem stating that Edward's remaining money is $0.42 -/
theorem edwards_remaining_money :
  remaining_money 18 16.35 (75 / 1000) = 42 / 100 := by
  sorry

end NUMINAMATH_CALUDE_edwards_remaining_money_l782_78285


namespace NUMINAMATH_CALUDE_no_integer_solutions_l782_78291

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x ≠ 1 ∧ (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l782_78291


namespace NUMINAMATH_CALUDE_quadratic_inequality_l782_78229

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 8 * x + 5 > 0 ↔ x < -1/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l782_78229


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l782_78298

theorem simplify_fraction_product : (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l782_78298


namespace NUMINAMATH_CALUDE_biased_coin_probabilities_l782_78276

theorem biased_coin_probabilities (p : ℝ) 
  (h_range : 0 < p ∧ p < 1)
  (h_equal_prob : (5 : ℝ) * p * (1 - p)^4 = 10 * p^2 * (1 - p)^3)
  (h_non_zero : (5 : ℝ) * p * (1 - p)^4 ≠ 0) : 
  p = 1/3 ∧ (10 : ℝ) * p^3 * (1 - p)^2 = 40/243 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probabilities_l782_78276


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l782_78234

theorem decimal_to_fraction (x : ℚ) (h : x = 368/100) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ (Nat.gcd n d = 1) :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l782_78234


namespace NUMINAMATH_CALUDE_condition1_condition2_degree_in_x_l782_78202

/-- A polynomial in three variables -/
def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

/-- The first condition of the polynomial -/
theorem condition1 (x y z : ℝ) : f x (z^2) y + f x (y^2) z = 0 := by sorry

/-- The second condition of the polynomial -/
theorem condition2 (x y z : ℝ) : f (z^3) y x + f (x^3) y z = 0 := by sorry

/-- The polynomial is of 4th degree in x -/
theorem degree_in_x : ∃ (a b c d e : ℝ → ℝ → ℝ), ∀ x y z, 
  f x y z = a y z * x^4 + b y z * x^3 + c y z * x^2 + d y z * x + e y z := by sorry

end NUMINAMATH_CALUDE_condition1_condition2_degree_in_x_l782_78202


namespace NUMINAMATH_CALUDE_find_m_l782_78263

theorem find_m : ∃ m : ℝ, (∀ x : ℝ, x - m > 5 ↔ x > 2) → m = -3 := by sorry

end NUMINAMATH_CALUDE_find_m_l782_78263


namespace NUMINAMATH_CALUDE_root_implies_ab_leq_one_l782_78296

theorem root_implies_ab_leq_one (a b : ℝ) : 
  ((a + b + a) * (a + b + b) = 9) → ab ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_ab_leq_one_l782_78296


namespace NUMINAMATH_CALUDE_ascending_order_l782_78238

theorem ascending_order (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 1) :
  b * c < a * c ∧ a * c < a * b ∧ a * b < a * b * c := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_l782_78238


namespace NUMINAMATH_CALUDE_add_518_276_base_12_l782_78204

/-- Addition in base 12 --/
def add_base_12 (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 12 to base 10 --/
def base_12_to_10 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 12 --/
def base_10_to_12 (n : ℕ) : ℕ :=
  sorry

theorem add_518_276_base_12 :
  add_base_12 (base_10_to_12 518) (base_10_to_12 276) = base_10_to_12 792 :=
sorry

end NUMINAMATH_CALUDE_add_518_276_base_12_l782_78204


namespace NUMINAMATH_CALUDE_number_of_coaches_l782_78247

theorem number_of_coaches (pouches_per_pack : ℕ) (packs_bought : ℕ) (team_members : ℕ) (helpers : ℕ) :
  pouches_per_pack = 6 →
  packs_bought = 3 →
  team_members = 13 →
  helpers = 2 →
  packs_bought * pouches_per_pack = team_members + helpers + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_coaches_l782_78247


namespace NUMINAMATH_CALUDE_complex_modulus_range_l782_78226

theorem complex_modulus_range (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) →
  -Real.sqrt 5 / 5 ≤ a ∧ a ≤ Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l782_78226


namespace NUMINAMATH_CALUDE_solve_equation_l782_78290

theorem solve_equation : ∃ x : ℝ, (5*x + 9*x = 350 - 10*(x - 4)) ∧ x = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l782_78290


namespace NUMINAMATH_CALUDE_proposition_p_true_q_false_l782_78200

theorem proposition_p_true_q_false :
  (∃ x : ℝ, Real.exp x ≥ x + 1) ∧
  (∃ a b : ℝ, a^2 < b^2 ∧ a ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_true_q_false_l782_78200


namespace NUMINAMATH_CALUDE_confucius_travel_equation_l782_78266

/-- Represents the scenario of Confucius and his students traveling to a school -/
def confucius_travel (x : ℝ) : Prop :=
  let student_speed := x
  let cart_speed := 1.5 * x
  let distance := 30
  let student_time := distance / student_speed
  let confucius_time := distance / cart_speed + 1
  student_time = confucius_time

/-- Theorem stating the equation that holds true for the travel scenario -/
theorem confucius_travel_equation (x : ℝ) (hx : x > 0) :
  confucius_travel x ↔ 30 / x = 30 / (1.5 * x) + 1 :=
sorry

end NUMINAMATH_CALUDE_confucius_travel_equation_l782_78266


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l782_78205

theorem least_number_with_remainder (n : ℕ) : n = 266 ↔ 
  (n > 0 ∧ 
   n % 33 = 2 ∧ 
   n % 8 = 2 ∧ 
   ∀ m : ℕ, m > 0 → m % 33 = 2 → m % 8 = 2 → m ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l782_78205


namespace NUMINAMATH_CALUDE_second_term_is_four_l782_78240

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The second term of an arithmetic sequence. -/
def ArithmeticSequence.secondTerm (seq : ArithmeticSequence) : ℝ :=
  seq.a + seq.d

/-- The third term of an arithmetic sequence. -/
def ArithmeticSequence.thirdTerm (seq : ArithmeticSequence) : ℝ :=
  seq.a + 2 * seq.d

/-- The sum of the first and third terms of an arithmetic sequence. -/
def ArithmeticSequence.sumFirstAndThird (seq : ArithmeticSequence) : ℝ :=
  seq.a + seq.thirdTerm

theorem second_term_is_four (seq : ArithmeticSequence) 
    (h : seq.sumFirstAndThird = 8) : 
    seq.secondTerm = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_four_l782_78240


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l782_78233

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 6 = 10 →
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l782_78233


namespace NUMINAMATH_CALUDE_line_intersects_circle_l782_78215

/-- The line intersects the circle given the conditions -/
theorem line_intersects_circle (a x₀ y₀ : ℝ) (h_a : a > 0) 
  (h_outside : x₀^2 + y₀^2 > a^2) :
  ∃ (x y : ℝ), x^2 + y^2 = a^2 ∧ x₀*x + y₀*y = a^2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l782_78215


namespace NUMINAMATH_CALUDE_buckets_taken_away_is_three_l782_78295

/-- Calculates the number of buckets taken away to reach the bath level -/
def buckets_taken_away (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (weekly_usage : ℕ) (baths_per_week : ℕ) : ℕ :=
  let full_tub := bucket_capacity * buckets_to_fill
  let bath_level := weekly_usage / baths_per_week
  let difference := full_tub - bath_level
  difference / bucket_capacity

/-- Proves that the number of buckets taken away is 3 given the problem conditions -/
theorem buckets_taken_away_is_three :
  buckets_taken_away 120 14 9240 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_buckets_taken_away_is_three_l782_78295


namespace NUMINAMATH_CALUDE_not_perfect_square_l782_78292

theorem not_perfect_square (n : ℕ) (h : n > 1) : ¬∃ (m : ℕ), 9*n^2 - 9*n + 9 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l782_78292


namespace NUMINAMATH_CALUDE_ab_greater_ac_l782_78279

theorem ab_greater_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_ac_l782_78279


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l782_78224

theorem president_and_committee_selection (n : ℕ) (h : n = 10) : 
  n * (Nat.choose (n - 1) 3) = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_selection_l782_78224


namespace NUMINAMATH_CALUDE_geometry_theorem_l782_78252

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planesParallel : Plane → Plane → Prop)
variable (planesPerpendicular : Plane → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) :
  (planesParallel α β ∧ subset m α → parallel m β) ∧
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α → perpendicular m β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l782_78252


namespace NUMINAMATH_CALUDE_square_area_proof_l782_78223

theorem square_area_proof (x : ℝ) :
  (5 * x - 18 = 25 - 2 * x) →
  (5 * x - 18 ≥ 0) →
  ((5 * x - 18)^2 : ℝ) = 7921 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l782_78223
