import Mathlib

namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1890_189044

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1890_189044


namespace NUMINAMATH_CALUDE_shell_difference_l1890_189066

theorem shell_difference (perfect_total : ℕ) (broken_total : ℕ)
  (broken_spiral_percent : ℚ) (broken_clam_percent : ℚ)
  (perfect_spiral_percent : ℚ) (perfect_clam_percent : ℚ)
  (h1 : perfect_total = 30)
  (h2 : broken_total = 80)
  (h3 : broken_spiral_percent = 35 / 100)
  (h4 : broken_clam_percent = 40 / 100)
  (h5 : perfect_spiral_percent = 25 / 100)
  (h6 : perfect_clam_percent = 50 / 100) :
  ⌊broken_total * broken_spiral_percent⌋ - ⌊perfect_total * perfect_spiral_percent⌋ = 21 :=
by sorry

end NUMINAMATH_CALUDE_shell_difference_l1890_189066


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1890_189064

theorem polynomial_evaluation : 
  let p (x : ℝ) := 2 * x^4 + 3 * x^3 - x^2 + 5 * x - 2
  p 2 = 60 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1890_189064


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l1890_189052

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l1890_189052


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1890_189077

/-- Proves that if the cost price of 50 articles equals the selling price of 40 articles, 
    then the gain percent is 25%. -/
theorem gain_percent_calculation (C S : ℝ) 
  (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1890_189077


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1890_189028

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_incr : ∀ n : ℕ, a n < a (n + 1))
  (h_sum : a 1 + a 2 + a 3 = 12)
  (h_prod : a 1 * a 2 * a 3 = 48) :
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1890_189028


namespace NUMINAMATH_CALUDE_lucas_chocolate_candy_l1890_189024

/-- The number of pieces of chocolate candy Lucas makes for each student on Monday -/
def pieces_per_student : ℕ := 4

/-- The number of students not coming to class this upcoming Monday -/
def absent_students : ℕ := 3

/-- The number of pieces of chocolate candy Lucas will make this upcoming Monday -/
def upcoming_monday_pieces : ℕ := 28

/-- The number of pieces of chocolate candy Lucas made last Monday -/
def last_monday_pieces : ℕ := pieces_per_student * (upcoming_monday_pieces / pieces_per_student + absent_students)

theorem lucas_chocolate_candy : last_monday_pieces = 40 := by sorry

end NUMINAMATH_CALUDE_lucas_chocolate_candy_l1890_189024


namespace NUMINAMATH_CALUDE_gadget_production_75_workers_4_hours_l1890_189061

/-- Represents the production rate of a worker per hour -/
structure ProductionRate :=
  (gadgets : ℝ)
  (gizmos : ℝ)

/-- Calculates the total production given workers, hours, and rate -/
def totalProduction (workers : ℕ) (hours : ℕ) (rate : ProductionRate) : ProductionRate :=
  { gadgets := workers * hours * rate.gadgets,
    gizmos := workers * hours * rate.gizmos }

theorem gadget_production_75_workers_4_hours 
  (rate1 : ProductionRate)
  (rate2 : ProductionRate)
  (h1 : totalProduction 150 1 rate1 = { gadgets := 450, gizmos := 300 })
  (h2 : totalProduction 100 2 rate2 = { gadgets := 400, gizmos := 500 }) :
  (totalProduction 75 4 rate2).gadgets = 600 := by
sorry

end NUMINAMATH_CALUDE_gadget_production_75_workers_4_hours_l1890_189061


namespace NUMINAMATH_CALUDE_field_trip_absentees_prove_girls_absent_l1890_189048

/-- Given a field trip scenario, calculate the number of girls who couldn't join. -/
theorem field_trip_absentees (total_students : ℕ) (boys : ℕ) (girls_present : ℕ) : ℕ :=
  let girls_assigned := total_students - boys
  girls_assigned - girls_present

/-- Prove the number of girls who couldn't join the field trip. -/
theorem prove_girls_absent : field_trip_absentees 18 8 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_absentees_prove_girls_absent_l1890_189048


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1890_189045

/-- Given a hyperbola with equation x^2 - ty^2 = 3t and focal distance 6, its eccentricity is √6/2 -/
theorem hyperbola_eccentricity (t : ℝ) :
  (∃ (x y : ℝ), x^2 - t*y^2 = 3*t) →  -- Hyperbola equation
  (∃ (c : ℝ), c = 3) →  -- Focal distance is 6, so half of it (c) is 3
  (∃ (e : ℝ), e = (Real.sqrt 6) / 2 ∧ e = (Real.sqrt (t + 1))) -- Eccentricity
  :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1890_189045


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1890_189035

/-- Given a function g: ℝ → ℝ, if g(-2) = g(2) = 3, then (-2, 3) is the unique
    intersection point of y = g(x) and y = g(x+4), and the sum of its coordinates is 1. -/
theorem intersection_point_sum (g : ℝ → ℝ) (h1 : g (-2) = 3) (h2 : g 2 = 3) :
  ∃! p : ℝ × ℝ, (g p.1 = p.2 ∧ g (p.1 + 4) = p.2) ∧ p = (-2, 3) ∧ p.1 + p.2 = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1890_189035


namespace NUMINAMATH_CALUDE_people_in_room_l1890_189098

theorem people_in_room (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  total_chairs = 25 →
  seated_people = (3 * total_people) / 4 →
  seated_people = (2 * total_chairs) / 3 →
  total_people = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_room_l1890_189098


namespace NUMINAMATH_CALUDE_resultant_calculation_l1890_189049

theorem resultant_calculation : 
  let original : ℕ := 13
  let doubled := 2 * original
  let added_seven := doubled + 7
  let trebled := 3 * added_seven
  trebled = 99 := by sorry

end NUMINAMATH_CALUDE_resultant_calculation_l1890_189049


namespace NUMINAMATH_CALUDE_total_gumballs_l1890_189092

/-- Represents the number of gumballs in a small package -/
def small_package : ℕ := 5

/-- Represents the number of gumballs in a medium package -/
def medium_package : ℕ := 12

/-- Represents the number of gumballs in a large package -/
def large_package : ℕ := 20

/-- Represents the number of small packages Nathan bought -/
def small_quantity : ℕ := 4

/-- Represents the number of medium packages Nathan bought -/
def medium_quantity : ℕ := 3

/-- Represents the number of large packages Nathan bought -/
def large_quantity : ℕ := 2

/-- Theorem stating the total number of gumballs Nathan ate -/
theorem total_gumballs : 
  small_quantity * small_package + 
  medium_quantity * medium_package + 
  large_quantity * large_package = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_gumballs_l1890_189092


namespace NUMINAMATH_CALUDE_cosine_in_special_triangle_l1890_189018

/-- Given a triangle ABC where the sides a, b, and c are in the ratio 2:3:4, 
    prove that cos C = -1/4 -/
theorem cosine_in_special_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (ratio : ∃ (x : ℝ), x > 0 ∧ a = 2*x ∧ b = 3*x ∧ c = 4*x) : 
    (a^2 + b^2 - c^2) / (2*a*b) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_in_special_triangle_l1890_189018


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l1890_189050

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ x * y = 16 ∧ 1 / x = 3 * (1 / y) → x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l1890_189050


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_no_common_points_l1890_189013

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the "contained within" relation for a line in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the "no common points" relation between two lines
variable (no_common_points : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_no_common_points 
  (l a : Line) (α : Plane) 
  (h1 : parallel l α) 
  (h2 : contained_in a α) : 
  no_common_points l a := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_no_common_points_l1890_189013


namespace NUMINAMATH_CALUDE_paul_chickens_left_l1890_189070

/-- The number of chickens Paul has left after selling some -/
def chickens_left (initial : ℕ) (sold_neighbor : ℕ) (sold_gate : ℕ) : ℕ :=
  initial - sold_neighbor - sold_gate

/-- Theorem stating that Paul is left with 43 chickens -/
theorem paul_chickens_left : chickens_left 80 12 25 = 43 := by
  sorry

end NUMINAMATH_CALUDE_paul_chickens_left_l1890_189070


namespace NUMINAMATH_CALUDE_jack_morning_emails_l1890_189078

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The total number of emails Jack received in the morning and afternoon -/
def total_morning_afternoon : ℕ := 13

/-- Theorem stating that Jack received 5 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails + afternoon_emails = total_morning_afternoon → 
  morning_emails = 5 := by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l1890_189078


namespace NUMINAMATH_CALUDE_star_two_three_l1890_189067

-- Define the star operation
def star (c d : ℝ) : ℝ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

-- State the theorem
theorem star_two_three : star 2 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l1890_189067


namespace NUMINAMATH_CALUDE_divisible_by_64_l1890_189094

theorem divisible_by_64 (n : ℕ+) : ∃ k : ℤ, 3^(2*n.val + 2) - 8*n.val - 9 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l1890_189094


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1890_189000

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1890_189000


namespace NUMINAMATH_CALUDE_existence_of_a_l1890_189086

theorem existence_of_a (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧ ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_l1890_189086


namespace NUMINAMATH_CALUDE_original_number_proof_l1890_189055

theorem original_number_proof :
  ∃ x : ℝ, (x - x / 3 = x - 48) ∧ (x = 144) := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1890_189055


namespace NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l1890_189059

-- Define the triangle
def triangle_side1 : ℝ := 15
def triangle_side2 : ℝ := 36
def triangle_side3 : ℝ := 39

-- Define the rectangle's area formula
def rectangle_area (α β ω : ℝ) : ℝ := α * ω - β * ω^2

-- State the theorem
theorem inscribed_rectangle_coefficient :
  ∃ (α β : ℝ),
    (∀ ω, rectangle_area α β ω ≥ 0) ∧
    (rectangle_area α β triangle_side2 = 0) ∧
    (rectangle_area α β (triangle_side2 / 2) = 
      (triangle_side1 + triangle_side2 + triangle_side3) * 
      (triangle_side1 + triangle_side2 - triangle_side3) * 
      (triangle_side1 - triangle_side2 + triangle_side3) * 
      (-triangle_side1 + triangle_side2 + triangle_side3) / 
      (4 * 16)) ∧
    β = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l1890_189059


namespace NUMINAMATH_CALUDE_cans_storage_l1890_189089

theorem cans_storage (cans_per_row : ℕ) (shelves_per_closet : ℕ) (cans_per_closet : ℕ) :
  cans_per_row = 12 →
  shelves_per_closet = 10 →
  cans_per_closet = 480 →
  (cans_per_closet / cans_per_row) / shelves_per_closet = 4 :=
by sorry

end NUMINAMATH_CALUDE_cans_storage_l1890_189089


namespace NUMINAMATH_CALUDE_new_train_distance_l1890_189007

theorem new_train_distance (old_distance : ℝ) (increase_percentage : ℝ) (new_distance : ℝ) : 
  old_distance = 180 → 
  increase_percentage = 0.5 → 
  new_distance = old_distance * (1 + increase_percentage) → 
  new_distance = 270 := by
sorry

end NUMINAMATH_CALUDE_new_train_distance_l1890_189007


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1890_189009

theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + a < 0) → a < 11 ∧
  ∃ a : ℝ, a < 11 ∧ ∀ x : ℝ, x^2 - 2*x + a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1890_189009


namespace NUMINAMATH_CALUDE_stairs_climbing_time_l1890_189043

def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem stairs_climbing_time : arithmeticSum 30 8 6 = 300 := by
  sorry

end NUMINAMATH_CALUDE_stairs_climbing_time_l1890_189043


namespace NUMINAMATH_CALUDE_line_inclination_l1890_189076

theorem line_inclination (a : ℝ) : 
  (((2 - (-3)) / (1 - a) = Real.tan (135 * π / 180)) → a = 6) := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_l1890_189076


namespace NUMINAMATH_CALUDE_problem_statement_l1890_189085

theorem problem_statement (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1890_189085


namespace NUMINAMATH_CALUDE_angle_bisector_length_formulas_l1890_189001

theorem angle_bisector_length_formulas (a b c : ℝ) (α β γ : ℝ) (p R : ℝ) (l_a : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  α > 0 ∧ β > 0 ∧ γ > 0 ∧
  α + β + γ = π ∧
  p = (a + b + c) / 2 ∧
  R > 0 →
  (l_a = Real.sqrt (4 * p * (p - a) * b * c / ((b + c)^2))) ∧
  (l_a = 2 * b * c * Real.cos (α / 2) / (b + c)) ∧
  (l_a = 2 * R * Real.sin β * Real.sin γ / Real.cos ((β - γ) / 2)) ∧
  (l_a = 4 * p * Real.sin (β / 2) * Real.sin (γ / 2) / (Real.sin β + Real.sin γ)) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_formulas_l1890_189001


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1890_189011

theorem necessary_but_not_sufficient (x : ℝ) :
  (x ≥ 2 → x ≠ 1) ∧ ¬(x ≠ 1 → x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1890_189011


namespace NUMINAMATH_CALUDE_inequality_proof_l1890_189021

theorem inequality_proof (a b c : ℕ) (ha : a = 8^53) (hb : b = 16^41) (hc : c = 64^27) :
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1890_189021


namespace NUMINAMATH_CALUDE_planes_parallel_if_infinitely_many_parallel_lines_l1890_189026

-- Define the concept of a plane in 3D space
variable (α β : Set (ℝ × ℝ × ℝ))

-- Define what it means for a line to be parallel to a plane
def LineParallelToPlane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for two planes to be parallel
def PlanesParallel (p1 p2 : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the concept of infinitely many lines in a plane
def InfinitelyManyParallelLines (p1 p2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (S : Set (Set (ℝ × ℝ × ℝ))), Infinite S ∧ (∀ l ∈ S, l ⊆ p1 ∧ LineParallelToPlane l p2)

-- State the theorem
theorem planes_parallel_if_infinitely_many_parallel_lines (α β : Set (ℝ × ℝ × ℝ)) :
  InfinitelyManyParallelLines α β → PlanesParallel α β := by sorry

end NUMINAMATH_CALUDE_planes_parallel_if_infinitely_many_parallel_lines_l1890_189026


namespace NUMINAMATH_CALUDE_ratio_of_segments_l1890_189063

/-- Given four points P, Q, R, and S on a line in that order, 
    with PQ = 3, QR = 7, and PS = 17, the ratio of PR to QS is 10/7. -/
theorem ratio_of_segments (P Q R S : ℝ) : 
  Q < R ∧ R < S ∧ Q - P = 3 ∧ R - Q = 7 ∧ S - P = 17 → 
  (R - P) / (S - Q) = 10 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l1890_189063


namespace NUMINAMATH_CALUDE_square_difference_l1890_189096

theorem square_difference (m n : ℝ) (h1 : m + n = 3) (h2 : m - n = 4) : m^2 - n^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1890_189096


namespace NUMINAMATH_CALUDE_limit_of_function_l1890_189099

theorem limit_of_function : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
  0 < |x| ∧ |x| < δ → 
  |((1 + x * Real.sin x - Real.cos (2 * x)) / (Real.sin x)^2) - 3| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_of_function_l1890_189099


namespace NUMINAMATH_CALUDE_probability_two_ties_l1890_189082

/-- The probability of selecting 2 ties from a boutique with shirts, pants, and ties -/
theorem probability_two_ties (shirts pants ties : ℕ) : 
  shirts = 4 → pants = 8 → ties = 18 → 
  (ties : ℚ) / (shirts + pants + ties) * ((ties - 1) : ℚ) / (shirts + pants + ties - 1) = 51 / 145 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_ties_l1890_189082


namespace NUMINAMATH_CALUDE_sum_equals_300_l1890_189075

theorem sum_equals_300 : 192 + 58 + 42 + 8 = 300 := by sorry

end NUMINAMATH_CALUDE_sum_equals_300_l1890_189075


namespace NUMINAMATH_CALUDE_y_squared_value_l1890_189036

theorem y_squared_value (y : ℝ) (hy : y > 0) (h : Real.tan (Real.arccos y) = 2 * y) :
  y^2 = (-1 + Real.sqrt 17) / 8 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_value_l1890_189036


namespace NUMINAMATH_CALUDE_parabola_coordinate_shift_l1890_189062

/-- Given a parabola y = 3x² in a Cartesian coordinate system, 
    if the coordinate system is shifted 3 units right and 3 units up,
    then the equation of the parabola in the new coordinate system is y = 3(x+3)² - 3 -/
theorem parabola_coordinate_shift (x y : ℝ) : 
  (y = 3 * x^2) → 
  (∃ x' y', x' = x - 3 ∧ y' = y - 3 ∧ y' = 3 * (x' + 3)^2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coordinate_shift_l1890_189062


namespace NUMINAMATH_CALUDE_sand_remaining_l1890_189034

/-- Given a truck with an initial amount of sand and an amount of sand lost during transit,
    prove that the remaining amount of sand is equal to the initial amount minus the lost amount. -/
theorem sand_remaining (initial_sand : ℝ) (sand_lost : ℝ) :
  initial_sand - sand_lost = initial_sand - sand_lost :=
by sorry

end NUMINAMATH_CALUDE_sand_remaining_l1890_189034


namespace NUMINAMATH_CALUDE_empty_fixed_implies_empty_stable_l1890_189083

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Set of fixed points -/
def A (a b c : ℝ) : Set ℝ := {x | f a b c x = x}

/-- Set of stable points -/
def B (a b c : ℝ) : Set ℝ := {x | f a b c (f a b c x) = x}

/-- Theorem: If A is empty, then B is empty for quadratic functions -/
theorem empty_fixed_implies_empty_stable (a b c : ℝ) (ha : a ≠ 0) :
  A a b c = ∅ → B a b c = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_fixed_implies_empty_stable_l1890_189083


namespace NUMINAMATH_CALUDE_ones_digit_73_power_l1890_189080

theorem ones_digit_73_power (n : ℕ) : 
  (73^n % 10 = 7) ↔ (n % 4 = 3) := by
sorry

end NUMINAMATH_CALUDE_ones_digit_73_power_l1890_189080


namespace NUMINAMATH_CALUDE_increase_in_circumference_l1890_189003

/-- The increase in circumference when the diameter of a circle increases by 2π units -/
theorem increase_in_circumference (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + 2 * π)
  let increase_in_circumference := new_circumference - original_circumference
  increase_in_circumference = 2 * π^2 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_circumference_l1890_189003


namespace NUMINAMATH_CALUDE_pizza_delivery_gas_remaining_l1890_189069

theorem pizza_delivery_gas_remaining (start_amount used_amount : ℚ) 
  (h1 : start_amount = 0.5)
  (h2 : used_amount = 0.33) : 
  start_amount - used_amount = 0.17 := by
sorry

end NUMINAMATH_CALUDE_pizza_delivery_gas_remaining_l1890_189069


namespace NUMINAMATH_CALUDE_vector_eq_quadratic_eq_l1890_189039

/-- The vector representing k(3, -4, 1) - (6, 9, -2) --/
def v (k : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 3*k - 6
  | 1 => -4*k - 9
  | 2 => k + 2

/-- The squared norm of the vector --/
def squared_norm (k : ℝ) : ℝ := (v k 0)^2 + (v k 1)^2 + (v k 2)^2

/-- The theorem stating the equivalence between the vector equation and the quadratic equation --/
theorem vector_eq_quadratic_eq (k : ℝ) :
  squared_norm k = (3 * Real.sqrt 26)^2 ↔ 26 * k^2 + 40 * k - 113 = 0 := by sorry

end NUMINAMATH_CALUDE_vector_eq_quadratic_eq_l1890_189039


namespace NUMINAMATH_CALUDE_percentage_problem_l1890_189090

theorem percentage_problem (P : ℝ) : 
  (0.2 * 580 = (P / 100) * 120 + 80) → P = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1890_189090


namespace NUMINAMATH_CALUDE_smallest_d_for_injective_g_l1890_189010

def g (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_d_for_injective_g :
  ∀ d : ℝ, (∀ x y : ℝ, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_injective_g_l1890_189010


namespace NUMINAMATH_CALUDE_rocky_training_total_l1890_189065

/-- Rocky's training schedule for the first three days -/
def rocky_training (day : Nat) : Nat :=
  match day with
  | 1 => 4
  | 2 => 2 * rocky_training 1
  | 3 => 3 * rocky_training 2
  | _ => 0

/-- The total miles Rocky ran in the first three days of training -/
def total_miles : Nat :=
  rocky_training 1 + rocky_training 2 + rocky_training 3

theorem rocky_training_total : total_miles = 36 := by
  sorry

end NUMINAMATH_CALUDE_rocky_training_total_l1890_189065


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1890_189022

/-- Given a point M in polar coordinates (r, θ), 
    prove that its Cartesian coordinates are (x, y) --/
theorem polar_to_cartesian 
  (r : ℝ) (θ : ℝ) 
  (x : ℝ) (y : ℝ) 
  (h1 : r = 2) 
  (h2 : θ = π/6) 
  (h3 : x = r * Real.cos θ) 
  (h4 : y = r * Real.sin θ) : 
  x = Real.sqrt 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1890_189022


namespace NUMINAMATH_CALUDE_fraction_value_l1890_189053

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : 
  x^2 / (x^4 + x^2 + 1) = 1/8 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1890_189053


namespace NUMINAMATH_CALUDE_equation_solution_l1890_189074

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 150 * x + 3) / (3 * x + 56) = 4 * x + 2 ↔ x = -1.5 ∨ x = -18.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1890_189074


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l1890_189041

def A (k m : ℕ) : ℕ := (k.factorial) / (k - m).factorial

theorem permutation_equation_solution :
  ∃! n : ℕ, n > 0 ∧ A (2*n) 3 = 10 * A n 3 :=
sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l1890_189041


namespace NUMINAMATH_CALUDE_milk_powder_sampling_l1890_189058

theorem milk_powder_sampling (total : ℕ) (sample_size : ℕ) (b : ℕ) :
  total = 240 →
  sample_size = 60 →
  (∃ (a d : ℕ), b = a ∧ total = (a - d) + a + (a + d)) →
  b * sample_size / total = 20 :=
by sorry

end NUMINAMATH_CALUDE_milk_powder_sampling_l1890_189058


namespace NUMINAMATH_CALUDE_side_to_base_ratio_l1890_189081

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  -- The length of one side of the isosceles triangle
  side : ℝ
  -- The length of the base of the isosceles triangle
  base : ℝ
  -- The distance from the vertex to the point of tangency on the side
  vertex_to_tangency : ℝ
  -- Ensure the triangle is isosceles
  isosceles : side > 0
  -- Ensure the point of tangency divides the side in 7:5 ratio
  tangency_ratio : vertex_to_tangency / (side - vertex_to_tangency) = 7 / 5

/-- 
Theorem: In an isosceles triangle with an inscribed circle, 
if the point of tangency on one side divides it in the ratio 7:5 (starting from the vertex), 
then the ratio of the side to the base is 6:5.
-/
theorem side_to_base_ratio 
  (triangle : IsoscelesTriangleWithInscribedCircle) : 
  triangle.side / triangle.base = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_side_to_base_ratio_l1890_189081


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l1890_189079

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length 
  (train_speed : Real) 
  (bridge_crossing_time : Real) 
  (bridge_length : Real) : Real :=
  let total_distance := train_speed * (1000 / 3600) * bridge_crossing_time
  total_distance - bridge_length

/-- Proves that a train traveling at 45 km/hr that crosses a 250 m bridge in 30 seconds has a length of 125 m. -/
theorem train_length_proof :
  train_length 45 30 250 = 125 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l1890_189079


namespace NUMINAMATH_CALUDE_max_true_statements_l1890_189032

theorem max_true_statements (a b : ℝ) : 
  (∃ (s : Finset (Prop)), s.card = 2 ∧ 
    (∀ (p : Prop), p ∈ s → p) ∧
    s ⊆ {1/a > 1/b, a^3 > b^3, a > b, a < 0, b > 0}) ∧
  (∀ (s : Finset (Prop)), s.card > 2 → 
    s ⊆ {1/a > 1/b, a^3 > b^3, a > b, a < 0, b > 0} → 
    ∃ (p : Prop), p ∈ s ∧ ¬p) :=
sorry

end NUMINAMATH_CALUDE_max_true_statements_l1890_189032


namespace NUMINAMATH_CALUDE_square_condition_l1890_189030

def a_n (n : ℕ+) : ℕ := (10^n.val - 1) / 9

theorem square_condition (n : ℕ+) (b : ℕ) : 
  0 < b ∧ b < 10 →
  (∃ k : ℕ, a_n (2*n) - b * a_n n = k^2) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_condition_l1890_189030


namespace NUMINAMATH_CALUDE_monotonic_decreasing_intervals_of_neg_tan_l1890_189088

open Real

noncomputable def f (x : ℝ) := -tan x

theorem monotonic_decreasing_intervals_of_neg_tan :
  ∀ (k : ℤ) (x y : ℝ),
    x ∈ Set.Ioo (k * π - π / 2) (k * π + π / 2) →
    y ∈ Set.Ioo (k * π - π / 2) (k * π + π / 2) →
    x < y →
    f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_intervals_of_neg_tan_l1890_189088


namespace NUMINAMATH_CALUDE_four_times_three_equals_thirtyone_l1890_189046

-- Define the multiplication operation based on the given condition
def special_mult (a b : ℤ) : ℤ := a^2 + 2*a*b - b^2

-- State the theorem
theorem four_times_three_equals_thirtyone : special_mult 4 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_four_times_three_equals_thirtyone_l1890_189046


namespace NUMINAMATH_CALUDE_shooting_is_impossible_coin_toss_is_random_triangle_angles_is_certain_l1890_189091

-- Define the types of events
inductive EventType
  | Impossible
  | Random
  | Certain

-- Define the events
def shooting_event : EventType := EventType.Impossible
def coin_toss_event : EventType := EventType.Random
def triangle_angles_event : EventType := EventType.Certain

-- Theorem statements
theorem shooting_is_impossible : shooting_event = EventType.Impossible := by sorry

theorem coin_toss_is_random : coin_toss_event = EventType.Random := by sorry

theorem triangle_angles_is_certain : triangle_angles_event = EventType.Certain := by sorry

end NUMINAMATH_CALUDE_shooting_is_impossible_coin_toss_is_random_triangle_angles_is_certain_l1890_189091


namespace NUMINAMATH_CALUDE_cannot_tile_removed_square_board_l1890_189040

/-- Represents a chessboard with one square removed -/
def RemovedSquareBoard : Nat := 63

/-- Represents the size of a domino -/
def DominoSize : Nat := 2

theorem cannot_tile_removed_square_board :
  ¬ ∃ (n : Nat), n * DominoSize = RemovedSquareBoard :=
sorry

end NUMINAMATH_CALUDE_cannot_tile_removed_square_board_l1890_189040


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1890_189014

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The maximum value function of |f(x)| on [0,t] -/
noncomputable def φ (t : ℝ) : ℝ :=
  if t ≤ 1 then 2*t - t^2
  else if t ≤ 1 + Real.sqrt 2 then 1
  else t^2 - 2*t

theorem quadratic_function_properties :
  (∀ x, f x ≥ -1) ∧  -- minimum value is -1
  (f 0 = 0) ∧        -- f(0) = 0
  (∀ x, f (1 + x) = f (1 - x)) ∧  -- symmetry property
  (∃ a b c, ∀ x, f x = a*x^2 + b*x + c ∧ a ≠ 0) →  -- f is quadratic
  (∀ x, f x = x^2 - 2*x) ∧  -- part 1
  (∀ m, (∀ x, -3 ≤ x ∧ x ≤ 3 → f x > 2*m*x - 4) ↔ -3 < m ∧ m < 1) ∧  -- part 2
  (∀ t, t > 0 → ∀ x, 0 ≤ x ∧ x ≤ t → |f x| ≤ φ t) ∧  -- part 3
  (∀ t, t > 0 → ∃ x, 0 ≤ x ∧ x ≤ t ∧ |f x| = φ t)  -- part 3 (maximum is achieved)
:= by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1890_189014


namespace NUMINAMATH_CALUDE_linear_term_coefficient_l1890_189017

/-- The coefficient of the linear term in the expansion of (x-1)(1/x + x)^6 is 20 -/
theorem linear_term_coefficient : ℕ :=
  20

#check linear_term_coefficient

end NUMINAMATH_CALUDE_linear_term_coefficient_l1890_189017


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1890_189071

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(5, 0, 4), (4, 1, 4), (3, 2, 4), (2, 3, 4), (1, 4, 4), (0, 5, 4),
   (3, 0, 0), (2, 1, 0), (1, 2, 0), (0, 3, 0)}

theorem diophantine_equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x^2 + y^2 - z^2 = 9 - 2*x*y} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1890_189071


namespace NUMINAMATH_CALUDE_square_has_four_axes_of_symmetry_l1890_189072

-- Define the shapes
inductive Shape
  | Square
  | Rhombus
  | Rectangle
  | IsoscelesTrapezoid

-- Define a function to count axes of symmetry
def axesOfSymmetry (s : Shape) : Nat :=
  match s with
  | Shape.Square => 4
  | Shape.Rhombus => 2
  | Shape.Rectangle => 2
  | Shape.IsoscelesTrapezoid => 1

-- Theorem statement
theorem square_has_four_axes_of_symmetry :
  ∀ s : Shape, axesOfSymmetry s = 4 → s = Shape.Square := by
  sorry

#check square_has_four_axes_of_symmetry

end NUMINAMATH_CALUDE_square_has_four_axes_of_symmetry_l1890_189072


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1890_189057

theorem sqrt_product_equality : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1890_189057


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l1890_189068

/-- The set of all numbers that can be represented as the sum of four consecutive positive integers -/
def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

/-- The greatest common divisor of all numbers in set B is 2 -/
theorem gcd_of_B_is_two : 
  ∃ (d : ℕ), d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l1890_189068


namespace NUMINAMATH_CALUDE_correct_squares_form_cube_net_l1890_189012

-- Define the grid paper with 5 squares
structure GridPaper :=
  (squares : Fin 5 → Bool)
  (shaded : Set (Fin 5))

-- Define a cube net
def is_cube_net (gp : GridPaper) : Prop :=
  ∃ (s1 s2 : Fin 5), s1 ≠ s2 ∧ 
    gp.shaded = {s1, s2} ∧
    -- Additional conditions to ensure it forms a valid cube net
    true  -- Placeholder for the specific geometric conditions

-- Theorem statement
theorem correct_squares_form_cube_net (gp : GridPaper) :
  is_cube_net {squares := gp.squares, shaded := {4, 5}} :=
sorry

end NUMINAMATH_CALUDE_correct_squares_form_cube_net_l1890_189012


namespace NUMINAMATH_CALUDE_dollar_hash_composition_l1890_189029

def dollar (N : ℝ) : ℝ := 2 * (N + 1)

def hash (N : ℝ) : ℝ := 0.5 * N + 1

theorem dollar_hash_composition : hash (dollar (dollar (dollar 5))) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dollar_hash_composition_l1890_189029


namespace NUMINAMATH_CALUDE_partnership_investment_l1890_189073

/-- Given the investments of partners A and B, the total profit, and A's share of the profit,
    calculate the investment of partner C in a partnership business. -/
theorem partnership_investment (a_invest b_invest total_profit a_profit : ℚ) (h1 : a_invest = 6300)
    (h2 : b_invest = 4200) (h3 : total_profit = 14200) (h4 : a_profit = 4260) :
    ∃ c_invest : ℚ, c_invest = 10500 ∧ 
    a_profit / a_invest = total_profit / (a_invest + b_invest + c_invest) :=
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l1890_189073


namespace NUMINAMATH_CALUDE_fraction_square_decimal_equivalent_l1890_189027

theorem fraction_square_decimal_equivalent : (1 / 9 : ℚ)^2 = 0.012345679012345678 := by
  sorry

end NUMINAMATH_CALUDE_fraction_square_decimal_equivalent_l1890_189027


namespace NUMINAMATH_CALUDE_morse_high_school_students_l1890_189038

/-- The number of students in the other three grades at Morse High School -/
def other_grades_students : ℕ := 1500

/-- The number of seniors at Morse High School -/
def seniors : ℕ := 300

/-- The percentage of seniors who have cars -/
def senior_car_percentage : ℚ := 40 / 100

/-- The percentage of students in other grades who have cars -/
def other_grades_car_percentage : ℚ := 10 / 100

/-- The percentage of all students who have cars -/
def total_car_percentage : ℚ := 15 / 100

theorem morse_high_school_students :
  (seniors * senior_car_percentage + other_grades_students * other_grades_car_percentage : ℚ) =
  (seniors + other_grades_students) * total_car_percentage :=
sorry

end NUMINAMATH_CALUDE_morse_high_school_students_l1890_189038


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l1890_189097

/-- Given two lines in a plane, this function returns the equation of the line 
    that is symmetric to the first line with respect to the second line. -/
def symmetricLine (l₁ l₂ : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The first given line l₁ -/
def l₁ : ℝ → ℝ → Prop :=
  fun x y ↦ 3 * x - y - 3 = 0

/-- The second given line l₂ -/
def l₂ : ℝ → ℝ → Prop :=
  fun x y ↦ x + y - 1 = 0

/-- The expected symmetric line l₃ -/
def l₃ : ℝ → ℝ → Prop :=
  fun x y ↦ x - 3 * y - 1 = 0

theorem symmetric_line_correct :
  symmetricLine l₁ l₂ = l₃ := by sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l1890_189097


namespace NUMINAMATH_CALUDE_apples_in_box_l1890_189019

/-- The number of apples in a box -/
def apples_per_box : ℕ := 14

/-- The number of people eating apples -/
def num_people : ℕ := 2

/-- The number of weeks spent eating apples -/
def num_weeks : ℕ := 3

/-- The number of boxes of apples -/
def num_boxes : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of apples eaten per person per day -/
def apples_per_person_per_day : ℕ := 1

theorem apples_in_box :
  apples_per_box * num_boxes = num_people * apples_per_person_per_day * num_weeks * days_per_week :=
by sorry

end NUMINAMATH_CALUDE_apples_in_box_l1890_189019


namespace NUMINAMATH_CALUDE_point_outside_circle_l1890_189087

theorem point_outside_circle (a b : ℝ) :
  (∃ x y, x^2 + y^2 = 1 ∧ a*x + b*y = 1) →
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1890_189087


namespace NUMINAMATH_CALUDE_min_value_sum_l1890_189051

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 19 / x + 98 / y = 1) :
  x + y ≥ 203 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1890_189051


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_geq_two_l1890_189004

theorem x_plus_reciprocal_geq_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_geq_two_l1890_189004


namespace NUMINAMATH_CALUDE_range_of_a_l1890_189008

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ≤ 0 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1890_189008


namespace NUMINAMATH_CALUDE_total_money_from_stone_sale_l1890_189047

def number_of_stones : ℕ := 8
def price_per_stone : ℕ := 1785

theorem total_money_from_stone_sale : number_of_stones * price_per_stone = 14280 := by
  sorry

end NUMINAMATH_CALUDE_total_money_from_stone_sale_l1890_189047


namespace NUMINAMATH_CALUDE_binomial_30_3_l1890_189005

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l1890_189005


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1890_189060

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1890_189060


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1890_189095

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 2) :
  5 * x / ((x - 4) * (x - 2)^2) = 5 / (x - 4) + (-5) / (x - 2) + (-5) / (x - 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1890_189095


namespace NUMINAMATH_CALUDE_megan_total_songs_l1890_189037

/-- Represents the number of albums bought in each genre -/
structure AlbumCounts where
  country : ℕ
  pop : ℕ
  rock : ℕ
  jazz : ℕ

/-- Represents the number of songs per album in each genre -/
structure SongsPerAlbum where
  country : ℕ
  pop : ℕ
  rock : ℕ
  jazz : ℕ

/-- Calculates the total number of songs bought -/
def totalSongs (counts : AlbumCounts) (songsPerAlbum : SongsPerAlbum) : ℕ :=
  counts.country * songsPerAlbum.country +
  counts.pop * songsPerAlbum.pop +
  counts.rock * songsPerAlbum.rock +
  counts.jazz * songsPerAlbum.jazz

/-- Theorem stating that Megan bought 160 songs in total -/
theorem megan_total_songs :
  let counts : AlbumCounts := ⟨2, 8, 5, 2⟩
  let songsPerAlbum : SongsPerAlbum := ⟨12, 7, 10, 15⟩
  totalSongs counts songsPerAlbum = 160 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_songs_l1890_189037


namespace NUMINAMATH_CALUDE_no_right_prism_with_diagonals_4_5_7_l1890_189033

theorem no_right_prism_with_diagonals_4_5_7 :
  ¬∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^2 + y^2 = 16 ∧ x^2 + z^2 = 25 ∧ y^2 + z^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_no_right_prism_with_diagonals_4_5_7_l1890_189033


namespace NUMINAMATH_CALUDE_cube_root_property_l1890_189006

theorem cube_root_property (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_property_l1890_189006


namespace NUMINAMATH_CALUDE_man_walking_speed_l1890_189054

/-- The speed of a man walking alongside a train --/
theorem man_walking_speed (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 900 →
  crossing_time = 53.99568034557235 →
  train_speed_kmh = 63 →
  ∃ (man_speed : ℝ), abs (man_speed - 0.832) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_man_walking_speed_l1890_189054


namespace NUMINAMATH_CALUDE_homework_pages_l1890_189020

theorem homework_pages (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 10 ∧ 
  reading_pages = math_pages + 3 →
  total_pages = math_pages + reading_pages →
  total_pages = 23 := by
sorry

end NUMINAMATH_CALUDE_homework_pages_l1890_189020


namespace NUMINAMATH_CALUDE_pencil_cost_l1890_189016

/-- Given that 120 pencils cost $36, prove that 3000 pencils cost $900 -/
theorem pencil_cost (pencils_per_box : ℕ) (cost_per_box : ℚ) (total_pencils : ℕ) :
  pencils_per_box = 120 →
  cost_per_box = 36 →
  total_pencils = 3000 →
  (cost_per_box / pencils_per_box) * total_pencils = 900 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l1890_189016


namespace NUMINAMATH_CALUDE_gcd_111_1850_l1890_189093

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_111_1850_l1890_189093


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1890_189015

-- Part 1
theorem calculation_proof : (1 * (1/2)⁻¹) + 2 * Real.cos (π/4) - Real.sqrt 8 + |1 - Real.sqrt 2| = 1 := by sorry

-- Part 2
theorem inequality_system_solution :
  ∀ x : ℝ, (x/2 + 1 > 0 ∧ 2*(x-1) + 3 ≥ 3*x) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1890_189015


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1890_189023

/-- Given a quadratic function f(x) = ax² + bx + 1 where a ≠ 0,
    if the solution set of f(x) > 0 is {x | x ∈ ℝ, x ≠ -b/(2a)},
    then the minimum value of (b⁴ + 4)/(4a) is 4. -/
theorem quadratic_minimum (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 > 0 ↔ x ≠ -b / (2 * a))) →
  (∃ m : ℝ, m = 4 ∧ ∀ y : ℝ, y = (b^4 + 4) / (4 * a) → y ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1890_189023


namespace NUMINAMATH_CALUDE_baduk_stone_difference_l1890_189042

theorem baduk_stone_difference (total : ℕ) (white : ℕ) (h1 : total = 928) (h2 : white = 713) :
  white - (total - white) = 498 := by
  sorry

end NUMINAMATH_CALUDE_baduk_stone_difference_l1890_189042


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1890_189025

theorem complex_expression_equality (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  4 * (x^y * (7^y * 24^x)) / (x*y) + 5 * (x * (13^y * 15^x)) - 2 * (y * (6^x * 28^y)) + 7 * (x*y * (3^x * 19^y)) / (x+y) = 11948716.8 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1890_189025


namespace NUMINAMATH_CALUDE_mean_temperature_is_87_5_l1890_189084

def temperatures : List ℝ := [82, 80, 83, 88, 90, 92, 90, 95]

theorem mean_temperature_is_87_5 :
  (temperatures.sum / temperatures.length : ℝ) = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_87_5_l1890_189084


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1890_189002

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 5 → x^2 - 4*x - 5 > 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x - 5 > 0 ∧ ¬(x > 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1890_189002


namespace NUMINAMATH_CALUDE_distance_driven_margies_car_distance_l1890_189031

/-- Proves that given a car's fuel efficiency and gas price, 
    we can calculate the distance driven with a certain amount of money. -/
theorem distance_driven (efficiency : ℝ) (gas_price : ℝ) (money : ℝ) :
  efficiency > 0 → gas_price > 0 → money > 0 →
  (efficiency * (money / gas_price) = 200) ↔ 
  (efficiency = 40 ∧ gas_price = 5 ∧ money = 25) :=
sorry

/-- Specific instance of the theorem for Margie's car -/
theorem margies_car_distance : 
  ∃ (efficiency gas_price money : ℝ),
    efficiency > 0 ∧ gas_price > 0 ∧ money > 0 ∧
    efficiency = 40 ∧ gas_price = 5 ∧ money = 25 ∧
    efficiency * (money / gas_price) = 200 :=
sorry

end NUMINAMATH_CALUDE_distance_driven_margies_car_distance_l1890_189031


namespace NUMINAMATH_CALUDE_max_digit_sum_diff_l1890_189056

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem max_digit_sum_diff :
  (∀ x : ℕ, x > 0 → S (x + 2019) - S x ≤ 12) ∧
  (∃ x : ℕ, x > 0 ∧ S (x + 2019) - S x = 12) :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_diff_l1890_189056
