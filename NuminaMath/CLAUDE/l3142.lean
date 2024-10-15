import Mathlib

namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3142_314262

/-- Given that Rahul will be 26 years old in 10 years and Deepak is currently 8 years old,
    prove that the ratio of Rahul's age to Deepak's age is 2:1. -/
theorem rahul_deepak_age_ratio :
  ∀ (rahul_age deepak_age : ℕ),
    rahul_age + 10 = 26 →
    deepak_age = 8 →
    rahul_age / deepak_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3142_314262


namespace NUMINAMATH_CALUDE_odot_1_43_47_l3142_314250

/-- Custom operation ⊙ -/
def odot (a b c : ℤ) : ℤ := a * b * c + (a * b + b * c + c * a) - (a + b + c)

/-- Theorem stating that 1 ⊙ 43 ⊙ 47 = 4041 -/
theorem odot_1_43_47 : odot 1 43 47 = 4041 := by
  sorry

end NUMINAMATH_CALUDE_odot_1_43_47_l3142_314250


namespace NUMINAMATH_CALUDE_fruit_shop_apples_l3142_314244

theorem fruit_shop_apples (total : ℕ) 
  (h1 : (3 : ℚ) / 10 * total + (4 : ℚ) / 10 * total = 140) : total = 200 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_apples_l3142_314244


namespace NUMINAMATH_CALUDE_functional_equation_not_surjective_l3142_314242

/-- A function from reals to natural numbers satisfying a specific functional equation -/
def FunctionalEquation (f : ℝ → ℕ) : Prop :=
  ∀ x y : ℝ, f (x + 1 / (f y : ℝ)) = f (y + 1 / (f x : ℝ))

/-- Theorem stating that a function satisfying the functional equation cannot map onto all natural numbers -/
theorem functional_equation_not_surjective (f : ℝ → ℕ) (h : FunctionalEquation f) : 
  ¬(Set.range f = Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_not_surjective_l3142_314242


namespace NUMINAMATH_CALUDE_tadpole_fish_ratio_l3142_314223

/-- The ratio of initial tadpoles to initial fish in a pond -/
theorem tadpole_fish_ratio :
  ∀ (initial_tadpoles : ℕ) (initial_fish : ℕ),
  initial_fish = 50 →
  ∃ (remaining_fish : ℕ) (remaining_tadpoles : ℕ),
  remaining_fish = initial_fish - 7 ∧
  remaining_tadpoles = initial_tadpoles / 2 ∧
  remaining_tadpoles = remaining_fish + 32 →
  (initial_tadpoles : ℚ) / initial_fish = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_tadpole_fish_ratio_l3142_314223


namespace NUMINAMATH_CALUDE_window_width_calculation_l3142_314251

/-- Calculates the width of each window in a room given the room dimensions,
    door dimensions, number of windows, window height, cost per square foot,
    and total cost of whitewashing. -/
theorem window_width_calculation (room_length room_width room_height : ℝ)
                                 (door_height door_width : ℝ)
                                 (num_windows : ℕ)
                                 (window_height : ℝ)
                                 (cost_per_sqft total_cost : ℝ) :
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_height = 6 ∧ door_width = 3 ∧
  num_windows = 3 ∧
  window_height = 3 ∧
  cost_per_sqft = 9 ∧
  total_cost = 8154 →
  ∃ (window_width : ℝ),
    window_width = 4 ∧
    total_cost = (2 * (room_length + room_width) * room_height -
                  door_height * door_width -
                  num_windows * window_height * window_width) * cost_per_sqft :=
by sorry

end NUMINAMATH_CALUDE_window_width_calculation_l3142_314251


namespace NUMINAMATH_CALUDE_amit_work_days_l3142_314246

theorem amit_work_days (amit_rate : ℚ) (ananthu_rate : ℚ) : 
  ananthu_rate = 1 / 45 →
  amit_rate * 3 + ananthu_rate * 36 = 1 →
  amit_rate = 1 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_amit_work_days_l3142_314246


namespace NUMINAMATH_CALUDE_elective_schemes_count_l3142_314266

/-- The number of elective courses available. -/
def total_courses : ℕ := 10

/-- The number of mutually exclusive courses. -/
def exclusive_courses : ℕ := 3

/-- The number of courses each student must elect. -/
def courses_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- 
Theorem: The number of ways to choose 3 courses out of 10, 
where 3 specific courses are mutually exclusive, is 98.
-/
theorem elective_schemes_count : 
  choose (total_courses - exclusive_courses) courses_to_choose + 
  exclusive_courses * choose (total_courses - exclusive_courses) (courses_to_choose - 1) = 98 := by
  sorry


end NUMINAMATH_CALUDE_elective_schemes_count_l3142_314266


namespace NUMINAMATH_CALUDE_initial_crayons_count_l3142_314293

/-- 
Given a person who:
1. Has an initial number of crayons
2. Loses half of their crayons
3. Buys 20 new crayons
4. Ends up with 29 crayons total
This theorem proves that the initial number of crayons was 18.
-/
theorem initial_crayons_count (initial : ℕ) 
  (h1 : initial / 2 + 20 = 29) : initial = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l3142_314293


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3142_314278

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3142_314278


namespace NUMINAMATH_CALUDE_chord_equation_l3142_314228

/-- Given a circle with equation x^2 + y^2 = 9 and a chord PQ with midpoint (1, 2),
    the equation of line PQ is x + 2y - 5 = 0 -/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9} → 
    (P ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9} ∧ 
     Q ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9})) →
  ((P.1 + Q.1) / 2 = 1 ∧ (P.2 + Q.2) / 2 = 2) →
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ 
                  a * Q.1 + b * Q.2 + c = 0 ∧
                  a = 1 ∧ b = 2 ∧ c = -5 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l3142_314228


namespace NUMINAMATH_CALUDE_b_is_positive_l3142_314264

theorem b_is_positive (a b : ℝ) (h : ∀ x : ℝ, (x - a)^2 + b > 0) : b > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_is_positive_l3142_314264


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3142_314275

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x < -1)) ↔ (∃ x : ℝ, x ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3142_314275


namespace NUMINAMATH_CALUDE_solve_cookies_problem_l3142_314237

def cookies_problem (total_cookies : ℕ) (cookies_per_guest : ℕ) : Prop :=
  total_cookies = 10 ∧ cookies_per_guest = 2 →
  total_cookies / cookies_per_guest = 5

theorem solve_cookies_problem : cookies_problem 10 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_cookies_problem_l3142_314237


namespace NUMINAMATH_CALUDE_elizabeth_haircut_l3142_314263

theorem elizabeth_haircut (first_day : ℝ) (second_day : ℝ) 
  (h1 : first_day = 0.38)
  (h2 : second_day = 0.5) :
  first_day + second_day = 0.88 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_haircut_l3142_314263


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3142_314200

/-- An isosceles triangle with two sides of length 8 and one side of length 4 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 8 → b = 8 → c = 4 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3142_314200


namespace NUMINAMATH_CALUDE_no_quadratic_composition_l3142_314210

/-- A quadratic polynomial is a polynomial of degree 2 -/
def IsQuadratic (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c

/-- The theorem states that there do not exist quadratic polynomials f and g
    such that their composition equals x^4 - 3x^3 + 3x^2 - x for all x -/
theorem no_quadratic_composition :
  ¬ ∃ (f g : ℝ → ℝ), IsQuadratic f ∧ IsQuadratic g ∧
    (∀ x, f (g x) = x^4 - 3*x^3 + 3*x^2 - x) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_composition_l3142_314210


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3142_314221

def is_valid_base_6_digit (n : ℕ) : Prop := n < 6
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8

def value_in_base_6 (a : ℕ) : ℕ := 6 * a + a
def value_in_base_8 (b : ℕ) : ℕ := 8 * b + b

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ),
    is_valid_base_6_digit a ∧
    is_valid_base_8_digit b ∧
    value_in_base_6 a = 63 ∧
    value_in_base_8 b = 63 ∧
    (∀ (x y : ℕ),
      is_valid_base_6_digit x ∧
      is_valid_base_8_digit y ∧
      value_in_base_6 x = value_in_base_8 y →
      value_in_base_6 x ≥ 63) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3142_314221


namespace NUMINAMATH_CALUDE_roof_ratio_l3142_314270

theorem roof_ratio (length width : ℝ) : 
  length * width = 576 →
  length - width = 36 →
  length / width = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_ratio_l3142_314270


namespace NUMINAMATH_CALUDE_sector_max_area_l3142_314285

/-- Given a sector with central angle α and radius R, if the perimeter of the sector
    is a constant C (C > 0), then the maximum area of the sector is C²/16. -/
theorem sector_max_area (α R C : ℝ) (h_pos : C > 0) :
  let perimeter := 2 * R + α * R
  let area := (1/2) * α * R^2
  (perimeter = C) → (∀ α' R', 2 * R' + α' * R' = C → (1/2) * α' * R'^2 ≤ C^2 / 16) ∧
  (∃ α' R', 2 * R' + α' * R' = C ∧ (1/2) * α' * R'^2 = C^2 / 16) :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l3142_314285


namespace NUMINAMATH_CALUDE_triathlon_bike_speed_l3142_314231

/-- Triathlon problem -/
theorem triathlon_bike_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ) 
  (run_distance run_speed : ℝ) 
  (bike_distance : ℝ) 
  (h1 : total_time = 3)
  (h2 : swim_distance = 0.5)
  (h3 : swim_speed = 1)
  (h4 : run_distance = 5)
  (h5 : run_speed = 5)
  (h6 : bike_distance = 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bike_speed_l3142_314231


namespace NUMINAMATH_CALUDE_complex_number_problem_l3142_314269

variable (z : ℂ)

theorem complex_number_problem (h1 : ∃ (r : ℝ), z + 2*I = r) 
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) : 
  z = 4 - 2*I ∧ Complex.abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3142_314269


namespace NUMINAMATH_CALUDE_triangle_existence_l3142_314277

/-- Represents a line in 2D space -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Theorem stating the existence of a triangle given specific conditions -/
theorem triangle_existence 
  (base_length : ℝ) 
  (base_direction : ℝ × ℝ) 
  (angle_difference : ℝ) 
  (third_vertex_line : Line) : 
  ∃ (t : Triangle), 
    -- The base of the triangle has the given length
    (Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = base_length) ∧
    -- The direction of the base matches the given direction
    ((t.B.1 - t.A.1, t.B.2 - t.A.2) = base_direction) ∧
    -- The difference between the base angles is as specified
    (∃ (α β : ℝ), α > β ∧ α - β = angle_difference) ∧
    -- The third vertex lies on the given line
    (∃ (k : ℝ), t.C = (third_vertex_line.point.1 + k * third_vertex_line.direction.1,
                       third_vertex_line.point.2 + k * third_vertex_line.direction.2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l3142_314277


namespace NUMINAMATH_CALUDE_intersection_M_N_l3142_314204

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {x ∈ U | x^2 ≤ x}

def N : Set Int := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3142_314204


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3142_314280

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (a^2 + 2*a - 3) + (a + 3)*Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3142_314280


namespace NUMINAMATH_CALUDE_unique_good_number_adjacent_to_power_of_two_l3142_314273

theorem unique_good_number_adjacent_to_power_of_two :
  ∃! n : ℕ, n > 0 ∧
  (∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ n = a^b) ∧
  (∃ t : ℕ, t > 0 ∧ (n = 2^t + 1 ∨ n = 2^t - 1)) ∧
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_good_number_adjacent_to_power_of_two_l3142_314273


namespace NUMINAMATH_CALUDE_fraction_equality_l3142_314220

theorem fraction_equality : (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3142_314220


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3142_314254

theorem divisibility_equivalence (n : ℕ+) :
  11 ∣ (n.val^5 + 5^n.val) ↔ 11 ∣ (n.val^5 * 5^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3142_314254


namespace NUMINAMATH_CALUDE_square_tiles_problem_l3142_314215

theorem square_tiles_problem (n : ℕ) : 
  (4 * n - 4 = 52) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_problem_l3142_314215


namespace NUMINAMATH_CALUDE_smallest_sum_of_primes_and_composites_l3142_314216

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 0 → d < n → n % d ≠ 0

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def formNumbers (digits : List ℕ) : List ℕ :=
  sorry

theorem smallest_sum_of_primes_and_composites (digits : List ℕ) :
  digits = [1, 2, 3, 4, 5, 6, 7, 8, 9] →
  (∃ nums : List ℕ,
    nums = formNumbers digits ∧
    (∀ n ∈ nums, isPrime n) ∧
    nums.sum = 318 ∧
    (∀ otherNums : List ℕ,
      otherNums = formNumbers digits →
      (∀ n ∈ otherNums, isPrime n) →
      otherNums.sum ≥ 318)) ∧
  (∃ nums : List ℕ,
    nums = formNumbers digits ∧
    (∀ n ∈ nums, isComposite n) ∧
    nums.sum = 127 ∧
    (∀ otherNums : List ℕ,
      otherNums = formNumbers digits →
      (∀ n ∈ otherNums, isComposite n) →
      otherNums.sum ≥ 127)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_sum_of_primes_and_composites_l3142_314216


namespace NUMINAMATH_CALUDE_f_a_equals_two_l3142_314271

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_a_equals_two (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ a → f x = f (-x)) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ a → f x = x^2 + 1) →
  f a = 2 := by sorry

end NUMINAMATH_CALUDE_f_a_equals_two_l3142_314271


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3142_314257

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 24*x^2 + 98*x - 75

-- Define the theorem
theorem root_sum_reciprocal (p q r A B C : ℝ) :
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →  -- p, q, r are distinct
  (f p = 0 ∧ f q = 0 ∧ f r = 0) →  -- p, q, r are roots of f
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    5 / (s^3 - 24*s^2 + 98*s - 75) = A / (s-p) + B / (s-q) + C / (s-r)) →
  1/A + 1/B + 1/C = 256 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3142_314257


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_calculate_profit_l3142_314249

-- Statement 1
theorem simplify_expression (a b : ℝ) :
  -3 * (a + b)^2 - 6 * (a + b)^2 + 8 * (a + b)^2 = -(a + b)^2 := by
  sorry

-- Statement 2
theorem calculate_expression (a b c d : ℝ) 
  (h1 : a - 2*b = 5) 
  (h2 : 2*b - c = -7) 
  (h3 : c - d = 12) :
  4*(a - c) + 4*(2*b - d) - 4*(2*b - c) = 40 := by
  sorry

-- Statement 3
theorem calculate_profit (initial_cost standard_price : ℝ) (sales : List ℝ) 
  (h1 : initial_cost = 400)
  (h2 : standard_price = 56)
  (h3 : sales = [-3, 7, -8, 9, -2, 0, -1, -6])
  (h4 : sales.length = 8) :
  (sales.sum + 8 * standard_price) - initial_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_calculate_profit_l3142_314249


namespace NUMINAMATH_CALUDE_l_shape_area_l3142_314245

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the cut-out position -/
structure CutOutPosition where
  fromRight : ℝ
  fromBottom : ℝ

theorem l_shape_area (large : Rectangle) (cutOut : Rectangle) (pos : CutOutPosition) : 
  large.width = 12 →
  large.height = 7 →
  cutOut.width = 4 →
  cutOut.height = 3 →
  pos.fromRight = large.width / 2 →
  pos.fromBottom = large.height / 2 →
  large.area - cutOut.area = 72 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_l3142_314245


namespace NUMINAMATH_CALUDE_second_agency_daily_charge_correct_l3142_314222

/-- The daily charge of the first agency -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first agency -/
def first_agency_mile_charge : ℝ := 0.14

/-- The per-mile charge of the second agency -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the agencies' costs are equal -/
def equal_cost_miles : ℝ := 25

/-- The daily charge of the second agency -/
def second_agency_daily_charge : ℝ := 18.25

theorem second_agency_daily_charge_correct :
  first_agency_daily_charge + first_agency_mile_charge * equal_cost_miles =
  second_agency_daily_charge + second_agency_mile_charge * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_second_agency_daily_charge_correct_l3142_314222


namespace NUMINAMATH_CALUDE_macaroon_problem_l3142_314217

theorem macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (total_bags : ℕ) (remaining_weight : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  total_bags = 4 →
  remaining_weight = 45 →
  total_macaroons % total_bags = 0 →
  (total_macaroons * weight_per_macaroon - remaining_weight) / (total_macaroons / total_bags * weight_per_macaroon) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_macaroon_problem_l3142_314217


namespace NUMINAMATH_CALUDE_bench_press_calculation_l3142_314229

theorem bench_press_calculation (initial_weight : ℝ) (injury_decrease : ℝ) (training_increase : ℝ) : 
  initial_weight = 500 →
  injury_decrease = 0.8 →
  training_increase = 3 →
  (initial_weight * (1 - injury_decrease) * training_increase) = 300 := by
sorry

end NUMINAMATH_CALUDE_bench_press_calculation_l3142_314229


namespace NUMINAMATH_CALUDE_absolute_value_multiplication_l3142_314287

theorem absolute_value_multiplication : -2 * |(-3)| = -6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_multiplication_l3142_314287


namespace NUMINAMATH_CALUDE_sector_area_90_degrees_l3142_314212

/-- The area of a sector with radius 2 and central angle 90° is π. -/
theorem sector_area_90_degrees : 
  let r : ℝ := 2
  let angle_degrees : ℝ := 90
  let angle_radians : ℝ := angle_degrees * (π / 180)
  let sector_area : ℝ := (1/2) * r^2 * angle_radians
  sector_area = π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_90_degrees_l3142_314212


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3142_314224

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis along y-axis, and focal length 4 -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1
  major_axis_y : m - 2 > 10 - m
  focal_length : ∃ (a b : ℝ), a^2 - b^2 = 16 ∧ a^2 = m - 2 ∧ b^2 = 10 - m

theorem ellipse_m_value (m : ℝ) (e : Ellipse m) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3142_314224


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3142_314256

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the complex fraction simplifies to the given result -/
theorem complex_fraction_simplification :
  (3 * (1 + i)^2) / (i - 1) = 3 - 3*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3142_314256


namespace NUMINAMATH_CALUDE_orderedPartitions_of_five_l3142_314299

/-- The number of ordered partitions of a positive integer n into positive integers -/
def orderedPartitions (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: The number of ordered partitions of 5 is 16 -/
theorem orderedPartitions_of_five :
  orderedPartitions 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_orderedPartitions_of_five_l3142_314299


namespace NUMINAMATH_CALUDE_sqrt_yz_times_sqrt_xy_l3142_314227

theorem sqrt_yz_times_sqrt_xy (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  Real.sqrt (y * z) * Real.sqrt (x * y) = 4 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_yz_times_sqrt_xy_l3142_314227


namespace NUMINAMATH_CALUDE_third_person_teeth_removal_l3142_314239

theorem third_person_teeth_removal (total_teeth : ℕ) (total_removed : ℕ) 
  (first_person_fraction : ℚ) (second_person_fraction : ℚ) (last_person_removed : ℕ) :
  total_teeth = 32 →
  total_removed = 40 →
  first_person_fraction = 1/4 →
  second_person_fraction = 3/8 →
  last_person_removed = 4 →
  (total_removed - 
    (first_person_fraction * total_teeth + 
     second_person_fraction * total_teeth + 
     last_person_removed)) / total_teeth = 1/2 := by
  sorry

#check third_person_teeth_removal

end NUMINAMATH_CALUDE_third_person_teeth_removal_l3142_314239


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3142_314209

theorem sum_of_two_numbers (x y : ℤ) : y = 2 * x - 3 → x = 14 → x + y = 39 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3142_314209


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l3142_314202

theorem baker_cakes_problem (pastries_made : ℕ) (pastries_sold : ℕ) (cakes_sold : ℕ) :
  pastries_made = 114 →
  pastries_sold = 154 →
  cakes_sold = 78 →
  pastries_sold = cakes_sold + 76 →
  cakes_sold = 78 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l3142_314202


namespace NUMINAMATH_CALUDE_degree_of_q_l3142_314247

-- Define polynomials p, q, and i
variable (p q i : Polynomial ℝ)

-- Define the relationship between i, p, and q
def poly_relation (p q i : Polynomial ℝ) : Prop :=
  i = p.comp q ^ 2 - q ^ 3

-- State the theorem
theorem degree_of_q (hp : Polynomial.degree p = 4)
                    (hi : Polynomial.degree i = 12)
                    (h_rel : poly_relation p q i) :
  Polynomial.degree q = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_q_l3142_314247


namespace NUMINAMATH_CALUDE_monomial_properties_l3142_314274

-- Define a monomial as a product of a coefficient and variables with non-negative integer exponents
structure Monomial (α : Type*) [CommRing α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the coefficient of a monomial
def coefficient {α : Type*} [CommRing α] (m : Monomial α) : α := m.coeff

-- Define the degree of a monomial
def degree {α : Type*} [CommRing α] (m : Monomial α) : Nat :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

-- The monomial -1/3 * x * y^2
def m : Monomial ℚ := ⟨-1/3, [(1, 1), (2, 2)]⟩

-- Theorem statement
theorem monomial_properties :
  coefficient m = -1/3 ∧ degree m = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_properties_l3142_314274


namespace NUMINAMATH_CALUDE_son_age_proof_l3142_314284

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_son_age_proof_l3142_314284


namespace NUMINAMATH_CALUDE_good_functions_count_l3142_314265

/-- A function f: ℤ → {1, 2, ..., n} is good if it satisfies the given condition -/
def IsGoodFunction (n : ℕ) (f : ℤ → Fin n) : Prop :=
  n ≥ 2 ∧ ∀ k : Fin (n-1), ∃ j : ℤ, ∀ m : ℤ,
    (f (m + j) : ℤ) ≡ (f (m + k) : ℤ) - (f m : ℤ) [ZMOD (n+1)]

/-- The number of good functions for a given n -/
def NumberOfGoodFunctions (n : ℕ) : ℕ := sorry

theorem good_functions_count (n : ℕ) :
  (n ≥ 2 ∧ NumberOfGoodFunctions n = n * Nat.totient n) ↔ Nat.Prime (n+1) :=
sorry

end NUMINAMATH_CALUDE_good_functions_count_l3142_314265


namespace NUMINAMATH_CALUDE_quadrilateral_wx_length_l3142_314260

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ × ℝ)

-- Define the circle
def Circle := (ℝ × ℝ) → Prop

-- Define the inscribed property
def inscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define the diameter property
def is_diameter (W Z : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a segment
def segment_length (A B : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_wx_length 
  (q : Quadrilateral) 
  (c : Circle) 
  (h1 : inscribed q c)
  (h2 : is_diameter q.W q.Z c)
  (h3 : segment_length q.W q.Z = 2)
  (h4 : segment_length q.X q.Z = segment_length q.Y q.W)
  (h5 : angle_measure q.W q.X q.Y = 72 * π / 180) :
  segment_length q.W q.X = Real.cos (18 * π / 180) * Real.sqrt (2 * (1 - Real.sin (18 * π / 180))) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_wx_length_l3142_314260


namespace NUMINAMATH_CALUDE_min_value_expression_l3142_314294

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), 
    x^2 + y^2 + 1/x^2 + y/x ≥ m) ∧
  (∃ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), 
    x^2 + y^2 + 1/x^2 + y/x = m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3142_314294


namespace NUMINAMATH_CALUDE_vector_addition_l3142_314236

def vector1 : Fin 2 → ℤ := ![5, -9]
def vector2 : Fin 2 → ℤ := ![-8, 14]

theorem vector_addition :
  (vector1 + vector2) = ![(-3 : ℤ), 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l3142_314236


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3142_314291

theorem quadratic_factorization_sum (a b c : ℤ) :
  (∀ x, x^2 + 19*x + 88 = (x + a) * (x + b)) →
  (∀ x, x^2 - 23*x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3142_314291


namespace NUMINAMATH_CALUDE_right_triangle_ab_length_l3142_314286

/-- Given a right triangle ABC in the x-y plane with ∠B = 90°, 
    if the length of AC is 25 and the slope of AC is 4/3, 
    then the length of AB is 15. -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25)
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_ab_length_l3142_314286


namespace NUMINAMATH_CALUDE_tony_packs_count_l3142_314283

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := 3

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of pens Kendra and Tony each keep for themselves -/
def pens_kept : ℕ := 2

/-- The number of friends who receive pens -/
def friends : ℕ := 14

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

theorem tony_packs_count :
  tony_packs * pens_per_pack + kendra_packs * pens_per_pack = 
  friends + 2 * pens_kept :=
by sorry

end NUMINAMATH_CALUDE_tony_packs_count_l3142_314283


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3142_314296

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 → m ≥ n) ∧
  n = 2519 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3142_314296


namespace NUMINAMATH_CALUDE_jessa_cupcakes_l3142_314295

/-- The number of cupcakes needed for a given number of classes and students per class -/
def cupcakes_needed (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  num_classes * students_per_class

theorem jessa_cupcakes : 
  let fourth_grade_cupcakes := cupcakes_needed 3 30
  let pe_class_cupcakes := cupcakes_needed 1 50
  fourth_grade_cupcakes + pe_class_cupcakes = 140 := by
  sorry

end NUMINAMATH_CALUDE_jessa_cupcakes_l3142_314295


namespace NUMINAMATH_CALUDE_friends_contribution_impossibility_l3142_314205

theorem friends_contribution_impossibility (a b c d e : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → e ≥ 0 →
  a + b + c + d + e > 0 →
  a + b < (a + b + c + d + e) / 3 →
  b + c < (a + b + c + d + e) / 3 →
  c + d < (a + b + c + d + e) / 3 →
  d + e < (a + b + c + d + e) / 3 →
  e + a < (a + b + c + d + e) / 3 →
  False :=
by sorry

end NUMINAMATH_CALUDE_friends_contribution_impossibility_l3142_314205


namespace NUMINAMATH_CALUDE_certain_number_problem_l3142_314207

theorem certain_number_problem : 
  (∃ n : ℕ, (∀ m > n, ¬∃ p q : ℕ+, 
    p > m ∧ 
    q > m ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43) ∧
  (∃ p q : ℕ+, 
    p > n ∧ 
    q > n ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43)) ∧
  (∀ n' > n, ¬∃ p q : ℕ+, 
    p > n' ∧ 
    q > n' ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43) →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3142_314207


namespace NUMINAMATH_CALUDE_problem_solution_l3142_314259

theorem problem_solution (x y z : ℝ) 
  (eq1 : 12 * x - 9 * y^2 = 7)
  (eq2 : 6 * y - 9 * z^2 = -2)
  (eq3 : 12 * z - 9 * x^2 = 4) :
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3142_314259


namespace NUMINAMATH_CALUDE_star_polygon_n_value_l3142_314297

/-- Represents an n-pointed regular star polygon -/
structure StarPolygon (n : ℕ) where
  -- All 2n edges are congruent (implicit in the structure)
  -- Alternate angles A₁, A₂, ..., Aₙ are congruent (implicit)
  -- Alternate angles B₁, B₂, ..., Bₙ are congruent (implicit)
  angle_A : ℝ  -- Acute angle at each Aᵢ
  angle_B : ℝ  -- Acute angle at each Bᵢ
  angle_diff : angle_B = angle_A + 20  -- Angle difference condition
  sum_external : n * (angle_A + angle_B) = 360  -- Sum of external angles

/-- Theorem: For a star polygon satisfying the given conditions, n = 36 -/
theorem star_polygon_n_value :
  ∀ (n : ℕ) (s : StarPolygon n), n = 36 :=
by sorry

end NUMINAMATH_CALUDE_star_polygon_n_value_l3142_314297


namespace NUMINAMATH_CALUDE_crocodile_count_l3142_314267

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 52

/-- The number of eyes each animal (frog or crocodile) has -/
def eyes_per_animal : ℕ := 2

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 6

theorem crocodile_count :
  num_crocodiles * eyes_per_animal + num_frogs * eyes_per_animal = total_eyes :=
by sorry

end NUMINAMATH_CALUDE_crocodile_count_l3142_314267


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3142_314288

/-- The quadratic function f(x) = 3x^2 + 6x + 9 has its minimum value at x = -1 -/
theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + 6 * x + 9
  ∀ y : ℝ, f (-1) ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3142_314288


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l3142_314211

theorem quadratic_complete_square (x : ℝ) : 
  (∃ r s : ℝ, (6 * x^2 - 24 * x - 54 = 0) ↔ ((x + r)^2 = s)) → 
  (∃ r s : ℝ, (6 * x^2 - 24 * x - 54 = 0) ↔ ((x + r)^2 = s) ∧ r + s = 11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l3142_314211


namespace NUMINAMATH_CALUDE_logarithm_equation_l3142_314208

theorem logarithm_equation (a b c : ℝ) 
  (eq1 : Real.log 3 = 2*a - b)
  (eq2 : Real.log 5 = a + c)
  (eq3 : Real.log 8 = 3 - 3*a - 3*c)
  (eq4 : Real.log 9 = 4*a - 2*b) :
  Real.log 15 = 3*a - b + c := by sorry

end NUMINAMATH_CALUDE_logarithm_equation_l3142_314208


namespace NUMINAMATH_CALUDE_guys_age_proof_l3142_314219

theorem guys_age_proof :
  ∃ (age : ℕ), 
    (((age + 8) * 8 - (age - 8) * 8) / 2 = age) ∧ 
    (age = 64) := by
  sorry

end NUMINAMATH_CALUDE_guys_age_proof_l3142_314219


namespace NUMINAMATH_CALUDE_divisibility_condition_l3142_314243

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3142_314243


namespace NUMINAMATH_CALUDE_santa_candy_remainders_l3142_314232

theorem santa_candy_remainders (N : ℕ) (x : ℕ) (h : N = 35 * x + 7) :
  N % 15 ∈ ({2, 7, 12} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_santa_candy_remainders_l3142_314232


namespace NUMINAMATH_CALUDE_girls_math_questions_l3142_314255

def total_questions (fiona_per_hour shirley_per_hour kiana_per_hour : ℕ) (hours : ℕ) : ℕ :=
  (fiona_per_hour + shirley_per_hour + kiana_per_hour) * hours

theorem girls_math_questions :
  ∀ (fiona_per_hour : ℕ),
    fiona_per_hour = 36 →
    ∀ (shirley_per_hour : ℕ),
      shirley_per_hour = 2 * fiona_per_hour →
      ∀ (kiana_per_hour : ℕ),
        kiana_per_hour = (fiona_per_hour + shirley_per_hour) / 2 →
        total_questions fiona_per_hour shirley_per_hour kiana_per_hour 2 = 324 :=
by
  sorry

#eval total_questions 36 72 54 2

end NUMINAMATH_CALUDE_girls_math_questions_l3142_314255


namespace NUMINAMATH_CALUDE_watch_price_equation_l3142_314234

/-- The original cost price of a watch satisfies the equation relating its discounted price and taxed price with profit. -/
theorem watch_price_equation (C : ℝ) : C > 0 → 0.855 * C + 540 = 1.2096 * C := by
  sorry

end NUMINAMATH_CALUDE_watch_price_equation_l3142_314234


namespace NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l3142_314279

/-- Given an angle θ in the second quadrant such that tan(θ + π/3) = 1/2,
    prove that sin θ + √3 cos θ = -2√5/5 -/
theorem sin_plus_sqrt3_cos (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (θ + π/3) = 1/2) : 
  Real.sin θ + Real.sqrt 3 * Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l3142_314279


namespace NUMINAMATH_CALUDE_johns_distance_conversion_l3142_314268

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 8^3 + d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- John's weekly hiking distance in base 8 is 3762 --/
def johns_distance_base8 : ℕ × ℕ × ℕ × ℕ := (3, 7, 6, 2)

theorem johns_distance_conversion :
  let (d₃, d₂, d₁, d₀) := johns_distance_base8
  base8_to_base10 d₃ d₂ d₁ d₀ = 2034 :=
by sorry

end NUMINAMATH_CALUDE_johns_distance_conversion_l3142_314268


namespace NUMINAMATH_CALUDE_area_triangle_OCD_l3142_314272

/-- Given a trapezoid ABCD and a parallelogram ABGH inscribed within it,
    this theorem calculates the area of triangle OCD. -/
theorem area_triangle_OCD (S_ABCD S_ABGH : ℝ) (h1 : S_ABCD = 320) (h2 : S_ABGH = 80) :
  ∃ (S_OCD : ℝ), S_OCD = 45 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_OCD_l3142_314272


namespace NUMINAMATH_CALUDE_integer_coordinates_cubic_l3142_314298

/-- A cubic function with integer coordinates for extrema and inflection point -/
structure IntegerCubic where
  n : ℤ
  p : ℤ
  c : ℤ

/-- The cubic function with the given coefficients -/
def cubic_function (f : IntegerCubic) (x : ℝ) : ℝ :=
  x^3 + 3 * f.n * x^2 + 3 * (f.n^2 - f.p^2) * x + f.c

/-- The first derivative of the cubic function -/
def cubic_derivative (f : IntegerCubic) (x : ℝ) : ℝ :=
  3 * x^2 + 6 * f.n * x + 3 * (f.n^2 - f.p^2)

/-- The second derivative of the cubic function -/
def cubic_second_derivative (f : IntegerCubic) (x : ℝ) : ℝ :=
  6 * x + 6 * f.n

/-- Theorem: The cubic function has integer coordinates for extrema and inflection point -/
theorem integer_coordinates_cubic (f : IntegerCubic) :
  ∃ (x1 x2 xi : ℤ),
    (cubic_derivative f x1 = 0 ∧ cubic_derivative f x2 = 0) ∧
    cubic_second_derivative f xi = 0 ∧
    (∀ x : ℤ, cubic_derivative f x = 0 → x = x1 ∨ x = x2) ∧
    (∀ x : ℤ, cubic_second_derivative f x = 0 → x = xi) :=
  sorry

end NUMINAMATH_CALUDE_integer_coordinates_cubic_l3142_314298


namespace NUMINAMATH_CALUDE_smallest_sum_l3142_314253

theorem smallest_sum (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = B - A ∧ B - A = d) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = B * r ∧ D = C * r) →  -- B, C, D form a geometric sequence
  C = (4 * B) / 3 →  -- C/B = 4/3
  (∀ A' B' C' D' : ℕ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = B' - A' ∧ B' - A' = d) →
    (∃ r : ℚ, C' = B' * r ∧ D' = C' * r) →
    C' = (4 * B') / 3 →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_l3142_314253


namespace NUMINAMATH_CALUDE_solve_school_supplies_problem_l3142_314281

/-- Represents the price and quantity of pens and notebooks -/
structure Supplies where
  pen_price : ℚ
  notebook_price : ℚ
  pen_quantity : ℕ
  notebook_quantity : ℕ

/-- Calculates the total cost of supplies -/
def total_cost (s : Supplies) : ℚ :=
  s.pen_price * s.pen_quantity + s.notebook_price * s.notebook_quantity

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  xiaofang_cost : ℚ
  xiaoliang_cost : ℚ
  xiaofang_supplies : Supplies
  xiaoliang_supplies : Supplies
  reward_fund : ℚ
  prize_sets : ℕ

/-- Theorem stating the solution to the problem -/
theorem solve_school_supplies_problem (c : ProblemConditions)
  (h1 : c.xiaofang_cost = 18)
  (h2 : c.xiaoliang_cost = 22)
  (h3 : c.xiaofang_supplies.pen_quantity = 2)
  (h4 : c.xiaofang_supplies.notebook_quantity = 3)
  (h5 : c.xiaoliang_supplies.pen_quantity = 3)
  (h6 : c.xiaoliang_supplies.notebook_quantity = 2)
  (h7 : c.reward_fund = 400)
  (h8 : c.prize_sets = 20)
  (h9 : total_cost c.xiaofang_supplies = c.xiaofang_cost)
  (h10 : total_cost c.xiaoliang_supplies = c.xiaoliang_cost) :
  ∃ (pen_price notebook_price : ℚ) (combinations : ℕ),
    pen_price = 6 ∧
    notebook_price = 2 ∧
    combinations = 4 ∧
    (∀ x y : ℕ, (x * pen_price + y * notebook_price) * c.prize_sets = c.reward_fund →
      (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 7) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_solve_school_supplies_problem_l3142_314281


namespace NUMINAMATH_CALUDE_students_passing_both_tests_l3142_314226

theorem students_passing_both_tests (total : ℕ) (long_jump : ℕ) (shot_put : ℕ) (failed_both : ℕ) :
  total = 50 →
  long_jump = 40 →
  shot_put = 31 →
  failed_both = 4 →
  ∃ x : ℕ, x = 25 ∧ total = (long_jump - x) + (shot_put - x) + x + failed_both :=
by sorry

end NUMINAMATH_CALUDE_students_passing_both_tests_l3142_314226


namespace NUMINAMATH_CALUDE_complex_fraction_opposite_parts_l3142_314206

theorem complex_fraction_opposite_parts (b : ℝ) : 
  let z₁ : ℂ := 1 + b * I
  let z₂ : ℂ := -2 + I
  (((z₁ / z₂).re = -(z₁ / z₂).im) → b = -1/3) ∧ 
  (b = -1/3 → (z₁ / z₂).re = -(z₁ / z₂).im) := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_opposite_parts_l3142_314206


namespace NUMINAMATH_CALUDE_original_class_size_l3142_314276

theorem original_class_size (x : ℕ) : 
  (x > 0) →                        -- Ensure the class has at least one student
  (40 * x + 12 * 32) / (x + 12) = 36 →  -- New average age equation
  x = 12 :=
by sorry

end NUMINAMATH_CALUDE_original_class_size_l3142_314276


namespace NUMINAMATH_CALUDE_chess_pawn_placement_l3142_314241

theorem chess_pawn_placement (n : ℕ) (hn : n = 5) : 
  (Finset.card (Finset.univ : Finset (Fin n → Fin n))) * 
  (Finset.card (Finset.univ : Finset (Equiv.Perm (Fin n)))) = 14400 :=
by sorry

end NUMINAMATH_CALUDE_chess_pawn_placement_l3142_314241


namespace NUMINAMATH_CALUDE_triangle_rectangle_area_l3142_314289

theorem triangle_rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : 
  square_area = 1600 ∧ rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2/5) * circle_radius
  let triangle_height := 3 * circle_radius
  let triangle_area := (1/2) * rectangle_length * triangle_height
  let rectangle_area := rectangle_length * rectangle_breadth
  triangle_area + rectangle_area = 1120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_rectangle_area_l3142_314289


namespace NUMINAMATH_CALUDE_years_until_arun_36_l3142_314235

/-- Proves the number of years that will pass before Arun's age is 36 years -/
theorem years_until_arun_36 (arun_age deepak_age : ℕ) (future_arun_age : ℕ) : 
  arun_age / deepak_age = 5 / 7 →
  deepak_age = 42 →
  future_arun_age = 36 →
  future_arun_age - arun_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_years_until_arun_36_l3142_314235


namespace NUMINAMATH_CALUDE_find_missing_mark_l3142_314203

/-- Represents the marks obtained in a subject, ranging from 0 to 100. -/
def Marks := Fin 101

/-- Calculates the sum of marks for the given subjects. -/
def sum_marks (marks : List Marks) : Nat :=
  marks.foldl (fun acc m => acc + m.val) 0

/-- Represents the problem of finding the missing subject mark. -/
theorem find_missing_mark (english : Marks) (math : Marks) (physics : Marks) (chemistry : Marks)
    (average : Nat) (h_average : average = 69) (h_english : english.val = 66)
    (h_math : math.val = 65) (h_physics : physics.val = 77) (h_chemistry : chemistry.val = 62) :
    ∃ (biology : Marks), sum_marks [english, math, physics, chemistry, biology] / 5 = average :=
  sorry

end NUMINAMATH_CALUDE_find_missing_mark_l3142_314203


namespace NUMINAMATH_CALUDE_closest_to_zero_minus_one_closest_l3142_314261

def integers : List ℤ := [-1, 2, -3, 4, -5]

theorem closest_to_zero (n : ℤ) (h : n ∈ integers) : 
  ∀ m ∈ integers, |n| ≤ |m| :=
by
  sorry

theorem minus_one_closest : 
  ∃ n ∈ integers, ∀ m ∈ integers, |n| ≤ |m| ∧ n = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_closest_to_zero_minus_one_closest_l3142_314261


namespace NUMINAMATH_CALUDE_exponent_division_l3142_314230

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3142_314230


namespace NUMINAMATH_CALUDE_direct_square_root_most_suitable_l3142_314214

/-- The quadratic equation to be solved -/
def quadratic_equation (x : ℝ) : Prop := (x - 1)^2 = 4

/-- Possible solution methods for quadratic equations -/
inductive SolutionMethod
  | CompletingSquare
  | QuadraticFormula
  | Factoring
  | DirectSquareRoot

/-- Predicate to determine if a method is the most suitable for solving a given equation -/
def is_most_suitable_method (eq : ℝ → Prop) (method : SolutionMethod) : Prop :=
  ∀ other_method : SolutionMethod, method = other_method ∨ 
    (∃ (complexity_measure : SolutionMethod → ℕ), 
      complexity_measure method < complexity_measure other_method)

/-- Theorem stating that the direct square root method is the most suitable for the given equation -/
theorem direct_square_root_most_suitable :
  is_most_suitable_method quadratic_equation SolutionMethod.DirectSquareRoot :=
sorry

end NUMINAMATH_CALUDE_direct_square_root_most_suitable_l3142_314214


namespace NUMINAMATH_CALUDE_initial_watermelons_l3142_314225

theorem initial_watermelons (eaten : ℕ) (left : ℕ) (initial : ℕ) : 
  eaten = 3 → left = 1 → initial = eaten + left → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_watermelons_l3142_314225


namespace NUMINAMATH_CALUDE_marias_car_trip_l3142_314282

theorem marias_car_trip (total_distance : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 400)
  (h2 : remaining_distance = 150) : 
  ∃ x : ℝ, x * total_distance + (1/4) * (total_distance - x * total_distance) = total_distance - remaining_distance ∧ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_marias_car_trip_l3142_314282


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3142_314238

/-- The length of the major axis of the ellipse x²/5 + y²/2 = 1 is 2√5 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 5 + y^2 / 2 = 1}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3142_314238


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l3142_314233

theorem equation_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l3142_314233


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l3142_314252

theorem factorization_left_to_right (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l3142_314252


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l3142_314201

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y → 1/(x-1) + 2/(y-2) ≥ 2 :=
by
  sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y ∧ 1/(x-1) + 2/(y-2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l3142_314201


namespace NUMINAMATH_CALUDE_min_length_line_segment_ellipse_l3142_314248

/-- The minimum length of a line segment AB on an ellipse -/
theorem min_length_line_segment_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}
  ∃ (A B : ℝ × ℝ), A ∈ ellipse ∧ B ∈ ellipse ∧ 
    (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA ⊥ OB
    ∀ (C D : ℝ × ℝ), C ∈ ellipse → D ∈ ellipse → (C.1 * D.1 + C.2 * D.2 = 0) →
      (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 ≤ (C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 ∧
      (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (2 * a * b * Real.sqrt (a ^ 2 + b ^ 2) / (a ^ 2 + b ^ 2)) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_length_line_segment_ellipse_l3142_314248


namespace NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l3142_314290

theorem smallest_b_in_geometric_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  (∃ r : ℝ, 0 < r ∧ a * r = b ∧ b * r = c) →  -- geometric sequence condition
  a * b * c = 125 →  -- product condition
  ∀ x : ℝ, (0 < x ∧ ∃ y z : ℝ, 0 < y ∧ 0 < z ∧ 
    (∃ r : ℝ, 0 < r ∧ y * r = x ∧ x * r = z) ∧ 
    y * x * z = 125) → 
  5 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l3142_314290


namespace NUMINAMATH_CALUDE_chloe_trivia_points_l3142_314213

/-- Chloe's trivia game points calculation -/
theorem chloe_trivia_points 
  (first_round : ℕ) 
  (last_round_loss : ℕ) 
  (total_points : ℕ) 
  (h1 : first_round = 40)
  (h2 : last_round_loss = 4)
  (h3 : total_points = 86) :
  ∃ (second_round : ℕ), 
    first_round + second_round - last_round_loss = total_points ∧ 
    second_round = 50 := by
sorry

end NUMINAMATH_CALUDE_chloe_trivia_points_l3142_314213


namespace NUMINAMATH_CALUDE_boys_pass_percentage_l3142_314292

theorem boys_pass_percentage (total_candidates : ℕ) (girls : ℕ) (girls_pass_rate : ℚ) (total_fail_rate : ℚ) :
  total_candidates = 2000 →
  girls = 900 →
  girls_pass_rate = 32 / 100 →
  total_fail_rate = 647 / 1000 →
  let boys := total_candidates - girls
  let total_pass_rate := 1 - total_fail_rate
  let total_pass := total_pass_rate * total_candidates
  let girls_pass := girls_pass_rate * girls
  let boys_pass := total_pass - girls_pass
  let boys_pass_rate := boys_pass / boys
  boys_pass_rate = 38 / 100 := by
sorry

end NUMINAMATH_CALUDE_boys_pass_percentage_l3142_314292


namespace NUMINAMATH_CALUDE_trapezoid_division_common_side_l3142_314240

theorem trapezoid_division_common_side
  (a b k p : ℝ)
  (h1 : a > b)
  (h2 : k > 0)
  (h3 : p > 0) :
  let x := Real.sqrt ((k * a^2 + p * b^2) / (p + k))
  ∃ (h1 h2 : ℝ), 
    h1 > 0 ∧ h2 > 0 ∧
    (b + x) * h1 / ((a + x) * h2) = k / p ∧
    x > b ∧ x < a :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_division_common_side_l3142_314240


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l3142_314258

/-- The number of tickets Amanda needs to sell on the third day -/
def tickets_to_sell_day3 (total_goal : ℕ) (sold_day1 : ℕ) (sold_day2 : ℕ) : ℕ :=
  total_goal - (sold_day1 + sold_day2)

/-- Theorem stating that Amanda needs to sell 28 tickets on the third day -/
theorem amanda_ticket_sales : tickets_to_sell_day3 80 20 32 = 28 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l3142_314258


namespace NUMINAMATH_CALUDE_tim_running_hours_l3142_314218

/-- The number of days Tim originally ran per week -/
def original_days : ℕ := 3

/-- The number of extra days Tim added to her running schedule -/
def extra_days : ℕ := 2

/-- The number of hours Tim runs in the morning each day she runs -/
def morning_hours : ℕ := 1

/-- The number of hours Tim runs in the evening each day she runs -/
def evening_hours : ℕ := 1

/-- Theorem stating that Tim now runs 10 hours a week -/
theorem tim_running_hours : 
  (original_days + extra_days) * (morning_hours + evening_hours) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tim_running_hours_l3142_314218
