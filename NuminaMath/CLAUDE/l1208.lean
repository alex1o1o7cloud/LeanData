import Mathlib

namespace NUMINAMATH_CALUDE_prime_divisors_of_p_cubed_plus_three_l1208_120897

theorem prime_divisors_of_p_cubed_plus_three (p : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime (p^2 + 2)) :
  ∃ (a b c : ℕ), Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ p^3 + 3 = a * b * c :=
sorry

end NUMINAMATH_CALUDE_prime_divisors_of_p_cubed_plus_three_l1208_120897


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l1208_120824

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions --/
theorem field_length_width_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    pond_side = 8 →
    field_length = 112 →
    pond_side^2 = (1/98) * (field_length * field_width) →
    field_length / field_width = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l1208_120824


namespace NUMINAMATH_CALUDE_product_of_sum_of_logs_l1208_120851

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem product_of_sum_of_logs (a b : ℝ) (h : log10 a + log10 b = 1) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_logs_l1208_120851


namespace NUMINAMATH_CALUDE_square_sum_divisors_l1208_120849

theorem square_sum_divisors (n : ℕ) : n ≥ 2 →
  (∃ a b : ℕ, a > 1 ∧ a ∣ n ∧ b ∣ n ∧
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    n = a^2 + b^2) →
  n = 5 ∨ n = 8 ∨ n = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_divisors_l1208_120849


namespace NUMINAMATH_CALUDE_mary_vacuum_charges_l1208_120812

/-- The number of times Mary needs to charge her vacuum cleaner to clean her whole house -/
def charges_needed (battery_duration : ℕ) (time_per_room : ℕ) (num_bedrooms : ℕ) (num_kitchen : ℕ) (num_living_room : ℕ) : ℕ :=
  let total_rooms := num_bedrooms + num_kitchen + num_living_room
  let total_time := time_per_room * total_rooms
  (total_time + battery_duration - 1) / battery_duration

theorem mary_vacuum_charges :
  charges_needed 10 4 3 1 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_vacuum_charges_l1208_120812


namespace NUMINAMATH_CALUDE_transaction_difference_l1208_120856

theorem transaction_difference : 
  ∀ (mabel anthony cal jade : ℕ),
    mabel = 90 →
    anthony = mabel + mabel / 10 →
    cal = (2 * anthony) / 3 →
    jade = 81 →
    jade - cal = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_transaction_difference_l1208_120856


namespace NUMINAMATH_CALUDE_alphabet_value_problem_l1208_120884

theorem alphabet_value_problem (H M A T E : ℤ) : 
  H = 8 →
  M + A + T + H = 31 →
  T + E + A + M = 40 →
  M + E + E + T = 44 →
  M + A + T + E = 39 →
  A = 12 := by
sorry

end NUMINAMATH_CALUDE_alphabet_value_problem_l1208_120884


namespace NUMINAMATH_CALUDE_average_score_is_71_l1208_120822

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def biology_score : ℕ := 85

def total_score : ℕ := mathematics_score + science_score + social_studies_score + english_score + biology_score
def number_of_subjects : ℕ := 5

theorem average_score_is_71 : (total_score : ℚ) / number_of_subjects = 71 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_71_l1208_120822


namespace NUMINAMATH_CALUDE_total_tickets_is_56_l1208_120892

/-- The total number of tickets spent during three trips to the arcade -/
def total_tickets : ℕ :=
  let first_trip := 2 + 10 + 2
  let second_trip := 3 + 7 + 5
  let third_trip := 8 + 15 + 4
  first_trip + second_trip + third_trip

/-- Theorem stating that the total number of tickets spent is 56 -/
theorem total_tickets_is_56 : total_tickets = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_56_l1208_120892


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1208_120883

/-- Given an ellipse C with equation x^2/4 + y^2/m^2 = 1 and focal length 4,
    prove that the length of its major axis is 4√2. -/
theorem ellipse_major_axis_length (m : ℝ) :
  let C := {(x, y) : ℝ × ℝ | x^2/4 + y^2/m^2 = 1}
  let focal_length : ℝ := 4
  ∃ (major_axis_length : ℝ), major_axis_length = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1208_120883


namespace NUMINAMATH_CALUDE_log_equation_solution_l1208_120810

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ y : ℝ, y > 0 ∧ log y 81 = 4/2 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1208_120810


namespace NUMINAMATH_CALUDE_problem_solution_l1208_120865

theorem problem_solution (t : ℝ) :
  let x := 3 - 2*t
  let y := t^2 + 3*t + 6
  x = -1 → y = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1208_120865


namespace NUMINAMATH_CALUDE_small_bottle_price_approx_l1208_120885

/-- The price of a small bottle that results in the given average price -/
def price_small_bottle (large_quantity : ℕ) (small_quantity : ℕ) (large_price : ℚ) (average_price : ℚ) : ℚ :=
  ((average_price * (large_quantity + small_quantity : ℚ)) - (large_quantity : ℚ) * large_price) / (small_quantity : ℚ)

/-- Theorem stating that the price of small bottles is approximately $1.38 -/
theorem small_bottle_price_approx :
  let large_quantity : ℕ := 1300
  let small_quantity : ℕ := 750
  let large_price : ℚ := 189/100
  let average_price : ℚ := 17034/10000
  let calculated_price := price_small_bottle large_quantity small_quantity large_price average_price
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |calculated_price - 138/100| < ε :=
sorry

end NUMINAMATH_CALUDE_small_bottle_price_approx_l1208_120885


namespace NUMINAMATH_CALUDE_expression_simplification_l1208_120895

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) (h3 : x ≠ -1) : 
  (2*x + 4) / (x^2 - 1) / ((x + 2) / (x^2 - 2*x + 1)) - 2*x / (x + 1) = -2 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1208_120895


namespace NUMINAMATH_CALUDE_bbq_ice_cost_l1208_120803

/-- The cost of ice for Chad's BBQ --/
theorem bbq_ice_cost (people : ℕ) (ice_per_person : ℕ) (bags_per_pack : ℕ) (price_per_pack : ℚ) : 
  people = 15 →
  ice_per_person = 2 →
  bags_per_pack = 10 →
  price_per_pack = 3 →
  (people * ice_per_person : ℚ) / bags_per_pack * price_per_pack = 9 :=
by
  sorry

#check bbq_ice_cost

end NUMINAMATH_CALUDE_bbq_ice_cost_l1208_120803


namespace NUMINAMATH_CALUDE_total_insects_theorem_l1208_120833

/-- The number of geckos -/
def num_geckos : ℕ := 5

/-- The number of insects eaten by each gecko -/
def insects_per_gecko : ℕ := 6

/-- The number of lizards -/
def num_lizards : ℕ := 3

/-- The number of insects eaten by each lizard -/
def insects_per_lizard : ℕ := 2 * insects_per_gecko

/-- The total number of insects eaten by both geckos and lizards -/
def total_insects_eaten : ℕ := num_geckos * insects_per_gecko + num_lizards * insects_per_lizard

theorem total_insects_theorem : total_insects_eaten = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_theorem_l1208_120833


namespace NUMINAMATH_CALUDE_congruence_problem_l1208_120838

theorem congruence_problem (m : ℕ) : 
  163 * 937 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1208_120838


namespace NUMINAMATH_CALUDE_area_outside_triangle_l1208_120802

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of side PQ -/
  pq : ℝ
  /-- The length of the hypotenuse PR -/
  pr : ℝ
  /-- The circle is inscribed in the triangle -/
  inscribed : Bool
  /-- The triangle is right-angled at Q -/
  right_angle : Bool
  /-- The circle is tangent to PQ at M and to QR at N -/
  tangent_points : Bool
  /-- The points diametrically opposite M and N lie on PR -/
  diametric_points : Bool

/-- The theorem stating the area of the portion of the circle outside the triangle -/
theorem area_outside_triangle (t : RightTriangleWithInscribedCircle) 
  (h1 : t.pq = 9)
  (h2 : t.pr = 15)
  (h3 : t.inscribed)
  (h4 : t.right_angle)
  (h5 : t.tangent_points)
  (h6 : t.diametric_points) :
  ∃ (area : ℝ), area = 28.125 * Real.pi - 56.25 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_triangle_l1208_120802


namespace NUMINAMATH_CALUDE_inequality_proof_l1208_120887

theorem inequality_proof (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b < 0) 
  (hab : a < b) 
  (hbc : b < c) : 
  a + b < b + c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1208_120887


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l1208_120823

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = (1 / (x - 1)) + 1) → x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l1208_120823


namespace NUMINAMATH_CALUDE_b_age_is_eight_l1208_120819

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The total of their ages is 22
    Prove that b is 8 years old. -/
theorem b_age_is_eight (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 22) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_eight_l1208_120819


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1208_120866

open Set

def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def U : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | -2 < x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1208_120866


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1208_120831

/-- Given that (1+i)(x-yi) = 2, where x and y are real numbers and i is the imaginary unit, prove that x + y = 2 -/
theorem complex_equation_sum (x y : ℝ) : (Complex.I + 1) * (x - y * Complex.I) = 2 → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1208_120831


namespace NUMINAMATH_CALUDE_second_class_size_l1208_120858

/-- Given two classes of students, prove that the second class has 50 students. -/
theorem second_class_size (first_class_size : ℕ) (first_class_avg : ℝ) 
  (second_class_avg : ℝ) (total_avg : ℝ) :
  first_class_size = 30 →
  first_class_avg = 30 →
  second_class_avg = 60 →
  total_avg = 48.75 →
  ∃ (second_class_size : ℕ),
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size : ℝ) = total_avg ∧
    second_class_size = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_class_size_l1208_120858


namespace NUMINAMATH_CALUDE_sum_P_equals_97335_l1208_120813

/-- P(n) represents the product of all non-zero digits of a positive integer n -/
def P (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n < 10 then n
  else let d := n % 10
       let r := n / 10
       if d = 0 then P r
       else d * P r

/-- The sum of P(n) for n from 1 to 999 -/
def sum_P : ℕ := (List.range 999).map (fun i => P (i + 1)) |>.sum

theorem sum_P_equals_97335 : sum_P = 97335 := by
  sorry

end NUMINAMATH_CALUDE_sum_P_equals_97335_l1208_120813


namespace NUMINAMATH_CALUDE_georges_socks_l1208_120867

theorem georges_socks (initial_socks : ℕ) (thrown_away : ℕ) (final_socks : ℕ) :
  initial_socks = 28 →
  thrown_away = 4 →
  final_socks = 60 →
  final_socks - (initial_socks - thrown_away) = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_georges_socks_l1208_120867


namespace NUMINAMATH_CALUDE_arrangements_count_l1208_120829

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the condition that B and C must be adjacent -/
def bc_adjacent : Prop := True

/-- Represents the condition that A cannot stand at either end -/
def a_not_at_ends : Prop := True

/-- The number of different arrangements satisfying the given conditions -/
def num_arrangements : ℕ := 144

/-- Theorem stating that the number of arrangements is 144 -/
theorem arrangements_count :
  (num_students = 6) →
  bc_adjacent →
  a_not_at_ends →
  num_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1208_120829


namespace NUMINAMATH_CALUDE_divisibility_by_8640_l1208_120874

theorem divisibility_by_8640 (x : ℤ) : ∃ k : ℤ, x^9 - 6*x^7 + 9*x^5 - 4*x^3 = 8640 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_8640_l1208_120874


namespace NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l1208_120870

theorem periodic_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 + (3 * (2 / 99))) → (2 + (3 * (2 / 99)) = 68 / 33) := by
  sorry

end NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l1208_120870


namespace NUMINAMATH_CALUDE_sphere_in_cone_l1208_120834

theorem sphere_in_cone (b d : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * Real.sqrt d - b
  sphere_radius = (cone_base_radius * cone_height) / (Real.sqrt (cone_base_radius^2 + cone_height^2) + cone_height) →
  b + d = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cone_l1208_120834


namespace NUMINAMATH_CALUDE_min_students_is_minimum_l1208_120837

/-- The minimum number of students in the circle -/
def min_students : ℕ := 37

/-- Congcong's numbers are congruent modulo the number of students -/
axiom congcong_congruence : 25 ≡ 99 [ZMOD min_students]

/-- Mingming's numbers are congruent modulo the number of students -/
axiom mingming_congruence : 8 ≡ 119 [ZMOD min_students]

/-- The number of students is the minimum positive integer satisfying both congruences -/
theorem min_students_is_minimum :
  ∀ m : ℕ, m > 0 → (25 ≡ 99 [ZMOD m] ∧ 8 ≡ 119 [ZMOD m]) → m ≥ min_students :=
by sorry

end NUMINAMATH_CALUDE_min_students_is_minimum_l1208_120837


namespace NUMINAMATH_CALUDE_corner_rectangles_area_sum_l1208_120843

/-- Given a square with side length 100 cm divided into 9 rectangles,
    where the central rectangle has dimensions 40 cm × 60 cm,
    the sum of the areas of the four corner rectangles is 2400 cm². -/
theorem corner_rectangles_area_sum (x y : ℝ) : x > 0 → y > 0 →
  x + 40 + (100 - x - 40) = 100 →
  y + 60 + (100 - y - 60) = 100 →
  x * y + (60 - x) * y + x * (40 - y) + (60 - x) * (40 - y) = 2400 := by
  sorry

#check corner_rectangles_area_sum

end NUMINAMATH_CALUDE_corner_rectangles_area_sum_l1208_120843


namespace NUMINAMATH_CALUDE_seventeen_in_base_three_l1208_120861

/-- Represents a number in base 3 as a list of digits (least significant digit first) -/
def BaseThreeRepresentation := List Nat

/-- Converts a base 3 representation to its decimal value -/
def toDecimal (rep : BaseThreeRepresentation) : Nat :=
  rep.enum.foldl (fun acc (i, digit) => acc + digit * (3^i)) 0

/-- Theorem: The base-3 representation of 17 is [2, 2, 1] (which represents 122₃) -/
theorem seventeen_in_base_three :
  ∃ (rep : BaseThreeRepresentation), toDecimal rep = 17 ∧ rep = [2, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_base_three_l1208_120861


namespace NUMINAMATH_CALUDE_part_I_part_II_l1208_120862

-- Define the function f
def f (a b x : ℝ) := 2 * x^2 - 2 * a * x + b

-- Define set A
def A (a b : ℝ) := {x : ℝ | f a b x > 0}

-- Define set B
def B (t : ℝ) := {x : ℝ | |x - t| ≤ 1}

-- Theorem for part (I)
theorem part_I (a b : ℝ) (h1 : f a b (-1) = -8) (h2 : ∀ x : ℝ, f a b x ≥ f a b (-1)) :
  (Set.univ \ A a b) ∪ B 1 = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

-- Theorem for part (II)
theorem part_II (a b : ℝ) (h1 : f a b (-1) = -8) (h2 : ∀ x : ℝ, f a b x ≥ f a b (-1)) :
  {t : ℝ | A a b ∩ B t = ∅} = {t : ℝ | -2 ≤ t ∧ t ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1208_120862


namespace NUMINAMATH_CALUDE_rachels_math_homework_l1208_120818

/-- Rachel's homework problem -/
theorem rachels_math_homework (reading_pages : ℕ) (extra_math_pages : ℕ) : 
  reading_pages = 4 → extra_math_pages = 3 → reading_pages + extra_math_pages = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_rachels_math_homework_l1208_120818


namespace NUMINAMATH_CALUDE_product_of_imaginary_parts_l1208_120820

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_eq (z : ℂ) : Prop := z^2 + 3*z + (4 - 7*i) = 0

-- Define a function to get the imaginary part of a complex number
def im (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem product_of_imaginary_parts :
  ∃ (z1 z2 : ℂ), quadratic_eq z1 ∧ quadratic_eq z2 ∧ z1 ≠ z2 ∧ (im z1 * im z2 = -14) :=
sorry

end NUMINAMATH_CALUDE_product_of_imaginary_parts_l1208_120820


namespace NUMINAMATH_CALUDE_square_expression_is_perfect_square_l1208_120817

theorem square_expression_is_perfect_square (n k l : ℕ) 
  (h : n^2 + k^2 = 2 * l^2) : 
  ((2 * l - n - k) * (2 * l - n + k)) / 2 = (l - n)^2 :=
sorry

end NUMINAMATH_CALUDE_square_expression_is_perfect_square_l1208_120817


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1208_120888

theorem line_slope_intercept_product (m b : ℚ) (h1 : m = 1/3) (h2 : b = -3/4) :
  -1 < m * b ∧ m * b < 0 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1208_120888


namespace NUMINAMATH_CALUDE_total_capacity_is_57600_l1208_120826

/-- The total capacity of James' fleet of vans -/
def total_capacity : ℕ := by
  -- Define the number of vans
  let total_vans : ℕ := 6
  let large_vans : ℕ := 2
  let medium_van : ℕ := 1
  let extra_large_vans : ℕ := 3

  -- Define the capacities
  let base_capacity : ℕ := 8000
  let medium_capacity : ℕ := base_capacity - (base_capacity * 30 / 100)
  let extra_large_capacity : ℕ := base_capacity + (base_capacity * 50 / 100)

  -- Calculate total capacity
  exact large_vans * base_capacity + 
        medium_van * medium_capacity + 
        extra_large_vans * extra_large_capacity

/-- Theorem stating that the total capacity is 57600 gallons -/
theorem total_capacity_is_57600 : total_capacity = 57600 := by
  sorry

end NUMINAMATH_CALUDE_total_capacity_is_57600_l1208_120826


namespace NUMINAMATH_CALUDE_problem_statement_l1208_120852

theorem problem_statement (x y z : ℚ) : 
  x = 1/3 → y = 2/3 → z = x * y → 3 * x^2 * y^5 * z^3 = 768/1594323 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1208_120852


namespace NUMINAMATH_CALUDE_small_shape_placement_exists_l1208_120800

/-- Represents a shape with a given area -/
structure Shape where
  area : ℝ

/-- Represents an infinite grid with cells of a given area -/
structure Grid where
  cellArea : ℝ

/-- Represents a placement of a shape on a grid -/
structure Placement where
  shape : Shape
  grid : Grid

/-- Predicate to check if a placement covers any grid vertex -/
def coversVertex (p : Placement) : Prop :=
  sorry -- Definition omitted as it's not explicitly given in the problem

/-- Theorem stating that a shape smaller than a grid cell can be placed without covering vertices -/
theorem small_shape_placement_exists (s : Shape) (g : Grid) 
    (h : s.area < g.cellArea) : 
    ∃ (p : Placement), p.shape = s ∧ p.grid = g ∧ ¬coversVertex p :=
  sorry

end NUMINAMATH_CALUDE_small_shape_placement_exists_l1208_120800


namespace NUMINAMATH_CALUDE_distribute_5_4_l1208_120898

/-- The number of ways to distribute n different books to k students,
    with each student receiving at least one book. -/
def distribute (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n - (k.choose 3) * (k-3)^n

/-- Theorem stating that distributing 5 different books to 4 students,
    with each student receiving at least one book, results in 240 different schemes. -/
theorem distribute_5_4 : distribute 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_4_l1208_120898


namespace NUMINAMATH_CALUDE_number_puzzle_l1208_120804

theorem number_puzzle (x : ℝ) : (1/2 : ℝ) * x - 300 = 350 → (x + 200) * 2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1208_120804


namespace NUMINAMATH_CALUDE_ariella_interest_rate_l1208_120894

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem ariella_interest_rate :
  ∀ (daniella_initial ariella_initial ariella_final : ℝ),
  daniella_initial = 400 →
  ariella_initial = daniella_initial + 200 →
  ariella_final = 720 →
  ∃ (rate : ℝ), 
    simple_interest ariella_initial rate 2 = ariella_final ∧
    rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ariella_interest_rate_l1208_120894


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1208_120821

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1208_120821


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l1208_120886

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem volleyball_team_selection :
  (Nat.choose total_players starters) - (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 31460 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l1208_120886


namespace NUMINAMATH_CALUDE_lottery_jackpot_probability_l1208_120889

def num_megaballs : ℕ := 30
def num_winnerballs : ℕ := 49
def num_chosen_winnerballs : ℕ := 6
def lower_sum_bound : ℕ := 100
def upper_sum_bound : ℕ := 150

def N : ℕ := sorry -- Number of ways to choose 6 numbers from 49 that sum to [100, 150]

theorem lottery_jackpot_probability :
  ∃ (p : ℚ), p = (1 : ℚ) / num_megaballs * (N : ℚ) / (Nat.choose num_winnerballs num_chosen_winnerballs) :=
by sorry

end NUMINAMATH_CALUDE_lottery_jackpot_probability_l1208_120889


namespace NUMINAMATH_CALUDE_remainder_theorem_l1208_120891

theorem remainder_theorem : ∃ q : ℕ, 3^202 + 303 = (3^101 + 3^51 + 1) * q + 302 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1208_120891


namespace NUMINAMATH_CALUDE_earphone_cost_l1208_120882

/-- The cost of an earphone given weekly expenditure data -/
theorem earphone_cost (mean_expenditure : ℕ) (mon tue wed thu sat sun : ℕ) (pen notebook : ℕ) :
  mean_expenditure = 500 →
  mon = 450 →
  tue = 600 →
  wed = 400 →
  thu = 500 →
  sat = 550 →
  sun = 300 →
  pen = 30 →
  notebook = 50 →
  ∃ (earphone : ℕ), earphone = 7 * mean_expenditure - (mon + tue + wed + thu + sat + sun) - pen - notebook :=
by
  sorry

end NUMINAMATH_CALUDE_earphone_cost_l1208_120882


namespace NUMINAMATH_CALUDE_no_injective_function_exists_l1208_120869

theorem no_injective_function_exists : ¬∃ f : ℝ → ℝ, 
  Function.Injective f ∧ ∀ x : ℝ, f (x^2) - (f x)^2 ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_no_injective_function_exists_l1208_120869


namespace NUMINAMATH_CALUDE_wooden_block_volume_l1208_120881

/-- Represents a rectangular wooden block -/
structure WoodenBlock where
  length : ℝ
  baseArea : ℝ

/-- Calculates the volume of a rectangular wooden block -/
def volume (block : WoodenBlock) : ℝ :=
  block.length * block.baseArea

/-- Theorem: The volume of the wooden block is 864 cubic decimeters -/
theorem wooden_block_volume :
  ∀ (block : WoodenBlock),
    block.length = 72 →
    (3 - 1) * 2 * block.baseArea = 48 →
    volume block = 864 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_volume_l1208_120881


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l1208_120893

def a (n : ℕ) : ℚ := 9/2 - n

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ, a (n + 1) - a n = -1) ∧
  (Finset.sum (Finset.range 20) a = -120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l1208_120893


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l1208_120877

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 6

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_bird_count : total_birds = 72 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l1208_120877


namespace NUMINAMATH_CALUDE_A_subset_B_l1208_120864

variable {X : Type*} -- Domain of functions f and g
variable (f g : X → ℝ) -- Real-valued functions f and g
variable (a : ℝ) -- Real number a

def A (f g : X → ℝ) (a : ℝ) : Set X :=
  {x : X | |f x| + |g x| < a}

def B (f g : X → ℝ) (a : ℝ) : Set X :=
  {x : X | |f x + g x| < a}

theorem A_subset_B (h : a > 0) : A f g a ⊆ B f g a := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l1208_120864


namespace NUMINAMATH_CALUDE_solve_final_grade_problem_l1208_120830

def final_grade_problem (total_students : ℕ) (fraction_A fraction_B fraction_C : ℚ) : Prop :=
  let fraction_D := 1 - (fraction_A + fraction_B + fraction_C)
  let num_D := total_students - (total_students * (fraction_A + fraction_B + fraction_C)).floor
  (total_students = 100) ∧
  (fraction_A = 1/5) ∧
  (fraction_B = 1/4) ∧
  (fraction_C = 1/2) ∧
  (num_D = 5)

theorem solve_final_grade_problem :
  ∃ (total_students : ℕ) (fraction_A fraction_B fraction_C : ℚ),
    final_grade_problem total_students fraction_A fraction_B fraction_C :=
by
  sorry

end NUMINAMATH_CALUDE_solve_final_grade_problem_l1208_120830


namespace NUMINAMATH_CALUDE_triangle_inequalities_l1208_120832

theorem triangle_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 + 4*a*b*c > a^3 + b^3 + c^3) ∧
  (2*a^2*b^2 + 2*b^2*c^2 + 2*c^2*a^2 > a^4 + b^4 + c^4) ∧
  (2*a*b + 2*b*c + 2*c*a > a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l1208_120832


namespace NUMINAMATH_CALUDE_rank_difference_bound_l1208_120871

variable (n : ℕ) 
variable (hn : n ≥ 2)

theorem rank_difference_bound 
  (X Y : Matrix (Fin n) (Fin n) ℂ) : 
  Matrix.rank (X * Y) - Matrix.rank (Y * X) ≤ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_rank_difference_bound_l1208_120871


namespace NUMINAMATH_CALUDE_factor_calculation_l1208_120863

theorem factor_calculation (original_number : ℝ) (final_result : ℝ) : 
  original_number = 7 →
  final_result = 69 →
  ∃ (factor : ℝ), factor * (2 * original_number + 9) = final_result ∧ factor = 3 :=
by sorry

end NUMINAMATH_CALUDE_factor_calculation_l1208_120863


namespace NUMINAMATH_CALUDE_brick_height_calculation_l1208_120859

/-- Proves that the height of each brick is 6 cm given the wall dimensions,
    brick length and width, and the number of bricks needed. -/
theorem brick_height_calculation (wall_length wall_width wall_thickness : ℝ)
                                 (brick_length brick_width : ℝ)
                                 (num_bricks : ℝ) :
  wall_length = 200 →
  wall_width = 300 →
  wall_thickness = 2 →
  brick_length = 25 →
  brick_width = 11 →
  num_bricks = 72.72727272727273 →
  (wall_length * wall_width * wall_thickness) / (brick_length * brick_width * num_bricks) = 6 :=
by sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l1208_120859


namespace NUMINAMATH_CALUDE_box_surface_area_l1208_120854

/-- Proves that a rectangular box with given edge sum and diagonal length has a specific surface area -/
theorem box_surface_area (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 168) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 1139 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l1208_120854


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1208_120827

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  -- The sum function for the first n terms
  S : ℕ → ℝ
  -- Property: The sequence of differences forms an arithmetic sequence
  difference_is_arithmetic : ∀ (k : ℕ), S (k + 1) - S k = S (k + 2) - S (k + 1)

/-- Theorem: For an arithmetic sequence with S_n = 30 and S_{2n} = 100, S_{3n} = 170 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) (n : ℕ) 
    (h1 : a.S n = 30) (h2 : a.S (2 * n) = 100) : 
    a.S (3 * n) = 170 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1208_120827


namespace NUMINAMATH_CALUDE_possible_solutions_l1208_120835

theorem possible_solutions (a b : ℝ) (h1 : a + 1 > b) (h2 : b > 2/a) (h3 : 2/a > 0) :
  (∃ a₀, a₀ = 2 ∧ a₀ + 1 > 2/a₀ ∧ 2/a₀ > 0) ∧
  (∃ b₀, b₀ = 1 ∧ (∃ a₁, a₁ + 1 > b₀ ∧ b₀ > 2/a₁ ∧ 2/a₁ > 0)) :=
sorry

end NUMINAMATH_CALUDE_possible_solutions_l1208_120835


namespace NUMINAMATH_CALUDE_hot_dog_price_l1208_120850

/-- The cost of a hamburger -/
def hamburger_cost : ℝ := sorry

/-- The cost of a hot dog -/
def hot_dog_cost : ℝ := sorry

/-- First day's purchase equation -/
axiom day1_equation : 3 * hamburger_cost + 4 * hot_dog_cost = 10

/-- Second day's purchase equation -/
axiom day2_equation : 2 * hamburger_cost + 3 * hot_dog_cost = 7

/-- Theorem stating that a hot dog costs 1 dollar -/
theorem hot_dog_price : hot_dog_cost = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dog_price_l1208_120850


namespace NUMINAMATH_CALUDE_clothing_store_problem_l1208_120844

-- Define the types of clothing
inductive ClothingType
| A
| B

-- Define the structure for clothing information
structure ClothingInfo where
  purchasePrice : ClothingType → ℕ
  sellingPrice : ClothingType → ℕ
  totalQuantity : ℕ

-- Define the problem conditions
def problemConditions (info : ClothingInfo) : Prop :=
  info.totalQuantity = 100 ∧
  2 * info.purchasePrice ClothingType.A + info.purchasePrice ClothingType.B = 260 ∧
  info.purchasePrice ClothingType.A + 3 * info.purchasePrice ClothingType.B = 380 ∧
  info.sellingPrice ClothingType.A = 120 ∧
  info.sellingPrice ClothingType.B = 150

-- Define the profit calculation function
def calculateProfit (info : ClothingInfo) (quantityA quantityB : ℕ) : ℕ :=
  (info.sellingPrice ClothingType.A - info.purchasePrice ClothingType.A) * quantityA +
  (info.sellingPrice ClothingType.B - info.purchasePrice ClothingType.B) * quantityB

-- Theorem statement
theorem clothing_store_problem (info : ClothingInfo) :
  problemConditions info →
  (info.purchasePrice ClothingType.A = 80 ∧
   info.purchasePrice ClothingType.B = 100 ∧
   calculateProfit info 50 50 = 4500 ∧
   (∀ m : ℕ, m ≤ 33 → (100 - m) ≥ 2 * m) ∧
   (∀ m : ℕ, m > 33 → (100 - m) < 2 * m) ∧
   calculateProfit info 67 33 = 4330) :=
by sorry


end NUMINAMATH_CALUDE_clothing_store_problem_l1208_120844


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l1208_120872

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 + a 13 = 10) : 
  a 3 + a 5 + a 7 + a 9 + a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l1208_120872


namespace NUMINAMATH_CALUDE_internal_tangent_length_l1208_120857

theorem internal_tangent_length (r₁ r₂ R : ℝ) (h₁ : r₁ = 19) (h₂ : r₂ = 32) (h₃ : R = 100) :
  let d := R - r₁ + R - r₂
  2 * (r₁ * r₂ / d) * Real.sqrt ((d / (r₁ + r₂))^2 - 1) = 140 :=
by sorry

end NUMINAMATH_CALUDE_internal_tangent_length_l1208_120857


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1208_120841

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.3) 
  (h2 : mindy_rate = 0.2) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1208_120841


namespace NUMINAMATH_CALUDE_billys_age_l1208_120896

theorem billys_age (billy brenda joe : ℚ) 
  (h1 : billy = 3 * brenda)
  (h2 : billy = 2 * joe)
  (h3 : billy + brenda + joe = 72) :
  billy = 432 / 11 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l1208_120896


namespace NUMINAMATH_CALUDE_least_valid_integer_l1208_120860

def is_valid (a : ℕ) : Prop :=
  a % 2 = 0 ∧ a % 3 = 1 ∧ a % 4 = 2

theorem least_valid_integer : ∃ (a : ℕ), is_valid a ∧ ∀ (b : ℕ), b < a → ¬(is_valid b) :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_least_valid_integer_l1208_120860


namespace NUMINAMATH_CALUDE_cake_cost_l1208_120875

/-- Represents the duration of vacations in days -/
def vacation_duration_1 : ℕ := 7
def vacation_duration_2 : ℕ := 4

/-- Represents the payment in CZK for each vacation period -/
def payment_1 : ℕ := 700
def payment_2 : ℕ := 340

/-- Represents the daily rate for dog walking and rabbit feeding -/
def daily_rate : ℕ := 120

theorem cake_cost (cake_price : ℕ) : 
  (cake_price + payment_1) / vacation_duration_1 = 
  (cake_price + payment_2) / vacation_duration_2 → 
  cake_price = 140 := by
  sorry

end NUMINAMATH_CALUDE_cake_cost_l1208_120875


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l1208_120839

theorem rightmost_three_digits_of_7_to_1993 : 7^1993 % 1000 = 343 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l1208_120839


namespace NUMINAMATH_CALUDE_five_in_set_A_l1208_120899

theorem five_in_set_A : 5 ∈ {x : ℕ | 1 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_five_in_set_A_l1208_120899


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1208_120868

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) := x^2 - c*x + 6

-- Define the condition for the inequality
def condition (c : ℝ) : Prop :=
  ∀ x : ℝ, f c x > 0 ↔ (x < -2 ∨ x > 3)

-- Theorem statement
theorem quadratic_coefficient : ∃ c : ℝ, condition c ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1208_120868


namespace NUMINAMATH_CALUDE_adam_wall_area_l1208_120853

/-- The total area of four rectangular walls with given dimensions -/
def totalWallArea (w1_width w1_height w2_width w2_height w3_width w3_height w4_width w4_height : ℝ) : ℝ :=
  w1_width * w1_height + w2_width * w2_height + w3_width * w3_height + w4_width * w4_height

/-- Theorem: The total area of the walls with the given dimensions is 160 square feet -/
theorem adam_wall_area :
  totalWallArea 4 8 6 8 4 8 6 8 = 160 := by
  sorry

#eval totalWallArea 4 8 6 8 4 8 6 8

end NUMINAMATH_CALUDE_adam_wall_area_l1208_120853


namespace NUMINAMATH_CALUDE_two_roots_implies_c_values_l1208_120825

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_roots_implies_c_values (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) →
  c = -2 ∨ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_values_l1208_120825


namespace NUMINAMATH_CALUDE_prob_three_even_d20_l1208_120880

/-- A fair 20-sided die -/
def D20 : Type := Fin 20

/-- The probability of rolling an even number on a D20 -/
def prob_even : ℚ := 1/2

/-- The number of dice rolled -/
def n : ℕ := 5

/-- The number of dice showing even numbers -/
def k : ℕ := 3

/-- The probability of rolling exactly k even numbers out of n rolls -/
def prob_k_even (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem prob_three_even_d20 :
  prob_k_even n k prob_even = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_d20_l1208_120880


namespace NUMINAMATH_CALUDE_inequality_proof_l1208_120878

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1208_120878


namespace NUMINAMATH_CALUDE_expand_expression_l1208_120814

theorem expand_expression (x : ℝ) : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1208_120814


namespace NUMINAMATH_CALUDE_line_equation_proof_l1208_120873

/-- Proves that the equation of a line with a slope angle of 135° and a y-intercept of -1 is y = -x - 1 -/
theorem line_equation_proof (x y : ℝ) : 
  (∃ (k b : ℝ), k = Real.tan (135 * π / 180) ∧ b = -1 ∧ y = k * x + b) ↔ y = -x - 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1208_120873


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1208_120801

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1208_120801


namespace NUMINAMATH_CALUDE_mike_car_expenses_l1208_120807

def speakers : ℚ := 118.54
def tires : ℚ := 106.33
def windowTints : ℚ := 85.27
def seatCovers : ℚ := 79.99
def maintenance : ℚ := 199.75
def steeringWheelCover : ℚ := 15.63
def airFresheners : ℚ := 6.48 * 2  -- Assuming one set of two
def carWash : ℚ := 25

def totalExpenses : ℚ := speakers + tires + windowTints + seatCovers + maintenance + steeringWheelCover + airFresheners + carWash

theorem mike_car_expenses :
  totalExpenses = 643.47 := by sorry

end NUMINAMATH_CALUDE_mike_car_expenses_l1208_120807


namespace NUMINAMATH_CALUDE_range_of_m_for_subset_l1208_120842

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m}

-- The main theorem
theorem range_of_m_for_subset (h : ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∈ B m) ∧ 
  ¬(∀ x : ℝ, x ∈ B m → x ∈ A)) : 
  {m : ℝ | A ⊆ B m} = {m : ℝ | m > 3} := by
  sorry


end NUMINAMATH_CALUDE_range_of_m_for_subset_l1208_120842


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l1208_120879

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
    (h_sum : 3 * x₁ + 2 * x₂ + x₃ = 30) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 125 := by
  sorry

theorem min_sum_squares_attained (ε : ℝ) (h_pos : ε > 0) : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
    3 * x₁ + 2 * x₂ + x₃ = 30 ∧ 
    x₁^2 + x₂^2 + x₃^2 < 125 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l1208_120879


namespace NUMINAMATH_CALUDE_second_vessel_ratio_l1208_120845

/-- Represents the ratio of milk to water in a mixture -/
structure MilkWaterRatio where
  milk : ℚ
  water : ℚ

/-- The mixture in a vessel -/
structure Mixture where
  volume : ℚ
  ratio : MilkWaterRatio

theorem second_vessel_ratio 
  (v1 v2 : Mixture) 
  (h1 : v1.volume = v2.volume) 
  (h2 : v1.ratio = MilkWaterRatio.mk 4 2) 
  (h3 : let combined_ratio := MilkWaterRatio.mk 
          (v1.ratio.milk * v1.volume + v2.ratio.milk * v2.volume) 
          (v1.ratio.water * v1.volume + v2.ratio.water * v2.volume)
        combined_ratio = MilkWaterRatio.mk 3 1) :
  v2.ratio = MilkWaterRatio.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_second_vessel_ratio_l1208_120845


namespace NUMINAMATH_CALUDE_fan_sales_analysis_fan_sales_analysis_application_l1208_120806

/-- Represents the sales data for a week -/
structure WeeklySales where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the fan models and their prices -/
structure FanModels where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceA : ℕ
  sellingPriceB : ℕ

/-- Represents the purchase constraints -/
structure PurchaseConstraints where
  totalUnits : ℕ
  maxBudget : ℕ

/-- Main theorem encompassing all parts of the problem -/
theorem fan_sales_analysis 
  (week1 : WeeklySales)
  (week2 : WeeklySales)
  (models : FanModels)
  (constraints : PurchaseConstraints) :
  (models.sellingPriceA = 200 ∧ models.sellingPriceB = 150) ∧
  (∀ m : ℕ, m ≤ 37 → m * models.purchasePriceA + (constraints.totalUnits - m) * models.purchasePriceB ≤ constraints.maxBudget) ∧
  (∃ m : ℕ, m ≤ 37 ∧ m * (models.sellingPriceA - models.purchasePriceA) + 
    (constraints.totalUnits - m) * (models.sellingPriceB - models.purchasePriceB) > 2850) :=
by
  sorry

/-- Given data for the problem -/
def problem_data : WeeklySales × WeeklySales × FanModels × PurchaseConstraints :=
  ({ modelA := 4, modelB := 3, revenue := 1250 },
   { modelA := 5, modelB := 5, revenue := 1750 },
   { purchasePriceA := 140, purchasePriceB := 100, sellingPriceA := 0, sellingPriceB := 0 },
   { totalUnits := 50, maxBudget := 6500 })

/-- Application of the main theorem to the given data -/
theorem fan_sales_analysis_application :
  let (week1, week2, models, constraints) := problem_data
  (models.sellingPriceA = 200 ∧ models.sellingPriceB = 150) ∧
  (∀ m : ℕ, m ≤ 37 → m * models.purchasePriceA + (constraints.totalUnits - m) * models.purchasePriceB ≤ constraints.maxBudget) ∧
  (∃ m : ℕ, m ≤ 37 ∧ m * (models.sellingPriceA - models.purchasePriceA) + 
    (constraints.totalUnits - m) * (models.sellingPriceB - models.purchasePriceB) > 2850) :=
by
  sorry

end NUMINAMATH_CALUDE_fan_sales_analysis_fan_sales_analysis_application_l1208_120806


namespace NUMINAMATH_CALUDE_max_value_z_minus_2i_l1208_120815

theorem max_value_z_minus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max : ℝ), max = 3 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w - 2*I) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_z_minus_2i_l1208_120815


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1208_120805

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| < 2
def q (x : ℝ) : Prop := x^2 < 2 - x

-- Define the relationship between ¬p and ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∃ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  (∃ x : ℝ, ¬(q x) ∧ p x) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1208_120805


namespace NUMINAMATH_CALUDE_open_box_volume_l1208_120809

/-- Calculate the volume of an open box created from a rectangular sheet --/
theorem open_box_volume (sheet_length sheet_width cut_side : ℝ) :
  sheet_length = 48 ∧ 
  sheet_width = 36 ∧ 
  cut_side = 8 →
  (sheet_length - 2 * cut_side) * (sheet_width - 2 * cut_side) * cut_side = 5120 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_l1208_120809


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1208_120828

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  (∃ (m : ℝ), ∀ (a b : ℝ), a + b = 5 →
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 = m) ∧
  (∀ (m : ℝ), (∀ (a b : ℝ), a + b = 5 →
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 = m) →
  m = 441/2) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1208_120828


namespace NUMINAMATH_CALUDE_square_of_geometric_is_geometric_l1208_120811

-- Define a geometric sequence
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Statement to prove
theorem square_of_geometric_is_geometric (a : ℕ → ℝ) (h : IsGeometric a) :
  IsGeometric (fun n ↦ (a n)^2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_geometric_is_geometric_l1208_120811


namespace NUMINAMATH_CALUDE_matrix_operation_proof_l1208_120876

theorem matrix_operation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 8; -3, 0]
  2 • A + B = !![3, 14; -5, 8] := by
  sorry

end NUMINAMATH_CALUDE_matrix_operation_proof_l1208_120876


namespace NUMINAMATH_CALUDE_system_solution_l1208_120847

theorem system_solution (a b : ℝ) : 
  (2 * a * 1 + b * 1 = 3) → 
  (a * 1 - b * 1 = 1) → 
  a + 2 * b = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1208_120847


namespace NUMINAMATH_CALUDE_regular_soda_bottles_l1208_120840

theorem regular_soda_bottles (diet_soda : ℕ) (difference : ℕ) : 
  diet_soda = 19 → difference = 41 → diet_soda + difference = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_soda_bottles_l1208_120840


namespace NUMINAMATH_CALUDE_franks_sunday_bags_l1208_120848

/-- Given that Frank filled 5 bags on Saturday, each bag contains 5 cans,
    and Frank collected a total of 40 cans over the weekend,
    prove that Frank filled 3 bags on Sunday. -/
theorem franks_sunday_bags (saturday_bags : ℕ) (cans_per_bag : ℕ) (total_cans : ℕ)
    (h1 : saturday_bags = 5)
    (h2 : cans_per_bag = 5)
    (h3 : total_cans = 40) :
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_franks_sunday_bags_l1208_120848


namespace NUMINAMATH_CALUDE_mary_total_spending_l1208_120846

/-- The total amount Mary spent on clothing, given the costs of a shirt and a jacket. -/
def total_spent (shirt_cost jacket_cost : ℚ) : ℚ :=
  shirt_cost + jacket_cost

/-- Theorem stating that Mary's total spending is $25.31 -/
theorem mary_total_spending :
  total_spent 13.04 12.27 = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_mary_total_spending_l1208_120846


namespace NUMINAMATH_CALUDE_sally_grew_six_carrots_l1208_120855

/-- The number of carrots Fred grew -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots Sally grew -/
def sally_carrots : ℕ := total_carrots - fred_carrots

theorem sally_grew_six_carrots : sally_carrots = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_six_carrots_l1208_120855


namespace NUMINAMATH_CALUDE_distribute_two_four_x_minus_one_l1208_120890

theorem distribute_two_four_x_minus_one (x : ℝ) : 2 * (4 * x - 1) = 8 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_two_four_x_minus_one_l1208_120890


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l1208_120808

/-- The equation of a line passing through a given point with a given angle -/
theorem line_equation_through_point_with_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) :
  x₀ = Real.sqrt 3 →
  y₀ = -2 * Real.sqrt 3 →
  θ = 135 * π / 180 →
  ∃ (A B C : ℝ), A * x₀ + B * y₀ + C = 0 ∧
                 A * x + B * y + C = 0 ∧
                 A = 1 ∧ B = 1 ∧ C = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l1208_120808


namespace NUMINAMATH_CALUDE_sample_size_comparison_l1208_120836

theorem sample_size_comparison (n m : ℕ+) (x_bar y_bar z a : ℝ) :
  x_bar ≠ y_bar →
  0 < a →
  a < 1/2 →
  z = a * x_bar + (1 - a) * y_bar →
  n > m :=
sorry

end NUMINAMATH_CALUDE_sample_size_comparison_l1208_120836


namespace NUMINAMATH_CALUDE_additional_cakes_is_21_l1208_120816

/-- Represents the number of cakes baked in a week -/
structure CakeQuantities where
  cheesecakes : ℕ
  muffins : ℕ
  redVelvet : ℕ
  chocolateMoist : ℕ
  fruitcakes : ℕ
  carrotCakes : ℕ

/-- Carter's usual cake quantities -/
def usualQuantities : CakeQuantities := {
  cheesecakes := 6,
  muffins := 5,
  redVelvet := 8,
  chocolateMoist := 0,
  fruitcakes := 0,
  carrotCakes := 0
}

/-- Calculate the new quantities based on the given rates -/
def newQuantities (usual : CakeQuantities) : CakeQuantities := {
  cheesecakes := (usual.cheesecakes * 3 + 1) / 2,
  muffins := (usual.muffins * 6 + 2) / 5,
  redVelvet := (usual.redVelvet * 9 + 2) / 5,
  chocolateMoist := ((usual.redVelvet * 9 + 2) / 5) / 2,
  fruitcakes := (((usual.muffins * 6 + 2) / 5) * 2) / 3,
  carrotCakes := 0
}

/-- Calculate the total additional cakes -/
def additionalCakes (usual new : CakeQuantities) : ℕ :=
  (new.cheesecakes - usual.cheesecakes) +
  (new.muffins - usual.muffins) +
  (new.redVelvet - usual.redVelvet) +
  (new.chocolateMoist - usual.chocolateMoist) +
  (new.fruitcakes - usual.fruitcakes) +
  (new.carrotCakes - usual.carrotCakes)

theorem additional_cakes_is_21 :
  additionalCakes usualQuantities (newQuantities usualQuantities) = 21 := by
  sorry

end NUMINAMATH_CALUDE_additional_cakes_is_21_l1208_120816
