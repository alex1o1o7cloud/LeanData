import Mathlib

namespace NUMINAMATH_CALUDE_inequality_addition_l3649_364918

theorem inequality_addition {a b c d : ℝ} (hab : a > b) (hcd : c > d) (hc : c ≠ 0) (hd : d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l3649_364918


namespace NUMINAMATH_CALUDE_james_living_room_cost_l3649_364966

def couch_price : ℝ := 2500
def sectional_price : ℝ := 3500
def entertainment_center_price : ℝ := 1500
def rug_price : ℝ := 800
def coffee_table_price : ℝ := 700
def accessories_price : ℝ := 500

def couch_discount : ℝ := 0.10
def sectional_discount : ℝ := 0.10
def entertainment_center_discount : ℝ := 0.05
def rug_discount : ℝ := 0.05
def coffee_table_discount : ℝ := 0.12
def accessories_discount : ℝ := 0.15

def sales_tax_rate : ℝ := 0.0825
def service_fee : ℝ := 250

def total_cost : ℝ := 9587.65

theorem james_living_room_cost : 
  (couch_price * (1 - couch_discount) + 
   sectional_price * (1 - sectional_discount) + 
   entertainment_center_price * (1 - entertainment_center_discount) + 
   rug_price * (1 - rug_discount) + 
   coffee_table_price * (1 - coffee_table_discount) + 
   accessories_price * (1 - accessories_discount)) * 
  (1 + sales_tax_rate) + service_fee = total_cost := by
  sorry

end NUMINAMATH_CALUDE_james_living_room_cost_l3649_364966


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3649_364982

theorem roots_of_polynomial (x : ℝ) : 
  x^2 * (x - 5)^2 * (x + 3) = 0 ↔ x = 0 ∨ x = 5 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3649_364982


namespace NUMINAMATH_CALUDE_intersection_range_l3649_364905

-- Define the line equation
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * x - 1

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the intersection condition
def always_intersects (k : ℝ) : Prop :=
  ∀ x y : ℝ, hyperbola_equation x y → ∃ x', y = line_equation k x'

-- State the theorem
theorem intersection_range :
  ∀ k : ℝ, always_intersects k ↔ (k = 1 ∨ k = -1 ∨ (-Real.sqrt 5 / 2 ≤ k ∧ k ≤ Real.sqrt 5 / 2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l3649_364905


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l3649_364998

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem intersection_implies_a_equals_one :
  ∀ a : ℝ, (A ∩ B a = {3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l3649_364998


namespace NUMINAMATH_CALUDE_slope_angle_of_y_equals_1_l3649_364924

-- Define a line parallel to the x-axis
def parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y

-- Define the slope angle of a line
def slope_angle (f : ℝ → ℝ) : ℝ := sorry

-- Theorem: The slope angle of the line y = 1 is 0
theorem slope_angle_of_y_equals_1 :
  let f : ℝ → ℝ := λ x => 1
  parallel_to_x_axis f ∧ slope_angle f = 0 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_y_equals_1_l3649_364924


namespace NUMINAMATH_CALUDE_fraction_product_equality_l3649_364972

theorem fraction_product_equality : (1 / 3 : ℚ)^4 * (1 / 8 : ℚ) = 1 / 648 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l3649_364972


namespace NUMINAMATH_CALUDE_equal_projections_imply_equal_areas_l3649_364903

/-- Represents a parabola -/
structure Parabola where
  -- Add necessary fields to define a parabola

/-- Represents a chord of a parabola -/
structure Chord (p : Parabola) where
  -- Add necessary fields to define a chord

/-- Represents the projection of a chord on the directrix -/
def projection (p : Parabola) (c : Chord p) : ℝ :=
  sorry

/-- Represents the area of the segment cut off by a chord -/
def segmentArea (p : Parabola) (c : Chord p) : ℝ :=
  sorry

/-- Theorem: If two chords of a parabola have equal projections on the directrix,
    then the areas of the segments they cut off are equal -/
theorem equal_projections_imply_equal_areas (p : Parabola) (c1 c2 : Chord p) :
  projection p c1 = projection p c2 → segmentArea p c1 = segmentArea p c2 :=
by sorry

end NUMINAMATH_CALUDE_equal_projections_imply_equal_areas_l3649_364903


namespace NUMINAMATH_CALUDE_parking_tickets_l3649_364970

theorem parking_tickets (total : ℕ) (alan : ℕ) (marcy : ℕ) 
  (h1 : total = 150)
  (h2 : alan = 26)
  (h3 : marcy = 5 * alan)
  (h4 : total = alan + marcy) :
  total - marcy = 104 := by
  sorry

end NUMINAMATH_CALUDE_parking_tickets_l3649_364970


namespace NUMINAMATH_CALUDE_partner_numbers_problem_l3649_364938

/-- Definition of "partner numbers" -/
def partner_numbers (m n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    m = 100 * a + 10 * b + c ∧
    n = 100 * d + 10 * e + f ∧
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    1 ≤ e ∧ e ≤ 9 ∧
    1 ≤ f ∧ f ≤ 9 ∧
    ∃ (k : ℤ), k * (b - c) = a + 4 * d + 4 * e + 4 * f

theorem partner_numbers_problem (x y z : ℕ) 
  (hx : x ≤ 3) 
  (hy : 0 < y ∧ y ≤ 4) 
  (hz : 3 < z ∧ z ≤ 9) 
  (h_partner : partner_numbers (467 + 110 * x) (200 * y + z + 37))
  (h_sum : (2 * y + z + 1) % 12 = 0) :
  467 + 110 * x = 467 ∨ 467 + 110 * x = 687 := by
  sorry

end NUMINAMATH_CALUDE_partner_numbers_problem_l3649_364938


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l3649_364992

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 138) (h2 : b = 85) (h3 : c = 130) (h4 : d = 120) (h5 : e = 95) :
  720 - (a + b + c + d + e) = 152 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l3649_364992


namespace NUMINAMATH_CALUDE_three_digit_squares_divisible_by_12_l3649_364978

theorem three_digit_squares_divisible_by_12 :
  (∃! (l : List Nat), l = (List.range 22).filter (fun n => 
    10 ≤ n ∧ n ≤ 31 ∧ (n^2 % 12 = 0)) ∧ l.length = 4) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_squares_divisible_by_12_l3649_364978


namespace NUMINAMATH_CALUDE_number_operations_l3649_364949

theorem number_operations (x : ℝ) : (3 * ((x - 50) / 4) + 28 = 73) ↔ (x = 110) := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l3649_364949


namespace NUMINAMATH_CALUDE_fewer_puzzles_than_kits_difference_is_nine_l3649_364959

/-- The Smart Mart sells educational toys -/
structure SmartMart where
  science_kits : ℕ
  puzzles : ℕ

/-- The number of science kits sold is 45 -/
def science_kits_sold : ℕ := 45

/-- The number of puzzles sold is 36 -/
def puzzles_sold : ℕ := 36

/-- The Smart Mart sold fewer puzzles than science kits -/
theorem fewer_puzzles_than_kits (sm : SmartMart) :
  sm.puzzles < sm.science_kits :=
sorry

/-- The difference between science kits and puzzles sold is 9 -/
theorem difference_is_nine (sm : SmartMart) 
  (h1 : sm.science_kits = science_kits_sold) 
  (h2 : sm.puzzles = puzzles_sold) : 
  sm.science_kits - sm.puzzles = 9 :=
sorry

end NUMINAMATH_CALUDE_fewer_puzzles_than_kits_difference_is_nine_l3649_364959


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3649_364933

-- Define the sets A and B
def A : Set ℝ := {x | |x + 3| - |x - 3| > 3}
def B : Set ℝ := {x | ∃ t > 0, x = (t^2 - 4*t + 1) / t}

-- State the theorem
theorem intersection_complement_theorem : B ∩ (Set.univ \ A) = Set.Icc (-2) (3/2) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3649_364933


namespace NUMINAMATH_CALUDE_regression_line_intercept_l3649_364907

/-- Given a regression line with slope 1.23 passing through (4, 5), prove its y-intercept is 0.08 -/
theorem regression_line_intercept (slope : ℝ) (x₀ y₀ : ℝ) (h1 : slope = 1.23) (h2 : x₀ = 4) (h3 : y₀ = 5) :
  y₀ = slope * x₀ + 0.08 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l3649_364907


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3649_364948

/-- Given a geometric sequence {a_n} with sum of first n terms S_n = 3^n + t,
    prove that t + a_3 = 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 3^n + t) →
  (∀ n, a (n+1) = S (n+1) - S n) →
  (a 1 * a 3 = (a 2)^2) →
  t + a 3 = 17 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3649_364948


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3649_364944

theorem quadratic_root_implies_k (k : ℝ) : 
  (2 * (5 : ℝ)^2 + 3 * 5 - k = 0) → k = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3649_364944


namespace NUMINAMATH_CALUDE_journey_speed_proof_l3649_364976

/-- Proves that given a journey of 108 miles completed in 90 minutes, 
    where the average speed for the first 30 minutes was 65 mph and 
    for the second 30 minutes was 70 mph, the average speed for the 
    last 30 minutes was 81 mph. -/
theorem journey_speed_proof 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_first_segment : ℝ) 
  (speed_second_segment : ℝ) 
  (h1 : total_distance = 108) 
  (h2 : total_time = 90 / 60) 
  (h3 : speed_first_segment = 65) 
  (h4 : speed_second_segment = 70) : 
  ∃ (speed_last_segment : ℝ), 
    speed_last_segment = 81 ∧ 
    (speed_first_segment + speed_second_segment + speed_last_segment) / 3 = 
      total_distance / total_time := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l3649_364976


namespace NUMINAMATH_CALUDE_fraction_puzzle_solvable_l3649_364996

def is_valid_fraction (a b : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ a ≠ b

def are_distinct (a b c d e f g h i : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem fraction_puzzle_solvable : 
  ∃ (a b c d e f g h i : ℕ),
    is_valid_fraction a b ∧ 
    is_valid_fraction c d ∧ 
    is_valid_fraction e f ∧ 
    is_valid_fraction g h ∧
    are_distinct a b c d e f g h i ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f + (g : ℚ) / h = i := by
  sorry

end NUMINAMATH_CALUDE_fraction_puzzle_solvable_l3649_364996


namespace NUMINAMATH_CALUDE_jake_bitcoin_proportion_l3649_364921

/-- The proportion of bitcoins Jake gave to his brother -/
def proportion_to_brother : ℚ := 1/2

/-- Jake's initial fortune in bitcoins -/
def initial_fortune : ℕ := 80

/-- First donation amount in bitcoins -/
def first_donation : ℕ := 20

/-- Second donation amount in bitcoins -/
def second_donation : ℕ := 10

/-- Jake's final amount of bitcoins -/
def final_amount : ℕ := 80

theorem jake_bitcoin_proportion :
  let remaining_after_first_donation := initial_fortune - first_donation
  let remaining_after_giving_to_brother := remaining_after_first_donation * (1 - proportion_to_brother)
  let amount_after_tripling := remaining_after_giving_to_brother * 3
  amount_after_tripling - second_donation = final_amount :=
by sorry

end NUMINAMATH_CALUDE_jake_bitcoin_proportion_l3649_364921


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l3649_364913

theorem smallest_perfect_cube_divisor
  (p q r : ℕ)
  (hp : Prime p)
  (hq : Prime q)
  (hr : Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)
  (h1_not_prime : ¬ Prime 1)
  (n : ℕ)
  (hn : n = p * q^3 * r^6) :
  ∃ (m : ℕ), m^3 = p^3 * q^3 * r^6 ∧
    ∀ (k : ℕ), (k^3 ≥ n) → (k^3 ≥ m^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l3649_364913


namespace NUMINAMATH_CALUDE_banana_permutations_l3649_364989

-- Define the word and its properties
def word : String := "BANANA"
def word_length : Nat := 6
def b_count : Nat := 1
def a_count : Nat := 3
def n_count : Nat := 2

-- Theorem statement
theorem banana_permutations :
  (Nat.factorial word_length) / 
  (Nat.factorial b_count * Nat.factorial a_count * Nat.factorial n_count) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l3649_364989


namespace NUMINAMATH_CALUDE_expression_value_l3649_364995

theorem expression_value : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3649_364995


namespace NUMINAMATH_CALUDE_remainder_prime_divisible_by_210_l3649_364973

theorem remainder_prime_divisible_by_210 (p r : ℕ) : 
  Prime p → 
  r = p % 210 → 
  0 < r → 
  r < 210 → 
  ¬ Prime r → 
  (∃ (a b : ℕ), r = a^2 + b^2) → 
  r = 169 := by sorry

end NUMINAMATH_CALUDE_remainder_prime_divisible_by_210_l3649_364973


namespace NUMINAMATH_CALUDE_invisible_dots_count_l3649_364945

/-- The sum of numbers on a single six-sided die -/
def die_sum : Nat := 21

/-- The total number of dots on four dice -/
def total_dots : Nat := 4 * die_sum

/-- The sum of visible numbers on the dice -/
def visible_sum : Nat := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6

/-- The number of dots not visible on the dice -/
def invisible_dots : Nat := total_dots - visible_sum

theorem invisible_dots_count : invisible_dots = 54 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l3649_364945


namespace NUMINAMATH_CALUDE_building_occupancy_ratio_l3649_364939

/-- Calculates the occupancy ratio of a building given the number of units,
    monthly rent per unit, and total annual rent received. -/
theorem building_occupancy_ratio
  (num_units : ℕ)
  (monthly_rent : ℝ)
  (annual_rent_received : ℝ)
  (h1 : num_units = 100)
  (h2 : monthly_rent = 400)
  (h3 : annual_rent_received = 360000) :
  annual_rent_received / (num_units * monthly_rent * 12) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_building_occupancy_ratio_l3649_364939


namespace NUMINAMATH_CALUDE_similar_triangles_leg_ratio_l3649_364925

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs x and 7,
    prove that x = 84/9 -/
theorem similar_triangles_leg_ratio (x : ℝ) : 
  (12 : ℝ) / x = 9 / 7 → x = 84 / 9 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_ratio_l3649_364925


namespace NUMINAMATH_CALUDE_square_area_proof_l3649_364988

theorem square_area_proof (side_length : ℝ) (h1 : side_length > 0) : 
  (3 * 4 * side_length - (2 * side_length + 2 * (3 * side_length)) = 28) → 
  side_length^2 = 49 := by
  sorry

#check square_area_proof

end NUMINAMATH_CALUDE_square_area_proof_l3649_364988


namespace NUMINAMATH_CALUDE_max_profit_l3649_364977

/-- Annual sales revenue function -/
noncomputable def Q (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then -x^2 + 1040*x + 1200
  else if x > 30 then 998*x - 2048/(x-2) + 1800
  else 0

/-- Annual total profit function (in million yuan) -/
noncomputable def W (x : ℝ) : ℝ :=
  (Q x - (1000*x + 600)) / 1000

/-- The maximum profit is 1068 million yuan -/
theorem max_profit :
  ∃ x : ℝ, x > 0 ∧ W x = 1068 ∧ ∀ y : ℝ, y > 0 → W y ≤ W x :=
sorry

end NUMINAMATH_CALUDE_max_profit_l3649_364977


namespace NUMINAMATH_CALUDE_inequality_proof_l3649_364947

theorem inequality_proof (A B C a b c r : ℝ) 
  (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  (A + a + B + b) / (A + a + B + b + c + r) + 
  (B + b + C + c) / (B + b + C + c + a + r) > 
  (C + c + A + a) / (C + c + A + a + b + r) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3649_364947


namespace NUMINAMATH_CALUDE_rebecca_earring_ratio_l3649_364967

/-- Proves the ratio of gemstones to buttons for Rebecca's earrings --/
theorem rebecca_earring_ratio 
  (magnets_per_earring : ℕ)
  (buttons_to_magnets_ratio : ℚ)
  (sets_of_earrings : ℕ)
  (total_gemstones : ℕ)
  (h1 : magnets_per_earring = 2)
  (h2 : buttons_to_magnets_ratio = 1/2)
  (h3 : sets_of_earrings = 4)
  (h4 : total_gemstones = 24) :
  (total_gemstones : ℚ) / ((sets_of_earrings * 2 * magnets_per_earring * buttons_to_magnets_ratio) : ℚ) = 3 := by
  sorry

#check rebecca_earring_ratio

end NUMINAMATH_CALUDE_rebecca_earring_ratio_l3649_364967


namespace NUMINAMATH_CALUDE_rhombus_area_l3649_364950

/-- A rhombus with side length √113 and diagonal difference 8 has area 194. -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 →
  diag_diff = 8 →
  area = (Real.sqrt 210)^2 - 4^2 →
  area = 194 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l3649_364950


namespace NUMINAMATH_CALUDE_log_equation_solution_l3649_364900

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 - 3 * (Real.log x / Real.log 2) = 6 →
  x = (1 : ℝ) / 2^(9/4) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3649_364900


namespace NUMINAMATH_CALUDE_square_perimeter_l3649_364920

/-- Given a square with area 625 cm², prove its perimeter is 100 cm -/
theorem square_perimeter (s : ℝ) (h_area : s^2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3649_364920


namespace NUMINAMATH_CALUDE_fraction_value_l3649_364954

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 10) :
  (x + y) / (x - y) = -Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3649_364954


namespace NUMINAMATH_CALUDE_right_triangle_with_angle_ratio_l3649_364934

theorem right_triangle_with_angle_ratio (a b c : ℝ) (h_right : a + b + c = 180) 
  (h_largest : c = 90) (h_ratio : a / b = 3 / 2) : 
  c = 90 ∧ a = 54 ∧ b = 36 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_angle_ratio_l3649_364934


namespace NUMINAMATH_CALUDE_sin_function_properties_l3649_364983

noncomputable def f (x φ A : ℝ) : ℝ := Real.sin (2 * x + φ) + A

theorem sin_function_properties (φ A : ℝ) :
  -- Amplitude is A
  (∃ (x : ℝ), f x φ A - A = 1) ∧
  (∀ (x : ℝ), f x φ A - A ≤ 1) ∧
  -- Period is π
  (∀ (x : ℝ), f (x + π) φ A = f x φ A) ∧
  -- Initial phase is φ
  (∀ (x : ℝ), f x φ A = Real.sin (2 * x + φ) + A) ∧
  -- Maximum value occurs when x = π/4 + kπ, k ∈ ℤ
  (∀ (x : ℝ), f x φ A = A + 1 ↔ ∃ (k : ℤ), x = π/4 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_sin_function_properties_l3649_364983


namespace NUMINAMATH_CALUDE_gwen_games_remaining_l3649_364932

/-- The number of games remaining after giving some away -/
def remaining_games (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Given 98 initial games and 7 games given away, 91 games remain -/
theorem gwen_games_remaining :
  remaining_games 98 7 = 91 := by
  sorry

end NUMINAMATH_CALUDE_gwen_games_remaining_l3649_364932


namespace NUMINAMATH_CALUDE_num_unique_labelings_eq_30_l3649_364915

/-- A cube is a three-dimensional object with 6 faces. -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- A labeling of a cube is valid if it uses the numbers 1 to 6 exactly once each. -/
def is_valid_labeling (c : Cube) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 6 → n + 1 ∈ Finset.image c.faces Finset.univ) ∧
  (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → c.faces f₁ ≠ c.faces f₂)

/-- Two labelings are equivalent up to rotation if they can be transformed into each other by rotating the cube. -/
def equivalent_up_to_rotation (c₁ c₂ : Cube) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 6)), ∀ (f : Fin 6), c₁.faces f = c₂.faces (perm f)

/-- The number of unique labelings of a cube up to rotation -/
def num_unique_labelings : ℕ := sorry

theorem num_unique_labelings_eq_30 : num_unique_labelings = 30 := by
  sorry

end NUMINAMATH_CALUDE_num_unique_labelings_eq_30_l3649_364915


namespace NUMINAMATH_CALUDE_intersection_with_complement_is_empty_l3649_364965

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3}
def B : Set Nat := {1, 3, 4}

theorem intersection_with_complement_is_empty :
  A ∩ (U \ B) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_is_empty_l3649_364965


namespace NUMINAMATH_CALUDE_y_value_theorem_l3649_364914

theorem y_value_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_theorem_l3649_364914


namespace NUMINAMATH_CALUDE_cube_gt_iff_gt_l3649_364923

theorem cube_gt_iff_gt (a b : ℝ) : a^3 > b^3 ↔ a > b := by sorry

end NUMINAMATH_CALUDE_cube_gt_iff_gt_l3649_364923


namespace NUMINAMATH_CALUDE_game_ends_in_49_rounds_l3649_364984

/-- Represents a player in the token game -/
inductive Player : Type
  | A | B | C | D

/-- The state of the game at any point -/
structure GameState :=
  (tokens : Player → Nat)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := fun p => match p with
    | Player.A => 16
    | Player.B => 15
    | Player.C => 14
    | Player.D => 13 }

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (state : GameState) (count : Nat := 0) : Nat :=
  sorry

theorem game_ends_in_49_rounds :
  countRounds initialState = 49 :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_49_rounds_l3649_364984


namespace NUMINAMATH_CALUDE_no_solution_cubic_equation_l3649_364946

theorem no_solution_cubic_equation (p : ℕ) (hp : Nat.Prime p) :
  ¬∃ (x y : ℤ), (x^3 + y^3 = 2001 * ↑p ∨ x^3 - y^3 = 2001 * ↑p) :=
sorry

end NUMINAMATH_CALUDE_no_solution_cubic_equation_l3649_364946


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3649_364968

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p ∧
  p = 307 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3649_364968


namespace NUMINAMATH_CALUDE_min_sum_of_system_l3649_364994

theorem min_sum_of_system (x y z : ℝ) 
  (eq1 : x + 3*y + 6*z = 1)
  (eq2 : x*y + 2*x*z + 6*y*z = -8)
  (eq3 : x*y*z = 2) :
  ∀ (a b c : ℝ), (a + 3*b + 6*c = 1 ∧ a*b + 2*a*c + 6*b*c = -8 ∧ a*b*c = 2) → 
  x + y + z ≤ a + b + c ∧ x + y + z = -8/3 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_system_l3649_364994


namespace NUMINAMATH_CALUDE_square_area_with_two_side_expressions_l3649_364963

theorem square_area_with_two_side_expressions (x : ℝ) :
  (5 * x + 10 = 35 - 2 * x) →
  ((5 * x + 10) ^ 2 : ℝ) = 38025 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_two_side_expressions_l3649_364963


namespace NUMINAMATH_CALUDE_score_theorem_l3649_364929

/-- Represents the bags from which balls are drawn -/
inductive Bag
| A
| B

/-- Represents the color of the balls -/
inductive Color
| Black
| White
| Red

/-- Represents the score obtained from drawing a ball -/
def score (bag : Bag) (color : Color) : ℕ :=
  match bag, color with
  | Bag.A, Color.Black => 2
  | Bag.B, Color.Black => 1
  | _, _ => 0

/-- The probability of drawing a black ball from bag B -/
def probBlackB : ℝ := 0.8

/-- The probability of getting a total score of 1 -/
def probScoreOne : ℝ := 0.24

/-- The expected value of the total score -/
def expectedScore : ℝ := 1.94

/-- Theorem stating the expected value of the total score and comparing probabilities -/
theorem score_theorem :
  ∃ (probBlackA : ℝ),
    0 ≤ probBlackA ∧ probBlackA ≤ 1 ∧
    (let pA := probBlackA * (1 - probBlackB) + (1 - probBlackA) * probBlackB
     let pB := probBlackB * probBlackB
     pB > pA) ∧
    expectedScore = 1.94 := by
  sorry

end NUMINAMATH_CALUDE_score_theorem_l3649_364929


namespace NUMINAMATH_CALUDE_parallel_lines_parallelograms_l3649_364962

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of parallelograms formed by intersecting parallel lines -/
def parallelograms_count (set1 : ℕ) (set2 : ℕ) : ℕ :=
  choose_two set1 * choose_two set2

theorem parallel_lines_parallelograms :
  parallelograms_count 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_parallelograms_l3649_364962


namespace NUMINAMATH_CALUDE_approx_root_e_2019_l3649_364936

/-- Approximation of the 2019th root of e using tangent line method -/
theorem approx_root_e_2019 (e : ℝ) (h : e = Real.exp 1) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |e^(1/2019) - (1 + 1/2019)| < ε :=
sorry

end NUMINAMATH_CALUDE_approx_root_e_2019_l3649_364936


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l3649_364919

theorem largest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 35 = 0 → n ≤ 9985 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l3649_364919


namespace NUMINAMATH_CALUDE_water_tower_shortage_l3649_364928

theorem water_tower_shortage : 
  let tower_capacity : ℝ := 2700
  let first_neighborhood : ℝ := 300
  let second_neighborhood : ℝ := 2 * first_neighborhood
  let third_neighborhood : ℝ := second_neighborhood + 100
  let fourth_neighborhood : ℝ := 3 * first_neighborhood
  let fifth_neighborhood : ℝ := third_neighborhood / 2
  let leakage_loss : ℝ := 50
  let first_increased : ℝ := first_neighborhood * 1.1
  let third_increased : ℝ := third_neighborhood * 1.1
  let second_decreased : ℝ := second_neighborhood * 0.95
  let fifth_decreased : ℝ := fifth_neighborhood * 0.95
  let total_consumption : ℝ := first_increased + second_decreased + third_increased + fourth_neighborhood + fifth_decreased + leakage_loss
  total_consumption - tower_capacity = 252.5 :=
by sorry

end NUMINAMATH_CALUDE_water_tower_shortage_l3649_364928


namespace NUMINAMATH_CALUDE_grocery_store_soda_l3649_364902

theorem grocery_store_soda (regular_soda : ℕ) (apples : ℕ) (total_bottles : ℕ) 
  (h1 : regular_soda = 72)
  (h2 : apples = 78)
  (h3 : total_bottles = apples + 26) :
  total_bottles - regular_soda = 32 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_l3649_364902


namespace NUMINAMATH_CALUDE_sum_radii_greater_incircle_radius_l3649_364935

-- Define the triangle and circles
variable (A B C : EuclideanPlane) (S S₁ S₂ : Circle EuclideanPlane)

-- Define the radii
variable (r r₁ r₂ : ℝ)

-- Assumptions
variable (h_triangle : Triangle A B C)
variable (h_incircle : S.IsIncircle h_triangle)
variable (h_S₁_tangent : S₁.IsTangentTo (SegmentND A B) ∧ S₁.IsTangentTo (SegmentND A C))
variable (h_S₂_tangent : S₂.IsTangentTo (SegmentND A B) ∧ S₂.IsTangentTo (SegmentND B C))
variable (h_S₁S₂_tangent : S₁.IsExternallyTangentTo S₂)
variable (h_r : S.radius = r)
variable (h_r₁ : S₁.radius = r₁)
variable (h_r₂ : S₂.radius = r₂)

-- Theorem statement
theorem sum_radii_greater_incircle_radius : r₁ + r₂ > r := by
  sorry

end NUMINAMATH_CALUDE_sum_radii_greater_incircle_radius_l3649_364935


namespace NUMINAMATH_CALUDE_braiding_time_for_dance_team_l3649_364926

/-- Calculates the time in minutes to braid dancers' hair -/
def braidingTime (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℕ :=
  let total_braids := num_dancers * braids_per_dancer
  let total_seconds := total_braids * seconds_per_braid
  total_seconds / 60

theorem braiding_time_for_dance_team :
  braidingTime 8 5 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_braiding_time_for_dance_team_l3649_364926


namespace NUMINAMATH_CALUDE_unique_number_property_l3649_364980

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3649_364980


namespace NUMINAMATH_CALUDE_triangle_angle_and_side_length_l3649_364931

theorem triangle_angle_and_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (m n : ℝ × ℝ) :
  A > 0 ∧ A < π ∧
  B > 0 ∧ B < π ∧
  C > 0 ∧ C < π ∧
  A + B + C = π ∧
  m = (Real.sqrt 3, Real.cos A + 1) ∧
  n = (Real.sin A, -1) ∧
  m.1 * n.1 + m.2 * n.2 = 0 ∧
  a = 2 ∧
  Real.cos B = Real.sqrt 3 / 3 →
  A = π / 3 ∧ b = 4 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_and_side_length_l3649_364931


namespace NUMINAMATH_CALUDE_inequality_for_positive_reals_l3649_364975

theorem inequality_for_positive_reals : ∀ x : ℝ, x > 0 → x + 4 / x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_for_positive_reals_l3649_364975


namespace NUMINAMATH_CALUDE_g_minus_g_is_zero_l3649_364901

def f : ℕ → ℕ
| 0 => 0
| (n + 1) => if n % 2 = 0 then 2 * f (n / 2) + 1 else 2 * f n

def g (n : ℕ) : ℕ := f (f n)

theorem g_minus_g_is_zero (n : ℕ) : g (n - g n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_minus_g_is_zero_l3649_364901


namespace NUMINAMATH_CALUDE_permutations_of_five_l3649_364953

theorem permutations_of_five (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_five_l3649_364953


namespace NUMINAMATH_CALUDE_greatest_fraction_with_same_digit_sum_l3649_364960

/-- A function that returns the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Predicate to check if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_fraction_with_same_digit_sum :
  ∀ m n : ℕ, isFourDigit m → isFourDigit n → sumOfDigits m = sumOfDigits n →
  (m : ℚ) / n ≤ 9900 / 1089 :=
sorry

end NUMINAMATH_CALUDE_greatest_fraction_with_same_digit_sum_l3649_364960


namespace NUMINAMATH_CALUDE_problem_statement_l3649_364904

theorem problem_statement (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (2 * x + y) = 155 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3649_364904


namespace NUMINAMATH_CALUDE_mistaken_division_l3649_364911

theorem mistaken_division (n : ℕ) : 
  (n % 32 = 0 ∧ n / 32 = 3) → n / 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_l3649_364911


namespace NUMINAMATH_CALUDE_min_tokens_correct_l3649_364912

/-- The minimum number of tokens required to fill an n × m grid -/
def min_tokens (n m : ℕ) : ℕ :=
  if n % 2 = 0 ∨ m % 2 = 0 then
    (n + 1) / 2 + (m + 1) / 2
  else
    (n + 1) / 2 + (m + 1) / 2 - 1

/-- A function that determines if a grid can be filled given initial token placement -/
def can_fill_grid (n m : ℕ) (initial_tokens : Finset (ℕ × ℕ)) : Prop :=
  sorry

theorem min_tokens_correct (n m : ℕ) :
  ∀ (k : ℕ), k < min_tokens n m →
    ¬∃ (initial_tokens : Finset (ℕ × ℕ)),
      initial_tokens.card = k ∧
      can_fill_grid n m initial_tokens :=
  sorry

end NUMINAMATH_CALUDE_min_tokens_correct_l3649_364912


namespace NUMINAMATH_CALUDE_larger_integer_value_l3649_364917

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  a = 21 ∨ b = 21 :=
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3649_364917


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l3649_364986

theorem triangle_tangent_product (A B C : Real) (h1 : C = 2 * Real.pi / 3) 
  (h2 : Real.tan A + Real.tan B = 2 * Real.sqrt 3 / 3) : 
  Real.tan A * Real.tan B = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l3649_364986


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3649_364958

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3649_364958


namespace NUMINAMATH_CALUDE_meal_combinations_l3649_364969

def fruit_count : ℕ := 3
def salad_count : ℕ := 4
def dessert_count : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem meal_combinations :
  fruit_count * choose salad_count 2 * dessert_count = 90 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l3649_364969


namespace NUMINAMATH_CALUDE_square_difference_l3649_364957

theorem square_difference : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3649_364957


namespace NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l3649_364985

def total_trees : ℕ := 17
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := total_trees - birch_trees

theorem birch_trees_not_adjacent_probability : 
  (Nat.choose (non_birch_trees + 1) birch_trees) / (Nat.choose total_trees birch_trees) = 77 / 1033 := by
  sorry

end NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l3649_364985


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l3649_364955

/-- Represents a person's standing state -/
inductive State
  | Standing
  | Seated

/-- Represents the circular arrangement of people -/
def Arrangement := Vector State 10

/-- Checks if two adjacent people are standing -/
def hasAdjacentStanding (arr : Arrangement) : Bool :=
  sorry

/-- Checks if an arrangement is valid according to the problem rules -/
def isValidArrangement (arr : Arrangement) : Bool :=
  sorry

/-- The total number of possible arrangements -/
def totalArrangements : Nat :=
  2^8

/-- The number of valid arrangements where no two adjacent people stand -/
def validArrangements : Nat :=
  sorry

theorem no_adjacent_standing_probability :
  (validArrangements : ℚ) / totalArrangements = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l3649_364955


namespace NUMINAMATH_CALUDE_mary_final_cards_l3649_364927

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought : ℕ := 40
def cards_exchanged : ℕ := 10
def cards_lost : ℕ := 5

theorem mary_final_cards : 
  initial_cards - torn_cards + cards_from_fred + cards_bought - cards_lost = 71 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_cards_l3649_364927


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3649_364937

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 3)^2) + Real.sqrt (x^2 + (y + 3)^2) = 10) ↔
  (x^2 / 25 + y^2 / 16 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3649_364937


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3649_364922

theorem rectangle_circle_area_ratio (w l r : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 2 * Real.pi * r) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3649_364922


namespace NUMINAMATH_CALUDE_exam_questions_count_l3649_364961

/-- Calculates the total number of questions in an examination given specific conditions. -/
theorem exam_questions_count 
  (type_a_count : ℕ)
  (type_a_time : ℕ)
  (total_time : ℕ)
  (h1 : type_a_count = 50)
  (h2 : type_a_time = 72)
  (h3 : total_time = 180)
  (h4 : type_a_time * 2 ≤ total_time) :
  ∃ (type_b_count : ℕ),
    (type_a_count + type_b_count = 200) ∧
    (type_a_time + type_b_count * (type_a_time / type_a_count / 2) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_exam_questions_count_l3649_364961


namespace NUMINAMATH_CALUDE_rals_age_is_26_l3649_364987

/-- Ral's current age -/
def rals_age : ℕ := 26

/-- Suri's current age -/
def suris_age : ℕ := 13

/-- Ral is twice as old as Suri -/
axiom ral_twice_suri : rals_age = 2 * suris_age

/-- In 3 years, Suri's current age will be 16 -/
axiom suri_age_in_3_years : suris_age + 3 = 16

/-- Theorem: Ral's current age is 26 years old -/
theorem rals_age_is_26 : rals_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_rals_age_is_26_l3649_364987


namespace NUMINAMATH_CALUDE_distance_between_points_l3649_364951

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 7)
  let p2 : ℝ × ℝ := (3, -2)
  dist p1 p2 = 9 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3649_364951


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3649_364906

def is_valid (A : ℕ+) : Prop :=
  ∃ (a b : ℕ), 
    A = 2^a * 3^b ∧
    (a + 1) * (b + 1) = 3 * a * b

theorem smallest_valid_number : 
  is_valid 12 ∧ ∀ A : ℕ+, A < 12 → ¬is_valid A :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3649_364906


namespace NUMINAMATH_CALUDE_largest_rectangle_area_l3649_364971

/-- Represents a rectangular area within a square grid -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a square grid -/
structure Grid where
  size : Nat
  center : Nat × Nat

/-- Checks if a rectangle contains the center of a grid -/
def containsCenter (r : Rectangle) (g : Grid) : Prop :=
  ∃ (x y : Nat), x ≥ 1 ∧ x ≤ r.width ∧ y ≥ 1 ∧ y ≤ r.height ∧ 
    (x + (g.size - r.width) / 2, y + (g.size - r.height) / 2) = g.center

/-- Checks if a rectangle fits within a grid -/
def fitsInGrid (r : Rectangle) (g : Grid) : Prop :=
  r.width ≤ g.size ∧ r.height ≤ g.size

/-- The area of a rectangle -/
def area (r : Rectangle) : Nat :=
  r.width * r.height

/-- The theorem to be proved -/
theorem largest_rectangle_area (g : Grid) (r : Rectangle) : 
  g.size = 11 → 
  g.center = (6, 6) → 
  fitsInGrid r g → 
  ¬containsCenter r g → 
  area r ≤ 55 := by
  sorry

end NUMINAMATH_CALUDE_largest_rectangle_area_l3649_364971


namespace NUMINAMATH_CALUDE_domain_of_g_l3649_364910

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 8*x + 18⌋

theorem domain_of_g : Set.range g = Set.univ :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3649_364910


namespace NUMINAMATH_CALUDE_union_eq_univ_complement_inter_eq_open_interval_range_of_a_l3649_364964

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statements
theorem union_eq_univ : A ∪ B = Set.univ := by sorry

theorem complement_inter_eq_open_interval :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

theorem range_of_a (h : ∀ a, C a ⊆ B) :
  {a | ∀ x, x ∈ C a → x ∈ B} = Set.Icc (-2) 8 := by sorry

end NUMINAMATH_CALUDE_union_eq_univ_complement_inter_eq_open_interval_range_of_a_l3649_364964


namespace NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l3649_364999

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard -/
structure Checkerboard where
  square_size : ℝ

/-- Calculates the maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered_2inch_card (card : Card) (board : Checkerboard) :
  card.side_length = 2 →
  board.square_size = 1 →
  max_squares_covered card board = 9 :=
sorry

end NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l3649_364999


namespace NUMINAMATH_CALUDE_investment_average_rate_l3649_364997

def total_investment : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

theorem investment_average_rate :
  ∃ (x y : ℝ),
    x + y = total_investment ∧
    x * rate1 = y * rate2 / 2 ∧
    (x * rate1 + y * rate2) / total_investment = 0.041 :=
by sorry

end NUMINAMATH_CALUDE_investment_average_rate_l3649_364997


namespace NUMINAMATH_CALUDE_f_difference_l3649_364956

def f (n : ℕ) : ℚ :=
  (Finset.range (n + 1)).sum (fun i => 1 / ((n + 1 + i) : ℚ))

theorem f_difference (n : ℕ) : f (n + 1) - f n = 1 / (2 * n + 3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l3649_364956


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3649_364991

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 2) :
  (1/a + 3/b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 2 ∧ 1/a₀ + 3/b₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3649_364991


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l3649_364979

def coin_distribution (x : ℕ) : ℕ := x * (x + 1) / 2

theorem pirate_treasure_distribution (x : ℕ) :
  (coin_distribution x = 5 * x) → (x + 5 * x = 54) :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l3649_364979


namespace NUMINAMATH_CALUDE_base_conversion_3275_to_octal_l3649_364930

theorem base_conversion_3275_to_octal :
  (6 * 8^3 + 3 * 8^2 + 2 * 8^1 + 3 * 8^0 : ℕ) = 3275 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_3275_to_octal_l3649_364930


namespace NUMINAMATH_CALUDE_correct_growth_rate_l3649_364952

/-- Represents the monthly average growth rate of mask production -/
def monthly_growth_rate : ℝ := 0.1

/-- Initial daily production of masks in January -/
def initial_production : ℝ := 20000

/-- Final daily production of masks in March -/
def final_production : ℝ := 24200

/-- Number of months between initial and final production -/
def months : ℕ := 2

/-- Theorem stating that the given monthly growth rate is correct -/
theorem correct_growth_rate : 
  initial_production * (1 + monthly_growth_rate) ^ months = final_production := by
  sorry

end NUMINAMATH_CALUDE_correct_growth_rate_l3649_364952


namespace NUMINAMATH_CALUDE_units_digit_63_plus_74_base9_l3649_364993

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (a b : ℕ) : ℕ := a * 9 + b

/-- Calculates the units digit of a number in base 9 -/
def unitsDigitBase9 (n : ℕ) : ℕ := n % 9

theorem units_digit_63_plus_74_base9 :
  unitsDigitBase9 (base9ToBase10 6 3 + base9ToBase10 7 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_63_plus_74_base9_l3649_364993


namespace NUMINAMATH_CALUDE_identical_lines_condition_no_identical_lines_l3649_364909

/-- Two lines are identical if and only if they have the same slope and y-intercept -/
theorem identical_lines_condition (a b : ℝ) : 
  (∀ x y : ℝ, 2*x + a*y + b = 0 ↔ b*x - 3*y + 15 = 0) ↔ 
  ((-2/a = b/3) ∧ (-b/a = -5)) :=
sorry

/-- There are no real pairs (a, b) such that the lines 2x + ay + b = 0 and bx - 3y + 15 = 0 have the same graph -/
theorem no_identical_lines : ¬∃ a b : ℝ, ∀ x y : ℝ, 2*x + a*y + b = 0 ↔ b*x - 3*y + 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_identical_lines_condition_no_identical_lines_l3649_364909


namespace NUMINAMATH_CALUDE_range_of_increasing_function_l3649_364908

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_increasing_function (f : ℝ → ℝ) (h : increasing_function f) :
  {m : ℝ | f (2 - m) < f (m^2)} = {m : ℝ | m < -2 ∨ m > 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_increasing_function_l3649_364908


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3649_364981

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  -- The unique positive solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the conditions
    constructor
    · -- Prove 3 > 0
      sorry
    · -- Prove 3 * 3^2 - 7 * 3 - 6 = 0
      sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3649_364981


namespace NUMINAMATH_CALUDE_quadratic_max_min_l3649_364990

def f (x : ℝ) := x^2 - 4*x + 2

theorem quadratic_max_min :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 5, f x ≤ max ∧ min ≤ f x) ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) 5, f x₁ = max) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) 5, f x₂ = min) ∧
    max = 14 ∧ min = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_min_l3649_364990


namespace NUMINAMATH_CALUDE_f_2008_value_l3649_364940

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_initial : f 0 = 2008
axiom f_inequality1 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality2 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x

-- State the theorem
theorem f_2008_value : f 2008 = 2007 + 2^2008 := by sorry

end NUMINAMATH_CALUDE_f_2008_value_l3649_364940


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3649_364974

theorem complex_equation_solution (m : ℝ) :
  (2 : ℂ) / (1 - Complex.I) = 1 + m * Complex.I → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3649_364974


namespace NUMINAMATH_CALUDE_clownfish_in_display_tank_l3649_364942

theorem clownfish_in_display_tank 
  (total_fish : ℕ)
  (clownfish blowfish : ℕ)
  (blowfish_in_own_tank : ℕ)
  (h1 : total_fish = 100)
  (h2 : clownfish = blowfish)
  (h3 : blowfish_in_own_tank = 26)
  (h4 : total_fish = clownfish + blowfish) :
  let blowfish_in_display := blowfish - blowfish_in_own_tank
  let initial_clownfish_in_display := blowfish_in_display
  let final_clownfish_in_display := initial_clownfish_in_display - initial_clownfish_in_display / 3
  final_clownfish_in_display = 16 := by
sorry

end NUMINAMATH_CALUDE_clownfish_in_display_tank_l3649_364942


namespace NUMINAMATH_CALUDE_identical_asymptotes_iff_M_eq_225_div_16_l3649_364941

/-- Two hyperbolas have identical asymptotes if and only if M = 225/16 -/
theorem identical_asymptotes_iff_M_eq_225_div_16 :
  ∀ (M : ℝ),
  (∀ (x y : ℝ), x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) ↔ M = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_identical_asymptotes_iff_M_eq_225_div_16_l3649_364941


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3649_364943

theorem cubic_equation_roots :
  ∃ (pos_roots : ℕ), 
    (pos_roots = 1 ∨ pos_roots = 3) ∧
    (∀ x : ℝ, x^3 - 3*x^2 + 4*x - 12 = 0 → x > 0) ∧
    (¬∃ x : ℝ, x < 0 ∧ x^3 - 3*x^2 + 4*x - 12 = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3649_364943


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3649_364916

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (a b : ℝ), x^2 - 6*x + 7 = 0 ↔ (x + a)^2 = b ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3649_364916
