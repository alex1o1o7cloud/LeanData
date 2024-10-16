import Mathlib

namespace NUMINAMATH_CALUDE_point_N_coordinates_l1430_143011

-- Define the point M and vector a
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)

-- Define the relation between MN and a
def MN_relation (N : ℝ × ℝ) : Prop :=
  (N.1 - M.1, N.2 - M.2) = (-3 * a.1, -3 * a.2)

-- Theorem statement
theorem point_N_coordinates :
  ∃ N : ℝ × ℝ, MN_relation N ∧ N = (2, 0) := by sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l1430_143011


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1430_143069

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_complement_theorem :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1430_143069


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_is_correct_l1430_143087

/-- The repeating decimal 3.71717171... -/
def repeating_decimal : ℚ := 3 + 71/99

/-- The sum of the numerator and denominator of the fraction representing
    the repeating decimal 3.71717171... in its lowest terms -/
def sum_of_fraction_parts : ℕ := 467

/-- Theorem stating that the sum of the numerator and denominator of the fraction
    representing 3.71717171... in its lowest terms is 467 -/
theorem sum_of_fraction_parts_is_correct :
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n + d = sum_of_fraction_parts := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_is_correct_l1430_143087


namespace NUMINAMATH_CALUDE_constant_term_implies_n_12_l1430_143050

/-- The general term formula for the expansion of (√x - 2/x)^n -/
def generalTerm (n : ℕ) (r : ℕ) : ℚ → ℚ := 
  λ x => (n.choose r) * (-2)^r * x^((n - 3*r) / 2)

/-- The condition that the 5th term (r = 4) is the constant term -/
def fifthTermIsConstant (n : ℕ) : Prop :=
  (n - 3*4) / 2 = 0

theorem constant_term_implies_n_12 : 
  ∀ n : ℕ, fifthTermIsConstant n → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_implies_n_12_l1430_143050


namespace NUMINAMATH_CALUDE_statue_weight_proof_l1430_143097

def original_weight : ℝ := 190

def week1_reduction : ℝ := 0.25
def week2_reduction : ℝ := 0.15
def week3_reduction : ℝ := 0.10

def final_weight : ℝ := original_weight * (1 - week1_reduction) * (1 - week2_reduction) * (1 - week3_reduction)

theorem statue_weight_proof : final_weight = 108.9125 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_proof_l1430_143097


namespace NUMINAMATH_CALUDE_black_circle_area_black_circle_area_proof_l1430_143079

theorem black_circle_area (cube_edge : Real) (yellow_paint_area : Real) : Real :=
  let cube_face_area := cube_edge ^ 2
  let total_surface_area := 6 * cube_face_area
  let yellow_area_per_face := yellow_paint_area / 6
  let black_circle_area := cube_face_area - yellow_area_per_face
  
  black_circle_area

theorem black_circle_area_proof :
  black_circle_area 12 432 = 72 := by
  sorry

end NUMINAMATH_CALUDE_black_circle_area_black_circle_area_proof_l1430_143079


namespace NUMINAMATH_CALUDE_moon_permutations_l1430_143074

-- Define the word as a list of characters
def moon : List Char := ['M', 'O', 'O', 'N']

-- Define the number of unique permutations
def uniquePermutations (word : List Char) : ℕ :=
  Nat.factorial word.length / (Nat.factorial (word.count 'O'))

-- Theorem statement
theorem moon_permutations :
  uniquePermutations moon = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_permutations_l1430_143074


namespace NUMINAMATH_CALUDE_tan_pi_twelve_l1430_143039

theorem tan_pi_twelve : Real.tan (π / 12) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_twelve_l1430_143039


namespace NUMINAMATH_CALUDE_distance_to_walk_back_l1430_143014

/-- Represents the distance traveled by Vintik and Shpuntik -/
def TravelDistance (x y : ℝ) : Prop :=
  -- Vintik's total distance is 12 km
  2 * x + y = 12 ∧
  -- Total fuel consumption is 75 liters
  3 * x + 15 * y = 75 ∧
  -- x represents half of Vintik's forward distance
  x > 0 ∧ y > 0

/-- The theorem stating the distance to walk back home -/
theorem distance_to_walk_back (x y : ℝ) (h : TravelDistance x y) : 
  3 * x - 3 * y = 9 :=
sorry

end NUMINAMATH_CALUDE_distance_to_walk_back_l1430_143014


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1430_143063

/-- Estimates the number of fish in a pond using the capture-recapture method. -/
def estimate_fish_population (initially_marked : ℕ) (second_sample : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initially_marked * second_sample) / marked_in_second

/-- Theorem stating that the estimated fish population is 500 given the specific conditions. -/
theorem fish_population_estimate :
  let initially_marked := 10
  let second_sample := 100
  let marked_in_second := 2
  estimate_fish_population initially_marked second_sample marked_in_second = 500 := by
  sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l1430_143063


namespace NUMINAMATH_CALUDE_beijing_hangzhou_temp_difference_l1430_143007

/-- The temperature difference between two cities --/
def temperature_difference (temp1 : Int) (temp2 : Int) : Int :=
  temp2 - temp1

/-- Theorem: The temperature difference between Beijing and Hangzhou is 9°C --/
theorem beijing_hangzhou_temp_difference :
  temperature_difference (-10) (-1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_beijing_hangzhou_temp_difference_l1430_143007


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1430_143065

-- Define the equation for the radii
def radius_equation (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Define the radii R and r as real numbers satisfying the equation
def R : ℝ := sorry
def r : ℝ := sorry

-- Define the distance between centers
def d : ℝ := 3

-- State the theorem
theorem circles_externally_tangent :
  radius_equation R ∧ radius_equation r ∧ R ≠ r ∧ d = 3 →
  d = R + r :=
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1430_143065


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l1430_143071

/-- Given a student who needs to obtain 40% to pass, got 150 marks, and failed by 50 marks, 
    the maximum possible marks are 500. -/
theorem maximum_marks_calculation (passing_threshold : ℝ) (marks_obtained : ℝ) (marks_short : ℝ) :
  passing_threshold = 0.4 →
  marks_obtained = 150 →
  marks_short = 50 →
  ∃ (max_marks : ℝ), max_marks = 500 ∧ passing_threshold * max_marks = marks_obtained + marks_short :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l1430_143071


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_value_b_l1430_143006

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def cosine_condition (t : Triangle) : Prop :=
  2 * Real.cos t.B * (t.c * Real.cos t.A + t.a * Real.cos t.C) = t.b

def area_condition (t : Triangle) : Prop :=
  (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2

-- Theorem 1: Arithmetic sequence
theorem arithmetic_sequence (t : Triangle) 
  (h1 : triangle_condition t) (h2 : cosine_condition t) : 
  ∃ r : Real, t.A = t.B - r ∧ t.C = t.B + r :=
sorry

-- Theorem 2: Minimum value of b
theorem min_value_b (t : Triangle) 
  (h1 : triangle_condition t) (h2 : area_condition t) : 
  t.b ≥ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_value_b_l1430_143006


namespace NUMINAMATH_CALUDE_two_car_garage_count_l1430_143090

theorem two_car_garage_count (total : ℕ) (pool : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 90 → pool = 40 → both = 35 → neither = 35 → 
  ∃ (garage : ℕ), garage = 50 ∧ garage + pool - both = total - neither :=
by
  sorry

end NUMINAMATH_CALUDE_two_car_garage_count_l1430_143090


namespace NUMINAMATH_CALUDE_expression_simplification_l1430_143082

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - a / (a - 2)) / ((a^2 + 2*a) / (a - 2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1430_143082


namespace NUMINAMATH_CALUDE_five_div_sqrt_five_times_one_over_sqrt_five_equals_one_l1430_143002

theorem five_div_sqrt_five_times_one_over_sqrt_five_equals_one :
  ∀ (sqrt_five : ℝ), sqrt_five > 0 → sqrt_five * sqrt_five = 5 →
  5 / sqrt_five * (1 / sqrt_five) = 1 := by
sorry

end NUMINAMATH_CALUDE_five_div_sqrt_five_times_one_over_sqrt_five_equals_one_l1430_143002


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1430_143010

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1430_143010


namespace NUMINAMATH_CALUDE_product_factor_proof_l1430_143047

theorem product_factor_proof (w : ℕ+) (h1 : 2^5 ∣ (936 * w)) (h2 : 3^3 ∣ (936 * w)) (h3 : w ≥ 144) :
  ∃ x : ℕ, 12^x ∣ (936 * w) ∧ ∀ y : ℕ, 12^y ∣ (936 * w) → y ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_product_factor_proof_l1430_143047


namespace NUMINAMATH_CALUDE_area_bisectors_perpendicular_l1430_143016

/-- Triangle with two sides of length 6 and one side of length 8 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : a = b ∧ a = 6 ∧ c = 8

/-- Area bisector of a triangle -/
def AreaBisector (t : IsoscelesTriangle) := ℝ → ℝ

/-- The angle between two lines -/
def AngleBetween (l1 l2 : ℝ → ℝ) : ℝ := sorry

theorem area_bisectors_perpendicular (t : IsoscelesTriangle) 
  (b1 b2 : AreaBisector t) (h : b1 ≠ b2) : 
  AngleBetween b1 b2 = π / 2 := by sorry

end NUMINAMATH_CALUDE_area_bisectors_perpendicular_l1430_143016


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1430_143083

theorem algebraic_expression_value (a b : ℝ) (h : a + b - 2 = 0) :
  a^2 - b^2 + 4*b = 4 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1430_143083


namespace NUMINAMATH_CALUDE_sabrina_pencils_l1430_143044

theorem sabrina_pencils (total : ℕ) (justin_extra : ℕ) : 
  total = 50 → justin_extra = 8 →
  ∃ (sabrina : ℕ), 
    sabrina + (2 * sabrina + justin_extra) = total ∧ 
    sabrina = 14 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_pencils_l1430_143044


namespace NUMINAMATH_CALUDE_emily_seeds_l1430_143059

/-- Calculates the total number of seeds Emily started with -/
def total_seeds (big_garden_seeds : ℕ) (num_small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + num_small_gardens * seeds_per_small_garden

/-- Proves that Emily started with 41 seeds -/
theorem emily_seeds : 
  total_seeds 29 3 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_emily_seeds_l1430_143059


namespace NUMINAMATH_CALUDE_count_valid_numbers_valid_numbers_are_l1430_143076

def digits : List Nat := [2, 3, 0]

def is_valid_number (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits

def valid_numbers : List Nat :=
  (List.range 1000).filter is_valid_number

theorem count_valid_numbers :
  valid_numbers.length = 4 := by sorry

theorem valid_numbers_are :
  valid_numbers = [230, 203, 302, 320] := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_valid_numbers_are_l1430_143076


namespace NUMINAMATH_CALUDE_circle_ratio_l1430_143080

theorem circle_ratio (r R a c : Real) (hr : r > 0) (hR : R > r) (ha : a > c) (hc : c > 0) :
  π * R^2 = (a - c) * (π * R^2 - π * r^2) →
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1430_143080


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_137_l1430_143098

theorem first_nonzero_digit_of_1_137 :
  ∃ (n : ℕ) (k : ℕ), 
    10^n > 137 ∧ 
    (1000 : ℚ) / 137 = k + (1000 - k * 137 : ℚ) / 137 ∧ 
    k = 7 := by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_137_l1430_143098


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l1430_143012

/-- The probability of getting heads in a single toss of a fair coin -/
def p_heads : ℚ := 1/2

/-- The number of tosses -/
def n : ℕ := 5

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def prob_exactly (k : ℕ) : ℚ :=
  ↑(n.choose k) * p_heads ^ k * (1 - p_heads) ^ (n - k)

/-- The probability of getting at least 2 heads in 5 tosses of a fair coin -/
def prob_at_least_two : ℚ :=
  1 - prob_exactly 0 - prob_exactly 1

theorem fair_coin_five_tosses :
  prob_at_least_two = 13/16 := by sorry

end NUMINAMATH_CALUDE_fair_coin_five_tosses_l1430_143012


namespace NUMINAMATH_CALUDE_total_players_l1430_143031

/-- The total number of players in a game scenario -/
theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) :
  kabaddi = 10 →
  kho_kho_only = 15 →
  both = 5 →
  kabaddi + kho_kho_only - both = 25 := by
sorry

end NUMINAMATH_CALUDE_total_players_l1430_143031


namespace NUMINAMATH_CALUDE_component_service_life_probability_l1430_143013

theorem component_service_life_probability 
  (P_exceed_1_year : ℝ) 
  (P_exceed_2_years : ℝ) 
  (h1 : P_exceed_1_year = 0.6) 
  (h2 : P_exceed_2_years = 0.3) :
  (P_exceed_2_years / P_exceed_1_year) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_component_service_life_probability_l1430_143013


namespace NUMINAMATH_CALUDE_curve_tangent_problem_l1430_143053

theorem curve_tangent_problem (a b : ℝ) : 
  (2 * a + b / 2 = -5) →  -- Curve passes through (2, -5)
  (4 * a - b / 4 = -7/2) →  -- Tangent slope at (2, -5) is -7/2
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_problem_l1430_143053


namespace NUMINAMATH_CALUDE_remainder_3042_div_29_l1430_143004

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3042_div_29_l1430_143004


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_product_l1430_143052

theorem min_sum_reciprocal_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_product_l1430_143052


namespace NUMINAMATH_CALUDE_even_sum_digits_all_residues_l1430_143073

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Main theorem -/
theorem even_sum_digits_all_residues (k : ℕ) (h : k ≥ 2) :
  ∀ r, r < k → ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r :=
by sorry

end NUMINAMATH_CALUDE_even_sum_digits_all_residues_l1430_143073


namespace NUMINAMATH_CALUDE_line_and_circle_proof_l1430_143024

-- Define the lines and circles
def l₁ (x y : ℝ) : Prop := x - 2*y + 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x - y - 2 = 0
def c₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def c₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line that we want to prove
def target_line (x y : ℝ) : Prop := y = x

-- Define the circle that we want to prove
def target_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

-- Define the line on which the center of the target circle should lie
def center_line (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

theorem line_and_circle_proof :
  -- Part 1: The target line passes through the origin and the intersection of l₁ and l₂
  (∃ x y : ℝ, l₁ x y ∧ l₂ x y ∧ target_line x y) ∧
  target_line 0 0 ∧
  -- Part 2: The target circle has its center on the center_line and passes through
  -- the intersection points of c₁ and c₂
  (∃ x y : ℝ, center_line x y ∧ 
    ∀ a b : ℝ, (c₁ a b ∧ c₂ a b) → target_circle a b) :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_proof_l1430_143024


namespace NUMINAMATH_CALUDE_lottery_possibility_l1430_143023

theorem lottery_possibility (win_chance : ℝ) (h : win_chance = 0.01) : 
  ∃ (outcome : Bool), outcome = true :=
sorry

end NUMINAMATH_CALUDE_lottery_possibility_l1430_143023


namespace NUMINAMATH_CALUDE_hamiltonian_circuit_theorem_l1430_143001

/-- Represents a rectangle on a grid with unit cells -/
structure GridRectangle where
  m : ℕ  -- width
  n : ℕ  -- height

/-- Determines if a Hamiltonian circuit exists on the grid rectangle -/
def has_hamiltonian_circuit (rect : GridRectangle) : Prop :=
  rect.m > 0 ∧ rect.n > 0 ∧ (Odd rect.m ∨ Odd rect.n)

/-- Calculates the length of the Hamiltonian circuit when it exists -/
def hamiltonian_circuit_length (rect : GridRectangle) : ℕ :=
  (rect.m + 1) * (rect.n + 1)

theorem hamiltonian_circuit_theorem (rect : GridRectangle) :
  has_hamiltonian_circuit rect ↔
    ∃ (path_length : ℕ), 
      path_length = hamiltonian_circuit_length rect ∧
      path_length > 0 :=
sorry

end NUMINAMATH_CALUDE_hamiltonian_circuit_theorem_l1430_143001


namespace NUMINAMATH_CALUDE_inf_a_plus_2b_is_3_l1430_143033

open Real

/-- Given 0 < a < b and |log a| = |log b|, the infimum of a + 2b is 3 -/
theorem inf_a_plus_2b_is_3 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |log a| = |log b|) :
  ∃ (inf : ℝ), inf = 3 ∧ ∀ x, (∃ (a' b' : ℝ), 0 < a' ∧ a' < b' ∧ |log a'| = |log b'| ∧ x = a' + 2*b') → inf ≤ x :=
sorry

end NUMINAMATH_CALUDE_inf_a_plus_2b_is_3_l1430_143033


namespace NUMINAMATH_CALUDE_tan_negative_240_degrees_l1430_143070

theorem tan_negative_240_degrees : Real.tan (-(240 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_240_degrees_l1430_143070


namespace NUMINAMATH_CALUDE_prism_volume_l1430_143075

/-- The volume of a right rectangular prism with face areas 40, 50, and 100 square centimeters -/
theorem prism_volume (x y z : ℝ) (hxy : x * y = 40) (hxz : x * z = 50) (hyz : y * z = 100) :
  x * y * z = 100 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1430_143075


namespace NUMINAMATH_CALUDE_simplify_expression_l1430_143005

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (b * c)) * (a * b / (a^2 - (b + c)^2)) = 1 / (c * (a - b - c)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1430_143005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1430_143040

theorem arithmetic_sequence_sum (n : ℕ) (d : ℝ) (total_sum : ℝ) :
  n = 2023 →
  d = 2 →
  total_sum = 2080 →
  let a₁ := (total_sum - (n - 1) * n * d / 2) / n
  let subsequence_sum := (n / 3) * (2 * a₁ + (n / 3 - 1) * 3 * d) / 2
  subsequence_sum = 674 * ((2080 - 2022 * 2011) / 2023 + 2019) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1430_143040


namespace NUMINAMATH_CALUDE_hyperbola_circumradius_l1430_143096

/-- The hyperbola in the xy-plane -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The centroid of triangle F₁PF₂ -/
def G : ℝ × ℝ := sorry

/-- The incenter of triangle F₁PF₂ -/
def I : ℝ × ℝ := sorry

/-- The circumradius of triangle F₁PF₂ -/
def R : ℝ := sorry

theorem hyperbola_circumradius :
  hyperbola P.1 P.2 ∧ 
  (G.2 = I.2) →  -- GI is parallel to x-axis
  R = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_circumradius_l1430_143096


namespace NUMINAMATH_CALUDE_solution_implies_relationship_l1430_143019

theorem solution_implies_relationship (a b c : ℝ) :
  (a * (-3) + c * (-2) = 1) →
  (c * (-3) - b * (-2) = 2) →
  9 * a + 4 * b = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_relationship_l1430_143019


namespace NUMINAMATH_CALUDE_sum_due_calculation_l1430_143045

/-- The relationship between banker's discount, true discount, and sum due -/
def banker_discount_relation (bd td sd : ℝ) : Prop :=
  bd = td + td^2 / sd

/-- The problem statement -/
theorem sum_due_calculation (bd td : ℝ) (h1 : bd = 36) (h2 : td = 30) :
  ∃ sd : ℝ, banker_discount_relation bd td sd ∧ sd = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_calculation_l1430_143045


namespace NUMINAMATH_CALUDE_expression_simplification_l1430_143072

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : 3 * x + y / 3 + 2 * z ≠ 0) :
  (3 * x + y / 3 + 2 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (2 * z)⁻¹) = 
  (2 * y + 18 * x * z + 3 * z * x) / (6 * x * y * z * (9 * x + y + 6 * z)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1430_143072


namespace NUMINAMATH_CALUDE_units_digit_problem_l1430_143034

theorem units_digit_problem : ∃ n : ℕ, (3 * 19 * 1933 - 3^4) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1430_143034


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1430_143061

/-- A dodecahedron is a polyhedron with 12 pentagonal faces and 20 vertices,
    where three faces meet at each vertex. -/
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_per_vertex : Nat
  faces_are_pentagonal : faces = 12
  vertex_count : vertices = 20
  three_faces_per_vertex : faces_per_vertex = 3

/-- An interior diagonal of a dodecahedron is a segment connecting two vertices
    which do not lie on the same face. -/
def interior_diagonal (d : Dodecahedron) := Unit

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : Nat :=
  (d.vertices * (d.vertices - 1 - d.faces_per_vertex)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  num_interior_diagonals d = 160 := by
  sorry

#check dodecahedron_interior_diagonals

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1430_143061


namespace NUMINAMATH_CALUDE_backpack_cost_theorem_l1430_143027

/-- Calculates the total cost of personalized backpacks with a discount -/
def total_cost (num_backpacks : ℕ) (original_price : ℚ) (discount_rate : ℚ) (monogram_fee : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_discounted := num_backpacks.cast * discounted_price
  let total_monogram := num_backpacks.cast * monogram_fee
  total_discounted + total_monogram

/-- Theorem stating that the total cost of 5 backpacks with given prices and discount is $140.00 -/
theorem backpack_cost_theorem :
  total_cost 5 20 (1/5) 12 = 140 := by
  sorry

end NUMINAMATH_CALUDE_backpack_cost_theorem_l1430_143027


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1430_143018

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def players_to_choose : ℕ := 8
def max_quadruplets : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose total_players players_to_choose) -
  (Nat.choose quadruplets 3 * Nat.choose (total_players - quadruplets) (players_to_choose - 3) +
   Nat.choose quadruplets 4 * Nat.choose (total_players - quadruplets) (players_to_choose - 4)) = 34749 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1430_143018


namespace NUMINAMATH_CALUDE_acute_triangle_side_constraint_acute_triangle_side_constraint_converse_l1430_143043

/-- A triangle with side lengths a, b, and c is acute if and only if a² + b² > c², where c is the longest side. -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  c ≥ a ∧ c ≥ b ∧ a^2 + b^2 > c^2

/-- The theorem states that for an acute triangle with side lengths x²+4, 4x, and x²+6,
    where x is a positive real number, x must be greater than √(15)/3. -/
theorem acute_triangle_side_constraint (x : ℝ) :
  x > 0 →
  is_acute_triangle (x^2 + 4) (4*x) (x^2 + 6) →
  x > Real.sqrt 15 / 3 :=
by sorry

/-- The converse of the theorem: if x > √(15)/3, then the triangle with side lengths
    x²+4, 4x, and x²+6 is acute. -/
theorem acute_triangle_side_constraint_converse (x : ℝ) :
  x > Real.sqrt 15 / 3 →
  is_acute_triangle (x^2 + 4) (4*x) (x^2 + 6) :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_side_constraint_acute_triangle_side_constraint_converse_l1430_143043


namespace NUMINAMATH_CALUDE_congruence_solution_l1430_143037

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 9 % 47 ↔ n % 47 = 39 % 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1430_143037


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l1430_143093

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 400 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l1430_143093


namespace NUMINAMATH_CALUDE_line_slope_intercept_form_l1430_143060

/-- Definition of the line using vector dot product -/
def line_equation (x y : ℝ) : Prop :=
  (3 * (x - 2)) + (-4 * (y - 8)) = 0

/-- Slope-intercept form of a line -/
def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

/-- Theorem stating that the given line equation is equivalent to y = (3/4)x + 6.5 -/
theorem line_slope_intercept_form :
  ∀ x y : ℝ, line_equation x y ↔ slope_intercept_form (3/4) (13/2) x y :=
by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_form_l1430_143060


namespace NUMINAMATH_CALUDE_balls_sold_l1430_143042

/-- Proves that the number of balls sold is 17 given the conditions of the problem -/
theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) :
  selling_price = 720 →
  loss = 5 * cost_price →
  cost_price = 60 →
  selling_price + loss = 17 * cost_price :=
by
  sorry

#check balls_sold

end NUMINAMATH_CALUDE_balls_sold_l1430_143042


namespace NUMINAMATH_CALUDE_expression_simplification_l1430_143025

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 1) :
  (x^2 - 1) / x / (1 + 1/x) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1430_143025


namespace NUMINAMATH_CALUDE_jack_final_apples_l1430_143099

def initial_apples : ℕ := 150
def sold_to_jill_percent : ℚ := 30 / 100
def sold_to_june_percent : ℚ := 20 / 100
def apples_eaten : ℕ := 2
def apples_given_to_teacher : ℕ := 1

theorem jack_final_apples :
  let after_jill := initial_apples - (initial_apples * sold_to_jill_percent).floor
  let after_june := after_jill - (after_jill * sold_to_june_percent).floor
  let after_eating := after_june - apples_eaten
  let final_apples := after_eating - apples_given_to_teacher
  final_apples = 81 := by sorry

end NUMINAMATH_CALUDE_jack_final_apples_l1430_143099


namespace NUMINAMATH_CALUDE_village_population_l1430_143054

theorem village_population (P : ℝ) : 
  P > 0 →
  (P * 1.05 * 0.95 = 9975) →
  P = 10000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l1430_143054


namespace NUMINAMATH_CALUDE_apps_deleted_l1430_143028

theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) 
  (h1 : initial_apps = 16) (h2 : remaining_apps = 8) : 
  initial_apps - remaining_apps = 8 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l1430_143028


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1430_143062

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1430_143062


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1430_143068

theorem perfect_square_condition (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  (∃ k : ℕ, 2^m + 3^n = k^2) ↔ 
  (∃ a b : ℕ, m = 2*a ∧ n = 2*b ∧ a ≥ 1 ∧ b ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1430_143068


namespace NUMINAMATH_CALUDE_number_of_distributions_l1430_143022

/-- The number of ways to distribute 5 students into 3 groups with constraints -/
def distribution_schemes : ℕ :=
  -- The actual calculation would go here, but we don't have the solution steps
  80

/-- Theorem stating the number of distribution schemes -/
theorem number_of_distributions :
  distribution_schemes = 80 :=
by
  -- The proof would go here
  sorry

#check number_of_distributions

end NUMINAMATH_CALUDE_number_of_distributions_l1430_143022


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l1430_143086

theorem arithmetic_geometric_progression (k : ℝ) :
  ∃ (x y z : ℝ),
    x + y + z = k ∧
    y - x = z - y ∧
    y^2 = x * (z + k/6) ∧
    ((x = k/6 ∧ y = k/3 ∧ z = k/2) ∨ (x = 2*k/3 ∧ y = k/3 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l1430_143086


namespace NUMINAMATH_CALUDE_possible_m_values_l1430_143036

-- Define set A
def A : Set ℤ := {-1, 1}

-- Define set B
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

-- Theorem statement
theorem possible_m_values :
  ∀ m : ℤ, B m ⊆ A → (m = 0 ∨ m = 1 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_possible_m_values_l1430_143036


namespace NUMINAMATH_CALUDE_graph_single_point_implies_d_value_l1430_143051

/-- The equation of the graph -/
def graph_equation (x y : ℝ) (d : ℝ) : Prop :=
  x^2 + 3*y^2 + 6*x - 18*y + d = 0

/-- The graph consists of a single point -/
def single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, graph_equation p.1 p.2 d

/-- If the graph of x^2 + 3y^2 + 6x - 18y + d = 0 consists of a single point, then d = -27 -/
theorem graph_single_point_implies_d_value :
  ∀ d : ℝ, single_point d → d = -27 := by
  sorry

end NUMINAMATH_CALUDE_graph_single_point_implies_d_value_l1430_143051


namespace NUMINAMATH_CALUDE_greatest_common_factor_45_75_90_l1430_143009

theorem greatest_common_factor_45_75_90 : Nat.gcd 45 (Nat.gcd 75 90) = 15 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_45_75_90_l1430_143009


namespace NUMINAMATH_CALUDE_calculate_expression_l1430_143046

theorem calculate_expression : -1^2023 + 8 / (-2)^2 - |-4| * 5 = -19 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1430_143046


namespace NUMINAMATH_CALUDE_unique_function_f_l1430_143095

/-- A function from [1,+∞) to [1,+∞) satisfying given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x ≥ 1 → f x ≥ 1) ∧ 
  (∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
  (∀ x : ℝ, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

/-- The unique function satisfying the conditions is f(x) = x + 1 -/
theorem unique_function_f :
  ∃! f : ℝ → ℝ, FunctionF f ∧ ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_f_l1430_143095


namespace NUMINAMATH_CALUDE_range_of_a_l1430_143029

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 2 ≤ 0) → 
  -2 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1430_143029


namespace NUMINAMATH_CALUDE_extra_oil_amount_l1430_143094

-- Define the given conditions
def price_reduction : ℚ := 25 / 100
def reduced_price : ℚ := 40
def total_money : ℚ := 800

-- Define the function to calculate the original price
def original_price : ℚ := reduced_price / (1 - price_reduction)

-- Define the function to calculate the amount of oil that can be bought
def oil_amount (price : ℚ) : ℚ := total_money / price

-- State the theorem
theorem extra_oil_amount : 
  oil_amount reduced_price - oil_amount original_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_extra_oil_amount_l1430_143094


namespace NUMINAMATH_CALUDE_young_photographer_club_l1430_143092

theorem young_photographer_club (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos :=
by sorry

end NUMINAMATH_CALUDE_young_photographer_club_l1430_143092


namespace NUMINAMATH_CALUDE_speech_arrangement_count_l1430_143057

/-- The number of ways to arrange speeches for 3 boys and 2 girls chosen from a group of 4 boys and 3 girls, where the girls do not give consecutive speeches. -/
def speech_arrangements (total_boys : ℕ) (total_girls : ℕ) (chosen_boys : ℕ) (chosen_girls : ℕ) : ℕ :=
  (Nat.choose total_boys chosen_boys) * 
  (Nat.choose total_girls chosen_girls) * 
  (Nat.factorial chosen_boys) * 
  (Nat.factorial (chosen_boys + 1))

theorem speech_arrangement_count :
  speech_arrangements 4 3 3 2 = 864 := by
  sorry

end NUMINAMATH_CALUDE_speech_arrangement_count_l1430_143057


namespace NUMINAMATH_CALUDE_p_range_nonnegative_reals_l1430_143021

/-- The function p(x) = x^4 - 6x^2 + 9 -/
def p (x : ℝ) : ℝ := x^4 - 6*x^2 + 9

theorem p_range_nonnegative_reals :
  Set.range (fun (x : ℝ) ↦ p x) = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_p_range_nonnegative_reals_l1430_143021


namespace NUMINAMATH_CALUDE_coefficients_of_equation_l1430_143030

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation 3x^2 - x - 2 = 0 -/
def equation : QuadraticEquation :=
  { a := 3, b := -1, c := -2 }

theorem coefficients_of_equation :
  equation.a = 3 ∧ equation.b = -1 ∧ equation.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_equation_l1430_143030


namespace NUMINAMATH_CALUDE_circle_area_outside_triangle_l1430_143056

-- Define the right triangle ABC
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  right_angle : AC^2 = AB^2 + BC^2

-- Define the circle
structure TangentCircle (t : RightTriangle) where
  radius : ℝ
  tangent_AB : radius = t.AB / 2
  diametric_point_on_BC : radius * 2 ≤ t.BC

-- Main theorem
theorem circle_area_outside_triangle (t : RightTriangle) (c : TangentCircle t)
  (h1 : t.AB = 8)
  (h2 : t.BC = 10) :
  (π * c.radius^2 / 4) - (c.radius^2 / 2) = 4*π - 8 := by
  sorry


end NUMINAMATH_CALUDE_circle_area_outside_triangle_l1430_143056


namespace NUMINAMATH_CALUDE_line_l_is_correct_l1430_143081

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-2, -3)

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0

-- Theorem statement
theorem line_l_is_correct :
  (∀ x y : ℝ, line_l x y → (x, y) = point_A ∨ (x, y) ≠ point_A) ∧
  (∀ x y : ℝ, line_l x y → given_line x y → False) ∧
  line_l point_A.1 point_A.2 :=
sorry

end NUMINAMATH_CALUDE_line_l_is_correct_l1430_143081


namespace NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_l1430_143091

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fifth_and_eighth (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) - 5 = 0 →
  (a 10)^2 - 3*(a 10) - 5 = 0 →
  a 5 + a 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_l1430_143091


namespace NUMINAMATH_CALUDE_vegetable_bins_l1430_143067

theorem vegetable_bins (soup_bins pasta_bins total_bins : Real) 
  (h1 : soup_bins = 0.125)
  (h2 : pasta_bins = 0.5)
  (h3 : total_bins = 0.75) :
  total_bins - soup_bins - pasta_bins = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_bins_l1430_143067


namespace NUMINAMATH_CALUDE_polyhedron_edge_face_relation_l1430_143064

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  face_counts : ℕ → ℕ
  convex : True

/-- The sum of k * f_k for all k ≥ 3 equals twice the number of edges -/
theorem polyhedron_edge_face_relation (P : ConvexPolyhedron) :
  2 * P.edges = ∑' k, k * P.face_counts k :=
sorry

end NUMINAMATH_CALUDE_polyhedron_edge_face_relation_l1430_143064


namespace NUMINAMATH_CALUDE_smallest_area_triangle_l1430_143032

-- Define the angle XAY
def Angle (X A Y : Point) : Prop := sorry

-- Define a point O inside the angle XAY
def InsideAngle (O X A Y : Point) : Prop := sorry

-- Define symmetry of angles with respect to a point
def SymmetricAngle (X A Y X' A' Y' O : Point) : Prop := sorry

-- Define the intersection points B and C
def IntersectionPoints (B C X A Y X' A' Y' O : Point) : Prop := sorry

-- Define a line passing through three points
def LineThroughPoints (P Q R : Point) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem smallest_area_triangle 
  (X A Y O : Point) 
  (h1 : Angle X A Y) 
  (h2 : InsideAngle O X A Y) 
  (X' A' Y' : Point) 
  (h3 : SymmetricAngle X A Y X' A' Y' O) 
  (B C : Point) 
  (h4 : IntersectionPoints B C X A Y X' A' Y' O) 
  (h5 : LineThroughPoints B O C) :
  ∀ P Q : Point, 
    LineThroughPoints P O Q → 
    TriangleArea A P Q ≥ TriangleArea A B C := 
by sorry

end NUMINAMATH_CALUDE_smallest_area_triangle_l1430_143032


namespace NUMINAMATH_CALUDE_milk_production_l1430_143020

theorem milk_production (M : ℝ) 
  (h1 : M > 0) 
  (h2 : M * 0.25 * 0.5 = 2) : M = 16 := by
  sorry

end NUMINAMATH_CALUDE_milk_production_l1430_143020


namespace NUMINAMATH_CALUDE_sales_tax_satisfies_conditions_l1430_143026

/-- The sales tax percentage that satisfies the given conditions -/
def sales_tax_percentage : ℝ :=
  -- Define the sales tax percentage
  -- We don't know its exact value yet
  sorry

/-- The cost of the lunch before tax and tip -/
def lunch_cost : ℝ := 100

/-- The tip percentage -/
def tip_percentage : ℝ := 0.06

/-- The total amount paid -/
def total_paid : ℝ := 110

/-- Theorem stating that the sales tax percentage satisfies the given conditions -/
theorem sales_tax_satisfies_conditions :
  lunch_cost + sales_tax_percentage + 
  tip_percentage * (lunch_cost + sales_tax_percentage) = total_paid :=
by
  sorry

end NUMINAMATH_CALUDE_sales_tax_satisfies_conditions_l1430_143026


namespace NUMINAMATH_CALUDE_committee_probability_l1430_143055

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 5

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_combinations := Nat.choose boys committee_size
  let all_girls_combinations := Nat.choose girls committee_size
  let favorable_combinations := total_combinations - (all_boys_combinations + all_girls_combinations)
  (favorable_combinations : ℚ) / total_combinations = 59 / 63 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1430_143055


namespace NUMINAMATH_CALUDE_N_value_l1430_143085

theorem N_value : 
  let N := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  N = (1 + Real.sqrt 6 - Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_N_value_l1430_143085


namespace NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l1430_143038

def vowel_count : ℕ := 20
def word_length : ℕ := 5

theorem acme_vowel_soup_combinations :
  vowel_count ^ word_length = 3200000 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l1430_143038


namespace NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l1430_143035

theorem smallest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    (b = (4/3) * a) → (c = (5/3) * a) →
    (a + b + c = 180) →
    a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l1430_143035


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1430_143078

variable (a b : ℝ)

theorem problem_1 : 2 * a * (a^2 - 3*a - 1) = 2*a^3 - 6*a^2 - 2*a := by
  sorry

theorem problem_2 : (a^2*b - 2*a*b^2 + b^3) / b - (a + b)^2 = -4*a*b := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1430_143078


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l1430_143048

/-- The rate of drawing barbed wire per meter given a square field's area, gate widths, and total cost --/
theorem barbed_wire_rate (field_area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : 
  field_area = 3136 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 2331 →
  (total_cost / (4 * Real.sqrt field_area - num_gates * gate_width)) = 10.5 := by
  sorry

#check barbed_wire_rate

end NUMINAMATH_CALUDE_barbed_wire_rate_l1430_143048


namespace NUMINAMATH_CALUDE_quadratic_properties_l1430_143058

/-- Quadratic function f(x) = 2x² - 8x + 6 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 6

/-- Vertex form of f(x) -/
def vertex_form (x : ℝ) : ℝ := 2 * (x - 2)^2 - 2

theorem quadratic_properties :
  (∀ x, f x = vertex_form x) ∧
  f 2 = -2 ∧
  (∀ x, f x = f (4 - x)) ∧
  f 1 = 0 ∧
  f 3 = 0 ∧
  f 0 = 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1430_143058


namespace NUMINAMATH_CALUDE_pencil_box_cost_is_280_l1430_143077

/-- Represents the school's purchase of pencils and markers -/
structure SchoolPurchase where
  pencil_cartons : ℕ
  boxes_per_pencil_carton : ℕ
  marker_cartons : ℕ
  boxes_per_marker_carton : ℕ
  marker_carton_cost : ℚ
  total_spent : ℚ

/-- Calculates the cost of each box of pencils -/
def pencil_box_cost (purchase : SchoolPurchase) : ℚ :=
  (purchase.total_spent - purchase.marker_cartons * purchase.marker_carton_cost) /
  (purchase.pencil_cartons * purchase.boxes_per_pencil_carton)

/-- Theorem stating that for the given purchase, each box of pencils costs $2.80 -/
theorem pencil_box_cost_is_280 (purchase : SchoolPurchase) 
  (h1 : purchase.pencil_cartons = 20)
  (h2 : purchase.boxes_per_pencil_carton = 10)
  (h3 : purchase.marker_cartons = 10)
  (h4 : purchase.boxes_per_marker_carton = 5)
  (h5 : purchase.marker_carton_cost = 4)
  (h6 : purchase.total_spent = 600) :
  pencil_box_cost purchase = 280 / 100 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_cost_is_280_l1430_143077


namespace NUMINAMATH_CALUDE_inequality_preservation_l1430_143041

theorem inequality_preservation (a b : ℝ) : a < b → -2 + 2*a < -2 + 2*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1430_143041


namespace NUMINAMATH_CALUDE_sphere_water_volume_l1430_143088

def hemisphere_volume : ℝ := 4
def num_hemispheres : ℕ := 2749

theorem sphere_water_volume :
  let total_volume := (num_hemispheres : ℝ) * hemisphere_volume
  total_volume = 10996 := by sorry

end NUMINAMATH_CALUDE_sphere_water_volume_l1430_143088


namespace NUMINAMATH_CALUDE_max_items_for_alex_washing_l1430_143015

/-- Represents a washing machine with its characteristics and items to wash -/
structure WashingMachine where
  total_items : ℕ
  cycle_duration : ℕ  -- in minutes
  total_wash_time : ℕ  -- in minutes

/-- Calculates the maximum number of items that can be washed per cycle -/
def max_items_per_cycle (wm : WashingMachine) : ℕ :=
  wm.total_items / (wm.total_wash_time / wm.cycle_duration)

/-- Theorem stating the maximum number of items per cycle for the given washing machine -/
theorem max_items_for_alex_washing (wm : WashingMachine) 
  (h1 : wm.total_items = 60)
  (h2 : wm.cycle_duration = 45)
  (h3 : wm.total_wash_time = 180) :
  max_items_per_cycle wm = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_items_for_alex_washing_l1430_143015


namespace NUMINAMATH_CALUDE_divisible_by_25_l1430_143049

theorem divisible_by_25 (n : ℕ) : ∃ k : ℤ, (2^(n+2) * 3^n + 5*n - 4 : ℤ) = 25 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_25_l1430_143049


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1430_143000

theorem system_of_inequalities (x : ℝ) : (x + 2 > 3) ∧ (2*x - 1 < 5) → 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1430_143000


namespace NUMINAMATH_CALUDE_fraction_simplification_l1430_143008

theorem fraction_simplification :
  (3/7 + 5/8) / (5/12 + 2/15) = 295/154 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1430_143008


namespace NUMINAMATH_CALUDE_kebul_family_children_l1430_143089

/-- Represents a family with children -/
structure Family where
  total_members : ℕ
  father_age : ℕ
  average_age : ℚ
  average_age_without_father : ℚ

/-- Calculates the number of children in a family -/
def number_of_children (f : Family) : ℕ :=
  f.total_members - 2

/-- Theorem stating the number of children in the Kebul family -/
theorem kebul_family_children (f : Family) 
  (h1 : f.average_age = 18)
  (h2 : f.father_age = 38)
  (h3 : f.average_age_without_father = 14) :
  number_of_children f = 4 := by
  sorry

#eval number_of_children { total_members := 6, father_age := 38, average_age := 18, average_age_without_father := 14 }

end NUMINAMATH_CALUDE_kebul_family_children_l1430_143089


namespace NUMINAMATH_CALUDE_valid_parameterizations_l1430_143003

/-- The line equation y = -3x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

/-- A point is on the line if it satisfies the line equation -/
def point_on_line (p : ℝ × ℝ) : Prop :=
  line_equation p.1 p.2

/-- The direction vector is valid if it's parallel to (1, -3) -/
def valid_direction (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, -3*k)

/-- A parameterization is valid if its point is on the line and its direction is valid -/
def valid_parameterization (p v : ℝ × ℝ) : Prop :=
  point_on_line p ∧ valid_direction v

theorem valid_parameterizations :
  valid_parameterization (4/3, 0) (1, -3) ∧
  valid_parameterization (-2, 10) (-3, 9) :=
by sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l1430_143003


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1430_143066

theorem inequality_solution_set (x : ℝ) : 
  (x * (1 - 3 * x) > 0) ↔ (x > 0 ∧ x < 1/3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1430_143066


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1430_143084

/-- A quadratic function y = (x + m - 3)(x - m) + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x + m - 3) * (x - m) + 3

theorem quadratic_function_property (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < x₂ →
  f m x₁ = y₁ →
  f m x₂ = y₂ →
  x₁ + x₂ < 3 →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1430_143084


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l1430_143017

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two convex polygons in a plane -/
structure TwoPolygonConfig where
  Q₁ : ConvexPolygon
  Q₂ : ConvexPolygon
  same_plane : Bool
  m₁_le_m₂ : Q₁.sides ≤ Q₂.sides
  share_at_most_one_vertex : Bool
  share_no_sides : Bool

/-- The maximum number of intersections between two convex polygons -/
def max_intersections (config : TwoPolygonConfig) : ℕ := 
  config.Q₁.sides * config.Q₂.sides

/-- Theorem: The maximum number of intersections between two convex polygons
    under the given conditions is the product of their number of sides -/
theorem max_intersections_theorem (config : TwoPolygonConfig) : 
  max_intersections config = config.Q₁.sides * config.Q₂.sides := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l1430_143017
