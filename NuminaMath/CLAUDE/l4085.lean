import Mathlib

namespace NUMINAMATH_CALUDE_pentagon_from_equal_segments_l4085_408573

theorem pentagon_from_equal_segments (segment_length : Real) 
  (h1 : segment_length = 2 / 5)
  (h2 : 5 * segment_length = 2) : 
  4 * segment_length > segment_length := by
  sorry

end NUMINAMATH_CALUDE_pentagon_from_equal_segments_l4085_408573


namespace NUMINAMATH_CALUDE_hyperbola_foci_l4085_408585

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- The coordinates of a focus of the hyperbola -/
def focus_coordinate : ℝ × ℝ := (3, 0)

/-- Theorem: The coordinates of the foci of the hyperbola x^2/4 - y^2/5 = 1 are (±3, 0) -/
theorem hyperbola_foci :
  (∀ x y, hyperbola_equation x y → 
    (x = focus_coordinate.1 ∧ y = focus_coordinate.2) ∨ 
    (x = -focus_coordinate.1 ∧ y = focus_coordinate.2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l4085_408585


namespace NUMINAMATH_CALUDE_sixth_score_for_mean_90_l4085_408576

def quiz_scores : List ℕ := [85, 90, 88, 92, 95]

def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem sixth_score_for_mean_90 (x : ℕ) :
  arithmetic_mean (quiz_scores ++ [x]) = 90 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_sixth_score_for_mean_90_l4085_408576


namespace NUMINAMATH_CALUDE_triangle_area_l4085_408517

theorem triangle_area (a b : ℝ) (h1 : a = 8) (h2 : b = 7) : (1/2) * a * b = 28 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l4085_408517


namespace NUMINAMATH_CALUDE_fifth_bowler_score_l4085_408530

/-- A bowling team with 5 members and their scores -/
structure BowlingTeam where
  total_points : ℕ
  p1 : ℕ
  p2 : ℕ
  p3 : ℕ
  p4 : ℕ
  p5 : ℕ

/-- The conditions of the bowling team's scores -/
def validBowlingTeam (team : BowlingTeam) : Prop :=
  team.total_points = 2000 ∧
  team.p1 = team.p2 / 4 ∧
  team.p2 = team.p3 * 5 / 3 ∧
  team.p3 ≤ 500 ∧
  team.p3 = team.p4 * 3 / 5 ∧
  team.p4 = team.p5 * 9 / 10 ∧
  team.p1 + team.p2 + team.p3 + team.p4 + team.p5 = team.total_points

theorem fifth_bowler_score (team : BowlingTeam) :
  validBowlingTeam team → team.p5 = 561 := by
  sorry

end NUMINAMATH_CALUDE_fifth_bowler_score_l4085_408530


namespace NUMINAMATH_CALUDE_inequality_abc_l4085_408555

theorem inequality_abc (a b c : ℝ) (ha : a = Real.log 2.1) (hb : b = Real.exp 0.1) (hc : c = 1.1) :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_l4085_408555


namespace NUMINAMATH_CALUDE_circle_radius_with_parabolas_l4085_408596

/-- A parabola with equation y = 4x^2 -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- A line at 45° angle to the x-axis -/
def line_45_deg (x : ℝ) : ℝ := x

/-- The number of parabolas arranged around the circle -/
def num_parabolas : ℕ := 8

/-- Theorem stating that the radius of the circle is 1/16 under given conditions -/
theorem circle_radius_with_parabolas :
  ∀ (r : ℝ),
  (∃ (x : ℝ), parabola x + r = line_45_deg x) →  -- Parabola is tangent to 45° line
  (num_parabolas = 8) →                          -- Eight parabolas
  (r > 0) →                                      -- Radius is positive
  (r = 1 / 16) :=                                -- Radius is 1/16
by sorry

end NUMINAMATH_CALUDE_circle_radius_with_parabolas_l4085_408596


namespace NUMINAMATH_CALUDE_lars_baking_hours_l4085_408581

/-- The number of loaves of bread Lars can bake per hour -/
def loaves_per_hour : ℕ := 10

/-- The number of baguettes Lars can bake per hour -/
def baguettes_per_hour : ℕ := 15

/-- The total number of breads Lars makes -/
def total_breads : ℕ := 150

/-- The number of hours Lars bakes each day -/
def baking_hours : ℕ := 6

theorem lars_baking_hours :
  loaves_per_hour * baking_hours + baguettes_per_hour * baking_hours = total_breads :=
sorry

end NUMINAMATH_CALUDE_lars_baking_hours_l4085_408581


namespace NUMINAMATH_CALUDE_square_field_area_l4085_408543

/-- Proves that a square field with given conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 1.30 →
  total_cost = 865.80 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    (4 * side_length - gate_width * num_gates) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l4085_408543


namespace NUMINAMATH_CALUDE_insects_in_lab_l4085_408552

/-- The number of insects in a laboratory given the total number of legs --/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem: The number of insects in the laboratory is 5 --/
theorem insects_in_lab : number_of_insects 30 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_insects_in_lab_l4085_408552


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l4085_408547

/-- Waiter's tip calculation problem -/
theorem waiter_tip_calculation
  (total_customers : ℕ)
  (non_tipping_customers : ℕ)
  (total_tip_amount : ℕ)
  (h1 : total_customers = 9)
  (h2 : non_tipping_customers = 5)
  (h3 : total_tip_amount = 32) :
  total_tip_amount / (total_customers - non_tipping_customers) = 8 := by
  sorry

#check waiter_tip_calculation

end NUMINAMATH_CALUDE_waiter_tip_calculation_l4085_408547


namespace NUMINAMATH_CALUDE_square_difference_division_eleven_l4085_408592

theorem square_difference_division_eleven : (121^2 - 110^2) / 11 = 231 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_eleven_l4085_408592


namespace NUMINAMATH_CALUDE_square_field_division_l4085_408540

/-- Represents a square field with side length and division properties -/
structure SquareField where
  side_length : ℝ
  division_fence_length : ℝ

/-- Theorem: A square field of side 33m can be divided into three equal areas with at most 54m of fencing -/
theorem square_field_division (field : SquareField) 
  (h1 : field.side_length = 33) 
  (h2 : field.division_fence_length ≤ 54) : 
  ∃ (area_1 area_2 area_3 : ℝ), 
    area_1 = area_2 ∧ 
    area_2 = area_3 ∧ 
    area_1 + area_2 + area_3 = field.side_length * field.side_length := by
  sorry

end NUMINAMATH_CALUDE_square_field_division_l4085_408540


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l4085_408593

def S (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4*n - 3) + if n > 1 then S (n-1) else 0

theorem sequence_sum_problem : S 15 + S 22 - S 31 = -76 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l4085_408593


namespace NUMINAMATH_CALUDE_pet_store_birds_l4085_408551

/-- Calculates the total number of birds in a pet store with the given conditions -/
def totalBirds (totalCages : Nat) (emptyCages : Nat) (initialParrots : Nat) (initialParakeets : Nat) : Nat :=
  let nonEmptyCages := totalCages - emptyCages
  let parrotSum := nonEmptyCages * (2 * initialParrots + (nonEmptyCages - 1)) / 2
  let parakeetSum := nonEmptyCages * (2 * initialParakeets + 2 * (nonEmptyCages - 1)) / 2
  parrotSum + parakeetSum

/-- Theorem stating that the total number of birds in the pet store is 399 -/
theorem pet_store_birds : totalBirds 17 3 2 7 = 399 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l4085_408551


namespace NUMINAMATH_CALUDE_clock_angle_at_3_25_l4085_408542

/-- The angle of the minute hand on a clock face at a given number of minutes past the hour -/
def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

/-- The angle of the hour hand on a clock face at a given hour and minute -/
def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  hour * 30 + minute * 0.5

/-- The angle between the hour hand and minute hand on a clock face -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  |hour_hand_angle hour minute - minute_hand_angle minute|

theorem clock_angle_at_3_25 :
  clock_angle 3 25 = 47.5 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_25_l4085_408542


namespace NUMINAMATH_CALUDE_cos_B_value_triangle_area_l4085_408507

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition_1 (t : Triangle) : Prop :=
  (Real.sin t.B) ^ 2 = (Real.sin t.A) * (Real.sin t.C) ∧ t.a = Real.sqrt 2 * t.b

def satisfies_condition_2 (t : Triangle) : Prop :=
  Real.cos t.B = 3 / 4 ∧ t.a = 2

-- Define the theorems to prove
theorem cos_B_value (t : Triangle) (h : satisfies_condition_1 t) :
  Real.cos t.B = 3 / 4 := by sorry

theorem triangle_area (t : Triangle) (h : satisfies_condition_2 t) :
  let area := 1 / 2 * t.a * t.c * Real.sin t.B
  area = Real.sqrt 7 / 4 ∨ area = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_cos_B_value_triangle_area_l4085_408507


namespace NUMINAMATH_CALUDE_average_of_remaining_quantities_l4085_408575

theorem average_of_remaining_quantities
  (total_count : ℕ)
  (subset_count : ℕ)
  (total_average : ℚ)
  (subset_average : ℚ)
  (h1 : total_count = 6)
  (h2 : subset_count = 4)
  (h3 : total_average = 8)
  (h4 : subset_average = 5) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 14 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_quantities_l4085_408575


namespace NUMINAMATH_CALUDE_alices_favorite_number_l4085_408512

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem alices_favorite_number :
  ∃! n : ℕ, 100 < n ∧ n < 150 ∧ 
  13 ∣ n ∧ ¬(2 ∣ n) ∧ 
  4 ∣ sum_of_digits n ∧
  n = 143 :=
sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l4085_408512


namespace NUMINAMATH_CALUDE_range_of_m_l4085_408550

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ∈ Set.range f) →  -- domain of f is [-2, 2]
  (∀ x y, x ∈ Set.Icc (-2 : ℝ) 2 → y ∈ Set.Icc (-2 : ℝ) 2 → x < y → f x < f y) →  -- f is increasing on [-2, 2]
  f (1 - m) < f m →  -- given condition
  m ∈ Set.Ioo (1/2 : ℝ) 2 :=  -- conclusion: m is in the open interval (1/2, 2]
by sorry

end NUMINAMATH_CALUDE_range_of_m_l4085_408550


namespace NUMINAMATH_CALUDE_chess_tournament_wins_l4085_408500

theorem chess_tournament_wins (total_games : ℕ) (total_points : ℚ)
  (h1 : total_games = 20)
  (h2 : total_points = 12.5) :
  ∃ (wins losses draws : ℕ),
    wins + losses + draws = total_games ∧
    wins - losses = 5 ∧
    wins + draws / 2 = total_points := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_wins_l4085_408500


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l4085_408580

/-- Number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 60 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l4085_408580


namespace NUMINAMATH_CALUDE_smallest_opposite_l4085_408588

theorem smallest_opposite (a b c d : ℝ) (ha : a = -1) (hb : b = 0) (hc : c = Real.sqrt 5) (hd : d = -1/3) :
  min (-a) (min (-b) (min (-c) (-d))) = -c :=
by sorry

end NUMINAMATH_CALUDE_smallest_opposite_l4085_408588


namespace NUMINAMATH_CALUDE_purely_imaginary_m_eq_3_second_quadrant_m_range_l4085_408565

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := (m^2 - m - 6 : ℝ) + (m^2 + 5*m + 6 : ℝ) * Complex.I

/-- Theorem: If z is purely imaginary, then m = 3 -/
theorem purely_imaginary_m_eq_3 :
  (∀ m : ℝ, z m = Complex.I * Complex.im (z m)) → (∃ m : ℝ, m = 3) :=
sorry

/-- Theorem: If z is in the second quadrant, then -2 < m < 3 -/
theorem second_quadrant_m_range :
  (∀ m : ℝ, Complex.re (z m) < 0 ∧ Complex.im (z m) > 0) → (∀ m : ℝ, -2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_m_eq_3_second_quadrant_m_range_l4085_408565


namespace NUMINAMATH_CALUDE_min_value_of_product_l4085_408544

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem min_value_of_product (a b c : ℝ) (x₁ x₂ x₃ : ℝ) :
  a ≠ 0 →
  f a b c (-1) = 0 →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ (x + 1)^2 / 4) →
  x₁ ∈ Set.Ioo 0 2 →
  x₂ ∈ Set.Ioo 0 2 →
  x₃ ∈ Set.Ioo 0 2 →
  1 / x₁ + 1 / x₂ + 1 / x₃ = 3 →
  (f a b c x₁) * (f a b c x₂) * (f a b c x₃) ≥ 1 :=
by sorry

#check min_value_of_product

end NUMINAMATH_CALUDE_min_value_of_product_l4085_408544


namespace NUMINAMATH_CALUDE_overlapping_area_of_strips_l4085_408513

theorem overlapping_area_of_strips (total_length width : ℝ) 
  (left_length right_length : ℝ) (left_only_area right_only_area : ℝ) :
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_length + right_length = total_length →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ),
    overlap_area = (left_length * width - left_only_area) ∧
    overlap_area = (right_length * width - right_only_area) ∧
    overlap_area = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_overlapping_area_of_strips_l4085_408513


namespace NUMINAMATH_CALUDE_snake_length_ratio_l4085_408539

/-- The length of the garden snake in inches -/
def garden_snake_length : ℕ := 10

/-- The length of the boa constrictor in inches -/
def boa_constrictor_length : ℕ := 70

/-- The ratio of the boa constrictor's length to the garden snake's length -/
def length_ratio : ℚ := boa_constrictor_length / garden_snake_length

theorem snake_length_ratio :
  length_ratio = 7 := by sorry

end NUMINAMATH_CALUDE_snake_length_ratio_l4085_408539


namespace NUMINAMATH_CALUDE_polynomial_roots_l4085_408598

theorem polynomial_roots : 
  let p : ℝ → ℝ := λ x => 3*x^4 + 17*x^3 - 32*x^2 - 12*x
  (p 0 = 0) ∧ 
  (p (-1/2) = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l4085_408598


namespace NUMINAMATH_CALUDE_solve_proportion_l4085_408519

theorem solve_proportion (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_proportion_l4085_408519


namespace NUMINAMATH_CALUDE_largest_integer_with_3_digit_square_base7_l4085_408508

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 7) :: aux (m / 7)
    aux n |>.reverse

theorem largest_integer_with_3_digit_square_base7 :
  (M ^ 2 ≥ 7^2) ∧ 
  (M ^ 2 < 7^3) ∧ 
  (∀ n : ℕ, n > M → n ^ 2 ≥ 7^3) ∧
  (toBase7 M = [6, 6]) := by
  sorry

#eval M
#eval toBase7 M

end NUMINAMATH_CALUDE_largest_integer_with_3_digit_square_base7_l4085_408508


namespace NUMINAMATH_CALUDE_basketball_team_grouping_probability_l4085_408586

theorem basketball_team_grouping_probability :
  let total_teams : ℕ := 7
  let group_size_1 : ℕ := 3
  let group_size_2 : ℕ := 4
  let specific_teams : ℕ := 2
  
  let total_arrangements : ℕ := (Nat.choose total_teams group_size_1) * (Nat.choose group_size_1 group_size_1) +
                                (Nat.choose total_teams group_size_2) * (Nat.choose group_size_2 group_size_2)
  
  let favorable_arrangements : ℕ := (Nat.choose specific_teams specific_teams) *
                                    ((Nat.choose (total_teams - specific_teams) (group_size_1 - specific_teams)) +
                                     (Nat.choose (total_teams - specific_teams) (group_size_2 - specific_teams))) *
                                    (Nat.factorial specific_teams)
  
  (favorable_arrangements : ℚ) / total_arrangements = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_basketball_team_grouping_probability_l4085_408586


namespace NUMINAMATH_CALUDE_base_13_conversion_l4085_408502

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Base13Digit to its numerical value -/
def base13DigitToNat (d : Base13Digit) : ℕ :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Converts a two-digit number in base 13 to its decimal (base 10) equivalent -/
def base13ToDecimal (d1 d2 : Base13Digit) : ℕ :=
  13 * (base13DigitToNat d1) + (base13DigitToNat d2)

theorem base_13_conversion :
  base13ToDecimal Base13Digit.C Base13Digit.D1 = 157 := by
  sorry

end NUMINAMATH_CALUDE_base_13_conversion_l4085_408502


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_expression_l4085_408563

theorem smallest_prime_dividing_expression : 
  ∃ (a : ℕ), a > 1 ∧ 179 ∣ (a^89 - 1) / (a - 1) ∧
  ∀ (p : ℕ), p > 100 → p < 179 → Prime p → 
    ¬(∃ (b : ℕ), b > 1 ∧ p ∣ (b^89 - 1) / (b - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_expression_l4085_408563


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l4085_408594

theorem triangle_side_sum_max (a c : ℝ) : 
  let B : ℝ := π / 3
  let b : ℝ := 2
  0 < a ∧ 0 < c ∧ 
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  a + c ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l4085_408594


namespace NUMINAMATH_CALUDE_reflect_A_x_axis_l4085_408514

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-3, 2)

theorem reflect_A_x_axis : reflect_x A = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_A_x_axis_l4085_408514


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l4085_408515

theorem p_necessary_not_sufficient (p q : Prop) : 
  (∀ (h : p ∧ q), p) ∧ 
  (∃ (h : p), ¬(p ∧ q)) := by
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l4085_408515


namespace NUMINAMATH_CALUDE_average_sequence_problem_l4085_408564

theorem average_sequence_problem (a b c d e : ℝ) : 
  a = 8 ∧ 
  d = 26 ∧
  b = (a + c) / 2 ∧
  c = (b + d) / 2 ∧
  d = (c + e) / 2 
  → e = 32 := by sorry

end NUMINAMATH_CALUDE_average_sequence_problem_l4085_408564


namespace NUMINAMATH_CALUDE_parallel_lines_set_l4085_408520

-- Define the plane
variable (Plane : Type)

-- Define points on the plane
variable (D E P : Plane)

-- Define the distance between two points
variable (distance : Plane → Plane → ℝ)

-- Define the area of a triangle given three points
variable (triangle_area : Plane → Plane → Plane → ℝ)

-- Define a set of points
variable (T : Set Plane)

-- State the theorem
theorem parallel_lines_set (h_distinct : D ≠ E) :
  T = {P | triangle_area D E P = 0.5} →
  ∃ (l₁ l₂ : Set Plane), 
    (∀ X ∈ l₁, ∀ Y ∈ l₂, distance X Y = 2 / distance D E) ∧
    T = l₁ ∪ l₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_set_l4085_408520


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l4085_408589

/-- Given a bag of popping corn with white and blue kernels, calculate the probability
    that a randomly selected kernel that popped was white. -/
theorem popped_kernel_probability (total_kernels : ℝ) (h_total_pos : 0 < total_kernels) : 
  let white_ratio : ℝ := 3/4
  let blue_ratio : ℝ := 1/4
  let white_pop_prob : ℝ := 3/5
  let blue_pop_prob : ℝ := 3/4
  let white_kernels := white_ratio * total_kernels
  let blue_kernels := blue_ratio * total_kernels
  let popped_white := white_pop_prob * white_kernels
  let popped_blue := blue_pop_prob * blue_kernels
  let total_popped := popped_white + popped_blue
  (popped_white / total_popped) = 12/13 :=
by sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l4085_408589


namespace NUMINAMATH_CALUDE_problem_solution_l4085_408535

theorem problem_solution :
  (∀ x : ℝ, (x + 1) * (x - 3) > (x + 2) * (x - 4)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * (x + y) = 36 →
    x * y ≤ 81 ∧
    (x * y = 81 ↔ x = 9 ∧ y = 9)) :=
by sorry


end NUMINAMATH_CALUDE_problem_solution_l4085_408535


namespace NUMINAMATH_CALUDE_ending_number_proof_l4085_408583

theorem ending_number_proof (n : ℕ) : 
  (∃ (evens : Finset ℕ), evens.card = 35 ∧ 
    (∀ x ∈ evens, 25 < x ∧ x ≤ n ∧ Even x) ∧
    (∀ y, 25 < y ∧ y ≤ n ∧ Even y → y ∈ evens)) ↔ 
  n = 94 :=
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l4085_408583


namespace NUMINAMATH_CALUDE_rocket_launch_l4085_408528

/-- Rocket launch problem -/
theorem rocket_launch (a : ℝ) (g : ℝ) (t : ℝ) (h_object : ℝ) : 
  a = 20 → g = 10 → t = 40 → h_object = 45000 →
  let v₀ : ℝ := a * t
  let h₀ : ℝ := (1/2) * a * t^2
  let t_max : ℝ := v₀ / g
  let h_max : ℝ := h₀ + v₀ * t_max - (1/2) * g * t_max^2
  h_max = 48000 ∧ h_max > h_object :=
by sorry

end NUMINAMATH_CALUDE_rocket_launch_l4085_408528


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l4085_408566

theorem rational_inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4*x + 13) ≤ 0 ↔ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l4085_408566


namespace NUMINAMATH_CALUDE_football_field_area_l4085_408582

theorem football_field_area (A : ℝ) 
  (h1 : 500 / 3500 = 1200 / A) : A = 8400 := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_l4085_408582


namespace NUMINAMATH_CALUDE_computer_arrangements_l4085_408554

theorem computer_arrangements : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_computer_arrangements_l4085_408554


namespace NUMINAMATH_CALUDE_ellipse_t_squared_range_l4085_408524

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the condition for points A, B, and P
def condition (A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  (A.1, A.2) + (B.1, B.2) = t • (P.1, P.2)

-- Define the inequality condition
def inequality (A B P : ℝ × ℝ) : Prop :=
  ‖(P.1 - A.1, P.2 - A.2) - (P.1 - B.1, P.2 - B.2)‖ < Real.sqrt 3

-- Theorem statement
theorem ellipse_t_squared_range :
  ∀ (A B P : ℝ × ℝ) (t : ℝ),
    ellipse A.1 A.2 → ellipse B.1 B.2 → ellipse P.1 P.2 →
    condition A B P t → inequality A B P →
    20 - Real.sqrt 283 < t^2 ∧ t^2 < 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_t_squared_range_l4085_408524


namespace NUMINAMATH_CALUDE_inequality_condition_l4085_408558

theorem inequality_condition (a b : ℝ) (h1 : a * b ≠ 0) :
  (a < b ∧ b < 0) → (1 / a^2 > 1 / b^2) ∧
  ¬(∀ a b : ℝ, a * b ≠ 0 → (1 / a^2 > 1 / b^2) → (a < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l4085_408558


namespace NUMINAMATH_CALUDE_probability_first_two_trials_l4085_408591

-- Define the probability of event A
def P_A : ℝ := 0.7

-- Define the number of trials
def num_trials : ℕ := 4

-- Define the probability of event A occurring exactly in the first two trials
def P_first_two : ℝ := P_A * P_A * (1 - P_A) * (1 - P_A)

-- Theorem statement
theorem probability_first_two_trials : P_first_two = 0.0441 := by
  sorry

end NUMINAMATH_CALUDE_probability_first_two_trials_l4085_408591


namespace NUMINAMATH_CALUDE_equation_solution_l4085_408523

theorem equation_solution : ∃! x : ℝ, (81 : ℝ)^(x - 2) / (9 : ℝ)^(x - 2) = (27 : ℝ)^(3*x + 2) ∧ x = -10/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4085_408523


namespace NUMINAMATH_CALUDE_stone_pile_total_l4085_408527

/-- Represents the number of stones in each pile -/
structure StonePiles where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- The conditions of the stone pile problem -/
def stone_pile_conditions (piles : StonePiles) : Prop :=
  piles.fifth = 6 * piles.third ∧
  piles.second = 2 * (piles.third + piles.fifth) ∧
  piles.first * 3 = piles.fifth ∧
  piles.first + 10 = piles.fourth ∧
  2 * piles.fourth = piles.second

/-- The theorem stating that under the given conditions, the total number of stones is 60 -/
theorem stone_pile_total (piles : StonePiles) 
  (h : stone_pile_conditions piles) : 
  piles.first + piles.second + piles.third + piles.fourth + piles.fifth = 60 := by
  sorry


end NUMINAMATH_CALUDE_stone_pile_total_l4085_408527


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l4085_408534

/-- Calculates the selling price of a cricket bat given the profit and profit percentage -/
theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) :
  profit = 225 →
  profit_percentage = 36 →
  ∃ (cost_price selling_price : ℝ),
    cost_price = profit * 100 / profit_percentage ∧
    selling_price = cost_price + profit ∧
    selling_price = 850 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l4085_408534


namespace NUMINAMATH_CALUDE_fraction_equality_l4085_408504

theorem fraction_equality (x y : ℚ) (hx : x = 4/6) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4085_408504


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l4085_408503

theorem ellipse_hyperbola_ab_value (a b : ℝ) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a*b| = 2 * Real.sqrt 111 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l4085_408503


namespace NUMINAMATH_CALUDE_AR_equals_six_l4085_408584

-- Define the triangle and points
variable (A B C R P Q : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (acute_triangle : AcuteTriangle A B C)
variable (R_on_perpendicular_bisector : OnPerpendicularBisector R A C)
variable (CA_bisects_BAR : AngleBisector C A (B, R))
variable (Q_intersection : OnLine Q A C ∧ OnLine Q B R)
variable (P_on_circumcircle : OnCircumcircle P A R C)
variable (P_on_AB : SegmentND P A B)
variable (AP_length : dist A P = 1)
variable (PB_length : dist P B = 5)
variable (AQ_length : dist A Q = 2)

-- State the theorem
theorem AR_equals_six : dist A R = 6 := by sorry

end NUMINAMATH_CALUDE_AR_equals_six_l4085_408584


namespace NUMINAMATH_CALUDE_expression_evaluation_l4085_408553

theorem expression_evaluation (x y : ℚ) (hx : x = 1/3) (hy : y = -1/2) :
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4085_408553


namespace NUMINAMATH_CALUDE_prism_problem_l4085_408571

theorem prism_problem (x : ℝ) (d : ℝ) : 
  x > 0 → 
  let a := Real.log x / Real.log 5
  let b := Real.log x / Real.log 7
  let c := Real.log x / Real.log 9
  let surface_area := 2 * (a * b + b * c + c * a)
  let volume := a * b * c
  surface_area * (1/3 * volume) = 54 →
  d = Real.sqrt (a^2 + b^2 + c^2) →
  x = 216 ∧ d = 7 := by
  sorry

#check prism_problem

end NUMINAMATH_CALUDE_prism_problem_l4085_408571


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l4085_408536

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l4085_408536


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l4085_408541

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l4085_408541


namespace NUMINAMATH_CALUDE_f_composition_one_ninth_l4085_408568

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_one_ninth : f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_ninth_l4085_408568


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4085_408557

theorem complex_equation_solution (x y : ℝ) : 
  (2 * x - y + 1 : ℂ) + (y - 2 : ℂ) * I = 0 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4085_408557


namespace NUMINAMATH_CALUDE_total_people_present_l4085_408537

/-- Represents the number of associate professors -/
def associate_profs : ℕ := sorry

/-- Represents the number of assistant professors -/
def assistant_profs : ℕ := sorry

/-- Total number of pencils brought to the meeting -/
def total_pencils : ℕ := 10

/-- Total number of charts brought to the meeting -/
def total_charts : ℕ := 14

/-- Theorem stating the total number of people present at the meeting -/
theorem total_people_present : associate_profs + assistant_profs = 8 :=
  sorry

end NUMINAMATH_CALUDE_total_people_present_l4085_408537


namespace NUMINAMATH_CALUDE_weekly_pie_sales_l4085_408538

/-- The number of pies sold daily by the restaurant -/
def daily_pies : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def total_pies : ℕ := daily_pies * days_in_week

theorem weekly_pie_sales : total_pies = 56 := by
  sorry

end NUMINAMATH_CALUDE_weekly_pie_sales_l4085_408538


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l4085_408595

theorem max_q_minus_r_for_1027 :
  ∀ q r : ℕ+, 
  1027 = 23 * q + r → 
  ∀ q' r' : ℕ+, 
  1027 = 23 * q' + r' → 
  q - r ≤ 29 ∧ ∃ q r : ℕ+, 1027 = 23 * q + r ∧ q - r = 29 :=
by sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l4085_408595


namespace NUMINAMATH_CALUDE_square_point_B_coordinates_l4085_408531

/-- A square in a 2D plane -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Predicate to check if a line is parallel to the x-axis -/
def parallelToXAxis (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : Prop :=
  p1.2 = p2.2

theorem square_point_B_coordinates :
  ∀ (s : Square),
    s.A = (1, -2) →
    s.C = (4, 1) →
    parallelToXAxis s.A s.B →
    s.B = (4, -2) := by
  sorry

end NUMINAMATH_CALUDE_square_point_B_coordinates_l4085_408531


namespace NUMINAMATH_CALUDE_mod_eight_difference_l4085_408574

theorem mod_eight_difference (n : ℕ) : (47^1824 - 25^1824) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_difference_l4085_408574


namespace NUMINAMATH_CALUDE_product_inequality_l4085_408577

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (1 + x + y^2) * (1 + y + x^2) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l4085_408577


namespace NUMINAMATH_CALUDE_dennis_purchase_cost_l4085_408545

/-- Calculates the total cost after discount for Dennis's purchase -/
def total_cost_after_discount (pants_price : ℝ) (socks_price : ℝ) (pants_quantity : ℕ) (socks_quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_before_discount := pants_price * pants_quantity + socks_price * socks_quantity
  total_before_discount * (1 - discount_rate)

/-- Proves that the total cost after discount for Dennis's purchase is $392 -/
theorem dennis_purchase_cost :
  total_cost_after_discount 110 60 4 2 0.3 = 392 := by
  sorry

end NUMINAMATH_CALUDE_dennis_purchase_cost_l4085_408545


namespace NUMINAMATH_CALUDE_function_properties_l4085_408511

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_eq : f ω (π/8) = f ω (5*π/8)) :
  (∃! (min max : ℝ), min ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    max ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    (∀ x ∈ Set.Ioo (π/8) (5*π/8), f ω x ≥ f ω min ∧ f ω x ≤ f ω max) →
    ω = 4) ∧
  (∃! (z₁ z₂ : ℝ), z₁ ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    z₂ ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    f ω z₁ = 0 ∧ f ω z₂ = 0 ∧ 
    (∀ x ∈ Set.Ioo (π/8) (5*π/8), f ω x = 0 → x = z₁ ∨ x = z₂) →
    ω = 10/3 ∨ ω = 4 ∨ ω = 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4085_408511


namespace NUMINAMATH_CALUDE_girls_in_class_correct_number_of_girls_l4085_408509

theorem girls_in_class (total_books : ℕ) (boys : ℕ) (girls_books : ℕ) : ℕ :=
  let boys_books := total_books - girls_books
  let books_per_student := boys_books / boys
  girls_books / books_per_student

theorem correct_number_of_girls :
  girls_in_class 375 10 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_correct_number_of_girls_l4085_408509


namespace NUMINAMATH_CALUDE_max_product_constrained_max_value_is_three_l4085_408526

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/3 + b/4 = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x/3 + y/4 = 1 → x*y ≤ a*b := by
  sorry

theorem max_value_is_three (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/3 + b/4 = 1) :
  a*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_value_is_three_l4085_408526


namespace NUMINAMATH_CALUDE_optimal_arrangement_l4085_408599

-- Define the harvester types
inductive HarvesterType
| A
| B

-- Define the properties of harvesters
def harvest_rate (t : HarvesterType) : ℕ :=
  match t with
  | HarvesterType.A => 5
  | HarvesterType.B => 3

def fee_per_hectare (t : HarvesterType) : ℕ :=
  match t with
  | HarvesterType.A => 50
  | HarvesterType.B => 45

-- Define the problem constraints
def total_harvesters : ℕ := 12
def min_hectares_per_day : ℕ := 50

-- Define the optimization problem
def is_valid_arrangement (num_A : ℕ) : Prop :=
  num_A ≤ total_harvesters ∧
  num_A * harvest_rate HarvesterType.A + (total_harvesters - num_A) * harvest_rate HarvesterType.B ≥ min_hectares_per_day

def total_cost (num_A : ℕ) : ℕ :=
  num_A * harvest_rate HarvesterType.A * fee_per_hectare HarvesterType.A +
  (total_harvesters - num_A) * harvest_rate HarvesterType.B * fee_per_hectare HarvesterType.B

-- State the theorem
theorem optimal_arrangement :
  ∃ (num_A : ℕ), is_valid_arrangement num_A ∧
  (∀ (m : ℕ), is_valid_arrangement m → total_cost num_A ≤ total_cost m) ∧
  num_A = 7 ∧
  total_cost num_A = 2425 := by sorry

end NUMINAMATH_CALUDE_optimal_arrangement_l4085_408599


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l4085_408567

/-- Given two perpendicular lines and the foot of their perpendicular, prove m - n + p = 20 -/
theorem perpendicular_lines_intersection (m n p : ℝ) : 
  (∀ x y, m * x + 4 * y - 2 = 0 ∨ 2 * x - 5 * y + n = 0) →  -- Two lines
  (m * 2 = -4 * 5) →  -- Perpendicularity condition
  (m * 1 + 4 * p - 2 = 0) →  -- Foot satisfies first line equation
  (2 * 1 - 5 * p + n = 0) →  -- Foot satisfies second line equation
  m - n + p = 20 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l4085_408567


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l4085_408522

/-- Given two solutions with different alcohol concentrations, prove that mixing them in specific quantities results in a desired alcohol concentration. -/
theorem alcohol_mixture_proof (x_volume : ℝ) (y_volume : ℝ) (x_concentration : ℝ) (y_concentration : ℝ) (target_concentration : ℝ) :
  x_volume = 300 →
  y_volume = 200 →
  x_concentration = 0.1 →
  y_concentration = 0.3 →
  target_concentration = 0.18 →
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = target_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l4085_408522


namespace NUMINAMATH_CALUDE_max_value_sum_products_l4085_408570

theorem max_value_sum_products (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 200) : 
  ∃ (max : ℝ), max = 10000 ∧ ∀ (x y z w : ℝ), 
    0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w ∧ x + y + z + w = 200 →
    x * y + y * z + z * w ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_products_l4085_408570


namespace NUMINAMATH_CALUDE_product_of_primes_with_square_sum_l4085_408506

theorem product_of_primes_with_square_sum (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
  p₁^2 + p₂^2 + p₃^2 + p₄^2 = 476 →
  p₁ * p₂ * p₃ * p₄ = 1989 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_with_square_sum_l4085_408506


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l4085_408572

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 → 
  side * side = area → 
  perimeter = 4 * side → 
  perimeter = 60 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l4085_408572


namespace NUMINAMATH_CALUDE_vertical_shift_theorem_l4085_408533

theorem vertical_shift_theorem (f : ℝ → ℝ) :
  ∀ x y : ℝ, y = f x + 3 ↔ ∃ y₀ : ℝ, y₀ = f x ∧ y = y₀ + 3 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_theorem_l4085_408533


namespace NUMINAMATH_CALUDE_discount_calculation_l4085_408548

/-- Proves the true discount and the difference between claimed and true discount for a given discount scenario. -/
theorem discount_calculation (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.25 →
  additional_discount = 0.1 →
  claimed_discount = 0.4 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let true_discount := 1 - remaining_after_additional
  true_discount = 0.325 ∧ claimed_discount - true_discount = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l4085_408548


namespace NUMINAMATH_CALUDE_distance_to_city_l4085_408549

theorem distance_to_city (D : ℝ) 
  (h1 : D / 2 + D / 4 + 6 = D) : D = 24 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_city_l4085_408549


namespace NUMINAMATH_CALUDE_chess_club_problem_l4085_408546

theorem chess_club_problem (total_members chess_players checkers_players both_players : ℕ) 
  (h1 : total_members = 70)
  (h2 : chess_players = 45)
  (h3 : checkers_players = 38)
  (h4 : both_players = 25) :
  total_members - (chess_players + checkers_players - both_players) = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_problem_l4085_408546


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l4085_408529

theorem sum_of_a_and_b (a b : ℝ) (h1 : |a| = 10) (h2 : |b| = 7) (h3 : a > b) :
  a + b = 17 ∨ a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l4085_408529


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4085_408587

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4085_408587


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l4085_408562

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101_equals_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l4085_408562


namespace NUMINAMATH_CALUDE_least_number_divisible_by_53_and_71_l4085_408521

theorem least_number_divisible_by_53_and_71 (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((1357 + y) % 53 = 0 ∧ (1357 + y) % 71 = 0)) ∧ 
  (1357 + x) % 53 = 0 ∧ (1357 + x) % 71 = 0 → 
  x = 2406 := by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_53_and_71_l4085_408521


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4085_408556

theorem right_triangle_hypotenuse (x y : ℝ) :
  x > 0 ∧ y > 0 →
  (1/3) * π * x * y^2 = 800 * π →
  (1/3) * π * y * x^2 = 1920 * π →
  Real.sqrt (x^2 + y^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4085_408556


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l4085_408510

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions, the swimmer's speed in still water is 4 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
  (effectiveSpeed s true * 6 = 30) →   -- Downstream condition
  (effectiveSpeed s false * 6 = 18) →  -- Upstream condition
  s.swimmer = 4 := by
sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l4085_408510


namespace NUMINAMATH_CALUDE_remainder_of_sum_first_six_primes_div_seventh_prime_l4085_408532

-- Define the first seven prime numbers
def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

-- Define the sum of the first six primes
def sum_first_six_primes : Nat := (first_seven_primes.take 6).sum

-- Define the seventh prime
def seventh_prime : Nat := first_seven_primes[6]

-- Theorem statement
theorem remainder_of_sum_first_six_primes_div_seventh_prime :
  sum_first_six_primes % seventh_prime = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_first_six_primes_div_seventh_prime_l4085_408532


namespace NUMINAMATH_CALUDE_quadratic_minimum_property_l4085_408597

/-- Given a quadratic function f(x) = ax^2 + bx + 1 with minimum value f(1) = 0, prove that a - b = 3 -/
theorem quadratic_minimum_property (a b : ℝ) : 
  (∀ x, a*x^2 + b*x + 1 ≥ a + b + 1) ∧ (a + b + 1 = 0) → a - b = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_property_l4085_408597


namespace NUMINAMATH_CALUDE_parallel_lines_same_black_cells_l4085_408579

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a line in the grid (horizontal, vertical, or diagonal) -/
inductive Line
  | Horizontal : Nat → Line
  | Vertical : Nat → Line
  | LeftDiagonal : Nat → Line
  | RightDiagonal : Nat → Line

/-- The grid configuration -/
structure GridConfig where
  n : Nat
  blackCells : Set Cell

/-- Two lines are parallel -/
def areLinesParallel (l1 l2 : Line) : Prop :=
  match l1, l2 with
  | Line.Horizontal _, Line.Horizontal _ => true
  | Line.Vertical _, Line.Vertical _ => true
  | Line.LeftDiagonal _, Line.LeftDiagonal _ => true
  | Line.RightDiagonal _, Line.RightDiagonal _ => true
  | _, _ => false

/-- Count black cells in a line -/
def countBlackCells (g : GridConfig) (l : Line) : Nat :=
  sorry

/-- Main theorem -/
theorem parallel_lines_same_black_cells 
  (g : GridConfig) 
  (h : g.n ≥ 3) :
  ∃ l1 l2 : Line, 
    areLinesParallel l1 l2 ∧ 
    l1 ≠ l2 ∧ 
    countBlackCells g l1 = countBlackCells g l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_same_black_cells_l4085_408579


namespace NUMINAMATH_CALUDE_reflections_on_circumcircle_l4085_408560

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the orthocenter H
variable (H : EuclideanSpace ℝ (Fin 2))

-- Define the circumcircle
variable (circumcircle : Sphere (EuclideanSpace ℝ (Fin 2)))

-- Assumptions
variable (h_acute : IsAcute A B C)
variable (h_orthocenter : IsOrthocenter H A B C)
variable (h_circumcircle : IsCircumcircle circumcircle A B C)

-- Define the reflections of H with respect to the sides
def reflect_H_BC : EuclideanSpace ℝ (Fin 2) := sorry
def reflect_H_CA : EuclideanSpace ℝ (Fin 2) := sorry
def reflect_H_AB : EuclideanSpace ℝ (Fin 2) := sorry

-- Theorem statement
theorem reflections_on_circumcircle :
  circumcircle.mem reflect_H_BC ∧
  circumcircle.mem reflect_H_CA ∧
  circumcircle.mem reflect_H_AB :=
sorry

end NUMINAMATH_CALUDE_reflections_on_circumcircle_l4085_408560


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l4085_408578

theorem sqrt_six_div_sqrt_two_eq_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l4085_408578


namespace NUMINAMATH_CALUDE_race_champion_is_C_l4085_408516

-- Define the participants
inductive Participant : Type
| A : Participant
| B : Participant
| C : Participant
| D : Participant

-- Define the opinions
def xiaozhangs_opinion (champion : Participant) : Prop :=
  champion = Participant.A ∨ champion = Participant.B

def xiaowangs_opinion (champion : Participant) : Prop :=
  champion ≠ Participant.C

def xiaolis_opinion (champion : Participant) : Prop :=
  champion ≠ Participant.A ∧ champion ≠ Participant.B

-- Theorem statement
theorem race_champion_is_C :
  ∀ (champion : Participant),
    (xiaozhangs_opinion champion ∨ xiaowangs_opinion champion ∨ xiaolis_opinion champion) ∧
    (¬(xiaozhangs_opinion champion ∧ xiaowangs_opinion champion) ∧
     ¬(xiaozhangs_opinion champion ∧ xiaolis_opinion champion) ∧
     ¬(xiaowangs_opinion champion ∧ xiaolis_opinion champion)) →
    champion = Participant.C :=
by sorry

end NUMINAMATH_CALUDE_race_champion_is_C_l4085_408516


namespace NUMINAMATH_CALUDE_arithmetic_equation_l4085_408569

theorem arithmetic_equation : (26.3 * 12 * 20) / 3 + 125 = 2229 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l4085_408569


namespace NUMINAMATH_CALUDE_calculation_proof_l4085_408561

theorem calculation_proof : 
  41 * ((2 + 2/7) - (3 + 3/5)) / ((3 + 1/5) + (2 + 1/4)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4085_408561


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4085_408505

theorem trigonometric_identity : 
  Real.sin (30 * π / 180) + Real.cos (120 * π / 180) + 2 * Real.cos (45 * π / 180) - Real.sqrt 3 * Real.tan (30 * π / 180) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4085_408505


namespace NUMINAMATH_CALUDE_hexagon_enclosure_l4085_408559

theorem hexagon_enclosure (m n : ℕ) (h1 : m = 6) (h2 : m + 1 = 7) : 
  (3 * (360 / n) = 2 * (180 - (m - 2) * 180 / m)) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_enclosure_l4085_408559


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4085_408501

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem stating the property of the arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.a 2 + seq.S 3 = 4)
  (h2 : seq.a 3 + seq.S 5 = 12) :
  seq.a 4 + seq.S 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4085_408501


namespace NUMINAMATH_CALUDE_expression_evaluation_l4085_408525

theorem expression_evaluation (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4085_408525


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l4085_408518

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l4085_408518


namespace NUMINAMATH_CALUDE_triangle_similarity_l4085_408590

theorem triangle_similarity (DC CB AD : ℝ) (h1 : DC = 9) (h2 : CB = 6) 
  (h3 : AD > 0) (h4 : ∃ (AB : ℝ), AB = (1/3) * AD) (h5 : ∃ (ED : ℝ), ED = (2/3) * AD) : 
  ∃ (FC : ℝ), FC = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_l4085_408590
