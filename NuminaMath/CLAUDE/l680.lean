import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l680_68088

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) : 
  a + b^2 + c^3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l680_68088


namespace NUMINAMATH_CALUDE_square_with_tens_digit_seven_l680_68098

/-- Given a number A with more than one digit, if the tens digit of A^2 is 7, 
    then the units digit of A^2 is 6. -/
theorem square_with_tens_digit_seven (A : ℕ) : 
  A > 9 → 
  (A^2 / 10) % 10 = 7 → 
  A^2 % 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_square_with_tens_digit_seven_l680_68098


namespace NUMINAMATH_CALUDE_kangaroo_hop_distance_l680_68047

theorem kangaroo_hop_distance :
  let a : ℚ := 1/2  -- first term
  let r : ℚ := 3/4  -- common ratio
  let n : ℕ := 7    -- number of terms
  (a * (1 - r^n) / (1 - r) : ℚ) = 14297/2048 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_hop_distance_l680_68047


namespace NUMINAMATH_CALUDE_complex_absolute_value_l680_68007

open Complex

theorem complex_absolute_value : ∀ (i : ℂ), i * i = -1 → Complex.abs (2 * i * (1 - 2 * i)) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l680_68007


namespace NUMINAMATH_CALUDE_max_value_expression_l680_68037

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⨆ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + b*x + c^2))) = 
    3/2 * (a^2 + a*b + b^2/4 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l680_68037


namespace NUMINAMATH_CALUDE_complex_arithmetic_calculation_l680_68083

theorem complex_arithmetic_calculation : 
  ((15 - 2) + (4 / (1 / 2)) - (6 * 8)) * (100 - 24) / 38 = -54 := by
sorry

end NUMINAMATH_CALUDE_complex_arithmetic_calculation_l680_68083


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_2a_l680_68005

theorem factorization_a_squared_minus_2a (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_2a_l680_68005


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l680_68095

theorem consecutive_even_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ 
    b = n + 2 ∧ 
    c = n + 4 ∧ 
    d = n + 6 ∧ 
    c = 14) → 
  (n + (n + 2) + (n + 4) + (n + 6) = 52) := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l680_68095


namespace NUMINAMATH_CALUDE_min_value_range_l680_68078

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem min_value_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 a, f x ≥ f a) → a ∈ Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_min_value_range_l680_68078


namespace NUMINAMATH_CALUDE_rectangular_prism_layers_l680_68065

theorem rectangular_prism_layers (prism_volume : ℕ) (block_volume : ℕ) (blocks_per_layer : ℕ) (h1 : prism_volume = 252) (h2 : block_volume = 1) (h3 : blocks_per_layer = 36) : 
  (prism_volume / (blocks_per_layer * block_volume) : ℕ) = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_layers_l680_68065


namespace NUMINAMATH_CALUDE_circle_tangents_theorem_l680_68085

/-- Given two circles with radii x and y touching a circle with radius R,
    and the distance between points of contact a, this theorem proves
    the squared lengths of their common tangents. -/
theorem circle_tangents_theorem
  (R x y a : ℝ)
  (h_pos_R : R > 0)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_pos_a : a > 0) :
  (∃ (l_ext : ℝ), l_ext^2 = (a/R)^2 * (R+x)*(R+y) ∨ l_ext^2 = (a/R)^2 * (R-x)*(R-y)) ∧
  (∃ (l_int : ℝ), l_int^2 = (a/R)^2 * (R+y)*(R-x)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_theorem_l680_68085


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l680_68093

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 + I) / (2 - I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l680_68093


namespace NUMINAMATH_CALUDE_doubled_factorial_30_trailing_zeros_l680_68013

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The number of trailing zeros in 2 * n! -/
def trailingZerosDoubled (n : ℕ) : ℕ := sorry

theorem doubled_factorial_30_trailing_zeros :
  trailingZerosDoubled 30 = 7 := by sorry

end NUMINAMATH_CALUDE_doubled_factorial_30_trailing_zeros_l680_68013


namespace NUMINAMATH_CALUDE_fifteen_more_than_two_thirds_of_120_l680_68023

theorem fifteen_more_than_two_thirds_of_120 : (2 / 3 : ℚ) * 120 + 15 = 95 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_more_than_two_thirds_of_120_l680_68023


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l680_68072

theorem chocolate_bar_cost (total_bars : ℕ) (bars_sold : ℕ) (revenue : ℝ) : 
  total_bars = 7 → 
  bars_sold = total_bars - 4 → 
  revenue = 9 → 
  revenue / bars_sold = 3 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l680_68072


namespace NUMINAMATH_CALUDE_min_ticket_cost_l680_68002

theorem min_ticket_cost (total_tickets : ℕ) (price_low price_high : ℕ) 
  (h_total : total_tickets = 140)
  (h_price_low : price_low = 6)
  (h_price_high : price_high = 10)
  (h_constraint : ∀ x : ℕ, x ≤ total_tickets → total_tickets - x ≥ 2 * x → x ≤ 46) :
  ∃ (low_count high_count : ℕ),
    low_count + high_count = total_tickets ∧
    high_count ≥ 2 * low_count ∧
    low_count = 46 ∧
    high_count = 94 ∧
    low_count * price_low + high_count * price_high = 1216 ∧
    (∀ (a b : ℕ), a + b = total_tickets → b ≥ 2 * a → 
      a * price_low + b * price_high ≥ 1216) :=
by sorry

end NUMINAMATH_CALUDE_min_ticket_cost_l680_68002


namespace NUMINAMATH_CALUDE_least_cans_required_l680_68075

theorem least_cans_required (maaza pepsi sprite cola fanta : ℕ) 
  (h_maaza : maaza = 200)
  (h_pepsi : pepsi = 288)
  (h_sprite : sprite = 736)
  (h_cola : cola = 450)
  (h_fanta : fanta = 625) :
  let gcd := Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd maaza pepsi) sprite) cola) fanta
  gcd = 1 ∧ maaza / gcd + pepsi / gcd + sprite / gcd + cola / gcd + fanta / gcd = 2299 :=
by sorry

end NUMINAMATH_CALUDE_least_cans_required_l680_68075


namespace NUMINAMATH_CALUDE_correct_calculation_l680_68012

theorem correct_calculation (a b : ℝ) : 2 * a * b + 3 * b * a = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l680_68012


namespace NUMINAMATH_CALUDE_alternating_arrangements_2_3_l680_68051

/-- The number of ways to arrange m men and w women in a row, such that no two men or two women are adjacent -/
def alternating_arrangements (m : ℕ) (w : ℕ) : ℕ := sorry

theorem alternating_arrangements_2_3 :
  alternating_arrangements 2 3 = 24 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangements_2_3_l680_68051


namespace NUMINAMATH_CALUDE_min_stamps_for_30_cents_l680_68021

/-- Represents the number of stamps needed to make a certain value -/
structure StampCombination :=
  (threes : ℕ)
  (fours : ℕ)

/-- Calculates the total value of stamps in cents -/
def value (s : StampCombination) : ℕ := 3 * s.threes + 4 * s.fours

/-- Calculates the total number of stamps -/
def total_stamps (s : StampCombination) : ℕ := s.threes + s.fours

/-- Checks if a StampCombination is valid for the given target value -/
def is_valid (s : StampCombination) (target : ℕ) : Prop :=
  value s = target

/-- Theorem: The minimum number of stamps needed to make 30 cents is 8 -/
theorem min_stamps_for_30_cents :
  ∃ (s : StampCombination), is_valid s 30 ∧
    total_stamps s = 8 ∧
    (∀ (t : StampCombination), is_valid t 30 → total_stamps s ≤ total_stamps t) :=
sorry

end NUMINAMATH_CALUDE_min_stamps_for_30_cents_l680_68021


namespace NUMINAMATH_CALUDE_greater_number_proof_l680_68035

theorem greater_number_proof (x y : ℝ) (sum_eq : x + y = 36) (diff_eq : x - y = 12) : 
  max x y = 24 := by
sorry

end NUMINAMATH_CALUDE_greater_number_proof_l680_68035


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l680_68001

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l680_68001


namespace NUMINAMATH_CALUDE_lake_depth_for_specific_cone_l680_68046

/-- Represents a conical hill partially submerged in a lake -/
structure SubmergedCone where
  total_height : ℝ
  volume_ratio_above_water : ℝ

/-- Calculates the depth of the lake at the base of a partially submerged conical hill -/
def lake_depth (cone : SubmergedCone) : ℝ :=
  cone.total_height * (1 - (1 - cone.volume_ratio_above_water) ^ (1/3))

theorem lake_depth_for_specific_cone :
  let cone : SubmergedCone := ⟨5000, 1/5⟩
  lake_depth cone = 660 := by
  sorry

end NUMINAMATH_CALUDE_lake_depth_for_specific_cone_l680_68046


namespace NUMINAMATH_CALUDE_roses_distribution_l680_68054

theorem roses_distribution (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) 
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  : (initial_roses - stolen_roses) / people = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_distribution_l680_68054


namespace NUMINAMATH_CALUDE_gcd_bn_bn_plus_2_is_one_max_en_is_one_l680_68039

theorem gcd_bn_bn_plus_2_is_one (n : ℕ) : 
  Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) = 1 := by
  sorry

theorem max_en_is_one : 
  ∀ n : ℕ, (∃ k : ℕ, Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) = k) → 
  Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_bn_bn_plus_2_is_one_max_en_is_one_l680_68039


namespace NUMINAMATH_CALUDE_largest_package_size_l680_68086

theorem largest_package_size (hazel_pencils leo_pencils mia_pencils : ℕ) 
  (h1 : hazel_pencils = 36)
  (h2 : leo_pencils = 54)
  (h3 : mia_pencils = 72) :
  Nat.gcd hazel_pencils (Nat.gcd leo_pencils mia_pencils) = 18 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l680_68086


namespace NUMINAMATH_CALUDE_certain_number_minus_one_l680_68096

theorem certain_number_minus_one (x : ℝ) (h : 15 * x = 45) : x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_one_l680_68096


namespace NUMINAMATH_CALUDE_quadratic_equation_b_range_l680_68053

theorem quadratic_equation_b_range :
  ∀ (b c : ℝ),
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 + b*x + c = 0) →
  (0 ≤ 3*b + c) →
  (3*b + c ≤ 3) →
  b ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_b_range_l680_68053


namespace NUMINAMATH_CALUDE_mn_max_and_m2n2_min_l680_68026

/-- Given real numbers m and n, where m > 0, n > 0, and 2m + n = 1,
    prove that the maximum value of mn is 1/8 and
    the minimum value of 4m^2 + n^2 is 1/2 -/
theorem mn_max_and_m2n2_min (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → m * n ≥ x * y) ∧
  (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → 4 * m^2 + n^2 ≤ 4 * x^2 + y^2) ∧
  m * n = 1/8 ∧ 4 * m^2 + n^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_mn_max_and_m2n2_min_l680_68026


namespace NUMINAMATH_CALUDE_multiply_is_enlarge_l680_68060

-- Define the concept of enlarging a number
def enlarge (n : ℕ) (times : ℕ) : ℕ := n * times

-- State the theorem
theorem multiply_is_enlarge :
  ∀ (n : ℕ), 28 * 5 = enlarge 28 5 :=
by
  sorry

end NUMINAMATH_CALUDE_multiply_is_enlarge_l680_68060


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l680_68090

def A : Set ℝ := {y | ∃ x, y = Real.cos x}
def B : Set ℝ := {x | x^2 < 9}

theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l680_68090


namespace NUMINAMATH_CALUDE_steps_correct_l680_68074

/-- The number of steps Xiao Gang takes from his house to his school -/
def steps : ℕ := 2000

/-- The distance from Xiao Gang's house to his school in meters -/
def distance : ℝ := 900

/-- Xiao Gang's step length in meters -/
def step_length : ℝ := 0.45

/-- Theorem stating that the number of steps multiplied by the step length equals the distance -/
theorem steps_correct : (steps : ℝ) * step_length = distance := by sorry

end NUMINAMATH_CALUDE_steps_correct_l680_68074


namespace NUMINAMATH_CALUDE_parabola_intersection_through_focus_l680_68097

/-- The parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- Theorem statement -/
theorem parabola_intersection_through_focus 
  (para : Parabola) 
  (l : Line)
  (A B : Point)
  (N : ℝ) -- x-coordinate of N
  (h_not_perpendicular : l.m ≠ 0)
  (h_intersect : A.y^2 = 2*para.p*A.x ∧ B.y^2 = 2*para.p*B.x)
  (h_on_line : A.y = l.m * A.x + l.b ∧ B.y = l.m * B.x + l.b)
  (h_different_quadrants : A.y * B.y < 0)
  (h_bisect : abs ((A.y / (A.x - N)) + (B.y / (B.x - N))) = abs (A.y / (A.x - N) - B.y / (B.x - N))) :
  ∃ (t : ℝ), l.m * (para.p / 2) + l.b = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_through_focus_l680_68097


namespace NUMINAMATH_CALUDE_didi_fundraiser_amount_l680_68040

/-- Calculates the total amount raised from cake sales and donations --/
def total_amount_raised (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
  (donation1_per_slice : ℚ) (donation2_per_slice : ℚ) : ℚ :=
  let total_slices := num_cakes * slices_per_cake
  let sales_amount := total_slices * price_per_slice
  let donation1_amount := total_slices * donation1_per_slice
  let donation2_amount := total_slices * donation2_per_slice
  sales_amount + donation1_amount + donation2_amount

/-- Theorem stating that under the given conditions, the total amount raised is $140 --/
theorem didi_fundraiser_amount :
  total_amount_raised 10 8 1 (1/2) (1/4) = 140 := by
  sorry

end NUMINAMATH_CALUDE_didi_fundraiser_amount_l680_68040


namespace NUMINAMATH_CALUDE_ellipse_theorem_l680_68041

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = 1/2
  h_e_def : e^2 = 1 - (b/a)^2

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The range of t for a line passing through (t,0) intersecting the ellipse -/
def t_range (t : ℝ) : Prop :=
  (t ≤ (4 - 6 * Real.sqrt 2) / 7 ∨ (4 + 6 * Real.sqrt 2) / 7 ≤ t) ∧ t ≠ 1

theorem ellipse_theorem (E : Ellipse) :
  (∀ x y, ellipse_equation E x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ t, t_range t ↔
    ∃ A B : ℝ × ℝ,
      ellipse_equation E A.1 A.2 ∧
      ellipse_equation E B.1 B.2 ∧
      (A.1 - 1) * (B.1 - 1) + A.2 * B.2 = 0 ∧
      A.1 = t ∧ B.1 = t) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l680_68041


namespace NUMINAMATH_CALUDE_exists_central_island_l680_68061

/-- A type representing the islands -/
def Island : Type := ℕ

/-- A structure representing the City of Islands -/
structure CityOfIslands (n : ℕ) where
  /-- The set of islands -/
  islands : Finset Island
  /-- The number of islands is n -/
  island_count : islands.card = n
  /-- Connectivity relation between islands -/
  connected : Island → Island → Prop
  /-- Any two islands are connected (directly or indirectly) -/
  all_connected : ∀ (a b : Island), a ∈ islands → b ∈ islands → connected a b
  /-- The special connectivity property for four islands -/
  four_island_property : ∀ (a b c d : Island), 
    a ∈ islands → b ∈ islands → c ∈ islands → d ∈ islands →
    connected a b → connected b c → connected c d →
    (connected a c ∨ connected b d)

/-- The main theorem: there exists an island connected to all others -/
theorem exists_central_island {n : ℕ} (h : n ≥ 1) (city : CityOfIslands n) : 
  ∃ (central : Island), central ∈ city.islands ∧ 
    ∀ (other : Island), other ∈ city.islands → city.connected central other :=
sorry

end NUMINAMATH_CALUDE_exists_central_island_l680_68061


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l680_68004

theorem christmas_tree_lights (red : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_red : red = 26)
  (h_yellow : yellow = 37)
  (h_blue : blue = 32) :
  red + yellow + blue = 95 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l680_68004


namespace NUMINAMATH_CALUDE_price_per_deck_l680_68045

def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3
def total_earnings : ℕ := 4

theorem price_per_deck :
  (total_earnings : ℚ) / (initial_decks - remaining_decks) = 2 := by
  sorry

end NUMINAMATH_CALUDE_price_per_deck_l680_68045


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l680_68079

theorem product_remainder_mod_five : ∃ k : ℕ, 2532 * 3646 * 2822 * 3716 * 101 = 5 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l680_68079


namespace NUMINAMATH_CALUDE_falcons_minimum_wins_l680_68000

/-- The minimum number of additional games the Falcons need to win -/
def min_additional_games : ℕ := 29

/-- The total number of initial games played -/
def initial_games : ℕ := 5

/-- The number of games won by the Falcons initially -/
def initial_falcons_wins : ℕ := 2

/-- The minimum winning percentage required for the Falcons -/
def min_winning_percentage : ℚ := 91 / 100

theorem falcons_minimum_wins (N : ℕ) :
  (N ≥ min_additional_games) →
  ((initial_falcons_wins + N : ℚ) / (initial_games + N)) ≥ min_winning_percentage ∧
  ∀ M : ℕ, M < min_additional_games →
    ((initial_falcons_wins + M : ℚ) / (initial_games + M)) < min_winning_percentage :=
by sorry

end NUMINAMATH_CALUDE_falcons_minimum_wins_l680_68000


namespace NUMINAMATH_CALUDE_value_of_d_l680_68091

theorem value_of_d (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 4 * d) : d = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l680_68091


namespace NUMINAMATH_CALUDE_road_repair_theorem_l680_68073

/-- The number of persons in the first group -/
def first_group : ℕ := 39

/-- The number of days for the first group to complete the work -/
def days_first : ℕ := 24

/-- The number of hours per day for the first group -/
def hours_first : ℕ := 5

/-- The number of days for the second group to complete the work -/
def days_second : ℕ := 26

/-- The number of hours per day for the second group -/
def hours_second : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * days_first * hours_first

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (days_second * hours_second)

theorem road_repair_theorem : second_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_theorem_l680_68073


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l680_68027

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  totalChairs : ℕ
  seatedPeople : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def validSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.totalChairs / table.seatedPeople ≤ 4

/-- The theorem stating the smallest number of people that can be seated while satisfying the condition. -/
theorem smallest_valid_seating (table : CircularTable) : 
  table.totalChairs = 72 → 
  (∀ n : ℕ, n < table.seatedPeople → ¬validSeating ⟨table.totalChairs, n⟩) →
  validSeating table →
  table.seatedPeople = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l680_68027


namespace NUMINAMATH_CALUDE_gummy_bear_spending_percentage_l680_68034

-- Define the given constants
def hourly_rate : ℚ := 12.5
def hours_worked : ℕ := 40
def tax_rate : ℚ := 0.2
def remaining_money : ℚ := 340

-- Define the function to calculate the percentage spent on gummy bears
def gummy_bear_percentage (rate : ℚ) (hours : ℕ) (tax : ℚ) (remaining : ℚ) : ℚ :=
  let gross_pay := rate * hours
  let net_pay := gross_pay * (1 - tax)
  let spent_on_gummy_bears := net_pay - remaining
  (spent_on_gummy_bears / net_pay) * 100

-- Theorem statement
theorem gummy_bear_spending_percentage :
  gummy_bear_percentage hourly_rate hours_worked tax_rate remaining_money = 15 :=
sorry

end NUMINAMATH_CALUDE_gummy_bear_spending_percentage_l680_68034


namespace NUMINAMATH_CALUDE_stair_climbing_time_l680_68019

theorem stair_climbing_time (a : ℕ) (d : ℕ) (n : ℕ) :
  a = 15 → d = 10 → n = 4 →
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 120 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l680_68019


namespace NUMINAMATH_CALUDE_m_range_l680_68043

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, 2 * x - x^2 < m

def q (m : ℝ) : Prop := |m - 1| ≥ 2

-- State the theorem
theorem m_range :
  (∀ m : ℝ, ¬(¬(p m))) ∧ (∀ m : ℝ, ¬(p m ∧ q m)) →
  ∀ m : ℝ, (m ∈ Set.Ioo 1 3) ↔ (p m ∧ ¬(q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l680_68043


namespace NUMINAMATH_CALUDE_tetrahedron_has_four_faces_l680_68056

/-- A tetrahedron is a type of pyramid with a triangular base -/
structure Tetrahedron where
  is_pyramid : Bool
  has_triangular_base : Bool

/-- The number of faces in a tetrahedron -/
def num_faces (t : Tetrahedron) : Nat :=
  4

theorem tetrahedron_has_four_faces (t : Tetrahedron) :
  t.is_pyramid = true → t.has_triangular_base = true → num_faces t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_has_four_faces_l680_68056


namespace NUMINAMATH_CALUDE_blue_balls_count_l680_68094

def total_balls : ℕ := 12

def prob_two_blue : ℚ := 1/22

theorem blue_balls_count :
  ∃ b : ℕ, 
    b ≤ total_balls ∧ 
    (b : ℚ) / total_balls * ((b - 1) : ℚ) / (total_balls - 1) = prob_two_blue ∧
    b = 3 :=
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l680_68094


namespace NUMINAMATH_CALUDE_triangle_area_l680_68081

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Vectors m and n
  let m := (Real.sin C, Real.sin B * Real.cos A)
  let n := (b, 2 * c)
  -- m · n = 0
  m.1 * n.1 + m.2 * n.2 = 0 →
  -- a = 2√3
  a = 2 * Real.sqrt 3 →
  -- sin B + sin C = 1
  Real.sin B + Real.sin C = 1 →
  -- Area of triangle ABC is √3
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l680_68081


namespace NUMINAMATH_CALUDE_classroom_tables_count_l680_68092

/-- The number of tables in Miss Smith's classroom --/
def number_of_tables : ℕ :=
  let total_students : ℕ := 47
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_students : ℕ := 3 * bathroom_students
  let new_group_students : ℕ := 2 * 4
  let exchange_students : ℕ := 3 * 3
  let missing_students : ℕ := bathroom_students + canteen_students + new_group_students + exchange_students
  let present_students : ℕ := total_students - missing_students
  present_students / students_per_table

theorem classroom_tables_count : number_of_tables = 6 := by
  sorry

end NUMINAMATH_CALUDE_classroom_tables_count_l680_68092


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l680_68059

theorem quadratic_minimum_value (x m : ℝ) : 
  (∀ x, x^2 - 4*x + m ≥ 4) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l680_68059


namespace NUMINAMATH_CALUDE_line_slope_l680_68044

theorem line_slope (A B : ℝ × ℝ) : 
  A.1 = 2 * Real.sqrt 3 ∧ A.2 = -1 ∧ B.1 = Real.sqrt 3 ∧ B.2 = 2 →
  (B.2 - A.2) / (B.1 - A.1) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l680_68044


namespace NUMINAMATH_CALUDE_rectangle_area_l680_68036

/-- Given a rectangle with diagonal length x and length three times its width, 
    prove that its area is (3/10)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
    w^2 + (3*w)^2 = x^2 ∧ 
    3 * w^2 = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l680_68036


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l680_68087

/-- The probability of picking 2 red balls from a bag with 3 red, 2 blue, and 3 green balls -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 3 →
  blue_balls = 2 →
  green_balls = 3 →
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 3 / 28 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l680_68087


namespace NUMINAMATH_CALUDE_pentagon_y_coordinate_of_C_l680_68063

/-- Pentagon with vertices A, B, C, D, E in 2D space -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a triangle given three vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- Main theorem -/
theorem pentagon_y_coordinate_of_C (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 4))
  (h3 : p.D = (4, 4))
  (h4 : p.E = (4, 0))
  (h5 : ∃ y, p.C = (2, y))
  (h6 : hasVerticalSymmetry p)
  (h7 : pentagonArea p = 40) :
  ∃ y, p.C = (2, y) ∧ y = 16 := by sorry

end NUMINAMATH_CALUDE_pentagon_y_coordinate_of_C_l680_68063


namespace NUMINAMATH_CALUDE_unique_positive_solution_l680_68029

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l680_68029


namespace NUMINAMATH_CALUDE_inequality_solution_l680_68055

theorem inequality_solution (x : ℝ) :
  (3 - x) / (5 + 2*x) ≤ 0 ↔ x < -5/2 ∨ x ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l680_68055


namespace NUMINAMATH_CALUDE_prop_false_implies_a_lt_neg_13_div_2_l680_68066

theorem prop_false_implies_a_lt_neg_13_div_2 (a : ℝ) :
  (¬ ∀ x ∈ Set.Icc 1 2, x^2 + a*x + 9 ≥ 0) → a < -13/2 := by
  sorry

end NUMINAMATH_CALUDE_prop_false_implies_a_lt_neg_13_div_2_l680_68066


namespace NUMINAMATH_CALUDE_cricketer_average_score_l680_68009

theorem cricketer_average_score (total_matches : ℕ) 
  (matches_group1 matches_group2 : ℕ)
  (avg_score_group1 avg_score_group2 : ℚ) :
  total_matches = matches_group1 + matches_group2 →
  matches_group1 = 2 →
  matches_group2 = 3 →
  avg_score_group1 = 40 →
  avg_score_group2 = 10 →
  (matches_group1 * avg_score_group1 + matches_group2 * avg_score_group2) / total_matches = 22 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l680_68009


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l680_68064

theorem infinitely_many_divisible_by_prime (p : ℕ) (hp : Prime p) :
  ∃ (N : Set ℕ), Set.Infinite N ∧ ∀ n ∈ N, p ∣ (2^n - n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l680_68064


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l680_68084

theorem imaginary_unit_power (i : ℂ) : i * i = -1 → i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l680_68084


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l680_68028

/-- The volume of a cube with edge length a -/
def cube_volume (a : ℝ) : ℝ := a ^ 3

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The volume of the remaining part after cutting small cubes from vertices -/
def remaining_volume (edge_length : ℝ) (small_cube_volume : ℝ) : ℝ :=
  cube_volume edge_length - (cube_vertices : ℝ) * small_cube_volume

/-- Theorem: The volume of a cube with edge length 3 cm, after removing 
    small cubes of volume 1 cm³ from each of its vertices, is 19 cm³ -/
theorem remaining_cube_volume : 
  remaining_volume 3 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l680_68028


namespace NUMINAMATH_CALUDE_book_pages_count_l680_68057

/-- Count the occurrences of digit 1 in a number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in page numbers from 1 to n -/
def countOnesInPages (n : ℕ) : ℕ := sorry

/-- The number of pages in the book -/
def numPages : ℕ := 318

/-- The total count of digit 1 in the book's page numbers -/
def totalOnes : ℕ := 171

theorem book_pages_count :
  (countOnesInPages numPages = totalOnes) ∧ 
  (∀ m : ℕ, m < numPages → countOnesInPages m < totalOnes) := by sorry

end NUMINAMATH_CALUDE_book_pages_count_l680_68057


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l680_68082

theorem quadratic_roots_relation (q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^2 - 5*x₁ + q = 0) ∧ 
    (x₂^2 - 5*x₂ + q = 0) ∧ 
    (x₃^2 - 7*x₃ + 2*q = 0) ∧ 
    (x₄^2 - 7*x₄ + 2*q = 0) ∧ 
    (x₃ = 2*x₁ ∨ x₃ = 2*x₂ ∨ x₄ = 2*x₁ ∨ x₄ = 2*x₂)) →
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l680_68082


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l680_68099

theorem arithmetic_evaluation : 1537 + 180 / 60 * 15 - 237 = 1345 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l680_68099


namespace NUMINAMATH_CALUDE_min_value_xy_l680_68024

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y + 12 = x * y) :
  x * y ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l680_68024


namespace NUMINAMATH_CALUDE_triangle_sides_from_perimeters_l680_68018

/-- Given the perimeters of three figures formed by two identical squares and two identical triangles,
    prove that the lengths of the sides of the triangle are 5, 12, and 10. -/
theorem triangle_sides_from_perimeters (p1 p2 p3 : ℕ) 
  (h1 : p1 = 74) (h2 : p2 = 84) (h3 : p3 = 82) : 
  ∃ (a b c : ℕ), a = 5 ∧ b = 12 ∧ c = 10 ∧ 
  (∃ (s : ℕ), 2 * s + a + b + c = p1) ∧
  (∃ (s : ℕ), 2 * s + a + b + c + 2 * a = p2) ∧
  (∃ (s : ℕ), 2 * s + 2 * b + 2 * a = p3) :=
by sorry


end NUMINAMATH_CALUDE_triangle_sides_from_perimeters_l680_68018


namespace NUMINAMATH_CALUDE_polynomial_C_value_l680_68089

def polynomial (A B C D : ℤ) (x : ℝ) : ℝ := x^6 - 12*x^5 + A*x^4 + B*x^3 + C*x^2 + D*x + 36

theorem polynomial_C_value (A B C D : ℤ) :
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ x : ℝ, polynomial A B C D x = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) * (x - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12)) →
  C = -171 := by
sorry

end NUMINAMATH_CALUDE_polynomial_C_value_l680_68089


namespace NUMINAMATH_CALUDE_product_equals_24255_l680_68031

theorem product_equals_24255 : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_24255_l680_68031


namespace NUMINAMATH_CALUDE_rose_cost_l680_68052

/-- Proves that the cost of each rose is $5 given the conditions of Nadia's flower purchase. -/
theorem rose_cost (num_roses : ℕ) (num_lilies : ℚ) (total_cost : ℚ) : 
  num_roses = 20 →
  num_lilies = 3/4 * num_roses →
  total_cost = 250 →
  ∃ (rose_cost : ℚ), 
    rose_cost * num_roses + (2 * rose_cost) * num_lilies = total_cost ∧
    rose_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_rose_cost_l680_68052


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l680_68050

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_slope_angle_at_zero :
  let slope := deriv f 0
  Real.arctan slope = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l680_68050


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_390_l680_68070

theorem sin_n_equals_cos_390 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (390 * π / 180) →
  n = 60 ∨ n = 120 := by
sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_390_l680_68070


namespace NUMINAMATH_CALUDE_shapes_can_form_both_rectangles_l680_68016

/-- Represents a pentagon -/
structure Pentagon where
  area : ℝ

/-- Represents a triangle -/
structure Triangle where
  area : ℝ

/-- Represents a set of shapes consisting of two pentagons and a triangle -/
structure ShapeSet where
  pentagon1 : Pentagon
  pentagon2 : Pentagon
  triangle : Triangle

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if a set of shapes can form a given rectangle -/
def can_form_rectangle (shapes : ShapeSet) (rect : Rectangle) : Prop :=
  shapes.pentagon1.area + shapes.pentagon2.area + shapes.triangle.area = rect.width * rect.height

/-- The main theorem stating that it's possible to have a set of shapes
    that can form both a 4x6 and a 3x8 rectangle -/
theorem shapes_can_form_both_rectangles :
  ∃ (shapes : ShapeSet),
    can_form_rectangle shapes (Rectangle.mk 4 6) ∧
    can_form_rectangle shapes (Rectangle.mk 3 8) := by
  sorry

end NUMINAMATH_CALUDE_shapes_can_form_both_rectangles_l680_68016


namespace NUMINAMATH_CALUDE_max_value_implies_a_l680_68038

def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

theorem max_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, f a x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 2, f a x = 3) → 
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l680_68038


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l680_68020

theorem students_playing_neither_sport (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 38)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_both : both = 17) :
  total - (football + tennis - both) = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l680_68020


namespace NUMINAMATH_CALUDE_sum_abcd_equals_21_l680_68022

theorem sum_abcd_equals_21 
  (a b c d : ℝ) 
  (h1 : a * c + a * d + b * c + b * d = 68) 
  (h2 : c + d = 4) : 
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_21_l680_68022


namespace NUMINAMATH_CALUDE_simplify_square_roots_l680_68042

theorem simplify_square_roots : 
  Real.sqrt (10 + 6 * Real.sqrt 2) + Real.sqrt (10 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l680_68042


namespace NUMINAMATH_CALUDE_probability_same_color_eq_l680_68071

def total_marbles : ℕ := 5 + 4 + 6 + 3 + 2

def black_marbles : ℕ := 5
def red_marbles : ℕ := 4
def green_marbles : ℕ := 6
def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 2

def probability_same_color : ℚ :=
  (black_marbles * (black_marbles - 1) * (black_marbles - 2) * (black_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) +
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_eq : probability_same_color = 129 / 31250 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_eq_l680_68071


namespace NUMINAMATH_CALUDE_sequence_terms_coprime_l680_68025

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => (sequence_a n)^2 - sequence_a n + 1

theorem sequence_terms_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (sequence_a m) (sequence_a n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_terms_coprime_l680_68025


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l680_68032

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem inscribed_sphere_volume (π : ℝ) :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = 256 * π / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l680_68032


namespace NUMINAMATH_CALUDE_x_minus_y_equals_nine_l680_68068

theorem x_minus_y_equals_nine (x y : ℕ) (h1 : 3^x * 4^y = 19683) (h2 : x = 9) :
  x - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_nine_l680_68068


namespace NUMINAMATH_CALUDE_b1f_hex_to_dec_l680_68015

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for this problem

/-- Converts a hexadecimal number represented as a string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldl (fun acc d => 16 * acc + hex_to_dec d) 0

theorem b1f_hex_to_dec :
  hex_string_to_dec "B1F" = 2847 := by
  sorry


end NUMINAMATH_CALUDE_b1f_hex_to_dec_l680_68015


namespace NUMINAMATH_CALUDE_third_vertex_y_coordinate_l680_68003

/-- An equilateral triangle with two vertices at (3, 4) and (13, 4), and the third vertex in the first quadrant -/
structure EquilateralTriangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  h1 : v1 = (3, 4)
  h2 : v2 = (13, 4)
  h3 : v3.1 > 0 ∧ v3.2 > 0  -- First quadrant condition
  h4 : (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (v1.1 - v3.1)^2 + (v1.2 - v3.2)^2
  h5 : (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (v2.1 - v3.1)^2 + (v2.2 - v3.2)^2

/-- The y-coordinate of the third vertex is 4 + 5√3 -/
theorem third_vertex_y_coordinate (t : EquilateralTriangle) : t.v3.2 = 4 + 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_y_coordinate_l680_68003


namespace NUMINAMATH_CALUDE_juvy_garden_chives_l680_68069

theorem juvy_garden_chives (total_rows : ℕ) (plants_per_row : ℕ) 
  (parsley_rows : ℕ) (rosemary_rows : ℕ) (mint_rows : ℕ) (thyme_rows : ℕ) :
  total_rows = 50 →
  plants_per_row = 15 →
  parsley_rows = 5 →
  rosemary_rows = 7 →
  mint_rows = 10 →
  thyme_rows = 12 →
  (total_rows - (parsley_rows + rosemary_rows + mint_rows + thyme_rows)) * plants_per_row = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_juvy_garden_chives_l680_68069


namespace NUMINAMATH_CALUDE_camp_provisions_duration_l680_68033

/-- Represents the camp provisions problem -/
theorem camp_provisions_duration (initial_men_1 initial_men_2 : ℕ) 
  (initial_days_1 initial_days_2 : ℕ) (additional_men : ℕ) 
  (consumption_rate : ℚ) (days_before_supply : ℕ) 
  (supply_men supply_days : ℕ) : 
  initial_men_1 = 800 →
  initial_men_2 = 200 →
  initial_days_1 = 20 →
  initial_days_2 = 10 →
  additional_men = 200 →
  consumption_rate = 3/2 →
  days_before_supply = 10 →
  supply_men = 300 →
  supply_days = 15 →
  ∃ (remaining_days : ℚ), 
    remaining_days > 7.30 ∧ 
    remaining_days < 7.32 ∧
    remaining_days = 
      (initial_men_1 * initial_days_1 + initial_men_2 * initial_days_2 - 
       (initial_men_1 + initial_men_2 + additional_men * consumption_rate) * days_before_supply +
       supply_men * supply_days) / 
      (initial_men_1 + initial_men_2 + additional_men * consumption_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_camp_provisions_duration_l680_68033


namespace NUMINAMATH_CALUDE_sqrt_factorial_squared_l680_68076

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem sqrt_factorial_squared :
  (((factorial 5 * factorial 4 : ℕ) : ℝ).sqrt ^ 2 : ℝ) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_factorial_squared_l680_68076


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l680_68008

theorem cube_surface_area_from_volume :
  ∀ (v : ℝ) (s : ℝ) (sa : ℝ),
    v = 729 →
    v = s^3 →
    sa = 6 * s^2 →
    sa = 486 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l680_68008


namespace NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l680_68017

/-- Represents the schedule for Charlotte's dog walking --/
structure DogWalkingSchedule where
  poodles_monday : ℕ
  chihuahuas_monday : ℕ
  labradors_wednesday : ℕ
  poodle_time : ℕ
  chihuahua_time : ℕ
  labrador_time : ℕ
  total_time : ℕ

/-- Calculates the number of poodles Charlotte can walk on Tuesday --/
def poodles_tuesday (s : DogWalkingSchedule) : ℕ :=
  let monday_time := s.poodles_monday * s.poodle_time + s.chihuahuas_monday * s.chihuahua_time
  let wednesday_time := s.labradors_wednesday * s.labrador_time
  let tuesday_time := s.total_time - monday_time - wednesday_time - s.chihuahuas_monday * s.chihuahua_time
  tuesday_time / s.poodle_time

/-- Theorem stating that Charlotte can walk 4 poodles on Tuesday --/
theorem charlotte_tuesday_poodles (s : DogWalkingSchedule) 
  (h1 : s.poodles_monday = 4)
  (h2 : s.chihuahuas_monday = 2)
  (h3 : s.labradors_wednesday = 4)
  (h4 : s.poodle_time = 2)
  (h5 : s.chihuahua_time = 1)
  (h6 : s.labrador_time = 3)
  (h7 : s.total_time = 32) :
  poodles_tuesday s = 4 := by
  sorry


end NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l680_68017


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l680_68010

theorem circle_area_from_circumference (circumference : ℝ) (area : ℝ) :
  circumference = 18 →
  area = 81 / Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l680_68010


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l680_68030

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- The problem statement -/
theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-6, y)
  collinear a b → y = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l680_68030


namespace NUMINAMATH_CALUDE_man_speed_man_speed_proof_l680_68048

/-- The speed of a man relative to a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- Proof that the speed of the man is approximately 0.833 m/s given the specified conditions. -/
theorem man_speed_proof :
  let train_length : ℝ := 500
  let train_speed_kmh : ℝ := 63
  let crossing_time : ℝ := 29.997600191984642
  abs (man_speed train_length train_speed_kmh crossing_time - 0.833) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_man_speed_man_speed_proof_l680_68048


namespace NUMINAMATH_CALUDE_triangle_side_length_l680_68062

/-- Given a triangle ABC where sin A, sin B, sin C form an arithmetic sequence,
    B = 30°, and the area is 3/2, prove that the length of side b is √3 + 1. -/
theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c →
  -- sin A, sin B, sin C form an arithmetic sequence
  2 * Real.sin B = Real.sin A + Real.sin C →
  -- B = 30°
  B = π / 6 →
  -- Area of triangle ABC is 3/2
  1/2 * a * c * Real.sin B = 3/2 →
  -- b is opposite to angle B
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  -- Conclusion: length of side b is √3 + 1
  b = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l680_68062


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_series_l680_68077

theorem smallest_b_in_arithmetic_series (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  (∃ d : ℝ, a = b - d ∧ c = b + d) →  -- arithmetic series condition
  a * b * c = 125 →  -- product condition
  ∀ x : ℝ, (∃ y z : ℝ, 
    0 < y ∧ 0 < x ∧ 0 < z ∧  -- positivity for new terms
    (∃ e : ℝ, y = x - e ∧ z = x + e) ∧  -- arithmetic series for new terms
    y * x * z = 125) →  -- product condition for new terms
  x ≥ b →
  b ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_series_l680_68077


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l680_68014

/-- The diagonal of a rectangle with length 30√3 cm and width 30 cm is 60 cm. -/
theorem rectangle_diagonal : 
  let length : ℝ := 30 * Real.sqrt 3
  let width : ℝ := 30
  let diagonal : ℝ := Real.sqrt (length^2 + width^2)
  diagonal = 60 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l680_68014


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l680_68067

/-- Represents the possible number of apples in a bag -/
inductive BagSize
| small : BagSize  -- 6 apples
| large : BagSize  -- 12 apples

/-- Returns true if the given number is a valid total number of apples -/
def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ ∃ (small large : ℕ), n = 6 * small + 12 * large

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l680_68067


namespace NUMINAMATH_CALUDE_planes_perpendicular_if_line_perpendicular_and_parallel_l680_68011

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_if_line_perpendicular_and_parallel
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel a β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_if_line_perpendicular_and_parallel_l680_68011


namespace NUMINAMATH_CALUDE_speed_difference_l680_68049

theorem speed_difference (distance : ℝ) (emma_time lucas_time : ℝ) 
  (h1 : distance = 8)
  (h2 : emma_time = 12 / 60)
  (h3 : lucas_time = 40 / 60) :
  (distance / emma_time) - (distance / lucas_time) = 28 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l680_68049


namespace NUMINAMATH_CALUDE_window_area_theorem_l680_68058

/-- Represents a rectangular glass pane with length and width in inches. -/
structure GlassPane where
  length : ℕ
  width : ℕ

/-- Calculates the area of a single glass pane in square inches. -/
def pane_area (pane : GlassPane) : ℕ :=
  pane.length * pane.width

/-- Represents a window composed of multiple identical glass panes. -/
structure Window where
  pane : GlassPane
  num_panes : ℕ

/-- Calculates the total area of a window in square inches. -/
def window_area (w : Window) : ℕ :=
  pane_area w.pane * w.num_panes

/-- Theorem: The area of a window with 8 panes, each 12 inches by 8 inches, is 768 square inches. -/
theorem window_area_theorem : 
  ∀ (w : Window), w.pane.length = 12 → w.pane.width = 8 → w.num_panes = 8 → 
  window_area w = 768 := by
  sorry

end NUMINAMATH_CALUDE_window_area_theorem_l680_68058


namespace NUMINAMATH_CALUDE_binomial_variance_example_l680_68080

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a random variable -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Given a random variable X following the binomial distribution B(6, 1/3), its variance D(X) is 4/3 -/
theorem binomial_variance_example :
  let X : BinomialDistribution := ⟨6, 1/3, by norm_num⟩
  variance X = 4/3 := by sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l680_68080


namespace NUMINAMATH_CALUDE_multiplication_of_negative_half_and_two_l680_68006

theorem multiplication_of_negative_half_and_two :
  (-1/2 : ℚ) * 2 = -1 := by sorry

end NUMINAMATH_CALUDE_multiplication_of_negative_half_and_two_l680_68006
