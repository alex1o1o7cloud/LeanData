import Mathlib

namespace NUMINAMATH_CALUDE_crow_probability_l3180_318041

/-- Represents the number of crows of each color on each tree -/
structure CrowDistribution where
  birch_white : ℕ
  birch_black : ℕ
  oak_white : ℕ
  oak_black : ℕ

/-- The probability that the number of white crows on the birch tree remains the same -/
def prob_same (d : CrowDistribution) : ℚ :=
  (d.birch_black * (d.oak_black + 1) + d.birch_white * (d.oak_white + 1)) / (50 * 51)

/-- The probability that the number of white crows on the birch tree changes -/
def prob_change (d : CrowDistribution) : ℚ :=
  (d.birch_black * d.oak_white + d.birch_white * d.oak_black) / (50 * 51)

theorem crow_probability (d : CrowDistribution) 
  (h1 : d.birch_white + d.birch_black = 50)
  (h2 : d.oak_white + d.oak_black = 50)
  (h3 : d.birch_white > 0)
  (h4 : d.oak_white > 0)
  (h5 : d.birch_black ≥ d.birch_white)
  (h6 : d.oak_black ≥ d.oak_white ∨ d.oak_black + 1 = d.oak_white) :
  prob_same d > prob_change d := by
  sorry

end NUMINAMATH_CALUDE_crow_probability_l3180_318041


namespace NUMINAMATH_CALUDE_complex_multiplication_l3180_318088

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) :
  (1 + i) * (1 - 2*i) = 3 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3180_318088


namespace NUMINAMATH_CALUDE_family_adults_count_l3180_318067

/-- Represents the number of adults in a family visiting an amusement park. -/
def adults : ℕ := sorry

/-- The cost of an adult ticket in dollars. -/
def adult_ticket_cost : ℕ := 22

/-- The cost of a child ticket in dollars. -/
def child_ticket_cost : ℕ := 7

/-- The number of children in the family. -/
def num_children : ℕ := 2

/-- The total cost for the family's admission in dollars. -/
def total_cost : ℕ := 58

/-- Theorem stating that the number of adults in the family is 2. -/
theorem family_adults_count : adults = 2 := by
  sorry

end NUMINAMATH_CALUDE_family_adults_count_l3180_318067


namespace NUMINAMATH_CALUDE_n_range_theorem_l3180_318015

theorem n_range_theorem (x y m n : ℝ) :
  n ≤ x ∧ x < y ∧ y ≤ n + 1 ∧
  m ∈ Set.Ioo x y ∧
  |y| = |m| + |x| →
  -1 < n ∧ n < 1 :=
by sorry

end NUMINAMATH_CALUDE_n_range_theorem_l3180_318015


namespace NUMINAMATH_CALUDE_fifth_bank_coins_l3180_318097

def coins_in_bank (n : ℕ) : ℕ := 72 + 9 * (n - 1)

theorem fifth_bank_coins :
  coins_in_bank 5 = 108 :=
by sorry

end NUMINAMATH_CALUDE_fifth_bank_coins_l3180_318097


namespace NUMINAMATH_CALUDE_petes_number_l3180_318078

theorem petes_number (x : ℚ) : 4 * (2 * x + 10) = 120 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l3180_318078


namespace NUMINAMATH_CALUDE_fraction_undefined_values_l3180_318018

def undefined_values (a : ℝ) : Prop :=
  a^3 - 4*a = 0

theorem fraction_undefined_values :
  {a : ℝ | undefined_values a} = {-2, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_values_l3180_318018


namespace NUMINAMATH_CALUDE_circle_radius_with_parabolas_l3180_318062

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

end NUMINAMATH_CALUDE_circle_radius_with_parabolas_l3180_318062


namespace NUMINAMATH_CALUDE_line_equation_sum_l3180_318084

/-- Given a line with slope -3 passing through the point (5, 2),
    prove that m + b = 14 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (m b : ℝ) : 
  m = -3 → 
  2 = m * 5 + b → 
  m + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_sum_l3180_318084


namespace NUMINAMATH_CALUDE_no_prime_pair_sum_65_l3180_318099

theorem no_prime_pair_sum_65 : ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65 ∧ ∃ (k : ℕ), p * q = k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pair_sum_65_l3180_318099


namespace NUMINAMATH_CALUDE_area_ratio_EFWZ_ZWGH_l3180_318023

-- Define the points
variable (E F G H O Q Z W : ℝ × ℝ)

-- Define the lengths
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
axiom EF_eq_EO : length E F = length E O
axiom EO_eq_OG : length E O = length O G
axiom OG_eq_GH : length O G = length G H
axiom EF_eq_12 : length E F = 12
axiom FG_eq_18 : length F G = 18
axiom EH_eq_18 : length E H = 18
axiom OH_eq_18 : length O H = 18

-- Define Q as the point on FG such that OQ is perpendicular to FG
axiom Q_on_FG : sorry
axiom OQ_perp_FG : sorry

-- Define Z as midpoint of EF
axiom Z_midpoint_EF : sorry

-- Define W as midpoint of GH
axiom W_midpoint_GH : sorry

-- Define the area function for trapezoids
def area_trapezoid (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_EFWZ_ZWGH : 
  area_trapezoid E F W Z = area_trapezoid Z W G H := by sorry

end NUMINAMATH_CALUDE_area_ratio_EFWZ_ZWGH_l3180_318023


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l3180_318019

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 30
  let hole_depth : ℝ := 8
  let small_hole_diameter : ℝ := 2
  let large_hole_diameter : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let small_hole_volume := π * (small_hole_diameter / 2) ^ 2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2) ^ 2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 4466 * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l3180_318019


namespace NUMINAMATH_CALUDE_quadratic_minimum_property_l3180_318063

/-- Given a quadratic function f(x) = ax^2 + bx + 1 with minimum value f(1) = 0, prove that a - b = 3 -/
theorem quadratic_minimum_property (a b : ℝ) : 
  (∀ x, a*x^2 + b*x + 1 ≥ a + b + 1) ∧ (a + b + 1 = 0) → a - b = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_property_l3180_318063


namespace NUMINAMATH_CALUDE_tan_nine_pi_fourth_l3180_318055

theorem tan_nine_pi_fourth : Real.tan (9 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_nine_pi_fourth_l3180_318055


namespace NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l3180_318069

-- Problem 1
def problem1 (x y : ℤ) : ℤ :=
  (2 * x^2 * y - 4 * x * y^2) - (-3 * x * y^2 + x^2 * y)

theorem problem1_solution :
  problem1 (-1) 2 = 6 := by sorry

-- Problem 2
def A (x y : ℤ) : ℤ := x^2 - x*y + y^2
def B (x y : ℤ) : ℤ := -x^2 + 2*x*y + y^2

theorem problem2_solution :
  A 2010 (-1) + B 2010 (-1) = -2008 := by sorry

end NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l3180_318069


namespace NUMINAMATH_CALUDE_roots_transformation_l3180_318096

theorem roots_transformation (a b c d : ℝ) : 
  (a^4 - 16*a - 2 = 0) ∧ 
  (b^4 - 16*b - 2 = 0) ∧ 
  (c^4 - 16*c - 2 = 0) ∧ 
  (d^4 - 16*d - 2 = 0) →
  ((a+b)/c^2)^4 - 16*((a+b)/c^2)^3 - 1/2 = 0 ∧
  ((a+c)/b^2)^4 - 16*((a+c)/b^2)^3 - 1/2 = 0 ∧
  ((b+c)/a^2)^4 - 16*((b+c)/a^2)^3 - 1/2 = 0 ∧
  ((b+d)/d^2)^4 - 16*((b+d)/d^2)^3 - 1/2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l3180_318096


namespace NUMINAMATH_CALUDE_chess_tournament_max_matches_l3180_318043

/-- Represents a chess tournament. -/
structure ChessTournament where
  participants : Nat
  max_matches_per_pair : Nat
  no_triples : Bool

/-- The maximum number of matches any participant can play. -/
def max_matches_per_participant (t : ChessTournament) : Nat :=
  sorry

/-- The theorem stating the maximum number of matches per participant
    in a chess tournament with the given conditions. -/
theorem chess_tournament_max_matches
  (t : ChessTournament)
  (h1 : t.participants = 300)
  (h2 : t.max_matches_per_pair = 1)
  (h3 : t.no_triples = true) :
  max_matches_per_participant t = 200 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_max_matches_l3180_318043


namespace NUMINAMATH_CALUDE_hall_width_proof_l3180_318011

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 25 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) 
    (h1 : length = 20)
    (h2 : height = 5)
    (h3 : cost_per_sqm = 20)
    (h4 : total_cost = 19000) :
  ∃ (width : ℝ), 
    total_cost = cost_per_sqm * (length * width + 2 * length * height + 2 * width * height) ∧ 
    width = 25 := by
  sorry

end NUMINAMATH_CALUDE_hall_width_proof_l3180_318011


namespace NUMINAMATH_CALUDE_wechat_payment_balance_l3180_318098

/-- Represents a transaction with a description and an amount -/
structure Transaction where
  description : String
  amount : Int

/-- Calculates the balance from a list of transactions -/
def calculate_balance (transactions : List Transaction) : Int :=
  transactions.foldl (fun acc t => acc + t.amount) 0

/-- Theorem stating that the WeChat change payment balance for the day is an expenditure of $32 -/
theorem wechat_payment_balance : 
  let transactions : List Transaction := [
    { description := "Transfer from LZT", amount := 48 },
    { description := "Blue Wisteria Culture", amount := -30 },
    { description := "Scan QR code payment", amount := -50 }
  ]
  calculate_balance transactions = -32 := by sorry

end NUMINAMATH_CALUDE_wechat_payment_balance_l3180_318098


namespace NUMINAMATH_CALUDE_probability_different_digits_l3180_318021

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def count_valid_numbers : ℕ :=
  999 - 100 + 1

def count_different_digit_numbers : ℕ :=
  9 * 9 * 8

theorem probability_different_digits :
  (count_different_digit_numbers : ℚ) / count_valid_numbers = 18 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_digits_l3180_318021


namespace NUMINAMATH_CALUDE_vinegar_mixture_percentage_l3180_318007

/-- The percentage of the second vinegar solution -/
def P : ℝ := 40

/-- Volume of each initial solution in milliliters -/
def initial_volume : ℝ := 10

/-- Volume of the final mixture in milliliters -/
def final_volume : ℝ := 50

/-- Percentage of the first solution -/
def first_percentage : ℝ := 5

/-- Percentage of the final mixture -/
def final_percentage : ℝ := 9

theorem vinegar_mixture_percentage :
  initial_volume * (first_percentage / 100) +
  initial_volume * (P / 100) =
  final_volume * (final_percentage / 100) :=
sorry

end NUMINAMATH_CALUDE_vinegar_mixture_percentage_l3180_318007


namespace NUMINAMATH_CALUDE_stratified_sampling_students_l3180_318035

theorem stratified_sampling_students (total_population : ℕ) (teachers : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 1600)
  (h2 : teachers = 100)
  (h3 : sample_size = 80) :
  (sample_size * (total_population - teachers)) / total_population = 75 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_students_l3180_318035


namespace NUMINAMATH_CALUDE_certain_number_equation_l3180_318081

theorem certain_number_equation (x : ℝ) : (10 + 20 + 60) / 3 = ((10 + x + 25) / 3) + 5 ↔ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3180_318081


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l3180_318048

theorem pure_imaginary_square (a : ℝ) : 
  (∃ b : ℝ, (1 + a * Complex.I)^2 = b * Complex.I) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l3180_318048


namespace NUMINAMATH_CALUDE_dilation_problem_l3180_318040

def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

theorem dilation_problem : 
  let center := (0 : ℂ) + 5*I
  let scale := (3 : ℂ)
  let z := (3 : ℂ) + 2*I
  dilation center scale z = (9 : ℂ) - 4*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_problem_l3180_318040


namespace NUMINAMATH_CALUDE_pages_per_day_l3180_318066

/-- Given a book with 144 pages, prove that reading two-thirds of it in 12 days results in 8 pages read per day. -/
theorem pages_per_day (total_pages : ℕ) (days_read : ℕ) (fraction_read : ℚ) : 
  total_pages = 144 → 
  days_read = 12 → 
  fraction_read = 2/3 →
  (fraction_read * total_pages) / days_read = 8 := by
sorry

end NUMINAMATH_CALUDE_pages_per_day_l3180_318066


namespace NUMINAMATH_CALUDE_line_equation_sum_of_squares_l3180_318009

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 4 * x - 7

-- Define the point (2,1) that the line passes through
def point_on_line : Prop := line_l 2 1

-- Define the equation ax = by + c
def line_equation (a b c : ℤ) (x y : ℝ) : Prop := a * x = b * y + c

-- State that a, b, and c are positive integers with gcd 1
def abc_conditions (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ Int.gcd a (Int.gcd b c) = 1

-- Theorem statement
theorem line_equation_sum_of_squares :
  ∀ a b c : ℤ,
  (∀ x y : ℝ, line_l x y ↔ line_equation a b c x y) →
  abc_conditions a b c →
  a^2 + b^2 + c^2 = 66 := by sorry

end NUMINAMATH_CALUDE_line_equation_sum_of_squares_l3180_318009


namespace NUMINAMATH_CALUDE_sum_of_divisors_30_l3180_318027

theorem sum_of_divisors_30 : (Finset.filter (· ∣ 30) (Finset.range 31)).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_30_l3180_318027


namespace NUMINAMATH_CALUDE_circle_segment_area_l3180_318074

theorem circle_segment_area (r chord_length intersection_dist : ℝ) 
  (hr : r = 45)
  (hc : chord_length = 84)
  (hi : intersection_dist = 15) : 
  ∃ (m n d : ℝ), 
    (m = 506.25 ∧ n = 1012.5 ∧ d = 1) ∧
    (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) :=
by sorry

end NUMINAMATH_CALUDE_circle_segment_area_l3180_318074


namespace NUMINAMATH_CALUDE_fraction_difference_zero_l3180_318039

theorem fraction_difference_zero (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  1 / ((x - 2) * (x - 3)) - 2 / ((x - 1) * (x - 3)) + 1 / ((x - 1) * (x - 2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_zero_l3180_318039


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l3180_318047

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : ℕ := 3

/-- Calculates the probability of Cheryl getting 3 marbles of the same color -/
theorem cheryl_same_color_probability :
  (num_colors * (Nat.choose (total_marbles - 2 * marbles_drawn) marbles_drawn)) /
  (Nat.choose total_marbles marbles_drawn * Nat.choose (total_marbles - marbles_drawn) marbles_drawn) = 1 / 28 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l3180_318047


namespace NUMINAMATH_CALUDE_complement_union_A_B_l3180_318034

def U : Set ℕ := {0, 1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 7*x + 12 = 0}

def B : Set ℕ := {1, 3, 5}

theorem complement_union_A_B : (U \ (A ∪ B)) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l3180_318034


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3180_318070

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis --/
def symmetricXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- Theorem: If A(3,a) is symmetric to B(b,4) with respect to x-axis, then a + b = -1 --/
theorem symmetric_points_sum (a b : ℝ) : 
  let A : Point := ⟨3, a⟩
  let B : Point := ⟨b, 4⟩
  symmetricXAxis A B → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3180_318070


namespace NUMINAMATH_CALUDE_last_three_average_l3180_318017

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 65 →
  (list.take 4).sum / 4 = 60 →
  (list.drop 4).sum / 3 = 215 / 3 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l3180_318017


namespace NUMINAMATH_CALUDE_inequality_solution_l3180_318092

theorem inequality_solution (x : ℝ) : x^2 - 3*x - 10 < 0 ∧ x > 1 → 1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3180_318092


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l3180_318083

theorem arctan_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 (π / 2) →
  Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (30 * π / 180)) = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l3180_318083


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l3180_318036

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → a ≤ b ∧ a ≤ c := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l3180_318036


namespace NUMINAMATH_CALUDE_total_sum_is_120_rupees_l3180_318003

/-- Represents the division of money among three people -/
structure MoneyDivision where
  a_share : ℕ  -- A's share in paisa per rupee
  b_share : ℕ  -- B's share in paisa per rupee
  c_share : ℕ  -- C's share in paisa per rupee

/-- The given problem setup -/
def problem_setup : MoneyDivision :=
  { a_share := 0,  -- We don't know A's exact share, so we leave it as 0
    b_share := 65,
    c_share := 40 }

/-- Theorem stating the total sum of money -/
theorem total_sum_is_120_rupees (md : MoneyDivision) 
  (h1 : md.b_share = 65)
  (h2 : md.c_share = 40)
  (h3 : md.a_share + md.b_share + md.c_share = 100)  -- Total per rupee is 100 paisa
  (h4 : md.c_share * 120 = 4800)  -- C's share is Rs. 48 (4800 paisa)
  : (4800 / md.c_share) * 100 = 12000 := by
  sorry

#check total_sum_is_120_rupees

end NUMINAMATH_CALUDE_total_sum_is_120_rupees_l3180_318003


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3180_318031

/-- Calculate the average speed for a round trip given specific segments and speeds -/
theorem round_trip_average_speed 
  (total_distance : ℝ)
  (train_distance train_speed : ℝ)
  (car_to_y_distance car_to_y_speed : ℝ)
  (bus_distance bus_speed : ℝ)
  (car_return_distance car_return_speed : ℝ)
  (plane_speed : ℝ)
  (h1 : total_distance = 1500)
  (h2 : train_distance = 500)
  (h3 : train_speed = 60)
  (h4 : car_to_y_distance = 700)
  (h5 : car_to_y_speed = 50)
  (h6 : bus_distance = 300)
  (h7 : bus_speed = 40)
  (h8 : car_return_distance = 600)
  (h9 : car_return_speed = 60)
  (h10 : plane_speed = 500)
  : ∃ (average_speed : ℝ), abs (average_speed - 72.03) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l3180_318031


namespace NUMINAMATH_CALUDE_relationship_abc_l3180_318006

theorem relationship_abc : 
  let a : ℝ := Real.sqrt 0.5
  let b : ℝ := Real.sqrt 0.3
  let c : ℝ := (Real.log 2) / (Real.log 0.3)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3180_318006


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l3180_318050

theorem power_zero_eq_one (a : ℝ) (h : a ≠ 0) : a ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l3180_318050


namespace NUMINAMATH_CALUDE_distance_point_to_parametric_line_l3180_318082

/-- The distance from a point to a line defined parametrically -/
theorem distance_point_to_parametric_line :
  let P : ℝ × ℝ := (2, 0)
  let line (t : ℝ) : ℝ × ℝ := (1 + 4*t, 2 + 3*t)
  let distance (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) : ℝ :=
    -- Define distance function here (implementation not provided)
    sorry
  distance P line = 11/5 :=
by sorry

end NUMINAMATH_CALUDE_distance_point_to_parametric_line_l3180_318082


namespace NUMINAMATH_CALUDE_infinitely_many_n_congruent_to_sum_of_digits_l3180_318085

/-- Sum of digits in base r -/
def S_r (r : ℕ) (n : ℕ) : ℕ := sorry

/-- There are infinitely many n such that S_r(n) ≡ n (mod p) -/
theorem infinitely_many_n_congruent_to_sum_of_digits 
  (r : ℕ) (p : ℕ) (hr : r > 1) (hp : Nat.Prime p) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, S_r r (f k) ≡ f k [MOD p] := by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_congruent_to_sum_of_digits_l3180_318085


namespace NUMINAMATH_CALUDE_system_solution_l3180_318095

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 7 * y = -9) ∧ (5 * x + 3 * y = -11) ∧ (x = -104/47) ∧ (y = 1/47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3180_318095


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3180_318013

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 8) :
  ∃ (k : ℕ+), k ≥ 32 ∧ Nat.gcd (8 * m) (12 * n) = k ∧
  ∀ (l : ℕ+), Nat.gcd (8 * m) (12 * n) = l → l ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3180_318013


namespace NUMINAMATH_CALUDE_integral_inequality_l3180_318073

variables {a b : ℝ} (f : ℝ → ℝ)

/-- The main theorem statement -/
theorem integral_inequality
  (hab : 0 < a ∧ a < b)
  (hf : Continuous f)
  (hf_int : ∫ x in a..b, f x = 0) :
  ∫ x in a..b, ∫ y in a..b, f x * f y * Real.log (x + y) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l3180_318073


namespace NUMINAMATH_CALUDE_function_range_implies_m_range_l3180_318058

-- Define the function f(x) = x^2 - 2x + 5
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Define the theorem
theorem function_range_implies_m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 4) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_function_range_implies_m_range_l3180_318058


namespace NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l3180_318030

/-- The equation of a circle given its center and a chord on a line. -/
theorem circle_equation_from_center_and_chord (x y : ℝ) :
  let center : ℝ × ℝ := (4, 7)
  let chord_length : ℝ := 8
  let line_eq : ℝ → ℝ → ℝ := fun x y => 3 * x - 4 * y + 1
  (∃ (a b : ℝ), (a - 4)^2 + (b - 7)^2 = 25 ∧ 
                line_eq a b = 0 ∧ 
                (a - center.1)^2 + (b - center.2)^2 = (chord_length / 2)^2) →
  (x - 4)^2 + (y - 7)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l3180_318030


namespace NUMINAMATH_CALUDE_intersection_equality_implies_m_value_l3180_318052

def A (m : ℝ) : Set ℝ := {2, 5, m^2 - m}
def B (m : ℝ) : Set ℝ := {2, m + 3}

theorem intersection_equality_implies_m_value :
  ∀ m : ℝ, A m ∩ B m = B m → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_m_value_l3180_318052


namespace NUMINAMATH_CALUDE_sqrt_two_inequality_l3180_318086

theorem sqrt_two_inequality (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_inequality_l3180_318086


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l3180_318004

theorem simplify_sqrt_product (y : ℝ) (hy : y > 0) :
  Real.sqrt (50 * y^3) * Real.sqrt (18 * y) * Real.sqrt (98 * y^5) = 210 * y^4 * Real.sqrt (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l3180_318004


namespace NUMINAMATH_CALUDE_power_of_seven_mod_2000_l3180_318026

theorem power_of_seven_mod_2000 : 7^2023 % 2000 = 1849 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_2000_l3180_318026


namespace NUMINAMATH_CALUDE_set_equality_l3180_318077

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3180_318077


namespace NUMINAMATH_CALUDE_baker_cakes_l3180_318014

/-- Calculates the remaining number of cakes after buying and selling -/
def remaining_cakes (initial bought sold : ℕ) : ℕ :=
  initial + bought - sold

/-- Theorem: The number of cakes Baker still has is 190 -/
theorem baker_cakes : remaining_cakes 173 103 86 = 190 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l3180_318014


namespace NUMINAMATH_CALUDE_p_is_true_q_is_false_p_or_q_is_true_p_and_not_q_is_true_l3180_318010

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem stating that p is true
theorem p_is_true : p := sorry

-- Theorem stating that q is false
theorem q_is_false : ¬q := sorry

-- Theorem stating that the disjunction of p and q is true
theorem p_or_q_is_true : p ∨ q := sorry

-- Theorem stating that the conjunction of p and not q is true
theorem p_and_not_q_is_true : p ∧ ¬q := sorry

end NUMINAMATH_CALUDE_p_is_true_q_is_false_p_or_q_is_true_p_and_not_q_is_true_l3180_318010


namespace NUMINAMATH_CALUDE_max_value_function_l3180_318000

theorem max_value_function (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 1 ≤ 14) → 
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 1 = 14) → 
  a = 1/3 ∨ a = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_function_l3180_318000


namespace NUMINAMATH_CALUDE_statements_correctness_l3180_318025

theorem statements_correctness : 
  (∃! n : ℕ, n = 3 ∧ 
    (2^3 = 8) ∧ 
    (∀ r : ℚ, ∃ s : ℚ, s < r) ∧ 
    (∀ x : ℝ, x + x = 0 → x = 0) ∧ 
    (Real.sqrt ((-4)^2) ≠ 4) ∧ 
    (∃ x : ℝ, x ≠ 1 ∧ 1 / x ≠ 1)) :=
by sorry

end NUMINAMATH_CALUDE_statements_correctness_l3180_318025


namespace NUMINAMATH_CALUDE_specific_triangle_perimeter_l3180_318071

/-- Triangle with parallel lines intersecting its interior --/
structure TriangleWithParallelLines where
  -- Side lengths of the original triangle
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  -- Lengths of segments formed by parallel lines intersecting the triangle
  m_P_length : ℝ
  m_Q_length : ℝ
  m_R_length : ℝ

/-- Calculate the perimeter of the inner triangle formed by parallel lines --/
def innerTrianglePerimeter (t : TriangleWithParallelLines) : ℝ :=
  sorry

/-- Theorem statement for the specific triangle problem --/
theorem specific_triangle_perimeter :
  let t : TriangleWithParallelLines := {
    PQ := 150,
    QR := 275,
    PR := 225,
    m_P_length := 65,
    m_Q_length := 55,
    m_R_length := 25
  }
  innerTrianglePerimeter t = 755 := by
    sorry

end NUMINAMATH_CALUDE_specific_triangle_perimeter_l3180_318071


namespace NUMINAMATH_CALUDE_yellow_packs_bought_l3180_318042

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℕ := 80

/-- The number of packs of green bouncy balls given away -/
def green_packs_given : ℕ := 4

/-- The number of packs of green bouncy balls bought -/
def green_packs_bought : ℕ := 4

/-- The theorem stating the number of packs of yellow bouncy balls Maggie bought -/
theorem yellow_packs_bought : 
  (total_balls / balls_per_pack : ℕ) = 8 :=
sorry

end NUMINAMATH_CALUDE_yellow_packs_bought_l3180_318042


namespace NUMINAMATH_CALUDE_imaginary_unit_multiplication_l3180_318053

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_multiplication :
  i * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_multiplication_l3180_318053


namespace NUMINAMATH_CALUDE_max_omega_for_monotonic_sine_l3180_318065

theorem max_omega_for_monotonic_sine (A ω : ℝ) (h_A : A > 0) (h_ω : ω > 0) :
  (∀ x ∈ Set.Icc (-3 * π / 4) (-π / 6),
    ∀ y ∈ Set.Icc (-3 * π / 4) (-π / 6),
    x < y → A * Real.sin (x + ω * π / 2) < A * Real.sin (y + ω * π / 2)) →
  ω ≤ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_omega_for_monotonic_sine_l3180_318065


namespace NUMINAMATH_CALUDE_polynomial_roots_l3180_318064

theorem polynomial_roots : 
  let p : ℝ → ℝ := λ x => 3*x^4 + 17*x^3 - 32*x^2 - 12*x
  (p 0 = 0) ∧ 
  (p (-1/2) = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3180_318064


namespace NUMINAMATH_CALUDE_apples_order_calculation_l3180_318045

/-- The number of apples Chandler can eat per week -/
def chandler_apples_per_week : ℕ := 23

/-- The number of apples Lucy can eat per week -/
def lucy_apples_per_week : ℕ := 19

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The total number of apples needed for a month -/
def total_apples_per_month : ℕ := (chandler_apples_per_week + lucy_apples_per_week) * weeks_in_month

theorem apples_order_calculation :
  total_apples_per_month = 168 := by sorry

end NUMINAMATH_CALUDE_apples_order_calculation_l3180_318045


namespace NUMINAMATH_CALUDE_car_speed_problem_l3180_318080

/-- Proves that given a car traveling for two hours with an average speed of 40 km/h,
    and a speed of 60 km/h in the second hour, the speed in the first hour must be 20 km/h. -/
theorem car_speed_problem (speed_first_hour : ℝ) (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 60 →
  average_speed = 40 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_first_hour = 20 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3180_318080


namespace NUMINAMATH_CALUDE_stephanies_remaining_payment_l3180_318060

/-- Calculates the remaining amount to pay for Stephanie's bills -/
def remaining_payment (electricity_bill gas_bill water_bill internet_bill : ℚ)
  (gas_paid_fraction : ℚ) (gas_additional_payment : ℚ)
  (water_paid_fraction : ℚ) (internet_payments : ℕ) (internet_payment_amount : ℚ) : ℚ :=
  let gas_remaining := gas_bill - (gas_paid_fraction * gas_bill + gas_additional_payment)
  let water_remaining := water_bill - (water_paid_fraction * water_bill)
  let internet_remaining := internet_bill - (internet_payments : ℚ) * internet_payment_amount
  gas_remaining + water_remaining + internet_remaining

/-- Stephanie's remaining bill payment is $30 -/
theorem stephanies_remaining_payment :
  remaining_payment 60 40 40 25 (3/4) 5 (1/2) 4 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_remaining_payment_l3180_318060


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3180_318032

theorem smallest_solution_abs_equation :
  ∀ x : ℝ, x * |x| = 3 * x + 4 → x ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3180_318032


namespace NUMINAMATH_CALUDE_age_problem_l3180_318037

/-- Given the ages of Matt, Kaylee, Bella, and Alex, prove that they satisfy the given conditions -/
theorem age_problem (matt_age kaylee_age bella_age alex_age : ℕ) : 
  matt_age = 5 ∧ 
  kaylee_age + 7 = 3 * matt_age ∧ 
  kaylee_age + bella_age = matt_age + 9 ∧ 
  bella_age = alex_age + 3 →
  kaylee_age = 8 ∧ matt_age = 5 ∧ bella_age = 6 ∧ alex_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3180_318037


namespace NUMINAMATH_CALUDE_missing_number_equation_l3180_318044

theorem missing_number_equation : ∃! x : ℝ, x + 3699 + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l3180_318044


namespace NUMINAMATH_CALUDE_wrench_can_turn_bolt_l3180_318054

/-- Represents a wrench with a regular hexagonal shape -/
structure Wrench where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Represents a bolt with a square head -/
structure Bolt where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Condition for a wrench to turn a bolt -/
def canTurn (w : Wrench) (b : Bolt) : Prop :=
  Real.sqrt 3 / Real.sqrt 2 < b.sideLength / w.sideLength ∧ 
  b.sideLength / w.sideLength ≤ 3 - Real.sqrt 3

/-- Theorem stating the condition for a wrench to turn a bolt -/
theorem wrench_can_turn_bolt (w : Wrench) (b : Bolt) : 
  canTurn w b ↔ 
    (∃ (x : ℝ), b.sideLength = x * w.sideLength ∧ 
      Real.sqrt 3 / Real.sqrt 2 < x ∧ x ≤ 3 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_wrench_can_turn_bolt_l3180_318054


namespace NUMINAMATH_CALUDE_triangle_theorem_l3180_318072

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.cos t.C = 1/4)
  (h2 : t.a^2 = t.b^2 + (1/2) * t.c^2) :
  Real.sin (t.A - t.B) = Real.sqrt 15 / 8 ∧
  (t.c = Real.sqrt 10 → t.a = 3 ∧ t.b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3180_318072


namespace NUMINAMATH_CALUDE_mango_lassi_price_l3180_318094

/-- The cost of a mango lassi at Delicious Delhi restaurant --/
def mango_lassi_cost (samosa_cost pakora_cost tip_percentage total_cost : ℚ) : ℚ :=
  total_cost - (samosa_cost + pakora_cost + (samosa_cost + pakora_cost) * tip_percentage / 100)

/-- Theorem stating the cost of the mango lassi --/
theorem mango_lassi_price :
  mango_lassi_cost 6 12 25 25 = (5/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_mango_lassi_price_l3180_318094


namespace NUMINAMATH_CALUDE_prime_divides_29_power_plus_one_l3180_318029

theorem prime_divides_29_power_plus_one (p : ℕ) : 
  Nat.Prime p ∧ p ∣ 29^p + 1 ↔ p = 2 ∨ p = 3 ∨ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_29_power_plus_one_l3180_318029


namespace NUMINAMATH_CALUDE_percentage_problem_l3180_318033

theorem percentage_problem (N P : ℝ) : 
  N = 150 → N = (P / 100) * N + 126 → P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3180_318033


namespace NUMINAMATH_CALUDE_taylor_family_reunion_l3180_318059

theorem taylor_family_reunion (kids : ℕ) (tables : ℕ) (people_per_table : ℕ) (adults : ℕ) : 
  kids = 45 → tables = 14 → people_per_table = 12 → 
  adults = tables * people_per_table - kids → adults = 123 :=
by
  sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_l3180_318059


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3180_318046

theorem constant_term_expansion (x : ℝ) :
  let expansion := (Real.sqrt x + 2) * (1 / Real.sqrt x - 1)^5
  ∃ (f : ℝ → ℝ), expansion = f x + 3 ∧ (∀ y, f y ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3180_318046


namespace NUMINAMATH_CALUDE_square_on_parabola_diagonal_l3180_318057

/-- Given a square ABOC where O is the origin, A and B are on the parabola y = -x^2,
    and C is opposite to O, the length of diagonal AC is 2a, where a is the x-coordinate of point A. -/
theorem square_on_parabola_diagonal (a : ℝ) :
  let A : ℝ × ℝ := (a, -a^2)
  let B : ℝ × ℝ := (-a, -a^2)
  let O : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (a, a^2)
  -- ABOC is a square
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = (O.1 - C.1)^2 + (O.2 - C.2)^2 ∧
  (O.1 - C.1)^2 + (O.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
  -- Length of AC is 2a
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (2*a)^2 :=
by sorry


end NUMINAMATH_CALUDE_square_on_parabola_diagonal_l3180_318057


namespace NUMINAMATH_CALUDE_apples_ordered_per_month_l3180_318087

def chandler_initial : ℕ := 23
def lucy_initial : ℕ := 19
def ross_initial : ℕ := 15
def chandler_increase : ℕ := 2
def lucy_decrease : ℕ := 1
def weeks_per_month : ℕ := 4

def total_apples_month : ℕ :=
  (chandler_initial + (chandler_initial + chandler_increase) + 
   (chandler_initial + 2 * chandler_increase) + 
   (chandler_initial + 3 * chandler_increase)) +
  (lucy_initial + (lucy_initial - lucy_decrease) + 
   (lucy_initial - 2 * lucy_decrease) + 
   (lucy_initial - 3 * lucy_decrease)) +
  (ross_initial * weeks_per_month)

theorem apples_ordered_per_month : 
  total_apples_month = 234 := by sorry

end NUMINAMATH_CALUDE_apples_ordered_per_month_l3180_318087


namespace NUMINAMATH_CALUDE_jeff_average_skips_l3180_318075

-- Define the number of rounds
def num_rounds : ℕ := 4

-- Define Sam's skips per round
def sam_skips : ℕ := 16

-- Define Jeff's skips for each round
def jeff_round1 : ℕ := sam_skips - 1
def jeff_round2 : ℕ := sam_skips - 3
def jeff_round3 : ℕ := sam_skips + 4
def jeff_round4 : ℕ := sam_skips / 2

-- Define Jeff's total skips
def jeff_total : ℕ := jeff_round1 + jeff_round2 + jeff_round3 + jeff_round4

-- Theorem to prove
theorem jeff_average_skips :
  jeff_total / num_rounds = 14 := by sorry

end NUMINAMATH_CALUDE_jeff_average_skips_l3180_318075


namespace NUMINAMATH_CALUDE_whitney_max_sets_l3180_318090

/-- Represents the number of items Whitney has --/
structure ItemCounts where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ
  keychains : ℕ

/-- Represents the requirements for each set --/
structure SetRequirements where
  tshirts : ℕ
  buttonToStickerRatio : ℕ
  keychains : ℕ

/-- Calculates the maximum number of sets that can be made --/
def maxSets (items : ItemCounts) (reqs : SetRequirements) : ℕ :=
  min (items.tshirts / reqs.tshirts)
    (min (items.buttons / reqs.buttonToStickerRatio)
      (min (items.stickers)
        (items.keychains / reqs.keychains)))

/-- Theorem stating that the maximum number of sets Whitney can make is 7 --/
theorem whitney_max_sets :
  let items := ItemCounts.mk 7 36 15 21
  let reqs := SetRequirements.mk 1 4 3
  maxSets items reqs = 7 := by
  sorry


end NUMINAMATH_CALUDE_whitney_max_sets_l3180_318090


namespace NUMINAMATH_CALUDE_nikolai_silver_decrease_l3180_318038

/-- Represents the number of each type of coin --/
structure CoinCount where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Represents a transaction at the exchange point --/
inductive Transaction
  | Type1 : Transaction  -- 2 gold for 3 silver and 1 copper
  | Type2 : Transaction  -- 5 silver for 3 gold and 1 copper

/-- Applies a single transaction to a CoinCount --/
def applyTransaction (t : Transaction) (c : CoinCount) : CoinCount :=
  match t with
  | Transaction.Type1 => CoinCount.mk (c.gold - 2) (c.silver + 3) (c.copper + 1)
  | Transaction.Type2 => CoinCount.mk (c.gold + 3) (c.silver - 5) (c.copper + 1)

/-- Applies a list of transactions to an initial CoinCount --/
def applyTransactions (ts : List Transaction) (initial : CoinCount) : CoinCount :=
  ts.foldl (fun acc t => applyTransaction t acc) initial

theorem nikolai_silver_decrease (initialSilver : ℕ) :
  ∃ (ts : List Transaction),
    let final := applyTransactions ts (CoinCount.mk 0 initialSilver 0)
    final.gold = 0 ∧
    final.copper = 50 ∧
    initialSilver - final.silver = 10 := by
  sorry

end NUMINAMATH_CALUDE_nikolai_silver_decrease_l3180_318038


namespace NUMINAMATH_CALUDE_initial_number_proof_l3180_318051

theorem initial_number_proof (x : ℤ) : x - 12 * 3 * 2 = 9938 → x = 10010 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3180_318051


namespace NUMINAMATH_CALUDE_mark_spending_l3180_318022

/-- Represents the grocery items Mark buys -/
inductive GroceryItem
  | Apple
  | Bread
  | Cheese
  | Cereal

/-- Represents Mark's grocery shopping trip -/
structure GroceryShopping where
  prices : GroceryItem → ℕ
  quantities : GroceryItem → ℕ
  appleBuyOneGetOneFree : Bool
  couponValue : ℕ
  couponThreshold : ℕ

def calculateTotalSpending (shopping : GroceryShopping) : ℕ :=
  sorry

theorem mark_spending (shopping : GroceryShopping) 
  (h1 : shopping.prices GroceryItem.Apple = 2)
  (h2 : shopping.prices GroceryItem.Bread = 3)
  (h3 : shopping.prices GroceryItem.Cheese = 6)
  (h4 : shopping.prices GroceryItem.Cereal = 5)
  (h5 : shopping.quantities GroceryItem.Apple = 4)
  (h6 : shopping.quantities GroceryItem.Bread = 5)
  (h7 : shopping.quantities GroceryItem.Cheese = 3)
  (h8 : shopping.quantities GroceryItem.Cereal = 4)
  (h9 : shopping.appleBuyOneGetOneFree = true)
  (h10 : shopping.couponValue = 10)
  (h11 : shopping.couponThreshold = 50)
  : calculateTotalSpending shopping = 47 := by
  sorry

end NUMINAMATH_CALUDE_mark_spending_l3180_318022


namespace NUMINAMATH_CALUDE_cut_rectangle_properties_l3180_318024

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the result of cutting a smaller rectangle from a larger one -/
structure CutRectangle where
  original : Rectangle
  cut : Rectangle

/-- The resulting figure after cutting -/
def resultingFigure (cr : CutRectangle) : ℝ := area cr.original - area cr.cut

theorem cut_rectangle_properties (R : Rectangle) (S : CutRectangle) 
    (h1 : S.original = R) 
    (h2 : area S.cut > 0) 
    (h3 : S.cut.length < R.length ∧ S.cut.width < R.width) :
  resultingFigure S < area R ∧ perimeter R = perimeter S.original :=
by sorry

end NUMINAMATH_CALUDE_cut_rectangle_properties_l3180_318024


namespace NUMINAMATH_CALUDE_find_first_fraction_l3180_318061

def compound_ratio : ℚ := 0.07142857142857142
def second_fraction : ℚ := 1/3
def third_fraction : ℚ := 3/8

theorem find_first_fraction :
  ∃ (first_fraction : ℚ), first_fraction * second_fraction * third_fraction = compound_ratio :=
sorry

end NUMINAMATH_CALUDE_find_first_fraction_l3180_318061


namespace NUMINAMATH_CALUDE_total_amount_proof_l3180_318028

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalAmount (n50 : ℕ) (n500 : ℕ) : ℕ := n50 * 50 + n500 * 500

/-- Proves that the total amount of money is 10350 rupees given the specified conditions -/
theorem total_amount_proof :
  let total_notes : ℕ := 108
  let n50 : ℕ := 97
  let n500 : ℕ := total_notes - n50
  totalAmount n50 n500 = 10350 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_proof_l3180_318028


namespace NUMINAMATH_CALUDE_seventeen_doors_max_attempts_l3180_318056

/-- The maximum number of attempts needed to open n doors with n keys --/
def max_attempts (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 17 doors and 17 keys, the maximum number of attempts is 136 --/
theorem seventeen_doors_max_attempts :
  max_attempts 17 = 136 := by sorry

end NUMINAMATH_CALUDE_seventeen_doors_max_attempts_l3180_318056


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l3180_318068

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_uniqueness 
  (f : ℝ → ℝ) 
  (h_quad : is_quadratic f)
  (h_sol : ∀ x : ℝ, f x > 0 ↔ 0 < x ∧ x < 4)
  (h_max : ∀ x : ℝ, x ∈ Set.Icc (-1) 5 → f x ≤ 12)
  (h_attain : ∃ x : ℝ, x ∈ Set.Icc (-1) 5 ∧ f x = 12) :
  ∀ x : ℝ, f x = -3 * x^2 + 12 * x :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l3180_318068


namespace NUMINAMATH_CALUDE_breakfast_cost_theorem_l3180_318002

/-- Represents the cost of breakfast items and the special offer. -/
structure BreakfastPrices where
  toast : ℚ
  egg : ℚ
  coffee : ℚ
  orange_juice : ℚ
  special_offer : ℚ

/-- Represents an individual's breakfast order. -/
structure BreakfastOrder where
  toast : ℕ
  egg : ℕ
  coffee : ℕ
  orange_juice : ℕ

/-- Calculates the cost of a breakfast order given the prices. -/
def orderCost (prices : BreakfastPrices) (order : BreakfastOrder) : ℚ :=
  prices.toast * order.toast +
  prices.egg * order.egg +
  (if order.coffee ≥ 2 then prices.special_offer else prices.coffee * order.coffee) +
  prices.orange_juice * order.orange_juice

/-- Calculates the total cost of all breakfast orders with service charge. -/
def totalCost (prices : BreakfastPrices) (orders : List BreakfastOrder) (serviceCharge : ℚ) : ℚ :=
  let subtotal := (orders.map (orderCost prices)).sum
  subtotal + subtotal * serviceCharge

/-- Theorem stating that the total breakfast cost is £48.40. -/
theorem breakfast_cost_theorem (prices : BreakfastPrices)
    (dale andrew melanie kevin : BreakfastOrder) :
    prices.toast = 1 →
    prices.egg = 3 →
    prices.coffee = 2 →
    prices.orange_juice = 3/2 →
    prices.special_offer = 7/2 →
    dale = { toast := 2, egg := 2, coffee := 1, orange_juice := 0 } →
    andrew = { toast := 1, egg := 2, coffee := 0, orange_juice := 1 } →
    melanie = { toast := 3, egg := 1, coffee := 0, orange_juice := 2 } →
    kevin = { toast := 4, egg := 3, coffee := 2, orange_juice := 0 } →
    totalCost prices [dale, andrew, melanie, kevin] (1/10) = 484/10 := by
  sorry


end NUMINAMATH_CALUDE_breakfast_cost_theorem_l3180_318002


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3180_318089

-- Define the number of red and white balls
def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_white_balls

-- Define the probability of drawing a white ball
def prob_white_ball : ℚ := num_white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball :
  prob_white_ball = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3180_318089


namespace NUMINAMATH_CALUDE_perimeter_is_60_l3180_318008

/-- Square with side length 9 inches -/
def square_side_length : ℝ := 9

/-- Equilateral triangle with side length equal to square's side length -/
def triangle_side_length : ℝ := square_side_length

/-- Figure ABFCE formed after translating the triangle -/
structure Figure where
  AB : ℝ := square_side_length
  BF : ℝ := triangle_side_length
  FC : ℝ := triangle_side_length
  CE : ℝ := square_side_length
  EA : ℝ := square_side_length

/-- Perimeter of the figure ABFCE -/
def perimeter (fig : Figure) : ℝ :=
  fig.AB + fig.BF + fig.FC + fig.CE + fig.EA

/-- Theorem: The perimeter of figure ABFCE is 60 inches -/
theorem perimeter_is_60 (fig : Figure) : perimeter fig = 60 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_60_l3180_318008


namespace NUMINAMATH_CALUDE_hcd_8580_330_minus_12_l3180_318091

theorem hcd_8580_330_minus_12 : Nat.gcd 8580 330 - 12 = 318 := by
  sorry

end NUMINAMATH_CALUDE_hcd_8580_330_minus_12_l3180_318091


namespace NUMINAMATH_CALUDE_library_digital_format_l3180_318012

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for a book being available in digital format
variable (isDigital : Book → Prop)

-- Define the theorem
theorem library_digital_format (h : ¬∀ (b : Book), isDigital b) :
  (∃ (b : Book), ¬isDigital b) ∧ (¬∀ (b : Book), isDigital b) := by
  sorry

end NUMINAMATH_CALUDE_library_digital_format_l3180_318012


namespace NUMINAMATH_CALUDE_tom_swim_time_l3180_318020

/-- Proves that Tom swam for 2 hours given the conditions of the problem -/
theorem tom_swim_time (swim_speed : ℝ) (run_speed_multiplier : ℝ) (total_distance : ℝ) :
  swim_speed = 2 →
  run_speed_multiplier = 4 →
  total_distance = 12 →
  ∃ (swim_time : ℝ),
    swim_time * swim_speed + (swim_time / 2) * (run_speed_multiplier * swim_speed) = total_distance ∧
    swim_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_swim_time_l3180_318020


namespace NUMINAMATH_CALUDE_good_numbers_in_set_l3180_318049

/-- A number n is a "good number" if there exists a permutation of 1..n such that
    k + a[k] is a perfect square for all k in 1..n -/
def is_good_number (n : ℕ) : Prop :=
  ∃ a : Fin n → Fin n, Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k : ℕ) + (a k : ℕ) + 1 = m * m

theorem good_numbers_in_set : 
  is_good_number 13 ∧ 
  is_good_number 15 ∧ 
  is_good_number 17 ∧ 
  is_good_number 19 ∧ 
  ¬is_good_number 11 := by
  sorry

#check good_numbers_in_set

end NUMINAMATH_CALUDE_good_numbers_in_set_l3180_318049


namespace NUMINAMATH_CALUDE_crayons_left_l3180_318079

theorem crayons_left (initial_crayons : ℕ) (percentage_lost : ℚ) : 
  initial_crayons = 253 → 
  percentage_lost = 35.5 / 100 →
  ↑⌊initial_crayons - percentage_lost * initial_crayons⌋ = 163 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l3180_318079


namespace NUMINAMATH_CALUDE_cos_difference_l3180_318001

theorem cos_difference (α : Real) (h : α = 2 * Real.pi / 3) :
  Real.cos (α + Real.pi / 2) - Real.cos (Real.pi + α) = -(Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_l3180_318001


namespace NUMINAMATH_CALUDE_expression_simplification_l3180_318076

theorem expression_simplification (x : ℝ) : 
  (12 * x^12 - 3 * x^10 + 5 * x^9) + (-x^12 + 2 * x^10 + x^9 + 4 * x^4 + 6 * x^2 + 9) = 
  11 * x^12 - x^10 + 6 * x^9 + 4 * x^4 + 6 * x^2 + 9 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3180_318076


namespace NUMINAMATH_CALUDE_parabola_comparison_l3180_318093

theorem parabola_comparison : ∀ x : ℝ, 
  x^2 - (1/3)*x + 3 < x^2 + (1/3)*x + 4 := by sorry

end NUMINAMATH_CALUDE_parabola_comparison_l3180_318093


namespace NUMINAMATH_CALUDE_average_after_17th_inning_l3180_318016

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (runsScored : Nat) : Rat :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 3 after scoring 66 runs in the 17th inning, 
    then his average after the 17th inning is 18 -/
theorem average_after_17th_inning 
  (stats : BatsmanStats) 
  (h1 : stats.innings = 16)
  (h2 : newAverage stats 66 = stats.average + 3) :
  newAverage stats 66 = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_after_17th_inning_l3180_318016


namespace NUMINAMATH_CALUDE_max_graduates_of_interest_l3180_318005

theorem max_graduates_of_interest (total_graduates : ℕ) (num_universities : ℕ) 
  (h_total : total_graduates = 100)
  (h_unis : num_universities = 5)
  (h_half_reached : ∀ u, u ≤ num_universities → (total_graduates / 2 : ℝ) = total_graduates / 2) :
  ∃ (max_interest : ℕ), max_interest ≤ 83 ∧ 
  (∀ n : ℕ, (n : ℝ) * 3 + (total_graduates - n : ℝ) * 2 ≤ (total_graduates : ℝ) * (num_universities : ℝ) / 2 → n ≤ max_interest) :=
sorry

end NUMINAMATH_CALUDE_max_graduates_of_interest_l3180_318005
