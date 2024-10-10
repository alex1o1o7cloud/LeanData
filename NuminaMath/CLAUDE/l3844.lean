import Mathlib

namespace rectangular_plot_ratio_l3844_384417

/-- For a rectangular plot with given conditions, prove the ratio of area to breadth -/
theorem rectangular_plot_ratio (b l : ℝ) (h1 : b = 5) (h2 : l - b = 10) : 
  (l * b) / b = 15 := by
  sorry

end rectangular_plot_ratio_l3844_384417


namespace point_B_coordinates_l3844_384434

-- Define the points and lines
def A : ℝ × ℝ := (0, -1)
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- State the theorem
theorem point_B_coordinates :
  ∀ B : ℝ × ℝ,
  line1 B.1 B.2 →
  perpendicular ((B.2 - A.2) / (B.1 - A.1)) (-1/2) →
  B = (2, 3) := by
sorry


end point_B_coordinates_l3844_384434


namespace gcd_factorial_eight_and_factorial_six_squared_l3844_384477

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 2880 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l3844_384477


namespace smallest_yellow_marbles_l3844_384451

theorem smallest_yellow_marbles (total : ℕ) (blue red green yellow : ℕ) : 
  blue = total / 5 →
  red = 2 * green →
  green = 10 →
  blue + red + green + yellow = total →
  yellow ≥ 10 ∧ ∀ y : ℕ, y < 10 → ¬(
    ∃ t : ℕ, t / 5 + 2 * 10 + 10 + y = t ∧ 
    t / 5 + 2 * 10 + 10 + y = blue + red + green + y
  ) :=
by sorry

end smallest_yellow_marbles_l3844_384451


namespace book_cost_l3844_384419

theorem book_cost (book bookmark : ℝ) 
  (total_cost : book + bookmark = 2.10)
  (price_difference : book = bookmark + 2) :
  book = 2.05 := by
sorry

end book_cost_l3844_384419


namespace vector_sum_equals_l3844_384450

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := 2 • a + b

theorem vector_sum_equals : c = (5, 0) := by sorry

end vector_sum_equals_l3844_384450


namespace mean_median_difference_zero_l3844_384430

/-- Represents the score distribution in a classroom --/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score90 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score90 + score95 = 1

/-- Calculates the mean score given a score distribution --/
def mean_score (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 95 * d.score95

/-- Calculates the median score given a score distribution --/
def median_score (d : ScoreDistribution) : ℝ := 85

/-- The main theorem stating that the difference between mean and median is zero --/
theorem mean_median_difference_zero (d : ScoreDistribution) :
  d.score60 = 0.05 →
  d.score75 = 0.20 →
  d.score85 = 0.30 →
  d.score90 = 0.25 →
  mean_score d - median_score d = 0 := by
  sorry

end mean_median_difference_zero_l3844_384430


namespace intersection_point_d_equals_two_l3844_384425

/-- A function f(x) = 4x + c where c is an integer -/
def f (c : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + c

/-- The inverse of f -/
noncomputable def f_inv (c : ℤ) : ℝ → ℝ := λ x ↦ (x - c) / 4

theorem intersection_point_d_equals_two (c d : ℤ) :
  f c 2 = d ∧ f_inv c d = 2 → d = 2 := by sorry

end intersection_point_d_equals_two_l3844_384425


namespace simplify_expression_solve_cubic_equation_l3844_384490

-- Problem 1
theorem simplify_expression (a b : ℝ) : 2*a*(a-2*b) - (2*a-b)^2 = -2*a^2 - b^2 := by
  sorry

-- Problem 2
theorem solve_cubic_equation : ∃ x : ℝ, (x-1)^3 - 3 = 3/8 ∧ x = 5/2 := by
  sorry

end simplify_expression_solve_cubic_equation_l3844_384490


namespace ginger_wears_size_8_l3844_384468

def anna_size : ℕ := 2

def becky_size (anna : ℕ) : ℕ := 3 * anna

def ginger_size (becky : ℕ) : ℕ := 2 * becky - 4

theorem ginger_wears_size_8 : 
  ginger_size (becky_size anna_size) = 8 := by sorry

end ginger_wears_size_8_l3844_384468


namespace counterexample_exists_l3844_384439

theorem counterexample_exists : ∃ (a b c d : ℝ), 
  ((a + b) / (3*a - b) = (b + c) / (3*b - c)) ∧
  ((b + c) / (3*b - c) = (c + d) / (3*c - d)) ∧
  ((c + d) / (3*c - d) = (d + a) / (3*d - a)) ∧
  (3*a - b ≠ 0) ∧ (3*b - c ≠ 0) ∧ (3*c - d ≠ 0) ∧ (3*d - a ≠ 0) ∧
  (a^2 + b^2 + c^2 + d^2 ≠ a*b + b*c + c*d + d*a) :=
sorry

end counterexample_exists_l3844_384439


namespace pi_sixth_to_degrees_l3844_384447

theorem pi_sixth_to_degrees : 
  (π / 6 : Real) * (180 / π) = 30 := by sorry

end pi_sixth_to_degrees_l3844_384447


namespace bus_ride_cost_l3844_384448

-- Define the cost of bus and train rides
def bus_cost : ℝ := 1.75
def train_cost : ℝ := bus_cost + 6.35

-- State the theorem
theorem bus_ride_cost : 
  (train_cost = bus_cost + 6.35) → 
  (train_cost + bus_cost = 9.85) → 
  (bus_cost = 1.75) :=
by
  sorry

end bus_ride_cost_l3844_384448


namespace min_sum_abs_min_sum_abs_achieved_l3844_384433

theorem min_sum_abs (x : ℝ) : 
  |x + 3| + |x + 4| + |x + 6| + |x + 8| ≥ 12 :=
by sorry

theorem min_sum_abs_achieved : 
  ∃ x : ℝ, |x + 3| + |x + 4| + |x + 6| + |x + 8| = 12 :=
by sorry

end min_sum_abs_min_sum_abs_achieved_l3844_384433


namespace seating_theorem_l3844_384410

/-- Number of seats in a row -/
def num_seats : ℕ := 7

/-- Number of people to be seated -/
def num_people : ℕ := 3

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements -/
theorem seating_theorem :
  seating_arrangements num_seats num_people = 100 :=
sorry

end seating_theorem_l3844_384410


namespace min_value_quadratic_l3844_384464

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 8*x - 6*y + 20 ≥ -5 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 8*a - 6*b + 20 = -5 := by
  sorry

end min_value_quadratic_l3844_384464


namespace quadratic_real_equal_roots_l3844_384444

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 5 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 5 = 0 → y = x) ↔ 
  (m = 2 - 2 * Real.sqrt 15 ∨ m = 2 + 2 * Real.sqrt 15) := by
sorry

end quadratic_real_equal_roots_l3844_384444


namespace pie_remainder_l3844_384491

theorem pie_remainder (carlos_share : ℝ) (maria_fraction : ℝ) : 
  carlos_share = 0.6 → 
  maria_fraction = 0.5 → 
  (1 - carlos_share) * (1 - maria_fraction) = 0.2 := by
sorry

end pie_remainder_l3844_384491


namespace root_transformation_l3844_384459

/-- Given a nonzero constant k and roots a, b, c, d of the equation kx^4 - 5kx - 12 = 0,
    the polynomial with roots (b+c+d)/(ka^2), (a+c+d)/(kb^2), (a+b+d)/(kc^2), (a+b+c)/(kd^2)
    is 12k^3x^4 - 5k^3x^3 - 1 = 0 -/
theorem root_transformation (k : ℝ) (a b c d : ℝ) : k ≠ 0 →
  (k * a^4 - 5*k*a - 12 = 0) →
  (k * b^4 - 5*k*b - 12 = 0) →
  (k * c^4 - 5*k*c - 12 = 0) →
  (k * d^4 - 5*k*d - 12 = 0) →
  ∃ (x : ℝ), 12*k^3*x^4 - 5*k^3*x^3 - 1 = 0 ∧
    (x = (b+c+d)/(k*a^2) ∨ x = (a+c+d)/(k*b^2) ∨ x = (a+b+d)/(k*c^2) ∨ x = (a+b+c)/(k*d^2)) :=
by sorry

end root_transformation_l3844_384459


namespace lakeisha_lawn_mowing_l3844_384488

/-- The amount LaKeisha charges per square foot of lawn -/
def charge_per_sqft : ℚ := 1/10

/-- The cost of the book set -/
def book_cost : ℚ := 150

/-- The length of each lawn -/
def lawn_length : ℕ := 20

/-- The width of each lawn -/
def lawn_width : ℕ := 15

/-- The number of lawns already mowed -/
def lawns_mowed : ℕ := 3

/-- The additional square feet LaKeisha needs to mow -/
def additional_sqft : ℕ := 600

theorem lakeisha_lawn_mowing :
  (lawn_length * lawn_width * lawns_mowed * charge_per_sqft) + 
  (additional_sqft * charge_per_sqft) = book_cost :=
sorry

end lakeisha_lawn_mowing_l3844_384488


namespace f_derivative_at_zero_l3844_384411

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x + 1)

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end f_derivative_at_zero_l3844_384411


namespace bicycle_sale_loss_percentage_l3844_384463

theorem bicycle_sale_loss_percentage
  (profit_A_to_B : ℝ)
  (profit_A_to_C : ℝ)
  (h1 : profit_A_to_B = 0.30)
  (h2 : profit_A_to_C = 0.040000000000000036) :
  ∃ (loss_B_to_C : ℝ), loss_B_to_C = 0.20 ∧ 
    (1 + profit_A_to_C) = (1 + profit_A_to_B) * (1 - loss_B_to_C) := by
  sorry

end bicycle_sale_loss_percentage_l3844_384463


namespace abracadabra_anagram_count_l3844_384443

/-- Represents the frequency of each letter in a word -/
structure LetterFrequency where
  a : Nat
  b : Nat
  r : Nat
  c : Nat
  d : Nat

/-- Calculates the number of anagrams for a word with given letter frequencies -/
def anagramCount (freq : LetterFrequency) : Nat :=
  Nat.factorial 11 / (Nat.factorial freq.a * Nat.factorial freq.b * Nat.factorial freq.r)

/-- The letter frequency of "ABRACADABRA" -/
def abracadabraFreq : LetterFrequency := {
  a := 5,
  b := 2,
  r := 2,
  c := 1,
  d := 1
}

theorem abracadabra_anagram_count :
  anagramCount abracadabraFreq = 83160 := by sorry

end abracadabra_anagram_count_l3844_384443


namespace line_equation_l3844_384497

/-- Given a line parameterized by (x, y) = (3t + 6, 5t - 10) where t is a real number,
    prove that the equation of this line in the form y = mx + b is y = (5/3)x - 20. -/
theorem line_equation (t x y : ℝ) : 
  (x = 3 * t + 6 ∧ y = 5 * t - 10) → 
  y = (5/3) * x - 20 := by sorry

end line_equation_l3844_384497


namespace stating_valid_arrangements_count_l3844_384435

/-- 
Given n players with distinct heights, this function returns the number of ways to 
arrange them such that for each player, the total number of players either to their 
left and taller or to their right and shorter is even.
-/
def validArrangements (n : ℕ) : ℕ :=
  (n / 2).factorial * ((n + 1) / 2).factorial

/-- 
Theorem stating that the number of valid arrangements for n players
is equal to ⌊n/2⌋! * ⌈n/2⌉!
-/
theorem valid_arrangements_count (n : ℕ) :
  validArrangements n = (n / 2).factorial * ((n + 1) / 2).factorial := by
  sorry

end stating_valid_arrangements_count_l3844_384435


namespace area_of_gray_part_l3844_384401

/-- Given two overlapping rectangles, prove the area of the gray part -/
theorem area_of_gray_part (rect1_width rect1_height rect2_width rect2_height black_area : ℕ) 
  (h1 : rect1_width = 8)
  (h2 : rect1_height = 10)
  (h3 : rect2_width = 12)
  (h4 : rect2_height = 9)
  (h5 : black_area = 37) : 
  rect2_width * rect2_height - (rect1_width * rect1_height - black_area) = 65 := by
  sorry

#check area_of_gray_part

end area_of_gray_part_l3844_384401


namespace middle_term_arithmetic_sequence_l3844_384456

def arithmetic_sequence (a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d ∧ a₅ - a₄ = d

theorem middle_term_arithmetic_sequence :
  ∀ (a c : ℝ), arithmetic_sequence 17 a 29 c 41 → 29 = (17 + 41) / 2 :=
by sorry

end middle_term_arithmetic_sequence_l3844_384456


namespace special_line_equation_l3844_384496

/-- A line passing through (1,2) with its y-intercept twice its x-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1,2) -/
  passes_through : m + b = 2
  /-- The y-intercept is twice the x-intercept -/
  intercept_condition : b = 2 * (-b / m)

/-- The equation of the special line is either y = 2x or 2x + y - 4 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = -2 ∧ l.b = 4) := by
  sorry

end special_line_equation_l3844_384496


namespace train_platform_length_equality_l3844_384432

def train_speed : ℝ := 144  -- km/hr
def crossing_time : ℝ := 1  -- minute
def train_length : ℝ := 1200  -- meters

theorem train_platform_length_equality :
  let platform_length := train_speed * 1000 / 60 * crossing_time - train_length
  platform_length = train_length :=
by sorry

end train_platform_length_equality_l3844_384432


namespace m_range_theorem_l3844_384454

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2)

/-- Line and parabola have no intersections -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x y : ℝ, x - 2*y + 3 = 0 → y^2 = m*x → m ≠ 0 → False

/-- Equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (5 - 2*m) + y^2 / m = 1 → m * (5 - 2*m) < 0

/-- Main theorem -/
theorem m_range_theorem (m : ℝ) :
  (no_intersection m ∨ is_hyperbola m) ∧ ¬(no_intersection m ∧ is_hyperbola m) →
  m_range m :=
sorry

end m_range_theorem_l3844_384454


namespace employee_remaining_hours_l3844_384402

/-- Calculates the remaining hours for an employee who uses half of their allotted sick and vacation days --/
def remaining_hours (sick_days : ℕ) (vacation_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  let remaining_sick_days := sick_days / 2
  let remaining_vacation_days := vacation_days / 2
  (remaining_sick_days + remaining_vacation_days) * hours_per_day

/-- Proves that an employee with 10 sick days and 10 vacation days, using half of each, has 80 hours left --/
theorem employee_remaining_hours :
  remaining_hours 10 10 8 = 80 := by
  sorry

end employee_remaining_hours_l3844_384402


namespace yardley_snowfall_l3844_384469

/-- The total snowfall in Yardley throughout the day -/
def total_snowfall (early_morning late_morning afternoon evening : Real) : Real :=
  early_morning + late_morning + afternoon + evening

/-- Theorem: The total snowfall in Yardley is 1.22 inches -/
theorem yardley_snowfall :
  total_snowfall 0.12 0.24 0.5 0.36 = 1.22 := by
  sorry

end yardley_snowfall_l3844_384469


namespace square_inequality_l3844_384431

theorem square_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end square_inequality_l3844_384431


namespace successive_integers_product_l3844_384442

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 4160 → n = 64 := by
  sorry

end successive_integers_product_l3844_384442


namespace fourth_root_simplification_l3844_384487

theorem fourth_root_simplification :
  (2^8 * 3^2 * 5^3)^(1/4 : ℝ) = 4 * (1125 : ℝ)^(1/4 : ℝ) := by
  sorry

end fourth_root_simplification_l3844_384487


namespace museum_trip_total_l3844_384423

/-- The total number of people going to the museum on four buses -/
def total_people (first_bus : ℕ) : ℕ :=
  let second_bus := 2 * first_bus
  let third_bus := second_bus - 6
  let fourth_bus := first_bus + 9
  first_bus + second_bus + third_bus + fourth_bus

/-- Theorem: Given the conditions about the four buses, 
    the total number of people going to the museum is 75 -/
theorem museum_trip_total : total_people 12 = 75 := by
  sorry

end museum_trip_total_l3844_384423


namespace photographer_photos_to_include_l3844_384460

/-- Given a photographer with pre-selected photos and choices to provide photos,
    calculate the number of photos to include in an envelope. -/
def photos_to_include (pre_selected : ℕ) (choices : ℕ) : ℕ :=
  choices / pre_selected

/-- Theorem stating that for a photographer with 7 pre-selected photos and 56 choices,
    the number of photos to include is 8. -/
theorem photographer_photos_to_include :
  photos_to_include 7 56 = 8 := by
  sorry

end photographer_photos_to_include_l3844_384460


namespace sqrt_four_twentyfifths_equals_two_fifths_l3844_384482

theorem sqrt_four_twentyfifths_equals_two_fifths : 
  Real.sqrt (4 / 25) = 2 / 5 := by
  sorry

end sqrt_four_twentyfifths_equals_two_fifths_l3844_384482


namespace total_score_approximation_l3844_384467

/-- Represents the types of shots in a basketball game -/
inductive ShotType
  | ThreePoint
  | TwoPoint
  | FreeThrow

/-- Represents the success rate for each shot type -/
def successRate (shot : ShotType) : ℝ :=
  match shot with
  | ShotType.ThreePoint => 0.25
  | ShotType.TwoPoint => 0.50
  | ShotType.FreeThrow => 0.80

/-- Represents the point value for each shot type -/
def pointValue (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.ThreePoint => 3
  | ShotType.TwoPoint => 2
  | ShotType.FreeThrow => 1

/-- The total number of shots attempted -/
def totalShots : ℕ := 40

/-- Calculates the number of attempts for each shot type, assuming equal distribution -/
def attemptsPerType : ℕ := totalShots / 3

/-- Calculates the points scored for a given shot type -/
def pointsScored (shot : ShotType) : ℝ :=
  (successRate shot) * (pointValue shot : ℝ) * (attemptsPerType : ℝ)

/-- Calculates the total points scored across all shot types -/
def totalPointsScored : ℝ :=
  pointsScored ShotType.ThreePoint + pointsScored ShotType.TwoPoint + pointsScored ShotType.FreeThrow

/-- Theorem stating that the total points scored is approximately 33 -/
theorem total_score_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |totalPointsScored - 33| < ε :=
sorry

end total_score_approximation_l3844_384467


namespace max_distance_on_circle_common_chord_equation_three_common_tangents_l3844_384414

-- Define the circles
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0
def circle_C3 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle_C4 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y + 4 = 0

-- Theorem 1
theorem max_distance_on_circle :
  ∀ x₁ y₁ : ℝ, circle_C x₁ y₁ → (∀ x y : ℝ, circle_C x y → (x - 1)^2 + (y - 2*Real.sqrt 2)^2 ≤ (x₁ - 1)^2 + (y₁ - 2*Real.sqrt 2)^2) →
  (x₁ - 1)^2 + (y₁ - 2*Real.sqrt 2)^2 = 25 :=
sorry

-- Theorem 2
theorem common_chord_equation :
  ∀ x y : ℝ, (circle_C1 x y ∧ circle_C2 x y) → x - 2*y + 6 = 0 :=
sorry

-- Theorem 3
theorem three_common_tangents :
  ∃! n : ℕ, n = 3 ∧ 
  (∀ l : ℝ → ℝ → Prop, (∀ x y : ℝ, (circle_C3 x y → l x y) ∧ (circle_C4 x y → l x y)) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂)) →
  n = 3 :=
sorry

end max_distance_on_circle_common_chord_equation_three_common_tangents_l3844_384414


namespace candy_block_pieces_l3844_384474

/-- The number of candy pieces per block in Jan's candy necklace problem -/
def candy_pieces_per_block (total_necklaces : ℕ) (pieces_per_necklace : ℕ) (total_blocks : ℕ) : ℕ :=
  (total_necklaces * pieces_per_necklace) / total_blocks

/-- Theorem stating that the number of candy pieces per block is 30 -/
theorem candy_block_pieces :
  candy_pieces_per_block 9 10 3 = 30 := by
  sorry

end candy_block_pieces_l3844_384474


namespace solve_linear_equation_l3844_384475

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 → x = -7/3 := by
  sorry

end solve_linear_equation_l3844_384475


namespace weight_of_replaced_person_l3844_384492

/-- Prove that in a group of 8 persons, if the average weight increases by 2.5 kg
    when a new person weighing 90 kg replaces one of them,
    then the weight of the replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (original_group_size : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : original_group_size = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 90)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end weight_of_replaced_person_l3844_384492


namespace specific_pyramid_volume_l3844_384445

/-- A triangular pyramid with mutually perpendicular lateral edges -/
structure TriangularPyramid where
  /-- The area of the first lateral face -/
  area1 : ℝ
  /-- The area of the second lateral face -/
  area2 : ℝ
  /-- The area of the third lateral face -/
  area3 : ℝ
  /-- The lateral edges are mutually perpendicular -/
  perpendicular : True

/-- The volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of the specific triangular pyramid is 2 cm³ -/
theorem specific_pyramid_volume :
  let p : TriangularPyramid := {
    area1 := 1.5,
    area2 := 2,
    area3 := 6,
    perpendicular := trivial
  }
  volume p = 2 := by sorry

end specific_pyramid_volume_l3844_384445


namespace min_balls_to_draw_correct_l3844_384424

/-- Represents the number of balls of each color in the container -/
structure BallContainer :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (purple : ℕ)
  (orange : ℕ)

/-- The initial distribution of balls in the container -/
def initialContainer : BallContainer :=
  { red := 40
  , green := 25
  , yellow := 20
  , blue := 15
  , purple := 10
  , orange := 5 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : ℕ := 18

/-- Function to calculate the minimum number of balls to draw -/
def minBallsToDraw (container : BallContainer) (target : ℕ) : ℕ :=
  sorry

theorem min_balls_to_draw_correct :
  minBallsToDraw initialContainer targetCount = 82 :=
sorry

end min_balls_to_draw_correct_l3844_384424


namespace same_last_five_digits_l3844_384472

theorem same_last_five_digits (N : ℕ) : N = 3125 ↔ 
  (N > 0) ∧ 
  (∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    N % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (N^2) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) ∧
  (∀ M : ℕ, M < N → 
    (M > 0) → 
    (∀ (a b c d e : ℕ), 
      a ≠ 0 → 
      a < 10 → b < 10 → c < 10 → d < 10 → e < 10 →
      M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e →
      (M^2) % 100000 ≠ a * 10000 + b * 1000 + c * 100 + d * 10 + e)) :=
by sorry

end same_last_five_digits_l3844_384472


namespace linear_equation_exponent_relation_l3844_384489

/-- If 2x^(m-1) + 3y^(2n-1) = 7 is a linear equation in x and y, then m - 2n = 0 -/
theorem linear_equation_exponent_relation (m n : ℕ) :
  (∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(m-1) + 3 * y^(2*n-1) = a * x + b * y + c) →
  m - 2*n = 0 := by
sorry

end linear_equation_exponent_relation_l3844_384489


namespace square_root_of_64_l3844_384484

theorem square_root_of_64 : {x : ℝ | x^2 = 64} = {8, -8} := by sorry

end square_root_of_64_l3844_384484


namespace star_arrangements_l3844_384408

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    where rotations are considered equivalent but reflections are not. -/
theorem star_arrangements : (factorial 12) / 6 = 79833600 := by
  sorry

end star_arrangements_l3844_384408


namespace yoojungs_initial_candies_l3844_384405

/-- The number of candies Yoojung gave to her older sister -/
def candies_to_older_sister : ℕ := 7

/-- The number of candies Yoojung gave to her younger sister -/
def candies_to_younger_sister : ℕ := 6

/-- The number of candies Yoojung had left after giving candies to her sisters -/
def candies_left : ℕ := 15

/-- The initial number of candies Yoojung had -/
def initial_candies : ℕ := candies_to_older_sister + candies_to_younger_sister + candies_left

theorem yoojungs_initial_candies : initial_candies = 28 := by
  sorry

end yoojungs_initial_candies_l3844_384405


namespace exists_divisible_by_3_and_19_l3844_384457

theorem exists_divisible_by_3_and_19 : ∃ x : ℝ, ∃ m n : ℤ, x = 3 * m ∧ x = 19 * n := by
  sorry

end exists_divisible_by_3_and_19_l3844_384457


namespace cubic_root_sum_l3844_384409

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 10*a^2 + 16*a - 2 = 0 →
  b^3 - 10*b^2 + 16*b - 2 = 0 →
  c^3 - 10*c^2 + 16*c - 2 = 0 →
  (a / (b*c + 2)) + (b / (a*c + 2)) + (c / (a*b + 2)) = 4 := by
sorry

end cubic_root_sum_l3844_384409


namespace profit_to_cost_ratio_l3844_384416

theorem profit_to_cost_ratio (sale_price cost_price : ℚ) : 
  sale_price > 0 ∧ cost_price > 0 ∧ sale_price / cost_price = 6 / 2 → 
  (sale_price - cost_price) / cost_price = 2 / 1 := by
  sorry

end profit_to_cost_ratio_l3844_384416


namespace carnival_prize_percentage_carnival_prize_percentage_proof_l3844_384476

theorem carnival_prize_percentage (total_minnows : ℕ) (minnows_per_prize : ℕ) 
  (total_players : ℕ) (leftover_minnows : ℕ) : ℕ → Prop :=
  λ percentage_winners =>
    total_minnows = 600 ∧
    minnows_per_prize = 3 ∧
    total_players = 800 ∧
    leftover_minnows = 240 →
    percentage_winners = 15 ∧
    (total_minnows - leftover_minnows) / minnows_per_prize * 100 / total_players = percentage_winners

-- Proof
theorem carnival_prize_percentage_proof : 
  ∃ (percentage_winners : ℕ), carnival_prize_percentage 600 3 800 240 percentage_winners :=
by
  sorry

end carnival_prize_percentage_carnival_prize_percentage_proof_l3844_384476


namespace green_candy_pieces_l3844_384420

theorem green_candy_pieces (total red blue : ℝ) (h1 : total = 3409.7) (h2 : red = 145.5) (h3 : blue = 785.2) :
  total - red - blue = 2479 := by
  sorry

end green_candy_pieces_l3844_384420


namespace trapezoid_total_area_l3844_384499

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the total possible area of the trapezoid with different configurations -/
def totalPossibleArea (t : Trapezoid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_total_area :
  let t := Trapezoid.mk 4 6 8 10
  totalPossibleArea t = 48 * Real.sqrt 2 := by
  sorry

end trapezoid_total_area_l3844_384499


namespace particle_speeds_l3844_384473

-- Define the distance between points A and B in centimeters
def distance : ℝ := 301

-- Define the time when m2 starts moving after m1 leaves A
def start_time : ℝ := 11

-- Define the times of the two meetings after m2 starts moving
def first_meeting : ℝ := 10
def second_meeting : ℝ := 45

-- Define the speeds of particles m1 and m2
def speed_m1 : ℝ := 11
def speed_m2 : ℝ := 7

-- Theorem statement
theorem particle_speeds :
  -- Condition: At the first meeting, the total distance covered equals the initial distance
  (distance - start_time * speed_m1 = first_meeting * (speed_m1 + speed_m2)) ∧
  -- Condition: The relative movement between the two meetings
  (2 * first_meeting * speed_m2 = (second_meeting - first_meeting) * (speed_m1 - speed_m2)) →
  -- Conclusion: The speeds are correct
  speed_m1 = 11 ∧ speed_m2 = 7 := by
  sorry

end particle_speeds_l3844_384473


namespace cheaper_candy_price_l3844_384429

/-- Proves that the price of the cheaper candy is $2 per pound -/
theorem cheaper_candy_price
  (total_weight : ℝ)
  (mixture_price : ℝ)
  (cheaper_weight : ℝ)
  (expensive_price : ℝ)
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheaper_weight = 64)
  (h4 : expensive_price = 3)
  : ∃ (cheaper_price : ℝ),
    cheaper_price * cheaper_weight + expensive_price * (total_weight - cheaper_weight) =
    mixture_price * total_weight ∧ cheaper_price = 2 := by
  sorry

end cheaper_candy_price_l3844_384429


namespace hours_in_year_correct_hours_in_year_l3844_384493

theorem hours_in_year : ℕ → ℕ → ℕ → Prop :=
  fun hours_per_day days_per_year hours_per_year =>
    hours_per_day = 24 ∧ days_per_year = 365 →
    hours_per_year = hours_per_day * days_per_year

theorem correct_hours_in_year : hours_in_year 24 365 8760 := by
  sorry

end hours_in_year_correct_hours_in_year_l3844_384493


namespace circle_ratio_theorem_l3844_384412

noncomputable def circle_ratio : ℝ :=
  let r1 : ℝ := Real.sqrt 2
  let r2 : ℝ := 2
  let d : ℝ := Real.sqrt 3 + 1
  let common_area : ℝ := (7 * Real.pi - 6 * (Real.sqrt 3 + 1)) / 6
  let inscribed_radius : ℝ := (Real.sqrt 2 + 1 - Real.sqrt 3) / 2
  let inscribed_area : ℝ := Real.pi * inscribed_radius ^ 2
  inscribed_area / common_area

theorem circle_ratio_theorem : circle_ratio = 
  (3 * Real.pi * (3 + Real.sqrt 2 - Real.sqrt 3 - Real.sqrt 6)) / 
  (7 * Real.pi - 6 * (Real.sqrt 3 + 1)) := by sorry

end circle_ratio_theorem_l3844_384412


namespace remainder_problem_l3844_384452

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 41) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
  sorry

end remainder_problem_l3844_384452


namespace three_fractions_inequality_l3844_384458

theorem three_fractions_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) :
  (a - b*c) / (a + b*c) + (b - c*a) / (b + c*a) + (c - a*b) / (c + a*b) ≤ 3/2 := by
sorry

end three_fractions_inequality_l3844_384458


namespace election_vote_difference_l3844_384486

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 4400 →
  candidate_percentage = 30 / 100 →
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -1760 := by
  sorry

end election_vote_difference_l3844_384486


namespace xiaoxia_exceeds_xiaoming_l3844_384471

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  let xiaoxia_initial : ℤ := 52
  let xiaoming_initial : ℤ := 70
  let xiaoxia_monthly : ℤ := 15
  let xiaoming_monthly : ℤ := 12
  let xiaoxia_savings : ℤ := xiaoxia_initial + xiaoxia_monthly * n
  let xiaoming_savings : ℤ := xiaoming_initial + xiaoming_monthly * n
  xiaoxia_savings > xiaoming_savings ↔ 52 + 15 * n > 70 + 12 * n :=
by sorry

end xiaoxia_exceeds_xiaoming_l3844_384471


namespace wage_decrease_increase_l3844_384478

theorem wage_decrease_increase (initial_wage : ℝ) :
  let decreased_wage := initial_wage * (1 - 0.5)
  let final_wage := decreased_wage * (1 + 0.5)
  final_wage = initial_wage * 0.75 ∧ (initial_wage - final_wage) / initial_wage = 0.25 := by
  sorry

end wage_decrease_increase_l3844_384478


namespace iphone_price_calculation_l3844_384465

theorem iphone_price_calculation (P : ℝ) : 
  (P * (1 - 0.1) * (1 - 0.2) = 720) → P = 1000 := by
  sorry

end iphone_price_calculation_l3844_384465


namespace sqrt_calculations_l3844_384495

theorem sqrt_calculations :
  (∀ (x : ℝ), x ≥ 0 → Real.sqrt (x ^ 2) = x) ∧
  (Real.sqrt 21 * Real.sqrt 3 / Real.sqrt 7 = 3) := by
  sorry

end sqrt_calculations_l3844_384495


namespace unit_vectors_parallel_to_a_l3844_384455

def vector_a : ℝ × ℝ := (12, 5)

theorem unit_vectors_parallel_to_a :
  let magnitude := Real.sqrt (vector_a.1^2 + vector_a.2^2)
  let unit_vector := (vector_a.1 / magnitude, vector_a.2 / magnitude)
  (unit_vector = (12/13, 5/13) ∨ unit_vector = (-12/13, -5/13)) ∧
  (∀ v : ℝ × ℝ, (v.1^2 + v.2^2 = 1 ∧ ∃ k : ℝ, v = (k * vector_a.1, k * vector_a.2)) →
    (v = (12/13, 5/13) ∨ v = (-12/13, -5/13))) :=
by sorry

end unit_vectors_parallel_to_a_l3844_384455


namespace floor_plus_self_unique_solution_l3844_384400

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 22.7 :=
by sorry

end floor_plus_self_unique_solution_l3844_384400


namespace positive_integer_triplets_l3844_384436

theorem positive_integer_triplets :
  ∀ x y z : ℕ+,
    x ≤ y ∧ y ≤ z ∧ (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1 ↔
    (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end positive_integer_triplets_l3844_384436


namespace total_profit_is_35000_l3844_384481

/-- Represents the business subscription and profit distribution problem --/
structure BusinessProblem where
  total_subscription : ℕ
  a_more_than_b : ℕ
  b_more_than_c : ℕ
  c_profit : ℕ

/-- Calculates the total profit based on the given business problem --/
def calculate_total_profit (problem : BusinessProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the total profit is 35000 --/
theorem total_profit_is_35000 : 
  let problem := BusinessProblem.mk 50000 4000 5000 8400
  calculate_total_profit problem = 35000 := by
  sorry

end total_profit_is_35000_l3844_384481


namespace total_distance_traveled_l3844_384428

-- Define the speeds and conversion factors
def two_sail_speed : ℝ := 50
def one_sail_speed : ℝ := 25
def nautical_to_land_miles : ℝ := 1.15

-- Define the journey segments
def segment1_hours : ℝ := 2
def segment2_hours : ℝ := 3
def segment3_hours : ℝ := 1
def segment4_hours : ℝ := 2
def segment4_speed_reduction : ℝ := 0.3

-- Define the theorem
theorem total_distance_traveled :
  let segment1_distance := one_sail_speed * segment1_hours
  let segment2_distance := two_sail_speed * segment2_hours
  let segment3_distance := one_sail_speed * segment3_hours
  let segment4_distance := (one_sail_speed * (1 - segment4_speed_reduction)) * segment4_hours
  let total_nautical_miles := segment1_distance + segment2_distance + segment3_distance + segment4_distance
  let total_land_miles := total_nautical_miles * nautical_to_land_miles
  total_land_miles = 299 := by sorry

end total_distance_traveled_l3844_384428


namespace book_selling_price_l3844_384479

/-- Given a book with cost price CP, prove that the original selling price is 720 Rs. -/
theorem book_selling_price (CP : ℝ) : 
  (1.1 * CP = 880) →  -- Condition for 10% gain
  (∃ OSP, OSP = 0.9 * CP) →  -- Condition for 10% loss
  (∃ OSP, OSP = 720) :=
by sorry

end book_selling_price_l3844_384479


namespace first_supplier_cars_l3844_384403

theorem first_supplier_cars (total_production : ℕ) 
  (second_supplier_extra : ℕ) (fourth_fifth_supplier : ℕ) : 
  total_production = 5650000 →
  second_supplier_extra = 500000 →
  fourth_fifth_supplier = 325000 →
  ∃ (first_supplier : ℕ),
    first_supplier + 
    (first_supplier + second_supplier_extra) + 
    (first_supplier + (first_supplier + second_supplier_extra)) + 
    (2 * fourth_fifth_supplier) = total_production ∧
    first_supplier = 1000000 :=
by sorry

end first_supplier_cars_l3844_384403


namespace quadratic_inequality_solution_l3844_384483

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ 1 < x ∧ x < 2) →
  a + b = -1 := by
  sorry

end quadratic_inequality_solution_l3844_384483


namespace joan_balloons_l3844_384426

/-- The number of blue balloons Joan has now, given her initial count and the number lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

theorem joan_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end joan_balloons_l3844_384426


namespace no_universal_divisor_l3844_384485

-- Define a function to represent the concatenation of digits
def concat_digits (a b : ℕ) : ℕ := sorry

-- Define a function to represent the concatenation of three digits
def concat_three_digits (a n b : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_universal_divisor :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, 
    a ≠ 0 → b ≠ 0 → a < 10 → b < 10 → 
    (concat_three_digits a n b) % (concat_digits a b) = 0 := by sorry

end no_universal_divisor_l3844_384485


namespace decagon_triangle_probability_l3844_384421

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose n k

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 70

/-- The probability of a triangle having at least one side that is a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 7 / 12 := by sorry

end decagon_triangle_probability_l3844_384421


namespace product_equals_three_l3844_384462

theorem product_equals_three (a b c d : ℚ) 
  (ha : a + 3 = 3 * a)
  (hb : b + 4 = 4 * b)
  (hc : c + 5 = 5 * c)
  (hd : d + 6 = 6 * d) : 
  a * b * c * d = 3 := by
  sorry

end product_equals_three_l3844_384462


namespace common_root_quadratic_equations_l3844_384415

theorem common_root_quadratic_equations (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔
  (p = -3 ∨ p = 9) ∧
  ((p = -3 → ∃ x : ℝ, x = -1 ∧ x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ∧
   (p = 9 → ∃ x : ℝ, x = 3 ∧ x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0)) :=
by sorry

#check common_root_quadratic_equations

end common_root_quadratic_equations_l3844_384415


namespace class_average_l3844_384446

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) (high_score : ℕ) (remaining_avg : ℚ) : 
  total_students = 40 →
  high_scorers = 6 →
  zero_scorers = 9 →
  high_score = 98 →
  remaining_avg = 57 →
  (high_scorers * high_score + zero_scorers * 0 + (total_students - high_scorers - zero_scorers) * remaining_avg) / total_students = 50.325 := by
  sorry

#eval (6 * 98 + 9 * 0 + (40 - 6 - 9) * 57) / 40

end class_average_l3844_384446


namespace b_arrives_first_l3844_384437

theorem b_arrives_first (x y S : ℝ) (hx : x > 0) (hy : y > 0) (hS : S > 0) (hxy : x < y) :
  (S * (x + y)) / (2 * x * y) > (2 * S) / (x + y) := by
  sorry

end b_arrives_first_l3844_384437


namespace sum_equals_14x_l3844_384449

-- Define variables
variable (x y z : ℝ)

-- State the theorem
theorem sum_equals_14x (h1 : y = 3 * x) (h2 : z = 3 * y + x) : 
  x + y + z = 14 * x := by
  sorry

end sum_equals_14x_l3844_384449


namespace square_plus_reciprocal_square_l3844_384427

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1/x) = 4) : x^2 + (1/x^2) = 14 := by
  sorry

end square_plus_reciprocal_square_l3844_384427


namespace product_divisible_by_sum_implies_inequality_l3844_384498

theorem product_divisible_by_sum_implies_inequality 
  (m n : ℕ) 
  (h : (m * n) % (m + n) = 0) : 
  m + n ≤ n^2 := by
sorry

end product_divisible_by_sum_implies_inequality_l3844_384498


namespace hyperbola_minor_axis_length_l3844_384441

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the distance from the foci to the asymptote is 3,
    then the length of the minor axis is 6. -/
theorem hyperbola_minor_axis_length (b : ℝ) (h1 : b > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∃ d : ℝ, d = 3 ∧ d = b) →
  (∃ l : ℝ, l = 6 ∧ l = 2 * b) :=
by sorry

end hyperbola_minor_axis_length_l3844_384441


namespace one_thirteenth_150th_digit_l3844_384480

def decimal_representation (n : ℕ) : ℕ := 
  match n % 6 with
  | 1 => 0
  | 2 => 7
  | 3 => 6
  | 4 => 9
  | 5 => 2
  | 0 => 3
  | _ => 0  -- This case should never occur, but Lean requires it for exhaustiveness

theorem one_thirteenth_150th_digit : 
  decimal_representation 150 = 3 := by
sorry


end one_thirteenth_150th_digit_l3844_384480


namespace cube_difference_multiple_implies_sum_squares_multiple_of_sum_l3844_384453

theorem cube_difference_multiple_implies_sum_squares_multiple_of_sum
  (a b c : ℕ+)
  (ha : a < 2017)
  (hb : b < 2017)
  (hc : c < 2017)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hca : c ≠ a)
  (hab_multiple : ∃ k : ℤ, (a ^ 3 : ℤ) - (b ^ 3 : ℤ) = k * 2017)
  (hbc_multiple : ∃ k : ℤ, (b ^ 3 : ℤ) - (c ^ 3 : ℤ) = k * 2017)
  (hca_multiple : ∃ k : ℤ, (c ^ 3 : ℤ) - (a ^ 3 : ℤ) = k * 2017) :
  ∃ m : ℕ, (a ^ 2 + b ^ 2 + c ^ 2 : ℕ) = m * (a + b + c) :=
by sorry

end cube_difference_multiple_implies_sum_squares_multiple_of_sum_l3844_384453


namespace unique_integer_fraction_l3844_384422

theorem unique_integer_fraction : ∃! n : ℕ, 
  1 ≤ n ∧ n ≤ 2014 ∧ ∃ k : ℤ, 8 * n = k * (9999 - n) := by
  sorry

end unique_integer_fraction_l3844_384422


namespace problem_solution_l3844_384494

-- Define the conditions
def conditions (a b t : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ t = a * b

-- Theorem statement
theorem problem_solution (a b t : ℝ) (h : conditions a b t) :
  (0 < a ∧ a < 1) ∧
  (0 < t ∧ t ≤ 1/4) ∧
  ((a + 1/a) * (b + 1/b) ≥ 25/4) :=
by sorry

end problem_solution_l3844_384494


namespace max_value_z_l3844_384413

/-- The maximum value of z = x - 2y subject to constraints -/
theorem max_value_z (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : 2 * x + y ≤ 2) :
  ∃ (max_z : ℝ), max_z = 1 ∧ ∀ (z : ℝ), z = x - 2 * y → z ≤ max_z :=
by sorry

end max_value_z_l3844_384413


namespace factorization_equality_l3844_384470

theorem factorization_equality (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := by
  sorry

end factorization_equality_l3844_384470


namespace class_representation_ratio_l3844_384404

theorem class_representation_ratio :
  ∀ (num_boys num_girls : ℕ),
  num_boys > 0 →
  num_girls > 0 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3 / 5 * (num_girls : ℚ) / (num_boys + num_girls : ℚ) →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3 / 8 := by
sorry

end class_representation_ratio_l3844_384404


namespace symmetric_point_correct_l3844_384466

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := x + 3 * y - 10 = 0

/-- The original point -/
def original_point : ℝ × ℝ := (3, 9)

/-- The symmetric point -/
def symmetric_point : ℝ × ℝ := (-1, -3)

/-- Predicate to check if a point is symmetric to another point with respect to a line -/
def is_symmetric (p1 p2 : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line midpoint.1 midpoint.2 ∧
  (p2.2 - p1.2) * (p2.1 - p1.1) = -(1 / 3)

theorem symmetric_point_correct : 
  is_symmetric original_point symmetric_point line_of_symmetry :=
sorry

end symmetric_point_correct_l3844_384466


namespace divisibility_condition_l3844_384461

def is_divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

theorem divisibility_condition (a b : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9) →
  (is_divisible_by (62684 * 10 + a * 10 + b) 8 ∧ 
   is_divisible_by (62684 * 10 + a * 10 + b) 5) →
  (b = 0 ∧ (a = 0 ∨ a = 8)) := by
  sorry

#check divisibility_condition

end divisibility_condition_l3844_384461


namespace product_mod_400_l3844_384440

theorem product_mod_400 : (1567 * 2150) % 400 = 50 := by
  sorry

end product_mod_400_l3844_384440


namespace total_coins_is_21_l3844_384438

/-- The number of quarters in the wallet -/
def num_quarters : ℕ := 8

/-- The number of nickels in the wallet -/
def num_nickels : ℕ := 13

/-- The total number of coins in the wallet -/
def total_coins : ℕ := num_quarters + num_nickels

/-- Theorem stating that the total number of coins is 21 -/
theorem total_coins_is_21 : total_coins = 21 := by
  sorry

end total_coins_is_21_l3844_384438


namespace work_completion_time_l3844_384418

theorem work_completion_time (work : ℝ) (a b : ℝ) 
  (h1 : a + b = work / 6)  -- A and B together complete work in 6 days
  (h2 : a = work / 14)     -- A alone completes work in 14 days
  : b = work / 10.5        -- B alone completes work in 10.5 days
:= by sorry

end work_completion_time_l3844_384418


namespace minimal_shots_to_hit_triangle_l3844_384406

/-- A point on the circle --/
structure Point where
  index : Nat
  h_index : index ≥ 1 ∧ index ≤ 29

/-- A shot is a pair of distinct points --/
structure Shot where
  p1 : Point
  p2 : Point
  h_distinct : p1.index ≠ p2.index

/-- A triangle on the circle --/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point
  h_distinct : v1.index ≠ v2.index ∧ v2.index ≠ v3.index ∧ v3.index ≠ v1.index

/-- A function to determine if a shot hits a triangle --/
def hits (s : Shot) (t : Triangle) : Prop :=
  sorry -- Implementation details omitted

/-- The main theorem --/
theorem minimal_shots_to_hit_triangle :
  ∀ t : Triangle, ∃ K : Nat, K = 100 ∧
    (∀ shots : Finset Shot, shots.card = K →
      (∀ s ∈ shots, hits s t)) ∧
    (∀ K' : Nat, K' < K →
      ∃ shots : Finset Shot, shots.card = K' ∧
        ∃ s ∈ shots, ¬hits s t) :=
sorry

end minimal_shots_to_hit_triangle_l3844_384406


namespace min_balls_same_color_l3844_384407

/-- Given a bag with 6 balls each of 4 different colors, the minimum number of balls
    that must be drawn to ensure two balls of the same color are drawn is 5. -/
theorem min_balls_same_color (num_colors : ℕ) (balls_per_color : ℕ) :
  num_colors = 4 →
  balls_per_color = 6 →
  5 = Nat.succ num_colors :=
by sorry

end min_balls_same_color_l3844_384407
