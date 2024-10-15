import Mathlib

namespace NUMINAMATH_CALUDE_exists_divisible_with_at_most_1988_ones_l3529_352963

/-- A natural number is representable with at most 1988 ones if its binary representation
    has at most 1988 ones. -/
def representable_with_at_most_1988_ones (n : ℕ) : Prop :=
  (n.digits 2).count 1 ≤ 1988

/-- For any natural number M, there exists a natural number N that is
    representable with at most 1988 ones and is divisible by M. -/
theorem exists_divisible_with_at_most_1988_ones (M : ℕ) :
  ∃ N : ℕ, representable_with_at_most_1988_ones N ∧ M ∣ N :=
by sorry


end NUMINAMATH_CALUDE_exists_divisible_with_at_most_1988_ones_l3529_352963


namespace NUMINAMATH_CALUDE_no_solutions_abs_equation_l3529_352985

theorem no_solutions_abs_equation : ¬∃ y : ℝ, |y - 2| = |y - 1| + |y - 4| := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_abs_equation_l3529_352985


namespace NUMINAMATH_CALUDE_sum_mod_seven_l3529_352959

theorem sum_mod_seven : (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l3529_352959


namespace NUMINAMATH_CALUDE_nates_scallop_cost_l3529_352932

/-- Calculates the cost of scallops for a dinner party. -/
def scallop_cost (scallops_per_pound : ℕ) (cost_per_pound : ℚ) 
                 (scallops_per_person : ℕ) (num_people : ℕ) : ℚ :=
  let total_scallops := num_people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  pounds_needed * cost_per_pound

/-- The cost of scallops for Nate's dinner party is $48.00. -/
theorem nates_scallop_cost :
  scallop_cost 8 24 2 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_nates_scallop_cost_l3529_352932


namespace NUMINAMATH_CALUDE_simplify_expression_l3529_352936

theorem simplify_expression : 5 * (14 / 3) * (9 / -42) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3529_352936


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_over_sums_l3529_352906

theorem min_value_sum_of_squares_over_sums (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_over_sums_l3529_352906


namespace NUMINAMATH_CALUDE_annual_growth_rate_proof_l3529_352947

-- Define the initial number of students
def initial_students : ℕ := 200

-- Define the final number of students
def final_students : ℕ := 675

-- Define the number of years
def years : ℕ := 3

-- Define the growth rate as a real number between 0 and 1
def growth_rate : ℝ := 0.5

-- Theorem statement
theorem annual_growth_rate_proof :
  (initial_students : ℝ) * (1 + growth_rate)^years = final_students :=
sorry

end NUMINAMATH_CALUDE_annual_growth_rate_proof_l3529_352947


namespace NUMINAMATH_CALUDE_min_ratio_four_digit_number_l3529_352944

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest value of n/s_n for four-digit numbers is 1099/19 -/
theorem min_ratio_four_digit_number :
  ∀ n : ℕ, 1000 ≤ n → n ≤ 9999 → (n : ℚ) / (sum_of_digits n) ≥ 1099 / 19 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_four_digit_number_l3529_352944


namespace NUMINAMATH_CALUDE_list_property_l3529_352997

theorem list_property (numbers : List ℝ) (n : ℝ) : 
  numbers.length = 21 →
  n ∈ numbers →
  n = 5 * ((numbers.sum - n) / 20) →
  n = 0.2 * numbers.sum →
  (numbers.filter (λ x => x ≠ n)).length = 20 := by
sorry

end NUMINAMATH_CALUDE_list_property_l3529_352997


namespace NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l3529_352988

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: The value of b that satisfies h(b) = 0 is b = 7/5 -/
theorem h_zero_at_seven_fifths : 
  ∃ b : ℝ, h b = 0 ∧ b = 7/5 := by sorry

end NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l3529_352988


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l3529_352954

theorem fixed_point_parabola :
  ∀ (t : ℝ), 5 = 5 * (-1)^2 + 2 * t * (-1) - 5 * t := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l3529_352954


namespace NUMINAMATH_CALUDE_special_number_not_divisible_l3529_352915

/-- Represents a 70-digit number with specific digit frequency properties -/
def SpecialNumber := { n : ℕ // 
  (Nat.digits 10 n).length = 70 ∧ 
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7] → (Nat.digits 10 n).count d = 10) ∧
  (∀ d : ℕ, d ∈ [8, 9, 0] → d ∉ (Nat.digits 10 n))
}

/-- Theorem stating that no SpecialNumber can divide another SpecialNumber -/
theorem special_number_not_divisible (n m : SpecialNumber) : ¬(n.val ∣ m.val) := by
  sorry

end NUMINAMATH_CALUDE_special_number_not_divisible_l3529_352915


namespace NUMINAMATH_CALUDE_prob_black_ball_is_one_fourth_l3529_352989

/-- Represents the number of black balls in the bag. -/
def black_balls : ℕ := 6

/-- Represents the number of red balls in the bag. -/
def red_balls : ℕ := 18

/-- Represents the total number of balls in the bag. -/
def total_balls : ℕ := black_balls + red_balls

/-- The probability of drawing a black ball from the bag. -/
def prob_black_ball : ℚ := black_balls / total_balls

theorem prob_black_ball_is_one_fourth : prob_black_ball = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_black_ball_is_one_fourth_l3529_352989


namespace NUMINAMATH_CALUDE_stating_spinner_points_east_l3529_352918

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a clockwise rotation in revolutions --/
def clockwise_rotation : ℚ := 7/2

/-- Represents a counterclockwise rotation in revolutions --/
def counterclockwise_rotation : ℚ := 11/4

/-- Represents the initial direction of the spinner --/
def initial_direction : Direction := Direction.South

/-- 
  Theorem stating that after the given rotations, 
  the spinner will point east
--/
theorem spinner_points_east : 
  ∃ (final_direction : Direction),
    final_direction = Direction.East :=
by sorry

end NUMINAMATH_CALUDE_stating_spinner_points_east_l3529_352918


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3529_352904

/-- Represents the inverse variation relationship between x^3 and ∛w -/
def inverse_variation (x w : ℝ) : Prop := ∃ k : ℝ, x^3 * w^(1/3) = k

/-- Given conditions and theorem statement -/
theorem inverse_variation_problem (x₀ w₀ x₁ w₁ : ℝ) 
  (h₀ : inverse_variation x₀ w₀)
  (h₁ : x₀ = 3)
  (h₂ : w₀ = 8)
  (h₃ : x₁ = 6)
  (h₄ : inverse_variation x₁ w₁) :
  w₁ = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3529_352904


namespace NUMINAMATH_CALUDE_piper_wing_count_l3529_352981

/-- The number of commercial planes in the air exhibition -/
def num_planes : ℕ := 45

/-- The number of wings on each commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted by Piper -/
def total_wings : ℕ := num_planes * wings_per_plane

theorem piper_wing_count : total_wings = 90 := by
  sorry

end NUMINAMATH_CALUDE_piper_wing_count_l3529_352981


namespace NUMINAMATH_CALUDE_distance_between_points_l3529_352987

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 6)
  let p2 : ℝ × ℝ := (-7, -2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3529_352987


namespace NUMINAMATH_CALUDE_binomial_60_3_l3529_352991

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3529_352991


namespace NUMINAMATH_CALUDE_existence_of_larger_element_l3529_352968

/-- A doubly infinite array of positive integers -/
def InfiniteArray := ℕ+ → ℕ+ → ℕ+

/-- The property that each positive integer appears exactly eight times in the array -/
def EightOccurrences (a : InfiniteArray) : Prop :=
  ∀ n : ℕ+, (∃ (s : Finset (ℕ+ × ℕ+)), s.card = 8 ∧ (∀ (p : ℕ+ × ℕ+), p ∈ s ↔ a p.1 p.2 = n))

/-- The main theorem -/
theorem existence_of_larger_element (a : InfiniteArray) (h : EightOccurrences a) :
  ∃ (m n : ℕ+), a m n > m * n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_larger_element_l3529_352968


namespace NUMINAMATH_CALUDE_movie_ticket_final_price_l3529_352920

def movie_ticket_price (initial_price : ℝ) (year1_increase year2_decrease year3_increase year4_decrease year5_increase tax discount : ℝ) : ℝ :=
  let price1 := initial_price * (1 + year1_increase)
  let price2 := price1 * (1 - year2_decrease)
  let price3 := price2 * (1 + year3_increase)
  let price4 := price3 * (1 - year4_decrease)
  let price5 := price4 * (1 + year5_increase)
  let price_with_tax := price5 * (1 + tax)
  price_with_tax * (1 - discount)

theorem movie_ticket_final_price :
  let initial_price : ℝ := 100
  let year1_increase : ℝ := 0.12
  let year2_decrease : ℝ := 0.05
  let year3_increase : ℝ := 0.08
  let year4_decrease : ℝ := 0.04
  let year5_increase : ℝ := 0.06
  let tax : ℝ := 0.07
  let discount : ℝ := 0.10
  ∃ ε > 0, |movie_ticket_price initial_price year1_increase year2_decrease year3_increase year4_decrease year5_increase tax discount - 112.61| < ε :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_final_price_l3529_352920


namespace NUMINAMATH_CALUDE_exists_set_equal_partitions_l3529_352925

/-- The type of positive integers -/
def PositiveInt : Type := { n : ℕ // n > 0 }

/-- Count of partitions where each number appears at most twice -/
def countLimitedPartitions (n : ℕ) : ℕ :=
  sorry

/-- Count of partitions using elements from a set -/
def countSetPartitions (n : ℕ) (S : Set PositiveInt) : ℕ :=
  sorry

/-- The existence of a set S satisfying the partition property -/
theorem exists_set_equal_partitions :
  ∃ (S : Set PositiveInt), ∀ (n : ℕ), n > 0 →
    countLimitedPartitions n = countSetPartitions n S :=
  sorry

end NUMINAMATH_CALUDE_exists_set_equal_partitions_l3529_352925


namespace NUMINAMATH_CALUDE_calligraphy_book_characters_l3529_352957

/-- The number of characters written per day in the first practice -/
def first_practice_chars_per_day : ℕ := 25

/-- The additional characters written per day in the second practice -/
def additional_chars_per_day : ℕ := 3

/-- The number of days fewer in the second practice compared to the first -/
def days_difference : ℕ := 3

/-- The total number of characters in the book -/
def total_characters : ℕ := 700

theorem calligraphy_book_characters :
  ∃ (x : ℕ), 
    x > days_difference ∧
    first_practice_chars_per_day * x = 
    (first_practice_chars_per_day + additional_chars_per_day) * (x - days_difference) ∧
    total_characters = first_practice_chars_per_day * x :=
by sorry

end NUMINAMATH_CALUDE_calligraphy_book_characters_l3529_352957


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l3529_352984

theorem geometric_sequence_minimum (b₁ b₂ b₃ : ℝ) : 
  b₁ = 1 → (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = b₁ * r' ∧ b₃' = b₂' * r') → 
    3 * b₂ + 4 * b₃ ≤ 3 * b₂' + 4 * b₃') →
  3 * b₂ + 4 * b₃ = -9/16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l3529_352984


namespace NUMINAMATH_CALUDE_slope_angle_expression_l3529_352949

theorem slope_angle_expression (x y : ℝ) (α : ℝ) : 
  (6 * x - 2 * y - 5 = 0) →
  (Real.tan α = 3) →
  ((Real.sin (π - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (π + α)) = -2) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_expression_l3529_352949


namespace NUMINAMATH_CALUDE_football_game_attendance_l3529_352994

/-- Proves that the number of children attending a football game is 80, given the ticket prices, total attendance, and total revenue. -/
theorem football_game_attendance
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_attendance : ℕ)
  (total_revenue : ℕ)
  (h1 : adult_price = 60)
  (h2 : child_price = 25)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 14000)
  : ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 80 := by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l3529_352994


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3529_352910

-- Define the number of pages read per night
def pages_per_night : ℝ := 120.0

-- Define the number of days of reading
def days_of_reading : ℝ := 10.0

-- Define the total number of pages in the book
def total_pages : ℝ := pages_per_night * days_of_reading

-- Theorem statement
theorem book_pages_calculation : total_pages = 1200.0 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l3529_352910


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3529_352955

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(-Real.sqrt 5, 0), (Real.sqrt 5, 0)}

/-- Theorem: The foci of the hyperbola are at (±√5, 0) -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3529_352955


namespace NUMINAMATH_CALUDE_inscribed_triangle_inequality_l3529_352902

/-- A triangle PQR inscribed in a semicircle with diameter PQ and R on the semicircle -/
structure InscribedTriangle where
  /-- The radius of the semicircle -/
  r : ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point R -/
  R : ℝ × ℝ
  /-- PQ is the diameter of the semicircle -/
  diameter : dist P Q = 2 * r
  /-- R is on the semicircle -/
  on_semicircle : dist P R = r ∨ dist Q R = r

/-- The sum of distances PR and QR -/
def t (triangle : InscribedTriangle) : ℝ :=
  dist triangle.P triangle.R + dist triangle.Q triangle.R

/-- Theorem: For all inscribed triangles, t^2 ≤ 8r^2 -/
theorem inscribed_triangle_inequality (triangle : InscribedTriangle) :
  (t triangle)^2 ≤ 8 * triangle.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_inequality_l3529_352902


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l3529_352952

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  Real.sqrt 3 * a * Real.cos C - c * Real.sin A = 0 →
  b = 6 →
  1/2 * a * b * Real.sin C = 6 * Real.sqrt 3 →
  C = π/3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l3529_352952


namespace NUMINAMATH_CALUDE_smallest_angle_divisible_isosceles_l3529_352939

/-- An isosceles triangle that can be divided into two isosceles triangles -/
structure DivisibleIsoscelesTriangle where
  /-- The measure of one of the equal angles in the original isosceles triangle -/
  α : Real
  /-- The triangle is isosceles -/
  isIsosceles : α ≥ 0 ∧ α ≤ 90
  /-- The triangle can be divided into two isosceles triangles -/
  isDivisible : ∃ (β γ : Real), (β > 0 ∧ γ > 0) ∧ 
    ((β = α ∧ γ = (180 - α) / 2) ∨ (β = (180 - α) / 2 ∧ γ = (3 * α - 180) / 2))

/-- The smallest angle in a divisible isosceles triangle is 180/7 degrees -/
theorem smallest_angle_divisible_isosceles (t : DivisibleIsoscelesTriangle) :
  min t.α (180 - 2 * t.α) ≥ 180 / 7 ∧ 
  ∃ (t' : DivisibleIsoscelesTriangle), min t'.α (180 - 2 * t'.α) = 180 / 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_divisible_isosceles_l3529_352939


namespace NUMINAMATH_CALUDE_pen_count_l3529_352945

theorem pen_count (initial : ℕ) (received : ℕ) (given_away : ℕ) : 
  initial = 20 → received = 22 → given_away = 19 → 
  ((initial + received) * 2 - given_away) = 65 := by
sorry

end NUMINAMATH_CALUDE_pen_count_l3529_352945


namespace NUMINAMATH_CALUDE_box_volume_increase_l3529_352908

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + h * l) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3529_352908


namespace NUMINAMATH_CALUDE_complex_power_2013_l3529_352926

def i : ℂ := Complex.I

theorem complex_power_2013 : ((1 + i) / (1 - i)) ^ 2013 = i := by sorry

end NUMINAMATH_CALUDE_complex_power_2013_l3529_352926


namespace NUMINAMATH_CALUDE_tangent_line_condition_range_of_a_l3529_352951

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a / x
def g (x : ℝ) : ℝ := 2 * x * Real.exp x - Real.log x - x - Real.log 2

-- Part 1: Tangent line condition
theorem tangent_line_condition (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ ∧ (deriv (f a)) x₀ = 1) → a = Real.exp 1 / 2 :=
sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂ > 0, f a x₁ ≥ g x₂) → a ≥ 1 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_condition_range_of_a_l3529_352951


namespace NUMINAMATH_CALUDE_same_gender_probability_l3529_352961

/-- The probability of selecting two students of the same gender -/
theorem same_gender_probability (n_male n_female : ℕ) (h_male : n_male = 2) (h_female : n_female = 8) :
  let total := n_male + n_female
  let same_gender_ways := Nat.choose n_male 2 + Nat.choose n_female 2
  let total_ways := Nat.choose total 2
  (same_gender_ways : ℚ) / total_ways = 29 / 45 := by sorry

end NUMINAMATH_CALUDE_same_gender_probability_l3529_352961


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l3529_352971

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 30) :
  Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2)) = 2 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l3529_352971


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3529_352903

theorem arccos_one_over_sqrt_two (π : Real) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3529_352903


namespace NUMINAMATH_CALUDE_tangent_product_upper_bound_l3529_352937

theorem tangent_product_upper_bound (α β : Real) 
  (sum_eq : α + β = Real.pi / 3)
  (α_pos : α > 0)
  (β_pos : β > 0) :
  Real.tan α * Real.tan β ≤ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_upper_bound_l3529_352937


namespace NUMINAMATH_CALUDE_toy_production_lot_l3529_352983

theorem toy_production_lot (total : ℕ) 
  (h_red : total * 2 / 5 = total * 40 / 100)
  (h_small : total / 2 = total * 50 / 100)
  (h_red_small : total / 10 = total * 10 / 100)
  (h_red_large : total * 3 / 10 = 60) :
  total * 2 / 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_toy_production_lot_l3529_352983


namespace NUMINAMATH_CALUDE_x_cubed_remainder_l3529_352940

theorem x_cubed_remainder (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^3 ≡ 8 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_remainder_l3529_352940


namespace NUMINAMATH_CALUDE_lineup_combinations_l3529_352913

/-- The number of ways to choose a starting lineup for a basketball team -/
def choose_lineup (total_players : ℕ) (center_players : ℕ) (point_guard_players : ℕ) : ℕ :=
  center_players * point_guard_players * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating the number of ways to choose a starting lineup -/
theorem lineup_combinations :
  choose_lineup 12 3 2 = 4320 :=
by sorry

end NUMINAMATH_CALUDE_lineup_combinations_l3529_352913


namespace NUMINAMATH_CALUDE_camera_price_theorem_l3529_352905

/-- The sticker price of the camera -/
def sticker_price : ℝ := 666.67

/-- The price at Store X after discount and rebate -/
def store_x_price (p : ℝ) : ℝ := 0.80 * p - 50

/-- The price at Store Y after discount -/
def store_y_price (p : ℝ) : ℝ := 0.65 * p

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem camera_price_theorem : 
  store_y_price sticker_price - store_x_price sticker_price = 40 := by
  sorry


end NUMINAMATH_CALUDE_camera_price_theorem_l3529_352905


namespace NUMINAMATH_CALUDE_fiona_reach_probability_l3529_352978

/-- Represents a lily pad with its number and whether it contains a predator -/
structure LilyPad :=
  (number : Nat)
  (hasPredator : Bool)

/-- Represents Fiona's possible moves -/
inductive Move
  | Hop
  | Jump

/-- Represents the frog's journey -/
def FrogJourney := List LilyPad

/-- The probability of each move -/
def moveProbability : Move → ℚ
  | Move.Hop => 1/2
  | Move.Jump => 1/2

/-- The number of pads to move for each move type -/
def moveDistance : Move → Nat
  | Move.Hop => 1
  | Move.Jump => 2

/-- The lily pads in the pond -/
def lilyPads : List LilyPad :=
  List.range 16 |> List.map (λ n => ⟨n, n ∈ [4, 7, 11]⟩)

/-- Check if a journey is safe (doesn't land on predator pads) -/
def isSafeJourney (journey : FrogJourney) : Bool :=
  journey.all (λ pad => !pad.hasPredator)

/-- Calculate the probability of a specific journey -/
def journeyProbability (journey : FrogJourney) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem fiona_reach_probability :
  ∃ (safeJourneys : List FrogJourney),
    (∀ j ∈ safeJourneys, j.head? = some ⟨0, false⟩ ∧
                         j.getLast? = some ⟨14, false⟩ ∧
                         isSafeJourney j) ∧
    (safeJourneys.map journeyProbability).sum = 3/256 :=
  sorry

end NUMINAMATH_CALUDE_fiona_reach_probability_l3529_352978


namespace NUMINAMATH_CALUDE_digit_150_of_1_13_l3529_352930

def decimal_representation_1_13 : List ℕ := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_13 : 
  (decimal_representation_1_13[(150 - 1) % decimal_representation_1_13.length] = 3) := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_1_13_l3529_352930


namespace NUMINAMATH_CALUDE_lucky_draw_probabilities_l3529_352950

def probability_wang_wins : ℝ := 0.4
def probability_zhang_wins : ℝ := 0.2

theorem lucky_draw_probabilities :
  let p_both_win := probability_wang_wins * probability_zhang_wins
  let p_only_one_wins := probability_wang_wins * (1 - probability_zhang_wins) + (1 - probability_wang_wins) * probability_zhang_wins
  let p_at_most_one_wins := 1 - p_both_win
  (p_both_win = 0.08) ∧
  (p_only_one_wins = 0.44) ∧
  (p_at_most_one_wins = 0.92) := by
  sorry

end NUMINAMATH_CALUDE_lucky_draw_probabilities_l3529_352950


namespace NUMINAMATH_CALUDE_axis_triangle_line_equation_l3529_352975

/-- A line passing through a point and forming a triangle with the axes --/
structure AxisTriangleLine where
  /-- The slope of the line --/
  k : ℝ
  /-- The line passes through the point (1, 2) --/
  passes_through : k * (1 - 0) = 2 - 0
  /-- The slope is negative --/
  negative_slope : k < 0
  /-- The area of the triangle formed with the axes is 4 --/
  triangle_area : (1/2) * (2 - k) * (1 - 2/k) = 4

/-- The equation of the line is 2x + y - 4 = 0 --/
theorem axis_triangle_line_equation (l : AxisTriangleLine) : 
  ∃ (a b c : ℝ), a * 1 + b * 2 + c = 0 ∧ 
                  ∀ x y, a * x + b * y + c = 0 ↔ y - 2 = l.k * (x - 1) :=
sorry

end NUMINAMATH_CALUDE_axis_triangle_line_equation_l3529_352975


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3529_352942

theorem parallel_line_through_point (x y : ℝ) : 
  let P : ℝ × ℝ := (0, 2)
  let L₁ : Set (ℝ × ℝ) := {(x, y) | 2 * x - y = 0}
  let L₂ : Set (ℝ × ℝ) := {(x, y) | 2 * x - y + 2 = 0}
  (P ∈ L₂) ∧ (∃ k : ℝ, k ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ L₁ ↔ (k * x, k * y) ∈ L₂) :=
by
  sorry

#check parallel_line_through_point

end NUMINAMATH_CALUDE_parallel_line_through_point_l3529_352942


namespace NUMINAMATH_CALUDE_sum_of_parts_zero_l3529_352938

theorem sum_of_parts_zero : 
  let z : ℂ := (3 - Complex.I) / (2 + Complex.I)
  (z.re + z.im) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_zero_l3529_352938


namespace NUMINAMATH_CALUDE_range_of_m_l3529_352999

-- Define the conditions
def condition1 (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def condition2 (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := condition2 x m

-- State the theorem
theorem range_of_m : 
  (∀ x m : ℝ, condition1 x → (¬(p x) → ¬(q x m)) ∧ ∃ y : ℝ, ¬(p y) ∧ q y m) →
  (∀ m : ℝ, m ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3529_352999


namespace NUMINAMATH_CALUDE_reflection_sum_l3529_352967

/-- Reflects a point over the y-axis -/
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

/-- Reflects a point over the x-axis -/
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)

/-- Sums the coordinates of a point -/
def sum_coordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

theorem reflection_sum (y : ℝ) :
  let C : ℝ × ℝ := (3, y)
  let D := reflect_y C.1 C.2
  let E := reflect_x D.1 D.2
  sum_coordinates C + sum_coordinates E = -6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_l3529_352967


namespace NUMINAMATH_CALUDE_mari_made_79_buttons_l3529_352974

/-- The number of buttons made by each person -/
structure ButtonCounts where
  kendra : ℕ
  mari : ℕ
  sue : ℕ
  jess : ℕ
  tom : ℕ

/-- The conditions of the button-making problem -/
def ButtonProblem (counts : ButtonCounts) : Prop :=
  counts.kendra = 15 ∧
  counts.mari = 5 * counts.kendra + 4 ∧
  counts.sue = 2 * counts.kendra / 3 ∧
  counts.jess = 2 * (counts.sue + counts.kendra) ∧
  counts.tom = 3 * counts.jess / 4

/-- Mari made 79 buttons -/
theorem mari_made_79_buttons (counts : ButtonCounts) 
  (h : ButtonProblem counts) : counts.mari = 79 := by
  sorry

end NUMINAMATH_CALUDE_mari_made_79_buttons_l3529_352974


namespace NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt2_over_2_l3529_352946

/-- The modulus of the complex number z = i / (1 - i) is equal to √2/2 -/
theorem modulus_of_z_equals_sqrt2_over_2 : 
  let z : ℂ := Complex.I / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt2_over_2_l3529_352946


namespace NUMINAMATH_CALUDE_prob_divisible_by_11_is_correct_l3529_352970

/-- The probability of reaching a number divisible by 11 in the described process -/
def prob_divisible_by_11 : ℚ := 11 / 20

/-- The process of building an integer as described in the problem -/
def build_integer (start : ℕ) (stop_condition : ℕ → Bool) : ℕ → ℚ := sorry

/-- The main theorem stating that the probability of reaching a number divisible by 11 is 11/20 -/
theorem prob_divisible_by_11_is_correct :
  build_integer 9 (λ n => n % 11 = 0 ∨ n % 11 = 1) 0 = prob_divisible_by_11 := by sorry

end NUMINAMATH_CALUDE_prob_divisible_by_11_is_correct_l3529_352970


namespace NUMINAMATH_CALUDE_division_and_subtraction_l3529_352958

theorem division_and_subtraction : (12 / (1/12)) - 5 = 139 := by
  sorry

end NUMINAMATH_CALUDE_division_and_subtraction_l3529_352958


namespace NUMINAMATH_CALUDE_cookie_fraction_with_nuts_l3529_352973

theorem cookie_fraction_with_nuts 
  (total_cookies : ℕ) 
  (nuts_per_cookie : ℕ) 
  (total_nuts : ℕ) 
  (h1 : total_cookies = 60) 
  (h2 : nuts_per_cookie = 2) 
  (h3 : total_nuts = 72) : 
  (total_nuts / nuts_per_cookie : ℚ) / total_cookies = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_fraction_with_nuts_l3529_352973


namespace NUMINAMATH_CALUDE_perpendicular_slope_to_OA_l3529_352953

/-- Given point A(3, 5) and O as the origin, prove that the slope of the line perpendicular to OA is -3/5 -/
theorem perpendicular_slope_to_OA :
  let A : ℝ × ℝ := (3, 5)
  let O : ℝ × ℝ := (0, 0)
  let slope_OA : ℝ := (A.2 - O.2) / (A.1 - O.1)
  let slope_perpendicular : ℝ := -1 / slope_OA
  slope_perpendicular = -3/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_to_OA_l3529_352953


namespace NUMINAMATH_CALUDE_tommy_balloons_l3529_352998

theorem tommy_balloons (initial : ℝ) : 
  initial + 34.5 - 12.75 = 60.75 → initial = 39 := by sorry

end NUMINAMATH_CALUDE_tommy_balloons_l3529_352998


namespace NUMINAMATH_CALUDE_equivalent_angle_for_negative_463_l3529_352995

-- Define the angle equivalence relation
def angle_equivalent (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

-- State the theorem
theorem equivalent_angle_for_negative_463 :
  ∀ k : ℤ, angle_equivalent (-463) (k * 360 + 257) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_angle_for_negative_463_l3529_352995


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3529_352927

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 200 * (1 / x) = 400 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3529_352927


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l3529_352923

/-- Represents the cost calculation for manuscript printing and binding --/
def manuscript_cost (
  num_copies : ℕ
  ) (
  total_pages : ℕ
  ) (
  color_pages : ℕ
  ) (
  bw_cost : ℚ
  ) (
  color_cost : ℚ
  ) (
  binding_cost : ℚ
  ) (
  index_cost : ℚ
  ) (
  rush_copies : ℕ
  ) (
  rush_cost : ℚ
  ) (
  binding_discount_rate : ℚ
  ) (
  bundle_discount : ℚ
  ) : ℚ :=
  let bw_pages := total_pages - color_pages
  let print_cost := (bw_pages : ℚ) * bw_cost + (color_pages : ℚ) * color_cost
  let additional_cost := binding_cost + index_cost - bundle_discount
  let copy_cost := print_cost + additional_cost
  let total_before_discount := (num_copies : ℚ) * copy_cost
  let binding_discount := (num_copies : ℚ) * binding_cost * binding_discount_rate
  let rush_fee := (rush_copies : ℚ) * rush_cost
  total_before_discount - binding_discount + rush_fee

/-- Theorem stating the total cost for the manuscript printing and binding --/
theorem manuscript_cost_theorem :
  manuscript_cost 10 400 50 (5/100) (1/10) 5 2 5 3 (1/10) (1/2) = 300 :=
by sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l3529_352923


namespace NUMINAMATH_CALUDE_new_person_weight_l3529_352972

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 6 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3529_352972


namespace NUMINAMATH_CALUDE_sandwiches_sold_out_l3529_352900

theorem sandwiches_sold_out (original : ℕ) (available : ℕ) (h1 : original = 9) (h2 : available = 4) :
  original - available = 5 := by
sorry

end NUMINAMATH_CALUDE_sandwiches_sold_out_l3529_352900


namespace NUMINAMATH_CALUDE_triangle_power_equality_l3529_352922

theorem triangle_power_equality (a b c : ℝ) 
  (h : ∀ n : ℕ, (a^n + b^n > c^n) ∧ (b^n + c^n > a^n) ∧ (c^n + a^n > b^n)) :
  (a = b) ∨ (b = c) ∨ (c = a) := by
sorry

end NUMINAMATH_CALUDE_triangle_power_equality_l3529_352922


namespace NUMINAMATH_CALUDE_equation_solution_l3529_352919

theorem equation_solution (x : ℂ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3*x - 6) / (x + 2) + (3*x^2 - 12) / (3 - x) = 3 ↔ x = -2 + 2*I ∨ x = -2 - 2*I :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3529_352919


namespace NUMINAMATH_CALUDE_angle_C_measure_l3529_352909

theorem angle_C_measure (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Given conditions
  (a = 2 * Real.sqrt 6) →
  (b = 6) →
  (Real.cos B = -1/2) →
  -- Conclusion
  C = π/12 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l3529_352909


namespace NUMINAMATH_CALUDE_simplification_proof_l3529_352960

theorem simplification_proof (a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : 3*a - b/3 ≠ 0) :
  (3*a - b/3)⁻¹ * ((3*a)⁻¹ - (b/3)⁻¹) = -(a*b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplification_proof_l3529_352960


namespace NUMINAMATH_CALUDE_solve_commencement_addresses_l3529_352980

def commencement_addresses_problem (sandoval hawkins sloan : ℕ) : Prop :=
  sandoval = 12 ∧
  hawkins = sandoval / 2 ∧
  sloan > sandoval ∧
  sandoval + hawkins + sloan = 40 ∧
  sloan - sandoval = 10

theorem solve_commencement_addresses :
  ∃ (sandoval hawkins sloan : ℕ), commencement_addresses_problem sandoval hawkins sloan :=
by
  sorry

end NUMINAMATH_CALUDE_solve_commencement_addresses_l3529_352980


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3529_352934

theorem initial_money_calculation (M : ℚ) : 
  (((M * (3/5) * (2/3) * (3/4) * (4/7)) : ℚ) = 700) → M = 24500/6 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3529_352934


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l3529_352901

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 3) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_proof (h1 : square_area = 4761) (h2 : rectangle_breadth = 13) :
  rectangle_area square_area rectangle_breadth = 598 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l3529_352901


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l3529_352924

def B_current_age : ℕ := 39
def A_current_age : ℕ := B_current_age + 9

def A_future_age : ℕ := A_current_age + 10
def B_past_age : ℕ := B_current_age - 10

theorem age_ratio_is_two_to_one :
  A_future_age / B_past_age = 2 ∧ B_past_age ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l3529_352924


namespace NUMINAMATH_CALUDE_complex_arithmetic_calculation_l3529_352992

theorem complex_arithmetic_calculation : 
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_calculation_l3529_352992


namespace NUMINAMATH_CALUDE_books_per_shelf_l3529_352911

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) (h1 : total_books = 12) (h2 : num_shelves = 3) :
  total_books / num_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l3529_352911


namespace NUMINAMATH_CALUDE_car_trading_theorem_l3529_352964

/-- Represents the profit and purchase constraints for a car trading company. -/
structure CarTrading where
  profit_A_2_B_5 : ℕ  -- Profit from selling 2 A and 5 B
  profit_A_1_B_2 : ℕ  -- Profit from selling 1 A and 2 B
  price_A : ℕ         -- Purchase price of model A
  price_B : ℕ         -- Purchase price of model B
  total_budget : ℕ    -- Total budget
  total_units : ℕ     -- Total number of cars to purchase

/-- Theorem stating the profit per unit and minimum purchase of model A -/
theorem car_trading_theorem (ct : CarTrading) 
  (h1 : ct.profit_A_2_B_5 = 31000)
  (h2 : ct.profit_A_1_B_2 = 13000)
  (h3 : ct.price_A = 120000)
  (h4 : ct.price_B = 150000)
  (h5 : ct.total_budget = 3000000)
  (h6 : ct.total_units = 22) :
  ∃ (profit_A profit_B min_A : ℕ),
    profit_A = 3000 ∧
    profit_B = 5000 ∧
    min_A = 10 ∧
    2 * profit_A + 5 * profit_B = ct.profit_A_2_B_5 ∧
    profit_A + 2 * profit_B = ct.profit_A_1_B_2 ∧
    min_A * ct.price_A + (ct.total_units - min_A) * ct.price_B ≤ ct.total_budget :=
by sorry

end NUMINAMATH_CALUDE_car_trading_theorem_l3529_352964


namespace NUMINAMATH_CALUDE_plums_added_l3529_352917

def initial_plums : ℕ := 17
def final_plums : ℕ := 21

theorem plums_added (initial : ℕ) (final : ℕ) (added : ℕ) 
  (h1 : initial = initial_plums) 
  (h2 : final = final_plums) 
  (h3 : final = initial + added) : 
  added = final - initial :=
by
  sorry

end NUMINAMATH_CALUDE_plums_added_l3529_352917


namespace NUMINAMATH_CALUDE_square_number_placement_l3529_352979

theorem square_number_placement :
  ∃ (a b c d e : ℕ),
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (Nat.gcd a b > 1 ∧ Nat.gcd b c > 1 ∧ Nat.gcd c d > 1 ∧ Nat.gcd d a > 1) ∧
    (Nat.gcd a e > 1 ∧ Nat.gcd b e > 1 ∧ Nat.gcd c e > 1 ∧ Nat.gcd d e > 1) ∧
    (Nat.gcd a c = 1 ∧ Nat.gcd b d = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_number_placement_l3529_352979


namespace NUMINAMATH_CALUDE_correct_calculation_l3529_352933

theorem correct_calculation (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3529_352933


namespace NUMINAMATH_CALUDE_unique_solution_for_all_y_l3529_352977

theorem unique_solution_for_all_y : ∃! x : ℝ, ∀ y : ℝ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_y_l3529_352977


namespace NUMINAMATH_CALUDE_range_of_g_l3529_352912

-- Define the functions
def f (x : ℝ) := x^2 - 7*x + 12
def g (x : ℝ) := x^2 - 7*x + 14

-- State the theorem
theorem range_of_g (x : ℝ) : 
  f x < 0 → ∃ y ∈ Set.Icc (1.75 : ℝ) 2, y = g x :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l3529_352912


namespace NUMINAMATH_CALUDE_sibling_age_sum_l3529_352941

/-- Given the ages and age differences of three siblings, prove the sum of the youngest and oldest siblings' ages. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = 10 ∧ 
  juliet = maggie + 3 ∧ 
  ralph = juliet + 2 → 
  maggie + ralph = 19 := by
  sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l3529_352941


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3529_352916

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 → (2021 * (x^2020)^(1/202) - 1 = 2020 * x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3529_352916


namespace NUMINAMATH_CALUDE_remainder_problem_l3529_352986

theorem remainder_problem (n : ℤ) : 
  (n % 4 = 3) → (n % 9 = 5) → (n % 36 = 23) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3529_352986


namespace NUMINAMATH_CALUDE_exists_composite_prime_product_plus_one_l3529_352928

/-- pₖ denotes the k-th prime number -/
def nth_prime (k : ℕ) : ℕ := sorry

/-- Product of first n prime numbers plus 1 -/
def prime_product_plus_one (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc * nth_prime (i + 1)) 1 + 1

theorem exists_composite_prime_product_plus_one :
  ∃ n : ℕ, ¬ Nat.Prime (prime_product_plus_one n) := by sorry

end NUMINAMATH_CALUDE_exists_composite_prime_product_plus_one_l3529_352928


namespace NUMINAMATH_CALUDE_vector_a_magnitude_l3529_352976

def vector_a : ℝ × ℝ := (3, -2)

theorem vector_a_magnitude : ‖vector_a‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_a_magnitude_l3529_352976


namespace NUMINAMATH_CALUDE_ounces_per_cup_l3529_352914

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) (h1 : total_ounces = 264) (h2 : total_cups = 33) :
  total_ounces / total_cups = 8 := by
sorry

end NUMINAMATH_CALUDE_ounces_per_cup_l3529_352914


namespace NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3529_352993

/-- Represents the cost of fruits in dollars -/
structure FruitCosts where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  cherries : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitProblemConditions (c : FruitCosts) : Prop :=
  c.apples + c.bananas + c.cantaloupe + c.dates + c.cherries = 30 ∧
  c.dates = 3 * c.apples ∧
  c.cantaloupe = c.apples - c.bananas ∧
  c.cherries = c.apples + c.bananas

/-- The theorem stating that under the given conditions, 
    the cost of bananas and cantaloupe is $5 -/
theorem banana_cantaloupe_cost (c : FruitCosts) 
  (h : fruitProblemConditions c) : 
  c.bananas + c.cantaloupe = 5 := by
  sorry

end NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3529_352993


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3529_352982

theorem triangle_inequalities (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b + c) (hbc : b < a + c) (hca : c < a + b) :
  (a + b + c = 2 → a^2 + b^2 + c^2 + 2*a*b*c < 2) ∧
  (a + b + c = 1 → a^2 + b^2 + c^2 + 4*a*b*c < 1/2) ∧
  (a + b + c = 1 → 5*(a^2 + b^2 + c^2) + 18*a*b*c > 7/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3529_352982


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3529_352948

theorem quadratic_polynomial_with_complex_root : 
  ∃ (a b c : ℝ), 
    (a = 3 ∧ 
     (Complex.I : ℂ)^2 = -1 ∧
     (3 : ℂ) * ((4 + 2 * Complex.I) ^ 2 - 8 * (4 + 2 * Complex.I) + 16 + 4) = 3 * (Complex.I : ℂ)^2 + b * (Complex.I : ℂ) + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3529_352948


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l3529_352956

def number_of_dogs : ℕ := 10
def group_sizes : List ℕ := [3, 5, 2]

theorem dog_grouping_theorem :
  let remaining_dogs := number_of_dogs - 2  -- Fluffy and Nipper are pre-placed
  let ways_to_fill_fluffy_group := Nat.choose remaining_dogs (group_sizes[0] - 1)
  let remaining_after_fluffy := remaining_dogs - (group_sizes[0] - 1)
  let ways_to_fill_nipper_group := Nat.choose remaining_after_fluffy (group_sizes[1] - 1)
  ways_to_fill_fluffy_group * ways_to_fill_nipper_group = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l3529_352956


namespace NUMINAMATH_CALUDE_men_who_left_l3529_352990

/-- Given a hostel with provisions for a certain number of men and days,
    calculate the number of men who left if the provisions last longer. -/
theorem men_who_left (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) :
  initial_men = 250 →
  initial_days = 32 →
  new_days = 40 →
  ∃ (men_left : ℕ),
    men_left = 50 ∧
    initial_men * initial_days = (initial_men - men_left) * new_days :=
by sorry

end NUMINAMATH_CALUDE_men_who_left_l3529_352990


namespace NUMINAMATH_CALUDE_probability_x_less_than_y_l3529_352962

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the condition x < y
def condition (p : ℝ × ℝ) : Prop := p.1 < p.2

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.ProbabilityMeasure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_less_than_y :
  prob {p ∈ rectangle | condition p} = 1/8 := by sorry

end NUMINAMATH_CALUDE_probability_x_less_than_y_l3529_352962


namespace NUMINAMATH_CALUDE_hillary_activities_lcm_l3529_352943

theorem hillary_activities_lcm : Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) = 48 := by
  sorry

end NUMINAMATH_CALUDE_hillary_activities_lcm_l3529_352943


namespace NUMINAMATH_CALUDE_cookies_per_person_l3529_352907

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) 
  (h1 : total_cookies = 420) (h2 : num_people = 14) :
  total_cookies / num_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l3529_352907


namespace NUMINAMATH_CALUDE_geometry_relations_l3529_352966

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)

-- Define the lines and planes
variable (l m : Line) (α β γ : Plane)

-- State the theorem
theorem geometry_relations :
  (perp l m ∧ perp_line_plane l α ∧ perp_line_plane m β → perp_plane α β) ∧
  (parallel α β ∧ parallel β γ → parallel α γ) ∧
  (perp_line_plane l α ∧ parallel α β → perp_line_plane l β) ∧
  (perp_line_plane l α ∧ perp_line_plane m α → parallel_line l m) :=
by sorry

end NUMINAMATH_CALUDE_geometry_relations_l3529_352966


namespace NUMINAMATH_CALUDE_disease_mortality_percentage_l3529_352996

theorem disease_mortality_percentage (population : ℝ) 
  (h1 : population > 0) 
  (affected_percentage : ℝ) 
  (h2 : affected_percentage = 15) 
  (death_percentage : ℝ) 
  (h3 : death_percentage = 8) : 
  (affected_percentage / 100) * (death_percentage / 100) * 100 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_disease_mortality_percentage_l3529_352996


namespace NUMINAMATH_CALUDE_reciprocal_difference_product_relation_l3529_352969

theorem reciprocal_difference_product_relation :
  ∃ (a b : ℕ), a > b ∧ (1 : ℚ) / (a - b) = 3 * (1 : ℚ) / (a * b) :=
by
  use 6, 2
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_product_relation_l3529_352969


namespace NUMINAMATH_CALUDE_four_five_equality_and_precision_l3529_352929

/-- Represents a decimal number with its value and precision -/
structure Decimal where
  value : ℚ
  precision : ℕ

/-- 4.5 as a Decimal -/
def d1 : Decimal := { value := 4.5, precision := 1 }

/-- 4.50 as a Decimal -/
def d2 : Decimal := { value := 4.5, precision := 2 }

/-- Two Decimals are equal in magnitude if their values are equal -/
def equal_magnitude (a b : Decimal) : Prop := a.value = b.value

/-- Two Decimals differ in precision if their precisions are different -/
def differ_precision (a b : Decimal) : Prop := a.precision ≠ b.precision

/-- Theorem stating that 4.5 and 4.50 are equal in magnitude but differ in precision -/
theorem four_five_equality_and_precision : 
  equal_magnitude d1 d2 ∧ differ_precision d1 d2 := by
  sorry

end NUMINAMATH_CALUDE_four_five_equality_and_precision_l3529_352929


namespace NUMINAMATH_CALUDE_remaining_statue_weight_l3529_352931

/-- Represents the weights of Hammond's statues and marble block -/
structure HammondStatues where
  initial_weight : ℝ
  first_statue : ℝ
  second_statue : ℝ
  discarded_marble : ℝ

/-- Theorem stating the weight of each remaining statue -/
theorem remaining_statue_weight (h : HammondStatues)
  (h_initial : h.initial_weight = 80)
  (h_first : h.first_statue = 10)
  (h_second : h.second_statue = 18)
  (h_discarded : h.discarded_marble = 22)
  (h_equal_remaining : ∃ x : ℝ, 
    h.initial_weight - h.discarded_marble - h.first_statue - h.second_statue = 2 * x) :
  ∃ x : ℝ, x = 15 ∧ 
    h.initial_weight - h.discarded_marble - h.first_statue - h.second_statue = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_remaining_statue_weight_l3529_352931


namespace NUMINAMATH_CALUDE_coupon1_best_l3529_352965

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

def coupon2_discount (x : ℝ) : ℝ := 30

def coupon3_discount (x : ℝ) : ℝ := 0.25 * (x - 150)

theorem coupon1_best (x : ℝ) (h1 : x > 100) : 
  (coupon1_discount x > coupon2_discount x ∧ coupon1_discount x > coupon3_discount x) ↔ 
  (200 < x ∧ x < 375) := by sorry

end NUMINAMATH_CALUDE_coupon1_best_l3529_352965


namespace NUMINAMATH_CALUDE_value_of_a_l3529_352935

theorem value_of_a (a : ℚ) : a + (2 * a / 5) = 9 / 5 → a = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3529_352935


namespace NUMINAMATH_CALUDE_sin_15_30_75_product_l3529_352921

theorem sin_15_30_75_product : Real.sin (15 * π / 180) * Real.sin (30 * π / 180) * Real.sin (75 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_30_75_product_l3529_352921
