import Mathlib

namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l570_57023

/-- Given an election with two candidates, prove that the winning candidate
    received 60% of the votes under the given conditions. -/
theorem winning_candidate_percentage
  (total_votes : ℕ)
  (winning_margin : ℕ)
  (h_total : total_votes = 1400)
  (h_margin : winning_margin = 280) :
  (winning_votes : ℕ) →
  (losing_votes : ℕ) →
  (winning_votes + losing_votes = total_votes) →
  (winning_votes = losing_votes + winning_margin) →
  (winning_votes : ℚ) / total_votes = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l570_57023


namespace NUMINAMATH_CALUDE_count_integer_pairs_l570_57048

theorem count_integer_pairs : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + 2*p.2 < 40) (Finset.product (Finset.range 40) (Finset.range 40))).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l570_57048


namespace NUMINAMATH_CALUDE_power_product_result_l570_57090

theorem power_product_result : (-1.5) ^ 2021 * (2/3) ^ 2023 = -(4/9) := by sorry

end NUMINAMATH_CALUDE_power_product_result_l570_57090


namespace NUMINAMATH_CALUDE_square_sum_not_equal_sum_squares_l570_57083

theorem square_sum_not_equal_sum_squares : ∃ (a b : ℝ), a^2 + b^2 ≠ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_not_equal_sum_squares_l570_57083


namespace NUMINAMATH_CALUDE_exists_n_divides_1991_l570_57024

theorem exists_n_divides_1991 : ∃ n : ℕ, n > 2 ∧ (2 * 10^(n+1) - 9) % 1991 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_divides_1991_l570_57024


namespace NUMINAMATH_CALUDE_gcf_of_2835_and_8960_l570_57030

theorem gcf_of_2835_and_8960 : Nat.gcd 2835 8960 = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_2835_and_8960_l570_57030


namespace NUMINAMATH_CALUDE_line_through_intersection_with_equal_intercepts_l570_57095

/-- The equation of a line passing through the intersection of two given lines and having equal intercepts on the coordinate axes -/
theorem line_through_intersection_with_equal_intercepts :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, x + 2*y - 6 = 0 ∧ x - 2*y + 2 = 0 → a*x + b*y + c = 0) ∧
    (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ a*x₁ + c = 0 ∧ b*x₂ + c = 0 ∧ x₁ = x₂) →
    (a = 1 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4) := by
  sorry

end NUMINAMATH_CALUDE_line_through_intersection_with_equal_intercepts_l570_57095


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l570_57087

theorem modulus_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 - i^3) * (1 + 2*i)
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l570_57087


namespace NUMINAMATH_CALUDE_number_fraction_problem_l570_57063

theorem number_fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (1/3 : ℝ) * (2/5 : ℝ) * N = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l570_57063


namespace NUMINAMATH_CALUDE_two_digit_number_property_l570_57031

theorem two_digit_number_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ x y : ℕ, 
    n = 10 * x + y ∧ 
    x < 10 ∧ 
    y < 10 ∧ 
    n = x^3 + y^2) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l570_57031


namespace NUMINAMATH_CALUDE_right_handed_players_count_l570_57034

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 120)
  (h2 : throwers = 45)
  (h3 : throwers ≤ total_players)
  (h4 : 5 * (total_players - throwers) % 5 = 0) -- Ensures divisibility by 5
  : (throwers + (3 * (total_players - throwers) / 5) : ℕ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l570_57034


namespace NUMINAMATH_CALUDE_minimum_additional_candies_l570_57021

theorem minimum_additional_candies 
  (initial_candies : ℕ) 
  (num_students : ℕ) 
  (additional_candies : ℕ) : 
  initial_candies = 237 →
  num_students = 31 →
  additional_candies = 11 →
  (∃ (candies_per_student : ℕ), 
    (initial_candies + additional_candies) = num_students * candies_per_student) ∧
  (∀ (x : ℕ), x < additional_candies →
    ¬(∃ (y : ℕ), (initial_candies + x) = num_students * y)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_candies_l570_57021


namespace NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l570_57013

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 2x = 0 -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem stating that 2x = 0 is a linear equation -/
theorem two_x_eq_zero_is_linear : is_linear_equation f := by
  sorry


end NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l570_57013


namespace NUMINAMATH_CALUDE_no_base_all_prime_l570_57072

/-- For any base b ≥ 2, there exists a number of the form 11...1 
    with (b^2 - 1) ones in base b that is not prime. -/
theorem no_base_all_prime (b : ℕ) (hb : b ≥ 2) : 
  ∃ N : ℕ, (∃ k : ℕ, N = (b^(2*k) - 1) / (b^2 - 1)) ∧ ¬ Prime N := by
  sorry

end NUMINAMATH_CALUDE_no_base_all_prime_l570_57072


namespace NUMINAMATH_CALUDE_example_is_fractional_equation_l570_57028

/-- Definition of a fractional equation -/
def is_fractional_equation (eq : Prop) : Prop :=
  ∃ (x : ℝ) (f g : ℝ → ℝ) (h : ℝ → ℝ), 
    (∀ y, f y ≠ 0 ∧ g y ≠ 0) ∧ 
    eq ↔ (h x / f x - 3 / g x = 1) ∧
    (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ f x = a * x + b) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ g x = c * x + d)

/-- The equation (x / (2x - 1)) - (3 / (2x + 1)) = 1 is a fractional equation -/
theorem example_is_fractional_equation : 
  is_fractional_equation (∃ x : ℝ, x / (2 * x - 1) - 3 / (2 * x + 1) = 1) :=
sorry

end NUMINAMATH_CALUDE_example_is_fractional_equation_l570_57028


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_12n_integer_l570_57001

theorem smallest_n_for_sqrt_12n_integer :
  ∀ n : ℕ+, (∃ k : ℕ+, (12 * n : ℕ) = k ^ 2) → n ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_12n_integer_l570_57001


namespace NUMINAMATH_CALUDE_slope_theorem_l570_57051

/-- Given two points A(-3, 8) and B(5, y) in a coordinate plane, 
    if the slope of the line through A and B is -1/2, then y = 4. -/
theorem slope_theorem (y : ℝ) : 
  let A : ℝ × ℝ := (-3, 8)
  let B : ℝ × ℝ := (5, y)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -1/2 → y = 4 := by
sorry


end NUMINAMATH_CALUDE_slope_theorem_l570_57051


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l570_57003

-- Define the original inequality
def original_inequality (x : ℝ) : Prop := 2 * x^2 - 5*x - 3 ≥ 0

-- Define the solution to the original inequality
def solution_inequality (x : ℝ) : Prop := x ≤ -1/2 ∨ x ≥ 3

-- Define the proposed necessary but not sufficient condition
def proposed_condition (x : ℝ) : Prop := x < -1 ∨ x > 4

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ x : ℝ, original_inequality x ↔ solution_inequality x) →
  (∀ x : ℝ, solution_inequality x → proposed_condition x) ∧
  ¬(∀ x : ℝ, proposed_condition x → solution_inequality x) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l570_57003


namespace NUMINAMATH_CALUDE_range_of_a_l570_57016

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, (1/a) - (1/x) ≤ 2*x) : a ≥ Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l570_57016


namespace NUMINAMATH_CALUDE_sequence_properties_l570_57081

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = b n - 1

theorem sequence_properties
  (a b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : sum_of_terms b S)
  (h_a2b1 : a 2 = b 1)
  (h_a5b2 : a 5 = b 2) :
  (∀ n : ℕ, a n = 2 * n - 6) ∧
  (∀ n : ℕ, S n = (-2)^n - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l570_57081


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l570_57076

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y - x * y = 0 → a + 2 * b ≤ x + 2 * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l570_57076


namespace NUMINAMATH_CALUDE_small_painting_price_l570_57022

/-- Represents the price of paintings and sales data for Noah's art business -/
structure PaintingSales where
  large_price : ℕ
  small_price : ℕ
  last_month_large : ℕ
  last_month_small : ℕ
  this_month_total : ℕ

/-- Theorem stating that given the conditions, the price of a small painting is $30 -/
theorem small_painting_price (sales : PaintingSales) 
  (h1 : sales.large_price = 60)
  (h2 : sales.last_month_large = 8)
  (h3 : sales.last_month_small = 4)
  (h4 : sales.this_month_total = 1200) :
  sales.small_price = 30 := by
  sorry

#check small_painting_price

end NUMINAMATH_CALUDE_small_painting_price_l570_57022


namespace NUMINAMATH_CALUDE_triangle_properties_l570_57042

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a ≠ abc.b)
  (h2 : abc.c = Real.sqrt 3)
  (h3 : (Real.cos abc.A)^2 - (Real.cos abc.B)^2 = Real.sqrt 3 * Real.sin abc.A * Real.cos abc.A - Real.sqrt 3 * Real.sin abc.B * Real.cos abc.B)
  (h4 : Real.sin abc.A = 4/5) :
  abc.C = π/3 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = (8 * Real.sqrt 3 + 18) / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l570_57042


namespace NUMINAMATH_CALUDE_modulus_of_z_l570_57069

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l570_57069


namespace NUMINAMATH_CALUDE_spadesuit_problem_l570_57078

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_problem_l570_57078


namespace NUMINAMATH_CALUDE_trig_identity_l570_57000

theorem trig_identity (θ : Real) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l570_57000


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l570_57046

/-- 
Given a line equation (2k-1)x-(k+3)y-(k-11)=0 where k is any real number,
prove that this line always passes through the point (2, 3).
-/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l570_57046


namespace NUMINAMATH_CALUDE_counterexample_exists_l570_57002

theorem counterexample_exists :
  ∃ (n : ℕ), n ≥ 2 ∧ ¬(∃ (k : ℕ), (2^(2^n) % (2^n - 1)) = 4^k) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l570_57002


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l570_57055

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of the quadratic equation 5x^2 - 11x + 4 is 41 -/
theorem quadratic_discriminant : discriminant 5 (-11) 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l570_57055


namespace NUMINAMATH_CALUDE_distance_A_l570_57010

def A : ℝ × ℝ := (0, 15)
def B : ℝ × ℝ := (0, 18)
def C : ℝ × ℝ := (4, 10)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

theorem distance_A'B' :
  ∀ (A' B' : ℝ × ℝ),
    on_line_y_eq_x A' →
    on_line_y_eq_x B' →
    collinear A A' C →
    collinear B B' C →
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_l570_57010


namespace NUMINAMATH_CALUDE_sandys_scooter_gain_percent_l570_57091

/-- Calculates the gain percent for a transaction given purchase price, repair cost, and selling price -/
def gainPercent (purchasePrice repairCost sellingPrice : ℚ) : ℚ :=
  let totalCost := purchasePrice + repairCost
  let gain := sellingPrice - totalCost
  (gain / totalCost) * 100

/-- Theorem: The gain percent for Sandy's scooter transaction is 10% -/
theorem sandys_scooter_gain_percent :
  gainPercent 900 300 1320 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sandys_scooter_gain_percent_l570_57091


namespace NUMINAMATH_CALUDE_league_games_count_l570_57044

/-- Calculates the number of games in a league season -/
def number_of_games (n : ℕ) (k : ℕ) : ℕ :=
  (n * (n - 1) / 2) * k

/-- Theorem: In a league with 50 teams, where each team plays every other team 4 times,
    the total number of games played in the season is 4900. -/
theorem league_games_count : number_of_games 50 4 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l570_57044


namespace NUMINAMATH_CALUDE_point_transformation_to_third_quadrant_l570_57027

/-- Given a point (a, b) in the fourth quadrant, prove that (a/b, 2b-a) is in the third quadrant -/
theorem point_transformation_to_third_quadrant (a b : ℝ) 
  (h1 : a > 0) (h2 : b < 0) : (a / b < 0) ∧ (2 * b - a < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_to_third_quadrant_l570_57027


namespace NUMINAMATH_CALUDE_meeting_arrangements_l570_57009

def number_of_schools : ℕ := 3
def members_per_school : ℕ := 6
def total_members : ℕ := 18
def host_representatives : ℕ := 3
def other_representatives : ℕ := 1

def arrange_meeting : ℕ := 
  number_of_schools * 
  (Nat.choose members_per_school host_representatives) * 
  (Nat.choose members_per_school other_representatives) * 
  (Nat.choose members_per_school other_representatives)

theorem meeting_arrangements :
  arrange_meeting = 2160 :=
sorry

end NUMINAMATH_CALUDE_meeting_arrangements_l570_57009


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l570_57062

-- Define the operation ⊙
noncomputable def bowtie (x y : ℝ) : ℝ := x + Real.sqrt (y + Real.sqrt (y + Real.sqrt (y + Real.sqrt y)))

-- State the theorem
theorem bowtie_equation_solution (h : ℝ) : 
  bowtie 8 h = 12 → h = 12 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l570_57062


namespace NUMINAMATH_CALUDE_no_real_solutions_count_l570_57070

theorem no_real_solutions_count : 
  ∀ b c : ℕ+, 
  (∃ x : ℝ, x^2 + (b:ℝ)*x + (c:ℝ) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c:ℝ)*x + (b:ℝ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_count_l570_57070


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l570_57019

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) (h : 2 * r = 4) :
  (1 / 2) * 4 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l570_57019


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l570_57026

theorem quadratic_roots_sum_squares_minimum (a : ℝ) 
  (x₁ x₂ : ℝ) (h₁ : x₁^2 + 2*a*x₁ + a^2 + 4*a - 2 = 0) 
  (h₂ : x₂^2 + 2*a*x₂ + a^2 + 4*a - 2 = 0) 
  (h₃ : x₁ ≠ x₂) :
  x₁^2 + x₂^2 ≥ 1/2 ∧ 
  (x₁^2 + x₂^2 = 1/2 ↔ a = 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l570_57026


namespace NUMINAMATH_CALUDE_basketball_win_requirement_l570_57025

theorem basketball_win_requirement (total_games : ℕ) (games_played : ℕ) (games_won : ℕ) (target_percentage : ℚ) :
  total_games = 100 →
  games_played = 60 →
  games_won = 30 →
  target_percentage = 65 / 100 →
  ∃ (remaining_wins : ℕ), 
    remaining_wins = 35 ∧
    (games_won + remaining_wins : ℚ) / total_games = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_requirement_l570_57025


namespace NUMINAMATH_CALUDE_race_track_length_l570_57082

/-- Represents a runner in the race --/
structure Runner where
  position : ℝ
  velocity : ℝ

/-- Represents the race --/
structure Race where
  track_length : ℝ
  alberto : Runner
  bernardo : Runner
  carlos : Runner

/-- The conditions of the race --/
def race_conditions (r : Race) : Prop :=
  r.alberto.velocity > 0 ∧
  r.bernardo.velocity > 0 ∧
  r.carlos.velocity > 0 ∧
  r.alberto.position = r.track_length ∧
  r.bernardo.position = r.track_length - 36 ∧
  r.carlos.position = r.track_length - 46 ∧
  (r.track_length / r.bernardo.velocity) * r.carlos.velocity = r.track_length - 16

theorem race_track_length (r : Race) (h : race_conditions r) : r.track_length = 96 := by
  sorry

#check race_track_length

end NUMINAMATH_CALUDE_race_track_length_l570_57082


namespace NUMINAMATH_CALUDE_bricks_used_total_bricks_used_l570_57057

/-- Calculates the total number of bricks used in a construction project -/
theorem bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) (incomplete_courses : ℕ) : ℕ :=
  let complete_walls := total_walls - 1
  let complete_wall_bricks := courses_per_wall * bricks_per_course
  let incomplete_wall_courses := courses_per_wall - incomplete_courses
  let incomplete_wall_bricks := incomplete_wall_courses * bricks_per_course
  complete_walls * complete_wall_bricks + incomplete_wall_bricks

/-- Proves that the total number of bricks used is 1140 given the specific conditions -/
theorem total_bricks_used :
  bricks_used 10 20 6 3 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_bricks_used_total_bricks_used_l570_57057


namespace NUMINAMATH_CALUDE_train_length_l570_57036

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 →
  crossing_time = 29.997600191984642 →
  bridge_length = 390 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l570_57036


namespace NUMINAMATH_CALUDE_certain_number_value_l570_57007

theorem certain_number_value : ∃ x : ℝ, 
  (3 - (1/5) * 390 = x - (1/7) * 210 + 114) ∧ 
  (3 - (1/5) * 390 - (x - (1/7) * 210) = 114) → 
  x = -159 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l570_57007


namespace NUMINAMATH_CALUDE_businessmen_who_drank_nothing_l570_57073

/-- The number of businessmen who drank neither coffee, tea, nor soda -/
theorem businessmen_who_drank_nothing (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_and_tea tea_and_soda coffee_and_soda : ℕ) (all_three : ℕ) : 
  total = 40 →
  coffee = 20 →
  tea = 15 →
  soda = 10 →
  coffee_and_tea = 8 →
  tea_and_soda = 4 →
  coffee_and_soda = 3 →
  all_three = 2 →
  total - (coffee + tea + soda - coffee_and_tea - tea_and_soda - coffee_and_soda + all_three) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_who_drank_nothing_l570_57073


namespace NUMINAMATH_CALUDE_d_value_for_lines_l570_57033

/-- Two straight lines pass through four points in 3D space -/
def line_through_points (a b c d : ℝ) (k : ℕ) : Prop :=
  ∃ (l₁ l₂ : Set (ℝ × ℝ × ℝ)),
    l₁ ≠ l₂ ∧
    (1, 0, a) ∈ l₁ ∧ (1, 0, a) ∈ l₂ ∧
    (b, 1, 0) ∈ l₁ ∧ (b, 1, 0) ∈ l₂ ∧
    (0, c, 1) ∈ l₁ ∧ (0, c, 1) ∈ l₂ ∧
    (k * d, k * d, -d) ∈ l₁ ∧ (k * d, k * d, -d) ∈ l₂

/-- The theorem stating the possible values of d -/
theorem d_value_for_lines (k : ℕ) (h1 : k ≠ 6) (h2 : k ≠ 1) :
  ∀ a b c d : ℝ, line_through_points a b c d k → d = -k / (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_d_value_for_lines_l570_57033


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l570_57018

theorem eight_digit_divisible_by_nine (n : Nat) : 
  (9673 * 10000 + n * 1000 + 432) % 9 = 0 ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l570_57018


namespace NUMINAMATH_CALUDE_smallest_n_fourth_fifth_power_l570_57054

theorem smallest_n_fourth_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), 3 * n = x^4) ∧ 
  (∃ (y : ℕ), 2 * n = y^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (a : ℕ), 3 * m = a^4) → 
    (∃ (b : ℕ), 2 * m = b^5) → 
    m ≥ 6912) ∧
  n = 6912 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_fourth_fifth_power_l570_57054


namespace NUMINAMATH_CALUDE_star_polygon_angle_sum_l570_57012

/-- A star polygon created from a regular n-gon --/
structure StarPolygon where
  n : ℕ
  n_ge_6 : n ≥ 6

/-- The sum of interior angles of a star polygon --/
def sum_interior_angles (s : StarPolygon) : ℝ :=
  180 * (s.n - 2)

/-- Theorem: The sum of interior angles of a star polygon is 180°(n-2) --/
theorem star_polygon_angle_sum (s : StarPolygon) :
  sum_interior_angles s = 180 * (s.n - 2) :=
by sorry

end NUMINAMATH_CALUDE_star_polygon_angle_sum_l570_57012


namespace NUMINAMATH_CALUDE_prime_cube_plus_one_l570_57093

theorem prime_cube_plus_one (p : ℕ) (hp : Prime p) :
  (∃ (x y : ℕ), p^x = y^3 + 1) ↔ p = 2 ∨ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_plus_one_l570_57093


namespace NUMINAMATH_CALUDE_conference_handshakes_l570_57011

/-- The number of handshakes in a conference with n participants -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that a conference with 10 participants results in 45 handshakes -/
theorem conference_handshakes : handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l570_57011


namespace NUMINAMATH_CALUDE_pencils_indeterminate_l570_57050

/-- Represents the contents of a drawer -/
structure Drawer where
  initial_crayons : ℕ
  added_crayons : ℕ
  final_crayons : ℕ
  pencils : ℕ

/-- Theorem stating that the number of pencils cannot be determined -/
theorem pencils_indeterminate (d : Drawer) 
  (h1 : d.initial_crayons = 41)
  (h2 : d.added_crayons = 12)
  (h3 : d.final_crayons = 53)
  : ¬ ∃ (n : ℕ), ∀ (d' : Drawer), 
    d'.initial_crayons = d.initial_crayons ∧ 
    d'.added_crayons = d.added_crayons ∧ 
    d'.final_crayons = d.final_crayons → 
    d'.pencils = n :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_indeterminate_l570_57050


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l570_57094

theorem trigonometric_expression_equality : 
  (Real.sin (10 * π / 180) * Real.sin (80 * π / 180)) / 
  (Real.cos (35 * π / 180)^2 - Real.sin (35 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l570_57094


namespace NUMINAMATH_CALUDE_rational_root_of_cubic_l570_57058

/-- Given a cubic polynomial with rational coefficients, if 3 + √5 is a root
    and another root is rational, then the rational root is -6 -/
theorem rational_root_of_cubic (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 3 + Real.sqrt 5) →
  (∃ r : ℚ, r^3 + a*r^2 + b*r + c = 0) →
  (∃ r : ℚ, r^3 + a*r^2 + b*r + c = 0 ∧ r = -6) :=
by sorry

end NUMINAMATH_CALUDE_rational_root_of_cubic_l570_57058


namespace NUMINAMATH_CALUDE_one_point_inside_circle_l570_57065

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a predicate for three points being collinear
def collinear (p q r : Point) : Prop := sorry

-- Define a predicate for a point being on a circle determined by three other points
def onCircle (p q r s : Point) : Prop := sorry

-- Define a predicate for a point being inside a circle determined by three other points
def insideCircle (p q r s : Point) : Prop := sorry

-- Theorem statement
theorem one_point_inside_circle (A B C D : Point) 
  (h_not_collinear : ¬(collinear A B C) ∧ ¬(collinear A B D) ∧ ¬(collinear A C D) ∧ ¬(collinear B C D))
  (h_not_on_circle : ¬(onCircle A B C D) ∧ ¬(onCircle A B D C) ∧ ¬(onCircle A C D B) ∧ ¬(onCircle B C D A)) :
  insideCircle A B C D ∨ insideCircle A B D C ∨ insideCircle A C D B ∨ insideCircle B C D A :=
by sorry

end NUMINAMATH_CALUDE_one_point_inside_circle_l570_57065


namespace NUMINAMATH_CALUDE_cafe_visits_l570_57061

/-- The number of people in the club -/
def n : ℕ := 9

/-- The number of people who visit the cafe each day -/
def k : ℕ := 3

/-- The number of days -/
def days : ℕ := 360

/-- The number of times each pair visits the cafe -/
def x : ℕ := 30

theorem cafe_visits :
  (n.choose 2) * x = days * (k.choose 2) := by sorry

end NUMINAMATH_CALUDE_cafe_visits_l570_57061


namespace NUMINAMATH_CALUDE_binary_multiplication_subtraction_l570_57059

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to a binary representation as a list of bits. -/
def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_subtraction :
  let a := binary_to_nat [true, false, true, true]  -- 1101₂
  let b := binary_to_nat [true, true, true]         -- 111₂
  let c := binary_to_nat [true, false, true]        -- 101₂
  nat_to_binary ((a * b) - c) = [false, false, false, true, false, false, true] -- 1001000₂
:= by sorry

end NUMINAMATH_CALUDE_binary_multiplication_subtraction_l570_57059


namespace NUMINAMATH_CALUDE_smallest_linear_combination_l570_57077

theorem smallest_linear_combination (m n : ℤ) : ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 2017 * a + 48576 * b) ∧ 
  ∀ (l : ℕ), l > 0 → (∃ (c d : ℤ), l = 2017 * c + 48576 * d) → k ≤ l :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_l570_57077


namespace NUMINAMATH_CALUDE_store_discount_is_ten_percent_l570_57041

/-- Calculates the discount percentage given the number of items, cost per item, 
    discount threshold, and final cost after discount. -/
def discount_percentage (num_items : ℕ) (cost_per_item : ℚ) 
  (discount_threshold : ℚ) (final_cost : ℚ) : ℚ :=
  let total_cost := num_items * cost_per_item
  let discount_amount := total_cost - final_cost
  let eligible_amount := total_cost - discount_threshold
  (discount_amount / eligible_amount) * 100

/-- Proves that the discount percentage is 10% for the given scenario. -/
theorem store_discount_is_ten_percent :
  discount_percentage 7 200 1000 1360 = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_is_ten_percent_l570_57041


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l570_57053

theorem wrapping_paper_fraction (total_fraction : Rat) (num_presents : Nat) 
  (h1 : total_fraction = 5 / 12)
  (h2 : num_presents = 4) :
  total_fraction / num_presents = 5 / 48 := by
sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l570_57053


namespace NUMINAMATH_CALUDE_integer_solutions_for_system_l570_57032

theorem integer_solutions_for_system (x y : ℤ) : 
  x^2 = (y+1)^2 + 1 ∧ 
  x^2 - (y+1)^2 = 1 ∧ 
  (x-y-1) * (x+y+1) = 1 → 
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_for_system_l570_57032


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l570_57004

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, x ≤ 4 ↔ (x : ℝ)^4 / (x : ℝ)^2 < 18 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l570_57004


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l570_57015

theorem complex_fraction_simplification :
  (2 + 4 * Complex.I) / ((1 + Complex.I)^2) = 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l570_57015


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l570_57099

theorem minimum_value_of_function (x : ℝ) (h : x > 3) :
  (1 / (x - 3) + x) ≥ 5 ∧ ∃ y > 3, 1 / (y - 3) + y = 5 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l570_57099


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l570_57039

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 ≥ 2*x ↔ x ∈ Set.Iic 0 ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l570_57039


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l570_57060

/-- The opposite of a real number x is the number that, when added to x, yields zero. -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of -3 is 3. -/
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l570_57060


namespace NUMINAMATH_CALUDE_gcd_properties_l570_57086

theorem gcd_properties (a b n : ℕ) (c : ℤ) (h1 : a ≠ 0) (h2 : c > 0) :
  let d := Nat.gcd a b
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧
  (Nat.gcd (a * c.natAbs) (b * c.natAbs) = c.natAbs * Nat.gcd a b) :=
by sorry

end NUMINAMATH_CALUDE_gcd_properties_l570_57086


namespace NUMINAMATH_CALUDE_fraction_scaling_l570_57071

theorem fraction_scaling (x y : ℝ) :
  (3*x + 3*y) / ((3*x)^2 + (3*y)^2) = (1/3) * ((x + y) / (x^2 + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_scaling_l570_57071


namespace NUMINAMATH_CALUDE_correct_substitution_l570_57074

theorem correct_substitution (x y : ℝ) : 
  (x = 3*y - 1 ∧ x - 2*y = 4) → (3*y - 1 - 2*y = 4) := by
  sorry

end NUMINAMATH_CALUDE_correct_substitution_l570_57074


namespace NUMINAMATH_CALUDE_four_integers_sum_l570_57037

theorem four_integers_sum (a b c d : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧
  a + b + c = 6 ∧
  a + b + d = 7 ∧
  a + c + d = 8 ∧
  b + c + d = 9 →
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_four_integers_sum_l570_57037


namespace NUMINAMATH_CALUDE_total_students_proof_l570_57067

/-- The number of students who knew about the event -/
def students_who_knew : ℕ := 40

/-- The frequency of students who knew about the event -/
def frequency : ℚ := 8/10

/-- The total number of students participating in the competition -/
def total_students : ℕ := 50

/-- Theorem stating that the total number of students is 50 given the conditions -/
theorem total_students_proof : 
  (students_who_knew : ℚ) / frequency = total_students := by sorry

end NUMINAMATH_CALUDE_total_students_proof_l570_57067


namespace NUMINAMATH_CALUDE_linda_money_l570_57049

/-- Represents the price of a single jean in dollars -/
def jean_price : ℕ := 11

/-- Represents the price of a single tee in dollars -/
def tee_price : ℕ := 8

/-- Represents the number of tees sold in a day -/
def tees_sold : ℕ := 7

/-- Represents the number of jeans sold in a day -/
def jeans_sold : ℕ := 4

/-- Calculates the total money Linda had at the end of the day -/
def total_money : ℕ := jean_price * jeans_sold + tee_price * tees_sold

theorem linda_money : total_money = 100 := by
  sorry

end NUMINAMATH_CALUDE_linda_money_l570_57049


namespace NUMINAMATH_CALUDE_inequality_solution_l570_57043

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4) / (x^2 - 1) > 0 ↔ x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l570_57043


namespace NUMINAMATH_CALUDE_age_sum_product_total_l570_57088

theorem age_sum_product_total (elvie_age arielle_age : ℕ) : 
  elvie_age = 10 → arielle_age = 11 → 
  (elvie_age + arielle_age) + (elvie_age * arielle_age) = 131 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_product_total_l570_57088


namespace NUMINAMATH_CALUDE_hyperbola_inequality_l570_57056

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 = 1

-- Define the line
def line (x : ℝ) : Prop := x = 3

-- Define the intersection points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (3, -1)

-- Define a point P on the hyperbola
def P (x y : ℝ) : Prop := hyperbola x y

-- Define the vector representation of OP
def OP (a b : ℝ) : ℝ × ℝ := (2*a + 2*b, a - b)

theorem hyperbola_inequality (a b : ℝ) :
  (∀ x y, P x y → OP a b = (x, y)) → |a + b| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_inequality_l570_57056


namespace NUMINAMATH_CALUDE_optimal_selling_price_l570_57017

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -20 * x^2 + 200 * x + 4000

/-- Theorem stating the optimal selling price to maximize profit -/
theorem optimal_selling_price :
  let original_price : ℝ := 80
  let initial_selling_price : ℝ := 90
  let initial_quantity : ℝ := 400
  let price_sensitivity : ℝ := 20  -- Units decrease per 1 yuan increase

  ∃ (max_profit_price : ℝ), 
    (∀ x, 0 < x → x ≤ 20 → profit_function x ≤ profit_function max_profit_price) ∧
    max_profit_price = 95 := by
  sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l570_57017


namespace NUMINAMATH_CALUDE_gcf_120_180_300_l570_57047

theorem gcf_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_120_180_300_l570_57047


namespace NUMINAMATH_CALUDE_eighth_term_is_15_l570_57079

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term function
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem eighth_term_is_15 : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_15_l570_57079


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l570_57075

theorem no_infinite_sequence_exists : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n, k n ≠ 0) ∧ 
  (∀ n, k (n + 1) = k n - 1 / k n) ∧ 
  (∀ n, k n * k (n + 1) ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l570_57075


namespace NUMINAMATH_CALUDE_two_out_of_three_correct_probability_l570_57084

def probability_correct_forecast : ℝ := 0.8

def probability_two_out_of_three_correct : ℝ :=
  3 * probability_correct_forecast^2 * (1 - probability_correct_forecast)

theorem two_out_of_three_correct_probability :
  probability_two_out_of_three_correct = 0.384 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_three_correct_probability_l570_57084


namespace NUMINAMATH_CALUDE_annie_gives_mary_25_crayons_l570_57080

/-- The number of crayons Annie gives to Mary -/
def crayons_given_to_mary (pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

/-- Theorem stating that Annie gives 25 crayons to Mary -/
theorem annie_gives_mary_25_crayons :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

#eval crayons_given_to_mary 21 36

end NUMINAMATH_CALUDE_annie_gives_mary_25_crayons_l570_57080


namespace NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l570_57097

theorem smallest_sum_consecutive_integers (n : ℕ) : 
  (∀ k < n, k * (k + 1) ≤ 420) → n * (n + 1) > 420 → n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l570_57097


namespace NUMINAMATH_CALUDE_percentage_to_decimal_two_percent_to_decimal_l570_57020

theorem percentage_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem two_percent_to_decimal : (2 : ℚ) / 100 = 0.02 := by sorry

end NUMINAMATH_CALUDE_percentage_to_decimal_two_percent_to_decimal_l570_57020


namespace NUMINAMATH_CALUDE_elevator_force_theorem_gavrila_force_l570_57029

/-- The force exerted by a person on the floor of a decelerating elevator -/
def elevatorForce (m : ℝ) (g a : ℝ) : ℝ := m * (g - a)

/-- Theorem: The force exerted by a person on the floor of a decelerating elevator
    is equal to the person's mass multiplied by the difference between
    gravitational acceleration and the elevator's deceleration -/
theorem elevator_force_theorem (m g a : ℝ) :
  elevatorForce m g a = m * (g - a) := by
  sorry

/-- Corollary: For Gavrila's specific case -/
theorem gavrila_force :
  elevatorForce 70 10 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_elevator_force_theorem_gavrila_force_l570_57029


namespace NUMINAMATH_CALUDE_tony_squat_weight_l570_57092

-- Define Tony's lifting capabilities
def curl_weight : ℕ := 90
def military_press_weight : ℕ := 2 * curl_weight
def squat_weight : ℕ := 5 * military_press_weight

-- Theorem to prove
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end NUMINAMATH_CALUDE_tony_squat_weight_l570_57092


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l570_57096

theorem unique_solution_quadratic (p : ℝ) : 
  p ≠ 0 ∧ (∃! x, p * x^2 - 8 * x + 2 = 0) ↔ p = 8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l570_57096


namespace NUMINAMATH_CALUDE_quadratic_root_value_l570_57038

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 7 * x + k = 0 ↔ x = (7 + Real.sqrt 17) / 4 ∨ x = (7 - Real.sqrt 17) / 4) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l570_57038


namespace NUMINAMATH_CALUDE_range_of_x_l570_57005

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the interval [-1, 1]
def I : Set ℝ := Set.Icc (-1 : ℝ) 1

-- Define the theorem
theorem range_of_x (h1 : Monotone f) (h2 : Set.MapsTo f I I) 
  (h3 : ∀ x, f (x - 2) < f (1 - x)) :
  ∃ S : Set ℝ, S = Set.Ico 1 (3/2) ∧ ∀ x, x ∈ S ↔ 
    (x - 2 ∈ I ∧ 1 - x ∈ I ∧ f (x - 2) < f (1 - x)) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l570_57005


namespace NUMINAMATH_CALUDE_square_sum_inequality_l570_57068

theorem square_sum_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l570_57068


namespace NUMINAMATH_CALUDE_license_plate_count_l570_57006

/-- The number of possible letters for each position in the license plate -/
def num_letters : ℕ := 26

/-- The number of possible odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits -/
def num_even_digits : ℕ := 5

/-- The total number of license plates with the given conditions -/
def total_license_plates : ℕ := num_letters^3 * num_odd_digits * (num_odd_digits * num_even_digits + num_even_digits * num_odd_digits)

theorem license_plate_count : total_license_plates = 455625 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l570_57006


namespace NUMINAMATH_CALUDE_theta_values_l570_57089

theorem theta_values (a b : ℝ) (θ : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.cos (x + 2 * θ) + b * x + 3
  (f 1 = 5 ∧ f (-1) = 1) → (θ = π / 4 ∨ θ = -π / 4) :=
by sorry

end NUMINAMATH_CALUDE_theta_values_l570_57089


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l570_57066

theorem largest_four_digit_divisible_by_88 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 88 = 0 → n ≤ 9944 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l570_57066


namespace NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l570_57085

/-- Represents the number of seeds Steven needs to collect for his assignment -/
def total_seeds_required : ℕ := 60

/-- Represents the average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- Represents the average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- Represents the average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- Represents the number of apples Steven has -/
def steven_apples : ℕ := 4

/-- Represents the number of pears Steven has -/
def steven_pears : ℕ := 3

/-- Represents the number of grapes Steven has -/
def steven_grapes : ℕ := 9

/-- Theorem stating that Steven needs 3 more seeds to fulfill his assignment -/
theorem steven_needs_three_more_seeds :
  total_seeds_required - (steven_apples * apple_seeds + steven_pears * pear_seeds + steven_grapes * grape_seeds) = 3 := by
  sorry

end NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l570_57085


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l570_57040

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_unique :
  ∀ a b c : ℝ,
  (f a b c (-1) = 0) →
  (∀ x : ℝ, x ≤ f a b c x) →
  (∀ x : ℝ, f a b c x ≤ (1 + x^2) / 2) →
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l570_57040


namespace NUMINAMATH_CALUDE_circle_symmetry_l570_57008

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define Circle C₁
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define Circle C₂
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry with respect to a line
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  line_of_symmetry ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  (x2 - x1) * (x2 - x1) = (y2 - y1) * (y2 - y1)

-- Theorem statement
theorem circle_symmetry :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_C1 x1 y1 →
  circle_C2 x2 y2 →
  symmetric_points x1 y1 x2 y2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l570_57008


namespace NUMINAMATH_CALUDE_two_ways_to_combine_fractions_l570_57064

theorem two_ways_to_combine_fractions : ∃ (f g : ℚ → ℚ → ℚ → ℚ),
  f (1/8) (1/9) (1/28) = 1/2016 ∧
  g (1/8) (1/9) (1/28) = 1/2016 ∧
  f ≠ g :=
by sorry

end NUMINAMATH_CALUDE_two_ways_to_combine_fractions_l570_57064


namespace NUMINAMATH_CALUDE_gabriel_diabetes_capsules_l570_57098

theorem gabriel_diabetes_capsules (forgot_days took_days : ℕ) 
  (h1 : forgot_days = 3) 
  (h2 : took_days = 28) : 
  forgot_days + took_days = 31 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_diabetes_capsules_l570_57098


namespace NUMINAMATH_CALUDE_batsman_average_l570_57052

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  (previous_total = 16 * previous_average) →
  (previous_total + 87) / 17 = previous_average + 3 →
  (previous_total + 87) / 17 = 39 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l570_57052


namespace NUMINAMATH_CALUDE_bird_families_left_l570_57014

theorem bird_families_left (total : ℕ) (to_africa : ℕ) (to_asia : ℕ) 
  (h1 : total = 85) (h2 : to_africa = 23) (h3 : to_asia = 37) : 
  total - (to_africa + to_asia) = 25 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_left_l570_57014


namespace NUMINAMATH_CALUDE_three_hundred_thousand_cubed_times_fifty_l570_57045

theorem three_hundred_thousand_cubed_times_fifty :
  (300000 ^ 3) * 50 = 1350000000000000000 := by sorry

end NUMINAMATH_CALUDE_three_hundred_thousand_cubed_times_fifty_l570_57045


namespace NUMINAMATH_CALUDE_complement_B_union_A_C_subset_B_implies_a_range_l570_57035

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem for part (1)
theorem complement_B_union_A :
  (Set.univ \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x} :=
sorry

-- Theorem for part (2)
theorem C_subset_B_implies_a_range (a : ℝ) :
  C a ⊆ B → 2 ≤ a ∧ a ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_complement_B_union_A_C_subset_B_implies_a_range_l570_57035
