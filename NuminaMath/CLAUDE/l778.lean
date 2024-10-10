import Mathlib

namespace wind_pressure_theorem_l778_77890

/-- The pressure-area-velocity relationship for wind on a sail -/
theorem wind_pressure_theorem (k : ℝ) :
  (∃ P A V : ℝ, P = k * A * V^2 ∧ P = 1.25 ∧ A = 1 ∧ V = 20) →
  (∃ P A V : ℝ, P = k * A * V^2 ∧ P = 20 ∧ A = 4 ∧ V = 40) :=
by sorry

end wind_pressure_theorem_l778_77890


namespace range_of_a_l778_77844

def proposition_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  proposition_p a ∧ proposition_q a → a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l778_77844


namespace benny_apples_l778_77848

def dan_apples : ℕ := 9
def total_apples : ℕ := 11

theorem benny_apples : total_apples - dan_apples = 2 := by
  sorry

end benny_apples_l778_77848


namespace schedule_count_is_576_l778_77855

/-- Represents a table tennis match between two schools -/
structure TableTennisMatch where
  /-- Number of players in each school -/
  players_per_school : Nat
  /-- Number of opponents each player faces from the other school -/
  opponents_per_player : Nat
  /-- Number of rounds in the match -/
  total_rounds : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat

/-- The specific match configuration from the problem -/
def match_config : TableTennisMatch :=
  { players_per_school := 4
  , opponents_per_player := 2
  , total_rounds := 6
  , games_per_round := 4
  }

/-- Calculate the number of ways to schedule the match -/
def schedule_count (m : TableTennisMatch) : Nat :=
  (Nat.factorial m.total_rounds) * (Nat.factorial m.games_per_round)

/-- Theorem stating that the number of ways to schedule the match is 576 -/
theorem schedule_count_is_576 : schedule_count match_config = 576 := by
  sorry


end schedule_count_is_576_l778_77855


namespace log_equation_solution_l778_77804

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 2 = 9 →
  x = 2^(27/10) := by
sorry

end log_equation_solution_l778_77804


namespace perpendicular_line_through_point_l778_77891

/-- Given a line L1 with equation x - 2y + m = 0 and a point P (-1, 3),
    this theorem states that the line L2 with equation 2x + y - 1 = 0
    passes through P and is perpendicular to L1. -/
theorem perpendicular_line_through_point (m : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + m = 0
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0
  let P : ℝ × ℝ := (-1, 3)
  (L2 P.1 P.2) ∧                   -- L2 passes through P
  (∀ x1 y1 x2 y2, L1 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x1 - 2*y1) + (y2 - y1) * (-2*x1 - y1) = 0) -- L2 is perpendicular to L1
  := by sorry

end perpendicular_line_through_point_l778_77891


namespace polynomial_product_sum_l778_77895

theorem polynomial_product_sum (a b c d e : ℝ) : 
  (∀ x : ℝ, (3 * x^3 - 5 * x^2 + 4 * x - 6) * (7 - 2 * x) = a * x^4 + b * x^3 + c * x^2 + d * x + e) →
  16 * a + 8 * b + 4 * c + 2 * d + e = 42 := by
sorry

end polynomial_product_sum_l778_77895


namespace electronics_store_theorem_l778_77893

theorem electronics_store_theorem (total : ℕ) (tv : ℕ) (computer : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : tv = 9)
  (h3 : computer = 7)
  (h4 : both = 3)
  : total - (tv + computer - both) = 2 :=
by sorry

end electronics_store_theorem_l778_77893


namespace smallest_positive_integer_d_l778_77871

theorem smallest_positive_integer_d : ∃ d : ℕ+, d = 4 ∧
  (∀ d' : ℕ+, d' < d →
    ¬∃ x y : ℝ, x^2 + y^2 = 100 ∧ y = 2*x + d' ∧ x^2 + y^2 = 100 * d') ∧
  ∃ x y : ℝ, x^2 + y^2 = 100 ∧ y = 2*x + d ∧ x^2 + y^2 = 100 * d :=
sorry

end smallest_positive_integer_d_l778_77871


namespace johns_total_time_l778_77803

/-- The total time John spent on his book and exploring is 5 years. -/
theorem johns_total_time (exploring_time note_writing_time book_writing_time : ℝ) :
  exploring_time = 3 →
  note_writing_time = exploring_time / 2 →
  book_writing_time = 0.5 →
  exploring_time + note_writing_time + book_writing_time = 5 :=
by sorry

end johns_total_time_l778_77803


namespace min_bottles_to_fill_l778_77808

theorem min_bottles_to_fill (small_capacity large_capacity : ℕ) 
  (h1 : small_capacity = 40)
  (h2 : large_capacity = 360) : 
  Nat.ceil (large_capacity / small_capacity) = 9 := by
  sorry

#check min_bottles_to_fill

end min_bottles_to_fill_l778_77808


namespace sum_of_squares_of_roots_l778_77857

theorem sum_of_squares_of_roots (a b c : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = -12) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 73/4 := by
sorry

end sum_of_squares_of_roots_l778_77857


namespace min_value_theorem_l778_77834

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x + 2)^2 / (y - 2) + (y + 2)^2 / (x - 2) ≥ 50 ∧
  ((x + 2)^2 / (y - 2) + (y + 2)^2 / (x - 2) = 50 ↔ x = 3 ∧ y = 3) :=
by sorry

end min_value_theorem_l778_77834


namespace rational_function_value_at_one_l778_77832

/-- A structure representing a rational function with specific properties. -/
structure RationalFunction where
  r : ℝ → ℝ  -- Numerator polynomial
  s : ℝ → ℝ  -- Denominator polynomial
  is_quadratic_r : ∃ a b c : ℝ, ∀ x, r x = a * x^2 + b * x + c
  is_quadratic_s : ∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c
  hole_at_4 : r 4 = 0 ∧ s 4 = 0
  zero_at_0 : r 0 = 0
  horizontal_asymptote : ∀ ε > 0, ∃ M, ∀ x > M, |r x / s x + 2| < ε
  vertical_asymptote : s 3 = 0 ∧ r 3 ≠ 0

/-- Theorem stating that for a rational function with the given properties, r(1)/s(1) = 1 -/
theorem rational_function_value_at_one (f : RationalFunction) : f.r 1 / f.s 1 = 1 := by
  sorry

end rational_function_value_at_one_l778_77832


namespace complex_in_second_quadrant_l778_77810

/-- The complex number z = (1+2i)/(1-2i) is in the second quadrant -/
theorem complex_in_second_quadrant : 
  let z : ℂ := (1 + 2*I) / (1 - 2*I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end complex_in_second_quadrant_l778_77810


namespace sum_remainder_is_two_l778_77884

theorem sum_remainder_is_two (n : ℤ) : (8 - n + (n + 4)) % 5 = 2 := by
  sorry

end sum_remainder_is_two_l778_77884


namespace num_event_committees_l778_77839

/-- The number of teams in the tournament -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of members in the event committee -/
def committee_size : ℕ := 16

/-- Theorem stating the number of possible event committees -/
theorem num_event_committees : 
  (num_teams : ℕ) * (Nat.choose team_size host_selection) * 
  (Nat.choose team_size non_host_selection)^(num_teams - 1) = 3443073600 := by
  sorry

end num_event_committees_l778_77839


namespace quadratic_root_condition_l778_77828

theorem quadratic_root_condition (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
   p * x₁^2 + (p - 1) * x₁ + p + 1 = 0 ∧
   p * x₂^2 + (p - 1) * x₂ + p + 1 = 0 ∧
   x₂ > 2 * x₁) →
  (0 < p ∧ p < 1/7) :=
by sorry

end quadratic_root_condition_l778_77828


namespace power_product_l778_77823

theorem power_product (x y : ℝ) (h1 : (10 : ℝ) ^ x = 3) (h2 : (10 : ℝ) ^ y = 4) : 
  (10 : ℝ) ^ (x * y) = 12 := by
  sorry

end power_product_l778_77823


namespace P_equals_59_when_V_is_9_l778_77853

-- Define the relationship between P, h, and V
def P (h V : ℝ) : ℝ := 3 * h * V + 5

-- State the theorem
theorem P_equals_59_when_V_is_9 : 
  ∃ (h : ℝ), (P h 6 = 41) → (P h 9 = 59) := by
  sorry

end P_equals_59_when_V_is_9_l778_77853


namespace factory_output_percentage_l778_77814

theorem factory_output_percentage (T X Y : ℝ) : 
  T > 0 →  -- Total output is positive
  X > 0 →  -- Machine-x output is positive
  Y > 0 →  -- Machine-y output is positive
  X + Y = T →  -- Total output is sum of both machines
  0.006 * T = 0.009 * X + 0.004 * Y →  -- Defective units equation
  X = 0.4 * T  -- Machine-x produces 40% of total output
  := by sorry

end factory_output_percentage_l778_77814


namespace perpendicular_lines_n_value_l778_77899

/-- Two perpendicular lines with a given foot of perpendicular -/
structure PerpendicularLines where
  m : ℝ
  n : ℝ
  p : ℝ
  line1_eq : ∀ x y, m * x + 4 * y - 2 = 0
  line2_eq : ∀ x y, 2 * x - 5 * y + n = 0
  perpendicular : m * 2 + 4 * 5 = 0
  foot_on_line1 : m * 1 + 4 * p - 2 = 0
  foot_on_line2 : 2 * 1 - 5 * p + n = 0

/-- The value of n in the given perpendicular lines setup is -12 -/
theorem perpendicular_lines_n_value (pl : PerpendicularLines) : pl.n = -12 := by
  sorry

end perpendicular_lines_n_value_l778_77899


namespace fraction_to_decimal_l778_77866

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l778_77866


namespace mode_and_median_of_data_l778_77835

def data : List ℕ := [6, 8, 3, 6, 4, 6, 5]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem mode_and_median_of_data :
  mode data = 6 ∧ median data = 6 := by sorry

end mode_and_median_of_data_l778_77835


namespace fraction_of_students_with_partners_l778_77889

theorem fraction_of_students_with_partners :
  ∀ (a b : ℕ), 
    a > 0 → b > 0 →
    (b : ℚ) / 4 = (3 : ℚ) * a / 7 →
    ((b : ℚ) / 4 + (3 : ℚ) * a / 7) / ((b : ℚ) + a) = 6 / 19 :=
by sorry

end fraction_of_students_with_partners_l778_77889


namespace larger_number_is_588_l778_77802

/-- Given two positive integers with HCF 42 and LCM factors 12 and 14, the larger number is 588 -/
theorem larger_number_is_588 (a b : ℕ+) (hcf : Nat.gcd a b = 42) 
  (lcm_factors : ∃ (x y : ℕ+), x = 12 ∧ y = 14 ∧ Nat.lcm a b = 42 * x * y) :
  max a b = 588 := by
  sorry

end larger_number_is_588_l778_77802


namespace toy_production_proof_l778_77818

/-- A factory produces toys. -/
structure ToyFactory where
  weekly_production : ℕ
  working_days : ℕ
  uniform_production : Bool

/-- Calculate the daily toy production for a given factory. -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.working_days

theorem toy_production_proof (factory : ToyFactory) 
  (h1 : factory.weekly_production = 5505)
  (h2 : factory.working_days = 5)
  (h3 : factory.uniform_production = true) :
  daily_production factory = 1101 := by
  sorry

#eval daily_production { weekly_production := 5505, working_days := 5, uniform_production := true }

end toy_production_proof_l778_77818


namespace tangent_line_parallel_to_given_line_l778_77885

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 10

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_parallel_to_given_line :
  ∀ x₀ y₀ : ℝ,
  f x₀ = y₀ →
  f' x₀ = 4 →
  ((x₀ = 1 ∧ y₀ = -8) ∨ (x₀ = -1 ∧ y₀ = -12)) ∧
  ((y₀ = 4 * x₀ - 12) ∨ (y₀ = 4 * x₀ - 8)) :=
by sorry

end tangent_line_parallel_to_given_line_l778_77885


namespace leapYearsIn200Years_l778_77805

/-- Definition of a leap year in the modified calendar system -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 == 0 && year % 128 ≠ 0

/-- Count of leap years in a given period -/
def countLeapYears (period : ℕ) : ℕ :=
  (List.range period).filter isLeapYear |>.length

/-- Theorem: There are 49 leap years in a 200-year period -/
theorem leapYearsIn200Years : countLeapYears 200 = 49 := by
  sorry

end leapYearsIn200Years_l778_77805


namespace comic_book_ratio_l778_77827

/-- Represents the number of comic books Sandy has at different stages -/
structure ComicBooks where
  initial : ℕ
  sold : ℕ
  bought : ℕ
  final : ℕ

/-- The ratio of sold books to initial books is 1:2 -/
theorem comic_book_ratio (s : ComicBooks) 
  (h1 : s.initial = 14)
  (h2 : s.bought = 6)
  (h3 : s.final = 13)
  (h4 : s.initial - s.sold + s.bought = s.final) :
  s.sold / s.initial = 1 / 2 := by
sorry

end comic_book_ratio_l778_77827


namespace cos_product_range_in_triangle_l778_77894

theorem cos_product_range_in_triangle (A B C : ℝ) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 
  A + B + C = π ∧ 
  B = π / 3 → 
  -1/2 ≤ Real.cos A * Real.cos C ∧ Real.cos A * Real.cos C ≤ 1/4 :=
sorry

end cos_product_range_in_triangle_l778_77894


namespace fraction_equality_l778_77829

theorem fraction_equality : (2 - (1/2) * (1 - 1/4)) / (2 - (1 - 1/3)) = 39/32 := by
  sorry

end fraction_equality_l778_77829


namespace xy_equals_three_l778_77878

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end xy_equals_three_l778_77878


namespace proposition_c_is_true_l778_77807

theorem proposition_c_is_true : ∀ x y : ℝ, x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1 := by
  sorry

end proposition_c_is_true_l778_77807


namespace purple_ring_weight_l778_77825

/-- The weight of the purple ring in Karin's science class experiment -/
theorem purple_ring_weight (orange_weight white_weight total_weight : ℚ)
  (h_orange : orange_weight = 8/100)
  (h_white : white_weight = 42/100)
  (h_total : total_weight = 83/100) :
  total_weight - orange_weight - white_weight = 33/100 := by
  sorry

end purple_ring_weight_l778_77825


namespace lcm_1008_672_l778_77858

theorem lcm_1008_672 : Nat.lcm 1008 672 = 2016 := by
  sorry

end lcm_1008_672_l778_77858


namespace always_true_inequality_l778_77815

theorem always_true_inequality (a b x y : ℝ) (h1 : x < a) (h2 : y < b) : x * y < a * b := by
  sorry

end always_true_inequality_l778_77815


namespace nearest_year_with_more_zeros_than_ones_l778_77821

/-- Given a natural number, returns the number of ones in its binary representation. -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Given a natural number, returns the number of zeros in its binary representation. -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Theorem: 2048 is the smallest integer greater than 2017 such that in its binary representation, 
    the number of ones is less than or equal to the number of zeros. -/
theorem nearest_year_with_more_zeros_than_ones : 
  ∀ k : ℕ, k > 2017 → k < 2048 → countOnes k > countZeros k :=
by sorry

end nearest_year_with_more_zeros_than_ones_l778_77821


namespace purely_imaginary_complex_number_l778_77863

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 4*m + 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = 5 := by
  sorry

end purely_imaginary_complex_number_l778_77863


namespace sum_mod_ten_zero_l778_77873

theorem sum_mod_ten_zero : (5000 + 5001 + 5002 + 5003 + 5004) % 10 = 0 := by
  sorry

end sum_mod_ten_zero_l778_77873


namespace oldest_child_age_l778_77816

-- Define the problem parameters
def num_children : ℕ := 4
def average_age : ℝ := 8
def younger_ages : List ℝ := [5, 7, 9]

-- State the theorem
theorem oldest_child_age :
  ∀ (oldest_age : ℝ),
  (List.sum younger_ages + oldest_age) / num_children = average_age →
  oldest_age = 11 := by
  sorry

end oldest_child_age_l778_77816


namespace cylinder_radius_l778_77840

structure Cone where
  diameter : ℚ
  altitude : ℚ

structure Cylinder where
  radius : ℚ

def inscribed_cylinder (cone : Cone) (cyl : Cylinder) : Prop :=
  cyl.radius * 2 = cyl.radius * 2 ∧  -- cylinder's diameter equals its height
  cone.diameter = 10 ∧
  cone.altitude = 12 ∧
  -- The axes of the cylinder and cone coincide (implicit in the problem setup)
  true

theorem cylinder_radius (cone : Cone) (cyl : Cylinder) :
  inscribed_cylinder cone cyl → cyl.radius = 30 / 11 := by
  sorry

end cylinder_radius_l778_77840


namespace mount_tai_temp_difference_l778_77860

/-- The temperature difference between two points is the absolute value of their difference. -/
def temperature_difference (t1 t2 : ℝ) : ℝ := |t1 - t2|

/-- The average temperature at the top of Mount Tai in January (in °C). -/
def temp_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai in January (in °C). -/
def temp_foot : ℝ := -1

/-- The temperature difference between the foot and top of Mount Tai is 8°C. -/
theorem mount_tai_temp_difference : temperature_difference temp_foot temp_top = 8 := by
  sorry

end mount_tai_temp_difference_l778_77860


namespace hyperbola_line_intersection_specific_a_value_l778_77868

/-- Hyperbola C: x²/a² - y² = 1 (a > 0) -/
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1 ∧ a > 0

/-- Line l: x + y = 1 -/
def line (x y : ℝ) : Prop := x + y = 1

/-- P is the intersection point of l and the y-axis -/
def P : ℝ × ℝ := (0, 1)

/-- A and B are distinct intersection points of C and l -/
def intersection_points (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    hyperbola a A.1 A.2 ∧ line A.1 A.2 ∧
    hyperbola a B.1 B.2 ∧ line B.1 B.2

/-- PA = (5/12)PB -/
def vector_relation (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (5/12 * (B.1 - P.1), 5/12 * (B.2 - P.2))

theorem hyperbola_line_intersection (a : ℝ) :
  intersection_points a → (0 < a ∧ a < 1) ∨ (1 < a ∧ a < Real.sqrt 2) :=
sorry

theorem specific_a_value (a : ℝ) (A B : ℝ × ℝ) :
  hyperbola a A.1 A.2 ∧ line A.1 A.2 ∧
  hyperbola a B.1 B.2 ∧ line B.1 B.2 ∧
  vector_relation A B →
  a = 17/13 :=
sorry

end hyperbola_line_intersection_specific_a_value_l778_77868


namespace spider_web_problem_l778_77897

theorem spider_web_problem (S : ℕ) : 
  (∃ (W D : ℕ), 
    S = W ∧              -- Number of spiders equals number of webs made by each spider
    S = D ∧              -- Number of spiders equals number of days taken
    7 * S = W * D) →     -- Relationship between 1 spider making 1 web in 7 days
  S = 7 := by
sorry

end spider_web_problem_l778_77897


namespace correct_solution_l778_77865

/-- The original equation -/
def original_equation (x : ℚ) : Prop :=
  (2 - 2*x) / 3 = (3*x - 3) / 7 + 3

/-- Xiao Jun's incorrect equation -/
def incorrect_equation (x m : ℚ) : Prop :=
  7*(2 - 2*x) = 3*(3*x - m) + 3

/-- Xiao Jun's solution -/
def xiao_jun_solution : ℚ := 14/23

/-- The correct value of m -/
def correct_m : ℚ := 3

theorem correct_solution :
  incorrect_equation xiao_jun_solution correct_m →
  ∃ x : ℚ, x = 2 ∧ original_equation x :=
by sorry

end correct_solution_l778_77865


namespace watch_loss_percentage_l778_77879

/-- Proves that the loss percentage is 10% given the conditions of the watch sale problem -/
theorem watch_loss_percentage (cost_price : ℝ) (additional_price : ℝ) (gain_percentage : ℝ) 
  (h1 : cost_price = 1428.57)
  (h2 : additional_price = 200)
  (h3 : gain_percentage = 4) : 
  ∃ (loss_percentage : ℝ), 
    loss_percentage = 10 ∧ 
    cost_price + additional_price = cost_price * (1 + gain_percentage / 100) ∧
    cost_price * (1 - loss_percentage / 100) + additional_price = cost_price * (1 + gain_percentage / 100) :=
by
  sorry


end watch_loss_percentage_l778_77879


namespace height_statistics_l778_77887

/-- Heights of students in Class A -/
def class_a_heights : Finset ℕ := sorry

/-- Heights of students in Class B -/
def class_b_heights : Finset ℕ := sorry

/-- The mode of a finite set of natural numbers -/
def mode (s : Finset ℕ) : ℕ := sorry

/-- The median of a finite set of natural numbers -/
def median (s : Finset ℕ) : ℕ := sorry

/-- Theorem stating the mode of Class A heights and median of Class B heights -/
theorem height_statistics :
  mode class_a_heights = 171 ∧ median class_b_heights = 170 := by sorry

end height_statistics_l778_77887


namespace volume_of_cubes_l778_77813

/-- Given two cubes where the ratio of their edges is 3:1 and the volume of the smaller cube is 8 units,
    the volume of the larger cube is 216 units. -/
theorem volume_of_cubes (a b : ℝ) (h1 : a / b = 3) (h2 : b^3 = 8) : a^3 = 216 := by
  sorry

end volume_of_cubes_l778_77813


namespace tie_distribution_impossibility_l778_77850

theorem tie_distribution_impossibility 
  (B : Type) -- Set of boys
  (G : Type) -- Set of girls
  (knows : B → G → Prop) -- Relation representing who knows whom
  (color : B ⊕ G → Fin 99) -- Function assigning colors to people
  : ¬ (
    -- For any boy who knows at least 2015 girls
    (∀ b : B, (∃ (girls : Finset G), girls.card ≥ 2015 ∧ ∀ g ∈ girls, knows b g) →
      -- There are two girls among them with different colored ties
      ∃ g1 g2 : G, g1 ≠ g2 ∧ knows b g1 ∧ knows b g2 ∧ color (Sum.inr g1) ≠ color (Sum.inr g2)) ∧
    -- For any girl who knows at least 2015 boys
    (∀ g : G, (∃ (boys : Finset B), boys.card ≥ 2015 ∧ ∀ b ∈ boys, knows b g) →
      -- There are two boys among them with different colored ties
      ∃ b1 b2 : B, b1 ≠ b2 ∧ knows b1 g ∧ knows b2 g ∧ color (Sum.inl b1) ≠ color (Sum.inl b2))
  ) :=
by sorry

end tie_distribution_impossibility_l778_77850


namespace simplify_expression_l778_77883

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^4 + b^4 = a + b) :
  a / b + b / a - 1 / (a * b^2) = -(a + b) / (a * b^2) := by
  sorry

end simplify_expression_l778_77883


namespace marble_collection_total_l778_77872

/-- Given a collection of orange, purple, and yellow marbles, where:
    - The number of orange marbles is o
    - There are 30% more orange marbles than purple marbles
    - There are 50% more yellow marbles than orange marbles
    Prove that the total number of marbles is 3.269o -/
theorem marble_collection_total (o : ℝ) (o_positive : o > 0) : ∃ (p y : ℝ),
  p > 0 ∧ y > 0 ∧
  o = 1.3 * p ∧
  y = 1.5 * o ∧
  o + p + y = 3.269 * o :=
sorry

end marble_collection_total_l778_77872


namespace correct_height_proof_l778_77800

/-- Proves the correct height of a boy in a class given certain conditions -/
theorem correct_height_proof (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ) :
  n = 35 →
  initial_avg = 184 →
  wrong_height = 166 →
  actual_avg = 182 →
  ∃ (correct_height : ℝ), correct_height = 236 ∧
    n * actual_avg = n * initial_avg - wrong_height + correct_height :=
by sorry

end correct_height_proof_l778_77800


namespace different_color_probability_l778_77877

theorem different_color_probability : 
  let total_balls : ℕ := 5
  let white_balls : ℕ := 2
  let black_balls : ℕ := 3
  let probability_different_colors : ℚ := 12 / 25
  (white_balls + black_balls = total_balls) →
  (probability_different_colors = 
    (white_balls * black_balls + black_balls * white_balls) / (total_balls * total_balls)) :=
by sorry

end different_color_probability_l778_77877


namespace complex_magnitude_product_l778_77870

theorem complex_magnitude_product : Complex.abs ((7 - 4*Complex.I) * (5 + 3*Complex.I)) = Real.sqrt 2210 := by
  sorry

end complex_magnitude_product_l778_77870


namespace intersection_locus_l778_77886

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in the form y² = x -/
def parabola (p : Point2D) : Prop :=
  p.y^2 = p.x

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.x = l.slope * p.y + l.intercept

/-- Checks if four points are concyclic (lie on the same circle) -/
def areConcyclic (p1 p2 p3 p4 : Point2D) : Prop :=
  ∃ (center : Point2D) (radius : ℝ),
    (center.x - p1.x)^2 + (center.y - p1.y)^2 = radius^2 ∧
    (center.x - p2.x)^2 + (center.y - p2.y)^2 = radius^2 ∧
    (center.x - p3.x)^2 + (center.y - p3.y)^2 = radius^2 ∧
    (center.x - p4.x)^2 + (center.y - p4.y)^2 = radius^2

theorem intersection_locus
  (a b : ℝ)
  (ha : 0 < a)
  (hab : a < b)
  (l m : Line2D)
  (hl : l.intercept = a)
  (hm : m.intercept = b)
  (p1 p2 p3 p4 : Point2D)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4)
  (h_on_parabola : parabola p1 ∧ parabola p2 ∧ parabola p3 ∧ parabola p4)
  (h_on_lines : (pointOnLine p1 l ∨ pointOnLine p1 m) ∧
                (pointOnLine p2 l ∨ pointOnLine p2 m) ∧
                (pointOnLine p3 l ∨ pointOnLine p3 m) ∧
                (pointOnLine p4 l ∨ pointOnLine p4 m))
  (h_concyclic : areConcyclic p1 p2 p3 p4)
  (P : Point2D)
  (h_intersection : pointOnLine P l ∧ pointOnLine P m) :
  P.x = (a + b) / 2 :=
by sorry

end intersection_locus_l778_77886


namespace rectangles_in_5x5_grid_l778_77867

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : ℕ := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating the number of different rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles = 100 := by
  sorry

end rectangles_in_5x5_grid_l778_77867


namespace diana_total_earnings_l778_77817

def july_earnings : ℝ := 150

def august_earnings : ℝ := 3 * july_earnings

def september_earnings : ℝ := 2 * august_earnings

def october_earnings : ℝ := september_earnings * 1.1

def november_earnings : ℝ := october_earnings * 0.95

def total_earnings : ℝ := july_earnings + august_earnings + september_earnings + october_earnings + november_earnings

theorem diana_total_earnings : total_earnings = 3430.50 := by
  sorry

end diana_total_earnings_l778_77817


namespace diophantine_approximation_l778_77822

theorem diophantine_approximation (x : ℝ) : 
  ∀ N : ℕ, ∃ p q : ℤ, q > N ∧ |x - (p : ℝ) / (q : ℝ)| < 1 / (q : ℝ)^2 := by
  sorry

end diophantine_approximation_l778_77822


namespace rectangular_plot_breadth_l778_77841

theorem rectangular_plot_breadth (area length breadth : ℝ) : 
  area = 24 * breadth →
  length = breadth + 10 →
  area = length * breadth →
  breadth = 14 := by
sorry

end rectangular_plot_breadth_l778_77841


namespace min_good_pairs_l778_77882

/-- A circular arrangement of natural numbers from 1 to 100 -/
def CircularArrangement := Fin 100 → ℕ

/-- Predicate to check if a number at index i satisfies the neighbor condition -/
def satisfies_neighbor_condition (arr : CircularArrangement) (i : Fin 100) : Prop :=
  (arr i > arr ((i + 1) % 100) ∧ arr i > arr ((i + 99) % 100)) ∨
  (arr i < arr ((i + 1) % 100) ∧ arr i < arr ((i + 99) % 100))

/-- Predicate to check if a pair at indices i and j form a "good pair" -/
def is_good_pair (arr : CircularArrangement) (i j : Fin 100) : Prop :=
  arr i > arr j ∧ satisfies_neighbor_condition arr i ∧ satisfies_neighbor_condition arr j

/-- The main theorem stating that any valid arrangement has at least 51 good pairs -/
theorem min_good_pairs (arr : CircularArrangement) 
  (h_valid : ∀ i, satisfies_neighbor_condition arr i)
  (h_distinct : ∀ i j, i ≠ j → arr i ≠ arr j)
  (h_range : ∀ i, arr i ∈ Finset.range 101 \ {0}) :
  ∃ (pairs : Finset (Fin 100 × Fin 100)), pairs.card ≥ 51 ∧ 
    ∀ (p : Fin 100 × Fin 100), p ∈ pairs → is_good_pair arr p.1 p.2 :=
sorry

end min_good_pairs_l778_77882


namespace batting_average_is_60_l778_77896

/-- A batsman's batting statistics -/
structure BattingStats where
  total_innings : ℕ
  highest_score : ℕ
  lowest_score : ℕ
  average_excluding_extremes : ℚ

/-- The batting average for all innings -/
def batting_average (stats : BattingStats) : ℚ :=
  let total_runs := stats.average_excluding_extremes * (stats.total_innings - 2 : ℚ) + stats.highest_score + stats.lowest_score
  total_runs / stats.total_innings

/-- Theorem stating the batting average for the given conditions -/
theorem batting_average_is_60 (stats : BattingStats) 
    (h1 : stats.total_innings = 46)
    (h2 : stats.highest_score = 194)
    (h3 : stats.highest_score - stats.lowest_score = 180)
    (h4 : stats.average_excluding_extremes = 58) :
    batting_average stats = 60 := by
  sorry

end batting_average_is_60_l778_77896


namespace infinite_power_tower_eq_four_l778_77831

/-- The limit of the infinite power tower x^(x^(x^...)) -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := Real.log x / Real.log (Real.log x)

/-- Theorem stating that if the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_eq_four (x : ℝ) (h : x > 0) :
  infinitePowerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end infinite_power_tower_eq_four_l778_77831


namespace circle_intersection_range_l778_77845

-- Define the circle C
def circle_C (a x y : ℝ) : Prop := (x - a)^2 + (y - a + 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition for point M
def condition_M (a x y : ℝ) : Prop :=
  circle_C a x y ∧ (x^2 + (y - 2)^2) + (x^2 + y^2) = 10

-- Main theorem
theorem circle_intersection_range (a : ℝ) :
  (∃ x y : ℝ, condition_M a x y) → a ∈ Set.Icc 0 3 := by
  sorry

end circle_intersection_range_l778_77845


namespace inverse_A_times_B_l778_77892

open Matrix

theorem inverse_A_times_B :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 2, 5]
  A⁻¹ * B = !![1/2, -1/2; 2, 5] := by
sorry

end inverse_A_times_B_l778_77892


namespace diagonal_sum_property_l778_77880

/-- A convex regular polygon with 3k sides, where k > 4 is an integer -/
structure RegularPolygon (k : ℕ) :=
  (sides : ℕ)
  (convex : Bool)
  (regular : Bool)
  (k_gt_4 : k > 4)
  (sides_eq_3k : sides = 3 * k)

/-- A diagonal in a polygon -/
structure Diagonal (P : RegularPolygon k) :=
  (length : ℝ)

/-- Theorem: In a convex regular polygon with 3k sides (k > 4), 
    there exist diagonals whose lengths are equal to the sum of 
    the lengths of two shorter diagonals -/
theorem diagonal_sum_property (k : ℕ) (P : RegularPolygon k) :
  ∃ (d1 d2 d3 : Diagonal P), 
    d1.length = d2.length + d3.length ∧ 
    d1.length > d2.length ∧ 
    d1.length > d3.length :=
  sorry

end diagonal_sum_property_l778_77880


namespace concert_longest_song_duration_l778_77888

/-- Represents the duration of the longest song in a concert --/
def longest_song_duration (total_time intermission_time num_songs regular_song_duration : ℕ) : ℕ :=
  total_time - intermission_time - (num_songs - 1) * regular_song_duration

/-- Theorem stating the duration of the longest song in the given concert scenario --/
theorem concert_longest_song_duration :
  longest_song_duration 80 10 13 5 = 10 := by sorry

end concert_longest_song_duration_l778_77888


namespace max_basketballs_min_basketballs_for_profit_max_profit_l778_77843

/-- Represents the sports equipment store problem -/
structure StoreProblem where
  total_balls : ℕ
  max_payment : ℕ
  basketball_wholesale : ℕ
  volleyball_wholesale : ℕ
  basketball_retail : ℕ
  volleyball_retail : ℕ
  min_profit : ℕ

/-- The specific instance of the store problem -/
def store_instance : StoreProblem :=
  { total_balls := 100
  , max_payment := 11815
  , basketball_wholesale := 130
  , volleyball_wholesale := 100
  , basketball_retail := 160
  , volleyball_retail := 120
  , min_profit := 2580
  }

/-- Calculates the total cost of purchasing basketballs and volleyballs -/
def total_cost (p : StoreProblem) (basketballs : ℕ) : ℕ :=
  p.basketball_wholesale * basketballs + p.volleyball_wholesale * (p.total_balls - basketballs)

/-- Calculates the profit from selling all balls -/
def profit (p : StoreProblem) (basketballs : ℕ) : ℕ :=
  (p.basketball_retail - p.basketball_wholesale) * basketballs +
  (p.volleyball_retail - p.volleyball_wholesale) * (p.total_balls - basketballs)

/-- Theorem stating the maximum number of basketballs that can be purchased -/
theorem max_basketballs (p : StoreProblem) :
  ∃ (max_basketballs : ℕ),
    (∀ (b : ℕ), total_cost p b ≤ p.max_payment → b ≤ max_basketballs) ∧
    total_cost p max_basketballs ≤ p.max_payment ∧
    max_basketballs = 60 :=
  sorry

/-- Theorem stating the minimum number of basketballs needed for desired profit -/
theorem min_basketballs_for_profit (p : StoreProblem) :
  ∃ (min_basketballs : ℕ),
    (∀ (b : ℕ), profit p b ≥ p.min_profit → b ≥ min_basketballs) ∧
    profit p min_basketballs ≥ p.min_profit ∧
    min_basketballs = 58 :=
  sorry

/-- Theorem stating the maximum profit achievable -/
theorem max_profit (p : StoreProblem) :
  ∃ (max_profit : ℕ),
    (∀ (b : ℕ), total_cost p b ≤ p.max_payment → profit p b ≤ max_profit) ∧
    (∃ (b : ℕ), total_cost p b ≤ p.max_payment ∧ profit p b = max_profit) ∧
    max_profit = 2600 :=
  sorry

end max_basketballs_min_basketballs_for_profit_max_profit_l778_77843


namespace polynomial_value_l778_77838

theorem polynomial_value (x : ℝ) (h : x = (1 + Real.sqrt 1994) / 2) :
  (4 * x^3 - 1997 * x - 1994)^20001 = -1 := by
  sorry

end polynomial_value_l778_77838


namespace cubic_equation_roots_of_unity_l778_77836

theorem cubic_equation_roots_of_unity :
  ∃ (a b c : ℤ), (1 : ℂ)^3 + a*(1 : ℂ)^2 + b*(1 : ℂ) + c = 0 ∧
                 (-1 : ℂ)^3 + a*(-1 : ℂ)^2 + b*(-1 : ℂ) + c = 0 :=
by sorry

end cubic_equation_roots_of_unity_l778_77836


namespace infinite_points_satisfying_condition_l778_77849

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the diameter endpoints
def DiameterEndpoints (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((center.1 - radius, center.2), (center.1 + radius, center.2))

-- Define the condition for points P
def SatisfiesCondition (p : ℝ × ℝ) (endpoints : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (a, b) := endpoints
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2 = 5

-- Theorem statement
theorem infinite_points_satisfying_condition 
  (center : ℝ × ℝ) : 
  ∃ (s : Set (ℝ × ℝ)), 
    (∀ p ∈ s, p ∈ Circle center 2 ∧ 
              SatisfiesCondition p (DiameterEndpoints center 2)) ∧
    (Set.Infinite s) := by
  sorry

end infinite_points_satisfying_condition_l778_77849


namespace power_of_two_not_sum_of_consecutive_integers_l778_77833

theorem power_of_two_not_sum_of_consecutive_integers :
  ∀ n : ℕ+, (∀ r : ℕ, r > 1 → ¬∃ k : ℕ, n = (k + r) * (k + r - 1) / 2 - k * (k - 1) / 2) ↔
  ∃ l : ℕ, n = 2^l := by sorry

end power_of_two_not_sum_of_consecutive_integers_l778_77833


namespace price_reduction_theorem_l778_77846

theorem price_reduction_theorem (x : ℝ) : 
  (1 - x / 100) * 1.8 = 1.17 → x = 35 := by sorry

end price_reduction_theorem_l778_77846


namespace increasing_f_implies_a_range_l778_77847

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (3/2) 3 := by sorry

end increasing_f_implies_a_range_l778_77847


namespace sides_in_nth_figure_formula_l778_77859

/-- The number of sides in the n-th figure of a sequence starting with a hexagon
    and increasing by 5 sides for each subsequent figure. -/
def sides_in_nth_figure (n : ℕ) : ℕ := 5 * n + 1

/-- Theorem stating that the number of sides in the n-th figure is 5n + 1 -/
theorem sides_in_nth_figure_formula (n : ℕ) :
  sides_in_nth_figure n = 5 * n + 1 := by sorry

end sides_in_nth_figure_formula_l778_77859


namespace xf_is_even_l778_77811

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- Theorem statement
theorem xf_is_even (f : ℝ → ℝ) (h : OddFunction f) :
  EvenFunction (fun x ↦ x * f x) := by
  sorry

end xf_is_even_l778_77811


namespace half_floors_full_capacity_l778_77862

/-- Represents a building with floors, apartments, and occupants. -/
structure Building where
  total_floors : ℕ
  apartments_per_floor : ℕ
  people_per_apartment : ℕ
  total_people : ℕ

/-- Calculates the number of full-capacity floors in the building. -/
def full_capacity_floors (b : Building) : ℕ :=
  let people_per_full_floor := b.apartments_per_floor * b.people_per_apartment
  let total_full_floor_capacity := b.total_floors * people_per_full_floor
  (2 * b.total_people - total_full_floor_capacity) / people_per_full_floor

/-- Theorem stating that for a building with specific parameters,
    the number of full-capacity floors is half the total floors. -/
theorem half_floors_full_capacity (b : Building)
    (h1 : b.total_floors = 12)
    (h2 : b.apartments_per_floor = 10)
    (h3 : b.people_per_apartment = 4)
    (h4 : b.total_people = 360) :
    full_capacity_floors b = b.total_floors / 2 := by
  sorry

end half_floors_full_capacity_l778_77862


namespace last_digit_sum_l778_77864

theorem last_digit_sum (x y : ℕ) : 
  (135^x + 31^y + 56^(x+y)) % 10 = 2 := by sorry

end last_digit_sum_l778_77864


namespace mutually_exclusive_not_contradictory_l778_77851

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first : BallColor)
  (second : BallColor)

/-- The bag containing 2 black balls and 2 white balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.Black} + 2 • {BallColor.White}

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

theorem mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∧ exactlyTwoWhite outcome)) ∧
  (∃ outcome : DrawOutcome, exactlyOneBlack outcome ∨ exactlyTwoWhite outcome) ∧
  (∃ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∨ exactlyTwoWhite outcome)) :=
sorry

end mutually_exclusive_not_contradictory_l778_77851


namespace lamps_remain_lighted_l778_77824

def toggle_lamps (n : ℕ) : ℕ :=
  n - (n / 2 + n / 3 + n / 5 - n / 6 - n / 10 - n / 15 + n / 30)

theorem lamps_remain_lighted :
  toggle_lamps 2015 = 1006 := by
  sorry

end lamps_remain_lighted_l778_77824


namespace find_S_l778_77876

theorem find_S : ∃ S : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ S = 120 := by
  sorry

end find_S_l778_77876


namespace inequality_solution_set_l778_77874

theorem inequality_solution_set (p : ℝ) :
  (p ≥ 0 ∧ ∀ q > 0, (5 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 4 * p * q)) / (p + 2 * q) > 3 * p^2 * q) ↔
  0 ≤ p ∧ p < 4 :=
by sorry

end inequality_solution_set_l778_77874


namespace tomato_cucumber_ratio_l778_77881

/-- Given the initial quantities of tomatoes and cucumbers, and the amounts picked,
    prove that the ratio of remaining tomatoes to remaining cucumbers is 7:68. -/
theorem tomato_cucumber_ratio
  (initial_tomatoes : ℕ)
  (initial_cucumbers : ℕ)
  (tomatoes_picked_yesterday : ℕ)
  (tomatoes_picked_today : ℕ)
  (cucumbers_picked_total : ℕ)
  (h1 : initial_tomatoes = 171)
  (h2 : initial_cucumbers = 225)
  (h3 : tomatoes_picked_yesterday = 134)
  (h4 : tomatoes_picked_today = 30)
  (h5 : cucumbers_picked_total = 157)
  : (initial_tomatoes - (tomatoes_picked_yesterday + tomatoes_picked_today)) /
    (initial_cucumbers - cucumbers_picked_total) = 7 / 68 :=
by sorry

end tomato_cucumber_ratio_l778_77881


namespace blocks_used_proof_l778_77830

/-- The number of blocks Randy used to build a tower -/
def tower_blocks : ℕ := 27

/-- The number of blocks Randy used to build a house -/
def house_blocks : ℕ := 53

/-- The total number of blocks Randy used for both the tower and the house -/
def total_blocks : ℕ := tower_blocks + house_blocks

theorem blocks_used_proof : total_blocks = 80 := by
  sorry

end blocks_used_proof_l778_77830


namespace fourth_derivative_y_l778_77861

noncomputable def y (x : ℝ) : ℝ := (5 * x - 8) * (2 ^ (-x))

theorem fourth_derivative_y (x : ℝ) :
  (deriv^[4] y) x = 2^(-x) * (Real.log 2)^4 * (5*x - 9) := by sorry

end fourth_derivative_y_l778_77861


namespace quadratic_always_positive_l778_77812

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > (1/2 : ℝ) := by sorry

end quadratic_always_positive_l778_77812


namespace cubic_function_properties_l778_77820

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define monotonically increasing function
def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement
theorem cubic_function_properties :
  isPowerFunction f ∧ isMonotonicallyIncreasing f :=
sorry

end cubic_function_properties_l778_77820


namespace train_passing_bridge_l778_77852

/-- Time for a train to pass a bridge -/
theorem train_passing_bridge (train_length : Real) (train_speed_kmph : Real) (bridge_length : Real) :
  train_length = 360 ∧ 
  train_speed_kmph = 45 ∧ 
  bridge_length = 140 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 40 := by
  sorry

end train_passing_bridge_l778_77852


namespace dads_final_strawberry_weight_l778_77898

/-- Given the initial total weight of strawberries collected by Marco and his dad,
    the additional weight Marco's dad found, and Marco's final weight of strawberries,
    prove that Marco's dad's final weight of strawberries is 46 pounds. -/
theorem dads_final_strawberry_weight
  (initial_total : ℕ)
  (dads_additional : ℕ)
  (marcos_final : ℕ)
  (h1 : initial_total = 22)
  (h2 : dads_additional = 30)
  (h3 : marcos_final = 36) :
  initial_total - (marcos_final - (initial_total - marcos_final)) + dads_additional = 46 :=
by sorry

end dads_final_strawberry_weight_l778_77898


namespace cubic_sum_minus_product_l778_77854

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 15) 
  (h2 : x*y + y*z + z*x = 34) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 1845 := by
  sorry

end cubic_sum_minus_product_l778_77854


namespace log3_one_over_81_l778_77806

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_one_over_81 : log3 (1/81) = -4 := by
  sorry

end log3_one_over_81_l778_77806


namespace cindys_cycling_speed_l778_77826

/-- Cindy's cycling problem -/
theorem cindys_cycling_speed :
  -- Cindy leaves school at the same time every day
  ∀ (leave_time : ℝ),
  -- Define the distance from school to home
  ∀ (distance : ℝ),
  -- If she cycles at 20 km/h, she arrives home at 4:30 PM
  (distance / 20 = 4.5 - leave_time) →
  -- If she cycles at 10 km/h, she arrives home at 5:15 PM
  (distance / 10 = 5.25 - leave_time) →
  -- Then the speed at which she must cycle to arrive home at 5:00 PM is 12 km/h
  (distance / 12 = 5 - leave_time) :=
by sorry

end cindys_cycling_speed_l778_77826


namespace race_length_proof_l778_77869

/-- The race length in meters -/
def race_length : ℕ := 210

/-- Runner A's constant speed in m/s -/
def runner_a_speed : ℕ := 10

/-- Runner B's initial speed in m/s -/
def runner_b_initial_speed : ℕ := 1

/-- Runner B's speed increase per second in m/s -/
def runner_b_speed_increase : ℕ := 1

/-- Time difference between runners at finish in seconds -/
def finish_time_difference : ℕ := 1

/-- Function to calculate the distance covered by Runner B in t seconds -/
def runner_b_distance (t : ℕ) : ℕ := t * (t + 1) / 2

theorem race_length_proof :
  ∃ (t : ℕ), 
    (t * runner_a_speed = race_length) ∧ 
    (runner_b_distance (t - 1) = race_length) ∧ 
    (t > finish_time_difference) :=
by sorry

end race_length_proof_l778_77869


namespace g_property_g_2022_l778_77837

/-- A function g that satisfies the given property for all real x and y -/
def g : ℝ → ℝ := fun x ↦ 2021 * x

/-- The theorem stating that g satisfies the required property -/
theorem g_property : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y) := by sorry

/-- The main theorem proving that g(2022) equals 4086462 -/
theorem g_2022 : g 2022 = 4086462 := by sorry

end g_property_g_2022_l778_77837


namespace no_inscribable_2010_gon_l778_77819

theorem no_inscribable_2010_gon : ¬ ∃ (sides : Fin 2010 → ℕ), 
  (∀ i : Fin 2010, 1 ≤ sides i ∧ sides i ≤ 2010) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 2010 → ∃ i : Fin 2010, sides i = n) ∧
  (∃ r : ℝ, r > 0 ∧ ∀ i : Fin 2010, 
    ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = (sides i)^2 ∧ a * b = r * (sides i)) :=
by sorry

end no_inscribable_2010_gon_l778_77819


namespace adams_change_l778_77875

/-- Given that Adam has $5 and an airplane costs $4.28, prove that the change Adam will receive is $0.72. -/
theorem adams_change (adam_money : ℝ) (airplane_cost : ℝ) (change : ℝ) 
  (h1 : adam_money = 5)
  (h2 : airplane_cost = 4.28)
  (h3 : change = adam_money - airplane_cost) :
  change = 0.72 := by
sorry

end adams_change_l778_77875


namespace geometric_sequence_problem_l778_77856

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_6 = 6 and a_9 = 9, prove that a_3 = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_6 : a 6 = 6) 
    (h_9 : a 9 = 9) : 
  a 3 = 4 := by
  sorry


end geometric_sequence_problem_l778_77856


namespace x_minus_y_value_l778_77809

theorem x_minus_y_value (x y : ℝ) 
  (eq1 : 3015 * x + 3020 * y = 3025)
  (eq2 : 3018 * x + 3024 * y = 3030) : 
  x - y = 11.1167 := by
sorry

end x_minus_y_value_l778_77809


namespace sum_of_counts_l778_77801

/-- A function that returns the count of four-digit even numbers -/
def count_four_digit_even : ℕ :=
  sorry

/-- A function that returns the count of four-digit numbers divisible by both 5 and 3 -/
def count_four_digit_div_by_5_and_3 : ℕ :=
  sorry

/-- Theorem stating that the sum of four-digit even numbers and four-digit numbers
    divisible by both 5 and 3 is equal to 5100 -/
theorem sum_of_counts : count_four_digit_even + count_four_digit_div_by_5_and_3 = 5100 :=
  sorry

end sum_of_counts_l778_77801


namespace solution_check_l778_77842

theorem solution_check (x : ℝ) : 
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 := by
  sorry

#check solution_check

end solution_check_l778_77842
