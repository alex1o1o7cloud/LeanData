import Mathlib

namespace NUMINAMATH_CALUDE_total_loaves_served_l1999_199970

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real)
  (h1 : wheat_bread = 0.2)
  (h2 : white_bread = 0.4) :
  wheat_bread + white_bread = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l1999_199970


namespace NUMINAMATH_CALUDE_sum_of_even_is_even_l1999_199938

theorem sum_of_even_is_even (a b : ℤ) (ha : Even a) (hb : Even b) : Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_is_even_l1999_199938


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1999_199917

/-- The area of an isosceles triangle with two sides of length 5 and base of length 6 is 12 -/
theorem isosceles_triangle_area : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 6 →
  (∃ (h : ℝ), h^2 = a^2 - (c/2)^2) →
  (1/2) * c * (a^2 - (c/2)^2).sqrt = 12 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1999_199917


namespace NUMINAMATH_CALUDE_circle_square_intersection_l1999_199906

theorem circle_square_intersection (r : ℝ) (s : ℝ) (x : ℝ) :
  r = 2 →
  s = 2 →
  (π * r^2 - (s^2 - (π * r^2 - 2 * r * x + x^2))) = 2 →
  x = π / 3 + Real.sqrt 3 / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_square_intersection_l1999_199906


namespace NUMINAMATH_CALUDE_isosceles_triangle_circle_centers_distance_l1999_199959

/-- For an isosceles triangle with circumradius R and inradius r, 
    the distance d between the centers of the circumscribed and inscribed circles 
    is given by d = √(R(R-2r)). -/
theorem isosceles_triangle_circle_centers_distance 
  (R r d : ℝ) 
  (h_R_pos : R > 0) 
  (h_r_pos : r > 0) 
  (h_isosceles : IsIsosceles) : 
  d = Real.sqrt (R * (R - 2 * r)) := by
  sorry

/-- Represents an isosceles triangle -/
structure IsIsosceles : Prop where
  -- Add necessary fields to represent an isosceles triangle
  -- This is left abstract as the problem doesn't provide specific details

#check isosceles_triangle_circle_centers_distance

end NUMINAMATH_CALUDE_isosceles_triangle_circle_centers_distance_l1999_199959


namespace NUMINAMATH_CALUDE_meeting_point_difference_l1999_199908

/-- The distance between points R and S in miles -/
def total_distance : ℕ := 80

/-- The constant speed of the man starting from R in miles per hour -/
def speed_R : ℕ := 5

/-- The initial speed of the man starting from S in miles per hour -/
def initial_speed_S : ℕ := 4

/-- The hourly increase in speed for the man starting from S in miles per hour -/
def speed_increase_S : ℕ := 1

/-- The number of hours it takes for the men to meet -/
def meeting_time : ℕ := 8

/-- The distance traveled by the man starting from R -/
def distance_R : ℕ := speed_R * meeting_time

/-- The distance traveled by the man starting from S -/
def distance_S : ℕ := initial_speed_S * meeting_time + (meeting_time - 1) * meeting_time / 2

/-- The difference in distances traveled by the two men -/
def x : ℤ := distance_S - distance_R

theorem meeting_point_difference : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_difference_l1999_199908


namespace NUMINAMATH_CALUDE_class_size_l1999_199934

/-- The number of students in Yuna's class -/
def total_students : ℕ := 33

/-- The number of students who like Korean -/
def korean_students : ℕ := 28

/-- The number of students who like math -/
def math_students : ℕ := 27

/-- The number of students who like both Korean and math -/
def both_subjects : ℕ := 22

/-- There is no student who does not like both Korean and math -/
axiom no_neither : total_students = korean_students + math_students - both_subjects

theorem class_size : total_students = 33 :=
sorry

end NUMINAMATH_CALUDE_class_size_l1999_199934


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_simplify_third_expression_l1999_199903

-- First expression
theorem simplify_first_expression (a b : ℝ) (h : (a - b)^2 + a*b ≠ 0) :
  (a^3 + b^3) / ((a - b)^2 + a*b) = a + b := by sorry

-- Second expression
theorem simplify_second_expression (x a : ℝ) (h : x^2 - 4*a^2 ≠ 0) :
  (x^2 - 4*a*x + 4*a^2) / (x^2 - 4*a^2) = (x - 2*a) / (x + 2*a) := by sorry

-- Third expression
theorem simplify_third_expression (x y : ℝ) (h : x*y - 2*x ≠ 0) :
  (x*y - 2*x - 3*y + 6) / (x*y - 2*x) = (x - 3) / x := by sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_simplify_third_expression_l1999_199903


namespace NUMINAMATH_CALUDE_chocolate_purchase_shortage_l1999_199952

theorem chocolate_purchase_shortage (chocolate_cost : ℕ) (initial_money : ℕ) (borrowed_money : ℕ) : 
  chocolate_cost = 500 ∧ initial_money = 400 ∧ borrowed_money = 59 →
  chocolate_cost - (initial_money + borrowed_money) = 41 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_purchase_shortage_l1999_199952


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1999_199982

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_lengths (x : ℕ) :
  (x > 0) →
  (is_valid_triangle (3 * x) 10 (x^2)) →
  (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1999_199982


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds_l1999_199947

theorem sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds :
  8 < Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) ∧
  Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) < 9 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds_l1999_199947


namespace NUMINAMATH_CALUDE_modular_congruence_existence_l1999_199999

theorem modular_congruence_existence (a c : ℕ+) (b : ℤ) :
  ∃ x : ℕ+, (c : ℤ) ∣ ((a : ℤ)^(x : ℕ) + x - b) := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_existence_l1999_199999


namespace NUMINAMATH_CALUDE_matrix_product_l1999_199926

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 2; 3, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![4, 3; 2, 1]

theorem matrix_product :
  A * B = !![8, 5; 20, 13] := by sorry

end NUMINAMATH_CALUDE_matrix_product_l1999_199926


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1999_199909

theorem smallest_solution_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 2 * x + 1 ∧ 
  ∀ (y : ℝ), y * |y| = 2 * y + 1 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1999_199909


namespace NUMINAMATH_CALUDE_expression_evaluation_l1999_199905

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(3*x)) / (y^(2*y) * x^(3*x)) = (x/y)^(2*y - 3*x) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1999_199905


namespace NUMINAMATH_CALUDE_max_guaranteed_profit_l1999_199963

/-- Represents the number of balls -/
def n : ℕ := 10

/-- Represents the cost of a test and the price of a non-radioactive ball -/
def cost : ℕ := 1

/-- Represents the triangular number function -/
def H (k : ℕ) : ℕ := k * (k + 1) / 2

/-- Theorem stating the maximum guaranteed profit for n balls -/
theorem max_guaranteed_profit :
  ∃ k : ℕ, H k < n ∧ n ≤ H (k + 1) ∧ n - (k + 1) = 5 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_profit_l1999_199963


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_diameter_squared_l1999_199972

/-- An equiangular hexagon with specified side lengths -/
structure EquiangularHexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  equiangular : True  -- We're not proving this property, just stating it

/-- The diameter of the largest inscribed circle in an equiangular hexagon -/
def largest_inscribed_circle_diameter (h : EquiangularHexagon) : ℝ :=
  sorry  -- Definition not provided, as it's part of what needs to be proved

/-- Theorem: The square of the diameter of the largest inscribed circle in the given hexagon is 147 -/
theorem largest_inscribed_circle_diameter_squared (h : EquiangularHexagon)
  (h_AB : h.AB = 6)
  (h_BC : h.BC = 8)
  (h_CD : h.CD = 10)
  (h_DE : h.DE = 12) :
  (largest_inscribed_circle_diameter h)^2 = 147 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_diameter_squared_l1999_199972


namespace NUMINAMATH_CALUDE_same_first_last_digit_exists_l1999_199942

-- Define a function to get the first digit of a natural number
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem same_first_last_digit_exists (n : ℕ) (h : n > 0 ∧ n % 10 ≠ 0) :
  ∃ k : ℕ, k > 0 ∧ firstDigit (n^k) = lastDigit (n^k) :=
sorry

end NUMINAMATH_CALUDE_same_first_last_digit_exists_l1999_199942


namespace NUMINAMATH_CALUDE_agrey_caught_more_l1999_199943

def fishing_problem (leo_fish agrey_fish total_fish : ℕ) : Prop :=
  leo_fish + agrey_fish = total_fish ∧ agrey_fish > leo_fish

theorem agrey_caught_more (leo_fish total_fish : ℕ) 
  (h : fishing_problem leo_fish (total_fish - leo_fish) total_fish) 
  (h_leo : leo_fish = 40) 
  (h_total : total_fish = 100) : 
  (total_fish - leo_fish) - leo_fish = 20 := by
  sorry

end NUMINAMATH_CALUDE_agrey_caught_more_l1999_199943


namespace NUMINAMATH_CALUDE_y_worked_days_proof_l1999_199904

/-- The number of days x needs to finish the entire work -/
def x_total_days : ℝ := 24

/-- The number of days y needs to finish the entire work -/
def y_total_days : ℝ := 16

/-- The number of days x needs to finish the remaining work after y leaves -/
def x_remaining_days : ℝ := 9

/-- The number of days y worked before leaving the job -/
def y_worked_days : ℝ := 10

theorem y_worked_days_proof :
  y_worked_days * (1 / y_total_days) + x_remaining_days * (1 / x_total_days) = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_worked_days_proof_l1999_199904


namespace NUMINAMATH_CALUDE_count_divisible_by_3_or_5_is_28_l1999_199922

/-- The count of numbers from 1 to 60 that are divisible by either 3 or 5 or both -/
def count_divisible_by_3_or_5 : ℕ :=
  let n := 60
  let divisible_by_3 := n / 3
  let divisible_by_5 := n / 5
  let divisible_by_15 := n / 15
  divisible_by_3 + divisible_by_5 - divisible_by_15

theorem count_divisible_by_3_or_5_is_28 : count_divisible_by_3_or_5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_3_or_5_is_28_l1999_199922


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l1999_199911

theorem binary_arithmetic_equality : 
  (0b10110 : Nat) + 0b1101 - 0b11100 + 0b11101 + 0b101 = 0b101101 := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l1999_199911


namespace NUMINAMATH_CALUDE_total_amount_pens_pencils_l1999_199907

/-- The total amount spent on pens and pencils -/
def total_amount (num_pens : ℕ) (num_pencils : ℕ) (price_pen : ℚ) (price_pencil : ℚ) : ℚ :=
  num_pens * price_pen + num_pencils * price_pencil

/-- Theorem stating the total amount spent on pens and pencils -/
theorem total_amount_pens_pencils :
  total_amount 30 75 12 2 = 510 := by
  sorry

#eval total_amount 30 75 12 2

end NUMINAMATH_CALUDE_total_amount_pens_pencils_l1999_199907


namespace NUMINAMATH_CALUDE_watch_time_calculation_l1999_199955

-- Define constants based on the problem conditions
def regular_season_episodes : ℕ := 22
def third_season_episodes : ℕ := 24
def last_season_extra_episodes : ℕ := 4
def previous_seasons : ℕ := 9
def early_episode_length : ℚ := 1/2
def later_episode_length : ℚ := 3/4
def bonus_episodes : ℕ := 5
def bonus_episode_length : ℚ := 1
def crossover_episode_length : ℚ := 3/2
def marathon_length : ℚ := 5
def daily_watch_time : ℚ := 2

-- Theorem to prove
theorem watch_time_calculation :
  let total_episodes := 
    3 * regular_season_episodes + 2 + -- First three seasons
    6 * regular_season_episodes + -- Seasons 4-9
    (regular_season_episodes + last_season_extra_episodes) -- Last season
  let total_hours := 
    (3 * regular_season_episodes + 2) * early_episode_length + -- First three seasons
    (6 * regular_season_episodes) * later_episode_length + -- Seasons 4-9
    (regular_season_episodes + last_season_extra_episodes) * later_episode_length + -- Last season
    bonus_episodes * bonus_episode_length + -- Bonus episodes
    crossover_episode_length -- Crossover episode
  let remaining_hours := total_hours - marathon_length
  let days_to_finish := remaining_hours / daily_watch_time
  days_to_finish = 77 := by sorry

end NUMINAMATH_CALUDE_watch_time_calculation_l1999_199955


namespace NUMINAMATH_CALUDE_m_range_l1999_199932

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being decreasing on [-1,1]
def is_decreasing_on (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f x > f y

-- State the theorem
theorem m_range (f : ℝ → ℝ) (m : ℝ) 
  (h1 : is_decreasing_on f) 
  (h2 : f (m - 1) > f (2*m - 1)) : 
  0 < m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_m_range_l1999_199932


namespace NUMINAMATH_CALUDE_reyansh_farm_cows_l1999_199910

/-- Represents the number of cows on Mr. Reyansh's farm -/
def num_cows : ℕ := sorry

/-- Represents the daily water consumption of one cow in liters -/
def cow_water_daily : ℕ := 80

/-- Represents the number of sheep on Mr. Reyansh's farm -/
def num_sheep : ℕ := 10 * num_cows

/-- Represents the daily water consumption of one sheep in liters -/
def sheep_water_daily : ℕ := cow_water_daily / 4

/-- Represents the total water consumption for all animals in a week in liters -/
def total_water_weekly : ℕ := 78400

/-- Theorem stating that the number of cows on Mr. Reyansh's farm is 40 -/
theorem reyansh_farm_cows :
  num_cows = 40 :=
by sorry

end NUMINAMATH_CALUDE_reyansh_farm_cows_l1999_199910


namespace NUMINAMATH_CALUDE_set_intersection_proof_l1999_199978

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

theorem set_intersection_proof : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_proof_l1999_199978


namespace NUMINAMATH_CALUDE_count_eight_digit_integers_l1999_199920

/-- The number of different 8-digit positive integers -/
def eight_digit_integers : ℕ :=
  9 * (10 ^ 7)

/-- Theorem: The number of different 8-digit positive integers is 90,000,000 -/
theorem count_eight_digit_integers :
  eight_digit_integers = 90000000 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_integers_l1999_199920


namespace NUMINAMATH_CALUDE_compare_fractions_l1999_199968

theorem compare_fractions : -3/4 > -|-(4/5)| := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l1999_199968


namespace NUMINAMATH_CALUDE_fraction_equality_l1999_199928

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 4 * y) = 3) : 
  (2 * x + 4 * y) / (4 * x - 2 * y) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1999_199928


namespace NUMINAMATH_CALUDE_toy_spending_ratio_l1999_199986

def Trevor_spending : ℕ := 80
def total_spending : ℕ := 680
def years : ℕ := 4

def spending_ratio (Reed Quinn : ℕ) : Prop :=
  Reed = 2 * Quinn

theorem toy_spending_ratio :
  ∀ Reed Quinn : ℕ,
  (Trevor_spending = Reed + 20) →
  (∃ k : ℕ, Reed = k * Quinn) →
  (years * (Trevor_spending + Reed + Quinn) = total_spending) →
  spending_ratio Reed Quinn :=
by
  sorry

#check toy_spending_ratio

end NUMINAMATH_CALUDE_toy_spending_ratio_l1999_199986


namespace NUMINAMATH_CALUDE_three_distinct_roots_l1999_199901

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem three_distinct_roots 
  (a b c x₁ x₂ : ℝ) 
  (h1 : ∃ x₁ x₂, x₁ ≠ x₂ ∧ (3*x₁^2 + 2*a*x₁ + b = 0) ∧ (3*x₂^2 + 2*a*x₂ + b = 0)) 
  (h2 : f a b c x₁ = x₁) 
  (h3 : x₁ < x₂) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 3*(f a b c x)^2 + 2*a*(f a b c x) + b = 0 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l1999_199901


namespace NUMINAMATH_CALUDE_fraction_equality_l1999_199969

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (3 * a) / (3 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1999_199969


namespace NUMINAMATH_CALUDE_inequality_properties_l1999_199975

theorem inequality_properties (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (1 / a > 1 / b) ∧
  (a^(1/5 : ℝ) < b^(1/5 : ℝ)) ∧
  (Real.sqrt (a^2 - a) > Real.sqrt (b^2 - b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l1999_199975


namespace NUMINAMATH_CALUDE_max_score_15_cards_l1999_199953

/-- Represents the score for a hand of cards -/
def score (r b y : ℕ) : ℕ := r + 2 * r * b + 3 * b * y

/-- The maximum score achievable with 15 cards -/
theorem max_score_15_cards : 
  ∃ (r b y : ℕ), r + b + y = 15 ∧ ∀ (r' b' y' : ℕ), r' + b' + y' = 15 → score r' b' y' ≤ score r b y ∧ score r b y = 168 := by
  sorry

end NUMINAMATH_CALUDE_max_score_15_cards_l1999_199953


namespace NUMINAMATH_CALUDE_trihedral_dihedral_planar_equality_l1999_199919

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

/-- Represents a dihedral angle -/
def DihedralAngle : Type := Real

/-- 
Given a trihedral angle, there exists a planar angle equal to 
the dihedral angle opposite to one of its plane angles.
-/
theorem trihedral_dihedral_planar_equality 
  (t : TrihedralAngle) : 
  ∃ (planar_angle : Real) (dihedral : DihedralAngle), 
    planar_angle = dihedral := by
  sorry


end NUMINAMATH_CALUDE_trihedral_dihedral_planar_equality_l1999_199919


namespace NUMINAMATH_CALUDE_ad_duration_l1999_199944

theorem ad_duration (num_ads : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) 
  (h1 : num_ads = 5)
  (h2 : cost_per_minute = 4000)
  (h3 : total_cost = 60000) :
  (total_cost / cost_per_minute) / num_ads = 3 := by
  sorry

end NUMINAMATH_CALUDE_ad_duration_l1999_199944


namespace NUMINAMATH_CALUDE_elective_course_arrangements_l1999_199921

def slots : ℕ := 6
def courses : ℕ := 3

theorem elective_course_arrangements : 
  (slots.factorial) / ((slots - courses).factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_arrangements_l1999_199921


namespace NUMINAMATH_CALUDE_difference_of_ones_and_zeros_237_l1999_199941

def base_2_representation (n : Nat) : List Nat :=
  sorry

def count_zeros (l : List Nat) : Nat :=
  sorry

def count_ones (l : List Nat) : Nat :=
  sorry

theorem difference_of_ones_and_zeros_237 :
  let binary_237 := base_2_representation 237
  let x := count_zeros binary_237
  let y := count_ones binary_237
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_difference_of_ones_and_zeros_237_l1999_199941


namespace NUMINAMATH_CALUDE_candy_mixture_price_prove_candy_mixture_price_l1999_199929

/-- Given two types of candies with equal amounts, priced at 2 and 3 rubles per kilogram respectively,
    the price of their mixture is 2.4 rubles per kilogram. -/
theorem candy_mixture_price : ℝ → Prop :=
  fun (s : ℝ) ↦
    let candy1_weight := s / 2
    let candy2_weight := s / 3
    let total_weight := candy1_weight + candy2_weight
    let total_cost := 2 * candy1_weight + 3 * candy2_weight
    let mixture_price := total_cost / total_weight
    mixture_price = 2.4

/-- Proof of the candy mixture price theorem -/
theorem prove_candy_mixture_price : ∃ s : ℝ, candy_mixture_price s := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_price_prove_candy_mixture_price_l1999_199929


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1999_199964

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 39) →
  (a 3 + a 6 + a 9 = 33) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1999_199964


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1999_199954

theorem sufficient_condition_for_inequality (a b : ℝ) :
  Real.sqrt (a - 1) > Real.sqrt (b - 1) → a > b ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1999_199954


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1999_199967

/-- The angle between two vectors in radians -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)  -- 60 degrees in radians
  (h2 : a = (1, Real.sqrt 3))
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :  -- |b| = 1
  Real.sqrt (((a.1 + 2*b.1)^2) + ((a.2 + 2*b.2)^2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1999_199967


namespace NUMINAMATH_CALUDE_triangle_properties_and_heron_l1999_199994

/-- Triangle properties and Heron's formula -/
theorem triangle_properties_and_heron (r r_a r_b r_c p a b c S : ℝ) 
  (hr : r > 0) (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0)
  (hp : p > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S > 0)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_inradius : r = S / p)
  (h_exradius_a : r_a = S / (p - a))
  (h_exradius_b : r_b = S / (p - b))
  (h_exradius_c : r_c = S / (p - c)) : 
  (r * p = r_a * (p - a)) ∧ 
  (r * r_a = (p - b) * (p - c)) ∧
  (r_b * r_c = p * (p - a)) ∧
  (S^2 = p * (p - a) * (p - b) * (p - c)) ∧
  (S^2 = r * r_a * r_b * r_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_and_heron_l1999_199994


namespace NUMINAMATH_CALUDE_system_solutions_l1999_199962

theorem system_solutions (x a : ℝ) : 
  (a = -3*x^2 + 5*x - 2 ∧ (x+2)*a = 4*(x^2 - 1) ∧ x ≠ -2) → 
  ((x = 1 ∧ a = 0) ∨ (x = 0 ∧ a = -2) ∨ (x = -8/3 ∧ a = -110/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1999_199962


namespace NUMINAMATH_CALUDE_trucks_meeting_l1999_199950

/-- Two trucks meeting on a highway --/
theorem trucks_meeting 
  (initial_distance : ℝ) 
  (speed_A speed_B : ℝ) 
  (delay : ℝ) :
  initial_distance = 940 →
  speed_A = 90 →
  speed_B = 80 →
  delay = 1 →
  ∃ (t : ℝ), 
    t > 0 ∧ 
    speed_A * (t + delay) + speed_B * t = initial_distance ∧ 
    speed_A * (t + delay) - speed_B * t = 140 :=
by sorry

end NUMINAMATH_CALUDE_trucks_meeting_l1999_199950


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l1999_199976

theorem sqrt_product_plus_one : 
  Real.sqrt ((41 : ℝ) * 40 * 39 * 38 + 1) = 1559 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l1999_199976


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1999_199946

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 3*x = 0) ∧
  (∃ x : ℝ, x^2 - 4*x - 1 = 0) ∧
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ (x = 0 ∨ x = 3)) ∧
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1999_199946


namespace NUMINAMATH_CALUDE_julie_work_hours_l1999_199937

/-- Given Julie's work conditions, prove she needs to work 18 hours per week during school year --/
theorem julie_work_hours : 
  ∀ (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
    (school_weeks : ℕ) (school_earnings : ℕ),
  summer_weeks = 10 →
  summer_hours_per_week = 60 →
  summer_earnings = 7500 →
  school_weeks = 40 →
  school_earnings = 9000 →
  (school_earnings * summer_weeks * summer_hours_per_week) / 
    (summer_earnings * school_weeks) = 18 := by
sorry

end NUMINAMATH_CALUDE_julie_work_hours_l1999_199937


namespace NUMINAMATH_CALUDE_extreme_points_inequality_l1999_199974

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - 1 - a * Real.log x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ →
  (∀ x, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂) →
  (f a x₁) / x₂ > -7/2 - Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_inequality_l1999_199974


namespace NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l1999_199991

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_7_balls_3_boxes : distribute 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l1999_199991


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1999_199981

theorem consecutive_numbers_sum (a b c d : ℤ) : 
  b = a + 1 → 
  c = b + 1 → 
  d = c + 1 → 
  b * c = 2970 → 
  a + d = 113 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1999_199981


namespace NUMINAMATH_CALUDE_ratio_reduction_l1999_199993

theorem ratio_reduction (x : ℕ) (h : x ≥ 3) :
  (∃ a b : ℕ, a < b ∧ (6 - x : ℚ) / (7 - x) < a / b) ∧
  (∀ a b : ℕ, a < b → (6 - x : ℚ) / (7 - x) < a / b → 4 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_ratio_reduction_l1999_199993


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l1999_199912

/-- Given two rectangles with equal area, where one rectangle has dimensions 4 inches by 30 inches,
    and the other has a length of 5 inches, prove that the width of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_width (area : ℝ) (length1 width1 length2 width2 : ℝ) : 
  area = length1 * width1 → -- Area of the first rectangle
  area = length2 * width2 → -- Area of the second rectangle
  length1 = 4 →             -- Length of the first rectangle
  width1 = 30 →             -- Width of the first rectangle
  length2 = 5 →             -- Length of the second rectangle
  width2 = 24 :=            -- Width of the second rectangle (to be proved)
by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l1999_199912


namespace NUMINAMATH_CALUDE_inscribed_parallelogram_theorem_l1999_199995

/-- A triangle with an inscribed parallelogram -/
structure InscribedParallelogram where
  -- Triangle side lengths
  side1 : ℝ
  side2 : ℝ
  -- Parallelogram side on triangle base
  para_side : ℝ

/-- Properties of the inscribed parallelogram -/
def inscribed_parallelogram_properties (t : InscribedParallelogram) : Prop :=
  t.side1 = 9 ∧ t.side2 = 15 ∧ t.para_side = 6

/-- Theorem about the inscribed parallelogram -/
theorem inscribed_parallelogram_theorem (t : InscribedParallelogram) 
  (h : inscribed_parallelogram_properties t) :
  ∃ (other_side base : ℝ),
    other_side = 4 * Real.sqrt 2 ∧ 
    base = 18 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_parallelogram_theorem_l1999_199995


namespace NUMINAMATH_CALUDE_prize_probability_l1999_199980

theorem prize_probability (p : ℝ) (h : p = 0.9) :
  Nat.choose 5 3 * p^3 * (1 - p)^2 = Nat.choose 5 3 * 0.9^3 * 0.1^2 := by
  sorry

end NUMINAMATH_CALUDE_prize_probability_l1999_199980


namespace NUMINAMATH_CALUDE_sarahs_earnings_proof_l1999_199948

/-- Sarah's earnings for an 8-hour day, given Connor's hourly wage and their wage ratio -/
def sarahs_daily_earnings (connors_hourly_wage : ℝ) (wage_ratio : ℝ) (hours_worked : ℝ) : ℝ :=
  connors_hourly_wage * wage_ratio * hours_worked

/-- Theorem stating Sarah's earnings for an 8-hour day -/
theorem sarahs_earnings_proof (connors_hourly_wage : ℝ) (wage_ratio : ℝ) (hours_worked : ℝ)
    (h1 : connors_hourly_wage = 7.20)
    (h2 : wage_ratio = 5)
    (h3 : hours_worked = 8) :
    sarahs_daily_earnings connors_hourly_wage wage_ratio hours_worked = 288 := by
  sorry

#eval sarahs_daily_earnings 7.20 5 8

end NUMINAMATH_CALUDE_sarahs_earnings_proof_l1999_199948


namespace NUMINAMATH_CALUDE_complement_of_40_degree_angle_l1999_199914

/-- Given an angle A of 40 degrees, its complement is 50 degrees. -/
theorem complement_of_40_degree_angle (A : ℝ) : 
  A = 40 → (90 - A) = 50 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_40_degree_angle_l1999_199914


namespace NUMINAMATH_CALUDE_exam_time_ratio_l1999_199992

/-- Given an examination with the following parameters:
  * Total duration: 3 hours
  * Total number of questions: 200
  * Number of type A problems: 25
  * Time spent on type A problems: 40 minutes

  Prove that the ratio of time spent on type A problems to time spent on type B problems is 2:7. -/
theorem exam_time_ratio :
  let total_time : ℕ := 3 * 60  -- Total time in minutes
  let type_a_time : ℕ := 40     -- Time spent on type A problems
  let type_b_time : ℕ := total_time - type_a_time  -- Time spent on type B problems
  (type_a_time : ℚ) / (type_b_time : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_exam_time_ratio_l1999_199992


namespace NUMINAMATH_CALUDE_remainder_of_power_sum_l1999_199984

theorem remainder_of_power_sum (n : ℕ) : (Nat.pow 6 83 + Nat.pow 8 83) % 49 = 35 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_sum_l1999_199984


namespace NUMINAMATH_CALUDE_horner_v3_value_l1999_199990

/-- Horner's Method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f : List ℝ := [12, 35, -8, 79, 6, 5, 3]

/-- The x-value at which to evaluate the polynomial -/
def x : ℝ := -4

/-- Theorem: The value of v3 in Horner's Method for f(x) at x = -4 is -57 -/
theorem horner_v3_value : 
  let v0 := f.reverse.head!
  let v1 := v0 * x + f.reverse.tail!.head!
  let v2 := v1 * x + f.reverse.tail!.tail!.head!
  let v3 := v2 * x + f.reverse.tail!.tail!.tail!.head!
  v3 = -57 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l1999_199990


namespace NUMINAMATH_CALUDE_equation_solution_l1999_199900

theorem equation_solution :
  ∃ x : ℚ, 
    (((x / 128 + (1 + 2 / 7)) / (5 - 4 * (2 / 21) * 0.75)) / 
    ((1 / 3 + 5 / 7 * 1.4) / ((4 - 2 * (2 / 3)) * 3)) = 4.5) ∧ 
    x = 1440 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1999_199900


namespace NUMINAMATH_CALUDE_factorial_sum_eq_power_of_three_l1999_199983

theorem factorial_sum_eq_power_of_three (a b c d : ℕ) : 
  a ≤ b → b ≤ c → a.factorial + b.factorial + c.factorial = 3^d →
  ((a, b, c, d) = (1, 1, 1, 1) ∨ (a, b, c, d) = (1, 2, 3, 2) ∨ (a, b, c, d) = (1, 2, 4, 3)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_eq_power_of_three_l1999_199983


namespace NUMINAMATH_CALUDE_average_weight_increase_l1999_199945

/-- Proves that replacing a person weighing 65 kg with a person weighing 77 kg
    in a group of 8 people increases the average weight by 1.5 kg. -/
theorem average_weight_increase (initial_average : ℝ) :
  let initial_total := 8 * initial_average
  let new_total := initial_total - 65 + 77
  let new_average := new_total / 8
  new_average - initial_average = 1.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1999_199945


namespace NUMINAMATH_CALUDE_f_diff_at_pi_l1999_199957

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 3 * x^2 + 7 * Real.sin x

theorem f_diff_at_pi : f Real.pi - f (-Real.pi) = -2 * Real.pi^3 := by
  sorry

end NUMINAMATH_CALUDE_f_diff_at_pi_l1999_199957


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1999_199985

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = m + 3 * x) ↔ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1999_199985


namespace NUMINAMATH_CALUDE_homogeneous_polynomial_on_circle_l1999_199902

-- Define a homogeneous polynomial
def IsHomogeneous (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (c x y : ℝ), P (c * x) (c * y) = c^n * P x y

-- Define the theorem
theorem homogeneous_polynomial_on_circle (P : ℝ → ℝ → ℝ) (n : ℕ) :
  IsHomogeneous P n →
  (∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) →
  n > 0 →
  ∃ k : ℕ, k > 0 ∧ ∀ x y : ℝ, P x y = (x^2 + y^2)^k :=
by sorry

end NUMINAMATH_CALUDE_homogeneous_polynomial_on_circle_l1999_199902


namespace NUMINAMATH_CALUDE_largest_divisor_of_prime_square_difference_l1999_199958

theorem largest_divisor_of_prime_square_difference (p q : ℕ) 
  (hp : Prime p) (hq : Prime q) (h_order : q < p) : 
  (∀ (d : ℕ), d > 2 → ∃ (p' q' : ℕ), Prime p' ∧ Prime q' ∧ q' < p' ∧ ¬(d ∣ (p'^2 - q'^2))) ∧ 
  (∀ (p' q' : ℕ), Prime p' → Prime q' → q' < p' → 2 ∣ (p'^2 - q'^2)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_prime_square_difference_l1999_199958


namespace NUMINAMATH_CALUDE_perimeter_of_square_III_is_four_l1999_199924

/-- Given three squares I, II, and III, prove that the perimeter of square III is 4 -/
theorem perimeter_of_square_III_is_four :
  ∀ (side_I side_II side_III : ℝ),
  side_I * 4 = 20 →
  side_II * 4 = 16 →
  side_III = side_I - side_II →
  side_III * 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_III_is_four_l1999_199924


namespace NUMINAMATH_CALUDE_min_distance_complex_points_l1999_199965

theorem min_distance_complex_points (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min : ℝ), min = 3 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_points_l1999_199965


namespace NUMINAMATH_CALUDE_horse_cost_problem_l1999_199923

theorem horse_cost_problem (selling_price : ℕ) (cost : ℕ) : 
  selling_price = 56 →
  selling_price = cost + (cost * cost) / 100 →
  cost = 40 := by
sorry

end NUMINAMATH_CALUDE_horse_cost_problem_l1999_199923


namespace NUMINAMATH_CALUDE_combination_sum_l1999_199931

theorem combination_sum : Nat.choose 99 2 + Nat.choose 99 3 = 161700 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_l1999_199931


namespace NUMINAMATH_CALUDE_two_digit_sum_divisibility_l1999_199987

theorem two_digit_sum_divisibility (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ∃ k : Int, (10 * a + b) + (10 * b + a) = 11 * k :=
by sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisibility_l1999_199987


namespace NUMINAMATH_CALUDE_didi_fundraising_price_per_slice_l1999_199918

/-- Proves that the price per slice is $1 given the conditions of Didi's fundraising event --/
theorem didi_fundraising_price_per_slice :
  ∀ (price_per_slice : ℚ),
    (10 : ℕ) * (8 : ℕ) * price_per_slice +  -- Revenue from slice sales
    (10 : ℕ) * (8 : ℕ) * (1/2 : ℚ) +        -- Donation from first business owner
    (10 : ℕ) * (8 : ℕ) * (1/4 : ℚ) = 140    -- Donation from second business owner
    → price_per_slice = 1 := by
  sorry

end NUMINAMATH_CALUDE_didi_fundraising_price_per_slice_l1999_199918


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1999_199961

/-- The complex number (1+i)/i is in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 + Complex.I) / Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1999_199961


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1999_199979

theorem chess_tournament_games (n : ℕ) (h : n = 24) : 
  n * (n - 1) / 2 = 552 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1999_199979


namespace NUMINAMATH_CALUDE_min_value_expression_l1999_199956

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 ∧
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) = 8 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1999_199956


namespace NUMINAMATH_CALUDE_spring_length_formula_l1999_199989

/-- Spring scale properties -/
structure SpringScale where
  initialLength : ℝ
  extensionRate : ℝ

/-- The analytical expression for the total length of a spring -/
def totalLength (s : SpringScale) (mass : ℝ) : ℝ :=
  s.initialLength + s.extensionRate * mass

/-- Theorem: The analytical expression for the total length of the spring is y = 10 + 2x -/
theorem spring_length_formula (s : SpringScale) (mass : ℝ) :
  s.initialLength = 10 ∧ s.extensionRate = 2 →
  totalLength s mass = 10 + 2 * mass := by
  sorry

end NUMINAMATH_CALUDE_spring_length_formula_l1999_199989


namespace NUMINAMATH_CALUDE_exam_questions_count_l1999_199996

theorem exam_questions_count :
  ∀ (a b c : ℕ),
    b = 23 →
    c = 1 →
    a ≥ 1 →
    b ≥ 1 →
    c ≥ 1 →
    a ≥ (6 : ℚ) / 10 * (a + 2 * b + 3 * c) →
    a + b + c = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_questions_count_l1999_199996


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1999_199936

theorem circle_area_ratio (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h_diameter : 2 * r = 0.2 * (2 * s)) : 
  (π * r^2) / (π * s^2) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1999_199936


namespace NUMINAMATH_CALUDE_continued_fraction_value_l1999_199915

def continued_fraction (a b c d : ℚ) : ℚ :=
  -1 / (a - 1 / (b - 1 / (c - 1 / d)))

theorem continued_fraction_value : continued_fraction 2 2 2 2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l1999_199915


namespace NUMINAMATH_CALUDE_largest_number_in_set_l1999_199988

theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  -3 * a = max (-3 * a) (max (5 * a) (max (24 / a) (max (a ^ 2) 1))) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l1999_199988


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1999_199949

theorem rectangular_solid_diagonal (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1999_199949


namespace NUMINAMATH_CALUDE_bowling_shoe_rental_cost_l1999_199977

/-- The cost to rent bowling shoes for a day, given the following conditions:
  1. The cost per game is $1.75.
  2. A person has $12.80 in total.
  3. The person can bowl a maximum of 7 complete games. -/
theorem bowling_shoe_rental_cost :
  let cost_per_game : ℚ := 175 / 100
  let total_money : ℚ := 1280 / 100
  let max_games : ℕ := 7
  let shoe_rental_cost : ℚ := total_money - (cost_per_game * max_games)
  shoe_rental_cost = 55 / 100 := by sorry

end NUMINAMATH_CALUDE_bowling_shoe_rental_cost_l1999_199977


namespace NUMINAMATH_CALUDE_remainder_equality_l1999_199973

theorem remainder_equality (P P' D Q R R' : ℕ) 
  (h1 : P > P') 
  (h2 : R = P % D) 
  (h3 : R' = P' % D) : 
  ((P + Q) * P') % D = (R * R') % D := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1999_199973


namespace NUMINAMATH_CALUDE_min_value_theorem_l1999_199998

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (y / (2*x) + 1 / y) ≥ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1999_199998


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1999_199951

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1999_199951


namespace NUMINAMATH_CALUDE_triangle_area_l1999_199913

/-- The area of a triangle with sides a = 4, b = 5, and angle C = 60° is 5√3 -/
theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 4) (h2 : b = 5) (h3 : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1999_199913


namespace NUMINAMATH_CALUDE_light_travel_distance_l1999_199971

/-- The distance light travels in one year in kilometers -/
def light_year : ℝ := 9460000000000

/-- The number of years we're considering -/
def years : ℕ := 120

/-- Theorem stating the distance light travels in 120 years -/
theorem light_travel_distance :
  light_year * years = 1.1352e15 := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l1999_199971


namespace NUMINAMATH_CALUDE_simplify_expression_l1999_199960

theorem simplify_expression (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a - 2*b ≠ 0) (h3 : a^2 - b^2 ≠ 0) (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1999_199960


namespace NUMINAMATH_CALUDE_average_parking_cost_for_9_hours_l1999_199933

/-- Calculates the average cost per hour for parking given the following conditions:
  * Base cost for up to 2 hours
  * Additional cost per hour after 2 hours
  * Total number of hours parked
-/
def averageParkingCost (baseCost hourlyRate : ℚ) (totalHours : ℕ) : ℚ :=
  let totalCost := baseCost + hourlyRate * (totalHours - 2)
  totalCost / totalHours

/-- Theorem stating that the average parking cost for 9 hours is $3.03 -/
theorem average_parking_cost_for_9_hours :
  averageParkingCost 15 (7/4) 9 = 303/100 := by
  sorry

#eval averageParkingCost 15 (7/4) 9

end NUMINAMATH_CALUDE_average_parking_cost_for_9_hours_l1999_199933


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_even_numbers_divisible_by_48_l1999_199930

theorem product_of_three_consecutive_even_numbers_divisible_by_48 (k : ℤ) :
  ∃ (n : ℤ), (2*k) * (2*k + 2) * (2*k + 4) = 48 * n :=
sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_even_numbers_divisible_by_48_l1999_199930


namespace NUMINAMATH_CALUDE_odd_solution_exists_l1999_199966

theorem odd_solution_exists (k m n : ℕ+) (h : m * n = k^2 + k + 3) :
  (∃ (x y : ℤ), x^2 + 11 * y^2 = 4 * m ∧ x % 2 ≠ 0 ∧ y % 2 ≠ 0) ∨
  (∃ (x y : ℤ), x^2 + 11 * y^2 = 4 * n ∧ x % 2 ≠ 0 ∧ y % 2 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_odd_solution_exists_l1999_199966


namespace NUMINAMATH_CALUDE_trig_identity_l1999_199940

theorem trig_identity (θ : ℝ) (h : θ ≠ 0) (h' : θ ≠ π/2) : 
  (Real.sin θ + 1 / Real.sin θ)^2 + (Real.cos θ + 1 / Real.cos θ)^2 = 
  6 + 2 * ((Real.sin θ / Real.cos θ)^2 + (Real.cos θ / Real.sin θ)^2) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l1999_199940


namespace NUMINAMATH_CALUDE_medals_count_l1999_199997

/-- The total number of medals displayed in the sports center -/
def total_medals (gold silver bronze : ℕ) : ℕ :=
  gold + silver + bronze

/-- Theorem: The total number of medals is 67 -/
theorem medals_count : total_medals 19 32 16 = 67 := by
  sorry

end NUMINAMATH_CALUDE_medals_count_l1999_199997


namespace NUMINAMATH_CALUDE_unbroken_seashells_l1999_199939

/-- Given that Tom found 7 seashells in total and 4 of them were broken,
    prove that the number of unbroken seashells is 3. -/
theorem unbroken_seashells (total : ℕ) (broken : ℕ) 
  (h1 : total = 7) 
  (h2 : broken = 4) : 
  total - broken = 3 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l1999_199939


namespace NUMINAMATH_CALUDE_worker_schedule_theorem_l1999_199916

/-- Represents a worker's daily schedule and pay --/
structure WorkerSchedule where
  baseHours : ℝ
  basePay : ℝ
  bonusPay : ℝ
  bonusHours : ℝ
  bonusHourlyRate : ℝ

/-- Theorem stating the conditions and conclusion about the worker's schedule --/
theorem worker_schedule_theorem (w : WorkerSchedule) 
  (h1 : w.basePay = 80)
  (h2 : w.bonusPay = 20)
  (h3 : w.bonusHours = 2)
  (h4 : w.bonusHourlyRate = 10)
  (h5 : w.bonusHourlyRate * (w.baseHours + w.bonusHours) = w.basePay + w.bonusPay) :
  w.baseHours = 8 := by
  sorry

#check worker_schedule_theorem

end NUMINAMATH_CALUDE_worker_schedule_theorem_l1999_199916


namespace NUMINAMATH_CALUDE_total_is_27_l1999_199927

def purchase1 : ℚ := 2.47
def purchase2 : ℚ := 7.51
def purchase3 : ℚ := 11.56
def purchase4 : ℚ := 4.98

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 0.5 then x.floor else x.ceil

def total_rounded : ℤ := 
  round_to_nearest_dollar (purchase1 + purchase2 + purchase3 + purchase4)

theorem total_is_27 : total_rounded = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_is_27_l1999_199927


namespace NUMINAMATH_CALUDE_distance_traveled_l1999_199935

/-- Given a speed of 65 km/hr and a time of 3 hr, the distance traveled is 195 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 65) (h2 : time = 3) :
  speed * time = 195 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l1999_199935


namespace NUMINAMATH_CALUDE_videotape_boxes_needed_l1999_199925

/-- Represents the duration of a program -/
structure Duration :=
  (value : ℝ)

/-- Represents a box of videotape -/
structure Box :=
  (capacity : ℝ)

/-- Represents the content to be recorded -/
structure Content :=
  (tvEpisodes : ℕ)
  (skits : ℕ)
  (songs : ℕ)

def Box.canRecord (b : Box) (d1 d2 : Duration) (n1 n2 : ℕ) : Prop :=
  n1 * d1.value + n2 * d2.value ≤ b.capacity

theorem videotape_boxes_needed 
  (tvDuration skitDuration songDuration : Duration)
  (box : Box)
  (content : Content)
  (h1 : box.canRecord tvDuration skitDuration 2 1)
  (h2 : box.canRecord skitDuration songDuration 2 3)
  (h3 : skitDuration.value > songDuration.value)
  (h4 : content.tvEpisodes = 7 ∧ content.skits = 11 ∧ content.songs = 20) :
  (∃ n : ℕ, n = 8 ∨ n = 9) ∧ 
  (∀ m : ℕ, m < 8 → 
    m * box.capacity < 
      content.tvEpisodes * tvDuration.value + 
      content.skits * skitDuration.value + 
      content.songs * songDuration.value) :=
sorry

end NUMINAMATH_CALUDE_videotape_boxes_needed_l1999_199925
