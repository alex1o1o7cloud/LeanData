import Mathlib

namespace circle_intersection_trajectory_l160_16058

/-- Given two circles with centers at (a₁, 0) and (a₂, 0), both passing through (1, 0),
    intersecting the positive y-axis at (0, y₁) and (0, y₂) respectively,
    prove that the trajectory of (1/a₁, 1/a₂) is a straight line when ln y₁ + ln y₂ = 0 -/
theorem circle_intersection_trajectory (a₁ a₂ y₁ y₂ : ℝ) 
    (h1 : (1 - a₁)^2 = a₁^2 + y₁^2)
    (h2 : (1 - a₂)^2 = a₂^2 + y₂^2)
    (h3 : Real.log y₁ + Real.log y₂ = 0) :
    ∃ (m b : ℝ), ∀ (x y : ℝ), (x = 1/a₁ ∧ y = 1/a₂) → y = m*x + b :=
  sorry


end circle_intersection_trajectory_l160_16058


namespace parallelogram_BJ_length_l160_16044

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram ABCD with additional points H, J, and K -/
structure Parallelogram :=
  (A B C D H J K : Point)

/-- Checks if three points are collinear -/
def collinear (P Q R : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Checks if two line segments are parallel -/
def parallel (P Q R S : Point) : Prop := sorry

theorem parallelogram_BJ_length
  (ABCD : Parallelogram)
  (h1 : collinear ABCD.A ABCD.D ABCD.H)
  (h2 : collinear ABCD.B ABCD.H ABCD.J)
  (h3 : collinear ABCD.B ABCD.H ABCD.K)
  (h4 : collinear ABCD.A ABCD.C ABCD.J)
  (h5 : collinear ABCD.D ABCD.C ABCD.K)
  (h6 : distance ABCD.J ABCD.H = 20)
  (h7 : distance ABCD.K ABCD.H = 30)
  (h8 : distance ABCD.A ABCD.D = 2 * distance ABCD.B ABCD.C)
  (h9 : parallel ABCD.A ABCD.B ABCD.D ABCD.C)
  (h10 : parallel ABCD.A ABCD.D ABCD.B ABCD.C) :
  distance ABCD.B ABCD.J = 5 := by sorry

end parallelogram_BJ_length_l160_16044


namespace quadratic_factorization_l160_16023

theorem quadratic_factorization : ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l160_16023


namespace inscribed_circle_radius_height_ratio_l160_16066

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The height corresponding to the hypotenuse -/
  m : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- m is positive -/
  m_pos : 0 < m
  /-- r is positive -/
  r_pos : 0 < r

/-- The ratio of the inscribed circle radius to the height is between 0.4 and 0.5 -/
theorem inscribed_circle_radius_height_ratio 
  (t : RightTriangleWithInscribedCircle) : 0.4 < t.r / t.m ∧ t.r / t.m < 0.5 := by
  sorry

end inscribed_circle_radius_height_ratio_l160_16066


namespace smallest_base_for_100_l160_16030

theorem smallest_base_for_100 : 
  ∃ b : ℕ, (b ≥ 5 ∧ b^2 ≤ 100 ∧ 100 < b^3) ∧ 
  (∀ c : ℕ, c < b → (c^2 > 100 ∨ 100 ≥ c^3)) := by
  sorry

end smallest_base_for_100_l160_16030


namespace greatest_integer_prime_absolute_value_l160_16040

theorem greatest_integer_prime_absolute_value : 
  ∃ (x : ℤ), (∀ (y : ℤ), y > x → ¬(Nat.Prime (Int.natAbs (8 * y^2 - 56 * y + 21)))) ∧ 
  (Nat.Prime (Int.natAbs (8 * x^2 - 56 * x + 21))) ∧ 
  x = 4 :=
sorry

end greatest_integer_prime_absolute_value_l160_16040


namespace lcm_problem_l160_16035

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by
  sorry

end lcm_problem_l160_16035


namespace cubic_equation_roots_l160_16025

theorem cubic_equation_roots (x : ℝ) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    (r₁ < 0 ∧ r₂ < 0 ∧ r₃ > 0) ∧
    (∀ y : ℝ, y^3 + 3*y^2 - 4*y - 12 = 0 ↔ y = r₁ ∨ y = r₂ ∨ y = r₃) :=
by sorry

end cubic_equation_roots_l160_16025


namespace inequality_max_value_inequality_range_l160_16016

theorem inequality_max_value (x y : ℝ) (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 1 3) :
  (2 * x^2 + y^2) / (x * y) ≤ 2 * Real.sqrt 2 :=
by sorry

theorem inequality_range (a : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 3 → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by sorry

end inequality_max_value_inequality_range_l160_16016


namespace smallest_consecutive_even_integer_l160_16070

/-- Represents three consecutive even integers -/
structure ConsecutiveEvenIntegers where
  middle : ℕ
  is_even : Even middle

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The property that the sum of the integers is one-fifth of their product -/
def sum_is_one_fifth_of_product (integers : ConsecutiveEvenIntegers) : Prop :=
  (integers.middle - 2) + integers.middle + (integers.middle + 2) = 
    ((integers.middle - 2) * integers.middle * (integers.middle + 2)) / 5

theorem smallest_consecutive_even_integer :
  ∃ (integers : ConsecutiveEvenIntegers),
    (is_two_digit (integers.middle - 2)) ∧
    (is_two_digit integers.middle) ∧
    (is_two_digit (integers.middle + 2)) ∧
    (sum_is_one_fifth_of_product integers) ∧
    (integers.middle - 2 = 86) := by
  sorry

end smallest_consecutive_even_integer_l160_16070


namespace math_score_proof_l160_16003

theorem math_score_proof (a b c : ℕ) : 
  (a + b + c = 288) →  -- Sum of scores is 288
  (∃ k : ℕ, a = 2*k ∧ b = 2*k + 2 ∧ c = 2*k + 4) →  -- Consecutive even numbers
  b = 96  -- Mathematics score is 96
:= by sorry

end math_score_proof_l160_16003


namespace number_equation_solution_l160_16045

theorem number_equation_solution : ∃ x : ℝ, 5 * x + 4 = 19 ∧ x = 3 := by
  sorry

end number_equation_solution_l160_16045


namespace circle_area_ratio_l160_16077

/-- Given a triangle with sides 13, 14, and 15, the ratio of the area of its 
    circumscribed circle to the area of its inscribed circle is (65/32)^2 -/
theorem circle_area_ratio (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let p := (a + b + c) / 2
  let s := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := s / p
  let R := (a * b * c) / (4 * s)
  (R / r) ^ 2 = (65 / 32) ^ 2 := by
  sorry


end circle_area_ratio_l160_16077


namespace binomial_coefficient_sum_l160_16010

theorem binomial_coefficient_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ : ℝ) :
  (∀ x : ℝ, (1 + x)^14 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + 
    a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ + 8*a₈ + 9*a₉ + 10*a₁₀ + 
    11*a₁₁ + 12*a₁₂ + 13*a₁₃ + 14*a₁₄ = 7 * 2^14 := by
  sorry

end binomial_coefficient_sum_l160_16010


namespace decimal_to_fraction_times_three_l160_16096

theorem decimal_to_fraction_times_three :
  (2.36 : ℚ) * 3 = 177 / 25 := by
sorry

end decimal_to_fraction_times_three_l160_16096


namespace average_weight_of_class_l160_16086

theorem average_weight_of_class (num_male num_female : ℕ) 
                                (avg_weight_male avg_weight_female : ℚ) :
  num_male = 20 →
  num_female = 20 →
  avg_weight_male = 42 →
  avg_weight_female = 38 →
  (num_male * avg_weight_male + num_female * avg_weight_female) / (num_male + num_female) = 40 := by
sorry

end average_weight_of_class_l160_16086


namespace factor_expression_l160_16082

theorem factor_expression (b : ℝ) : 43 * b^2 + 129 * b = 43 * b * (b + 3) := by
  sorry

end factor_expression_l160_16082


namespace remaining_flight_time_l160_16024

def flight_duration : ℕ := 10 * 60  -- in minutes
def tv_episode_duration : ℕ := 25  -- in minutes
def num_tv_episodes : ℕ := 3
def sleep_duration : ℕ := 270  -- 4.5 hours in minutes
def movie_duration : ℕ := 105  -- 1 hour 45 minutes in minutes
def num_movies : ℕ := 2

theorem remaining_flight_time :
  flight_duration - (num_tv_episodes * tv_episode_duration + sleep_duration + num_movies * movie_duration) = 45 := by
  sorry

end remaining_flight_time_l160_16024


namespace regular_polygon_exterior_angle_20_l160_16090

/-- A regular polygon with exterior angles measuring 20 degrees has 18 sides. -/
theorem regular_polygon_exterior_angle_20 : 
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 20 → 
  n = 18 := by
  sorry

end regular_polygon_exterior_angle_20_l160_16090


namespace sheila_hourly_rate_l160_16098

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday
  weekly_earnings : ℕ    -- Weekly earnings in dollars

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly rate --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 8
  , tue_thu_hours := 6
  , weekly_earnings := 288 }

/-- Theorem stating that Sheila's hourly rate is $8 --/
theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 8 := by
  sorry

end sheila_hourly_rate_l160_16098


namespace min_T_minus_S_and_max_T_l160_16087

/-- Given non-negative real numbers a, b, and c, S and T are defined as follows:
    S = a + 2b + 3c
    T = a + b^2 + c^3 -/
def S (a b c : ℝ) : ℝ := a + 2*b + 3*c
def T (a b c : ℝ) : ℝ := a + b^2 + c^3

theorem min_T_minus_S_and_max_T (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (∀ a' b' c' : ℝ, 0 ≤ a' → 0 ≤ b' → 0 ≤ c' → -3 ≤ T a' b' c' - S a' b' c') ∧
  (S a b c = 4 → T a b c ≤ 4) :=
by sorry

end min_T_minus_S_and_max_T_l160_16087


namespace log_equation_solution_l160_16083

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1)
  (h_eq : Real.log x / (3 * Real.log b) + Real.log b / (3 * Real.log x) = 1) :
  x = b ∨ x = b ^ ((3 - Real.sqrt 5) / 2) := by
  sorry

end log_equation_solution_l160_16083


namespace iphone_average_cost_l160_16063

/-- Proves that the average cost of an iPhone is $1000 given the sales data --/
theorem iphone_average_cost (iphone_count : Nat) (ipad_count : Nat) (appletv_count : Nat)
  (ipad_cost : ℝ) (appletv_cost : ℝ) (total_average : ℝ)
  (h1 : iphone_count = 100)
  (h2 : ipad_count = 20)
  (h3 : appletv_count = 80)
  (h4 : ipad_cost = 900)
  (h5 : appletv_cost = 200)
  (h6 : total_average = 670) :
  (iphone_count * 1000 + ipad_count * ipad_cost + appletv_count * appletv_cost) /
    (iphone_count + ipad_count + appletv_count : ℝ) = total_average :=
by sorry

end iphone_average_cost_l160_16063


namespace linear_coefficient_of_quadratic_equation_l160_16095

theorem linear_coefficient_of_quadratic_equation :
  let equation := fun x : ℝ => x^2 - 2022*x - 2023
  ∃ a b c : ℝ, (∀ x, equation x = a*x^2 + b*x + c) ∧ b = -2022 :=
sorry

end linear_coefficient_of_quadratic_equation_l160_16095


namespace cookie_count_pastry_shop_cookies_l160_16017

/-- Given a ratio of doughnuts, cookies, and muffins, and the number of doughnuts and muffins,
    calculate the number of cookies. -/
theorem cookie_count (doughnut_ratio cookie_ratio muffin_ratio : ℕ) 
                     (doughnut_count muffin_count : ℕ) : ℕ :=
  let total_ratio := doughnut_ratio + cookie_ratio + muffin_ratio
  let part_value := doughnut_count / doughnut_ratio
  cookie_ratio * part_value

/-- Prove that given the ratio of doughnuts, cookies, and muffins is 5 : 3 : 1,
    and there are 50 doughnuts and 10 muffins, the number of cookies is 30. -/
theorem pastry_shop_cookies : cookie_count 5 3 1 50 10 = 30 := by
  sorry

end cookie_count_pastry_shop_cookies_l160_16017


namespace samoa_price_is_4_l160_16069

/-- The price of a box of samoas -/
def samoa_price : ℝ := sorry

/-- The number of boxes of samoas sold -/
def samoa_boxes : ℕ := 3

/-- The price of a box of thin mints -/
def thin_mint_price : ℝ := 3.5

/-- The number of boxes of thin mints sold -/
def thin_mint_boxes : ℕ := 2

/-- The price of a box of fudge delights -/
def fudge_delight_price : ℝ := 5

/-- The number of boxes of fudge delights sold -/
def fudge_delight_boxes : ℕ := 1

/-- The price of a box of sugar cookies -/
def sugar_cookie_price : ℝ := 2

/-- The number of boxes of sugar cookies sold -/
def sugar_cookie_boxes : ℕ := 9

/-- The total sales amount -/
def total_sales : ℝ := 42

theorem samoa_price_is_4 : 
  samoa_price = 4 :=
by sorry

end samoa_price_is_4_l160_16069


namespace archie_marbles_l160_16068

theorem archie_marbles (initial : ℕ) : 
  (initial : ℝ) * (1 - 0.6) * 0.5 = 20 → initial = 100 := by
  sorry

end archie_marbles_l160_16068


namespace mabels_tomatoes_l160_16034

/-- The number of tomatoes Mabel has -/
def total_tomatoes (plant1 plant2 plant3 plant4 : ℕ) : ℕ :=
  plant1 + plant2 + plant3 + plant4

/-- Theorem stating the total number of tomatoes Mabel has -/
theorem mabels_tomatoes :
  ∃ (plant1 plant2 plant3 plant4 : ℕ),
    plant1 = 8 ∧
    plant2 = plant1 + 4 ∧
    plant3 = 3 * (plant1 + plant2) ∧
    plant4 = 3 * (plant1 + plant2) ∧
    total_tomatoes plant1 plant2 plant3 plant4 = 140 :=
by
  sorry

end mabels_tomatoes_l160_16034


namespace door_opening_probability_l160_16020

/-- The probability of opening a door on the second try given specific conditions -/
theorem door_opening_probability (total_keys : ℕ) (working_keys : ℕ) : 
  total_keys = 4 → working_keys = 2 → 
  (working_keys : ℚ) / total_keys * working_keys / (total_keys - 1) = 1/3 := by
sorry

end door_opening_probability_l160_16020


namespace cards_distribution_l160_16080

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 52) (h2 : num_people = 9) :
  let base_cards := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_base := num_people - people_with_extra
  people_with_base = 2 ∧ base_cards + 1 < 7 := by sorry

end cards_distribution_l160_16080


namespace jeremy_remaining_money_l160_16018

/-- Given an initial amount and the costs of various items, calculate the remaining amount --/
def remaining_amount (initial : ℕ) (jersey_cost : ℕ) (jersey_count : ℕ) (basketball_cost : ℕ) (shorts_cost : ℕ) : ℕ :=
  initial - (jersey_cost * jersey_count + basketball_cost + shorts_cost)

/-- Prove that Jeremy has $14 left after his purchases --/
theorem jeremy_remaining_money :
  remaining_amount 50 2 5 18 8 = 14 := by
  sorry

end jeremy_remaining_money_l160_16018


namespace negation_equivalence_l160_16081

theorem negation_equivalence :
  ¬(∀ x : ℝ, x^3 - x^2 + 2 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 2 ≥ 0 := by sorry

end negation_equivalence_l160_16081


namespace seth_boxes_theorem_l160_16055

/-- The number of boxes Seth bought at the market -/
def market_boxes : ℕ := 3

/-- The number of boxes Seth bought at the farm -/
def farm_boxes : ℕ := 2 * market_boxes

/-- The total number of boxes Seth initially had -/
def initial_boxes : ℕ := market_boxes + farm_boxes

/-- The number of boxes Seth gave to his mother -/
def mother_boxes : ℕ := 1

/-- The number of boxes Seth had after giving to his mother -/
def after_mother_boxes : ℕ := initial_boxes - mother_boxes

/-- The number of boxes Seth donated to charity -/
def charity_boxes : ℕ := after_mother_boxes / 4

/-- The number of boxes Seth had after donating to charity -/
def after_charity_boxes : ℕ := after_mother_boxes - charity_boxes

/-- The number of boxes Seth had left at the end -/
def final_boxes : ℕ := 4

/-- The number of boxes Seth gave to his friends -/
def friend_boxes : ℕ := after_charity_boxes - final_boxes

/-- The total number of boxes Seth bought -/
def total_boxes : ℕ := initial_boxes

theorem seth_boxes_theorem : total_boxes = 14 := by
  sorry

end seth_boxes_theorem_l160_16055


namespace relay_race_time_difference_l160_16019

theorem relay_race_time_difference 
  (total_time : ℕ) 
  (jen_time : ℕ) 
  (susan_time : ℕ) 
  (mary_time : ℕ) 
  (tiffany_time : ℕ) :
  total_time = 223 →
  jen_time = 30 →
  susan_time = jen_time + 10 →
  mary_time = 2 * susan_time →
  tiffany_time < mary_time →
  total_time = mary_time + susan_time + jen_time + tiffany_time →
  mary_time - tiffany_time = 7 :=
by sorry

end relay_race_time_difference_l160_16019


namespace markup_percentage_is_30_l160_16049

/-- Represents the markup percentage applied by a merchant -/
def markup_percentage : ℝ → ℝ := sorry

/-- Represents the discount percentage applied to the marked price -/
def discount_percentage : ℝ := 10

/-- Represents the profit percentage after discount -/
def profit_percentage : ℝ := 17

/-- Theorem stating that given the conditions, the markup percentage is 30% -/
theorem markup_percentage_is_30 :
  ∀ (cost_price : ℝ),
  cost_price > 0 →
  let marked_price := cost_price * (1 + markup_percentage cost_price / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  selling_price = cost_price * (1 + profit_percentage / 100) →
  markup_percentage cost_price = 30 :=
by sorry

end markup_percentage_is_30_l160_16049


namespace reverse_order_product_sum_l160_16013

/-- Checks if two positive integers have reverse digit order -/
def are_reverse_order (a b : ℕ) : Prop := sorry

/-- Given two positive integers m and n with reverse digit order and m * n = 1446921630, prove m + n = 79497 -/
theorem reverse_order_product_sum (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : are_reverse_order m n) 
  (h4 : m * n = 1446921630) : 
  m + n = 79497 := by sorry

end reverse_order_product_sum_l160_16013


namespace mollys_age_l160_16091

/-- Given Sandy's age and the ratio of Sandy's age to Molly's age, calculate Molly's age -/
theorem mollys_age (sandy_age : ℕ) (ratio : ℚ) (h1 : sandy_age = 49) (h2 : ratio = 7/9) :
  sandy_age / ratio = 63 :=
sorry

end mollys_age_l160_16091


namespace janice_age_proof_l160_16052

/-- Calculates a person's age given their birth year and the current year -/
def calculate_age (birth_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - birth_year

theorem janice_age_proof (current_year : ℕ) (mark_birth_year : ℕ) :
  current_year = 2021 →
  mark_birth_year = 1976 →
  let mark_age := calculate_age mark_birth_year current_year
  let graham_age := mark_age - 3
  let janice_age := graham_age / 2
  janice_age = 21 := by sorry

end janice_age_proof_l160_16052


namespace triangle_special_cosine_identity_l160_16053

theorem triangle_special_cosine_identity (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  Real.sin A = Real.cos B ∧ 
  Real.sin A = Real.tan C → 
  Real.cos A ^ 3 + Real.cos A ^ 2 - Real.cos A = 1/2 := by
sorry

end triangle_special_cosine_identity_l160_16053


namespace minimum_students_l160_16004

theorem minimum_students (b g : ℕ) : 
  b > 0 → 
  g > 0 → 
  2 * (b / 2) = g * 2 / 3 → 
  ∀ b' g', b' > 0 → g' > 0 → 2 * (b' / 2) = g' * 2 / 3 → b' + g' ≥ b + g →
  b + g = 5 := by
sorry

end minimum_students_l160_16004


namespace cell_population_growth_l160_16060

/-- The number of cells in a population after n hours, given the specified conditions -/
def cell_count (n : ℕ) : ℕ :=
  2^(n-1) + 4

/-- Theorem stating that the cell_count function correctly models the cell population growth -/
theorem cell_population_growth (n : ℕ) :
  let initial_cells := 5
  let cells_lost_per_hour := 2
  let division_factor := 2
  cell_count n = (initial_cells - cells_lost_per_hour) * division_factor^n + cells_lost_per_hour :=
by sorry

end cell_population_growth_l160_16060


namespace dog_food_cost_l160_16054

def initial_amount : ℕ := 167
def meat_cost : ℕ := 17
def chicken_cost : ℕ := 22
def veggie_cost : ℕ := 43
def egg_cost : ℕ := 5
def remaining_amount : ℕ := 35

theorem dog_food_cost : 
  initial_amount - (meat_cost + chicken_cost + veggie_cost + egg_cost + remaining_amount) = 45 := by
  sorry

end dog_food_cost_l160_16054


namespace max_backpacks_filled_fifteen_backpacks_possible_max_backpacks_is_fifteen_l160_16067

def pencils : ℕ := 150
def notebooks : ℕ := 255
def pens : ℕ := 315

theorem max_backpacks_filled (n : ℕ) : 
  (pencils % n = 0 ∧ notebooks % n = 0 ∧ pens % n = 0) →
  n ≤ 15 :=
by
  sorry

theorem fifteen_backpacks_possible : 
  pencils % 15 = 0 ∧ notebooks % 15 = 0 ∧ pens % 15 = 0 :=
by
  sorry

theorem max_backpacks_is_fifteen : 
  ∀ n : ℕ, (pencils % n = 0 ∧ notebooks % n = 0 ∧ pens % n = 0) → n ≤ 15 :=
by
  sorry

end max_backpacks_filled_fifteen_backpacks_possible_max_backpacks_is_fifteen_l160_16067


namespace arithmetic_geometric_mean_ratio_l160_16092

theorem arithmetic_geometric_mean_ratio : ∃ (c d : ℝ), 
  c > d ∧ d > 0 ∧ 
  (c + d) / 2 = 3 * Real.sqrt (c * d) ∧
  |(c / d) - 34| < 1 := by
  sorry

end arithmetic_geometric_mean_ratio_l160_16092


namespace inequality_proof_l160_16062

theorem inequality_proof (a b c : ℝ) (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  (a + b + c ≥ 3) ∧ (a * b * c ≤ 1) ∧
  ((a + b + c = 3 ∧ a * b * c = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1)) := by
  sorry

end inequality_proof_l160_16062


namespace solve_equation_l160_16033

theorem solve_equation (x n : ℝ) (h1 : x / 4 - (x - 3) / n = 1) (h2 : x = 6) : n = 6 := by
  sorry

end solve_equation_l160_16033


namespace geometric_sequence_sum_l160_16047

theorem geometric_sequence_sum (n : ℕ) : 
  (1/3 : ℝ) * (1 - (1/3)^n) / (1 - 1/3) = 728/729 → n = 6 := by
  sorry

end geometric_sequence_sum_l160_16047


namespace function_second_derivative_at_e_l160_16038

open Real

theorem function_second_derivative_at_e (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = 2 * x * (deriv^[2] f e) + log x) : 
  deriv^[2] f e = -1 / e := by
  sorry

end function_second_derivative_at_e_l160_16038


namespace max_available_is_two_l160_16057

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the colleagues
inductive Colleague
| Alice
| Bob
| Charlie
| Diana

-- Define a function that represents the availability of a colleague on a given day
def isAvailable (c : Colleague) (d : Day) : Bool :=
  match c, d with
  | Colleague.Alice, Day.Monday => false
  | Colleague.Alice, Day.Tuesday => true
  | Colleague.Alice, Day.Wednesday => false
  | Colleague.Alice, Day.Thursday => true
  | Colleague.Alice, Day.Friday => false
  | Colleague.Bob, Day.Monday => true
  | Colleague.Bob, Day.Tuesday => false
  | Colleague.Bob, Day.Wednesday => true
  | Colleague.Bob, Day.Thursday => false
  | Colleague.Bob, Day.Friday => true
  | Colleague.Charlie, Day.Monday => false
  | Colleague.Charlie, Day.Tuesday => false
  | Colleague.Charlie, Day.Wednesday => true
  | Colleague.Charlie, Day.Thursday => true
  | Colleague.Charlie, Day.Friday => false
  | Colleague.Diana, Day.Monday => true
  | Colleague.Diana, Day.Tuesday => true
  | Colleague.Diana, Day.Wednesday => false
  | Colleague.Diana, Day.Thursday => false
  | Colleague.Diana, Day.Friday => true

-- Define a function that counts the number of available colleagues on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun c => isAvailable c d) [Colleague.Alice, Colleague.Bob, Colleague.Charlie, Colleague.Diana]).length

-- Theorem: The maximum number of available colleagues on any day is 2
theorem max_available_is_two :
  (List.map countAvailable [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]).maximum? = some 2 := by
  sorry


end max_available_is_two_l160_16057


namespace company_picnic_attendance_l160_16006

theorem company_picnic_attendance 
  (total_employees : ℕ) 
  (men_attendance_rate : ℚ) 
  (women_attendance_rate : ℚ) 
  (men_percentage : ℚ) 
  (h1 : men_attendance_rate = 1/5) 
  (h2 : women_attendance_rate = 2/5) 
  (h3 : men_percentage = 7/20) :
  let women_percentage := 1 - men_percentage
  let men_attended := (men_attendance_rate * men_percentage * total_employees).floor
  let women_attended := (women_attendance_rate * women_percentage * total_employees).floor
  let total_attended := men_attended + women_attended
  (total_attended : ℚ) / total_employees = 33/100 :=
sorry

end company_picnic_attendance_l160_16006


namespace inequality_proof_l160_16072

theorem inequality_proof (a b x₁ x₂ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) :
  (a * x₁ + b * x₂) * (a * x₂ + b * x₁) ≥ x₁ * x₂ := by
  sorry

end inequality_proof_l160_16072


namespace probability_one_of_each_color_is_9_28_l160_16008

def total_balls : ℕ := 9
def balls_per_color : ℕ := 3
def selected_balls : ℕ := 3

def probability_one_of_each_color : ℚ :=
  (balls_per_color ^ 3 : ℚ) / (total_balls.choose selected_balls)

theorem probability_one_of_each_color_is_9_28 :
  probability_one_of_each_color = 9 / 28 := by
  sorry

end probability_one_of_each_color_is_9_28_l160_16008


namespace smallest_integer_with_given_remainders_l160_16012

theorem smallest_integer_with_given_remainders :
  ∃ (n : ℕ), n > 0 ∧
    n % 5 = 4 ∧
    n % 7 = 5 ∧
    n % 11 = 9 ∧
    n % 13 = 11 ∧
    (∀ m : ℕ, m > 0 ∧
      m % 5 = 4 ∧
      m % 7 = 5 ∧
      m % 11 = 9 ∧
      m % 13 = 11 → m ≥ n) ∧
    n = 999 :=
by sorry

end smallest_integer_with_given_remainders_l160_16012


namespace polynomial_division_degree_l160_16001

theorem polynomial_division_degree (f q d r : Polynomial ℝ) : 
  Polynomial.degree f = 17 →
  Polynomial.degree q = 10 →
  r = 5 * X^4 - 3 * X^3 + 2 * X^2 - X + 15 →
  f = d * q + r →
  Polynomial.degree d = 7 := by
sorry

end polynomial_division_degree_l160_16001


namespace part1_part2_l160_16097

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Part 1
theorem part1 (k : ℝ) : 
  (∀ x, f k x < 0 ↔ 2 < x ∧ x < 3) → k = 2/5 := by sorry

-- Part 2
theorem part2 (k : ℝ) :
  k > 0 ∧ (∀ x, 2 < x ∧ x < 3 → f k x < 0) → 0 < k ∧ k ≤ 2/5 := by sorry

end part1_part2_l160_16097


namespace unique_albums_count_l160_16073

/-- Represents the album collections of Andrew, John, and Samantha -/
structure AlbumCollections where
  andrew_total : ℕ
  john_total : ℕ
  samantha_total : ℕ
  andrew_john_shared : ℕ
  andrew_samantha_shared : ℕ
  john_samantha_shared : ℕ

/-- Calculates the number of unique albums given the album collections -/
def uniqueAlbums (c : AlbumCollections) : ℕ :=
  (c.andrew_total - c.andrew_john_shared - c.andrew_samantha_shared) +
  (c.john_total - c.andrew_john_shared - c.john_samantha_shared) +
  (c.samantha_total - c.andrew_samantha_shared - c.john_samantha_shared)

/-- Theorem stating that the number of unique albums is 26 for the given collection -/
theorem unique_albums_count :
  let c : AlbumCollections := {
    andrew_total := 23,
    john_total := 20,
    samantha_total := 15,
    andrew_john_shared := 12,
    andrew_samantha_shared := 3,
    john_samantha_shared := 5
  }
  uniqueAlbums c = 26 := by
  sorry

end unique_albums_count_l160_16073


namespace ratio_of_65_to_13_l160_16074

theorem ratio_of_65_to_13 (certain_number : ℚ) (h : certain_number = 65) : 
  certain_number / 13 = 5 := by
sorry

end ratio_of_65_to_13_l160_16074


namespace profit_and_marginal_profit_max_not_equal_l160_16022

def marginal_function (f : ℕ → ℝ) : ℕ → ℝ := λ x => f (x + 1) - f x

def revenue (a : ℝ) : ℕ → ℝ := λ x => 3000 * x + a * x^2

def cost (k : ℝ) : ℕ → ℝ := λ x => k * x + 4000

def profit (a k : ℝ) : ℕ → ℝ := λ x => revenue a x - cost k x

def marginal_profit (a k : ℝ) : ℕ → ℝ := marginal_function (profit a k)

theorem profit_and_marginal_profit_max_not_equal :
  ∃ (a k : ℝ),
    (∀ x : ℕ, 0 < x ∧ x ≤ 100 → profit a k x ≤ 74120) ∧
    (∃ x : ℕ, 0 < x ∧ x ≤ 100 ∧ profit a k x = 74120) ∧
    (∀ x : ℕ, 0 < x ∧ x ≤ 100 → marginal_profit a k x ≤ 2440) ∧
    (∃ x : ℕ, 0 < x ∧ x ≤ 100 ∧ marginal_profit a k x = 2440) ∧
    (cost k 10 = 9000) ∧
    (profit a k 10 = 19000) ∧
    74120 ≠ 2440 :=
by
  sorry

end profit_and_marginal_profit_max_not_equal_l160_16022


namespace xyz_inequality_l160_16005

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x*y + y*z + z*x = 1) : 
  x*y*z*(x+y)*(y+z)*(z+x) ≥ (1-x^2)*(1-y^2)*(1-z^2) := by
  sorry

end xyz_inequality_l160_16005


namespace circle_line_intersection_l160_16079

/-- Given a circle and a line, if their intersection forms a chord of length 4, then the parameter 'a' in the circle equation equals -4 -/
theorem circle_line_intersection (x y : ℝ) (a : ℝ) : 
  (x^2 + y^2 + 2*x - 2*y + a = 0) →  -- Circle equation
  (x + y + 2 = 0) →                  -- Line equation
  (∃ p q : ℝ × ℝ, p ≠ q ∧            -- Existence of two distinct intersection points
    (p.1^2 + p.2^2 + 2*p.1 - 2*p.2 + a = 0) ∧
    (p.1 + p.2 + 2 = 0) ∧
    (q.1^2 + q.2^2 + 2*q.1 - 2*q.2 + a = 0) ∧
    (q.1 + q.2 + 2 = 0) ∧
    ((p.1 - q.1)^2 + (p.2 - q.2)^2 = 16)) → -- Chord length is 4
  a = -4 := by sorry

end circle_line_intersection_l160_16079


namespace brandon_gecko_sales_l160_16094

/-- The number of geckos Brandon sold in the first half of last year -/
def first_half_last_year : ℕ := 46

/-- The number of geckos Brandon sold in the second half of last year -/
def second_half_last_year : ℕ := 55

/-- The number of geckos Brandon sold in the first half two years ago -/
def first_half_two_years_ago : ℕ := 3 * first_half_last_year

/-- The number of geckos Brandon sold in the second half two years ago -/
def second_half_two_years_ago : ℕ := 117

/-- The total number of geckos Brandon sold in the last two years -/
def total_geckos : ℕ := first_half_last_year + second_half_last_year + first_half_two_years_ago + second_half_two_years_ago

theorem brandon_gecko_sales : total_geckos = 356 := by
  sorry

end brandon_gecko_sales_l160_16094


namespace polynomial_existence_l160_16078

theorem polynomial_existence : 
  ∃ (p : ℝ → ℝ), 
    (∃ (a b c : ℝ), ∀ x, p x = a * x^2 + b * x + c) ∧ 
    p 0 = 100 ∧ 
    p 1 = 90 ∧ 
    p 2 = 70 ∧ 
    p 3 = 40 ∧ 
    p 4 = 0 :=
by sorry

end polynomial_existence_l160_16078


namespace sqrt_two_minus_a_l160_16011

theorem sqrt_two_minus_a (a : ℝ) : a = -2 → Real.sqrt (2 - a) = 2 := by
  sorry

end sqrt_two_minus_a_l160_16011


namespace correct_num_non_officers_l160_16000

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := 495

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Represents the average salary of all employees in Rs/month -/
def avg_salary_all : ℚ := 120

/-- Represents the average salary of officers in Rs/month -/
def avg_salary_officers : ℚ := 450

/-- Represents the average salary of non-officers in Rs/month -/
def avg_salary_non_officers : ℚ := 110

/-- Theorem stating that the number of non-officers is correct given the conditions -/
theorem correct_num_non_officers :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / 
  (num_officers + num_non_officers : ℚ) = avg_salary_all := by
  sorry


end correct_num_non_officers_l160_16000


namespace middle_school_students_l160_16015

theorem middle_school_students (band_percentage : ℚ) (band_students : ℕ) 
  (h1 : band_percentage = 1/5) 
  (h2 : band_students = 168) : 
  ∃ total_students : ℕ, 
    (band_percentage * total_students = band_students) ∧ 
    total_students = 840 := by
  sorry

end middle_school_students_l160_16015


namespace triangle_area_l160_16043

theorem triangle_area (a b c : ℝ) (ha : a = 10) (hb : b = 24) (hc : c = 26) :
  (1 / 2) * a * b = 120 :=
by sorry

end triangle_area_l160_16043


namespace smallest_integer_greater_than_neg_seven_thirds_l160_16046

theorem smallest_integer_greater_than_neg_seven_thirds :
  Int.ceil (-7/3 : ℚ) = -2 :=
sorry

end smallest_integer_greater_than_neg_seven_thirds_l160_16046


namespace f_equals_g_l160_16042

/-- Two functions are considered the same if they have the same domain, codomain, and function value for all inputs. -/
def same_function (α β : Type) (f g : α → β) : Prop :=
  ∀ x, f x = g x

/-- Function f defined as f(x) = x - 1 -/
def f : ℝ → ℝ := λ x ↦ x - 1

/-- Function g defined as g(t) = t - 1 -/
def g : ℝ → ℝ := λ t ↦ t - 1

/-- Theorem stating that f and g are the same function -/
theorem f_equals_g : same_function ℝ ℝ f g := by
  sorry


end f_equals_g_l160_16042


namespace prob_angle_AQB_obtuse_l160_16007

/-- Pentagon ABCDE with vertices A(0,3), B(5,0), C(2π,0), D(2π,5), E(0,5) -/
def pentagon : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 ≤ 2*Real.pi ∧ p.2 ≥ 0 ∧ p.2 ≤ 5 ∧
       (p.2 ≥ 3 - 3/5 * p.1 ∨ p.1 ≥ 2*Real.pi)}

/-- Point A -/
def A : ℝ × ℝ := (0, 3)

/-- Point B -/
def B : ℝ × ℝ := (5, 0)

/-- Random point Q in the pentagon -/
def Q : ℝ × ℝ := sorry

/-- Angle AQB -/
def angle_AQB : ℝ := sorry

/-- Probability measure on the pentagon -/
def prob : MeasureTheory.Measure (ℝ × ℝ) := sorry

/-- The probability that angle AQB is obtuse -/
theorem prob_angle_AQB_obtuse :
  prob {q ∈ pentagon | angle_AQB > Real.pi/2} / prob pentagon = 17/128 := by sorry

end prob_angle_AQB_obtuse_l160_16007


namespace specific_det_value_det_equation_solution_l160_16061

-- Define the determinant of order 2
def det2 (a b c d : ℤ) : ℤ := a * d - b * c

-- Theorem 1: The value of the specific determinant is 1
theorem specific_det_value : det2 2022 2023 2021 2022 = 1 := by sorry

-- Theorem 2: If the given determinant equals 32, then m = 4
theorem det_equation_solution (m : ℤ) : 
  det2 (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 := by sorry

end specific_det_value_det_equation_solution_l160_16061


namespace triangle_inequality_l160_16064

theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end triangle_inequality_l160_16064


namespace x_squared_mod_25_l160_16029

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 15 [ZMOD 25]) (h2 : 4 * x ≡ 12 [ZMOD 25]) :
  x^2 ≡ 9 [ZMOD 25] := by
  sorry

end x_squared_mod_25_l160_16029


namespace medical_team_selection_l160_16037

theorem medical_team_selection (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 6) (h2 : female_doctors = 5) :
  (Nat.choose male_doctors 2) * (Nat.choose female_doctors 1) = 75 := by
  sorry

end medical_team_selection_l160_16037


namespace sin_double_sum_eq_four_sin_product_l160_16051

/-- Given that α + β + γ = π, prove that sin 2α + sin 2β + sin 2γ = 4 sin α sin β sin γ -/
theorem sin_double_sum_eq_four_sin_product (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 4 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end sin_double_sum_eq_four_sin_product_l160_16051


namespace project_work_time_difference_l160_16021

theorem project_work_time_difference (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 90 →
  2 * t1 = 3 * t2 →
  3 * t2 = 4 * t3 →
  t3 - t1 = 20 := by
sorry

end project_work_time_difference_l160_16021


namespace journey_distance_l160_16031

/-- Proves that a journey with given conditions has a total distance of 126 km -/
theorem journey_distance (total_time : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_time = 12 ∧
  speed1 = 21 ∧
  speed2 = 14 ∧
  speed3 = 6 →
  (1 / speed1 + 1 / speed2 + 1 / speed3) * (total_time / 3) = 126 := by
  sorry


end journey_distance_l160_16031


namespace problem_solution_l160_16041

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 21 * x * y = x^3 + 3 * x^2 * y^2) : x = 7 := by
  sorry

end problem_solution_l160_16041


namespace f_equals_three_l160_16056

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 1
  else if x < 2 then x^2
  else 2*x

theorem f_equals_three (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end f_equals_three_l160_16056


namespace reach_probability_is_15_1024_l160_16002

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a step direction --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of a single step in any direction --/
def stepProbability : Rat := 1 / 4

/-- The starting point --/
def start : Point := ⟨0, 0⟩

/-- The target point --/
def target : Point := ⟨3, 1⟩

/-- The maximum number of steps allowed --/
def maxSteps : Nat := 8

/-- Calculates the probability of reaching the target from the start in at most maxSteps --/
def reachProbability (start : Point) (target : Point) (maxSteps : Nat) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem reach_probability_is_15_1024 : 
  reachProbability start target maxSteps = 15 / 1024 := by sorry

end reach_probability_is_15_1024_l160_16002


namespace square_area_from_rectangle_l160_16085

theorem square_area_from_rectangle (r l b : ℝ) : 
  l = r / 4 →  -- length of rectangle is 1/4 of circle radius
  l * b = 35 → -- area of rectangle is 35
  b = 5 →      -- breadth of rectangle is 5
  r^2 = 784 := by sorry

end square_area_from_rectangle_l160_16085


namespace other_bases_with_square_property_existence_of_other_bases_l160_16036

theorem other_bases_with_square_property (B : ℕ) (V : ℕ) : Prop :=
  2 < B ∧ 1 < V ∧ V < B ∧ V * V % B = V % B

theorem existence_of_other_bases :
  ∃ B V, B ≠ 50 ∧ other_bases_with_square_property B V := by
  sorry

end other_bases_with_square_property_existence_of_other_bases_l160_16036


namespace range_of_a_l160_16093

theorem range_of_a (a : ℝ) : 
  (∀ t ∈ Set.Ioo 0 2, t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2) → 
  a ∈ Set.Icc (2/13) 1 := by
sorry

end range_of_a_l160_16093


namespace men_in_second_scenario_l160_16028

/-- Calculates the number of men working in the second scenario given the conditions --/
theorem men_in_second_scenario 
  (hours_per_day_first : ℕ) 
  (hours_per_day_second : ℕ)
  (men_first : ℕ)
  (earnings_first : ℚ)
  (earnings_second : ℚ)
  (days_per_week : ℕ) :
  hours_per_day_first = 10 →
  hours_per_day_second = 6 →
  men_first = 4 →
  earnings_first = 1400 →
  earnings_second = 1890.0000000000002 →
  days_per_week = 7 →
  ∃ (men_second : ℕ), men_second = 9 ∧ 
    (men_second * hours_per_day_second * days_per_week : ℚ) * 
    (earnings_first / (men_first * hours_per_day_first * days_per_week : ℚ)) = 
    earnings_second :=
by sorry

end men_in_second_scenario_l160_16028


namespace baseball_card_pages_l160_16027

theorem baseball_card_pages (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 8)
  (h3 : old_cards = 10) :
  (new_cards + old_cards) / cards_per_page = 6 := by
  sorry

end baseball_card_pages_l160_16027


namespace max_min_sum_implies_a_value_l160_16026

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

theorem max_min_sum_implies_a_value (a : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 2 3, f a x ≤ max) ∧
    (∃ y ∈ Set.Icc 2 3, f a y = max) ∧
    (∀ x ∈ Set.Icc 2 3, min ≤ f a x) ∧
    (∃ y ∈ Set.Icc 2 3, f a y = min) ∧
    max + min = 5) →
  a = 1 := by
sorry

end max_min_sum_implies_a_value_l160_16026


namespace triangle_angle_problem_l160_16032

theorem triangle_angle_problem (A B C : Real) (BC AC : Real) :
  BC = Real.sqrt 3 →
  AC = Real.sqrt 2 →
  A = π / 3 →
  B = π / 4 :=
by
  sorry

end triangle_angle_problem_l160_16032


namespace similar_triangles_height_l160_16039

/-- Given two similar triangles with an area ratio of 1:9 and the smaller triangle
    having a height of 5 cm, prove that the corresponding height of the larger triangle
    is 15 cm. -/
theorem similar_triangles_height (small_height large_height : ℝ) :
  small_height = 5 →
  (9 : ℝ) * small_height^2 = large_height^2 →
  large_height = 15 :=
by
  sorry

end similar_triangles_height_l160_16039


namespace least_subtraction_for_divisibility_problem_solution_l160_16065

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 := by
  sorry

theorem problem_solution : 
  let n := 568219
  let d := 89
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 ∧ k = 45 := by
  sorry

end least_subtraction_for_divisibility_problem_solution_l160_16065


namespace gala_luncheon_croissant_cost_l160_16050

/-- Calculates the cost of croissants for a gala luncheon --/
theorem gala_luncheon_croissant_cost
  (people : ℕ)
  (sandwiches_per_person : ℕ)
  (croissants_per_set : ℕ)
  (cost_per_set : ℚ)
  (h1 : people = 24)
  (h2 : sandwiches_per_person = 2)
  (h3 : croissants_per_set = 12)
  (h4 : cost_per_set = 8) :
  (people * sandwiches_per_person / croissants_per_set : ℚ) * cost_per_set = 32 := by
  sorry

#check gala_luncheon_croissant_cost

end gala_luncheon_croissant_cost_l160_16050


namespace tennis_ball_count_l160_16076

theorem tennis_ball_count : 
  ∀ (lily frodo brian sam : ℕ),
  lily = 12 →
  frodo = lily + 15 →
  brian = 3 * frodo →
  sam = frodo + lily - 5 →
  sam = 34 := by
sorry

end tennis_ball_count_l160_16076


namespace max_a_for_zero_points_l160_16059

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / x^2 - x - a/x + 2*Real.exp 1

theorem max_a_for_zero_points :
  (∃ a : ℝ, ∃ x : ℝ, x > 0 ∧ f a x = 0) →
  (∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1) ∧
  (∃ x : ℝ, x > 0 ∧ f (Real.exp 2 + 1 / Real.exp 1) x = 0) := by
  sorry

end max_a_for_zero_points_l160_16059


namespace minimizes_f_l160_16071

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := 3 * (x - a)^2 + 3 * (x - b)^2

/-- The statement that (a+b)/2 minimizes f(x) -/
theorem minimizes_f (a b : ℝ) :
  ∀ x : ℝ, f a b ((a + b) / 2) ≤ f a b x :=
sorry

end minimizes_f_l160_16071


namespace only_points_in_circle_form_set_l160_16084

-- Define a type for the objects in question
inductive Object
| MaleStudents
| DifficultProblems
| OutgoingGirls
| PointsInCircle

-- Define a predicate for whether an object can form a set
def CanFormSet (obj : Object) : Prop :=
  match obj with
  | Object.PointsInCircle => True
  | _ => False

-- State the theorem
theorem only_points_in_circle_form_set :
  ∀ (obj : Object), CanFormSet obj ↔ obj = Object.PointsInCircle :=
by sorry

end only_points_in_circle_form_set_l160_16084


namespace unique_solution_for_system_l160_16009

theorem unique_solution_for_system (x y : ℝ) :
  (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x - 2) + (y - 2)) →
  x = 5 ∧ y = 2 := by
  sorry

end unique_solution_for_system_l160_16009


namespace egg_cost_l160_16088

/-- The cost of breakfast items and breakfasts for Dale and Andrew -/
structure BreakfastCosts where
  toast : ℝ  -- Cost of a slice of toast
  egg : ℝ    -- Cost of an egg
  dale : ℝ   -- Cost of Dale's breakfast
  andrew : ℝ  -- Cost of Andrew's breakfast
  total : ℝ  -- Total cost of both breakfasts

/-- Theorem stating the cost of an egg given the breakfast costs -/
theorem egg_cost (b : BreakfastCosts) 
  (h_toast : b.toast = 1)
  (h_dale : b.dale = 2 * b.toast + 2 * b.egg)
  (h_andrew : b.andrew = b.toast + 2 * b.egg)
  (h_total : b.total = b.dale + b.andrew)
  (h_total_value : b.total = 15) :
  b.egg = 3 := by
  sorry

end egg_cost_l160_16088


namespace implication_not_equivalence_l160_16075

theorem implication_not_equivalence :
  ∃ (a : ℝ), (∀ (x : ℝ), (abs (5 * x - 1) > a) → (x^2 - (3/2) * x + 1/2 > 0)) ∧
             (∃ (y : ℝ), (y^2 - (3/2) * y + 1/2 > 0) ∧ (abs (5 * y - 1) ≤ a)) :=
by
  -- The proof goes here
  sorry

end implication_not_equivalence_l160_16075


namespace profit_starts_fourth_year_option_two_more_profitable_l160_16014

def initial_investment : ℕ := 81
def annual_rental_income : ℕ := 30
def first_year_renovation : ℕ := 1
def yearly_renovation_increase : ℕ := 2

def total_renovation_cost (n : ℕ) : ℕ := n^2

def total_income (n : ℕ) : ℕ := annual_rental_income * n

def profit (n : ℕ) : ℤ := (total_income n : ℤ) - (initial_investment : ℤ) - (total_renovation_cost n : ℤ)

def average_profit (n : ℕ) : ℚ := (profit n : ℚ) / n

theorem profit_starts_fourth_year :
  ∀ n : ℕ, n < 4 → profit n ≤ 0 ∧ profit 4 > 0 := by sorry

theorem option_two_more_profitable :
  profit 15 + 10 < profit 9 + 50 := by sorry

#eval profit 4
#eval profit 15 + 10
#eval profit 9 + 50

end profit_starts_fourth_year_option_two_more_profitable_l160_16014


namespace square_side_length_l160_16089

-- Define the rectangle's dimensions
def rectangle_length : ℝ := 7
def rectangle_width : ℝ := 5

-- Define the theorem
theorem square_side_length : 
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side := rectangle_perimeter / 4
  square_side = 6 := by
  sorry

end square_side_length_l160_16089


namespace inequality_and_equality_condition_l160_16048

theorem inequality_and_equality_condition (p q : ℝ) (hp : 0 < p) (hq : p < q)
  (α β γ δ ε : ℝ) (hα : p ≤ α ∧ α ≤ q) (hβ : p ≤ β ∧ β ≤ q)
  (hγ : p ≤ γ ∧ γ ≤ q) (hδ : p ≤ δ ∧ δ ≤ q) (hε : p ≤ ε ∧ ε ≤ q) :
  (α + β + γ + δ + ε) * (1/α + 1/β + 1/γ + 1/δ + 1/ε) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ∧
  ((α + β + γ + δ + ε) * (1/α + 1/β + 1/γ + 1/δ + 1/ε) = 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ↔
   ((α = p ∧ β = p ∧ γ = q ∧ δ = q ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = p ∧ δ = q ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = q ∧ δ = p ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = q ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = p ∧ γ = p ∧ δ = q ∧ ε = q) ∨
    (α = q ∧ β = p ∧ γ = q ∧ δ = p ∧ ε = q) ∨
    (α = q ∧ β = p ∧ γ = q ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = q ∧ γ = p ∧ δ = p ∧ ε = q) ∨
    (α = q ∧ β = q ∧ γ = p ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = q ∧ γ = q ∧ δ = p ∧ ε = p))) :=
sorry

end inequality_and_equality_condition_l160_16048


namespace largest_absolute_value_l160_16099

theorem largest_absolute_value : let S : Finset Int := {2, 3, -3, -4}
  ∃ x ∈ S, ∀ y ∈ S, |y| ≤ |x| ∧ x = -4 := by
  sorry

end largest_absolute_value_l160_16099
