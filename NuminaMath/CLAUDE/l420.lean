import Mathlib

namespace NUMINAMATH_CALUDE_square_area_five_equal_rectangles_l420_42055

/-- A square divided into five rectangles of equal area, where one rectangle has a width of 5, has a total area of 400. -/
theorem square_area_five_equal_rectangles (s : ℝ) (w : ℝ) : 
  s > 0 ∧ w > 0 ∧ w = 5 ∧ 
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  a = b ∧ b = c ∧ c = d ∧ d = e ∧
  s * s = a + b + c + d + e ∧
  w * (s - w) = a →
  s * s = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_five_equal_rectangles_l420_42055


namespace NUMINAMATH_CALUDE_reflect_h_twice_l420_42005

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_y_eq_x_minus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let q := (p.1, p.2 + 2)  -- Translate up by 2
  let r := (q.2, q.1)      -- Reflect across y = x
  (r.1, r.2 - 2)           -- Translate down by 2

theorem reflect_h_twice (h : ℝ × ℝ) :
  h = (5, 3) →
  reflect_y_eq_x_minus_2 (reflect_x h) = (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_h_twice_l420_42005


namespace NUMINAMATH_CALUDE_euro_equation_solution_l420_42046

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem euro_equation_solution :
  ∀ y : ℝ, euro y (euro 7 5) = 560 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_euro_equation_solution_l420_42046


namespace NUMINAMATH_CALUDE_two_students_choose_A_l420_42058

/-- The number of ways to choose exactly two students from four to take course A -/
def waysToChooseTwoForA : ℕ := 24

/-- The number of students -/
def numStudents : ℕ := 4

/-- The number of courses -/
def numCourses : ℕ := 3

theorem two_students_choose_A :
  waysToChooseTwoForA = (numStudents.choose 2) * (2^(numStudents - 2)) :=
sorry

end NUMINAMATH_CALUDE_two_students_choose_A_l420_42058


namespace NUMINAMATH_CALUDE_total_distance_is_75_miles_l420_42049

/-- Calculates the total distance traveled given initial speed and time, where the second part of the journey is twice as long at twice the speed. -/
def totalDistance (initialSpeed : ℝ) (initialTime : ℝ) : ℝ :=
  let distance1 := initialSpeed * initialTime
  let distance2 := (2 * initialSpeed) * (2 * initialTime)
  distance1 + distance2

/-- Proves that given an initial speed of 30 mph and an initial time of 0.5 hours, the total distance traveled is 75 miles. -/
theorem total_distance_is_75_miles :
  totalDistance 30 0.5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_75_miles_l420_42049


namespace NUMINAMATH_CALUDE_polynomial_negative_l420_42036

theorem polynomial_negative (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < a) : 
  (a - x)^6 - 3*a*(a - x)^5 + (5/2)*a^2*(a - x)^4 - (1/2)*a^4*(a - x)^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_negative_l420_42036


namespace NUMINAMATH_CALUDE_daniel_video_game_collection_l420_42035

/-- The number of video games Daniel bought for $12 each -/
def games_at_12 : ℕ := 80

/-- The price of the first group of games -/
def price_1 : ℕ := 12

/-- The price of the second group of games -/
def price_2 : ℕ := 7

/-- The price of the third group of games -/
def price_3 : ℕ := 3

/-- The total amount Daniel spent on all games -/
def total_spent : ℕ := 2290

/-- Theorem stating the total number of video games in Daniel's collection -/
theorem daniel_video_game_collection :
  ∃ (games_at_7 games_at_3 : ℕ),
    games_at_7 = games_at_3 ∧
    games_at_12 * price_1 + games_at_7 * price_2 + games_at_3 * price_3 = total_spent ∧
    games_at_12 + games_at_7 + games_at_3 = 346 :=
by sorry

end NUMINAMATH_CALUDE_daniel_video_game_collection_l420_42035


namespace NUMINAMATH_CALUDE_balance_forces_l420_42042

/-- A force is represented by a pair of real numbers -/
def Force : Type := ℝ × ℝ

/-- Addition of forces -/
def add_forces (f1 f2 : Force) : Force :=
  (f1.1 + f2.1, f1.2 + f2.2)

/-- The zero force -/
def zero_force : Force := (0, 0)

/-- Two forces are balanced by a third force if their sum is the zero force -/
def balances (f1 f2 f3 : Force) : Prop :=
  add_forces (add_forces f1 f2) f3 = zero_force

theorem balance_forces :
  let f1 : Force := (1, 1)
  let f2 : Force := (2, 3)
  let f3 : Force := (-3, -4)
  balances f1 f2 f3 := by sorry

end NUMINAMATH_CALUDE_balance_forces_l420_42042


namespace NUMINAMATH_CALUDE_sum_of_possible_S_values_l420_42016

theorem sum_of_possible_S_values : ∃ (a b c x y z : ℕ+) (S : ℕ),
  (a^2 - 2 : ℚ) / x = (b^2 - 37 : ℚ) / y ∧
  (b^2 - 37 : ℚ) / y = (c^2 - 41 : ℚ) / z ∧
  (c^2 - 41 : ℚ) / z = (a + b + c : ℚ) ∧
  S = a + b + c + x + y + z ∧
  (∀ (a' b' c' x' y' z' : ℕ+) (S' : ℕ),
    ((a'^2 - 2 : ℚ) / x' = (b'^2 - 37 : ℚ) / y' ∧
     (b'^2 - 37 : ℚ) / y' = (c'^2 - 41 : ℚ) / z' ∧
     (c'^2 - 41 : ℚ) / z' = (a' + b' + c' : ℚ) ∧
     S' = a' + b' + c' + x' + y' + z') →
    S = 98 ∨ S = 211) ∧
  (∃ (a₁ b₁ c₁ x₁ y₁ z₁ : ℕ+) (S₁ : ℕ),
    (a₁^2 - 2 : ℚ) / x₁ = (b₁^2 - 37 : ℚ) / y₁ ∧
    (b₁^2 - 37 : ℚ) / y₁ = (c₁^2 - 41 : ℚ) / z₁ ∧
    (c₁^2 - 41 : ℚ) / z₁ = (a₁ + b₁ + c₁ : ℚ) ∧
    S₁ = a₁ + b₁ + c₁ + x₁ + y₁ + z₁ ∧
    S₁ = 98) ∧
  (∃ (a₂ b₂ c₂ x₂ y₂ z₂ : ℕ+) (S₂ : ℕ),
    (a₂^2 - 2 : ℚ) / x₂ = (b₂^2 - 37 : ℚ) / y₂ ∧
    (b₂^2 - 37 : ℚ) / y₂ = (c₂^2 - 41 : ℚ) / z₂ ∧
    (c₂^2 - 41 : ℚ) / z₂ = (a₂ + b₂ + c₂ : ℚ) ∧
    S₂ = a₂ + b₂ + c₂ + x₂ + y₂ + z₂ ∧
    S₂ = 211) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_S_values_l420_42016


namespace NUMINAMATH_CALUDE_selenas_remaining_money_is_38_l420_42096

/-- Calculates the remaining money for Selena after her meal -/
def selenas_remaining_money (tip : ℚ) (steak_price : ℚ) (steak_count : ℕ) 
  (burger_price : ℚ) (burger_count : ℕ) (icecream_price : ℚ) (icecream_count : ℕ) : ℚ :=
  tip - (steak_price * steak_count + burger_price * burger_count + icecream_price * icecream_count)

/-- Theorem stating that Selena will be left with $38 after her meal -/
theorem selenas_remaining_money_is_38 :
  selenas_remaining_money 99 24 2 3.5 2 2 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_selenas_remaining_money_is_38_l420_42096


namespace NUMINAMATH_CALUDE_non_red_percentage_is_27_percent_l420_42001

/-- Represents the car population data for a city --/
structure CarPopulation where
  total : ℕ
  honda : ℕ
  toyota : ℕ
  nissan : ℕ
  honda_red_ratio : ℚ
  toyota_red_ratio : ℚ
  nissan_red_ratio : ℚ

/-- Calculate the percentage of non-red cars in the given car population --/
def non_red_percentage (pop : CarPopulation) : ℚ :=
  let total_red := pop.honda * pop.honda_red_ratio +
                   pop.toyota * pop.toyota_red_ratio +
                   pop.nissan * pop.nissan_red_ratio
  let total_non_red := pop.total - total_red
  (total_non_red / pop.total) * 100

/-- The theorem stating that the percentage of non-red cars is 27% --/
theorem non_red_percentage_is_27_percent (pop : CarPopulation)
  (h1 : pop.total = 30000)
  (h2 : pop.honda = 12000)
  (h3 : pop.toyota = 10000)
  (h4 : pop.nissan = 8000)
  (h5 : pop.honda_red_ratio = 80 / 100)
  (h6 : pop.toyota_red_ratio = 75 / 100)
  (h7 : pop.nissan_red_ratio = 60 / 100) :
  non_red_percentage pop = 27 := by
  sorry

end NUMINAMATH_CALUDE_non_red_percentage_is_27_percent_l420_42001


namespace NUMINAMATH_CALUDE_house_resale_price_l420_42009

theorem house_resale_price (initial_value : ℝ) (loss_percent : ℝ) (interest_rate : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  interest_rate = 0.05 ∧ 
  gain_percent = 0.2 → 
  initial_value * (1 - loss_percent) * (1 + interest_rate) * (1 + gain_percent) = 12852 :=
by sorry

end NUMINAMATH_CALUDE_house_resale_price_l420_42009


namespace NUMINAMATH_CALUDE_birds_reduced_correct_l420_42067

/-- The number of birds reduced on the third day, given the initial number of birds,
    the doubling on the second day, and the total number of birds seen in three days. -/
def birds_reduced (initial : ℕ) (total : ℕ) : ℕ :=
  initial * 2 - (total - (initial + initial * 2))

/-- Theorem stating that the number of birds reduced on the third day is 200,
    given the conditions from the problem. -/
theorem birds_reduced_correct : birds_reduced 300 1300 = 200 := by
  sorry

end NUMINAMATH_CALUDE_birds_reduced_correct_l420_42067


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_four_l420_42022

theorem sum_of_fractions_geq_four (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a * d + b * c) / (b * d) + (b * c + a * d) / (a * c) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_four_l420_42022


namespace NUMINAMATH_CALUDE_cos_equality_implies_43_l420_42018

theorem cos_equality_implies_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_implies_43_l420_42018


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l420_42099

/-- The probability of selecting two non-defective pens from a box with defective pens -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 10) 
  (h2 : defective_pens = 2) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 28 / 45 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l420_42099


namespace NUMINAMATH_CALUDE_perpendicular_tangents_imply_m_value_l420_42066

/-- The original function F1 -/
def F1 (x : ℝ) : ℝ := x^2

/-- The translated function F2 -/
def F2 (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 - 1

/-- The derivative of F1 -/
def F1_derivative (x : ℝ) : ℝ := 2 * x

/-- The derivative of F2 -/
def F2_derivative (m : ℝ) (x : ℝ) : ℝ := 2 * (x - m)

theorem perpendicular_tangents_imply_m_value :
  ∀ m : ℝ, (F1_derivative 1 * F2_derivative m 1 = -1) → m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_imply_m_value_l420_42066


namespace NUMINAMATH_CALUDE_zero_variance_median_equals_mean_l420_42087

-- Define a sample as a finite multiset of real numbers
def Sample := Multiset ℝ

-- Define the variance of a sample
def variance (s : Sample) : ℝ := sorry

-- Define the median of a sample
def median (s : Sample) : ℝ := sorry

-- Define the mean of a sample
def mean (s : Sample) : ℝ := sorry

-- Theorem statement
theorem zero_variance_median_equals_mean (s : Sample) (a : ℝ) :
  variance s = 0 ∧ median s = a → mean s = a := by sorry

end NUMINAMATH_CALUDE_zero_variance_median_equals_mean_l420_42087


namespace NUMINAMATH_CALUDE_quadruple_solution_l420_42008

theorem quadruple_solution :
  ∀ a b c d : ℕ+,
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a + b = c * d ∧ a * b = c + d →
    (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
    (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
    (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
    (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
    (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
    (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
    (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
    (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solution_l420_42008


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l420_42053

theorem scientific_notation_equivalence :
  ∃ (a : ℝ) (n : ℤ), 
    27017800000000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ a < 10 ∧
    n = 13 ∧
    a = 2.70178 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l420_42053


namespace NUMINAMATH_CALUDE_solve_system_l420_42088

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 3 * q = 10) 
  (eq2 : 3 * p + 5 * q = 20) : 
  q = 35 / 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l420_42088


namespace NUMINAMATH_CALUDE_geometric_region_equivalence_l420_42079

theorem geometric_region_equivalence (x y : ℝ) :
  (x^2 + y^2 - 4 ≥ 0 ∧ x^2 - 1 ≥ 0 ∧ y^2 - 1 ≥ 0) ↔
  ((x^2 + y^2 ≥ 4) ∧ (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_region_equivalence_l420_42079


namespace NUMINAMATH_CALUDE_second_fund_interest_rate_l420_42044

/-- Proves that the interest rate of the second fund is 8.5% given the problem conditions --/
theorem second_fund_interest_rate : 
  ∀ (total_investment : ℝ) 
    (fund1_rate : ℝ) 
    (annual_interest : ℝ) 
    (fund1_investment : ℝ),
  total_investment = 50000 →
  fund1_rate = 8 →
  annual_interest = 4120 →
  fund1_investment = 26000 →
  ∃ (fund2_rate : ℝ),
    fund2_rate = 8.5 ∧
    annual_interest = (fund1_investment * fund1_rate / 100) + 
                      ((total_investment - fund1_investment) * fund2_rate / 100) :=
by
  sorry


end NUMINAMATH_CALUDE_second_fund_interest_rate_l420_42044


namespace NUMINAMATH_CALUDE_outside_county_attendance_l420_42030

/-- The number of kids from Lawrence county who went to camp -/
def lawrence_camp : ℕ := 34044

/-- The total number of kids who attended the camp -/
def total_camp : ℕ := 458988

/-- The number of kids from outside the county who attended the camp -/
def outside_county : ℕ := total_camp - lawrence_camp

theorem outside_county_attendance : outside_county = 424944 := by
  sorry

end NUMINAMATH_CALUDE_outside_county_attendance_l420_42030


namespace NUMINAMATH_CALUDE_xian_temp_difference_l420_42000

/-- Given the highest and lowest temperatures on a day, calculate the maximum temperature difference. -/
def max_temp_difference (highest lowest : ℝ) : ℝ :=
  highest - lowest

/-- Theorem: The maximum temperature difference on January 1, 2008 in Xi'an was 6°C. -/
theorem xian_temp_difference :
  let highest : ℝ := 3
  let lowest : ℝ := -3
  max_temp_difference highest lowest = 6 := by
  sorry

end NUMINAMATH_CALUDE_xian_temp_difference_l420_42000


namespace NUMINAMATH_CALUDE_first_chapter_has_13_pages_l420_42029

/-- Represents a book with chapters of increasing length -/
structure Book where
  num_chapters : ℕ
  total_pages : ℕ
  page_increase : ℕ

/-- Calculates the number of pages in the first chapter of a book -/
def first_chapter_pages (b : Book) : ℕ :=
  let x := (b.total_pages - (b.num_chapters * (b.num_chapters - 1) * b.page_increase / 2)) / b.num_chapters
  x

/-- Theorem stating that for a specific book, the first chapter has 13 pages -/
theorem first_chapter_has_13_pages :
  let b : Book := { num_chapters := 5, total_pages := 95, page_increase := 3 }
  first_chapter_pages b = 13 := by
  sorry


end NUMINAMATH_CALUDE_first_chapter_has_13_pages_l420_42029


namespace NUMINAMATH_CALUDE_seventh_root_of_negative_two_plus_fourth_root_of_negative_three_l420_42015

theorem seventh_root_of_negative_two_plus_fourth_root_of_negative_three : 
  ((-2 : ℝ) ^ 7) ^ (1/7) + ((-3 : ℝ) ^ 4) ^ (1/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_negative_two_plus_fourth_root_of_negative_three_l420_42015


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l420_42086

theorem prime_pairs_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → 
  (∃ k : ℤ, 30 * q - 1 = k * p) → 
  (∃ m : ℤ, 30 * p - 1 = m * q) → 
  ((p = 7 ∧ q = 11) ∨ (p = 11 ∧ q = 7) ∨ (p = 59 ∧ q = 61) ∨ (p = 61 ∧ q = 59)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l420_42086


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l420_42010

/-- The probability of picking two red balls from a bag containing 4 red, 4 blue, and 2 green balls -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 4 →
  blue_balls = 4 →
  green_balls = 2 →
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l420_42010


namespace NUMINAMATH_CALUDE_total_movie_time_is_172_l420_42091

/-- Represents a segment of movie watching, including the time spent watching and rewinding --/
structure MovieSegment where
  watchTime : ℕ
  rewindTime : ℕ

/-- Calculates the total time for a movie segment --/
def segmentTime (segment : MovieSegment) : ℕ :=
  segment.watchTime + segment.rewindTime

/-- The sequence of movie segments as described in the problem --/
def movieSegments : List MovieSegment := [
  ⟨30, 5⟩,
  ⟨20, 7⟩,
  ⟨10, 12⟩,
  ⟨15, 8⟩,
  ⟨25, 15⟩,
  ⟨15, 10⟩
]

/-- Theorem stating that the total time to watch the movie is 172 minutes --/
theorem total_movie_time_is_172 :
  (movieSegments.map segmentTime).sum = 172 := by
  sorry

end NUMINAMATH_CALUDE_total_movie_time_is_172_l420_42091


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_beta_l420_42043

theorem sin_2alpha_plus_beta (p α β : ℝ) : 
  (∀ x, x^2 - 4*p*x - 2 = 1 → x = Real.tan α ∨ x = Real.tan β) →
  Real.sin (2 * (α + β)) = (2 * p) / (p^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_beta_l420_42043


namespace NUMINAMATH_CALUDE_solve_for_y_l420_42077

theorem solve_for_y (x y : ℝ) (h : 3 * x + 5 * y = 10) : y = 2 - (3/5) * x := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l420_42077


namespace NUMINAMATH_CALUDE_complementary_event_is_at_most_one_wins_l420_42089

-- Define the sample space
inductive Outcome
  | BothWin
  | AWinsBLoses
  | ALosesBWins
  | BothLose

-- Define the event A
def eventA (outcome : Outcome) : Prop :=
  outcome = Outcome.BothWin

-- Define the complementary event
def complementaryEventA (outcome : Outcome) : Prop :=
  outcome = Outcome.AWinsBLoses ∨ outcome = Outcome.ALosesBWins ∨ outcome = Outcome.BothLose

-- Theorem statement
theorem complementary_event_is_at_most_one_wins :
  ∀ (outcome : Outcome), ¬(eventA outcome) ↔ complementaryEventA outcome :=
sorry

end NUMINAMATH_CALUDE_complementary_event_is_at_most_one_wins_l420_42089


namespace NUMINAMATH_CALUDE_min_socks_for_ten_pairs_five_colors_l420_42006

/-- The minimum number of socks needed to guarantee a certain number of pairs, given a number of colors -/
def min_socks (colors : ℕ) (pairs : ℕ) : ℕ := 2 * pairs + colors - 1

/-- Theorem stating that 24 socks are needed to guarantee 10 pairs with 5 colors -/
theorem min_socks_for_ten_pairs_five_colors :
  min_socks 5 10 = 24 := by sorry

end NUMINAMATH_CALUDE_min_socks_for_ten_pairs_five_colors_l420_42006


namespace NUMINAMATH_CALUDE_ellipse_and_circle_properties_l420_42062

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle D
def circle_D (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1/4

-- Theorem statement
theorem ellipse_and_circle_properties :
  -- The eccentricity of ellipse C is √3/2
  (∃ e : ℝ, e = Real.sqrt 3 / 2 ∧
    ∀ x y : ℝ, ellipse_C x y → 
      e = Real.sqrt (1 - (Real.sqrt (1 - x^2/4))^2) / 2) ∧
  -- Circle D lies entirely inside ellipse C
  (∀ x y : ℝ, circle_D x y → ellipse_C x y) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_properties_l420_42062


namespace NUMINAMATH_CALUDE_construction_valid_l420_42069

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

def isConvex (q : Quadrilateral) : Prop := sorry

def areNotConcyclic (q : Quadrilateral) : Prop := sorry

-- Define the construction steps
def rotateAroundPoint (p : Point) (center : Point) (angle : ℝ) : Point := sorry

def lineIntersection (p1 p2 q1 q2 : Point) : Point := sorry

def circumcircle (p1 p2 p3 : Point) : Set Point := sorry

-- Define the construction method
def constructCD (A B : Point) (angleBCD angleADC angleBCA angleACD : ℝ) : Quadrilateral := sorry

-- The main theorem
theorem construction_valid (A B : Point) (angleBCD angleADC angleBCA angleACD : ℝ) :
  let q := constructCD A B angleBCD angleADC angleBCA angleACD
  isConvex q ∧ areNotConcyclic q →
  ∃ (C D : Point), q = Quadrilateral.mk A B C D :=
sorry

end NUMINAMATH_CALUDE_construction_valid_l420_42069


namespace NUMINAMATH_CALUDE_polynomial_factorization_l420_42014

theorem polynomial_factorization (x : ℝ) : 
  x^12 - 3*x^9 + 3*x^3 + 1 = (x+1)^4 * (x^2-x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l420_42014


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l420_42050

theorem trigonometric_equation_solution :
  ∀ x : ℝ, ((7/2 * Real.cos (2*x) + 2) * abs (2 * Real.cos (2*x) - 1) = 
            Real.cos x * (Real.cos x + Real.cos (5*x))) ↔
           (∃ k : ℤ, x = π/6 + k*π/2 ∨ x = -π/6 + k*π/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l420_42050


namespace NUMINAMATH_CALUDE_square_sum_over_sum_ge_sqrt_product_l420_42081

theorem square_sum_over_sum_ge_sqrt_product (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y^2) / (x + y) ≥ Real.sqrt (x * y) := by sorry

end NUMINAMATH_CALUDE_square_sum_over_sum_ge_sqrt_product_l420_42081


namespace NUMINAMATH_CALUDE_average_equality_l420_42094

theorem average_equality (n : ℕ) (scores : Fin n → ℝ) :
  let original_avg : ℝ := (Finset.sum Finset.univ (λ i => scores i)) / n
  let new_sum : ℝ := (Finset.sum Finset.univ (λ i => scores i)) + 2 * original_avg
  new_sum / (n + 2) = original_avg := by
  sorry

end NUMINAMATH_CALUDE_average_equality_l420_42094


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_AE_squared_l420_42051

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E on AC -/
  E : ℝ × ℝ
  /-- AB is parallel to CD -/
  parallel_AB_CD : (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)
  /-- Length of AB is 6 -/
  AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6
  /-- Length of CD is 14 -/
  CD_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 14
  /-- ∠AEC is a right angle -/
  AEC_right_angle : (E.1 - A.1) * (E.1 - C.1) + (E.2 - A.2) * (E.2 - C.2) = 0
  /-- CE = CB -/
  CE_eq_CB : (E.1 - C.1)^2 + (E.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- The theorem to be proved -/
theorem isosceles_trapezoid_AE_squared (t : IsoscelesTrapezoid) :
  (t.E.1 - t.A.1)^2 + (t.E.2 - t.A.2)^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_AE_squared_l420_42051


namespace NUMINAMATH_CALUDE_no_100_equilateral_division_l420_42007

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  -- This is a simplified representation
  is_convex : Bool

/-- An equilateral triangle -/
structure EquilateralTriangle where
  -- Add necessary fields and conditions for an equilateral triangle
  -- This is a simplified representation
  is_equilateral : Bool

/-- A division of a convex polygon into equilateral triangles -/
structure PolygonDivision (P : ConvexPolygon) where
  triangles : List EquilateralTriangle
  is_valid_division : Bool  -- This would ensure the division is valid

/-- Theorem stating that no convex polygon can be divided into 100 different equilateral triangles -/
theorem no_100_equilateral_division (P : ConvexPolygon) :
  ¬∃ (d : PolygonDivision P), d.is_valid_division ∧ d.triangles.length = 100 := by
  sorry

end NUMINAMATH_CALUDE_no_100_equilateral_division_l420_42007


namespace NUMINAMATH_CALUDE_simplify_and_multiply_l420_42078

theorem simplify_and_multiply :
  (3 / 504 - 17 / 72) * (5 / 7) = -145 / 882 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_multiply_l420_42078


namespace NUMINAMATH_CALUDE_jonathan_typing_time_l420_42045

/-- Represents the time it takes for Jonathan to type the document alone -/
def jonathan_time : ℝ := 40

/-- Represents the time it takes for Susan to type the document alone -/
def susan_time : ℝ := 30

/-- Represents the time it takes for Jack to type the document alone -/
def jack_time : ℝ := 24

/-- Represents the time it takes for all three to type the document together -/
def combined_time : ℝ := 10

/-- Theorem stating that Jonathan's individual typing time satisfies the given conditions -/
theorem jonathan_typing_time :
  1 / jonathan_time + 1 / susan_time + 1 / jack_time = 1 / combined_time :=
by sorry

end NUMINAMATH_CALUDE_jonathan_typing_time_l420_42045


namespace NUMINAMATH_CALUDE_sandwiches_al_can_order_correct_l420_42020

/-- Represents the types of ingredients available at the deli -/
structure DeliIngredients where
  breads : Nat
  meats : Nat
  cheeses : Nat

/-- Represents the specific ingredients mentioned in the problem -/
structure SpecificIngredients where
  turkey : Bool
  salami : Bool
  swissCheese : Bool
  multiGrainBread : Bool

/-- Calculates the number of sandwiches Al can order -/
def sandwichesAlCanOrder (d : DeliIngredients) (s : SpecificIngredients) : Nat :=
  d.breads * d.meats * d.cheeses - d.breads - d.cheeses

/-- The theorem stating the number of sandwiches Al can order -/
theorem sandwiches_al_can_order_correct (d : DeliIngredients) (s : SpecificIngredients) :
  d.breads = 5 → d.meats = 7 → d.cheeses = 6 →
  s.turkey = true → s.salami = true → s.swissCheese = true → s.multiGrainBread = true →
  sandwichesAlCanOrder d s = 199 := by
  sorry

#check sandwiches_al_can_order_correct

end NUMINAMATH_CALUDE_sandwiches_al_can_order_correct_l420_42020


namespace NUMINAMATH_CALUDE_oranges_left_l420_42074

def initial_oranges : ℕ := 55
def oranges_taken : ℕ := 35

theorem oranges_left : initial_oranges - oranges_taken = 20 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l420_42074


namespace NUMINAMATH_CALUDE_triangle_ratio_l420_42080

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points N, D, E, F
variable (N D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : N = ((A.1 + C.1)/2, (A.2 + C.2)/2))  -- N is midpoint of AC
variable (h2 : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 100) -- AB = 10
variable (h3 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 324) -- BC = 18
variable (h4 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1-t) * C.1, t * B.2 + (1-t) * C.2)) -- D on BC
variable (h5 : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (s * A.1 + (1-s) * B.1, s * A.2 + (1-s) * B.2)) -- E on AB
variable (h6 : ∃ r u : ℝ, F = (r * D.1 + (1-r) * E.1, r * D.2 + (1-r) * E.2) ∧
                          F = (u * A.1 + (1-u) * N.1, u * A.2 + (1-u) * N.2)) -- F is intersection of DE and AN
variable (h7 : (D.1 - B.1)^2 + (D.2 - B.2)^2 = 9 * ((E.1 - B.1)^2 + (E.2 - B.2)^2)) -- BD = 3BE

-- Theorem statement
theorem triangle_ratio :
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 1/9 * ((F.1 - E.1)^2 + (F.2 - E.2)^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l420_42080


namespace NUMINAMATH_CALUDE_expression_evaluation_l420_42004

theorem expression_evaluation :
  let f (x : ℝ) := (x^2 - 5*x + 6) / (x - 2)
  f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l420_42004


namespace NUMINAMATH_CALUDE_simultaneous_sound_arrival_l420_42052

/-- Given a shooting range of length d meters, a bullet speed of c m/sec, and a speed of sound of s m/sec,
    the point x where the sound of the gunshot and the sound of the bullet hitting the target 
    arrive simultaneously is (d/2) * (1 + s/c) meters from the shooting position. -/
theorem simultaneous_sound_arrival (d c s : ℝ) (hd : d > 0) (hc : c > 0) (hs : s > 0) :
  let x := (d / 2) * (1 + s / c)
  (x / s) = (d / c + (d - x) / s) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_sound_arrival_l420_42052


namespace NUMINAMATH_CALUDE_charge_account_interest_l420_42024

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that the total amount owed after one year on a $75 charge with 7% simple annual interest is $80.25 -/
theorem charge_account_interest : total_amount_owed 75 0.07 1 = 80.25 := by
  sorry

end NUMINAMATH_CALUDE_charge_account_interest_l420_42024


namespace NUMINAMATH_CALUDE_production_theorem_l420_42003

/-- Represents the production scenario -/
structure ProductionScenario where
  women : ℕ
  hours_per_day : ℕ
  days : ℕ
  units_produced : ℚ

/-- The production function that calculates the units produced given a scenario -/
def production_function (x : ProductionScenario) (z : ProductionScenario) : ℚ :=
  (z.women * z.hours_per_day * z.days : ℚ) * x.units_produced / (x.women * x.hours_per_day * x.days : ℚ)

theorem production_theorem (x z : ProductionScenario) 
  (h : x.women = x.hours_per_day ∧ x.hours_per_day = x.days ∧ x.units_produced = x.women ^ 2) :
  production_function x z = (z.women * z.hours_per_day * z.days : ℚ) / x.women := by
  sorry

#check production_theorem

end NUMINAMATH_CALUDE_production_theorem_l420_42003


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l420_42057

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 7

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem volleyball_team_selection :
  choose total_players starters - choose (total_players - quadruplets) starters = 28392 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l420_42057


namespace NUMINAMATH_CALUDE_car_rental_cost_equality_l420_42095

/-- The fixed amount Samuel paid for car rental -/
def samuel_fixed_amount : ℝ := 24

/-- The per-kilometer rate for Samuel's rental -/
def samuel_rate : ℝ := 0.16

/-- The fixed amount Carrey paid for car rental -/
def carrey_fixed_amount : ℝ := 20

/-- The per-kilometer rate for Carrey's rental -/
def carrey_rate : ℝ := 0.25

/-- The distance driven by both Samuel and Carrey -/
def distance_driven : ℝ := 44.44444444444444

theorem car_rental_cost_equality :
  samuel_fixed_amount + samuel_rate * distance_driven =
  carrey_fixed_amount + carrey_rate * distance_driven :=
by sorry


end NUMINAMATH_CALUDE_car_rental_cost_equality_l420_42095


namespace NUMINAMATH_CALUDE_singer_tip_percentage_l420_42092

/-- Proves that the tip percentage is 20% given the conditions of the problem -/
theorem singer_tip_percentage (hours : ℕ) (hourly_rate : ℚ) (total_paid : ℚ) :
  hours = 3 →
  hourly_rate = 15 →
  total_paid = 54 →
  (total_paid - hours * hourly_rate) / (hours * hourly_rate) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_singer_tip_percentage_l420_42092


namespace NUMINAMATH_CALUDE_cylinder_line_segment_distance_l420_42084

/-- Represents a cylinder with a square axial cross-section -/
structure SquareCylinder where
  -- We don't need to define specific properties here

/-- Represents a line segment connecting points on the top and bottom bases of the cylinder -/
structure LineSegment where
  length : ℝ
  angle : ℝ

/-- 
Theorem: For a cylinder with a square axial cross-section, given a line segment of length l 
connecting points on the top and bottom base circumferences and making an angle α with the base plane, 
the distance d from this line segment to the cylinder axis is (l/2) * sqrt(-cos(2α)), 
and the valid range for α is π/4 < α < 3π/4.
-/
theorem cylinder_line_segment_distance (c : SquareCylinder) (seg : LineSegment) :
  let l := seg.length
  let α := seg.angle
  let d := (l / 2) * Real.sqrt (-Real.cos (2 * α))
  d > 0 ∧ π / 4 < α ∧ α < 3 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_line_segment_distance_l420_42084


namespace NUMINAMATH_CALUDE_m_squared_divisors_l420_42090

/-- A number with exactly 4 divisors -/
def HasFourDivisors (m : ℕ) : Prop :=
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 4

/-- The number of divisors of a natural number -/
def NumberOfDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem m_squared_divisors (m : ℕ) (h : HasFourDivisors m) : 
  NumberOfDivisors (m^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_m_squared_divisors_l420_42090


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_160_420_l420_42028

theorem lcm_gcf_ratio_160_420 : 
  (Nat.lcm 160 420) / (Nat.gcd 160 420 - 2) = 187 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_160_420_l420_42028


namespace NUMINAMATH_CALUDE_four_propositions_correct_l420_42017

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define symmetry about a point
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x) + 2 * b

-- Define symmetry about a line
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Define periodicity
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem four_propositions_correct (f : ℝ → ℝ) :
  (IsOdd f → SymmetricAboutPoint (fun x => f (x - 1)) 1 0) ∧
  (SymmetricAboutLine (fun x => f (x - 1)) 1 → IsEven f) ∧
  ((∀ x, f (x - 1) = -f x) → HasPeriod f 2) ∧
  (SymmetricAboutLine (fun x => f (x - 1)) 1 ∧ SymmetricAboutLine (fun x => f (1 - x)) 1) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_correct_l420_42017


namespace NUMINAMATH_CALUDE_circles_separate_l420_42071

theorem circles_separate (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂) :
  (∃ x : ℝ, x^2 - 2*R₁*x + R₂^2 - d*(R₂ - R₁) = 0 ∧
   ∀ y : ℝ, y^2 - 2*R₁*y + R₂^2 - d*(R₂ - R₁) = 0 → y = x) →
  R₁ + R₂ = d ∧ d > R₁ + R₂ := by
sorry

end NUMINAMATH_CALUDE_circles_separate_l420_42071


namespace NUMINAMATH_CALUDE_equality_condition_l420_42093

theorem equality_condition (a b c : ℝ) : a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l420_42093


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l420_42098

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ (n + 10) ∣ (n^3 + 100) ∧ ∀ (m : ℕ), m > n → ¬((m + 10) ∣ (m^3 + 100)) :=
by
  use 890
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l420_42098


namespace NUMINAMATH_CALUDE_fraction_problem_l420_42073

theorem fraction_problem (f : ℝ) : 
  (f * 8.0 = 0.25 * 8.0 + 2) → f = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l420_42073


namespace NUMINAMATH_CALUDE_prove_equation_l420_42032

theorem prove_equation (c d : ℝ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 3 + d = 8 + c) : 
  5 - c = 7 := by
sorry

end NUMINAMATH_CALUDE_prove_equation_l420_42032


namespace NUMINAMATH_CALUDE_two_digit_number_solution_l420_42056

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  n.val / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  n.val / 99

theorem two_digit_number_solution (cd : TwoDigitNumber) :
  54 * (toRepeatingDecimal cd - toDecimal cd) = (36 : ℚ) / 100 →
  cd.val = 65 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_solution_l420_42056


namespace NUMINAMATH_CALUDE_largest_possible_b_l420_42082

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_b_l420_42082


namespace NUMINAMATH_CALUDE_annies_final_crayons_l420_42065

/-- The number of crayons Annie has at the end, given the initial conditions. -/
def anniesCrayons : ℕ :=
  let initialCrayons : ℕ := 4
  let samsCrayons : ℕ := 36
  let matthewsCrayons : ℕ := 5 * samsCrayons
  initialCrayons + samsCrayons + matthewsCrayons

/-- Theorem stating that Annie will have 220 crayons at the end. -/
theorem annies_final_crayons : anniesCrayons = 220 := by
  sorry

end NUMINAMATH_CALUDE_annies_final_crayons_l420_42065


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_rising_right_side_simplest_form_f_satisfies_conditions_l420_42047

/-- A quadratic function that satisfies the given conditions -/
def f (x : ℝ) : ℝ := x^2

/-- The vertex of f is on the x-axis -/
theorem vertex_on_x_axis : ∃ h : ℝ, f h = 0 ∧ ∀ x : ℝ, f x ≥ f h :=
sorry

/-- f is rising on the right side of the y-axis -/
theorem rising_right_side : ∀ x > 0, ∀ y > x, f y > f x :=
sorry

/-- f is in its simplest form -/
theorem simplest_form : ∀ a b c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c = f x) → a = 1 ∧ b = 0 ∧ c = 0 :=
sorry

/-- f satisfies all the required conditions -/
theorem f_satisfies_conditions : 
  (∃ h : ℝ, f h = 0 ∧ ∀ x : ℝ, f x ≥ f h) ∧ 
  (∀ x > 0, ∀ y > x, f y > f x) ∧
  (∀ a b c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c = f x) → a = 1 ∧ b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_rising_right_side_simplest_form_f_satisfies_conditions_l420_42047


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_faces_pentagonal_pyramid_faces_proof_l420_42034

/-- A pentagonal pyramid is a three-dimensional shape with a pentagonal base and triangular faces connecting the base to an apex. -/
structure PentagonalPyramid where
  base : Pentagon
  triangular_faces : Fin 5 → Triangle

/-- A pentagon is a polygon with 5 sides. -/
structure Pentagon where
  sides : Fin 5 → Segment

/-- Theorem: The number of faces of a pentagonal pyramid is 6. -/
theorem pentagonal_pyramid_faces (p : PentagonalPyramid) : Nat :=
  6

#check pentagonal_pyramid_faces

/-- Proof of the theorem -/
theorem pentagonal_pyramid_faces_proof (p : PentagonalPyramid) : 
  pentagonal_pyramid_faces p = 6 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_faces_pentagonal_pyramid_faces_proof_l420_42034


namespace NUMINAMATH_CALUDE_polygon_product_symmetric_l420_42083

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  
/-- Calculates the sum of products of side lengths and distances for two polygons -/
def polygonProduct (P Q : ConvexPolygon) : ℝ :=
  sorry

/-- Theorem stating that the polygon product is symmetric -/
theorem polygon_product_symmetric (P Q : ConvexPolygon) :
  polygonProduct P Q = polygonProduct Q P := by
  sorry

end NUMINAMATH_CALUDE_polygon_product_symmetric_l420_42083


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l420_42026

/-- The distance from the origin to the line x = 1 is 1. -/
theorem distance_origin_to_line : ∃ d : ℝ, d = 1 ∧ 
  ∀ (x y : ℝ), x = 1 → d = |x| := by sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l420_42026


namespace NUMINAMATH_CALUDE_middle_school_students_l420_42039

theorem middle_school_students (band_percentage : ℝ) (band_students : ℕ) (total_students : ℕ) : 
  band_percentage = 0.20 →
  band_students = 168 →
  (band_percentage * total_students : ℝ) = band_students →
  total_students = 840 := by
sorry

end NUMINAMATH_CALUDE_middle_school_students_l420_42039


namespace NUMINAMATH_CALUDE_smallest_x_value_l420_42040

theorem smallest_x_value : 
  let f (x : ℚ) := 7 * (4 * x^2 + 4 * x + 5) - x * (4 * x - 35)
  ∃ (x : ℚ), f x = 0 ∧ ∀ (y : ℚ), f y = 0 → x ≤ y ∧ x = -5/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l420_42040


namespace NUMINAMATH_CALUDE_railway_length_scientific_notation_l420_42013

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem railway_length_scientific_notation :
  toScientificNotation 95500 = ScientificNotation.mk 9.55 4 (by norm_num) := by
  sorry

end NUMINAMATH_CALUDE_railway_length_scientific_notation_l420_42013


namespace NUMINAMATH_CALUDE_mod_nineteen_problem_l420_42075

theorem mod_nineteen_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ 38574 ≡ n [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_mod_nineteen_problem_l420_42075


namespace NUMINAMATH_CALUDE_circle_intersection_and_common_chord_l420_42019

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + 45 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 4*x + 3*y - 23 = 0

-- Define the intersection of circles
def circles_intersect (C₁ C₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, C₁ x y ∧ C₂ x y

-- Define the common chord
def common_chord (C₁ C₂ line_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (C₁ x y ∧ C₂ x y) → line_eq x y

-- Theorem statement
theorem circle_intersection_and_common_chord :
  (circles_intersect C₁ C₂) ∧
  (common_chord C₁ C₂ line_eq) ∧
  (∃ a b, C₁ a b ∧ C₂ a b ∧ (a - 1)^2 + (b - 3)^2 = 7) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_common_chord_l420_42019


namespace NUMINAMATH_CALUDE_dried_mushrooms_from_fresh_l420_42012

/-- Calculates the amount of dried mushrooms obtained from fresh mushrooms -/
theorem dried_mushrooms_from_fresh (fresh_mass : ℝ) (fresh_water_percent : ℝ) 
  (dried_water_percent : ℝ) (h1 : fresh_water_percent = 0.9) 
  (h2 : dried_water_percent = 0.12) (h3 : fresh_mass = 44) : 
  (fresh_mass * (1 - fresh_water_percent)) / (1 - dried_water_percent) = 5 := by
  sorry

end NUMINAMATH_CALUDE_dried_mushrooms_from_fresh_l420_42012


namespace NUMINAMATH_CALUDE_maggie_picked_40_apples_l420_42072

/-- The number of apples Kelsey picked -/
def kelsey_apples : ℕ := 28

/-- The number of apples Layla picked -/
def layla_apples : ℕ := 22

/-- The average number of apples picked by the three -/
def average_apples : ℕ := 30

/-- The number of people who picked apples -/
def num_people : ℕ := 3

/-- The number of apples Maggie picked -/
def maggie_apples : ℕ := 40

theorem maggie_picked_40_apples :
  kelsey_apples + layla_apples + maggie_apples = average_apples * num_people :=
by sorry

end NUMINAMATH_CALUDE_maggie_picked_40_apples_l420_42072


namespace NUMINAMATH_CALUDE_speed_ratio_l420_42070

/-- The speed of object A -/
def v_A : ℝ := sorry

/-- The speed of object B -/
def v_B : ℝ := sorry

/-- The initial distance of B from O -/
def initial_distance : ℝ := 800

/-- The time when A and B are first equidistant from O -/
def t1 : ℝ := 3

/-- The additional time until A and B are again equidistant from O -/
def t2 : ℝ := 5

theorem speed_ratio : 
  (∀ t : ℝ, t1 * v_A = |initial_distance - t1 * v_B|) ∧
  (∀ t : ℝ, (t1 + t2) * v_A = |initial_distance - (t1 + t2) * v_B|) →
  v_A / v_B = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_speed_ratio_l420_42070


namespace NUMINAMATH_CALUDE_cindy_earnings_l420_42061

/-- Calculates the earnings for teaching one math course in a month -/
def earnings_per_course (total_courses : ℕ) (total_hours_per_week : ℕ) (hourly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (total_hours_per_week / total_courses) * weeks_per_month * hourly_rate

/-- Theorem: Cindy's earnings for one math course in a month is $1200 -/
theorem cindy_earnings : 
  earnings_per_course 4 48 25 4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_cindy_earnings_l420_42061


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l420_42060

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x - 2 / Real.sqrt x) ^ 5
  ∃ c : ℝ, c = 40 ∧ 
    ∃ other_terms : ℝ → ℝ, 
      expansion = c * x^2 + other_terms x :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l420_42060


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l420_42033

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ),
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 12 ≠ 0) ∧
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + (c : ℝ) * x + 12 = 0) ∧
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l420_42033


namespace NUMINAMATH_CALUDE_square_perimeter_l420_42076

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) :
  4 * side = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l420_42076


namespace NUMINAMATH_CALUDE_correct_propositions_l420_42068

/-- A structure representing a plane with lines -/
structure Plane where
  /-- The type of lines in the plane -/
  Line : Type
  /-- Perpendicularity relation between lines -/
  perp : Line → Line → Prop
  /-- Parallelism relation between lines -/
  parallel : Line → Line → Prop

/-- The main theorem stating the two correct propositions -/
theorem correct_propositions (P : Plane) 
  (a b c α β γ : P.Line) : 
  (P.perp a α ∧ P.perp b β ∧ P.perp α β → P.perp a b) ∧
  (P.parallel α β ∧ P.parallel β γ ∧ P.perp a α → P.perp a γ) := by
  sorry


end NUMINAMATH_CALUDE_correct_propositions_l420_42068


namespace NUMINAMATH_CALUDE_max_value_of_d_l420_42085

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + 5 * Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_d_l420_42085


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l420_42097

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence (α : Type*) [Field α] where
  first_term : α
  common_difference : α

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1 : ℚ) * seq.common_difference

theorem twelfth_term_of_specific_sequence :
  let seq := ArithmeticSequence.mk (1/2 : ℚ) ((5/6 - 1/2) : ℚ)
  nth_term seq 2 = 5/6 → nth_term seq 3 = 7/6 → nth_term seq 12 = 25/6 := by
  sorry


end NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l420_42097


namespace NUMINAMATH_CALUDE_distribution_methods_count_l420_42011

/-- The number of ways to distribute tickets to tourists -/
def distribute_tickets : ℕ :=
  Nat.choose 6 2 * Nat.choose 4 2 * (Nat.factorial 2)

/-- Theorem stating that the number of distribution methods is 180 -/
theorem distribution_methods_count : distribute_tickets = 180 := by
  sorry

end NUMINAMATH_CALUDE_distribution_methods_count_l420_42011


namespace NUMINAMATH_CALUDE_quadratic_factoring_l420_42064

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The result of factoring a quadratic equation -/
inductive FactoredForm
  | Product : FactoredForm

/-- Factoring a quadratic equation results in a product form -/
theorem quadratic_factoring (eq : QuadraticEquation) : ∃ (f : FactoredForm), f = FactoredForm.Product := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factoring_l420_42064


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l420_42031

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.exp (x^2) - Real.cos x) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = (3/2) := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l420_42031


namespace NUMINAMATH_CALUDE_three_numbers_sum_l420_42027

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →  -- Ascending order
  y = 7 →  -- Median is 7
  (x + y + z) / 3 = x + 12 →  -- Mean is 12 more than least
  (x + y + z) / 3 = z - 18 →  -- Mean is 18 less than greatest
  x + y + z = 39 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l420_42027


namespace NUMINAMATH_CALUDE_remaining_insects_l420_42063

def playground_insects (spiders ants initial_ladybugs departed_ladybugs : ℕ) : ℕ :=
  spiders + ants + initial_ladybugs - departed_ladybugs

theorem remaining_insects : 
  playground_insects 3 12 8 2 = 21 := by sorry

end NUMINAMATH_CALUDE_remaining_insects_l420_42063


namespace NUMINAMATH_CALUDE_quadratic_equation_theorem_l420_42021

theorem quadratic_equation_theorem (m : ℝ) (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 1 = 0 ∧ y^2 - 2*y + m - 1 = 0) →  -- two real roots condition
  (p^2 - 2*p + m - 1 = 0) →  -- p is a root
  ((p^2 - 2*p + 3)*(m + 4) = 7) →  -- given equation
  (m = -3 ∧ m ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_theorem_l420_42021


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l420_42038

theorem fractional_equation_solution :
  ∃ x : ℚ, x = -3/4 ∧ x / (x + 1) = 2 * x / (3 * x + 3) - 1 :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l420_42038


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l420_42054

theorem complex_multiplication_sum (z a b : ℂ) : 
  z = 3 + I → 
  Complex.I * z = a + b * I → 
  (z.re : ℝ) + (z.im : ℝ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l420_42054


namespace NUMINAMATH_CALUDE_system_solution_condition_l420_42048

theorem system_solution_condition (a : ℝ) :
  (∃ (x y b : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2*b^2 = 2*b*(x - y) + 1) →
  a ≤ Real.sqrt 2 + 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_condition_l420_42048


namespace NUMINAMATH_CALUDE_tile_arrangement_l420_42037

/-- The internal angle of a square in degrees -/
def square_angle : ℝ := 90

/-- The internal angle of an octagon in degrees -/
def octagon_angle : ℝ := 135

/-- The sum of angles around a vertex in degrees -/
def vertex_sum : ℝ := 360

/-- The number of square tiles around a vertex -/
def num_square_tiles : ℕ := 1

/-- The number of octagonal tiles around a vertex -/
def num_octagon_tiles : ℕ := 2

theorem tile_arrangement :
  num_square_tiles * square_angle + num_octagon_tiles * octagon_angle = vertex_sum :=
by sorry

end NUMINAMATH_CALUDE_tile_arrangement_l420_42037


namespace NUMINAMATH_CALUDE_greatest_constant_for_triangle_inequality_l420_42023

theorem greatest_constant_for_triangle_inequality (a b c : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (a + b > c) → (b + c > a) → (c + a > b) →
  (∃ (N : ℝ), ∀ (a b c : ℝ), 
    (a > 0) → (b > 0) → (c > 0) →
    (a + b > c) → (b + c > a) → (c + a > b) →
    (a^2 + b^2 + a*b) / c^2 > N) ∧
  (∀ (M : ℝ), 
    (∀ (a b c : ℝ), 
      (a > 0) → (b > 0) → (c > 0) →
      (a + b > c) → (b + c > a) → (c + a > b) →
      (a^2 + b^2 + a*b) / c^2 > M) →
    M ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_greatest_constant_for_triangle_inequality_l420_42023


namespace NUMINAMATH_CALUDE_simple_interest_problem_l420_42025

theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) : 
  interest = 4016.25 →
  rate = 9 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 8925 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l420_42025


namespace NUMINAMATH_CALUDE_spadesuit_example_l420_42059

def spadesuit (a b : ℝ) : ℝ := |a - b|

theorem spadesuit_example : spadesuit 3 (spadesuit 5 8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_example_l420_42059


namespace NUMINAMATH_CALUDE_compound_interest_rate_calculation_l420_42002

theorem compound_interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (CI : ℝ)  -- Compound Interest
  (t : ℝ)  -- Time in years
  (n : ℝ)  -- Number of times interest is compounded per year
  (h1 : P = 8000)
  (h2 : CI = 484.76847061839544)
  (h3 : t = 1.5)
  (h4 : n = 2)
  : ∃ (r : ℝ), abs (r - 0.0397350993377484) < 0.0000000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_calculation_l420_42002


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l420_42041

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -5)

theorem reflection_across_y_axis :
  reflect_y_axis P = (-3, -5) := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l420_42041
