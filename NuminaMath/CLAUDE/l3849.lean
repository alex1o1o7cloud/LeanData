import Mathlib

namespace NUMINAMATH_CALUDE_olivia_weekly_earnings_l3849_384954

/-- Olivia's weekly earnings calculation -/
theorem olivia_weekly_earnings 
  (hourly_wage : ℕ) 
  (monday_hours wednesday_hours friday_hours : ℕ) : 
  hourly_wage = 9 → 
  monday_hours = 4 → 
  wednesday_hours = 3 → 
  friday_hours = 6 → 
  hourly_wage * (monday_hours + wednesday_hours + friday_hours) = 117 := by
  sorry

end NUMINAMATH_CALUDE_olivia_weekly_earnings_l3849_384954


namespace NUMINAMATH_CALUDE_f_zero_values_l3849_384934

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x y : ℝ, f (x + y) = f x * f y)
variable (h3 : deriv f 0 = 2)

-- Theorem statement
theorem f_zero_values : f 0 = 0 ∨ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_values_l3849_384934


namespace NUMINAMATH_CALUDE_boat_distance_difference_l3849_384968

/-- The difference in distance traveled between two boats, one traveling downstream
    and one upstream, is 30 km. -/
theorem boat_distance_difference
  (a : ℝ)  -- Speed of both boats in still water (km/h)
  (h : a > 5)  -- Assumption that the boat speed is greater than the water flow speed
  : (3 * (a + 5)) - (3 * (a - 5)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_boat_distance_difference_l3849_384968


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3849_384990

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 - 3*x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3849_384990


namespace NUMINAMATH_CALUDE_not_prime_fourth_power_minus_four_l3849_384915

theorem not_prime_fourth_power_minus_four (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ¬∃ q : ℕ, Nat.Prime q ∧ p - 4 = q^4 := by
  sorry

end NUMINAMATH_CALUDE_not_prime_fourth_power_minus_four_l3849_384915


namespace NUMINAMATH_CALUDE_cafeteria_cottage_pies_l3849_384946

/-- The number of lasagnas made by the cafeteria -/
def num_lasagnas : ℕ := 100

/-- The amount of ground mince used per lasagna (in pounds) -/
def mince_per_lasagna : ℕ := 2

/-- The amount of ground mince used per cottage pie (in pounds) -/
def mince_per_cottage_pie : ℕ := 3

/-- The total amount of ground mince used (in pounds) -/
def total_mince : ℕ := 500

/-- The number of cottage pies made by the cafeteria -/
def num_cottage_pies : ℕ := (total_mince - num_lasagnas * mince_per_lasagna) / mince_per_cottage_pie

theorem cafeteria_cottage_pies :
  num_cottage_pies = 100 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_cottage_pies_l3849_384946


namespace NUMINAMATH_CALUDE_pedro_has_200_squares_l3849_384950

/-- The number of squares Jesus has -/
def jesus_squares : ℕ := 60

/-- The number of squares Linden has -/
def linden_squares : ℕ := 75

/-- The number of additional squares Pedro has compared to Jesus and Linden combined -/
def pedro_additional_squares : ℕ := 65

/-- The total number of squares Pedro has -/
def pedro_squares : ℕ := jesus_squares + linden_squares + pedro_additional_squares

theorem pedro_has_200_squares : pedro_squares = 200 := by
  sorry

end NUMINAMATH_CALUDE_pedro_has_200_squares_l3849_384950


namespace NUMINAMATH_CALUDE_divisible_by_five_l3849_384969

theorem divisible_by_five (a b : ℕ) : 
  (∃ k : ℕ, a * b = 5 * k) → (∃ m : ℕ, a = 5 * m) ∨ (∃ n : ℕ, b = 5 * n) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_five_l3849_384969


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3849_384952

/-- Given that the solution set of ax^2 + bx + c ≤ 0 is {x | x ≤ -1/3 ∨ x ≥ 2},
    prove that the solution set of cx^2 + bx + a > 0 is {x | x < -3 ∨ x > 1/2} -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c ≤ 0 ↔ x ≤ -1/3 ∨ x ≥ 2) :
  ∀ x, c*x^2 + b*x + a > 0 ↔ x < -3 ∨ x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3849_384952


namespace NUMINAMATH_CALUDE_ben_win_probability_l3849_384912

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : ∀ p : ℚ, p ≠ lose_prob → p = 1 - lose_prob) : 1 - lose_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l3849_384912


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l3849_384938

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l3849_384938


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3849_384993

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - k*x₁ - 4 = 0 →
  x₂^2 - k*x₂ - 4 = 0 →
  x₁^2 + x₂^2 + x₁*x₂ = 6 →
  k^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3849_384993


namespace NUMINAMATH_CALUDE_ruble_payment_combinations_l3849_384925

theorem ruble_payment_combinations : 
  ∃! n : ℕ, n = (Finset.filter (λ (x : ℕ × ℕ) => 3 * x.1 + 5 * x.2 = 78) (Finset.product (Finset.range 79) (Finset.range 79))).card ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_ruble_payment_combinations_l3849_384925


namespace NUMINAMATH_CALUDE_initial_solution_volume_l3849_384983

/-- Given an initial solution with 42% alcohol, prove that its volume is 11 litres
    when 3 litres of water is added, resulting in a new mixture with 33% alcohol. -/
theorem initial_solution_volume (initial_percentage : Real) (added_water : Real) (final_percentage : Real) :
  initial_percentage = 0.42 →
  added_water = 3 →
  final_percentage = 0.33 →
  ∃ (initial_volume : Real),
    initial_volume * initial_percentage = (initial_volume + added_water) * final_percentage ∧
    initial_volume = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_solution_volume_l3849_384983


namespace NUMINAMATH_CALUDE_team_captain_selection_l3849_384958

def total_team_size : ℕ := 15
def shortlisted_size : ℕ := 5
def captains_to_choose : ℕ := 4

theorem team_captain_selection :
  (Nat.choose total_team_size captains_to_choose) -
  (Nat.choose (total_team_size - shortlisted_size) captains_to_choose) = 1155 :=
by sorry

end NUMINAMATH_CALUDE_team_captain_selection_l3849_384958


namespace NUMINAMATH_CALUDE_perfect_game_score_l3849_384973

/-- Given that a perfect score is 21 points, prove that the total points after 3 perfect games is 63. -/
theorem perfect_game_score (perfect_score : ℕ) (h : perfect_score = 21) :
  3 * perfect_score = 63 := by
  sorry

end NUMINAMATH_CALUDE_perfect_game_score_l3849_384973


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l3849_384961

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℝ
  b : ℝ
  y : ℝ
  first_term : ℝ := 2 * a
  second_term : ℝ := y
  third_term : ℝ := 3 * b
  fourth_term : ℝ := 4 * y
  is_arithmetic : ∃ (d : ℝ), second_term - first_term = d ∧ 
                              third_term - second_term = d ∧ 
                              fourth_term - third_term = d

/-- The ratio of a to b in the arithmetic sequence is -1/5 -/
theorem ratio_a_to_b (seq : ArithmeticSequence) : seq.a / seq.b = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l3849_384961


namespace NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l3849_384930

theorem max_value_x_2y_plus_1 (x y : ℝ) 
  (hx : |x - 1| ≤ 1) 
  (hy : |y - 2| ≤ 1) : 
  |x - 2*y + 1| ≤ 5 := by sorry

end NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l3849_384930


namespace NUMINAMATH_CALUDE_linear_function_through_points_l3849_384931

/-- A linear function passing through point A(1, -1) -/
def f (a : ℝ) (x : ℝ) : ℝ := -x + a

theorem linear_function_through_points :
  ∃ (a : ℝ), f a 1 = -1 ∧ f a (-2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_points_l3849_384931


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3849_384963

/-- The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3 -/
theorem min_value_quadratic (x : ℝ) : 
  ∃ (min : ℝ), (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≥ 3 * y^2 - 18 * y + 7) ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3849_384963


namespace NUMINAMATH_CALUDE_endpoint_is_200_l3849_384933

/-- The endpoint of a range of even integers starting from 20, given that its average
    is 35 greater than the average of even integers from 10 to 140 inclusive. -/
def endpoint : ℕ :=
  let start1 := 20
  let start2 := 10
  let end2 := 140
  let diff := 35
  let avg2 := (start2 + end2) / 2
  let endpoint := 2 * (avg2 + diff) - start1
  endpoint

theorem endpoint_is_200 : endpoint = 200 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_is_200_l3849_384933


namespace NUMINAMATH_CALUDE_john_percentage_increase_l3849_384992

/-- Represents the data for John's work at two hospitals --/
structure HospitalData where
  patients_first_hospital : ℕ  -- patients per day at first hospital
  total_patients_per_year : ℕ  -- total patients per year
  days_per_week : ℕ           -- working days per week
  weeks_per_year : ℕ          -- working weeks per year

/-- Calculates the percentage increase in patients at the second hospital compared to the first --/
def percentage_increase (data : HospitalData) : ℚ :=
  let total_working_days := data.days_per_week * data.weeks_per_year
  let patients_second_hospital := (data.total_patients_per_year - data.patients_first_hospital * total_working_days) / total_working_days
  ((patients_second_hospital - data.patients_first_hospital) / data.patients_first_hospital) * 100

/-- Theorem stating that given John's work conditions, the percentage increase is 20% --/
theorem john_percentage_increase :
  let john_data : HospitalData := {
    patients_first_hospital := 20,
    total_patients_per_year := 11000,
    days_per_week := 5,
    weeks_per_year := 50
  }
  percentage_increase john_data = 20 := by
  sorry


end NUMINAMATH_CALUDE_john_percentage_increase_l3849_384992


namespace NUMINAMATH_CALUDE_complex_number_ratio_l3849_384987

theorem complex_number_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : ((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = (3.5 : ℂ) := by
sorry

end NUMINAMATH_CALUDE_complex_number_ratio_l3849_384987


namespace NUMINAMATH_CALUDE_solution_to_equation_l3849_384977

theorem solution_to_equation : ∃ x y : ℝ, x + 2 * y = 4 ∧ x = 0 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3849_384977


namespace NUMINAMATH_CALUDE_complex_number_range_angle_between_vectors_l3849_384947

-- Problem 1
theorem complex_number_range (Z : ℂ) (a : ℝ) 
  (h1 : (Z + 2*I).im = 0)
  (h2 : ((Z / (2 - I)).im = 0))
  (h3 : ((Z + a*I)^2).re > 0)
  (h4 : ((Z + a*I)^2).im > 0) :
  2 < a ∧ a < 6 := by sorry

-- Problem 2
theorem angle_between_vectors (z₁ z₂ : ℂ) 
  (h1 : z₁ = 3)
  (h2 : z₂ = -5 + 5*I) :
  Real.arccos ((z₁.re * z₂.re + z₁.im * z₂.im) / (Complex.abs z₁ * Complex.abs z₂)) = 3 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_complex_number_range_angle_between_vectors_l3849_384947


namespace NUMINAMATH_CALUDE_ball_draw_probability_l3849_384924

theorem ball_draw_probability (n : ℕ) : 
  (200 ≤ n) ∧ (n ≤ 1000) ∧ 
  (∃ k : ℕ, n = k^2) ∧
  (∃ x y : ℕ, x + y = n ∧ (x - y)^2 = n) →
  (∃ l : List ℕ, l.length = 17 ∧ n ∈ l) :=
sorry

end NUMINAMATH_CALUDE_ball_draw_probability_l3849_384924


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_l3849_384986

theorem chinese_remainder_theorem (x : ℤ) :
  (2 + x) % (2^4) = 3^2 % (2^4) ∧
  (3 + x) % (3^4) = 2^3 % (3^4) ∧
  (5 + x) % (5^4) = 7^2 % (5^4) →
  x % 30 = 14 := by
sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_l3849_384986


namespace NUMINAMATH_CALUDE_total_loaves_served_l3849_384996

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real)
  (h1 : wheat_bread = 0.5)
  (h2 : white_bread = 0.4) :
  wheat_bread + white_bread = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l3849_384996


namespace NUMINAMATH_CALUDE_production_performance_l3849_384922

/-- Represents the production schedule and actual performance of a team of workers. -/
structure ProductionSchedule where
  total_parts : ℕ
  days_ahead : ℕ
  extra_parts_per_day : ℕ

/-- Calculates the intended time frame and daily overachievement percentage. -/
def calculate_performance (schedule : ProductionSchedule) : ℕ × ℚ :=
  sorry

/-- Theorem stating that for the given production schedule, 
    the intended time frame was 40 days and the daily overachievement was 25%. -/
theorem production_performance :
  let schedule := ProductionSchedule.mk 8000 8 50
  calculate_performance schedule = (40, 25/100) := by
  sorry

end NUMINAMATH_CALUDE_production_performance_l3849_384922


namespace NUMINAMATH_CALUDE_expression_evaluation_l3849_384921

theorem expression_evaluation (a b c : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  (c^2 + a^2 + b)^2 - (c^2 + a^2 - b)^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3849_384921


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3849_384945

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers without any zero -/
def six_digit_numbers_without_zero : ℕ := 531441

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero : 
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3849_384945


namespace NUMINAMATH_CALUDE_shaded_area_sum_l3849_384978

/-- Given an equilateral triangle with side length 10 cm and an inscribed circle
    whose diameter is a side of the triangle, the sum of the areas of the two regions
    between the circle and the triangle can be expressed as a*π - b*√c,
    where a + b + c = 143/6. -/
theorem shaded_area_sum (a b c : ℝ) : 
  let side_length : ℝ := 10
  let triangle_area := side_length^2 * Real.sqrt 3 / 4
  let circle_radius := side_length / 2
  let sector_area := π * circle_radius^2 / 3
  let shaded_area := 2 * (sector_area - triangle_area / 2)
  (∃ (a b c : ℝ), shaded_area = a * π - b * Real.sqrt c ∧ a + b + c = 143/6) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l3849_384978


namespace NUMINAMATH_CALUDE_chicken_pot_pie_customers_l3849_384956

/-- The number of pieces in a shepherd's pie -/
def shepherds_pie_pieces : ℕ := 4

/-- The number of pieces in a chicken pot pie -/
def chicken_pot_pie_pieces : ℕ := 5

/-- The number of customers who ordered slices of shepherd's pie -/
def shepherds_pie_customers : ℕ := 52

/-- The total number of pies sold -/
def total_pies_sold : ℕ := 29

/-- Theorem stating the number of customers who ordered slices of chicken pot pie -/
theorem chicken_pot_pie_customers : ℕ := by
  sorry

end NUMINAMATH_CALUDE_chicken_pot_pie_customers_l3849_384956


namespace NUMINAMATH_CALUDE_problem_solution_l3849_384953

theorem problem_solution (x y : ℝ) : 
  x / y = 15 / 5 → y = 25 → x = 75 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3849_384953


namespace NUMINAMATH_CALUDE_sum_of_ten_and_thousand_cube_equals_1010_scientific_notation_of_1010_sum_equals_scientific_notation_l3849_384937

theorem sum_of_ten_and_thousand_cube_equals_1010 : 10 + 10^3 = 1010 := by
  sorry

theorem scientific_notation_of_1010 : 1010 = 1.01 * 10^3 := by
  sorry

theorem sum_equals_scientific_notation : 10 + 10^3 = 1.01 * 10^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_and_thousand_cube_equals_1010_scientific_notation_of_1010_sum_equals_scientific_notation_l3849_384937


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l3849_384948

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l3849_384948


namespace NUMINAMATH_CALUDE_password_identification_l3849_384972

def is_valid_password (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n % 9 = 0 ∧ n / 1000 = 5

def alice_knows (n : ℕ) : Prop :=
  ∃ a b : ℕ, n / 100 % 10 = a ∧ n / 10 % 10 = b

def bob_knows (n : ℕ) : Prop :=
  ∃ b c : ℕ, n / 10 % 10 = b ∧ n % 10 = c

def initially_unknown (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_valid_password m ∧ alice_knows m ∧ bob_knows m

theorem password_identification :
  ∃ n : ℕ,
    is_valid_password n ∧
    alice_knows n ∧
    bob_knows n ∧
    initially_unknown n ∧
    (∀ m : ℕ, is_valid_password m ∧ alice_knows m ∧ bob_knows m ∧ initially_unknown m → m ≤ n) ∧
    n = 5940 :=
  sorry

end NUMINAMATH_CALUDE_password_identification_l3849_384972


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3849_384919

def parabola (x y : ℝ) : Prop := y = -(x + 2)^2 + 6

theorem parabola_y_axis_intersection :
  ∃ (y : ℝ), parabola 0 y ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3849_384919


namespace NUMINAMATH_CALUDE_percentage_problem_l3849_384962

theorem percentage_problem (N : ℝ) (P : ℝ) (h1 : N = 140) 
  (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3849_384962


namespace NUMINAMATH_CALUDE_probability_three_different_suits_l3849_384936

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of selecting three cards of different suits from a standard deck without replacement -/
def probabilityDifferentSuits : ℚ :=
  (CardsPerSuit * (StandardDeck - CardsPerSuit) * (StandardDeck - 2 * CardsPerSuit)) /
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem probability_three_different_suits :
  probabilityDifferentSuits = 169 / 425 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_different_suits_l3849_384936


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3849_384998

open Set

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_deriv : ∀ x, f x > deriv f x) (h_init : f 0 = 2) :
  {x : ℝ | f x < 2 * Real.exp x} = {x : ℝ | x > 0} := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3849_384998


namespace NUMINAMATH_CALUDE_monogram_count_l3849_384970

def alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def before_m : List Char := alphabet.take 12
def after_m : List Char := alphabet.drop 13

def is_valid_monogram (first middle last : Char) : Prop :=
  first ∈ before_m ∧ middle = 'M' ∧ last ∈ after_m ∧ first < middle ∧ middle < last

def count_valid_monograms : Nat :=
  (before_m.length) * (after_m.length)

theorem monogram_count :
  count_valid_monograms = 156 := by sorry

end NUMINAMATH_CALUDE_monogram_count_l3849_384970


namespace NUMINAMATH_CALUDE_tv_ad_sequences_l3849_384966

/-- Represents the number of different broadcast sequences for advertisements -/
def num_broadcast_sequences (total_ads : ℕ) (commercial_ads : ℕ) (public_service_ads : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of different broadcast sequences for the given conditions -/
theorem tv_ad_sequences :
  let total_ads := 5
  let commercial_ads := 3
  let public_service_ads := 2
  num_broadcast_sequences total_ads commercial_ads public_service_ads = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_ad_sequences_l3849_384966


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3849_384905

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 3*α - 2 = 0) → 
  (β^2 - 3*β - 2 = 0) → 
  7*α^4 + 10*β^3 = 544 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3849_384905


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3849_384971

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧ 
    n = 2 * a + 3 ∧ 
    n = 3 * b + 2

theorem smallest_dual_base_representation : 
  (∀ m < 17, ¬ is_valid_representation m) ∧ 
  is_valid_representation 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3849_384971


namespace NUMINAMATH_CALUDE_ladder_problem_l3849_384989

theorem ladder_problem (ladder_length height : Real) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : Real), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3849_384989


namespace NUMINAMATH_CALUDE_ball_max_height_l3849_384914

/-- The height function of the ball -/
def height_function (t : ℝ) : ℝ := 180 * t - 20 * t^2

/-- The maximum height reached by the ball -/
def max_height : ℝ := 405

theorem ball_max_height : 
  ∃ t : ℝ, height_function t = max_height ∧ 
  ∀ u : ℝ, height_function u ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3849_384914


namespace NUMINAMATH_CALUDE_log_relationship_l3849_384913

theorem log_relationship (c d : ℝ) (hc : c = Real.log 625 / Real.log 4) (hd : d = Real.log 25 / Real.log 5) :
  c = 2 * d := by
  sorry

end NUMINAMATH_CALUDE_log_relationship_l3849_384913


namespace NUMINAMATH_CALUDE_train_length_l3849_384982

/-- Given a train that can cross an electric pole in 10 seconds at a speed of 180 km/h,
    prove that its length is 500 meters. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 180 →
  time_s = 10 →
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3849_384982


namespace NUMINAMATH_CALUDE_yuna_division_l3849_384980

theorem yuna_division (x : ℚ) : 8 * x = 56 → 42 / x = 6 := by
  sorry

end NUMINAMATH_CALUDE_yuna_division_l3849_384980


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3849_384975

theorem binomial_expansion_properties :
  let n : ℕ := 5
  let a : ℝ := 1
  let b : ℝ := 2
  -- The coefficient of the third term
  (Finset.sum (Finset.range 1) (fun k => (n.choose k) * a^(n-k) * b^k)) = 40 ∧
  -- The sum of all binomial coefficients
  (Finset.sum (Finset.range (n+1)) (fun k => n.choose k)) = 32 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3849_384975


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l3849_384926

theorem tax_percentage_calculation (paycheck : ℝ) (savings : ℝ) : 
  paycheck = 125 →
  savings = 20 →
  (1 - 0.2) * (1 - (20 : ℝ) / 100) * paycheck = savings →
  (20 : ℝ) / 100 * paycheck = paycheck - ((1 - (20 : ℝ) / 100) * paycheck) :=
by sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l3849_384926


namespace NUMINAMATH_CALUDE_coefficient_sum_l3849_384943

theorem coefficient_sum (x : ℝ) (a₀ a₁ a₂ a₃ : ℝ) 
  (h : (1 - 2/x)^3 = a₀ + a₁*(1/x) + a₂*(1/x)^2 + a₃*(1/x)^3) :
  a₁ + a₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_l3849_384943


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l3849_384944

theorem min_value_abs_sum (a : ℝ) (h : 0 ≤ a ∧ a < 4) : 
  ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, 0 ≤ x ∧ x < 4 → m ≤ |x - 2| + |3 - x| :=
by sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l3849_384944


namespace NUMINAMATH_CALUDE_last_fish_is_sudak_l3849_384988

-- Define the types of fish
inductive Fish : Type
| Perch : Fish
| Pike : Fish
| Sudak : Fish

-- Define the initial fish counts
def initial_perches : Nat := 6
def initial_pikes : Nat := 7
def initial_sudaks : Nat := 8

-- Define the eating rules
def can_eat (eater prey : Fish) : Prop :=
  match eater, prey with
  | Fish.Perch, Fish.Pike => True
  | Fish.Pike, Fish.Pike => True
  | Fish.Pike, Fish.Perch => True
  | _, _ => False

-- Define the restriction on eating fish that have eaten an odd number
def odd_eater_restriction (f : Fish → Nat) : Prop :=
  ∀ fish, f fish % 2 = 1 → ∀ other, ¬(can_eat other fish)

-- Define the theorem
theorem last_fish_is_sudak :
  ∃ (final_state : Fish → Nat),
    (final_state Fish.Perch + final_state Fish.Pike + final_state Fish.Sudak = 1) ∧
    (final_state Fish.Sudak = 1) ∧
    (∃ (intermediate_state : Fish → Nat),
      odd_eater_restriction intermediate_state ∧
      (∀ fish, final_state fish ≤ intermediate_state fish) ∧
      (intermediate_state Fish.Perch ≤ initial_perches) ∧
      (intermediate_state Fish.Pike ≤ initial_pikes) ∧
      (intermediate_state Fish.Sudak ≤ initial_sudaks)) :=
sorry

end NUMINAMATH_CALUDE_last_fish_is_sudak_l3849_384988


namespace NUMINAMATH_CALUDE_least_multiple_divisible_l3849_384904

theorem least_multiple_divisible (x : ℕ) : 
  (∀ y : ℕ, y > 0 ∧ y < 57 → ¬(57 ∣ 23 * y)) ∧ (57 ∣ 23 * 57) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_divisible_l3849_384904


namespace NUMINAMATH_CALUDE_rational_number_conditions_l3849_384957

theorem rational_number_conditions (a b : ℚ) : 
  a ≠ 0 → b ≠ 0 → abs a = a → abs b = -b → a + b < 0 → 
  ∃ (a b : ℚ), a = 1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_number_conditions_l3849_384957


namespace NUMINAMATH_CALUDE_quadratic_sets_problem_l3849_384974

theorem quadratic_sets_problem (p q : ℝ) :
  let A := {x : ℝ | x^2 + p*x + 15 = 0}
  let B := {x : ℝ | x^2 - 5*x + q = 0}
  (A ∩ B = {3}) →
  (p = -8 ∧ q = 6 ∧ A ∪ B = {2, 3, 5}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sets_problem_l3849_384974


namespace NUMINAMATH_CALUDE_line_slope_hyperbola_intersection_l3849_384959

/-- A line intersecting a hyperbola x^2 - y^2 = 1 at two points has a slope of 2 
    if the midpoint of the line segment between these points is (2,1) -/
theorem line_slope_hyperbola_intersection (A B : ℝ × ℝ) : 
  (A.1^2 - A.2^2 = 1) →  -- A is on the hyperbola
  (B.1^2 - B.2^2 = 1) →  -- B is on the hyperbola
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) →  -- Midpoint is (2,1)
  (B.2 - A.2) / (B.1 - A.1) = 2 :=  -- Slope is 2
by sorry

end NUMINAMATH_CALUDE_line_slope_hyperbola_intersection_l3849_384959


namespace NUMINAMATH_CALUDE_number_of_juniors_l3849_384994

/-- Represents the number of students in a school program -/
def total_students : ℕ := 40

/-- Represents the ratio of juniors on the debate team -/
def junior_debate_ratio : ℚ := 3/10

/-- Represents the ratio of seniors on the debate team -/
def senior_debate_ratio : ℚ := 1/5

/-- Represents the ratio of juniors in the science club -/
def junior_science_ratio : ℚ := 2/5

/-- Represents the ratio of seniors in the science club -/
def senior_science_ratio : ℚ := 1/4

/-- Theorem stating that the number of juniors in the program is 16 -/
theorem number_of_juniors :
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total_students ∧
    (junior_debate_ratio * juniors : ℚ) = (senior_debate_ratio * seniors : ℚ) ∧
    juniors = 16 :=
by sorry

end NUMINAMATH_CALUDE_number_of_juniors_l3849_384994


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l3849_384942

theorem cookie_jar_problem (C : ℕ) : (C - 1 = (C + 5) / 2) → C = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l3849_384942


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3849_384906

def cubic_equation (x : Int) : Int :=
  x^3 - 4*x^2 - 11*x + 24

def is_root (x : Int) : Prop :=
  cubic_equation x = 0

theorem integer_roots_of_cubic :
  ∀ x : Int, is_root x ↔ x = -4 ∨ x = 3 ∨ x = 8 := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3849_384906


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l3849_384964

theorem fly_distance_from_ceiling :
  ∀ (x y z : ℝ),
  x = 3 →
  y = 4 →
  Real.sqrt (x^2 + y^2 + z^2) = 7 →
  z = 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l3849_384964


namespace NUMINAMATH_CALUDE_solution_set_correct_l3849_384979

def solution_set : Set (ℚ × ℚ) :=
  {(-2/3, 1), (1, 1), (-1/3, -3), (-1/3, 2)}

def satisfies_equations (p : ℚ × ℚ) : Prop :=
  let x := p.1
  let y := p.2
  (3*x - y - 3*x*y = -1) ∧ (9*x^2*y^2 + 9*x^2 + y^2 - 6*x*y = 13)

theorem solution_set_correct :
  ∀ p : ℚ × ℚ, p ∈ solution_set ↔ satisfies_equations p :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l3849_384979


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3849_384965

/-- The equation of a hyperbola passing through (6, √3) with asymptotes y = ±x/3 -/
theorem hyperbola_equation (x y : ℝ) :
  (∀ k : ℝ, k * x = 3 * y → k = 1 ∨ k = -1) →  -- asymptotes condition
  6^2 / 9 - (Real.sqrt 3)^2 = 1 →               -- point condition
  x^2 / 9 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3849_384965


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l3849_384991

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
def Line (p1 p2 : Point2D) :=
  {p : Point2D | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

-- Define a vertical line by a point
def VerticalLine (p : Point2D) :=
  {q : Point2D | q.x = p.x}

-- Define the intersection of two lines
def Intersection (l1 l2 : Set Point2D) :=
  {p : Point2D | p ∈ l1 ∧ p ∈ l2}

theorem intersection_in_fourth_quadrant :
  let l := Line ⟨-3, 0⟩ ⟨0, -5⟩
  let l' := VerticalLine ⟨2, 4⟩
  let i := Intersection l l'
  ∀ p ∈ i, p.x > 0 ∧ p.y < 0 := by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l3849_384991


namespace NUMINAMATH_CALUDE_tim_prank_combinations_l3849_384908

/-- Represents the number of choices Tim has for each day of the week. -/
structure PrankChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of prank combinations given the choices for each day. -/
def totalCombinations (choices : PrankChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Tim's prank choices for the week -/
def timChoices : PrankChoices :=
  { monday := 1
    tuesday := 3
    wednesday := 4
    thursday := 3
    friday := 1 }

/-- Theorem stating that the total number of combinations for Tim's prank is 36 -/
theorem tim_prank_combinations :
    totalCombinations timChoices = 36 := by
  sorry


end NUMINAMATH_CALUDE_tim_prank_combinations_l3849_384908


namespace NUMINAMATH_CALUDE_triangle_special_area_implies_30_degree_angle_l3849_384984

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area of the triangle is (a² + b² - c²) / (4√3),
    then angle C equals 30°. -/
theorem triangle_special_area_implies_30_degree_angle
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a^2 + b^2 - c^2) / (4 * Real.sqrt 3) = 1/2 * a * b * Real.sin (Real.pi / 6)) :
  ∃ A B C : ℝ,
    0 < A ∧ A < Real.pi ∧
    0 < B ∧ B < Real.pi ∧
    0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    C = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_special_area_implies_30_degree_angle_l3849_384984


namespace NUMINAMATH_CALUDE_p_less_than_negative_one_l3849_384932

theorem p_less_than_negative_one (x y p : ℝ) 
  (eq1 : 3 * x - 2 * y = 4 - p)
  (eq2 : 4 * x - 3 * y = 2 + p)
  (ineq : x > y) : 
  p < -1 := by
sorry

end NUMINAMATH_CALUDE_p_less_than_negative_one_l3849_384932


namespace NUMINAMATH_CALUDE_lisa_photos_contradiction_l3849_384929

theorem lisa_photos_contradiction (animal_photos : ℕ) (flower_photos : ℕ) 
  (scenery_photos : ℕ) (abstract_photos : ℕ) :
  animal_photos = 20 ∧
  flower_photos = (3/2 : ℚ) * animal_photos ∧
  scenery_photos + abstract_photos = (2/5 : ℚ) * (animal_photos + flower_photos) ∧
  3 * abstract_photos = 2 * scenery_photos →
  ¬(80 ≤ animal_photos + flower_photos + scenery_photos + abstract_photos ∧
    animal_photos + flower_photos + scenery_photos + abstract_photos ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_lisa_photos_contradiction_l3849_384929


namespace NUMINAMATH_CALUDE_smallest_a_value_l3849_384920

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) : 
  a ≥ 17 ∧ ∃ (a₀ : ℝ), a₀ ≥ 17 ∧ (∀ a' ≥ 17, (∀ x : ℤ, Real.sin (a' * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) → a' ≥ a₀) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3849_384920


namespace NUMINAMATH_CALUDE_ab_value_l3849_384935

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3849_384935


namespace NUMINAMATH_CALUDE_problem_solution_l3849_384976

theorem problem_solution (a b A : ℝ) 
  (h1 : 3^a = A) 
  (h2 : 5^b = A) 
  (h3 : 1/a + 1/b = 2) : 
  A = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3849_384976


namespace NUMINAMATH_CALUDE_multiply_by_97_preserves_form_l3849_384941

theorem multiply_by_97_preserves_form (a b : ℕ) :
  ∃ (a' b' : ℕ), 97 * (3 * a^2 + 32 * b^2) = 3 * a'^2 + 32 * b'^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_97_preserves_form_l3849_384941


namespace NUMINAMATH_CALUDE_andrews_eggs_l3849_384955

theorem andrews_eggs (total_needed : ℕ) (still_to_buy : ℕ) 
  (h1 : total_needed = 222) 
  (h2 : still_to_buy = 67) : 
  total_needed - still_to_buy = 155 := by
sorry

end NUMINAMATH_CALUDE_andrews_eggs_l3849_384955


namespace NUMINAMATH_CALUDE_inverse_proportion_constant_l3849_384981

/-- Given two points A(3,m) and B(3m-1,2) on the graph of y = k/x, prove that k = 2 -/
theorem inverse_proportion_constant (k m : ℝ) : 
  (3 * m = k) ∧ (2 * (3 * m - 1) = k) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_constant_l3849_384981


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3849_384928

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3849_384928


namespace NUMINAMATH_CALUDE_shoe_donation_percentage_l3849_384939

theorem shoe_donation_percentage (initial_shoes : ℕ) (final_shoes : ℕ) (purchased_shoes : ℕ) : 
  initial_shoes = 80 → 
  final_shoes = 62 → 
  purchased_shoes = 6 → 
  (initial_shoes - (final_shoes - purchased_shoes)) / initial_shoes * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_shoe_donation_percentage_l3849_384939


namespace NUMINAMATH_CALUDE_julie_rowing_distance_l3849_384907

theorem julie_rowing_distance (downstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) 
  (h1 : downstream_distance = 72)
  (h2 : time = 4)
  (h3 : stream_speed = 0.5) :
  ∃ (upstream_distance : ℝ), 
    upstream_distance = 68 ∧ 
    time = upstream_distance / (downstream_distance / (2 * time) - stream_speed) ∧
    time = downstream_distance / (downstream_distance / (2 * time) + stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_julie_rowing_distance_l3849_384907


namespace NUMINAMATH_CALUDE_min_sum_pqrs_l3849_384923

theorem min_sum_pqrs (p q r s : ℕ) : 
  p > 1 → q > 1 → r > 1 → s > 1 →
  31 * (p + 1) = 37 * (q + 1) →
  41 * (r + 1) = 43 * (s + 1) →
  p + q + r + s ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_pqrs_l3849_384923


namespace NUMINAMATH_CALUDE_computer_price_reduction_l3849_384910

/-- The average percentage decrease per price reduction for a computer model -/
theorem computer_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 5000)
  (h2 : final_price = 2560)
  (h3 : ∃ x : ℝ, original_price * (1 - x/100)^3 = final_price) :
  ∃ x : ℝ, x = 20 ∧ original_price * (1 - x/100)^3 = final_price := by
sorry


end NUMINAMATH_CALUDE_computer_price_reduction_l3849_384910


namespace NUMINAMATH_CALUDE_min_sum_bound_l3849_384951

theorem min_sum_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b^2 / (6 * c^2) + c^3 / (9 * a^3) ≥ 3 / Real.rpow 162 (1/3) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' / (3 * b') + b'^2 / (6 * c'^2) + c'^3 / (9 * a'^3) = 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_bound_l3849_384951


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3849_384917

theorem smallest_x_absolute_value_equation :
  ∃ x : ℝ, (∀ y : ℝ, y * |y| = 3 * y + 2 → x ≤ y) ∧ x * |x| = 3 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3849_384917


namespace NUMINAMATH_CALUDE_ohara_triple_36_25_l3849_384985

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt a + Real.sqrt b = x

/-- Theorem: If (36,25,x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_36_25 (x : ℕ) :
  is_ohara_triple 36 25 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_36_25_l3849_384985


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3849_384927

theorem quadratic_inequality (y : ℝ) : y^2 - 8*y + 12 < 0 ↔ 2 < y ∧ y < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3849_384927


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l3849_384911

/-- Represents a position on the circular table -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the game state -/
structure GameState :=
  (placedCoins : Set Position)
  (currentPlayer : Bool)  -- true for Alice, false for Bob

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (pos : Position) : Prop :=
  pos ∉ state.placedCoins ∧ pos.x^2 + pos.y^2 ≤ 1

/-- Defines a winning strategy for a player -/
def hasWinningStrategy (player : Bool) : Prop :=
  ∀ (state : GameState), 
    state.currentPlayer = player → 
    ∃ (move : Position), isValidMove state move ∧
      ¬∃ (opponentMove : Position), 
        isValidMove (GameState.mk (state.placedCoins ∪ {move}) (¬player)) opponentMove

/-- The main theorem stating that Alice (the starting player) has a winning strategy -/
theorem alice_has_winning_strategy : 
  hasWinningStrategy true :=
sorry

end NUMINAMATH_CALUDE_alice_has_winning_strategy_l3849_384911


namespace NUMINAMATH_CALUDE_total_bugs_eaten_l3849_384940

def gecko_bugs : ℕ := 12

def lizard_bugs : ℕ := gecko_bugs / 2

def frog_bugs : ℕ := 3 * lizard_bugs

def toad_bugs : ℕ := frog_bugs + frog_bugs / 2

theorem total_bugs_eaten :
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_bugs_eaten_l3849_384940


namespace NUMINAMATH_CALUDE_friends_in_group_l3849_384903

def number_of_friends (initial_wings : ℕ) (additional_wings : ℕ) (wings_per_person : ℕ) : ℕ :=
  (initial_wings + additional_wings) / wings_per_person

theorem friends_in_group :
  number_of_friends 8 10 6 = 3 :=
by sorry

end NUMINAMATH_CALUDE_friends_in_group_l3849_384903


namespace NUMINAMATH_CALUDE_ratio_equality_l3849_384918

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a / 4) / (b / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3849_384918


namespace NUMINAMATH_CALUDE_colored_paper_distribution_l3849_384916

/-- Proves that each female student receives 6 sheets of colored paper given the problem conditions -/
theorem colored_paper_distribution (total_students : ℕ) (total_paper : ℕ) (leftover : ℕ) :
  total_students = 24 →
  total_paper = 50 →
  leftover = 2 →
  ∃ (female_students : ℕ) (male_students : ℕ),
    female_students + male_students = total_students ∧
    male_students = 2 * female_students ∧
    (total_paper - leftover) % female_students = 0 ∧
    (total_paper - leftover) / female_students = 6 :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_distribution_l3849_384916


namespace NUMINAMATH_CALUDE_debbys_friend_photos_l3849_384949

theorem debbys_friend_photos (total_photos family_photos : ℕ) 
  (h1 : total_photos = 86) 
  (h2 : family_photos = 23) : 
  total_photos - family_photos = 63 := by
  sorry

end NUMINAMATH_CALUDE_debbys_friend_photos_l3849_384949


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3849_384995

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ∧
  ∀ C > 3, ∃ y₁ y₂ y₃ y₄ y₅ y₆ : ℝ, (y₁ + y₂ + y₃ + y₄ + y₅ + y₆)^2 < C * (y₁*(y₂ + y₃) + y₂*(y₃ + y₄) + y₃*(y₄ + y₅) + y₄*(y₅ + y₆) + y₅*(y₆ + y₁) + y₆*(y₁ + y₂)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3849_384995


namespace NUMINAMATH_CALUDE_nested_square_roots_simplification_l3849_384901

theorem nested_square_roots_simplification :
  Real.sqrt (36 * Real.sqrt (12 * Real.sqrt 9)) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_roots_simplification_l3849_384901


namespace NUMINAMATH_CALUDE_square_root_of_four_l3849_384960

theorem square_root_of_four : ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3849_384960


namespace NUMINAMATH_CALUDE_boys_under_six_feet_l3849_384999

theorem boys_under_six_feet (total_students : ℕ) (boys_ratio : ℚ) (under_six_feet_ratio : ℚ) :
  total_students = 38 →
  boys_ratio = 2 / 3 →
  under_six_feet_ratio = 3 / 4 →
  ⌊boys_ratio * total_students⌋ * under_six_feet_ratio = 18 :=
by sorry

end NUMINAMATH_CALUDE_boys_under_six_feet_l3849_384999


namespace NUMINAMATH_CALUDE_isabela_cucumber_purchase_l3849_384909

/-- The number of cucumbers Isabela bought -/
def cucumbers : ℕ := 100

/-- The number of pencils Isabela bought -/
def pencils : ℕ := cucumbers / 2

/-- The original price of each item in dollars -/
def original_price : ℕ := 20

/-- The discount percentage on pencils -/
def discount_percentage : ℚ := 20 / 100

/-- The discounted price of pencils in dollars -/
def discounted_pencil_price : ℚ := original_price * (1 - discount_percentage)

/-- The total amount spent in dollars -/
def total_spent : ℕ := 2800

theorem isabela_cucumber_purchase :
  cucumbers = 100 ∧
  cucumbers = 2 * pencils ∧
  (pencils : ℚ) * discounted_pencil_price + (cucumbers : ℚ) * original_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_isabela_cucumber_purchase_l3849_384909


namespace NUMINAMATH_CALUDE_james_future_age_l3849_384967

/-- Represents the ages and relationships of Justin, Jessica, and James -/
structure FamilyAges where
  justin_age : ℕ
  jessica_age_when_justin_born : ℕ
  james_age_diff_from_jessica : ℕ
  james_age_in_five_years : ℕ

/-- Calculates James' age after a given number of years -/
def james_age_after_years (f : FamilyAges) (years : ℕ) : ℕ :=
  f.james_age_in_five_years - 5 + years

/-- Theorem stating James' age after some years -/
theorem james_future_age (f : FamilyAges) (x : ℕ) :
  f.justin_age = 26 →
  f.jessica_age_when_justin_born = 6 →
  f.james_age_diff_from_jessica = 7 →
  f.james_age_in_five_years = 44 →
  james_age_after_years f x = 39 + x :=
by
  sorry

end NUMINAMATH_CALUDE_james_future_age_l3849_384967


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3849_384902

/-- The maximum value of x-2y for points (x,y) on the ellipse x^2/16 + y^2/9 = 1 is 2√13 -/
theorem max_value_on_ellipse :
  (∃ (x y : ℝ), x^2/16 + y^2/9 = 1 ∧ x - 2*y = 2*Real.sqrt 13) ∧
  (∀ (x y : ℝ), x^2/16 + y^2/9 = 1 → x - 2*y ≤ 2*Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l3849_384902


namespace NUMINAMATH_CALUDE_original_number_is_five_l3849_384900

theorem original_number_is_five : ∃ x : ℚ, ((x / 4) * 12) - 6 = 9 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_five_l3849_384900


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l3849_384997

theorem greatest_integer_fraction (x : ℤ) : 
  (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l3849_384997
