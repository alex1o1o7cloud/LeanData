import Mathlib

namespace equation_solution_l281_28184

theorem equation_solution : ∃ x : ℝ, 
  (Real.sqrt (7 * x - 3) + Real.sqrt (2 * x - 2) = 5) ∧ 
  (7 * x - 3 ≥ 0) ∧ 
  (2 * x - 2 ≥ 0) ∧ 
  (abs (x - 20.14) < 0.01) := by
  sorry

end equation_solution_l281_28184


namespace gcd_of_840_and_1764_l281_28108

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l281_28108


namespace complement_subset_relation_l281_28168

open Set

theorem complement_subset_relation (P Q : Set ℝ) : 
  (P = {x : ℝ | 0 < x ∧ x < 1}) → 
  (Q = {x : ℝ | x^2 + x - 2 ≤ 0}) → 
  ((compl Q) ⊆ (compl P)) :=
by
  sorry

end complement_subset_relation_l281_28168


namespace internet_discount_percentage_l281_28134

theorem internet_discount_percentage
  (monthly_rate : ℝ)
  (total_payment : ℝ)
  (num_months : ℕ)
  (h1 : monthly_rate = 50)
  (h2 : total_payment = 190)
  (h3 : num_months = 4) :
  (monthly_rate - total_payment / num_months) / monthly_rate * 100 = 5 := by
  sorry

end internet_discount_percentage_l281_28134


namespace triangle_properties_l281_28133

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about the properties of a triangle -/
theorem triangle_properties (t : Triangle) :
  (Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) →
  (2 * t.a^2 = t.b^2 + t.c^2) ∧
  (t.a = 5 ∧ Real.cos t.A = 25/31 → t.a + t.b + t.c = 14) :=
by sorry

end triangle_properties_l281_28133


namespace exists_containing_quadrilateral_l281_28104

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool

/-- A point in 2D space -/
def Point := Real × Real

/-- Check if a point is inside a convex polygon -/
def is_inside (p : Point) (poly : ConvexPolygon) : Bool := sorry

/-- Check if four points form a quadrilateral -/
def is_quadrilateral (a b c d : Point) : Bool := sorry

/-- Check if a quadrilateral contains a point -/
def quadrilateral_contains (a b c d : Point) (p : Point) : Bool := sorry

theorem exists_containing_quadrilateral 
  (poly : ConvexPolygon) (p1 p2 : Point) 
  (h1 : is_inside p1 poly) (h2 : is_inside p2 poly) :
  ∃ (a b c d : Point), 
    a ∈ poly.vertices ∧ 
    b ∈ poly.vertices ∧ 
    c ∈ poly.vertices ∧ 
    d ∈ poly.vertices ∧
    is_quadrilateral a b c d ∧
    quadrilateral_contains a b c d p1 ∧
    quadrilateral_contains a b c d p2 := by
  sorry

end exists_containing_quadrilateral_l281_28104


namespace sum_of_squares_theorem_l281_28180

theorem sum_of_squares_theorem (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 13 := by
  sorry

end sum_of_squares_theorem_l281_28180


namespace hazel_lemonade_cups_l281_28132

/-- The number of cups of lemonade Hazel sold to kids on bikes -/
def cups_sold_to_kids : ℕ := 18

/-- The number of cups of lemonade Hazel gave to her friends -/
def cups_given_to_friends : ℕ := cups_sold_to_kids / 2

/-- The number of cups of lemonade Hazel drank herself -/
def cups_drunk_by_hazel : ℕ := 1

/-- The total number of cups of lemonade Hazel made -/
def total_cups : ℕ := 56

theorem hazel_lemonade_cups : 
  2 * (cups_sold_to_kids + cups_given_to_friends + cups_drunk_by_hazel) = total_cups := by
  sorry

#check hazel_lemonade_cups

end hazel_lemonade_cups_l281_28132


namespace quadratic_discriminant_l281_28120

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation x^2 + 3x - 1 = 0 -/
def a : ℝ := 1
def b : ℝ := 3
def c : ℝ := -1

theorem quadratic_discriminant : discriminant a b c = 13 := by
  sorry

end quadratic_discriminant_l281_28120


namespace fraction_evaluation_l281_28181

theorem fraction_evaluation : (2 + 1/2) / (1 - 3/4) = 10 := by
  sorry

end fraction_evaluation_l281_28181


namespace problem_1_l281_28167

theorem problem_1 : 2 * Real.sqrt 28 + 7 * Real.sqrt 7 - Real.sqrt 7 * Real.sqrt (4/7) = 11 * Real.sqrt 7 - 2 := by
  sorry

end problem_1_l281_28167


namespace exist_x_y_sequences_l281_28126

def sequence_a : ℕ → ℚ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * sequence_a (n + 1) - sequence_a n

theorem exist_x_y_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n, 
    sequence_a n = (y n ^ 2 + 7 : ℚ) / (x n - y n : ℚ) ∧
    x n > y n ∧ 
    x n > 0 ∧ 
    y n > 0 :=
by sorry

end exist_x_y_sequences_l281_28126


namespace increasing_function_range_increasing_function_and_hyperbola_range_l281_28159

/-- The function f(x) = x² + (a-1)x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x

/-- The property that f is increasing on (1, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x > 1 → y > 1 → x < y → f a x < f a y

/-- The equation x² - ay² = 1 represents a hyperbola -/
def is_hyperbola (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x^2 - a*y^2 = 1

theorem increasing_function_range (a : ℝ) :
  is_increasing_on_interval a → a > -1 :=
sorry

theorem increasing_function_and_hyperbola_range (a : ℝ) :
  is_increasing_on_interval a → is_hyperbola a → a > 0 :=
sorry

end increasing_function_range_increasing_function_and_hyperbola_range_l281_28159


namespace joan_football_games_l281_28122

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to this year and last year -/
def total_games : ℕ := 9

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 5 := by
  sorry

end joan_football_games_l281_28122


namespace hyperbola_m_range_l281_28118

-- Define the equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / (5 + m) = 1 ∧ m * (5 + m) < 0

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ -5 < m ∧ m < 0 :=
by sorry

end hyperbola_m_range_l281_28118


namespace perpendicular_necessary_not_sufficient_l281_28148

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_necessary_not_sufficient 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicularToPlane l α) 
  (h2 : subset m β) :
  (∀ (α β : Plane), parallel α β → perpendicular l m) ∧ 
  (∃ (l m : Line) (α β : Plane), 
    perpendicularToPlane l α ∧ 
    subset m β ∧ 
    perpendicular l m ∧ 
    ¬(parallel α β)) := by
  sorry

end perpendicular_necessary_not_sufficient_l281_28148


namespace sum_of_x_and_y_is_three_l281_28100

theorem sum_of_x_and_y_is_three (x y : ℝ) 
  (hx : (x - 1)^2003 + 2002*(x - 1) = -1)
  (hy : (y - 2)^2003 + 2002*(y - 2) = 1) : 
  x + y = 3 := by
  sorry

end sum_of_x_and_y_is_three_l281_28100


namespace rem_prime_specific_value_l281_28182

/-- Modified remainder function -/
def rem' (x y : ℚ) : ℚ := x - y * ⌊x / (2 * y)⌋

/-- Theorem stating the value of rem'(5/9, -3/7) -/
theorem rem_prime_specific_value : rem' (5/9) (-3/7) = 62/63 := by
  sorry

end rem_prime_specific_value_l281_28182


namespace hyperbola_asymptotes_l281_28197

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    imaginary axis length of 4, and focal distance of 4√3,
    prove that its asymptotes are given by y = ±(√2/2)x -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_imaginary_axis : b = 2) 
  (h_focal_distance : 2 * Real.sqrt ((a^2 + b^2) : ℝ) = 4 * Real.sqrt 3) :
  ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) := by
  sorry


end hyperbola_asymptotes_l281_28197


namespace inequality_solution_set_l281_28106

-- Define the inequality
def inequality (x : ℝ) : Prop := (1 - x) * (x - 3) < 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end inequality_solution_set_l281_28106


namespace green_pill_cost_proof_l281_28199

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 22.5

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℚ := 903

theorem green_pill_cost_proof :
  green_pill_cost = 22.5 ∧
  pink_pill_cost = green_pill_cost - 2 ∧
  treatment_days = 21 ∧
  total_cost = 903 ∧
  total_cost = treatment_days * (green_pill_cost + pink_pill_cost) :=
by sorry

end green_pill_cost_proof_l281_28199


namespace line_through_point_l281_28174

/-- Given a line 3x + ay - 5 = 0 that passes through the point (1, 2), prove that a = 1 --/
theorem line_through_point (a : ℝ) : (3 * 1 + a * 2 - 5 = 0) → a = 1 := by
  sorry

end line_through_point_l281_28174


namespace route_b_faster_l281_28128

/-- Represents a route with multiple segments, each with its own distance and speed. -/
structure Route where
  segments : List (Float × Float)
  total_distance : Float

/-- Calculates the total time taken to travel a route -/
def travel_time (r : Route) : Float :=
  r.segments.foldl (fun acc (d, s) => acc + d / s) 0

/-- Route A details -/
def route_a : Route :=
  { segments := [(8, 40)], total_distance := 8 }

/-- Route B details -/
def route_b : Route :=
  { segments := [(5.5, 45), (1, 25), (0.5, 15)], total_distance := 7 }

/-- The time difference between Route A and Route B in minutes -/
def time_difference : Float :=
  travel_time route_a - travel_time route_b

theorem route_b_faster : 
  0.26 < time_difference ∧ time_difference < 0.28 :=
sorry

end route_b_faster_l281_28128


namespace largest_four_digit_divisible_by_five_l281_28157

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ, 
  (n ≤ 9999 ∧ n ≥ 1000) ∧ 
  n % 5 = 0 ∧
  ∀ m : ℕ, (m ≤ 9999 ∧ m ≥ 1000 ∧ m % 5 = 0) → m ≤ n :=
by sorry

end largest_four_digit_divisible_by_five_l281_28157


namespace square_area_ratio_l281_28111

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 3 * (4 * b)) : a^2 = 9 * b^2 := by
  sorry

end square_area_ratio_l281_28111


namespace equal_to_2x_6_l281_28162

theorem equal_to_2x_6 (x : ℝ) : 2 * x^7 / x = 2 * x^6 := by sorry

end equal_to_2x_6_l281_28162


namespace watch_time_theorem_l281_28145

/-- Represents a season of the TV show -/
structure Season where
  episodes : Nat
  minutesPerEpisode : Nat

/-- Calculates the total number of days needed to watch the show -/
def daysToWatchShow (seasons : List Season) (hoursPerDay : Nat) : Nat :=
  let totalMinutes := seasons.foldl (fun acc s => acc + s.episodes * s.minutesPerEpisode) 0
  let minutesPerDay := hoursPerDay * 60
  (totalMinutes + minutesPerDay - 1) / minutesPerDay

/-- The main theorem stating it takes 35 days to watch the show -/
theorem watch_time_theorem (seasons : List Season) (hoursPerDay : Nat) :
  seasons = [
    ⟨30, 22⟩, ⟨28, 25⟩, ⟨27, 29⟩, ⟨20, 31⟩, ⟨25, 27⟩, ⟨20, 35⟩
  ] →
  hoursPerDay = 2 →
  daysToWatchShow seasons hoursPerDay = 35 := by
  sorry

#eval daysToWatchShow [
  ⟨30, 22⟩, ⟨28, 25⟩, ⟨27, 29⟩, ⟨20, 31⟩, ⟨25, 27⟩, ⟨20, 35⟩
] 2

end watch_time_theorem_l281_28145


namespace cars_ratio_after_days_l281_28101

/-- Represents the number of days after which Station A will have 7 times as many cars as Station B -/
def days_to_reach_ratio : ℕ :=
  let initial_a : ℕ := 192
  let initial_b : ℕ := 48
  let daily_a_to_b : ℕ := 21
  let daily_b_to_a : ℕ := 24
  6

/-- Theorem stating that after the calculated number of days, 
    Station A will have 7 times as many cars as Station B -/
theorem cars_ratio_after_days :
  let initial_a : ℕ := 192
  let initial_b : ℕ := 48
  let daily_a_to_b : ℕ := 21
  let daily_b_to_a : ℕ := 24
  let days := days_to_reach_ratio
  let final_a := initial_a + days * (daily_b_to_a - daily_a_to_b)
  let final_b := initial_b + days * (daily_a_to_b - daily_b_to_a)
  final_a = 7 * final_b :=
by
  sorry

end cars_ratio_after_days_l281_28101


namespace average_percent_increase_l281_28191

theorem average_percent_increase (initial_population final_population : ℕ) 
  (years : ℕ) (h1 : initial_population = 175000) (h2 : final_population = 262500) 
  (h3 : years = 10) :
  (((final_population - initial_population) / years) / initial_population) * 100 = 5 := by
sorry

end average_percent_increase_l281_28191


namespace tiktok_twitter_ratio_l281_28116

/-- Represents the number of followers on different social media platforms --/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms --/
def total_followers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube

/-- Theorem stating the relationship between TikTok and Twitter followers --/
theorem tiktok_twitter_ratio (f : Followers) (x : ℕ) : 
  f.instagram = 240 →
  f.facebook = 500 →
  f.twitter = (f.instagram + f.facebook) / 2 →
  f.tiktok = x * f.twitter →
  f.youtube = f.tiktok + 510 →
  total_followers f = 3840 →
  x = 3 := by
  sorry

end tiktok_twitter_ratio_l281_28116


namespace initial_customers_count_l281_28190

/-- The number of customers who left -/
def customers_left : ℕ := 5

/-- The number of customers remaining -/
def customers_remaining : ℕ := 9

/-- The initial number of customers -/
def initial_customers : ℕ := customers_left + customers_remaining

theorem initial_customers_count : initial_customers = 14 := by
  sorry

end initial_customers_count_l281_28190


namespace asima_integer_possibilities_l281_28154

theorem asima_integer_possibilities (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : 4 * (2 * a - 10) + 4 * (2 * b - 10) = 440) :
  ∃ (n : ℕ), n = 64 ∧ (∀ x : ℕ, x > 0 ∧ x ≤ n → ∃ y : ℕ, y > 0 ∧ 4 * (2 * x - 10) + 4 * (2 * y - 10) = 440) :=
sorry

end asima_integer_possibilities_l281_28154


namespace same_wage_proportional_earnings_l281_28151

/-- Proves that maintaining the same hourly wage and weekly hours results in proportional earnings -/
theorem same_wage_proportional_earnings
  (seasonal_weeks : ℕ)
  (seasonal_earnings : ℝ)
  (new_weeks : ℕ)
  (new_earnings : ℝ)
  (h_seasonal_weeks : seasonal_weeks = 36)
  (h_seasonal_earnings : seasonal_earnings = 7200)
  (h_new_weeks : new_weeks = 18)
  (h_new_earnings : new_earnings = 3600)
  : (new_earnings / new_weeks) = (seasonal_earnings / seasonal_weeks) :=
by sorry

end same_wage_proportional_earnings_l281_28151


namespace min_value_2n_plus_k_l281_28103

theorem min_value_2n_plus_k (n k : ℕ) : 
  (144 + n) * 2 = n * k → -- total coins after sharing
  n > 0 → -- at least one person joins
  k > 0 → -- each person carries at least one coin
  2 * n + k ≥ 50 ∧ ∃ (n' k' : ℕ), 2 * n' + k' = 50 ∧ (144 + n') * 2 = n' * k' ∧ n' > 0 ∧ k' > 0 :=
sorry

end min_value_2n_plus_k_l281_28103


namespace inequality_proof_l281_28172

theorem inequality_proof (a b c : ℝ) 
  (sum_cond : a + b + c = 3)
  (nonzero_cond : (6*a + b^2 + c^2) * (6*b + c^2 + a^2) * (6*c + a^2 + b^2) ≠ 0) :
  a / (6*a + b^2 + c^2) + b / (6*b + c^2 + a^2) + c / (6*c + a^2 + b^2) ≤ 3/8 := by
sorry

end inequality_proof_l281_28172


namespace square_sum_product_inequality_l281_28177

theorem square_sum_product_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end square_sum_product_inequality_l281_28177


namespace profit_percent_calculation_l281_28105

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.9 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 100 / 9 := by
  sorry

end profit_percent_calculation_l281_28105


namespace count_flippable_numbers_is_1500_l281_28141

/-- A digit that remains valid when flipped -/
inductive ValidDigit
| Zero
| One
| Eight
| Six
| Nine

/-- A nine-digit number that remains unchanged when flipped -/
structure FlippableNumber :=
(d1 d2 d3 d4 d5 : ValidDigit)

/-- The count of FlippableNumbers -/
def count_flippable_numbers : ℕ := sorry

/-- The first digit cannot be zero -/
axiom first_digit_nonzero :
  ∀ (n : FlippableNumber), n.d1 ≠ ValidDigit.Zero

/-- The theorem to be proved -/
theorem count_flippable_numbers_is_1500 :
  count_flippable_numbers = 1500 := by sorry

end count_flippable_numbers_is_1500_l281_28141


namespace average_pages_is_23_l281_28135

/-- The number of pages in the storybook Taesoo read -/
def total_pages : ℕ := 161

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of pages read per day -/
def average_pages : ℚ := total_pages / days_in_week

/-- Theorem stating that the average number of pages read per day is 23 -/
theorem average_pages_is_23 : average_pages = 23 := by
  sorry

end average_pages_is_23_l281_28135


namespace sqrt_sum_equals_eleven_sqrt_two_over_six_l281_28192

theorem sqrt_sum_equals_eleven_sqrt_two_over_six :
  Real.sqrt (9/2) + Real.sqrt (2/9) = 11 * Real.sqrt 2 / 6 := by
  sorry

end sqrt_sum_equals_eleven_sqrt_two_over_six_l281_28192


namespace sum_first_10_terms_sequence_is_constant_sum_equals_first_term_times_n_l281_28150

/-- Sum of the first n terms of a geometric sequence with a₁ = 2 and r = 1 -/
def geometricSum (n : ℕ) : ℝ := 2 * n

/-- The geometric sequence with a₁ = 2 and r = 1 -/
def geometricSequence : ℕ → ℝ
  | 0 => 2
  | n + 1 => geometricSequence n

theorem sum_first_10_terms :
  geometricSum 10 = 20 := by sorry

theorem sequence_is_constant (n : ℕ) :
  geometricSequence n = 2 := by sorry

theorem sum_equals_first_term_times_n (n : ℕ) :
  geometricSum n = 2 * n := by sorry

end sum_first_10_terms_sequence_is_constant_sum_equals_first_term_times_n_l281_28150


namespace sum_of_coefficients_l281_28125

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3)) →
  A + B = 15 := by
  sorry

end sum_of_coefficients_l281_28125


namespace fifth_week_hours_l281_28143

-- Define the required average hours per week
def required_average : ℝ := 12

-- Define the number of weeks
def num_weeks : ℕ := 5

-- Define the study hours for the first four weeks
def week1_hours : ℝ := 10
def week2_hours : ℝ := 14
def week3_hours : ℝ := 9
def week4_hours : ℝ := 13

-- Define the sum of study hours for the first four weeks
def sum_first_four_weeks : ℝ := week1_hours + week2_hours + week3_hours + week4_hours

-- Theorem to prove
theorem fifth_week_hours : 
  ∃ (x : ℝ), (sum_first_four_weeks + x) / num_weeks = required_average ∧ x = 14 := by
  sorry

end fifth_week_hours_l281_28143


namespace star_operation_divisors_l281_28185

-- Define the star operation
def star (a b : ℤ) : ℚ := (a^2 : ℚ) / b

-- Define the count of positive integer divisors of a number
def countPositiveDivisors (n : ℕ) : ℕ := sorry

-- Define the count of integer x for which (20 ★ x) is a positive integer
def countValidX : ℕ := sorry

-- Theorem statement
theorem star_operation_divisors : 
  countPositiveDivisors 400 = countValidX := by sorry

end star_operation_divisors_l281_28185


namespace even_odd_difference_3000_l281_28147

/-- Sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * n

/-- Sum of the first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even numbers and the sum of the first n odd numbers -/
def evenOddDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem even_odd_difference_3000 : evenOddDifference 3000 = 3000 := by
  sorry

end even_odd_difference_3000_l281_28147


namespace pascal_triangle_interior_sum_l281_28169

theorem pascal_triangle_interior_sum (row_6_sum : ℕ) (row_8_sum : ℕ) : 
  row_6_sum = 30 → row_8_sum = 126 := by
  sorry

end pascal_triangle_interior_sum_l281_28169


namespace minoxidil_mixture_l281_28102

theorem minoxidil_mixture (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_volume : ℝ) (added_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 70 ∧ 
  initial_concentration = 0.02 ∧ 
  added_volume = 35 ∧ 
  added_concentration = 0.05 ∧ 
  final_concentration = 0.03 →
  (initial_volume * initial_concentration + added_volume * added_concentration) / 
    (initial_volume + added_volume) = final_concentration :=
by sorry

end minoxidil_mixture_l281_28102


namespace cube_plus_one_expansion_problem_solution_l281_28188

theorem cube_plus_one_expansion (n : ℕ) : 
  n^3 + 3*(n^2) + 3*n + 1 = (n + 1)^3 :=
by sorry

theorem problem_solution : 
  98^3 + 3*(98^2) + 3*98 + 1 = 970299 :=
by sorry

end cube_plus_one_expansion_problem_solution_l281_28188


namespace age_of_b_l281_28140

theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 29 →
  (a + c) / 2 = 32 →
  b = 23 := by
sorry

end age_of_b_l281_28140


namespace intersection_of_A_and_B_l281_28195

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 2 4 := by
  sorry

end intersection_of_A_and_B_l281_28195


namespace line_equation_through_point_with_slope_l281_28136

/-- The equation of a line passing through (0, 2) with slope 2 is 2x - y + 2 = 0 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (y - 2 = 2 * (x - 0)) ↔ (2 * x - y + 2 = 0) := by sorry

end line_equation_through_point_with_slope_l281_28136


namespace space_shuttle_speed_conversion_l281_28152

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) (seconds_per_hour : ℕ) : ℝ :=
  speed_km_per_second * (seconds_per_hour : ℝ)

/-- Theorem: A space shuttle orbiting at 9 km/s is equivalent to 32400 km/h -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 9 3600 = 32400 := by
  sorry

end space_shuttle_speed_conversion_l281_28152


namespace conic_section_eccentricity_l281_28124

/-- The conic section defined by the equation 10x - 2xy - 2y + 1 = 0 -/
def ConicSection (x y : ℝ) : Prop :=
  10 * x - 2 * x * y - 2 * y + 1 = 0

/-- The eccentricity of a conic section -/
def Eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 2

theorem conic_section_eccentricity :
  ∀ x y : ℝ, ConicSection x y → ∃ e : ℝ, Eccentricity e := by
  sorry

end conic_section_eccentricity_l281_28124


namespace parabola_focus_directrix_distance_l281_28127

/-- For a parabola with equation y^2 = 4x, the distance from its focus to its directrix is 2 -/
theorem parabola_focus_directrix_distance : 
  ∀ (x y : ℝ), y^2 = 4*x → ∃ (f d : ℝ × ℝ), 
    (f.1 = 1 ∧ f.2 = 0) ∧ -- focus coordinates
    (d.1 = -1 ∧ ∀ t, d.2 = t) ∧ -- directrix equation
    (f.1 - d.1 = 2) -- distance between focus and directrix
  := by sorry

end parabola_focus_directrix_distance_l281_28127


namespace circle_common_chord_l281_28109

theorem circle_common_chord (a : ℝ) (h : a > 0) : 
  ∃ (x y : ℝ), 
    (x^2 + y^2 = 4 ∧ 
     x^2 + y^2 + 2*x + 2*a*y - 6 = 0 ∧ 
     ∃ (x₁ y₁ x₂ y₂ : ℝ), 
       (x₁^2 + y₁^2 = 4 ∧ 
        x₁^2 + y₁^2 + 2*x₁ + 2*a*y₁ - 6 = 0 ∧
        x₂^2 + y₂^2 = 4 ∧ 
        x₂^2 + y₂^2 + 2*x₂ + 2*a*y₂ - 6 = 0 ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12)) →
    a = 0 := by
  sorry

end circle_common_chord_l281_28109


namespace non_officers_count_l281_28194

/-- Proves that the number of non-officers is 525 given the salary information --/
theorem non_officers_count (total_avg : ℝ) (officer_avg : ℝ) (non_officer_avg : ℝ) (officer_count : ℕ) :
  total_avg = 120 →
  officer_avg = 470 →
  non_officer_avg = 110 →
  officer_count = 15 →
  ∃ (non_officer_count : ℕ),
    (officer_count * officer_avg + non_officer_count * non_officer_avg) / (officer_count + non_officer_count) = total_avg ∧
    non_officer_count = 525 :=
by sorry

end non_officers_count_l281_28194


namespace honey_market_optimization_l281_28179

/-- Represents the honey market in Milnlandia -/
structure HoneyMarket where
  /-- Inverse demand function: P = 310 - 3Q -/
  demand : ℝ → ℝ
  /-- Production cost per jar in milns -/
  cost : ℝ
  /-- Tax per jar in milns -/
  tax : ℝ

/-- Profit function for the honey producer -/
def profit (market : HoneyMarket) (quantity : ℝ) : ℝ :=
  (market.demand quantity) * quantity - market.cost * quantity - market.tax * quantity

/-- Tax revenue function for the government -/
def taxRevenue (market : HoneyMarket) (quantity : ℝ) : ℝ :=
  market.tax * quantity

/-- The statement to be proved -/
theorem honey_market_optimization (market : HoneyMarket) 
    (h_demand : ∀ q, market.demand q = 310 - 3 * q)
    (h_cost : market.cost = 10) :
  (∃ q_max : ℝ, q_max = 50 ∧ 
    ∀ q, profit market q ≤ profit market q_max) ∧
  (∃ t_max : ℝ, t_max = 150 ∧
    ∀ t, market.tax = t → 
      taxRevenue { market with tax := t } 
        ((310 - t) / 6) ≤ 
      taxRevenue { market with tax := t_max } 
        ((310 - t_max) / 6)) := by
  sorry


end honey_market_optimization_l281_28179


namespace nikolai_is_petrs_son_l281_28139

/-- Represents a person who went fishing -/
structure Fisher where
  name : String
  fish_caught : ℕ

/-- Represents a father-son pair who went fishing -/
structure FishingPair where
  father : Fisher
  son : Fisher

/-- The total number of fish caught by all fishers -/
def total_fish : ℕ := 25

/-- Theorem stating that given the conditions, Nikolai must be Petr's son -/
theorem nikolai_is_petrs_son (pair1 pair2 : FishingPair) 
  (h1 : pair1.father.name = "Petr")
  (h2 : pair1.father.fish_caught = 3 * pair1.son.fish_caught)
  (h3 : pair2.father.fish_caught = pair2.son.fish_caught)
  (h4 : pair1.father.fish_caught + pair1.son.fish_caught + 
        pair2.father.fish_caught + pair2.son.fish_caught = total_fish)
  : pair1.son.name = "Nikolai" := by
  sorry

end nikolai_is_petrs_son_l281_28139


namespace male_population_in_village_l281_28107

theorem male_population_in_village (total_population : ℕ) 
  (h1 : total_population = 800) 
  (num_groups : ℕ) 
  (h2 : num_groups = 4) 
  (h3 : total_population % num_groups = 0) 
  (h4 : ∃ (male_group : ℕ), male_group ≤ num_groups ∧ 
    male_group * (total_population / num_groups) = total_population / num_groups) :
  total_population / num_groups = 200 :=
by sorry

end male_population_in_village_l281_28107


namespace intersection_unique_l281_28110

/-- The line is defined by (x-2)/1 = (y-3)/1 = (z-4)/2 -/
def line (x y z : ℝ) : Prop :=
  (x - 2) = (y - 3) ∧ (x - 2) = (z - 4) / 2

/-- The plane is defined by 2X + Y + Z = 0 -/
def plane (x y z : ℝ) : Prop :=
  2 * x + y + z = 0

/-- The point of intersection -/
def intersection_point : ℝ × ℝ × ℝ := (-0.2, 0.8, -0.4)

theorem intersection_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 ∧ p = intersection_point :=
by sorry

end intersection_unique_l281_28110


namespace circle_centered_at_parabola_focus_l281_28142

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  (x - parabola_focus.1)^2 + (y - parabola_focus.2)^2 = circle_radius^2

theorem circle_centered_at_parabola_focus :
  ∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + y^2 = 4 :=
sorry

end circle_centered_at_parabola_focus_l281_28142


namespace pascal_triangle_51_numbers_l281_28156

theorem pascal_triangle_51_numbers (n : ℕ) : 
  n = 50 → (n.choose 4) = 230150 := by
  sorry

end pascal_triangle_51_numbers_l281_28156


namespace tree_planting_group_size_l281_28137

/-- Proves that the number of people in the first group is 3, given the conditions of the tree planting activity. -/
theorem tree_planting_group_size :
  ∀ (x : ℕ), 
    (12 : ℚ) / x = (36 : ℚ) / (x + 6) →
    x = 3 :=
by
  sorry

end tree_planting_group_size_l281_28137


namespace subtraction_puzzle_l281_28183

theorem subtraction_puzzle :
  ∀ (A B C D E F H I J : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
     E ≠ F ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
     F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
     H ≠ I ∧ H ≠ J ∧
     I ≠ J) →
    (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧
    (1 ≤ D ∧ D ≤ 9) ∧ (1 ≤ E ∧ E ≤ 9) ∧ (1 ≤ F ∧ F ≤ 9) ∧
    (1 ≤ H ∧ H ≤ 9) ∧ (1 ≤ I ∧ I ≤ 9) ∧ (1 ≤ J ∧ J ≤ 9) →
    100 * A + 10 * B + C - (100 * D + 10 * E + F) = 100 * H + 10 * I + J →
    A + B + C + D + E + F + H + I + J = 45 →
    A + B + C = 18 := by
  sorry

end subtraction_puzzle_l281_28183


namespace square_area_with_four_circles_l281_28163

theorem square_area_with_four_circles (r : ℝ) (h : r = 3) :
  let circle_diameter := 2 * r
  let square_side := 2 * circle_diameter
  square_side ^ 2 = 144 := by sorry

end square_area_with_four_circles_l281_28163


namespace inequality_solution_l281_28170

theorem inequality_solution :
  ∃! (a b : ℝ), ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    |a * x + b - Real.sqrt (1 - x^2)| ≤ (Real.sqrt 2 - 1) / 2 ∧
    a = 0 ∧ b = 3/2 := by
  sorry

end inequality_solution_l281_28170


namespace simplify_and_evaluate_l281_28164

theorem simplify_and_evaluate (x y : ℝ) 
  (hx : x = 2 + 3 * Real.sqrt 3) 
  (hy : y = 2 - 3 * Real.sqrt 3) : 
  (x^2 / (x - y)) - (y^2 / (x - y)) = 4 := by
  sorry

end simplify_and_evaluate_l281_28164


namespace smallest_b_in_arithmetic_sequence_l281_28161

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  c = 2 * b - a →          -- a, b, c form an arithmetic sequence
  a * b * c = 125 →        -- product condition
  b ≥ 5 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    c' = 2 * b' - a' ∧ a' * b' * c' = 125 ∧ b' = 5 :=
by sorry

end smallest_b_in_arithmetic_sequence_l281_28161


namespace distance_from_origin_l281_28176

theorem distance_from_origin (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end distance_from_origin_l281_28176


namespace tobys_journey_l281_28198

/-- Toby's sled-pulling journey --/
theorem tobys_journey (unloaded_speed loaded_speed : ℝ)
  (distance1 distance2 distance3 distance4 : ℝ)
  (h1 : unloaded_speed = 20)
  (h2 : loaded_speed = 10)
  (h3 : distance1 = 180)
  (h4 : distance2 = 120)
  (h5 : distance3 = 80)
  (h6 : distance4 = 140) :
  distance1 / loaded_speed + distance2 / unloaded_speed +
  distance3 / loaded_speed + distance4 / unloaded_speed = 39 := by
  sorry

end tobys_journey_l281_28198


namespace lucys_cookies_l281_28178

/-- Lucy's grocery shopping problem -/
theorem lucys_cookies (total_packs cake_packs cookie_packs : ℕ) : 
  total_packs = 27 → cake_packs = 4 → total_packs = cookie_packs + cake_packs → cookie_packs = 23 := by
  sorry

end lucys_cookies_l281_28178


namespace wireless_mice_ratio_l281_28117

/-- Proves that the ratio of wireless mice to total mice sold is 1:2 -/
theorem wireless_mice_ratio (total_mice : ℕ) (optical_mice : ℕ) (trackball_mice : ℕ) :
  total_mice = 80 →
  optical_mice = total_mice / 4 →
  trackball_mice = 20 →
  let wireless_mice := total_mice - (optical_mice + trackball_mice)
  (wireless_mice : ℚ) / total_mice = 1 / 2 := by
  sorry

#check wireless_mice_ratio

end wireless_mice_ratio_l281_28117


namespace sequence_is_arithmetic_progression_first_term_is_three_common_difference_is_six_l281_28186

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) : ℝ := 3 * n^2

/-- The n-th term of the sequence -/
def u (n : ℕ) : ℝ := S n - S (n-1)

theorem sequence_is_arithmetic_progression :
  ∃ (a d : ℝ), ∀ n : ℕ, u n = a + (n - 1) * d :=
sorry

theorem first_term_is_three : u 1 = 3 :=
sorry

theorem common_difference_is_six :
  ∀ n : ℕ, n > 1 → u n - u (n-1) = 6 :=
sorry

end sequence_is_arithmetic_progression_first_term_is_three_common_difference_is_six_l281_28186


namespace contest_scores_l281_28173

theorem contest_scores (n k : ℕ) (hn : n ≥ 2) :
  (∀ (i : ℕ), i ≤ k → ∃! (f : ℕ → ℕ), (∀ x, x ≤ n → f x ≤ n) ∧ 
    (∀ x y, x ≠ y → f x ≠ f y) ∧ (Finset.sum (Finset.range n) f = Finset.sum (Finset.range n) id)) →
  (∀ x, x ≤ n → k * (Finset.sum (Finset.range n) id) = 26 * n) →
  (n = 25 ∧ k = 2) ∨ (n = 12 ∧ k = 4) ∨ (n = 3 ∧ k = 13) :=
by sorry

end contest_scores_l281_28173


namespace at_least_one_not_greater_than_negative_four_l281_28119

theorem at_least_one_not_greater_than_negative_four
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
  sorry

end at_least_one_not_greater_than_negative_four_l281_28119


namespace dice_product_composite_probability_l281_28171

def num_dice : ℕ := 6
def num_sides : ℕ := 8

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

def total_outcomes : ℕ := num_sides ^ num_dice

def non_composite_outcomes : ℕ := 25

theorem dice_product_composite_probability :
  (total_outcomes - non_composite_outcomes) / total_outcomes = 262119 / 262144 :=
sorry

end dice_product_composite_probability_l281_28171


namespace alcohol_mixture_proof_l281_28149

/-- Proves that adding 3.6 liters of pure alcohol to a 6-liter solution
    that is 20% alcohol results in a solution that is 50% alcohol. -/
theorem alcohol_mixture_proof
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_alcohol : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.20)
  (h3 : added_alcohol = 3.6)
  (h4 : final_concentration = 0.50) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end alcohol_mixture_proof_l281_28149


namespace integer_root_values_l281_28112

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 9 = 0) ↔ 
  a ∈ ({-109, -21, -13, 3, 11, 53} : Set ℤ) :=
sorry

end integer_root_values_l281_28112


namespace unique_solution_cubic_rational_equation_l281_28146

theorem unique_solution_cubic_rational_equation :
  ∃! x : ℝ, (x^3 - 3*x^2 + 2*x)/(x^2 + 2*x + 1) + 2*x = -8 := by
  sorry

end unique_solution_cubic_rational_equation_l281_28146


namespace total_wristbands_distributed_l281_28155

/-- Represents the number of wristbands given to each spectator -/
def wristbands_per_spectator : ℕ := 2

/-- Represents the total number of wristbands distributed -/
def total_wristbands : ℕ := 125

/-- Theorem stating that the total number of wristbands distributed is 125 -/
theorem total_wristbands_distributed :
  total_wristbands = 125 := by sorry

end total_wristbands_distributed_l281_28155


namespace f_sum_zero_four_l281_28175

def f (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem f_sum_zero_four (a b c d : ℝ) :
  f a b c d 1 = 1 →
  f a b c d 2 = 2 →
  f a b c d 3 = 3 →
  f a b c d 0 + f a b c d 4 = 28 :=
by
  sorry

end f_sum_zero_four_l281_28175


namespace infinite_solutions_condition_l281_28158

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end infinite_solutions_condition_l281_28158


namespace polygon_division_euler_characteristic_l281_28196

/-- A polygon division represents the result of dividing a polygon into several polygons. -/
structure PolygonDivision where
  p : ℕ  -- number of resulting polygons
  q : ℕ  -- number of segments that are the sides of these polygons
  r : ℕ  -- number of points that are their vertices

/-- The Euler characteristic of a polygon division is always 1. -/
theorem polygon_division_euler_characteristic (d : PolygonDivision) : 
  d.p - d.q + d.r = 1 := by
  sorry

end polygon_division_euler_characteristic_l281_28196


namespace players_satisfy_distances_l281_28123

/-- Represents the positions of four players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two player positions -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required distances between players -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem players_satisfy_distances : 
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end players_satisfy_distances_l281_28123


namespace fraction_equality_l281_28189

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7^2) :
  d / a = 1 / 122.5 := by
  sorry

end fraction_equality_l281_28189


namespace sphere_volume_l281_28187

theorem sphere_volume (A : Real) (V : Real) :
  A = 9 * Real.pi →  -- area of the main view (circle)
  V = (4 / 3) * Real.pi * (3 ^ 3) →  -- volume formula with radius 3
  V = 36 * Real.pi :=  -- expected volume
by
  sorry

end sphere_volume_l281_28187


namespace log_equation_solution_l281_28130

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end log_equation_solution_l281_28130


namespace prob_select_AB_l281_28131

/-- The number of employees -/
def total_employees : ℕ := 4

/-- The number of employees to be selected -/
def selected_employees : ℕ := 2

/-- The probability of selecting at least one of A and B -/
def prob_at_least_one_AB : ℚ := 5/6

/-- Theorem stating the probability of selecting at least one of A and B -/
theorem prob_select_AB : 
  1 - (Nat.choose (total_employees - 2) selected_employees : ℚ) / (Nat.choose total_employees selected_employees : ℚ) = prob_at_least_one_AB :=
sorry

end prob_select_AB_l281_28131


namespace cone_volume_l281_28153

/-- Given a cone with base radius 1 and lateral area 2π, its volume is (√3/3)π -/
theorem cone_volume (r h : ℝ) : 
  r = 1 → 
  π * r * (r^2 + h^2).sqrt = 2 * π → 
  (1/3) * π * r^2 * h = (Real.sqrt 3 / 3) * π :=
by sorry

end cone_volume_l281_28153


namespace unique_solution_l281_28113

/-- Represents the ages of two people satisfying the given conditions -/
structure AgesPair where
  first : ℕ
  second : ℕ
  sum_is_35 : first + second = 35
  age_relation : 2 * first - second = second - first

/-- The unique solution to the age problem -/
theorem unique_solution : ∃! (ages : AgesPair), ages.first = 20 ∧ ages.second = 15 := by
  sorry

end unique_solution_l281_28113


namespace remaining_roots_equation_l281_28138

theorem remaining_roots_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) :
  ∃ x₁ : ℝ, (x₁^2 + a*x₁ + b*c = 0 ∧ x₁^2 + b*x₁ + c*a = 0) →
  ∃ x₂ x₃ : ℝ, x₂ ≠ x₁ ∧ x₃ ≠ x₁ ∧ x₂^2 + c*x₂ + a*b = 0 ∧ x₃^2 + c*x₃ + a*b = 0 :=
sorry

end remaining_roots_equation_l281_28138


namespace least_number_of_pennies_l281_28160

theorem least_number_of_pennies :
  ∃ (p : ℕ), p > 0 ∧ p % 7 = 3 ∧ p % 4 = 1 ∧
  ∀ (q : ℕ), q > 0 ∧ q % 7 = 3 ∧ q % 4 = 1 → p ≤ q :=
by
  -- The proof goes here
  sorry

end least_number_of_pennies_l281_28160


namespace cara_in_middle_groups_l281_28193

theorem cara_in_middle_groups (n : ℕ) (h : n = 6) : Nat.choose n 2 = 15 := by
  sorry

end cara_in_middle_groups_l281_28193


namespace smallest_b_value_l281_28165

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end smallest_b_value_l281_28165


namespace root_difference_implies_k_value_l281_28114

theorem root_difference_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
  (r^2 + k*r + 12 = 0) ∧ (s^2 + k*s + 12 = 0) →
  ((r+3)^2 - k*(r+3) + 12 = 0) ∧ ((s+3)^2 - k*(s+3) + 12 = 0) →
  k = 3 := by
sorry

end root_difference_implies_k_value_l281_28114


namespace lucy_calculation_l281_28115

theorem lucy_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 13) 
  (h2 : x - y - z = -1) : 
  x - y = 6 := by sorry

end lucy_calculation_l281_28115


namespace tan_theta_two_implies_expression_equals_negative_two_l281_28129

theorem tan_theta_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (π/2 + θ) + Real.sin (π + θ)) = -2 := by
  sorry

end tan_theta_two_implies_expression_equals_negative_two_l281_28129


namespace one_nonnegative_solution_l281_28121

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x :=
sorry

end one_nonnegative_solution_l281_28121


namespace wizard_hat_theorem_l281_28144

/-- Represents a strategy for the wizard hat problem -/
def Strategy : Type := Unit

/-- Represents the outcome of applying a strategy -/
def Outcome (n : ℕ) : Type := Fin n → Bool

/-- A wizard can see hats in front but not their own -/
axiom can_see_forward (n : ℕ) (i : Fin n) : ∀ j : Fin n, i < j → Prop

/-- Each wizard says a unique number between 1 and 1001 -/
axiom unique_numbers (n : ℕ) (outcome : Outcome n) : 
  ∀ i j : Fin n, i ≠ j → outcome i ≠ outcome j

/-- Wizards speak from back to front -/
axiom speak_order (n : ℕ) (i j : Fin n) : i < j → Prop

/-- Applying a strategy produces an outcome -/
def apply_strategy (n : ℕ) (s : Strategy) : Outcome n := sorry

/-- Counts the number of correct identifications in an outcome -/
def count_correct (n : ℕ) (outcome : Outcome n) : ℕ := sorry

theorem wizard_hat_theorem (n : ℕ) (h : n > 1000) :
  ∃ (s : Strategy), 
    (count_correct n (apply_strategy n s) > 500) ∧ 
    (count_correct n (apply_strategy n s) ≥ 999) := by
  sorry

end wizard_hat_theorem_l281_28144


namespace cousins_ages_sum_l281_28166

theorem cousins_ages_sum : ∃ (a b c d : ℕ),
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧  -- single-digit
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧      -- positive
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct
  ((a * b = 20 ∧ c * d = 21) ∨ (a * c = 20 ∧ b * d = 21) ∨ 
   (a * d = 20 ∧ b * c = 21) ∨ (b * c = 20 ∧ a * d = 21) ∨ 
   (b * d = 20 ∧ a * c = 21) ∧ (c * d = 20 ∧ a * b = 21)) ∧
  (a + b + c + d = 19) :=
by sorry

end cousins_ages_sum_l281_28166
