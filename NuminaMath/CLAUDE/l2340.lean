import Mathlib

namespace NUMINAMATH_CALUDE_team_size_l2340_234074

/-- A soccer team with goalies, defenders, midfielders, and strikers -/
structure SoccerTeam where
  goalies : ℕ
  defenders : ℕ
  midfielders : ℕ
  strikers : ℕ

/-- The total number of players in a soccer team -/
def totalPlayers (team : SoccerTeam) : ℕ :=
  team.goalies + team.defenders + team.midfielders + team.strikers

/-- Theorem stating the total number of players in the given team -/
theorem team_size (team : SoccerTeam) 
  (h1 : team.goalies = 3)
  (h2 : team.defenders = 10)
  (h3 : team.midfielders = 2 * team.defenders)
  (h4 : team.strikers = 7) :
  totalPlayers team = 40 := by
  sorry

#eval totalPlayers { goalies := 3, defenders := 10, midfielders := 20, strikers := 7 }

end NUMINAMATH_CALUDE_team_size_l2340_234074


namespace NUMINAMATH_CALUDE_sequence_problem_l2340_234092

theorem sequence_problem (a : Fin 100 → ℝ) 
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2340_234092


namespace NUMINAMATH_CALUDE_smallest_odd_divisor_of_difference_of_squares_l2340_234067

theorem smallest_odd_divisor_of_difference_of_squares (m n : ℕ) : 
  Odd m → Odd n → n < m → 
  (∃ (k : ℕ), ∀ (a b : ℕ), Odd a → Odd b → b < a → k ∣ (a^2 - b^2)) → 
  (∃ (d : ℕ), Odd d ∧ d ∣ (m^2 - n^2) ∧ 
    ∀ (e : ℕ), Odd e → e ∣ (m^2 - n^2) → d ≤ e) → 
  ∃ (d : ℕ), d = 1 ∧ Odd d ∧ d ∣ (m^2 - n^2) ∧ 
    ∀ (e : ℕ), Odd e → e ∣ (m^2 - n^2) → d ≤ e :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_divisor_of_difference_of_squares_l2340_234067


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2340_234023

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = -3 * x^2 + 9 * x + 54) ∧
    q (-3) = 0 ∧
    q 6 = 0 ∧
    q 0 = -54 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2340_234023


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2340_234057

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to reflect a point about the y-axis -/
def reflectAboutYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Function to get the equation of a line given two points -/
def lineFromPoints (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p1.y * p2.x - p1.x * p2.y }

/-- Theorem stating the equation of the reflected ray -/
theorem reflected_ray_equation
  (start : Point)
  (slope : ℝ)
  (h_start : start = { x := 2, y := 3 })
  (h_slope : slope = 1/2) :
  let intersect : Point := { x := 0, y := 2 }
  let reflected_start : Point := reflectAboutYAxis start
  let reflected_line : Line := lineFromPoints intersect reflected_start
  reflected_line = { a := 1, b := 2, c := -4 } :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2340_234057


namespace NUMINAMATH_CALUDE_percentage_difference_l2340_234060

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) :
  (x - y) / x * 100 = 91.67 :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2340_234060


namespace NUMINAMATH_CALUDE_inequality_holds_iff_equal_l2340_234030

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The inequality holds for all real α and β iff m = n -/
theorem inequality_holds_iff_equal (m n : ℕ+) : 
  (∀ α β : ℝ, floor ((m + n : ℝ) * α) + floor ((m + n : ℝ) * β) ≥ 
    floor (m * α) + floor (n * β) + floor (n * (α + β))) ↔ m = n := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_equal_l2340_234030


namespace NUMINAMATH_CALUDE_barn_size_calculation_barn_size_is_1000_l2340_234084

/-- Given a property with a house and a barn, calculate the size of the barn. -/
theorem barn_size_calculation (price_per_sqft : ℝ) (house_size : ℝ) (total_value : ℝ) : ℝ :=
  let house_value := price_per_sqft * house_size
  let barn_value := total_value - house_value
  barn_value / price_per_sqft

/-- The size of the barn is 1000 square feet. -/
theorem barn_size_is_1000 :
  barn_size_calculation 98 2400 333200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_barn_size_calculation_barn_size_is_1000_l2340_234084


namespace NUMINAMATH_CALUDE_probability_one_white_ball_l2340_234038

/-- The probability of drawing exactly one white ball when drawing three balls from a bag containing
    four white balls and three black balls of the same size. -/
theorem probability_one_white_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 4)
  (h3 : black_balls = 3)
  (h4 : total_balls > 0) :
  (white_balls : ℚ) / total_balls * 
  (black_balls : ℚ) / (total_balls - 1) * 
  (black_balls - 1 : ℚ) / (total_balls - 2) * 3 = 12 / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_one_white_ball_l2340_234038


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l2340_234082

/-- The minimum value of a for which x^2 + ax + 1 ≥ 0 holds for all x ∈ (0, 1] is -2 -/
theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l2340_234082


namespace NUMINAMATH_CALUDE_bananas_per_friend_l2340_234001

/-- Given Virginia has 40 bananas and shares them equally among 40 friends,
    prove that each friend receives 1 banana. -/
theorem bananas_per_friend (total_bananas : ℕ) (num_friends : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_friends = 40) :
  total_bananas / num_friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_friend_l2340_234001


namespace NUMINAMATH_CALUDE_equation_solution_l2340_234073

theorem equation_solution : 
  ∃! x : ℚ, (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2)) ∧ x = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2340_234073


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l2340_234099

theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  -- a, b, c form a geometric sequence
  (b ^ 2 = a * c) ∧
  -- Given trigonometric ratios
  (Real.sin B = 5 / 13) ∧
  (Real.cos B = 12 / (a * c)) →
  -- Conclusion
  a + c = 3 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_l2340_234099


namespace NUMINAMATH_CALUDE_min_draws_for_sum_30_l2340_234010

-- Define the set of integers from 0 to 20
def integerSet : Set ℕ := {n : ℕ | n ≤ 20}

-- Define a function to check if two numbers in a list sum to 30
def hasPairSum30 (list : List ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ list ∧ b ∈ list ∧ a ≠ b ∧ a + b = 30

-- Theorem: The minimum number of integers to guarantee a pair summing to 30 is 10
theorem min_draws_for_sum_30 :
  ∀ (drawn : List ℕ),
    (∀ n ∈ drawn, n ∈ integerSet) →
    (drawn.length ≥ 10 → hasPairSum30 drawn) ∧
    (∃ subset : List ℕ, subset.length = 9 ∧ ∀ n ∈ subset, n ∈ integerSet ∧ ¬hasPairSum30 subset) :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_sum_30_l2340_234010


namespace NUMINAMATH_CALUDE_fashion_show_evening_wear_correct_evening_wear_count_l2340_234039

theorem fashion_show_evening_wear (num_models : ℕ) (bathing_suits_per_model : ℕ) 
  (runway_time : ℕ) (total_show_time : ℕ) : ℕ :=
  let total_bathing_suit_trips := num_models * bathing_suits_per_model
  let bathing_suit_time := total_bathing_suit_trips * runway_time
  let evening_wear_time := total_show_time - bathing_suit_time
  let evening_wear_trips := evening_wear_time / runway_time
  evening_wear_trips / num_models

theorem correct_evening_wear_count : 
  fashion_show_evening_wear 6 2 2 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fashion_show_evening_wear_correct_evening_wear_count_l2340_234039


namespace NUMINAMATH_CALUDE_abc_equality_l2340_234085

theorem abc_equality (a b c : ℕ) 
  (h : ∀ n : ℕ, (a * b * c)^n ∣ ((a^n - 1) * (b^n - 1) * (c^n - 1) + 1)^3) : 
  a = b ∧ b = c :=
sorry

end NUMINAMATH_CALUDE_abc_equality_l2340_234085


namespace NUMINAMATH_CALUDE_kitten_price_l2340_234088

theorem kitten_price (kitten_count puppy_count : ℕ) 
                     (puppy_price total_earnings : ℚ) :
  kitten_count = 2 →
  puppy_count = 1 →
  puppy_price = 5 →
  total_earnings = 17 →
  ∃ kitten_price : ℚ, 
    kitten_price * kitten_count + puppy_price * puppy_count = total_earnings ∧
    kitten_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_kitten_price_l2340_234088


namespace NUMINAMATH_CALUDE_profit_percentage_l2340_234016

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.75 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2340_234016


namespace NUMINAMATH_CALUDE_victoria_remaining_balance_l2340_234055

/-- Calculates Victoria's remaining balance after shopping --/
theorem victoria_remaining_balance :
  let initial_amount : ℕ := 500
  let rice_price : ℕ := 20
  let rice_quantity : ℕ := 2
  let wheat_price : ℕ := 25
  let wheat_quantity : ℕ := 3
  let soda_price : ℕ := 150
  let soda_quantity : ℕ := 1
  let total_spent : ℕ := rice_price * rice_quantity + wheat_price * wheat_quantity + soda_price * soda_quantity
  let remaining_balance : ℕ := initial_amount - total_spent
  remaining_balance = 235 := by
  sorry

end NUMINAMATH_CALUDE_victoria_remaining_balance_l2340_234055


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_implies_k_ge_160_l2340_234050

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Theorem statement
theorem f_monotonically_decreasing_implies_k_ge_160 :
  ∀ k : ℝ, monotonically_decreasing_on (f k) 5 20 → k ≥ 160 :=
sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_implies_k_ge_160_l2340_234050


namespace NUMINAMATH_CALUDE_ratio_transformation_l2340_234009

theorem ratio_transformation (a c : ℝ) (h : c ≠ 0) :
  (3 * a) / (c / 3) = 9 * (a / c) := by sorry

end NUMINAMATH_CALUDE_ratio_transformation_l2340_234009


namespace NUMINAMATH_CALUDE_card_sum_problem_l2340_234059

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_problem_l2340_234059


namespace NUMINAMATH_CALUDE_quadratic_equation_c_value_l2340_234033

theorem quadratic_equation_c_value : 
  ∀ c : ℝ, 
  (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 40) / 4 ∨ x = (-8 - Real.sqrt 40) / 4) → 
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_c_value_l2340_234033


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l2340_234052

theorem mean_equality_implies_z (z : ℝ) : 
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l2340_234052


namespace NUMINAMATH_CALUDE_greatest_number_under_150_with_odd_factors_l2340_234008

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_under_150_with_odd_factors : 
  (∀ n : ℕ, n < 150 ∧ has_odd_number_of_factors n → n ≤ 144) ∧ 
  144 < 150 ∧ 
  has_odd_number_of_factors 144 :=
sorry

end NUMINAMATH_CALUDE_greatest_number_under_150_with_odd_factors_l2340_234008


namespace NUMINAMATH_CALUDE_fourth_group_frequency_l2340_234032

/-- Given a set of data with 50 items divided into 5 groups, prove that the frequency of the fourth group is 12 -/
theorem fourth_group_frequency
  (total_items : ℕ)
  (num_groups : ℕ)
  (freq_group1 : ℕ)
  (freq_group2 : ℕ)
  (freq_group3 : ℕ)
  (freq_group5 : ℕ)
  (h_total : total_items = 50)
  (h_groups : num_groups = 5)
  (h_freq1 : freq_group1 = 10)
  (h_freq2 : freq_group2 = 8)
  (h_freq3 : freq_group3 = 11)
  (h_freq5 : freq_group5 = 9) :
  total_items - (freq_group1 + freq_group2 + freq_group3 + freq_group5) = 12 :=
by sorry

end NUMINAMATH_CALUDE_fourth_group_frequency_l2340_234032


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2340_234019

theorem triangle_angle_c (a b : ℝ) (A B C : ℝ) :
  a = 2 →
  A = π / 6 →
  b = 2 * Real.sqrt 3 →
  a * Real.sin B = b * Real.sin A →
  A + B + C = π →
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2340_234019


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_specific_circle_equation_l2340_234036

/-- Given two points A and B as the endpoints of a circle's diameter, 
    prove that the equation of the circle is (x-h)^2 + (y-k)^2 = r^2,
    where (h,k) is the midpoint of AB and r is half the distance between A and B. -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := Real.sqrt (((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4)
  ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x - x₁)^2 + (y - y₁)^2) * ((x - x₂)^2 + (y - y₂)^2) = 
    ((x - x₁)^2 + (y - y₁)^2 + (x - x₂)^2 + (y - y₂)^2)^2 / 4 :=
by sorry

/-- The equation of the circle with diameter endpoints A(4,9) and B(6,3) is (x-5)^2 + (y-6)^2 = 10 -/
theorem specific_circle_equation : 
  ∀ (x y : ℝ), (x - 5)^2 + (y - 6)^2 = 10 ↔ 
    ((x - 4)^2 + (y - 9)^2) * ((x - 6)^2 + (y - 3)^2) = 
    ((x - 4)^2 + (y - 9)^2 + (x - 6)^2 + (y - 3)^2)^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_specific_circle_equation_l2340_234036


namespace NUMINAMATH_CALUDE_concert_duration_in_minutes_l2340_234077

/-- Converts hours and minutes to total minutes -/
def hours_minutes_to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

/-- Theorem: A concert lasting 7 hours and 45 minutes is 465 minutes long -/
theorem concert_duration_in_minutes : 
  hours_minutes_to_minutes 7 45 = 465 := by
  sorry

end NUMINAMATH_CALUDE_concert_duration_in_minutes_l2340_234077


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2340_234026

/-- The imaginary unit -/
def i : ℂ := Complex.I

theorem complex_simplification_and_multiplication :
  ((6 - 3 * i) - (2 - 5 * i)) * (1 + 2 * i) = 10 * i := by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2340_234026


namespace NUMINAMATH_CALUDE_weight_of_a_l2340_234043

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 75 kg -/
theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l2340_234043


namespace NUMINAMATH_CALUDE_car_distance_proof_l2340_234051

/-- Calculates the distance traveled by a car given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The speed of the car in miles per hour -/
def car_speed : ℝ := 80

/-- The time the car traveled in hours -/
def travel_time : ℝ := 4.5

theorem car_distance_proof : 
  distance_traveled car_speed travel_time = 360 := by sorry

end NUMINAMATH_CALUDE_car_distance_proof_l2340_234051


namespace NUMINAMATH_CALUDE_total_jogging_distance_l2340_234037

/-- The total distance jogged over three days is the sum of the distances jogged each day. -/
theorem total_jogging_distance 
  (monday_distance tuesday_distance wednesday_distance : ℕ) 
  (h1 : monday_distance = 2)
  (h2 : tuesday_distance = 5)
  (h3 : wednesday_distance = 9) :
  monday_distance + tuesday_distance + wednesday_distance = 16 := by
sorry

end NUMINAMATH_CALUDE_total_jogging_distance_l2340_234037


namespace NUMINAMATH_CALUDE_paco_cookies_l2340_234087

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 34

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := 97

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 56

/-- The number of sweet cookies Paco had left after eating -/
def remaining_sweet_cookies : ℕ := 19

theorem paco_cookies : initial_sweet_cookies = eaten_sweet_cookies + remaining_sweet_cookies :=
by sorry

end NUMINAMATH_CALUDE_paco_cookies_l2340_234087


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2340_234049

theorem rectangle_measurement_error (L W : ℝ) (p : ℝ) (h_positive : L > 0 ∧ W > 0) :
  (1.05 * L) * ((1 - p) * W) = (1 + 0.008) * (L * W) → p = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2340_234049


namespace NUMINAMATH_CALUDE_president_and_committee_count_l2340_234066

/-- The number of ways to choose a president and a 2-person committee -/
def choose_president_and_committee (total_people : ℕ) (people_over_30 : ℕ) : ℕ :=
  total_people * (people_over_30 * (people_over_30 - 1) / 2 + 
  (total_people - people_over_30) * people_over_30 * (people_over_30 - 1) / 2)

/-- Theorem stating the number of ways to choose a president and committee -/
theorem president_and_committee_count :
  choose_president_and_committee 10 6 = 120 := by sorry

end NUMINAMATH_CALUDE_president_and_committee_count_l2340_234066


namespace NUMINAMATH_CALUDE_inequality_solution_l2340_234056

theorem inequality_solution (x : ℝ) : 
  x ≥ 0 → (2021 * (x^2020)^(1/202) - 1 ≥ 2020*x ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2340_234056


namespace NUMINAMATH_CALUDE_new_members_weight_l2340_234045

/-- Theorem: Calculate the combined weight of new group members -/
theorem new_members_weight (original_size : ℕ) (weight_increase : ℝ) 
  (original_member1 original_member2 original_member3 : ℝ) :
  original_size = 8 →
  weight_increase = 4.2 →
  original_member1 = 60 →
  original_member2 = 75 →
  original_member3 = 65 →
  (original_member1 + original_member2 + original_member3 + 
    original_size * weight_increase) = 233.6 := by
  sorry

end NUMINAMATH_CALUDE_new_members_weight_l2340_234045


namespace NUMINAMATH_CALUDE_exponential_inequality_l2340_234042

theorem exponential_inequality (x : ℝ) : 
  (1/4 : ℝ)^(x^2 - 8) > (4 : ℝ)^(-2*x) ↔ -2 < x ∧ x < 4 := by
sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2340_234042


namespace NUMINAMATH_CALUDE_area_inequalities_l2340_234027

/-- An acute-angled triangle -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

/-- Area of the orthic triangle -/
def orthic_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the tangential triangle -/
def tangential_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the contact triangle -/
def contact_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the excentral triangle -/
def excentral_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the medial triangle -/
def medial_area (t : AcuteTriangle) : ℝ := sorry

/-- A triangle is equilateral if all its angles are equal -/
def is_equilateral (t : AcuteTriangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

/-- The main theorem -/
theorem area_inequalities (t : AcuteTriangle) :
  orthic_area t ≤ tangential_area t ∧
  tangential_area t = contact_area t ∧
  contact_area t ≤ excentral_area t ∧
  excentral_area t ≤ medial_area t ∧
  (orthic_area t = medial_area t ↔ is_equilateral t) :=
sorry

end NUMINAMATH_CALUDE_area_inequalities_l2340_234027


namespace NUMINAMATH_CALUDE_total_sales_is_28_l2340_234078

/-- The number of crates of eggs Gabrielle sells on Monday -/
def monday_sales : ℕ := 5

/-- The number of crates of eggs Gabrielle sells on Tuesday -/
def tuesday_sales : ℕ := 2 * monday_sales

/-- The number of crates of eggs Gabrielle sells on Wednesday -/
def wednesday_sales : ℕ := tuesday_sales - 2

/-- The number of crates of eggs Gabrielle sells on Thursday -/
def thursday_sales : ℕ := tuesday_sales / 2

/-- The total number of crates of eggs Gabrielle sells over 4 days -/
def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales

theorem total_sales_is_28 : total_sales = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_is_28_l2340_234078


namespace NUMINAMATH_CALUDE_divisibility_and_sum_of_primes_l2340_234012

theorem divisibility_and_sum_of_primes :
  ∃ (p₁ p₂ p₃ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    (p₁ ∣ (2^10 - 1)) ∧ (p₂ ∣ (2^10 - 1)) ∧ (p₃ ∣ (2^10 - 1)) ∧
    (∀ q : ℕ, Prime q → (q ∣ (2^10 - 1)) → (q = p₁ ∨ q = p₂ ∨ q = p₃)) ∧
    p₁ + p₂ + p₃ = 45 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_sum_of_primes_l2340_234012


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2340_234072

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z - 4) = 3 + 2 * Complex.I) : 
  z.im = 3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2340_234072


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l2340_234054

theorem angle_sum_around_point (y : ℝ) : 
  y > 0 ∧ 150 + y + y = 360 → y = 105 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l2340_234054


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l2340_234089

/-- Given a function f(x) = x + 1/x - 2 and a real number a such that f(a) = 3,
    prove that f(-a) = -7. -/
theorem function_value_at_negative_a 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x + 1/x - 2) 
  (h2 : f a = 3) : 
  f (-a) = -7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l2340_234089


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l2340_234011

def A : Set ℤ := {0, 1}
def B (a : ℤ) : Set ℤ := {-1, 0, a+3}

theorem subset_implies_a_value (h : A ⊆ B a) : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l2340_234011


namespace NUMINAMATH_CALUDE_apple_cost_price_l2340_234086

-- Define the selling price
def selling_price : ℚ := 19

-- Define the ratio of selling price to cost price
def selling_to_cost_ratio : ℚ := 5/6

-- Theorem statement
theorem apple_cost_price :
  ∃ (cost_price : ℚ), 
    cost_price = selling_price / selling_to_cost_ratio ∧ 
    cost_price = 114/5 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l2340_234086


namespace NUMINAMATH_CALUDE_equality_of_sets_l2340_234070

theorem equality_of_sets (x y a : ℝ) : 
  (3 * x^2 = x^2 + x^2 + x^2) ∧ 
  ((x - y)^2 = (y - x)^2) ∧ 
  ((a^2)^3 = (a^3)^2) := by
sorry

end NUMINAMATH_CALUDE_equality_of_sets_l2340_234070


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l2340_234053

theorem average_of_a_and_b (a b c : ℝ) (h1 : (b + c) / 2 = 90) (h2 : c - a = 90) : 
  (a + b) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l2340_234053


namespace NUMINAMATH_CALUDE_perfect_squares_identification_l2340_234063

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def option_A : ℕ := 3^4 * 4^5 * 7^7
def option_B : ℕ := 3^6 * 4^4 * 7^6
def option_C : ℕ := 3^5 * 4^6 * 7^5
def option_D : ℕ := 3^4 * 4^7 * 7^4
def option_E : ℕ := 3^6 * 4^6 * 7^6

theorem perfect_squares_identification :
  ¬(is_perfect_square option_A) ∧
  (is_perfect_square option_B) ∧
  ¬(is_perfect_square option_C) ∧
  (is_perfect_square option_D) ∧
  (is_perfect_square option_E) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_identification_l2340_234063


namespace NUMINAMATH_CALUDE_floor_sqrt_245_l2340_234000

theorem floor_sqrt_245 : ⌊Real.sqrt 245⌋ = 15 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_245_l2340_234000


namespace NUMINAMATH_CALUDE_lillys_daily_savings_l2340_234020

/-- Proves that the daily savings amount is $2 given the conditions of Lilly's flower-buying plan for Maria's birthday. -/
theorem lillys_daily_savings 
  (saving_period : ℕ) 
  (flower_cost : ℚ) 
  (total_flowers : ℕ) 
  (h1 : saving_period = 22)
  (h2 : flower_cost = 4)
  (h3 : total_flowers = 11) : 
  (total_flowers : ℚ) * flower_cost / saving_period = 2 := by
  sorry

end NUMINAMATH_CALUDE_lillys_daily_savings_l2340_234020


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2340_234096

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2 * n + 1) = 225 := by sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2340_234096


namespace NUMINAMATH_CALUDE_min_xy_value_l2340_234093

theorem min_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1 / (2 + x) + 1 / (2 + y) = 1 / 3) :
  ∀ z, x * y ≥ z → z ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l2340_234093


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2340_234017

theorem sqrt_expression_equality : 
  |Real.sqrt 2 - Real.sqrt 3| - Real.sqrt 4 + Real.sqrt 2 * (Real.sqrt 2 + 1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2340_234017


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_units_digit_zero_l2340_234075

theorem units_digit_of_quotient (n : ℕ) : 
  (7^n + 4^n) % 9 = 2 :=
sorry

theorem units_digit_zero : 
  (7^2023 + 4^2023) / 9 % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_units_digit_zero_l2340_234075


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l2340_234062

theorem multiples_of_four_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ 100 < n ∧ n < 350) (Finset.range 350)).card = 62 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l2340_234062


namespace NUMINAMATH_CALUDE_expansion_terms_count_l2340_234076

/-- The number of terms in the expansion of a product of two polynomials with distinct variables -/
def num_terms_in_expansion (n m : ℕ) : ℕ := n * m

theorem expansion_terms_count : 
  let first_factor_terms : ℕ := 3
  let second_factor_terms : ℕ := 6
  num_terms_in_expansion first_factor_terms second_factor_terms = 18 := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l2340_234076


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l2340_234025

theorem sphere_radius_from_surface_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) :
  A = 4 * Real.pi * r^2 → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l2340_234025


namespace NUMINAMATH_CALUDE_digit_count_proof_l2340_234068

theorem digit_count_proof (total_count : ℕ) (available_digits : ℕ) 
  (h1 : total_count = 28672) 
  (h2 : available_digits = 8) : 
  ∃ n : ℕ, available_digits ^ n = total_count ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_proof_l2340_234068


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2340_234044

/-- Given a principal amount and an interest rate, if the simple interest for 2 years is 320
    and the compound interest for 2 years is 340, then the interest rate is 12.5% per annum. -/
theorem interest_rate_calculation (P R : ℝ) 
  (h_simple : (P * R * 2) / 100 = 320)
  (h_compound : P * ((1 + R / 100)^2 - 1) = 340) :
  R = 12.5 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2340_234044


namespace NUMINAMATH_CALUDE_ratio_of_segments_l2340_234071

/-- Given collinear points A, B, C in the Cartesian plane where:
    A = (a, 0) lies on the x-axis
    B lies on the line y = x
    C lies on the line y = 2x
    AB/BC = 2
    D = (a, a)
    E is the second intersection of the circumcircle of triangle ADC with y = x
    F is the intersection of ray AE with y = 2x
    Prove that AE/EF = √2/2 -/
theorem ratio_of_segments (a : ℝ) : ∃ (B C E F : ℝ × ℝ),
  let A := (a, 0)
  let D := (a, a)
  -- B lies on y = x
  B.2 = B.1 ∧
  -- C lies on y = 2x
  C.2 = 2 * C.1 ∧
  -- AB/BC = 2
  (((B.1 - A.1)^2 + (B.2 - A.2)^2) : ℝ) / ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 4 ∧
  -- E lies on y = x
  E.2 = E.1 ∧
  -- E is on the circumcircle of ADC
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  (E.1 - C.1)^2 + (E.2 - C.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
  -- F lies on y = 2x
  F.2 = 2 * F.1 ∧
  -- F lies on ray AE
  ∃ (t : ℝ), t > 0 ∧ F.1 - A.1 = t * (E.1 - A.1) ∧ F.2 - A.2 = t * (E.2 - A.2) →
  -- Conclusion: AE/EF = √2/2
  (((E.1 - A.1)^2 + (E.2 - A.2)^2) : ℝ) / ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l2340_234071


namespace NUMINAMATH_CALUDE_chord_length_polar_l2340_234090

/-- Chord length intercepted by a line on a circle in polar coordinates -/
theorem chord_length_polar (ρ θ : ℝ) (h1 : ρ = 4 * Real.sin θ) (h2 : Real.tan θ = 1/2) :
  ρ = 4 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l2340_234090


namespace NUMINAMATH_CALUDE_round_0_6457_to_hundredth_l2340_234018

/-- Rounds a number to the nearest hundredth -/
def roundToHundredth (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

/-- The theorem states that rounding 0.6457 to the nearest hundredth results in 0.65 -/
theorem round_0_6457_to_hundredth :
  roundToHundredth (6457 / 10000) = 65 / 100 := by sorry

end NUMINAMATH_CALUDE_round_0_6457_to_hundredth_l2340_234018


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2340_234091

theorem imaginary_part_of_complex_number (z : ℂ) : z = (3 - 2 * Complex.I^2) / (1 + Complex.I) → z.im = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2340_234091


namespace NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l2340_234083

/-- Charlie's schedule cycle length -/
def charlie_cycle : Nat := 6

/-- Dana's schedule cycle length -/
def dana_cycle : Nat := 10

/-- Number of days in the period -/
def total_days : Nat := 1200

/-- Number of rest days in Charlie's cycle -/
def charlie_rest_days : Nat := 2

/-- Number of rest days in Dana's cycle -/
def dana_rest_days : Nat := 1

/-- Function to calculate the number of coinciding rest days -/
def coinciding_rest_days (charlie_cycle dana_cycle total_days : Nat) : Nat :=
  sorry

theorem coinciding_rest_days_theorem :
  coinciding_rest_days charlie_cycle dana_cycle total_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l2340_234083


namespace NUMINAMATH_CALUDE_triangle_problem_l2340_234048

theorem triangle_problem (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧
  ∃ (x y z : ℝ), x^2 + y^2 = z^2 ∧ 
  Set.toFinset {x, y, z} = Set.toFinset {a, b, c} :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2340_234048


namespace NUMINAMATH_CALUDE_light_source_height_l2340_234006

/-- Given a cube with edge length 3 cm, illuminated by a light source x cm directly
    above and 3 cm horizontally from a top vertex, if the shadow area outside the
    cube's base is 75 square cm, then x = 7 cm. -/
theorem light_source_height (x : ℝ) : 
  let cube_edge : ℝ := 3
  let horizontal_distance : ℝ := 3
  let shadow_area : ℝ := 75
  let total_area : ℝ := cube_edge^2 + shadow_area
  let shadow_side : ℝ := Real.sqrt total_area
  let height_increase : ℝ := shadow_side - cube_edge
  x = (cube_edge * (cube_edge + height_increase)) / height_increase → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_light_source_height_l2340_234006


namespace NUMINAMATH_CALUDE_opposite_of_two_and_two_thirds_l2340_234014

theorem opposite_of_two_and_two_thirds :
  -(2 + 2/3) = -(2 + 2/3) := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_and_two_thirds_l2340_234014


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2340_234065

/-- Given that i is the imaginary unit and zi = 2i - z, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (i : ℂ) (z : ℂ) 
  (h_i : i * i = -1) 
  (h_z : z * i = 2 * i - z) : 
  Real.sqrt 2 / 2 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2340_234065


namespace NUMINAMATH_CALUDE_construct_equilateral_triangle_l2340_234064

/-- A triangle with two 70° angles and one 40° angle -/
structure WoodenTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  two_70 : (angle1 = 70 ∧ angle2 = 70) ∨ (angle1 = 70 ∧ angle3 = 70) ∨ (angle2 = 70 ∧ angle3 = 70)
  one_40 : angle1 = 40 ∨ angle2 = 40 ∨ angle3 = 40

/-- An equilateral triangle has three 60° angles -/
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = 60 ∧ b = 60 ∧ c = 60

/-- The theorem stating that an equilateral triangle can be constructed using only the wooden triangle -/
theorem construct_equilateral_triangle (wt : WoodenTriangle) :
  ∃ a b c : ℝ, is_equilateral_triangle a b c ∧
  (∃ (n : ℕ), n > 0 ∧ a + b + c = n * (wt.angle1 + wt.angle2 + wt.angle3)) :=
sorry

end NUMINAMATH_CALUDE_construct_equilateral_triangle_l2340_234064


namespace NUMINAMATH_CALUDE_mod_thirteen_problem_l2340_234069

theorem mod_thirteen_problem (a : ℤ) 
  (h1 : 0 < a) (h2 : a < 13) 
  (h3 : (53^2017 + a) % 13 = 0) : 
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_problem_l2340_234069


namespace NUMINAMATH_CALUDE_fraction_equals_91_when_x_is_3_l2340_234002

theorem fraction_equals_91_when_x_is_3 :
  let x : ℝ := 3
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
sorry

end NUMINAMATH_CALUDE_fraction_equals_91_when_x_is_3_l2340_234002


namespace NUMINAMATH_CALUDE_horner_v4_at_2_l2340_234041

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ := 
  let v1 := x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_v4_at_2 : horner_v4 2 = 240 := by sorry

end NUMINAMATH_CALUDE_horner_v4_at_2_l2340_234041


namespace NUMINAMATH_CALUDE_cards_distribution_l2340_234034

theorem cards_distribution (total_cards : Nat) (num_people : Nat) (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  let people_with_extra := extra_cards
  num_people - people_with_extra = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2340_234034


namespace NUMINAMATH_CALUDE_exists_divisible_by_2022_l2340_234058

def concatenate_numbers (n m : ℕ) : ℕ :=
  sorry

theorem exists_divisible_by_2022 :
  ∃ n m : ℕ, n > m ∧ m ≥ 1 ∧ (concatenate_numbers n m) % 2022 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_2022_l2340_234058


namespace NUMINAMATH_CALUDE_tiles_count_theorem_l2340_234080

/-- Represents a square floor tiled with congruent square tiles -/
structure TiledSquare where
  side_length : ℕ

/-- The number of tiles along the diagonals and central line of a tiled square -/
def diagonal_and_central_count (s : TiledSquare) : ℕ :=
  3 * s.side_length - 2

/-- The total number of tiles covering the floor -/
def total_tiles (s : TiledSquare) : ℕ :=
  s.side_length ^ 2

/-- Theorem stating that if the diagonal and central count is 55, 
    then the total number of tiles is 361 -/
theorem tiles_count_theorem (s : TiledSquare) :
  diagonal_and_central_count s = 55 → total_tiles s = 361 := by
  sorry

end NUMINAMATH_CALUDE_tiles_count_theorem_l2340_234080


namespace NUMINAMATH_CALUDE_moving_circles_touch_times_l2340_234015

/-- The problem of two moving circles touching each other --/
theorem moving_circles_touch_times
  (r₁ : ℝ) (v₁ : ℝ) (d₁ : ℝ)
  (r₂ : ℝ) (v₂ : ℝ) (d₂ : ℝ)
  (h₁ : r₁ = 981)
  (h₂ : v₁ = 7)
  (h₃ : d₁ = 2442)
  (h₄ : r₂ = 980)
  (h₅ : v₂ = 5)
  (h₆ : d₂ = 1591) :
  ∃ (t₁ t₂ : ℝ),
    t₁ = 111 ∧ t₂ = 566 ∧
    (∀ t, (d₁ - v₁ * t)^2 + (d₂ - v₂ * t)^2 = (r₁ + r₂)^2 → t = t₁ ∨ t = t₂) :=
by sorry

end NUMINAMATH_CALUDE_moving_circles_touch_times_l2340_234015


namespace NUMINAMATH_CALUDE_bow_collection_problem_l2340_234035

theorem bow_collection_problem (total : ℕ) (yellow : ℕ) :
  yellow = 36 →
  (1 : ℚ) / 4 * total + (1 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + yellow = total →
  (1 : ℚ) / 6 * total = 24 := by
  sorry

end NUMINAMATH_CALUDE_bow_collection_problem_l2340_234035


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l2340_234098

theorem complex_square_root_of_negative_four :
  ∀ z : ℂ, z^2 = -4 ↔ z = 2*I ∨ z = -2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l2340_234098


namespace NUMINAMATH_CALUDE_javier_children_count_l2340_234040

/-- The number of children in Javier's household -/
def num_children : ℕ := 
  let total_legs : ℕ := 22
  let javier_wife_legs : ℕ := 2 + 2
  let dog_legs : ℕ := 2 * 4
  let cat_legs : ℕ := 1 * 4
  let remaining_legs : ℕ := total_legs - (javier_wife_legs + dog_legs + cat_legs)
  remaining_legs / 2

theorem javier_children_count : num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_javier_children_count_l2340_234040


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l2340_234047

/-- A round-robin tournament where each player plays every other player exactly once. -/
structure RoundRobinTournament where
  num_players : ℕ
  num_players_pos : 0 < num_players

/-- The number of matches in a round-robin tournament. -/
def num_matches (t : RoundRobinTournament) : ℕ := t.num_players.choose 2

theorem ten_player_tournament_matches :
  ∀ t : RoundRobinTournament, t.num_players = 10 → num_matches t = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l2340_234047


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l2340_234095

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (k : ℝ), 29 * 39 * x^4 + 4 = k * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l2340_234095


namespace NUMINAMATH_CALUDE_no_integer_solution_quadratic_l2340_234021

theorem no_integer_solution_quadratic (x : ℤ) : x^2 + 3 ≥ 2*x := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_quadratic_l2340_234021


namespace NUMINAMATH_CALUDE_edwards_spending_l2340_234031

theorem edwards_spending (initial_amount : ℚ) : 
  initial_amount - 130 - (0.25 * (initial_amount - 130)) = 270 → 
  initial_amount = 490 := by
sorry

end NUMINAMATH_CALUDE_edwards_spending_l2340_234031


namespace NUMINAMATH_CALUDE_third_artist_set_duration_l2340_234079

/-- The duration of the music festival in minutes -/
def festival_duration : ℕ := 6 * 60

/-- The duration of the first artist's set in minutes -/
def first_artist_set : ℕ := 70 + 5

/-- The duration of the second artist's set in minutes -/
def second_artist_set : ℕ := 15 * 4 + 6 * 7 + 15 + 2 * 10

/-- The duration of the third artist's set in minutes -/
def third_artist_set : ℕ := festival_duration - first_artist_set - second_artist_set

theorem third_artist_set_duration : third_artist_set = 148 := by
  sorry

end NUMINAMATH_CALUDE_third_artist_set_duration_l2340_234079


namespace NUMINAMATH_CALUDE_intersection_line_l2340_234046

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x-1)^2 + (y-3)^2 = 10

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y - 5 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_l2340_234046


namespace NUMINAMATH_CALUDE_polar_to_rectangular_min_a_for_inequality_l2340_234061

-- Part A
theorem polar_to_rectangular (ρ θ : ℝ) (x y : ℝ) :
  ρ^2 * Real.cos θ - ρ = 0 ↔ x = 1 :=
sorry

-- Part B
theorem min_a_for_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 5, |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_min_a_for_inequality_l2340_234061


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l2340_234005

/-- Given three squares with side lengths satisfying certain conditions, 
    prove that the sum of their areas is 189. -/
theorem sum_of_square_areas (x a b : ℝ) 
  (h1 : a + b + x = 23)
  (h2 : 9 ≤ (min a b)^2)
  (h3 : (min a b)^2 ≤ 25)
  (h4 : max a b ≥ 5) :
  x^2 + a^2 + b^2 = 189 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l2340_234005


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_250_125_l2340_234097

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def is_two_digit_prime (p : ℕ) : Prop := 10 ≤ p ∧ p < 100 ∧ Nat.Prime p

theorem largest_two_digit_prime_factor_of_binom_250_125 :
  ∃ (p : ℕ), is_two_digit_prime p ∧
             p ∣ binomial_coefficient 250 125 ∧
             ∀ (q : ℕ), is_two_digit_prime q ∧ q ∣ binomial_coefficient 250 125 → q ≤ p ∧
             p = 83 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_250_125_l2340_234097


namespace NUMINAMATH_CALUDE_gcf_of_180_150_210_l2340_234094

theorem gcf_of_180_150_210 : Nat.gcd 180 (Nat.gcd 150 210) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_150_210_l2340_234094


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l2340_234029

def matrix_element (i j : Nat) : Int :=
  if j % i = 0 then 1 else -1

def sum_3j : Int :=
  (matrix_element 3 2) + (matrix_element 3 3) + (matrix_element 3 4) + (matrix_element 3 5)

def sum_i4 : Int :=
  (matrix_element 2 4) + (matrix_element 3 4) + (matrix_element 4 4)

theorem matrix_sum_theorem : sum_3j + sum_i4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l2340_234029


namespace NUMINAMATH_CALUDE_drink_equality_l2340_234081

theorem drink_equality (x : ℝ) : 
  let eric_initial := x
  let sara_initial := 1.4 * x
  let eric_consumed := (2/3) * eric_initial
  let sara_consumed := (2/3) * sara_initial
  let eric_remaining := eric_initial - eric_consumed
  let sara_remaining := sara_initial - sara_consumed
  let transfer := (1/2) * sara_remaining + 3
  let eric_final := eric_consumed + transfer
  let sara_final := sara_consumed + (sara_remaining - transfer)
  eric_final = sara_final ∧ eric_final = 23 ∧ sara_final = 23 :=
by sorry

#check drink_equality

end NUMINAMATH_CALUDE_drink_equality_l2340_234081


namespace NUMINAMATH_CALUDE_frequency_20_plus_l2340_234024

-- Define the sample size
def sample_size : ℕ := 35

-- Define the frequencies for each interval
def freq_5_10 : ℕ := 5
def freq_10_15 : ℕ := 12
def freq_15_20 : ℕ := 7
def freq_20_25 : ℕ := 5
def freq_25_30 : ℕ := 4
def freq_30_35 : ℕ := 2

-- Theorem to prove
theorem frequency_20_plus (h : freq_5_10 + freq_10_15 + freq_15_20 + freq_20_25 + freq_25_30 + freq_30_35 = sample_size) :
  (freq_20_25 + freq_25_30 + freq_30_35 : ℚ) / sample_size = 11 / 35 := by
  sorry

end NUMINAMATH_CALUDE_frequency_20_plus_l2340_234024


namespace NUMINAMATH_CALUDE_lizzies_group_size_l2340_234022

theorem lizzies_group_size (total : ℕ) (difference : ℕ) : 
  total = 91 → difference = 17 → ∃ (other : ℕ), other + (other + difference) = total ∧ other + difference = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_lizzies_group_size_l2340_234022


namespace NUMINAMATH_CALUDE_opposite_face_color_l2340_234013

/-- Represents the colors used on the cube faces -/
inductive Color
| Orange
| Silver
| Yellow
| Violet
| Indigo
| Turquoise

/-- Represents a face of the cube -/
inductive Face
| Top
| Bottom
| Front
| Back
| Left
| Right

/-- Represents a view of the cube -/
structure View where
  top : Color
  front : Color
  right : Color

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Face → Color

/-- Checks if all colors in a list are unique -/
def allUnique (colors : List Color) : Prop :=
  colors.Nodup

/-- The theorem to be proved -/
theorem opposite_face_color (c : Cube)
  (view1 : View)
  (view2 : View)
  (view3 : View)
  (h1 : view1 = { top := Color.Orange, front := Color.Yellow, right := Color.Silver })
  (h2 : view2 = { top := Color.Orange, front := Color.Indigo, right := Color.Silver })
  (h3 : view3 = { top := Color.Orange, front := Color.Violet, right := Color.Silver })
  (h_unique : allUnique [Color.Orange, Color.Silver, Color.Yellow, Color.Violet, Color.Indigo, Color.Turquoise])
  (h_cube : c.faces Face.Top = Color.Orange ∧
            c.faces Face.Right = Color.Silver ∧
            c.faces Face.Front ∈ [Color.Yellow, Color.Indigo, Color.Violet] ∧
            c.faces Face.Left ∈ [Color.Yellow, Color.Indigo, Color.Violet] ∧
            c.faces Face.Back ∈ [Color.Yellow, Color.Indigo, Color.Violet])
  : c.faces Face.Bottom = Color.Turquoise → c.faces Face.Top = Color.Orange :=
by sorry

end NUMINAMATH_CALUDE_opposite_face_color_l2340_234013


namespace NUMINAMATH_CALUDE_weekly_earnings_calculation_l2340_234003

/- Define the basic fees and attendance -/
def kidFee : ℚ := 3
def adultFee : ℚ := 6
def weekdayKids : ℕ := 8
def weekdayAdults : ℕ := 10
def weekendKids : ℕ := 12
def weekendAdults : ℕ := 15

/- Define the discounts and special rates -/
def weekendRate : ℚ := 1.5
def groupDiscountRate : ℚ := 0.8
def membershipDiscountRate : ℚ := 0.9
def weekdayGroupBookings : ℕ := 2
def weekendMemberships : ℕ := 8

/- Calculate earnings -/
def weekdayEarnings : ℚ := 5 * (weekdayKids * kidFee + weekdayAdults * adultFee)
def weekendEarnings : ℚ := 2 * (weekendKids * kidFee * weekendRate + weekendAdults * adultFee * weekendRate)

/- Calculate discounts -/
def weekdayGroupDiscount : ℚ := 5 * weekdayGroupBookings * (kidFee + adultFee) * (1 - groupDiscountRate)
def weekendMembershipDiscount : ℚ := 2 * weekendMemberships * adultFee * weekendRate * (1 - membershipDiscountRate)

/- Define the total weekly earnings -/
def totalWeeklyEarnings : ℚ := weekdayEarnings + weekendEarnings - weekdayGroupDiscount - weekendMembershipDiscount

/- The theorem to prove -/
theorem weekly_earnings_calculation : totalWeeklyEarnings = 738.6 := by
  sorry


end NUMINAMATH_CALUDE_weekly_earnings_calculation_l2340_234003


namespace NUMINAMATH_CALUDE_rod_cutting_l2340_234004

/-- Given a rod of 17 meters long from which 20 pieces can be cut,
    prove that the length of each piece is 85 centimeters. -/
theorem rod_cutting (rod_length : ℝ) (num_pieces : ℕ) (piece_length_cm : ℝ) :
  rod_length = 17 →
  num_pieces = 20 →
  piece_length_cm = (rod_length / num_pieces) * 100 →
  piece_length_cm = 85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l2340_234004


namespace NUMINAMATH_CALUDE_line_y_intercept_l2340_234007

/-- A line with slope -3 and x-intercept (7,0) has y-intercept (0, 21) -/
theorem line_y_intercept (f : ℝ → ℝ) (h1 : ∀ x y, f y - f x = -3 * (y - x)) 
  (h2 : f 7 = 0) : f 0 = 21 := by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2340_234007


namespace NUMINAMATH_CALUDE_andrew_age_proof_l2340_234028

/-- Andrew's age in years -/
def andrew_age : ℚ := 30 / 7

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℚ := 15 * andrew_age

theorem andrew_age_proof :
  andrew_age = 30 / 7 ∧
  grandfather_age = 15 * andrew_age ∧
  grandfather_age - andrew_age = 60 :=
sorry

end NUMINAMATH_CALUDE_andrew_age_proof_l2340_234028
