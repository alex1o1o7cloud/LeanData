import Mathlib

namespace later_purchase_cost_l320_32047

/-- The cost of a single bat in dollars -/
def bat_cost : ℕ := 500

/-- The cost of a single ball in dollars -/
def ball_cost : ℕ := 100

/-- The number of bats in the later purchase -/
def num_bats : ℕ := 3

/-- The number of balls in the later purchase -/
def num_balls : ℕ := 5

/-- The total cost of the later purchase -/
def total_cost : ℕ := num_bats * bat_cost + num_balls * ball_cost

theorem later_purchase_cost : total_cost = 2000 := by
  sorry

end later_purchase_cost_l320_32047


namespace base_conversion_3500_to_base_7_l320_32030

theorem base_conversion_3500_to_base_7 :
  (1 * 7^4 + 3 * 7^3 + 1 * 7^2 + 3 * 7^1 + 0 * 7^0 : ℕ) = 3500 := by
  sorry

end base_conversion_3500_to_base_7_l320_32030


namespace square_sum_equality_l320_32007

theorem square_sum_equality (n : ℤ) : n + n + n + n = 4 * n := by
  sorry

end square_sum_equality_l320_32007


namespace hania_age_in_5_years_l320_32076

-- Define the current year as a reference point
def current_year : ℕ := 2023

-- Define Samir's age in 5 years
def samir_age_in_5_years : ℕ := 20

-- Define the relationship between Samir's current age and Hania's age 10 years ago
axiom samir_hania_age_relation : 
  ∃ (samir_current_age hania_age_10_years_ago : ℕ),
    samir_current_age = samir_age_in_5_years - 5 ∧
    samir_current_age = hania_age_10_years_ago / 2

-- Theorem to prove
theorem hania_age_in_5_years : 
  ∃ (hania_current_age : ℕ),
    hania_current_age + 5 = 45 :=
sorry

end hania_age_in_5_years_l320_32076


namespace gcd_count_for_360_l320_32068

theorem gcd_count_for_360 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃! (s : Finset ℕ+), (∀ x ∈ s, ∃ a b : ℕ+, Nat.gcd a b = x ∧ Nat.lcm a b * x = 360) ∧ s.card = 6 :=
sorry

end gcd_count_for_360_l320_32068


namespace gasoline_price_decrease_l320_32055

theorem gasoline_price_decrease (a : ℝ) : 
  (∃ (initial_price final_price : ℝ), 
    initial_price = 8.1 ∧ 
    final_price = 7.8 ∧ 
    initial_price * (1 - a/100)^2 = final_price) → 
  8.1 * (1 - a/100)^2 = 7.8 :=
by sorry

end gasoline_price_decrease_l320_32055


namespace task_completion_probability_l320_32080

theorem task_completion_probability (p_task1 p_task1_not_task2 : ℝ) 
  (h1 : p_task1 = 5/8)
  (h2 : p_task1_not_task2 = 1/4)
  (h3 : 0 ≤ p_task1 ∧ p_task1 ≤ 1)
  (h4 : 0 ≤ p_task1_not_task2 ∧ p_task1_not_task2 ≤ 1) :
  ∃ p_task2 : ℝ, p_task2 = 3/5 ∧ p_task1 * (1 - p_task2) = p_task1_not_task2 := by
  sorry

end task_completion_probability_l320_32080


namespace window_dimensions_correct_l320_32099

/-- Represents the dimensions of a window -/
structure WindowDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the dimensions of a rectangular window with panes -/
def calculateWindowDimensions (x : ℝ) : WindowDimensions :=
  { width := 4 * x + 10,
    height := 9 * x + 8 }

theorem window_dimensions_correct (x : ℝ) :
  let numRows : ℕ := 3
  let numCols : ℕ := 4
  let numPanes : ℕ := 12
  let paneHeightToWidthRatio : ℝ := 3
  let borderWidth : ℝ := 2
  let dimensions := calculateWindowDimensions x
  (numRows * numCols = numPanes) ∧
  (numRows * (x * paneHeightToWidthRatio) + (numRows + 1) * borderWidth = dimensions.height) ∧
  (numCols * x + (numCols + 1) * borderWidth = dimensions.width) :=
by sorry

end window_dimensions_correct_l320_32099


namespace sum_of_three_consecutive_even_numbers_l320_32097

theorem sum_of_three_consecutive_even_numbers : 
  ∀ (a b c : ℕ), 
    a = 80 → 
    b = a + 2 → 
    c = b + 2 → 
    a + b + c = 246 := by
  sorry

end sum_of_three_consecutive_even_numbers_l320_32097


namespace p_necessary_not_sufficient_for_q_l320_32096

-- Define the conditions
def condition_p (x : ℝ) : Prop := |x + 1| ≤ 4
def condition_q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x, condition_q x → condition_p x) ∧
  (∃ x, condition_p x ∧ ¬condition_q x) :=
sorry

end p_necessary_not_sufficient_for_q_l320_32096


namespace distribution_count_4_3_l320_32087

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribution_count (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 4 distinct objects into 3 distinct groups,
    where each group must contain at least one object, is equal to 36 -/
theorem distribution_count_4_3 :
  distribution_count 4 3 = 36 := by sorry

end distribution_count_4_3_l320_32087


namespace age_of_replaced_man_l320_32065

/-- Given a group of 8 men where two are replaced by two women, prove the age of one of the replaced men. -/
theorem age_of_replaced_man
  (n : ℕ) -- Total number of people
  (m : ℕ) -- Number of men initially
  (w : ℕ) -- Number of women replacing men
  (A : ℝ) -- Initial average age of men
  (increase : ℝ) -- Increase in average age after replacement
  (known_man_age : ℕ) -- Age of one of the replaced men
  (women_avg_age : ℝ) -- Average age of the women
  (h1 : n = 8)
  (h2 : m = 8)
  (h3 : w = 2)
  (h4 : increase = 2)
  (h5 : known_man_age = 10)
  (h6 : women_avg_age = 23)
  : ∃ (other_man_age : ℕ), other_man_age = 20 :=
by
  sorry


end age_of_replaced_man_l320_32065


namespace tangent_perpendicular_point_l320_32057

def f (x : ℝ) := x^4 - x

theorem tangent_perpendicular_point :
  ∃! p : ℝ × ℝ, 
    p.2 = f p.1 ∧ 
    (4 * p.1^3 - 1) * (-1/3) = -1 ∧
    p = (1, 0) := by
  sorry

end tangent_perpendicular_point_l320_32057


namespace range_of_a_l320_32090

-- Define the conditions
def p (x : ℝ) : Prop := (x + 1)^2 > 4
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(q x a) ∧ p x)) →
  (∀ a : ℝ, a ≥ 1 ↔ (∀ x : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(q x a) ∧ p x))) :=
by sorry

end range_of_a_l320_32090


namespace exists_geometric_progression_shift_l320_32026

/-- Given a sequence {a_n} defined by a_n = q * a_{n-1} + d where q ≠ 1,
    there exists a constant c such that b_n = a_n + c forms a geometric progression. -/
theorem exists_geometric_progression_shift 
  (q d : ℝ) (hq : q ≠ 1) (a : ℕ → ℝ) 
  (ha : ∀ n : ℕ, a (n + 1) = q * a n + d) :
  ∃ c : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n + c = r^n * (a 0 + c) :=
sorry

end exists_geometric_progression_shift_l320_32026


namespace curve_E_perpendicular_points_sum_inverse_squares_l320_32013

-- Define the curve E
def curve_E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the property of perpendicular vectors
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem curve_E_perpendicular_points_sum_inverse_squares (x₁ y₁ x₂ y₂ : ℝ) :
  curve_E x₁ y₁ → curve_E x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2) = 7 / 12 :=
by sorry

end curve_E_perpendicular_points_sum_inverse_squares_l320_32013


namespace handshake_problem_l320_32073

/-- The number of handshakes in a complete graph with n vertices -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: Given 435 handshakes, there are 30 men -/
theorem handshake_problem :
  ∃ (n : ℕ), n > 0 ∧ handshakes n = 435 ∧ n = 30 := by
  sorry

end handshake_problem_l320_32073


namespace rectangle_area_y_value_l320_32011

/-- A rectangle with vertices at (-2, y), (10, y), (-2, 1), and (10, 1) has an area of 108 square units. Prove that y = 10. -/
theorem rectangle_area_y_value (y : ℝ) : 
  y > 0 → -- y is positive
  (10 - (-2)) * (y - 1) = 108 → -- area of the rectangle is 108 square units
  y = 10 := by
sorry

end rectangle_area_y_value_l320_32011


namespace smallest_valid_number_l320_32094

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10^11 ∧ n < 10^12) ∧  -- 12-digit number
  (n % 36 = 0) ∧             -- divisible by 36
  (∀ d : ℕ, d < 10 → ∃ k : ℕ, (n / 10^k) % 10 = d)  -- contains each digit 0-9

theorem smallest_valid_number :
  (is_valid_number 100023457896) ∧
  (∀ m : ℕ, m < 100023457896 → ¬(is_valid_number m)) :=
sorry

end smallest_valid_number_l320_32094


namespace parabola_symmetry_l320_32003

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

-- Define the translation
def translate_left : ℝ := 3
def translate_up : ℝ := 2

-- Define parabola C after translation
def parabola_C (x : ℝ) : ℝ := original_parabola (x + translate_left) + translate_up

-- Define the symmetric parabola
def symmetric_parabola (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

-- Theorem statement
theorem parabola_symmetry :
  ∀ x : ℝ, parabola_C (-x) = symmetric_parabola x :=
by sorry

end parabola_symmetry_l320_32003


namespace square_side_length_l320_32053

/-- Given a square and a regular hexagon where:
    1) The perimeter of the square equals the perimeter of the hexagon
    2) Each side of the hexagon measures 6 cm
    Prove that the length of one side of the square is 9 cm -/
theorem square_side_length (square_perimeter hexagon_perimeter : ℝ) 
  (hexagon_side : ℝ) (h1 : square_perimeter = hexagon_perimeter) 
  (h2 : hexagon_side = 6) (h3 : hexagon_perimeter = 6 * hexagon_side) :
  square_perimeter / 4 = 9 := by
  sorry

end square_side_length_l320_32053


namespace power_product_equals_four_l320_32000

theorem power_product_equals_four (a b : ℕ+) (h : (3 ^ a.val) ^ b.val = 3 ^ 3) :
  3 ^ a.val * 3 ^ b.val = 3 ^ 4 := by
  sorry

end power_product_equals_four_l320_32000


namespace line_equivalence_slope_and_intercept_l320_32091

/-- The vector representation of the line -/
def line_vector (x y : ℝ) : ℝ := 2 * (x - 3) + (-1) * (y - (-4))

/-- The slope-intercept form of the line -/
def line_slope_intercept (x y : ℝ) : Prop := y = 2 * x - 10

theorem line_equivalence :
  ∀ x y : ℝ, line_vector x y = 0 ↔ line_slope_intercept x y :=
sorry

theorem slope_and_intercept :
  ∃ m b : ℝ, (∀ x y : ℝ, line_vector x y = 0 → y = m * x + b) ∧ m = 2 ∧ b = -10 :=
sorry

end line_equivalence_slope_and_intercept_l320_32091


namespace remainder_theorem_l320_32014

theorem remainder_theorem (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : v < y) :
  (x + 4 * u * y) % y = v := by
  sorry

end remainder_theorem_l320_32014


namespace definite_integral_3x_squared_l320_32010

theorem definite_integral_3x_squared : ∫ x in (1:ℝ)..2, 3 * x^2 = 7 := by sorry

end definite_integral_3x_squared_l320_32010


namespace prob_both_counterfeit_value_l320_32061

/-- Represents the total number of banknotes --/
def total_notes : ℕ := 20

/-- Represents the number of counterfeit notes --/
def counterfeit_notes : ℕ := 5

/-- Represents the number of notes drawn --/
def drawn_notes : ℕ := 2

/-- Calculates the probability that both drawn notes are counterfeit given that at least one is counterfeit --/
def prob_both_counterfeit : ℚ :=
  (Nat.choose counterfeit_notes drawn_notes) / 
  (Nat.choose counterfeit_notes drawn_notes + 
   Nat.choose counterfeit_notes 1 * Nat.choose (total_notes - counterfeit_notes) 1)

theorem prob_both_counterfeit_value : 
  prob_both_counterfeit = 2 / 17 := by sorry

end prob_both_counterfeit_value_l320_32061


namespace last_two_average_l320_32067

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 60 →
  ((list.take 3).sum / 3 : ℝ) = 45 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 72.5 := by
sorry

end last_two_average_l320_32067


namespace rectangle_in_circle_area_l320_32064

theorem rectangle_in_circle_area (r : ℝ) (w h : ℝ) :
  r = 5 ∧ w = 6 ∧ h = 2 →
  w * h ≤ π * r^2 →
  w * h = 12 :=
by sorry

end rectangle_in_circle_area_l320_32064


namespace f_increasing_on_interval_l320_32054

def f (x : ℝ) := -x^2 + 2*x + 8

theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ y ≤ 1 → f x < f y :=
by
  sorry

end f_increasing_on_interval_l320_32054


namespace sqrt_six_times_sqrt_three_equals_three_sqrt_two_l320_32032

theorem sqrt_six_times_sqrt_three_equals_three_sqrt_two :
  Real.sqrt 6 * Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_six_times_sqrt_three_equals_three_sqrt_two_l320_32032


namespace least_positive_angle_theorem_l320_32028

theorem least_positive_angle_theorem (θ : Real) : 
  (θ > 0 ∧ Real.cos (5 * π / 180) = Real.sin (25 * π / 180) + Real.sin θ) →
  θ = 35 * π / 180 := by
  sorry

end least_positive_angle_theorem_l320_32028


namespace line_segment_endpoint_l320_32056

theorem line_segment_endpoint (y : ℝ) : 
  y < 0 → 
  ((3 - 1)^2 + (-2 - y)^2)^(1/2) = 15 → 
  y = -2 - (221 : ℝ)^(1/2) := by
sorry

end line_segment_endpoint_l320_32056


namespace mean_proportional_segments_l320_32048

/-- Given that segment b is the mean proportional between segments a and c,
    prove that if a = 2 and b = 4, then c = 8. -/
theorem mean_proportional_segments (a b c : ℝ) 
  (h1 : b^2 = a * c) -- b is the mean proportional between a and c
  (h2 : a = 2)       -- a = 2 cm
  (h3 : b = 4)       -- b = 4 cm
  : c = 8 := by
  sorry

end mean_proportional_segments_l320_32048


namespace square_sum_given_sum_and_product_l320_32035

theorem square_sum_given_sum_and_product (a b : ℝ) : 
  a + b = 6 → a * b = 3 → a^2 + b^2 = 30 := by
  sorry

end square_sum_given_sum_and_product_l320_32035


namespace cubic_tangent_ratio_l320_32093

-- Define the cubic function
def cubic (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the points A, T, B on the x-axis
structure RootPoints where
  α : ℝ
  γ : ℝ
  β : ℝ

-- Define the theorem
theorem cubic_tangent_ratio 
  (a b c : ℝ) 
  (roots : RootPoints) 
  (h1 : cubic a b c roots.α = 0)
  (h2 : cubic a b c roots.γ = 0)
  (h3 : cubic a b c roots.β = 0)
  (h4 : roots.α < roots.γ)
  (h5 : roots.γ < roots.β) :
  (roots.β - roots.α) / ((roots.α + roots.γ)/2 - (roots.β + roots.γ)/2) = -2 := by
  sorry

end cubic_tangent_ratio_l320_32093


namespace range_of_f_triangle_properties_l320_32071

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x - 1/2

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

def triangle : Triangle where
  A := Real.pi / 3
  a := 2 * Real.sqrt 3
  b := 2
  c := 4

-- Theorem statements
theorem range_of_f : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-1/2) 1 := sorry

theorem triangle_properties (t : Triangle) (h1 : 0 < t.A) (h2 : t.A < Real.pi / 2) 
  (h3 : t.a = 2 * Real.sqrt 3) (h4 : t.c = 4) (h5 : f t.A = 1) : 
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3) := sorry

end

end range_of_f_triangle_properties_l320_32071


namespace line_m_plus_b_l320_32050

/-- A line passing through three given points has m + b = -1 -/
theorem line_m_plus_b (m b : ℝ) : 
  (3 = m * 3 + b) →  -- Line passes through (3, 3)
  (-1 = m * 1 + b) →  -- Line passes through (1, -1)
  (1 = m * 2 + b) →  -- Line passes through (2, 1)
  m + b = -1 := by
sorry

end line_m_plus_b_l320_32050


namespace train_distance_proof_l320_32058

-- Define the speeds of the trains
def speed_train1 : ℝ := 20
def speed_train2 : ℝ := 25

-- Define the difference in distance traveled
def distance_difference : ℝ := 55

-- Define the total distance between stations
def total_distance : ℝ := 495

-- Theorem statement
theorem train_distance_proof :
  ∃ (time : ℝ) (distance1 distance2 : ℝ),
    time > 0 ∧
    distance1 = speed_train1 * time ∧
    distance2 = speed_train2 * time ∧
    distance2 = distance1 + distance_difference ∧
    total_distance = distance1 + distance2 :=
by
  sorry

end train_distance_proof_l320_32058


namespace expected_heads_is_56_l320_32069

/-- The number of fair coins --/
def n : ℕ := 90

/-- The probability of getting heads on a single fair coin toss --/
def p_heads : ℚ := 1/2

/-- The probability of getting tails followed by two consecutive heads --/
def p_tails_then_heads : ℚ := 1/2 * 1/4

/-- The total probability of a coin showing heads under the given rules --/
def p_total : ℚ := p_heads + p_tails_then_heads

/-- The expected number of coins showing heads --/
def expected_heads : ℚ := n * p_total

theorem expected_heads_is_56 : expected_heads = 56 := by sorry

end expected_heads_is_56_l320_32069


namespace cloth_sale_calculation_l320_32001

theorem cloth_sale_calculation (selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  selling_price = 8925 ∧ 
  profit_per_meter = 10 ∧ 
  cost_price_per_meter = 95 →
  (selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
sorry

end cloth_sale_calculation_l320_32001


namespace two_digit_product_777_l320_32060

theorem two_digit_product_777 :
  ∀ a b : ℕ,
    10 ≤ a ∧ a < 100 →
    10 ≤ b ∧ b < 100 →
    a * b = 777 →
    ((a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21)) :=
by sorry

end two_digit_product_777_l320_32060


namespace cosine_sum_identity_l320_32059

theorem cosine_sum_identity : 
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end cosine_sum_identity_l320_32059


namespace edward_tickets_l320_32070

/-- The number of tickets Edward won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 3

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 4

/-- The number of candies Edward could buy -/
def candies_bought : ℕ := 2

/-- The number of tickets Edward won playing 'skee ball' -/
def skee_ball_tickets : ℕ := sorry

theorem edward_tickets : skee_ball_tickets = 5 := by
  sorry

end edward_tickets_l320_32070


namespace ads_ratio_l320_32052

def problem (ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked : ℕ) : Prop :=
  ads_page1 = 12 ∧
  ads_page2 = 2 * ads_page1 ∧
  ads_page3 = ads_page2 + 24 ∧
  ads_page4 = (3 * ads_page2) / 4 ∧
  ads_clicked = 68

theorem ads_ratio (ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked : ℕ) :
  problem ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked →
  (ads_clicked : ℚ) / (ads_page1 + ads_page2 + ads_page3 + ads_page4 : ℚ) = 2 / 3 := by
  sorry

end ads_ratio_l320_32052


namespace min_pie_pieces_correct_l320_32023

/-- The minimum number of pieces a pie can be cut into to be equally divided among either 10 or 11 guests -/
def min_pie_pieces : ℕ := 20

/-- The number of expected guests -/
def possible_guests : Set ℕ := {10, 11}

/-- A function that checks if a given number of pieces can be equally divided among a given number of guests -/
def is_divisible (pieces : ℕ) (guests : ℕ) : Prop :=
  ∃ (k : ℕ), pieces = k * guests

theorem min_pie_pieces_correct :
  (∀ g ∈ possible_guests, is_divisible min_pie_pieces g) ∧
  (∀ p < min_pie_pieces, ∃ g ∈ possible_guests, ¬is_divisible p g) :=
sorry

end min_pie_pieces_correct_l320_32023


namespace michael_singles_percentage_l320_32031

/-- Calculates the percentage of singles in a player's hits -/
def percentage_singles (total_hits : ℕ) (home_runs triples doubles : ℕ) : ℚ :=
  let non_singles := home_runs + triples + doubles
  let singles := total_hits - non_singles
  (singles : ℚ) / (total_hits : ℚ) * 100

theorem michael_singles_percentage :
  percentage_singles 50 2 3 8 = 74 := by
  sorry

end michael_singles_percentage_l320_32031


namespace no_simple_algebraic_solution_l320_32051

variable (g V₀ a S V t : ℝ)

def velocity_equation := V = g * t + V₀

def displacement_equation := S = (1/2) * g * t^2 + V₀ * t + (1/3) * a * t^3

theorem no_simple_algebraic_solution :
  ∀ g V₀ a S V t : ℝ,
  velocity_equation g V₀ V t →
  displacement_equation g V₀ a S t →
  ¬∃ f : ℝ → ℝ → ℝ → ℝ → ℝ, t = f S g V₀ a :=
by sorry

end no_simple_algebraic_solution_l320_32051


namespace runners_speed_l320_32086

/-- The speeds of two runners on a circular track -/
theorem runners_speed (speed_a speed_b : ℝ) (track_length : ℝ) : 
  speed_a > 0 ∧ 
  speed_b > 0 ∧ 
  track_length > 0 ∧ 
  (speed_a + speed_b) * 48 = track_length ∧ 
  (speed_a - speed_b) * 600 = track_length ∧ 
  speed_a = speed_b + 2/3 → 
  speed_a = 9/2 ∧ speed_b = 23/6 := by sorry

end runners_speed_l320_32086


namespace roots_of_equation_l320_32008

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => 3*x*(x-1) - 2*(x-1)
  (f 1 = 0 ∧ f (2/3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 2/3) := by sorry

end roots_of_equation_l320_32008


namespace eight_ampersand_five_l320_32095

def ampersand (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem eight_ampersand_five : ampersand 8 5 = 39 := by
  sorry

end eight_ampersand_five_l320_32095


namespace tank_fill_time_l320_32098

/-- Given two pipes A and B, where A fills a tank in 56 minutes and B fills the tank 7 times as fast as A,
    the time to fill the tank when both pipes are open is 7 minutes. -/
theorem tank_fill_time (time_A : ℝ) (rate_B_multiplier : ℝ) : 
  time_A = 56 → rate_B_multiplier = 7 → 
  1 / (1 / time_A + rate_B_multiplier / time_A) = 7 := by
  sorry

end tank_fill_time_l320_32098


namespace system_solution_implies_m_zero_l320_32012

theorem system_solution_implies_m_zero (x y m : ℝ) :
  (2 * x + 3 * y = 4) →
  (3 * x + 2 * y = 2 * m - 3) →
  (x + y = 1 / 5) →
  m = 0 := by
sorry

end system_solution_implies_m_zero_l320_32012


namespace sqrt_equation_solution_l320_32081

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (x + 15) = 12 → x = 129 := by sorry

end sqrt_equation_solution_l320_32081


namespace cubic_divisibility_l320_32039

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluation of the cubic polynomial at a given point -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℤ) : ℤ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Condition that one root is the product of the other two -/
def has_product_root (p : CubicPolynomial) : Prop :=
  ∃ (u v : ℚ), u ≠ 0 ∧ v ≠ 0 ∧ 
    (u + v + u*v = -p.a) ∧
    (u*v*(1 + u + v) = p.b) ∧
    (u^2 * v^2 = -p.c)

/-- Main theorem statement -/
theorem cubic_divisibility (p : CubicPolynomial) (h : has_product_root p) :
  (2 * p.eval (-1)) ∣ (p.eval 1 + p.eval (-1) - 2 * (1 + p.eval 0)) :=
sorry

end cubic_divisibility_l320_32039


namespace sum_of_prime_factors_195195_l320_32049

theorem sum_of_prime_factors_195195 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (195195 + 1))) id = 39) ∧ 
  (∀ p : ℕ, p ∈ Finset.filter Nat.Prime (Finset.range (195195 + 1)) ↔ p.Prime ∧ 195195 % p = 0) :=
by sorry

end sum_of_prime_factors_195195_l320_32049


namespace factorization_problems_l320_32044

theorem factorization_problems :
  (∀ x : ℝ, x^2 - 16 = (x + 4) * (x - 4)) ∧
  (∀ a b : ℝ, a^3*b - 2*a^2*b + a*b = a*b*(a - 1)^2) := by
  sorry

end factorization_problems_l320_32044


namespace fencing_length_l320_32018

/-- Given a rectangular field with area 400 sq. ft and one side of 20 feet,
    prove that the fencing required for three sides is 60 feet. -/
theorem fencing_length (area : ℝ) (side : ℝ) (h1 : area = 400) (h2 : side = 20) :
  2 * (area / side) + side = 60 := by
  sorry

end fencing_length_l320_32018


namespace subset_relation_l320_32040

def set_A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def set_B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

theorem subset_relation (a : ℝ) :
  (∀ x : ℝ, x ∈ set_B → x ∈ set_A 1) ∧
  (∀ x : ℝ, x ∈ set_A a → x ∈ set_B ↔ a < -8 ∨ a ≥ 2) :=
sorry

end subset_relation_l320_32040


namespace circle_radius_three_inches_l320_32036

theorem circle_radius_three_inches (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 := by
  sorry

end circle_radius_three_inches_l320_32036


namespace max_digit_sum_l320_32021

/-- A_n is an n-digit integer with all digits equal to a -/
def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^n - 1) / 9

/-- B_n is a 2n-digit integer with all digits equal to b -/
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9

/-- C_n is a 3n-digit integer with all digits equal to c -/
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(3*n) - 1) / 9

/-- The theorem statement -/
theorem max_digit_sum (a b c : ℕ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) (hc : 0 < c ∧ c < 10) :
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ 0 < n₁ ∧ 0 < n₂ ∧ 
    C_n c n₁ - A_n a n₁ = (B_n b n₁)^2 ∧
    C_n c n₂ - A_n a n₂ = (B_n b n₂)^2) →
  a + b + c ≤ 13 :=
sorry

end max_digit_sum_l320_32021


namespace hotel_price_per_night_l320_32027

def car_value : ℕ := 30000
def house_value : ℕ := 4 * car_value
def total_value : ℕ := 158000

theorem hotel_price_per_night :
  ∃ (price_per_night : ℕ), 
    car_value + house_value + 2 * price_per_night = total_value ∧
    price_per_night = 4000 :=
by sorry

end hotel_price_per_night_l320_32027


namespace arithmetic_sequence_35th_term_l320_32041

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The 35th term of the arithmetic sequence is 99. -/
theorem arithmetic_sequence_35th_term 
  (seq : ArithmeticSequence) 
  (h15 : seq.a 15 = 33) 
  (h25 : seq.a 25 = 66) : 
  seq.a 35 = 99 := by
  sorry

end arithmetic_sequence_35th_term_l320_32041


namespace equation_solutions_l320_32078

theorem equation_solutions : 
  ∃! (s : Set ℝ), s = {x : ℝ | (x + 3)^4 + (x + 1)^4 = 82} ∧ s = {0, -4} := by
  sorry

end equation_solutions_l320_32078


namespace quadratic_roots_condition_l320_32063

theorem quadratic_roots_condition (p q : ℝ) : 
  (q < 0) ↔ (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) := by sorry

end quadratic_roots_condition_l320_32063


namespace inverse_g_sum_l320_32082

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3*x - x^2

theorem inverse_g_sum : 
  ∃ (a b c : ℝ), g a = -2 ∧ g b = 0 ∧ g c = 4 ∧ a + b + c = 6 :=
by sorry

end inverse_g_sum_l320_32082


namespace digit1Sequence_1482_to_1484_l320_32033

/-- A sequence of positive integers starting with digit 1 in increasing order -/
def digit1Sequence : ℕ → ℕ := sorry

/-- The nth digit in the concatenated sequence of digit1Sequence -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1482nd, 1483rd, and 1484th digits -/
def targetNumber : ℕ := 100 * (nthDigit 1482) + 10 * (nthDigit 1483) + (nthDigit 1484)

theorem digit1Sequence_1482_to_1484 : targetNumber = 129 := by sorry

end digit1Sequence_1482_to_1484_l320_32033


namespace greatest_three_digit_number_l320_32020

theorem greatest_three_digit_number : ∃ n : ℕ, 
  (n ≤ 999) ∧ 
  (n ≥ 100) ∧ 
  (∃ k : ℕ, n = 8 * k - 1) ∧ 
  (∃ m : ℕ, n = 7 * m + 4) ∧ 
  (∀ x : ℕ, x ≤ 999 ∧ x ≥ 100 ∧ (∃ a : ℕ, x = 8 * a - 1) ∧ (∃ b : ℕ, x = 7 * b + 4) → x ≤ n) ∧
  n = 967 :=
by sorry

end greatest_three_digit_number_l320_32020


namespace angle_measure_in_triangle_l320_32024

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c² = (a-b)² + 6 and the area of the triangle is 3√3/2,
    then the measure of angle C is π/3 -/
theorem angle_measure_in_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c^2 = (a - b)^2 + 6 →
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = π / 3 := by sorry

end angle_measure_in_triangle_l320_32024


namespace opposites_power_2004_l320_32017

theorem opposites_power_2004 (x y : ℝ) 
  (h : |x + 1| + |y + 2*x| = 0) : 
  (x + y)^2004 = 1 := by
  sorry

end opposites_power_2004_l320_32017


namespace no_quinary_country_46_airlines_l320_32074

/-- A quinary country is a country where each city is connected by air lines with exactly five other cities. -/
structure QuinaryCountry where
  cities : ℕ
  airLines : ℕ
  isQuinary : airLines = (cities * 5) / 2

/-- Theorem: There cannot exist a quinary country with exactly 46 air lines. -/
theorem no_quinary_country_46_airlines : ¬ ∃ (q : QuinaryCountry), q.airLines = 46 := by
  sorry

end no_quinary_country_46_airlines_l320_32074


namespace simplest_quadratic_radical_l320_32006

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := x ≥ 0

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, y > 1 → ¬(∃ z : ℝ, x = y * z^2)

-- Define the set of given expressions
def GivenExpressions : Set ℝ := {8, 1/3, 6, 0.1}

-- Theorem statement
theorem simplest_quadratic_radical :
  ∀ x ∈ GivenExpressions, 
    SimplestQuadraticRadical (Real.sqrt x) → x = 6 :=
by sorry

end simplest_quadratic_radical_l320_32006


namespace inscribed_square_side_length_l320_32025

/-- A rhombus with an inscribed square -/
structure RhombusWithSquare where
  /-- Length of the first diagonal of the rhombus -/
  d1 : ℝ
  /-- Length of the second diagonal of the rhombus -/
  d2 : ℝ
  /-- The first diagonal is positive -/
  d1_pos : 0 < d1
  /-- The second diagonal is positive -/
  d2_pos : 0 < d2
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed with sides parallel to rhombus diagonals -/
  inscribed : square_side > 0

/-- Theorem stating the side length of the inscribed square in a rhombus with given diagonals -/
theorem inscribed_square_side_length (r : RhombusWithSquare) (h1 : r.d1 = 8) (h2 : r.d2 = 12) : 
  r.square_side = 4.8 := by
  sorry

end inscribed_square_side_length_l320_32025


namespace fish_problem_l320_32016

theorem fish_problem (trout_weight salmon_weight campers fish_per_camper bass_weight : ℕ) 
  (h1 : trout_weight = 8)
  (h2 : salmon_weight = 24)
  (h3 : campers = 22)
  (h4 : fish_per_camper = 2)
  (h5 : bass_weight = 2) :
  (campers * fish_per_camper - (trout_weight + salmon_weight)) / bass_weight = 6 := by
  sorry

end fish_problem_l320_32016


namespace total_books_theorem_melanie_books_l320_32019

/-- Calculates the total number of books after a purchase -/
def total_books_after_purchase (initial_books : ℕ) (books_bought : ℕ) : ℕ :=
  initial_books + books_bought

/-- Theorem: The total number of books after a purchase is the sum of initial books and books bought -/
theorem total_books_theorem (initial_books books_bought : ℕ) :
  total_books_after_purchase initial_books books_bought = initial_books + books_bought :=
by
  sorry

/-- Melanie's book collection problem -/
theorem melanie_books :
  total_books_after_purchase 41 46 = 87 :=
by
  sorry

end total_books_theorem_melanie_books_l320_32019


namespace shaded_area_problem_l320_32046

/-- The area of the shaded region in a square with side length 40 units, 
    where two congruent triangles with base 20 units and height 20 units 
    are removed, is equal to 1200 square units. -/
theorem shaded_area_problem (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 40 →
  triangle_base = 20 →
  triangle_height = 20 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 1200 := by
  sorry

end shaded_area_problem_l320_32046


namespace collinear_vectors_l320_32043

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (e₁ e₂ : V)
variable (A B C : V)
variable (k : ℝ)

theorem collinear_vectors (h1 : e₁ ≠ 0 ∧ e₂ ≠ 0 ∧ ¬ ∃ (r : ℝ), e₁ = r • e₂)
  (h2 : B - A = 2 • e₁ + k • e₂)
  (h3 : C - B = e₁ - 3 • e₂)
  (h4 : ∃ (t : ℝ), C - A = t • (B - A)) :
  k = -6 := by sorry

end collinear_vectors_l320_32043


namespace quadratic_opens_downwards_iff_a_negative_l320_32088

/-- A quadratic function of the form y = ax² - 2 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2

/-- The property that the graph of a quadratic function opens downwards -/
def opens_downwards (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f ((x + y) / 2) > (f x + f y) / 2

theorem quadratic_opens_downwards_iff_a_negative (a : ℝ) :
  opens_downwards (quadratic_function a) ↔ a < 0 := by
  sorry

end quadratic_opens_downwards_iff_a_negative_l320_32088


namespace distance_after_seven_seconds_l320_32066

/-- The distance fallen by a freely falling body after t seconds -/
def distance_fallen (t : ℝ) : ℝ := 4.9 * t^2

/-- The time difference between the start of the two falling bodies -/
def time_difference : ℝ := 5

/-- The distance between the two falling bodies after t seconds -/
def distance_between (t : ℝ) : ℝ :=
  distance_fallen t - distance_fallen (t - time_difference)

/-- Theorem: The distance between the two falling bodies is 220.5 meters after 7 seconds -/
theorem distance_after_seven_seconds :
  distance_between 7 = 220.5 := by sorry

end distance_after_seven_seconds_l320_32066


namespace tan_double_angle_solution_l320_32085

theorem tan_double_angle_solution (α : ℝ) (h : Real.tan (2 * α) = 4 / 3) :
  Real.tan α = -2 ∨ Real.tan α = 1 / 2 := by
  sorry

end tan_double_angle_solution_l320_32085


namespace weight_difference_l320_32075

-- Define the weights as real numbers
variable (W_A W_B W_C W_D W_E : ℝ)

-- Define the conditions
def condition1 : Prop := (W_A + W_B + W_C) / 3 = 80
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 82
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 81
def condition4 : Prop := W_A = 95
def condition5 : Prop := W_E > W_D

-- Theorem statement
theorem weight_difference (h1 : condition1 W_A W_B W_C)
                          (h2 : condition2 W_A W_B W_C W_D)
                          (h3 : condition3 W_B W_C W_D W_E)
                          (h4 : condition4 W_A)
                          (h5 : condition5 W_D W_E) : 
  W_E - W_D = 3 := by
  sorry

end weight_difference_l320_32075


namespace triangle_properties_l320_32077

noncomputable section

/-- Triangle ABC with internal angles A, B, C opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C ∧
  t.a * t.c * Real.cos t.B = -3

/-- The area of the triangle -/
def TriangleArea (t : Triangle) : ℝ :=
  (3 * Real.sqrt 3) / 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  TriangleArea t = (3 * Real.sqrt 3) / 2 ∧
  t.b ≥ Real.sqrt 6 :=
sorry

end triangle_properties_l320_32077


namespace recurring_decimal_sum_l320_32034

theorem recurring_decimal_sum : 
  let x := 1 / 3
  let y := 5 / 999
  let z := 7 / 9999
  x + y + z = 10170 / 29997 := by sorry

end recurring_decimal_sum_l320_32034


namespace garden_tomatoes_count_l320_32084

/-- Represents the garden layout and vegetable distribution --/
structure Garden where
  tomato_kinds : Nat
  cucumber_kinds : Nat
  cucumbers_per_kind : Nat
  potatoes : Nat
  rows : Nat
  spaces_per_row : Nat
  additional_capacity : Nat

/-- Calculates the number of tomatoes of each kind in the garden --/
def tomatoes_per_kind (g : Garden) : Nat :=
  let total_spaces := g.rows * g.spaces_per_row
  let occupied_spaces := g.cucumber_kinds * g.cucumbers_per_kind + g.potatoes
  let tomato_spaces := total_spaces - occupied_spaces - g.additional_capacity
  tomato_spaces / g.tomato_kinds

/-- Theorem stating that for the given garden configuration, 
    there are 5 tomatoes of each kind --/
theorem garden_tomatoes_count :
  let g : Garden := {
    tomato_kinds := 3,
    cucumber_kinds := 5,
    cucumbers_per_kind := 4,
    potatoes := 30,
    rows := 10,
    spaces_per_row := 15,
    additional_capacity := 85
  }
  tomatoes_per_kind g = 5 := by sorry

end garden_tomatoes_count_l320_32084


namespace monic_quartic_polynomial_value_l320_32083

-- Define a monic quartic polynomial
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + (f 0)

-- State the theorem
theorem monic_quartic_polynomial_value (f : ℝ → ℝ) :
  MonicQuarticPolynomial f →
  f (-1) = -1 →
  f 2 = -4 →
  f (-3) = -9 →
  f 4 = -16 →
  f 1 = 23 := by
  sorry

end monic_quartic_polynomial_value_l320_32083


namespace angle_q_measure_l320_32005

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  -- Angle measures in degrees
  angle_p : ℝ
  angle_q : ℝ
  angle_r : ℝ
  -- Triangle conditions
  sum_of_angles : angle_p + angle_q + angle_r = 180
  isosceles : angle_q = angle_r
  angle_r_five_times_p : angle_r = 5 * angle_p

/-- The measure of angle Q in the specified isosceles triangle is 900/11 degrees -/
theorem angle_q_measure (t : IsoscelesTriangle) : t.angle_q = 900 / 11 := by
  sorry

#check angle_q_measure

end angle_q_measure_l320_32005


namespace smallest_k_for_periodic_sum_l320_32089

/-- Represents a rational number with a periodic decimal representation -/
structure PeriodicDecimal where
  numerator : ℤ
  period : ℕ+

/-- Returns true if the given natural number is the minimal period of the decimal representation -/
def is_minimal_period (r : ℚ) (p : ℕ+) : Prop :=
  ∃ (m : ℤ), r = m / (10^p.val - 1) ∧ 
  ∀ (q : ℕ+), q < p → ¬∃ (n : ℤ), r = n / (10^q.val - 1)

theorem smallest_k_for_periodic_sum (a b : PeriodicDecimal) : 
  (is_minimal_period (a.numerator / (10^30 - 1)) 30) →
  (is_minimal_period (b.numerator / (10^30 - 1)) 30) →
  (is_minimal_period ((a.numerator - b.numerator) / (10^30 - 1)) 15) →
  (∀ k : ℕ, k < 6 → ¬is_minimal_period ((a.numerator + k * b.numerator) / (10^30 - 1)) 15) →
  is_minimal_period ((a.numerator + 6 * b.numerator) / (10^30 - 1)) 15 :=
by sorry

end smallest_k_for_periodic_sum_l320_32089


namespace f_inequality_l320_32037

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the condition that f(x) > f'(x) for all x
axiom f_greater_than_f' : ∀ x, f x > f' x

-- State the theorem to be proved
theorem f_inequality : 3 * f (Real.log 2) > 2 * f (Real.log 3) := by
  sorry

end f_inequality_l320_32037


namespace delivery_pay_difference_l320_32072

/-- Calculates the difference in pay between two delivery workers given their delivery counts and pay rate. -/
theorem delivery_pay_difference 
  (oula_deliveries : ℕ) 
  (tona_deliveries_ratio : ℚ) 
  (pay_per_delivery : ℕ) 
  (h1 : oula_deliveries = 96)
  (h2 : tona_deliveries_ratio = 3/4)
  (h3 : pay_per_delivery = 100) :
  (oula_deliveries * pay_per_delivery : ℕ) - (((tona_deliveries_ratio * oula_deliveries) : ℚ).floor * pay_per_delivery) = 2400 := by
  sorry

#check delivery_pay_difference

end delivery_pay_difference_l320_32072


namespace hyperbola_standard_form_l320_32015

-- Define the asymptotes of the hyperbola
def asymptote_slope : ℝ := 2

-- Define the ellipse that shares foci with the hyperbola
def ellipse_equation (x y : ℝ) : Prop := x^2 / 49 + y^2 / 24 = 1

-- Define the standard form of a hyperbola
def hyperbola_equation (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_standard_form :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), y = asymptote_slope * x ∨ y = -asymptote_slope * x) →
  (∀ (x y : ℝ), ellipse_equation x y ↔ 
    ∃ (x' y' : ℝ), hyperbola_equation a b x' y' ∧ 
    (x - x')^2 + (y - y')^2 = 0) →
  a^2 = 5 ∧ b^2 = 20 :=
sorry

end hyperbola_standard_form_l320_32015


namespace chess_team_probability_l320_32022

def chess_club_size : ℕ := 20
def num_boys : ℕ := 12
def num_girls : ℕ := 8
def team_size : ℕ := 4

theorem chess_team_probability :
  let total_combinations := Nat.choose chess_club_size team_size
  let all_boys_combinations := Nat.choose num_boys team_size
  let all_girls_combinations := Nat.choose num_girls team_size
  let probability_at_least_one_each := 1 - (all_boys_combinations + all_girls_combinations : ℚ) / total_combinations
  probability_at_least_one_each = 4280 / 4845 := by
  sorry

end chess_team_probability_l320_32022


namespace midpoint_one_sixth_one_ninth_l320_32092

theorem midpoint_one_sixth_one_ninth :
  (1 / 6 + 1 / 9) / 2 = 5 / 36 := by sorry

end midpoint_one_sixth_one_ninth_l320_32092


namespace triangle_value_l320_32042

def base_7_to_10 (a b : ℕ) : ℕ := a * 7 + b

def base_9_to_10 (a b : ℕ) : ℕ := a * 9 + b

theorem triangle_value :
  ∃! t : ℕ, t < 10 ∧ base_7_to_10 5 t = base_9_to_10 t 3 :=
by sorry

end triangle_value_l320_32042


namespace cubic_function_property_l320_32038

/-- Given a cubic function f(x) = ax³ - bx + 1 where a and b are real numbers,
    prove that if f(-2) = -1, then f(2) = 3. -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 - b * x + 1)
    (h2 : f (-2) = -1) : 
  f 2 = 3 := by
  sorry

end cubic_function_property_l320_32038


namespace coin_problem_l320_32002

theorem coin_problem (x : ℕ) :
  (x : ℚ) + x / 2 + x / 4 = 105 →
  x = 60 := by
sorry

end coin_problem_l320_32002


namespace apples_used_l320_32079

def initial_apples : ℕ := 40
def remaining_apples : ℕ := 39

theorem apples_used : initial_apples - remaining_apples = 1 := by
  sorry

end apples_used_l320_32079


namespace count_non_divisible_eq_31_l320_32009

/-- The product of proper positive integer divisors of n -/
def g_hat (n : ℕ) : ℕ := sorry

/-- Counts the number of integers n between 2 and 100 (inclusive) for which n does not divide g_hat(n) -/
def count_non_divisible : ℕ := sorry

/-- Theorem stating that the count of non-divisible numbers is 31 -/
theorem count_non_divisible_eq_31 : count_non_divisible = 31 := by sorry

end count_non_divisible_eq_31_l320_32009


namespace cryptarithmetic_puzzle_l320_32029

theorem cryptarithmetic_puzzle (F I V E G H T : ℕ) : 
  (F = 8) →
  (V % 2 = 1) →
  (100 * F + 10 * I + V + 100 * F + 10 * I + V = 10000 * E + 1000 * I + 100 * G + 10 * H + T) →
  (F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ G ∧ F ≠ H ∧ F ≠ T ∧
   I ≠ V ∧ I ≠ E ∧ I ≠ G ∧ I ≠ H ∧ I ≠ T ∧
   V ≠ E ∧ V ≠ G ∧ V ≠ H ∧ V ≠ T ∧
   E ≠ G ∧ E ≠ H ∧ E ≠ T ∧
   G ≠ H ∧ G ≠ T ∧
   H ≠ T) →
  (F < 10 ∧ I < 10 ∧ V < 10 ∧ E < 10 ∧ G < 10 ∧ H < 10 ∧ T < 10) →
  I = 2 := by
sorry

end cryptarithmetic_puzzle_l320_32029


namespace sqrt_equation_solution_l320_32004

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end sqrt_equation_solution_l320_32004


namespace remainder_divisibility_l320_32062

theorem remainder_divisibility (x : ℕ) (h1 : x > 1) (h2 : ¬ Nat.Prime x) 
  (h3 : 5000 % x = 25) : 9995 % x = 25 := by
  sorry

end remainder_divisibility_l320_32062


namespace blue_face_probability_is_five_eighths_l320_32045

/-- An octahedron with blue and red faces -/
structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (total_is_sum : total_faces = blue_faces + red_faces)
  (total_is_eight : total_faces = 8)

/-- The probability of rolling a blue face on an octahedron -/
def blue_face_probability (o : Octahedron) : ℚ :=
  o.blue_faces / o.total_faces

/-- Theorem: The probability of rolling a blue face on an octahedron with 5 blue faces out of 8 total faces is 5/8 -/
theorem blue_face_probability_is_five_eighths (o : Octahedron) 
  (h : o.blue_faces = 5) : blue_face_probability o = 5 / 8 := by
  sorry

end blue_face_probability_is_five_eighths_l320_32045
