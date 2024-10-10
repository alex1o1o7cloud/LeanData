import Mathlib

namespace tangent_line_at_one_l949_94913

def f (x : ℝ) := x^3 + x

theorem tangent_line_at_one :
  ∀ y : ℝ, (4 * 1 - y - 2 = 0) ↔ (∃ m : ℝ, m = (f 1 - f x) / (1 - x) ∧ y = m * (1 - x) + f 1) :=
by sorry

end tangent_line_at_one_l949_94913


namespace tea_mixture_ratio_l949_94989

/-- Proves that the ratio of tea at Rs. 64 per kg to tea at Rs. 74 per kg is 1:1 in a mixture worth Rs. 69 per kg -/
theorem tea_mixture_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  64 * x + 74 * y = 69 * (x + y) → x = y := by
  sorry

end tea_mixture_ratio_l949_94989


namespace contest_awards_l949_94975

theorem contest_awards (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  (n.factorial / (n - k).factorial) = 60 := by
  sorry

end contest_awards_l949_94975


namespace sphere_surface_area_from_volume_l949_94952

/-- Given a sphere with volume 36π cubic inches, its surface area is 36π square inches. -/
theorem sphere_surface_area_from_volume : 
  ∀ (r : ℝ), (4 / 3 : ℝ) * π * r^3 = 36 * π → 4 * π * r^2 = 36 * π :=
by
  sorry

end sphere_surface_area_from_volume_l949_94952


namespace f_negative_l949_94934

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x > 0
def f_positive (x : ℝ) : ℝ := x * (1 - x)

-- Theorem to prove
theorem f_negative (f : ℝ → ℝ) (h_odd : odd_function f) (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = x * (1 + x) :=
by sorry

end f_negative_l949_94934


namespace gymnastics_competition_participants_l949_94996

/-- Represents the structure of a gymnastics competition layout --/
structure GymnasticsCompetition where
  rows : ℕ
  columns : ℕ
  front_position : ℕ
  back_position : ℕ
  left_position : ℕ
  right_position : ℕ

/-- Calculates the total number of participants in the gymnastics competition --/
def total_participants (gc : GymnasticsCompetition) : ℕ :=
  gc.rows * gc.columns

/-- Theorem stating that the total number of participants is 425 --/
theorem gymnastics_competition_participants :
  ∀ (gc : GymnasticsCompetition),
    gc.front_position = 6 →
    gc.back_position = 12 →
    gc.left_position = 15 →
    gc.right_position = 11 →
    gc.columns = gc.front_position + gc.back_position - 1 →
    gc.rows = gc.left_position + gc.right_position - 1 →
    total_participants gc = 425 := by
  sorry

#check gymnastics_competition_participants

end gymnastics_competition_participants_l949_94996


namespace function_exponent_proof_l949_94942

theorem function_exponent_proof (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) → f 3 = Real.sqrt 3 → n = 1/2 := by
sorry

end function_exponent_proof_l949_94942


namespace function_bound_l949_94993

theorem function_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : 
  |a * x^2 + x - a| ≤ 5/4 := by sorry

end function_bound_l949_94993


namespace inequality_proof_l949_94920

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x) : 
  1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 ≥ 4 / (x*y + y*z + z*x) :=
by sorry

end inequality_proof_l949_94920


namespace shopping_trip_proof_l949_94971

def shopping_trip (initial_amount bag_price lunch_price : ℚ) : Prop :=
  let shoe_price : ℚ := 45
  let remaining : ℚ := 78
  initial_amount = 158 ∧
  bag_price = shoe_price - 17 ∧
  initial_amount = shoe_price + bag_price + lunch_price + remaining ∧
  lunch_price / bag_price = 1/4

theorem shopping_trip_proof : ∃ bag_price lunch_price : ℚ, shopping_trip 158 bag_price lunch_price :=
sorry

end shopping_trip_proof_l949_94971


namespace cubs_series_win_probability_l949_94939

def probability_cubs_win_game : ℚ := 2/3

def probability_cubs_win_series : ℚ :=
  (1 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^0) +
  (3 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^1) +
  (6 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^2)

theorem cubs_series_win_probability :
  probability_cubs_win_series = 64/81 :=
by sorry

end cubs_series_win_probability_l949_94939


namespace distance_between_cities_l949_94949

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- Yesterday's travel time from A to B in hours -/
def yesterday_time : ℝ := 6

/-- Today's travel time from B to A in hours -/
def today_time : ℝ := 4.5

/-- Time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- Average speed for the round trip if time were saved, in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  yesterday_time = 6 ∧
  today_time = 4.5 ∧
  (2 * distance) / (yesterday_time + today_time - 2 * time_saved) = average_speed :=
by sorry

end distance_between_cities_l949_94949


namespace max_regions_circular_disk_l949_94933

/-- 
Given a circular disk divided by 2n equally spaced radii (n > 0) and one chord,
the maximum number of non-overlapping regions is 3n + 1.
-/
theorem max_regions_circular_disk (n : ℕ) (h : n > 0) : 
  ∃ (num_regions : ℕ), num_regions = 3 * n + 1 ∧ 
  (∀ (m : ℕ), m ≤ num_regions) := by
sorry

end max_regions_circular_disk_l949_94933


namespace hyperbola_a_minus_h_l949_94951

/-- The standard form equation of a hyperbola -/
def is_hyperbola (a b h k x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- The equation of an asymptote -/
def is_asymptote (m c x y : ℝ) : Prop :=
  y = m * x + c

theorem hyperbola_a_minus_h (a b h k : ℝ) :
  a > 0 →
  b > 0 →
  is_asymptote 3 4 h k →
  is_asymptote (-3) 6 h k →
  is_hyperbola a b h k 1 9 →
  a - h = 2 * Real.sqrt 3 - 1/3 := by sorry

end hyperbola_a_minus_h_l949_94951


namespace min_max_sum_of_a_l949_94958

theorem min_max_sum_of_a (a b c : ℝ) (sum_eq : a + b + c = 5) (sum_sq_eq : a^2 + b^2 + c^2 = 8) :
  ∃ (m M : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8) → m ≤ x ∧ x ≤ M) ∧ m + M = 4 := by
  sorry

end min_max_sum_of_a_l949_94958


namespace payment_calculation_l949_94910

/-- Represents the pricing and discount options for suits and ties -/
structure StorePolicy where
  suit_price : ℕ
  tie_price : ℕ
  option1_free_ties : ℕ
  option2_discount : ℚ

/-- Calculates the payment for Option 1 -/
def option1_payment (policy : StorePolicy) (suits : ℕ) (ties : ℕ) : ℕ :=
  policy.suit_price * suits + policy.tie_price * (ties - suits)

/-- Calculates the payment for Option 2 -/
def option2_payment (policy : StorePolicy) (suits : ℕ) (ties : ℕ) : ℚ :=
  (1 - policy.option2_discount) * (policy.suit_price * suits + policy.tie_price * ties)

/-- Theorem statement for the payment calculations -/
theorem payment_calculation (x : ℕ) (h : x > 10) :
  let policy : StorePolicy := {
    suit_price := 1000,
    tie_price := 200,
    option1_free_ties := 1,
    option2_discount := 1/10
  }
  option1_payment policy 10 x = 200 * x + 8000 ∧
  option2_payment policy 10 x = 180 * x + 9000 := by
  sorry

end payment_calculation_l949_94910


namespace min_value_x_plus_y_l949_94990

theorem min_value_x_plus_y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 9 / (x + 1) + 1 / (y + 1) = 1) : 
  x + y ≥ 14 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9 / (x₀ + 1) + 1 / (y₀ + 1) = 1 ∧ x₀ + y₀ = 14 :=
sorry

end min_value_x_plus_y_l949_94990


namespace union_of_S_and_T_l949_94954

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_of_S_and_T : S ∪ T = {0, 1, 3} := by sorry

end union_of_S_and_T_l949_94954


namespace focus_of_parabola_is_correct_l949_94957

/-- The focus of a parabola y^2 = 12x --/
def focus_of_parabola : ℝ × ℝ := (3, 0)

/-- The equation of the parabola --/
def parabola_equation (x y : ℝ) : Prop := y^2 = 12 * x

/-- Theorem: The focus of the parabola y^2 = 12x is at the point (3, 0) --/
theorem focus_of_parabola_is_correct :
  let (a, b) := focus_of_parabola
  ∀ x y : ℝ, parabola_equation x y → (x - a)^2 + y^2 = (x + a)^2 :=
sorry

end focus_of_parabola_is_correct_l949_94957


namespace quadratic_root_problem_l949_94978

theorem quadratic_root_problem (v : ℚ) : 
  (3 * ((-12 - Real.sqrt 400) / 15)^2 + 12 * ((-12 - Real.sqrt 400) / 15) + v = 0) → 
  v = 704/75 := by
sorry

end quadratic_root_problem_l949_94978


namespace greater_number_problem_l949_94968

theorem greater_number_problem (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) (h3 : a > b) : a = 26 := by
  sorry

end greater_number_problem_l949_94968


namespace statement_B_is_incorrect_l949_94988

-- Define the basic types
def Chromosome : Type := String
def Allele : Type := String
def Genotype : Type := List Allele

-- Define the meiosis process
def meiosis (g : Genotype) : List Genotype := sorry

-- Define the normal chromosome distribution
def normalChromosomeDistribution (g : Genotype) : Prop := sorry

-- Define the statement B
def statementB : Prop :=
  ∃ (parent : Genotype) (sperm : Genotype),
    parent = ["A", "a", "X^b", "Y"] ∧
    sperm ∈ meiosis parent ∧
    sperm = ["A", "A", "a", "Y"] ∧
    (∃ (other_sperms : List Genotype),
      other_sperms.length = 3 ∧
      (∀ s ∈ other_sperms, s ∈ meiosis parent) ∧
      other_sperms = [["a", "Y"], ["X^b"], ["X^b"]])

-- Theorem stating that B is incorrect
theorem statement_B_is_incorrect :
  ¬statementB :=
sorry

end statement_B_is_incorrect_l949_94988


namespace ellipse_sum_parameters_l949_94901

/-- An ellipse with foci F₁ and F₂, and constant sum of distances 2a -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  a : ℝ

/-- The standard form equation of an ellipse -/
structure EllipseEquation where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, this function returns its standard form equation -/
def ellipse_to_equation (e : Ellipse) : EllipseEquation :=
  sorry

theorem ellipse_sum_parameters (e : Ellipse) (eq : EllipseEquation) :
  e.F₁ = (0, 0) →
  e.F₂ = (6, 0) →
  e.a = 5 →
  eq = ellipse_to_equation e →
  eq.h + eq.k + eq.a + eq.b = 12 := by
  sorry

end ellipse_sum_parameters_l949_94901


namespace product_of_numbers_l949_94908

theorem product_of_numbers (x y : ℝ) : 
  x + y = 24 → x^2 + y^2 = 404 → x * y = 86 := by
sorry

end product_of_numbers_l949_94908


namespace geometric_sequence_property_l949_94931

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 * a 5 = 1 →
  a 8 * a 9 = 16 →
  a 6 * a 7 = 4 := by
sorry

end geometric_sequence_property_l949_94931


namespace minimum_nickels_needed_l949_94967

def sneaker_cost : ℚ := 42.5
def tax_rate : ℚ := 0.08
def five_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 6
def quarters : ℕ := 10

def total_cost : ℚ := sneaker_cost * (1 + tax_rate)

def money_without_nickels : ℚ := 
  (five_dollar_bills * 5) + one_dollar_bills + (quarters * 0.25)

theorem minimum_nickels_needed :
  ∃ n : ℕ, 
    (money_without_nickels + n * 0.05 ≥ total_cost) ∧
    (∀ m : ℕ, m < n → money_without_nickels + m * 0.05 < total_cost) ∧
    n = 348 := by
  sorry

end minimum_nickels_needed_l949_94967


namespace prob_at_least_one_head_four_coins_prob_at_least_one_head_four_coins_is_15_16_l949_94947

/-- The probability of getting at least one head when tossing four fair coins -/
theorem prob_at_least_one_head_four_coins : ℝ :=
  let p_tail : ℝ := 1 / 2  -- probability of getting a tail on one coin toss
  let p_all_tails : ℝ := p_tail ^ 4  -- probability of getting all tails
  1 - p_all_tails

/-- Proof that the probability of getting at least one head when tossing four fair coins is 15/16 -/
theorem prob_at_least_one_head_four_coins_is_15_16 :
  prob_at_least_one_head_four_coins = 15 / 16 := by
  sorry

end prob_at_least_one_head_four_coins_prob_at_least_one_head_four_coins_is_15_16_l949_94947


namespace parabola_intercepts_sum_l949_94985

/-- Represents a parabola of the form x = 3y² - 9y + 5 --/
def Parabola (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- Theorem stating that for the given parabola, the sum of its x-intercept and y-intercepts is 8 --/
theorem parabola_intercepts_sum (a b c : ℝ) 
  (h_x_intercept : Parabola a 0)
  (h_y_intercept1 : Parabola 0 b)
  (h_y_intercept2 : Parabola 0 c)
  : a + b + c = 8 := by
  sorry

end parabola_intercepts_sum_l949_94985


namespace square_difference_value_l949_94927

theorem square_difference_value (a b : ℝ) 
  (h1 : 3 * (a + b) = 18) 
  (h2 : a - b = 4) : 
  a^2 - b^2 = 24 := by
sorry

end square_difference_value_l949_94927


namespace profit_percentage_is_25_percent_l949_94932

/-- Calculates the profit percentage given cost price, marked price, and discount rate. -/
def profit_percentage (cost_price marked_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let selling_price := marked_price * (1 - discount_rate)
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is 25% for the given conditions. -/
theorem profit_percentage_is_25_percent :
  profit_percentage (47.50 : ℚ) (62.5 : ℚ) (0.05 : ℚ) = 25 :=
by sorry

end profit_percentage_is_25_percent_l949_94932


namespace inverse_proportion_problem_l949_94966

/-- Given that x is inversely proportional to y, prove that when x = 8 and y = 16,
    then x = -4 when y = -32 -/
theorem inverse_proportion_problem (x y : ℝ) (c : ℝ) 
    (h1 : x * y = c)  -- x is inversely proportional to y
    (h2 : 8 * 16 = c) -- When x = 8, y = 16
    (h3 : y = -32)    -- Given y = -32
    : x = -4 := by
  sorry

end inverse_proportion_problem_l949_94966


namespace most_accurate_estimate_l949_94979

-- Define the temperature range
def lower_bound : Float := 98.6
def upper_bound : Float := 99.1

-- Define a type for temperature readings
structure TemperatureReading where
  value : Float
  is_within_range : lower_bound ≤ value ∧ value ≤ upper_bound

-- Define a function to determine if a reading is closer to the upper bound
def closer_to_upper_bound (reading : TemperatureReading) : Prop :=
  reading.value > (lower_bound + upper_bound) / 2

-- Theorem statement
theorem most_accurate_estimate (reading : TemperatureReading) 
  (h : closer_to_upper_bound reading) : 
  upper_bound = 99.1 ∧ upper_bound - reading.value < reading.value - lower_bound :=
by
  sorry

end most_accurate_estimate_l949_94979


namespace smallest_sum_c_d_l949_94937

theorem smallest_sum_c_d (c d : ℝ) (hc : c > 0) (hd : d > 0)
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ (192/81)^(1/3) + (12 * (192/81)^(1/3))^(1/2) := by
  sorry

end smallest_sum_c_d_l949_94937


namespace expression_value_l949_94921

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 := by
  sorry

end expression_value_l949_94921


namespace rainbow_preschool_full_day_students_l949_94925

theorem rainbow_preschool_full_day_students 
  (total_students : ℕ) 
  (half_day_percentage : ℚ) 
  (h1 : total_students = 80)
  (h2 : half_day_percentage = 1/4) : 
  (1 - half_day_percentage) * total_students = 60 := by
  sorry

end rainbow_preschool_full_day_students_l949_94925


namespace two_point_zero_six_recurring_l949_94902

def recurring_decimal_02 : ℚ := 2 / 99

theorem two_point_zero_six_recurring (h : recurring_decimal_02 = 2 / 99) :
  2 + 3 * recurring_decimal_02 = 68 / 33 := by
  sorry

end two_point_zero_six_recurring_l949_94902


namespace no_equal_roots_for_quadratic_l949_94911

theorem no_equal_roots_for_quadratic :
  ¬ ∃ k : ℝ, ∃ x : ℝ, x^2 - (k + 1) * x + (k - 3) = 0 ∧
    ∀ y : ℝ, y^2 - (k + 1) * y + (k - 3) = 0 → y = x := by
  sorry

end no_equal_roots_for_quadratic_l949_94911


namespace hyperbola_axis_ratio_l949_94980

/-- The ratio of the real semi-axis length to the imaginary axis length of the hyperbola 2x^2 - y^2 = 8 -/
theorem hyperbola_axis_ratio : ∃ (a b : ℝ), 
  (∀ x y : ℝ, 2 * x^2 - y^2 = 8 ↔ x^2 / (2 * a^2) - y^2 / (2 * b^2) = 1) ∧ 
  (a / (2 * b) = Real.sqrt 2 / 4) := by
  sorry

end hyperbola_axis_ratio_l949_94980


namespace complete_quadrilateral_l949_94983

/-- A point in the projective plane -/
structure ProjPoint where
  x : ℝ
  y : ℝ
  z : ℝ
  nontrivial : (x, y, z) ≠ (0, 0, 0)

/-- A line in the projective plane -/
structure ProjLine where
  a : ℝ
  b : ℝ
  c : ℝ
  nontrivial : (a, b, c) ≠ (0, 0, 0)

/-- The cross ratio of four collinear points -/
def cross_ratio (A B C D : ProjPoint) : ℝ := sorry

/-- Intersection of two lines -/
def intersect (l1 l2 : ProjLine) : ProjPoint := sorry

/-- Line passing through two points -/
def line_through (A B : ProjPoint) : ProjLine := sorry

theorem complete_quadrilateral 
  (A B C D : ProjPoint) 
  (P : ProjPoint := intersect (line_through A B) (line_through C D))
  (Q : ProjPoint := intersect (line_through A D) (line_through B C))
  (R : ProjPoint := intersect (line_through A C) (line_through B D))
  (K : ProjPoint := intersect (line_through Q R) (line_through A B))
  (L : ProjPoint := intersect (line_through Q R) (line_through C D)) :
  cross_ratio Q R K L = -1 := by
  sorry

end complete_quadrilateral_l949_94983


namespace sum_of_coefficients_l949_94991

theorem sum_of_coefficients (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end sum_of_coefficients_l949_94991


namespace sum_of_roots_l949_94950

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by sorry

end sum_of_roots_l949_94950


namespace school_boys_count_l949_94997

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 80 →
  boys = 50 := by
sorry

end school_boys_count_l949_94997


namespace triangle_area_l949_94976

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (7, 1)
def C : ℝ × ℝ := (5, 6)

-- State the theorem
theorem triangle_area : 
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 13 := by sorry

end triangle_area_l949_94976


namespace edge_coloring_theorem_l949_94940

/-- Given a complete graph K_n with n vertices, this theorem states that:
    1) If we color the edges with at least n colors, there will be a triangle with all edges in different colors.
    2) If we color the edges with at most n-3 colors, there will be a cycle of length 3 or 4 with all edges in the same color.
    3) For n = 2023, it's possible to color the edges using 2022 colors without violating the conditions,
       and it's also possible using 2020 colors without violating the conditions.
    4) The difference between the maximum and minimum number of colors that satisfy the conditions is 2. -/
theorem edge_coloring_theorem (n : ℕ) (h : n = 2023) :
  (∀ (coloring : Fin n → Fin n → Fin n), 
    (∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
      coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
  (∀ (coloring : Fin n → Fin n → Fin (n-3)), 
    (∃ (i j k l : Fin n), (i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i) ∧ 
      ((coloring i j = coloring j k ∧ coloring j k = coloring k i) ∨
       (coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i)))) ∧
  (∃ (coloring : Fin n → Fin n → Fin 2022), 
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring i k) ∧
      ¬(coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
    (∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i))) ∧
  (∃ (coloring : Fin n → Fin n → Fin 2020), 
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring i k) ∧
      ¬(coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
    (∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i))) ∧
  (2022 - 2020 = 2) :=
by sorry


end edge_coloring_theorem_l949_94940


namespace hyperbola_eccentricity_l949_94930

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the circle F
def circle_F (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 3 = 0

-- Define the right focus F of hyperbola C
def right_focus (c : ℝ) : Prop :=
  c = 2

-- Define the distance from F to asymptote
def distance_to_asymptote (b : ℝ) : Prop :=
  b = 1

-- Theorem statement
theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ),
  (∃ x y, hyperbola_C x y a b) →
  (∃ x y, circle_F x y) →
  right_focus c →
  distance_to_asymptote b →
  c^2 = a^2 + b^2 →
  c / a = 2 * Real.sqrt 3 / 3 :=
sorry

end hyperbola_eccentricity_l949_94930


namespace loan_principal_calculation_l949_94955

/-- Simple interest calculation -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_principal_calculation (rate time interest : ℚ) 
  (h_rate : rate = 12)
  (h_time : time = 3)
  (h_interest : interest = 7200) :
  ∃ principal : ℚ, simpleInterest principal rate time = interest ∧ principal = 20000 := by
  sorry

#check loan_principal_calculation

end loan_principal_calculation_l949_94955


namespace log3_of_9_cubed_l949_94915

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_of_9_cubed : log3 (9^3) = 6 := by
  sorry

end log3_of_9_cubed_l949_94915


namespace shifted_function_is_linear_l949_94929

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Represents a horizontal shift transformation on a function -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x - shift)

/-- The original direct proportion function y = -2x -/
def originalFunction : ℝ → ℝ :=
  λ x => -2 * x

/-- The result of shifting the original function 3 units to the right -/
def shiftedFunction : ℝ → ℝ :=
  horizontalShift originalFunction 3

theorem shifted_function_is_linear :
  ∃ (k b : ℝ), k ≠ 0 ∧ (∀ x, shiftedFunction x = k * x + b) ∧ k = -2 ∧ b = 6 := by
  sorry

end shifted_function_is_linear_l949_94929


namespace greatest_divisor_with_remainders_l949_94972

theorem greatest_divisor_with_remainders : Nat.gcd (60 - 6) (190 - 10) = 18 := by
  sorry

end greatest_divisor_with_remainders_l949_94972


namespace star_operation_example_l949_94995

def star_operation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem star_operation_example :
  let A : Set ℕ := {1,3,5,7}
  let B : Set ℕ := {2,3,5}
  star_operation A B = {1,7} := by
sorry

end star_operation_example_l949_94995


namespace savanna_animal_count_l949_94944

/-- The number of animals in Savanna National Park -/
def savanna_total (safari_lions : ℕ) : ℕ :=
  let safari_snakes := safari_lions / 2
  let safari_giraffes := safari_snakes - 10
  let savanna_lions := safari_lions * 2
  let savanna_snakes := safari_snakes * 3
  let savanna_giraffes := safari_giraffes + 20
  savanna_lions + savanna_snakes + savanna_giraffes

/-- Theorem stating the total number of animals in Savanna National Park -/
theorem savanna_animal_count : savanna_total 100 = 410 := by
  sorry

end savanna_animal_count_l949_94944


namespace other_root_of_quadratic_l949_94935

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = 2) :=
sorry

end other_root_of_quadratic_l949_94935


namespace max_value_trig_sum_l949_94973

theorem max_value_trig_sum (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end max_value_trig_sum_l949_94973


namespace horner_method_v3_l949_94909

def horner_polynomial (x : ℚ) : ℚ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def horner_v3 (x : ℚ) : ℚ :=
  let v0 := 2
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 23

theorem horner_method_v3 :
  horner_v3 (-4) = -49 :=
by sorry

end horner_method_v3_l949_94909


namespace planes_parallel_conditions_l949_94953

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the theorem
theorem planes_parallel_conditions 
  (α β : Plane) 
  (h_different : α ≠ β) :
  (∃ a : Line, perpendicular a α ∧ perpendicular a β → plane_parallel α β) ∧
  (∃ a b : Line, skew a b ∧ contains α a ∧ contains β b ∧ 
    parallel a β ∧ parallel b α → plane_parallel α β) :=
sorry

end planes_parallel_conditions_l949_94953


namespace inverse_of_A_l949_94904

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -3; -2, 1]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![-1/2, -3/2; -1, -2]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l949_94904


namespace isosceles_trapezoid_area_l949_94977

/-- Isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  /-- Length of lateral sides AB and CD -/
  lateral_side : ℝ
  /-- Ratio of AH : AK : AC -/
  ratio_ah : ℝ
  ratio_ak : ℝ
  ratio_ac : ℝ
  /-- Conditions -/
  lateral_positive : lateral_side > 0
  ratio_positive : ratio_ah > 0 ∧ ratio_ak > 0 ∧ ratio_ac > 0
  ratio_order : ratio_ah < ratio_ak ∧ ratio_ak < ratio_ac

/-- The area of the isosceles trapezoid with given properties is 180 -/
theorem isosceles_trapezoid_area
  (t : IsoscelesTrapezoid)
  (h1 : t.lateral_side = 10)
  (h2 : t.ratio_ah = 5 ∧ t.ratio_ak = 14 ∧ t.ratio_ac = 15) :
  ∃ (area : ℝ), area = 180 :=
sorry

end isosceles_trapezoid_area_l949_94977


namespace imaginary_part_of_z_times_i_l949_94962

-- Define the complex number z
def z : ℂ := 4 - 8 * Complex.I

-- State the theorem
theorem imaginary_part_of_z_times_i : Complex.im (z * Complex.I) = 4 := by
  sorry

end imaginary_part_of_z_times_i_l949_94962


namespace square_sum_reciprocal_l949_94963

theorem square_sum_reciprocal (x : ℝ) (h : 18 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 20 := by
  sorry

end square_sum_reciprocal_l949_94963


namespace intersection_and_range_l949_94982

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 4 / (x + 3) ≤ 1}
def C (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 ≤ 0}

theorem intersection_and_range :
  (A ∩ B = {x : ℝ | -4 < x ∧ x < -3 ∨ 1 ≤ x ∧ x < 3}) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ C m → x ∈ A) ∧ (∃ x : ℝ, x ∈ C m ∧ x ∉ A) ↔ -3 < m ∧ m < 2) :=
sorry

end intersection_and_range_l949_94982


namespace min_balls_for_three_same_color_l949_94917

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  white : Nat
  black : Nat
  blue : Nat

/-- Calculates the minimum number of balls to draw to ensure at least three of the same color -/
def minBallsToEnsureThreeSameColor (bag : BagContents) : Nat :=
  7

/-- Theorem stating that for a bag with 5 white, 5 black, and 2 blue balls,
    the minimum number of balls to draw to ensure at least three of the same color is 7 -/
theorem min_balls_for_three_same_color :
  let bag : BagContents := { white := 5, black := 5, blue := 2 }
  minBallsToEnsureThreeSameColor bag = 7 := by
  sorry

end min_balls_for_three_same_color_l949_94917


namespace basketball_highlight_film_avg_player_footage_l949_94998

/-- Calculates the average player footage in minutes for a basketball highlight film --/
theorem basketball_highlight_film_avg_player_footage
  (point_guard_footage : ℕ)
  (shooting_guard_footage : ℕ)
  (small_forward_footage : ℕ)
  (power_forward_footage : ℕ)
  (center_footage : ℕ)
  (game_footage : ℕ)
  (interview_footage : ℕ)
  (opening_closing_footage : ℕ)
  (h1 : point_guard_footage = 130)
  (h2 : shooting_guard_footage = 145)
  (h3 : small_forward_footage = 85)
  (h4 : power_forward_footage = 60)
  (h5 : center_footage = 180)
  (h6 : game_footage = 120)
  (h7 : interview_footage = 90)
  (h8 : opening_closing_footage = 30) :
  (point_guard_footage + shooting_guard_footage + small_forward_footage + power_forward_footage + center_footage) / (5 * 60) = 2 :=
by sorry

end basketball_highlight_film_avg_player_footage_l949_94998


namespace weight_of_smaller_cube_l949_94961

/-- Given two cubes of the same material, where the second cube has sides twice as long as the first
and weighs 64 pounds, the weight of the first cube is 8 pounds. -/
theorem weight_of_smaller_cube (s : ℝ) (weight_first : ℝ) (weight_second : ℝ) : 
  s > 0 → 
  weight_second = 64 → 
  (2 * s)^3 / s^3 * weight_first = weight_second → 
  weight_first = 8 :=
by sorry

end weight_of_smaller_cube_l949_94961


namespace euclidean_division_123456789_by_37_l949_94959

theorem euclidean_division_123456789_by_37 :
  ∃ (q r : ℤ), 123456789 = 37 * q + r ∧ 0 ≤ r ∧ r < 37 ∧ q = 3336669 ∧ r = 36 := by
  sorry

end euclidean_division_123456789_by_37_l949_94959


namespace complex_exponential_sum_l949_94946

theorem complex_exponential_sum : 
  12 * Complex.exp (Complex.I * Real.pi / 7) + 12 * Complex.exp (Complex.I * 19 * Real.pi / 14) = 
  24 * Real.cos (5 * Real.pi / 28) * Complex.exp (Complex.I * 3 * Real.pi / 4) := by
  sorry

end complex_exponential_sum_l949_94946


namespace intersection_point_unique_l949_94981

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/3, -2)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x - 1

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y + 4 = -6 * x

theorem intersection_point_unique :
  (∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point) :=
sorry

end intersection_point_unique_l949_94981


namespace quadratic_inequality_properties_l949_94923

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 3) ↔ a * x^2 + b * x + c ≥ 0) →
  (a > 0 ∧
   (∀ x : ℝ, bx + c > 0 ↔ x < -6) ∧
   a + b + c < 0) :=
by sorry

end quadratic_inequality_properties_l949_94923


namespace stamp_distribution_l949_94994

theorem stamp_distribution (total : ℕ) (x y : ℕ) 
  (h1 : total = 70)
  (h2 : x = 4 * y + 5)
  (h3 : x + y = total) :
  x = 57 ∧ y = 13 := by
  sorry

end stamp_distribution_l949_94994


namespace escalator_travel_time_l949_94984

/-- Proves that a person walking on a moving escalator takes 8 seconds to cover its length --/
theorem escalator_travel_time 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (person_speed : ℝ) 
  (h1 : escalator_speed = 10) 
  (h2 : escalator_length = 112) 
  (h3 : person_speed = 4) : 
  escalator_length / (escalator_speed + person_speed) = 8 := by
  sorry

end escalator_travel_time_l949_94984


namespace problem_solution_l949_94906

theorem problem_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h1 : 1/p + 1/q = 2) (h2 : p*q = 1) : p = 1 := by
  sorry

end problem_solution_l949_94906


namespace complex_magnitude_equation_l949_94907

theorem complex_magnitude_equation :
  ∃! (x : ℝ), x > 0 ∧ Complex.abs (x - 3 * Complex.I * Real.sqrt 5) * Complex.abs (8 - 5 * Complex.I) = 50 := by
  sorry

end complex_magnitude_equation_l949_94907


namespace mean_of_three_numbers_l949_94919

theorem mean_of_three_numbers (p q r : ℝ) : 
  (p + q) / 2 = 13 →
  (q + r) / 2 = 16 →
  (r + p) / 2 = 7 →
  (p + q + r) / 3 = 12 := by
sorry

end mean_of_three_numbers_l949_94919


namespace rent_during_harvest_l949_94987

/-- The total rent paid during the harvest season -/
def total_rent (weekly_rent : ℕ) (weeks : ℕ) : ℕ :=
  weekly_rent * weeks

/-- Proof that the total rent paid during the harvest season is $526,692 -/
theorem rent_during_harvest : total_rent 388 1359 = 526692 := by
  sorry

end rent_during_harvest_l949_94987


namespace distance_traveled_l949_94970

/-- Calculates the actual distance traveled given the conditions of the problem -/
def actual_distance (actual_speed hours_walked : ℝ) : ℝ :=
  actual_speed * hours_walked

/-- Represents the additional distance that would be covered at the higher speed -/
def additional_distance (actual_speed higher_speed hours_walked : ℝ) : ℝ :=
  (higher_speed - actual_speed) * hours_walked

theorem distance_traveled (actual_speed higher_speed additional : ℝ) 
  (h1 : actual_speed = 12)
  (h2 : higher_speed = 20)
  (h3 : additional = 30) :
  ∃ (hours_walked : ℝ), 
    additional_distance actual_speed higher_speed hours_walked = additional ∧ 
    actual_distance actual_speed hours_walked = 45 := by
  sorry

end distance_traveled_l949_94970


namespace books_loaned_out_l949_94964

/-- The number of books in the special collection at the beginning of the month -/
def initial_books : ℕ := 150

/-- The percentage of loaned books that are returned -/
def return_rate : ℚ := 85 / 100

/-- The number of books in the special collection at the end of the month -/
def final_books : ℕ := 135

/-- The number of books damaged or lost and replaced -/
def replaced_books : ℕ := 5

/-- The number of books loaned out during the month -/
def loaned_books : ℕ := 133

theorem books_loaned_out : 
  initial_books - loaned_books + (return_rate * loaned_books).floor + replaced_books = final_books :=
sorry

end books_loaned_out_l949_94964


namespace solve_system_l949_94992

theorem solve_system (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := by
  sorry

end solve_system_l949_94992


namespace product_of_one_plus_tangents_sine_double_angle_l949_94986

-- Part I
theorem product_of_one_plus_tangents (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : α + β = π/4) : 
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

-- Part II
theorem sine_double_angle (α β : Real) 
  (h1 : π/2 < β ∧ β < α ∧ α < 3*π/4) 
  (h2 : Real.cos (α - β) = 12/13) 
  (h3 : Real.sin (α + β) = -3/5) : 
  Real.sin (2 * α) = -56/65 := by
  sorry

end product_of_one_plus_tangents_sine_double_angle_l949_94986


namespace flower_planting_cost_l949_94956

theorem flower_planting_cost (flower_cost soil_cost clay_pot_cost total_cost : ℕ) : 
  flower_cost = 9 →
  clay_pot_cost = flower_cost + 20 →
  soil_cost < flower_cost →
  total_cost = 45 →
  total_cost = flower_cost + clay_pot_cost + soil_cost →
  flower_cost - soil_cost = 2 := by
sorry

end flower_planting_cost_l949_94956


namespace walker_catch_up_equations_l949_94903

theorem walker_catch_up_equations 
  (good_efficiency bad_efficiency initial_lead : ℕ) 
  (h_efficiency : good_efficiency > bad_efficiency) 
  (h_initial_lead : initial_lead > 0) : 
  ∃ (x y : ℚ), 
    x - y = initial_lead ∧ 
    x = (good_efficiency : ℚ) / bad_efficiency * y ∧ 
    x > 0 ∧ y > 0 := by
  sorry

end walker_catch_up_equations_l949_94903


namespace prime_square_minus_one_divisible_by_thirty_l949_94969

theorem prime_square_minus_one_divisible_by_thirty (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  30 ∣ p^2 - 1 := by
  sorry

end prime_square_minus_one_divisible_by_thirty_l949_94969


namespace inequality_sum_l949_94926

theorem inequality_sum (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) (h6 : d ≠ 0) :
  a + c > b + d := by
  sorry

end inequality_sum_l949_94926


namespace q_polynomial_expression_l949_94974

theorem q_polynomial_expression (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 8 * x^2) = (5 * x^4 + 18 * x^3 + 20 * x^2 + 2)) →
  (∀ x, q x = -2 * x^6 + x^4 + 18 * x^3 + 12 * x^2 + 2) :=
by
  sorry

end q_polynomial_expression_l949_94974


namespace right_triangle_side_length_l949_94965

/-- Given a right triangle with acute angles in the ratio 5:4 and hypotenuse 10 cm,
    the length of the side opposite the smaller angle is 10 * sin(40°) -/
theorem right_triangle_side_length (a b c : ℝ) (θ₁ θ₂ : Real) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right triangle condition)
  c = 10 →  -- hypotenuse length
  θ₁ / θ₂ = 5 / 4 →  -- ratio of acute angles
  θ₁ + θ₂ = π / 2 →  -- sum of acute angles in a right triangle
  θ₂ < θ₁ →  -- θ₂ is the smaller angle
  b = 10 * Real.sin (40 * π / 180) :=
by sorry

end right_triangle_side_length_l949_94965


namespace ivy_morning_cupcakes_l949_94999

/-- The number of cupcakes Ivy baked in the morning -/
def morning_cupcakes : ℕ := sorry

/-- The number of cupcakes Ivy baked in the afternoon -/
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

/-- The total number of cupcakes Ivy baked -/
def total_cupcakes : ℕ := 55

/-- Theorem stating that Ivy baked 20 cupcakes in the morning -/
theorem ivy_morning_cupcakes : 
  morning_cupcakes = 20 ∧ 
  afternoon_cupcakes = morning_cupcakes + 15 ∧ 
  total_cupcakes = morning_cupcakes + afternoon_cupcakes := by
  sorry

end ivy_morning_cupcakes_l949_94999


namespace total_toys_count_l949_94924

/-- The number of toy cars given to boys -/
def toy_cars : ℕ := 134

/-- The number of dolls given to girls -/
def dolls : ℕ := 269

/-- The total number of toys given -/
def total_toys : ℕ := toy_cars + dolls

theorem total_toys_count : total_toys = 403 := by
  sorry

end total_toys_count_l949_94924


namespace bouncing_ball_distance_l949_94945

/-- The total distance traveled by a bouncing ball -/
def total_distance (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + initial_height * rebound_ratio + initial_height * rebound_ratio

/-- Theorem: A ball dropped from 100 cm with 50% rebound travels 200 cm when it touches the floor the third time -/
theorem bouncing_ball_distance :
  total_distance 100 0.5 = 200 := by
  sorry

end bouncing_ball_distance_l949_94945


namespace output_for_15_l949_94943

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 23 := by
  sorry

end output_for_15_l949_94943


namespace no_factor_of_polynomial_l949_94900

theorem no_factor_of_polynomial : ¬ ∃ (p : Polynomial ℝ), 
  (p = X^2 + 4*X + 4 ∨ 
   p = X^2 - 4*X + 4 ∨ 
   p = X^2 + 2*X + 4 ∨ 
   p = X^2 + 4) ∧ 
  (∃ (q : Polynomial ℝ), X^4 - 4*X^2 + 16 = p * q) := by
  sorry

end no_factor_of_polynomial_l949_94900


namespace circle_point_range_l949_94941

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Main theorem
theorem circle_point_range (m : ℝ) :
  m > 0 →
  (∃ a b : ℝ, C a b ∧
    dot_product (a + m, b) (a - m, b) = 0) →
  4 ≤ m ∧ m ≤ 6 :=
by sorry

end circle_point_range_l949_94941


namespace max_k_value_l949_94928

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) :=
sorry

end max_k_value_l949_94928


namespace negation_of_proposition_l949_94938

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x > 0, (x + 1) * Real.exp x > 1) ↔ (∃ x₀ > 0, (x₀ + 1) * Real.exp x₀ ≤ 1) := by
  sorry

end negation_of_proposition_l949_94938


namespace first_cube_weight_l949_94936

/-- Given two cubical blocks of the same metal, where the sides of the second cube
    are twice as long as the first cube, and the second cube weighs 24 pounds,
    prove that the weight of the first cubical block is 3 pounds. -/
theorem first_cube_weight (s : ℝ) (weight : ℝ → ℝ) :
  (∀ x, weight (8 * x) = 8 * weight x) →  -- Weight is proportional to volume
  weight (8 * s^3) = 24 →                 -- Second cube weighs 24 pounds
  weight (s^3) = 3 :=
by sorry

end first_cube_weight_l949_94936


namespace existence_of_ratio_triplet_l949_94914

def TwoColorFunction := ℕ → Bool

theorem existence_of_ratio_triplet (color : TwoColorFunction) :
  ∃ A B C : ℕ, (color A = color B) ∧ (color B = color C) ∧ (A * B = C * C) := by
  sorry

end existence_of_ratio_triplet_l949_94914


namespace tetrahedron_distance_altitude_inequality_l949_94905

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The minimum distance between any pair of opposite edges -/
  d : ℝ
  /-- The length of the shortest altitude -/
  h : ℝ
  /-- Assumption that d and h are positive -/
  d_pos : d > 0
  h_pos : h > 0

/-- Theorem: For any tetrahedron, twice the minimum distance between opposite edges
    is greater than the length of the shortest altitude -/
theorem tetrahedron_distance_altitude_inequality (t : Tetrahedron) : 2 * t.d > t.h := by
  sorry

end tetrahedron_distance_altitude_inequality_l949_94905


namespace least_coins_ten_coins_coins_in_wallet_l949_94918

theorem least_coins (b : ℕ) : b ≡ 3 [ZMOD 7] ∧ b ≡ 2 [ZMOD 4] → b ≥ 10 := by
  sorry

theorem ten_coins : 10 ≡ 3 [ZMOD 7] ∧ 10 ≡ 2 [ZMOD 4] := by
  sorry

theorem coins_in_wallet : ∃ (b : ℕ), b ≡ 3 [ZMOD 7] ∧ b ≡ 2 [ZMOD 4] ∧ 
  ∀ (n : ℕ), n ≡ 3 [ZMOD 7] ∧ n ≡ 2 [ZMOD 4] → b ≤ n := by
  sorry

end least_coins_ten_coins_coins_in_wallet_l949_94918


namespace not_divisible_by_fifteen_l949_94916

theorem not_divisible_by_fifteen (a : ℤ) : ¬ (15 ∣ (a^2 + a + 2)) := by
  sorry

end not_divisible_by_fifteen_l949_94916


namespace piano_harmonies_count_l949_94922

theorem piano_harmonies_count : 
  (Nat.choose 7 3) + (Nat.choose 7 4) + (Nat.choose 7 5) + (Nat.choose 7 6) + (Nat.choose 7 7) = 99 := by
  sorry

end piano_harmonies_count_l949_94922


namespace m_value_l949_94948

theorem m_value (a b m : ℝ) 
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end m_value_l949_94948


namespace square_diff_div_81_l949_94960

theorem square_diff_div_81 : (2500 - 2409)^2 / 81 = 102 := by sorry

end square_diff_div_81_l949_94960


namespace integral_reciprocal_plus_one_l949_94912

theorem integral_reciprocal_plus_one (u : ℝ) : 
  ∫ x in (0:ℝ)..(1:ℝ), 1 / (x + 1) = Real.log 2 := by
  sorry

end integral_reciprocal_plus_one_l949_94912
