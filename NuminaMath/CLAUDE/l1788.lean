import Mathlib

namespace set_operations_l1788_178875

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) := by
  sorry

end set_operations_l1788_178875


namespace fullPriceRevenue_is_600_l1788_178856

/-- Represents the fundraiser event ticket sales -/
structure FundraiserEvent where
  totalTickets : ℕ
  totalRevenue : ℕ
  fullPrice : ℕ
  halfPrice : ℕ
  fullPriceRevenue : ℕ

/-- The fundraiser event satisfies the given conditions -/
def validFundraiserEvent (e : FundraiserEvent) : Prop :=
  e.totalTickets = 200 ∧
  e.totalRevenue = 2700 ∧
  e.fullPrice > 0 ∧
  e.halfPrice = e.fullPrice / 2 ∧
  e.totalTickets = (e.totalRevenue - e.fullPriceRevenue) / e.halfPrice + e.fullPriceRevenue / e.fullPrice

/-- The theorem stating that the full-price ticket revenue is $600 -/
theorem fullPriceRevenue_is_600 (e : FundraiserEvent) (h : validFundraiserEvent e) : 
  e.fullPriceRevenue = 600 := by
  sorry

end fullPriceRevenue_is_600_l1788_178856


namespace quadratic_real_root_condition_l1788_178864

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := by
  sorry

end quadratic_real_root_condition_l1788_178864


namespace shifted_quadratic_function_l1788_178877

/-- The original quadratic function -/
def original_function (x : ℝ) : ℝ := x^2

/-- The shifted function -/
def shifted_function (x : ℝ) : ℝ := (x - 3)^2 - 2

/-- Theorem stating that the shifted function is equivalent to shifting the original function -/
theorem shifted_quadratic_function (x : ℝ) : 
  shifted_function x = original_function (x - 3) - 2 := by sorry

end shifted_quadratic_function_l1788_178877


namespace equation_result_l1788_178805

theorem equation_result (x : ℝ) : 
  14 * x + 5 - 21 * x^2 = -2 → 6 * x^2 - 4 * x + 5 = 7 := by
  sorry

end equation_result_l1788_178805


namespace units_digit_of_17_pow_2041_l1788_178882

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the main theorem
theorem units_digit_of_17_pow_2041 : unitsDigit (17^2041) = 7 := by
  sorry

end units_digit_of_17_pow_2041_l1788_178882


namespace units_digit_of_7_cubed_l1788_178893

theorem units_digit_of_7_cubed : (7^3) % 10 = 3 := by
  sorry

end units_digit_of_7_cubed_l1788_178893


namespace expression_evaluation_l1788_178809

theorem expression_evaluation :
  let x : ℝ := 3
  let numerator := 4 + x^2 - x*(2+x) - 2^2
  let denominator := x^2 - 2*x + 3
  numerator / denominator = -1 := by
sorry

end expression_evaluation_l1788_178809


namespace income_calculation_l1788_178851

theorem income_calculation (a b c d e : ℝ) : 
  (a + b) / 2 = 4050 →
  (b + c) / 2 = 5250 →
  (a + c) / 2 = 4200 →
  (a + b + d) / 3 = 4800 →
  (c + d + e) / 3 = 6000 →
  (b + a + e) / 3 = 4500 →
  a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400 :=
by sorry

end income_calculation_l1788_178851


namespace reflection_of_circle_center_l1788_178852

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (8, -3)

/-- The expected center of the reflected circle -/
def expected_reflected_center : ℝ × ℝ := (3, -8)

theorem reflection_of_circle_center :
  reflect_about_diagonal original_center = expected_reflected_center := by
  sorry

end reflection_of_circle_center_l1788_178852


namespace hotel_room_encoding_l1788_178847

theorem hotel_room_encoding (x : ℕ) : 
  1 ≤ x ∧ x ≤ 30 ∧ x % 5 = 1 ∧ x % 7 = 6 → x = 13 := by
  sorry

end hotel_room_encoding_l1788_178847


namespace other_metal_price_l1788_178879

/-- Given the price of Metal A, the ratio of Metal A to another metal, and the cost of their alloy,
    this theorem proves the price of the other metal. -/
theorem other_metal_price
  (price_a : ℝ)
  (ratio : ℝ)
  (alloy_cost : ℝ)
  (h1 : price_a = 68)
  (h2 : ratio = 3)
  (h3 : alloy_cost = 75) :
  (4 * alloy_cost - 3 * price_a) = 96 := by
  sorry

end other_metal_price_l1788_178879


namespace inequality_solution_set_l1788_178870

theorem inequality_solution_set (x : ℝ) : x^2 - |x| - 6 < 0 ↔ -3 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l1788_178870


namespace figure_can_form_square_l1788_178878

/-- Represents a figure drawn on squared paper -/
structure Figure where
  -- Add necessary fields to represent the figure

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to cut a figure into triangles -/
def cut_into_triangles (f : Figure) : List Triangle :=
  sorry

/-- Function to check if a list of triangles can form a square -/
def can_form_square (triangles : List Triangle) : Bool :=
  sorry

/-- Theorem stating that the figure can be cut into 5 triangles that form a square -/
theorem figure_can_form_square (f : Figure) :
  ∃ (triangles : List Triangle), 
    cut_into_triangles f = triangles ∧ 
    triangles.length = 5 ∧ 
    can_form_square triangles = true :=
  sorry

end figure_can_form_square_l1788_178878


namespace gcd_45045_30030_l1788_178815

theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 := by
  sorry

end gcd_45045_30030_l1788_178815


namespace triangle_problem_l1788_178828

def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_problem (a b c : ℝ) (h_triangle : triangle_ABC a b c) 
  (h_angle : Real.cos (π/3) = (b^2 + c^2 - a^2) / (2*b*c))
  (h_sides : a^2 - c^2 = (2/3) * b^2) :
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = Real.sqrt 3 / 5 ∧
  (1/2 * b * c * Real.sin (π/3) = 3 * Real.sqrt 3 / 4 → a = Real.sqrt 7) := by
  sorry

end triangle_problem_l1788_178828


namespace max_values_for_constrained_expressions_l1788_178845

theorem max_values_for_constrained_expressions (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1) :
  (∃ (max_ab : ℝ), ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 → x * y ≤ max_ab) ∧
  (∃ (max_sqrt : ℝ), ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 → Real.sqrt x + Real.sqrt y ≤ max_sqrt) ∧
  (∀ (M : ℝ), ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x^2 + y^2 > M) ∧
  (∀ (M : ℝ), ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1/x + 4/y > M) :=
by sorry

end max_values_for_constrained_expressions_l1788_178845


namespace exists_ten_segments_no_triangle_l1788_178810

/-- A sequence of 10 positive real numbers in geometric progression -/
def geometricSequence : Fin 10 → ℝ
  | ⟨n, _⟩ => 2^n

/-- Predicate to check if three numbers can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem stating that there exists a set of 10 segments where no three segments can form a triangle -/
theorem exists_ten_segments_no_triangle :
  ∃ (s : Fin 10 → ℝ), ∀ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(canFormTriangle (s i) (s j) (s k)) := by
  sorry

end exists_ten_segments_no_triangle_l1788_178810


namespace sample_in_range_l1788_178818

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) (n : ℕ) : ℕ :=
  start + (total / sampleSize) * n

/-- Theorem: The sample in the range [37, 54] is 42 -/
theorem sample_in_range (total : ℕ) (sampleSize : ℕ) (start : ℕ) :
  total = 900 →
  sampleSize = 50 →
  start = 6 →
  ∃ n : ℕ, 
    37 ≤ systematicSample total sampleSize start n ∧ 
    systematicSample total sampleSize start n ≤ 54 ∧
    systematicSample total sampleSize start n = 42 :=
by
  sorry


end sample_in_range_l1788_178818


namespace imaginary_number_product_l1788_178826

theorem imaginary_number_product (z : ℂ) (a : ℝ) : 
  (z.im ≠ 0 ∧ z.re = 0) → (Complex.I * z.im = z) → ((3 - Complex.I) * z = a + Complex.I) → a = 1/3 := by
  sorry

end imaginary_number_product_l1788_178826


namespace squares_below_line_l1788_178829

/-- Represents a line in the form ax + by = c --/
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Counts the number of integer points strictly below a line in the first quadrant --/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The line 8x + 245y = 1960 --/
def problemLine : Line := { a := 8, b := 245, c := 1960 }

theorem squares_below_line :
  countPointsBelowLine problemLine = 853 :=
sorry

end squares_below_line_l1788_178829


namespace stone_123_l1788_178855

/-- The number of stones in the sequence -/
def num_stones : ℕ := 15

/-- The number of counts in a complete cycle -/
def cycle_length : ℕ := 29

/-- The count we're interested in -/
def target_count : ℕ := 123

/-- The function that maps a count to its corresponding initial stone number -/
def count_to_stone (count : ℕ) : ℕ :=
  (count % cycle_length) % num_stones + 1

theorem stone_123 : count_to_stone target_count = 8 := by
  sorry

end stone_123_l1788_178855


namespace chemistry_marks_proof_l1788_178861

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  biology : ℕ
  chemistry : ℕ

/-- Calculates the average of marks --/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.biology + m.chemistry : ℚ) / 5

theorem chemistry_marks_proof (m : Marks) 
  (h1 : m.english = 73)
  (h2 : m.mathematics = 69)
  (h3 : m.physics = 92)
  (h4 : m.biology = 82)
  (h5 : average m = 76) :
  m.chemistry = 64 := by
sorry


end chemistry_marks_proof_l1788_178861


namespace image_of_square_l1788_178872

/-- Transformation from xy-plane to uv-plane -/
def transform (x y : ℝ) : ℝ × ℝ :=
  (x^3 - y^3, x^2 * y^2)

/-- Square OABC in xy-plane -/
def square_vertices : List (ℝ × ℝ) :=
  [(0, 0), (2, 0), (2, 2), (0, 2)]

/-- Theorem: Image of square OABC in uv-plane -/
theorem image_of_square :
  (square_vertices.map (λ (x, y) => transform x y)) =
  [(0, 0), (8, 0), (0, 16), (-8, 0)] := by
  sorry


end image_of_square_l1788_178872


namespace smarties_remainder_l1788_178824

theorem smarties_remainder (m : ℕ) (h : m % 11 = 5) : (2 * m) % 11 = 10 := by
  sorry

end smarties_remainder_l1788_178824


namespace theater_ticket_sales_l1788_178802

/-- Calculates the total money taken in on ticket sales given the prices and number of tickets sold. -/
def totalTicketSales (adultPrice childPrice : ℕ) (totalTickets adultTickets : ℕ) : ℕ :=
  adultPrice * adultTickets + childPrice * (totalTickets - adultTickets)

/-- Theorem stating that given the specific ticket prices and sales, the total money taken in is $206. -/
theorem theater_ticket_sales :
  totalTicketSales 8 5 34 12 = 206 := by
  sorry

end theater_ticket_sales_l1788_178802


namespace pencil_count_l1788_178876

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 75

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 112

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 448

/-- The number of pencils Bob bought -/
def bob_pencils : ℕ := cindi_pencils + 20

/-- The total number of pencils bought by Donna, Marcia, and Bob -/
def total_pencils : ℕ := donna_pencils + marcia_pencils + bob_pencils

theorem pencil_count : total_pencils = 655 := by
  sorry

end pencil_count_l1788_178876


namespace positive_roots_l1788_178868

theorem positive_roots (x y z : ℝ) 
  (sum_pos : x + y + z > 0) 
  (sum_prod_pos : x*y + y*z + z*x > 0) 
  (prod_pos : x*y*z > 0) : 
  x > 0 ∧ y > 0 ∧ z > 0 := by
sorry

end positive_roots_l1788_178868


namespace solve_system_l1788_178833

theorem solve_system (u v : ℚ) 
  (eq1 : 4 * u - 5 * v = 23)
  (eq2 : 2 * u + 4 * v = -8) :
  u + v = -1 := by
sorry

end solve_system_l1788_178833


namespace min_participants_l1788_178849

/-- Represents a single-round robin tournament --/
structure Tournament where
  participants : ℕ
  matches_per_player : ℕ
  winner_wins : ℕ

/-- Conditions for the tournament --/
def valid_tournament (t : Tournament) : Prop :=
  t.participants > 1 ∧
  t.matches_per_player = t.participants - 1 ∧
  (t.winner_wins : ℝ) / t.matches_per_player > 0.68 ∧
  (t.winner_wins : ℝ) / t.matches_per_player < 0.69

/-- The theorem to be proved --/
theorem min_participants (t : Tournament) (h : valid_tournament t) :
  t.participants ≥ 17 :=
sorry

end min_participants_l1788_178849


namespace power_of_two_special_case_l1788_178866

theorem power_of_two_special_case :
  let n : ℝ := 2^(0.15 : ℝ)
  let b : ℝ := 33.333333333333314
  n^b = 32 := by
  sorry

end power_of_two_special_case_l1788_178866


namespace rectangle_area_l1788_178807

/-- Given a rectangle where the length is four times the width and the perimeter is 250 cm,
    prove that its area is 2500 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 250 → l * w = 2500 := by sorry

end rectangle_area_l1788_178807


namespace p_sufficient_not_necessary_l1788_178896

-- Define propositions p and q
def p (x : ℝ) : Prop := 5 * x - 6 ≥ x^2
def q (x : ℝ) : Prop := |x + 1| > 2

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by sorry

end p_sufficient_not_necessary_l1788_178896


namespace nested_fraction_equality_l1788_178812

theorem nested_fraction_equality : 
  1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 11 / 7 := by
  sorry

end nested_fraction_equality_l1788_178812


namespace distribute_seven_balls_to_three_people_l1788_178895

/-- The number of ways to distribute n identical balls to k people, 
    with each person getting at least 1 ball -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 15 ways to distribute 7 identical balls to 3 people, 
    with each person getting at least 1 ball -/
theorem distribute_seven_balls_to_three_people : 
  distribute_balls 7 3 = 15 := by sorry

end distribute_seven_balls_to_three_people_l1788_178895


namespace intersection_M_complement_N_l1788_178822

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def N : Set ℝ := {x : ℝ | Real.exp (Real.log 2 * (1 - x)) > 1}

-- Define the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Icc (-2) 1 :=
sorry

end intersection_M_complement_N_l1788_178822


namespace expression_value_for_x_2_l1788_178898

theorem expression_value_for_x_2 : 
  let x : ℝ := 2
  (x + x * (x * x)) = 10 := by
sorry

end expression_value_for_x_2_l1788_178898


namespace algebraic_expression_value_l1788_178837

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : x * y = 1) : 
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 := by
  sorry

end algebraic_expression_value_l1788_178837


namespace rectangle_area_l1788_178835

/-- The area of a rectangle with perimeter 60 and length-to-width ratio 3:2 is 216 -/
theorem rectangle_area (l w : ℝ) : 
  (2 * l + 2 * w = 60) →  -- Perimeter condition
  (l = (3/2) * w) →       -- Length-to-width ratio condition
  (l * w = 216) :=        -- Area calculation
by sorry

end rectangle_area_l1788_178835


namespace miriam_initial_marbles_l1788_178819

/-- The number of marbles Miriam initially had -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Miriam gave to her brother -/
def marbles_to_brother : ℕ := 60

/-- The number of marbles Miriam gave to her sister -/
def marbles_to_sister : ℕ := 2 * marbles_to_brother

/-- The number of marbles Miriam currently has -/
def current_marbles : ℕ := 300

/-- The number of marbles Miriam gave to her friend Savanna -/
def marbles_to_savanna : ℕ := 3 * current_marbles

theorem miriam_initial_marbles :
  initial_marbles = marbles_to_brother + marbles_to_sister + marbles_to_savanna + current_marbles :=
by sorry

end miriam_initial_marbles_l1788_178819


namespace walking_time_is_half_time_saved_l1788_178816

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure HomeCommuteScenario where
  usual_arrival_time : ℕ  -- Time in minutes when they usually arrive home
  early_station_arrival : ℕ  -- Time in minutes the man arrives early at the station
  actual_arrival_time : ℕ  -- Time in minutes when they actually arrive home
  walking_time : ℕ  -- Time in minutes the man spends walking

/-- Theorem stating that the walking time is half of the time saved --/
theorem walking_time_is_half_time_saved (scenario : HomeCommuteScenario) 
  (h1 : scenario.early_station_arrival = 60)
  (h2 : scenario.usual_arrival_time - scenario.actual_arrival_time = 30) :
  scenario.walking_time = (scenario.usual_arrival_time - scenario.actual_arrival_time) / 2 := by
  sorry

#check walking_time_is_half_time_saved

end walking_time_is_half_time_saved_l1788_178816


namespace no_valid_pair_l1788_178804

def s : Finset ℤ := {2, 3, 4, 5, 9, 12, 18}
def b : Finset ℤ := {4, 5, 6, 7, 8, 11, 14, 19}

theorem no_valid_pair : ¬∃ (x y : ℤ), 
  x ∈ s ∧ y ∈ b ∧ 
  x % 3 = 2 ∧ y % 4 = 1 ∧ 
  (x % 2 = 0 ∧ y % 2 = 1 ∨ x % 2 = 1 ∧ y % 2 = 0) ∧ 
  x + y = 20 := by
sorry

end no_valid_pair_l1788_178804


namespace star_calculation_l1788_178836

/-- The star operation on rational numbers -/
def star (a b : ℚ) : ℚ := 2 * a - b + 1

/-- Theorem stating that 1 ☆ [3 ☆ (-2)] = -6 -/
theorem star_calculation : star 1 (star 3 (-2)) = -6 := by
  sorry

end star_calculation_l1788_178836


namespace minute_hand_angle_half_hour_l1788_178853

/-- The angle traversed by the minute hand of a clock in a given time period -/
def minute_hand_angle (time_period : ℚ) : ℚ :=
  360 * time_period

theorem minute_hand_angle_half_hour :
  minute_hand_angle (1/2) = 180 := by
  sorry

#check minute_hand_angle_half_hour

end minute_hand_angle_half_hour_l1788_178853


namespace square_root_meaningful_implies_x_geq_5_l1788_178800

theorem square_root_meaningful_implies_x_geq_5 (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
  sorry

end square_root_meaningful_implies_x_geq_5_l1788_178800


namespace sqrt_calculation_l1788_178827

theorem sqrt_calculation : (Real.sqrt 12 - Real.sqrt (1/3)) * Real.sqrt 6 = 5 * Real.sqrt 2 := by
  sorry

end sqrt_calculation_l1788_178827


namespace quadratic_equation_solution_l1788_178885

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -3 ∧ x₂ = 5 ∧ 
  (x₁^2 - 2*x₁ - 15 = 0) ∧ 
  (x₂^2 - 2*x₂ - 15 = 0) ∧
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 → x = x₁ ∨ x = x₂) :=
by
  sorry

end quadratic_equation_solution_l1788_178885


namespace coupon_savings_difference_l1788_178862

-- Define the coupon savings functions
def couponA (price : ℝ) : ℝ := 0.18 * price
def couponB : ℝ := 35
def couponC (price : ℝ) : ℝ := 0.20 * (price - 120)

-- Define the conditions for Coupon A to be at least as good as B and C
def couponABestCondition (price : ℝ) : Prop :=
  couponA price ≥ couponB ∧ couponA price ≥ couponC price

-- Define the range of prices where Coupon A is the best
def priceRange : Set ℝ := {price | price > 120 ∧ couponABestCondition price}

-- Theorem statement
theorem coupon_savings_difference :
  ∃ (x y : ℝ), x ∈ priceRange ∧ y ∈ priceRange ∧
  (∀ p ∈ priceRange, x ≤ p ∧ p ≤ y) ∧
  y - x = 1005.56 :=
sorry

end coupon_savings_difference_l1788_178862


namespace closest_fraction_l1788_178887

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]
def team_gamma_fraction : ℚ := 13/80

theorem closest_fraction :
  ∀ f ∈ fractions, |team_gamma_fraction - 1/6| ≤ |team_gamma_fraction - f| :=
by sorry

end closest_fraction_l1788_178887


namespace curve_is_ellipse_iff_l1788_178890

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k

/-- The condition for the curve to be a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -19

/-- Theorem stating that the curve is a non-degenerate ellipse iff k > -19 -/
theorem curve_is_ellipse_iff (x y k : ℝ) :
  (∀ x y, curve_equation x y k) ↔ is_non_degenerate_ellipse k :=
sorry

end curve_is_ellipse_iff_l1788_178890


namespace gcd_266_209_l1788_178806

theorem gcd_266_209 : Nat.gcd 266 209 = 19 := by
  sorry

end gcd_266_209_l1788_178806


namespace a_spending_percentage_l1788_178843

/-- Proves that A spends 95% of his salary given the conditions of the problem -/
theorem a_spending_percentage (total_salary : ℝ) (a_salary : ℝ) (b_spending_percentage : ℝ) :
  total_salary = 6000 →
  a_salary = 4500 →
  b_spending_percentage = 0.85 →
  let b_salary := total_salary - a_salary
  let a_savings := a_salary * (1 - (95 / 100))
  let b_savings := b_salary * (1 - b_spending_percentage)
  a_savings = b_savings :=
by sorry

end a_spending_percentage_l1788_178843


namespace village_population_l1788_178857

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 3294 → P = 4080 := by
sorry

end village_population_l1788_178857


namespace eleventh_term_value_l1788_178838

/-- An arithmetic progression with specified properties -/
structure ArithmeticProgression where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first 15 terms is 56.25
  sum_15_terms : (15 / 2 : ℝ) * (2 * a + 14 * d) = 56.25
  -- 7th term is 3.25
  term_7 : a + 6 * d = 3.25

/-- Theorem: The 11th term of the specified arithmetic progression is 5.25 -/
theorem eleventh_term_value (ap : ArithmeticProgression) : ap.a + 10 * ap.d = 5.25 := by
  sorry

end eleventh_term_value_l1788_178838


namespace problem_statement_l1788_178839

theorem problem_statement (P Q : ℝ) 
  (h1 : P^2 - P*Q = 1) 
  (h2 : 4*P*Q - 3*Q^2 = 2) : 
  P^2 + 3*P*Q - 3*Q^2 = 3 := by
  sorry

end problem_statement_l1788_178839


namespace triangle_area_l1788_178817

/-- Given a triangle with one side of length 14 units, the angle opposite to this side
    being 60 degrees, and the ratio of the other two sides being 8:5,
    prove that the area of the triangle is 40√3 square units. -/
theorem triangle_area (a b c : ℝ) (θ : ℝ) :
  a = 14 →
  θ = 60 * π / 180 →
  b / c = 8 / 5 →
  (1 / 2) * b * c * Real.sin θ = 40 * Real.sqrt 3 :=
by sorry

end triangle_area_l1788_178817


namespace sufficient_not_necessary_condition_l1788_178823

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a + b = 2 → a * b ≤ 1) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a * b ≤ 1 ∧ a + b ≠ 2) :=
by sorry

end sufficient_not_necessary_condition_l1788_178823


namespace point_in_fourth_quadrant_l1788_178808

theorem point_in_fourth_quadrant (a : ℝ) :
  let A : ℝ × ℝ := (Real.sqrt a + 1, -3)
  A.1 > 0 ∧ A.2 < 0 :=
by sorry

end point_in_fourth_quadrant_l1788_178808


namespace election_votes_theorem_l1788_178888

/-- Theorem: In an election with 3 candidates, where one candidate received 71.42857142857143% 
    of the total votes, and the other two candidates received 3000 and 5000 votes respectively, 
    the winning candidate received 20,000 votes. -/
theorem election_votes_theorem : 
  let total_votes : ℝ := (20000 + 3000 + 5000 : ℝ)
  let winning_percentage : ℝ := 71.42857142857143
  let other_votes_1 : ℝ := 3000
  let other_votes_2 : ℝ := 5000
  let winning_votes : ℝ := 20000
  (winning_votes / total_votes) * 100 = winning_percentage ∧
  winning_votes + other_votes_1 + other_votes_2 = total_votes :=
by
  sorry

#check election_votes_theorem

end election_votes_theorem_l1788_178888


namespace tangent_slope_at_negative_five_l1788_178865

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem tangent_slope_at_negative_five
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_diff : Differentiable ℝ f)
  (hf_deriv_one : deriv f 1 = 1)
  (hf_periodic : ∀ x, f (x + 2) = f (x - 2)) :
  deriv f (-5) = -1 :=
sorry

end tangent_slope_at_negative_five_l1788_178865


namespace binary_sum_equals_result_l1788_178820

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 11011₂ -/
def num1 : List Bool := [true, true, false, true, true]

/-- Represents the binary number 1010₂ -/
def num2 : List Bool := [true, false, true, false]

/-- Represents the binary number 11100₂ -/
def num3 : List Bool := [true, true, true, false, false]

/-- Represents the binary number 1001₂ -/
def num4 : List Bool := [true, false, false, true]

/-- Represents the binary number 100010₂ (the expected result) -/
def result : List Bool := [true, false, false, false, true, false]

/-- The main theorem stating that the sum of the binary numbers equals the expected result -/
theorem binary_sum_equals_result :
  binaryToNat num1 + binaryToNat num2 - binaryToNat num3 + binaryToNat num4 = binaryToNat result :=
by sorry

end binary_sum_equals_result_l1788_178820


namespace sandras_sock_purchase_l1788_178867

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions --/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 5 * p.five_dollar = 36 ∧
  p.two_dollar ≤ 6 ∧ p.three_dollar ≤ 6 ∧ p.five_dollar ≤ 6

/-- Theorem stating that the only valid purchase has 11 pairs of $2 socks --/
theorem sandras_sock_purchase :
  ∀ p : SockPurchase, is_valid_purchase p → p.two_dollar = 11 := by
  sorry

end sandras_sock_purchase_l1788_178867


namespace max_y_value_l1788_178894

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) :
  y ≤ 27 ∧ ∃ (x₀ : ℤ), x₀ * 27 + 7 * x₀ + 6 * 27 = -8 :=
sorry

end max_y_value_l1788_178894


namespace circle_division_evenness_l1788_178841

theorem circle_division_evenness (N : ℕ) : 
  (∃ (chords : Fin N → Fin (2 * N) × Fin (2 * N)),
    (∀ i : Fin N, (chords i).1 ≠ (chords i).2) ∧ 
    (∀ i j : Fin N, i ≠ j → (chords i).1 ≠ (chords j).1 ∧ (chords i).1 ≠ (chords j).2 ∧
                            (chords i).2 ≠ (chords j).1 ∧ (chords i).2 ≠ (chords j).2) ∧
    (∀ i : Fin N, ∃ k l : ℕ, 
      (((chords i).2 - (chords i).1 : ℤ) % (2 * N : ℤ) = 2 * k ∨
       ((chords i).1 - (chords i).2 : ℤ) % (2 * N : ℤ) = 2 * k) ∧
      (((chords i).2 - (chords i).1 : ℤ) % (2 * N : ℤ) = 2 * l ∨
       ((chords i).1 - (chords i).2 : ℤ) % (2 * N : ℤ) = 2 * l) ∧
      k + l = N)) →
  Even N :=
by sorry

end circle_division_evenness_l1788_178841


namespace limit_a_over_3n_l1788_178831

def S (n : ℕ) : ℝ := -3 * (n ^ 2 : ℝ) + 2 * n + 1

def a (n : ℕ) : ℝ := S (n + 1) - S n

theorem limit_a_over_3n :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n / (3 * (n + 1)) + 2| < ε :=
sorry

end limit_a_over_3n_l1788_178831


namespace five_wednesdays_theorem_l1788_178897

/-- The year of the Gregorian calendar reform -/
def gregorian_reform_year : ℕ := 1752

/-- The cycle length for years with 5 Wednesdays in February -/
def cycle_length : ℕ := 28

/-- The reference year with 5 Wednesdays in February -/
def reference_year : ℕ := 1928

/-- Predicate to check if a year has 5 Wednesdays in February -/
def has_five_wednesdays (year : ℕ) : Prop :=
  (year ≥ gregorian_reform_year) ∧ 
  (year = reference_year ∨ (year - reference_year) % cycle_length = 0)

/-- The nearest year before the reference year with 5 Wednesdays in February -/
def nearest_before : ℕ := 1888

/-- The nearest year after the reference year with 5 Wednesdays in February -/
def nearest_after : ℕ := 1956

theorem five_wednesdays_theorem :
  (has_five_wednesdays nearest_before) ∧
  (has_five_wednesdays nearest_after) ∧
  (∀ y : ℕ, nearest_before < y ∧ y < reference_year → ¬(has_five_wednesdays y)) ∧
  (∀ y : ℕ, reference_year < y ∧ y < nearest_after → ¬(has_five_wednesdays y)) :=
sorry

end five_wednesdays_theorem_l1788_178897


namespace intersection_of_logarithmic_functions_l1788_178874

open Real

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * log x = log (3 * x) := by sorry

end intersection_of_logarithmic_functions_l1788_178874


namespace one_third_coloring_ways_l1788_178803

/-- The number of ways to choose k items from a set of n items -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The number of triangles in the square -/
def total_triangles : ℕ := 18

/-- The number of triangles to be colored -/
def colored_triangles : ℕ := 6

/-- Theorem stating that the number of ways to color one-third of the square is 18564 -/
theorem one_third_coloring_ways :
  binomial total_triangles colored_triangles = 18564 := by
  sorry

end one_third_coloring_ways_l1788_178803


namespace eight_power_problem_l1788_178811

theorem eight_power_problem (x : ℝ) (h : (8 : ℝ) ^ (3 * x) = 64) : (8 : ℝ) ^ (-x) = 1/4 := by
  sorry

end eight_power_problem_l1788_178811


namespace holiday_approval_count_l1788_178869

theorem holiday_approval_count (total : ℕ) (oppose_percent : ℚ) (indifferent_percent : ℚ) 
  (h_total : total = 600)
  (h_oppose : oppose_percent = 6 / 100)
  (h_indifferent : indifferent_percent = 14 / 100) :
  ↑total * (1 - oppose_percent - indifferent_percent) = 480 :=
by sorry

end holiday_approval_count_l1788_178869


namespace product_of_differences_l1788_178880

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2023) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2022)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2023) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2022)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2023) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2022)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/2023 := by
  sorry

end product_of_differences_l1788_178880


namespace function_value_at_pi_sixth_l1788_178881

/-- Given a function f(x) = 3sin(ωx + φ) that satisfies f(π/3 + x) = f(-x) for any x,
    prove that f(π/6) = -3 or f(π/6) = 3 -/
theorem function_value_at_pi_sixth (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (-x)) →
  f (π / 6) = -3 ∨ f (π / 6) = 3 := by
  sorry

end function_value_at_pi_sixth_l1788_178881


namespace y_value_proof_l1788_178813

theorem y_value_proof : ∀ y : ℚ, (1/3 - 1/4 = 4/y) → y = 48 := by
  sorry

end y_value_proof_l1788_178813


namespace foci_of_hyperbola_l1788_178883

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 7 - y^2 / 3 = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(Real.sqrt 10, 0), (-Real.sqrt 10, 0)}

/-- Theorem: The given coordinates are the foci of the hyperbola -/
theorem foci_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates ↔ 
    ∃ (x' y' : ℝ), hyperbola_equation x' y' ∧ 
      (x - x')^2 + (y - y')^2 = ((Real.sqrt 10 + x')^2 + y'^2).sqrt * 
                                ((Real.sqrt 10 - x')^2 + y'^2).sqrt :=
sorry

end foci_of_hyperbola_l1788_178883


namespace smallest_natural_power_l1788_178889

theorem smallest_natural_power (n : ℕ) : n^(Nat.zero) = 1 := by
  sorry

#check smallest_natural_power 2009

end smallest_natural_power_l1788_178889


namespace internal_diagonal_cubes_l1788_178834

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def num_cubes_passed (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating that the number of unit cubes an internal diagonal passes through
    in a 200 × 300 × 450 rectangular solid is 700 -/
theorem internal_diagonal_cubes :
  num_cubes_passed 200 300 450 = 700 := by sorry

end internal_diagonal_cubes_l1788_178834


namespace franks_breakfast_shopping_l1788_178832

/-- Frank's breakfast shopping problem -/
theorem franks_breakfast_shopping
  (num_buns : ℕ)
  (num_milk_bottles : ℕ)
  (milk_price : ℚ)
  (egg_price_multiplier : ℕ)
  (total_paid : ℚ)
  (h_num_buns : num_buns = 10)
  (h_num_milk_bottles : num_milk_bottles = 2)
  (h_milk_price : milk_price = 2)
  (h_egg_price : egg_price_multiplier = 3)
  (h_total_paid : total_paid = 11)
  : (total_paid - (num_milk_bottles * milk_price + egg_price_multiplier * milk_price)) / num_buns = 0.1 := by
  sorry

end franks_breakfast_shopping_l1788_178832


namespace morning_campers_count_l1788_178873

/-- Given a total number of campers and a ratio for morning:afternoon:evening,
    calculate the number of campers who went rowing in the morning. -/
def campers_in_morning (total : ℕ) (morning_ratio afternoon_ratio evening_ratio : ℕ) : ℕ :=
  let total_ratio := morning_ratio + afternoon_ratio + evening_ratio
  let part_size := total / total_ratio
  morning_ratio * part_size

/-- Theorem stating that given 60 total campers and a ratio of 3:2:4,
    the number of campers who went rowing in the morning is 18. -/
theorem morning_campers_count :
  campers_in_morning 60 3 2 4 = 18 := by
  sorry

#eval campers_in_morning 60 3 2 4

end morning_campers_count_l1788_178873


namespace correlation_coefficient_relationship_l1788_178821

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x y : List ℝ) : ℝ := sorry

def r₁ : ℝ := linear_correlation_coefficient X Y
def r₂ : ℝ := linear_correlation_coefficient U V

theorem correlation_coefficient_relationship : r₂ < 0 ∧ 0 < r₁ := by sorry

end correlation_coefficient_relationship_l1788_178821


namespace least_addition_for_divisibility_l1788_178830

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), (1021 + x) % 25 = 0 ∧ 
  ∀ (y : ℕ), (1021 + y) % 25 = 0 → x ≤ y :=
by
  -- The proof would go here
  sorry

end least_addition_for_divisibility_l1788_178830


namespace cone_base_radius_l1788_178854

/-- The base radius of a cone with height 4 and volume 4π is √3 -/
theorem cone_base_radius (h : ℝ) (V : ℝ) (r : ℝ) :
  h = 4 → V = 4 * Real.pi → V = (1/3) * Real.pi * r^2 * h → r = Real.sqrt 3 := by
  sorry

end cone_base_radius_l1788_178854


namespace intersection_of_lines_l1788_178801

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that the intersection point of lines AB and CD is (-4/3, 35, 3/2) --/
theorem intersection_of_lines :
  let A : ℝ × ℝ × ℝ := (3, -5, 4)
  let B : ℝ × ℝ × ℝ := (13, -15, 9)
  let C : ℝ × ℝ × ℝ := (-6, 6, -12)
  let D : ℝ × ℝ × ℝ := (-4, -2, 8)
  intersection_point A B C D = (-4/3, 35, 3/2) := by sorry

end intersection_of_lines_l1788_178801


namespace polynomial_interpolation_l1788_178899

def p (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 2*x + 2

theorem polynomial_interpolation :
  (p (-2) = 2) ∧
  (p (-1) = 1) ∧
  (p 0 = 2) ∧
  (p 1 = -1) ∧
  (p 2 = 10) ∧
  (∀ q : ℝ → ℝ, (q (-2) = 2) ∧ (q (-1) = 1) ∧ (q 0 = 2) ∧ (q 1 = -1) ∧ (q 2 = 10) →
    (∃ a b c d e : ℝ, ∀ x, q x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
    (∀ x, q x = p x)) :=
sorry

end polynomial_interpolation_l1788_178899


namespace decimal_to_percentage_l1788_178848

theorem decimal_to_percentage (x : ℝ) : x = 5.02 → (x * 100 : ℝ) = 502 := by
  sorry

end decimal_to_percentage_l1788_178848


namespace maximum_marks_calculation_maximum_marks_is_750_l1788_178871

theorem maximum_marks_calculation (passing_percentage : ℝ) (student_score : ℕ) (shortfall : ℕ) : ℝ :=
  let passing_score : ℕ := student_score + shortfall
  let maximum_marks : ℝ := passing_score / passing_percentage
  maximum_marks

-- Proof that the maximum marks is 750 given the conditions
theorem maximum_marks_is_750 :
  maximum_marks_calculation 0.3 212 13 = 750 :=
by sorry

end maximum_marks_calculation_maximum_marks_is_750_l1788_178871


namespace extreme_value_condition_decreasing_function_condition_l1788_178842

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 - b*x

-- Theorem for part (1)
theorem extreme_value_condition (a b : ℝ) :
  (∃ x : ℝ, f a b x = 2 ∧ ∀ y : ℝ, f a b y ≤ f a b x) ∧ f a b 1 = 2 →
  a = 1 ∧ b = 3 :=
sorry

-- Theorem for part (2)
theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 2 → f a (9*a) x > f a (9*a) y) →
  a ≥ 1 :=
sorry

end extreme_value_condition_decreasing_function_condition_l1788_178842


namespace jennifers_spending_l1788_178863

theorem jennifers_spending (initial_amount : ℚ) : 
  initial_amount / 5 + initial_amount / 6 + initial_amount / 2 + 20 = initial_amount →
  initial_amount = 150 := by
  sorry

end jennifers_spending_l1788_178863


namespace max_pyramid_volume_is_four_l1788_178825

/-- A triangular prism with given ratios on its lateral edges -/
structure TriangularPrism where
  volume : ℝ
  AM_ratio : ℝ
  BN_ratio : ℝ
  CK_ratio : ℝ

/-- The maximum volume of a pyramid formed inside the prism -/
def max_pyramid_volume (prism : TriangularPrism) : ℝ := sorry

/-- Theorem stating the maximum volume of the pyramid MNKP -/
theorem max_pyramid_volume_is_four (prism : TriangularPrism) 
  (h1 : prism.volume = 16)
  (h2 : prism.AM_ratio = 1/2)
  (h3 : prism.BN_ratio = 1/3)
  (h4 : prism.CK_ratio = 1/4) :
  max_pyramid_volume prism = 4 := by sorry

end max_pyramid_volume_is_four_l1788_178825


namespace square_area_proof_l1788_178884

theorem square_area_proof (x : ℝ) : 
  (5 * x - 21 = 36 - 4 * x) → 
  (5 * x - 21)^2 = 113.4225 := by
  sorry

end square_area_proof_l1788_178884


namespace min_sum_given_product_min_sum_value_l1788_178891

theorem min_sum_given_product (x y : ℝ) (h1 : x * y = 4) (h2 : x > 0) (h3 : y > 0) :
  ∀ a b : ℝ, a * b = 4 ∧ a > 0 ∧ b > 0 → x + y ≤ a + b :=
by
  sorry

theorem min_sum_value (x y : ℝ) (h1 : x * y = 4) (h2 : x > 0) (h3 : y > 0) :
  ∃ M : ℝ, M = 4 ∧ x + y ≥ M :=
by
  sorry

end min_sum_given_product_min_sum_value_l1788_178891


namespace ellipse_foci_distance_l1788_178850

/-- The distance between the foci of an ellipse defined by 25x^2 - 100x + 4y^2 + 8y + 36 = 0 -/
theorem ellipse_foci_distance : 
  let ellipse_eq := fun (x y : ℝ) => 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36
  ∃ (h k a b : ℝ), 
    (∀ x y, ellipse_eq x y = 0 ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧
    2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 14.28 :=
by sorry

end ellipse_foci_distance_l1788_178850


namespace frog_hops_l1788_178846

theorem frog_hops (frog1 frog2 frog3 : ℕ) : 
  frog1 = 4 * frog2 →
  frog2 = 2 * frog3 →
  frog2 = 18 →
  frog1 + frog2 + frog3 = 99 := by
sorry

end frog_hops_l1788_178846


namespace m_range_l1788_178840

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 > 0

-- Define the set of x values satisfying p
def A : Set ℝ := {x | p x}

-- Define the set of x values satisfying q
def B (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem m_range :
  (∀ m : ℝ, m > 0 → (A ⊂ B m) ∧ (A ≠ B m)) →
  {m : ℝ | 0 < m ∧ m ≤ 2} = {m : ℝ | ∃ x, q x m ∧ ¬p x} :=
sorry

end m_range_l1788_178840


namespace counterexample_fifth_power_l1788_178858

theorem counterexample_fifth_power : 144^5 + 121^5 + 95^5 + 30^5 = 159^5 := by
  sorry

end counterexample_fifth_power_l1788_178858


namespace prob_two_co_captains_all_teams_l1788_178892

/-- Represents a math team with a given number of students and co-captains -/
structure MathTeam where
  students : Nat
  coCaptains : Nat
  h : coCaptains ≤ students

/-- Calculates the probability of choosing two co-captains from a given team -/
def probTwoCoCaptains (team : MathTeam) : Rat :=
  (Nat.choose team.coCaptains 2 : Rat) / (Nat.choose team.students 2 : Rat)

/-- The list of math teams in the area -/
def mathTeams : List MathTeam := [
  ⟨6, 3, by norm_num⟩,
  ⟨9, 2, by norm_num⟩,
  ⟨10, 4, by norm_num⟩
]

theorem prob_two_co_captains_all_teams : 
  (List.sum (mathTeams.map probTwoCoCaptains) / (mathTeams.length : Rat)) = 65 / 540 := by
  sorry

end prob_two_co_captains_all_teams_l1788_178892


namespace victor_finished_last_l1788_178814

-- Define the set of runners
inductive Runner : Type
| Lotar : Runner
| Manfred : Runner
| Victor : Runner
| Jan : Runner
| Eddy : Runner

-- Define the relation "finished before"
def finished_before : Runner → Runner → Prop := sorry

-- State the conditions
axiom lotar_before_manfred : finished_before Runner.Lotar Runner.Manfred
axiom victor_after_jan : finished_before Runner.Jan Runner.Victor
axiom manfred_before_jan : finished_before Runner.Manfred Runner.Jan
axiom eddy_before_victor : finished_before Runner.Eddy Runner.Victor

-- Define what it means to finish last
def finished_last (r : Runner) : Prop :=
  ∀ other : Runner, other ≠ r → finished_before other r

-- State the theorem
theorem victor_finished_last :
  finished_last Runner.Victor :=
sorry

end victor_finished_last_l1788_178814


namespace nonagon_diagonals_l1788_178844

/-- The number of diagonals in a regular polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : numDiagonals 9 = 27 := by sorry

end nonagon_diagonals_l1788_178844


namespace ines_shopping_result_l1788_178860

/-- Represents the shopping scenario for Ines at the farmers' market -/
def shopping_scenario (initial_amount : ℚ) (peach_price peach_qty cherry_price cherry_qty
                       baguette_price baguette_qty strawberry_price strawberry_qty
                       salad_price salad_qty : ℚ) : ℚ :=
  let total_cost := peach_price * peach_qty + cherry_price * cherry_qty +
                    baguette_price * baguette_qty + strawberry_price * strawberry_qty +
                    salad_price * salad_qty
  let discount_rate := if total_cost > 10 then 0.1 else 0 +
                       if peach_qty > 0 && cherry_qty > 0 && baguette_qty > 0 &&
                          strawberry_qty > 0 && salad_qty > 0
                       then 0.05 else 0
  let discounted_total := total_cost * (1 - discount_rate)
  let with_tax := discounted_total * 1.05
  let final_total := with_tax * 1.02
  initial_amount - final_total

/-- Theorem stating that Ines will be short by $4.58 after her shopping trip -/
theorem ines_shopping_result :
  shopping_scenario 20 2 3 3.5 2 1.25 4 4 1 2.5 2 = -4.58 := by
  sorry

end ines_shopping_result_l1788_178860


namespace tan_theta_value_l1788_178859

theorem tan_theta_value (θ : Real) 
  (h : (1 + Real.sin (2 * θ)) / (Real.cos θ ^ 2 - Real.sin θ ^ 2) = -3) : 
  Real.tan θ = 2 := by
  sorry

end tan_theta_value_l1788_178859


namespace repeating_decimal_difference_l1788_178886

theorem repeating_decimal_difference : 
  let x : ℚ := 72 / 99  -- $0.\overline{72}$ as a fraction
  let y : ℚ := 72 / 100 -- $0.72$ as a fraction
  x - y = 2 / 275 := by
sorry

end repeating_decimal_difference_l1788_178886
