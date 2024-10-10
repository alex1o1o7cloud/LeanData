import Mathlib

namespace solution_set_of_inequality_l756_75665

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (2 - x) > 0 ↔ x ∈ Set.Ioo 1 2 :=
by sorry

end solution_set_of_inequality_l756_75665


namespace binary_110010_equals_50_l756_75696

-- Define the binary number as a list of digits
def binary_number : List Nat := [1, 1, 0, 0, 1, 0]

-- Function to convert binary to decimal
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem statement
theorem binary_110010_equals_50 :
  binary_to_decimal binary_number = 50 := by
  sorry

end binary_110010_equals_50_l756_75696


namespace smaller_square_area_l756_75664

/-- The area of the smaller square formed by inscribing two right triangles in a larger square --/
theorem smaller_square_area (s : ℝ) (h : s = 4) : 
  let diagonal_smaller := s
  let side_smaller := diagonal_smaller / Real.sqrt 2
  side_smaller ^ 2 = 8 := by
  sorry

end smaller_square_area_l756_75664


namespace work_completion_days_l756_75659

-- Define the daily work done by a man and a boy
variable (M B : ℝ)

-- Define the total work to be done
variable (W : ℝ)

-- Define the number of days for the first group
variable (D : ℝ)

-- Theorem statement
theorem work_completion_days 
  (h1 : M = 2 * B) -- A man's daily work is twice that of a boy
  (h2 : (13 * M + 24 * B) * 4 = W) -- 13 men and 24 boys complete the work in 4 days
  (h3 : (12 * M + 16 * B) * D = W) -- 12 men and 16 boys complete the work in D days
  : D = 5 := by
  sorry

end work_completion_days_l756_75659


namespace particle_speed_is_sqrt_34_l756_75674

/-- A particle moves along a path. Its position at time t is (3t + 1, 5t - 2). -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 1, 5 * t - 2)

/-- The speed of the particle is defined as the distance traveled per unit time. -/
def particle_speed : ℝ := sorry

/-- Theorem: The speed of the particle is √34 units of distance per unit of time. -/
theorem particle_speed_is_sqrt_34 : particle_speed = Real.sqrt 34 := by sorry

end particle_speed_is_sqrt_34_l756_75674


namespace pool_swimmers_l756_75612

theorem pool_swimmers (total : ℕ) (first_day : ℕ) (second_day_diff : ℕ) :
  total = 246 →
  first_day = 79 →
  second_day_diff = 47 →
  ∃ (third_day : ℕ), 
    total = first_day + (third_day + second_day_diff) + third_day ∧
    third_day = 60 :=
by sorry

end pool_swimmers_l756_75612


namespace point_on_x_axis_l756_75611

/-- Given a point P with coordinates (m+3, m+1) on the x-axis,
    prove that its coordinates are (2, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 3, m + 1)
  P.2 = 0 → P = (2, 0) := by
sorry

end point_on_x_axis_l756_75611


namespace sufficient_not_necessary_l756_75608

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end sufficient_not_necessary_l756_75608


namespace child_ticket_cost_l756_75641

theorem child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_cost : ℚ) (extra_cost : ℚ) :
  num_adults = 9 →
  num_children = 7 →
  adult_ticket_cost = 11 →
  extra_cost = 50 →
  ∃ (child_ticket_cost : ℚ),
    num_adults * adult_ticket_cost = num_children * child_ticket_cost + extra_cost ∧
    child_ticket_cost = 7 :=
by sorry

end child_ticket_cost_l756_75641


namespace triathlon_bicycle_speed_specific_triathlon_problem_l756_75600

/-- Triathlon problem -/
theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ) 
  (run_distance run_speed : ℝ) 
  (bike_distance : ℝ) : ℝ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

/-- The specific triathlon problem -/
theorem specific_triathlon_problem : 
  triathlon_bicycle_speed 1.75 (1/3) 1.5 2.5 8 12 = 1728/175 := by
  sorry

end triathlon_bicycle_speed_specific_triathlon_problem_l756_75600


namespace remainder_proof_l756_75616

theorem remainder_proof :
  let n : ℕ := 174
  let d₁ : ℕ := 34
  let d₂ : ℕ := 5
  (n % d₁ = 4) ∧ (n % d₂ = 4) :=
by sorry

end remainder_proof_l756_75616


namespace decagon_ratio_l756_75645

/-- Represents a decagon with the properties described in the problem -/
structure Decagon :=
  (area : ℝ)
  (bisector_line : Set ℝ × Set ℝ)
  (below_area : ℝ)
  (triangle_base : ℝ)
  (xq : ℝ)
  (qy : ℝ)

/-- The theorem corresponding to the problem -/
theorem decagon_ratio (d : Decagon) : 
  d.area = 15 ∧ 
  d.below_area = 7.5 ∧ 
  d.triangle_base = 7 ∧ 
  d.xq + d.qy = 7 →
  d.xq / d.qy = 2 / 5 :=
by sorry

end decagon_ratio_l756_75645


namespace circle_center_on_line_ab_range_l756_75690

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop :=
  a*x - b*y + 1 = 0

-- Define the center of the circle
def center (x y : ℝ) : Prop :=
  x = -1 ∧ y = 2

-- Theorem statement
theorem circle_center_on_line_ab_range :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), circle_eq x y ∧ center x y ∧ line_eq a b x y) →
  0 < a * b ∧ a * b ≤ 2 - Real.sqrt 3 :=
sorry

end circle_center_on_line_ab_range_l756_75690


namespace smallest_n_divisible_by_four_l756_75628

theorem smallest_n_divisible_by_four :
  ∃ (n : ℕ), (7 * (n - 3)^5 - n^2 + 16*n - 30) % 4 = 0 ∧
  ∀ (m : ℕ), m < n → (7 * (m - 3)^5 - m^2 + 16*m - 30) % 4 ≠ 0 ∧
  n = 1 := by
  sorry

end smallest_n_divisible_by_four_l756_75628


namespace writer_book_frequency_l756_75618

theorem writer_book_frequency
  (years_writing : ℕ)
  (avg_earnings_per_book : ℝ)
  (total_earnings : ℝ)
  (h1 : years_writing = 20)
  (h2 : avg_earnings_per_book = 30000)
  (h3 : total_earnings = 3600000) :
  (years_writing * 12 : ℝ) / (total_earnings / avg_earnings_per_book) = 2 := by
  sorry

end writer_book_frequency_l756_75618


namespace sin_symmetric_angles_l756_75604

def symmetric_angles (α β : Real) : Prop :=
  ∃ k : Int, α + β = Real.pi + 2 * k * Real.pi

theorem sin_symmetric_angles (α β : Real) 
  (h_symmetric : symmetric_angles α β) (h_sin_α : Real.sin α = 1/3) : 
  Real.sin β = 1/3 := by
  sorry

end sin_symmetric_angles_l756_75604


namespace jessie_weight_loss_l756_75668

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (weight_lost : ℝ) (current_weight : ℝ) (loss_per_day : ℝ) :
  weight_lost = 126 →
  current_weight = 66 →
  loss_per_day = 0.5 →
  ∃ (initial_weight : ℝ) (days : ℝ),
    initial_weight = current_weight + weight_lost ∧
    initial_weight = 192 ∧
    days * loss_per_day = weight_lost :=
by sorry

end jessie_weight_loss_l756_75668


namespace principal_calculation_l756_75671

/-- Prove that the principal is 9200 given the specified conditions -/
theorem principal_calculation (r t : ℝ) (h1 : r = 0.12) (h2 : t = 3) : 
  ∃ P : ℝ, P - (P * r * t) = P - 5888 ∧ P = 9200 := by
  sorry

end principal_calculation_l756_75671


namespace fibonacci_harmonic_sum_l756_75643

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the harmonic series
def H : ℕ → ℚ
  | 0 => 0
  | (n + 1) => H n + 1 / (n + 1)

-- State the theorem
theorem fibonacci_harmonic_sum :
  (∑' n : ℕ, (fib (n + 1) : ℚ) / ((n + 2) * H (n + 1) * H (n + 2))) = 1 := by
  sorry

end fibonacci_harmonic_sum_l756_75643


namespace difference_between_numbers_l756_75619

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 84) (h2 : a = 36) (h3 : b = 48) :
  b - a = 12 := by
sorry

end difference_between_numbers_l756_75619


namespace quadratic_inequality_condition_l756_75644

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^2 + 2 * a * x < 1 - 3 * a) ↔ a < 1/6 := by
  sorry

end quadratic_inequality_condition_l756_75644


namespace harolds_remaining_money_l756_75683

/-- Represents Harold's financial situation and calculates his remaining money --/
def harolds_finances (income rent car_payment groceries : ℚ) : ℚ := 
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement_savings := remaining / 2
  remaining - retirement_savings

/-- Theorem stating that Harold will have $650.00 left after expenses and retirement savings --/
theorem harolds_remaining_money :
  harolds_finances 2500 700 300 50 = 650 := by
  sorry

end harolds_remaining_money_l756_75683


namespace difference_of_cubes_divisible_by_eight_l756_75661

theorem difference_of_cubes_divisible_by_eight (a b : ℤ) : 
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 = 8*k := by
  sorry

end difference_of_cubes_divisible_by_eight_l756_75661


namespace expression_simplification_l756_75631

theorem expression_simplification (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 6) :
  (a - b)^2 + b*(3*a - b) - a^2 = 2 * Real.sqrt 3 := by
  sorry

end expression_simplification_l756_75631


namespace books_loaned_out_l756_75677

/-- Proves that the number of books loaned out is 50, given the initial number of books,
    the return rate, and the final number of books. -/
theorem books_loaned_out
  (initial_books : ℕ)
  (return_rate : ℚ)
  (final_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_books = 65) :
  ∃ (loaned_books : ℕ), loaned_books = 50 ∧
    final_books = initial_books - (1 - return_rate) * loaned_books :=
by sorry

end books_loaned_out_l756_75677


namespace floor_neg_three_point_seven_l756_75622

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem floor_neg_three_point_seven :
  floor (-3.7) = -4 := by sorry

end floor_neg_three_point_seven_l756_75622


namespace flower_expenses_l756_75639

/-- The total expenses for ordering flowers at Parc Municipal -/
theorem flower_expenses : 
  let tulips : ℕ := 250
  let carnations : ℕ := 375
  let roses : ℕ := 320
  let price_per_flower : ℕ := 2
  (tulips + carnations + roses) * price_per_flower = 1890 := by
  sorry

end flower_expenses_l756_75639


namespace three_tenths_plus_four_thousandths_l756_75663

theorem three_tenths_plus_four_thousandths : 
  (3 : ℚ) / 10 + (4 : ℚ) / 1000 = (304 : ℚ) / 1000 := by sorry

end three_tenths_plus_four_thousandths_l756_75663


namespace track_width_l756_75617

theorem track_width (r₁ r₂ : ℝ) : 
  (2 * π * r₁ = 100 * π) →
  (2 * π * r₁ - 2 * π * r₂ = 16 * π) →
  (r₁ - r₂ = 8) := by
sorry

end track_width_l756_75617


namespace factorization_of_4x_squared_minus_1_l756_75666

theorem factorization_of_4x_squared_minus_1 (x : ℝ) : 4 * x^2 - 1 = (2*x + 1) * (2*x - 1) := by
  sorry

end factorization_of_4x_squared_minus_1_l756_75666


namespace cos_180_degrees_l756_75697

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l756_75697


namespace marble_distribution_l756_75651

theorem marble_distribution (total : ℕ) (first second third : ℚ) : 
  total = 78 →
  first = 3 * second + 2 →
  second = third / 2 →
  first + second + third = total →
  (first = 40 ∧ second = 38/3 ∧ third = 76/3) := by
  sorry

end marble_distribution_l756_75651


namespace modulus_complex_power_eight_l756_75698

theorem modulus_complex_power_eight :
  Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end modulus_complex_power_eight_l756_75698


namespace nested_fraction_simplification_l756_75607

theorem nested_fraction_simplification : 
  1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 11 / 7 := by
  sorry

end nested_fraction_simplification_l756_75607


namespace marks_father_gave_85_l756_75672

/-- The amount of money Mark's father gave him. -/
def fathers_money (num_books : ℕ) (book_price : ℕ) (money_left : ℕ) : ℕ :=
  num_books * book_price + money_left

/-- Theorem stating that Mark's father gave him $85. -/
theorem marks_father_gave_85 :
  fathers_money 10 5 35 = 85 := by
  sorry

end marks_father_gave_85_l756_75672


namespace right_shift_two_units_l756_75602

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Transformation that moves a function horizontally -/
def horizontalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b - f.m * shift }

theorem right_shift_two_units (f : LinearFunction) :
  f.m = 2 ∧ f.b = 1 →
  (horizontalShift f 2).m = 2 ∧ (horizontalShift f 2).b = -3 := by
  sorry

end right_shift_two_units_l756_75602


namespace total_octopus_legs_l756_75624

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The number of octopuses Sawyer saw -/
def octopuses_seen : ℕ := 5

/-- Theorem: The total number of octopus legs Sawyer saw is 40 -/
theorem total_octopus_legs : octopuses_seen * legs_per_octopus = 40 := by
  sorry

end total_octopus_legs_l756_75624


namespace arithmetic_sequence_length_l756_75694

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℕ),
    a₁ = 1 →
    aₙ = 46 →
    d = 3 →
    aₙ = a₁ + (n - 1) * d →
    n = 16 :=
by
  sorry

end arithmetic_sequence_length_l756_75694


namespace inequality_system_solution_range_l756_75601

/-- Define the @ operation for real numbers -/
def at_op (p q : ℝ) : ℝ := p + q - p * q

/-- The theorem statement -/
theorem inequality_system_solution_range (m : ℝ) :
  (∃! (a b : ℤ), (a ≠ b) ∧ 
    (at_op 2 (a : ℝ) > 0) ∧ (at_op (a : ℝ) 3 ≤ m) ∧
    (at_op 2 (b : ℝ) > 0) ∧ (at_op (b : ℝ) 3 ≤ m) ∧
    (∀ x : ℤ, x ≠ a ∧ x ≠ b → 
      ¬((at_op 2 (x : ℝ) > 0) ∧ (at_op (x : ℝ) 3 ≤ m))))
  → 3 ≤ m ∧ m < 5 :=
by sorry

end inequality_system_solution_range_l756_75601


namespace fraction_value_l756_75655

theorem fraction_value (y : ℝ) (h : 4 - 9/y + 9/(y^2) = 0) : 3/y = 2 := by
  sorry

end fraction_value_l756_75655


namespace smallest_q_in_geometric_sequence_l756_75689

def is_geometric_sequence (p q r : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ q = p * k ∧ r = q * k

theorem smallest_q_in_geometric_sequence (p q r : ℝ) :
  p > 0 → q > 0 → r > 0 →
  is_geometric_sequence p q r →
  p * q * r = 216 →
  q ≥ 6 ∧ ∃ p' q' r' : ℝ, p' > 0 ∧ q' > 0 ∧ r' > 0 ∧
    is_geometric_sequence p' q' r' ∧ p' * q' * r' = 216 ∧ q' = 6 :=
by sorry

end smallest_q_in_geometric_sequence_l756_75689


namespace complex_square_plus_four_l756_75676

theorem complex_square_plus_four : 
  let i : ℂ := Complex.I
  (2 - 3*i)^2 + 4 = -1 - 12*i :=
by sorry

end complex_square_plus_four_l756_75676


namespace expected_sixes_is_one_third_l756_75633

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 
  0 * (prob_not_six ^ 2) + 
  1 * (2 * prob_six * prob_not_six) + 
  2 * (prob_six ^ 2)

theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 := by sorry

end expected_sixes_is_one_third_l756_75633


namespace cucumber_price_is_four_l756_75615

/-- Represents the price of cucumbers per kilo -/
def cucumber_price : ℝ := sorry

/-- Theorem: Given Peter's shopping details, the price of cucumbers per kilo is $4 -/
theorem cucumber_price_is_four :
  let initial_amount : ℝ := 500
  let potatoes_kilo : ℝ := 6
  let potatoes_price : ℝ := 2
  let tomatoes_kilo : ℝ := 9
  let tomatoes_price : ℝ := 3
  let cucumbers_kilo : ℝ := 5
  let bananas_kilo : ℝ := 3
  let bananas_price : ℝ := 5
  let remaining_amount : ℝ := 426
  initial_amount - 
    (potatoes_kilo * potatoes_price + 
     tomatoes_kilo * tomatoes_price + 
     cucumbers_kilo * cucumber_price + 
     bananas_kilo * bananas_price) = remaining_amount →
  cucumber_price = 4 := by sorry

end cucumber_price_is_four_l756_75615


namespace total_distance_walked_l756_75640

/-- Given a constant walking pace and duration, calculate the total distance walked. -/
theorem total_distance_walked (pace : ℝ) (duration : ℝ) (total_distance : ℝ) : 
  pace = 2 → duration = 8 → total_distance = pace * duration → total_distance = 16 := by
  sorry

end total_distance_walked_l756_75640


namespace expected_rainfall_theorem_l756_75682

/-- The number of days considered in the weather forecast -/
def days : ℕ := 5

/-- The probability of no rain on a given day -/
def prob_no_rain : ℝ := 0.3

/-- The probability of 3 inches of rain on a given day -/
def prob_3_inches : ℝ := 0.5

/-- The probability of 8 inches of rain on a given day -/
def prob_8_inches : ℝ := 0.2

/-- The amount of rainfall (in inches) for the "no rain" scenario -/
def rain_0 : ℝ := 0

/-- The amount of rainfall (in inches) for the "3 inches" scenario -/
def rain_3 : ℝ := 3

/-- The amount of rainfall (in inches) for the "8 inches" scenario -/
def rain_8 : ℝ := 8

/-- The expected total rainfall over the given number of days -/
def expected_total_rainfall : ℝ := days * (prob_no_rain * rain_0 + prob_3_inches * rain_3 + prob_8_inches * rain_8)

theorem expected_rainfall_theorem : expected_total_rainfall = 15.5 := by
  sorry

end expected_rainfall_theorem_l756_75682


namespace simplify_power_product_l756_75686

theorem simplify_power_product (x : ℝ) : (x^5 * x^3)^2 = x^16 := by
  sorry

end simplify_power_product_l756_75686


namespace simplify_expression_l756_75670

theorem simplify_expression (x y : ℝ) : (35*x - 24*y) + (15*x + 40*y) - (25*x - 49*y) = 25*x + 65*y := by
  sorry

end simplify_expression_l756_75670


namespace initial_wall_count_l756_75681

theorem initial_wall_count (total_containers ceiling_containers leftover_containers tiled_walls : ℕ) 
  (h1 : total_containers = 16)
  (h2 : ceiling_containers = 1)
  (h3 : leftover_containers = 3)
  (h4 : tiled_walls = 1)
  (h5 : ∀ w1 w2 : ℕ, w1 ≠ 0 → w2 ≠ 0 → (total_containers - ceiling_containers - leftover_containers) / w1 = 
                     (total_containers - ceiling_containers - leftover_containers) / w2 → w1 = w2) :
  total_containers - ceiling_containers - leftover_containers + tiled_walls = 13 := by
  sorry

end initial_wall_count_l756_75681


namespace x_plus_2y_equals_10_l756_75610

theorem x_plus_2y_equals_10 (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : x + 2*y = 10 := by
  sorry

end x_plus_2y_equals_10_l756_75610


namespace cube_root_nested_expression_l756_75634

theorem cube_root_nested_expression (x : ℝ) (h : x ≥ 0) :
  (x * (x * x^(1/3))^(1/2))^(1/3) = x^(5/9) := by
  sorry

end cube_root_nested_expression_l756_75634


namespace tan_product_eighths_pi_l756_75647

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = -Real.sqrt 2 := by
  sorry

end tan_product_eighths_pi_l756_75647


namespace sum_P_Q_equals_52_l756_75688

theorem sum_P_Q_equals_52 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)) →
  P + Q = 52 := by
sorry

end sum_P_Q_equals_52_l756_75688


namespace ram_bicycle_sale_loss_percentage_l756_75669

/-- Calculates the percentage loss on the second bicycle sold by Ram -/
theorem ram_bicycle_sale_loss_percentage :
  let selling_price : ℚ := 990
  let total_cost : ℚ := 1980
  let profit_percentage_first : ℚ := 10 / 100

  let cost_price_first : ℚ := selling_price / (1 + profit_percentage_first)
  let cost_price_second : ℚ := total_cost - cost_price_first
  let loss_second : ℚ := cost_price_second - selling_price
  let loss_percentage_second : ℚ := (loss_second / cost_price_second) * 100

  loss_percentage_second = 25 / 3 := by sorry

end ram_bicycle_sale_loss_percentage_l756_75669


namespace arithmetic_sequence_common_difference_l756_75684

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l756_75684


namespace inequality_solution_set_l756_75649

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-4 : ℝ) 2) = {x | (2 - x) / (x + 4) > 0} :=
by sorry

end inequality_solution_set_l756_75649


namespace g_range_l756_75692

noncomputable def f (a x : ℝ) : ℝ := a^x / (1 + a^x)

noncomputable def g (a x : ℝ) : ℤ := 
  ⌊f a x - 1/2⌋ + ⌊f a (-x) - 1/2⌋

theorem g_range (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  ∀ x : ℝ, g a x ∈ ({0, -1} : Set ℤ) := by sorry

end g_range_l756_75692


namespace negation_of_proposition_is_true_l756_75695

theorem negation_of_proposition_is_true :
  let p := (∀ x y : ℝ, x + y = 5 → x = 2 ∧ y = 3)
  ¬p = True :=
by
  sorry

end negation_of_proposition_is_true_l756_75695


namespace gcd_problem_l756_75678

theorem gcd_problem (a : ℤ) (h : 1610 ∣ a) :
  Nat.gcd (Int.natAbs (a^2 + 9*a + 35)) (Int.natAbs (a + 5)) = 15 := by
  sorry

end gcd_problem_l756_75678


namespace perimeter_semicircular_arcs_square_l756_75629

/-- The perimeter of a region bounded by four semicircular arcs, each constructed on the sides of a square with side length √2, is equal to 2π√2. -/
theorem perimeter_semicircular_arcs_square (side_length : ℝ) : 
  side_length = Real.sqrt 2 → 
  (4 : ℝ) * (π / 2 * side_length) = 2 * π * Real.sqrt 2 := by
  sorry

end perimeter_semicircular_arcs_square_l756_75629


namespace arithmetic_sum_33_l756_75632

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum_33 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 := by
sorry

end arithmetic_sum_33_l756_75632


namespace floor_tiles_count_l756_75605

/-- Represents a square floor tiled with square tiles -/
structure SquareFloor where
  side_length : ℕ
  is_square : side_length > 0

/-- The number of black tiles on the diagonals of a square floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

theorem floor_tiles_count (floor : SquareFloor) 
  (h : black_tiles floor = 101) : 
  total_tiles floor = 2601 := by
  sorry

end floor_tiles_count_l756_75605


namespace complex_square_equation_l756_75627

theorem complex_square_equation (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  a + b * Complex.I = 4 + 3 * Complex.I :=
by sorry

end complex_square_equation_l756_75627


namespace star_example_l756_75656

/-- The ⬥ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + 2*y) * (x - y)

/-- Theorem stating that 5 ⬥ (2 ⬥ 3) = -143 -/
theorem star_example : star 5 (star 2 3) = -143 := by
  sorry

end star_example_l756_75656


namespace unique_perpendicular_tangent_perpendicular_tangent_equation_angle_of_inclination_range_l756_75658

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Statement for the unique perpendicular tangent line
theorem unique_perpendicular_tangent (a : ℝ) :
  (∃! x : ℝ, f' a x = -1) ↔ a = 3 :=
sorry

-- Statement for the equation of the perpendicular tangent line
theorem perpendicular_tangent_equation (a : ℝ) (h : a = 3) :
  ∃ x y : ℝ, 3*x + y - 8 = 0 ∧ y = f a x ∧ f' a x = -1 :=
sorry

-- Statement for the range of the angle of inclination
theorem angle_of_inclination_range (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, -π/4 < Real.arctan (f' a x) ∧ Real.arctan (f' a x) < π/2 :=
sorry

end unique_perpendicular_tangent_perpendicular_tangent_equation_angle_of_inclination_range_l756_75658


namespace fourth_quadrant_angle_l756_75642

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

/-- An angle is in the fourth quadrant if it's between 270° and 360° -/
def is_fourth_quadrant (α : ℝ) : Prop := 270 < α ∧ α < 360

/-- If α is in the first quadrant, then 360° - α is in the fourth quadrant -/
theorem fourth_quadrant_angle (α : ℝ) (h : is_first_quadrant α) : 
  is_fourth_quadrant (360 - α) := by
  sorry

end fourth_quadrant_angle_l756_75642


namespace complex_multiplication_l756_75625

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - 2*i) = 2 + i := by
  sorry

end complex_multiplication_l756_75625


namespace min_rain_day4_exceeds_21_inches_l756_75623

/-- Represents the rainfall and drainage scenario over 4 days -/
structure RainfallScenario where
  capacity : ℝ  -- Total capacity in inches
  drainRate : ℝ  -- Drainage rate in inches per day
  day1Rain : ℝ  -- Rainfall on day 1 in inches
  day2Rain : ℝ  -- Rainfall on day 2 in inches
  day3Rain : ℝ  -- Rainfall on day 3 in inches

/-- Calculates the minimum rainfall on day 4 to cause flooding -/
def minRainDay4ToFlood (scenario : RainfallScenario) : ℝ :=
  scenario.capacity - (scenario.day1Rain + scenario.day2Rain + scenario.day3Rain - 3 * scenario.drainRate)

/-- Theorem stating the minimum rainfall on day 4 to cause flooding is more than 21 inches -/
theorem min_rain_day4_exceeds_21_inches (scenario : RainfallScenario) 
    (h1 : scenario.capacity = 72) -- 6 feet = 72 inches
    (h2 : scenario.drainRate = 3)
    (h3 : scenario.day1Rain = 10)
    (h4 : scenario.day2Rain = 2 * scenario.day1Rain)
    (h5 : scenario.day3Rain = 1.5 * scenario.day2Rain) : 
  minRainDay4ToFlood scenario > 21 := by
  sorry

#eval minRainDay4ToFlood { capacity := 72, drainRate := 3, day1Rain := 10, day2Rain := 20, day3Rain := 30 }

end min_rain_day4_exceeds_21_inches_l756_75623


namespace odd_function_value_l756_75687

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_value (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f 3 = 7) :
  f (-3) = -7 := by
  sorry

end odd_function_value_l756_75687


namespace antonette_age_l756_75637

theorem antonette_age (a t : ℝ) 
  (h1 : t = 3 * a)  -- Tom is thrice as old as Antonette
  (h2 : a + t = 54) -- The sum of their ages is 54
  : a = 13.5 := by  -- Prove that Antonette's age is 13.5
  sorry

end antonette_age_l756_75637


namespace quadratic_equivalence_l756_75635

/-- The quadratic function y = x^2 - 4x + 3 is equivalent to y = (x-2)^2 - 1 -/
theorem quadratic_equivalence :
  ∀ x : ℝ, x^2 - 4*x + 3 = (x - 2)^2 - 1 := by
  sorry

end quadratic_equivalence_l756_75635


namespace full_bucket_weight_l756_75626

/-- Represents the weight of a bucket with water -/
structure BucketWeight where
  empty : ℝ  -- Weight of the empty bucket
  full : ℝ   -- Weight of water when bucket is full

/-- Given conditions about the bucket weights -/
def bucket_conditions (p q : ℝ) (b : BucketWeight) : Prop :=
  b.empty + (3/4 * b.full) = p ∧ b.empty + (1/3 * b.full) = q

/-- Theorem stating the weight of a fully full bucket -/
theorem full_bucket_weight (p q : ℝ) (b : BucketWeight) 
  (h : bucket_conditions p q b) : 
  b.empty + b.full = (8*p - 3*q) / 5 := by
  sorry

end full_bucket_weight_l756_75626


namespace exists_function_1995_double_l756_75636

/-- The number of iterations in the problem -/
def iterations : ℕ := 1995

/-- Definition of function iteration -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, n => n
  | k + 1, n => f (iterate f k n)

/-- Theorem stating the existence of a function satisfying the condition -/
theorem exists_function_1995_double :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, iterate f iterations n = 2 * n := by
  sorry

end exists_function_1995_double_l756_75636


namespace two_numbers_sum_and_digit_removal_l756_75662

theorem two_numbers_sum_and_digit_removal (x y : ℕ) : 
  x + y = 2014 ∧ 
  3 * (x / 100) = y + 6 ∧ 
  x > y → 
  (x = 1963 ∧ y = 51) ∨ (x = 51 ∧ y = 1963) :=
by sorry

end two_numbers_sum_and_digit_removal_l756_75662


namespace house_position_l756_75693

theorem house_position (total_houses : Nat) (product_difference : Nat) : 
  total_houses = 11 → product_difference = 5 → 
  ∃ (position : Nat), position = 4 ∧ 
    (position - 1) * (total_houses - position) = 
    (position - 2) * (total_houses - position + 1) + product_difference := by
  sorry

end house_position_l756_75693


namespace train_speed_l756_75660

/-- Proves that the speed of a train is 45 km/hr given specific conditions --/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : bridge_length = 225)
  (h3 : crossing_time = 30)
  (h4 : (1 : ℝ) / 3.6 = 1 / 3.6) : -- Conversion factor
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_l756_75660


namespace sqrt_sin_sum_equals_neg_two_cos_three_l756_75630

theorem sqrt_sin_sum_equals_neg_two_cos_three :
  Real.sqrt (1 + Real.sin 6) + Real.sqrt (1 - Real.sin 6) = -2 * Real.cos 3 := by
  sorry

end sqrt_sin_sum_equals_neg_two_cos_three_l756_75630


namespace unique_three_config_score_l756_75621

/-- Represents a quiz score configuration -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- The scoring system for the quiz -/
def score (qs : QuizScore) : ℚ :=
  5 * qs.correct + 1.5 * qs.unanswered

/-- Predicate to check if a QuizScore is valid -/
def is_valid_score (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = 20

/-- Predicate to check if a rational number is a possible quiz score -/
def is_possible_score (s : ℚ) : Prop :=
  ∃ qs : QuizScore, is_valid_score qs ∧ score qs = s

/-- Predicate to check if a rational number has exactly three distinct valid quiz configurations -/
def has_three_configurations (s : ℚ) : Prop :=
  ∃ qs1 qs2 qs3 : QuizScore,
    is_valid_score qs1 ∧ is_valid_score qs2 ∧ is_valid_score qs3 ∧
    score qs1 = s ∧ score qs2 = s ∧ score qs3 = s ∧
    qs1 ≠ qs2 ∧ qs1 ≠ qs3 ∧ qs2 ≠ qs3 ∧
    ∀ qs : QuizScore, is_valid_score qs → score qs = s → (qs = qs1 ∨ qs = qs2 ∨ qs = qs3)

theorem unique_three_config_score :
  ∀ s : ℚ, 0 ≤ s ∧ s ≤ 100 → has_three_configurations s → s = 75 :=
sorry

end unique_three_config_score_l756_75621


namespace total_students_correct_l756_75614

/-- The total number of students at the college -/
def total_students : ℝ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_percentage : ℝ := 32.5

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 594

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct :
  (1 - biology_percentage / 100) * total_students = non_biology_students :=
sorry

end total_students_correct_l756_75614


namespace square_sum_theorem_l756_75606

theorem square_sum_theorem (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90) :
  (x + y)^2 = 130 := by
sorry

end square_sum_theorem_l756_75606


namespace running_percentage_is_fifty_percent_l756_75699

/-- Represents a cricket batsman's score -/
structure BatsmanScore where
  total_runs : ℕ
  boundaries : ℕ
  sixes : ℕ

/-- Calculates the percentage of runs made by running between wickets -/
def runningPercentage (score : BatsmanScore) : ℚ :=
  let boundary_runs := 4 * score.boundaries
  let six_runs := 6 * score.sixes
  let running_runs := score.total_runs - (boundary_runs + six_runs)
  (running_runs : ℚ) / score.total_runs * 100

/-- Theorem: The percentage of runs made by running is 50% for the given score -/
theorem running_percentage_is_fifty_percent (score : BatsmanScore) 
    (h_total : score.total_runs = 120)
    (h_boundaries : score.boundaries = 3)
    (h_sixes : score.sixes = 8) : 
  runningPercentage score = 50 := by
  sorry

end running_percentage_is_fifty_percent_l756_75699


namespace camping_trip_percentage_l756_75653

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (22 : ℝ) / 100 * total_students = (students_more_than_100 : ℝ))
  (h2 : (75 : ℝ) / 100 * (students_on_trip : ℝ) = (students_not_more_than_100 : ℝ))
  (h3 : students_on_trip = students_more_than_100 + students_not_more_than_100) :
  (students_on_trip : ℝ) / total_students = 88 / 100 :=
by sorry

end camping_trip_percentage_l756_75653


namespace column_of_1985_l756_75652

/-- The column number (1-indexed) in which a given odd positive integer appears in the arrangement -/
def columnNumber (n : ℕ) : ℕ :=
  (n % 16 + 15) % 16 / 2 + 1

theorem column_of_1985 : columnNumber 1985 = 1 := by sorry

end column_of_1985_l756_75652


namespace sqrt_sum_rational_iff_equal_and_in_set_l756_75654

def is_valid_pair (m n : ℤ) : Prop :=
  ∃ (q : ℚ), (Real.sqrt (n + Real.sqrt 2016) + Real.sqrt (m - Real.sqrt 2016) : ℝ) = q

def valid_n_set : Set ℤ := {505, 254, 130, 65, 50, 46, 45}

theorem sqrt_sum_rational_iff_equal_and_in_set (m n : ℤ) :
  is_valid_pair m n ↔ (m = n ∧ n ∈ valid_n_set) :=
sorry

end sqrt_sum_rational_iff_equal_and_in_set_l756_75654


namespace left_handed_jazz_lovers_l756_75620

/-- Represents a club with members and their music preferences -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_dislike_both : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 25)
  (h2 : c.left_handed = 10)
  (h3 : c.jazz_lovers = 18)
  (h4 : c.right_handed_dislike_both = 3)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members)
  (h6 : ∀ (m : ℕ), m < c.total_members → 
    (m ∈ (Finset.range c.jazz_lovers) ∨ 
     m ∈ (Finset.range (c.total_members - c.jazz_lovers - c.right_handed_dislike_both))))
  : {x : ℕ // x = 10 ∧ x ≤ c.left_handed ∧ x ≤ c.jazz_lovers} :=
by
  sorry

#check left_handed_jazz_lovers

end left_handed_jazz_lovers_l756_75620


namespace cube_cannot_cover_5x5_square_l756_75638

/-- Represents the four possible directions on a chessboard -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the state of the cube -/
structure CubeState :=
  (position : Position)
  (topFace : Fin 6)
  (faceDirections : Fin 6 → Direction)

/-- The set of all positions a cube can visit given its initial state -/
def visitablePositions (initialState : CubeState) : Set Position :=
  sorry

/-- A 5x5 square on the chessboard -/
def square5x5 (topLeft : Position) : Set Position :=
  { p : Position | 
    topLeft.x ≤ p.x ∧ p.x < topLeft.x + 5 ∧
    topLeft.y - 4 ≤ p.y ∧ p.y ≤ topLeft.y }

/-- Theorem stating that the cube cannot cover any 5x5 square -/
theorem cube_cannot_cover_5x5_square (initialState : CubeState) :
  ∀ topLeft : Position, ¬(square5x5 topLeft ⊆ visitablePositions initialState) :=
sorry

end cube_cannot_cover_5x5_square_l756_75638


namespace vector_range_l756_75603

/-- Given unit vectors i and j along x and y axes respectively, and a vector a satisfying 
    |a - i| + |a - 2j| = √5, prove that the range of |a + 2i| is [6√5/5, 3]. -/
theorem vector_range (i j a : ℝ × ℝ) : 
  i = (1, 0) → 
  j = (0, 1) → 
  ‖a - i‖ + ‖a - 2 • j‖ = Real.sqrt 5 → 
  6 * Real.sqrt 5 / 5 ≤ ‖a + 2 • i‖ ∧ ‖a + 2 • i‖ ≤ 3 := by
  sorry

end vector_range_l756_75603


namespace cool_drink_jasmine_percentage_l756_75657

/-- Represents the initial percentage of jasmine water in the solution -/
def initial_percentage : ℝ := 5

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 90

/-- The volume of jasmine added in liters -/
def added_jasmine : ℝ := 8

/-- The volume of water added in liters -/
def added_water : ℝ := 2

/-- The final percentage of jasmine in the solution -/
def final_percentage : ℝ := 12.5

/-- The final volume of the solution in liters -/
def final_volume : ℝ := initial_volume + added_jasmine + added_water

theorem cool_drink_jasmine_percentage :
  (initial_percentage / 100) * initial_volume + added_jasmine = 
  (final_percentage / 100) * final_volume :=
sorry

end cool_drink_jasmine_percentage_l756_75657


namespace smallest_integer_with_remainders_l756_75609

theorem smallest_integer_with_remainders : 
  ∃ n : ℕ, 
    n > 1 ∧
    n % 3 = 2 ∧ 
    n % 7 = 2 ∧ 
    n % 5 = 1 ∧
    (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 7 = 2 ∧ m % 5 = 1 → n ≤ m) ∧
    n = 86 := by
  sorry

end smallest_integer_with_remainders_l756_75609


namespace simplify_absolute_value_l756_75679

theorem simplify_absolute_value : |(-5^2 + 6)| = 19 := by
  sorry

end simplify_absolute_value_l756_75679


namespace first_day_over_500_l756_75667

def paperclips (k : ℕ) : ℕ := 4 * 3^k

theorem first_day_over_500 : 
  (∃ k : ℕ, paperclips k > 500) ∧ 
  (∀ j : ℕ, j < 5 → paperclips j ≤ 500) ∧ 
  (paperclips 5 > 500) :=
by sorry

end first_day_over_500_l756_75667


namespace election_vote_majority_l756_75691

/-- In an election with two candidates, prove the vote majority for the winner. -/
theorem election_vote_majority
  (total_votes : ℕ)
  (winning_percentage : ℚ)
  (h_total : total_votes = 700)
  (h_percentage : winning_percentage = 70 / 100) :
  (winning_percentage * total_votes : ℚ).floor -
  ((1 - winning_percentage) * total_votes : ℚ).floor = 280 := by
  sorry

end election_vote_majority_l756_75691


namespace not_arithmetic_sequence_l756_75650

theorem not_arithmetic_sequence : ¬∃ (a d : ℝ) (m n k : ℤ), 
  a + (m - 1 : ℝ) * d = 1 ∧ 
  a + (n - 1 : ℝ) * d = Real.sqrt 2 ∧ 
  a + (k - 1 : ℝ) * d = 3 ∧ 
  n = m + 1 ∧ 
  k = n + 1 := by
  sorry

end not_arithmetic_sequence_l756_75650


namespace expansion_equals_difference_of_squares_l756_75648

theorem expansion_equals_difference_of_squares (x y : ℝ) : 
  (5*y - x) * (-5*y - x) = x^2 - 25*y^2 := by
  sorry

end expansion_equals_difference_of_squares_l756_75648


namespace green_green_pairs_l756_75685

/-- Represents the distribution of shirt colors and pairs in a classroom --/
structure Classroom where
  total_students : ℕ
  red_students : ℕ
  green_students : ℕ
  total_pairs : ℕ
  red_red_pairs : ℕ

/-- The theorem states that given the classroom conditions, 
    the number of pairs where both students wear green is 35 --/
theorem green_green_pairs (c : Classroom) 
  (h1 : c.total_students = 144)
  (h2 : c.red_students = 63)
  (h3 : c.green_students = 81)
  (h4 : c.total_pairs = 72)
  (h5 : c.red_red_pairs = 26)
  (h6 : c.total_students = c.red_students + c.green_students) :
  c.total_pairs - c.red_red_pairs - (c.red_students - 2 * c.red_red_pairs) = 35 := by
  sorry

#check green_green_pairs

end green_green_pairs_l756_75685


namespace six_arts_arrangement_l756_75673

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  -- n: total number of lectures
  -- k: position limit for the specific lecture (Mathematics)
  -- m: number of lectures that must be adjacent (Archery and Charioteering)
  sorry

theorem six_arts_arrangement : number_of_arrangements 6 3 2 = 120 := by
  sorry

end six_arts_arrangement_l756_75673


namespace mika_stickers_bought_l756_75675

/-- The number of stickers Mika bought from the store -/
def stickers_bought : ℕ := 26

/-- The number of stickers Mika started with -/
def initial_stickers : ℕ := 20

/-- The number of stickers Mika got for her birthday -/
def birthday_stickers : ℕ := 20

/-- The number of stickers Mika gave to her sister -/
def stickers_given : ℕ := 6

/-- The number of stickers Mika used to decorate a greeting card -/
def stickers_used : ℕ := 58

/-- The number of stickers Mika is left with -/
def remaining_stickers : ℕ := 2

theorem mika_stickers_bought : 
  initial_stickers + birthday_stickers + stickers_bought = 
  stickers_given + stickers_used + remaining_stickers :=
sorry

end mika_stickers_bought_l756_75675


namespace power_of_5000_times_2_l756_75680

theorem power_of_5000_times_2 : ∃ n : ℕ, 2 * (5000 ^ 150) = 10 ^ n ∧ n = 600 := by
  sorry

end power_of_5000_times_2_l756_75680


namespace team_B_better_image_l756_75646

-- Define the structure for a team
structure Team where
  members : ℕ
  avg_height : ℝ
  height_variance : ℝ

-- Define the two teams
def team_A : Team := { members := 20, avg_height := 160, height_variance := 10.5 }
def team_B : Team := { members := 20, avg_height := 160, height_variance := 1.2 }

-- Define a function to determine which team has a better performance image
def better_performance_image (t1 t2 : Team) : Prop :=
  t1.avg_height = t2.avg_height ∧ t1.height_variance < t2.height_variance

-- Theorem statement
theorem team_B_better_image : 
  better_performance_image team_B team_A :=
sorry

end team_B_better_image_l756_75646


namespace circle_radius_is_ten_l756_75613

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 6*x + 12*y

/-- The radius of the circle -/
def circle_radius : ℝ := 10

theorem circle_radius_is_ten :
  ∃ (center_x center_y : ℝ),
    ∀ (x y : ℝ), circle_equation x y ↔ 
      (x - center_x)^2 + (y - center_y)^2 = circle_radius^2 :=
sorry

end circle_radius_is_ten_l756_75613
