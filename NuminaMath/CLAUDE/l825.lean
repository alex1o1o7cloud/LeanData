import Mathlib

namespace ben_age_l825_82526

def Ages : List ℕ := [6, 8, 10, 12, 14]

def ParkPair (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ Ages ∧ b ∈ Ages ∧ a ≠ b

def LibraryPair (a b : ℕ) : Prop := a + b < 20 ∧ a ∈ Ages ∧ b ∈ Ages ∧ a ≠ b

def RemainingAges (park1 park2 lib1 lib2 : ℕ) : List ℕ :=
  Ages.filter (λ x => x ∉ [park1, park2, lib1, lib2])

theorem ben_age :
  ∀ park1 park2 lib1 lib2 youngest_home,
    ParkPair park1 park2 →
    LibraryPair lib1 lib2 →
    youngest_home = (RemainingAges park1 park2 lib1 lib2).minimum →
    10 ∈ RemainingAges park1 park2 lib1 lib2 →
    10 ≠ youngest_home →
    10 = (RemainingAges park1 park2 lib1 lib2).maximum :=
by sorry

end ben_age_l825_82526


namespace range_of_x_minus_y_l825_82512

theorem range_of_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) :
  ∃ a b : ℝ, a = -4 ∧ b = 2 ∧ a < x - y ∧ x - y < b :=
sorry

end range_of_x_minus_y_l825_82512


namespace batsman_highest_score_l825_82508

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_without_extremes : ℚ) 
  (h : total_innings = 46)
  (i : overall_average = 62)
  (j : score_difference = 150)
  (k : average_without_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    highest_score + lowest_score = total_innings * overall_average - (total_innings - 2) * average_without_extremes ∧
    highest_score = 221 :=
by sorry

end batsman_highest_score_l825_82508


namespace sum_consecutive_odd_numbers_remainder_l825_82522

theorem sum_consecutive_odd_numbers_remainder (start : ℕ) (h : start = 10999) :
  (List.sum (List.map (λ i => start + 2 * i) (List.range 7))) % 14 = 7 := by
  sorry

end sum_consecutive_odd_numbers_remainder_l825_82522


namespace product_of_powers_of_ten_l825_82535

theorem product_of_powers_of_ten : (10^0.4) * (10^0.6) * (10^0.3) * (10^0.2) * (10^0.5) = 100 := by
  sorry

end product_of_powers_of_ten_l825_82535


namespace hyperbola_equation_l825_82567

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, x t^2 / a^2 - y t^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 4 * Real.sqrt 5

-- Define the asymptotes
def asymptotes (x y : ℝ → ℝ) : Prop :=
  ∀ t, (2 * x t = y t) ∨ (2 * x t = -y t)

theorem hyperbola_equation (a b : ℝ) (x y : ℝ → ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : is_hyperbola a b x y)
  (h4 : focal_length (Real.sqrt (a^2 + b^2)))
  (h5 : asymptotes x y) :
  is_hyperbola 2 4 x y :=
sorry

end hyperbola_equation_l825_82567


namespace forall_positive_implies_square_plus_greater_than_one_is_false_l825_82523

theorem forall_positive_implies_square_plus_greater_than_one_is_false :
  ¬(∀ x : ℝ, x > 0 → x^2 + x > 1) :=
sorry

end forall_positive_implies_square_plus_greater_than_one_is_false_l825_82523


namespace parallelogram_area_l825_82577

def v : Fin 2 → ℝ := ![6, -4]
def w : Fin 2 → ℝ := ![13, -1]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 46 := by sorry

end parallelogram_area_l825_82577


namespace sufficient_not_necessary_l825_82541

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, b > a ∧ a > 0 → 1 / a^2 > 1 / b^2) ∧
  (∃ a b : ℝ, 1 / a^2 > 1 / b^2 ∧ ¬(b > a ∧ a > 0)) :=
by sorry

end sufficient_not_necessary_l825_82541


namespace complex_product_equals_112_l825_82590

theorem complex_product_equals_112 (y : ℂ) (h : y = Complex.exp (2 * Real.pi * Complex.I / 9)) :
  (3 * y + y^3) * (3 * y^3 + y^9) * (3 * y^6 + y^18) * 
  (3 * y^2 + y^6) * (3 * y^5 + y^15) * (3 * y^7 + y^21) = 112 := by
  sorry

end complex_product_equals_112_l825_82590


namespace divided_square_longer_side_l825_82555

/-- Represents a square divided into a trapezoid and hexagon -/
structure DividedSquare where
  side_length : ℝ
  trapezoid_area : ℝ
  hexagon_area : ℝ
  longer_parallel_side : ℝ

/-- The properties of our specific divided square -/
def my_square : DividedSquare where
  side_length := 2
  trapezoid_area := 2
  hexagon_area := 2
  longer_parallel_side := 2  -- This is what we want to prove

theorem divided_square_longer_side (s : DividedSquare) 
  (h1 : s.side_length = 2)
  (h2 : s.trapezoid_area = s.hexagon_area)
  (h3 : s.trapezoid_area + s.hexagon_area = s.side_length ^ 2)
  (h4 : s.trapezoid_area = (s.longer_parallel_side + s.side_length) * (s.side_length / 2) / 2) :
  s.longer_parallel_side = 2 := by
  sorry

#check divided_square_longer_side my_square

end divided_square_longer_side_l825_82555


namespace cos_105_degrees_l825_82531

theorem cos_105_degrees : Real.cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l825_82531


namespace power_fraction_simplification_l825_82581

theorem power_fraction_simplification :
  (1 : ℚ) / ((-5^4)^2) * (-5)^9 = -5 := by sorry

end power_fraction_simplification_l825_82581


namespace triangle_area_half_parallelogram_area_l825_82573

/-- The area of a triangle with equal base and height is half the area of a parallelogram with the same base and height. -/
theorem triangle_area_half_parallelogram_area (b h : ℝ) (b_pos : 0 < b) (h_pos : 0 < h) :
  (1 / 2 * b * h) = (1 / 2) * (b * h) :=
by sorry

end triangle_area_half_parallelogram_area_l825_82573


namespace symmetric_line_correct_l825_82587

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-axis -/
def yAxis : Line := { a := 1, b := 0, c := 0 }

/-- Check if a line is symmetric to another line with respect to the y-axis -/
def isSymmetricToYAxis (l1 l2 : Line) : Prop :=
  l1.a = -l2.a ∧ l1.b = l2.b ∧ l1.c = l2.c

/-- The original line x - y + 1 = 0 -/
def originalLine : Line := { a := 1, b := -1, c := 1 }

/-- The symmetric line we want to prove -/
def symmetricLine : Line := { a := 1, b := 1, c := -1 }

theorem symmetric_line_correct : 
  isSymmetricToYAxis originalLine symmetricLine :=
sorry

end symmetric_line_correct_l825_82587


namespace piggy_bank_savings_l825_82571

-- Define the initial amount in the piggy bank
def initial_amount : ℕ := 200

-- Define the cost per store trip
def cost_per_trip : ℕ := 2

-- Define the number of trips per month
def trips_per_month : ℕ := 4

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Define the function to calculate the remaining amount
def remaining_amount : ℕ :=
  initial_amount - (cost_per_trip * trips_per_month * months_in_year)

-- Theorem to prove
theorem piggy_bank_savings : remaining_amount = 104 := by
  sorry

end piggy_bank_savings_l825_82571


namespace red_markers_count_l825_82525

theorem red_markers_count (total_markers blue_markers : ℕ) 
  (h1 : total_markers = 105)
  (h2 : blue_markers = 64) :
  total_markers - blue_markers = 41 := by
sorry

end red_markers_count_l825_82525


namespace sale_price_increase_l825_82585

theorem sale_price_increase (regular_price : ℝ) (regular_price_positive : regular_price > 0) : 
  let sale_price := regular_price * (1 - 0.2)
  let price_increase := regular_price - sale_price
  let percent_increase := (price_increase / sale_price) * 100
  percent_increase = 25 := by
sorry

end sale_price_increase_l825_82585


namespace horse_and_saddle_value_l825_82594

/-- The total value of a horse and saddle is $100, given that the horse is worth 7 times as much as the saddle, and the saddle is worth $12.5. -/
theorem horse_and_saddle_value :
  let saddle_value : ℝ := 12.5
  let horse_value : ℝ := 7 * saddle_value
  horse_value + saddle_value = 100 := by
  sorry

end horse_and_saddle_value_l825_82594


namespace percentage_of_a_to_b_l825_82534

theorem percentage_of_a_to_b (A B C D : ℝ) 
  (h1 : A = 0.125 * C)
  (h2 : B = 0.375 * D)
  (h3 : D = 1.225 * C)
  (h4 : C = 0.805 * B) :
  A = 0.100625 * B := by
sorry

end percentage_of_a_to_b_l825_82534


namespace rhinestone_project_l825_82506

theorem rhinestone_project (total : ℚ) : 
  (1 / 3 : ℚ) * total + (1 / 5 : ℚ) * total + 21 = total → 
  total = 45 := by
sorry

end rhinestone_project_l825_82506


namespace gathering_handshakes_l825_82529

/-- The number of handshakes in a gathering of elves and dwarves -/
def total_handshakes (num_elves num_dwarves : ℕ) : ℕ :=
  let elf_handshakes := num_elves * (num_elves - 1) / 2
  let elf_dwarf_handshakes := num_elves * num_dwarves
  elf_handshakes + elf_dwarf_handshakes

/-- Theorem stating the total number of handshakes in the gathering -/
theorem gathering_handshakes :
  total_handshakes 25 18 = 750 := by
  sorry

#eval total_handshakes 25 18

end gathering_handshakes_l825_82529


namespace remainder_problem_l825_82565

theorem remainder_problem (k : ℕ+) (h : ∃ a : ℕ, 120 = a * k ^ 2 + 12) :
  ∃ b : ℕ, 144 = b * k + 0 :=
by sorry

end remainder_problem_l825_82565


namespace apple_ratio_proof_l825_82586

theorem apple_ratio_proof (red_apples green_apples : ℕ) : 
  red_apples = 32 →
  red_apples + green_apples = 44 →
  (red_apples : ℚ) / green_apples = 8 / 3 :=
by
  sorry

end apple_ratio_proof_l825_82586


namespace purely_imaginary_implies_a_eq_two_root_implies_abs_z_eq_two_sqrt_two_l825_82597

/-- Given a complex number z = (a^2 - 5a + 6) + (a - 3)i where a ∈ ℝ -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 5*a + 6) (a - 3)

/-- Part 1: If z is purely imaginary, then a = 2 -/
theorem purely_imaginary_implies_a_eq_two (a : ℝ) :
  z a = Complex.I * Complex.im (z a) → a = 2 := by sorry

/-- Part 2: If z is a root of the equation x^2 - 4x + 8 = 0, then |z| = 2√2 -/
theorem root_implies_abs_z_eq_two_sqrt_two (a : ℝ) :
  (z a)^2 - 4*(z a) + 8 = 0 → Complex.abs (z a) = 2 * Real.sqrt 2 := by sorry

end purely_imaginary_implies_a_eq_two_root_implies_abs_z_eq_two_sqrt_two_l825_82597


namespace ratio_twenty_ten_l825_82582

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  prop1 : a 2 * a 6 = 16
  prop2 : a 4 + a 8 = 8

/-- The ratio of the 20th term to the 10th term is 1 -/
theorem ratio_twenty_ten (seq : GeometricSequence) : seq.a 20 / seq.a 10 = 1 := by
  sorry

#check ratio_twenty_ten

end ratio_twenty_ten_l825_82582


namespace world_cup_knowledge_competition_l825_82517

theorem world_cup_knowledge_competition (p_know : ℝ) (p_guess : ℝ) (num_options : ℕ) :
  p_know = 2/3 →
  p_guess = 1/3 →
  num_options = 4 →
  (p_know * 1 + p_guess * (1 / num_options)) / (p_know + p_guess * (1 / num_options)) = 8/9 :=
by sorry

end world_cup_knowledge_competition_l825_82517


namespace small_painting_price_is_80_l825_82515

/-- The price of a small painting given the conditions of Michael's art sale -/
def small_painting_price (large_price : ℕ) (large_sold small_sold : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - large_price * large_sold) / small_sold

/-- Theorem stating that the price of a small painting is $80 under the given conditions -/
theorem small_painting_price_is_80 :
  small_painting_price 100 5 8 1140 = 80 := by
  sorry

end small_painting_price_is_80_l825_82515


namespace mean_proportional_problem_l825_82599

theorem mean_proportional_problem (x : ℝ) :
  (56.5 : ℝ) = Real.sqrt (x * 64) → x = 3192.25 / 64 := by
  sorry

end mean_proportional_problem_l825_82599


namespace inequality_proof_l825_82592

theorem inequality_proof (x y : ℝ) (h1 : x > -1) (h2 : y > -1) (h3 : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2/3 ∧ 
  (x / (y + 1) + y / (x + 1) = 2/3 ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end inequality_proof_l825_82592


namespace hcf_problem_l825_82552

theorem hcf_problem (a b : ℕ) (h1 : a = 345) (h2 : b < a) 
  (h3 : Nat.lcm a b = Nat.gcd a b * 14 * 15) : Nat.gcd a b = 5 := by
  sorry

end hcf_problem_l825_82552


namespace ryan_english_hours_l825_82530

/-- The number of hours Ryan spends on learning Chinese -/
def chinese_hours : ℕ := 2

/-- The number of hours Ryan spends on learning English -/
def english_hours : ℕ := chinese_hours + 4

/-- Theorem: Ryan spends 6 hours on learning English -/
theorem ryan_english_hours : english_hours = 6 := by
  sorry

end ryan_english_hours_l825_82530


namespace half_abs_diff_squares_20_15_l825_82558

theorem half_abs_diff_squares_20_15 : 
  (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
sorry

end half_abs_diff_squares_20_15_l825_82558


namespace compute_expression_l825_82513

theorem compute_expression : 8 * (1/3)^4 = 8/81 := by
  sorry

end compute_expression_l825_82513


namespace mr_green_garden_yield_l825_82569

/-- Calculates the total expected yield from a rectangular garden -/
def gardenYield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) 
                (potato_yield : ℝ) (carrot_yield : ℝ) : ℝ :=
  let area := (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length
  area * (potato_yield + carrot_yield)

/-- Theorem stating the expected yield from Mr. Green's garden -/
theorem mr_green_garden_yield :
  gardenYield 20 25 2.5 0.5 0.25 = 2343.75 := by
  sorry

end mr_green_garden_yield_l825_82569


namespace jolene_earnings_180_l825_82500

/-- Represents Jolene's earnings from various jobs -/
structure JoleneEarnings where
  babysitting_families : ℕ
  babysitting_rate : ℕ
  car_washing_jobs : ℕ
  car_washing_rate : ℕ

/-- Calculates Jolene's total earnings -/
def total_earnings (e : JoleneEarnings) : ℕ :=
  e.babysitting_families * e.babysitting_rate + e.car_washing_jobs * e.car_washing_rate

/-- Theorem stating that Jolene's total earnings are $180 -/
theorem jolene_earnings_180 :
  ∃ (e : JoleneEarnings),
    e.babysitting_families = 4 ∧
    e.babysitting_rate = 30 ∧
    e.car_washing_jobs = 5 ∧
    e.car_washing_rate = 12 ∧
    total_earnings e = 180 := by
  sorry

end jolene_earnings_180_l825_82500


namespace quadratic_sum_and_reciprocal_l825_82574

theorem quadratic_sum_and_reciprocal (t : ℝ) (h : t^2 - 3*t + 1 = 0) : t + 1/t = 3 := by
  sorry

end quadratic_sum_and_reciprocal_l825_82574


namespace possible_values_of_2a_plus_b_l825_82543

theorem possible_values_of_2a_plus_b (a b x y z : ℕ) :
  a^x = b^y ∧ 
  a^x = 1994^z ∧ 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / z →
  2*a + b = 1001 ∨ 2*a + b = 1996 := by
  sorry

end possible_values_of_2a_plus_b_l825_82543


namespace product_modulo_25_l825_82579

theorem product_modulo_25 (n : ℕ) : 
  65 * 74 * 89 ≡ n [ZMOD 25] → 0 ≤ n ∧ n < 25 → n = 15 := by
  sorry

end product_modulo_25_l825_82579


namespace rectangle_perimeter_problem_l825_82576

/-- Given a rectangle A with perimeter 32 cm and length twice its width,
    and a square B with area one-third of rectangle A's area,
    prove that the perimeter of square B is 64√3/9 cm. -/
theorem rectangle_perimeter_problem (width_A : ℝ) (length_A : ℝ) (side_B : ℝ) :
  width_A > 0 →
  length_A = 2 * width_A →
  2 * (length_A + width_A) = 32 →
  side_B^2 = (1/3) * (length_A * width_A) →
  4 * side_B = (64 * Real.sqrt 3) / 9 :=
by sorry

end rectangle_perimeter_problem_l825_82576


namespace min_both_mozart_bach_l825_82545

theorem min_both_mozart_bach (total : ℕ) (mozart_fans : ℕ) (bach_fans : ℕ)
  (h1 : total = 150)
  (h2 : mozart_fans = 120)
  (h3 : bach_fans = 110)
  : ∃ (both : ℕ), both ≥ 80 ∧ 
    both ≤ mozart_fans ∧ 
    both ≤ bach_fans ∧ 
    ∀ (x : ℕ), x < both → 
      (mozart_fans - x) + (bach_fans - x) > total := by
  sorry

end min_both_mozart_bach_l825_82545


namespace car_speed_decrease_l825_82509

/-- Proves that the speed decrease per interval is 3 mph given the conditions of the problem -/
theorem car_speed_decrease (initial_speed : ℝ) (distance_fifth : ℝ) (interval_duration : ℝ) :
  initial_speed = 45 →
  distance_fifth = 4.4 →
  interval_duration = 8 / 60 →
  ∃ (speed_decrease : ℝ),
    speed_decrease = 3 ∧
    initial_speed - 4 * speed_decrease = distance_fifth / interval_duration :=
by sorry

end car_speed_decrease_l825_82509


namespace last_digit_is_four_l825_82532

/-- Represents the process of repeatedly removing digits in odd positions -/
def remove_odd_positions (n : ℕ) : ℕ → ℕ
| 0 => 0
| 1 => n % 10
| m + 2 => remove_odd_positions (n / 100) m

/-- The initial 100-digit number -/
def initial_number : ℕ := 1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

/-- The theorem stating that the last remaining digit is 4 -/
theorem last_digit_is_four :
  ∃ k, remove_odd_positions initial_number k = 4 ∧ 
       ∀ m > k, remove_odd_positions initial_number m = 0 :=
sorry

end last_digit_is_four_l825_82532


namespace edric_monthly_salary_l825_82551

/-- Calculates the monthly salary given working hours per day, days per week, hourly rate, and weeks per month. -/
def monthly_salary (hours_per_day : ℝ) (days_per_week : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ) : ℝ :=
  hours_per_day * days_per_week * hourly_rate * weeks_per_month

/-- Proves that Edric's monthly salary is approximately $623.52 given the specified working conditions. -/
theorem edric_monthly_salary :
  let hours_per_day : ℝ := 8
  let days_per_week : ℝ := 6
  let hourly_rate : ℝ := 3
  let weeks_per_month : ℝ := 52 / 12
  ∃ ε > 0, |monthly_salary hours_per_day days_per_week hourly_rate weeks_per_month - 623.52| < ε :=
by
  sorry

end edric_monthly_salary_l825_82551


namespace sin_sum_to_product_l825_82595

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end sin_sum_to_product_l825_82595


namespace inequality_proof_l825_82519

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 :=
by sorry

end inequality_proof_l825_82519


namespace decision_box_distinguishes_l825_82542

-- Define the components of control structures
inductive ControlComponent
  | ProcessingBox
  | DecisionBox
  | StartEndBox
  | InputOutputBox

-- Define the types of control structures
structure ControlStructure where
  components : List ControlComponent

-- Define a selection structure
def SelectionStructure : ControlStructure := {
  components := [ControlComponent.ProcessingBox, ControlComponent.DecisionBox, 
                 ControlComponent.StartEndBox, ControlComponent.InputOutputBox]
}

-- Define a sequential structure
def SequentialStructure : ControlStructure := {
  components := [ControlComponent.ProcessingBox, ControlComponent.StartEndBox, 
                 ControlComponent.InputOutputBox]
}

-- Define the distinguishing feature
def isDistinguishingFeature (component : ControlComponent) 
                            (struct1 struct2 : ControlStructure) : Prop :=
  (component ∈ struct1.components) ∧ (component ∉ struct2.components)

-- Theorem stating that the decision box is the distinguishing feature
theorem decision_box_distinguishes :
  isDistinguishingFeature ControlComponent.DecisionBox SelectionStructure SequentialStructure :=
by
  sorry


end decision_box_distinguishes_l825_82542


namespace ratio_of_numbers_with_special_average_l825_82538

theorem ratio_of_numbers_with_special_average (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : (a + b) / 2 = a - b) : a / b = 3 := by
  sorry

end ratio_of_numbers_with_special_average_l825_82538


namespace mothers_age_l825_82520

theorem mothers_age (daughter_age_in_3_years : ℕ) 
  (h1 : daughter_age_in_3_years = 26) 
  (h2 : ∃ (mother_age_5_years_ago daughter_age_5_years_ago : ℕ), 
    mother_age_5_years_ago = 2 * daughter_age_5_years_ago) : 
  ∃ (mother_current_age : ℕ), mother_current_age = 41 := by
sorry

end mothers_age_l825_82520


namespace min_total_books_l825_82537

/-- Represents the number of books for each subject in the library. -/
structure LibraryBooks where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ
  history : ℕ

/-- Defines the conditions for the library books problem. -/
def LibraryBooksProblem (books : LibraryBooks) : Prop :=
  books.physics * 2 = books.chemistry * 3 ∧
  books.chemistry * 3 = books.biology * 4 ∧
  books.biology * 6 = books.mathematics * 5 ∧
  books.mathematics * 8 = books.history * 7 ∧
  books.mathematics ≥ 1000 ∧
  books.physics + books.chemistry + books.biology + books.mathematics + books.history > 10000

/-- Theorem stating the minimum possible total number of books in the library. -/
theorem min_total_books (books : LibraryBooks) (h : LibraryBooksProblem books) :
  books.physics + books.chemistry + books.biology + books.mathematics + books.history = 10050 :=
by
  sorry


end min_total_books_l825_82537


namespace train_bridge_crossing_time_l825_82528

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 215) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 :=
by sorry

end train_bridge_crossing_time_l825_82528


namespace unique_square_friendly_l825_82559

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (n : ℤ) : Prop := ∃ k : ℤ, n = k^2

/-- An integer c is square-friendly if for all integers m, m^2 + 18m + c is a perfect square. -/
def IsSquareFriendly (c : ℤ) : Prop :=
  ∀ m : ℤ, IsPerfectSquare (m^2 + 18*m + c)

/-- Theorem: 81 is the only square-friendly integer. -/
theorem unique_square_friendly : ∃! c : ℤ, IsSquareFriendly c ∧ c = 81 := by
  sorry

end unique_square_friendly_l825_82559


namespace min_dot_product_l825_82501

/-- Ellipse C with foci at (0,-√3) and (0,√3) passing through (√3/2, 1) -/
def ellipse_C (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 = 1

/-- Parabola E with vertex at (0,0) and focus at (1,0) -/
def parabola_E (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- Point on parabola E -/
def point_on_E (x y : ℝ) : Prop :=
  parabola_E x y

/-- Line through focus (1,0) with slope k -/
def line_through_focus (k x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- Perpendicular line through focus (1,0) with slope -1/k -/
def perp_line_through_focus (k x y : ℝ) : Prop :=
  y = -1/k * (x - 1)

/-- Theorem: Minimum value of AG · HB is 16 -/
theorem min_dot_product :
  ∃ (min : ℝ),
    (∀ (k x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
      point_on_E x₁ y₁ ∧ point_on_E x₂ y₂ ∧ point_on_E x₃ y₃ ∧ point_on_E x₄ y₄ ∧
      line_through_focus k x₁ y₁ ∧ line_through_focus k x₂ y₂ ∧
      perp_line_through_focus k x₃ y₃ ∧ perp_line_through_focus k x₄ y₄ →
      ((x₁ - x₃) * (x₄ - x₂) + (y₁ - y₃) * (y₄ - y₂) ≥ min)) ∧
    min = 16 := by
  sorry

end min_dot_product_l825_82501


namespace logo_area_difference_l825_82548

/-- The logo problem -/
theorem logo_area_difference :
  let triangle_side : ℝ := 12
  let square_side : ℝ := 2 * (9 - 3 * Real.sqrt 3)
  let overlapped_area : ℝ := square_side^2 - (square_side / 2) * (triangle_side - square_side / 2)
  let non_overlapping_area : ℝ := 2 * (square_side / 2) * (triangle_side - square_side / 2) / Real.sqrt 3
  overlapped_area - non_overlapping_area = 102.6 - 57.6 * Real.sqrt 3 := by
  sorry

end logo_area_difference_l825_82548


namespace solution_set_part1_a_upper_bound_l825_82540

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem a_upper_bound (a : ℝ) :
  (∀ x ∈ Set.Ici 1, f a x ≥ -x^2 - 2) → a ≤ 4 :=
sorry

end solution_set_part1_a_upper_bound_l825_82540


namespace subcommittee_formation_count_l825_82549

def number_of_ways_to_form_subcommittee (total_republicans : ℕ) (total_democrats : ℕ) (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_count :
  number_of_ways_to_form_subcommittee 10 7 4 3 = 7350 := by
  sorry

end subcommittee_formation_count_l825_82549


namespace freshman_count_l825_82502

theorem freshman_count (f o j s : ℕ) : 
  f * 4 = o * 5 →
  o * 8 = j * 7 →
  j * 7 = s * 9 →
  f + o + j + s = 2158 →
  f = 630 :=
by sorry

end freshman_count_l825_82502


namespace polygon_sides_when_interior_thrice_exterior_polygon_sides_when_interior_thrice_exterior_proof_l825_82575

theorem polygon_sides_when_interior_thrice_exterior : ℕ → Prop :=
  fun n =>
    (180 * (n - 2) = 3 * 360) →
    n = 8

-- The proof is omitted
theorem polygon_sides_when_interior_thrice_exterior_proof :
  polygon_sides_when_interior_thrice_exterior 8 :=
sorry

end polygon_sides_when_interior_thrice_exterior_polygon_sides_when_interior_thrice_exterior_proof_l825_82575


namespace f_properties_l825_82527

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ x, f a x = -f a (-x) ↔ a = 1/2) ∧
  (a = 1/2 →
    (∀ y, -1/2 < y ∧ y < 1/2 → ∃ x, f a x = y) ∧
    (∀ x, -1/2 < f a x ∧ f a x < 1/2)) :=
by sorry

end f_properties_l825_82527


namespace fraction_division_subtraction_l825_82505

theorem fraction_division_subtraction :
  (5 / 6 : ℚ) / (9 / 10 : ℚ) - 1 = -2 / 27 := by sorry

end fraction_division_subtraction_l825_82505


namespace methods_B_and_D_are_correct_l825_82516

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x + 5 * y = 18
def equation2 (x y : ℝ) : Prop := 7 * x + 4 * y = 36

-- Define method B
def methodB (x y : ℝ) : Prop :=
  ∃ (z : ℝ), 9 * x + 9 * y = 54 ∧ z * (9 * x + 9 * y) - (2 * x + 5 * y) = z * 54 - 18

-- Define method D
def methodD (x y : ℝ) : Prop :=
  5 * (7 * x + 4 * y) - 4 * (2 * x + 5 * y) = 5 * 36 - 4 * 18

-- Theorem statement
theorem methods_B_and_D_are_correct :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → methodB x y ∧ methodD x y :=
sorry

end methods_B_and_D_are_correct_l825_82516


namespace triangle_classification_l825_82584

/-- Triangle classification based on side lengths --/
def TriangleType (a b c : ℝ) : Type :=
  { type : String // 
    type = "acute" ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2 ∨
    type = "right" ∧ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) ∨
    type = "obtuse" ∧ (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2) }

theorem triangle_classification :
  ∃ (t1 : TriangleType 4 6 8) (t2 : TriangleType 10 24 26) (t3 : TriangleType 10 12 14),
    t1.val = "obtuse" ∧ t2.val = "right" ∧ t3.val = "acute" := by
  sorry


end triangle_classification_l825_82584


namespace wage_decrease_compensation_l825_82583

/-- Proves that a 25% increase in working hours maintains the same income after a 20% wage decrease --/
theorem wage_decrease_compensation (W H S : ℝ) (C : ℝ) (H_pos : H > 0) (W_pos : W > 0) :
  let original_income := W * H + C * S
  let new_wage := W * 0.8
  let new_hours := H * 1.25
  new_wage * new_hours + C * S = original_income := by
  sorry

end wage_decrease_compensation_l825_82583


namespace min_value_on_line_l825_82598

/-- The minimum value of ((a+1)^2 + b^2) is 3, given that (a,b) is on y = √3x - √3 -/
theorem min_value_on_line :
  ∀ a b : ℝ, b = Real.sqrt 3 * a - Real.sqrt 3 →
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (x + 1)^2 + y^2 ≥ 3) ∧
  ∃ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 ∧ (x + 1)^2 + y^2 = 3 :=
by sorry

end min_value_on_line_l825_82598


namespace parabola_with_vertex_and_focus_parabola_through_point_l825_82568

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Theorem 1
theorem parabola_with_vertex_and_focus
  (p : Parabola)
  (vertex : Point)
  (focus : Point)
  (h1 : vertex.x = 0 ∧ vertex.y = 0)
  (h2 : focus.x = 6 ∧ focus.y = 0) :
  p.equation = fun x y ↦ y^2 = 24*x :=
sorry

-- Theorem 2
theorem parabola_through_point
  (p : Parabola)
  (point : Point)
  (h : point.x = 1 ∧ point.y = 2) :
  (p.equation = fun x y ↦ x^2 = (1/2)*y) ∨
  (p.equation = fun x y ↦ y^2 = 4*x) :=
sorry

end parabola_with_vertex_and_focus_parabola_through_point_l825_82568


namespace total_grading_time_l825_82533

def math_worksheets : ℕ := 45
def science_worksheets : ℕ := 37
def history_worksheets : ℕ := 32

def math_grading_time : ℕ := 15
def science_grading_time : ℕ := 20
def history_grading_time : ℕ := 25

theorem total_grading_time :
  math_worksheets * math_grading_time +
  science_worksheets * science_grading_time +
  history_worksheets * history_grading_time = 2215 := by
sorry

end total_grading_time_l825_82533


namespace complement_of_A_l825_82589

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | |x - 1| > 2}

theorem complement_of_A : 
  Set.compl A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end complement_of_A_l825_82589


namespace pushup_difference_l825_82570

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 44

/-- The number of push-ups David did -/
def david_pushups : ℕ := 78

/-- The difference in crunches between Zachary and David -/
def crunch_difference : ℕ := 27

/-- Theorem stating the difference in push-ups between David and Zachary -/
theorem pushup_difference : david_pushups - zachary_pushups = 19 := by
  sorry

end pushup_difference_l825_82570


namespace hash_difference_l825_82510

def hash (x y : ℤ) : ℤ := 2 * x * y - 3 * x + y

theorem hash_difference : (hash 6 4) - (hash 4 6) = -8 := by
  sorry

end hash_difference_l825_82510


namespace shape_has_four_sides_l825_82591

/-- The shape being fenced -/
structure Shape where
  sides : ℕ
  cost_per_side : ℕ
  total_cost : ℕ

/-- The shape satisfies the given conditions -/
def satisfies_conditions (s : Shape) : Prop :=
  s.cost_per_side = 69 ∧ s.total_cost = 276 ∧ s.total_cost = s.cost_per_side * s.sides

theorem shape_has_four_sides (s : Shape) (h : satisfies_conditions s) : s.sides = 4 := by
  sorry

end shape_has_four_sides_l825_82591


namespace fraction_simplification_l825_82503

theorem fraction_simplification (x : ℝ) : 
  (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end fraction_simplification_l825_82503


namespace min_value_theorem_l825_82588

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  2/x + 1/y ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 2 ∧ 2/x₀ + 1/y₀ = 4 :=
by sorry

end min_value_theorem_l825_82588


namespace max_distance_complex_l825_82560

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 ∧
             ∀ (v : ℂ), Complex.abs (v + 2 - 2*I) = 1 →
                        Complex.abs (v - 2 - 2*I) ≤ Complex.abs (w - 2 - 2*I) ∧
             Complex.abs (w - 2 - 2*I) = 5 :=
by sorry

end max_distance_complex_l825_82560


namespace shiela_painting_distribution_l825_82511

/-- Given Shiela has 18 paintings and 2 grandmothers, prove that each grandmother
    receives 9 paintings when the paintings are distributed equally. -/
theorem shiela_painting_distribution
  (total_paintings : ℕ)
  (num_grandmothers : ℕ)
  (h1 : total_paintings = 18)
  (h2 : num_grandmothers = 2)
  : total_paintings / num_grandmothers = 9 := by
  sorry

end shiela_painting_distribution_l825_82511


namespace speed_ratio_theorem_l825_82580

/-- Given two objects A and B with speeds v₁ and v₂ respectively, 
    if they meet after a hours when moving towards each other
    and A overtakes B after b hours when moving in the same direction,
    then the ratio of their speeds v₁/v₂ = (a + b) / (b - a). -/
theorem speed_ratio_theorem (v₁ v₂ a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : v₁ > v₂) :
  (∃ S : ℝ, S > 0 ∧ S = a * (v₁ + v₂) ∧ S = b * (v₁ - v₂)) →
  v₁ / v₂ = (a + b) / (b - a) :=
by sorry

end speed_ratio_theorem_l825_82580


namespace average_of_solutions_is_zero_l825_82536

theorem average_of_solutions_is_zero :
  let f : ℝ → ℝ := fun x => Real.sqrt (3 * x^2 + 4)
  let solutions := {x : ℝ | f x = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → (x = x₁ ∨ x = x₂) := by
  sorry

end average_of_solutions_is_zero_l825_82536


namespace fraction_product_l825_82539

theorem fraction_product : (2 : ℚ) / 3 * (4 : ℚ) / 7 * (9 : ℚ) / 13 = (24 : ℚ) / 91 := by
  sorry

end fraction_product_l825_82539


namespace not_necessarily_true_squared_l825_82553

theorem not_necessarily_true_squared (a b : ℝ) (h : a < b) : 
  ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end not_necessarily_true_squared_l825_82553


namespace largest_integer_divisibility_l825_82550

theorem largest_integer_divisibility : ∃ (n : ℕ), n = 1956 ∧ 
  (∀ m : ℕ, m > n → ¬(∃ k : ℤ, (m^2 - 2012 : ℤ) = k * (m + 7))) ∧
  (∃ k : ℤ, (n^2 - 2012 : ℤ) = k * (n + 7)) :=
sorry

end largest_integer_divisibility_l825_82550


namespace total_dog_legs_l825_82518

/-- The standard number of legs for a dog -/
def standard_dog_legs : ℕ := 4

/-- The number of dogs in the park -/
def dogs_in_park : ℕ := 109

/-- Theorem: The total number of dog legs in the park is 436 -/
theorem total_dog_legs : dogs_in_park * standard_dog_legs = 436 := by
  sorry

end total_dog_legs_l825_82518


namespace smallest_multiple_of_seven_all_nines_l825_82546

def is_all_nines (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^k - 1

theorem smallest_multiple_of_seven_all_nines :
  ∃ N : ℕ, (N = 142857 ∧
            is_all_nines (7 * N) ∧
            ∀ m : ℕ, m < N → ¬is_all_nines (7 * m)) :=
sorry

end smallest_multiple_of_seven_all_nines_l825_82546


namespace sum_of_consecutive_integers_l825_82572

theorem sum_of_consecutive_integers :
  let start : Int := -9
  let count : Nat := 20
  let sequence := List.range count |>.map (λ i => start + i)
  sequence.sum = 10 := by
  sorry

end sum_of_consecutive_integers_l825_82572


namespace trigonometric_expression_equality_l825_82521

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  ((4 * (Real.cos (12 * π / 180))^2 - 2) * Real.sin (12 * π / 180)) = 
  -4 * Real.sqrt 3 := by
  sorry

end trigonometric_expression_equality_l825_82521


namespace collinear_points_q_value_l825_82524

/-- 
If the points (7, q), (5, 3), and (1, -1) are collinear, then q = 5.
-/
theorem collinear_points_q_value (q : ℝ) : 
  (∃ (t : ℝ), (7 - 1) = t * (5 - 1) ∧ (q + 1) = t * (3 + 1)) → q = 5 := by
  sorry

end collinear_points_q_value_l825_82524


namespace triangle_area_l825_82544

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) : 
  let S := (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)
  S = 84 := by sorry

end triangle_area_l825_82544


namespace surface_area_of_T_l825_82561

-- Define the cube
structure Cube where
  edge_length : ℝ
  vertex_A : ℝ × ℝ × ℝ

-- Define points on the cube
def L (c : Cube) : ℝ × ℝ × ℝ := (3, 0, 0)
def M (c : Cube) : ℝ × ℝ × ℝ := (0, 3, 0)
def N (c : Cube) : ℝ × ℝ × ℝ := (0, 0, 3)
def P (c : Cube) : ℝ × ℝ × ℝ := (c.edge_length, c.edge_length, c.edge_length)

-- Define the solid T
structure SolidT (c : Cube) where
  tunnel_sides : Set (ℝ × ℝ × ℝ)

-- Define the surface area of T
def surface_area (t : SolidT c) : ℝ := sorry

-- Theorem statement
theorem surface_area_of_T (c : Cube) (t : SolidT c) :
  c.edge_length = 10 →
  surface_area t = 582 + 9 * Real.sqrt 6 :=
sorry

end surface_area_of_T_l825_82561


namespace tree_planting_event_percentage_l825_82578

theorem tree_planting_event_percentage (boys : ℕ) (girls : ℕ) : 
  boys = 600 →
  girls = boys + 400 →
  (960 : ℚ) / (boys + girls : ℚ) = 60 / 100 := by
  sorry

end tree_planting_event_percentage_l825_82578


namespace expression_equality_l825_82562

theorem expression_equality : 3 * Real.sqrt 2 + |1 - Real.sqrt 2| + (8 : ℝ) ^ (1/3) = 4 * Real.sqrt 2 + 1 := by
  sorry

end expression_equality_l825_82562


namespace sample_size_is_450_l825_82593

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ

/-- Theorem: Given a population of 5000 students and a sample of 450 students,
    the sample size is 450. -/
theorem sample_size_is_450 (pop : Population) (sample : Sample) 
    (h1 : pop.size = 5000) (h2 : sample.size = 450) : 
  sample.size = 450 := by
  sorry

#check sample_size_is_450

end sample_size_is_450_l825_82593


namespace circle_tangent_perpendicular_l825_82556

-- Define the types for our geometric objects
variable (Point Circle Line : Type)

-- Define the necessary operations and relations
variable (radius : Circle → ℝ)
variable (intersect : Circle → Circle → Set Point)
variable (tangent_point : Circle → Line → Point)
variable (line_through : Point → Point → Line)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem circle_tangent_perpendicular 
  (Γ Γ' : Circle) 
  (A B C D : Point) 
  (t : Line) :
  radius Γ = radius Γ' →
  A ∈ intersect Γ Γ' →
  B ∈ intersect Γ Γ' →
  C = tangent_point Γ t →
  D = tangent_point Γ' t →
  perpendicular (line_through A C) (line_through B D) :=
by sorry

end circle_tangent_perpendicular_l825_82556


namespace sufficient_but_not_necessary_l825_82504

theorem sufficient_but_not_necessary (a : ℝ) :
  (((1 / a) > (1 / 4)) → (∀ x : ℝ, a * x^2 + a * x + 1 > 0)) ∧
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ ((1 / a) ≤ (1 / 4))) := by
  sorry

end sufficient_but_not_necessary_l825_82504


namespace sequence_property_l825_82507

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => a (i + 1))

theorem sequence_property (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, n > 0 → sequence_sum a n = 2 * a n - n) :
  (a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7) ∧
  (∀ n : ℕ, n > 0 → a n = 2^n - 1) := by
  sorry

end sequence_property_l825_82507


namespace imaginary_unit_sum_l825_82596

theorem imaginary_unit_sum (i : ℂ) (hi : i^2 = -1) : i^11 + i^111 + i^222 = -2*i - 1 := by
  sorry

end imaginary_unit_sum_l825_82596


namespace equation_solution_l825_82566

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = (4 * Real.sqrt 3) / 3 ∧ 
  (∀ x : ℝ, Real.sqrt 3 * x * (x - 5) + 4 * (5 - x) = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l825_82566


namespace xy_value_from_absolute_sum_l825_82554

theorem xy_value_from_absolute_sum (x y : ℝ) :
  |x - 5| + |y + 3| = 0 → x * y = -15 := by
  sorry

end xy_value_from_absolute_sum_l825_82554


namespace product_of_conjugates_equals_one_l825_82563

theorem product_of_conjugates_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end product_of_conjugates_equals_one_l825_82563


namespace magical_stack_size_l825_82547

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards

/-- Checks if a card number is in its original position after restacking -/
def retains_position (stack : CardStack) (card : ℕ) : Prop :=
  card ≤ stack.n → card = 2 * card - 1
  ∧ card > stack.n → card = 2 * (card - stack.n)

/-- Defines a magical stack -/
def is_magical (stack : CardStack) : Prop :=
  ∃ (a b : ℕ), a ≤ stack.n ∧ b > stack.n ∧ retains_position stack a ∧ retains_position stack b

/-- Main theorem: A magical stack where card 161 retains its position has 482 cards -/
theorem magical_stack_size :
  ∀ (stack : CardStack),
    is_magical stack →
    retains_position stack 161 →
    2 * stack.n = 482 :=
by sorry

end magical_stack_size_l825_82547


namespace lines_perpendicular_to_plane_are_parallel_l825_82514

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Define the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (l m : Line) (α : Plane) :
  l ≠ m →
  perpendicular l α →
  perpendicular m α →
  parallel l m :=
sorry

end lines_perpendicular_to_plane_are_parallel_l825_82514


namespace sqrt_sum_equals_abs_sum_l825_82557

theorem sqrt_sum_equals_abs_sum (x : ℝ) :
  Real.sqrt (x^2 + 6*x + 9) + Real.sqrt (x^2 - 6*x + 9) = |x - 3| + |x + 3| := by
  sorry

end sqrt_sum_equals_abs_sum_l825_82557


namespace train_passing_time_l825_82564

/-- Calculates the time taken for a train to pass a man moving in the opposite direction. -/
theorem train_passing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 200 →
  train_speed = 80 →
  man_speed = 10 →
  (train_length / ((train_speed + man_speed) * 1000 / 3600)) = 8 := by
  sorry

#check train_passing_time

end train_passing_time_l825_82564
