import Mathlib

namespace NUMINAMATH_CALUDE_medical_team_selection_l1331_133104

theorem medical_team_selection (male_doctors : ℕ) (female_doctors : ℕ) : 
  male_doctors = 6 → female_doctors = 5 → 
  (male_doctors.choose 2) * (female_doctors.choose 1) = 75 := by
sorry

end NUMINAMATH_CALUDE_medical_team_selection_l1331_133104


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l1331_133194

/-- Defines the equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 16*y^2 - 8*x + 16*y + 32 = 0

/-- Theorem stating that the conic equation represents a hyperbola -/
theorem conic_is_hyperbola :
  ∃ (h k a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ (x y : ℝ), conic_equation x y ↔ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l1331_133194


namespace NUMINAMATH_CALUDE_buttons_given_to_mary_l1331_133119

theorem buttons_given_to_mary (initial_buttons : ℕ) (buttons_left : ℕ) : initial_buttons - buttons_left = 4 :=
by
  sorry

#check buttons_given_to_mary 9 5

end NUMINAMATH_CALUDE_buttons_given_to_mary_l1331_133119


namespace NUMINAMATH_CALUDE_combination_equation_solution_permutation_equation_solution_l1331_133155

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Permutation (Arrangement) -/
def permutation (n k : ℕ) : ℕ := sorry

theorem combination_equation_solution :
  ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9 →
  (binomial 9 x = binomial 9 (2*x - 3)) ↔ (x = 3 ∨ x = 4) := by sorry

theorem permutation_equation_solution :
  ∀ x : ℕ, 0 < x ∧ x ≤ 8 →
  (permutation 8 x = 6 * permutation 8 (x - 2)) ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_combination_equation_solution_permutation_equation_solution_l1331_133155


namespace NUMINAMATH_CALUDE_population_change_theorem_l1331_133142

/-- Represents the population change factor for a given percentage change -/
def change_factor (percent : ℚ) : ℚ := 1 + percent / 100

/-- Calculates the net change in population over 5 years given the yearly changes -/
def net_change (year1 year2 year3 year4 year5 : ℚ) : ℚ :=
  (change_factor year1 * change_factor year2 * change_factor year3 * 
   change_factor year4 * change_factor year5 - 1) * 100

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  if q - ⌊q⌋ < 1/2 then ⌊q⌋ else ⌈q⌉

theorem population_change_theorem :
  round_to_nearest (net_change 20 10 (-30) (-20) 10) = -19 := by sorry

end NUMINAMATH_CALUDE_population_change_theorem_l1331_133142


namespace NUMINAMATH_CALUDE_dave_total_earnings_l1331_133173

/-- Calculates daily earnings after tax -/
def dailyEarnings (hourlyWage : ℚ) (hoursWorked : ℚ) (unpaidBreak : ℚ) : ℚ :=
  let actualHours := hoursWorked - unpaidBreak
  let earningsBeforeTax := actualHours * hourlyWage
  let taxDeduction := earningsBeforeTax * (1 / 10)
  earningsBeforeTax - taxDeduction

/-- Represents Dave's total earnings for the week -/
def daveEarnings : ℚ :=
  dailyEarnings 6 6 (1/2) +  -- Monday
  dailyEarnings 7 2 (1/4) +  -- Tuesday
  dailyEarnings 9 3 0 +      -- Wednesday
  dailyEarnings 8 5 (1/2)    -- Thursday

theorem dave_total_earnings :
  daveEarnings = 9743 / 100 := by sorry

end NUMINAMATH_CALUDE_dave_total_earnings_l1331_133173


namespace NUMINAMATH_CALUDE_product_squared_l1331_133122

theorem product_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by sorry

end NUMINAMATH_CALUDE_product_squared_l1331_133122


namespace NUMINAMATH_CALUDE_digit_145_of_49_div_686_l1331_133181

/-- The decimal expansion of 49/686 has a period of 6 -/
def period : ℕ := 6

/-- The repeating sequence in the decimal expansion of 49/686 -/
def repeating_sequence : Fin 6 → ℕ
| 0 => 0
| 1 => 7
| 2 => 1
| 3 => 4
| 4 => 2
| 5 => 8

/-- The 145th digit after the decimal point in the decimal expansion of 49/686 is 8 -/
theorem digit_145_of_49_div_686 : 
  repeating_sequence ((145 - 1) % period) = 8 := by sorry

end NUMINAMATH_CALUDE_digit_145_of_49_div_686_l1331_133181


namespace NUMINAMATH_CALUDE_stating_school_travel_time_l1331_133158

/-- Represents the time in minutes to get from home to school -/
def time_to_school : ℕ := 12

/-- Represents the fraction of the way Kolya walks before realizing he forgot his book -/
def initial_fraction : ℚ := 1/4

/-- Represents the time in minutes Kolya arrives early if he doesn't go back -/
def early_time : ℕ := 5

/-- Represents the time in minutes Kolya arrives late if he goes back -/
def late_time : ℕ := 1

/-- 
Theorem stating that the time to get to school is 12 minutes, given the conditions of the problem.
-/
theorem school_travel_time :
  time_to_school = 12 ∧
  initial_fraction = 1/4 ∧
  early_time = 5 ∧
  late_time = 1 →
  time_to_school = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_stating_school_travel_time_l1331_133158


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l1331_133179

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  (2/a + 1/b) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    Real.sqrt 2 = Real.sqrt (4^a₀ * 2^b₀) ∧ 2/a₀ + 1/b₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l1331_133179


namespace NUMINAMATH_CALUDE_computation_problem_points_l1331_133171

theorem computation_problem_points :
  ∀ (total_problems : ℕ) 
    (computation_problems : ℕ) 
    (word_problem_points : ℕ) 
    (total_points : ℕ),
  total_problems = 30 →
  computation_problems = 20 →
  word_problem_points = 5 →
  total_points = 110 →
  ∃ (computation_problem_points : ℕ),
    computation_problem_points * computation_problems +
    word_problem_points * (total_problems - computation_problems) = total_points ∧
    computation_problem_points = 3 :=
by sorry

end NUMINAMATH_CALUDE_computation_problem_points_l1331_133171


namespace NUMINAMATH_CALUDE_product_mod_seven_l1331_133133

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l1331_133133


namespace NUMINAMATH_CALUDE_circle_square_area_difference_l1331_133193

theorem circle_square_area_difference :
  let square_side : ℝ := 12
  let circle_diameter : ℝ := 16
  let π : ℝ := 3
  let square_area := square_side ^ 2
  let circle_area := π * (circle_diameter / 2) ^ 2
  circle_area - square_area = 48 := by sorry

end NUMINAMATH_CALUDE_circle_square_area_difference_l1331_133193


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_36_l1331_133106

theorem arithmetic_square_root_of_36 : 
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 36 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_36_l1331_133106


namespace NUMINAMATH_CALUDE_meeting_attendees_l1331_133178

theorem meeting_attendees (total_handshakes : ℕ) (h : total_handshakes = 91) :
  ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = total_handshakes ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendees_l1331_133178


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l1331_133143

theorem largest_n_satisfying_inequality : 
  (∀ n : ℕ, n^6033 < 2011^2011 → n ≤ 12) ∧ 12^6033 < 2011^2011 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l1331_133143


namespace NUMINAMATH_CALUDE_g_of_2_eq_3_l1331_133167

def g (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem g_of_2_eq_3 : g 2 = 3 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_3_l1331_133167


namespace NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l1331_133127

theorem a_fourth_zero_implies_a_squared_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l1331_133127


namespace NUMINAMATH_CALUDE_reading_time_difference_l1331_133108

/-- Given Xanthia's and Molly's reading speeds and a book length, 
    calculate the difference in reading time in minutes. -/
theorem reading_time_difference 
  (xanthia_speed molly_speed book_length : ℕ) 
  (hx : xanthia_speed = 120)
  (hm : molly_speed = 40)
  (hb : book_length = 360) : 
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 360 := by
  sorry

#check reading_time_difference

end NUMINAMATH_CALUDE_reading_time_difference_l1331_133108


namespace NUMINAMATH_CALUDE_red_ant_percentage_l1331_133123

/-- Proves that the percentage of red ants in the population is 85%, given the specified conditions. -/
theorem red_ant_percentage (female_red_percentage : ℝ) (male_red_total_percentage : ℝ) :
  female_red_percentage = 45 →
  male_red_total_percentage = 46.75 →
  ∃ (red_percentage : ℝ),
    red_percentage = 85 ∧
    (100 - female_red_percentage) / 100 * red_percentage = male_red_total_percentage :=
by sorry

end NUMINAMATH_CALUDE_red_ant_percentage_l1331_133123


namespace NUMINAMATH_CALUDE_fifth_number_in_row_l1331_133170

-- Define Pascal's triangle
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the row we're interested in
def targetRow : ℕ → ℕ
  | 0 => 1
  | 1 => 15
  | k => pascal 15 (k - 1)

-- State the theorem
theorem fifth_number_in_row : targetRow 5 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_row_l1331_133170


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_endpoints_locus_l1331_133120

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  A : Point
  B : Point
  C : Point
  centroid : Point
  orthocenter : Point
  isIsosceles : A.x = -B.x ∧ A.y = B.y
  centroidOrigin : centroid = ⟨0, 0⟩
  orthocenterOnYAxis : orthocenter = ⟨0, 1⟩
  thirdVertexOnYAxis : C.x = 0

/-- The locus of base endpoints of an isosceles triangle -/
def locusOfBaseEndpoints (p : Point) : Prop :=
  p.x ≠ 0 ∧ 3 * (p.y - 1/2)^2 - p.x^2 = 3/4

/-- Theorem stating that the base endpoints of the isosceles triangle lie on the specified locus -/
theorem isosceles_triangle_base_endpoints_locus (triangle : IsoscelesTriangle) :
  locusOfBaseEndpoints triangle.A ∧ locusOfBaseEndpoints triangle.B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_endpoints_locus_l1331_133120


namespace NUMINAMATH_CALUDE_johnny_video_game_cost_l1331_133162

/-- The amount Johnny spent on the video game -/
def video_game_cost (september_savings october_savings november_savings amount_left : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - amount_left

/-- Theorem: Johnny spent $58 on the video game -/
theorem johnny_video_game_cost :
  video_game_cost 30 49 46 67 = 58 := by
  sorry

end NUMINAMATH_CALUDE_johnny_video_game_cost_l1331_133162


namespace NUMINAMATH_CALUDE_alissa_has_more_present_difference_l1331_133153

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := 53

/-- Alissa has more presents than Ethan -/
theorem alissa_has_more : alissa_presents > ethan_presents := by sorry

/-- The difference between Alissa's and Ethan's presents is 22 -/
theorem present_difference : alissa_presents - ethan_presents = 22 := by sorry

end NUMINAMATH_CALUDE_alissa_has_more_present_difference_l1331_133153


namespace NUMINAMATH_CALUDE_largest_number_with_property_l1331_133196

theorem largest_number_with_property : ∃ n : ℕ, 
  (n ≤ 100) ∧ 
  ((n - 2) % 7 = 0) ∧ 
  ((n - 2) % 8 = 0) ∧ 
  (∀ m : ℕ, m ≤ 100 → ((m - 2) % 7 = 0) ∧ ((m - 2) % 8 = 0) → m ≤ n) ∧
  n = 58 := by
sorry

end NUMINAMATH_CALUDE_largest_number_with_property_l1331_133196


namespace NUMINAMATH_CALUDE_inverse_of_A_l1331_133139

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  A⁻¹ = !![(-1 : ℝ), -3; -2, -5] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1331_133139


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1331_133100

/-- Calculates the weight loss for Jessie based on her exercise routine --/
def weight_loss (initial_weight : ℝ) (exercise_days : ℕ) (even_day_loss : ℝ) (odd_day_loss : ℝ) : ℝ :=
  let even_days := (exercise_days - 1) / 2
  let odd_days := exercise_days - even_days
  even_days * even_day_loss + odd_days * odd_day_loss

/-- Theorem stating that Jessie's weight loss is 8.1 kg --/
theorem jessie_weight_loss :
  let initial_weight : ℝ := 74
  let exercise_days : ℕ := 25
  let even_day_loss : ℝ := 0.2 + 0.15
  let odd_day_loss : ℝ := 0.3
  weight_loss initial_weight exercise_days even_day_loss odd_day_loss = 8.1 := by
  sorry

#eval weight_loss 74 25 (0.2 + 0.15) 0.3

end NUMINAMATH_CALUDE_jessie_weight_loss_l1331_133100


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1331_133197

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 - 8 * X^3 - 12 * X^2 + 5 * X - 9
  let divisor : Polynomial ℚ := 3 * X^2 - 2
  let quotient := dividend / divisor
  (quotient.coeff 2 = 10/3) ∧ (quotient.coeff 1 = -8/3) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1331_133197


namespace NUMINAMATH_CALUDE_ellipse_line_slope_product_l1331_133146

/-- Given an ellipse C and a line l intersecting C at two points, 
    prove that the product of slopes of OM and l is -9 --/
theorem ellipse_line_slope_product (x₁ x₂ y₁ y₂ : ℝ) 
  (hC₁ : 9 * x₁^2 + y₁^2 = 1)
  (hC₂ : 9 * x₂^2 + y₂^2 = 1)
  (h_not_origin : x₁ ≠ 0 ∨ y₁ ≠ 0)
  (h_not_parallel : x₁ ≠ x₂ ∧ y₁ ≠ y₂) :
  let k_OM := (y₁ + y₂) / (x₁ + x₂)
  let k_l := (y₁ - y₂) / (x₁ - x₂)
  k_OM * k_l = -9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_product_l1331_133146


namespace NUMINAMATH_CALUDE_johns_zoo_l1331_133175

theorem johns_zoo (snakes : ℕ) (monkeys : ℕ) (lions : ℕ) (pandas : ℕ) (dogs : ℕ) :
  snakes = 15 ∧
  monkeys = 2 * snakes ∧
  lions = monkeys - 5 ∧
  pandas = lions + 8 ∧
  dogs = pandas / 3 →
  snakes + monkeys + lions + pandas + dogs = 114 := by
sorry

end NUMINAMATH_CALUDE_johns_zoo_l1331_133175


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l1331_133138

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- State the theorem
theorem sqrt_two_irrational : ¬ IsRational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l1331_133138


namespace NUMINAMATH_CALUDE_geometric_sequence_s6_l1331_133154

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1

/-- Theorem stating the result for S_6 given the conditions -/
theorem geometric_sequence_s6 (seq : GeometricSequence) 
  (h1 : seq.a 3 = 4)
  (h2 : seq.S 3 = 7) :
  seq.S 6 = 63 ∨ seq.S 6 = 133/27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_s6_l1331_133154


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l1331_133128

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l1331_133128


namespace NUMINAMATH_CALUDE_remainder_theorem_l1331_133117

theorem remainder_theorem : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1331_133117


namespace NUMINAMATH_CALUDE_robin_gum_count_l1331_133129

/-- The number of gum pieces Robin has after his purchases -/
def total_gum_pieces : ℕ :=
  let initial_packages := 27
  let initial_pieces_per_package := 18
  let additional_packages_1 := 15
  let additional_pieces_per_package_1 := 12
  let additional_packages_2 := 8
  let additional_pieces_per_package_2 := 25
  initial_packages * initial_pieces_per_package +
  additional_packages_1 * additional_pieces_per_package_1 +
  additional_packages_2 * additional_pieces_per_package_2

theorem robin_gum_count : total_gum_pieces = 866 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l1331_133129


namespace NUMINAMATH_CALUDE_average_lawn_cuts_per_month_l1331_133121

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months Mr. Roper cuts his lawn 15 times per month -/
def high_frequency_months : ℕ := 6

/-- The number of months Mr. Roper cuts his lawn 3 times per month -/
def low_frequency_months : ℕ := 6

/-- The number of times Mr. Roper cuts his lawn in high frequency months -/
def high_frequency_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn in low frequency months -/
def low_frequency_cuts : ℕ := 3

/-- Theorem stating that the average number of times Mr. Roper cuts his lawn per month is 9 -/
theorem average_lawn_cuts_per_month :
  (high_frequency_months * high_frequency_cuts + low_frequency_months * low_frequency_cuts) / months_in_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_lawn_cuts_per_month_l1331_133121


namespace NUMINAMATH_CALUDE_three_digit_number_subtraction_l1331_133199

theorem three_digit_number_subtraction (c : ℕ) 
  (h1 : c < 10) 
  (h2 : 2 * c < 10) 
  (h3 : c + 3 < 10) : 
  (100 * (c + 3) + 10 * (2 * c) + c) - (100 * c + 10 * (2 * c) + (c + 3)) ≡ 7 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_subtraction_l1331_133199


namespace NUMINAMATH_CALUDE_tea_cups_filled_l1331_133183

theorem tea_cups_filled (total_tea : ℕ) (tea_per_cup : ℕ) (h1 : total_tea = 1050) (h2 : tea_per_cup = 65) :
  (total_tea / tea_per_cup : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_tea_cups_filled_l1331_133183


namespace NUMINAMATH_CALUDE_max_pangs_proof_l1331_133180

/-- The maximum number of pangs that can be purchased given the constraints -/
def max_pangs : ℕ := 9

/-- The price of a pin in dollars -/
def pin_price : ℕ := 3

/-- The price of a pon in dollars -/
def pon_price : ℕ := 4

/-- The price of a pang in dollars -/
def pang_price : ℕ := 9

/-- The total budget in dollars -/
def total_budget : ℕ := 100

/-- The minimum number of pins that must be purchased -/
def min_pins : ℕ := 2

/-- The minimum number of pons that must be purchased -/
def min_pons : ℕ := 3

theorem max_pangs_proof :
  ∃ (pins pons : ℕ),
    pins ≥ min_pins ∧
    pons ≥ min_pons ∧
    pin_price * pins + pon_price * pons + pang_price * max_pangs = total_budget ∧
    ∀ (pangs : ℕ), pangs > max_pangs →
      ∀ (p q : ℕ),
        p ≥ min_pins →
        q ≥ min_pons →
        pin_price * p + pon_price * q + pang_price * pangs ≠ total_budget :=
by sorry

end NUMINAMATH_CALUDE_max_pangs_proof_l1331_133180


namespace NUMINAMATH_CALUDE_number_puzzle_l1331_133103

theorem number_puzzle (N : ℚ) : 
  (N / (4/5)) = ((4/5) * N + 18) → N = 40 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_l1331_133103


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l1331_133134

theorem smallest_fourth_number (a b : ℕ) (h1 : a * 10 + b < 100) (h2 : a * 10 + b > 0) :
  (21 + 34 + 65 + a * 10 + b) = 4 * ((2 + 1 + 3 + 4 + 6 + 5 + a + b)) →
  12 ≤ a * 10 + b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l1331_133134


namespace NUMINAMATH_CALUDE_ladder_geometric_sequence_a10_l1331_133166

/-- A sequence {a_n} is an m-th order ladder geometric sequence if it satisfies
    a_{n+m}^2 = a_n × a_{n+2m} for any positive integers n and m. -/
def is_ladder_geometric (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∀ n : ℕ, (a (n + m))^2 = a n * a (n + 2*m)

theorem ladder_geometric_sequence_a10 (a : ℕ → ℝ) :
  is_ladder_geometric a 3 → a 1 = 1 → a 4 = 2 → a 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ladder_geometric_sequence_a10_l1331_133166


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1331_133185

/-- Proves that given 8 persons, if replacing one person with a new person weighing 93 kg
    increases the average weight by 3.5 kg, then the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 3.5)
  (h3 : new_person_weight = 93)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1331_133185


namespace NUMINAMATH_CALUDE_solution_set_is_correct_l1331_133182

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The solution set of the inequality floor(x)^2 - 5*floor(x) + 6 ≤ 0 -/
def solution_set : Set ℝ :=
  {x : ℝ | (floor x)^2 - 5*(floor x) + 6 ≤ 0}

/-- Theorem stating that the solution set is [2,4) -/
theorem solution_set_is_correct : solution_set = Set.Icc 2 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_correct_l1331_133182


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1331_133144

theorem arithmetic_sequence_length : 
  ∀ (a₁ n d : ℕ) (aₙ : ℕ), 
    a₁ = 3 → 
    d = 3 → 
    aₙ = 144 → 
    aₙ = a₁ + (n - 1) * d → 
    n = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1331_133144


namespace NUMINAMATH_CALUDE_mario_poster_count_l1331_133137

/-- The number of posters Mario made -/
def mario_posters : ℕ := 18

/-- The number of posters Samantha made -/
def samantha_posters : ℕ := mario_posters + 15

/-- The total number of posters made -/
def total_posters : ℕ := 51

theorem mario_poster_count : 
  mario_posters = 18 ∧ 
  samantha_posters = mario_posters + 15 ∧ 
  mario_posters + samantha_posters = total_posters :=
by sorry

end NUMINAMATH_CALUDE_mario_poster_count_l1331_133137


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1331_133151

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 8 < 0 ↔ -4/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1331_133151


namespace NUMINAMATH_CALUDE_ratio_to_percentage_l1331_133168

theorem ratio_to_percentage (x : ℝ) (h : x ≠ 0) :
  (x / 2) / (3 * x / 5) = 3 / 5 → (x / 2) / (3 * x / 5) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_l1331_133168


namespace NUMINAMATH_CALUDE_min_value_theorem_l1331_133195

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 3 / (x + 1) ≥ 2 * Real.sqrt 3 - 1 ∧
  (x + 3 / (x + 1) = 2 * Real.sqrt 3 - 1 ↔ x = Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1331_133195


namespace NUMINAMATH_CALUDE_x_age_is_63_l1331_133177

/-- Given the ages of three people X, Y, and Z, prove that X's current age is 63 years. -/
theorem x_age_is_63 (x y z : ℕ) : 
  (x - 3 = 2 * (y - 3)) →  -- Three years ago, X's age was twice that of Y's age
  (y - 3 = 3 * (z - 3)) →  -- Three years ago, Y's age was three times that of Z's age
  ((x + 7) + (y + 7) + (z + 7) = 130) →  -- Seven years from now, the sum of their ages will be 130 years
  x = 63 := by
sorry

end NUMINAMATH_CALUDE_x_age_is_63_l1331_133177


namespace NUMINAMATH_CALUDE_doudou_mother_age_l1331_133130

/-- Represents the ages of Doudou's family members -/
structure FamilyAges where
  doudou : ℕ
  brother : ℕ
  mother : ℕ
  father : ℕ

/-- The conditions of the problem -/
def problemConditions (ages : FamilyAges) : Prop :=
  ages.brother = ages.doudou + 3 ∧
  ages.mother = ages.father - 2 ∧
  ages.doudou + ages.brother + ages.mother + ages.father - 20 = 59 ∧
  ages.doudou + ages.brother + ages.mother + ages.father + 20 = 97

/-- The theorem to be proved -/
theorem doudou_mother_age (ages : FamilyAges) :
  problemConditions ages → ages.mother = 33 := by
  sorry


end NUMINAMATH_CALUDE_doudou_mother_age_l1331_133130


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1331_133112

theorem min_distance_to_line (x y : ℝ) (h : 2 * x + y + 5 = 0) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x' y' : ℝ), 2 * x' + y' + 5 = 0 → Real.sqrt (x'^2 + y'^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1331_133112


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l1331_133114

theorem ab_positive_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, a * b > 0 ∧ b / a + a / b ≤ 2) ∧
  (∀ a b : ℝ, b / a + a / b > 2 → a * b > 0) :=
sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l1331_133114


namespace NUMINAMATH_CALUDE_train_length_l1331_133102

theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 870)
  (h2 : platform_time = 39)
  (h3 : pole_time = 10) :
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by sorry

end NUMINAMATH_CALUDE_train_length_l1331_133102


namespace NUMINAMATH_CALUDE_regular_square_prism_volume_l1331_133150

theorem regular_square_prism_volume (h : ℝ) (sa : ℝ) (v : ℝ) : 
  h = 2 →
  sa = 12 * Real.pi →
  (∃ (r : ℝ), sa = 4 * Real.pi * r^2 ∧ 
    ∃ (a : ℝ), (2*r)^2 = 2*a^2 + h^2 ∧ 
    v = a^2 * h) →
  v = 8 := by sorry

end NUMINAMATH_CALUDE_regular_square_prism_volume_l1331_133150


namespace NUMINAMATH_CALUDE_expression_simplification_l1331_133161

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 2 + 1) 
  (hb : b = Real.sqrt 2 - 1) : 
  (a^2 - b^2) / a / (a + (2*a*b + b^2) / a) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1331_133161


namespace NUMINAMATH_CALUDE_xiao_dong_jump_record_l1331_133107

/-- Represents the recording of a long jump result -/
def record_jump (standard : ℝ) (jump : ℝ) : ℝ :=
  jump - standard

/-- The standard for the long jump -/
def long_jump_standard : ℝ := 4.00

/-- Xiao Dong's jump distance -/
def xiao_dong_jump : ℝ := 3.85

/-- Theorem stating how Xiao Dong's jump should be recorded -/
theorem xiao_dong_jump_record :
  record_jump long_jump_standard xiao_dong_jump = -0.15 := by
  sorry

end NUMINAMATH_CALUDE_xiao_dong_jump_record_l1331_133107


namespace NUMINAMATH_CALUDE_percentage_b_grades_l1331_133118

def scores : List Nat := [91, 82, 56, 99, 86, 95, 88, 79, 77, 68, 83, 81, 65, 84, 93, 72, 89, 78]

def is_b_grade (score : Nat) : Bool := 85 ≤ score ∧ score ≤ 93

def count_b_grades (scores : List Nat) : Nat :=
  scores.filter is_b_grade |>.length

theorem percentage_b_grades :
  let total_students := scores.length
  let b_grade_students := count_b_grades scores
  (b_grade_students : Rat) / total_students * 100 = 27.78 := by
  sorry

end NUMINAMATH_CALUDE_percentage_b_grades_l1331_133118


namespace NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l1331_133152

/-- If a, b, c form a geometric sequence, then ax^2 + bx + c = 0 has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) (h : b^2 = a*c) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l1331_133152


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1331_133105

theorem cubic_equation_solution (x : ℝ) : x^3 + 64 = 0 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1331_133105


namespace NUMINAMATH_CALUDE_total_results_l1331_133172

theorem total_results (avg_all : ℝ) (avg_first_six : ℝ) (avg_last_six : ℝ) (sixth_result : ℝ) : 
  avg_all = 52 → 
  avg_first_six = 49 → 
  avg_last_six = 52 → 
  sixth_result = 34 → 
  ∃ n : ℕ, n = 11 ∧ n * avg_all = (6 * avg_first_six + 6 * avg_last_six - sixth_result) :=
by
  sorry

#check total_results

end NUMINAMATH_CALUDE_total_results_l1331_133172


namespace NUMINAMATH_CALUDE_ab_value_l1331_133136

/-- The value of a letter in the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | _ => 0

/-- The number value of a word -/
def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

/-- Theorem: The number value of "ab" is 6 -/
theorem ab_value : word_value "ab" = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1331_133136


namespace NUMINAMATH_CALUDE_unique_solution_l1331_133198

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 2

/-- The main theorem stating that the function g(x) = x + 3 is the unique solution -/
theorem unique_solution :
  ∀ g : ℝ → ℝ, SatisfiesFunctionalEquation g → ∀ x : ℝ, g x = x + 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1331_133198


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l1331_133187

theorem ball_distribution_ratio : 
  let total_balls : ℕ := 20
  let num_bins : ℕ := 5
  let config_A : List ℕ := [2, 6, 4, 4, 4]
  let config_B : List ℕ := [4, 4, 4, 4, 4]
  
  let prob_A := (Nat.choose num_bins 1) * (Nat.choose (num_bins - 1) 1) * 
                (Nat.choose total_balls 2) * (Nat.choose (total_balls - 2) 6) * 
                (Nat.choose (total_balls - 2 - 6) 4) * (Nat.choose (total_balls - 2 - 6 - 4) 4) * 
                (Nat.choose (total_balls - 2 - 6 - 4 - 4) 4)
  
  let prob_B := (Nat.choose total_balls 4) * (Nat.choose (total_balls - 4) 4) * 
                (Nat.choose (total_balls - 4 - 4) 4) * (Nat.choose (total_balls - 4 - 4 - 4) 4) * 
                (Nat.choose (total_balls - 4 - 4 - 4 - 4) 4)
  
  prob_A / prob_B = 10 := by
  sorry

#check ball_distribution_ratio

end NUMINAMATH_CALUDE_ball_distribution_ratio_l1331_133187


namespace NUMINAMATH_CALUDE_paths_avoiding_diagonal_l1331_133160

/-- The number of paths on an 8x8 grid from corner to corner, avoiding a diagonal line --/
def num_paths : ℕ := sorry

/-- Binomial coefficient function --/
def binom (n k : ℕ) : ℕ := sorry

theorem paths_avoiding_diagonal :
  num_paths = binom 7 1 * binom 7 1 + (binom 7 3) ^ 2 := by sorry

end NUMINAMATH_CALUDE_paths_avoiding_diagonal_l1331_133160


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1331_133148

/-- A predicate that checks if a number is prime -/
def IsPrime (p : ℕ) : Prop := Nat.Prime p

/-- A predicate that checks if a number is not divisible by 3 or by another number -/
def NotDivisibleBy3OrY (z y : ℕ) : Prop := ¬(z % 3 = 0) ∧ ¬(z % y = 0)

theorem unique_solution_cube_difference_square :
  ∀ x y z : ℕ,
    x > 0 → y > 0 → z > 0 →
    IsPrime y →
    NotDivisibleBy3OrY z y →
    x^3 - y^3 = z^2 →
    x = 8 ∧ y = 7 ∧ z = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1331_133148


namespace NUMINAMATH_CALUDE_forgot_capsules_days_l1331_133190

/-- The number of days in July -/
def july_days : ℕ := 31

/-- The number of days Adam took his capsules in July -/
def days_took_capsules : ℕ := 27

/-- The number of days Adam forgot to take his capsules in July -/
def days_forgot_capsules : ℕ := july_days - days_took_capsules

theorem forgot_capsules_days : days_forgot_capsules = 4 := by
  sorry

end NUMINAMATH_CALUDE_forgot_capsules_days_l1331_133190


namespace NUMINAMATH_CALUDE_fiftieth_rising_number_excludes_one_three_four_l1331_133113

/-- A rising number is a number where each digit is strictly greater than the previous digit. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The set of digits used to construct the rising numbers. -/
def DigitSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The function that generates the nth four-digit rising number from the DigitSet. -/
def NthRisingNumber (n : ℕ) : Finset ℕ := sorry

/-- Theorem stating that the 50th rising number does not contain 1, 3, or 4. -/
theorem fiftieth_rising_number_excludes_one_three_four :
  1 ∉ NthRisingNumber 50 ∧ 3 ∉ NthRisingNumber 50 ∧ 4 ∉ NthRisingNumber 50 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_rising_number_excludes_one_three_four_l1331_133113


namespace NUMINAMATH_CALUDE_sin_90_degrees_l1331_133189

theorem sin_90_degrees : 
  Real.sin (90 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l1331_133189


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1331_133115

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  -- Sides of the right triangle
  de : ℝ
  ef : ℝ
  df : ℝ
  -- The triangle is right-angled
  is_right : de^2 + ef^2 = df^2
  -- Side lengths are positive
  de_pos : de > 0
  ef_pos : ef > 0
  df_pos : df > 0
  -- The square is inscribed in the triangle
  square_inscribed : True

/-- The side length of the inscribed square -/
def square_side_length (t : RightTriangleWithSquare) : ℝ := sorry

/-- Theorem stating that for a right triangle with sides 6, 8, and 10, 
    the inscribed square has side length 120/37 -/
theorem inscribed_square_side_length :
  let t : RightTriangleWithSquare := {
    de := 6,
    ef := 8,
    df := 10,
    is_right := by norm_num,
    de_pos := by norm_num,
    ef_pos := by norm_num,
    df_pos := by norm_num,
    square_inscribed := trivial
  }
  square_side_length t = 120 / 37 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1331_133115


namespace NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1331_133147

theorem sum_of_squares_not_prime (a b c d : ℕ+) (h : a * b = c * d) :
  ¬ Nat.Prime (a^2 + b^2 + c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1331_133147


namespace NUMINAMATH_CALUDE_isosceles_triangles_independent_of_coloring_l1331_133101

/-- The number of isosceles triangles with vertices of the same color in a regular (6n+1)-gon -/
def num_isosceles_triangles (n : ℕ) (K : ℕ) : ℕ :=
  (1/2) * ((6*n+1 - K)*(6*n - K) + K*(K-1) - K*(6*n+1-K))

/-- Theorem stating that the number of isosceles triangles with vertices of the same color
    in a regular (6n+1)-gon is independent of the coloring scheme -/
theorem isosceles_triangles_independent_of_coloring (n : ℕ) (K : ℕ) 
    (h1 : K ≤ 6*n+1) : 
  ∀ (K' : ℕ), K' ≤ 6*n+1 → num_isosceles_triangles n K = num_isosceles_triangles n K' :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_independent_of_coloring_l1331_133101


namespace NUMINAMATH_CALUDE_units_digit_17_310_l1331_133111

/-- The units digit of 7^n for n ≥ 1 -/
def unitsDigit7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | 0 => 1
  | _ => 0  -- This case should never occur

/-- The units digit of 17^n follows the same pattern as 7^n -/
axiom unitsDigit17 (n : ℕ) : n ≥ 1 → unitsDigit7 n = (17^n) % 10

theorem units_digit_17_310 : (17^310) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_310_l1331_133111


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1331_133126

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 3) * x^2 - 4 * x + 2 = 0) ↔ k ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1331_133126


namespace NUMINAMATH_CALUDE_root_product_sum_l1331_133140

theorem root_product_sum (p q r : ℂ) : 
  (6 * p^3 - 9 * p^2 + 16 * p - 12 = 0) →
  (6 * q^3 - 9 * q^2 + 16 * q - 12 = 0) →
  (6 * r^3 - 9 * r^2 + 16 * r - 12 = 0) →
  p * q + p * r + q * r = 8/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1331_133140


namespace NUMINAMATH_CALUDE_total_red_balloons_l1331_133149

/-- The number of red balloons Sara has -/
def sara_red : ℕ := 31

/-- The number of green balloons Sara has -/
def sara_green : ℕ := 15

/-- The number of red balloons Sandy has -/
def sandy_red : ℕ := 24

/-- Theorem stating the total number of red balloons Sara and Sandy have -/
theorem total_red_balloons : sara_red + sandy_red = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_red_balloons_l1331_133149


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_4410_l1331_133125

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors_4410 :
  prime_factorization 4410 = [(2, 1), (3, 2), (5, 1), (7, 2)] →
  count_perfect_square_factors 4410 = 4 := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_4410_l1331_133125


namespace NUMINAMATH_CALUDE_right_triangle_proof_l1331_133192

open Real

theorem right_triangle_proof (A B C : ℝ) (a b c : ℝ) (h1 : b ≠ 1) 
  (h2 : C / A = 2) (h3 : sin B / sin A = 2) (h4 : A + B + C = π) :
  A = π / 6 ∧ B = π / 2 ∧ C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_proof_l1331_133192


namespace NUMINAMATH_CALUDE_ipod_final_price_l1331_133141

/-- Calculates the final price of an item after two discounts and a compound sales tax. -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  price_after_discount2 * (1 + tax_rate)

/-- Theorem stating that the final price of the iPod is approximately $77.08 -/
theorem ipod_final_price :
  ∃ ε > 0, |final_price 128 (7/20) 0.15 0.09 - 77.08| < ε :=
sorry

end NUMINAMATH_CALUDE_ipod_final_price_l1331_133141


namespace NUMINAMATH_CALUDE_inequality_proof_l1331_133145

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a * Real.sqrt (c^2 + 1))) + (1 / (b * Real.sqrt (a^2 + 1))) + (1 / (c * Real.sqrt (b^2 + 1))) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1331_133145


namespace NUMINAMATH_CALUDE_sara_pears_l1331_133131

theorem sara_pears (total_pears sally_pears : ℕ) 
  (h1 : total_pears = 56)
  (h2 : sally_pears = 11) :
  total_pears - sally_pears = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_pears_l1331_133131


namespace NUMINAMATH_CALUDE_all_semifinalists_advanced_no_semifinalists_eliminated_l1331_133188

/-- The number of semifinalists -/
def total_semifinalists : ℕ := 8

/-- The number of medal winners in the final round -/
def medal_winners : ℕ := 3

/-- The number of possible groups of medal winners -/
def possible_groups : ℕ := 56

/-- The number of semifinalists who advanced to the final round -/
def advanced_semifinalists : ℕ := total_semifinalists

theorem all_semifinalists_advanced :
  advanced_semifinalists = total_semifinalists ∧
  Nat.choose advanced_semifinalists medal_winners = possible_groups :=
by sorry

theorem no_semifinalists_eliminated :
  total_semifinalists - advanced_semifinalists = 0 :=
by sorry

end NUMINAMATH_CALUDE_all_semifinalists_advanced_no_semifinalists_eliminated_l1331_133188


namespace NUMINAMATH_CALUDE_raw_material_expenditure_l1331_133110

theorem raw_material_expenditure (x : ℝ) :
  (x ≥ 0) →
  (x ≤ 1) →
  (1 - x - (1/10) * (1 - x) = 0.675) →
  (x = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_raw_material_expenditure_l1331_133110


namespace NUMINAMATH_CALUDE_maze_paths_count_l1331_133186

/-- Represents a maze with specific branching structure -/
structure Maze where
  initial_branches : Nat
  subsequent_branches : Nat
  final_paths : Nat

/-- Calculates the number of unique paths through the maze -/
def count_paths (m : Maze) : Nat :=
  m.initial_branches * m.subsequent_branches.pow m.final_paths

/-- Theorem stating that a maze with given properties has 16 unique paths -/
theorem maze_paths_count :
  ∀ (m : Maze), m.initial_branches = 2 ∧ m.subsequent_branches = 2 ∧ m.final_paths = 3 →
  count_paths m = 16 := by
  sorry

#eval count_paths ⟨2, 2, 3⟩  -- Should output 16

end NUMINAMATH_CALUDE_maze_paths_count_l1331_133186


namespace NUMINAMATH_CALUDE_pascal_interior_sum_8_9_l1331_133184

/-- Sum of interior numbers in row n of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The sum of interior numbers of the 8th and 9th rows of Pascal's Triangle is 380 -/
theorem pascal_interior_sum_8_9 : interior_sum 8 + interior_sum 9 = 380 := by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_8_9_l1331_133184


namespace NUMINAMATH_CALUDE_circle_C_and_point_M_l1331_133191

/-- Circle C passing through points A and B, bisected by a line, with point M satisfying certain conditions -/
structure CircleC where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- Point A on the circle -/
  A : ℝ × ℝ
  /-- Point B on the circle -/
  B : ℝ × ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point M on the circle -/
  M : ℝ × ℝ
  /-- Circle passes through A -/
  passes_through_A : (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2
  /-- Circle passes through B -/
  passes_through_B : (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2
  /-- Line x-3y-4=0 bisects the circle -/
  bisected_by_line : center.1 - 3 * center.2 - 4 = 0
  /-- |MP|/|MQ| = 2 -/
  distance_ratio : ((M.1 - P.1)^2 + (M.2 - P.2)^2) = 4 * ((M.1 - Q.1)^2 + (M.2 - Q.2)^2)

/-- Theorem about the equation of circle C and coordinates of point M -/
theorem circle_C_and_point_M (c : CircleC)
  (h_A : c.A = (0, 2))
  (h_B : c.B = (6, 4))
  (h_P : c.P = (-6, 0))
  (h_Q : c.Q = (6, 0)) :
  (c.center = (4, 0) ∧ c.radius^2 = 20) ∧
  (c.M = (10/3, 4*Real.sqrt 11/3) ∨ c.M = (10/3, -4*Real.sqrt 11/3)) := by
  sorry

end NUMINAMATH_CALUDE_circle_C_and_point_M_l1331_133191


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_fifteen_is_dual_base_palindrome_fifteen_is_smallest_dual_base_palindrome_l1331_133169

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem smallest_dual_base_palindrome : 
  ∀ n : ℕ, n > 10 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n ≥ 15 :=
by sorry

theorem fifteen_is_dual_base_palindrome : 
  isPalindrome 15 2 ∧ isPalindrome 15 4 :=
by sorry

theorem fifteen_is_smallest_dual_base_palindrome : 
  ∀ n : ℕ, n > 10 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_fifteen_is_dual_base_palindrome_fifteen_is_smallest_dual_base_palindrome_l1331_133169


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l1331_133135

theorem like_terms_exponent_difference (a b : ℝ) (m n : ℤ) :
  (∃ k : ℝ, a^(m-2) * b^(n+7) = k * a^4 * b^4) →
  m - n = 9 := by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l1331_133135


namespace NUMINAMATH_CALUDE_parabola_focus_l1331_133176

/-- A parabola is defined by its equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k) -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -2x^2 + 5, its focus is (0, 9/2) -/
theorem parabola_focus (p : Parabola) (f : Focus) : 
  p.a = -2 ∧ p.b = 0 ∧ p.c = 5 → f.h = 0 ∧ f.k = 9/2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1331_133176


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1331_133156

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {2,3,5,6}
def B : Set ℕ := {1,3,4,6,7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1331_133156


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1331_133116

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = Complex.I + z) :
  z = 1/2 - (1/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1331_133116


namespace NUMINAMATH_CALUDE_cubes_after_removing_layer_l1331_133109

/-- The number of smaller cubes in one dimension of the large cube -/
def cube_dimension : ℕ := 10

/-- The total number of smaller cubes in the large cube -/
def total_cubes : ℕ := cube_dimension ^ 3

/-- The number of smaller cubes in one layer -/
def layer_cubes : ℕ := cube_dimension ^ 2

/-- Theorem: Removing one layer from a cube of 10x10x10 smaller cubes leaves 900 cubes -/
theorem cubes_after_removing_layer :
  total_cubes - layer_cubes = 900 := by
  sorry


end NUMINAMATH_CALUDE_cubes_after_removing_layer_l1331_133109


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1331_133132

theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : total_students = 416) (h2 : girls = 160) :
  (total_students - girls) / girls = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1331_133132


namespace NUMINAMATH_CALUDE_cat_kittens_count_l1331_133174

/-- The number of kittens born to a cat, given specific weight conditions -/
def number_of_kittens (weight_two_lightest weight_four_heaviest total_weight : ℕ) : ℕ :=
  2 + 4 + (total_weight - weight_two_lightest - weight_four_heaviest) / ((weight_four_heaviest / 4 + weight_two_lightest / 2) / 2)

/-- Theorem stating that under the given conditions, the cat gave birth to 11 kittens -/
theorem cat_kittens_count :
  number_of_kittens 80 200 500 = 11 :=
by
  sorry

#eval number_of_kittens 80 200 500

end NUMINAMATH_CALUDE_cat_kittens_count_l1331_133174


namespace NUMINAMATH_CALUDE_median_moons_theorem_l1331_133157

/-- Represents the two categories of planets -/
inductive PlanetCategory
| Rocky
| GasGiant

/-- Represents a planet with its category and number of moons -/
structure Planet where
  name : String
  category : PlanetCategory
  moons : ℕ

/-- The list of all planets with their data -/
def planets : List Planet := [
  ⟨"Mercury", PlanetCategory.Rocky, 0⟩,
  ⟨"Venus", PlanetCategory.Rocky, 0⟩,
  ⟨"Earth", PlanetCategory.Rocky, 1⟩,
  ⟨"Mars", PlanetCategory.Rocky, 3⟩,
  ⟨"Jupiter", PlanetCategory.GasGiant, 20⟩,
  ⟨"Saturn", PlanetCategory.GasGiant, 25⟩,
  ⟨"Uranus", PlanetCategory.GasGiant, 17⟩,
  ⟨"Neptune", PlanetCategory.GasGiant, 3⟩,
  ⟨"Pluto", PlanetCategory.GasGiant, 8⟩
]

/-- Calculate the median number of moons for a given category -/
def medianMoons (category : PlanetCategory) : ℚ := sorry

/-- The theorem stating the median number of moons for each category -/
theorem median_moons_theorem :
  medianMoons PlanetCategory.Rocky = 1/2 ∧
  medianMoons PlanetCategory.GasGiant = 17 := by sorry

end NUMINAMATH_CALUDE_median_moons_theorem_l1331_133157


namespace NUMINAMATH_CALUDE_expected_allergies_in_sample_l1331_133159

/-- The probability that an American suffers from allergies -/
def allergy_probability : ℚ := 1 / 5

/-- The size of the random sample -/
def sample_size : ℕ := 250

/-- The expected number of Americans with allergies in the sample -/
def expected_allergies : ℚ := allergy_probability * sample_size

theorem expected_allergies_in_sample :
  expected_allergies = 50 := by sorry

end NUMINAMATH_CALUDE_expected_allergies_in_sample_l1331_133159


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_product_l1331_133164

theorem simplify_inverse_sum_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_product_l1331_133164


namespace NUMINAMATH_CALUDE_class_size_proof_l1331_133124

theorem class_size_proof (S : ℕ) : 
  S / 2 + S / 3 + 4 = S → S = 24 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l1331_133124


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1331_133163

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  (n = 3^15 + 11^13) → (Nat.minFac n = 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1331_133163


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l1331_133165

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l1331_133165
