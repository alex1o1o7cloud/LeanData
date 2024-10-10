import Mathlib

namespace largest_number_l3951_395146

def numbers : List ℝ := [0.988, 0.9808, 0.989, 0.9809, 0.998]

theorem largest_number (n : ℝ) (hn : n ∈ numbers) : n ≤ 0.998 :=
by sorry

end largest_number_l3951_395146


namespace gumballs_last_42_days_l3951_395139

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on day 1 -/
def earrings_day1 : ℕ := 3

/-- The number of pairs of earrings Kim brings on day 2 -/
def earrings_day2 : ℕ := 2 * earrings_day1

/-- The number of pairs of earrings Kim brings on day 3 -/
def earrings_day3 : ℕ := earrings_day2 - 1

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The total number of gumballs Kim receives -/
def total_gumballs : ℕ := 
  gumballs_per_pair * (earrings_day1 + earrings_day2 + earrings_day3)

/-- The number of days the gumballs will last -/
def days_gumballs_last : ℕ := total_gumballs / gumballs_eaten_per_day

theorem gumballs_last_42_days : days_gumballs_last = 42 := by
  sorry

end gumballs_last_42_days_l3951_395139


namespace three_numbers_sum_l3951_395128

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a - 15 → 
  (a + b + c) / 3 = c + 10 → 
  a + b + c = 45 := by
sorry

end three_numbers_sum_l3951_395128


namespace typist_salary_problem_l3951_395156

/-- Given a salary that is first increased by 10% and then decreased by 5%,
    resulting in Rs. 2090, prove that the original salary was Rs. 2000. -/
theorem typist_salary_problem (S : ℝ) : 
  S * 1.1 * 0.95 = 2090 → S = 2000 := by
  sorry

#check typist_salary_problem

end typist_salary_problem_l3951_395156


namespace percentage_problem_l3951_395178

theorem percentage_problem (P : ℝ) : 
  (P ≥ 0 ∧ P ≤ 100) → 
  (P / 100) * 3200 = (20 / 100) * 650 + 190 → 
  P = 10 := by sorry

end percentage_problem_l3951_395178


namespace expected_adjacent_pairs_standard_deck_l3951_395164

/-- A standard deck of cards. -/
structure Deck :=
  (total : Nat)
  (black : Nat)
  (red : Nat)
  (h_total : total = 52)
  (h_half : black = red)
  (h_sum : black + red = total)

/-- The expected number of adjacent pairs with one black and one red card
    in a circular arrangement of cards from a standard deck. -/
def expectedAdjacentPairs (d : Deck) : Rat :=
  (d.total : Rat) * (d.black : Rat) * (d.red : Rat) / ((d.total - 1) : Rat)

theorem expected_adjacent_pairs_standard_deck :
  ∃ (d : Deck), expectedAdjacentPairs d = 1352 / 51 := by
  sorry

end expected_adjacent_pairs_standard_deck_l3951_395164


namespace debbys_museum_pictures_l3951_395163

theorem debbys_museum_pictures 
  (zoo_pictures : ℕ) 
  (deleted_pictures : ℕ) 
  (remaining_pictures : ℕ) 
  (h1 : zoo_pictures = 24)
  (h2 : deleted_pictures = 14)
  (h3 : remaining_pictures = 22)
  (h4 : remaining_pictures = zoo_pictures + museum_pictures - deleted_pictures) :
  museum_pictures = 12 := by
  sorry

#check debbys_museum_pictures

end debbys_museum_pictures_l3951_395163


namespace original_denominator_proof_l3951_395102

theorem original_denominator_proof (d : ℚ) : 
  (5 / d : ℚ) ≠ 0 → -- Ensure the original fraction is well-defined
  ((5 - 3) / (d + 4) : ℚ) = 1 / 3 →
  d = 2 := by
sorry

end original_denominator_proof_l3951_395102


namespace condition_sufficient_not_necessary_l3951_395147

-- Define the condition for m and n
def condition (m n : ℝ) : Prop := m < 0 ∧ 0 < n

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m n : ℝ) : Prop := 
  ∃ (x y : ℝ), n * x^2 + m * y^2 = 1 ∧ (m < 0 ∧ n > 0) ∨ (m > 0 ∧ n < 0)

-- State the theorem
theorem condition_sufficient_not_necessary (m n : ℝ) :
  (condition m n → is_hyperbola m n) ∧ 
  ¬(is_hyperbola m n → condition m n) :=
sorry

end condition_sufficient_not_necessary_l3951_395147


namespace problem_1_problem_2_l3951_395115

-- Problem 1
theorem problem_1 : (-1)^2023 + 2 * Real.cos (π / 4) - |Real.sqrt 2 - 2| - (1 / 2)⁻¹ = 2 * Real.sqrt 2 - 5 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x ≠ 0) : 
  (1 - 1 / (x + 1)) / ((x^2) / (x^2 + 2*x + 1)) = (x + 1) / x := by sorry

end problem_1_problem_2_l3951_395115


namespace boys_age_problem_l3951_395125

theorem boys_age_problem (x : ℕ) : x + 4 = 2 * (x - 6) → x = 16 := by
  sorry

end boys_age_problem_l3951_395125


namespace reflection_result_l3951_395106

/-- Reflect a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The initial point C -/
def C : ℝ × ℝ := (-1, 4)

theorem reflection_result :
  (reflect_x (reflect_y C)) = (1, -4) := by
  sorry

end reflection_result_l3951_395106


namespace special_triangle_properties_l3951_395101

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  angle_sum : A + B + C = π
  side_angle_relation : (Real.cos B) / (Real.cos C) = b / (2 * a - c)
  b_value : b = Real.sqrt 7
  area : (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2

/-- Main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) : t.B = π/3 ∧ t.a + t.c = 5 := by
  sorry

end special_triangle_properties_l3951_395101


namespace board_cutting_l3951_395132

theorem board_cutting (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) : 
  total_length = 120 →
  shorter_piece + (2 * shorter_piece + difference) = total_length →
  difference = 15 →
  shorter_piece = 35 := by
sorry

end board_cutting_l3951_395132


namespace johans_house_rooms_l3951_395154

theorem johans_house_rooms (walls_per_room : ℕ) (green_ratio : ℚ) (purple_walls : ℕ) : 
  walls_per_room = 8 →
  green_ratio = 3/5 →
  purple_walls = 32 →
  ∃ (total_rooms : ℕ), total_rooms = 10 ∧ 
    (purple_walls : ℚ) / walls_per_room = (1 - green_ratio) * total_rooms :=
by sorry

end johans_house_rooms_l3951_395154


namespace triangle_angle_theorem_l3951_395141

theorem triangle_angle_theorem (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) (h5 : a = b) :
  let C := Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2)))
  C = Real.arccos (1/4) :=
sorry

end triangle_angle_theorem_l3951_395141


namespace tournament_has_king_l3951_395185

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Lose

/-- A tournament with m teams -/
structure Tournament (m : ℕ) where
  /-- The result of a match between two teams -/
  result : Fin m → Fin m → MatchResult
  /-- Each pair of teams has competed exactly once -/
  competed_once : ∀ i j : Fin m, i ≠ j → (result i j = MatchResult.Win ∧ result j i = MatchResult.Lose) ∨
                                        (result i j = MatchResult.Lose ∧ result j i = MatchResult.Win)

/-- Definition of a king in the tournament -/
def is_king (t : Tournament m) (x : Fin m) : Prop :=
  ∀ y : Fin m, y ≠ x → 
    (t.result x y = MatchResult.Win) ∨ 
    (∃ z : Fin m, t.result x z = MatchResult.Win ∧ t.result z y = MatchResult.Win)

/-- Theorem: Every tournament has a king -/
theorem tournament_has_king (m : ℕ) (t : Tournament m) : ∃ x : Fin m, is_king t x := by
  sorry

end tournament_has_king_l3951_395185


namespace sum_of_squares_power_of_three_l3951_395137

theorem sum_of_squares_power_of_three (n : ℕ) :
  ∃ x y z : ℤ, (Nat.gcd (Nat.gcd x.natAbs y.natAbs) z.natAbs = 1) ∧
  (x^2 + y^2 + z^2 = 3^(2^n)) := by
sorry

end sum_of_squares_power_of_three_l3951_395137


namespace candy_probability_theorem_l3951_395191

/-- Represents a packet of candies -/
structure CandyPacket where
  blue : ℕ
  total : ℕ
  h_total_pos : total > 0
  h_blue_le_total : blue ≤ total

/-- Represents a box containing two packets of candies -/
structure CandyBox where
  packet1 : CandyPacket
  packet2 : CandyPacket

/-- The probability of drawing a blue candy from the box -/
def blue_probability (box : CandyBox) : ℚ :=
  (box.packet1.blue + box.packet2.blue : ℚ) / (box.packet1.total + box.packet2.total)

theorem candy_probability_theorem :
  (∃ box : CandyBox, blue_probability box = 5/13) ∧
  (∃ box : CandyBox, blue_probability box = 7/18) ∧
  (∀ box : CandyBox, blue_probability box ≠ 17/40) :=
sorry

end candy_probability_theorem_l3951_395191


namespace blanket_average_price_l3951_395109

theorem blanket_average_price : 
  let blanket_group1 := (3, 100)
  let blanket_group2 := (5, 150)
  let blanket_group3 := (2, 570)
  let total_blankets := blanket_group1.1 + blanket_group2.1 + blanket_group3.1
  let total_cost := blanket_group1.1 * blanket_group1.2 + blanket_group2.1 * blanket_group2.2 + blanket_group3.1 * blanket_group3.2
  total_cost / total_blankets = 219 := by
sorry

end blanket_average_price_l3951_395109


namespace jake_earnings_l3951_395173

def calculate_earnings (viper_count cobra_count python_count : ℕ)
                       (viper_eggs cobra_eggs python_eggs : ℕ)
                       (viper_price cobra_price python_price : ℚ)
                       (viper_discount cobra_discount : ℚ) : ℚ :=
  let viper_babies := viper_count * viper_eggs
  let cobra_babies := cobra_count * cobra_eggs
  let python_babies := python_count * python_eggs
  let viper_earnings := viper_babies * (viper_price * (1 - viper_discount))
  let cobra_earnings := cobra_babies * (cobra_price * (1 - cobra_discount))
  let python_earnings := python_babies * python_price
  viper_earnings + cobra_earnings + python_earnings

theorem jake_earnings :
  calculate_earnings 2 3 1 3 2 4 300 250 450 (1/10) (1/20) = 4845 := by
  sorry

end jake_earnings_l3951_395173


namespace committee_selection_count_l3951_395134

theorem committee_selection_count : Nat.choose 12 5 = 792 := by
  sorry

end committee_selection_count_l3951_395134


namespace consecutive_integers_sum_46_l3951_395149

theorem consecutive_integers_sum_46 :
  ∃ (x : ℕ), x > 0 ∧ x + (x + 1) + (x + 2) + (x + 3) = 46 := by
  sorry

end consecutive_integers_sum_46_l3951_395149


namespace half_fourth_of_twelve_y_plus_three_l3951_395193

theorem half_fourth_of_twelve_y_plus_three (y : ℝ) :
  (1/2) * (1/4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 := by
  sorry

end half_fourth_of_twelve_y_plus_three_l3951_395193


namespace fraction_simplification_l3951_395169

theorem fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end fraction_simplification_l3951_395169


namespace range_of_a_l3951_395180

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, 2 * x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) → (a > 2 ∨ (-2 < a ∧ a < 1)) := by
  sorry

end range_of_a_l3951_395180


namespace swimmer_distance_l3951_395144

/-- Proves that the distance swam against the current is 6 km given the specified conditions -/
theorem swimmer_distance (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) 
  (h1 : swimmer_speed = 4)
  (h2 : current_speed = 1)
  (h3 : time = 2) :
  (swimmer_speed - current_speed) * time = 6 := by
  sorry

end swimmer_distance_l3951_395144


namespace ferris_wheel_small_seats_l3951_395135

/-- The number of small seats on the Ferris wheel -/
def small_seats : ℕ := 2

/-- The number of people each small seat can hold -/
def people_per_small_seat : ℕ := 14

/-- The total number of people who can ride on small seats -/
def total_people_on_small_seats : ℕ := small_seats * people_per_small_seat

theorem ferris_wheel_small_seats :
  total_people_on_small_seats = 28 := by sorry

end ferris_wheel_small_seats_l3951_395135


namespace jade_savings_l3951_395179

/-- Calculates Jade's monthly savings given her earnings and spending patterns. -/
theorem jade_savings (monthly_earnings : ℝ) (living_expenses_ratio : ℝ) (insurance_ratio : ℝ) :
  monthly_earnings = 1600 →
  living_expenses_ratio = 0.75 →
  insurance_ratio = 1/5 →
  monthly_earnings * (1 - living_expenses_ratio - insurance_ratio) = 80 :=
by sorry

end jade_savings_l3951_395179


namespace system_solution_unique_l3951_395165

theorem system_solution_unique :
  ∃! (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = -1 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l3951_395165


namespace gcd_factorial_problem_l3951_395119

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 12) / (Nat.factorial 4)) = 240 := by
  sorry

end gcd_factorial_problem_l3951_395119


namespace tangent_line_at_one_l3951_395110

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2

-- State the theorem
theorem tangent_line_at_one (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 * f (2 - x) - x^2 + 8*x - 8) : 
  ∃ m b, ∀ x, (x - 1) * (f 1) + m * (x - 1) = m * x + b := by
  sorry

end tangent_line_at_one_l3951_395110


namespace tan_150_degrees_l3951_395112

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end tan_150_degrees_l3951_395112


namespace sum_of_two_and_repeating_third_l3951_395138

-- Define the repeating decimal 0.3333...
def repeating_third : ℚ := 1 / 3

-- Theorem statement
theorem sum_of_two_and_repeating_third :
  2 + repeating_third = 7 / 3 := by sorry

end sum_of_two_and_repeating_third_l3951_395138


namespace example_is_quadratic_l3951_395166

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 3x + 1 = 0 is a quadratic equation -/
theorem example_is_quadratic : is_quadratic_equation (fun x ↦ x^2 - 3*x + 1) := by
  sorry

end example_is_quadratic_l3951_395166


namespace tangent_line_slope_relation_l3951_395113

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_slope_relation (a b : ℝ) :
  a^2 + b = 0 →
  ∃ (m n : ℝ),
    let k1 := 3*m^2 + 2*a*m + b
    let k2 := 3*n^2 + 2*a*n + b
    k2 = 4*k1 →
    a^2 = 3*b :=
sorry

end tangent_line_slope_relation_l3951_395113


namespace sum_of_fractions_equals_seven_l3951_395150

theorem sum_of_fractions_equals_seven : 
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
  1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
  1 / (Real.sqrt 12 - 3) = 7 := by
  sorry

end sum_of_fractions_equals_seven_l3951_395150


namespace fraction_value_l3951_395199

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 5) (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end fraction_value_l3951_395199


namespace opposite_of_negative_2022_l3951_395108

theorem opposite_of_negative_2022 : -((-2022 : ℤ)) = 2022 := by sorry

end opposite_of_negative_2022_l3951_395108


namespace five_people_arrangement_l3951_395190

/-- The number of arrangements of n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements of n people in a row where two specific people are next to each other -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of arrangements of n people in a row where two specific people are not next to each other -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem five_people_arrangement :
  nonAdjacentArrangements 5 = 72 :=
by sorry

end five_people_arrangement_l3951_395190


namespace microscope_magnification_factor_l3951_395118

/-- The magnification factor of an electron microscope, given the magnified image diameter and actual tissue diameter. -/
theorem microscope_magnification_factor 
  (magnified_diameter : ℝ) 
  (actual_diameter : ℝ) 
  (h1 : magnified_diameter = 2) 
  (h2 : actual_diameter = 0.002) : 
  magnified_diameter / actual_diameter = 1000 := by
sorry

end microscope_magnification_factor_l3951_395118


namespace ones_digit_73_pow_355_l3951_395105

theorem ones_digit_73_pow_355 : (73^355) % 10 = 7 := by
  sorry

end ones_digit_73_pow_355_l3951_395105


namespace last_three_digits_is_419_l3951_395114

/-- A function that generates the nth digit in the list of increasing positive integers starting with 2 -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1998th, 1999th, and 2000th digits -/
def lastThreeDigits : ℕ := 
  100 * (nthDigit 1998) + 10 * (nthDigit 1999) + (nthDigit 2000)

/-- Theorem stating that the last three digits form the number 419 -/
theorem last_three_digits_is_419 : lastThreeDigits = 419 := by sorry

end last_three_digits_is_419_l3951_395114


namespace roots_sum_of_squares_l3951_395160

theorem roots_sum_of_squares (m n : ℝ) : 
  (m^2 - 5*m + 3 = 0) → (n^2 - 5*n + 3 = 0) → (m^2 + n^2 = 19) := by
  sorry

end roots_sum_of_squares_l3951_395160


namespace square_area_from_diagonal_l3951_395153

theorem square_area_from_diagonal (d : ℝ) (h : d = 2) : 
  (d^2 / 2) = 2 := by sorry

end square_area_from_diagonal_l3951_395153


namespace circle_trajectory_l3951_395104

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The fixed point A -/
def A : Point := ⟨2, 0⟩

/-- Checks if a circle passes through a given point -/
def passesThrough (c : Circle) (p : Point) : Prop :=
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

/-- Checks if a circle intersects the y-axis forming a chord of length 4 -/
def intersectsYAxis (c : Circle) : Prop :=
  c.radius^2 = c.center.x^2 + 4

/-- The trajectory of the center of the moving circle -/
def trajectory (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Theorem: The trajectory of the center of a circle that passes through (2,0) 
    and intersects the y-axis forming a chord of length 4 is y² = 4x -/
theorem circle_trajectory : 
  ∀ (c : Circle), 
    passesThrough c A → 
    intersectsYAxis c → 
    trajectory c.center :=
sorry

end circle_trajectory_l3951_395104


namespace euro_calculation_l3951_395188

/-- The € operation as defined in the problem -/
def euro (x y z : ℕ) : ℕ := (2 * x * y + y^2) * z^3

/-- The statement to be proven -/
theorem euro_calculation : euro 7 (euro 4 5 3) 2 = 24844760 := by
  sorry

end euro_calculation_l3951_395188


namespace total_cement_is_15_1_l3951_395177

/-- The amount of cement used for Lexi's street in tons -/
def lexiStreetCement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tessStreetCement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def totalCement : ℝ := lexiStreetCement + tessStreetCement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_1 : totalCement = 15.1 := by sorry

end total_cement_is_15_1_l3951_395177


namespace teachers_count_correct_teachers_count_l3951_395103

theorem teachers_count (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ) : ℕ :=
  let total_tickets := total_cost / ticket_cost
  let num_teachers := total_tickets - num_students
  num_teachers

theorem correct_teachers_count :
  teachers_count 20 5 115 = 3 := by
  sorry

end teachers_count_correct_teachers_count_l3951_395103


namespace average_monthly_balance_l3951_395159

def monthly_balances : List ℝ := [100, 200, 250, 250, 150, 100]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 175 := by
  sorry

end average_monthly_balance_l3951_395159


namespace square_roots_problem_l3951_395198

theorem square_roots_problem (m : ℝ) :
  (2*m - 4)^2 = (3*m - 1)^2 → (2*m - 4)^2 = 4 :=
by sorry

end square_roots_problem_l3951_395198


namespace ratio_x_to_y_l3951_395131

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (17 * x - 3 * y) = 5 / 7) :
  x / y = -20 := by
  sorry

end ratio_x_to_y_l3951_395131


namespace geometric_sequence_third_term_l3951_395120

/-- Given a geometric sequence with common ratio greater than 1,
    if the difference between the 5th and 1st term is 15,
    and the difference between the 4th and 2nd term is 6,
    then the 3rd term is 4. -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- The sequence
  (q : ℝ)      -- Common ratio
  (h_geom : ∀ n, a (n + 1) = a n * q)  -- Geometric sequence property
  (h_q : q > 1)  -- Common ratio > 1
  (h_diff1 : a 5 - a 1 = 15)  -- Given condition
  (h_diff2 : a 4 - a 2 = 6)   -- Given condition
  : a 3 = 4 := by
  sorry

end geometric_sequence_third_term_l3951_395120


namespace ramanujan_number_l3951_395170

theorem ramanujan_number (r h : ℂ) : 
  r * h = 40 + 24 * I ∧ h = 7 + I → r = 28/5 + 64/25 * I :=
by sorry

end ramanujan_number_l3951_395170


namespace train_length_problem_l3951_395121

theorem train_length_problem (speed1 speed2 : ℝ) (pass_time : ℝ) (h1 : speed1 = 55) (h2 : speed2 = 50) (h3 : pass_time = 11.657142857142858) :
  let relative_speed := (speed1 + speed2) * (5 / 18)
  let total_distance := relative_speed * pass_time
  let train_length := total_distance / 2
  train_length = 170 := by sorry

end train_length_problem_l3951_395121


namespace bus_speed_increase_l3951_395133

/-- The speed increase of a bus per hour, given initial speed and total distance traveled. -/
theorem bus_speed_increase 
  (S₀ : ℝ) 
  (total_distance : ℝ) 
  (x : ℝ) 
  (h1 : S₀ = 35) 
  (h2 : total_distance = 552) 
  (h3 : total_distance = S₀ * 12 + x * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11)) : 
  x = 2 := by
sorry

end bus_speed_increase_l3951_395133


namespace cheap_handcuff_time_is_6_l3951_395157

/-- The time it takes to pick the lock on a cheap pair of handcuffs -/
def cheap_handcuff_time : ℝ := 6

/-- The time it takes to pick the lock on an expensive pair of handcuffs -/
def expensive_handcuff_time : ℝ := 8

/-- The number of friends to rescue -/
def num_friends : ℕ := 3

/-- The total time it takes to free all friends -/
def total_rescue_time : ℝ := 42

/-- Theorem stating that the time to pick a cheap handcuff lock is 6 minutes -/
theorem cheap_handcuff_time_is_6 :
  cheap_handcuff_time = 6 ∧
  num_friends * (cheap_handcuff_time + expensive_handcuff_time) = total_rescue_time :=
by sorry

end cheap_handcuff_time_is_6_l3951_395157


namespace smallest_solution_for_floor_equation_l3951_395184

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^3⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 18 ∧
  (∀ (y : ℝ), y > 0 → (⌊y^3⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 18 → y ≥ x) ∧
  x = 369 / 19 := by
  sorry

end smallest_solution_for_floor_equation_l3951_395184


namespace divisibility_property_l3951_395123

theorem divisibility_property (n : ℕ) : 
  ∃ (x y : ℤ), (x^2 + y^2 - 2018) % n = 0 := by
sorry

end divisibility_property_l3951_395123


namespace area_of_intersection_region_l3951_395148

noncomputable def f₀ (x : ℝ) : ℝ := |x|

noncomputable def f₁ (x : ℝ) : ℝ := |f₀ x - 1|

noncomputable def f₂ (x : ℝ) : ℝ := |f₁ x - 2|

theorem area_of_intersection_region (f₀ f₁ f₂ : ℝ → ℝ) :
  (f₀ = fun x ↦ |x|) →
  (f₁ = fun x ↦ |f₀ x - 1|) →
  (f₂ = fun x ↦ |f₁ x - 2|) →
  (∫ x in (-3)..(3), min (f₂ x) 0) = -7 :=
by sorry

end area_of_intersection_region_l3951_395148


namespace max_individual_score_l3951_395100

theorem max_individual_score (n : ℕ) (total : ℕ) (min_score : ℕ) 
  (h1 : n = 12)
  (h2 : total = 100)
  (h3 : min_score = 7)
  (h4 : ∀ p : ℕ, p ≤ n → min_score ≤ p) :
  ∃ max_score : ℕ, 
    (∀ p : ℕ, p ≤ n → p ≤ max_score) ∧ 
    (∃ player : ℕ, player ≤ n ∧ player = max_score) ∧
    max_score = 23 :=
sorry

end max_individual_score_l3951_395100


namespace perpendicular_diagonals_imply_cyclic_projections_l3951_395145

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the concept of perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let AC := (q.C.1 - q.A.1, q.C.2 - q.A.2)
  let BD := (q.D.1 - q.B.1, q.D.2 - q.B.2)
  AC.1 * BD.1 + AC.2 * BD.2 = 0

-- Define the projection of a point onto a line segment
def project_point (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the intersection point of diagonals
def diagonal_intersection (q : Quadrilateral) : ℝ × ℝ :=
  sorry

-- Define a cyclic quadrilateral
def is_cyclic (A B C D : ℝ × ℝ) : Prop :=
  sorry

-- Main theorem
theorem perpendicular_diagonals_imply_cyclic_projections (q : Quadrilateral) :
  has_perpendicular_diagonals q →
  let I := diagonal_intersection q
  let A1 := project_point I q.A q.B
  let B1 := project_point I q.B q.C
  let C1 := project_point I q.C q.D
  let D1 := project_point I q.D q.A
  is_cyclic A1 B1 C1 D1 :=
by
  sorry

end perpendicular_diagonals_imply_cyclic_projections_l3951_395145


namespace jackson_meat_problem_l3951_395162

theorem jackson_meat_problem (M : ℝ) : 
  M > 0 → 
  M - (1/4 * M) - 3 = 12 → 
  M = 20 :=
by sorry

end jackson_meat_problem_l3951_395162


namespace sum_remainder_l3951_395124

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 := by
  sorry

end sum_remainder_l3951_395124


namespace power_of_three_equation_l3951_395175

theorem power_of_three_equation (k : ℤ) : 
  3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 := by
  sorry

end power_of_three_equation_l3951_395175


namespace alcohol_concentration_after_addition_l3951_395189

/-- Proves that adding 5.5 liters of alcohol and 4.5 liters of water to a 40-liter solution
    with 5% alcohol concentration results in a 15% alcohol solution. -/
theorem alcohol_concentration_after_addition :
  let initial_volume : ℝ := 40
  let initial_concentration : ℝ := 0.05
  let added_alcohol : ℝ := 5.5
  let added_water : ℝ := 4.5
  let final_concentration : ℝ := 0.15
  let final_volume : ℝ := initial_volume + added_alcohol + added_water
  initial_volume * initial_concentration + added_alcohol =
    final_volume * final_concentration :=
by sorry

end alcohol_concentration_after_addition_l3951_395189


namespace particular_number_proof_l3951_395192

theorem particular_number_proof : ∃! x : ℚ, ((x + 2 - 6) * 3) / 4 = 3 := by
  sorry

end particular_number_proof_l3951_395192


namespace sum_of_three_numbers_l3951_395155

theorem sum_of_three_numbers : 3/8 + 0.125 + 9.51 = 10.01 := by
  sorry

end sum_of_three_numbers_l3951_395155


namespace even_function_implies_m_equals_one_l3951_395158

def f (m : ℝ) (x : ℝ) : ℝ := x^4 + (m-1)*x + 1

theorem even_function_implies_m_equals_one (m : ℝ) :
  (∀ x, f m x = f m (-x)) → m = 1 := by
  sorry

end even_function_implies_m_equals_one_l3951_395158


namespace arcsin_neg_half_equals_neg_pi_sixth_l3951_395122

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end arcsin_neg_half_equals_neg_pi_sixth_l3951_395122


namespace binary_1010101_equals_octal_125_l3951_395197

/-- Converts a binary number represented as a list of bits to a natural number -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

/-- The binary representation of 1010101₂ -/
def binary_1010101 : List Bool := [true, false, true, false, true, false, true]

/-- The octal representation of 125₈ -/
def octal_125 : List ℕ := [5, 2, 1]

theorem binary_1010101_equals_octal_125 :
  natural_to_octal (binary_to_natural binary_1010101) = octal_125 := by
  sorry

end binary_1010101_equals_octal_125_l3951_395197


namespace area_between_curves_l3951_395117

-- Define the functions f and g
def f (x : ℝ) : ℝ := -2 * x^2 + 7 * x - 6
def g (x : ℝ) : ℝ := -x

-- Define the theorem
theorem area_between_curves : 
  ∃ (a b : ℝ), a < b ∧ 
  (∫ (x : ℝ) in a..b, f x - g x) = 8/3 := by
sorry

end area_between_curves_l3951_395117


namespace sum_of_remainders_consecutive_integers_l3951_395116

theorem sum_of_remainders_consecutive_integers (n : ℕ) : 
  (n % 4) + ((n + 1) % 4) + ((n + 2) % 4) + ((n + 3) % 4) = 6 := by
  sorry

end sum_of_remainders_consecutive_integers_l3951_395116


namespace isosceles_triangle_not_unique_l3951_395176

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  /-- The base angle of the isosceles triangle -/
  base_angle : ℝ
  /-- The altitude of the isosceles triangle -/
  altitude : ℝ
  /-- The side length of the isosceles triangle -/
  side_length : ℝ
  /-- Constraint that the base angle is between 0 and π/2 -/
  angle_constraint : 0 < base_angle ∧ base_angle < π / 2
  /-- Constraint that the altitude is positive -/
  altitude_positive : altitude > 0
  /-- Constraint that the side length is positive -/
  side_positive : side_length > 0

/-- Theorem stating that there exist multiple non-congruent isosceles triangles
    with the same base angle and altitude -/
theorem isosceles_triangle_not_unique :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1.base_angle = t2.base_angle ∧
    t1.altitude = t2.altitude ∧
    t1.side_length ≠ t2.side_length :=
  sorry

end isosceles_triangle_not_unique_l3951_395176


namespace ellipse_equation_l3951_395129

/-- An ellipse with focal length 2 passing through (-√5, 0) has a standard equation of either x²/5 + y²/4 = 1 or y²/6 + x²/5 = 1 -/
theorem ellipse_equation (f : ℝ) (P : ℝ × ℝ) : 
  f = 2 → P = (-Real.sqrt 5, 0) → 
  (∃ (x y : ℝ), x^2/5 + y^2/4 = 1) ∨ (∃ (x y : ℝ), y^2/6 + x^2/5 = 1) :=
by sorry

end ellipse_equation_l3951_395129


namespace wage_increase_calculation_l3951_395127

theorem wage_increase_calculation (W H W' H' : ℝ) : 
  W > 0 → H > 0 → W' > W → -- Initial conditions
  H' = H * (1 - 0.20) → -- 20% reduction in hours
  W * H = W' * H' → -- Total weekly income remains the same
  (W' - W) / W = 0.25 := by -- The wage increase is 25%
  sorry

end wage_increase_calculation_l3951_395127


namespace unique_solution_quadratic_l3951_395183

theorem unique_solution_quadratic (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x - a = 0) ↔ a = 0 := by
sorry

end unique_solution_quadratic_l3951_395183


namespace girls_in_circle_l3951_395126

/-- The number of girls in a circular arrangement where one girl is
    both the fifth to the left and the eighth to the right of another girl. -/
def num_girls_in_circle : ℕ := 13

/-- Proposition: In a circular arrangement of girls, if one girl is both
    the fifth to the left and the eighth to the right of another girl,
    then the total number of girls in the circle is 13. -/
theorem girls_in_circle :
  ∀ (n : ℕ), n > 0 →
  (∃ (a b : ℕ), a < n ∧ b < n ∧ a ≠ b ∧
   ((a + 5) % n = b) ∧ ((b + 8) % n = a)) →
  n = num_girls_in_circle :=
sorry

end girls_in_circle_l3951_395126


namespace union_of_sets_l3951_395140

theorem union_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | -2 < x ∧ x < 0} →
  B = {x : ℝ | -1 < x ∧ x < 1} →
  A ∪ B = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end union_of_sets_l3951_395140


namespace intersection_of_three_lines_l3951_395174

/-- Given three lines in a plane, if they intersect at the same point, 
    we can determine the value of a parameter in one of the lines. -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃ (x y : ℝ), (a*x + 2*y + 6 = 0) ∧ (x + y - 4 = 0) ∧ (2*x - y + 1 = 0)) →
  a = -12 := by
  sorry

end intersection_of_three_lines_l3951_395174


namespace relationship_2x_3sinx_l3951_395195

theorem relationship_2x_3sinx :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < θ → 2 * x < 3 * Real.sin x) ∧
  (2 * θ = 3 * Real.sin θ) ∧
  (∀ x : ℝ, θ < x ∧ x < π / 2 → 2 * x > 3 * Real.sin x) := by
  sorry

end relationship_2x_3sinx_l3951_395195


namespace same_solution_implies_a_value_l3951_395168

theorem same_solution_implies_a_value :
  ∀ x a : ℚ,
  (3 * x + 5 = 11) →
  (6 * x + 3 * a = 22) →
  a = 10 / 3 :=
by sorry

end same_solution_implies_a_value_l3951_395168


namespace union_of_A_and_B_l3951_395171

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 2, 3, 4} := by sorry

end union_of_A_and_B_l3951_395171


namespace coin_value_calculation_l3951_395194

def total_value (num_dimes num_quarters : ℕ) (dime_value quarter_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value + (num_quarters : ℚ) * quarter_value

theorem coin_value_calculation :
  total_value 22 10 (10 / 100) (25 / 100) = 470 / 100 := by
  sorry

end coin_value_calculation_l3951_395194


namespace least_number_divisible_by_five_primes_l3951_395136

theorem least_number_divisible_by_five_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ q₁ q₂ q₃ q₄ q₅ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → 
    m ≥ n) ∧
  n = 2310 :=
by sorry

end least_number_divisible_by_five_primes_l3951_395136


namespace arithmetic_sequence_properties_l3951_395143

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * a 1 + (n : ℝ) * (n - 1) / 2 * (a 2 - a 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h : seq.a 1 + 5 * seq.a 3 = seq.S 8) :
    seq.a 10 = 0 ∧ seq.S 7 = seq.S 12 := by
  sorry

end arithmetic_sequence_properties_l3951_395143


namespace highest_power_of_three_dividing_M_l3951_395181

def M : ℕ := sorry

theorem highest_power_of_three_dividing_M : 
  ∃ (k : ℕ), (3^1 ∣ M) ∧ ¬(3^(k+2) ∣ M) :=
sorry

end highest_power_of_three_dividing_M_l3951_395181


namespace house_bedrooms_count_l3951_395196

/-- A house with two floors and a certain number of bedrooms on each floor. -/
structure House where
  second_floor_bedrooms : ℕ
  first_floor_bedrooms : ℕ

/-- The total number of bedrooms in a house. -/
def total_bedrooms (h : House) : ℕ :=
  h.second_floor_bedrooms + h.first_floor_bedrooms

/-- Theorem stating that a house with 2 bedrooms on the second floor and 8 on the first floor has 10 bedrooms in total. -/
theorem house_bedrooms_count :
  ∀ (h : House), h.second_floor_bedrooms = 2 → h.first_floor_bedrooms = 8 →
  total_bedrooms h = 10 := by
  sorry

end house_bedrooms_count_l3951_395196


namespace julio_lime_cost_l3951_395107

/-- Represents the cost of limes for Julio's mocktails over 30 days -/
def lime_cost (mocktails_per_day : ℕ) (tbsp_per_mocktail : ℕ) (tbsp_per_lime : ℕ) (limes_per_dollar : ℕ) (days : ℕ) : ℚ :=
  let limes_needed := (mocktails_per_day * tbsp_per_mocktail * days) / tbsp_per_lime
  let lime_sets := (limes_needed + limes_per_dollar - 1) / limes_per_dollar
  lime_sets

theorem julio_lime_cost : 
  lime_cost 1 1 2 3 30 = 5 := by
  sorry

end julio_lime_cost_l3951_395107


namespace sampling_suitable_for_large_events_sampling_suitable_for_beijing_olympics_l3951_395142

/-- Represents a survey method -/
inductive SurveyMethod
  | Sampling
  | Comprehensive

/-- Represents a large-scale event -/
structure LargeEvent where
  name : String
  potential_viewers : ℕ

/-- Defines when a survey method is suitable for an event -/
def is_suitable_survey_method (method : SurveyMethod) (event : LargeEvent) : Prop :=
  match method with
  | SurveyMethod.Sampling => 
      event.potential_viewers > 1000000 ∧ 
      (∀ n : ℕ, n < event.potential_viewers → ∃ sample : Finset ℕ, sample.card = n)
  | SurveyMethod.Comprehensive => 
      event.potential_viewers ≤ 1000000

/-- The main theorem stating that sampling survey is suitable for large events -/
theorem sampling_suitable_for_large_events (event : LargeEvent) 
  (h1 : event.potential_viewers > 1000000) 
  (h2 : ∀ n : ℕ, n < event.potential_viewers → ∃ sample : Finset ℕ, sample.card = n) :
  is_suitable_survey_method SurveyMethod.Sampling event :=
by sorry

/-- The Beijing Winter Olympics as an instance of LargeEvent -/
def beijing_winter_olympics : LargeEvent :=
  { name := "Beijing Winter Olympics"
  , potential_viewers := 2000000000 }  -- An example large number

/-- Theorem stating that sampling survey is suitable for the Beijing Winter Olympics -/
theorem sampling_suitable_for_beijing_olympics :
  is_suitable_survey_method SurveyMethod.Sampling beijing_winter_olympics :=
by sorry

end sampling_suitable_for_large_events_sampling_suitable_for_beijing_olympics_l3951_395142


namespace intersection_of_lines_l3951_395172

/-- Given four points in 3D space, this theorem states that the intersection of the lines
    passing through the first two points and the last two points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (5, -6, 8) →
  B = (15, -16, 13) →
  C = (1, 4, -5) →
  D = (3, -4, 11) →
  ∃ t s : ℝ, 
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (1 + 2*s, 4 - 8*s, -5 + 16*s) ∧
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (3, -4, 7) :=
by sorry

end intersection_of_lines_l3951_395172


namespace m_range_l3951_395111

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom even_func : ∀ x : ℝ, f (-x) = f x
axiom increasing_neg : ∀ a b : ℝ, a < b → b < 0 → (f a - f b) / (a - b) > 0
axiom condition : ∀ m : ℝ, f (2*m + 1) > f (2*m)

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x : ℝ, f (-x) = f x) → 
  (∀ a b : ℝ, a < b → b < 0 → (f a - f b) / (a - b) > 0) → 
  (f (2*m + 1) > f (2*m)) → 
  m < -1/4 :=
sorry

end m_range_l3951_395111


namespace room_selection_equivalence_l3951_395186

def total_rooms : ℕ := 6

def select_at_least_two (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ k => if k ≥ 2 then Nat.choose n k else 0)

def sum_of_combinations (n : ℕ) : ℕ :=
  Nat.choose n 3 + 2 * Nat.choose n 4 + Nat.choose n 5 + Nat.choose n 6

def power_minus_seven (n : ℕ) : ℕ :=
  2^n - 7

theorem room_selection_equivalence :
  select_at_least_two total_rooms = sum_of_combinations total_rooms ∧
  select_at_least_two total_rooms = power_minus_seven total_rooms := by
  sorry

end room_selection_equivalence_l3951_395186


namespace multiply_and_add_equality_l3951_395152

theorem multiply_and_add_equality : 24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end multiply_and_add_equality_l3951_395152


namespace triangle_height_proof_l3951_395130

/-- Given a triangle with base 4 meters and a constant k = 2 meters, 
    prove that its height is 4 meters when its area satisfies two equations. -/
theorem triangle_height_proof (height : ℝ) (k : ℝ) (base : ℝ) : 
  k = 2 →
  base = 4 →
  (base^2) / (4 * (height - k)) = (1/2) * base * height →
  height = 4 := by
  sorry

end triangle_height_proof_l3951_395130


namespace sqrt_one_plus_xy_rational_l3951_395151

theorem sqrt_one_plus_xy_rational (x y : ℚ) 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (x*y + 1)^2 = 0) : 
  ∃ (q : ℚ), q^2 = 1 + x*y := by
  sorry

end sqrt_one_plus_xy_rational_l3951_395151


namespace cos_product_20_40_60_80_l3951_395187

theorem cos_product_20_40_60_80 : 
  Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (60 * π / 180) * Real.cos (80 * π / 180) = 1 / 16 := by
  sorry

end cos_product_20_40_60_80_l3951_395187


namespace pushup_difference_l3951_395161

theorem pushup_difference (zachary_pushups : Real) (david_more_than_zachary : Real) (john_less_than_david : Real)
  (h1 : zachary_pushups = 15.5)
  (h2 : david_more_than_zachary = 39.2)
  (h3 : john_less_than_david = 9.3) :
  let david_pushups := zachary_pushups + david_more_than_zachary
  let john_pushups := david_pushups - john_less_than_david
  john_pushups - zachary_pushups = 29.9 := by
sorry

end pushup_difference_l3951_395161


namespace inequality_system_solution_implies_a_greater_than_negative_one_l3951_395182

theorem inequality_system_solution_implies_a_greater_than_negative_one :
  (∃ x : ℝ, x + a ≥ 0 ∧ 1 - 2*x > x - 2) → a > -1 :=
by sorry

end inequality_system_solution_implies_a_greater_than_negative_one_l3951_395182


namespace grade_c_boxes_l3951_395167

theorem grade_c_boxes (total : ℕ) (m n t : ℕ) 
  (h1 : total = 420)
  (h2 : 2 * t = m + n) : 
  (total / 3 : ℕ) = 140 := by sorry

end grade_c_boxes_l3951_395167
