import Mathlib

namespace personal_income_tax_l1331_133136

/-- Personal income tax calculation -/
theorem personal_income_tax (salary : ℕ) (tax_free : ℕ) (rate1 : ℚ) (rate2 : ℚ) (threshold : ℕ) : 
  salary = 2900 ∧ 
  tax_free = 2000 ∧ 
  rate1 = 5/100 ∧ 
  rate2 = 10/100 ∧ 
  threshold = 500 → 
  (min threshold (salary - tax_free) : ℚ) * rate1 + 
  (max 0 ((salary - tax_free) - threshold) : ℚ) * rate2 = 65 := by
sorry

end personal_income_tax_l1331_133136


namespace multiples_of_seven_l1331_133146

theorem multiples_of_seven (x : ℕ) : 
  (∃ n : ℕ, n = 47 ∧ 
   (∀ k : ℕ, x ≤ 7 * k ∧ 7 * k ≤ 343 → k ≤ n) ∧
   (∀ k : ℕ, k ≤ n → x ≤ 7 * k ∧ 7 * k ≤ 343)) →
  x = 14 := by
sorry

end multiples_of_seven_l1331_133146


namespace power_division_simplification_l1331_133161

theorem power_division_simplification : 8^15 / 64^3 = 8^9 := by
  sorry

end power_division_simplification_l1331_133161


namespace right_triangle_legs_from_altitude_areas_l1331_133187

/-- Given a right-angled triangle ABC with right angle at C, and altitude CD to hypotenuse AB
    dividing the triangle into two triangles with areas Q and q, 
    the legs of the triangle are √(2(q + Q)√(q/Q)) and √(2(q + Q)√(Q/q)). -/
theorem right_triangle_legs_from_altitude_areas (Q q : ℝ) (hQ : Q > 0) (hq : q > 0) :
  ∃ (AC BC : ℝ),
    AC = Real.sqrt (2 * (q + Q) * Real.sqrt (q / Q)) ∧
    BC = Real.sqrt (2 * (q + Q) * Real.sqrt (Q / q)) ∧
    AC^2 + BC^2 = (AC * BC)^2 / (Q + q) := by
  sorry

end right_triangle_legs_from_altitude_areas_l1331_133187


namespace fruit_cup_cost_calculation_l1331_133160

/-- The cost of a fruit cup in dollars -/
def fruit_cup_cost : ℝ := 1.80

/-- The cost of a muffin in dollars -/
def muffin_cost : ℝ := 2

/-- The number of muffins Francis had -/
def francis_muffins : ℕ := 2

/-- The number of fruit cups Francis had -/
def francis_fruit_cups : ℕ := 2

/-- The number of muffins Kiera had -/
def kiera_muffins : ℕ := 2

/-- The number of fruit cups Kiera had -/
def kiera_fruit_cups : ℕ := 1

/-- The total cost of their breakfast in dollars -/
def total_cost : ℝ := 17

theorem fruit_cup_cost_calculation :
  (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost) +
  (kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost) = total_cost :=
by sorry

end fruit_cup_cost_calculation_l1331_133160


namespace polynomial_multiplication_simplification_l1331_133110

theorem polynomial_multiplication_simplification (x : ℝ) :
  (3 * x - 2) * (5 * x^12 + 3 * x^11 + 5 * x^10 + 3 * x^9) =
  15 * x^13 - x^12 + 9 * x^11 - x^10 - 6 * x^9 := by
  sorry

end polynomial_multiplication_simplification_l1331_133110


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1331_133191

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c => 
    (a = 9 ∧ b = 9 ∧ c = 4) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
    (a = b) →  -- Isosceles condition
    (a + b + c = 22)  -- Perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 9 9 4 := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1331_133191


namespace probability_multiple_of_four_l1331_133129

/-- A set of digits from 1 to 5 -/
def DigitSet : Finset ℕ := {1, 2, 3, 4, 5}

/-- A function to check if a three-digit number is divisible by 4 -/
def isDivisibleByFour (a b c : ℕ) : Prop := (10 * b + c) % 4 = 0

/-- The total number of ways to draw three digits from five -/
def totalWays : ℕ := 5 * 4 * 3

/-- The number of ways to draw three digits that form a number divisible by 4 -/
def validWays : ℕ := 15

theorem probability_multiple_of_four :
  (validWays : ℚ) / totalWays = 1 / 4 := by sorry

end probability_multiple_of_four_l1331_133129


namespace steve_pages_written_l1331_133137

/-- Calculates the total number of pages Steve writes in a month -/
def total_pages_written (days_per_month : ℕ) (days_between_letters : ℕ) 
  (minutes_per_regular_letter : ℕ) (minutes_per_page : ℕ) 
  (minutes_for_long_letter : ℕ) : ℕ :=
  let regular_letters := days_per_month / days_between_letters
  let pages_per_regular_letter := minutes_per_regular_letter / minutes_per_page
  let regular_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := minutes_for_long_letter / (2 * minutes_per_page)
  regular_pages + long_letter_pages

theorem steve_pages_written :
  total_pages_written 30 3 20 10 80 = 24 := by sorry

end steve_pages_written_l1331_133137


namespace min_distance_to_line_l1331_133145

/-- Given a right-angled triangle with sides a, b, and hypotenuse c,
    and a point (m,n) on the line ax+by+2c=0,
    the minimum value of m^2 + n^2 is 4. -/
theorem min_distance_to_line (a b c : ℝ) (m n : ℝ → ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  (∀ t, a * (m t) + b * (n t) + 2*c = 0) →
  (∃ t₀, ∀ t, (m t)^2 + (n t)^2 ≥ (m t₀)^2 + (n t₀)^2) →
  ∃ t₀, (m t₀)^2 + (n t₀)^2 = 4 :=
by sorry

end min_distance_to_line_l1331_133145


namespace evaluate_expression_l1331_133158

theorem evaluate_expression :
  (2^1501 + 5^1502)^2 - (2^1501 - 5^1502)^2 = 20 * 10^1501 := by
  sorry

end evaluate_expression_l1331_133158


namespace smarties_remainder_l1331_133127

theorem smarties_remainder (n : ℕ) (h : n % 11 = 6) : (4 * n) % 11 = 2 := by
  sorry

end smarties_remainder_l1331_133127


namespace arithmetic_sequence_problem_l1331_133111

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where a_2 + a_6 = 10, prove that a_4 = 5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 6 = 10) : 
  a 4 = 5 := by
sorry

end arithmetic_sequence_problem_l1331_133111


namespace blue_marble_difference_l1331_133119

theorem blue_marble_difference (total_green : ℕ) 
  (ratio_a_blue ratio_a_green ratio_b_blue ratio_b_green : ℕ) : 
  total_green = 162 →
  ratio_a_blue = 5 →
  ratio_a_green = 3 →
  ratio_b_blue = 4 →
  ratio_b_green = 1 →
  ∃ (a b : ℕ), 
    ratio_a_green * a + ratio_b_green * b = total_green ∧
    (ratio_a_blue + ratio_a_green) * a = (ratio_b_blue + ratio_b_green) * b ∧
    ratio_b_blue * b - ratio_a_blue * a = 49 :=
by
  sorry

#check blue_marble_difference

end blue_marble_difference_l1331_133119


namespace adam_tattoo_count_l1331_133147

/-- The number of tattoos Jason has on each arm -/
def jason_arm_tattoos : ℕ := 2

/-- The number of tattoos Jason has on each leg -/
def jason_leg_tattoos : ℕ := 3

/-- The number of arms Jason has -/
def jason_arms : ℕ := 2

/-- The number of legs Jason has -/
def jason_legs : ℕ := 2

/-- The total number of tattoos Jason has -/
def jason_total_tattoos : ℕ := jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

/-- The number of tattoos Adam has -/
def adam_tattoos : ℕ := 2 * jason_total_tattoos + 3

theorem adam_tattoo_count : adam_tattoos = 23 := by
  sorry

end adam_tattoo_count_l1331_133147


namespace center_locus_is_conic_l1331_133198

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A conic section in a 2D plane -/
structure Conic where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a conic section -/
def center (c : Conic) : Point2D :=
  { x := 0, y := 0 }  -- Placeholder definition

/-- Checks if a point lies on a conic -/
def lies_on (p : Point2D) (c : Conic) : Prop :=
  c.a * p.x^2 + c.b * p.x * p.y + c.c * p.y^2 + c.d * p.x + c.e * p.y + c.f = 0

/-- The set of all conics passing through four given points -/
def conics_through_four_points (A B C D : Point2D) : Set Conic :=
  { c | lies_on A c ∧ lies_on B c ∧ lies_on C c ∧ lies_on D c }

/-- The locus of centers of conics passing through four points -/
def center_locus (A B C D : Point2D) : Set Point2D :=
  { p | ∃ c ∈ conics_through_four_points A B C D, center c = p }

/-- Theorem: The locus of centers of conics passing through four points is a conic -/
theorem center_locus_is_conic (A B C D : Point2D) :
  ∃ Γ : Conic, ∀ p ∈ center_locus A B C D, lies_on p Γ :=
sorry

end center_locus_is_conic_l1331_133198


namespace circle_line_tangency_l1331_133149

/-- The circle equation -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 2*k*x - 2*y = 0

/-- The line equation -/
def line_equation (x y k : ℝ) : Prop :=
  x + y = 2*k

/-- The tangency condition -/
def are_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y k ∧ line_equation x y k

/-- The main theorem -/
theorem circle_line_tangency (k : ℝ) :
  are_tangent k → k = -1 := by sorry

end circle_line_tangency_l1331_133149


namespace f_triple_3_l1331_133128

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_triple_3 : f (f (f 3)) = 107 := by
  sorry

end f_triple_3_l1331_133128


namespace first_car_departure_time_l1331_133103

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv : minutes < 60

/-- Represents a car with its speed -/
structure Car where
  speed : ℝ  -- speed in miles per hour

def problem (first_car : Car) (second_car : Car) (trip_distance : ℝ) (time_difference : ℝ) (meeting_time : Time) : Prop :=
  first_car.speed = 30 ∧
  second_car.speed = 60 ∧
  trip_distance = 80 ∧
  time_difference = 1/6 ∧  -- 10 minutes in hours
  meeting_time.hours = 10 ∧
  meeting_time.minutes = 30

theorem first_car_departure_time 
  (first_car : Car) (second_car : Car) (trip_distance : ℝ) (time_difference : ℝ) (meeting_time : Time) :
  problem first_car second_car trip_distance time_difference meeting_time →
  ∃ (departure_time : Time), 
    departure_time.hours = 10 ∧ departure_time.minutes = 10 :=
sorry

end first_car_departure_time_l1331_133103


namespace det_sin_matrix_zero_l1331_133151

theorem det_sin_matrix_zero :
  let A : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    match i, j with
    | 0, 0 => Real.sin 1
    | 0, 1 => Real.sin 2
    | 0, 2 => Real.sin 3
    | 1, 0 => Real.sin 4
    | 1, 1 => Real.sin 5
    | 1, 2 => Real.sin 6
    | 2, 0 => Real.sin 7
    | 2, 1 => Real.sin 8
    | 2, 2 => Real.sin 9
  Matrix.det A = 0 :=
by sorry

end det_sin_matrix_zero_l1331_133151


namespace probability_same_color_is_correct_l1331_133192

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 4

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_is_correct : probability_same_color = 106 / 109725 := by
  sorry

end probability_same_color_is_correct_l1331_133192


namespace julia_rental_cost_l1331_133115

/-- Calculates the total cost of a car rental --/
def calculateRentalCost (dailyRate : ℝ) (mileageRate : ℝ) (days : ℝ) (miles : ℝ) : ℝ :=
  dailyRate * days + mileageRate * miles

/-- Proves that Julia's car rental cost is $46.12 --/
theorem julia_rental_cost :
  let dailyRate : ℝ := 29
  let mileageRate : ℝ := 0.08
  let days : ℝ := 1
  let miles : ℝ := 214
  calculateRentalCost dailyRate mileageRate days miles = 46.12 := by
  sorry

end julia_rental_cost_l1331_133115


namespace mona_unique_players_l1331_133155

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (groups : ℕ) (players_per_group : ℕ) (groups_with_two_repeats : ℕ) (groups_with_one_repeat : ℕ) : ℕ :=
  groups * players_per_group - (2 * groups_with_two_repeats + groups_with_one_repeat)

/-- Theorem stating the number of unique players Mona grouped with --/
theorem mona_unique_players :
  unique_players 25 4 8 5 = 79 := by
  sorry

end mona_unique_players_l1331_133155


namespace min_value_theorem_l1331_133134

theorem min_value_theorem (x : ℝ) (h1 : x > 0) (h2 : Real.log x + 1 ≤ x) :
  (x^2 - Real.log x + x) / x ≥ 2 ∧
  (∃ y > 0, (y^2 - Real.log y + y) / y = 2) :=
sorry

end min_value_theorem_l1331_133134


namespace stock_price_increase_l1331_133131

theorem stock_price_increase (X : ℝ) : 
  (1 + X / 100) * (1 - 25 / 100) * (1 + 15 / 100) = 103.5 / 100 → X = 20 := by
  sorry

end stock_price_increase_l1331_133131


namespace z_values_l1331_133139

theorem z_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let z := ((x - 3)^2 * (x + 4)) / (2*x - 4)
  z = 64.8 ∨ z = -10.125 := by
sorry

end z_values_l1331_133139


namespace complex_equation_solution_l1331_133165

/-- Given a and b are real numbers satisfying a + bi = (1 + i)i^3, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) (h : (↑a + ↑b * I) = (1 + I) * I^3) : a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l1331_133165


namespace smallest_prime_six_less_than_square_l1331_133197

theorem smallest_prime_six_less_than_square : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.Prime n) ∧ 
    (∃ (m : ℕ), n = m^2 - 6) ∧
    (∀ (k : ℕ), k > 0 → Nat.Prime k → (∃ (j : ℕ), k = j^2 - 6) → k ≥ n) ∧
    n = 3 := by
  sorry

end smallest_prime_six_less_than_square_l1331_133197


namespace orange_sales_theorem_l1331_133138

def planned_daily_sales : ℕ := 10
def deviations : List ℤ := [4, -3, -5, 7, -8, 21, -6]
def selling_price : ℕ := 80
def shipping_fee : ℕ := 7

theorem orange_sales_theorem :
  let first_five_days_sales := planned_daily_sales * 5 + (deviations.take 5).sum
  let total_deviation := deviations.sum
  let total_sales := planned_daily_sales * 7 + total_deviation
  let total_earnings := total_sales * selling_price - total_sales * shipping_fee
  (first_five_days_sales = 45) ∧
  (total_deviation > 0) ∧
  (total_earnings = 5840) := by
  sorry

end orange_sales_theorem_l1331_133138


namespace f_properties_l1331_133157

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem f_properties :
  (∀ x > 0, f x = x^2 + x - Real.log x) →
  (∃ m b : ℝ, ∀ x : ℝ, (x = 1 → f x = m * x + b) ∧ m = 2 ∧ b = 0) ∧
  (∃ x_min : ℝ, x_min > 0 ∧ ∀ x > 0, f x ≥ f x_min ∧ f x_min = 3/4 + Real.log 2) ∧
  (¬ ∃ x_max : ℝ, x_max > 0 ∧ ∀ x > 0, f x ≤ f x_max) :=
by sorry

end f_properties_l1331_133157


namespace combined_machine_time_l1331_133135

theorem combined_machine_time (t1 t2 : ℝ) (h1 : t1 = 20) (h2 : t2 = 30) :
  1 / (1 / t1 + 1 / t2) = 12 := by
  sorry

end combined_machine_time_l1331_133135


namespace logistics_personnel_in_sample_l1331_133113

theorem logistics_personnel_in_sample
  (total_staff : ℕ)
  (logistics_staff : ℕ)
  (sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : logistics_staff = 24)
  (h3 : sample_size = 20) :
  (logistics_staff : ℚ) / (total_staff : ℚ) * (sample_size : ℚ) = 3 :=
by sorry

end logistics_personnel_in_sample_l1331_133113


namespace matrix_power_four_l1331_133124

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end matrix_power_four_l1331_133124


namespace parallel_vectors_m_value_l1331_133173

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two parallel vectors (1,2) and (m,1), prove that m = 1/2 -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, 1)
  are_parallel a b → m = 1/2 :=
by sorry

end parallel_vectors_m_value_l1331_133173


namespace cone_height_from_sphere_waste_l1331_133121

/-- Given a sphere and a cone carved from it, prove the height of the cone when 75% of wood is wasted -/
theorem cone_height_from_sphere_waste (r : ℝ) (h : ℝ) : 
  r = 9 →  -- sphere radius
  (4/3) * Real.pi * r^3 * (1 - 0.75) = (1/3) * Real.pi * r^2 * h → -- 75% wood wasted
  h = 27 := by
  sorry

end cone_height_from_sphere_waste_l1331_133121


namespace golden_ratio_comparison_l1331_133107

theorem golden_ratio_comparison : (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end golden_ratio_comparison_l1331_133107


namespace franks_remaining_money_l1331_133143

/-- 
Given:
- Frank initially had $600.
- He spent 1/5 of his money on groceries.
- He then spent 1/4 of the remaining money on a magazine.

Prove that Frank has $360 left after buying groceries and the magazine.
-/
theorem franks_remaining_money (initial_amount : ℚ) 
  (h1 : initial_amount = 600)
  (grocery_fraction : ℚ) (h2 : grocery_fraction = 1/5)
  (magazine_fraction : ℚ) (h3 : magazine_fraction = 1/4) :
  let remaining_after_groceries := initial_amount - grocery_fraction * initial_amount
  let remaining_after_magazine := remaining_after_groceries - magazine_fraction * remaining_after_groceries
  remaining_after_magazine = 360 := by
sorry

end franks_remaining_money_l1331_133143


namespace area_between_concentric_circles_l1331_133116

/-- Given two concentric circles where the radius of the outer circle is twice
    the radius of the inner circle, and the width of the gray region between
    them is 3 feet, prove that the area of the gray region is 21π square feet. -/
theorem area_between_concentric_circles (r : ℝ) : 
  r > 0 → -- Inner circle radius is positive
  2 * r - r = 3 → -- Width of gray region is 3
  π * (2 * r)^2 - π * r^2 = 21 * π := by
  sorry

end area_between_concentric_circles_l1331_133116


namespace cricket_run_rate_l1331_133142

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario. -/
theorem cricket_run_rate : required_run_rate 50 10 (34/10) 282 = 62/10 := by
  sorry

end cricket_run_rate_l1331_133142


namespace lifespan_survey_is_sample_l1331_133126

/-- Represents a collection of data from a survey --/
structure SurveyData where
  size : Nat
  provinces : Nat
  dataType : Type

/-- Defines what constitutes a sample in statistical terms --/
def IsSample (data : SurveyData) : Prop :=
  data.size < population_size ∧ data.size > 0
  where population_size : Nat := 1000000  -- Arbitrary large number for illustration

/-- The theorem to be proved --/
theorem lifespan_survey_is_sample :
  let survey : SurveyData := {
    size := 2500,
    provinces := 11,
    dataType := Nat  -- Assuming lifespan is measured in years
  }
  IsSample survey := by sorry


end lifespan_survey_is_sample_l1331_133126


namespace twenty_dollars_combinations_l1331_133104

/-- The number of ways to make 20 dollars with nickels, dimes, and quarters -/
def ways_to_make_20_dollars : ℕ :=
  (Finset.filter (fun (n, d, q) => 
    5 * n + 10 * d + 25 * q = 2000 ∧ 
    n ≥ 2 ∧ 
    q ≥ 1) 
  (Finset.product (Finset.range 401) (Finset.product (Finset.range 201) (Finset.range 81)))).card

/-- Theorem stating that there are exactly 130 ways to make 20 dollars 
    with nickels, dimes, and quarters, using at least two nickels and one quarter -/
theorem twenty_dollars_combinations : ways_to_make_20_dollars = 130 := by
  sorry

end twenty_dollars_combinations_l1331_133104


namespace tutors_next_meeting_l1331_133194

theorem tutors_next_meeting (elena fiona george harry : ℕ) 
  (h_elena : elena = 5)
  (h_fiona : fiona = 6)
  (h_george : george = 8)
  (h_harry : harry = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm elena fiona) george) harry = 360 := by
  sorry

end tutors_next_meeting_l1331_133194


namespace least_number_with_remainder_five_l1331_133188

def is_valid_number (n : ℕ) : Prop :=
  ∃ (S : Set ℕ), 15 ∈ S ∧ ∀ m ∈ S, m > 0 ∧ n % m = 5

theorem least_number_with_remainder_five :
  is_valid_number 125 ∧ ∀ k < 125, ¬(is_valid_number k) :=
sorry

end least_number_with_remainder_five_l1331_133188


namespace reading_time_calculation_l1331_133117

theorem reading_time_calculation (total_time math_time spelling_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18) :
  total_time - (math_time + spelling_time) = 27 :=
by
  sorry

end reading_time_calculation_l1331_133117


namespace perimeter_approx_l1331_133178

/-- A right triangle with area 150 and one leg 15 units longer than the other -/
structure RightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  area_eq : (1/2) * shorter_leg * longer_leg = 150
  leg_diff : longer_leg = shorter_leg + 15
  pythagorean : shorter_leg^2 + longer_leg^2 = hypotenuse^2

/-- The perimeter of the triangle -/
def perimeter (t : RightTriangle) : ℝ :=
  t.shorter_leg + t.longer_leg + t.hypotenuse

/-- Theorem stating that the perimeter is approximately 66.47 -/
theorem perimeter_approx (t : RightTriangle) :
  abs (perimeter t - 66.47) < 0.01 := by
  sorry

end perimeter_approx_l1331_133178


namespace president_and_committee_selection_l1331_133199

/-- The number of ways to choose a president and a 3-person committee from a group of 10 people,
    where the order of committee selection doesn't matter and the president cannot be on the committee. -/
def select_president_and_committee (total_people : ℕ) (committee_size : ℕ) : ℕ :=
  total_people * (Nat.choose (total_people - 1) committee_size)

/-- Theorem stating that the number of ways to choose a president and a 3-person committee
    from a group of 10 people, where the order of committee selection doesn't matter and
    the president cannot be on the committee, is equal to 840. -/
theorem president_and_committee_selection :
  select_president_and_committee 10 3 = 840 := by
  sorry


end president_and_committee_selection_l1331_133199


namespace gene_mutation_not_valid_for_AaB_l1331_133100

/-- Represents a genotype --/
inductive Genotype
  | AaB
  | AABb

/-- Represents possible reasons for lacking a gene --/
inductive Reason
  | GeneMutation
  | ChromosomalVariation
  | ChromosomalStructuralVariation
  | MaleIndividual

/-- Determines if a reason is valid for explaining the lack of a gene --/
def is_valid_reason (g : Genotype) (r : Reason) : Prop :=
  match g, r with
  | Genotype.AaB, Reason.GeneMutation => False
  | _, _ => True

/-- Theorem stating that gene mutation is not a valid reason for individual A's genotype --/
theorem gene_mutation_not_valid_for_AaB :
  ¬(is_valid_reason Genotype.AaB Reason.GeneMutation) :=
by
  sorry


end gene_mutation_not_valid_for_AaB_l1331_133100


namespace eldest_age_difference_l1331_133180

/-- Represents the ages of three grandchildren -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.middle = ages.youngest + 3 ∧
  ages.eldest = 3 * ages.youngest ∧
  ages.eldest = 15

theorem eldest_age_difference (ages : GrandchildrenAges) :
  satisfiesConditions ages →
  ages.eldest = ages.youngest + ages.middle + 2 := by
  sorry

end eldest_age_difference_l1331_133180


namespace course_choice_related_to_gender_l1331_133164

-- Define the contingency table
def contingency_table := (40, 10, 30, 20)

-- Define the total number of students
def total_students : Nat := 100

-- Define the critical value for α = 0.05
def critical_value : Float := 3.841

-- Function to calculate χ²
def calculate_chi_square (a b c d : Nat) : Float :=
  let n := a + b + c + d
  let numerator := n * (a * d - b * c) ^ 2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  numerator.toFloat / denominator.toFloat

-- Theorem statement
theorem course_choice_related_to_gender (a b c d : Nat) 
  (h1 : (a, b, c, d) = contingency_table) 
  (h2 : a + b + c + d = total_students) : 
  calculate_chi_square a b c d > critical_value :=
by
  sorry


end course_choice_related_to_gender_l1331_133164


namespace unpainted_cubes_in_four_cube_l1331_133190

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  total_cubes : ℕ
  painted_corners : ℕ
  painted_edges : ℕ

/-- Properties of a 4x4x4 cube with painted corners -/
def four_cube : Cube 4 :=
  { side_length := 4
  , total_cubes := 64
  , painted_corners := 8
  , painted_edges := 12 }

/-- The number of unpainted cubes in a cube with painted corners -/
def unpainted_cubes (c : Cube n) : ℕ :=
  c.total_cubes - (c.painted_corners + c.painted_edges)

/-- Theorem: The number of unpainted cubes in a 4x4x4 cube with painted corners is 44 -/
theorem unpainted_cubes_in_four_cube :
  unpainted_cubes four_cube = 44 := by
  sorry

end unpainted_cubes_in_four_cube_l1331_133190


namespace veggie_votes_l1331_133181

theorem veggie_votes (total_votes meat_votes : ℕ) 
  (h1 : total_votes = 672)
  (h2 : meat_votes = 335) : 
  total_votes - meat_votes = 337 := by
sorry

end veggie_votes_l1331_133181


namespace largest_of_three_consecutive_odds_l1331_133122

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (a + b + c = 75) →  -- sum is 75
  (c - a = 4) →       -- difference between largest and smallest is 4
  (Odd a ∧ Odd b ∧ Odd c) →  -- all numbers are odd
  (b = a + 2 ∧ c = b + 2) →  -- numbers are consecutive
  (c = 27) :=         -- largest number is 27
by sorry

end largest_of_three_consecutive_odds_l1331_133122


namespace gcf_is_correct_l1331_133168

def term1 (x y : ℕ) : ℕ := 9 * x^3 * y^2
def term2 (x y : ℕ) : ℕ := 12 * x^2 * y^3

def gcf (x y : ℕ) : ℕ := 3 * x^2 * y^2

theorem gcf_is_correct (x y : ℕ) :
  (gcf x y) ∣ (term1 x y) ∧ (gcf x y) ∣ (term2 x y) ∧
  ∀ (d : ℕ), d ∣ (term1 x y) ∧ d ∣ (term2 x y) → d ∣ (gcf x y) :=
sorry

end gcf_is_correct_l1331_133168


namespace unsatisfactory_tests_l1331_133159

theorem unsatisfactory_tests (n : ℕ) (k : ℕ) : 
  n < 50 →
  n % 7 = 0 →
  n % 3 = 0 →
  n % 2 = 0 →
  n / 7 + n / 3 + n / 2 + k = n →
  k = 1 := by
sorry

end unsatisfactory_tests_l1331_133159


namespace bus_row_capacity_l1331_133163

/-- Represents a school bus with a given number of rows and total capacity. -/
structure SchoolBus where
  rows : ℕ
  totalCapacity : ℕ

/-- Calculates the capacity of each row in the school bus. -/
def rowCapacity (bus : SchoolBus) : ℕ :=
  bus.totalCapacity / bus.rows

/-- Theorem stating that for a bus with 20 rows and a total capacity of 80,
    the capacity of each row is 4. -/
theorem bus_row_capacity :
  let bus : SchoolBus := { rows := 20, totalCapacity := 80 }
  rowCapacity bus = 4 := by
  sorry

end bus_row_capacity_l1331_133163


namespace dans_initial_money_l1331_133144

/-- Represents Dan's money transactions -/
def dans_money (initial : ℕ) (candy_cost : ℕ) (chocolate_cost : ℕ) (remaining : ℕ) : Prop :=
  initial = candy_cost + chocolate_cost + remaining

theorem dans_initial_money : 
  ∃ (initial : ℕ), dans_money initial 2 3 2 ∧ initial = 7 := by
  sorry

end dans_initial_money_l1331_133144


namespace normal_distribution_symmetry_l1331_133112

/-- A random variable following a normal distribution with mean 2 and variance 4 -/
def X : Real → Real := sorry

/-- The probability density function of X -/
def pdf_X : Real → Real := sorry

/-- The cumulative distribution function of X -/
def cdf_X : Real → Real := sorry

/-- The value 'a' such that P(X < a) = 0.2 -/
def a : Real := sorry

theorem normal_distribution_symmetry (h1 : ∀ x, pdf_X x = pdf_X (4 - x))
  (h2 : cdf_X a = 0.2) : cdf_X (4 - a) = 0.8 := by sorry

end normal_distribution_symmetry_l1331_133112


namespace custom_operation_result_l1331_133189

def custom_operation (A B : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y * (x + y)}

theorem custom_operation_result :
  let A : Set ℕ := {0, 1}
  let B : Set ℕ := {2, 3}
  custom_operation A B = {0, 6, 12} := by
  sorry

end custom_operation_result_l1331_133189


namespace sqrt_a_div_sqrt_b_eq_five_halves_l1331_133185

theorem sqrt_a_div_sqrt_b_eq_five_halves (a b : ℝ) 
  (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*a)/(61*b)) : 
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end sqrt_a_div_sqrt_b_eq_five_halves_l1331_133185


namespace power_of_product_l1331_133172

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by sorry

end power_of_product_l1331_133172


namespace correct_num_schedules_l1331_133140

/-- Represents a subject in the school schedule -/
inductive Subject
| Chinese
| Mathematics
| English
| ScienceComprehensive

/-- Represents a class period -/
inductive ClassPeriod
| First
| Second
| Third

/-- A schedule is a function that assigns subjects to class periods -/
def Schedule := ClassPeriod → List Subject

/-- Checks if a schedule is valid according to the problem constraints -/
def isValidSchedule (s : Schedule) : Prop :=
  (∀ subject : Subject, ∃ period : ClassPeriod, subject ∈ s period) ∧
  (∀ period : ClassPeriod, s period ≠ []) ∧
  (∀ period : ClassPeriod, Subject.Mathematics ∈ s period → Subject.ScienceComprehensive ∉ s period) ∧
  (∀ period : ClassPeriod, Subject.ScienceComprehensive ∈ s period → Subject.Mathematics ∉ s period)

/-- The number of valid schedules -/
def numValidSchedules : ℕ := sorry

theorem correct_num_schedules : numValidSchedules = 30 := by sorry

end correct_num_schedules_l1331_133140


namespace power_of_64_l1331_133133

theorem power_of_64 : 64^(5/3) = 1024 := by
  have h : 64 = 2^6 := by sorry
  sorry

end power_of_64_l1331_133133


namespace brians_trip_distance_l1331_133154

/-- Calculates the distance traveled given car efficiency and gas used -/
def distance_traveled (efficiency : ℝ) (gas_used : ℝ) : ℝ :=
  efficiency * gas_used

/-- Proves that Brian's car travels 60 miles given the conditions -/
theorem brians_trip_distance :
  let efficiency : ℝ := 20
  let gas_used : ℝ := 3
  distance_traveled efficiency gas_used = 60 := by
  sorry

end brians_trip_distance_l1331_133154


namespace complex_cube_sum_magnitude_l1331_133176

theorem complex_cube_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs (z₁ + z₂) = 20) 
  (h2 : Complex.abs (z₁^2 + z₂^2) = 16) : 
  Complex.abs (z₁^3 + z₂^3) = 3520 := by
  sorry

end complex_cube_sum_magnitude_l1331_133176


namespace area_of_triangle_AKF_l1331_133171

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define points A, B, K, and F
variable (A B K : ℝ × ℝ)
def F : ℝ × ℝ := focus

-- State that A is on the parabola
axiom A_on_parabola : parabola A.1 A.2

-- State that B is on the directrix
axiom B_on_directrix : directrix B.1

-- State that K is on the directrix
axiom K_on_directrix : directrix K.1

-- State that A, F, and B are collinear
axiom A_F_B_collinear : ∃ (t : ℝ), A = F + t • (B - F) ∨ B = F + t • (A - F)

-- State that AK is perpendicular to the directrix
axiom AK_perp_directrix : (A.1 - K.1) * 0 + (A.2 - K.2) * 1 = 0

-- State that |AF| = |BF|
axiom AF_eq_BF : (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2

-- Theorem to prove
theorem area_of_triangle_AKF : 
  (1/2) * abs ((A.1 - F.1) * (K.2 - F.2) - (K.1 - F.1) * (A.2 - F.2)) = 4 * Real.sqrt 3 :=
sorry

end area_of_triangle_AKF_l1331_133171


namespace inequality_proof_l1331_133101

theorem inequality_proof (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4/3 := by
sorry

end inequality_proof_l1331_133101


namespace average_children_in_families_with_children_l1331_133162

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (avg_children_all : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : avg_children_all = 3)
  (h3 : childless_families = 3)
  : (total_families * avg_children_all) / (total_families - childless_families) = 4 := by
  sorry

end average_children_in_families_with_children_l1331_133162


namespace parabola_vertex_l1331_133179

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 2(x-1)^2 + 2 is at the point (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
  sorry

end parabola_vertex_l1331_133179


namespace sum_of_polynomials_l1331_133120

/-- Given three polynomial functions f, g, and h, prove their sum equals a specific polynomial. -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => -4*x^2 + 2*x - 5
  let g := fun (x : ℝ) => -6*x^2 + 4*x - 9
  let h := fun (x : ℝ) => 6*x^2 + 6*x + 2
  f x + g x + h x = -4*x^2 + 12*x - 12 := by
  sorry

end sum_of_polynomials_l1331_133120


namespace max_value_sine_function_l1331_133109

theorem max_value_sine_function (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x ∈ Set.Icc 0 (π/3), 2 * Real.sin (ω * x) ≤ Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π/3), 2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3/4 := by
  sorry

end max_value_sine_function_l1331_133109


namespace f_has_max_and_min_l1331_133170

-- Define the function
def f (x : ℝ) : ℝ := -x^3 - x^2 + 2

-- Theorem statement
theorem f_has_max_and_min :
  (∃ x_max : ℝ, ∀ x : ℝ, f x ≤ f x_max) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f x_min ≤ f x) :=
sorry

end f_has_max_and_min_l1331_133170


namespace water_depth_approx_0_6_l1331_133169

/-- Represents a horizontal cylindrical tank partially filled with water -/
structure WaterTank where
  length : ℝ
  diameter : ℝ
  exposedArea : ℝ

/-- Calculates the depth of water in the tank -/
def waterDepth (tank : WaterTank) : ℝ :=
  sorry

/-- Theorem stating that the water depth is approximately 0.6 feet for the given tank -/
theorem water_depth_approx_0_6 (tank : WaterTank) 
  (h1 : tank.length = 12)
  (h2 : tank.diameter = 8)
  (h3 : tank.exposedArea = 50) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |waterDepth tank - 0.6| < ε :=
  sorry

end water_depth_approx_0_6_l1331_133169


namespace arithmetic_geometric_sequence_l1331_133184

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  geometric_sequence (a 1) (a 2) (a 5) →
  d = 2 :=
by sorry

end arithmetic_geometric_sequence_l1331_133184


namespace black_hair_ratio_l1331_133193

/-- Represents the ratio of hair colors in the class -/
structure HairColorRatio :=
  (red : ℕ)
  (blonde : ℕ)
  (black : ℕ)

/-- Represents the class information -/
structure ClassInfo :=
  (ratio : HairColorRatio)
  (redHairedKids : ℕ)
  (totalKids : ℕ)

/-- The main theorem -/
theorem black_hair_ratio (c : ClassInfo) 
  (h1 : c.ratio = HairColorRatio.mk 3 6 7)
  (h2 : c.redHairedKids = 9)
  (h3 : c.totalKids = 48) : 
  (c.ratio.black * c.redHairedKids / c.ratio.red : ℚ) / c.totalKids = 7 / 16 := by
  sorry

#check black_hair_ratio

end black_hair_ratio_l1331_133193


namespace greatest_integer_value_five_satisfies_condition_no_greater_integer_greatest_integer_is_five_l1331_133114

theorem greatest_integer_value (x : ℤ) : (3 * Int.natAbs x + 4 ≤ 19) → x ≤ 5 :=
by sorry

theorem five_satisfies_condition : 3 * Int.natAbs 5 + 4 ≤ 19 :=
by sorry

theorem no_greater_integer (y : ℤ) : y > 5 → (3 * Int.natAbs y + 4 > 19) :=
by sorry

theorem greatest_integer_is_five : 
  ∃ (x : ℤ), (3 * Int.natAbs x + 4 ≤ 19) ∧ (∀ (y : ℤ), (3 * Int.natAbs y + 4 ≤ 19) → y ≤ x) ∧ x = 5 :=
by sorry

end greatest_integer_value_five_satisfies_condition_no_greater_integer_greatest_integer_is_five_l1331_133114


namespace cubic_root_sum_l1331_133106

/-- Given a cubic equation x³ + px + q = 0 where p and q are real numbers,
    if 2 + i is a root, then p + q = 9 -/
theorem cubic_root_sum (p q : ℝ) : 
  (Complex.I : ℂ) ^ 3 + p * (Complex.I : ℂ) + q = 0 → p + q = 9 := by
  sorry

end cubic_root_sum_l1331_133106


namespace alex_shirt_count_l1331_133174

/-- Given that:
  - Alex has some new shirts
  - Joe has 3 more new shirts than Alex
  - Ben has a certain number of new shirts more than Joe
  - Ben has 15 new shirts
Prove that Alex has 12 new shirts. -/
theorem alex_shirt_count :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
  joe_shirts = alex_shirts + 3 →
  ben_shirts > joe_shirts →
  ben_shirts = 15 →
  alex_shirts = 12 := by
sorry

end alex_shirt_count_l1331_133174


namespace probability_n_power_16_mod_6_equals_1_l1331_133102

theorem probability_n_power_16_mod_6_equals_1 (N : ℕ) (h : 1 ≤ N ∧ N ≤ 2000) :
  (Nat.card {n : ℕ | 1 ≤ n ∧ n ≤ 2000 ∧ n^16 % 6 = 1}) / 2000 = 1/2 := by
  sorry

end probability_n_power_16_mod_6_equals_1_l1331_133102


namespace hotel_revenue_calculation_l1331_133156

/-- The total revenue of a hotel for one night, given the number of single and double rooms booked and their respective prices. -/
def hotel_revenue (total_rooms single_price double_price double_rooms : ℕ) : ℕ :=
  let single_rooms := total_rooms - double_rooms
  single_rooms * single_price + double_rooms * double_price

/-- Theorem stating that under the given conditions, the hotel's revenue for one night is $14,000. -/
theorem hotel_revenue_calculation :
  hotel_revenue 260 35 60 196 = 14000 := by
  sorry

end hotel_revenue_calculation_l1331_133156


namespace mathilda_debt_repayment_l1331_133167

/-- Mathilda's debt repayment problem -/
theorem mathilda_debt_repayment 
  (original_debt : ℝ) 
  (remaining_percentage : ℝ) 
  (initial_installment : ℝ) :
  original_debt = 500 ∧ 
  remaining_percentage = 75 ∧ 
  initial_installment = original_debt * (100 - remaining_percentage) / 100 →
  initial_installment = 125 := by
sorry

end mathilda_debt_repayment_l1331_133167


namespace rhombus_diagonals_not_always_equal_l1331_133150

/-- A rhombus is a quadrilateral with all four sides of equal length -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The diagonals of a rhombus -/
def diagonals (r : Rhombus) : ℝ × ℝ :=
  sorry

/-- Theorem: The diagonals of a rhombus are not always equal -/
theorem rhombus_diagonals_not_always_equal :
  ¬ (∀ r : Rhombus, (diagonals r).1 = (diagonals r).2) :=
sorry

end rhombus_diagonals_not_always_equal_l1331_133150


namespace min_value_of_a_l1331_133152

theorem min_value_of_a (a b c : ℝ) (ha : a > 0) (hroots : ∃ x y : ℝ, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) (hineq : ∀ c' : ℝ, c' ≥ 1 → 25 * a + 10 * b + 4 * c' ≥ 4) : a ≥ 16/25 := by
  sorry

end min_value_of_a_l1331_133152


namespace max_value_cos_sin_sum_l1331_133105

theorem max_value_cos_sin_sum :
  ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5 := by
  sorry

end max_value_cos_sin_sum_l1331_133105


namespace service_provider_selection_l1331_133177

theorem service_provider_selection (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 :=
by sorry

end service_provider_selection_l1331_133177


namespace inequality_solution_l1331_133141

theorem inequality_solution (y : ℝ) : 
  (1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4) ↔ 
  (y < -4 ∨ (-2 < y ∧ y < 0) ∨ 1 < y) :=
by sorry

end inequality_solution_l1331_133141


namespace smallest_enclosing_circle_l1331_133166

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := y - 2 = 0
def line3 (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  line1 A.1 A.2 ∧ line2 A.1 A.2 ∧
  line2 B.1 B.2 ∧ line3 B.1 B.2 ∧
  line1 C.1 C.2 ∧ line3 C.1 C.2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1.5)^2 = 1.25

-- Theorem statement
theorem smallest_enclosing_circle 
  (A B C : ℝ × ℝ) 
  (h : triangle A B C) :
  ∀ x y : ℝ, 
  (∀ px py : ℝ, (px = A.1 ∧ py = A.2) ∨ (px = B.1 ∧ py = B.2) ∨ (px = C.1 ∧ py = C.2) → 
    (x - px)^2 + (y - py)^2 ≤ 1.25) ↔ 
  circle_equation x y :=
sorry

end smallest_enclosing_circle_l1331_133166


namespace nine_times_polygon_properties_l1331_133148

/-- A polygon with interior angles 9 times the exterior angles -/
structure NineTimesPolygon where
  n : ℕ -- number of sides
  interior_angles : Fin n → ℝ
  exterior_angles : Fin n → ℝ
  h_positive : ∀ i, interior_angles i > 0 ∧ exterior_angles i > 0
  h_relation : ∀ i, interior_angles i = 9 * exterior_angles i
  h_exterior_sum : (Finset.univ.sum exterior_angles) = 360

theorem nine_times_polygon_properties (Q : NineTimesPolygon) :
  (Finset.univ.sum Q.interior_angles = 3240) ∧
  (∃ (i j : Fin Q.n), Q.interior_angles i ≠ Q.interior_angles j ∨ Q.interior_angles i = Q.interior_angles j) :=
by sorry

end nine_times_polygon_properties_l1331_133148


namespace triangle_side_lengths_l1331_133186

theorem triangle_side_lengths :
  ∀ x y z : ℕ,
    x ≥ y ∧ y ≥ z →
    x + y + z = 240 →
    3 * x - 2 * (y + z) = 5 * z + 10 →
    ((x = 113 ∧ y = 112 ∧ z = 15) ∨
     (x = 114 ∧ y = 110 ∧ z = 16) ∨
     (x = 115 ∧ y = 108 ∧ z = 17) ∨
     (x = 116 ∧ y = 106 ∧ z = 18) ∨
     (x = 117 ∧ y = 104 ∧ z = 19) ∨
     (x = 118 ∧ y = 102 ∧ z = 20) ∨
     (x = 119 ∧ y = 100 ∧ z = 21)) :=
by
  sorry

end triangle_side_lengths_l1331_133186


namespace opposite_sign_pair_l1331_133125

theorem opposite_sign_pair : ∃! (x : ℝ), (x > 0 ∧ x * x = 7) ∧ 
  (∀ a b : ℝ, (a = 131 ∧ b = 1 - 31) ∨ 
              (a = x ∧ b = -x) ∨ 
              (a = 1/3 ∧ b = Real.sqrt (1/9)) ∨ 
              (a = 5^2 ∧ b = (-5)^2) →
   (a + b = 0 ∧ a * b < 0) ↔ (a = x ∧ b = -x)) :=
by sorry

end opposite_sign_pair_l1331_133125


namespace exists_n_pow_half_n_eq_twelve_l1331_133195

theorem exists_n_pow_half_n_eq_twelve :
  ∃ n : ℝ, n > 0 ∧ n^(n/2) = 12 :=
by sorry

end exists_n_pow_half_n_eq_twelve_l1331_133195


namespace morgan_change_calculation_l1331_133153

/-- Calculates the change Morgan receives after buying lunch items and paying with a $50 bill. -/
theorem morgan_change_calculation (hamburger onion_rings smoothie side_salad chocolate_cake : ℚ)
  (h1 : hamburger = 5.75)
  (h2 : onion_rings = 2.50)
  (h3 : smoothie = 3.25)
  (h4 : side_salad = 3.75)
  (h5 : chocolate_cake = 4.20) :
  50 - (hamburger + onion_rings + smoothie + side_salad + chocolate_cake) = 30.55 := by
  sorry

#eval 50 - (5.75 + 2.50 + 3.25 + 3.75 + 4.20)

end morgan_change_calculation_l1331_133153


namespace kyro_are_fylol_and_glyk_l1331_133118

-- Define the types
variable (U : Type) -- Universe of discourse
variable (Fylol Glyk Kyro Mylo : Set U)

-- State the given conditions
variable (h1 : Fylol ⊆ Glyk)
variable (h2 : Kyro ⊆ Glyk)
variable (h3 : Mylo ⊆ Fylol)
variable (h4 : Kyro ⊆ Mylo)

-- Theorem to prove
theorem kyro_are_fylol_and_glyk : Kyro ⊆ Fylol ∩ Glyk := by sorry

end kyro_are_fylol_and_glyk_l1331_133118


namespace summer_determination_l1331_133196

def has_entered_summer (temperatures : List ℤ) : Prop :=
  temperatures.length = 5 ∧ ∀ t ∈ temperatures, t ≥ 22

def median (l : List ℤ) : ℤ := sorry
def mode (l : List ℤ) : ℤ := sorry
def mean (l : List ℤ) : ℚ := sorry
def variance (l : List ℤ) : ℚ := sorry

theorem summer_determination :
  ∀ (temps_A temps_B temps_C temps_D : List ℤ),
    (median temps_A = 24 ∧ mode temps_A = 22) →
    (median temps_B = 25 ∧ mean temps_B = 24) →
    (mean temps_C = 22 ∧ mode temps_C = 22) →
    (28 ∈ temps_D ∧ mean temps_D = 24 ∧ variance temps_D = 4.8) →
    (has_entered_summer temps_A ∧
     has_entered_summer temps_D ∧
     ¬(has_entered_summer temps_B ∧ has_entered_summer temps_C)) :=
by sorry

end summer_determination_l1331_133196


namespace perimeter_APR_is_50_l1331_133108

/-- A circle with two tangents from an exterior point A touching at B and C,
    and a third tangent touching at Q and intersecting AB at P and AC at R. -/
structure TangentCircle where
  /-- The length of tangent AB -/
  AB : ℝ
  /-- The distance from A to Q along the tangent -/
  AQ : ℝ

/-- The perimeter of triangle APR in the TangentCircle configuration -/
def perimeterAPR (tc : TangentCircle) : ℝ :=
  tc.AB - tc.AQ + tc.AQ + tc.AQ

/-- Theorem stating that for a TangentCircle with AB = 25 and AQ = 12.5,
    the perimeter of triangle APR is 50 -/
theorem perimeter_APR_is_50 (tc : TangentCircle)
    (h1 : tc.AB = 25) (h2 : tc.AQ = 12.5) :
    perimeterAPR tc = 50 := by
  sorry

end perimeter_APR_is_50_l1331_133108


namespace twigs_to_find_l1331_133182

/-- The number of twigs already in the nest circle -/
def twigs_in_circle : ℕ := 12

/-- The number of additional twigs needed for each twig in the circle -/
def twigs_per_existing : ℕ := 6

/-- The fraction of needed twigs dropped by the tree -/
def tree_dropped_fraction : ℚ := 1/3

/-- Theorem stating how many twigs the bird still needs to find -/
theorem twigs_to_find : 
  (twigs_in_circle * twigs_per_existing : ℕ) - 
  (twigs_in_circle * twigs_per_existing : ℕ) * tree_dropped_fraction = 48 := by
  sorry

end twigs_to_find_l1331_133182


namespace sine_inequality_l1331_133175

theorem sine_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) := by
  sorry

end sine_inequality_l1331_133175


namespace rotate_A_180_l1331_133183

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Rotates a point 180 degrees clockwise about the origin -/
def rotate180 (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The original point A -/
def A : Point := { x := -4, y := 1 }

/-- The expected result after rotation -/
def A_rotated : Point := { x := 4, y := -1 }

/-- Theorem stating that rotating A 180 degrees clockwise about the origin results in A_rotated -/
theorem rotate_A_180 : rotate180 A = A_rotated := by sorry

end rotate_A_180_l1331_133183


namespace certain_number_problem_l1331_133132

theorem certain_number_problem (x : ℤ) (h : x + 5 * 8 = 340) : x = 300 := by
  sorry

end certain_number_problem_l1331_133132


namespace total_spent_is_638_l1331_133123

/-- The total amount spent by Elizabeth, Emma, and Elsa -/
def total_spent (emma_spent : ℕ) : ℕ :=
  let elsa_spent := 2 * emma_spent
  let elizabeth_spent := 4 * elsa_spent
  emma_spent + elsa_spent + elizabeth_spent

/-- Theorem stating that the total amount spent is 638 -/
theorem total_spent_is_638 : total_spent 58 = 638 := by
  sorry

end total_spent_is_638_l1331_133123


namespace f_increasing_implies_F_decreasing_l1331_133130

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Definition of F in terms of f -/
def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (1 - x) - f (1 + x)

/-- A function f is decreasing on ℝ -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem f_increasing_implies_F_decreasing (f : ℝ → ℝ) (h : IsIncreasing f) : IsDecreasing (F f) := by
  sorry

end f_increasing_implies_F_decreasing_l1331_133130
