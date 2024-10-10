import Mathlib

namespace mary_baseball_cards_l879_87928

theorem mary_baseball_cards (x : ℕ) : 
  x - 8 + 26 + 40 = 84 → x = 26 := by
  sorry

end mary_baseball_cards_l879_87928


namespace average_page_count_l879_87940

theorem average_page_count (n : ℕ) (g1 g2 g3 : ℕ) (p1 p2 p3 : ℕ) :
  n = g1 + g2 + g3 →
  g1 = g2 →
  g2 = g3 →
  g1 = 5 →
  p1 = 2 →
  p2 = 3 →
  p3 = 1 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 := by
  sorry

end average_page_count_l879_87940


namespace sin_negative_1740_degrees_l879_87965

theorem sin_negative_1740_degrees : Real.sin (-(1740 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end sin_negative_1740_degrees_l879_87965


namespace students_left_is_30_percent_l879_87911

/-- The percentage of students left in a classroom after some students leave for activities -/
def students_left_percentage (total : ℕ) (painting : ℚ) (playing : ℚ) (workshop : ℚ) : ℚ :=
  (1 - (painting + playing + workshop)) * 100

/-- Theorem: Given the conditions, the percentage of students left in the classroom is 30% -/
theorem students_left_is_30_percent :
  students_left_percentage 250 (3/10) (2/10) (1/5) = 30 := by
  sorry

end students_left_is_30_percent_l879_87911


namespace expression_evaluation_l879_87988

theorem expression_evaluation (x : ℝ) (h : x = 2) : 
  (x - 1) / (x - 2) * ((x^2 - 4) / (x^2 - 2*x + 1)) - 2 / (x - 1) = 2 := by
  sorry

end expression_evaluation_l879_87988


namespace regions_99_lines_l879_87980

/-- The number of regions created by lines in a plane -/
def num_regions (num_lines : ℕ) (all_parallel : Bool) (all_intersect_one_point : Bool) : ℕ :=
  if all_parallel then
    num_lines + 1
  else if all_intersect_one_point then
    2 * num_lines
  else
    0  -- This case is not used in our theorem, but included for completeness

/-- Theorem stating the possible number of regions created by 99 lines in a plane -/
theorem regions_99_lines :
  ∀ (n : ℕ), n < 199 →
  (∃ (all_parallel all_intersect_one_point : Bool),
    num_regions 99 all_parallel all_intersect_one_point = n) →
  (n = 100 ∨ n = 198) :=
by
  sorry

#check regions_99_lines

end regions_99_lines_l879_87980


namespace fraction_product_cube_l879_87968

theorem fraction_product_cube (x : ℝ) (hx : x ≠ 0) : 
  (8 / 9)^3 * (x / 3)^3 * (3 / x)^3 = 512 / 729 := by
  sorry

end fraction_product_cube_l879_87968


namespace sum_difference_1500_l879_87920

/-- The sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * n

/-- The sum of the first n even counting numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sumDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem sum_difference_1500 : sumDifference 1500 = 1500 := by
  sorry

end sum_difference_1500_l879_87920


namespace geometric_sequence_eighth_term_l879_87985

theorem geometric_sequence_eighth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^5 = 32) 
  (h3 : r > 0) : 
  a * r^7 = 4 := by
sorry

end geometric_sequence_eighth_term_l879_87985


namespace common_ratio_of_geometric_series_l879_87972

def geometric_series (n : ℕ) : ℚ := (7 / 3) * (7 / 3) ^ n

theorem common_ratio_of_geometric_series :
  ∀ n : ℕ, geometric_series (n + 1) / geometric_series n = 7 / 3 :=
by
  sorry

#check common_ratio_of_geometric_series

end common_ratio_of_geometric_series_l879_87972


namespace certain_amount_less_than_twice_l879_87983

theorem certain_amount_less_than_twice (n : ℤ) (x : ℤ) : n = 16 ∧ 2 * n - x = 20 → x = 12 := by
  sorry

end certain_amount_less_than_twice_l879_87983


namespace digit_79_is_2_l879_87914

/-- The sequence of digits formed by writing consecutive integers from 65 to 1 in descending order -/
def descending_sequence : List Nat := sorry

/-- The 79th digit in the descending sequence -/
def digit_79 : Nat := sorry

/-- Theorem stating that the 79th digit in the descending sequence is 2 -/
theorem digit_79_is_2 : digit_79 = 2 := by sorry

end digit_79_is_2_l879_87914


namespace inverse_proportion_y_relationship_l879_87913

/-- Given two points A(-3, y₁) and B(2, y₂) on the graph of y = 6/x, prove that y₁ < y₂ -/
theorem inverse_proportion_y_relationship (y₁ y₂ : ℝ) : 
  y₁ = 6 / (-3) → y₂ = 6 / 2 → y₁ < y₂ := by
  sorry


end inverse_proportion_y_relationship_l879_87913


namespace stripe_area_on_cylinder_l879_87909

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylinder 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40)
  (h2 : stripe_width = 2)
  (h3 : revolutions = 3) :
  stripe_width * revolutions * π * diameter = 240 * π := by
sorry

end stripe_area_on_cylinder_l879_87909


namespace specific_ohara_triple_l879_87982

/-- O'Hara triple condition -/
def is_ohara_triple (a b x : ℝ) : Prop := Real.sqrt a + Real.sqrt b = x

/-- Proof of the specific O'Hara triple -/
theorem specific_ohara_triple :
  let a : ℝ := 49
  let b : ℝ := 16
  ∃ x : ℝ, is_ohara_triple a b x ∧ x = 11 := by
  sorry

end specific_ohara_triple_l879_87982


namespace aunt_gemma_feeding_times_l879_87957

/-- Calculates the number of times Aunt Gemma feeds her dogs per day -/
def feeding_times_per_day (num_dogs : ℕ) (food_per_meal : ℕ) (num_sacks : ℕ) (sack_weight : ℕ) (days : ℕ) : ℕ :=
  let total_food := num_sacks * sack_weight * 1000
  let food_per_day := total_food / days
  let food_per_dog_per_day := food_per_day / num_dogs
  food_per_dog_per_day / food_per_meal

theorem aunt_gemma_feeding_times : 
  feeding_times_per_day 4 250 2 50 50 = 2 := by sorry

end aunt_gemma_feeding_times_l879_87957


namespace pentagon_area_half_octagon_l879_87976

/-- Regular octagon with vertices labeled CHILDREN -/
structure RegularOctagon :=
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : sorry)

/-- Pentagon formed by 5 consecutive vertices of the octagon -/
def Pentagon (o : RegularOctagon) : Set (ℝ × ℝ) :=
  {p | ∃ i : Fin 5, p = o.vertices i}

/-- Area of a shape in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

theorem pentagon_area_half_octagon (o : RegularOctagon) 
  (h : area {p | ∃ i : Fin 8, p = o.vertices i} = 1) : 
  area (Pentagon o) = 1/2 := by sorry

end pentagon_area_half_octagon_l879_87976


namespace sin_pi_minus_alpha_l879_87959

theorem sin_pi_minus_alpha (α : Real) : 
  (∃ (x y : Real), x = Real.sqrt 3 ∧ y = 1 ∧ x = Real.tan α * y) →
  Real.sin (Real.pi - α) = 1/2 := by
  sorry

end sin_pi_minus_alpha_l879_87959


namespace plate_color_probability_l879_87969

/-- The probability of selecting two plates of the same color -/
def same_color_probability (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  let same_color_combinations := (red.choose 2) + (blue.choose 2) + (yellow.choose 2)
  let total_combinations := total.choose 2
  same_color_combinations / total_combinations

/-- Theorem: The probability of selecting two plates of the same color
    given 7 red plates, 5 blue plates, and 3 yellow plates is 34/105 -/
theorem plate_color_probability :
  same_color_probability 7 5 3 = 34 / 105 := by
  sorry

end plate_color_probability_l879_87969


namespace box_ball_count_l879_87956

theorem box_ball_count (red_balls : ℕ) (red_prob : ℚ) (total_balls : ℕ) : 
  red_balls = 12 → red_prob = 3/5 → (red_balls : ℚ) / total_balls = red_prob → total_balls = 20 := by
  sorry

end box_ball_count_l879_87956


namespace youngest_child_age_l879_87991

theorem youngest_child_age (age1 age2 age3 : ℕ) : 
  age1 < age2 ∧ age2 < age3 →
  6 + (0.60 * (age1 + age2 + age3 : ℝ)) + (3 * 0.90) = 15.30 →
  age1 = 1 :=
sorry

end youngest_child_age_l879_87991


namespace linear_systems_solutions_l879_87934

theorem linear_systems_solutions :
  -- System 1
  (2 : ℝ) + 2 * (1 : ℝ) = (4 : ℝ) ∧
  (2 : ℝ) + 3 * (1 : ℝ) = (5 : ℝ) ∧
  -- System 2
  2 * (2 : ℝ) - 5 * (5 : ℝ) = (-21 : ℝ) ∧
  4 * (2 : ℝ) + 3 * (5 : ℝ) = (23 : ℝ) :=
by sorry

end linear_systems_solutions_l879_87934


namespace wheel_radius_increase_l879_87900

/-- Calculates the increase in wheel radius given original and new odometer readings -/
theorem wheel_radius_increase
  (original_radius : ℝ)
  (original_reading : ℝ)
  (new_reading : ℝ)
  (inches_per_mile : ℝ)
  (h1 : original_radius = 16)
  (h2 : original_reading = 1000)
  (h3 : new_reading = 980)
  (h4 : inches_per_mile = 62560) :
  ∃ (increase : ℝ), abs (increase - 0.33) < 0.005 :=
by sorry

end wheel_radius_increase_l879_87900


namespace inequality_solution_set_l879_87961

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 0 → 
  2 - x ≥ 0 → 
  (((Real.sqrt (2 - x) + 4 * x - 3) / x ≥ 2) ↔ (x < 0 ∨ (1 ≤ x ∧ x ≤ 2))) :=
by sorry

end inequality_solution_set_l879_87961


namespace gcd_7584_18027_l879_87924

theorem gcd_7584_18027 : Nat.gcd 7584 18027 = 3 := by
  sorry

end gcd_7584_18027_l879_87924


namespace largest_divisor_of_expression_l879_87927

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (8*x + 4) * (8*x + 8) * (4*x + 2) = 384 * k) ∧
  (∀ (d : ℤ), d > 384 → ¬(∀ (y : ℤ), Odd y → ∃ (m : ℤ), (8*y + 4) * (8*y + 8) * (4*y + 2) = d * m)) :=
sorry

end largest_divisor_of_expression_l879_87927


namespace exists_special_function_l879_87997

def I : Set ℝ := Set.Icc (-1) 1

def is_piecewise_continuous (f : ℝ → ℝ) : Prop :=
  ∃ (s : Set ℝ), Set.Finite s ∧
  ∀ x ∈ I, x ∉ s → ∃ ε > 0, ∀ y ∈ I, |y - x| < ε → f y = f x

theorem exists_special_function :
  ∃ f : ℝ → ℝ,
    (∀ x ∈ I, f (f x) = -x) ∧
    (∀ x ∉ I, f x = 0) ∧
    is_piecewise_continuous f :=
sorry

end exists_special_function_l879_87997


namespace first_platform_length_l879_87979

/-- Given a train and two platforms, calculates the length of the first platform. -/
theorem first_platform_length 
  (train_length : ℝ) 
  (first_platform_time : ℝ) 
  (second_platform_length : ℝ) 
  (second_platform_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : first_platform_time = 15)
  (h3 : second_platform_length = 500)
  (h4 : second_platform_time = 20) :
  ∃ L : ℝ, (L + train_length) / first_platform_time = 
           (second_platform_length + train_length) / second_platform_time ∧ 
           L = 350 :=
by sorry

end first_platform_length_l879_87979


namespace parabola_focus_coordinates_l879_87995

/-- Given a parabola with equation x = (1/(4*m)) * y^2, prove that its focus has coordinates (m, 0) -/
theorem parabola_focus_coordinates (m : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | x = (1 / (4 * m)) * y^2}
  let focus := (m, 0)
  focus ∈ parabola ∧ ∀ p ∈ parabola, ∃ d : ℝ, (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = d^2 :=
by sorry

end parabola_focus_coordinates_l879_87995


namespace binomial_equation_unique_solution_l879_87931

theorem binomial_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13) ∧ n = 13 := by
  sorry

end binomial_equation_unique_solution_l879_87931


namespace final_price_calculation_l879_87901

/-- Calculates the final price of a set containing coffee, cheesecake, and sandwich -/
theorem final_price_calculation (coffee_price cheesecake_price sandwich_price : ℝ)
  (coffee_discount : ℝ) (additional_discount : ℝ) :
  coffee_price = 6 →
  cheesecake_price = 10 →
  sandwich_price = 8 →
  coffee_discount = 0.25 * coffee_price →
  additional_discount = 3 →
  (coffee_price - coffee_discount + cheesecake_price + sandwich_price) - additional_discount = 19.5 :=
by sorry

end final_price_calculation_l879_87901


namespace smallest_valid_n_l879_87949

def is_valid_arrangement (n : ℕ) (a : Fin 9 → ℕ) : Prop :=
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i j, (i.val - j.val) % 9 ≥ 2 → n ∣ a i * a j) ∧
  (∀ i, ¬(n ∣ a i * a (i + 1)))

theorem smallest_valid_n : 
  (∃ (a : Fin 9 → ℕ), is_valid_arrangement 485100 a) ∧
  (∀ n < 485100, ¬∃ (a : Fin 9 → ℕ), is_valid_arrangement n a) :=
sorry

end smallest_valid_n_l879_87949


namespace netPopulationIncreaseIs345600_l879_87989

/-- Calculates the net population increase in one day given birth and death rates -/
def netPopulationIncreaseInOneDay (birthRate : ℕ) (deathRate : ℕ) : ℕ :=
  let netIncreasePerTwoSeconds := birthRate - deathRate
  let netIncreasePerSecond := netIncreasePerTwoSeconds / 2
  let secondsInDay : ℕ := 24 * 60 * 60
  netIncreasePerSecond * secondsInDay

/-- Theorem stating that the net population increase in one day is 345,600 given the specified birth and death rates -/
theorem netPopulationIncreaseIs345600 :
  netPopulationIncreaseInOneDay 10 2 = 345600 := by
  sorry

#eval netPopulationIncreaseInOneDay 10 2

end netPopulationIncreaseIs345600_l879_87989


namespace CH4_yield_is_zero_l879_87904

-- Define the molecules and their amounts
structure Molecule :=
  (C : ℕ) (H : ℕ) (O : ℕ)

-- Define the reactions
def reaction_CH4 (m : Molecule) : Molecule :=
  {C := m.C - 1, H := m.H - 4, O := m.O}

def reaction_CO2 (m : Molecule) : Molecule :=
  {C := m.C - 1, H := m.H, O := m.O - 2}

def reaction_H2O (m : Molecule) : Molecule :=
  {C := m.C, H := m.H - 4, O := m.O - 2}

-- Define the initial amounts
def initial_amounts : Molecule :=
  {C := 3, H := 12, O := 8}  -- 3 moles C, 6 moles H2 (12 H atoms), 4 moles O2 (8 O atoms)

-- Define the theoretical yield of CH4
def theoretical_yield_CH4 (m : Molecule) : ℕ :=
  min m.C (m.H / 4)

-- Theorem statement
theorem CH4_yield_is_zero :
  theoretical_yield_CH4 (reaction_H2O (reaction_CO2 initial_amounts)) = 0 :=
sorry

end CH4_yield_is_zero_l879_87904


namespace intended_number_is_five_l879_87915

theorem intended_number_is_five : ∃! x : ℚ, (((3 * x * 10 + 2) / 19) + 7) = 3 * x := by
  sorry

end intended_number_is_five_l879_87915


namespace wire_circle_square_area_l879_87950

/-- The area of a square formed by a wire that can also form a circle of radius 56 cm is 784π² cm² -/
theorem wire_circle_square_area :
  let r : ℝ := 56  -- radius of the circle in cm
  let circle_circumference : ℝ := 2 * Real.pi * r
  let square_side : ℝ := circle_circumference / 4
  let square_area : ℝ := square_side * square_side
  square_area = 784 * Real.pi ^ 2 := by
    sorry

end wire_circle_square_area_l879_87950


namespace candidate_vote_percentage_l879_87967

theorem candidate_vote_percentage
  (total_votes : ℝ)
  (vote_difference : ℝ)
  (h_total : total_votes = 25000.000000000007)
  (h_diff : vote_difference = 5000) :
  let candidate_percentage := (total_votes - vote_difference) / (2 * total_votes) * 100
  candidate_percentage = 40 := by
sorry

end candidate_vote_percentage_l879_87967


namespace complex_magnitude_equation_l879_87907

theorem complex_magnitude_equation (t : ℝ) : t > 2 ∧ 
  Complex.abs (t + 4 * Complex.I * Real.sqrt 3) * Complex.abs (7 - 2 * Complex.I) = 17 * Real.sqrt 13 ↔ 
  t = Real.sqrt (1213 / 53) := by
sorry

end complex_magnitude_equation_l879_87907


namespace range_of_2x_minus_y_l879_87941

theorem range_of_2x_minus_y (x y : ℝ) (hx : 2 < x ∧ x < 4) (hy : -1 < y ∧ y < 3) :
  1 < 2 * x - y ∧ 2 * x - y < 9 := by
  sorry

end range_of_2x_minus_y_l879_87941


namespace inequality_solution_implies_m_range_l879_87937

theorem inequality_solution_implies_m_range (m : ℝ) : 
  (∀ x, (m - 1) * x > m - 1 ↔ x < 1) → m < 1 := by
  sorry

end inequality_solution_implies_m_range_l879_87937


namespace angle_three_measure_l879_87951

def minutes_to_degrees (m : ℕ) : ℚ := m / 60

def angle_measure (degrees : ℕ) (minutes : ℕ) : ℚ := degrees + minutes_to_degrees minutes

theorem angle_three_measure 
  (angle1 angle2 angle3 : ℚ) 
  (h1 : angle1 + angle2 = 90) 
  (h2 : angle2 + angle3 = 180) 
  (h3 : angle1 = angle_measure 67 12) : 
  angle3 = angle_measure 157 12 := by
sorry

end angle_three_measure_l879_87951


namespace triple_consecutive_primes_l879_87932

theorem triple_consecutive_primes (p : ℤ) : 
  (Nat.Prime p.natAbs ∧ Nat.Prime (p + 2).natAbs ∧ Nat.Prime (p + 4).natAbs) ↔ p = 3 :=
sorry

end triple_consecutive_primes_l879_87932


namespace mr_A_net_gain_l879_87994

def initial_cash_A : ℕ := 20000
def initial_house_value : ℕ := 20000
def initial_car_value : ℕ := 5000
def initial_cash_B : ℕ := 25000

def house_sale_price : ℕ := 21000
def car_sale_price : ℕ := 4500

def house_buyback_price : ℕ := 19000
def car_depreciation_rate : ℚ := 1/10
def car_buyback_price : ℕ := 4050

theorem mr_A_net_gain :
  let first_transaction_cash_A := initial_cash_A + house_sale_price + car_sale_price
  let second_transaction_cash_A := first_transaction_cash_A - house_buyback_price - car_buyback_price
  second_transaction_cash_A - initial_cash_A = 2000 := by sorry

end mr_A_net_gain_l879_87994


namespace circle_extrema_l879_87942

theorem circle_extrema (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 6) :
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ y₀ / x₀ = 3 + 2 * Real.sqrt 2 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → y₁ / x₁ ≤ 3 + 2 * Real.sqrt 2) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ y₀ / x₀ = 3 - 2 * Real.sqrt 2 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → y₁ / x₁ ≥ 3 - 2 * Real.sqrt 2) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ Real.sqrt ((x₀ - 2)^2 + y₀^2) = Real.sqrt 10 + Real.sqrt 6 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → Real.sqrt ((x₁ - 2)^2 + y₁^2) ≤ Real.sqrt 10 + Real.sqrt 6) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ Real.sqrt ((x₀ - 2)^2 + y₀^2) = Real.sqrt 10 - Real.sqrt 6 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → Real.sqrt ((x₁ - 2)^2 + y₁^2) ≥ Real.sqrt 10 - Real.sqrt 6) :=
by sorry

end circle_extrema_l879_87942


namespace rectangle_diagonal_rectangle_diagonal_proof_l879_87970

/-- The length of the diagonal of a rectangle with length 100 and width 100√2 is 100√3 -/
theorem rectangle_diagonal : Real → Prop :=
  fun d =>
    let length := 100
    let width := 100 * Real.sqrt 2
    d = 100 * Real.sqrt 3 ∧ d^2 = length^2 + width^2

/-- Proof of the theorem -/
theorem rectangle_diagonal_proof : ∃ d, rectangle_diagonal d := by
  sorry

end rectangle_diagonal_rectangle_diagonal_proof_l879_87970


namespace austonHeightCm_l879_87966

/-- Converts inches to centimeters -/
def inchesToCm (inches : ℝ) : ℝ := inches * 2.54

/-- Auston's height in inches -/
def austonHeightInches : ℝ := 60

/-- Theorem stating Auston's height in centimeters -/
theorem austonHeightCm : inchesToCm austonHeightInches = 152.4 := by
  sorry

end austonHeightCm_l879_87966


namespace triangle_side_equations_l879_87992

theorem triangle_side_equations (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ¬(∃ x y z : ℝ, x^2 - 2*b*x + 2*a*c = 0 ∧ y^2 - 2*c*y + 2*a*b = 0 ∧ z^2 - 2*a*z + 2*b*c = 0) :=
by sorry

end triangle_side_equations_l879_87992


namespace inequalities_representation_l879_87954

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a shape in 2D space -/
class Shape where
  contains : Point2D → Prop

/-- Diamond shape defined by |x| + |y| ≤ r -/
def Diamond (r : ℝ) : Shape where
  contains p := abs p.x + abs p.y ≤ r

/-- Circle shape defined by x² + y² ≤ r² -/
def Circle (r : ℝ) : Shape where
  contains p := p.x^2 + p.y^2 ≤ r^2

/-- Hexagon shape defined by 3Max(|x|, |y|) ≤ r -/
def Hexagon (r : ℝ) : Shape where
  contains p := 3 * max (abs p.x) (abs p.y) ≤ r

/-- Theorem stating that the inequalities represent the described geometric shapes -/
theorem inequalities_representation (r : ℝ) (p : Point2D) :
  (Diamond r).contains p → (Circle r).contains p → (Hexagon r).contains p :=
by sorry

end inequalities_representation_l879_87954


namespace function_form_l879_87953

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating the form of the function satisfying the equation -/
theorem function_form (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x :=
sorry

end function_form_l879_87953


namespace geometric_series_sum_l879_87971

theorem geometric_series_sum :
  let a : ℕ := 1  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  let series_sum := (a * (r^n - 1)) / (r - 1)
  series_sum = 3280 := by sorry

end geometric_series_sum_l879_87971


namespace cameron_donation_ratio_l879_87916

theorem cameron_donation_ratio :
  let boris_initial : ℕ := 24
  let boris_donation_fraction : ℚ := 1/4
  let cameron_initial : ℕ := 30
  let total_after_donation : ℕ := 38
  let boris_after := boris_initial - boris_initial * boris_donation_fraction
  let cameron_after := total_after_donation - boris_after
  let cameron_donated := cameron_initial - cameron_after
  cameron_donated / cameron_initial = 1/3 := by
sorry

end cameron_donation_ratio_l879_87916


namespace sum_of_digits_of_10_pow_97_minus_97_l879_87960

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the sum of digits of 10^97 - 97 is 858 -/
theorem sum_of_digits_of_10_pow_97_minus_97 : sum_of_digits (10^97 - 97) = 858 := by sorry

end sum_of_digits_of_10_pow_97_minus_97_l879_87960


namespace inequality_solution_l879_87973

theorem inequality_solution (x : ℝ) : 
  ((1 + x) / 2 - (2 * x + 1) / 3 ≤ 1) ↔ (x ≥ -5) :=
by sorry

end inequality_solution_l879_87973


namespace factorial_equation_solutions_l879_87943

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 3^x + 5^y + 14 = z! ↔ (x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) :=
by sorry

end factorial_equation_solutions_l879_87943


namespace square_gt_when_abs_lt_l879_87902

theorem square_gt_when_abs_lt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_gt_when_abs_lt_l879_87902


namespace product_357_sum_28_l879_87925

theorem product_357_sum_28 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 357 →
  (a : ℕ) + b + c + d = 28 := by
sorry

end product_357_sum_28_l879_87925


namespace geometric_sequence_first_term_l879_87945

/-- Given a geometric sequence {a_n} with sum S_n = 2010^n + t, prove a_1 = 2009 -/
theorem geometric_sequence_first_term (n : ℕ) (t : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ k, S k = 2010^k + t) →
  (a 1 * a 3 = (a 2)^2) →
  a 1 = 2009 :=
by sorry

end geometric_sequence_first_term_l879_87945


namespace y_work_time_l879_87955

/-- Given workers x, y, and z, and their work rates, prove that y alone takes 24 hours to complete the work. -/
theorem y_work_time (x y z : ℝ) (hx : x = 1 / 8) (hyz : y + z = 1 / 6) (hxz : x + z = 1 / 4) :
  1 / y = 24 := by
  sorry

end y_work_time_l879_87955


namespace ab_squared_equals_twelve_l879_87938

theorem ab_squared_equals_twelve (a b : ℝ) : (a + 2)^2 + |b - 3| = 0 → a^2 * b = 12 := by
  sorry

end ab_squared_equals_twelve_l879_87938


namespace ellipse_theorem_l879_87903

/-- Ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  right_focus_dist : ℝ → ℝ → ℝ
  h_focus : right_focus_dist 1 (-1) = 3
  h_point : -1^2 / a^2 + (-Real.sqrt 6 / 2)^2 / b^2 = 1

/-- Line l intersecting the ellipse -/
def Line (m t : ℝ) (x y : ℝ) : Prop :=
  x - m * y - t = 0

/-- Statement of the theorem -/
theorem ellipse_theorem (E : Ellipse) :
  E.a^2 = 4 ∧ E.b^2 = 2 ∧
  ∀ m t, ∃ M N : ℝ × ℝ,
    M ≠ N ∧
    M ≠ (-E.a, 0) ∧ N ≠ (-E.a, 0) ∧
    Line m t M.1 M.2 ∧ Line m t N.1 N.2 ∧
    M.1^2 / 4 + M.2^2 / 2 = 1 ∧
    N.1^2 / 4 + N.2^2 / 2 = 1 ∧
    ((M.1 + E.a)^2 + M.2^2) * ((N.1 + E.a)^2 + N.2^2) =
      ((M.1 - N.1)^2 + (M.2 - N.2)^2) * ((M.1 + N.1 + 2*E.a)^2 + (M.2 + N.2)^2) / 4 →
    t = -2/3 := by
  sorry

end ellipse_theorem_l879_87903


namespace centroid_tetrahedron_volume_ratio_l879_87952

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Calculates the centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- Checks if a point is inside a tetrahedron -/
def isInterior (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Main theorem: volume ratio of centroids' tetrahedron to original tetrahedron -/
theorem centroid_tetrahedron_volume_ratio 
  (ABCD : Tetrahedron) (P : Point3D) 
  (h : isInterior P ABCD) : 
  let G1 := centroid ⟨P, ABCD.A, ABCD.B, ABCD.C⟩
  let G2 := centroid ⟨P, ABCD.B, ABCD.C, ABCD.D⟩
  let G3 := centroid ⟨P, ABCD.C, ABCD.D, ABCD.A⟩
  let G4 := centroid ⟨P, ABCD.D, ABCD.A, ABCD.B⟩
  volume ⟨G1, G2, G3, G4⟩ / volume ABCD = 1 / 64 := by sorry

end centroid_tetrahedron_volume_ratio_l879_87952


namespace minimum_guests_l879_87978

theorem minimum_guests (total_food : ℝ) (max_individual_food : ℝ) (h1 : total_food = 520) (h2 : max_individual_food = 1.5) :
  ∃ n : ℕ, n * max_individual_food ≥ total_food ∧ ∀ m : ℕ, m * max_individual_food ≥ total_food → m ≥ n ∧ n = 347 :=
sorry

end minimum_guests_l879_87978


namespace min_d_value_l879_87990

theorem min_d_value (a b d : ℕ+) (h1 : a < b) (h2 : b < d + 1)
  (h3 : ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2023 ∧ 
       p.2 = |p.1 - a| + |p.1 - b| + |p.1 - (d + 1)|) :
  (∀ d' : ℕ+, d' ≥ d → 
    ∃ a' b' : ℕ+, a' < b' ∧ b' < d' + 1 ∧
    ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2023 ∧ 
    p.2 = |p.1 - a'| + |p.1 - b'| + |p.1 - (d' + 1)|) →
  d = 2020 := by
sorry

end min_d_value_l879_87990


namespace power_of_1024_is_16_l879_87987

theorem power_of_1024_is_16 :
  (1024 : ℝ) ^ (2/5 : ℝ) = 16 :=
by
  have h : 1024 = 2^10 := by norm_num
  sorry

end power_of_1024_is_16_l879_87987


namespace greatest_divisor_of_consecutive_multiples_of_four_l879_87999

theorem greatest_divisor_of_consecutive_multiples_of_four : ∃ (k : ℕ), 
  k > 0 ∧ 
  (∀ (n : ℕ), 
    (4*n * 4*(n+1) * 4*(n+2)) % k = 0) ∧
  (∀ (m : ℕ), 
    m > k → 
    ∃ (n : ℕ), (4*n * 4*(n+1) * 4*(n+2)) % m ≠ 0) ∧
  k = 768 :=
by sorry

end greatest_divisor_of_consecutive_multiples_of_four_l879_87999


namespace probability_of_draw_l879_87964

/-- Given two players A and B playing chess, this theorem proves the probability of a draw. -/
theorem probability_of_draw 
  (p_win : ℝ) 
  (p_not_lose : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.5 := by
  sorry

end probability_of_draw_l879_87964


namespace trees_on_rectangular_plot_l879_87948

/-- The number of trees planted on a rectangular plot -/
def num_trees (length width spacing : ℕ) : ℕ :=
  ((length / spacing) + 1) * ((width / spacing) + 1)

/-- Theorem: The number of trees planted at a five-foot distance from each other
    on a rectangular plot of land with sides 120 feet and 70 feet is 375 -/
theorem trees_on_rectangular_plot :
  num_trees 120 70 5 = 375 := by
  sorry

end trees_on_rectangular_plot_l879_87948


namespace problem_statements_l879_87923

theorem problem_statements :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) ∧
  (∀ P Q : Set ℝ, ∀ a : ℝ, a ∈ P ∩ Q → a ∈ P) ∧
  (∀ x : ℝ, (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)) ∧
  (∀ a b c : ℝ, (1 : ℝ) = 0 ↔ a + b + c = 0) :=
by sorry

end problem_statements_l879_87923


namespace similar_triangles_leg_sum_l879_87939

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a * b / 2 = 18) →  -- Area of smaller triangle
  (c * d / 2 = 288) →  -- Area of larger triangle
  (a^2 + b^2 = 10^2) →  -- Pythagorean theorem for smaller triangle
  (c / a = d / b) →  -- Similar triangles condition
  (c + d = 52) := by
sorry

end similar_triangles_leg_sum_l879_87939


namespace min_value_of_function_min_value_achievable_l879_87975

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x^2 + 3/x ≥ (3/2) * Real.rpow 18 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, x^2 + 3/x = (3/2) * Real.rpow 18 (1/3) :=
by sorry

end min_value_of_function_min_value_achievable_l879_87975


namespace solve_linear_equation_l879_87905

theorem solve_linear_equation (x : ℝ) (h : x + 1 = 2) : x = 1 := by
  sorry

end solve_linear_equation_l879_87905


namespace inequality_system_solution_set_l879_87919

theorem inequality_system_solution_set :
  ∀ x : ℝ, (2 * x ≤ -1 ∧ x > -1) ↔ (-1 < x ∧ x ≤ -1/2) :=
by sorry

end inequality_system_solution_set_l879_87919


namespace evaluate_expression_l879_87917

theorem evaluate_expression : 9^6 * 3^4 / 27^5 = 3 := by
  sorry

end evaluate_expression_l879_87917


namespace specific_weekly_profit_l879_87908

/-- Represents a business owner's financial situation --/
structure BusinessOwner where
  daily_earnings : ℕ
  weekly_rent : ℕ

/-- Calculates the weekly profit for a business owner --/
def weekly_profit (owner : BusinessOwner) : ℕ :=
  owner.daily_earnings * 7 - owner.weekly_rent

/-- Theorem stating that a business owner with specific earnings and rent has a weekly profit of $36 --/
theorem specific_weekly_profit :
  ∀ (owner : BusinessOwner),
    owner.daily_earnings = 8 →
    owner.weekly_rent = 20 →
    weekly_profit owner = 36 := by
  sorry

#eval weekly_profit { daily_earnings := 8, weekly_rent := 20 }

end specific_weekly_profit_l879_87908


namespace mentor_fraction_l879_87929

theorem mentor_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  n = 2 * s / 3 → (n / 2 + s / 3) / (n + s) = 2 / 5 := by
  sorry

end mentor_fraction_l879_87929


namespace circle_center_l879_87944

/-- The equation of a circle in the x-y plane --/
def CircleEquation (x y : ℝ) : Prop :=
  16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 80 = 0

/-- The center of a circle --/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem: The center of the circle with the given equation is (1, -2) --/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 1 ∧ c.y = -2 ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 0 :=
sorry

end circle_center_l879_87944


namespace fraction_of_product_l879_87926

theorem fraction_of_product (total : ℝ) (result : ℝ) : 
  total = 5020 →
  (3/4 : ℝ) * (1/2 : ℝ) * total = (3/4 : ℝ) * (1/2 : ℝ) * 5020 →
  result = 753.0000000000001 →
  (result / ((3/4 : ℝ) * (1/2 : ℝ) * total) : ℝ) = 0.4 :=
by sorry

end fraction_of_product_l879_87926


namespace sqrt_two_decomposition_l879_87936

theorem sqrt_two_decomposition :
  ∃ (a : ℤ) (b : ℝ), 
    (Real.sqrt 2 = a + b) ∧ 
    (0 ≤ b) ∧ 
    (b < 1) ∧ 
    (a = 1) ∧ 
    (1 / b = Real.sqrt 2 + 1) := by
  sorry

end sqrt_two_decomposition_l879_87936


namespace cost_per_credit_l879_87921

/-- Calculates the cost per credit given college expenses -/
theorem cost_per_credit
  (total_credits : ℕ)
  (cost_per_textbook : ℕ)
  (num_textbooks : ℕ)
  (facilities_fee : ℕ)
  (total_expenses : ℕ)
  (h1 : total_credits = 14)
  (h2 : cost_per_textbook = 120)
  (h3 : num_textbooks = 5)
  (h4 : facilities_fee = 200)
  (h5 : total_expenses = 7100) :
  (total_expenses - (cost_per_textbook * num_textbooks + facilities_fee)) / total_credits = 450 :=
by sorry

end cost_per_credit_l879_87921


namespace modulo_congruence_unique_solution_l879_87918

theorem modulo_congruence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 27514 [MOD 16] ∧ n = 10 := by
  sorry

end modulo_congruence_unique_solution_l879_87918


namespace square_of_negative_product_l879_87998

theorem square_of_negative_product (x y : ℝ) : (-x * y^2)^2 = x^2 * y^4 := by sorry

end square_of_negative_product_l879_87998


namespace pushups_sum_is_350_l879_87930

/-- The number of push-ups done by Zachary, David, and Emily -/
def total_pushups (zachary_pushups : ℕ) (david_extra : ℕ) : ℕ :=
  let david_pushups := zachary_pushups + david_extra
  let emily_pushups := 2 * david_pushups
  zachary_pushups + david_pushups + emily_pushups

/-- Theorem stating that the total number of push-ups is 350 -/
theorem pushups_sum_is_350 : total_pushups 44 58 = 350 := by
  sorry

end pushups_sum_is_350_l879_87930


namespace largest_valid_number_l879_87906

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a : Fin 10 → Fin 10),
    n = a 9 * 10^9 + a 8 * 10^8 + a 7 * 10^7 + a 6 * 10^6 + a 5 * 10^5 + 
        a 4 * 10^4 + a 3 * 10^3 + a 2 * 10^2 + a 1 * 10 + a 0 ∧
    ∀ i : Fin 10, (List.count (a i) (List.map a (List.range 10)) = a (9 - i))

def is_largest_valid_number (n : ℕ) : Prop :=
  is_valid_number n ∧ 
  ∀ m : ℕ, is_valid_number m → m ≤ n

theorem largest_valid_number : 
  is_largest_valid_number 8888228888 :=
sorry

end largest_valid_number_l879_87906


namespace original_selling_price_l879_87962

/-- The original selling price given the profit rates and price difference -/
theorem original_selling_price 
  (original_profit_rate : ℝ)
  (reduced_purchase_rate : ℝ)
  (new_profit_rate : ℝ)
  (price_difference : ℝ)
  (h1 : original_profit_rate = 0.1)
  (h2 : reduced_purchase_rate = 0.1)
  (h3 : new_profit_rate = 0.3)
  (h4 : price_difference = 49) :
  ∃ (purchase_price : ℝ),
    (1 + original_profit_rate) * purchase_price = 770 ∧
    ((1 - reduced_purchase_rate) * (1 + new_profit_rate) - (1 + original_profit_rate)) * purchase_price = price_difference :=
by sorry

end original_selling_price_l879_87962


namespace x_squared_minus_y_squared_l879_87986

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 10/21) 
  (h2 : x - y = 1/63) : 
  x^2 - y^2 = 10/1323 := by
sorry

end x_squared_minus_y_squared_l879_87986


namespace rectangle_area_difference_rectangle_area_difference_proof_l879_87974

theorem rectangle_area_difference : ℕ → Prop :=
  fun d => ∀ l w : ℕ,
    (l + w = 30) →  -- Perimeter condition: 2l + 2w = 60 simplified
    (∃ l' w' : ℕ, l' + w' = 30 ∧ l' * w' = l * w + d) →  -- Larger area exists
    (∀ l'' w'' : ℕ, l'' + w'' = 30 → l'' * w'' ≤ l * w + d) →  -- No larger area exists
    d = 196

-- The proof goes here
theorem rectangle_area_difference_proof : rectangle_area_difference 196 := by
  sorry

end rectangle_area_difference_rectangle_area_difference_proof_l879_87974


namespace cars_cannot_meet_l879_87947

-- Define the network structure
structure TriangleNetwork where
  -- Assume the network is infinite and regular
  -- Each vertex has exactly 6 edges connected to it
  vertex_degree : ℕ
  vertex_degree_eq : vertex_degree = 6

-- Define a car's position and movement
structure Car where
  position : ℕ × ℕ  -- Represent position as discrete coordinates
  direction : ℕ     -- 0, 1, or 2 representing the three possible directions

-- Define the movement options
inductive Move
  | straight
  | left
  | right

-- Function to update car position based on move
def update_position (c : Car) (m : Move) : Car :=
  sorry  -- Implementation details omitted for brevity

-- Theorem statement
theorem cars_cannot_meet 
  (network : TriangleNetwork) 
  (car1 car2 : Car) 
  (start_same_edge : car1.position.1 = car2.position.1 ∧ car1.direction = car2.direction)
  (t : ℕ) :
  ∀ (moves1 moves2 : List Move),
  moves1.length = t ∧ moves2.length = t →
  (moves1.foldl update_position car1).position ≠ (moves2.foldl update_position car2).position :=
sorry

end cars_cannot_meet_l879_87947


namespace triangle_inequality_l879_87935

/-- Prove that for positive integers x, y, z and angles α, β, γ in [0, π) where any two angles 
    sum to more than the third, the following inequality holds:
    √(x²+y²-2xy cos α) + √(y²+z²-2yz cos β) ≥ √(z²+x²-2zx cos γ) -/
theorem triangle_inequality (x y z : ℕ+) (α β γ : ℝ)
  (h_α : 0 ≤ α ∧ α < π)
  (h_β : 0 ≤ β ∧ β < π)
  (h_γ : 0 ≤ γ ∧ γ < π)
  (h_sum1 : α + β > γ)
  (h_sum2 : β + γ > α)
  (h_sum3 : γ + α > β) :
  Real.sqrt (x.val^2 + y.val^2 - 2*x.val*y.val*(Real.cos α)) + 
  Real.sqrt (y.val^2 + z.val^2 - 2*y.val*z.val*(Real.cos β)) ≥
  Real.sqrt (z.val^2 + x.val^2 - 2*z.val*x.val*(Real.cos γ)) := by
  sorry

end triangle_inequality_l879_87935


namespace last_problem_number_l879_87912

theorem last_problem_number 
  (start : ℕ) 
  (total : ℕ) 
  (h1 : start = 75) 
  (h2 : total = 51) : 
  start + total - 1 = 125 := by
sorry

end last_problem_number_l879_87912


namespace unique_positive_solution_l879_87981

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ Real.sqrt ((7 * x) / 3) = x :=
  ⟨7/3, by sorry⟩

end unique_positive_solution_l879_87981


namespace total_passengers_is_120_l879_87958

/-- The total number of passengers on the flight -/
def total_passengers : ℕ := 120

/-- The proportion of female passengers -/
def female_proportion : ℚ := 55 / 100

/-- The proportion of passengers in first class -/
def first_class_proportion : ℚ := 10 / 100

/-- The proportion of male passengers in first class -/
def male_first_class_proportion : ℚ := 1 / 3

/-- The number of females in coach class -/
def females_in_coach : ℕ := 58

/-- Theorem stating that the total number of passengers is 120 -/
theorem total_passengers_is_120 : 
  total_passengers = 120 ∧
  female_proportion * total_passengers = 
    (females_in_coach : ℚ) + 
    (1 - male_first_class_proportion) * first_class_proportion * total_passengers :=
by sorry

end total_passengers_is_120_l879_87958


namespace mean_equality_implies_x_value_mean_equality_proof_l879_87922

theorem mean_equality_implies_x_value : ℝ → Prop :=
  fun x =>
    (11 + 14 + 25) / 3 = (18 + x + 4) / 3 → x = 28

-- Proof
theorem mean_equality_proof : mean_equality_implies_x_value 28 := by
  sorry

end mean_equality_implies_x_value_mean_equality_proof_l879_87922


namespace inverse_trig_sum_equals_pi_l879_87946

theorem inverse_trig_sum_equals_pi : 
  Real.arctan (Real.sqrt 3) - Real.arcsin (-1/2) + Real.arccos 0 = π := by
  sorry

end inverse_trig_sum_equals_pi_l879_87946


namespace log_32_2_l879_87984

theorem log_32_2 : Real.log 2 / Real.log 32 = 1 / 5 := by
  have h : 32 = 2^5 := by sorry
  sorry

end log_32_2_l879_87984


namespace binomial_divisibility_l879_87977

theorem binomial_divisibility (n k : ℕ) (h : k ≤ n - 1) :
  (∀ k ≤ n - 1, n ∣ Nat.choose n k) ↔ Nat.Prime n :=
sorry

end binomial_divisibility_l879_87977


namespace quadrilateral_area_is_16_l879_87963

/-- A regular six-pointed star -/
structure RegularSixPointedStar :=
  (area : ℝ)
  (star_formed_by_two_triangles : Bool)
  (each_triangle_area : ℝ)

/-- The area of a quadrilateral formed by two adjacent points and the center of the star -/
def quadrilateral_area (star : RegularSixPointedStar) : ℝ := sorry

/-- Theorem stating the area of the quadrilateral is 16 cm² -/
theorem quadrilateral_area_is_16 (star : RegularSixPointedStar) 
  (h1 : star.star_formed_by_two_triangles = true) 
  (h2 : star.each_triangle_area = 72) : quadrilateral_area star = 16 := by
  sorry

end quadrilateral_area_is_16_l879_87963


namespace count_valid_formations_l879_87933

/-- The total number of musicians in the marching band -/
def total_musicians : ℕ := 420

/-- The minimum number of musicians per row -/
def min_per_row : ℕ := 12

/-- The maximum number of musicians per row -/
def max_per_row : ℕ := 50

/-- A function that checks if a pair of positive integers (s, t) forms a valid rectangular formation -/
def is_valid_formation (s t : ℕ+) : Prop :=
  s * t = total_musicians ∧ min_per_row ≤ t ∧ t ≤ max_per_row

/-- The theorem stating that there are exactly 8 valid rectangular formations -/
theorem count_valid_formations :
  ∃! (formations : Finset (ℕ+ × ℕ+)),
    formations.card = 8 ∧
    ∀ pair : ℕ+ × ℕ+, pair ∈ formations ↔ is_valid_formation pair.1 pair.2 :=
sorry

end count_valid_formations_l879_87933


namespace finite_decimal_fraction_condition_l879_87996

def is_finite_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ q = a / b ∧ ∀ p : ℕ, Nat.Prime p → p ∣ b → p = 2 ∨ p = 5

theorem finite_decimal_fraction_condition (n : ℕ) :
  n > 0 → (is_finite_decimal (1 / (n * (n + 1))) ↔ n = 1 ∨ n = 4) :=
by sorry

end finite_decimal_fraction_condition_l879_87996


namespace roots_sum_absolute_value_l879_87993

theorem roots_sum_absolute_value (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + x + m = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    |x₁| + |x₂| = 3) →
  m = -2 := by
sorry

end roots_sum_absolute_value_l879_87993


namespace corveus_sleep_deficit_l879_87910

/-- Calculates the total sleep deficit for Corveus in a week --/
def corveusWeeklySleepDeficit : ℤ :=
  let weekdaySleep : ℤ := 5 * 5  -- 4 hours night sleep + 1 hour nap, for 5 days
  let weekendSleep : ℤ := 5 * 2  -- 5 hours night sleep for 2 days
  let daylightSavingAdjustment : ℤ := 1  -- Extra hour due to daylight saving
  let midnightAwakenings : ℤ := 2  -- Loses 1 hour twice a week
  let actualSleep : ℤ := weekdaySleep + weekendSleep + daylightSavingAdjustment - midnightAwakenings
  let recommendedSleep : ℤ := 6 * 7  -- 6 hours per day for 7 days
  recommendedSleep - actualSleep

/-- Theorem stating that Corveus's weekly sleep deficit is 8 hours --/
theorem corveus_sleep_deficit : corveusWeeklySleepDeficit = 8 := by
  sorry

end corveus_sleep_deficit_l879_87910
