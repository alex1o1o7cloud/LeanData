import Mathlib

namespace least_common_multiple_first_ten_l1105_110519

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 :=
by sorry

end least_common_multiple_first_ten_l1105_110519


namespace a_to_b_equals_negative_one_l1105_110508

theorem a_to_b_equals_negative_one (a b : ℝ) (h : |a + 1| = -(b - 3)^2) : a^b = -1 := by
  sorry

end a_to_b_equals_negative_one_l1105_110508


namespace first_note_denomination_l1105_110504

/-- Proves that given the conditions of the problem, the denomination of the first type of notes must be 1 rupee -/
theorem first_note_denomination (total_amount : ℕ) (total_notes : ℕ) (x : ℕ) : 
  total_amount = 400 →
  total_notes = 75 →
  total_amount = (total_notes / 3 * x) + (total_notes / 3 * 5) + (total_notes / 3 * 10) →
  x = 1 := by
  sorry

#check first_note_denomination

end first_note_denomination_l1105_110504


namespace sqrt_prime_irrational_l1105_110566

theorem sqrt_prime_irrational (p : ℕ) (h : Prime p) : Irrational (Real.sqrt p) := by
  sorry

end sqrt_prime_irrational_l1105_110566


namespace triangle_50_40_l1105_110520

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a * b + (a - b) + 6

-- Theorem statement
theorem triangle_50_40 : triangle 50 40 = 2016 := by
  sorry

end triangle_50_40_l1105_110520


namespace sin_B_in_triangle_l1105_110554

theorem sin_B_in_triangle (A B C : Real) (AB BC : Real) :
  A = 2 * π / 3 →  -- 120° in radians
  AB = 5 →
  BC = 7 →
  Real.sin B = 3 * Real.sqrt 3 / 14 :=
by sorry

end sin_B_in_triangle_l1105_110554


namespace min_value_on_circle_l1105_110534

theorem min_value_on_circle :
  ∃ (min : ℝ), min = -5 ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → 3*x + 4*y ≥ min) ∧
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ 3*x + 4*y = min) := by
  sorry

end min_value_on_circle_l1105_110534


namespace coefficient_sum_equals_negative_eight_l1105_110502

/-- Given a polynomial equation, prove that a specific linear combination of its coefficients equals -8 -/
theorem coefficient_sum_equals_negative_eight 
  (a : Fin 9 → ℝ) 
  (h : ∀ x : ℝ, x^5 * (x+3)^3 = (a 8)*(x+1)^8 + (a 7)*(x+1)^7 + (a 6)*(x+1)^6 + 
                               (a 5)*(x+1)^5 + (a 4)*(x+1)^4 + (a 3)*(x+1)^3 + 
                               (a 2)*(x+1)^2 + (a 1)*(x+1) + (a 0)) : 
  7*(a 7) + 5*(a 5) + 3*(a 3) + (a 1) = -8 := by
  sorry

end coefficient_sum_equals_negative_eight_l1105_110502


namespace circle_center_radius_sum_l1105_110582

/-- Given a circle D defined by the equation x^2 - 20x + y^2 + 6y + 25 = 0,
    prove that the sum of its center coordinates and radius is 7 + √66 -/
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), x^2 - 20*x + y^2 + 6*y + 25 = 0 ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = 7 + Real.sqrt 66 :=
by sorry

end circle_center_radius_sum_l1105_110582


namespace hotel_charge_difference_l1105_110596

theorem hotel_charge_difference (G R P : ℝ) 
  (hR : R = G * 3.0000000000000006)
  (hP : P = G * 0.9) :
  (R - P) / R * 100 = 70 := by sorry

end hotel_charge_difference_l1105_110596


namespace building_C_floors_l1105_110515

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := floors_A + 9

/-- The number of floors in Building C -/
def floors_C : ℕ := 5 * floors_B - 6

theorem building_C_floors : floors_C = 59 := by
  sorry

end building_C_floors_l1105_110515


namespace largest_x_value_l1105_110539

theorem largest_x_value (x : ℝ) :
  (x / 7 + 3 / (7 * x) = 2 / 3) →
  x ≤ (7 + Real.sqrt 22) / 3 := by
  sorry

end largest_x_value_l1105_110539


namespace problem_statement_l1105_110567

theorem problem_statement (x y : ℝ) 
  (hx : x > 4) 
  (hy : y > 9) 
  (h : (Real.log x / Real.log 4)^4 + (Real.log y / Real.log 9)^4 + 18 = 18 * (Real.log x / Real.log 4) * (Real.log y / Real.log 9)) : 
  x^2 + y^2 = 4^(2 * Real.sqrt 3) + 9^(2 * Real.sqrt 3) := by
  sorry

end problem_statement_l1105_110567


namespace largest_percent_error_rectangular_plot_l1105_110521

theorem largest_percent_error_rectangular_plot (length width : ℝ) 
  (h_length : length = 15)
  (h_width : width = 10)
  (h_error : ℝ) (h_error_bound : h_error = 0.1) : 
  let actual_area := length * width
  let max_length := length * (1 + h_error)
  let max_width := width * (1 + h_error)
  let max_area := max_length * max_width
  let max_percent_error := (max_area - actual_area) / actual_area * 100
  max_percent_error = 21 := by sorry

end largest_percent_error_rectangular_plot_l1105_110521


namespace smallest_value_zero_l1105_110547

-- Define the function y
def y (x p q : ℝ) : ℝ := x^3 + x^2 + p*x + q

-- State the theorem
theorem smallest_value_zero (p : ℝ) :
  ∃ q : ℝ, (∀ x : ℝ, y x p q ≥ 0) ∧ (∃ x : ℝ, y x p q = 0) ∧ q = -2/27 :=
sorry

end smallest_value_zero_l1105_110547


namespace candy_probability_theorem_l1105_110552

-- Define the type for a packet of candies
structure Packet where
  blue : ℕ
  total : ℕ

-- Define the function to calculate the probability of drawing a blue candy from a box
def boxProbability (p1 p2 : Packet) : ℚ :=
  (p1.blue + p2.blue : ℚ) / (p1.total + p2.total : ℚ)

-- Theorem statement
theorem candy_probability_theorem :
  ∃ (p1 p2 p3 p4 : Packet),
    (boxProbability p1 p2 = 5/13 ∨ boxProbability p1 p2 = 7/18) ∧
    (boxProbability p3 p4 ≠ 17/40) ∧
    (∀ (p5 p6 : Packet), 3/8 ≤ boxProbability p5 p6 ∧ boxProbability p5 p6 ≤ 2/5) :=
by sorry

end candy_probability_theorem_l1105_110552


namespace small_prob_event_cannot_occur_is_false_l1105_110573

-- Define a probability space
variable (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)

-- Define an event as a measurable set
def Event (Ω : Type) [MeasurableSpace Ω] := {A : Set Ω // MeasurableSet A}

-- Define a very small probability
def VerySmallProbability (ε : ℝ) : Prop := 0 < ε ∧ ε < 1/1000000

-- Statement: An event with a very small probability cannot occur
theorem small_prob_event_cannot_occur_is_false :
  ∃ (A : Event Ω) (ε : ℝ), VerySmallProbability ε ∧ P A < ε ∧ ¬(P A = 0) :=
sorry

end small_prob_event_cannot_occur_is_false_l1105_110573


namespace derivative_of_even_function_l1105_110581

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the condition that f(-x) = f(x) for all x
variable (h : ∀ x, f (-x) = f x)

-- Define g as the derivative of f
variable (g : ℝ → ℝ)
variable (hg : ∀ x, HasDerivAt f (g x) x)

-- State the theorem
theorem derivative_of_even_function :
  ∀ x, g (-x) = -g x := by sorry

end derivative_of_even_function_l1105_110581


namespace dana_soda_consumption_l1105_110507

/-- The number of milliliters in one liter -/
def ml_per_liter : ℕ := 1000

/-- The size of the soda bottle in liters -/
def bottle_size : ℕ := 2

/-- The number of days the bottle lasts -/
def days_lasted : ℕ := 4

/-- Dana's daily soda consumption in milliliters -/
def daily_consumption : ℕ := (bottle_size * ml_per_liter) / days_lasted

theorem dana_soda_consumption :
  daily_consumption = 500 :=
sorry

end dana_soda_consumption_l1105_110507


namespace complex_conversion_l1105_110572

theorem complex_conversion :
  (2 * Real.sqrt 3) * Complex.exp (Complex.I * (17 * Real.pi / 6)) = -3 + Complex.I * Real.sqrt 3 := by
  sorry

end complex_conversion_l1105_110572


namespace rainfall_depth_calculation_l1105_110585

/-- Calculates the approximate rainfall depth given container dimensions and collected water depth -/
theorem rainfall_depth_calculation (container_side : ℝ) (container_height : ℝ) (water_depth : ℝ) 
  (h1 : container_side = 20)
  (h2 : container_height = 40)
  (h3 : water_depth = 10) : 
  ∃ (rainfall_depth : ℝ), abs (rainfall_depth - 12.7) < 0.1 := by
  sorry

end rainfall_depth_calculation_l1105_110585


namespace two_digit_factorizations_of_2210_l1105_110511

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2210

def distinct_factorizations (f1 f2 : ℕ × ℕ) : Prop :=
  f1.1 ≠ f2.1 ∧ f1.1 ≠ f2.2

theorem two_digit_factorizations_of_2210 :
  ∃ (f1 f2 : ℕ × ℕ),
    valid_factorization f1.1 f1.2 ∧
    valid_factorization f2.1 f2.2 ∧
    distinct_factorizations f1 f2 ∧
    ∀ (f : ℕ × ℕ), valid_factorization f.1 f.2 →
      (f = f1 ∨ f = f2 ∨ f = (f1.2, f1.1) ∨ f = (f2.2, f2.1)) :=
by sorry

end two_digit_factorizations_of_2210_l1105_110511


namespace total_seeds_planted_l1105_110529

theorem total_seeds_planted (num_flowerbeds : ℕ) (seeds_per_flowerbed : ℕ) 
  (h1 : num_flowerbeds = 8) 
  (h2 : seeds_per_flowerbed = 4) : 
  num_flowerbeds * seeds_per_flowerbed = 32 := by
  sorry

end total_seeds_planted_l1105_110529


namespace sticks_difference_l1105_110579

theorem sticks_difference (picked_up left : ℕ) 
  (h1 : picked_up = 14) 
  (h2 : left = 4) : 
  picked_up - left = 10 := by
sorry

end sticks_difference_l1105_110579


namespace sphere_cylinder_volume_l1105_110565

/-- Given a sphere with surface area 256π cm², prove that the volume of a cylinder
    with the same radius as the sphere and height equal to the sphere's diameter
    is 1024π cm³. -/
theorem sphere_cylinder_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 256 * Real.pi) :
  Real.pi * r^2 * (2 * r) = 1024 * Real.pi := by
  sorry

end sphere_cylinder_volume_l1105_110565


namespace partnership_profit_share_l1105_110558

/-- Partnership profit sharing problem -/
theorem partnership_profit_share
  (x : ℝ)  -- A's investment amount
  (annual_gain : ℝ)  -- Total annual gain
  (h1 : annual_gain = 18900)  -- Given annual gain
  (h2 : x > 0)  -- Assumption that A's investment is positive
  : x * 12 / (x * 12 + 2 * x * 6 + 3 * x * 4) * annual_gain = 6300 :=
by sorry

end partnership_profit_share_l1105_110558


namespace calculation_proof_l1105_110563

theorem calculation_proof : (-3)^2 - Real.sqrt 4 + (1/2)⁻¹ = 9 := by
  sorry

end calculation_proof_l1105_110563


namespace riverside_academy_statistics_l1105_110574

/-- The number of students taking statistics at Riverside Academy -/
def students_taking_statistics (total_students : ℕ) (physics_students : ℕ) (both_subjects : ℕ) : ℕ :=
  total_students - (physics_students - both_subjects)

/-- Theorem: The number of students taking statistics is 21 -/
theorem riverside_academy_statistics :
  let total_students : ℕ := 25
  let physics_students : ℕ := 10
  let both_subjects : ℕ := 6
  students_taking_statistics total_students physics_students both_subjects = 21 := by
  sorry

end riverside_academy_statistics_l1105_110574


namespace division_remainder_l1105_110562

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 199 →
  divisor = 18 →
  quotient = 11 →
  remainder = 1 := by
sorry

end division_remainder_l1105_110562


namespace sqrt_sum_equals_seven_l1105_110595

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_equals_seven_l1105_110595


namespace cube_lateral_surface_area_l1105_110525

theorem cube_lateral_surface_area (volume : ℝ) (lateral_surface_area : ℝ) :
  volume = 125 →
  lateral_surface_area = 4 * (volume ^ (1/3))^2 →
  lateral_surface_area = 100 := by
  sorry

end cube_lateral_surface_area_l1105_110525


namespace classroom_fundraising_l1105_110589

/-- The amount each classroom needs to raise -/
def classroom_goal : ℕ := 200

/-- The number of families that contributed $10 each -/
def num_families_10 : ℕ := 8

/-- The number of families that contributed $5 each -/
def num_families_5 : ℕ := 10

/-- The contribution from families giving $10 each -/
def contribution_10 : ℕ := 10 * num_families_10

/-- The contribution from families giving $5 each -/
def contribution_5 : ℕ := 5 * num_families_5

/-- The amount still needed to reach the goal -/
def amount_needed : ℕ := 30

/-- The number of families with unknown contribution -/
def num_unknown_families : ℕ := 2

theorem classroom_fundraising (x : ℕ) : 
  x * num_unknown_families + contribution_10 + contribution_5 = classroom_goal - amount_needed →
  x = 20 := by
  sorry

end classroom_fundraising_l1105_110589


namespace pen_problem_solution_l1105_110527

/-- Represents the number of pens of each color in Maria's desk drawer. -/
structure PenCounts where
  red : ℕ
  black : ℕ
  blue : ℕ

/-- The conditions of the pen problem. -/
def penProblem (p : PenCounts) : Prop :=
  p.red = 8 ∧
  p.black > p.red ∧
  p.blue = p.red + 7 ∧
  p.red + p.black + p.blue = 41

/-- The theorem stating the solution to the pen problem. -/
theorem pen_problem_solution (p : PenCounts) (h : penProblem p) : 
  p.black - p.red = 10 := by
  sorry

end pen_problem_solution_l1105_110527


namespace f_behavior_l1105_110578

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

def has_min_value (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → c ≤ f x

-- State the theorem
theorem f_behavior :
  is_even f →
  increasing_on f 5 7 →
  has_min_value f 5 7 6 →
  decreasing_on f (-7) (-5) ∧ has_min_value f (-7) (-5) 6 :=
sorry

end f_behavior_l1105_110578


namespace pet_shop_pricing_l1105_110549

theorem pet_shop_pricing (puppy_cost kitten_cost parakeet_cost : ℚ) : 
  puppy_cost = 3 * parakeet_cost →
  parakeet_cost = kitten_cost / 2 →
  2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost = 130 →
  parakeet_cost = 10 := by
sorry

end pet_shop_pricing_l1105_110549


namespace twenty_first_term_is_4641_l1105_110544

/-- The sequence where each term is the sum of consecutive integers, 
    and the number of integers in each group increases by 1 -/
def sequence_term (n : ℕ) : ℕ :=
  let first_num := 1 + (n * (n - 1)) / 2
  let last_num := first_num + n - 1
  n * (first_num + last_num) / 2

/-- The 21st term of the sequence is 4641 -/
theorem twenty_first_term_is_4641 : sequence_term 21 = 4641 := by
  sorry

end twenty_first_term_is_4641_l1105_110544


namespace largest_four_digit_square_base7_l1105_110555

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem largest_four_digit_square_base7 :
  (M * M ≥ 7^3) ∧ 
  (M * M < 7^4) ∧ 
  (∀ n : ℕ, n > M → n * n ≥ 7^4) ∧
  (toBase7 M = [6, 6]) := by sorry

end largest_four_digit_square_base7_l1105_110555


namespace other_number_proof_l1105_110500

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 24)
  (h2 : Nat.gcd a b = 4)
  (h3 : a = 12) : 
  b = 8 := by
  sorry

end other_number_proof_l1105_110500


namespace boys_speed_l1105_110580

/-- The speed of a boy traveling from home to school on the first day, given certain conditions. -/
theorem boys_speed (distance : ℝ) (late_time : ℝ) (early_time : ℝ) (second_day_speed : ℝ) : 
  distance = 2.5 ∧ 
  late_time = 7 / 60 ∧ 
  early_time = 8 / 60 ∧ 
  second_day_speed = 10 → 
  ∃ (first_day_speed : ℝ), first_day_speed = 9.375 := by
  sorry

#eval (9.375 : Float)

end boys_speed_l1105_110580


namespace problem_1_problem_2_l1105_110551

-- Problem 1
theorem problem_1 : (1) - 1^2 + Real.sqrt 12 + Real.sqrt (4/3) = -1 + (8 * Real.sqrt 3) / 3 := by sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = -1/2 ∨ x = 1 := by sorry

end problem_1_problem_2_l1105_110551


namespace fran_ate_15_green_macaroons_l1105_110522

/-- The number of green macaroons Fran ate -/
def green_eaten : ℕ := sorry

/-- The number of red macaroons Fran baked -/
def red_baked : ℕ := 50

/-- The number of green macaroons Fran baked -/
def green_baked : ℕ := 40

/-- The number of macaroons remaining -/
def remaining : ℕ := 45

theorem fran_ate_15_green_macaroons :
  green_eaten = 15 ∧
  red_baked = 50 ∧
  green_baked = 40 ∧
  remaining = 45 ∧
  red_baked + green_baked = green_eaten + 2 * green_eaten + remaining :=
sorry

end fran_ate_15_green_macaroons_l1105_110522


namespace triangle_problem_l1105_110509

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a * sin A / sin B →
  c = 2 * a * sin C / sin B →
  b / 2 = (2 * a * sin A * cos C + c * sin (2 * A)) / 2 →
  (A = π/6 ∧
   (a = 2 →
    ∀ (b' c' : ℝ),
      b' > 0 ∧ c' > 0 →
      b' = 2 * sin A / sin B →
      c' = 2 * sin C / sin B →
      1/2 * b' * c' * sin A ≤ 2 + sqrt 3)) :=
by sorry

end triangle_problem_l1105_110509


namespace purely_imaginary_complex_number_l1105_110594

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - a - 2) (a + 1)
  (z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
sorry

end purely_imaginary_complex_number_l1105_110594


namespace hyperbola_asymptote_l1105_110512

/-- Given a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has the equation y = 2x, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, x^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = 2*x ∧ x^2 - y^2 / b^2 = 1) →
  b = 2 := by
sorry

end hyperbola_asymptote_l1105_110512


namespace molecular_weight_N2O3_l1105_110583

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms in N2O3
def N_count : ℕ := 2
def O_count : ℕ := 3

-- Define the number of moles
def moles : ℝ := 4

-- Theorem statement
theorem molecular_weight_N2O3 : 
  moles * (N_count * atomic_weight_N + O_count * atomic_weight_O) = 304.08 := by
  sorry


end molecular_weight_N2O3_l1105_110583


namespace new_average_age_l1105_110593

def initial_people : ℕ := 8
def initial_average_age : ℚ := 25
def leaving_person_age : ℕ := 20
def remaining_people : ℕ := 7

theorem new_average_age :
  (initial_people * initial_average_age - leaving_person_age) / remaining_people = 180 / 7 := by
  sorry

end new_average_age_l1105_110593


namespace divisibility_properties_l1105_110597

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬ ((a + b) ∣ (a^(2*k) + b^(2*k))) ∧ ¬ ((a - b) ∣ (a^(2*k) + b^(2*k)))) ∧
  ((a + b) ∣ (a^(2*k) - b^(2*k)) ∧ (a - b) ∣ (a^(2*k) - b^(2*k))) ∧
  ((a + b) ∣ (a^(2*k+1) + b^(2*k+1))) ∧
  ((a - b) ∣ (a^(2*k+1) - b^(2*k+1))) :=
by sorry

end divisibility_properties_l1105_110597


namespace tom_marbles_groups_l1105_110557

/-- Represents the colors of marbles --/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents Tom's collection of marbles --/
structure MarbleCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of different groups of two marbles that can be chosen --/
def countDifferentGroups (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that Tom's specific collection results in 12 different groups --/
theorem tom_marbles_groups :
  let toms_collection : MarbleCollection := {
    red := 1,
    green := 1,
    blue := 2,
    yellow := 3
  }
  countDifferentGroups toms_collection = 12 := by
  sorry

end tom_marbles_groups_l1105_110557


namespace hyperbola_standard_equation_l1105_110550

/-- A hyperbola passing through the point (4, √3) with asymptote equation y = 1/2x has the standard equation x²/4 - y² = 1 -/
theorem hyperbola_standard_equation (x y : ℝ) :
  (∃ k : ℝ, x^2 / 4 - y^2 = k) →  -- Assuming the general form of the hyperbola equation
  (∀ x, y = 1/2 * x) →           -- Asymptote equation
  (4^2 / 4 - (Real.sqrt 3)^2 = 1) →  -- The hyperbola passes through (4, √3)
  x^2 / 4 - y^2 = 1 :=            -- Standard equation of the hyperbola
by sorry

end hyperbola_standard_equation_l1105_110550


namespace hyperbola_points_m_range_l1105_110591

/-- Given points A(-1, y₁) and B(2, y₂) on the hyperbola y = (3+m)/x with y₁ > y₂, 
    the range of values for m is m < -3 -/
theorem hyperbola_points_m_range (m : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = (3 + m) / (-1) → 
  y₂ = (3 + m) / 2 → 
  y₁ > y₂ → 
  m < -3 := by
sorry

end hyperbola_points_m_range_l1105_110591


namespace rainfall_difference_l1105_110535

/-- The difference in rainfall between March and April -/
theorem rainfall_difference (march_rainfall april_rainfall : ℝ) 
  (h1 : march_rainfall = 0.81)
  (h2 : april_rainfall = 0.46) : 
  march_rainfall - april_rainfall = 0.35 := by
  sorry

end rainfall_difference_l1105_110535


namespace remainder_sum_l1105_110561

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 47) (hd : d % 45 = 14) : (c + d) % 15 = 1 := by
  sorry

end remainder_sum_l1105_110561


namespace power_mod_thirteen_l1105_110506

theorem power_mod_thirteen :
  5^2023 ≡ 8 [ZMOD 13] := by
sorry

end power_mod_thirteen_l1105_110506


namespace geoffrey_money_left_l1105_110576

-- Define the given amounts
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def total_money : ℕ := 125
def game_cost : ℕ := 35
def num_games : ℕ := 3

-- Theorem to prove
theorem geoffrey_money_left :
  total_money - (grandmother_gift + aunt_gift + uncle_gift + num_games * game_cost) = 20 := by
  sorry

end geoffrey_money_left_l1105_110576


namespace boat_speed_in_still_water_l1105_110598

/-- The speed of a boat in still water given its travel distances with and against a stream -/
theorem boat_speed_in_still_water 
  (along_stream : ℝ) 
  (against_stream : ℝ) 
  (h1 : along_stream = 16) 
  (h2 : against_stream = 6) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = along_stream ∧ 
    boat_speed - stream_speed = against_stream ∧ 
    boat_speed = 11 :=
by sorry

end boat_speed_in_still_water_l1105_110598


namespace sum_in_base6_l1105_110542

/-- Represents a number in base 6 -/
def Base6 : Type := List Nat

/-- Converts a base 6 number to its decimal representation -/
def to_decimal (n : Base6) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a decimal number to its base 6 representation -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- Adds two base 6 numbers -/
def add_base6 (a b : Base6) : Base6 :=
  to_base6 (to_decimal a + to_decimal b)

theorem sum_in_base6 (a b c : Base6) :
  a = [0, 5, 6] ∧ b = [5, 0, 1] ∧ c = [2] →
  add_base6 (add_base6 a b) c = [1, 1, 1, 1] :=
sorry

end sum_in_base6_l1105_110542


namespace twenty_new_homes_l1105_110548

/-- Calculates the number of new trailer homes added -/
def new_trailer_homes (initial_count : ℕ) (initial_avg_age : ℕ) (time_passed : ℕ) (current_avg_age : ℕ) : ℕ :=
  let total_age := initial_count * (initial_avg_age + time_passed)
  let k := (total_age - initial_count * current_avg_age) / (current_avg_age - time_passed)
  k

/-- Theorem stating that 20 new trailer homes were added -/
theorem twenty_new_homes :
  new_trailer_homes 30 15 3 12 = 20 := by
  sorry

#eval new_trailer_homes 30 15 3 12

end twenty_new_homes_l1105_110548


namespace cake_recipe_flour_amount_l1105_110517

/-- The total number of cups of flour in Mary's cake recipe -/
def total_flour : ℕ := 9

/-- The total number of cups of sugar in the recipe -/
def total_sugar : ℕ := 11

/-- The number of cups of flour already added -/
def flour_added : ℕ := 4

/-- The difference between remaining sugar and remaining flour to be added -/
def sugar_flour_diff : ℕ := 6

theorem cake_recipe_flour_amount :
  total_flour = 9 ∧
  total_sugar = 11 ∧
  flour_added = 4 ∧
  sugar_flour_diff = 6 →
  total_flour = 9 :=
by sorry

end cake_recipe_flour_amount_l1105_110517


namespace red_mushrooms_with_spots_l1105_110537

/-- Represents the number of mushrooms gathered by Bill and Ted -/
structure MushroomGathering where
  red : ℕ
  brown : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the fraction of red mushrooms with white spots -/
def fraction_red_with_spots (g : MushroomGathering) (total_spotted : ℕ) : ℚ :=
  (total_spotted - g.brown - g.blue / 2) / g.red

/-- The main theorem stating the fraction of red mushrooms with white spots -/
theorem red_mushrooms_with_spots :
  let g := MushroomGathering.mk 12 6 14 6
  let total_spotted := 17
  fraction_red_with_spots g total_spotted = 2 / 3 := by
  sorry


end red_mushrooms_with_spots_l1105_110537


namespace floor_of_sum_l1105_110530

theorem floor_of_sum (x : ℝ) (h : x = -3.7 + 1.5) : ⌊x⌋ = -3 := by
  sorry

end floor_of_sum_l1105_110530


namespace evelyn_initial_skittles_l1105_110592

/-- The number of Skittles Evelyn shared with Christine -/
def shared_skittles : ℕ := 72

/-- The number of Skittles Evelyn had left after sharing -/
def remaining_skittles : ℕ := 4

/-- The initial number of Skittles Evelyn had -/
def initial_skittles : ℕ := shared_skittles + remaining_skittles

theorem evelyn_initial_skittles : initial_skittles = 76 := by
  sorry

end evelyn_initial_skittles_l1105_110592


namespace seventh_term_approx_l1105_110503

/-- Represents a geometric sequence with 10 terms -/
structure GeometricSequence where
  a₁ : ℝ
  r : ℝ
  len : ℕ
  h_len : len = 10
  h_a₁ : a₁ = 4
  h_a₄ : a₁ * r^3 = 64
  h_a₁₀ : a₁ * r^9 = 39304

/-- The 7th term of the geometric sequence -/
def seventh_term (seq : GeometricSequence) : ℝ :=
  seq.a₁ * seq.r^6

/-- Theorem stating that the 7th term is approximately 976 -/
theorem seventh_term_approx (seq : GeometricSequence) :
  ∃ ε > 0, |seventh_term seq - 976| < ε :=
sorry

end seventh_term_approx_l1105_110503


namespace wire_length_problem_l1105_110501

theorem wire_length_problem (total_wires : ℕ) (total_avg_length : ℝ) 
  (quarter_avg_length : ℝ) (third_avg_length : ℝ) :
  total_wires = 12 →
  total_avg_length = 95 →
  quarter_avg_length = 120 →
  third_avg_length = 75 →
  let quarter_wires := total_wires / 4
  let third_wires := total_wires / 3
  let remaining_wires := total_wires - quarter_wires - third_wires
  let total_length := total_wires * total_avg_length
  let quarter_length := quarter_wires * quarter_avg_length
  let third_length := third_wires * third_avg_length
  let remaining_length := total_length - quarter_length - third_length
  remaining_length / remaining_wires = 96 := by
sorry

end wire_length_problem_l1105_110501


namespace earth_angle_calculation_l1105_110526

/-- The angle between two points on a spherical Earth given their coordinates --/
def spherical_angle (lat1 : Real) (lon1 : Real) (lat2 : Real) (lon2 : Real) : Real :=
  sorry

theorem earth_angle_calculation :
  let p_lat : Real := 0
  let p_lon : Real := 100
  let q_lat : Real := 30
  let q_lon : Real := -100 -- Negative for West longitude
  spherical_angle p_lat p_lon q_lat q_lon = 160 := by
  sorry

end earth_angle_calculation_l1105_110526


namespace line_through_point_with_equal_intercepts_l1105_110543

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a line passes through a point
def passes_through (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

-- Define a function to check if a line has equal intercepts
def has_equal_intercepts (l : Line) : Prop :=
  l.a = l.b ∨ (l.a = -l.b ∧ l.c = 0)

-- State the theorem
theorem line_through_point_with_equal_intercepts :
  ∀ l : Line,
    passes_through l 3 (-6) →
    has_equal_intercepts l →
    (l = Line.mk 1 1 3 ∨ l = Line.mk 2 1 0) :=
by sorry

end line_through_point_with_equal_intercepts_l1105_110543


namespace metal_bars_per_set_l1105_110505

theorem metal_bars_per_set (total_bars : ℕ) (num_sets : ℕ) (bars_per_set : ℕ) : 
  total_bars = 14 → num_sets = 2 → total_bars = num_sets * bars_per_set → bars_per_set = 7 := by
  sorry

end metal_bars_per_set_l1105_110505


namespace coupon_savings_difference_l1105_110588

/-- Represents the savings from a coupon given a price -/
def CouponSavings (price : ℝ) : (ℝ → ℝ) → ℝ := fun f => f price

/-- Coupon A: 20% off the listed price -/
def CouponA (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: $50 off the listed price -/
def CouponB (_ : ℝ) : ℝ := 50

/-- Coupon C: 30% off the amount by which the listed price exceeds $120 -/
def CouponC (price : ℝ) : ℝ := 0.3 * (price - 120)

theorem coupon_savings_difference (price_min price_max : ℝ) :
  (price_min > 120) →
  (price_max > 120) →
  (∀ p : ℝ, p ≥ price_min → p ≤ price_max → 
    CouponSavings p CouponA ≥ max (CouponSavings p CouponB) (CouponSavings p CouponC)) →
  (∀ p : ℝ, p < price_min ∨ p > price_max → 
    CouponSavings p CouponA < max (CouponSavings p CouponB) (CouponSavings p CouponC)) →
  price_max - price_min = 110 := by
  sorry

end coupon_savings_difference_l1105_110588


namespace more_women_than_men_l1105_110571

theorem more_women_than_men (total : ℕ) (ratio : ℚ) : 
  total = 18 → ratio = 7/11 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 4 :=
sorry

end more_women_than_men_l1105_110571


namespace range_of_b_l1105_110536

theorem range_of_b (a b c : ℝ) (h1 : a * c = b^2) (h2 : a + b + c = 3) :
  -3 ≤ b ∧ b ≤ 1 := by
  sorry

end range_of_b_l1105_110536


namespace cow_problem_l1105_110510

theorem cow_problem (purchase_price daily_food_cost additional_costs selling_price profit : ℕ) 
  (h1 : purchase_price = 600)
  (h2 : daily_food_cost = 20)
  (h3 : additional_costs = 500)
  (h4 : selling_price = 2500)
  (h5 : profit = 600) :
  ∃ days : ℕ, days = (selling_price - profit - purchase_price - additional_costs) / daily_food_cost ∧ days = 40 :=
sorry

end cow_problem_l1105_110510


namespace total_money_l1105_110513

theorem total_money (A B C : ℕ) : 
  A + C = 200 →
  B + C = 340 →
  C = 40 →
  A + B + C = 500 := by
sorry

end total_money_l1105_110513


namespace sqrt_15_has_two_roots_l1105_110559

-- Define √15 as a real number
noncomputable def sqrt15 : ℝ := Real.sqrt 15

-- State the theorem
theorem sqrt_15_has_two_roots :
  ∃ (x : ℝ), x ≠ sqrt15 ∧ x * x = 15 :=
by
  -- The proof would go here, but we're using sorry as instructed
  sorry

end sqrt_15_has_two_roots_l1105_110559


namespace two_sides_and_angle_unique_two_angles_and_side_unique_three_sides_unique_two_sides_and_included_angle_not_unique_l1105_110514

/-- Represents a triangle -/
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

/-- Determines if a triangle is uniquely defined -/
def is_unique_triangle (t : Triangle) : Prop := sorry

/-- Two sides and an angle uniquely determine a triangle -/
theorem two_sides_and_angle_unique (a b : ℝ) (α : ℝ) : 
  ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.α = α := sorry

/-- Two angles and a side uniquely determine a triangle -/
theorem two_angles_and_side_unique (α β : ℝ) (a : ℝ) : 
  ∃! t : Triangle, t.α = α ∧ t.β = β ∧ t.a = a := sorry

/-- Three sides uniquely determine a triangle -/
theorem three_sides_unique (a b c : ℝ) : 
  ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.c = c := sorry

/-- Two sides and their included angle do not uniquely determine a triangle -/
theorem two_sides_and_included_angle_not_unique (a b : ℝ) (γ : ℝ) : 
  ¬(∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.γ = γ) := sorry

end two_sides_and_angle_unique_two_angles_and_side_unique_three_sides_unique_two_sides_and_included_angle_not_unique_l1105_110514


namespace initial_apples_count_l1105_110516

/-- The number of apples Rachel picked from the tree -/
def apples_picked : ℕ := 4

/-- The number of apples remaining on the tree -/
def apples_remaining : ℕ := 3

/-- The initial number of apples on the tree -/
def initial_apples : ℕ := apples_picked + apples_remaining

theorem initial_apples_count : initial_apples = 7 := by
  sorry

end initial_apples_count_l1105_110516


namespace sum_of_coefficients_fifth_power_one_plus_sqrt_two_l1105_110538

theorem sum_of_coefficients_fifth_power_one_plus_sqrt_two (a b : ℚ) : 
  (1 + Real.sqrt 2) ^ 5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end sum_of_coefficients_fifth_power_one_plus_sqrt_two_l1105_110538


namespace max_k_for_f_greater_than_k_l1105_110575

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + x * Real.log x + b

theorem max_k_for_f_greater_than_k (b : ℝ) :
  (∀ x > 0, (3 * x - f x 1 - 4 = 0)) →
  (∃ k : ℤ, ∀ x > 0, f x b > k) →
  (∀ k : ℤ, (∀ x > 0, f x b > k) → k ≤ -3) ∧
  (∀ x > 0, f x b > -3) :=
sorry

end max_k_for_f_greater_than_k_l1105_110575


namespace no_triangle_with_heights_1_2_3_l1105_110528

theorem no_triangle_with_heights_1_2_3 : 
  ¬ ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧  -- triangle inequality
    (1 : ℝ) = (2 * (a * b * c).sqrt) / (b * c) ∧  -- height 1
    (2 : ℝ) = (2 * (a * b * c).sqrt) / (a * c) ∧  -- height 2
    (3 : ℝ) = (2 * (a * b * c).sqrt) / (a * b) :=  -- height 3
by sorry


end no_triangle_with_heights_1_2_3_l1105_110528


namespace real_roots_of_polynomial_l1105_110531

/-- The polynomial function we're investigating -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 3*x^2 + 2*x - 6

/-- Theorem stating that -1 and 3 are the only real roots of the polynomial -/
theorem real_roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 3) :=
by sorry

end real_roots_of_polynomial_l1105_110531


namespace blue_ball_probability_l1105_110533

/-- Represents a container with red and blue balls -/
structure Container where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue ball from a container -/
def blueProbability (c : Container) : ℚ :=
  c.blue / (c.red + c.blue)

/-- The containers X, Y, and Z -/
def X : Container := ⟨3, 7⟩
def Y : Container := ⟨5, 5⟩
def Z : Container := ⟨6, 4⟩

/-- The list of all containers -/
def containers : List Container := [X, Y, Z]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

/-- The overall probability of selecting a blue ball -/
def overallBlueProbability : ℚ :=
  (containers.map blueProbability).sum / containers.length

theorem blue_ball_probability :
  overallBlueProbability = 8 / 15 := by
  sorry

end blue_ball_probability_l1105_110533


namespace inequality_implication_l1105_110586

theorem inequality_implication (x y : ℝ) : x > y → -2*x < -2*y := by
  sorry

end inequality_implication_l1105_110586


namespace production_average_problem_l1105_110570

theorem production_average_problem (n : ℕ) : 
  (∀ (past_total : ℕ), past_total = n * 50 →
   (past_total + 90) / (n + 1) = 58) →
  n = 4 := by
sorry

end production_average_problem_l1105_110570


namespace bridge_length_bridge_length_proof_l1105_110546

/-- The length of a bridge given train specifications --/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length problem --/
theorem bridge_length_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |bridge_length 180 60 45 - 570.15| < ε :=
sorry

end bridge_length_bridge_length_proof_l1105_110546


namespace modular_inverse_40_mod_61_l1105_110599

theorem modular_inverse_40_mod_61 :
  (∃ x : ℤ, 21 * x ≡ 1 [ZMOD 61] ∧ x ≡ 15 [ZMOD 61]) →
  (∃ y : ℤ, 40 * y ≡ 1 [ZMOD 61] ∧ y ≡ 46 [ZMOD 61]) :=
by sorry

end modular_inverse_40_mod_61_l1105_110599


namespace tv_count_indeterminate_l1105_110590

structure GroupInfo where
  total : ℕ
  married : ℕ
  radio : ℕ
  ac : ℕ
  tv_radio_ac_married : ℕ

def has_tv (info : GroupInfo) : Set ℕ :=
  { n | n ≥ info.tv_radio_ac_married ∧ n ≤ info.total }

theorem tv_count_indeterminate (info : GroupInfo) 
  (h_total : info.total = 100)
  (h_married : info.married = 81)
  (h_radio : info.radio = 85)
  (h_ac : info.ac = 70)
  (h_tram : info.tv_radio_ac_married = 11) :
  ∃ (n : ℕ), n ∈ has_tv info ∧ 
  ∀ (m : ℕ), m ≠ n → (m ∈ has_tv info ↔ n ∈ has_tv info) :=
sorry

end tv_count_indeterminate_l1105_110590


namespace calculation_proof_l1105_110524

theorem calculation_proof : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end calculation_proof_l1105_110524


namespace expand_expression_l1105_110564

theorem expand_expression (x : ℝ) : (x + 2) * (3 * x - 6) = 3 * x^2 - 12 := by
  sorry

end expand_expression_l1105_110564


namespace bianca_recycling_points_l1105_110556

/-- Calculates the points earned by Bianca for recycling bottles and cans --/
def points_earned (aluminum_points plastic_points glass_points : ℕ)
                  (aluminum_bags plastic_bags glass_bags : ℕ)
                  (aluminum_not_recycled plastic_not_recycled glass_not_recycled : ℕ) : ℕ :=
  (aluminum_points * (aluminum_bags - aluminum_not_recycled)) +
  (plastic_points * (plastic_bags - plastic_not_recycled)) +
  (glass_points * (glass_bags - glass_not_recycled))

theorem bianca_recycling_points :
  points_earned 5 8 10 10 5 5 3 2 1 = 99 := by
  sorry

end bianca_recycling_points_l1105_110556


namespace inequality_solution_l1105_110518

def inequality (x : ℝ) : Prop :=
  2*x^4 + x^2 - 2*x - 3*x^2*|x-1| + 1 ≥ 0

def solution_set : Set ℝ :=
  {x | x ≤ -(1 + Real.sqrt 5)/2 ∨ 
       (-1 ≤ x ∧ x ≤ 1/2) ∨ 
       x ≥ (Real.sqrt 5 - 1)/2}

theorem inequality_solution : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end inequality_solution_l1105_110518


namespace jake_audrey_ball_difference_l1105_110569

theorem jake_audrey_ball_difference :
  ∀ (jake_balls audrey_balls : ℕ),
    jake_balls = 7 →
    audrey_balls = 41 →
    audrey_balls - jake_balls = 34 :=
by
  sorry

end jake_audrey_ball_difference_l1105_110569


namespace student_arrangement_count_l1105_110523

/-- The number of ways to arrange 6 students in a line with 3 friends not adjacent -/
def arrangement_count : ℕ := 576

/-- Total number of students -/
def total_students : ℕ := 6

/-- Number of friends who refuse to stand next to each other -/
def friend_count : ℕ := 3

/-- Number of non-friend students -/
def non_friend_count : ℕ := total_students - friend_count

theorem student_arrangement_count :
  arrangement_count =
    (Nat.factorial total_students) -
    ((Nat.factorial non_friend_count) *
     (Nat.choose (non_friend_count + 1) friend_count) *
     (Nat.factorial friend_count)) :=
by sorry

end student_arrangement_count_l1105_110523


namespace frances_pencil_collection_l1105_110568

/-- The number of groups of pencils in Frances's collection -/
def num_groups : ℕ := 5

/-- The number of pencils in each group -/
def pencils_per_group : ℕ := 5

/-- The total number of pencils in Frances's collection -/
def total_pencils : ℕ := num_groups * pencils_per_group

theorem frances_pencil_collection : total_pencils = 25 := by
  sorry

end frances_pencil_collection_l1105_110568


namespace probability_of_red_ball_l1105_110545

def num_red_balls : ℕ := 2
def num_white_balls : ℕ := 6

def total_balls : ℕ := num_red_balls + num_white_balls

theorem probability_of_red_ball :
  (num_red_balls : ℚ) / (total_balls : ℚ) = 1 / 4 := by
  sorry

end probability_of_red_ball_l1105_110545


namespace product_sum_relation_l1105_110541

theorem product_sum_relation (a b x : ℤ) : 
  b = 9 → b - a = 5 → a * b = 2 * (a + b) + x → x = 10 := by
  sorry

end product_sum_relation_l1105_110541


namespace quadrilateral_fixed_point_theorem_l1105_110584

-- Define the plane
variable (Plane : Type)

-- Define points in the plane
variable (Point : Type)
variable (A B C D P : Point)

-- Define the distance function
variable (distance : Point → Point → ℝ)

-- Define the angle function
variable (angle : Point → Point → Point → ℝ)

-- Define the line through two points
variable (line_through : Point → Point → Set Point)

-- Define the "lies on" relation
variable (lies_on : Point → Set Point → Prop)

-- Theorem statement
theorem quadrilateral_fixed_point_theorem :
  ∃ P : Point,
    ∀ C D : Point,
      distance A B = distance B C →
      distance A D = distance D C →
      angle A D C = Real.pi / 2 →
      lies_on P (line_through C D) :=
sorry

end quadrilateral_fixed_point_theorem_l1105_110584


namespace price_problem_solution_l1105_110553

/-- The price of sugar and salt -/
def price_problem (sugar_price salt_price : ℝ) : Prop :=
  let sugar_3kg_salt_1kg := 3 * sugar_price + salt_price
  sugar_price = 1.5 ∧ sugar_3kg_salt_1kg = 5 →
  2 * sugar_price + 5 * salt_price = 5.5

/-- The solution to the price problem -/
theorem price_problem_solution :
  ∃ (sugar_price salt_price : ℝ), price_problem sugar_price salt_price :=
sorry

end price_problem_solution_l1105_110553


namespace shopping_mall_investment_strategy_l1105_110560

/-- Profit when selling at the beginning of the month -/
def profit_beginning (x : ℝ) : ℝ := x * (1 + 0.15) * (1 + 0.10) - x

/-- Profit when selling at the end of the month -/
def profit_end (x : ℝ) : ℝ := x * (1 + 0.30) - x - 700

theorem shopping_mall_investment_strategy :
  (profit_beginning 15000 > profit_end 15000) ∧
  (profit_end 30000 > profit_beginning 30000) ∧
  (∀ x y : ℝ, profit_beginning x = 6000 ∧ profit_end y = 6000 → y < x) ∧
  (∀ x y : ℝ, profit_beginning x = 5300 ∧ profit_end y = 5300 → y < x) :=
sorry

end shopping_mall_investment_strategy_l1105_110560


namespace marys_tickets_l1105_110540

theorem marys_tickets (total_tickets : ℕ) (probability : ℚ) (marys_tickets : ℕ) : 
  total_tickets = 120 →
  probability = 1 / 15 →
  (marys_tickets : ℚ) / total_tickets = probability →
  marys_tickets = 8 := by
  sorry

end marys_tickets_l1105_110540


namespace negation_equivalence_l1105_110577

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) := by sorry

end negation_equivalence_l1105_110577


namespace base_representation_comparison_l1105_110587

theorem base_representation_comparison (n : ℕ) (h : n = 1357) :
  (Nat.log 3 n + 1) = (Nat.log 5 n + 1) + (Nat.log 8 n + 1) - 2 :=
by sorry

end base_representation_comparison_l1105_110587


namespace solve_josie_problem_l1105_110532

def josie_problem (initial_amount : ℕ) (cassette_cost : ℕ) (num_cassettes : ℕ) (remaining_amount : ℕ) : Prop :=
  let total_cassette_cost := cassette_cost * num_cassettes
  let amount_after_cassettes := initial_amount - total_cassette_cost
  let headphone_cost := amount_after_cassettes - remaining_amount
  headphone_cost = 25

theorem solve_josie_problem :
  josie_problem 50 9 2 7 := by sorry

end solve_josie_problem_l1105_110532
