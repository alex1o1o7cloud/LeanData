import Mathlib

namespace intersecting_planes_not_imply_intersecting_lines_l383_38321

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation for lines and for planes
variable (lines_intersect : Line → Line → Prop)
variable (planes_intersect : Plane → Plane → Prop)

-- State the theorem
theorem intersecting_planes_not_imply_intersecting_lines 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : perpendicular a α) 
  (h4 : perpendicular b β) :
  ∃ (α β : Plane), planes_intersect α β ∧ ¬ lines_intersect a b :=
sorry

end intersecting_planes_not_imply_intersecting_lines_l383_38321


namespace sqrt_x_minus_3_real_range_l383_38327

theorem sqrt_x_minus_3_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) → x ≥ 3 := by
  sorry

end sqrt_x_minus_3_real_range_l383_38327


namespace infinite_solutions_imply_equal_coefficients_l383_38367

theorem infinite_solutions_imply_equal_coefficients (a b : ℝ) :
  (∀ x : ℝ, a * (a - x) - b * (b - x) = 0) →
  a - b = 0 := by
sorry

end infinite_solutions_imply_equal_coefficients_l383_38367


namespace sum_distinct_prime_divisors_of_1260_l383_38372

/-- The sum of the distinct prime integer divisors of 1260 is 17. -/
theorem sum_distinct_prime_divisors_of_1260 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 1260)) id) = 17 := by
  sorry

end sum_distinct_prime_divisors_of_1260_l383_38372


namespace expression_sum_l383_38368

theorem expression_sum (d e : ℤ) (h : d ≠ 0) : 
  let original := (16 * d + 17 + 18 * d^2) + (4 * d + 3) + 2 * e
  ∃ (a b c : ℤ), 
    original = a * d + b + c * d^2 + d * e ∧ 
    a + b + c + e = 60 := by
  sorry

end expression_sum_l383_38368


namespace vectors_same_direction_l383_38336

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define points A, B, C
variable (A B C : V)

-- Define the vectors
def AB : V := B - A
def AC : V := C - A
def BC : V := C - B

-- Define the theorem
theorem vectors_same_direction (h : ‖AB A B‖ = ‖AC A C‖ + ‖BC B C‖) :
  ∃ (k : ℝ), k > 0 ∧ AC A C = k • (BC B C) := by
  sorry

end vectors_same_direction_l383_38336


namespace circle_area_in_square_l383_38363

theorem circle_area_in_square (square_area : Real) (circle_area : Real) : 
  square_area = 400 →
  circle_area = Real.pi * (Real.sqrt square_area / 2)^2 →
  circle_area = 100 * Real.pi :=
by sorry

end circle_area_in_square_l383_38363


namespace no_all_ones_reverse_product_l383_38332

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number consists only of the digit 1 -/
def allOnes (n : ℕ) : Prop := sorry

/-- 
There does not exist a natural number n > 1 such that n multiplied by 
the number formed by reversing its digits results in a number comprised 
entirely of the digit one.
-/
theorem no_all_ones_reverse_product : 
  ¬ ∃ (n : ℕ), n > 1 ∧ allOnes (n * reverseDigits n) := by
  sorry

end no_all_ones_reverse_product_l383_38332


namespace max_values_on_sphere_l383_38355

theorem max_values_on_sphere (x y z : ℝ) :
  x^2 + y^2 + z^2 = 4 →
  (∃ (max_xz_yz : ℝ), ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 4 → x' * z' + y' * z' ≤ max_xz_yz ∧ max_xz_yz = 2 * Real.sqrt 2) ∧
  (x + y + z = 0 →
    ∃ (max_z : ℝ), ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 4 ∧ x' + y' + z' = 0 → z' ≤ max_z ∧ max_z = (2 * Real.sqrt 6) / 3) := by
  sorry

end max_values_on_sphere_l383_38355


namespace roots_have_different_signs_l383_38351

/-- Given two quadratic polynomials with specific properties, prove that the roots of the first polynomial have different signs -/
theorem roots_have_different_signs (a b c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0) →  -- First polynomial has two distinct roots
  (∀ x : ℝ, a^2 * x^2 + 2*b^2*x + c^2 ≠ 0) →                                        -- Second polynomial has no roots
  ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0  -- Roots of first polynomial have different signs
:= by sorry

end roots_have_different_signs_l383_38351


namespace f_log_one_third_36_l383_38389

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x % 3 ∧ x % 3 < 1 then 3^(x % 3) - 1
  else if 1 ≤ x % 3 ∧ x % 3 < 2 then -(3^(2 - (x % 3)) - 1)
  else -(3^((x % 3) - 2) - 1)

-- State the theorem
theorem f_log_one_third_36 (h1 : ∀ x, f (-x) = -f x) 
                            (h2 : ∀ x, f (x + 3) = f x) 
                            (h3 : ∀ x, 0 ≤ x → x < 1 → f x = 3^x - 1) :
  f (Real.log 36 / Real.log (1/3)) = -2/3 := by
  sorry

end f_log_one_third_36_l383_38389


namespace dot_product_not_sufficient_nor_necessary_for_parallel_l383_38397

-- Define the type for plane vectors
def PlaneVector := ℝ × ℝ

-- Define dot product for plane vectors
def dot_product (a b : PlaneVector) : ℝ :=
  (a.1 * b.1) + (a.2 * b.2)

-- Define parallelism for plane vectors
def parallel (a b : PlaneVector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

-- Theorem statement
theorem dot_product_not_sufficient_nor_necessary_for_parallel :
  ∃ (a b : PlaneVector),
    (dot_product a b > 0 ∧ ¬parallel a b) ∧
    (parallel a b ∧ ¬(dot_product a b > 0)) :=
sorry

end dot_product_not_sufficient_nor_necessary_for_parallel_l383_38397


namespace range_of_m_l383_38307

/-- The proposition p -/
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0

/-- The proposition q -/
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

/-- q is a sufficient but not necessary condition for p -/
def q_sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ ¬(∀ x, p x m → q x)

theorem range_of_m :
  ∀ m : ℝ, q_sufficient_not_necessary m ↔ -1/3 ≤ m ∧ m ≤ 3/2 :=
sorry

end range_of_m_l383_38307


namespace indeterminate_m_l383_38377

/-- An odd function from ℝ to ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem indeterminate_m (f : ℝ → ℝ) (m : ℝ) 
  (hodd : OddFunction f) (hm : f m = 2) (hm2 : f (m^2 - 2) = -2) :
  ¬ (∀ n : ℝ, f n = 2 → n = m) :=
sorry

end indeterminate_m_l383_38377


namespace multiple_of_second_number_l383_38320

theorem multiple_of_second_number (x y m : ℤ) : 
  y = m * x + 3 → 
  x + y = 27 → 
  y = 19 → 
  m = 2 := by
  sorry

end multiple_of_second_number_l383_38320


namespace absolute_value_equation_solution_l383_38300

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 24| + |x - 20| = |2*x - 44| :=
by
  -- The unique solution is x = 22
  use 22
  sorry

end absolute_value_equation_solution_l383_38300


namespace isosceles_triangle_side_length_l383_38326

/-- Isosceles triangle with given side length and area -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab = ac
  -- Given side length
  bcLength : bc = 16
  -- Area
  area : ℝ
  areaValue : area = 120

/-- The length of AB in the isosceles triangle -/
def sideLength (t : IsoscelesTriangle) : ℝ := t.ab

/-- Theorem: The length of AB in the given isosceles triangle is 17 -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) :
  sideLength t = 17 := by
  sorry

end isosceles_triangle_side_length_l383_38326


namespace element_uniquely_identified_l383_38361

/-- Represents a 6x6 grid of distinct elements -/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- The column of an element in the original grid -/
def OriginalColumn := Fin 6

/-- The column of an element in the new grid -/
def NewColumn := Fin 6

/-- Given a grid, an original column, and a new column, 
    returns the unique position of the element in both grids -/
def findElement (g : Grid) (oc : OriginalColumn) (nc : NewColumn) : 
  (Fin 6 × Fin 6) × (Fin 6 × Fin 6) :=
sorry

theorem element_uniquely_identified (g : Grid) (oc : OriginalColumn) (nc : NewColumn) :
  ∃! (p₁ p₂ : Fin 6 × Fin 6), 
    (findElement g oc nc).1 = p₁ ∧ 
    (findElement g oc nc).2 = p₂ ∧
    g p₁.1 p₁.2 = g p₂.2 p₂.1 :=
sorry

end element_uniquely_identified_l383_38361


namespace arithmetic_sequence_tangent_sum_l383_38341

theorem arithmetic_sequence_tangent_sum (x y z : Real) 
  (h1 : y - x = π/3) 
  (h2 : z - y = π/3) : 
  Real.tan x * Real.tan y + Real.tan y * Real.tan z + Real.tan z * Real.tan x = -3 := by
sorry

end arithmetic_sequence_tangent_sum_l383_38341


namespace bike_tractor_speed_ratio_l383_38379

/-- Given the speeds and distances of vehicles, prove the ratio of bike speed to tractor speed --/
theorem bike_tractor_speed_ratio :
  ∀ (car_speed bike_speed tractor_speed : ℝ),
  car_speed = (9/5) * bike_speed →
  tractor_speed = 575 / 25 →
  car_speed = 331.2 / 4 →
  bike_speed / tractor_speed = 2 := by
  sorry

end bike_tractor_speed_ratio_l383_38379


namespace samias_walking_distance_l383_38392

/-- Represents the problem of calculating Samia's walking distance --/
theorem samias_walking_distance
  (total_time : ℝ)
  (bike_speed : ℝ)
  (walk_speed : ℝ)
  (wait_time : ℝ)
  (h_total_time : total_time = 1.25)  -- 1 hour and 15 minutes
  (h_bike_speed : bike_speed = 20)
  (h_walk_speed : walk_speed = 4)
  (h_wait_time : wait_time = 0.25)  -- 15 minutes
  : ∃ (total_distance : ℝ),
    let bike_distance := total_distance / 3
    let walk_distance := 2 * total_distance / 3
    bike_distance / bike_speed + wait_time + walk_distance / walk_speed = total_time ∧
    (walk_distance ≥ 3.55 ∧ walk_distance ≤ 3.65) :=
by
  sorry

#check samias_walking_distance

end samias_walking_distance_l383_38392


namespace fraction_cube_multiply_l383_38302

theorem fraction_cube_multiply (a b : ℚ) : (1 / 3 : ℚ)^3 * (1 / 5 : ℚ) = 1 / 135 := by
  sorry

end fraction_cube_multiply_l383_38302


namespace rain_gear_needed_l383_38395

structure WeatherForecast where
  rain_probability : ℝ
  rain_probability_valid : 0 ≤ rain_probability ∧ rain_probability ≤ 1

def high_possibility (p : ℝ) : Prop := p > 0.5

theorem rain_gear_needed (forecast : WeatherForecast) 
  (h : forecast.rain_probability = 0.95) : 
  high_possibility forecast.rain_probability :=
by
  sorry

#check rain_gear_needed

end rain_gear_needed_l383_38395


namespace palic_characterization_l383_38356

/-- Palic function definition -/
def isPalic (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  Continuous f ∧
  ∀ x y z : ℝ, f x + f y + f z = f (a*x + b*y + c*z) + f (b*x + c*y + a*z) + f (c*x + a*y + b*z)

/-- Theorem: Characterization of Palic functions -/
theorem palic_characterization (a b c : ℝ) 
    (h1 : a + b + c = 1)
    (h2 : a^2 + b^2 + c^2 = 1)
    (h3 : a^3 + b^3 + c^3 ≠ 1)
    (f : ℝ → ℝ)
    (hf : isPalic f a b c) :
  ∃ p q r : ℝ, ∀ x : ℝ, f x = p * x^2 + q * x + r :=
sorry

end palic_characterization_l383_38356


namespace savings_is_six_dollars_l383_38305

-- Define the number of notebooks
def num_notebooks : ℕ := 8

-- Define the original price per notebook
def original_price : ℚ := 3

-- Define the discount rate
def discount_rate : ℚ := 1/4

-- Define the function to calculate savings
def calculate_savings (n : ℕ) (p : ℚ) (d : ℚ) : ℚ :=
  n * p * d

-- Theorem stating that the savings is $6.00
theorem savings_is_six_dollars :
  calculate_savings num_notebooks original_price discount_rate = 6 := by
  sorry

end savings_is_six_dollars_l383_38305


namespace sin_sum_arcsin_arctan_l383_38325

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end sin_sum_arcsin_arctan_l383_38325


namespace mixed_number_calculation_l383_38328

theorem mixed_number_calculation :
  let a := 5 + 1 / 2
  let b := 2 + 2 / 3
  let c := 1 + 1 / 5
  let d := 3 + 1 / 4
  (a - b) / (c + d) = 170 / 267 :=
by sorry

end mixed_number_calculation_l383_38328


namespace cos_2x_satisfies_conditions_l383_38319

theorem cos_2x_satisfies_conditions (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x)
  (f x = f (-x)) ∧ (f (x - π) = f x) := by sorry

end cos_2x_satisfies_conditions_l383_38319


namespace least_positive_integer_with_remainders_l383_38373

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 5 = 2 ∧
  n % 4 = 2 ∧
  n % 3 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 5 = 2 ∧ m % 4 = 2 ∧ m % 3 = 0 → n ≤ m :=
by sorry

end least_positive_integer_with_remainders_l383_38373


namespace eight_digit_numbers_divisibility_l383_38384

def first_number (A B C : ℕ) : ℕ := 84000000 + A * 100000 + 53000 + B * 100 + 10 + C
def second_number (A B C D : ℕ) : ℕ := 32700000 + A * 10000 + B * 1000 + 500 + C * 10 + D

theorem eight_digit_numbers_divisibility (A B C D : ℕ) 
  (h1 : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) 
  (h2 : first_number A B C % 4 = 0) 
  (h3 : second_number A B C D % 3 = 0) : 
  D = 2 := by
sorry

end eight_digit_numbers_divisibility_l383_38384


namespace polar_equation_circle_l383_38374

theorem polar_equation_circle (ρ : ℝ → ℝ → ℝ) (x y : ℝ) :
  (ρ = λ _ _ => 5) → (x^2 + y^2 = 25) :=
sorry

end polar_equation_circle_l383_38374


namespace quotient_approx_l383_38382

theorem quotient_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000001 ∧ |0.284973 / 29 - 0.009827| < ε :=
sorry

end quotient_approx_l383_38382


namespace joanna_initial_gumballs_l383_38333

/-- 
Given:
- Jacques had 60 gumballs initially
- They purchased 4 times their initial total
- After sharing equally, each got 250 gumballs
Prove that Joanna initially had 40 gumballs
-/
theorem joanna_initial_gumballs : 
  ∀ (j : ℕ), -- j represents Joanna's initial number of gumballs
  let jacques_initial := 60
  let total_initial := j + jacques_initial
  let purchased := 4 * total_initial
  let total_final := total_initial + purchased
  let each_after_sharing := 250
  total_final = 2 * each_after_sharing →
  j = 40 := by
sorry

end joanna_initial_gumballs_l383_38333


namespace sqrt_of_sqrt_16_l383_38343

theorem sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end sqrt_of_sqrt_16_l383_38343


namespace min_value_of_ratio_l383_38378

theorem min_value_of_ratio (x y : ℝ) (h1 : x + y - 3 ≤ 0) (h2 : x - y + 1 ≥ 0) (h3 : y ≥ 1) :
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 3 ≤ 0 ∧ x₀ - y₀ + 1 ≥ 0 ∧ y₀ ≥ 1 ∧
    ∀ (x' y' : ℝ), x' + y' - 3 ≤ 0 → x' - y' + 1 ≥ 0 → y' ≥ 1 → y₀ / x₀ ≤ y' / x' :=
by sorry

end min_value_of_ratio_l383_38378


namespace roots_shift_l383_38309

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the resulting polynomial
def resulting_poly (x : ℝ) : ℝ := x^3 + 9*x^2 + 21*x + 14

theorem roots_shift :
  ∀ (a b c : ℝ),
  (original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0) →
  (∀ x : ℝ, resulting_poly x = 0 ↔ (x = a - 3 ∨ x = b - 3 ∨ x = c - 3)) :=
by sorry

end roots_shift_l383_38309


namespace total_oil_volume_l383_38391

-- Define the volume of each bottle in mL
def bottle_volume : ℕ := 200

-- Define the number of bottles
def num_bottles : ℕ := 20

-- Define the conversion factor from mL to L
def ml_per_liter : ℕ := 1000

-- Theorem to prove
theorem total_oil_volume (bottle_volume : ℕ) (num_bottles : ℕ) (ml_per_liter : ℕ) :
  bottle_volume = 200 → num_bottles = 20 → ml_per_liter = 1000 →
  (bottle_volume * num_bottles) / ml_per_liter = 4 := by
  sorry

end total_oil_volume_l383_38391


namespace arithmetic_evaluation_l383_38338

theorem arithmetic_evaluation : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end arithmetic_evaluation_l383_38338


namespace odd_integers_equality_l383_38371

theorem odd_integers_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 :=
by sorry

end odd_integers_equality_l383_38371


namespace midpoint_on_grid_l383_38381

theorem midpoint_on_grid (points : Fin 5 → ℤ × ℤ) :
  ∃ i j, i ≠ j ∧ i < 5 ∧ j < 5 ∧
  (((points i).1 + (points j).1) % 2 = 0) ∧
  (((points i).2 + (points j).2) % 2 = 0) :=
sorry

end midpoint_on_grid_l383_38381


namespace initial_alcohol_percentage_l383_38344

/-- Proves that the initial alcohol percentage is 25% given the problem conditions -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (final_percentage : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_added_alcohol : added_alcohol = 3)
  (h_final_percentage : final_percentage = 50)
  (h_alcohol_balance : initial_volume * (initial_percentage / 100) + added_alcohol = 
                       (initial_volume + added_alcohol) * (final_percentage / 100)) :
  initial_percentage = 25 :=
by
  sorry

#check initial_alcohol_percentage

end initial_alcohol_percentage_l383_38344


namespace uncle_bob_parking_probability_l383_38339

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars that have already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces needed for Uncle Bob's truck -/
def needed_spaces : ℕ := 2

/-- The probability of having at least two adjacent empty spaces -/
def probability_adjacent_spaces : ℚ := 232 / 323

theorem uncle_bob_parking_probability :
  let total_combinations := Nat.choose total_spaces parked_cars
  let unfavorable_combinations := Nat.choose (parked_cars + needed_spaces + 1) (needed_spaces + 1)
  (1 : ℚ) - (unfavorable_combinations : ℚ) / (total_combinations : ℚ) = probability_adjacent_spaces :=
sorry

end uncle_bob_parking_probability_l383_38339


namespace cricket_average_problem_l383_38358

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : Nat
  totalRuns : Nat
  deriving Repr

/-- Calculates the average runs per innings -/
def averageRuns (player : CricketPlayer) : Rat :=
  player.totalRuns / player.innings

theorem cricket_average_problem (player : CricketPlayer) 
  (h1 : player.innings = 20)
  (h2 : averageRuns { innings := player.innings + 1, totalRuns := player.totalRuns + 158 } = 
        averageRuns player + 6) :
  averageRuns player = 32 := by
  sorry


end cricket_average_problem_l383_38358


namespace no_standard_operation_satisfies_equation_l383_38330

theorem no_standard_operation_satisfies_equation : ¬∃ (op : ℝ → ℝ → ℝ), 
  (op = (·+·) ∨ op = (·-·) ∨ op = (·*·) ∨ op = (·/·)) ∧ 
  (op 12 4) - 3 + (6 - 2) = 7 := by
sorry

end no_standard_operation_satisfies_equation_l383_38330


namespace convoy_problem_l383_38380

/-- Represents the convoy of vehicles -/
structure Convoy where
  num_vehicles : ℕ
  departure_interval : ℚ
  first_departure : ℚ
  stop_time : ℚ
  speed : ℚ

/-- Calculate the travel time of the last vehicle in the convoy -/
def last_vehicle_travel_time (c : Convoy) : ℚ :=
  c.stop_time - (c.first_departure + (c.num_vehicles - 1) * c.departure_interval)

/-- Calculate the total distance traveled by the convoy -/
def total_distance_traveled (c : Convoy) : ℚ :=
  let total_time := c.num_vehicles * (c.stop_time - c.first_departure) - 
    (c.num_vehicles * (c.num_vehicles - 1) / 2) * c.departure_interval
  total_time * c.speed

/-- The main theorem statement -/
theorem convoy_problem (c : Convoy) 
  (h1 : c.num_vehicles = 15)
  (h2 : c.departure_interval = 1/6)
  (h3 : c.first_departure = 2)
  (h4 : c.stop_time = 6)
  (h5 : c.speed = 60) : 
  last_vehicle_travel_time c = 5/3 ∧ 
  total_distance_traveled c = 2550 := by
  sorry

#eval last_vehicle_travel_time ⟨15, 1/6, 2, 6, 60⟩
#eval total_distance_traveled ⟨15, 1/6, 2, 6, 60⟩

end convoy_problem_l383_38380


namespace line_of_symmetry_l383_38394

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property
axiom symmetry_property : ∀ x, g x = g (3 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem line_of_symmetry :
  (∀ x, g x = g (3 - x)) → is_axis_of_symmetry g 1.5 :=
by sorry

end line_of_symmetry_l383_38394


namespace undefined_fraction_l383_38342

theorem undefined_fraction (a b : ℝ) (h1 : a = 4) (h2 : b = -4) :
  ¬∃x : ℝ, x = 3 / (a + b) := by
  sorry

end undefined_fraction_l383_38342


namespace min_value_theorem_l383_38318

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m > 0) (h3 : n > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → (1 / m + 2 / n) ≤ (1 / x + 2 / y) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 2 ∧ 1 / a + 2 / b = 4 :=
by sorry

end min_value_theorem_l383_38318


namespace square_fraction_count_l383_38324

theorem square_fraction_count : 
  ∃! (count : ℕ), count = 2 ∧ 
    (∀ n : ℤ, (∃ k : ℤ, n / (30 - 2*n) = k^2) ↔ (n = 0 ∨ n = 10)) := by
  sorry

end square_fraction_count_l383_38324


namespace apples_handed_out_to_students_l383_38359

theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (pies : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 62) 
  (h2 : pies = 6) 
  (h3 : apples_per_pie = 9) :
  initial_apples - pies * apples_per_pie = 8 := by
sorry

end apples_handed_out_to_students_l383_38359


namespace quadratic_function_property_l383_38308

/-- Given a quadratic function f(x) = x^2 - 2ax + b where a > 1,
    if both the domain and range of f are [1, a], then b = 5. -/
theorem quadratic_function_property (a b : ℝ) (f : ℝ → ℝ) :
  a > 1 →
  (∀ x, f x = x^2 - 2*a*x + b) →
  (∀ x, x ∈ Set.Icc 1 a ↔ f x ∈ Set.Icc 1 a) →
  b = 5 := by
  sorry

end quadratic_function_property_l383_38308


namespace gcd_of_three_numbers_l383_38304

theorem gcd_of_three_numbers : Nat.gcd 9118 (Nat.gcd 12173 33182) = 47 := by
  sorry

end gcd_of_three_numbers_l383_38304


namespace quadratic_inequality_solution_l383_38331

theorem quadratic_inequality_solution (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, ax^2 - 6*x + a^2 < 0 ↔ 1 < x ∧ x < m) →
  m = 2 := by
sorry

end quadratic_inequality_solution_l383_38331


namespace doubled_to_original_ratio_l383_38329

theorem doubled_to_original_ratio (x : ℝ) (h : 3 * (2 * x + 5) = 135) : 
  (2 * x) / x = 2 := by
  sorry

end doubled_to_original_ratio_l383_38329


namespace area_le_sqrt_product_area_eq_sqrt_product_iff_rectangle_l383_38303

/-- A quadrilateral circumscribed about a circle -/
structure CircumscribedQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  area_pos : 0 < area

/-- The theorem stating that the area of a circumscribed quadrilateral is at most the square root of the product of its side lengths -/
theorem area_le_sqrt_product (q : CircumscribedQuadrilateral) : 
  q.area ≤ Real.sqrt (q.a * q.b * q.c * q.d) := by
  sorry

/-- The condition for equality in the above inequality -/
def is_rectangle (q : CircumscribedQuadrilateral) : Prop :=
  (q.a = q.c ∧ q.b = q.d) ∨ (q.a = q.b ∧ q.c = q.d)

/-- The theorem stating that equality holds if and only if the quadrilateral is a rectangle -/
theorem area_eq_sqrt_product_iff_rectangle (q : CircumscribedQuadrilateral) :
  q.area = Real.sqrt (q.a * q.b * q.c * q.d) ↔ is_rectangle q := by
  sorry

end area_le_sqrt_product_area_eq_sqrt_product_iff_rectangle_l383_38303


namespace product_even_implies_factor_even_l383_38352

theorem product_even_implies_factor_even (a b : ℕ) : 
  Even (a * b) → Even a ∨ Even b := by sorry

end product_even_implies_factor_even_l383_38352


namespace aladdin_journey_theorem_l383_38347

-- Define the circle (equator)
def Equator : Real := 40000

-- Define Aladdin's path
def AladdinPath : Set ℝ → Prop :=
  λ path => ∀ x, x ∈ path → 0 ≤ x ∧ x < Equator

-- Define the property of covering every point on the equator
def CoversEquator (path : Set ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x < Equator → ∃ y ∈ path, y % Equator = x

-- Define the westward travel limit
def WestwardLimit : Real := 19000

-- Define the theorem
theorem aladdin_journey_theorem (path : Set ℝ) 
  (h_path : AladdinPath path)
  (h_covers : CoversEquator path)
  (h_westward : ∀ x ∈ path, x ≤ WestwardLimit) :
  ∃ x ∈ path, abs (x % Equator - x) ≥ Equator / 2 := by
sorry

end aladdin_journey_theorem_l383_38347


namespace product_equals_square_l383_38393

theorem product_equals_square : 500 * 49.95 * 4.995 * 5000 = (24975 : ℝ)^2 := by
  sorry

end product_equals_square_l383_38393


namespace johns_pill_cost_l383_38375

/-- Calculates the out-of-pocket cost for pills in a 30-day month given the following conditions:
  * Daily pill requirement
  * Cost per pill
  * Insurance coverage percentage
  * Number of days in a month
-/
def outOfPocketCost (dailyPills : ℕ) (costPerPill : ℚ) (insuranceCoverage : ℚ) (daysInMonth : ℕ) : ℚ :=
  let totalPills := dailyPills * daysInMonth
  let totalCost := totalPills * costPerPill
  let insuranceAmount := totalCost * insuranceCoverage
  totalCost - insuranceAmount

/-- Proves that given the specified conditions, John's out-of-pocket cost for pills in a 30-day month is $54 -/
theorem johns_pill_cost :
  outOfPocketCost 2 (3/2) (2/5) 30 = 54 := by
  sorry

end johns_pill_cost_l383_38375


namespace intersection_two_elements_l383_38399

/-- The set M represents lines passing through (1,1) with slope k -/
def M (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 1) + 1}

/-- The set N represents a circle with center (0,1) and radius 1 -/
def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

/-- The intersection of M and N contains exactly two elements -/
theorem intersection_two_elements (k : ℝ) : ∃ (p q : ℝ × ℝ), p ≠ q ∧
  M k ∩ N = {p, q} :=
sorry

end intersection_two_elements_l383_38399


namespace ten_bags_of_bags_l383_38316

/-- The number of ways to create a "bag of bags" structure with n identical bags. -/
def bagsOfBags : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => sorry  -- The actual recursive definition would go here

/-- The number of ways to create a "bag of bags" structure with 10 identical bags is 719. -/
theorem ten_bags_of_bags : bagsOfBags 10 = 719 := by sorry

end ten_bags_of_bags_l383_38316


namespace sqrt_mixed_fraction_equality_l383_38366

theorem sqrt_mixed_fraction_equality (k n : ℝ) (h1 : k > 0) (h2 : n > 0) (h3 : n + 1 = k^2) :
  Real.sqrt (k * (k / n)) = k * Real.sqrt (k / n) := by
  sorry

end sqrt_mixed_fraction_equality_l383_38366


namespace spider_permutations_l383_38312

/-- Represents the number of legs a spider has -/
def num_legs : ℕ := 8

/-- Represents the number of items per leg -/
def items_per_leg : ℕ := 3

/-- Represents the total number of items -/
def total_items : ℕ := num_legs * items_per_leg

/-- Represents the number of valid orderings per leg -/
def valid_orderings_per_leg : ℕ := 3

/-- Represents the total number of orderings per leg -/
def total_orderings_per_leg : ℕ := 6

/-- Represents the probability of a valid ordering for one leg -/
def prob_valid_ordering : ℚ := 1 / 2

/-- Theorem: The number of valid permutations for a spider to put on its items
    with the given constraints is equal to 24! / 2^8 -/
theorem spider_permutations :
  (Nat.factorial total_items) / (2 ^ num_legs) =
  (Nat.factorial total_items) * (prob_valid_ordering ^ num_legs) :=
sorry

end spider_permutations_l383_38312


namespace stratified_sampling_sum_l383_38369

def total_population : ℕ := 40 + 10 + 30 + 20

def strata : List ℕ := [40, 10, 30, 20]

def sample_size : ℕ := 20

def stratified_sample (stratum : ℕ) : ℕ :=
  (stratum * sample_size) / total_population

theorem stratified_sampling_sum :
  stratified_sample (strata[1]) + stratified_sample (strata[3]) = 6 := by
  sorry

end stratified_sampling_sum_l383_38369


namespace intersection_theorem_subset_theorem_l383_38390

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | -x^2 + 2*m*x + 4 - m^2 ≥ 0}
def B : Set ℝ := {x | 2*x^2 - 5*x - 7 < 0}

-- Define the intersection of A and B
def A_intersect_B (m : ℝ) : Set ℝ := A m ∩ B

-- Define the complement of A in ℝ
def complement_A (m : ℝ) : Set ℝ := {x | x ∉ A m}

-- Theorem for part (1)
theorem intersection_theorem (m : ℝ) :
  A_intersect_B m = {x | 0 ≤ x ∧ x < 7/2} ↔ m = 2 :=
sorry

-- Theorem for part (2)
theorem subset_theorem (m : ℝ) :
  B ⊆ complement_A m ↔ m ≤ -3 ∨ m ≥ 11/2 :=
sorry

end intersection_theorem_subset_theorem_l383_38390


namespace upstream_downstream_time_ratio_l383_38322

theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 63) 
  (h2 : stream_speed = 21) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
sorry

end upstream_downstream_time_ratio_l383_38322


namespace common_difference_is_three_l383_38376

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_three
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end common_difference_is_three_l383_38376


namespace min_value_theorem_l383_38396

/-- Given a function f(x) = x(x-a)(x-b) where f'(0) = 4, 
    the minimum value of a^2 + 2b^2 is 8√2 -/
theorem min_value_theorem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (x - a) * (x - b)
  let f' : ℝ → ℝ := λ x ↦ (3 * x^2) - 2 * (a + b) * x + a * b
  (f' 0 = 4) → (∀ a b : ℝ, a^2 + 2*b^2 ≥ 8 * Real.sqrt 2) :=
by sorry

end min_value_theorem_l383_38396


namespace sum_of_numbers_l383_38323

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 := by
  sorry

end sum_of_numbers_l383_38323


namespace rectangular_plot_breadth_l383_38311

/-- Represents a rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 21 * breadth
  length_eq : length = breadth + 10

/-- Theorem stating that a rectangular plot with the given properties has a breadth of 11 meters -/
theorem rectangular_plot_breadth (plot : RectangularPlot) : plot.breadth = 11 := by
  sorry

end rectangular_plot_breadth_l383_38311


namespace min_value_parallel_vectors_l383_38354

/-- Given parallel vectors a and b, prove the minimum value of 3/x + 2/y is 8 -/
theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, y - 1]
  (∃ (k : ℝ), b = k • a) →
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 3 / x' + 2 / y' ≥ 8) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end min_value_parallel_vectors_l383_38354


namespace cos_585_degrees_l383_38349

theorem cos_585_degrees : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_585_degrees_l383_38349


namespace revenue_change_l383_38370

theorem revenue_change 
  (T : ℝ) -- Original tax rate
  (C : ℝ) -- Original consumption
  (h1 : T > 0) -- Assumption: tax rate is positive
  (h2 : C > 0) -- Assumption: consumption is positive
  : 
  let T_new := T * (1 - 0.15) -- New tax rate after 15% decrease
  let C_new := C * (1 + 0.10) -- New consumption after 10% increase
  let R := T * C -- Original revenue
  let R_new := T_new * C_new -- New revenue
  (R_new / R) = 0.935 -- Ratio of new revenue to original revenue
  :=
by sorry

end revenue_change_l383_38370


namespace regression_slope_effect_l383_38334

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Predicted y value for a given x --/
def LinearRegression.predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

theorem regression_slope_effect (model : LinearRegression) 
  (h : model.slope = -1 ∧ model.intercept = 2) :
  ∀ x : ℝ, model.predict (x + 1) = model.predict x - 1 := by
  sorry

#check regression_slope_effect

end regression_slope_effect_l383_38334


namespace elegant_interval_p_values_l383_38340

theorem elegant_interval_p_values (a b : ℕ) (m : ℝ) (p : ℕ) :
  (a < m ∧ m < b) →  -- m is in the "elegant interval" (a, b)
  (b = a + 1) →  -- a and b are consecutive positive integers
  (3 < Real.sqrt a + b ∧ Real.sqrt a + b ≤ 13) →  -- satisfies the given inequality
  (∃ x y : ℕ, x = b ∧ y * y = a ∧ b * x + a * y = p) →  -- x = b, y = √a, and bx + ay = p
  (p = 33 ∨ p = 127) :=
by sorry

end elegant_interval_p_values_l383_38340


namespace areas_product_eq_volume_squared_l383_38388

/-- A rectangular prism with dimensions x, y, and z. -/
structure RectangularPrism where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The volume of a rectangular prism. -/
def volume (p : RectangularPrism) : ℝ :=
  p.x * p.y * p.z

/-- The areas of the top, back, and lateral face of a rectangular prism. -/
def areas (p : RectangularPrism) : ℝ × ℝ × ℝ :=
  (p.x * p.y, p.y * p.z, p.z * p.x)

/-- The theorem stating that the product of the areas equals the square of the volume. -/
theorem areas_product_eq_volume_squared (p : RectangularPrism) :
  let (top, back, lateral) := areas p
  top * back * lateral = (volume p) ^ 2 := by
  sorry


end areas_product_eq_volume_squared_l383_38388


namespace equation_has_three_solutions_l383_38346

/-- The number of distinct complex solutions to the equation (z^4 - 1) / (z^3 - 3z + 2) = 0 -/
def num_solutions : ℕ := 3

/-- The equation (z^4 - 1) / (z^3 - 3z + 2) = 0 -/
def equation (z : ℂ) : Prop :=
  (z^4 - 1) / (z^3 - 3*z + 2) = 0

theorem equation_has_three_solutions :
  ∃ (S : Finset ℂ), S.card = num_solutions ∧
    (∀ z ∈ S, equation z) ∧
    (∀ z : ℂ, equation z → z ∈ S) :=
by sorry

end equation_has_three_solutions_l383_38346


namespace greg_sharon_harvest_difference_l383_38350

theorem greg_sharon_harvest_difference :
  let greg_harvest : Real := 0.4
  let sharon_harvest : Real := 0.1
  greg_harvest - sharon_harvest = 0.3 := by
  sorry

end greg_sharon_harvest_difference_l383_38350


namespace possible_b_values_l383_38314

/-- The cubic polynomial p(x) = x^3 + ax + b -/
def p (a b x : ℝ) : ℝ := x^3 + a*x + b

/-- The cubic polynomial q(x) = x^3 + ax + b + 150 -/
def q (a b x : ℝ) : ℝ := x^3 + a*x + b + 150

/-- Theorem stating the possible values of b given the conditions -/
theorem possible_b_values (a b r s : ℝ) : 
  (p a b r = 0 ∧ p a b s = 0) →  -- r and s are roots of p(x)
  (q a b (r+3) = 0 ∧ q a b (s-5) = 0) →  -- r+3 and s-5 are roots of q(x)
  b = 0 ∨ b = 12082 := by
sorry

end possible_b_values_l383_38314


namespace mystery_discount_rate_l383_38301

theorem mystery_discount_rate 
  (biography_price : ℝ) 
  (mystery_price : ℝ) 
  (biography_count : ℕ) 
  (mystery_count : ℕ) 
  (total_savings : ℝ) 
  (total_discount_rate : ℝ) 
  (h1 : biography_price = 20)
  (h2 : mystery_price = 12)
  (h3 : biography_count = 5)
  (h4 : mystery_count = 3)
  (h5 : total_savings = 19)
  (h6 : total_discount_rate = 0.43)
  : ∃ (biography_discount : ℝ) (mystery_discount : ℝ),
    biography_discount + mystery_discount = total_discount_rate ∧ 
    mystery_discount = 0.375 := by
  sorry

end mystery_discount_rate_l383_38301


namespace max_squares_covered_l383_38386

/-- Represents a square card with side length 2 inches -/
structure Card where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- Represents a checkerboard with 1-inch squares -/
structure Checkerboard where
  square_size : ℝ
  square_size_eq : square_size = 1

/-- The maximum number of squares that can be covered by the card -/
def max_covered_squares : ℕ := 16

/-- Theorem stating the maximum number of squares that can be covered -/
theorem max_squares_covered (card : Card) (board : Checkerboard) :
  ∃ (n : ℕ), n = max_covered_squares ∧ 
  n = (max_covered_squares : ℝ) ∧
  ∀ (m : ℕ), (m : ℝ) ≤ (card.side_length / board.square_size) ^ 2 → m ≤ n :=
sorry

end max_squares_covered_l383_38386


namespace quadratic_function_property_l383_38362

-- Define a quadratic function
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Define the inverse function property
def HasInverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x)

theorem quadratic_function_property (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : HasInverse f)
  (h3 : ∀ x, f x = 3 * (Classical.choose h2) x + 5)
  (h4 : f 1 = 5) :
  f 2 = 8 := by
  sorry

end quadratic_function_property_l383_38362


namespace bag_empty_probability_l383_38398

/-- Probability of forming a pair when drawing 3 cards from n pairs -/
def prob_pair (n : ℕ) : ℚ :=
  (3 : ℚ) / (2 * n - 1)

/-- Probability of emptying the bag with n pairs of cards -/
def P (n : ℕ) : ℚ :=
  if n ≤ 2 then 1
  else (prob_pair n) * (P (n - 1))

theorem bag_empty_probability :
  P 6 = 9 / 385 :=
sorry

#eval (9 : ℕ) + 385

end bag_empty_probability_l383_38398


namespace perpendicular_line_equation_l383_38317

/-- Given a line l with equation Ax + By + C = 0, 
    a line perpendicular to l has the equation Bx - Ay + C' = 0, 
    where C' is some constant. -/
theorem perpendicular_line_equation 
  (A B C : ℝ) (x y : ℝ → ℝ) (l : ℝ → Prop) :
  (l = λ t => A * (x t) + B * (y t) + C = 0) →
  ∃ C', ∃ l_perp : ℝ → Prop,
    (l_perp = λ t => B * (x t) - A * (y t) + C' = 0) ∧
    (∀ t, l_perp t → (∀ s, l s → 
      (x t - x s) * (A * (x t - x s) + B * (y t - y s)) + 
      (y t - y s) * (B * (x t - x s) - A * (y t - y s)) = 0)) :=
by sorry

end perpendicular_line_equation_l383_38317


namespace det_A_squared_minus_3A_l383_38360

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 3, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 140 := by
  sorry

end det_A_squared_minus_3A_l383_38360


namespace wild_weatherman_proof_l383_38385

structure TextContent where
  content : String

structure WritingStyle where
  style : String

structure CareerAspiration where
  aspiration : String

structure WeatherForecastingTechnology where
  accuracy : String
  perfection : Bool

structure WeatherScienceStudy where
  name : String

def text_content : TextContent := ⟨"[Full text content]"⟩

theorem wild_weatherman_proof 
  (text : TextContent) 
  (writing_style : WritingStyle) 
  (sam_aspiration : CareerAspiration) 
  (weather_tech : WeatherForecastingTechnology) 
  (weather_study : WeatherScienceStudy) : 
  writing_style.style = "interview" ∧ 
  sam_aspiration.aspiration = "news reporter" ∧ 
  weather_tech.accuracy = "more exact" ∧ 
  ¬weather_tech.perfection ∧
  weather_study.name = "meteorology" := by
  sorry

#check wild_weatherman_proof text_content

end wild_weatherman_proof_l383_38385


namespace sum_between_equals_1999_l383_38348

theorem sum_between_equals_1999 :
  ∀ x y : ℕ, x < y →
  (((x + 1 + (y - 1)) / 2) * (y - x - 1) = 1999) →
  ((x = 1998 ∧ y = 2000) ∨ (x = 998 ∧ y = 1001)) :=
by sorry

end sum_between_equals_1999_l383_38348


namespace angle_properties_l383_38313

theorem angle_properties (α : Real) (y : Real) :
  -- Angle α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- Point P on its terminal side has coordinates (-√2, y)
  ∃ P : Real × Real, P = (-Real.sqrt 2, y) →
  -- sin α = (√2/4)y
  Real.sin α = (Real.sqrt 2 / 4) * y →
  -- Prove: tan α = -√3
  Real.tan α = -Real.sqrt 3 ∧
  -- Prove: (3sin α · cos α) / (4sin²α + 2cos²α) = -3√3/14
  (3 * Real.sin α * Real.cos α) / (4 * Real.sin α ^ 2 + 2 * Real.cos α ^ 2) = -3 * Real.sqrt 3 / 14 := by
  sorry

end angle_properties_l383_38313


namespace perimeter_triangle_PF₁F₂_shortest_distance_opposite_branches_l383_38383

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
def P_on_C (P : ℝ × ℝ) : Prop := hyperbola_C P.1 P.2

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: Perimeter of triangle PF₁F₂
theorem perimeter_triangle_PF₁F₂ (P : ℝ × ℝ) (h₁ : P_on_C P) (h₂ : distance P F₁ = 2 * distance P F₂) :
  distance P F₁ + distance P F₂ + distance F₁ F₂ = 28 := sorry

-- Theorem 2: Shortest distance between opposite branches
theorem shortest_distance_opposite_branches :
  ∃ (P Q : ℝ × ℝ), P_on_C P ∧ P_on_C Q ∧ 
    (∀ (R S : ℝ × ℝ), P_on_C R → P_on_C S → R.1 * S.1 < 0 → distance P Q ≤ distance R S) ∧
    distance P Q = 6 := sorry

end perimeter_triangle_PF₁F₂_shortest_distance_opposite_branches_l383_38383


namespace complex_number_equality_l383_38306

theorem complex_number_equality (b : ℝ) : 
  (Complex.re ((1 + b * Complex.I) / (2 + Complex.I)) = 
   Complex.im ((1 + b * Complex.I) / (2 + Complex.I))) → 
  b = 3 := by
  sorry

end complex_number_equality_l383_38306


namespace vehicle_value_fraction_l383_38364

def vehicle_value_this_year : ℚ := 16000
def vehicle_value_last_year : ℚ := 20000

theorem vehicle_value_fraction :
  vehicle_value_this_year / vehicle_value_last_year = 4 / 5 := by
  sorry

end vehicle_value_fraction_l383_38364


namespace mean_salary_proof_l383_38353

def salaries : List ℝ := [1000, 2500, 3100, 3650, 1500, 2000]

theorem mean_salary_proof :
  (salaries.sum / salaries.length : ℝ) = 2458.33 := by
  sorry

end mean_salary_proof_l383_38353


namespace line_points_k_value_l383_38365

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 2 = 2 * (n + k) + 5) → 
  k = 1 := by
sorry

end line_points_k_value_l383_38365


namespace ashleys_notebooks_l383_38335

theorem ashleys_notebooks :
  ∀ (notebook_price pencil_price : ℕ) (notebooks_in_93 : ℕ),
    notebook_price + pencil_price = 5 →
    21 * pencil_price + notebooks_in_93 * notebook_price = 93 →
    notebooks_in_93 = 15 →
    ∃ (notebooks_in_5 : ℕ),
      notebooks_in_5 * notebook_price + 1 * pencil_price = 5 ∧
      notebooks_in_5 = 1 :=
by sorry

end ashleys_notebooks_l383_38335


namespace binomial_congruence_l383_38337

theorem binomial_congruence (p a b : ℕ) (hp : Nat.Prime p) (hab : a ≥ b) (hb : b ≥ 0) :
  (Nat.choose (p * (a - b)) p) ≡ (Nat.choose a b) [MOD p] := by
  sorry

end binomial_congruence_l383_38337


namespace perfect_squares_of_cube_sums_l383_38357

theorem perfect_squares_of_cube_sums : 
  ∃ (a b c d : ℕ),
    (1^3 + 2^3 = a^2) ∧ 
    (1^3 + 2^3 + 3^3 = b^2) ∧ 
    (1^3 + 2^3 + 3^3 + 4^3 = c^2) ∧ 
    (1^3 + 2^3 + 3^3 + 4^3 + 5^3 = d^2) := by
  sorry

end perfect_squares_of_cube_sums_l383_38357


namespace roses_per_friend_l383_38345

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- Prove that each dancer friend gave Bella 2 roses -/
theorem roses_per_friend (
  parents_roses : ℕ) 
  (dancer_friends : ℕ) 
  (total_roses : ℕ) 
  (h1 : parents_roses = 2 * dozen)
  (h2 : dancer_friends = 10)
  (h3 : total_roses = 44) :
  (total_roses - parents_roses) / dancer_friends = 2 := by
  sorry

#check roses_per_friend

end roses_per_friend_l383_38345


namespace binary_multiplication_subtraction_l383_38387

-- Define binary numbers as natural numbers
def binary_11011 : ℕ := 27
def binary_1101 : ℕ := 13
def binary_1010 : ℕ := 10

-- Define the expected result
def expected_result : ℕ := 409

-- Theorem statement
theorem binary_multiplication_subtraction :
  (binary_11011 * binary_1101) - binary_1010 = expected_result :=
by
  sorry

end binary_multiplication_subtraction_l383_38387


namespace cosine_value_l383_38310

theorem cosine_value (α : ℝ) (h : 2 * Real.cos (2 * α) + 9 * Real.sin α = 4) :
  Real.cos α = Real.sqrt 15 / 4 ∨ Real.cos α = -Real.sqrt 15 / 4 := by
  sorry

end cosine_value_l383_38310


namespace forty_men_handshakes_l383_38315

/-- The maximum number of handshakes without cyclic handshakes for n people -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 40 men, the maximum number of handshakes without cyclic handshakes is 780 -/
theorem forty_men_handshakes : max_handshakes 40 = 780 := by
  sorry

end forty_men_handshakes_l383_38315
