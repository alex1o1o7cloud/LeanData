import Mathlib

namespace largest_y_coordinate_l3832_383239

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by
  sorry

end largest_y_coordinate_l3832_383239


namespace complex_exp_add_l3832_383294

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_add (z w : ℂ) : cexp z * cexp w = cexp (z + w) := by
  sorry

end complex_exp_add_l3832_383294


namespace equal_points_per_game_l3832_383279

/-- 
Given a player who scores a total of 36 points in 3 games, 
with points equally distributed among the games,
prove that the player scores 12 points in each game.
-/
theorem equal_points_per_game 
  (total_points : ℕ) 
  (num_games : ℕ) 
  (h1 : total_points = 36) 
  (h2 : num_games = 3) 
  (h3 : total_points % num_games = 0) : 
  total_points / num_games = 12 := by
sorry


end equal_points_per_game_l3832_383279


namespace parallel_planes_normal_vectors_l3832_383289

/-- Given two planes α and β with normal vectors (1, 2, -2) and (-2, -4, k) respectively,
    if α is parallel to β, then k = 4. -/
theorem parallel_planes_normal_vectors (k : ℝ) :
  let nα : ℝ × ℝ × ℝ := (1, 2, -2)
  let nβ : ℝ × ℝ × ℝ := (-2, -4, k)
  (∃ (t : ℝ), t ≠ 0 ∧ nα = t • nβ) →
  k = 4 := by
sorry

end parallel_planes_normal_vectors_l3832_383289


namespace absolute_difference_of_mn_l3832_383276

theorem absolute_difference_of_mn (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
  sorry

end absolute_difference_of_mn_l3832_383276


namespace largest_divisor_consecutive_odd_squares_l3832_383235

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) : 
  (∃ k : ℤ, n = 2*k + 1 ∧ m = 2*k + 3) →  -- m and n are consecutive odd integers
  n < m →                                 -- n is less than m
  (∀ d : ℤ, d ∣ (m^2 - n^2) → d ≤ 8) ∧    -- 8 is an upper bound for divisors
  8 ∣ (m^2 - n^2)                         -- 8 divides m^2 - n^2
  := by sorry

end largest_divisor_consecutive_odd_squares_l3832_383235


namespace range_of_a_l3832_383250

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ 4 * a * x) → |a| ≤ (1/4 : ℝ) := by
  sorry

end range_of_a_l3832_383250


namespace original_price_sum_l3832_383292

/-- The original price of all items before price increases -/
def original_total_price (candy_box soda_can chips_bag chocolate_bar : ℝ) : ℝ :=
  candy_box + soda_can + chips_bag + chocolate_bar

/-- Theorem stating that the original total price is 22 pounds -/
theorem original_price_sum :
  original_total_price 10 6 4 2 = 22 := by
  sorry

end original_price_sum_l3832_383292


namespace specific_theater_seats_l3832_383273

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row : ℕ
  seat_increase : ℕ
  last_row : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let num_rows := (t.last_row - t.first_row) / t.seat_increase + 1
  num_rows * (t.first_row + t.last_row) / 2

/-- Theorem stating that a theater with specific parameters has 570 seats -/
theorem specific_theater_seats :
  let t : Theater := { first_row := 12, seat_increase := 2, last_row := 48 }
  total_seats t = 570 := by sorry

end specific_theater_seats_l3832_383273


namespace initial_bottles_correct_l3832_383261

/-- The number of water bottles initially in Samira's box -/
def initial_bottles : ℕ := 48

/-- The number of players on the field -/
def num_players : ℕ := 11

/-- The number of bottles each player takes in the first break -/
def bottles_first_break : ℕ := 2

/-- The number of bottles each player takes at the end of the game -/
def bottles_end_game : ℕ := 1

/-- The number of bottles remaining after the game -/
def remaining_bottles : ℕ := 15

/-- Theorem stating that the initial number of bottles is correct -/
theorem initial_bottles_correct :
  initial_bottles = num_players * (bottles_first_break + bottles_end_game) + remaining_bottles :=
by sorry

end initial_bottles_correct_l3832_383261


namespace ant_path_circle_containment_l3832_383210

/-- A closed path in a plane -/
structure ClosedPath where
  path : Set (ℝ × ℝ)
  is_closed : path.Nonempty ∧ ∃ p, p ∈ path ∧ p ∈ frontier path
  length : ℝ

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem statement -/
theorem ant_path_circle_containment (γ : ClosedPath) (h : γ.length = 1) :
  ∃ (c : Circle), c.radius = 1/4 ∧ γ.path ⊆ {p : ℝ × ℝ | dist p c.center ≤ c.radius } :=
sorry

end ant_path_circle_containment_l3832_383210


namespace existence_of_parameters_l3832_383223

theorem existence_of_parameters : ∃ (a b c : ℝ), ∀ (x : ℝ), 
  (x + a)^2 + (2*x + b)^2 + (2*x + c)^2 = (3*x + 1)^2 := by
  sorry

end existence_of_parameters_l3832_383223


namespace log_sum_equals_three_l3832_383259

theorem log_sum_equals_three : Real.log 50 + Real.log 20 = 3 * Real.log 10 := by
  sorry

end log_sum_equals_three_l3832_383259


namespace beach_conditions_l3832_383293

-- Define the weather conditions
structure WeatherConditions where
  temperature : ℝ
  sunny : Prop
  windSpeed : ℝ

-- Define when the beach is crowded
def isCrowded (w : WeatherConditions) : Prop :=
  w.temperature ≥ 85 ∧ w.sunny ∧ w.windSpeed < 15

-- Theorem statement
theorem beach_conditions (w : WeatherConditions) :
  ¬(isCrowded w) →
  (w.temperature < 85 ∨ ¬w.sunny ∨ w.windSpeed ≥ 15) :=
by
  sorry

end beach_conditions_l3832_383293


namespace combined_average_age_l3832_383207

theorem combined_average_age (room_a_count : ℕ) (room_b_count : ℕ) 
  (room_a_avg : ℝ) (room_b_avg : ℝ) :
  room_a_count = 8 →
  room_b_count = 3 →
  room_a_avg = 45 →
  room_b_avg = 20 →
  (room_a_count * room_a_avg + room_b_count * room_b_avg) / (room_a_count + room_b_count) = 38 := by
  sorry

end combined_average_age_l3832_383207


namespace probability_theorem_l3832_383247

/-- The number of tiles in box A -/
def num_tiles_A : ℕ := 20

/-- The number of tiles in box B -/
def num_tiles_B : ℕ := 30

/-- The lowest number on tiles in box A -/
def min_num_A : ℕ := 1

/-- The highest number on tiles in box A -/
def max_num_A : ℕ := 20

/-- The lowest number on tiles in box B -/
def min_num_B : ℕ := 10

/-- The highest number on tiles in box B -/
def max_num_B : ℕ := 39

/-- The probability of drawing a tile less than 10 from box A -/
def prob_A : ℚ := 9 / 20

/-- The probability of drawing a tile that is either odd or greater than 35 from box B -/
def prob_B : ℚ := 17 / 30

theorem probability_theorem :
  prob_A * prob_B = 51 / 200 := by sorry

end probability_theorem_l3832_383247


namespace functional_equation_solution_l3832_383275

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_eq : ∀ x y, f x * f y = f (x - y)) :
  (∀ x, f x = 1) ∨ (∀ x, f x = -1) := by sorry

end functional_equation_solution_l3832_383275


namespace complex_fraction_sum_l3832_383258

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) = (a + Complex.I) / (b + 2 * Complex.I) → a + b = 4 := by
  sorry

end complex_fraction_sum_l3832_383258


namespace negative_5643_mod_10_l3832_383254

theorem negative_5643_mod_10 :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5643 [ZMOD 10] := by
  sorry

end negative_5643_mod_10_l3832_383254


namespace prob_select_B_is_one_fourth_prob_select_B_and_C_is_one_sixth_l3832_383221

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the total number of students
def total_students : ℕ := 4

-- Define the probability of selecting one student
def prob_select_one (s : Student) : ℚ :=
  1 / total_students

-- Define the probability of selecting two specific students
def prob_select_two (s1 s2 : Student) : ℚ :=
  2 / (total_students * (total_students - 1))

-- Theorem for part 1
theorem prob_select_B_is_one_fourth :
  prob_select_one Student.B = 1 / 4 := by sorry

-- Theorem for part 2
theorem prob_select_B_and_C_is_one_sixth :
  prob_select_two Student.B Student.C = 1 / 6 := by sorry

end prob_select_B_is_one_fourth_prob_select_B_and_C_is_one_sixth_l3832_383221


namespace count_even_perfect_square_factors_l3832_383297

/-- The number of even perfect square factors of 2^6 * 7^3 * 3^4 -/
def evenPerfectSquareFactors : ℕ := 18

/-- The exponent of 2 in the given number -/
def exponent2 : ℕ := 6

/-- The exponent of 7 in the given number -/
def exponent7 : ℕ := 3

/-- The exponent of 3 in the given number -/
def exponent3 : ℕ := 4

theorem count_even_perfect_square_factors :
  evenPerfectSquareFactors = (exponent2 / 2 + 1) * ((exponent7 / 2) + 1) * ((exponent3 / 2) + 1) :=
by sorry

end count_even_perfect_square_factors_l3832_383297


namespace victor_final_books_l3832_383249

/-- The number of books Victor has after various transactions -/
def final_books (initial : ℝ) (bought : ℝ) (given : ℝ) (donated : ℝ) : ℝ :=
  initial + bought - given - donated

/-- Theorem stating that Victor ends up with 19.8 books -/
theorem victor_final_books :
  final_books 35.5 12.3 7.2 20.8 = 19.8 := by
  sorry

end victor_final_books_l3832_383249


namespace square_between_squares_l3832_383204

theorem square_between_squares (n k ℓ x : ℕ) : 
  (x^2 < n) → (n < (x+1)^2) → (n - x^2 = k) → ((x+1)^2 - n = ℓ) →
  n - k*ℓ = (x^2 - n + x)^2 := by
sorry

end square_between_squares_l3832_383204


namespace at_least_one_alarm_probability_l3832_383244

theorem at_least_one_alarm_probability (pA pB : ℝ) 
  (hpA : 0 ≤ pA ∧ pA ≤ 1) (hpB : 0 ≤ pB ∧ pB ≤ 1) :
  1 - (1 - pA) * (1 - pB) = pA + pB - pA * pB :=
by sorry

end at_least_one_alarm_probability_l3832_383244


namespace complex_power_twelve_l3832_383213

/-- If z = 2 cos(π/8) * (sin(3π/4) + i*cos(3π/4) + i), then z^12 = -64i. -/
theorem complex_power_twelve (z : ℂ) : 
  z = 2 * Real.cos (π/8) * (Real.sin (3*π/4) + Complex.I * Real.cos (3*π/4) + Complex.I) → 
  z^12 = -64 * Complex.I := by
  sorry

end complex_power_twelve_l3832_383213


namespace F_range_l3832_383270

def F (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem F_range : Set.range F = Set.Ici 4 := by sorry

end F_range_l3832_383270


namespace spade_calculation_l3832_383217

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 2 (spade 6 1) = -1221 := by
  sorry

end spade_calculation_l3832_383217


namespace complex_modulus_of_z_l3832_383299

theorem complex_modulus_of_z (z : ℂ) : z = 1 - (1 / Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_of_z_l3832_383299


namespace smallest_congruent_to_zero_l3832_383291

theorem smallest_congruent_to_zero : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → k < 10 → n % k = 0) ∧
  (∀ (m : ℕ), m > 0 → (∀ (k : ℕ), k > 0 → k < 10 → m % k = 0) → m ≥ n) :=
by sorry

end smallest_congruent_to_zero_l3832_383291


namespace cube_volume_from_space_diagonal_l3832_383203

/-- The volume of a cube with a space diagonal of 10 units is 1000 cubic units -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 10 → s^3 = 1000 := by
  sorry

end cube_volume_from_space_diagonal_l3832_383203


namespace factorization_sum_l3832_383200

theorem factorization_sum (a b : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 125 * x - 100 = (5 * x + a) * (5 * x + b)) →
  a + b = -25 :=
by sorry

end factorization_sum_l3832_383200


namespace inverse_47_mod_48_l3832_383228

theorem inverse_47_mod_48 : ∃! x : ℕ, x ∈ Finset.range 48 ∧ (47 * x) % 48 = 1 :=
by
  use 47
  sorry

end inverse_47_mod_48_l3832_383228


namespace union_of_sets_l3832_383257

def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) :
  A a ∩ B a b = {1/4} → A a ∪ B a b = {-2, 1, 1/4} := by
  sorry

end union_of_sets_l3832_383257


namespace conic_sections_l3832_383214

-- Hyperbola
def hyperbola_equation (e : ℝ) (c : ℝ) : Prop :=
  e = Real.sqrt 3 ∧ c = 5 * Real.sqrt 3 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 25 - y^2 / 50 = 1)

-- Ellipse
def ellipse_equation (e : ℝ) (d : ℝ) : Prop :=
  e = 1/2 ∧ d = 4 * Real.sqrt 3 →
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y : ℝ, y^2 / a^2 + x^2 / b^2 = 1 ↔ y^2 / 12 + x^2 / 9 = 1)

-- Parabola
def parabola_equation (p : ℝ) : Prop :=
  p = 4 →
  ∀ x y : ℝ, x^2 = 4 * p * y ↔ x^2 = 8 * y

theorem conic_sections :
  ∀ (e_hyp e_ell c d p : ℝ),
    hyperbola_equation e_hyp c ∧
    ellipse_equation e_ell d ∧
    parabola_equation p :=
by sorry

end conic_sections_l3832_383214


namespace solve_equation_l3832_383285

theorem solve_equation (x : ℝ) :
  let y := 1 / (4 * x^2 + 2 * x + 1)
  y = 1 → x = 0 ∨ x = -1/2 := by
  sorry

end solve_equation_l3832_383285


namespace tax_calculation_l3832_383262

/-- Calculate tax given income and tax rate -/
def calculate_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

/-- Calculate total tax for given gross pay and tax brackets -/
def total_tax (gross_pay : ℝ) : ℝ :=
  let tax1 := calculate_tax 1500 0.10
  let tax2 := calculate_tax 2000 0.15
  let tax3 := calculate_tax (gross_pay - 1500 - 2000) 0.20
  tax1 + tax2 + tax3

/-- Apply standard deduction to total tax -/
def tax_after_deduction (total_tax : ℝ) (deduction : ℝ) : ℝ :=
  total_tax - deduction

theorem tax_calculation (gross_pay : ℝ) (deduction : ℝ) 
  (h1 : gross_pay = 4500)
  (h2 : deduction = 100) :
  tax_after_deduction (total_tax gross_pay) deduction = 550 := by
  sorry

#eval tax_after_deduction (total_tax 4500) 100

end tax_calculation_l3832_383262


namespace ryosuke_trip_gas_cost_l3832_383227

/-- Calculates the cost of gas for a trip given odometer readings, fuel efficiency, and gas price -/
def gas_cost_for_trip (initial_reading final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gas_used := (distance : ℚ) / fuel_efficiency
  gas_used * gas_price

/-- Theorem: The cost of gas for Ryosuke's trip is approximately $3.47 -/
theorem ryosuke_trip_gas_cost :
  let cost := gas_cost_for_trip 74568 74592 28 (405/100)
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/100) ∧ |cost - (347/100)| < ε := by
  sorry

#eval gas_cost_for_trip 74568 74592 28 (405/100)

end ryosuke_trip_gas_cost_l3832_383227


namespace apple_picking_multiple_l3832_383220

theorem apple_picking_multiple (K : ℕ) (M : ℕ) : 
  K + 274 = 340 → 
  274 = M * K + 10 →
  M = 4 := by sorry

end apple_picking_multiple_l3832_383220


namespace students_taking_music_l3832_383274

theorem students_taking_music (total : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) :
  total = 500 →
  art = 20 →
  both = 10 →
  neither = 470 →
  total - neither - art + both = 20 :=
by
  sorry

end students_taking_music_l3832_383274


namespace sum_of_fractions_l3832_383226

theorem sum_of_fractions : 
  let fractions : List ℚ := [2/8, 4/8, 6/8, 8/8, 10/8, 12/8, 14/8, 16/8, 18/8, 20/8]
  fractions.sum = 13.75 := by
sorry

end sum_of_fractions_l3832_383226


namespace combined_weight_l3832_383236

/-- Given weights of John, Mary, and Jamison, prove their combined weight -/
theorem combined_weight 
  (mary_weight : ℝ) 
  (john_weight : ℝ) 
  (jamison_weight : ℝ)
  (h1 : john_weight = mary_weight * (5/4))
  (h2 : mary_weight = jamison_weight - 20)
  (h3 : mary_weight = 160) :
  mary_weight + john_weight + jamison_weight = 540 := by
  sorry

end combined_weight_l3832_383236


namespace inequality_proof_l3832_383268

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
sorry

end inequality_proof_l3832_383268


namespace difference_of_squares_75_25_l3832_383248

theorem difference_of_squares_75_25 : 75^2 - 25^2 = 5000 := by
  sorry

end difference_of_squares_75_25_l3832_383248


namespace tomato_basket_price_l3832_383219

-- Define the given values
def strawberry_plants : ℕ := 5
def tomato_plants : ℕ := 7
def strawberries_per_plant : ℕ := 14
def tomatoes_per_plant : ℕ := 16
def fruits_per_basket : ℕ := 7
def strawberry_basket_price : ℕ := 9
def total_revenue : ℕ := 186

-- Calculate total strawberries and tomatoes
def total_strawberries : ℕ := strawberry_plants * strawberries_per_plant
def total_tomatoes : ℕ := tomato_plants * tomatoes_per_plant

-- Calculate number of baskets
def strawberry_baskets : ℕ := total_strawberries / fruits_per_basket
def tomato_baskets : ℕ := total_tomatoes / fruits_per_basket

-- Define the theorem
theorem tomato_basket_price :
  (total_revenue - strawberry_baskets * strawberry_basket_price) / tomato_baskets = 6 :=
by sorry

end tomato_basket_price_l3832_383219


namespace polygon_sides_when_interior_thrice_exterior_l3832_383266

theorem polygon_sides_when_interior_thrice_exterior :
  ∀ n : ℕ,
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 :=
by
  sorry

end polygon_sides_when_interior_thrice_exterior_l3832_383266


namespace resulting_number_divisibility_l3832_383237

theorem resulting_number_divisibility : ∃ k : ℕ, (722425 + 335) = 30 * k := by
  sorry

end resulting_number_divisibility_l3832_383237


namespace percentage_calculation_l3832_383251

theorem percentage_calculation (x y : ℝ) (h : x = 875.3 ∧ y = 318.65) : 
  (y / x) * 100 = 36.4 := by sorry

end percentage_calculation_l3832_383251


namespace intersection_M_N_l3832_383265

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 2, 3}
def complement_N : Finset ℕ := {1, 2, 4}

theorem intersection_M_N :
  (M ∩ (U \ complement_N) : Finset ℕ) = {0, 3} := by sorry

end intersection_M_N_l3832_383265


namespace rectangle_area_change_l3832_383286

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let A := L * B
  let L' := L / 2
  let A' := (3 / 2) * A
  let B' := A' / L'
  B' = 3 * B :=
by sorry

end rectangle_area_change_l3832_383286


namespace perpendicular_preservation_l3832_383252

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_preservation 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perpendicular m α) 
  (h4 : parallel_lines m n) 
  (h5 : parallel_planes α β) : 
  perpendicular n β :=
sorry

end perpendicular_preservation_l3832_383252


namespace triangle_angle_45_degrees_l3832_383216

theorem triangle_angle_45_degrees (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- angles are positive
  A + B + C = 180 → -- sum of angles in a triangle is 180°
  B + C = 3 * A → -- given condition
  A = 45 ∨ B = 45 ∨ C = 45 := by sorry

end triangle_angle_45_degrees_l3832_383216


namespace imaginary_part_of_z_l3832_383278

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (2 + i) / (1 + i)^2
  (z.im : ℝ) = -1 := by
  sorry

end imaginary_part_of_z_l3832_383278


namespace square_area_ratio_sqrt_l3832_383230

theorem square_area_ratio_sqrt (side_C side_D : ℝ) (h1 : side_C = 45) (h2 : side_D = 60) :
  Real.sqrt ((side_C ^ 2) / (side_D ^ 2)) = 3 / 4 := by
  sorry

end square_area_ratio_sqrt_l3832_383230


namespace quadratic_value_l3832_383298

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_value (a b c : ℝ) :
  (∃ (x : ℝ), f a b c x = -6 ∧ ∀ (y : ℝ), f a b c y ≥ -6) ∧  -- Minimum value is -6
  (∀ (x : ℝ), f a b c x ≥ f a b c (-2)) ∧                   -- Minimum occurs at x = -2
  f a b c 0 = 20 →                                          -- Passes through (0, 20)
  f a b c (-3) = 0.5 :=                                     -- Value at x = -3 is 0.5
by sorry

end quadratic_value_l3832_383298


namespace union_of_A_and_B_l3832_383260

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 4} := by sorry

end union_of_A_and_B_l3832_383260


namespace building_floors_l3832_383238

/-- Given information about three buildings A, B, and C, prove that Building C has 59 floors. -/
theorem building_floors :
  let floors_A : ℕ := 4
  let floors_B : ℕ := floors_A + 9
  let floors_C : ℕ := 5 * floors_B - 6
  floors_C = 59 := by sorry

end building_floors_l3832_383238


namespace octal_addition_theorem_l3832_383229

/-- Represents a number in base 8 --/
def OctalNumber := List Nat

/-- Converts a natural number to its octal representation --/
def toOctal (n : Nat) : OctalNumber := sorry

/-- Converts an octal number to its decimal representation --/
def fromOctal (o : OctalNumber) : Nat := sorry

/-- Adds two octal numbers --/
def addOctal (a b : OctalNumber) : OctalNumber := sorry

theorem octal_addition_theorem :
  let a := [5, 3, 2, 6]
  let b := [1, 4, 7, 3]
  addOctal a b = [7, 0, 4, 3] := by sorry

end octal_addition_theorem_l3832_383229


namespace inverse_proportion_l3832_383206

/-- Given that γ is inversely proportional to δ, prove that if γ = 5 when δ = 15, then γ = 5/3 when δ = 45. -/
theorem inverse_proportion (γ δ : ℝ) (h : ∃ k : ℝ, ∀ x y, γ * x = k ∧ y * δ = k) 
  (h1 : γ = 5 ∧ δ = 15) : 
  (γ = 5/3 ∧ δ = 45) :=
sorry

end inverse_proportion_l3832_383206


namespace compound_composition_l3832_383233

def atomic_weight_N : ℕ := 14
def atomic_weight_H : ℕ := 1
def atomic_weight_Br : ℕ := 80
def molecular_weight : ℕ := 98

theorem compound_composition (n : ℕ) : 
  atomic_weight_N + n * atomic_weight_H + atomic_weight_Br = molecular_weight → n = 4 := by
  sorry

end compound_composition_l3832_383233


namespace tracing_time_5x5_l3832_383224

/-- Represents a rectangular grid with width and height -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Calculates the total length of lines in a grid -/
def totalLength (g : Grid) : ℕ :=
  (g.width + 1) * g.height + (g.height + 1) * g.width

/-- Time taken to trace a grid given a reference grid and its tracing time -/
def tracingTime (refGrid : Grid) (refTime : ℕ) (targetGrid : Grid) : ℕ :=
  (totalLength targetGrid * refTime) / (totalLength refGrid)

theorem tracing_time_5x5 :
  let refGrid : Grid := { width := 7, height := 3 }
  let targetGrid : Grid := { width := 5, height := 5 }
  tracingTime refGrid 26 targetGrid = 30 := by
  sorry

end tracing_time_5x5_l3832_383224


namespace ellipse_foci_distance_l3832_383242

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 4) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 21 := by
  sorry

end ellipse_foci_distance_l3832_383242


namespace divisibility_by_35_l3832_383282

theorem divisibility_by_35 : ∃! n : ℕ, n < 10 ∧ 35 ∣ (80000 + 10000 * n + 975) := by
  sorry

end divisibility_by_35_l3832_383282


namespace intersection_when_a_is_4_subset_condition_l3832_383283

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 7 < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}

-- Theorem 1: When a = 4, A ∩ B = (1, 6)
theorem intersection_when_a_is_4 : A ∩ (B 4) = Set.Ioo 1 6 := by sorry

-- Theorem 2: A ⊆ B if and only if a ∈ (-∞, -7] ∪ [5, +∞)
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ≤ -7 ∨ a ≥ 5 := by sorry

end intersection_when_a_is_4_subset_condition_l3832_383283


namespace beam_equation_l3832_383243

/-- The equation for buying beams problem -/
theorem beam_equation (x : ℕ+) (h : x > 1) : 
  (3 : ℚ) * ((x : ℚ) - 1) = 6210 / (x : ℚ) :=
sorry

/-- The total cost of beams in wen -/
def total_cost : ℕ := 6210

/-- The transportation cost per beam in wen -/
def transport_cost : ℕ := 3

/-- The number of beams that can be bought -/
def num_beams : ℕ+ := sorry

end beam_equation_l3832_383243


namespace negation_of_existence_l3832_383212

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end negation_of_existence_l3832_383212


namespace cube_sum_and_reciprocal_l3832_383225

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end cube_sum_and_reciprocal_l3832_383225


namespace intersection_implies_a_zero_l3832_383231

theorem intersection_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {a^2, a + 1, -1}
  let B : Set ℝ := {2*a - 1, |a - 2|, 3*a^2 + 4}
  A ∩ B = {-1} → a = 0 := by
sorry

end intersection_implies_a_zero_l3832_383231


namespace sqrt_less_than_3y_iff_l3832_383209

theorem sqrt_less_than_3y_iff (y : ℝ) (h : y > 0) : 
  Real.sqrt y < 3 * y ↔ y > 1 / 9 := by
sorry

end sqrt_less_than_3y_iff_l3832_383209


namespace woman_lawyer_probability_l3832_383240

/-- Represents a study group with members, women, and lawyers. -/
structure StudyGroup where
  total_members : ℕ
  women_percentage : ℝ
  lawyer_percentage : ℝ
  women_percentage_valid : 0 ≤ women_percentage ∧ women_percentage ≤ 1
  lawyer_percentage_valid : 0 ≤ lawyer_percentage ∧ lawyer_percentage ≤ 1

/-- Calculates the probability of selecting a woman lawyer from the study group. -/
def probability_woman_lawyer (group : StudyGroup) : ℝ :=
  group.women_percentage * group.lawyer_percentage

/-- Theorem stating that the probability of selecting a woman lawyer is 0.08
    given the specified conditions. -/
theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.women_percentage = 0.4)
  (h2 : group.lawyer_percentage = 0.2) : 
  probability_woman_lawyer group = 0.08 := by
  sorry

#check woman_lawyer_probability

end woman_lawyer_probability_l3832_383240


namespace meeting_point_l3832_383215

/-- Represents the walking speed of a person -/
structure WalkingSpeed where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents a person walking around the block -/
structure Walker where
  name : String
  speed : WalkingSpeed

/-- The scenario of Jane and Hector walking around the block -/
structure WalkingScenario where
  jane : Walker
  hector : Walker
  block_size : ℝ
  jane_speed_ratio : ℝ
  start_point : ℝ
  jane_speed_twice_hector : jane.speed.speed = 2 * hector.speed.speed
  block_size_positive : block_size > 0
  jane_speed_ratio_half : jane_speed_ratio = 1/2
  start_point_zero : start_point = 0

/-- The theorem stating where Jane and Hector meet -/
theorem meeting_point (scenario : WalkingScenario) : 
  ∃ t : ℝ, t > 0 ∧ 
  (scenario.hector.speed.speed * t + scenario.jane.speed.speed * t = scenario.block_size) ∧
  (scenario.hector.speed.speed * t = 12) := by
  sorry

end meeting_point_l3832_383215


namespace track_team_composition_l3832_383222

/-- The number of children on a track team after changes in composition -/
theorem track_team_composition (initial_girls initial_boys girls_joined boys_quit : ℕ) :
  initial_girls = 18 →
  initial_boys = 15 →
  girls_joined = 7 →
  boys_quit = 4 →
  (initial_girls + girls_joined) + (initial_boys - boys_quit) = 36 := by
  sorry


end track_team_composition_l3832_383222


namespace sergio_total_amount_l3832_383256

/-- Represents the total amount Mr. Sergio got from selling his fruits -/
def total_amount (mango_produce : ℕ) (price_per_kg : ℕ) : ℕ :=
  let apple_produce := 2 * mango_produce
  let orange_produce := mango_produce + 200
  (apple_produce + mango_produce + orange_produce) * price_per_kg

/-- Theorem stating that Mr. Sergio's total amount is $90,000 -/
theorem sergio_total_amount :
  total_amount 400 50 = 90000 := by
  sorry

end sergio_total_amount_l3832_383256


namespace weekly_profit_calculation_l3832_383263

def planned_daily_sales : ℕ := 10

def daily_differences : List ℤ := [4, -3, -2, 7, -6, 18, -5]

def selling_price : ℕ := 65

def num_workers : ℕ := 3

def daily_expense_per_worker : ℕ := 80

def packaging_fee : ℕ := 5

def total_days : ℕ := 7

theorem weekly_profit_calculation :
  let total_sales := planned_daily_sales * total_days + daily_differences.sum
  let revenue := total_sales * (selling_price - packaging_fee)
  let expenses := num_workers * daily_expense_per_worker * total_days
  let profit := revenue - expenses
  profit = 3300 := by sorry

end weekly_profit_calculation_l3832_383263


namespace distance_between_points_l3832_383296

theorem distance_between_points :
  let x₁ : ℝ := 2
  let y₁ : ℝ := 3
  let x₂ : ℝ := 5
  let y₂ : ℝ := 9
  let distance := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  distance = 3 * Real.sqrt 5 :=
by sorry

end distance_between_points_l3832_383296


namespace max_remainder_239_div_n_l3832_383232

theorem max_remainder_239_div_n (n : ℕ) (h : n < 135) :
  (Finset.range n).sup (λ m => 239 % m) = 119 := by
  sorry

end max_remainder_239_div_n_l3832_383232


namespace real_part_of_i_squared_times_one_plus_i_l3832_383295

theorem real_part_of_i_squared_times_one_plus_i :
  Complex.re (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end real_part_of_i_squared_times_one_plus_i_l3832_383295


namespace original_figure_area_l3832_383280

/-- The area of the original figure given the properties of its intuitive diagram --/
theorem original_figure_area (height : ℝ) (top_angle : ℝ) (area_ratio : ℝ) : 
  height = 2 → 
  top_angle = 120 * π / 180 → 
  area_ratio = 2 * Real.sqrt 2 → 
  (1 / 2) * (4 * height) * (4 * height) * Real.sin top_angle * area_ratio = 8 * Real.sqrt 6 := by
  sorry

#check original_figure_area

end original_figure_area_l3832_383280


namespace correct_good_carrots_l3832_383287

/-- The number of good carrots given the number of carrots picked by Haley and her mother, and the number of bad carrots. -/
def goodCarrots (haleyCarrots motherCarrots badCarrots : ℕ) : ℕ :=
  haleyCarrots + motherCarrots - badCarrots

/-- Theorem stating that the number of good carrots is 64 given the specific conditions. -/
theorem correct_good_carrots :
  goodCarrots 39 38 13 = 64 := by
  sorry

end correct_good_carrots_l3832_383287


namespace equation_solution_l3832_383246

theorem equation_solution :
  ∃ x : ℝ, (x + Real.sqrt (x^2 - x) = 2) ∧ (x = 4/3) := by
  sorry

end equation_solution_l3832_383246


namespace smallest_consecutive_cubes_with_square_difference_l3832_383205

theorem smallest_consecutive_cubes_with_square_difference :
  (∀ n : ℕ, n < 7 → ¬∃ k : ℕ, (n + 1)^3 - n^3 = k^2) ∧
  ∃ k : ℕ, 8^3 - 7^3 = k^2 := by
sorry

end smallest_consecutive_cubes_with_square_difference_l3832_383205


namespace max_sum_of_squares_l3832_383284

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 100 →
  a * d + b * c = 250 →
  c * d = 144 →
  a^2 + b^2 + c^2 + d^2 ≤ 1760 :=
by sorry

end max_sum_of_squares_l3832_383284


namespace divisibility_equivalence_l3832_383290

theorem divisibility_equivalence (m n k : ℕ) (h : m > n) :
  (∃ a : ℤ, 4^m - 4^n = a * 3^(k+1)) ↔ (∃ b : ℤ, m - n = b * 3^k) :=
sorry

end divisibility_equivalence_l3832_383290


namespace rhombus_area_l3832_383271

/-- The area of a rhombus with side length 10 and angle 60 degrees between sides is 50√3 -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 10 →
  angle = 60 * π / 180 →
  side_length * side_length * Real.sin angle = 50 * Real.sqrt 3 :=
by sorry

end rhombus_area_l3832_383271


namespace congruence_solutions_count_l3832_383211

theorem congruence_solutions_count : 
  ∃! (s : Finset Nat), 
    (∀ x ∈ s, x > 0 ∧ x < 150 ∧ (x + 17) % 45 = 80 % 45) ∧ 
    (∀ x, x > 0 ∧ x < 150 ∧ (x + 17) % 45 = 80 % 45 → x ∈ s) ∧
    s.card = 3 := by
  sorry

end congruence_solutions_count_l3832_383211


namespace coefficient_sum_l3832_383272

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end coefficient_sum_l3832_383272


namespace election_votes_l3832_383277

theorem election_votes (total_votes : ℕ) : 
  (0.7 * (0.85 * total_votes : ℝ) = 333200) → 
  total_votes = 560000 := by
sorry

end election_votes_l3832_383277


namespace geometric_sequence_sum_l3832_383208

/-- Given a geometric sequence {aₙ} satisfying a₂ + a₄ = 20 and a₃ + a₅ = 40, prove a₅ + a₇ = 160 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_sum1 : a 2 + a 4 = 20) (h_sum2 : a 3 + a 5 = 40) : 
  a 5 + a 7 = 160 := by
  sorry

end geometric_sequence_sum_l3832_383208


namespace distance_covered_l3832_383264

/-- Proves that the total distance covered is 10 km given the specified conditions --/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 3.75)
  (h4 : (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time) :
  total_distance = 10 :=
by sorry

#check distance_covered

end distance_covered_l3832_383264


namespace exactly_one_positive_integer_solution_l3832_383288

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ+), 24 - 6 * (n : ℝ) > 12 :=
sorry

end exactly_one_positive_integer_solution_l3832_383288


namespace initial_distance_between_cars_l3832_383202

/-- 
Given two cars A and B traveling in the same direction:
- Car A's speed is 58 mph
- Car B's speed is 50 mph
- After 6 hours, Car A is 8 miles ahead of Car B
Prove that the initial distance between Car A and Car B is 40 miles
-/
theorem initial_distance_between_cars (speed_A speed_B time_elapsed final_distance : ℝ) 
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : time_elapsed = 6)
  (h4 : final_distance = 8) :
  speed_A * time_elapsed - speed_B * time_elapsed - final_distance = 40 :=
by sorry

end initial_distance_between_cars_l3832_383202


namespace airplane_seats_theorem_l3832_383218

theorem airplane_seats_theorem :
  ∀ (total_seats : ℕ),
  (30 : ℕ) +                            -- First Class seats
  (total_seats * 20 / 100 : ℕ) +         -- Business Class seats (20% of total)
  (15 : ℕ) +                            -- Premium Economy Class seats
  (total_seats - (30 + (total_seats * 20 / 100) + 15) : ℕ) -- Economy Class seats
  = total_seats →
  total_seats = 288 := by
sorry

end airplane_seats_theorem_l3832_383218


namespace inscribed_circle_rectangle_area_l3832_383281

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (l w : ℝ),
  r = 7 →
  l = 3 * w →
  w = 2 * r →
  l * w = 588 :=
by
  sorry

end inscribed_circle_rectangle_area_l3832_383281


namespace total_boxes_in_cases_l3832_383267

/-- The number of cases Jenny needs to deliver -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 8

/-- Theorem: The total number of boxes in the cases Jenny needs to deliver is 24 -/
theorem total_boxes_in_cases : num_cases * boxes_per_case = 24 := by
  sorry

end total_boxes_in_cases_l3832_383267


namespace product_xy_l3832_383253

theorem product_xy (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := by
  sorry

end product_xy_l3832_383253


namespace number_problem_l3832_383241

theorem number_problem (x y a : ℝ) :
  x * y = 1 →
  (a^((x + y)^2)) / (a^((x - y)^2)) = 1296 →
  a = 6 := by sorry

end number_problem_l3832_383241


namespace sector_central_angle_l3832_383201

/-- Given a sector with radius 2 and area 4, its central angle (in absolute value) is 2 radians -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 4) :
  |2 * area / r^2| = 2 :=
by sorry

end sector_central_angle_l3832_383201


namespace smallest_c_value_l3832_383269

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos (b * (-π/4) + c)) →
  c ≥ π/4 :=
sorry

end smallest_c_value_l3832_383269


namespace lg_root_relationship_l3832_383245

theorem lg_root_relationship : ∃ (M1 M2 M3 : ℝ),
  M1 > 0 ∧ M2 > 0 ∧ M3 > 0 ∧
  Real.log M1 / Real.log 10 < M1 ^ (1/10) ∧
  Real.log M2 / Real.log 10 > M2 ^ (1/10) ∧
  Real.log M3 / Real.log 10 = M3 ^ (1/10) :=
by sorry

end lg_root_relationship_l3832_383245


namespace prob_at_least_two_same_correct_l3832_383234

/-- The number of sides on each die -/
def num_sides : Nat := 8

/-- The number of dice rolled -/
def num_dice : Nat := 7

/-- The probability of rolling 7 fair 8-sided dice and getting at least two dice showing the same number -/
def prob_at_least_two_same : ℚ := 319 / 320

/-- Theorem stating that the probability of at least two dice showing the same number
    when rolling 7 fair 8-sided dice is equal to 319/320 -/
theorem prob_at_least_two_same_correct :
  (1 : ℚ) - (Nat.factorial num_sides / Nat.factorial (num_sides - num_dice)) / (num_sides ^ num_dice) = prob_at_least_two_same := by
  sorry


end prob_at_least_two_same_correct_l3832_383234


namespace unique_determination_from_subset_sums_l3832_383255

/-- Given a set of n integers, this function returns all possible subset sums excluding the empty subset -/
def allSubsetSums (s : Finset Int) : Finset Int :=
  sorry

theorem unique_determination_from_subset_sums
  (n : Nat)
  (s : Finset Int)
  (h1 : s.card = n)
  (h2 : 0 ∉ allSubsetSums s)
  (h3 : (allSubsetSums s).card = 2^n - 1) :
  ∀ t : Finset Int, allSubsetSums s = allSubsetSums t → s = t :=
sorry

end unique_determination_from_subset_sums_l3832_383255
