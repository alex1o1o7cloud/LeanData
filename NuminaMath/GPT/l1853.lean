import Mathlib

namespace NUMINAMATH_GPT_difference_of_numbers_l1853_185391

theorem difference_of_numbers (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l1853_185391


namespace NUMINAMATH_GPT_total_cost_tom_pays_for_trip_l1853_185308

/-- Tom needs to get 10 different vaccines and a doctor's visit to go to Barbados.
    Each vaccine costs $45.
    The doctor's visit costs $250.
    Insurance will cover 80% of these medical bills.
    The trip itself costs $1200.
    Prove that the total amount Tom has to pay for his trip to Barbados, including medical expenses, is $1340. -/
theorem total_cost_tom_pays_for_trip : 
  let cost_per_vaccine := 45
  let number_of_vaccines := 10
  let cost_doctor_visit := 250
  let insurance_coverage_rate := 0.8
  let trip_cost := 1200
  let total_medical_cost := (number_of_vaccines * cost_per_vaccine) + cost_doctor_visit
  let insurance_coverage := insurance_coverage_rate * total_medical_cost
  let net_medical_cost := total_medical_cost - insurance_coverage
  let total_cost := trip_cost + net_medical_cost
  total_cost = 1340 := 
by 
  sorry

end NUMINAMATH_GPT_total_cost_tom_pays_for_trip_l1853_185308


namespace NUMINAMATH_GPT_gcf_factorial_5_6_l1853_185368

theorem gcf_factorial_5_6 : Nat.gcd (Nat.factorial 5) (Nat.factorial 6) = Nat.factorial 5 := by
  sorry

end NUMINAMATH_GPT_gcf_factorial_5_6_l1853_185368


namespace NUMINAMATH_GPT_circle_tangent_radius_l1853_185365

theorem circle_tangent_radius (k : ℝ) (r : ℝ) (hk : k > 4) 
  (h_tangent1 : dist (0, k) (x, x) = r)
  (h_tangent2 : dist (0, k) (x, -x) = r) 
  (h_tangent3 : dist (0, k) (x, 4) = r) : 
  r = 4 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_circle_tangent_radius_l1853_185365


namespace NUMINAMATH_GPT_evaluate_g_expressions_l1853_185338

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g_expressions : 3 * g 5 + 4 * g (-2) = 287 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_expressions_l1853_185338


namespace NUMINAMATH_GPT_total_fish_correct_l1853_185309

def Billy_fish : ℕ := 10
def Tony_fish : ℕ := 3 * Billy_fish
def Sarah_fish : ℕ := Tony_fish + 5
def Bobby_fish : ℕ := 2 * Sarah_fish
def Jenny_fish : ℕ := Bobby_fish - 4
def total_fish : ℕ := Billy_fish + Tony_fish + Sarah_fish + Bobby_fish + Jenny_fish

theorem total_fish_correct : total_fish = 211 := by
  sorry

end NUMINAMATH_GPT_total_fish_correct_l1853_185309


namespace NUMINAMATH_GPT_total_candies_count_l1853_185310

variable (purple_candies orange_candies yellow_candies : ℕ)

theorem total_candies_count
  (ratio_condition : purple_candies / orange_candies = 2 / 4 ∧ purple_candies / yellow_candies = 2 / 5)
  (yellow_candies_count : yellow_candies = 40) :
  purple_candies + orange_candies + yellow_candies = 88 :=
by
  sorry

end NUMINAMATH_GPT_total_candies_count_l1853_185310


namespace NUMINAMATH_GPT_product_of_remaining_numbers_l1853_185374

theorem product_of_remaining_numbers {a b c d : ℕ} (h1 : a = 11) (h2 : b = 22) (h3 : c = 33) (h4 : d = 44) :
  ∃ (x y z : ℕ), 
  (∃ n: ℕ, (a + b + c + d) - n * 3 = 3 ∧ -- We removed n groups of 3 different numbers
             x + y + z = 2 * n + (a + b + c + d)) ∧ -- We added 2 * n numbers back
  x * y * z = 12 := 
sorry

end NUMINAMATH_GPT_product_of_remaining_numbers_l1853_185374


namespace NUMINAMATH_GPT_sum_largest_and_smallest_l1853_185358

-- Define the three-digit number properties
def hundreds_digit := 4
def tens_digit := 8
def A : ℕ := sorry  -- Placeholder for the digit A

-- Define the number based on the digits
def number (A : ℕ) : ℕ := 100 * hundreds_digit + 10 * tens_digit + A

-- Hypotheses
axiom A_range : 0 ≤ A ∧ A ≤ 9

-- Largest and smallest possible numbers
def largest_number := number 9
def smallest_number := number 0

-- Prove the sum
theorem sum_largest_and_smallest : largest_number + smallest_number = 969 :=
by
  sorry

end NUMINAMATH_GPT_sum_largest_and_smallest_l1853_185358


namespace NUMINAMATH_GPT_rectangle_height_l1853_185363

theorem rectangle_height (y : ℝ) (h_pos : 0 < y) 
  (h_area : let length := 5 - (-3)
            let height := y - (-2)
            length * height = 112) : y = 12 := 
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_rectangle_height_l1853_185363


namespace NUMINAMATH_GPT_return_speed_is_48_l1853_185312

variable (d r : ℕ)
variable (t_1 t_2 : ℚ)

-- Given conditions
def distance_each_way : Prop := d = 120
def time_to_travel_A_to_B : Prop := t_1 = d / 80
def time_to_travel_B_to_A : Prop := t_2 = d / r
def average_speed_round_trip : Prop := 60 * (t_1 + t_2) = 2 * d

-- Statement to prove
theorem return_speed_is_48 :
  distance_each_way d ∧
  time_to_travel_A_to_B d t_1 ∧
  time_to_travel_B_to_A d r t_2 ∧
  average_speed_round_trip d t_1 t_2 →
  r = 48 :=
by
  intros
  sorry

end NUMINAMATH_GPT_return_speed_is_48_l1853_185312


namespace NUMINAMATH_GPT_rectangle_perimeter_l1853_185339

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 :=
by
  -- Proving the theorem here
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1853_185339


namespace NUMINAMATH_GPT_n_is_square_l1853_185336

theorem n_is_square (n m : ℕ) (h1 : 3 ≤ n) (h2 : m = (n * (n - 1)) / 2) (h3 : ∃ (cards : Finset ℕ), 
  (cards.card = n) ∧ (∀ i ∈ cards, i ∈ Finset.range (m + 1)) ∧ 
  (∀ (i j : ℕ) (hi : i ∈ cards) (hj : j ∈ cards), i ≠ j → 
    ((i + j) % m) ≠ ((i + j) % m))) : 
  ∃ k : ℕ, n = k * k := 
sorry

end NUMINAMATH_GPT_n_is_square_l1853_185336


namespace NUMINAMATH_GPT_James_bought_3_CDs_l1853_185350

theorem James_bought_3_CDs :
  ∃ (cd1 cd2 cd3 : ℝ), cd1 = 1.5 ∧ cd2 = 1.5 ∧ cd3 = 2 * cd1 ∧ cd1 + cd2 + cd3 = 6 ∧ 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_James_bought_3_CDs_l1853_185350


namespace NUMINAMATH_GPT_calvin_total_insects_l1853_185330

-- Definitions based on the conditions
def roaches := 12
def scorpions := 3
def crickets := roaches / 2
def caterpillars := scorpions * 2

-- Statement of the problem
theorem calvin_total_insects : 
  roaches + scorpions + crickets + caterpillars = 27 :=
  by
    sorry

end NUMINAMATH_GPT_calvin_total_insects_l1853_185330


namespace NUMINAMATH_GPT_numbers_equal_l1853_185314

theorem numbers_equal (a b c d : ℕ)
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  a = b ∨ b = c ∨ c = d ∨ a = c ∨ a = d ∨ b = d ∨ (a = b ∧ b = c) ∨ (b = c ∧ c = d) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) :=
sorry

end NUMINAMATH_GPT_numbers_equal_l1853_185314


namespace NUMINAMATH_GPT_distance_from_M_to_x_axis_l1853_185393

-- Define the point M and its coordinates.
def point_M : ℤ × ℤ := (-9, 12)

-- Define the distance to the x-axis is simply the absolute value of the y-coordinate.
def distance_to_x_axis (p : ℤ × ℤ) : ℤ := Int.natAbs p.snd

-- Theorem stating the distance from point M to the x-axis is 12.
theorem distance_from_M_to_x_axis : distance_to_x_axis point_M = 12 := by
  sorry

end NUMINAMATH_GPT_distance_from_M_to_x_axis_l1853_185393


namespace NUMINAMATH_GPT_merchants_and_cost_l1853_185335

theorem merchants_and_cost (n C : ℕ) (h1 : 8 * n = C + 3) (h2 : 7 * n = C - 4) : n = 7 ∧ C = 53 := 
by 
  sorry

end NUMINAMATH_GPT_merchants_and_cost_l1853_185335


namespace NUMINAMATH_GPT_distance_P_to_outer_circle_l1853_185386

theorem distance_P_to_outer_circle
  (r_large r_small : ℝ) 
  (h_tangent_inner : true) 
  (h_tangent_diameter : true) 
  (P : ℝ) 
  (O1P : ℝ)
  (O2P : ℝ := r_small)
  (O1O2 : ℝ := r_large - r_small)
  (h_O1O2_eq_680 : O1O2 = 680)
  (h_O2P_eq_320 : O2P = 320) :
  r_large - O1P = 400 :=
by
  sorry

end NUMINAMATH_GPT_distance_P_to_outer_circle_l1853_185386


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1853_185306

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 0 → x^2 + x > 0) ∧ (∃ y : ℝ, y < -1 ∧ y^2 + y > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1853_185306


namespace NUMINAMATH_GPT_negation_of_all_honest_l1853_185337

-- Define the needed predicates
variable {Man : Type} -- Type for men
variable (man : Man → Prop)
variable (age : Man → ℕ)
variable (honest : Man → Prop)

-- Define the conditions and the statement we want to prove
theorem negation_of_all_honest :
  (∀ x, man x → age x > 30 → honest x) →
  (∃ x, man x ∧ age x > 30 ∧ ¬ honest x) :=
sorry

end NUMINAMATH_GPT_negation_of_all_honest_l1853_185337


namespace NUMINAMATH_GPT_sandwiches_sold_out_l1853_185316

-- Define the parameters as constant values
def original : ℕ := 9
def available : ℕ := 4

-- The theorem stating the problem and the expected result
theorem sandwiches_sold_out : (original - available) = 5 :=
by
  -- This is the placeholder for the proof
  sorry

end NUMINAMATH_GPT_sandwiches_sold_out_l1853_185316


namespace NUMINAMATH_GPT_num_solutions_eq_4_l1853_185388

theorem num_solutions_eq_4 (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  ∃ n : ℕ, n = 4 ∧ (2 + 4 * Real.cos θ - 6 * Real.sin (2 * θ) + 3 * Real.tan θ = 0) :=
sorry

end NUMINAMATH_GPT_num_solutions_eq_4_l1853_185388


namespace NUMINAMATH_GPT_smallest_number_l1853_185381

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1/2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end NUMINAMATH_GPT_smallest_number_l1853_185381


namespace NUMINAMATH_GPT_additional_hours_to_travel_l1853_185317

theorem additional_hours_to_travel (distance1 time1 distance2 : ℝ) (rate : ℝ) 
  (h1 : distance1 = 270) 
  (h2 : time1 = 3)
  (h3 : distance2 = 180)
  (h4 : rate = distance1 / time1) :
  distance2 / rate = 2 := by
  sorry

end NUMINAMATH_GPT_additional_hours_to_travel_l1853_185317


namespace NUMINAMATH_GPT_quadrilateral_area_l1853_185373

-- Define the angles in the quadrilateral ABCD
def ABD : ℝ := 20
def DBC : ℝ := 60
def ADB : ℝ := 30
def BDC : ℝ := 70

-- Define the side lengths
variables (AB CD AD BC AC BD : ℝ)

-- Prove that the area of the quadrilateral ABCD is half the product of its sides
theorem quadrilateral_area (h1 : ABD = 20) (h2 : DBC = 60) (h3 : ADB = 30) (h4 : BDC = 70)
  : (1 / 2) * (AB * CD + AD * BC) = (1 / 2) * (AB * CD + AD * BC) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1853_185373


namespace NUMINAMATH_GPT_product_of_two_numbers_l1853_185345

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by 
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1853_185345


namespace NUMINAMATH_GPT_tire_circumference_is_one_meter_l1853_185332

-- Definitions for the given conditions
def car_speed : ℕ := 24 -- in km/h
def tire_rotations_per_minute : ℕ := 400

-- Conversion factors
def km_to_m : ℕ := 1000
def hour_to_min : ℕ := 60

-- The equivalent proof problem
theorem tire_circumference_is_one_meter 
  (hs : car_speed * km_to_m / hour_to_min = 400 * tire_rotations_per_minute)
  : 400 = 400 * 1 := 
by
  sorry

end NUMINAMATH_GPT_tire_circumference_is_one_meter_l1853_185332


namespace NUMINAMATH_GPT_coordinates_of_Q_l1853_185320

theorem coordinates_of_Q (m : ℤ) (P Q : ℤ × ℤ) (hP : P = (m + 2, 2 * m + 4))
  (hQ_move : Q = (P.1, P.2 + 2)) (hQ_x_axis : Q.2 = 0) : Q = (-1, 0) :=
sorry

end NUMINAMATH_GPT_coordinates_of_Q_l1853_185320


namespace NUMINAMATH_GPT_problem_I4_1_l1853_185343

theorem problem_I4_1 
  (x y : ℝ)
  (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
sorry

end NUMINAMATH_GPT_problem_I4_1_l1853_185343


namespace NUMINAMATH_GPT_locker_count_proof_l1853_185372

theorem locker_count_proof (cost_per_digit : ℕ := 3)
  (total_cost : ℚ := 224.91) :
  (N : ℕ) = 2151 :=
by
  sorry

end NUMINAMATH_GPT_locker_count_proof_l1853_185372


namespace NUMINAMATH_GPT_car_kilometers_per_gallon_l1853_185331

theorem car_kilometers_per_gallon :
  ∀ (distance gallon_used : ℝ), distance = 120 → gallon_used = 6 →
  distance / gallon_used = 20 :=
by
  intros distance gallon_used h_distance h_gallon_used
  sorry

end NUMINAMATH_GPT_car_kilometers_per_gallon_l1853_185331


namespace NUMINAMATH_GPT_additional_interest_rate_l1853_185340

variable (P A1 A2 T SI1 SI2 R AR : ℝ)
variable (h_P : P = 9000)
variable (h_A1 : A1 = 10200)
variable (h_A2 : A2 = 10740)
variable (h_T : T = 3)
variable (h_SI1 : SI1 = A1 - P)
variable (h_SI2 : SI2 = A2 - A1)
variable (h_R : SI1 = P * R * T / 100)
variable (h_AR : SI2 = P * AR * T / 100)

theorem additional_interest_rate :
  AR = 2 := by
  sorry

end NUMINAMATH_GPT_additional_interest_rate_l1853_185340


namespace NUMINAMATH_GPT_problem_l1853_185333

theorem problem (x y : ℚ) (h1 : x + y = 10 / 21) (h2 : x - y = 1 / 63) : 
  x^2 - y^2 = 10 / 1323 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1853_185333


namespace NUMINAMATH_GPT_LCM_activities_l1853_185324

theorem LCM_activities :
  ∃ (d : ℕ), d = Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) ∧ d = 48 :=
by
  sorry

end NUMINAMATH_GPT_LCM_activities_l1853_185324


namespace NUMINAMATH_GPT_quotient_of_37_div_8_l1853_185313

theorem quotient_of_37_div_8 : (37 / 8) = 4 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_37_div_8_l1853_185313


namespace NUMINAMATH_GPT_revenue_increase_l1853_185351

theorem revenue_increase (P Q : ℝ) :
    let R := P * Q
    let P_new := 1.7 * P
    let Q_new := 0.8 * Q
    let R_new := P_new * Q_new
    R_new = 1.36 * R :=
sorry

end NUMINAMATH_GPT_revenue_increase_l1853_185351


namespace NUMINAMATH_GPT_rectangular_plot_area_l1853_185364

-- Define the conditions
def breadth := 11  -- breadth in meters
def length := 3 * breadth  -- length is thrice the breadth

-- Define the function to calculate area
def area (length breadth : ℕ) := length * breadth

-- The theorem to prove
theorem rectangular_plot_area : area length breadth = 363 := by
  sorry

end NUMINAMATH_GPT_rectangular_plot_area_l1853_185364


namespace NUMINAMATH_GPT_expression_negativity_l1853_185375

-- Given conditions: a, b, and c are lengths of the sides of a triangle
variables (a b c : ℝ)
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- To prove: (a - b)^2 - c^2 < 0
theorem expression_negativity (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a - b)^2 - c^2 < 0 :=
sorry

end NUMINAMATH_GPT_expression_negativity_l1853_185375


namespace NUMINAMATH_GPT_largest_integer_solution_l1853_185357

theorem largest_integer_solution (x : ℤ) (h : 3 - 2 * x > 0) : x ≤ 1 :=
by sorry

end NUMINAMATH_GPT_largest_integer_solution_l1853_185357


namespace NUMINAMATH_GPT_x_is_integer_if_conditions_hold_l1853_185302

theorem x_is_integer_if_conditions_hold (x : ℝ)
  (h1 : ∃ (k : ℤ), x^2 - x = k)
  (h2 : ∃ (n : ℕ), n ≥ 3 ∧ ∃ (m : ℤ), x^n - x = m) :
  ∃ (z : ℤ), x = z :=
sorry

end NUMINAMATH_GPT_x_is_integer_if_conditions_hold_l1853_185302


namespace NUMINAMATH_GPT_same_quadratic_function_b_l1853_185380

theorem same_quadratic_function_b (a c b : ℝ) :
    (∀ x : ℝ, a * (x - 2)^2 + c = (2 * x - 5) * (x - b)) → b = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_same_quadratic_function_b_l1853_185380


namespace NUMINAMATH_GPT_seashells_increase_l1853_185369

def initial_seashells : ℕ := 50
def final_seashells : ℕ := 130
def week_increment (x : ℕ) : ℕ := 4 * x + initial_seashells

theorem seashells_increase (x : ℕ) (h: final_seashells = week_increment x) : x = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_seashells_increase_l1853_185369


namespace NUMINAMATH_GPT_min_value_of_y_l1853_185360

theorem min_value_of_y (x : ℝ) : ∃ x0 : ℝ, (∀ x : ℝ, 4 * x^2 + 8 * x + 16 ≥ 12) ∧ (4 * x0^2 + 8 * x0 + 16 = 12) :=
sorry

end NUMINAMATH_GPT_min_value_of_y_l1853_185360


namespace NUMINAMATH_GPT_slope_range_l1853_185387

theorem slope_range (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ k : ℝ, k = (y + 2) / (x + 1) ∧ k ∈ Set.Ici (3 / 4) :=
sorry

end NUMINAMATH_GPT_slope_range_l1853_185387


namespace NUMINAMATH_GPT_roden_total_fish_l1853_185347

def total_goldfish : Nat :=
  15 + 10 + 3 + 4

def total_blue_fish : Nat :=
  7 + 12 + 7 + 8

def total_green_fish : Nat :=
  5 + 9 + 6

def total_purple_fish : Nat :=
  2

def total_red_fish : Nat :=
  1

def total_fish : Nat :=
  total_goldfish + total_blue_fish + total_green_fish + total_purple_fish + total_red_fish

theorem roden_total_fish : total_fish = 89 :=
by
  unfold total_fish total_goldfish total_blue_fish total_green_fish total_purple_fish total_red_fish
  sorry

end NUMINAMATH_GPT_roden_total_fish_l1853_185347


namespace NUMINAMATH_GPT_system_solutions_l1853_185399

theorem system_solutions (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = -1) : 
  b = -22 :=
by 
  sorry

end NUMINAMATH_GPT_system_solutions_l1853_185399


namespace NUMINAMATH_GPT_range_of_a_l1853_185353

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| > Real.logb 2 a) →
  0 < a ∧ a < 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1853_185353


namespace NUMINAMATH_GPT_rectangle_perimeter_l1853_185376

theorem rectangle_perimeter 
(area : ℝ) (width : ℝ) (h1 : area = 200) (h2 : width = 10) : 
    ∃ (perimeter : ℝ), perimeter = 60 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1853_185376


namespace NUMINAMATH_GPT_emily_lives_lost_l1853_185322

variable (L : ℕ)
variable (initial_lives : ℕ) (extra_lives : ℕ) (final_lives : ℕ)

-- Conditions based on the problem statement
axiom initial_lives_def : initial_lives = 42
axiom extra_lives_def : extra_lives = 24
axiom final_lives_def : final_lives = 41

-- Mathematically equivalent proof statement
theorem emily_lives_lost : initial_lives - L + extra_lives = final_lives → L = 25 := by
  sorry

end NUMINAMATH_GPT_emily_lives_lost_l1853_185322


namespace NUMINAMATH_GPT_cow_calf_ratio_l1853_185396

theorem cow_calf_ratio (cost_cow cost_calf : ℕ) (h_cow : cost_cow = 880) (h_calf : cost_calf = 110) :
  cost_cow / cost_calf = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_cow_calf_ratio_l1853_185396


namespace NUMINAMATH_GPT_problem_l1853_185356

theorem problem 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 996 / 1005 :=
sorry

end NUMINAMATH_GPT_problem_l1853_185356


namespace NUMINAMATH_GPT_find_a_l1853_185389

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 18 * x^3 + ((86 : ℝ)) * x^2 + 200 * x - 1984

-- Define the condition and statement
theorem find_a (α β γ δ : ℝ) (hαβγδ : α * β * γ * δ = -1984)
  (hαβ : α * β = -32) (hγδ : γ * δ = 62) :
  (∀ a : ℝ, a = 86) :=
  sorry

end NUMINAMATH_GPT_find_a_l1853_185389


namespace NUMINAMATH_GPT_perpendicular_value_of_k_parallel_value_of_k_l1853_185327

variables (a b : ℝ × ℝ) (k : ℝ)

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-3, 1)
def ka_plus_b (k : ℝ) : ℝ × ℝ := (2*k - 3, 3*k + 1)
def a_minus_3b : ℝ × ℝ := (11, 0)

theorem perpendicular_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  a - ka_plus_b k = a_minus_3b → k = (3 / 2) :=
sorry

theorem parallel_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  ∃ k, (ka_plus_b (-1/3)) = (-1/3 * 11, -1/3 * 0) ∧ k = -1 / 3 :=
sorry

end NUMINAMATH_GPT_perpendicular_value_of_k_parallel_value_of_k_l1853_185327


namespace NUMINAMATH_GPT_scallops_per_person_l1853_185366

theorem scallops_per_person 
    (scallops_per_pound : ℕ)
    (cost_per_pound : ℝ)
    (total_cost : ℝ)
    (people : ℕ)
    (total_pounds : ℝ)
    (total_scallops : ℕ)
    (scallops_per_person : ℕ)
    (h1 : scallops_per_pound = 8)
    (h2 : cost_per_pound = 24)
    (h3 : total_cost = 48)
    (h4 : people = 8)
    (h5 : total_pounds = total_cost / cost_per_pound)
    (h6 : total_scallops = scallops_per_pound * total_pounds)
    (h7 : scallops_per_person = total_scallops / people) : 
    scallops_per_person = 2 := 
by {
    sorry
}

end NUMINAMATH_GPT_scallops_per_person_l1853_185366


namespace NUMINAMATH_GPT_find_g_2_l1853_185383

variable (g : ℝ → ℝ)

-- Function satisfying the given conditions
axiom g_functional : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom g_nonzero : ∀ (x : ℝ), g x ≠ 0

-- The proof statement
theorem find_g_2 : g 2 = 1 := by
  sorry

end NUMINAMATH_GPT_find_g_2_l1853_185383


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1853_185305

def l1 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => m * x + (m + 1) * y + 2

def l2 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => (m + 1) * x + (m + 4) * y - 3

def perpendicular_slopes (m : ℝ) : Prop :=
  let slope_l1 := -m / (m + 1)
  let slope_l2 := -(m + 1) / (m + 4)
  slope_l1 * slope_l2 = -1

theorem sufficient_but_not_necessary (m : ℝ) : m = -2 → (∃ k, m = -k ∧ perpendicular_slopes k) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1853_185305


namespace NUMINAMATH_GPT_travel_time_l1853_185318

theorem travel_time (speed distance : ℕ) (h_speed : speed = 100) (h_distance : distance = 500) :
  distance / speed = 5 := by
  sorry

end NUMINAMATH_GPT_travel_time_l1853_185318


namespace NUMINAMATH_GPT_tourist_groupings_l1853_185321

-- Assume a function to count valid groupings exists
noncomputable def num_groupings (guides tourists : ℕ) :=
  if tourists < guides * 2 then 0 
  else sorry -- placeholder for the actual combinatorial function

theorem tourist_groupings : num_groupings 4 8 = 105 := 
by
  -- The proof is omitted intentionally 
  sorry

end NUMINAMATH_GPT_tourist_groupings_l1853_185321


namespace NUMINAMATH_GPT_polynomial_coefficient_sum_equality_l1853_185326

theorem polynomial_coefficient_sum_equality :
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
    (∀ x : ℝ, (2 * x + 1)^4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
    (a₀ - a₁ + a₂ - a₃ + a₄ = 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_polynomial_coefficient_sum_equality_l1853_185326


namespace NUMINAMATH_GPT_rate_per_kg_for_grapes_l1853_185390

theorem rate_per_kg_for_grapes (G : ℝ) (h : 9 * G + 9 * 55 = 1125) : G = 70 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_rate_per_kg_for_grapes_l1853_185390


namespace NUMINAMATH_GPT_equivalent_expression_l1853_185359

theorem equivalent_expression :
  (5+3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * (5^16 + 3^16) * 
  (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := 
  sorry

end NUMINAMATH_GPT_equivalent_expression_l1853_185359


namespace NUMINAMATH_GPT_max_value_expr_l1853_185348

theorem max_value_expr (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
sorry

end NUMINAMATH_GPT_max_value_expr_l1853_185348


namespace NUMINAMATH_GPT_find_expression_value_find_m_value_find_roots_and_theta_l1853_185362

-- Define the conditions
variable (θ : ℝ) (m : ℝ)
variable (h1 : θ > 0) (h2 : θ < 2 * Real.pi)
variable (h3 : ∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0) → (x = Real.sin θ ∨ x = Real.cos θ))

-- Theorem 1: Find the value of a given expression
theorem find_expression_value :
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + Real.cos θ / (1 - Real.tan θ) = (Real.sqrt 3 + 1) / 2 :=
  sorry

-- Theorem 2: Find the value of m
theorem find_m_value :
  m = Real.sqrt 3 / 2 :=
  sorry

-- Theorem 3: Find the roots of the equation and the value of θ
theorem find_roots_and_theta :
  (∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + Real.sqrt 3 / 2 = 0) → (x = Real.sqrt 3 / 2 ∨ x = 1 / 2)) ∧
  (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
  sorry

end NUMINAMATH_GPT_find_expression_value_find_m_value_find_roots_and_theta_l1853_185362


namespace NUMINAMATH_GPT_parallelogram_area_l1853_185329

theorem parallelogram_area (θ : ℝ) (a b : ℝ) (hθ : θ = 100) (ha : a = 20) (hb : b = 10):
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  area = 200 * Real.cos 10 := 
by
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1853_185329


namespace NUMINAMATH_GPT_telescope_visual_range_increased_l1853_185311

/-- A certain telescope increases the visual range from 100 kilometers to 150 kilometers. 
    Proof that the visual range is increased by 50% using the telescope.
-/
theorem telescope_visual_range_increased :
  let original_range := 100
  let new_range := 150
  (new_range - original_range) / original_range * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_telescope_visual_range_increased_l1853_185311


namespace NUMINAMATH_GPT_BigDigMiningCopperOutput_l1853_185354

theorem BigDigMiningCopperOutput :
  (∀ (total_output : ℝ) (nickel_percentage : ℝ) (iron_percentage : ℝ) (amount_of_nickel : ℝ),
      nickel_percentage = 0.10 → 
      iron_percentage = 0.60 → 
      amount_of_nickel = 720 →
      total_output = amount_of_nickel / nickel_percentage →
      (1 - nickel_percentage - iron_percentage) * total_output = 2160) :=
sorry

end NUMINAMATH_GPT_BigDigMiningCopperOutput_l1853_185354


namespace NUMINAMATH_GPT_area_of_square_B_l1853_185349

theorem area_of_square_B (c : ℝ) (hA : ∃ sA, sA * sA = 2 * c^2) (hB : ∃ sA, exists sB, sB * sB = 3 * (sA * sA)) : 
∃ sB, sB * sB = 6 * c^2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_B_l1853_185349


namespace NUMINAMATH_GPT_percentage_difference_l1853_185300

theorem percentage_difference (w x y z : ℝ) (h1 : w = 0.6 * x) (h2 : x = 0.6 * y) (h3 : z = 0.54 * y) : 
  ((z - w) / w) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1853_185300


namespace NUMINAMATH_GPT_calculate_F_l1853_185370

def f(a : ℝ) : ℝ := a^2 - 5 * a + 6
def F(a b c : ℝ) : ℝ := b^2 + a * c + 1

theorem calculate_F : F 3 (f 3) (f 5) = 19 :=
by
  sorry

end NUMINAMATH_GPT_calculate_F_l1853_185370


namespace NUMINAMATH_GPT_number_of_rows_with_exactly_7_students_l1853_185371

theorem number_of_rows_with_exactly_7_students 
  (total_students : ℕ) (rows_with_6_students rows_with_7_students : ℕ) 
  (total_students_eq : total_students = 53)
  (seats_condition : total_students = 6 * rows_with_6_students + 7 * rows_with_7_students) 
  (no_seat_unoccupied : rows_with_6_students + rows_with_7_students = rows_with_6_students + rows_with_7_students) :
  rows_with_7_students = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_rows_with_exactly_7_students_l1853_185371


namespace NUMINAMATH_GPT_sequence_period_2016_l1853_185334

theorem sequence_period_2016 : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 1 / (1 - a n)) → 
  a 1 = 1 / 2 → 
  a 2016 = -1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_period_2016_l1853_185334


namespace NUMINAMATH_GPT_sarah_flour_total_l1853_185385

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def whole_wheat_pastry_flour : ℕ := 2

def total_flour : ℕ := rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour

theorem sarah_flour_total : total_flour = 20 := by
  sorry

end NUMINAMATH_GPT_sarah_flour_total_l1853_185385


namespace NUMINAMATH_GPT_Hoelder_l1853_185367

variable (A B p q : ℝ)

theorem Hoelder (hA : 0 < A) (hB : 0 < B) (hp : 0 < p) (hq : 0 < q) (h : 1 / p + 1 / q = 1) : 
  A^(1/p) * B^(1/q) ≤ A / p + B / q := 
sorry

end NUMINAMATH_GPT_Hoelder_l1853_185367


namespace NUMINAMATH_GPT_team_members_count_l1853_185341

theorem team_members_count (x : ℕ) (h1 : 3 * x + 2 * x = 33 ∨ 4 * x + 2 * x = 33) : x = 6 := by
  sorry

end NUMINAMATH_GPT_team_members_count_l1853_185341


namespace NUMINAMATH_GPT_pipe_A_fills_tank_in_16_hours_l1853_185325

theorem pipe_A_fills_tank_in_16_hours
  (A : ℝ)
  (h1 : ∀ t : ℝ, t = 12.000000000000002 → (1/A + 1/24) * t = 5/4) :
  A = 16 :=
by sorry

end NUMINAMATH_GPT_pipe_A_fills_tank_in_16_hours_l1853_185325


namespace NUMINAMATH_GPT_mass_percentage_of_C_in_benzene_l1853_185315

theorem mass_percentage_of_C_in_benzene :
  let C_molar_mass := 12.01 -- g/mol
  let H_molar_mass := 1.008 -- g/mol
  let benzene_C_atoms := 6
  let benzene_H_atoms := 6
  let C_total_mass := benzene_C_atoms * C_molar_mass
  let H_total_mass := benzene_H_atoms * H_molar_mass
  let benzene_total_mass := C_total_mass + H_total_mass
  let mass_percentage_C := (C_total_mass / benzene_total_mass) * 100
  (mass_percentage_C = 92.26) :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_C_in_benzene_l1853_185315


namespace NUMINAMATH_GPT_triangle_PQR_area_l1853_185382

/-

Define the points P, Q, and R.
Define a function to calculate the area of a triangle given three points.
Then write a theorem to state that the area of triangle PQR is 12.

-/

structure Point where
  x : ℕ
  y : ℕ

def P : Point := ⟨2, 6⟩
def Q : Point := ⟨2, 2⟩
def R : Point := ⟨8, 5⟩

def area (A B C : Point) : ℚ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

theorem triangle_PQR_area : area P Q R = 12 := by
  /- 
    The proof should involve calculating the area using the given points.
   -/
  sorry

end NUMINAMATH_GPT_triangle_PQR_area_l1853_185382


namespace NUMINAMATH_GPT_abs_eq_5_iff_l1853_185352

theorem abs_eq_5_iff (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_5_iff_l1853_185352


namespace NUMINAMATH_GPT_quadratic_root_one_is_minus_one_l1853_185392

theorem quadratic_root_one_is_minus_one (m : ℝ) (h : ∃ x : ℝ, x = -1 ∧ m * x^2 + x - m^2 + 1 = 0) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_one_is_minus_one_l1853_185392


namespace NUMINAMATH_GPT_binomial_7_2_eq_21_l1853_185378

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_binomial_7_2_eq_21_l1853_185378


namespace NUMINAMATH_GPT_simplify_expression_l1853_185395

theorem simplify_expression (x : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 9 = 45 * x + 27 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1853_185395


namespace NUMINAMATH_GPT_trig_identity_condition_l1853_185307

open Real

theorem trig_identity_condition (a : Real) (h : ∃ x ≥ 0, (tan a = -1 ∧ cos a ≠ 0)) :
  (sin a / sqrt (1 - sin a ^ 2) + sqrt (1 - cos a ^ 2) / cos a) = 0 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_condition_l1853_185307


namespace NUMINAMATH_GPT_population_net_increase_period_l1853_185394

def period_in_hours (birth_rate : ℕ) (death_rate : ℕ) (net_increase : ℕ) : ℕ :=
  let net_rate_per_second := (birth_rate / 2) - (death_rate / 2)
  let period_in_seconds := net_increase / net_rate_per_second
  period_in_seconds / 3600

theorem population_net_increase_period :
  period_in_hours 10 2 345600 = 24 :=
by
  unfold period_in_hours
  sorry

end NUMINAMATH_GPT_population_net_increase_period_l1853_185394


namespace NUMINAMATH_GPT_find_d_l1853_185398

-- Define the proportional condition
def in_proportion (a b c d : ℕ) : Prop := a * d = b * c

-- Given values as parameters
variables {a b c d : ℕ}

-- Theorem to be proven
theorem find_d (h : in_proportion a b c d) (ha : a = 1) (hb : b = 2) (hc : c = 3) : d = 6 :=
sorry

end NUMINAMATH_GPT_find_d_l1853_185398


namespace NUMINAMATH_GPT_activity_participants_l1853_185355

variable (A B C D : Prop)

theorem activity_participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) : B ∧ C ∧ ¬A ∧ ¬D :=
by
  sorry

end NUMINAMATH_GPT_activity_participants_l1853_185355


namespace NUMINAMATH_GPT_unique_solution_k_l1853_185379

theorem unique_solution_k (k : ℝ) :
  (∀ x : ℝ, (x + 3) / (k * x + 2) = x) ↔ (k = -1 / 12) :=
  sorry

end NUMINAMATH_GPT_unique_solution_k_l1853_185379


namespace NUMINAMATH_GPT_sqrt_12_minus_sqrt_27_l1853_185361

theorem sqrt_12_minus_sqrt_27 :
  (Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_sqrt_12_minus_sqrt_27_l1853_185361


namespace NUMINAMATH_GPT_geometric_progression_value_l1853_185304

variable (a : ℕ → ℕ)
variable (r : ℕ)
variable (h_geo : ∀ n, a (n + 1) = a n * r)

theorem geometric_progression_value (h2 : a 2 = 2) (h6 : a 6 = 162) : a 10 = 13122 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_value_l1853_185304


namespace NUMINAMATH_GPT_pyramid_base_side_length_l1853_185346

theorem pyramid_base_side_length (area : ℕ) (slant_height : ℕ) (s : ℕ) 
  (h1 : area = 100) 
  (h2 : slant_height = 20) 
  (h3 : area = (1 / 2) * s * slant_height) :
  s = 10 := 
by 
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l1853_185346


namespace NUMINAMATH_GPT_cos_8_identity_l1853_185397

theorem cos_8_identity (m : ℝ) (h : Real.sin 74 = m) : 
  Real.cos 8 = Real.sqrt ((1 + m) / 2) :=
sorry

end NUMINAMATH_GPT_cos_8_identity_l1853_185397


namespace NUMINAMATH_GPT_trigonometric_relationship_l1853_185323

noncomputable def a : ℝ := Real.sin (46 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (46 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (46 * Real.pi / 180)

theorem trigonometric_relationship : c > a ∧ a > b :=
by
  -- This is the statement part; the proof will be handled here
  sorry

end NUMINAMATH_GPT_trigonometric_relationship_l1853_185323


namespace NUMINAMATH_GPT_counterpositive_prop_l1853_185344

theorem counterpositive_prop (a b c : ℝ) (h : a^2 + b^2 + c^2 < 3) : a + b + c ≠ 3 := 
sorry

end NUMINAMATH_GPT_counterpositive_prop_l1853_185344


namespace NUMINAMATH_GPT_vector_CD_l1853_185303

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b c : V)
variable (h1 : B - A = a)
variable (h2 : B - C = b)
variable (h3 : D - A = c)

theorem vector_CD :
  D - C = -a + b + c :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_vector_CD_l1853_185303


namespace NUMINAMATH_GPT_initial_average_correct_l1853_185319

theorem initial_average_correct (A : ℕ) 
  (num_students : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (wrong_avg : ℕ) (correct_avg : ℕ) 
  (h1 : num_students = 30)
  (h2 : wrong_mark = 70)
  (h3 : correct_mark = 10)
  (h4 : correct_avg = 98)
  (h5 : num_students * correct_avg = (num_students * A) - (wrong_mark - correct_mark)) :
  A = 100 := 
sorry

end NUMINAMATH_GPT_initial_average_correct_l1853_185319


namespace NUMINAMATH_GPT_larger_number_is_391_l1853_185328

-- Define the H.C.F and factors
def HCF := 23
def factor1 := 13
def factor2 := 17
def LCM := HCF * factor1 * factor2

-- Define the two numbers based on the factors
def number1 := HCF * factor1
def number2 := HCF * factor2

-- Theorem statement
theorem larger_number_is_391 : max number1 number2 = 391 := 
by
  sorry

end NUMINAMATH_GPT_larger_number_is_391_l1853_185328


namespace NUMINAMATH_GPT_staff_discount_l1853_185377

theorem staff_discount (d : ℝ) (S : ℝ) (h1 : d > 0)
    (h2 : 0.455 * d = (1 - S / 100) * (0.65 * d)) : S = 30 := by
    sorry

end NUMINAMATH_GPT_staff_discount_l1853_185377


namespace NUMINAMATH_GPT_fraction_comparison_l1853_185301

theorem fraction_comparison : (9 / 16) > (5 / 9) :=
by {
  sorry -- the detailed proof is not required for this task
}

end NUMINAMATH_GPT_fraction_comparison_l1853_185301


namespace NUMINAMATH_GPT_find_number_of_small_gardens_l1853_185342

-- Define the conditions
def seeds_total : Nat := 52
def seeds_big_garden : Nat := 28
def seeds_per_small_garden : Nat := 4

-- Define the target value
def num_small_gardens : Nat := 6

-- The statement of the proof problem
theorem find_number_of_small_gardens 
  (H1 : seeds_total = 52) 
  (H2 : seeds_big_garden = 28) 
  (H3 : seeds_per_small_garden = 4) 
  : seeds_total - seeds_big_garden = 24 ∧ (seeds_total - seeds_big_garden) / seeds_per_small_garden = num_small_gardens := 
sorry

end NUMINAMATH_GPT_find_number_of_small_gardens_l1853_185342


namespace NUMINAMATH_GPT_smallest_mul_seven_perfect_square_l1853_185384

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the problem statement
theorem smallest_mul_seven_perfect_square :
  ∀ x : ℕ, x > 0 → (is_perfect_square (7 * x) ↔ x = 7) := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_mul_seven_perfect_square_l1853_185384
