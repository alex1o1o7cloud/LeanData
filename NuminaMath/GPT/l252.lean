import Mathlib

namespace NUMINAMATH_GPT_Brenda_bakes_20_cakes_a_day_l252_25202

-- Define the conditions
variables (x : ℕ)

-- Other necessary definitions
def cakes_baked_in_9_days (x : ℕ) : ℕ := 9 * x
def cakes_after_selling_half (total_cakes : ℕ) : ℕ := total_cakes.div2

-- Given condition that Brenda has 90 cakes after selling half
def final_cakes_after_selling : ℕ := 90

-- Mathematical statement we want to prove
theorem Brenda_bakes_20_cakes_a_day (x : ℕ) (h : cakes_after_selling_half (cakes_baked_in_9_days x) = final_cakes_after_selling) : x = 20 :=
by sorry

end NUMINAMATH_GPT_Brenda_bakes_20_cakes_a_day_l252_25202


namespace NUMINAMATH_GPT_slope_of_tangent_line_at_x_2_l252_25299

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3*x

theorem slope_of_tangent_line_at_x_2 : (deriv curve 2) = 7 := by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_line_at_x_2_l252_25299


namespace NUMINAMATH_GPT_total_net_gain_computation_l252_25201

noncomputable def house1_initial_value : ℝ := 15000
noncomputable def house2_initial_value : ℝ := 20000

noncomputable def house1_selling_price : ℝ := 1.15 * house1_initial_value
noncomputable def house2_selling_price : ℝ := 1.2 * house2_initial_value

noncomputable def house1_buy_back_price : ℝ := 0.85 * house1_selling_price
noncomputable def house2_buy_back_price : ℝ := 0.8 * house2_selling_price

noncomputable def house1_profit : ℝ := house1_selling_price - house1_buy_back_price
noncomputable def house2_profit : ℝ := house2_selling_price - house2_buy_back_price

noncomputable def total_net_gain : ℝ := house1_profit + house2_profit

theorem total_net_gain_computation : total_net_gain = 7387.5 :=
by
  sorry

end NUMINAMATH_GPT_total_net_gain_computation_l252_25201


namespace NUMINAMATH_GPT_mul_example_l252_25276

theorem mul_example : (3.6 * 0.5 = 1.8) := by
  sorry

end NUMINAMATH_GPT_mul_example_l252_25276


namespace NUMINAMATH_GPT_denis_neighbors_l252_25225

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end NUMINAMATH_GPT_denis_neighbors_l252_25225


namespace NUMINAMATH_GPT_slower_train_speed_l252_25244

theorem slower_train_speed (faster_speed : ℝ) (time_passed : ℝ) (train_length : ℝ) (slower_speed: ℝ) :
  faster_speed = 50 ∧ time_passed = 15 ∧ train_length = 75 →
  slower_speed = 32 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_slower_train_speed_l252_25244


namespace NUMINAMATH_GPT_driving_distance_l252_25297

theorem driving_distance:
  ∀ a b: ℕ, (a + b = 500 ∧ a ≥ 150 ∧ b ≥ 150) → 
  (⌊Real.sqrt (a^2 + b^2)⌋ = 380) :=
by
  intro a b
  intro h
  sorry

end NUMINAMATH_GPT_driving_distance_l252_25297


namespace NUMINAMATH_GPT_calc1_calc2_l252_25208

variable (a b : ℝ) 

theorem calc1 : (-b)^2 * (-b)^3 * (-b)^5 = b^10 :=
by sorry

theorem calc2 : (2 * a * b^2)^3 = 8 * a^3 * b^6 :=
by sorry

end NUMINAMATH_GPT_calc1_calc2_l252_25208


namespace NUMINAMATH_GPT_bread_consumption_snacks_per_day_l252_25285

theorem bread_consumption_snacks_per_day (members : ℕ) (breakfast_slices_per_member : ℕ) (slices_per_loaf : ℕ) (loaves : ℕ) (days : ℕ) (total_slices_breakfast : ℕ) (total_slices_all : ℕ) (snack_slices_per_member_per_day : ℕ) :
  members = 4 →
  breakfast_slices_per_member = 3 →
  slices_per_loaf = 12 →
  loaves = 5 →
  days = 3 →
  total_slices_breakfast = members * breakfast_slices_per_member * days →
  total_slices_all = slices_per_loaf * loaves →
  snack_slices_per_member_per_day = ((total_slices_all - total_slices_breakfast) / members / days) →
  snack_slices_per_member_per_day = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- We can insert the proof outline here based on the calculations from the solution steps
  sorry

end NUMINAMATH_GPT_bread_consumption_snacks_per_day_l252_25285


namespace NUMINAMATH_GPT_problem_l252_25269

theorem problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 7 - x = k^2) : x = 3 ∨ x = 6 ∨ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_l252_25269


namespace NUMINAMATH_GPT_intersection_A_B_l252_25217

def A := {x : ℝ | x < -1 ∨ x > 1}
def B := {x : ℝ | Real.log x / Real.log 2 > 0}

theorem intersection_A_B:
  A ∩ B = {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l252_25217


namespace NUMINAMATH_GPT_fraction_is_perfect_square_l252_25263

theorem fraction_is_perfect_square (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end NUMINAMATH_GPT_fraction_is_perfect_square_l252_25263


namespace NUMINAMATH_GPT_sum_of_largest_100_l252_25239

theorem sum_of_largest_100 (a : Fin 123 → ℝ) (h1 : (Finset.univ.sum a) = 3813) 
  (h2 : ∀ i j : Fin 123, i ≤ j → a i ≤ a j) : 
  ∃ s : Finset (Fin 123), s.card = 100 ∧ (s.sum a) ≥ 3100 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_largest_100_l252_25239


namespace NUMINAMATH_GPT_sticks_difference_l252_25257

-- Definitions of the conditions
def d := 14  -- number of sticks Dave picked up
def a := 9   -- number of sticks Amy picked up
def total := 50  -- initial total number of sticks in the yard

-- The proof problem statement
theorem sticks_difference : (d + a) - (total - (d + a)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sticks_difference_l252_25257


namespace NUMINAMATH_GPT_smallest_sum_B_c_l252_25289

theorem smallest_sum_B_c (B : ℕ) (c : ℕ) (hB : B < 5) (hc : c > 6) :
  31 * B = 4 * c + 4 → (B + c) = 34 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_B_c_l252_25289


namespace NUMINAMATH_GPT_part2_inequality_l252_25259

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- The main theorem we want to prove
theorem part2_inequality (a b c : ℝ) (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  |a + 2 * b + 3 * c| ≤ 6 :=
by {
-- Proof goes here
sorry
}

end NUMINAMATH_GPT_part2_inequality_l252_25259


namespace NUMINAMATH_GPT_karl_total_miles_l252_25288

def car_mileage_per_gallon : ℕ := 30
def full_tank_gallons : ℕ := 14
def initial_drive_miles : ℕ := 300
def gas_bought_gallons : ℕ := 10
def final_tank_fraction : ℚ := 1 / 3

theorem karl_total_miles (initial_fuel : ℕ) :
  initial_fuel = full_tank_gallons →
  (initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons) = initial_fuel - (initial_fuel * final_tank_fraction) / car_mileage_per_gallon + (580 - initial_drive_miles) / car_mileage_per_gallon →
  initial_drive_miles + (initial_fuel - initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons - initial_fuel * final_tank_fraction / car_mileage_per_gallon) * car_mileage_per_gallon = 580 := 
sorry

end NUMINAMATH_GPT_karl_total_miles_l252_25288


namespace NUMINAMATH_GPT_no_valid_n_for_conditions_l252_25212

theorem no_valid_n_for_conditions :
  ∀ (n : ℕ), (100 ≤ n / 5 ∧ n / 5 ≤ 999) ∧ (100 ≤ 5 * n ∧ 5 * n ≤ 999) → false :=
by
  sorry

end NUMINAMATH_GPT_no_valid_n_for_conditions_l252_25212


namespace NUMINAMATH_GPT_train_length_is_correct_l252_25254

-- Definitions
def speed_kmh := 48.0 -- in km/hr
def time_sec := 9.0 -- in seconds

-- Conversion function
def convert_speed (s_kmh : Float) : Float :=
  s_kmh * 1000 / 3600

-- Function to calculate length of train
def length_of_train (speed_kmh : Float) (time_sec : Float) : Float :=
  let speed_ms := convert_speed speed_kmh
  speed_ms * time_sec

-- Proof problem: Given the speed of the train and the time it takes to cross a pole, prove the length of the train
theorem train_length_is_correct : length_of_train speed_kmh time_sec = 119.97 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l252_25254


namespace NUMINAMATH_GPT_angela_deliveries_l252_25222

theorem angela_deliveries
  (n_meals : ℕ)
  (h_meals : n_meals = 3)
  (n_packages : ℕ)
  (h_packages : n_packages = 8 * n_meals) :
  n_meals + n_packages = 27 := by
  sorry

end NUMINAMATH_GPT_angela_deliveries_l252_25222


namespace NUMINAMATH_GPT_alberto_bjorn_distance_difference_l252_25236

-- Definitions based on given conditions
def alberto_speed : ℕ := 12  -- miles per hour
def bjorn_speed : ℕ := 10    -- miles per hour
def total_time : ℕ := 6      -- hours
def bjorn_rest_time : ℕ := 1 -- hours

def alberto_distance : ℕ := alberto_speed * total_time
def bjorn_distance : ℕ := bjorn_speed * (total_time - bjorn_rest_time)

-- The statement to prove
theorem alberto_bjorn_distance_difference :
  (alberto_distance - bjorn_distance) = 22 :=
by
  sorry

end NUMINAMATH_GPT_alberto_bjorn_distance_difference_l252_25236


namespace NUMINAMATH_GPT_sandy_bought_6_books_l252_25277

variable (initialBooks soldBooks boughtBooks remainingBooks : ℕ)

def half (n : ℕ) : ℕ := n / 2

theorem sandy_bought_6_books :
  initialBooks = 14 →
  soldBooks = half initialBooks →
  remainingBooks = initialBooks - soldBooks →
  remainingBooks + boughtBooks = 13 →
  boughtBooks = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sandy_bought_6_books_l252_25277


namespace NUMINAMATH_GPT_range_of_a_l252_25224

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a*x + 2*a > 0) : 0 < a ∧ a < 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l252_25224


namespace NUMINAMATH_GPT_total_lunch_cost_l252_25287

theorem total_lunch_cost
  (children chaperones herself additional_lunches cost_per_lunch : ℕ)
  (h1 : children = 35)
  (h2 : chaperones = 5)
  (h3 : herself = 1)
  (h4 : additional_lunches = 3)
  (h5 : cost_per_lunch = 7) :
  (children + chaperones + herself + additional_lunches) * cost_per_lunch = 308 :=
by
  sorry

end NUMINAMATH_GPT_total_lunch_cost_l252_25287


namespace NUMINAMATH_GPT_inequality_proof_l252_25237

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l252_25237


namespace NUMINAMATH_GPT_gcd_of_g_and_y_l252_25292

-- Define the function g(y)
def g (y : ℕ) := (3 * y + 4) * (8 * y + 3) * (14 * y + 9) * (y + 14)

-- Define that y is a multiple of 45678
def isMultipleOf (y divisor : ℕ) : Prop := ∃ k, y = k * divisor

-- Define the proof problem
theorem gcd_of_g_and_y (y : ℕ) (h : isMultipleOf y 45678) : Nat.gcd (g y) y = 1512 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_g_and_y_l252_25292


namespace NUMINAMATH_GPT_production_company_keeps_60_percent_l252_25210

noncomputable def openingWeekendRevenue : ℝ := 120
noncomputable def productionCost : ℝ := 60
noncomputable def profit : ℝ := 192
noncomputable def totalRevenue : ℝ := 3.5 * openingWeekendRevenue
noncomputable def amountKept : ℝ := profit + productionCost
noncomputable def percentageKept : ℝ := (amountKept / totalRevenue) * 100

theorem production_company_keeps_60_percent :
  percentageKept = 60 :=
by
  sorry

end NUMINAMATH_GPT_production_company_keeps_60_percent_l252_25210


namespace NUMINAMATH_GPT_average_age_of_choir_l252_25238

theorem average_age_of_choir 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (total_people : ℕ) (total_people_eq : total_people = num_females + num_males) :
  num_females = 12 → avg_age_females = 28 → num_males = 18 → avg_age_males = 38 → total_people = 30 →
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 34 := by
  intros
  sorry

end NUMINAMATH_GPT_average_age_of_choir_l252_25238


namespace NUMINAMATH_GPT_value_of_2x_plus_3y_l252_25261

theorem value_of_2x_plus_3y {x y : ℝ} (h1 : 2 * x - 1 = 5) (h2 : 3 * y + 2 = 17) : 2 * x + 3 * y = 21 :=
by
  sorry

end NUMINAMATH_GPT_value_of_2x_plus_3y_l252_25261


namespace NUMINAMATH_GPT_capacity_of_other_bottle_l252_25245

theorem capacity_of_other_bottle 
  (total_milk : ℕ) (capacity_bottle_one : ℕ) (fraction_filled_other_bottle : ℚ)
  (equal_fraction : ℚ) (other_bottle_milk : ℚ) (capacity_other_bottle : ℚ) : 
  total_milk = 8 ∧ capacity_bottle_one = 4 ∧ other_bottle_milk = 16/3 ∧ 
  (equal_fraction * capacity_bottle_one + equal_fraction * capacity_other_bottle = total_milk) ∧ 
  (fraction_filled_other_bottle = 5.333333333333333) → capacity_other_bottle = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_capacity_of_other_bottle_l252_25245


namespace NUMINAMATH_GPT_factor_expression_l252_25282

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 2 * x * (x + 3) + (x + 3) = (x + 1)^2 * (x + 3) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l252_25282


namespace NUMINAMATH_GPT_probability_of_continuous_stripe_loop_l252_25234

-- Definitions corresponding to identified conditions:
def cube_faces : ℕ := 6

def diagonal_orientations_per_face : ℕ := 2

def total_stripe_combinations (faces : ℕ) (orientations : ℕ) : ℕ :=
  orientations ^ faces

def satisfying_stripe_combinations : ℕ := 2

-- Proof statement:
theorem probability_of_continuous_stripe_loop :
  (satisfying_stripe_combinations : ℚ) / (total_stripe_combinations cube_faces diagonal_orientations_per_face : ℚ) = 1 / 32 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_continuous_stripe_loop_l252_25234


namespace NUMINAMATH_GPT_handshake_count_l252_25229

-- Defining the conditions
def number_of_companies : ℕ := 5
def representatives_per_company : ℕ := 5
def total_participants : ℕ := number_of_companies * representatives_per_company

-- Defining the number of handshakes each person makes
def handshakes_per_person : ℕ := total_participants - 1 - (representatives_per_company - 1)

-- Defining the total number of handshakes
def total_handshakes : ℕ := (total_participants * handshakes_per_person) / 2

theorem handshake_count :
  total_handshakes = 250 :=
by
  sorry

end NUMINAMATH_GPT_handshake_count_l252_25229


namespace NUMINAMATH_GPT_exist_midpoints_l252_25200
open Classical

noncomputable def h (a b c : ℝ) := (a + b + c) / 3

theorem exist_midpoints (a b c : ℝ) (X Y Z : ℝ) (AX BY CZ : ℝ) :
  (0 < X) ∧ (X < a) ∧
  (0 < Y) ∧ (Y < b) ∧
  (0 < Z) ∧ (Z < c) ∧
  (X + (a - X) = (h a b c)) ∧
  (Y + (b - Y) = (h a b c)) ∧
  (Z + (c - Z) = (h a b c)) ∧
  (AX * BY * CZ = (a - X) * (b - Y) * (c - Z))
  → ∃ (X Y Z : ℝ), X = (a / 2) ∧ Y = (b / 2) ∧ Z = (c / 2) :=
by
  sorry

end NUMINAMATH_GPT_exist_midpoints_l252_25200


namespace NUMINAMATH_GPT_mac_total_loss_l252_25246

-- Definitions based on conditions in part a)
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_quarter : ℝ := 0.25
def dimes_per_quarter : ℕ := 3
def nickels_per_quarter : ℕ := 7
def quarters_traded_dimes : ℕ := 20
def quarters_traded_nickels : ℕ := 20

-- Lean statement for the proof problem
theorem mac_total_loss : (dimes_per_quarter * value_dime * quarters_traded_dimes 
                          + nickels_per_quarter * value_nickel * quarters_traded_nickels
                          - 40 * value_quarter) = 3.00 := 
sorry

end NUMINAMATH_GPT_mac_total_loss_l252_25246


namespace NUMINAMATH_GPT_tan_alpha_eq_7_over_5_l252_25265

theorem tan_alpha_eq_7_over_5
  (α : ℝ)
  (h : Real.tan (α - π / 4) = 1 / 6) :
  Real.tan α = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_7_over_5_l252_25265


namespace NUMINAMATH_GPT_solve_for_x_l252_25274

theorem solve_for_x (x : ℝ) : 64 = 4 * (16:ℝ)^(x - 2) → x = 3 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l252_25274


namespace NUMINAMATH_GPT_number_of_paths_l252_25271

-- Definition of vertices
inductive Vertex
| A | B | C | D | E | F | G

-- Edges based on the description
def edges : List (Vertex × Vertex) := [
  (Vertex.A, Vertex.G), (Vertex.G, Vertex.C), (Vertex.G, Vertex.D), (Vertex.C, Vertex.B),
  (Vertex.D, Vertex.C), (Vertex.D, Vertex.F), (Vertex.D, Vertex.E), (Vertex.E, Vertex.F),
  (Vertex.F, Vertex.B), (Vertex.C, Vertex.F), (Vertex.A, Vertex.C), (Vertex.A, Vertex.D)
]

-- Function to count paths from A to B without revisiting any vertex
def countPaths (start : Vertex) (goal : Vertex) (adj : List (Vertex × Vertex)) : Nat :=
sorry

-- The theorem statement
theorem number_of_paths : countPaths Vertex.A Vertex.B edges = 10 :=
sorry

end NUMINAMATH_GPT_number_of_paths_l252_25271


namespace NUMINAMATH_GPT_absolute_value_inequality_range_of_xyz_l252_25280

-- Question 1 restated
theorem absolute_value_inequality (x : ℝ) :
  (|x + 2| + |x + 3| ≤ 2) ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

-- Question 2 restated
theorem range_of_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  -1/2 ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ 1 :=
sorry

end NUMINAMATH_GPT_absolute_value_inequality_range_of_xyz_l252_25280


namespace NUMINAMATH_GPT_tv_show_duration_l252_25216

theorem tv_show_duration (total_air_time : ℝ) (num_commercials : ℕ) (commercial_duration_min : ℝ) :
  total_air_time = 1.5 ∧ num_commercials = 3 ∧ commercial_duration_min = 10 →
  (total_air_time - (num_commercials * commercial_duration_min / 60)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tv_show_duration_l252_25216


namespace NUMINAMATH_GPT_negation_of_exists_x_lt_0_l252_25295

theorem negation_of_exists_x_lt_0 :
  (¬ ∃ x : ℝ, x + |x| < 0) ↔ (∀ x : ℝ, x + |x| ≥ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_exists_x_lt_0_l252_25295


namespace NUMINAMATH_GPT_geometric_sequence_tenth_term_l252_25266

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (12 / 3) / 4
  let nth_term (n : ℕ) := a * r^(n-1)
  nth_term 10 = 4 :=
  by sorry

end NUMINAMATH_GPT_geometric_sequence_tenth_term_l252_25266


namespace NUMINAMATH_GPT_solution_set_of_inequality_l252_25296

theorem solution_set_of_inequality :
  {x : ℝ | x * (x - 1) * (x - 2) > 0} = {x | (0 < x ∧ x < 1) ∨ x > 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l252_25296


namespace NUMINAMATH_GPT_rogers_coaches_l252_25267

-- Define the structure for the problem conditions
structure snacks_problem :=
  (team_members : ℕ)
  (helpers : ℕ)
  (packs_purchased : ℕ)
  (pouches_per_pack : ℕ)

-- Create an instance of the problem with given conditions
def rogers_problem : snacks_problem :=
  { team_members := 13,
    helpers := 2,
    packs_purchased := 3,
    pouches_per_pack := 6 }

-- Define the theorem to state that given the conditions, the number of coaches is 3
theorem rogers_coaches (p : snacks_problem) : p.packs_purchased * p.pouches_per_pack - p.team_members - p.helpers = 3 :=
by
  sorry

end NUMINAMATH_GPT_rogers_coaches_l252_25267


namespace NUMINAMATH_GPT_compare_neg_fractions_l252_25293

theorem compare_neg_fractions : (-5/4 : ℚ) > (-4/3 : ℚ) := 
sorry

end NUMINAMATH_GPT_compare_neg_fractions_l252_25293


namespace NUMINAMATH_GPT_grandma_mushrooms_l252_25235

theorem grandma_mushrooms (M : ℕ) (h₁ : ∀ t : ℕ, t = 2 * M)
                         (h₂ : ∀ p : ℕ, p = 4 * t)
                         (h₃ : ∀ b : ℕ, b = 4 * p)
                         (h₄ : ∀ r : ℕ, r = b / 3)
                         (h₅ : r = 32) :
  M = 3 :=
by
  -- We are expected to fill the steps here to provide the proof if required
  sorry

end NUMINAMATH_GPT_grandma_mushrooms_l252_25235


namespace NUMINAMATH_GPT_goats_at_farm_l252_25252

theorem goats_at_farm (G C D P : ℕ) 
  (h1: C = 2 * G)
  (h2: D = (G + C) / 2)
  (h3: P = D / 3)
  (h4: G = P + 33) :
  G = 66 :=
by
  sorry

end NUMINAMATH_GPT_goats_at_farm_l252_25252


namespace NUMINAMATH_GPT_balls_left_correct_l252_25284

def initial_balls : ℕ := 10
def balls_removed : ℕ := 3
def balls_left : ℕ := initial_balls - balls_removed

theorem balls_left_correct : balls_left = 7 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_balls_left_correct_l252_25284


namespace NUMINAMATH_GPT_inequality_necessary_not_sufficient_l252_25214

theorem inequality_necessary_not_sufficient (m : ℝ) : 
  (-3 < m ∧ m < 5) → (5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inequality_necessary_not_sufficient_l252_25214


namespace NUMINAMATH_GPT_faye_gave_away_books_l252_25272

theorem faye_gave_away_books (x : ℕ) (H1 : 34 - x + 48 = 79) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_faye_gave_away_books_l252_25272


namespace NUMINAMATH_GPT_algebraic_expression_l252_25250

-- Given conditions in the problem.
variables (x y : ℝ)

-- The statement to be proved: If 2x - 3y = 1, then 6y - 4x + 8 = 6.
theorem algebraic_expression (h : 2 * x - 3 * y = 1) : 6 * y - 4 * x + 8 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_l252_25250


namespace NUMINAMATH_GPT_men_took_dip_l252_25290

theorem men_took_dip 
  (tank_length : ℝ) (tank_breadth : ℝ) (water_rise_cm : ℝ) (man_displacement : ℝ)
  (H1 : tank_length = 40) (H2 : tank_breadth = 20) (H3 : water_rise_cm = 25) (H4 : man_displacement = 4) :
  let water_rise_m := water_rise_cm / 100
  let total_volume_displaced := tank_length * tank_breadth * water_rise_m
  let number_of_men := total_volume_displaced / man_displacement
  number_of_men = 50 :=
by
  sorry

end NUMINAMATH_GPT_men_took_dip_l252_25290


namespace NUMINAMATH_GPT_find_a_b_solution_set_l252_25232

-- Given function
def f (x : ℝ) (a b : ℝ) := x^2 - (a + b) * x + 3 * a

-- Part 1: Prove the values of a and b given the solution set of the inequality
theorem find_a_b (a b : ℝ) 
  (h1 : 1^2 - (a + b) * 1 + 3 * 1 = 0)
  (h2 : 3^2 - (a + b) * 3 + 3 * 1 = 0) :
  a = 1 ∧ b = 3 :=
sorry

-- Part 2: Find the solution set of the inequality f(x) > 0 given b = 3
theorem solution_set (a : ℝ)
  (h : b = 3) :
  (a > 3 → (∀ x, f x a 3 > 0 ↔ x < 3 ∨ x > a)) ∧
  (a < 3 → (∀ x, f x a 3 > 0 ↔ x < a ∨ x > 3)) ∧
  (a = 3 → (∀ x, f x a 3 > 0 ↔ x ≠ 3)) :=
sorry

end NUMINAMATH_GPT_find_a_b_solution_set_l252_25232


namespace NUMINAMATH_GPT_num_valid_constants_m_l252_25268

theorem num_valid_constants_m : 
  ∃ (m1 m2 : ℝ), 
  m1 ≠ m2 ∧ 
  (∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∧ 8 = m1 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∧ (1 / 2) = m2 ∨ 2 * c / d = 2)) ∧
  (∀ (m : ℝ), 
    (m = m1 ∨ m = m2) →
    ∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∨ 2 * c / d = 2)) :=
sorry

end NUMINAMATH_GPT_num_valid_constants_m_l252_25268


namespace NUMINAMATH_GPT_circle_equation_l252_25298

theorem circle_equation {a b c : ℝ} (hc : c ≠ 0) :
  ∃ D E F : ℝ, 
    (D = -(a + b)) ∧
    (E = - (c + ab / c)) ∧ 
    (F = ab) ∧
    ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 :=
sorry

end NUMINAMATH_GPT_circle_equation_l252_25298


namespace NUMINAMATH_GPT_modulo_remainder_l252_25256

theorem modulo_remainder : (7^2023) % 17 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_modulo_remainder_l252_25256


namespace NUMINAMATH_GPT_find_function_l252_25220

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 2 = x + y + 1) : 
  ∀ x : ℝ, f x = x + 2 := sorry

end NUMINAMATH_GPT_find_function_l252_25220


namespace NUMINAMATH_GPT_inequality_proof_l252_25218

variables {x y z : ℝ}

theorem inequality_proof 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x) 
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) : 
  4 * x + y ≥ 4 * z :=
sorry

end NUMINAMATH_GPT_inequality_proof_l252_25218


namespace NUMINAMATH_GPT_tailor_trimming_l252_25243

theorem tailor_trimming (x : ℝ) (A B : ℝ)
  (h1 : ∃ (L : ℝ), L = 22) -- Original length of a side of the cloth is 22 feet
  (h2 : 6 = 6) -- Feet trimmed from two opposite edges
  (h3 : ∃ (remaining_area : ℝ), remaining_area = 120) -- 120 square feet of cloth remain after trimming
  (h4 : A = 22 - 2 * 6) -- New length of the side after trimming 6 feet from opposite edges
  (h5 : B = 22 - x) -- New length of the side after trimming x feet from the other two edges
  (h6 : remaining_area = A * B) -- Relationship of the remaining area
: x = 10 :=
by
  sorry

end NUMINAMATH_GPT_tailor_trimming_l252_25243


namespace NUMINAMATH_GPT_find_a_l252_25204

theorem find_a {a b c : ℕ} (h₁ : a + b = c) (h₂ : b + c = 8) (h₃ : c = 4) : a = 0 := by
  sorry

end NUMINAMATH_GPT_find_a_l252_25204


namespace NUMINAMATH_GPT_steve_final_height_l252_25247

-- Define the initial height of Steve in inches.
def initial_height : ℕ := 5 * 12 + 6

-- Define how many inches Steve grew.
def growth : ℕ := 6

-- Define Steve's final height after growing.
def final_height : ℕ := initial_height + growth

-- The final height should be 72 inches.
theorem steve_final_height : final_height = 72 := by
  -- we don't provide the proof here
  sorry

end NUMINAMATH_GPT_steve_final_height_l252_25247


namespace NUMINAMATH_GPT_number_of_employees_l252_25219

-- Definitions
def emily_original_salary : ℕ := 1000000
def emily_new_salary : ℕ := 850000
def employee_original_salary : ℕ := 20000
def employee_new_salary : ℕ := 35000
def salary_difference : ℕ := emily_original_salary - emily_new_salary
def salary_increase_per_employee : ℕ := employee_new_salary - employee_original_salary

-- Theorem: Prove Emily has n employees where n = 10
theorem number_of_employees : salary_difference / salary_increase_per_employee = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_employees_l252_25219


namespace NUMINAMATH_GPT_dot_product_AB_BC_l252_25221

theorem dot_product_AB_BC (AB BC : ℝ) (B : ℝ) 
  (h1 : AB = 3) (h2 : BC = 4) (h3 : B = π/6) :
  (AB * BC * Real.cos (π - B) = -6 * Real.sqrt 3) :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_dot_product_AB_BC_l252_25221


namespace NUMINAMATH_GPT_determine_gx_l252_25213

/-
  Given two polynomials f(x) and h(x), we need to show that g(x) is a certain polynomial
  when f(x) + g(x) = h(x).
-/

def f (x : ℝ) : ℝ := 4 * x^5 + 3 * x^3 + x - 2
def h (x : ℝ) : ℝ := 7 * x^3 - 5 * x + 4
def g (x : ℝ) : ℝ := -4 * x^5 + 4 * x^3 - 4 * x + 6

theorem determine_gx (x : ℝ) : f x + g x = h x :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_determine_gx_l252_25213


namespace NUMINAMATH_GPT_infinitely_many_n_divide_b_pow_n_plus_1_l252_25231

theorem infinitely_many_n_divide_b_pow_n_plus_1 (b : ℕ) (h1 : b > 2) :
  (∃ᶠ n in at_top, n^2 ∣ b^n + 1) ↔ ¬ ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_GPT_infinitely_many_n_divide_b_pow_n_plus_1_l252_25231


namespace NUMINAMATH_GPT_parallel_lines_condition_l252_25249

theorem parallel_lines_condition (k1 k2 b : ℝ) (l1 l2 : ℝ → ℝ) (H1 : ∀ x, l1 x = k1 * x + 1)
  (H2 : ∀ x, l2 x = k2 * x + b) : (∀ x, l1 x = l2 x ↔ k1 = k2 ∧ b = 1) → (k1 = k2) ↔ (∀ x, l1 x ≠ l2 x ∧ l1 x - l2 x = 1 - b) := 
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l252_25249


namespace NUMINAMATH_GPT_calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l252_25223

noncomputable def volume_of_parallelepiped (R : ℝ) : ℝ := R^3 * Real.sqrt 6

noncomputable def diagonal_A_C_prime (R: ℝ) : ℝ := R * Real.sqrt 6

noncomputable def volume_of_rotation (R: ℝ) : ℝ := R^3 * Real.sqrt 12

theorem calculate_volume_and_diagonal (R : ℝ) : 
  volume_of_parallelepiped R = R^3 * Real.sqrt 6 ∧ 
  diagonal_A_C_prime R = R * Real.sqrt 6 :=
by sorry

theorem calculate_volume_and_surface_rotation (R : ℝ) :
  volume_of_rotation R = R^3 * Real.sqrt 12 :=
by sorry

theorem calculate_radius_given_volume (V : ℝ) (h : V = 0.034786) : 
  ∃ R : ℝ, V = volume_of_parallelepiped R :=
by sorry

end NUMINAMATH_GPT_calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l252_25223


namespace NUMINAMATH_GPT_dessert_menu_count_is_192_l252_25241

-- Defining the set of desserts
inductive Dessert
| cake | pie | ice_cream

-- Function to count valid dessert menus (not repeating on consecutive days) with cake on Friday
def countDessertMenus : Nat :=
  -- Let's denote Sunday as day 1 and Saturday as day 7
  let sunday_choices := 3
  let weekday_choices := 2 -- for Monday to Thursday (no repeats consecutive)
  let weekend_choices := 2 -- for Saturday and Sunday after
  sunday_choices * weekday_choices^4 * 1 * weekend_choices^2

-- Theorem stating the number of valid dessert menus for the week
theorem dessert_menu_count_is_192 : countDessertMenus = 192 :=
  by
    -- Actual proof is omitted
    sorry

end NUMINAMATH_GPT_dessert_menu_count_is_192_l252_25241


namespace NUMINAMATH_GPT_production_value_equation_l252_25242

theorem production_value_equation (x : ℝ) :
  (2000000 * (1 + x)^2) - (2000000 * (1 + x)) = 220000 := 
sorry

end NUMINAMATH_GPT_production_value_equation_l252_25242


namespace NUMINAMATH_GPT_matrix_determinant_zero_l252_25226

noncomputable def matrix_example : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]

theorem matrix_determinant_zero : matrix_example.det = 0 := 
by 
  sorry

end NUMINAMATH_GPT_matrix_determinant_zero_l252_25226


namespace NUMINAMATH_GPT_increasing_condition_sufficient_not_necessary_l252_25228

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x

theorem increasing_condition_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0) → (a ≥ 0) ∧ ¬ (a > 0 ↔ (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0)) :=
by
  sorry

end NUMINAMATH_GPT_increasing_condition_sufficient_not_necessary_l252_25228


namespace NUMINAMATH_GPT_initial_dogs_l252_25270

theorem initial_dogs (D : ℕ) (h : D + 5 + 3 = 10) : D = 2 :=
by sorry

end NUMINAMATH_GPT_initial_dogs_l252_25270


namespace NUMINAMATH_GPT_area_of_octagon_l252_25209

theorem area_of_octagon (a b : ℝ) (hsquare : a ^ 2 = 16)
  (hperimeter : 4 * a = 8 * b) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_octagon_l252_25209


namespace NUMINAMATH_GPT_arithmetic_problem_l252_25294

theorem arithmetic_problem : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l252_25294


namespace NUMINAMATH_GPT_lcm_of_48_and_14_is_56_l252_25262

theorem lcm_of_48_and_14_is_56 :
  ∀ n : ℕ, (n = 48 ∧ Nat.gcd n 14 = 12) → Nat.lcm n 14 = 56 :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_lcm_of_48_and_14_is_56_l252_25262


namespace NUMINAMATH_GPT_sum_in_base_b_l252_25207

noncomputable def s_in_base (b : ℕ) := 13 + 15 + 17

theorem sum_in_base_b (b : ℕ) (h : (13 * 15 * 17 : ℕ) = 4652) : s_in_base b = 51 := by
  sorry

end NUMINAMATH_GPT_sum_in_base_b_l252_25207


namespace NUMINAMATH_GPT_intersection_M_N_l252_25206

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l252_25206


namespace NUMINAMATH_GPT_range_of_m_l252_25283

noncomputable def f (x : ℝ) : ℝ := |x - 3| - 2
noncomputable def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem range_of_m :
  (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l252_25283


namespace NUMINAMATH_GPT_original_rectangle_area_l252_25215

-- Define the original rectangle sides, square side, and perimeters of rectangles adjacent to the square
variables {a b x : ℝ}
variable (h1 : a + x = 10)
variable (h2 : b + x = 8)

-- Define the area calculation
def area (a b : ℝ) := a * b

-- The area of the original rectangle should be 80 cm²
theorem original_rectangle_area : area (10 - x) (8 - x) = 80 := by
  sorry

end NUMINAMATH_GPT_original_rectangle_area_l252_25215


namespace NUMINAMATH_GPT_negation_of_proposition_l252_25260

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l252_25260


namespace NUMINAMATH_GPT_algebraic_expression_value_l252_25203

-- Define given condition
def condition (x : ℝ) : Prop := 3 * x^2 - 2 * x - 1 = 2

-- Define the target expression
def target_expression (x : ℝ) : ℝ := -9 * x^2 + 6 * x - 1

-- The theorem statement
theorem algebraic_expression_value (x : ℝ) (h : condition x) : target_expression x = -10 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l252_25203


namespace NUMINAMATH_GPT_faye_coloring_books_l252_25251

theorem faye_coloring_books (x : ℕ) : 34 - x + 48 = 79 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_faye_coloring_books_l252_25251


namespace NUMINAMATH_GPT_simplify_expression_l252_25253

theorem simplify_expression : ( (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l252_25253


namespace NUMINAMATH_GPT_unique_fraction_satisfying_condition_l252_25291

theorem unique_fraction_satisfying_condition : ∃! (x y : ℕ), Nat.gcd x y = 1 ∧ y ≠ 0 ∧ (x + 1) * 5 * y = (y + 1) * 6 * x :=
by
  sorry

end NUMINAMATH_GPT_unique_fraction_satisfying_condition_l252_25291


namespace NUMINAMATH_GPT_water_distribution_scheme_l252_25275

theorem water_distribution_scheme (a b c : ℚ) : 
  a + b + c = 1 ∧ 
  (∀ x : ℂ, ∃ n : ℕ, x^n = 1 → x = 1) ∧
  (∀ (x : ℂ), (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11 + x^12 + x^13 + x^14 + x^15 + x^16 + x^17 + x^18 + x^19 + x^20 + x^21 + x^22 = 0) → false) → 
  a = 0 ∧ b = 0 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_GPT_water_distribution_scheme_l252_25275


namespace NUMINAMATH_GPT_inequality_solution_real_roots_range_l252_25273

noncomputable def f (x : ℝ) : ℝ :=
|2 * x - 4| - |x - 3|

theorem inequality_solution :
  ∀ x, f x ≤ 2 → x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

theorem real_roots_range (k : ℝ) :
  (∃ x, f x = 0) → k ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end NUMINAMATH_GPT_inequality_solution_real_roots_range_l252_25273


namespace NUMINAMATH_GPT_smallest_number_of_tins_needed_l252_25211

variable (A : ℤ) (C : ℚ)

-- Conditions
def wall_area_valid : Prop := 1915 ≤ A ∧ A < 1925
def coverage_per_tin_valid : Prop := 17.5 ≤ C ∧ C < 18.5
def tins_needed_to_cover_wall (A : ℤ) (C : ℚ) : ℚ := A / C
def smallest_tins_needed : ℚ := 111

-- Proof problem statement
theorem smallest_number_of_tins_needed (A : ℤ) (C : ℚ)
    (h1 : wall_area_valid A)
    (h2 : coverage_per_tin_valid C)
    (h3 : 1915 ≤ A)
    (h4 : A < 1925)
    (h5 : 17.5 ≤ C)
    (h6 : C < 18.5) : 
  tins_needed_to_cover_wall A C + 1 ≥ smallest_tins_needed := by
    sorry

end NUMINAMATH_GPT_smallest_number_of_tins_needed_l252_25211


namespace NUMINAMATH_GPT_triangle_sine_inequality_l252_25278

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤
  1 + (1 / 2) * Real.cos ((A - B) / 4) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sine_inequality_l252_25278


namespace NUMINAMATH_GPT_remainder_3_pow_19_mod_10_l252_25255

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_19_mod_10_l252_25255


namespace NUMINAMATH_GPT_polynomial_coeff_properties_l252_25233

theorem polynomial_coeff_properties :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 : ℤ,
  (∀ x : ℤ, (1 - 2 * x)^7 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) ∧
  a0 = 1 ∧
  (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = -1) ∧
  (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3^7)) :=
sorry

end NUMINAMATH_GPT_polynomial_coeff_properties_l252_25233


namespace NUMINAMATH_GPT_selection_at_most_one_l252_25230

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_at_most_one (A B : ℕ) :
  (combination 5 3) - (combination 3 1) = 7 :=
by
  sorry

end NUMINAMATH_GPT_selection_at_most_one_l252_25230


namespace NUMINAMATH_GPT_sequence_a_1000_l252_25286

theorem sequence_a_1000 (a : ℕ → ℕ)
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n, a (n + 1) = 3 * a n - 2 * a (n - 1)) : 
  a 1000 = 2^1000 - 1 := 
sorry

end NUMINAMATH_GPT_sequence_a_1000_l252_25286


namespace NUMINAMATH_GPT_journey_speed_l252_25227

theorem journey_speed
  (v : ℝ) -- Speed during the first four hours
  (total_distance : ℝ) (total_time : ℝ) -- Total distance and time of the journey
  (distance_part1 : ℝ) (time_part1 : ℝ) -- Distance and time for the first part of journey
  (distance_part2 : ℝ) (time_part2 : ℝ) -- Distance and time for the second part of journey
  (speed_part2 : ℝ) : -- Speed during the second part of journey
  total_distance = 24 ∧ total_time = 8 ∧ speed_part2 = 2 ∧ 
  time_part1 = 4 ∧ time_part2 = 4 ∧ 
  distance_part1 = v * time_part1 ∧ distance_part2 = speed_part2 * time_part2 →
  v = 4 := 
by
  sorry

end NUMINAMATH_GPT_journey_speed_l252_25227


namespace NUMINAMATH_GPT_problem1_problem2_l252_25258

-- The first problem
theorem problem1 (x : ℝ) (h : Real.tan x = 3) :
  (2 * Real.sin (Real.pi - x) + 3 * Real.cos (-x)) /
  (Real.sin (x + Real.pi / 2) - Real.sin (x + Real.pi)) = 9 / 4 :=
by
  sorry

-- The second problem
theorem problem2 (x : ℝ) (h : Real.tan x = 3) :
  2 * Real.sin x ^ 2 - Real.sin (2 * x) + Real.cos x ^ 2 = 13 / 10 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l252_25258


namespace NUMINAMATH_GPT_find_b_l252_25264

theorem find_b (x : ℝ) (b : ℝ) :
  (3 * x + 9 = 0) → (2 * b * x - 15 = -5) → b = -5 / 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_b_l252_25264


namespace NUMINAMATH_GPT_diff_one_tenth_and_one_tenth_percent_of_6000_l252_25248

def one_tenth_of_6000 := 6000 / 10
def one_tenth_percent_of_6000 := (1 / 1000) * 6000

theorem diff_one_tenth_and_one_tenth_percent_of_6000 : 
  (one_tenth_of_6000 - one_tenth_percent_of_6000) = 594 :=
by
  sorry

end NUMINAMATH_GPT_diff_one_tenth_and_one_tenth_percent_of_6000_l252_25248


namespace NUMINAMATH_GPT_grandpa_movie_time_l252_25281

theorem grandpa_movie_time
  (each_movie_time : ℕ := 90)
  (max_movies_2_days : ℕ := 9)
  (x_movies_tuesday : ℕ)
  (movies_wednesday := 2 * x_movies_tuesday)
  (total_movies := x_movies_tuesday + movies_wednesday)
  (h : total_movies = max_movies_2_days) :
  90 * x_movies_tuesday = 270 :=
by
  sorry

end NUMINAMATH_GPT_grandpa_movie_time_l252_25281


namespace NUMINAMATH_GPT_rounding_no_order_l252_25279

theorem rounding_no_order (x : ℝ) (hx : x > 0) :
  let a := round (x * 100) / 100
  let b := round (x * 1000) / 1000
  let c := round (x * 10000) / 10000
  (¬((a ≥ b ∧ b ≥ c) ∨ (a ≤ b ∧ b ≤ c))) :=
sorry

end NUMINAMATH_GPT_rounding_no_order_l252_25279


namespace NUMINAMATH_GPT_mean_score_of_all_students_l252_25205

-- Conditions
def M : ℝ := 90
def A : ℝ := 75
def ratio (m a : ℝ) : Prop := m / a = 2 / 3

-- Question and correct answer
theorem mean_score_of_all_students (m a : ℝ) (hm : ratio m a) : (60 * a + 75 * a) / (5 * a / 3) = 81 := by
  sorry

end NUMINAMATH_GPT_mean_score_of_all_students_l252_25205


namespace NUMINAMATH_GPT_four_digit_square_number_divisible_by_11_with_unit_1_l252_25240

theorem four_digit_square_number_divisible_by_11_with_unit_1 
  : ∃ y : ℕ, y >= 1000 ∧ y <= 9999 ∧ (∃ n : ℤ, y = n^2) ∧ y % 11 = 0 ∧ y % 10 = 1 ∧ y = 9801 := 
by {
  -- sorry statement to skip the proof.
  sorry 
}

end NUMINAMATH_GPT_four_digit_square_number_divisible_by_11_with_unit_1_l252_25240
