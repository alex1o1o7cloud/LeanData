import Mathlib

namespace evaluate_exponentiation_l1124_112400

theorem evaluate_exponentiation : (3 ^ 3) ^ 4 = 531441 := by
  sorry

end evaluate_exponentiation_l1124_112400


namespace number_of_terms_is_10_l1124_112435

noncomputable def arith_seq_number_of_terms (a : ℕ) (n : ℕ) (d : ℕ) : Prop :=
  (n % 2 = 0) ∧ ((n-1)*d = 16) ∧ (n * (2*a + (n-2)*d) = 56) ∧ (n * (2*a + n*d) = 76)

theorem number_of_terms_is_10 (a d n : ℕ) (h : arith_seq_number_of_terms a n d) : n = 10 := by
  sorry

end number_of_terms_is_10_l1124_112435


namespace p_2015_coordinates_l1124_112420

namespace AaronWalk

def position (n : ℕ) : ℤ × ℤ :=
sorry

theorem p_2015_coordinates : position 2015 = (22, 57) := 
sorry

end AaronWalk

end p_2015_coordinates_l1124_112420


namespace distance_between_cars_l1124_112455

-- Definitions representing the initial conditions and distances traveled by the cars
def initial_distance : ℕ := 113
def first_car_distance_on_road : ℕ := 50
def second_car_distance_on_road : ℕ := 35

-- Statement of the theorem to be proved
theorem distance_between_cars : initial_distance - (first_car_distance_on_road + second_car_distance_on_road) = 28 :=
by
  sorry

end distance_between_cars_l1124_112455


namespace MikaWaterLeft_l1124_112479

def MikaWaterRemaining (startWater : ℚ) (usedWater : ℚ) : ℚ :=
  startWater - usedWater

theorem MikaWaterLeft :
  MikaWaterRemaining 3 (11 / 8) = 13 / 8 :=
by 
  sorry

end MikaWaterLeft_l1124_112479


namespace swimming_problem_l1124_112498

/-- The swimming problem where a man swims downstream 30 km and upstream a certain distance 
    taking 6 hours each time. Given his speed in still water is 4 km/h, we aim to prove the 
    distance swam upstream is 18 km. -/
theorem swimming_problem 
  (V_m : ℝ) (Distance_downstream : ℝ) (Time_downstream : ℝ) (Time_upstream : ℝ) 
  (Distance_upstream : ℝ) (V_s : ℝ)
  (h1 : V_m = 4)
  (h2 : Distance_downstream = 30)
  (h3 : Time_downstream = 6)
  (h4 : Time_upstream = 6)
  (h5 : V_m + V_s = Distance_downstream / Time_downstream)
  (h6 : V_m - V_s = Distance_upstream / Time_upstream) :
  Distance_upstream = 18 := 
sorry

end swimming_problem_l1124_112498


namespace range_of_a_l1124_112469

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) → -1 ≤ a ∧ a < 0 :=
by
  sorry

end range_of_a_l1124_112469


namespace Isabel_afternoon_runs_l1124_112426

theorem Isabel_afternoon_runs (circuit_length morning_runs weekly_distance afternoon_runs : ℕ)
  (h_circuit_length : circuit_length = 365)
  (h_morning_runs : morning_runs = 7)
  (h_weekly_distance : weekly_distance = 25550)
  (h_afternoon_runs : weekly_distance = morning_runs * circuit_length * 7 + afternoon_runs * circuit_length) :
  afternoon_runs = 21 :=
by
  -- The actual proof goes here
  sorry

end Isabel_afternoon_runs_l1124_112426


namespace ab_cd_value_l1124_112477

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = -3) 
  (h3 : a + c + d = 10) 
  (h4 : b + c + d = -1) : 
  ab + cd = -346 / 9 :=
by 
  sorry

end ab_cd_value_l1124_112477


namespace toys_of_Jason_l1124_112459

theorem toys_of_Jason (R J Jason : ℕ) 
  (hR : R = 1) 
  (hJ : J = R + 6) 
  (hJason : Jason = 3 * J) : 
  Jason = 21 :=
by
  sorry

end toys_of_Jason_l1124_112459


namespace painter_completion_time_l1124_112442

def hours_elapsed (start_time end_time : String) : ℕ :=
  match (start_time, end_time) with
  | ("9:00 AM", "12:00 PM") => 3
  | _ => 0

-- The initial conditions, the start time is 9:00 AM, and 3 hours later 1/4th is done
def start_time := "9:00 AM"
def partial_completion_time := "12:00 PM"
def partial_completion_fraction := 1 / 4
def partial_time_hours := hours_elapsed start_time partial_completion_time

-- The painter works consistently, so it would take 4 times the partial time to complete the job
def total_time_hours := 4 * partial_time_hours

-- Calculate the completion time by adding total_time_hours to the start_time
def completion_time : String :=
  match start_time with
  | "9:00 AM" => "9:00 PM"
  | _         => "unknown"

theorem painter_completion_time :
  completion_time = "9:00 PM" :=
by
  -- Definitions and calculations already included in the setup
  sorry

end painter_completion_time_l1124_112442


namespace steve_annual_salary_l1124_112454

variable (S : ℝ)

theorem steve_annual_salary :
  (0.70 * S - 800 = 27200) → (S = 40000) :=
by
  intro h
  sorry

end steve_annual_salary_l1124_112454


namespace probability_of_success_l1124_112448

theorem probability_of_success 
  (pA : ℚ) (pB : ℚ) 
  (hA : pA = 2 / 3) 
  (hB : pB = 3 / 5) :
  1 - ((1 - pA) * (1 - pB)) = 13 / 15 :=
by
  sorry

end probability_of_success_l1124_112448


namespace delaney_travel_time_l1124_112445

def bus_leaves_at := 8 * 60
def delaney_left_at := 7 * 60 + 50
def missed_by := 20

theorem delaney_travel_time
  (bus_leaves_at : ℕ) (delaney_left_at : ℕ) (missed_by : ℕ) :
  delaney_left_at + (bus_leaves_at + missed_by - bus_leaves_at) - delaney_left_at = 30 :=
by
  exact sorry

end delaney_travel_time_l1124_112445


namespace units_digit_of_45_pow_125_plus_7_pow_87_l1124_112421

theorem units_digit_of_45_pow_125_plus_7_pow_87 :
  (45 ^ 125 + 7 ^ 87) % 10 = 8 :=
by
  -- sorry to skip the proof
  sorry

end units_digit_of_45_pow_125_plus_7_pow_87_l1124_112421


namespace exists_xyz_prime_expression_l1124_112407

theorem exists_xyz_prime_expression (a b c p : ℤ) (h_prime : Prime p)
    (h_div : p ∣ (a^2 + b^2 + c^2 - ab - bc - ca))
    (h_gcd : Int.gcd p ((a^2 + b^2 + c^2 - ab - bc - ca) / p) = 1) :
    ∃ x y z : ℤ, p = x^2 + y^2 + z^2 - xy - yz - zx := by
  sorry

end exists_xyz_prime_expression_l1124_112407


namespace melina_age_l1124_112497

theorem melina_age (A M : ℕ) (alma_score : ℕ := 40) 
    (h1 : A + M = 2 * alma_score) 
    (h2 : M = 3 * A) : 
    M = 60 :=
by 
  sorry

end melina_age_l1124_112497


namespace range_of_x_l1124_112492

theorem range_of_x
  (x : ℝ)
  (h1 : ∀ m, -1 ≤ m ∧ m ≤ 4 → m * (x^2 - 1) - 1 - 8 * x < 0) :
  0 < x ∧ x < 5 / 2 :=
sorry

end range_of_x_l1124_112492


namespace total_students_1150_l1124_112429

theorem total_students_1150 (T G : ℝ) (h1 : 92 + G = T) (h2 : G = 0.92 * T) : T = 1150 := 
by
  sorry

end total_students_1150_l1124_112429


namespace roberta_started_with_8_records_l1124_112456

variable (R : ℕ)

def received_records := 12
def bought_records := 30
def total_received_and_bought := received_records + bought_records

theorem roberta_started_with_8_records (h : R + total_received_and_bought = 50) : R = 8 :=
by
  sorry

end roberta_started_with_8_records_l1124_112456


namespace number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l1124_112446

theorem number_of_pentagonal_faces_is_12_more_than_heptagonal_faces
  (convex : Prop)
  (trihedral : Prop)
  (faces_have_5_6_or_7_sides : Prop)
  (V E F : ℕ)
  (a b c : ℕ)
  (euler : V - E + F = 2)
  (edges_def : E = (5 * a + 6 * b + 7 * c) / 2)
  (vertices_def : V = (5 * a + 6 * b + 7 * c) / 3) :
  a = c + 12 :=
  sorry

end number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l1124_112446


namespace salary_for_may_l1124_112493

theorem salary_for_may (J F M A May : ℝ) 
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8200)
  (h3 : J = 5700) : 
  May = 6500 :=
by 
  have eq1 : J + F + M + A = 32000 := by
    linarith
  have eq2 : F + M + A + May = 32800 := by
    linarith
  have eq3 : May - J = 800 := by
    linarith [eq1, eq2]
  have eq4 : May = 6500 := by
    linarith [eq3, h3]
  exact eq4

end salary_for_may_l1124_112493


namespace riddles_ratio_l1124_112433

theorem riddles_ratio (Josh_riddles : ℕ) (Ivory_riddles : ℕ) (Taso_riddles : ℕ) 
  (h1 : Josh_riddles = 8) 
  (h2 : Ivory_riddles = Josh_riddles + 4) 
  (h3 : Taso_riddles = 24) : 
  Taso_riddles / Ivory_riddles = 2 := 
by sorry

end riddles_ratio_l1124_112433


namespace total_stars_l1124_112483

/-- Let n be the number of students, and s be the number of stars each student makes.
    We need to prove that the total number of stars is n * s. --/
theorem total_stars (n : ℕ) (s : ℕ) (h_n : n = 186) (h_s : s = 5) : n * s = 930 :=
by {
  sorry
}

end total_stars_l1124_112483


namespace solve_problem_l1124_112461

noncomputable def problem_statement : Prop :=
  let a := Real.arcsin (4/5)
  let b := Real.arccos (1/2)
  Real.sin (a + b) = (4 + 3 * Real.sqrt 3) / 10

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l1124_112461


namespace find_f1_find_f3_range_of_x_l1124_112464

-- Define f as described
axiom f : ℝ → ℝ
axiom f_domain : ∀ (x : ℝ), x > 0 → ∃ (y : ℝ), f y = f x

-- Given conditions
axiom condition1 : ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0
axiom condition2 : ∀ (x y : ℝ), 0 < x ∧ 0 < y → f (x * y) = f x + f y
axiom condition3 : f (1 / 3) = 1

-- Prove f(1) = 0
theorem find_f1 : f 1 = 0 := by sorry

-- Prove f(3) = -1
theorem find_f3 : f 3 = -1 := by sorry

-- Given inequality condition
axiom condition4 : ∀ x : ℝ, 0 < x → f x < 2 + f (2 - x)

-- Prove range of x for given inequality
theorem range_of_x : ∀ x, x > 1 / 5 ∧ x < 2 ↔ f x < 2 + f (2 - x) := by sorry

end find_f1_find_f3_range_of_x_l1124_112464


namespace cone_volume_l1124_112484

theorem cone_volume (V_cyl : ℝ) (d : ℝ) (π : ℝ) (V_cyl_eq : V_cyl = 81 * π) (h_eq : 2 * (d / 2) = 2 * d) :
  ∃ (V_cone : ℝ), V_cone = 27 * π * (6 ^ (1/3)) :=
by 
  sorry

end cone_volume_l1124_112484


namespace sequence_contains_infinitely_many_powers_of_two_l1124_112481

theorem sequence_contains_infinitely_many_powers_of_two (a : ℕ → ℕ) (b : ℕ → ℕ) : 
  (∃ a1, a1 % 5 ≠ 0 ∧ a 0 = a1) →
  (∀ n : ℕ, a (n + 1) = a n + b n) →
  (∀ n : ℕ, b n = a n % 10) →
  (∃ n : ℕ, ∃ k : ℕ, 2^k = a n) :=
by
  sorry

end sequence_contains_infinitely_many_powers_of_two_l1124_112481


namespace each_friend_pays_20_l1124_112427

def rent_cottage_cost_per_hour : ℕ := 5
def rent_cottage_hours : ℕ := 8
def total_rent_cost := rent_cottage_cost_per_hour * rent_cottage_hours
def number_of_friends : ℕ := 2
def each_friend_pays := total_rent_cost / number_of_friends

theorem each_friend_pays_20 :
  each_friend_pays = 20 := by
  sorry

end each_friend_pays_20_l1124_112427


namespace abigail_spent_in_store_l1124_112411

theorem abigail_spent_in_store (initial_amount : ℕ) (amount_left : ℕ) (amount_lost : ℕ) (spent : ℕ) 
  (h1 : initial_amount = 11) 
  (h2 : amount_left = 3)
  (h3 : amount_lost = 6) :
  spent = initial_amount - (amount_left + amount_lost) :=
by
  sorry

end abigail_spent_in_store_l1124_112411


namespace absolute_error_2175000_absolute_error_1730000_l1124_112478

noncomputable def absolute_error (a : ℕ) : ℕ :=
  if a = 2175000 then 1
  else if a = 1730000 then 10000
  else 0

theorem absolute_error_2175000 : absolute_error 2175000 = 1 :=
by sorry

theorem absolute_error_1730000 : absolute_error 1730000 = 10000 :=
by sorry

end absolute_error_2175000_absolute_error_1730000_l1124_112478


namespace area_of_sine_triangle_l1124_112415

-- We define the problem conditions and the statement we want to prove
theorem area_of_sine_triangle (A B C : Real) (area_ABC : ℝ) (unit_circle : ℝ) :
  unit_circle = 1 → area_ABC = 1 / 2 →
  let a := 2 * Real.sin A
  let b := 2 * Real.sin B
  let c := 2 * Real.sin C
  let s := (a + b + c) / 2
  let area_sine_triangle := 
    (s * (s - a) * (s - b) * (s - c)).sqrt / 4 
  area_sine_triangle = 1 / 8 :=
by
  intros
  sorry -- Proof is left as an exercise

end area_of_sine_triangle_l1124_112415


namespace difference_in_probabilities_is_twenty_percent_l1124_112422

-- Definition of the problem conditions
def prob_win_first_lawsuit : ℝ := 0.30
def prob_lose_first_lawsuit : ℝ := 0.70
def prob_win_second_lawsuit : ℝ := 0.50
def prob_lose_second_lawsuit : ℝ := 0.50

-- We need to prove that the difference in probability of losing both lawsuits and winning both lawsuits is 20%
theorem difference_in_probabilities_is_twenty_percent :
  (prob_lose_first_lawsuit * prob_lose_second_lawsuit) -
  (prob_win_first_lawsuit * prob_win_second_lawsuit) = 0.20 := 
by
  sorry

end difference_in_probabilities_is_twenty_percent_l1124_112422


namespace green_beads_in_each_necklace_l1124_112465

theorem green_beads_in_each_necklace (G : ℕ) :
  (∀ n, (n = 5) → (6 * n ≤ 45) ∧ (3 * n ≤ 45) ∧ (G * n = 45)) → G = 9 :=
by
  intros h
  have hn : 5 = 5 := rfl
  cases h 5 hn
  sorry

end green_beads_in_each_necklace_l1124_112465


namespace marbles_in_larger_container_l1124_112496

-- Defining the conditions
def volume1 := 24 -- in cm³
def marbles1 := 30 -- number of marbles in the first container
def volume2 := 72 -- in cm³

-- Statement of the theorem
theorem marbles_in_larger_container : (marbles1 / volume1 : ℚ) * volume2 = 90 := by
  sorry

end marbles_in_larger_container_l1124_112496


namespace tilling_time_in_minutes_l1124_112491

-- Definitions
def plot_width : ℕ := 110
def plot_length : ℕ := 120
def tiller_width : ℕ := 2
def tilling_rate : ℕ := 2 -- 2 seconds per foot

-- Theorem: The time to till the entire plot in minutes
theorem tilling_time_in_minutes : (plot_width / tiller_width * plot_length * tilling_rate) / 60 = 220 := by
  sorry

end tilling_time_in_minutes_l1124_112491


namespace prime_factorization_675_l1124_112425

theorem prime_factorization_675 :
  ∃ (n h : ℕ), n > 1 ∧ n = 3 ∧ h = 225 ∧ 675 = (3^3) * (5^2) :=
by
  sorry

end prime_factorization_675_l1124_112425


namespace plan_b_rate_l1124_112458

noncomputable def cost_plan_a (duration : ℕ) : ℝ :=
  if duration ≤ 4 then 0.60
  else 0.60 + 0.06 * (duration - 4)

def cost_plan_b (duration : ℕ) (rate : ℝ) : ℝ :=
  rate * duration

theorem plan_b_rate (rate : ℝ) : 
  cost_plan_a 18 = cost_plan_b 18 rate → rate = 0.08 := 
by
  -- proof goes here
  sorry

end plan_b_rate_l1124_112458


namespace john_total_trip_cost_l1124_112466

noncomputable def total_trip_cost
  (hotel_nights : ℕ) 
  (hotel_rate_per_night : ℝ) 
  (discount : ℝ) 
  (loyal_customer_discount_rate : ℝ) 
  (service_tax_rate : ℝ) 
  (room_service_cost_per_day : ℝ) 
  (cab_cost_per_ride : ℝ) : ℝ :=
  let hotel_cost := hotel_nights * hotel_rate_per_night
  let cost_after_discount := hotel_cost - discount
  let loyal_customer_discount := loyal_customer_discount_rate * cost_after_discount
  let cost_after_loyalty_discount := cost_after_discount - loyal_customer_discount
  let service_tax := service_tax_rate * cost_after_loyalty_discount
  let final_hotel_cost := cost_after_loyalty_discount + service_tax
  let room_service_cost := hotel_nights * room_service_cost_per_day
  let cab_cost := cab_cost_per_ride * 2 * hotel_nights
  final_hotel_cost + room_service_cost + cab_cost

theorem john_total_trip_cost : total_trip_cost 3 250 100 0.10 0.12 50 30 = 985.20 :=
by 
  -- We are skipping the proof but our focus is the statement
  sorry

end john_total_trip_cost_l1124_112466


namespace middle_integer_is_six_l1124_112439

def valid_even_integer (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = x ∧ x = n - 2 ∧ y = n ∧ z = n + 2 ∧ x < y ∧ y < z ∧
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9

theorem middle_integer_is_six (n : ℕ) (h : valid_even_integer n) :
  n = 6 :=
by
  sorry

end middle_integer_is_six_l1124_112439


namespace area_of_smallest_square_l1124_112403

theorem area_of_smallest_square (radius : ℝ) (h : radius = 6) : 
    ∃ s : ℝ, s = 2 * radius ∧ s^2 = 144 :=
by
  sorry

end area_of_smallest_square_l1124_112403


namespace jacket_cost_is_30_l1124_112402

-- Let's define the given conditions
def num_dresses := 5
def cost_per_dress := 20 -- dollars
def num_pants := 3
def cost_per_pant := 12 -- dollars
def num_jackets := 4
def transport_cost := 5 -- dollars
def initial_amount := 400 -- dollars
def remaining_amount := 139 -- dollars

-- Define the cost per jacket
def cost_per_jacket := 30 -- dollars

-- Final theorem statement to be proved
theorem jacket_cost_is_30:
  num_dresses * cost_per_dress + num_pants * cost_per_pant + num_jackets * cost_per_jacket + transport_cost = initial_amount - remaining_amount :=
sorry

end jacket_cost_is_30_l1124_112402


namespace incorrect_statement_l1124_112486

-- Define the operation (x * y)
def op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem to show the incorrectness of the given statement
theorem incorrect_statement (x y z : ℝ) : op x (y + z) ≠ op x y + op x z :=
  sorry

end incorrect_statement_l1124_112486


namespace ratio_of_Carla_to_Cosima_l1124_112468

variables (C M : ℝ)

-- Natasha has 3 times as much money as Carla
axiom h1 : 3 * C = 60

-- Carla has the same amount of money as Cosima
axiom h2 : C = M

-- Prove: the ratio of Carla's money to Cosima's money is 1:1
theorem ratio_of_Carla_to_Cosima : C / M = 1 :=
by sorry

end ratio_of_Carla_to_Cosima_l1124_112468


namespace infinite_triples_exists_l1124_112401

/-- There are infinitely many ordered triples (a, b, c) of positive integers such that 
the greatest common divisor of a, b, and c is 1, and the sum a^2b^2 + b^2c^2 + c^2a^2 
is the square of an integer. -/
theorem infinite_triples_exists : ∃ (a b c : ℕ), (∀ p q : ℕ, p ≠ q ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ 2 < p ∧ 2 < q →
  let a := p * q
  let b := 2 * p^2
  let c := q^2
  gcd (gcd a b) c = 1 ∧
  ∃ k : ℕ, a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = k^2) :=
sorry

end infinite_triples_exists_l1124_112401


namespace sum_a_b_l1124_112413

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 5 * b = 47) (h2 : 4 * a + 2 * b = 38) : a + b = 85 / 7 :=
by
  sorry

end sum_a_b_l1124_112413


namespace minimum_value_x2_y2_l1124_112449

variable {x y : ℝ}

theorem minimum_value_x2_y2 (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x * y = 1) : x^2 + y^2 = 2 :=
sorry

end minimum_value_x2_y2_l1124_112449


namespace find_natural_number_l1124_112480

variable {A : ℕ}

theorem find_natural_number (h1 : A = 8 * 2 + 7) : A = 23 :=
sorry

end find_natural_number_l1124_112480


namespace fraction_spent_at_arcade_l1124_112404

theorem fraction_spent_at_arcade :
  ∃ f : ℝ, 
    (2.25 - (2.25 * f) - ((2.25 - (2.25 * f)) / 3) = 0.60) → 
    f = 3 / 5 :=
by
  sorry

end fraction_spent_at_arcade_l1124_112404


namespace final_selling_price_correct_l1124_112471

noncomputable def purchase_price_inr : ℝ := 8000
noncomputable def depreciation_rate_annual : ℝ := 0.10
noncomputable def profit_rate : ℝ := 0.10
noncomputable def discount_rate : ℝ := 0.05
noncomputable def sales_tax_rate : ℝ := 0.12
noncomputable def exchange_rate_at_purchase : ℝ := 80
noncomputable def exchange_rate_at_selling : ℝ := 75

noncomputable def depreciated_value_after_2_years (initial_value : ℝ) : ℝ :=
  initial_value * (1 - depreciation_rate_annual) * (1 - depreciation_rate_annual)

noncomputable def marked_price (initial_value : ℝ) : ℝ :=
  initial_value * (1 + profit_rate)

noncomputable def selling_price_before_tax (marked_price : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

noncomputable def final_selling_price_inr (selling_price_before_tax : ℝ) : ℝ :=
  selling_price_before_tax * (1 + sales_tax_rate)

noncomputable def final_selling_price_usd (final_selling_price_inr : ℝ) : ℝ :=
  final_selling_price_inr / exchange_rate_at_selling

theorem final_selling_price_correct :
  final_selling_price_usd (final_selling_price_inr (selling_price_before_tax (marked_price purchase_price_inr))) = 124.84 := 
sorry

end final_selling_price_correct_l1124_112471


namespace inhabitants_reach_ball_on_time_l1124_112405

theorem inhabitants_reach_ball_on_time
  (kingdom_side_length : ℝ)
  (messenger_sent_at : ℕ)
  (ball_begins_at : ℕ)
  (inhabitant_speed : ℝ)
  (time_available : ℝ)
  (max_distance_within_square : ℝ)
  (H_side_length : kingdom_side_length = 2)
  (H_messenger_time : messenger_sent_at = 12)
  (H_ball_time : ball_begins_at = 19)
  (H_speed : inhabitant_speed = 3)
  (H_time_avail : time_available = 7)
  (H_max_distance : max_distance_within_square = 2 * Real.sqrt 2) :
  ∃ t : ℝ, t ≤ time_available ∧ max_distance_within_square / inhabitant_speed ≤ t :=
by
  -- You would write the proof here.
  sorry

end inhabitants_reach_ball_on_time_l1124_112405


namespace triangle_area_is_2_l1124_112432

noncomputable def area_of_triangle_OAB {x₀ : ℝ} (h₀ : 0 < x₀) : ℝ :=
  let y₀ := 1 / x₀
  let slope := -1 / x₀^2
  let tangent_line (x : ℝ) := y₀ + slope * (x - x₀)
  let A : ℝ × ℝ := (2 * x₀, 0) -- Intersection with x-axis
  let B : ℝ × ℝ := (0, 2 * y₀) -- Intersection with y-axis
  1 / 2 * abs (2 * y₀ * 2 * x₀)

theorem triangle_area_is_2 (x₀ : ℝ) (h₀ : 0 < x₀) : area_of_triangle_OAB h₀ = 2 :=
by
  sorry

end triangle_area_is_2_l1124_112432


namespace divisible_by_six_l1124_112487

theorem divisible_by_six (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 6 ∣ n :=
sorry

end divisible_by_six_l1124_112487


namespace maria_fraction_of_remaining_distance_l1124_112406

theorem maria_fraction_of_remaining_distance (total_distance remaining_distance distance_travelled : ℕ) 
(h_total : total_distance = 480) 
(h_first_stop : distance_travelled = total_distance / 2) 
(h_remaining : remaining_distance = total_distance - distance_travelled)
(h_final_leg : remaining_distance - distance_travelled = 180) : 
(distance_travelled / remaining_distance) = (1 / 4) := 
by
  sorry

end maria_fraction_of_remaining_distance_l1124_112406


namespace circle_area_and_circumference_changes_l1124_112430

noncomputable section

structure Circle :=
  (r : ℝ)

def area (c : Circle) : ℝ := Real.pi * c.r^2

def circumference (c : Circle) : ℝ := 2 * Real.pi * c.r

def percentage_change (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

theorem circle_area_and_circumference_changes
  (r1 r2 : ℝ) (c1 : Circle := {r := r1}) (c2 : Circle := {r := r2})
  (h1 : r1 = 5) (h2 : r2 = 4) :
  let original_area := area c1
  let new_area := area c2
  let original_circumference := circumference c1
  let new_circumference := circumference c2
  percentage_change original_area new_area = 36 ∧
  new_circumference = 8 * Real.pi ∧
  percentage_change original_circumference new_circumference = 20 :=
by
  sorry

end circle_area_and_circumference_changes_l1124_112430


namespace pizzas_ordered_l1124_112434

variable (m : ℕ) (x : ℕ)

theorem pizzas_ordered (h1 : m * 2 * x = 14) (h2 : x = 1 / 2 * m) (h3 : m > 13) : 
  14 + 13 * x = 15 := 
sorry

end pizzas_ordered_l1124_112434


namespace inequality_holds_for_a_in_interval_l1124_112494

theorem inequality_holds_for_a_in_interval:
  (∀ x y : ℝ, 
     2 ≤ x ∧ x ≤ 3 ∧ 3 ≤ y ∧ y ≤ 4 → (3*x - 2*y - a) * (3*x - 2*y - a^2) ≤ 0) ↔ a ∈ Set.Iic (-4) :=
by
  sorry

end inequality_holds_for_a_in_interval_l1124_112494


namespace juanita_sunscreen_cost_l1124_112440

theorem juanita_sunscreen_cost:
  let bottles_per_month := 1
  let months_in_year := 12
  let cost_per_bottle := 30.0
  let discount_rate := 0.30
  let total_bottles := bottles_per_month * months_in_year
  let total_cost_before_discount := total_bottles * cost_per_bottle
  let discount_amount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  total_cost_after_discount = 252.00 := 
by
  sorry

end juanita_sunscreen_cost_l1124_112440


namespace total_questions_attempted_l1124_112417

theorem total_questions_attempted (C W T : ℕ) 
    (hC : C = 36) 
    (hScore : 120 = (4 * C) - W) 
    (hT : T = C + W) : 
    T = 60 := 
by 
  sorry

end total_questions_attempted_l1124_112417


namespace opposite_of_neg_one_third_l1124_112457

theorem opposite_of_neg_one_third : -(-1/3) = 1/3 := 
sorry

end opposite_of_neg_one_third_l1124_112457


namespace percentage_profit_l1124_112499

theorem percentage_profit (cp sp : ℝ) (h1 : cp = 1200) (h2 : sp = 1680) : ((sp - cp) / cp) * 100 = 40 := 
by 
  sorry

end percentage_profit_l1124_112499


namespace find_c_l1124_112451

open Real

noncomputable def triangle_side_c (a b c : ℝ) (A B C : ℝ) :=
  A = (π / 4) ∧
  2 * b * sin B - c * sin C = 2 * a * sin A ∧
  (1/2) * b * c * (sqrt 2)/2 = 3 →
  c = 2 * sqrt 2
  
theorem find_c {a b c A B C : ℝ} (h : triangle_side_c a b c A B C) : c = 2 * sqrt 2 :=
sorry

end find_c_l1124_112451


namespace original_price_of_tennis_racket_l1124_112443

theorem original_price_of_tennis_racket
  (sneaker_cost : ℝ) (outfit_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ)
  (price_of_tennis_racket : ℝ) :
  sneaker_cost = 200 → 
  outfit_cost = 250 → 
  discount_rate = 0.20 → 
  total_spent = 750 → 
  price_of_tennis_racket = 289.77 :=
by
  intros hs ho hd ht
  have ht := ht.symm   -- To rearrange the equation
  sorry

end original_price_of_tennis_racket_l1124_112443


namespace distance_from_plate_to_bottom_edge_l1124_112488

theorem distance_from_plate_to_bottom_edge (d : ℝ) : 
  (10 + d + 63 = 20 + d + 53) :=
by
  -- The proof can be completed here.
  sorry

end distance_from_plate_to_bottom_edge_l1124_112488


namespace mooney_ate_correct_l1124_112450

-- Define initial conditions
def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mother_added : ℕ := 24
def final_brownies : ℕ := 36

-- Define Mooney ate some brownies
variable (mooney_ate : ℕ)

-- Prove that Mooney ate 4 brownies
theorem mooney_ate_correct :
  (initial_brownies - father_ate) - mooney_ate + mother_added = final_brownies →
  mooney_ate = 4 :=
by
  sorry

end mooney_ate_correct_l1124_112450


namespace maximize_A_plus_C_l1124_112460

theorem maximize_A_plus_C (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
 (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (hB : B = 2) (h7 : (A + C) % (B + D) = 0) 
 (h8 : A < 10) (h9 : B < 10) (h10 : C < 10) (h11 : D < 10) : 
 A + C ≤ 15 :=
sorry

end maximize_A_plus_C_l1124_112460


namespace calculate_taxi_fare_l1124_112431

theorem calculate_taxi_fare :
  ∀ (f_80 f_120: ℝ), f_80 = 160 ∧ f_80 = 20 + (80 * (140/80)) →
                      f_120 = 20 + (120 * (140/80)) →
                      f_120 = 230 :=
by
  intro f_80 f_120
  rintro ⟨h80, h_proportional⟩ h_120
  sorry

end calculate_taxi_fare_l1124_112431


namespace car_speed_l1124_112463

theorem car_speed (rev_per_min : ℕ) (circ : ℝ) (h_rev : rev_per_min = 400) (h_circ : circ = 5) : 
  (rev_per_min * circ) * 60 / 1000 = 120 :=
by
  sorry

end car_speed_l1124_112463


namespace algebraic_expression_value_l1124_112474

theorem algebraic_expression_value (a b : ℕ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b * b) / a) / ((a * a - b * b) / a) = 1 / 2 := 
sorry

end algebraic_expression_value_l1124_112474


namespace error_percentage_in_area_l1124_112447

theorem error_percentage_in_area
  (L W : ℝ)          -- Actual length and width of the rectangle
  (hL' : ℝ)          -- Measured length with 8% excess
  (hW' : ℝ)          -- Measured width with 5% deficit
  (hL'_def : hL' = 1.08 * L)  -- Condition for length excess
  (hW'_def : hW' = 0.95 * W)  -- Condition for width deficit
  :
  ((hL' * hW' - L * W) / (L * W) * 100 = 2.6) := sorry

end error_percentage_in_area_l1124_112447


namespace total_berries_l1124_112414

theorem total_berries (S_stacy S_steve S_skylar : ℕ) 
  (h1 : S_stacy = 800)
  (h2 : S_stacy = 4 * S_steve)
  (h3 : S_steve = 2 * S_skylar) :
  S_stacy + S_steve + S_skylar = 1100 :=
by
  sorry

end total_berries_l1124_112414


namespace xiao_li_place_l1124_112436

def guess_A (place : String) : Prop :=
  place ≠ "first" ∧ place ≠ "second"

def guess_B (place : String) : Prop :=
  place ≠ "first" ∧ place = "third"

def guess_C (place : String) : Prop :=
  place ≠ "third" ∧ place = "first"

def correct_guesses (guess : String → Prop) (place : String) : Prop :=
  guess place

def half_correct_guesses (guess : String → Prop) (place : String) : Prop :=
  (guess "first" = (place = "first")) ∨
  (guess "second" = (place = "second")) ∨
  (guess "third" = (place = "third"))

theorem xiao_li_place :
  ∃ (place : String),
  (correct_guesses guess_A place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_B place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_B place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_A place) :=
sorry

end xiao_li_place_l1124_112436


namespace units_digit_of_24_pow_4_add_42_pow_4_l1124_112444

theorem units_digit_of_24_pow_4_add_42_pow_4 : 
  (24^4 + 42^4) % 10 = 2 := 
by sorry

end units_digit_of_24_pow_4_add_42_pow_4_l1124_112444


namespace total_donations_correct_l1124_112428

def num_basketball_hoops : Nat := 60

def num_hoops_with_balls : Nat := num_basketball_hoops / 2

def num_pool_floats : Nat := 120
def num_damaged_floats : Nat := num_pool_floats / 4
def num_remaining_floats : Nat := num_pool_floats - num_damaged_floats

def num_footballs : Nat := 50
def num_tennis_balls : Nat := 40

def num_hoops_without_balls : Nat := num_basketball_hoops - num_hoops_with_balls

def total_donations : Nat := 
  num_hoops_without_balls + num_hoops_with_balls + num_remaining_floats + num_footballs + num_tennis_balls

theorem total_donations_correct : total_donations = 240 := by
  sorry

end total_donations_correct_l1124_112428


namespace expression_evaluation_l1124_112452

theorem expression_evaluation : 
  3 / 5 * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := 
by
  sorry

end expression_evaluation_l1124_112452


namespace parallelogram_point_D_l1124_112409

/-- Given points A, B, and C, the coordinates of point D in parallelogram ABCD -/
theorem parallelogram_point_D (A B C D : (ℝ × ℝ))
  (hA : A = (1, 1))
  (hB : B = (3, 2))
  (hC : C = (6, 3))
  (hMid : (2 * (A.1 + C.1), 2 * (A.2 + C.2)) = (2 * (B.1 + D.1), 2 * (B.2 + D.2))) :
  D = (4, 2) :=
sorry

end parallelogram_point_D_l1124_112409


namespace three_digit_numbers_subtract_297_l1124_112453

theorem three_digit_numbers_subtract_297:
  (∃ (p q r : ℕ), 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ 0 ≤ r ∧ r ≤ 9 ∧ (100 * p + 10 * q + r - 297 = 100 * r + 10 * q + p)) →
  (num_valid_three_digit_numbers = 60) :=
by
  sorry

end three_digit_numbers_subtract_297_l1124_112453


namespace directrix_of_parabola_l1124_112412

theorem directrix_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 4 * x^2 - 6) : 
    ∃ d, (∀ x, y x = 4 * x^2 - 6) ∧ d = -97/16 ↔ (y (-6 - d)) = -10 := 
    sorry

end directrix_of_parabola_l1124_112412


namespace most_lines_of_symmetry_circle_l1124_112437

-- Define the figures and their lines of symmetry
def regular_pentagon_lines_of_symmetry : ℕ := 5
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def circle_lines_of_symmetry : ℕ := 0  -- Representing infinite lines of symmetry in Lean is unconventional; we'll use a special case.
def regular_hexagon_lines_of_symmetry : ℕ := 6
def ellipse_lines_of_symmetry : ℕ := 2

-- Define a predicate to check if one figure has more lines of symmetry than all others
def most_lines_of_symmetry {α : Type} [LinearOrder α] (f : α) (others : List α) : Prop :=
  ∀ x ∈ others, f ≥ x

-- Define the problem statement in Lean
theorem most_lines_of_symmetry_circle :
  most_lines_of_symmetry circle_lines_of_symmetry [
    regular_pentagon_lines_of_symmetry,
    isosceles_triangle_lines_of_symmetry,
    regular_hexagon_lines_of_symmetry,
    ellipse_lines_of_symmetry ] :=
by {
  -- To represent infinite lines, we consider 0 as a larger "dummy" number in this context,
  -- since in Lean we don't have a built-in representation for infinity in finite ordering.
  -- Replace with a suitable model if necessary.
  sorry
}

end most_lines_of_symmetry_circle_l1124_112437


namespace first_train_speed_l1124_112485

-- Definitions
def train_speeds_opposite (v₁ v₂ t : ℝ) : Prop := v₁ * t + v₂ * t = 910

def train_problem_conditions (v₁ v₂ t : ℝ) : Prop :=
  train_speeds_opposite v₁ v₂ t ∧ v₂ = 80 ∧ t = 6.5

-- Theorem
theorem first_train_speed (v : ℝ) (h : train_problem_conditions v 80 6.5) : v = 60 :=
  sorry

end first_train_speed_l1124_112485


namespace men_per_table_correct_l1124_112495

def tables := 6
def women_per_table := 3
def total_customers := 48
def total_women := women_per_table * tables
def total_men := total_customers - total_women
def men_per_table := total_men / tables

theorem men_per_table_correct : men_per_table = 5 := by
  sorry

end men_per_table_correct_l1124_112495


namespace no_such_natural_numbers_l1124_112418

theorem no_such_natural_numbers :
  ¬ ∃ (x y : ℕ), (∃ (a b : ℕ), x^2 + y = a^2 ∧ x - y = b^2) := 
sorry

end no_such_natural_numbers_l1124_112418


namespace solve_for_a_l1124_112473

def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 6

theorem solve_for_a (a : ℝ) (h : a > 0) (h1 : f (g a) = 18) : a = Real.sqrt (2 * Real.sqrt 2 + 6) :=
by
  sorry

end solve_for_a_l1124_112473


namespace solve_for_a_l1124_112423

theorem solve_for_a : ∀ (a : ℝ), (2 * a - 16 = 9) → (a = 12.5) :=
by
  intro a h
  sorry

end solve_for_a_l1124_112423


namespace hexagon_area_eq_l1124_112470

theorem hexagon_area_eq (s t : ℝ) (hs : s^2 = 16) (heq : 4 * s = 6 * t) :
  6 * (t^2 * (Real.sqrt 3) / 4) = 32 * (Real.sqrt 3) / 3 := by
  sorry

end hexagon_area_eq_l1124_112470


namespace find_k_l1124_112424

-- Definitions of conditions
def equation1 (x k : ℝ) : Prop := x^2 + k*x + 10 = 0
def equation2 (x k : ℝ) : Prop := x^2 - k*x + 10 = 0
def roots_relation (a b k : ℝ) : Prop :=
  equation1 a k ∧ 
  equation1 b k ∧ 
  equation2 (a + 3) k ∧
  equation2 (b + 3) k

-- Statement to be proven
theorem find_k (a b k : ℝ) (h : roots_relation a b k) : k = 3 :=
sorry

end find_k_l1124_112424


namespace piecewise_function_not_composed_of_multiple_functions_l1124_112441

theorem piecewise_function_not_composed_of_multiple_functions :
  ∀ (f : ℝ → ℝ), (∃ (I : ℝ → Prop) (f₁ f₂ : ℝ → ℝ),
    (∀ x, I x → f x = f₁ x) ∧ (∀ x, ¬I x → f x = f₂ x)) →
    ¬(∃ (g₁ g₂ : ℝ → ℝ), (∀ x, f x = g₁ x ∨ f x = g₂ x)) :=
by
  sorry

end piecewise_function_not_composed_of_multiple_functions_l1124_112441


namespace t_shirts_per_package_l1124_112490

theorem t_shirts_per_package (total_t_shirts : ℕ) (total_packages : ℕ) (h1 : total_t_shirts = 39) (h2 : total_packages = 3) : total_t_shirts / total_packages = 13 :=
by {
  sorry
}

end t_shirts_per_package_l1124_112490


namespace polygon_sides_l1124_112467

theorem polygon_sides (n : ℕ) (a1 d : ℝ) (h1 : a1 = 100) (h2 : d = 10)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d < 180) : n = 8 :=
by
  sorry

end polygon_sides_l1124_112467


namespace value_added_to_each_number_is_11_l1124_112408

-- Given definitions and conditions
def initial_average : ℝ := 40
def number_count : ℕ := 15
def new_average : ℝ := 51

-- Mathematically equivalent proof statement
theorem value_added_to_each_number_is_11 (x : ℝ) 
  (h1 : number_count * initial_average = 600)
  (h2 : (600 + number_count * x) / number_count = new_average) : 
  x = 11 := 
by 
  sorry

end value_added_to_each_number_is_11_l1124_112408


namespace problem_1_problem_2_l1124_112489

-- Problem 1: Prove that sqrt(6) * sqrt(1/3) - sqrt(16) * sqrt(18) = -11 * sqrt(2)
theorem problem_1 : Real.sqrt 6 * Real.sqrt (1 / 3) - Real.sqrt 16 * Real.sqrt 18 = -11 * Real.sqrt 2 := 
by
  sorry

-- Problem 2: Prove that (2 - sqrt(5)) * (2 + sqrt(5)) + (2 - sqrt(2))^2 = 5 - 4 * sqrt(2)
theorem problem_2 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * Real.sqrt 2 := 
by
  sorry

end problem_1_problem_2_l1124_112489


namespace marbles_lost_l1124_112438

theorem marbles_lost (initial_marbs remaining_marbs marbles_lost : ℕ)
  (h1 : initial_marbs = 38)
  (h2 : remaining_marbs = 23)
  : marbles_lost = initial_marbs - remaining_marbs :=
by
  sorry

end marbles_lost_l1124_112438


namespace correct_algorithm_option_l1124_112472

def OptionA := ("Sequential structure", "Flow structure", "Loop structure")
def OptionB := ("Sequential structure", "Conditional structure", "Nested structure")
def OptionC := ("Sequential structure", "Conditional structure", "Loop structure")
def OptionD := ("Flow structure", "Conditional structure", "Loop structure")

-- The correct structures of an algorithm are sequential, conditional, and loop.
def algorithm_structures := ("Sequential structure", "Conditional structure", "Loop structure")

theorem correct_algorithm_option : algorithm_structures = OptionC := 
by 
  -- This would be proven by logic and checking the options; omitted here with 'sorry'
  sorry

end correct_algorithm_option_l1124_112472


namespace problem1_problem2_l1124_112476

namespace MathProof

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem problem1 (m : ℝ) :
  (∀ x, 0 < x → f x m > 0) → -2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5 :=
sorry

theorem problem2 (m : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ f x m = 0) → -2 < m ∧ m < 0 :=
sorry

end MathProof

end problem1_problem2_l1124_112476


namespace find_b_l1124_112462

-- Given conditions
def varies_inversely (a b : ℝ) := ∃ K : ℝ, K = a * b
def constant_a (a : ℝ) := a = 1500
def constant_b (b : ℝ) := b = 0.25

-- The theorem to prove
theorem find_b (a : ℝ) (b : ℝ) (h_inv: varies_inversely a b)
  (h_a: constant_a a) (h_b: constant_b b): b = 0.125 := 
sorry

end find_b_l1124_112462


namespace students_who_like_both_apple_pie_and_chocolate_cake_l1124_112416

def total_students := 50
def students_who_like_apple_pie := 22
def students_who_like_chocolate_cake := 20
def students_who_like_neither := 10
def students_who_like_only_cookies := 5

theorem students_who_like_both_apple_pie_and_chocolate_cake :
  (students_who_like_apple_pie + students_who_like_chocolate_cake - (total_students - students_who_like_neither - students_who_like_only_cookies)) = 7 := 
by
  sorry

end students_who_like_both_apple_pie_and_chocolate_cake_l1124_112416


namespace inequality_proof_l1124_112475

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : x^12 - y^12 + 2 * x^6 * y^6 ≤ (Real.pi / 2) := 
by 
  sorry

end inequality_proof_l1124_112475


namespace burger_cost_l1124_112419

theorem burger_cost 
    (b s : ℕ) 
    (h1 : 5 * b + 3 * s = 500) 
    (h2 : 3 * b + 2 * s = 310) :
    b = 70 := by
  sorry

end burger_cost_l1124_112419


namespace smallest_sum_l1124_112410

theorem smallest_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) 
  (h : (1/x + 1/y = 1/10)) : x + y = 49 := 
sorry

end smallest_sum_l1124_112410


namespace dog_food_l1124_112482

theorem dog_food (weights : List ℕ) (h_weights : weights = [20, 40, 10, 30, 50]) (h_ratio : ∀ w ∈ weights, 1 ≤ w / 10):
  (weights.sum / 10) = 15 := by
  sorry

end dog_food_l1124_112482
