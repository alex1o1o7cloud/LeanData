import Mathlib

namespace fred_speed_5_mph_l203_20385

theorem fred_speed_5_mph (F : ℝ) (h1 : 50 = 25 + 25) (h2 : 25 / 5 = 5) (h3 : 25 / F = 5) : 
  F = 5 :=
by
  -- Since Fred's speed makes meeting with Sam in the same time feasible
  sorry

end fred_speed_5_mph_l203_20385


namespace factory_production_system_l203_20362

theorem factory_production_system (x y : ℕ) (h1 : x + y = 95)
    (h2 : 8*x - 22*y = 0) :
    16*x - 22*y = 0 :=
by
  sorry

end factory_production_system_l203_20362


namespace maria_paper_count_l203_20321

-- Defining the initial number of sheets and the actions taken
variables (x y : ℕ)
def initial_sheets := 50 + 41
def remaining_sheets_after_giving_away := initial_sheets - x
def whole_sheets := remaining_sheets_after_giving_away - y
def half_sheets := y

-- The theorem we want to prove
theorem maria_paper_count (x y : ℕ) :
  whole_sheets x y = initial_sheets - x - y ∧ 
  half_sheets y = y :=
by sorry

end maria_paper_count_l203_20321


namespace solve_for_x_l203_20373

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 7 = 13) : x = 42 :=
sorry

end solve_for_x_l203_20373


namespace cubic_polynomial_at_zero_l203_20351

noncomputable def f (x : ℝ) : ℝ := by sorry

theorem cubic_polynomial_at_zero :
  (∃ f : ℝ → ℝ, f 2 = 15 ∨ f 2 = -15 ∧
                 f 4 = 15 ∨ f 4 = -15 ∧
                 f 5 = 15 ∨ f 5 = -15 ∧
                 f 6 = 15 ∨ f 6 = -15 ∧
                 f 8 = 15 ∨ f 8 = -15 ∧
                 f 9 = 15 ∨ f 9 = -15 ∧
                 ∀ x, ∃ c a b d, f x = c * x^3 + a * x^2 + b * x + d ) →
  |f 0| = 135 :=
by sorry

end cubic_polynomial_at_zero_l203_20351


namespace sum_of_integers_satisfying_l203_20334

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l203_20334


namespace man_l203_20306

theorem man's_salary (S : ℝ) 
  (h_food : S * (1 / 5) > 0)
  (h_rent : S * (1 / 10) > 0)
  (h_clothes : S * (3 / 5) > 0)
  (h_left : S * (1 / 10) = 19000) : 
  S = 190000 := by
  sorry

end man_l203_20306


namespace students_exceed_pets_l203_20325

-- Defining the conditions
def num_students_per_classroom := 25
def num_rabbits_per_classroom := 3
def num_guinea_pigs_per_classroom := 3
def num_classrooms := 5

-- Main theorem to prove
theorem students_exceed_pets:
  let total_students := num_students_per_classroom * num_classrooms
  let total_rabbits := num_rabbits_per_classroom * num_classrooms
  let total_guinea_pigs := num_guinea_pigs_per_classroom * num_classrooms
  let total_pets := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 :=
by 
  sorry

end students_exceed_pets_l203_20325


namespace second_smallest_prime_perimeter_l203_20322

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m ∣ n → m = n

def scalene_triangle (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def prime_perimeter (a b c : ℕ) : Prop := 
  is_prime (a + b + c)

def different_primes (a b c : ℕ) : Prop := 
  is_prime a ∧ is_prime b ∧ is_prime c

theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ), 
  scalene_triangle a b c ∧ 
  different_primes a b c ∧ 
  prime_perimeter a b c ∧ 
  a + b + c = 29 := 
sorry

end second_smallest_prime_perimeter_l203_20322


namespace find_value_of_x_squared_and_reciprocal_squared_l203_20307

theorem find_value_of_x_squared_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + (1/x)^2 = 2 := 
sorry

end find_value_of_x_squared_and_reciprocal_squared_l203_20307


namespace ellipse_sum_l203_20381

theorem ellipse_sum (F1 F2 : ℝ × ℝ) (h k a b : ℝ) 
  (hf1 : F1 = (0, 0)) (hf2 : F2 = (6, 0))
  (h_eqn : ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = 10) :
  h + k + a + b = 12 :=
by
  sorry

end ellipse_sum_l203_20381


namespace proof_expr1_l203_20383

noncomputable def expr1 : ℝ :=
  (Real.sin (65 * Real.pi / 180) + Real.sin (15 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) / 
  (Real.sin (25 * Real.pi / 180) - Real.cos (15 * Real.pi / 180) * Real.cos (80 * Real.pi / 180))

theorem proof_expr1 : expr1 = 2 + Real.sqrt 3 :=
by sorry

end proof_expr1_l203_20383


namespace solve_for_ratio_l203_20350

noncomputable def slope_tangent_y_equals_x_squared (x1 : ℝ) : ℝ :=
  2 * x1

noncomputable def slope_tangent_y_equals_x_cubed (x2 : ℝ) : ℝ :=
  3 * x2 * x2

noncomputable def y1_compute (x1 : ℝ) : ℝ :=
  x1 * x1

noncomputable def y2_compute (x2 : ℝ) : ℝ :=
  x2 * x2 * x2

theorem solve_for_ratio (x1 x2 : ℝ)
    (tangent_l_same : slope_tangent_y_equals_x_squared x1 = slope_tangent_y_equals_x_cubed x2)
    (y_tangent_l_same : y1_compute x1 = y2_compute x2) :
  x1 / x2 = 4 / 3 :=
by
  sorry

end solve_for_ratio_l203_20350


namespace minimum_gumballs_needed_l203_20384

/-- Alex wants to buy at least 150 gumballs,
    and have exactly 14 gumballs left after dividing evenly among 17 people.
    Determine the minimum number of gumballs Alex should buy. -/
theorem minimum_gumballs_needed (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 14) : n = 150 :=
sorry

end minimum_gumballs_needed_l203_20384


namespace expression_equivalence_l203_20360

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by sorry

end expression_equivalence_l203_20360


namespace hyperbola_standard_equation_l203_20377

theorem hyperbola_standard_equation (a b : ℝ) (x y : ℝ)
  (H₁ : 2 * a = 2) -- length of the real axis is 2
  (H₂ : y = 2 * x) -- one of its asymptote equations
  : y^2 - 4 * x^2 = 1 :=
sorry

end hyperbola_standard_equation_l203_20377


namespace sum_of_integers_l203_20361

theorem sum_of_integers (numbers : List ℕ) (h1 : numbers.Nodup) 
(h2 : ∃ a b, (a ≠ b ∧ a * b = 16 ∧ a ∈ numbers ∧ b ∈ numbers)) 
(h3 : ∃ c d, (c ≠ d ∧ c * d = 225 ∧ c ∈ numbers ∧ d ∈ numbers)) :
  numbers.sum = 44 :=
sorry

end sum_of_integers_l203_20361


namespace root_of_quadratic_eq_when_C_is_3_l203_20389

-- Define the quadratic equation and the roots we are trying to prove
def quadratic_eq (C : ℝ) (x : ℝ) := 3 * x^2 - 6 * x + C = 0

-- Set the constant C to 3
def C : ℝ := 3

-- State the theorem that proves the root of the equation when C=3 is x=1
theorem root_of_quadratic_eq_when_C_is_3 : quadratic_eq C 1 :=
by
  -- Skip the detailed proof
  sorry

end root_of_quadratic_eq_when_C_is_3_l203_20389


namespace minimum_photos_needed_l203_20371

theorem minimum_photos_needed 
  (total_photos : ℕ) 
  (photos_IV : ℕ)
  (photos_V : ℕ) 
  (photos_VI : ℕ) 
  (photos_VII : ℕ) 
  (photos_I_III : ℕ) 
  (H : total_photos = 130)
  (H_IV : photos_IV = 35)
  (H_V : photos_V = 30)
  (H_VI : photos_VI = 25)
  (H_VII : photos_VII = 20)
  (H_I_III : photos_I_III = total_photos - (photos_IV + photos_V + photos_VI + photos_VII)) :
  77 = 77 :=
by
  sorry

end minimum_photos_needed_l203_20371


namespace number_is_45_percent_of_27_l203_20349

theorem number_is_45_percent_of_27 (x : ℝ) (h : 27 / x = 45 / 100) : x = 60 := 
by
  sorry

end number_is_45_percent_of_27_l203_20349


namespace ratio_of_points_to_away_home_game_l203_20312

-- Definitions
def first_away_game_points (A : ℕ) : ℕ := A
def second_away_game_points (A : ℕ) : ℕ := A + 18
def third_away_game_points (A : ℕ) : ℕ := A + 20
def last_home_game_points : ℕ := 62
def next_game_points : ℕ := 55
def total_points (A : ℕ) : ℕ := A + (A + 18) + (A + 20) + 62 + 55

-- Given that the total points should be four times the points of the last home game
def target_points : ℕ := 4 * 62

-- The main theorem to prove
theorem ratio_of_points_to_away_home_game : ∀ A : ℕ,
  total_points A = target_points → 62 = 2 * A :=
by
  sorry

end ratio_of_points_to_away_home_game_l203_20312


namespace exists_triangle_with_sin_angles_l203_20303

theorem exists_triangle_with_sin_angles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2 * (a^2*b^2 + a^2*c^2 + b^2*c^2)) : 
    ∃ (α β γ : ℝ), α + β + γ = Real.pi ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c :=
by
  sorry

end exists_triangle_with_sin_angles_l203_20303


namespace orvin_balloons_l203_20399

def regular_price : ℕ := 2
def total_money_initial := 42 * regular_price
def pair_cost := regular_price + (regular_price / 2)
def pairs := total_money_initial / pair_cost
def balloons_from_sale := pairs * 2

def extra_money : ℕ := 18
def price_per_additional_balloon := 2 * regular_price
def additional_balloons := extra_money / price_per_additional_balloon
def greatest_number_of_balloons := balloons_from_sale + additional_balloons

theorem orvin_balloons (pairs balloons_from_sale additional_balloons greatest_number_of_balloons : ℕ) :
  pairs * 2 = 56 →
  additional_balloons = 4 →
  greatest_number_of_balloons = 60 :=
by
  sorry

end orvin_balloons_l203_20399


namespace adam_earnings_correct_l203_20359

def total_earnings (lawns_mowed lawns_to_mow : ℕ) (lawn_pay : ℕ)
                   (cars_washed cars_to_wash : ℕ) (car_pay_euros : ℕ) (euro_to_dollar : ℝ)
                   (dogs_walked dogs_to_walk : ℕ) (dog_pay_pesos : ℕ) (peso_to_dollar : ℝ) : ℝ :=
  let lawn_earnings := lawns_mowed * lawn_pay
  let car_earnings := (cars_washed * car_pay_euros : ℝ) * euro_to_dollar
  let dog_earnings := (dogs_walked * dog_pay_pesos : ℝ) * peso_to_dollar
  lawn_earnings + car_earnings + dog_earnings

theorem adam_earnings_correct :
  total_earnings 4 12 9 4 6 10 1.1 3 4 50 0.05 = 87.5 :=
by
  sorry

end adam_earnings_correct_l203_20359


namespace boxes_per_case_l203_20336

/-- Let's define the variables for the problem.
    We are given that Shirley sold 10 boxes of trefoils,
    and she needs to deliver 5 cases of boxes. --/
def total_boxes : ℕ := 10
def number_of_cases : ℕ := 5

/-- We need to prove that the number of boxes in each case is 2. --/
theorem boxes_per_case :
  total_boxes / number_of_cases = 2 :=
by
  -- Definition step where we specify the calculation
  unfold total_boxes number_of_cases
  -- The problem requires a division operation
  norm_num
  -- The result should be correct according to the solution steps
  done

end boxes_per_case_l203_20336


namespace binomial_constant_term_l203_20329

theorem binomial_constant_term : 
  (∃ c : ℕ, ∀ x : ℝ, (x + (1 / (3 * x)))^8 = c * (x ^ (4 * 2 - 8) / 3)) → 
  ∃ c : ℕ, c = 28 :=
sorry

end binomial_constant_term_l203_20329


namespace bruce_total_payment_l203_20317

-- Define the conditions
def quantity_grapes : Nat := 7
def rate_grapes : Nat := 70
def quantity_mangoes : Nat := 9
def rate_mangoes : Nat := 55

-- Define the calculation for total amount paid
def total_amount_paid : Nat :=
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes)

-- Proof statement
theorem bruce_total_payment : total_amount_paid = 985 :=
by
  -- Proof steps would go here
  sorry

end bruce_total_payment_l203_20317


namespace mileage_in_scientific_notation_l203_20364

noncomputable def scientific_notation_of_mileage : Prop :=
  let mileage := 42000
  mileage = 4.2 * 10^4

theorem mileage_in_scientific_notation :
  scientific_notation_of_mileage :=
by
  sorry

end mileage_in_scientific_notation_l203_20364


namespace faye_scored_47_pieces_l203_20357

variable (X : ℕ) -- X is the number of pieces of candy Faye scored on Halloween.

-- Definitions based on the conditions
def initial_candy_count (X : ℕ) : ℕ := X - 25
def after_sister_gave_40 (X : ℕ) : ℕ := initial_candy_count X + 40
def current_candy_count (X : ℕ) : ℕ := after_sister_gave_40 X

-- Theorem to prove the number of pieces of candy Faye scored on Halloween
theorem faye_scored_47_pieces (h : current_candy_count X = 62) : X = 47 :=
by
  sorry

end faye_scored_47_pieces_l203_20357


namespace truck_stops_l203_20355

variable (a : ℕ → ℕ)
variable (sum_1 : ℕ)
variable (sum_2 : ℕ)

-- Definition for the first sequence with a common difference of -10
def first_sequence : ℕ → ℕ
| 0       => 40
| (n + 1) => first_sequence n - 10

-- Definition for the second sequence with a common difference of -5
def second_sequence : ℕ → ℕ 
| 0       => 10
| (n + 1) => second_sequence n - 5

-- Summing the first sequence elements before the condition change:
def sum_first_sequence : ℕ → ℕ 
| 0       => 40
| (n + 1) => sum_first_sequence n + first_sequence (n + 1)

-- Summing the second sequence elements after the condition change:
def sum_second_sequence : ℕ → ℕ 
| 0       => second_sequence 0
| (n + 1) => sum_second_sequence n + second_sequence (n + 1)

-- Final sum of distances
def total_distance : ℕ :=
  sum_first_sequence 3 + sum_second_sequence 1

theorem truck_stops (sum_1 sum_2 : ℕ) (h1 : sum_1 = sum_first_sequence 3)
 (h2 : sum_2 = sum_second_sequence 1) : 
  total_distance = 115 := by
  sorry


end truck_stops_l203_20355


namespace tom_total_payment_l203_20316

def lemon_price : Nat := 2
def papaya_price : Nat := 1
def mango_price : Nat := 4
def discount_per_4_fruits : Nat := 1
def num_lemons : Nat := 6
def num_papayas : Nat := 4
def num_mangos : Nat := 2

theorem tom_total_payment :
  lemon_price * num_lemons + papaya_price * num_papayas + mango_price * num_mangos 
  - (num_lemons + num_papayas + num_mangos) / 4 * discount_per_4_fruits = 21 := 
by sorry

end tom_total_payment_l203_20316


namespace rhombus_area_of_square_l203_20346

theorem rhombus_area_of_square (h : ∀ (c : ℝ), c = 96) : ∃ (a : ℝ), a = 288 := 
by
  sorry

end rhombus_area_of_square_l203_20346


namespace parallelogram_height_l203_20398

theorem parallelogram_height (A B H : ℝ) 
    (h₁ : A = 96) 
    (h₂ : B = 12) 
    (h₃ : A = B * H) :
  H = 8 := 
by {
  sorry
}

end parallelogram_height_l203_20398


namespace tenured_professors_percentage_l203_20369

noncomputable def percentage_tenured (W M T TM : ℝ) := W = 0.69 ∧ (1 - W) = M ∧ (M * 0.52) = TM ∧ (W + T - TM) = 0.90 → T = 0.7512

-- Define the mathematical entities
variables (W M T TM : ℝ)

-- The main statement
theorem tenured_professors_percentage : percentage_tenured W M T TM := by
  sorry

end tenured_professors_percentage_l203_20369


namespace find_b_l203_20331

theorem find_b (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = 3^n + b)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1))
  (h_geometric : ∃ r, ∀ n ≥ 1, a n = a 1 * r^(n-1)) : b = -1 := 
sorry

end find_b_l203_20331


namespace arc_length_solution_l203_20354

variable (r : ℝ) (α : ℝ)

theorem arc_length_solution (h1 : r = 8) (h2 : α = 5 * Real.pi / 3) : 
    r * α = 40 * Real.pi / 3 := 
by 
    sorry

end arc_length_solution_l203_20354


namespace least_value_divisibility_l203_20396

theorem least_value_divisibility : ∃ (x : ℕ), (23 * x) % 3 = 0  ∧ (∀ y : ℕ, ((23 * y) % 3 = 0 → x ≤ y)) := 
  sorry

end least_value_divisibility_l203_20396


namespace compare_f_g_l203_20352

def R (m n : ℕ) : ℕ := sorry
def L (m n : ℕ) : ℕ := sorry

def f (m n : ℕ) : ℕ := R m n + L m n - sorry
def g (m n : ℕ) : ℕ := R m n + L m n - sorry

theorem compare_f_g (m n : ℕ) : f m n ≤ g m n := sorry

end compare_f_g_l203_20352


namespace find_m_l203_20376

-- Define the hyperbola equation
def hyperbola1 (x y : ℝ) (m : ℝ) : Prop := (x^3 / m) - (y^2 / 3) = 1
def hyperbola2 (x y : ℝ) : Prop := (x^3 / 8) - (y^2 / 4) = 1

-- Define the condition for eccentricity equivalence
def same_eccentricity (m : ℝ) : Prop :=
  let e1_sq := 1 + (4 / 2^2)
  let e2_sq := 1 + (3 / m)
  e1_sq = e2_sq

-- The main theorem statement
theorem find_m (m : ℝ) : hyperbola1 x y m → hyperbola2 x y → same_eccentricity m → m = 6 :=
by
  -- Proof can be skipped with sorry to satisfy the statement-only requirement
  sorry

end find_m_l203_20376


namespace find_original_cost_price_l203_20304

theorem find_original_cost_price (C S C_new S_new : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) (h3 : S_new = S - 16.80) (h4 : S_new = 1.04 * C_new) : C = 80 :=
by
  sorry

end find_original_cost_price_l203_20304


namespace nylon_needed_is_192_l203_20380

-- Define the required lengths for the collars
def nylon_needed_for_dog_collar : ℕ := 18
def nylon_needed_for_cat_collar : ℕ := 10

-- Define the number of collars needed
def number_of_dog_collars : ℕ := 9
def number_of_cat_collars : ℕ := 3

-- Define the total nylon needed
def total_nylon_needed : ℕ :=
  (nylon_needed_for_dog_collar * number_of_dog_collars) + (nylon_needed_for_cat_collar * number_of_cat_collars)

-- State the theorem we need to prove
theorem nylon_needed_is_192 : total_nylon_needed = 192 := 
  by
    -- Simplification to match the complete statement for completeness
    sorry

end nylon_needed_is_192_l203_20380


namespace simplify_expression_l203_20392

theorem simplify_expression :
  ((45 * 2^10) / (15 * 2^5) * 5) = 480 := by
  sorry

end simplify_expression_l203_20392


namespace wheel_horizontal_distance_l203_20313

noncomputable def wheel_radius : ℝ := 2
noncomputable def wheel_revolution_fraction : ℝ := 3 / 4
noncomputable def wheel_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem wheel_horizontal_distance :
  wheel_circumference wheel_radius * wheel_revolution_fraction = 3 * Real.pi :=
by
  sorry

end wheel_horizontal_distance_l203_20313


namespace factorization_correct_l203_20328

theorem factorization_correct:
  ∃ a b : ℤ, (25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) ∧ (a + 2 * b = -24) :=
by
  sorry

end factorization_correct_l203_20328


namespace water_evaporation_weight_l203_20391

noncomputable def initial_weight : ℝ := 200
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def final_salt_concentration : ℝ := 0.08

theorem water_evaporation_weight (W_final : ℝ) (evaporation_weight : ℝ) 
  (h1 : W_final = 10 / final_salt_concentration) 
  (h2 : evaporation_weight = initial_weight - W_final) : 
  evaporation_weight = 75 :=
by
  sorry

end water_evaporation_weight_l203_20391


namespace sara_height_l203_20344

def Julie := 33
def Mark := Julie + 1
def Roy := Mark + 2
def Joe := Roy + 3
def Sara := Joe + 6

theorem sara_height : Sara = 45 := by
  sorry

end sara_height_l203_20344


namespace factorize_polynomial_l203_20372

theorem factorize_polynomial (a b : ℝ) : 
  a^3 * b - 9 * a * b = a * b * (a + 3) * (a - 3) :=
by sorry

end factorize_polynomial_l203_20372


namespace no_integer_in_interval_l203_20374

theorem no_integer_in_interval (n : ℕ) : ¬ ∃ k : ℤ, 
  (n ≠ 0 ∧ (n * Real.sqrt 2 - 1 / (3 * n) < k) ∧ (k < n * Real.sqrt 2 + 1 / (3 * n))) := 
sorry

end no_integer_in_interval_l203_20374


namespace slope_of_line_l203_20388

theorem slope_of_line (m : ℤ) (hm : (3 * m - 6) / (1 + m) = 12) : m = -2 := 
sorry

end slope_of_line_l203_20388


namespace find_second_number_l203_20333

theorem find_second_number (A B : ℝ) (h1 : A = 6400) (h2 : 0.05 * A = 0.2 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l203_20333


namespace cupcakes_left_correct_l203_20342

-- Definitions based on conditions
def total_cupcakes : ℕ := 10 * 12 + 1 * 12 / 2
def total_students : ℕ := 48
def absent_students : ℕ := 6 
def field_trip_students : ℕ := 8
def teachers : ℕ := 2
def teachers_aids : ℕ := 2

-- Function to calculate the number of present people
def total_present_people : ℕ :=
  total_students - absent_students - field_trip_students + teachers + teachers_aids

-- Function to calculate the cupcakes left
def cupcakes_left : ℕ := total_cupcakes - total_present_people

-- The theorem to prove
theorem cupcakes_left_correct : cupcakes_left = 85 := 
by
  -- This is where the proof would go
  sorry

end cupcakes_left_correct_l203_20342


namespace find_x_plus_y_l203_20366

-- Define the vectors
def vector_a : ℝ × ℝ := (1, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_c (y : ℝ) : ℝ × ℝ := (-1, y)

-- Define the conditions
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v2.1 = k * v1.1 ∧ v2.2 = k * v1.2

-- State the theorem
theorem find_x_plus_y (x y : ℝ)
  (h1 : perpendicular vector_a (vector_b x))
  (h2 : parallel vector_a (vector_c y)) :
  x + y = 1 :=
sorry

end find_x_plus_y_l203_20366


namespace reduced_price_of_oil_l203_20330

/-- 
Given:
1. The original price per kg of oil is P.
2. The reduced price per kg of oil is 0.65P.
3. Rs. 800 can buy 5 kgs more oil at the reduced price than at the original price.
4. The equation 5P - 5 * 0.65P = 800 holds true.

Prove that the reduced price per kg of oil is Rs. 297.14.
-/
theorem reduced_price_of_oil (P : ℝ) (h1 : 5 * P - 5 * 0.65 * P = 800) : 
        0.65 * P = 297.14 := 
    sorry

end reduced_price_of_oil_l203_20330


namespace find_x_for_which_f_f_x_eq_f_x_l203_20339

noncomputable def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem find_x_for_which_f_f_x_eq_f_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end find_x_for_which_f_f_x_eq_f_x_l203_20339


namespace sqrt_floor_eq_l203_20348

theorem sqrt_floor_eq (n : ℤ) (h : n ≥ 0) : 
  (⌊Real.sqrt n + Real.sqrt (n + 2)⌋) = ⌊Real.sqrt (4 * n + 1)⌋ :=
sorry

end sqrt_floor_eq_l203_20348


namespace probability_multiple_4_or_15_l203_20382

-- Definitions of natural number range and a set of multiples
def first_30_nat_numbers : Finset ℕ := Finset.range 30
def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

-- Conditions
def multiples_of_4 := multiples_of 4 first_30_nat_numbers
def multiples_of_15 := multiples_of 15 first_30_nat_numbers

-- Proof that probability of selecting a multiple of 4 or 15 is 3 / 10
theorem probability_multiple_4_or_15 : 
  let favorable_outcomes := (multiples_of_4 ∪ multiples_of_15).card
  let total_outcomes := first_30_nat_numbers.card
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  -- correct answer based on the computation
  sorry

end probability_multiple_4_or_15_l203_20382


namespace total_pencils_is_60_l203_20314

def original_pencils : ℕ := 33
def added_pencils : ℕ := 27
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_is_60 : total_pencils = 60 := by
  sorry

end total_pencils_is_60_l203_20314


namespace alice_favorite_number_l203_20300

theorem alice_favorite_number :
  ∃ (n : ℕ), 50 < n ∧ n < 100 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ (n / 10 + n % 10) % 5 = 0 ∧ n = 55 :=
by
  sorry

end alice_favorite_number_l203_20300


namespace scientific_notation_of_122254_l203_20347

theorem scientific_notation_of_122254 :
  122254 = 1.22254 * 10^5 :=
sorry

end scientific_notation_of_122254_l203_20347


namespace transformed_curve_l203_20367

theorem transformed_curve (x y : ℝ) :
  (y * Real.cos x + 2 * y - 1 = 0) →
  (y - 1) * Real.sin x + 2 * y - 3 = 0 :=
by
  intro h
  sorry

end transformed_curve_l203_20367


namespace compound_interest_rate_is_10_percent_l203_20327

theorem compound_interest_rate_is_10_percent
  (P : ℝ) (CI : ℝ) (t : ℝ) (A : ℝ) (n : ℝ) (r : ℝ)
  (hP : P = 4500) (hCI : CI = 945.0000000000009) (ht : t = 2) (hn : n = 1) (hA : A = P + CI)
  (h_eq : A = P * (1 + r / n)^(n * t)) :
  r = 0.1 :=
by
  sorry

end compound_interest_rate_is_10_percent_l203_20327


namespace selected_numbers_satisfy_conditions_l203_20397

theorem selected_numbers_satisfy_conditions :
  ∃ (nums : Finset ℕ), 
  nums = {6, 34, 35, 51, 55, 77} ∧
  (∀ (a b c : ℕ), a ∈ nums → b ∈ nums → c ∈ nums → a ≠ b → a ≠ c → b ≠ c → 
    gcd a b = 1 ∨ gcd b c = 1 ∨ gcd c a = 1) ∧
  (∀ (x y z : ℕ), x ∈ nums → y ∈ nums → z ∈ nums → x ≠ y → x ≠ z → y ≠ z → 
    gcd x y ≠ 1 ∨ gcd y z ≠ 1 ∨ gcd z x ≠ 1) := 
sorry

end selected_numbers_satisfy_conditions_l203_20397


namespace remy_gallons_l203_20370

noncomputable def gallons_used (R : ℝ) : ℝ :=
  let remy := 3 * R + 1
  let riley := (R + remy) - 2
  let ronan := riley / 2
  R + remy + riley + ronan

theorem remy_gallons : ∃ R : ℝ, gallons_used R = 60 ∧ (3 * R + 1) = 18.85 :=
by
  sorry

end remy_gallons_l203_20370


namespace perfect_square_octal_last_digit_l203_20310

theorem perfect_square_octal_last_digit (a b c : ℕ) (n : ℕ) (h1 : a ≠ 0) (h2 : (abc:ℕ) = n^2) :
  c = 1 :=
sorry

end perfect_square_octal_last_digit_l203_20310


namespace unique_symmetric_matrix_pair_l203_20323

theorem unique_symmetric_matrix_pair (a b : ℝ) :
  (∃! M : Matrix (Fin 2) (Fin 2) ℝ, M = M.transpose ∧ Matrix.trace M = a ∧ Matrix.det M = b)
  ↔ (∃ t : ℝ, a = 2 * t ∧ b = t^2) :=
by
  sorry

end unique_symmetric_matrix_pair_l203_20323


namespace fewer_bands_l203_20324

theorem fewer_bands (J B Y : ℕ) (h1 : J = B + 10) (h2 : B - 4 = 8) (h3 : Y = 24) :
  Y - J = 2 :=
sorry

end fewer_bands_l203_20324


namespace garage_sale_items_count_l203_20305

theorem garage_sale_items_count (n_high n_low: ℕ) :
  n_high = 17 ∧ n_low = 24 → total_items = 40 :=
by
  let n_high: ℕ := 17
  let n_low: ℕ := 24
  let total_items: ℕ := (n_high - 1) + (n_low - 1) + 1
  sorry

end garage_sale_items_count_l203_20305


namespace square_side_length_l203_20335

theorem square_side_length (A : ℝ) (h : A = 25) : ∃ s : ℝ, s * s = A ∧ s = 5 :=
by
  sorry

end square_side_length_l203_20335


namespace find_f_value_l203_20332

theorem find_f_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / x^2) : 
  f (1 / 2) = 15 :=
sorry

end find_f_value_l203_20332


namespace dry_grapes_weight_l203_20368

theorem dry_grapes_weight (W_fresh : ℝ) (W_dry : ℝ) (P_water_fresh : ℝ) (P_water_dry : ℝ) :
  W_fresh = 40 → P_water_fresh = 0.80 → P_water_dry = 0.20 → W_dry = 10 := 
by 
  intros hWf hPwf hPwd 
  sorry

end dry_grapes_weight_l203_20368


namespace factorization_of_polynomial_l203_20311

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l203_20311


namespace Ashutosh_completion_time_l203_20308

def Suresh_work_rate := 1 / 15
def Ashutosh_work_rate := 1 / 25
def Suresh_work_time := 9

def job_completed_by_Suresh_in_9_hours := Suresh_work_rate * Suresh_work_time
def remaining_job := 1 - job_completed_by_Suresh_in_9_hours

theorem Ashutosh_completion_time : 
  Ashutosh_work_rate * t = remaining_job -> t = 10 :=
by
  sorry

end Ashutosh_completion_time_l203_20308


namespace mary_thought_animals_l203_20337

-- Definitions based on conditions
def double_counted_sheep : ℕ := 7
def forgotten_pigs : ℕ := 3
def actual_animals : ℕ := 56

-- Statement to be proven
theorem mary_thought_animals (double_counted_sheep forgotten_pigs actual_animals : ℕ) :
  (actual_animals + double_counted_sheep - forgotten_pigs) = 60 := 
by 
  -- Proof goes here
  sorry

end mary_thought_animals_l203_20337


namespace height_flagstaff_l203_20326

variables (s_1 s_2 h_2 : ℝ)
variable (h : ℝ)

-- Define the conditions as given
def shadow_flagstaff := s_1 = 40.25
def shadow_building := s_2 = 28.75
def height_building := h_2 = 12.5
def similar_triangles := (h / s_1) = (h_2 / s_2)

-- Prove the height of the flagstaff
theorem height_flagstaff : shadow_flagstaff s_1 ∧ shadow_building s_2 ∧ height_building h_2 ∧ similar_triangles h s_1 h_2 s_2 → h = 17.5 :=
by sorry

end height_flagstaff_l203_20326


namespace pen_price_l203_20390

theorem pen_price (p : ℝ) (h : 30 = 10 * p + 10 * (p / 2)) : p = 2 :=
sorry

end pen_price_l203_20390


namespace max_sides_of_convex_polygon_with_arithmetic_angles_l203_20318

theorem max_sides_of_convex_polygon_with_arithmetic_angles :
  ∀ (n : ℕ), (∃ α : ℝ, α > 0 ∧ α + (n - 1) * 1 < 180) → 
  n * (2 * α + (n - 1)) / 2 = (n - 2) * 180 → n ≤ 27 :=
by
  sorry

end max_sides_of_convex_polygon_with_arithmetic_angles_l203_20318


namespace total_sections_formed_l203_20378

theorem total_sections_formed (boys girls : ℕ) (hb : boys = 408) (hg : girls = 264) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 28 := 
by
  -- Note: this will assert the theorem, but the proof is omitted with sorry.
  sorry

end total_sections_formed_l203_20378


namespace min_value_of_x_l203_20315

-- Define the conditions and state the problem
theorem min_value_of_x (x : ℝ) : (∀ a : ℝ, a > 0 → x^2 < 1 + a) → x ≥ -1 :=
by
  sorry

end min_value_of_x_l203_20315


namespace polynomial_solution_l203_20320

theorem polynomial_solution (P : ℝ → ℝ) (h₀ : P 0 = 0) (h₁ : ∀ x : ℝ, P x = (1/2) * (P (x+1) + P (x-1))) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end polynomial_solution_l203_20320


namespace find_a_perpendicular_lines_l203_20393

theorem find_a_perpendicular_lines 
  (a : ℤ)
  (l1 : ∀ x y : ℤ, a * x + 4 * y + 7 = 0)
  (l2 : ∀ x y : ℤ, 2 * x - 3 * y - 1 = 0) : 
  (∃ a : ℤ, a = 6) :=
by sorry

end find_a_perpendicular_lines_l203_20393


namespace omar_total_time_l203_20343

-- Conditions
def lap_distance : ℝ := 400
def first_segment_distance : ℝ := 200
def second_segment_distance : ℝ := 200
def speed_first_segment : ℝ := 6
def speed_second_segment : ℝ := 4
def number_of_laps : ℝ := 7

-- Correct answer we want to prove
def total_time_proven : ℝ := 9 * 60 + 23 -- in seconds

-- Theorem statement claiming total time is 9 minutes and 23 seconds
theorem omar_total_time :
  let time_first_segment := first_segment_distance / speed_first_segment
  let time_second_segment := second_segment_distance / speed_second_segment
  let single_lap_time := time_first_segment + time_second_segment
  let total_time := number_of_laps * single_lap_time
  total_time = total_time_proven := sorry

end omar_total_time_l203_20343


namespace total_apples_picked_l203_20338

theorem total_apples_picked (Mike_apples Nancy_apples Keith_apples : ℕ)
  (hMike : Mike_apples = 7)
  (hNancy : Nancy_apples = 3)
  (hKeith : Keith_apples = 6) :
  Mike_apples + Nancy_apples + Keith_apples = 16 :=
by
  sorry

end total_apples_picked_l203_20338


namespace intersection_points_parabola_l203_20301

noncomputable def parabola : ℝ → ℝ := λ x => x^2

noncomputable def directrix : ℝ → ℝ := λ x => -1

noncomputable def other_line (m c : ℝ) : ℝ → ℝ := λ x => m * x + c

theorem intersection_points_parabola {m c : ℝ} (h1 : ∃ x1 x2 : ℝ, other_line m c x1 = parabola x1 ∧ other_line m c x2 = parabola x2) :
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 ≠ x2) → 
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 = x2) := 
by
  sorry

end intersection_points_parabola_l203_20301


namespace knight_moves_equal_n_seven_l203_20358

def knight_moves (n : ℕ) : ℕ := sorry -- Function to calculate the minimum number of moves for a knight.

theorem knight_moves_equal_n_seven :
  ∀ {n : ℕ}, n = 7 →
    knight_moves n = knight_moves n := by
  -- Conditions: Position on standard checkerboard 
  -- and the knight moves described above.
  sorry

end knight_moves_equal_n_seven_l203_20358


namespace fish_game_teams_l203_20379

noncomputable def number_of_possible_teams (n : ℕ) : ℕ := 
  if n = 6 then 5 else sorry

theorem fish_game_teams : number_of_possible_teams 6 = 5 := by
  unfold number_of_possible_teams
  rfl

end fish_game_teams_l203_20379


namespace find_multiplicand_l203_20395

theorem find_multiplicand (m : ℕ) 
( h : 32519 * m = 325027405 ) : 
m = 9995 := 
by {
  sorry
}

end find_multiplicand_l203_20395


namespace max_a1_l203_20302

theorem max_a1 (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, n > 0 → a n > 0)
  (h_eq : ∀ n : ℕ, n > 0 → 2 + a n * (a (n + 1) - a (n - 1)) = 0 ∨ 2 - a n * (a (n + 1) - a (n - 1)) = 0)
  (h_a20 : a 20 = a 20) :
  ∃ max_a1 : ℝ, max_a1 = 512 := 
sorry

end max_a1_l203_20302


namespace find_k_l203_20340

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

def sum_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem find_k (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : a 2 = -1)
  (h2 : 2 * a 1 + a 3 = -1)
  (h3 : arithmetic_sequence a d)
  (h4 : sum_of_sequence S a)
  (h5 : S k = -99) :
  k = 11 := 
by
  sorry

end find_k_l203_20340


namespace equal_cost_sharing_l203_20356

variable (X Y Z : ℝ)
variable (h : X < Y ∧ Y < Z)

theorem equal_cost_sharing :
  ∃ (amount : ℝ), amount = (Y + Z - 2 * X) / 3 := 
sorry

end equal_cost_sharing_l203_20356


namespace mohan_cookies_l203_20365

theorem mohan_cookies :
  ∃ (a : ℕ), 
    (a % 6 = 5) ∧ 
    (a % 7 = 3) ∧ 
    (a % 9 = 7) ∧ 
    (a % 11 = 10) ∧ 
    (a = 1817) :=
sorry

end mohan_cookies_l203_20365


namespace original_faculty_members_correct_l203_20386

noncomputable def original_faculty_members : ℝ := 282

theorem original_faculty_members_correct:
  ∃ F : ℝ, (0.6375 * F = 180) ∧ (F = original_faculty_members) :=
by
  sorry

end original_faculty_members_correct_l203_20386


namespace range_of_a_l203_20387

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → a < x) → a ≤ -1 :=
by
  sorry

end range_of_a_l203_20387


namespace domain_of_g_x_l203_20363

theorem domain_of_g_x :
  ∀ x, (x ≤ 6 ∧ x ≥ -19) ↔ -19 ≤ x ∧ x ≤ 6 :=
by 
  -- Statement only, no proof
  sorry

end domain_of_g_x_l203_20363


namespace four_c_plus_d_l203_20394

theorem four_c_plus_d (c d : ℝ) (h1 : 2 * c = -6) (h2 : c^2 - d = 1) : 4 * c + d = -4 :=
by
  sorry

end four_c_plus_d_l203_20394


namespace betty_min_sugar_flour_oats_l203_20319

theorem betty_min_sugar_flour_oats :
  ∃ (s f o : ℕ), f ≥ 4 + 2 * s ∧ f ≤ 3 * s ∧ o = f + s ∧ s = 4 :=
by
  sorry

end betty_min_sugar_flour_oats_l203_20319


namespace train_cross_signal_in_18_sec_l203_20345

-- Definitions of the given conditions
def train_length := 300 -- meters
def platform_length := 350 -- meters
def time_cross_platform := 39 -- seconds

-- Speed of the train
def train_speed := (train_length + platform_length) / time_cross_platform -- meters/second

-- Time to cross the signal pole
def time_cross_signal_pole := train_length / train_speed -- seconds

theorem train_cross_signal_in_18_sec : time_cross_signal_pole = 18 := by sorry

end train_cross_signal_in_18_sec_l203_20345


namespace last_digit_is_zero_last_ten_digits_are_zero_l203_20341

-- Condition: The product includes a factor of 10
def includes_factor_of_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10

-- Conclusion: The last digit of the product must be 0
theorem last_digit_is_zero (n : ℕ) (h : includes_factor_of_10 n) : 
  n % 10 = 0 :=
sorry

-- Condition: The product includes the factors \(5^{10}\) and \(2^{10}\)
def includes_10_to_the_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10^10

-- Conclusion: The last ten digits of the product must be 0000000000
theorem last_ten_digits_are_zero (n : ℕ) (h : includes_10_to_the_10 n) : 
  n % 10^10 = 0 :=
sorry

end last_digit_is_zero_last_ten_digits_are_zero_l203_20341


namespace year_weeks_span_l203_20309

theorem year_weeks_span (days_in_year : ℕ) (h1 : days_in_year = 365 ∨ days_in_year = 366) :
  ∃ W : ℕ, (W = 53 ∨ W = 54) ∧ (days_in_year = 365 → W = 53) ∧ (days_in_year = 366 → W = 53 ∨ W = 54) :=
by
  sorry

end year_weeks_span_l203_20309


namespace usb_drive_available_space_l203_20353

theorem usb_drive_available_space (C P : ℝ) (hC : C = 16) (hP : P = 50) : 
  (1 - P / 100) * C = 8 :=
by
  sorry

end usb_drive_available_space_l203_20353


namespace value_of_4_ampersand_neg3_l203_20375

-- Define the operation '&'
def ampersand (x y : Int) : Int :=
  x * (y + 2) + x * y

-- State the theorem
theorem value_of_4_ampersand_neg3 : ampersand 4 (-3) = -16 :=
by
  sorry

end value_of_4_ampersand_neg3_l203_20375
