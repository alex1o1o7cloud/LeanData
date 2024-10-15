import Mathlib

namespace NUMINAMATH_GPT_solve_inequality_l1839_183983

theorem solve_inequality (a x : ℝ) :
  (a = 1/2 → (x ≠ 1/2 → (x - a) * (x + a - 1) > 0)) ∧
  (a < 1/2 → ((x > (1 - a) ∨ x < a) → (x - a) * (x + a - 1) > 0)) ∧
  (a > 1/2 → ((x > a ∨ x < (1 - a)) → (x - a) * (x + a - 1) > 0)) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1839_183983


namespace NUMINAMATH_GPT_rate_of_interest_l1839_183993

variable (P SI T R : ℝ)
variable (hP : P = 400)
variable (hSI : SI = 160)
variable (hT : T = 2)

theorem rate_of_interest :
  (SI = (P * R * T) / 100) → R = 20 :=
by
  intro h
  have h1 : P = 400 := hP
  have h2 : SI = 160 := hSI
  have h3 : T = 2 := hT
  sorry

end NUMINAMATH_GPT_rate_of_interest_l1839_183993


namespace NUMINAMATH_GPT_initial_bags_of_rice_l1839_183925

theorem initial_bags_of_rice (sold restocked final initial : Int) 
  (h1 : sold = 23)
  (h2 : restocked = 132)
  (h3 : final = 164) 
  : ((initial - sold) + restocked = final) ↔ initial = 55 :=
by 
  have eq1 : ((initial - sold) + restocked = final) ↔ initial - 23 + 132 = 164 := by rw [h1, h2, h3]
  simp [eq1]
  sorry

end NUMINAMATH_GPT_initial_bags_of_rice_l1839_183925


namespace NUMINAMATH_GPT_consecutive_grouping_probability_l1839_183981

theorem consecutive_grouping_probability :
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements = 1 / 4620 :=
by
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  have h : (block_arrangements * green_factorial * orange_factorial * blue_factorial) = 103680 := sorry
  have h1 : (total_arrangements) = 479001600 := sorry
  calc
    (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements
    _ = 103680 / 479001600 := by rw [h, h1]
    _ = 1 / 4620 := sorry

end NUMINAMATH_GPT_consecutive_grouping_probability_l1839_183981


namespace NUMINAMATH_GPT_find_paintings_l1839_183951

noncomputable def cost_painting (P : ℕ) : ℝ := 40 * P
noncomputable def cost_toy : ℝ := 20 * 8
noncomputable def total_cost (P : ℕ) : ℝ := cost_painting P + cost_toy

noncomputable def sell_painting (P : ℕ) : ℝ := 36 * P
noncomputable def sell_toy : ℝ := 17 * 8
noncomputable def total_sell (P : ℕ) : ℝ := sell_painting P + sell_toy

noncomputable def total_loss (P : ℕ) : ℝ := total_cost P - total_sell P

theorem find_paintings : ∀ (P : ℕ), total_loss P = 64 → P = 10 :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_find_paintings_l1839_183951


namespace NUMINAMATH_GPT_employee_payment_l1839_183998

theorem employee_payment
  (A B C : ℝ)
  (h_total : A + B + C = 1500)
  (h_A : A = 1.5 * B)
  (h_C : C = 0.8 * B) :
  A = 682 ∧ B = 454 ∧ C = 364 := by
  sorry

end NUMINAMATH_GPT_employee_payment_l1839_183998


namespace NUMINAMATH_GPT_max_cubes_submerged_l1839_183956

noncomputable def cylinder_radius (diameter: ℝ) : ℝ := diameter / 2

noncomputable def water_volume (radius height: ℝ) : ℝ := Real.pi * radius^2 * height

noncomputable def cube_volume (edge: ℝ) : ℝ := edge^3

noncomputable def height_of_cubes (edge n: ℝ) : ℝ := edge * n

theorem max_cubes_submerged (diameter height water_height edge: ℝ) 
  (h1: diameter = 2.9)
  (h2: water_height = 4)
  (h3: edge = 2):
  ∃ max_n: ℝ, max_n = 5 := 
  sorry

end NUMINAMATH_GPT_max_cubes_submerged_l1839_183956


namespace NUMINAMATH_GPT_compute_four_at_seven_l1839_183971

def operation (a b : ℤ) : ℤ :=
  5 * a - 2 * b

theorem compute_four_at_seven : operation 4 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_compute_four_at_seven_l1839_183971


namespace NUMINAMATH_GPT_family_vacation_rain_days_l1839_183922

theorem family_vacation_rain_days (r_m r_a : ℕ) 
(h_rain_days : r_m + r_a = 13)
(clear_mornings : r_a = 11)
(clear_afternoons : r_m = 12) : 
r_m + r_a = 23 := 
by 
  sorry

end NUMINAMATH_GPT_family_vacation_rain_days_l1839_183922


namespace NUMINAMATH_GPT_probability_heart_and_face_card_club_l1839_183906

-- Conditions
def num_cards : ℕ := 52
def num_hearts : ℕ := 13
def num_face_card_clubs : ℕ := 3

-- Define the probabilities
def prob_heart_first : ℚ := num_hearts / num_cards
def prob_face_card_club_given_heart : ℚ := num_face_card_clubs / (num_cards - 1)

-- Proof statement
theorem probability_heart_and_face_card_club :
  prob_heart_first * prob_face_card_club_given_heart = 3 / 204 :=
by
  sorry

end NUMINAMATH_GPT_probability_heart_and_face_card_club_l1839_183906


namespace NUMINAMATH_GPT_gain_percentage_l1839_183950

theorem gain_percentage (selling_price gain : ℝ) (h1 : selling_price = 225) (h2 : gain = 75) : 
  (gain / (selling_price - gain) * 100) = 50 :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_l1839_183950


namespace NUMINAMATH_GPT_game_winning_starting_numbers_count_l1839_183966

theorem game_winning_starting_numbers_count : 
  ∃ win_count : ℕ, (win_count = 6) ∧ 
                  ∀ n : ℕ, (1 ≤ n ∧ n < 10) → 
                  (n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9) ↔ 
                  ((∃ m, (2 * n ≤ m ∧ m ≤ 3 * n) ∧ m < 2007)  → 
                   (∃ k, (2 * m ≤ k ∧ k ≤ 3 * m) ∧ k ≥ 2007) = false) := 
sorry

end NUMINAMATH_GPT_game_winning_starting_numbers_count_l1839_183966


namespace NUMINAMATH_GPT_dealer_gross_profit_l1839_183967

theorem dealer_gross_profit (purchase_price : ℝ) (markup_rate : ℝ) (selling_price : ℝ) (gross_profit : ℝ) 
  (purchase_price_cond : purchase_price = 150)
  (markup_rate_cond : markup_rate = 0.25)
  (selling_price_eq : selling_price = purchase_price + (markup_rate * selling_price))
  (gross_profit_eq : gross_profit = selling_price - purchase_price) : 
  gross_profit = 50 :=
by
  sorry

end NUMINAMATH_GPT_dealer_gross_profit_l1839_183967


namespace NUMINAMATH_GPT_find_square_l1839_183940

theorem find_square (y : ℝ) (h : (y + 5)^(1/3) = 3) : (y + 5)^2 = 729 := 
sorry

end NUMINAMATH_GPT_find_square_l1839_183940


namespace NUMINAMATH_GPT_two_candidates_solve_all_problems_l1839_183901

-- Definitions for the conditions and problem context
def candidates : Nat := 200
def problems : Nat := 6 
def solved_by (p : Nat) : Nat := 120 -- at least 120 participants solve each problem.

-- The main theorem representing the proof problem
theorem two_candidates_solve_all_problems :
  (∃ c1 c2 : Fin candidates, ∀ p : Fin problems, (solved_by p ≥ 120)) :=
by
  sorry

end NUMINAMATH_GPT_two_candidates_solve_all_problems_l1839_183901


namespace NUMINAMATH_GPT_number_of_soccer_balls_in_first_set_l1839_183972

noncomputable def cost_of_soccer_ball : ℕ := 50
noncomputable def first_cost_condition (F c : ℕ) : Prop := 3 * F + c = 155
noncomputable def second_cost_condition (F : ℕ) : Prop := 2 * F + 3 * cost_of_soccer_ball = 220

theorem number_of_soccer_balls_in_first_set (F : ℕ) :
  (first_cost_condition F 50) ∧ (second_cost_condition F) → 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_soccer_balls_in_first_set_l1839_183972


namespace NUMINAMATH_GPT_range_of_a_increasing_function_l1839_183987

noncomputable def f (x a : ℝ) := x^3 + a * x + 1 / x

noncomputable def f' (x a : ℝ) := 3 * x^2 - 1 / x^2 + a

theorem range_of_a_increasing_function (a : ℝ) :
  (∀ x : ℝ, x > 1/2 → f' x a ≥ 0) ↔ a ≥ 13 / 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_increasing_function_l1839_183987


namespace NUMINAMATH_GPT_janet_overtime_multiple_l1839_183918

theorem janet_overtime_multiple :
  let hourly_rate := 20
  let weekly_hours := 52
  let regular_hours := 40
  let car_price := 4640
  let weeks_needed := 4
  let normal_weekly_earning := regular_hours * hourly_rate
  let overtime_hours := weekly_hours - regular_hours
  let required_weekly_earning := car_price / weeks_needed
  let overtime_weekly_earning := required_weekly_earning - normal_weekly_earning
  let overtime_rate := overtime_weekly_earning / overtime_hours
  (overtime_rate / hourly_rate = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_janet_overtime_multiple_l1839_183918


namespace NUMINAMATH_GPT_equation_root_a_plus_b_l1839_183965

theorem equation_root_a_plus_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b ≥ 0) 
(h_root : (∃ x : ℝ, x > 0 ∧ x^3 - x^2 + 18 * x - 320 = 0 ∧ x = Real.sqrt a - ↑b)) : 
a + b = 25 := by
  sorry

end NUMINAMATH_GPT_equation_root_a_plus_b_l1839_183965


namespace NUMINAMATH_GPT_inequality_proof_l1839_183912

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + x + 2 * x^2) * (2 + 3 * y + y^2) * (4 + z + z^2) ≥ 60 * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1839_183912


namespace NUMINAMATH_GPT_find_a_b_of_solution_set_l1839_183937

theorem find_a_b_of_solution_set :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + (a + 1) * x + a * b = 0 ↔ x = -1 ∨ x = 4) → a + b = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_of_solution_set_l1839_183937


namespace NUMINAMATH_GPT_cultural_festival_recommendation_schemes_l1839_183913

theorem cultural_festival_recommendation_schemes :
  (∃ (females : Finset ℕ) (males : Finset ℕ),
    females.card = 3 ∧ males.card = 2 ∧
    ∃ (dance : Finset ℕ) (singing : Finset ℕ) (instruments : Finset ℕ),
      dance.card = 2 ∧ dance ⊆ females ∧
      singing.card = 2 ∧ singing ∩ females ≠ ∅ ∧
      instruments.card = 1 ∧ instruments ⊆ males ∧
      (females ∪ males).card = 5) → 
  ∃ (recommendation_schemes : ℕ), recommendation_schemes = 18 :=
by
  sorry

end NUMINAMATH_GPT_cultural_festival_recommendation_schemes_l1839_183913


namespace NUMINAMATH_GPT_scientific_notation_of_300_million_l1839_183920

theorem scientific_notation_of_300_million : 
  300000000 = 3 * 10^8 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_300_million_l1839_183920


namespace NUMINAMATH_GPT_agreed_upon_service_period_l1839_183942

theorem agreed_upon_service_period (x : ℕ) (hx : 900 + 100 = 1000) 
(assumed_service : x * 1000 = 9 * (650 + 100)) :
  x = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_agreed_upon_service_period_l1839_183942


namespace NUMINAMATH_GPT_average_candies_correct_l1839_183926

def candy_counts : List ℕ := [16, 22, 30, 26, 18, 20]
def num_members : ℕ := 6
def total_candies : ℕ := List.sum candy_counts
def average_candies : ℕ := total_candies / num_members

theorem average_candies_correct : average_candies = 22 := by
  -- Proof is omitted, as per instructions
  sorry

end NUMINAMATH_GPT_average_candies_correct_l1839_183926


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_x_l1839_183902

noncomputable def sum_of_roots_of_equation : ℚ :=
  let eq : Polynomial ℚ := 4 * Polynomial.X ^ 2 + 3 * Polynomial.X - 5
  let roots := eq.roots
  roots.sum

theorem sum_of_all_possible_values_of_x :
  sum_of_roots_of_equation = -3/4 := 
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_x_l1839_183902


namespace NUMINAMATH_GPT_gcd_poly_l1839_183947

-- Defining the conditions as stated in part a:
def is_even_multiple_of_1171 (b : ℤ) : Prop :=
  ∃ k : ℤ, b = 1171 * k * 2

-- Stating the main theorem based on the conditions and required proof in part c:
theorem gcd_poly (b : ℤ) (h : is_even_multiple_of_1171 b) : Int.gcd (3 * b ^ 2 + 47 * b + 79) (b + 17) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_poly_l1839_183947


namespace NUMINAMATH_GPT_find_sum_of_squares_of_roots_l1839_183916

theorem find_sum_of_squares_of_roots (a b c : ℝ) (h_ab : a < b) (h_bc : b < c)
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 2 * x^2 - 3 * x + 4)
  (h_eq : f a = f b ∧ f b = f c) :
  a^2 + b^2 + c^2 = 10 :=
sorry

end NUMINAMATH_GPT_find_sum_of_squares_of_roots_l1839_183916


namespace NUMINAMATH_GPT_battery_current_l1839_183932

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end NUMINAMATH_GPT_battery_current_l1839_183932


namespace NUMINAMATH_GPT_exists_zero_in_interval_l1839_183989

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem exists_zero_in_interval : ∃ c ∈ Set.Ioo 0 (1/2 : ℝ), f c = 0 := by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_exists_zero_in_interval_l1839_183989


namespace NUMINAMATH_GPT_new_area_after_increasing_length_and_width_l1839_183979

theorem new_area_after_increasing_length_and_width
  (L W : ℝ)
  (hA : L * W = 450)
  (hL' : 1.2 * L = L')
  (hW' : 1.3 * W = W') :
  (1.2 * L) * (1.3 * W) = 702 :=
by sorry

end NUMINAMATH_GPT_new_area_after_increasing_length_and_width_l1839_183979


namespace NUMINAMATH_GPT_question1_question2_l1839_183954

namespace MathProofs

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

-- Definitions based on conditions
def isA := ∀ x, A x ↔ (-3 < x ∧ x < 2)
def isB := ∀ x, B x ↔ (Real.exp (x - 1) ≥ 1)
def isCuA := ∀ x, (U \ A) x ↔ (x ≤ -3 ∨ x ≥ 2)

-- Proof of Question 1
theorem question1 : (∀ x, (A ∪ B) x ↔ (x > -3)) := by
  sorry

-- Proof of Question 2
theorem question2 : (∀ x, ((U \ A) ∩ B) x ↔ (x ≥ 2)) := by
  sorry

end MathProofs

end NUMINAMATH_GPT_question1_question2_l1839_183954


namespace NUMINAMATH_GPT_circle_center_coordinates_l1839_183955

-- Definition of the circle's equation
def circle_eq : Prop := ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 3

-- Proof of the circle's center coordinates
theorem circle_center_coordinates : ∃ h k : ℝ, (h, k) = (2, -1) := 
sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1839_183955


namespace NUMINAMATH_GPT_Linda_sold_7_tees_l1839_183961

variables (T : ℕ)
variables (jeans_price tees_price total_money_from_jeans total_money total_money_from_tees : ℕ)
variables (jeans_sold : ℕ)

def tees_sold :=
  jeans_price = 11 ∧ tees_price = 8 ∧ jeans_sold = 4 ∧
  total_money = 100 ∧ total_money_from_jeans = jeans_sold * jeans_price ∧
  total_money_from_tees = total_money - total_money_from_jeans ∧
  T = total_money_from_tees / tees_price
  
theorem Linda_sold_7_tees (h : tees_sold T jeans_price tees_price total_money_from_jeans total_money total_money_from_tees jeans_sold) : T = 7 :=
by
  sorry

end NUMINAMATH_GPT_Linda_sold_7_tees_l1839_183961


namespace NUMINAMATH_GPT_find_a_and_b_l1839_183963

open Function

theorem find_a_and_b (a b : ℚ) (k : ℚ)  (hA : (6 : ℚ) = k * (-3))
    (hB : (a : ℚ) = k * 2)
    (hC : (-1 : ℚ) = k * b) : 
    a = -4 ∧ b = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1839_183963


namespace NUMINAMATH_GPT_geometric_sequence_property_l1839_183970

open Classical

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property :
  ∃ (a : ℕ → ℝ) (q : ℝ), q < 0 ∧ geometric_sequence a q ∧
    a 1 = 1 - a 0 ∧ a 3 = 4 - a 2 ∧ a 3 + a 4 = -8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_property_l1839_183970


namespace NUMINAMATH_GPT_pool_houses_count_l1839_183973

-- Definitions based on conditions
def total_houses : ℕ := 65
def num_garage : ℕ := 50
def num_both : ℕ := 35
def num_neither : ℕ := 10
def num_pool : ℕ := total_houses - num_garage - num_neither + num_both

theorem pool_houses_count :
  num_pool = 40 := by
  -- Simplified form of the problem expressed in Lean 4 theorem statement.
  sorry

end NUMINAMATH_GPT_pool_houses_count_l1839_183973


namespace NUMINAMATH_GPT_positive_difference_eq_250_l1839_183930

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end NUMINAMATH_GPT_positive_difference_eq_250_l1839_183930


namespace NUMINAMATH_GPT_minimum_students_for_200_candies_l1839_183982

theorem minimum_students_for_200_candies (candies : ℕ) (students : ℕ) (h_candies : candies = 200) : students = 21 :=
by
  sorry

end NUMINAMATH_GPT_minimum_students_for_200_candies_l1839_183982


namespace NUMINAMATH_GPT_geom_series_sum_l1839_183910

def geom_sum (b1 : ℚ) (r : ℚ) (n : ℕ) : ℚ := 
  b1 * (1 - r^n) / (1 - r)

def b1 : ℚ := 3 / 4
def r : ℚ := 3 / 4
def n : ℕ := 15

theorem geom_series_sum :
  geom_sum b1 r n = 3177884751 / 1073741824 :=
by sorry

end NUMINAMATH_GPT_geom_series_sum_l1839_183910


namespace NUMINAMATH_GPT_both_students_given_correct_l1839_183943

open ProbabilityTheory

variables (P_A P_B : ℝ)

-- Define the conditions from part a)
def student_a_correct := P_A = 3 / 5
def student_b_correct := P_B = 1 / 3

-- Define the event that both students correctly answer
def both_students_correct := P_A * P_B

-- Define the event that the question is answered correctly
def question_answered_correctly := (P_A * (1 - P_B)) + ((1 - P_A) * P_B) + (P_A * P_B)

-- Define the conditional probability we need to prove
theorem both_students_given_correct (hA : student_a_correct P_A) (hB : student_b_correct P_B) :
  both_students_correct P_A P_B / question_answered_correctly P_A P_B = 3 / 11 := 
sorry

end NUMINAMATH_GPT_both_students_given_correct_l1839_183943


namespace NUMINAMATH_GPT_num_ducks_l1839_183923

variable (D G : ℕ)

theorem num_ducks (h1 : D + G = 8) (h2 : 2 * D + 4 * G = 24) : D = 4 := by
  sorry

end NUMINAMATH_GPT_num_ducks_l1839_183923


namespace NUMINAMATH_GPT_ajax_weight_after_two_weeks_l1839_183939

/-- Initial weight of Ajax in kilograms. -/
def initial_weight_kg : ℝ := 80

/-- Conversion factor from kilograms to pounds. -/
def kg_to_pounds : ℝ := 2.2

/-- Weight lost per hour of each exercise type. -/
def high_intensity_loss_per_hour : ℝ := 4
def moderate_intensity_loss_per_hour : ℝ := 2.5
def low_intensity_loss_per_hour : ℝ := 1.5

/-- Ajax's weekly exercise routine. -/
def weekly_high_intensity_hours : ℝ := 1 * 3 + 1.5 * 1
def weekly_moderate_intensity_hours : ℝ := 0.5 * 5
def weekly_low_intensity_hours : ℝ := 1 * 2 + 0.5 * 1

/-- Calculate the total weight loss in pounds per week. -/
def total_weekly_weight_loss_pounds : ℝ :=
  weekly_high_intensity_hours * high_intensity_loss_per_hour +
  weekly_moderate_intensity_hours * moderate_intensity_loss_per_hour +
  weekly_low_intensity_hours * low_intensity_loss_per_hour

/-- Calculate the total weight loss in pounds for two weeks. -/
def total_weight_loss_pounds_for_two_weeks : ℝ :=
  total_weekly_weight_loss_pounds * 2

/-- Calculate Ajax's initial weight in pounds. -/
def initial_weight_pounds : ℝ :=
  initial_weight_kg * kg_to_pounds

/-- Calculate Ajax's new weight after two weeks. -/
def new_weight_pounds : ℝ :=
  initial_weight_pounds - total_weight_loss_pounds_for_two_weeks

/-- Prove that Ajax's new weight in pounds is 120 after following the workout schedule for two weeks. -/
theorem ajax_weight_after_two_weeks :
  new_weight_pounds = 120 :=
by
  sorry

end NUMINAMATH_GPT_ajax_weight_after_two_weeks_l1839_183939


namespace NUMINAMATH_GPT_roots_of_quadratic_l1839_183994

theorem roots_of_quadratic :
  ∃ m n : ℝ, (∀ x : ℝ, x^2 - 4 * x - 1 = 0 → (x = m ∨ x = n)) ∧
            (m + n = 4) ∧
            (m * n = -1) ∧
            (m + n - m * n = 5) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1839_183994


namespace NUMINAMATH_GPT_totalPeoplePresent_l1839_183911

-- Defining the constants based on the problem conditions
def associateProfessors := 2
def assistantProfessors := 7

def totalPencils := 11
def totalCharts := 16

-- The main proof statement
theorem totalPeoplePresent :
  (∃ (A B : ℕ), (2 * A + B = totalPencils) ∧ (A + 2 * B = totalCharts)) →
  (associateProfessors + assistantProfessors = 9) :=
  by
  sorry

end NUMINAMATH_GPT_totalPeoplePresent_l1839_183911


namespace NUMINAMATH_GPT_tenth_term_ar_sequence_l1839_183931

-- Variables for the first term and common difference
variables (a1 d : ℕ) (n : ℕ)

-- Specific given values
def a1_fixed := 3
def d_fixed := 2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) := a1 + (n - 1) * d

-- The statement to prove
theorem tenth_term_ar_sequence : a_n 10 = 21 := by
  -- Definitions for a1 and d
  let a1 := a1_fixed
  let d := d_fixed
  -- The rest of the proof
  sorry

end NUMINAMATH_GPT_tenth_term_ar_sequence_l1839_183931


namespace NUMINAMATH_GPT_question_l1839_183900

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end NUMINAMATH_GPT_question_l1839_183900


namespace NUMINAMATH_GPT_find_divisor_l1839_183964

def positive_integer := {e : ℕ // e > 0}

theorem find_divisor (d : ℕ) :
  (∃ e : positive_integer, (e.val % 13 = 2)) →
  (∃ n : ℕ, n < 180 ∧ n % d = 5 ∧ ∀ m < 180, m % d = 5 → m = n) →
  d = 175 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1839_183964


namespace NUMINAMATH_GPT_complex_expression_value_l1839_183907

theorem complex_expression_value {i : ℂ} (h : i^2 = -1) : i^3 * (1 + i)^2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_complex_expression_value_l1839_183907


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1839_183992

-- Define the main problem conditions
variables (a b : ℝ)

-- State the problem in Lean
theorem value_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = 3) (h3 : |a - b| = - (a - b)) :
  a + b = 5 ∨ a + b = 1 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1839_183992


namespace NUMINAMATH_GPT_largest_k_inequality_l1839_183949

noncomputable def k : ℚ := 39 / 2

theorem largest_k_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b + c)^3 ≥ (5 / 2) * (a^3 + b^3 + c^3) + k * a * b * c := 
sorry

end NUMINAMATH_GPT_largest_k_inequality_l1839_183949


namespace NUMINAMATH_GPT_find_ab_plus_a_plus_b_l1839_183935

-- Define the polynomial
def quartic_poly (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 - 6*x - 1

-- Define the roots conditions
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

-- State the proof problem
theorem find_ab_plus_a_plus_b :
  ∃ a b : ℝ,
    is_root quartic_poly a ∧
    is_root quartic_poly b ∧
    ab = a * b ∧
    a_plus_b = a + b ∧
    ab + a_plus_b = 4 :=
by sorry

end NUMINAMATH_GPT_find_ab_plus_a_plus_b_l1839_183935


namespace NUMINAMATH_GPT_product_of_two_numbers_l1839_183978

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x + y = 8 * (x - y)) 
  (h3 : x * y = 40 * (x - y)) 
  : x * y = 63 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1839_183978


namespace NUMINAMATH_GPT_problem_solution_l1839_183959

variable (a : ℝ)

theorem problem_solution (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1839_183959


namespace NUMINAMATH_GPT_Liza_rent_l1839_183984

theorem Liza_rent :
  (800 - R + 1500 - 117 - 100 - 70 = 1563) -> R = 450 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_Liza_rent_l1839_183984


namespace NUMINAMATH_GPT_motorcycles_count_l1839_183996

/-- 
Prove that the number of motorcycles in the parking lot is 28 given the conditions:
1. Each car has 5 wheels (including one spare).
2. Each motorcycle has 2 wheels.
3. Each tricycle has 3 wheels.
4. There are 19 cars in the parking lot.
5. There are 11 tricycles in the parking lot.
6. Altogether all vehicles have 184 wheels.
-/
theorem motorcycles_count 
  (cars := 19) 
  (tricycles := 11) 
  (total_wheels := 184) 
  (wheels_per_car := 5) 
  (wheels_per_tricycle := 3) 
  (wheels_per_motorcycle := 2) :
  (184 - (19 * 5 + 11 * 3)) / 2 = 28 :=
by 
  sorry

end NUMINAMATH_GPT_motorcycles_count_l1839_183996


namespace NUMINAMATH_GPT_find_a_l1839_183936

-- Define the conditions for the lines l1 and l2
def line1 (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def line2 (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - (3/2) = 0

-- Define the condition for parallel lines
def parallel (a : ℝ) : Prop := a^2 - 4 * ((3/4) * a + 1) = 0 ∧ 4 * (-3/2) - 6 * a ≠ 0

-- Define the condition for perpendicular lines
def perpendicular (a : ℝ) : Prop := a * ((3/4) * a + 1) + 4 * a = 0

-- The theorem to prove values of a for which l1 is parallel or perpendicular to l2
theorem find_a (a : ℝ) :
  (parallel a → a = 4) ∧ (perpendicular a → a = 0 ∨ a = -20/3) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1839_183936


namespace NUMINAMATH_GPT_problem_statement_l1839_183968

theorem problem_statement (a b c : ℤ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 3) 
  (h3 : |c| = 6) 
  (h4 : |a + b| = - (a + b)) 
  (h5 : |a + c| = a + c) : 
  a - b + c = -2 ∨ a - b + c = 4 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1839_183968


namespace NUMINAMATH_GPT_age_of_john_l1839_183921

theorem age_of_john (J S : ℕ) 
  (h1 : S = 2 * J)
  (h2 : S + (50 - J) = 60) :
  J = 10 :=
sorry

end NUMINAMATH_GPT_age_of_john_l1839_183921


namespace NUMINAMATH_GPT_toms_final_stamp_count_l1839_183944

-- Definitions of the given conditions

def initial_stamps : ℕ := 3000
def mike_gift : ℕ := 17
def harry_gift : ℕ := 2 * mike_gift + 10
def sarah_gift : ℕ := 3 * mike_gift - 5
def damaged_stamps : ℕ := 37

-- Statement of the goal
theorem toms_final_stamp_count :
  initial_stamps + mike_gift + harry_gift + sarah_gift - damaged_stamps = 3070 :=
by
  sorry

end NUMINAMATH_GPT_toms_final_stamp_count_l1839_183944


namespace NUMINAMATH_GPT_remainder_of_sum_l1839_183985

theorem remainder_of_sum (a b : ℤ) (k m : ℤ)
  (h1 : a = 84 * k + 78)
  (h2 : b = 120 * m + 114) :
  (a + b) % 42 = 24 :=
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l1839_183985


namespace NUMINAMATH_GPT_part_one_solution_set_part_two_m_range_l1839_183917

theorem part_one_solution_set (m : ℝ) (x : ℝ) (h : m = 0) : ((m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

theorem part_two_m_range (m : ℝ) : (∀ x : ℝ, (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
by
  sorry

end NUMINAMATH_GPT_part_one_solution_set_part_two_m_range_l1839_183917


namespace NUMINAMATH_GPT_sulfuric_acid_moles_l1839_183953

-- Definitions based on the conditions
def iron_moles := 2
def hydrogen_moles := 2

-- The reaction equation in the problem
def reaction (Fe H₂SO₄ : ℕ) : Prop :=
  Fe + H₂SO₄ = hydrogen_moles

-- Goal: prove the number of moles of sulfuric acid used is 2
theorem sulfuric_acid_moles (Fe : ℕ) (H₂SO₄ : ℕ) (h : reaction Fe H₂SO₄) :
  H₂SO₄ = 2 :=
sorry

end NUMINAMATH_GPT_sulfuric_acid_moles_l1839_183953


namespace NUMINAMATH_GPT_shuttle_speed_l1839_183969

theorem shuttle_speed (v : ℕ) (h : v = 9) : v * 3600 = 32400 :=
by
  sorry

end NUMINAMATH_GPT_shuttle_speed_l1839_183969


namespace NUMINAMATH_GPT_marks_fathers_gift_l1839_183927

noncomputable def total_spent (books : ℕ) (cost_per_book : ℕ) : ℕ :=
  books * cost_per_book

noncomputable def total_money_given (spent : ℕ) (left_over : ℕ) : ℕ :=
  spent + left_over

theorem marks_fathers_gift :
  total_money_given (total_spent 10 5) 35 = 85 := by
  sorry

end NUMINAMATH_GPT_marks_fathers_gift_l1839_183927


namespace NUMINAMATH_GPT_checkerboard_corner_sum_is_164_l1839_183980

def checkerboard_sum_corners : ℕ :=
  let top_left := 1
  let top_right := 9
  let bottom_left := 73
  let bottom_right := 81
  top_left + top_right + bottom_left + bottom_right

theorem checkerboard_corner_sum_is_164 :
  checkerboard_sum_corners = 164 :=
by
  sorry

end NUMINAMATH_GPT_checkerboard_corner_sum_is_164_l1839_183980


namespace NUMINAMATH_GPT_greatest_three_digit_number_l1839_183924

theorem greatest_three_digit_number 
  (n : ℕ)
  (h1 : n % 7 = 2)
  (h2 : n % 6 = 4)
  (h3 : n ≥ 100)
  (h4 : n < 1000) :
  n = 994 :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_number_l1839_183924


namespace NUMINAMATH_GPT_existence_of_same_remainder_mod_36_l1839_183990

theorem existence_of_same_remainder_mod_36
  (a : Fin 7 → ℕ) :
  ∃ (i j k l : Fin 7), i < j ∧ k < l ∧ (a i)^2 + (a j)^2 % 36 = (a k)^2 + (a l)^2 % 36 := by
  sorry

end NUMINAMATH_GPT_existence_of_same_remainder_mod_36_l1839_183990


namespace NUMINAMATH_GPT_solve_for_y_l1839_183929

def diamond (a b : ℕ) : ℕ := 2 * a + b

theorem solve_for_y (y : ℕ) (h : diamond 4 (diamond 3 y) = 17) : y = 3 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l1839_183929


namespace NUMINAMATH_GPT_average_value_l1839_183908

variable (z : ℝ)

theorem average_value : (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 :=
by
  sorry

end NUMINAMATH_GPT_average_value_l1839_183908


namespace NUMINAMATH_GPT_oli_scoops_l1839_183905

theorem oli_scoops : ∃ x : ℤ, ∀ y : ℤ, y = 2 * x ∧ y = x + 4 → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_oli_scoops_l1839_183905


namespace NUMINAMATH_GPT_pet_store_problem_l1839_183960

theorem pet_store_problem 
  (initial_puppies : ℕ) 
  (sold_day1 : ℕ) 
  (sold_day2 : ℕ) 
  (sold_day3 : ℕ) 
  (sold_day4 : ℕ)
  (sold_day5 : ℕ) 
  (puppies_per_cage : ℕ)
  (initial_puppies_eq : initial_puppies = 120) 
  (sold_day1_eq : sold_day1 = 25) 
  (sold_day2_eq : sold_day2 = 10) 
  (sold_day3_eq : sold_day3 = 30) 
  (sold_day4_eq : sold_day4 = 15) 
  (sold_day5_eq : sold_day5 = 28) 
  (puppies_per_cage_eq : puppies_per_cage = 6) : 
  (initial_puppies - (sold_day1 + sold_day2 + sold_day3 + sold_day4 + sold_day5)) / puppies_per_cage = 2 := 
by 
  sorry

end NUMINAMATH_GPT_pet_store_problem_l1839_183960


namespace NUMINAMATH_GPT_value_of_m_l1839_183952

theorem value_of_m (m : ℤ) : 
  (∃ f : ℤ → ℤ, ∀ x : ℤ, x^2 - (m+1)*x + 1 = (f x)^2) → (m = 1 ∨ m = -3) := 
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1839_183952


namespace NUMINAMATH_GPT_chickens_and_rabbits_l1839_183928

theorem chickens_and_rabbits (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end NUMINAMATH_GPT_chickens_and_rabbits_l1839_183928


namespace NUMINAMATH_GPT_largest_value_after_2001_presses_l1839_183975

noncomputable def max_value_after_presses (n : ℕ) : ℝ :=
if n = 0 then 1 else sorry -- Placeholder for the actual function definition

theorem largest_value_after_2001_presses :
  max_value_after_presses 2001 = 1 :=
sorry

end NUMINAMATH_GPT_largest_value_after_2001_presses_l1839_183975


namespace NUMINAMATH_GPT_tractor_planting_rate_l1839_183946

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end NUMINAMATH_GPT_tractor_planting_rate_l1839_183946


namespace NUMINAMATH_GPT_sum_of_squares_of_coeffs_l1839_183948

theorem sum_of_squares_of_coeffs :
  let p := 3 * (x^5 + 5 * x^3 + 2 * x + 1)
  let coeffs := [3, 15, 6, 3]
  coeffs.map (λ c => c^2) |>.sum = 279 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coeffs_l1839_183948


namespace NUMINAMATH_GPT_exists_base_for_1994_no_base_for_1993_l1839_183915

-- Problem 1: Existence of a base for 1994 with identical digits
theorem exists_base_for_1994 :
  ∃ b : ℕ, 1 < b ∧ b < 1993 ∧ (∃ a : ℕ, ∀ n : ℕ, 1994 = a * ((b ^ n - 1) / (b - 1)) ∧ a = 2) :=
sorry

-- Problem 2: Non-existence of a base for 1993 with identical digits
theorem no_base_for_1993 :
  ¬∃ b : ℕ, 1 < b ∧ b < 1992 ∧ (∃ a : ℕ, ∀ n : ℕ, 1993 = a * ((b ^ n - 1) / (b - 1))) :=
sorry

end NUMINAMATH_GPT_exists_base_for_1994_no_base_for_1993_l1839_183915


namespace NUMINAMATH_GPT_complement_intersection_l1839_183941

universe u

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

def complement_U (A : Finset Int) (U : Finset Int) : Finset Int :=
  U.filter (λ x => x ∉ A)

theorem complement_intersection :
  (complement_U A U) ∩ B = {-3, 0} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1839_183941


namespace NUMINAMATH_GPT_medicine_supply_duration_l1839_183962

theorem medicine_supply_duration
  (pills_per_three_days : ℚ := 1 / 3)
  (total_pills : ℕ := 60)
  (days_per_month : ℕ := 30) :
  (((total_pills : ℚ) * ( 3 / pills_per_three_days)) / days_per_month) = 18 := sorry

end NUMINAMATH_GPT_medicine_supply_duration_l1839_183962


namespace NUMINAMATH_GPT_choose_four_socks_from_seven_l1839_183995

theorem choose_four_socks_from_seven : (Nat.choose 7 4) = 35 :=
by
  sorry

end NUMINAMATH_GPT_choose_four_socks_from_seven_l1839_183995


namespace NUMINAMATH_GPT_find_b_l1839_183909

-- Definitions from the conditions
variables (a b : ℝ)

-- Theorem statement using the conditions and the correct answer
theorem find_b (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1839_183909


namespace NUMINAMATH_GPT_batsman_new_average_l1839_183997

variable (A : ℝ) -- Assume that A is the average before the 17th inning
variable (score : ℝ) -- The score in the 17th inning
variable (new_average : ℝ) -- The new average after the 17th inning

-- The conditions
axiom H1 : score = 85
axiom H2 : new_average = A + 3

-- The statement to prove
theorem batsman_new_average : 
    new_average = 37 :=
by 
  sorry

end NUMINAMATH_GPT_batsman_new_average_l1839_183997


namespace NUMINAMATH_GPT_segments_after_cuts_l1839_183976

-- Definitions from the conditions
def cuts : ℕ := 10

-- Mathematically equivalent proof statement
theorem segments_after_cuts : (cuts + 1 = 11) :=
by sorry

end NUMINAMATH_GPT_segments_after_cuts_l1839_183976


namespace NUMINAMATH_GPT_lcm_5_6_10_15_l1839_183904

theorem lcm_5_6_10_15 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 10 15) = 30 := 
by
  sorry

end NUMINAMATH_GPT_lcm_5_6_10_15_l1839_183904


namespace NUMINAMATH_GPT_vectors_projection_l1839_183945

noncomputable def p := (⟨-44 / 53, 154 / 53⟩ : ℝ × ℝ)

theorem vectors_projection :
  let u := (⟨-4, 2⟩ : ℝ × ℝ)
  let v := (⟨3, 4⟩ : ℝ × ℝ)
  let w := (⟨7, 2⟩ : ℝ × ℝ)
  (⟨(7 * (24 / 53)) - 4, (2 * (24 / 53)) + 2⟩ : ℝ × ℝ) = p :=
by {
  -- proof skipped
  sorry
}

end NUMINAMATH_GPT_vectors_projection_l1839_183945


namespace NUMINAMATH_GPT_solution_eq1_solution_eq2_l1839_183991

-- Definitions corresponding to the conditions of the problem.
def eq1 (x : ℝ) : Prop := 16 * x^2 = 49
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 64

-- Statements for the proof problem.
theorem solution_eq1 (x : ℝ) : eq1 x → (x = 7 / 4 ∨ x = - (7 / 4)) :=
by
  intro h
  sorry

theorem solution_eq2 (x : ℝ) : eq2 x → (x = 10 ∨ x = -6) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_eq1_solution_eq2_l1839_183991


namespace NUMINAMATH_GPT_sin_cos_value_sin_minus_cos_value_tan_value_l1839_183934

variable (x : ℝ)

theorem sin_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x * Real.cos x = - 12 / 25 := 
sorry

theorem sin_minus_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x - Real.cos x = - 7 / 5 := 
sorry

theorem tan_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.tan x = - 3 / 4 := 
sorry

end NUMINAMATH_GPT_sin_cos_value_sin_minus_cos_value_tan_value_l1839_183934


namespace NUMINAMATH_GPT_BoatCrafters_total_canoes_l1839_183999

def canoe_production (n : ℕ) : ℕ :=
  if n = 0 then 5 else 3 * canoe_production (n-1) - 1

theorem BoatCrafters_total_canoes : 
  (canoe_production 0 - 1) + (canoe_production 1 - 1) + (canoe_production 2 - 1) + (canoe_production 3 - 1) = 196 := 
by
  sorry

end NUMINAMATH_GPT_BoatCrafters_total_canoes_l1839_183999


namespace NUMINAMATH_GPT_figurine_cost_is_one_l1839_183988

-- Definitions from the conditions
def cost_per_tv : ℕ := 50
def num_tvs : ℕ := 5
def num_figurines : ℕ := 10
def total_spent : ℕ := 260

-- The price of a single figurine
def cost_per_figurine (total_spent num_tvs cost_per_tv num_figurines : ℕ) : ℕ :=
  (total_spent - num_tvs * cost_per_tv) / num_figurines

-- The theorem statement
theorem figurine_cost_is_one : cost_per_figurine total_spent num_tvs cost_per_tv num_figurines = 1 :=
by
  sorry

end NUMINAMATH_GPT_figurine_cost_is_one_l1839_183988


namespace NUMINAMATH_GPT_heather_aprons_l1839_183974

theorem heather_aprons :
  ∀ (total sewn already_sewn sewn_today half_remaining tomorrow_sew : ℕ),
    total = 150 →
    already_sewn = 13 →
    sewn_today = 3 * already_sewn →
    sewn = already_sewn + sewn_today →
    remaining = total - sewn →
    half_remaining = remaining / 2 →
    tomorrow_sew = half_remaining →
    tomorrow_sew = 49 := 
by 
  -- The proof is left as an exercise.
  sorry

end NUMINAMATH_GPT_heather_aprons_l1839_183974


namespace NUMINAMATH_GPT_rain_probability_l1839_183903

theorem rain_probability :
  let PM : ℝ := 0.62
  let PT : ℝ := 0.54
  let PMcTc : ℝ := 0.28
  let PMT : ℝ := PM + PT - (1 - PMcTc)
  PMT = 0.44 :=
by
  sorry

end NUMINAMATH_GPT_rain_probability_l1839_183903


namespace NUMINAMATH_GPT_total_amount_l1839_183938

-- Define the conditions in Lean
variables (X Y Z: ℝ)
variable (h1 : Y = 0.75 * X)
variable (h2 : Z = (2/3) * X)
variable (h3 : Y = 48)

-- The theorem stating that the total amount of money is Rs. 154.67
theorem total_amount (X Y Z : ℝ) (h1 : Y = 0.75 * X) (h2 : Z = (2/3) * X) (h3 : Y = 48) : 
  X + Y + Z = 154.67 := 
by
  sorry

end NUMINAMATH_GPT_total_amount_l1839_183938


namespace NUMINAMATH_GPT_nilpotent_matrix_squared_zero_l1839_183957

variable {R : Type*} [Field R]
variable (A : Matrix (Fin 2) (Fin 2) R)

theorem nilpotent_matrix_squared_zero (h : A^4 = 0) : A^2 = 0 := 
sorry

end NUMINAMATH_GPT_nilpotent_matrix_squared_zero_l1839_183957


namespace NUMINAMATH_GPT_distance_a_beats_b_l1839_183919

noncomputable def time_a : ℕ := 90 -- A's time in seconds 
noncomputable def time_b : ℕ := 180 -- B's time in seconds 
noncomputable def distance : ℝ := 4.5 -- distance in km

theorem distance_a_beats_b : distance = (distance / time_a) * (time_b - time_a) :=
by
  -- sorry placeholder for proof
  sorry

end NUMINAMATH_GPT_distance_a_beats_b_l1839_183919


namespace NUMINAMATH_GPT_solve_for_x_l1839_183986

open Real

-- Define the condition and the target result
def target (x : ℝ) : Prop :=
  sqrt (9 + sqrt (16 + 3 * x)) + sqrt (3 + sqrt (4 + x)) = 3 + 3 * sqrt 2

theorem solve_for_x (x : ℝ) (h : target x) : x = 8 * sqrt 2 / 3 :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l1839_183986


namespace NUMINAMATH_GPT_max_of_three_numbers_l1839_183933

theorem max_of_three_numbers : ∀ (a b c : ℕ), a = 10 → b = 11 → c = 12 → max (max a b) c = 12 :=
by
  intros a b c h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_max_of_three_numbers_l1839_183933


namespace NUMINAMATH_GPT_range_of_a_l1839_183914

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - 2 * a * x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1839_183914


namespace NUMINAMATH_GPT_total_cost_is_2160_l1839_183977

variables (x y z : ℝ)

-- Conditions
def cond1 : Prop := x = 0.45 * y
def cond2 : Prop := y = 0.8 * z
def cond3 : Prop := z = x + 640

-- Goal
def total_cost := x + y + z

theorem total_cost_is_2160 (x y z : ℝ) (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 x z) :
  total_cost x y z = 2160 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_2160_l1839_183977


namespace NUMINAMATH_GPT_find_b_l1839_183958

def has_exactly_one_real_solution (f : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = 0

theorem find_b (b : ℝ) :
  (∃! (x : ℝ), x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) ↔ b < 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1839_183958
