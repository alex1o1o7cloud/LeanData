import Mathlib

namespace y2_minus_x2_l1016_101650

theorem y2_minus_x2 (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h1 : 56 ≤ x + y) (h2 : x + y ≤ 59) (h3 : 9 < 10 * x) (h4 : 10 * x < 91 * y) : y^2 - x^2 = 177 :=
by
  sorry

end y2_minus_x2_l1016_101650


namespace hex_to_decimal_B4E_l1016_101610

def hex_B := 11
def hex_4 := 4
def hex_E := 14
def base := 16
def hex_value := hex_B * base^2 + hex_4 * base^1 + hex_E * base^0

theorem hex_to_decimal_B4E : hex_value = 2894 :=
by
  -- here we would write the proof steps, this is skipped with "sorry"
  sorry

end hex_to_decimal_B4E_l1016_101610


namespace frictional_force_is_12N_l1016_101640

-- Given conditions
variables (m1 m2 a μ : ℝ)
-- Constants
def g : ℝ := 9.8

-- Frictional force on the tank
def F_friction : ℝ := μ * m1 * g

-- Proof statement
theorem frictional_force_is_12N (m1_value : m1 = 3) (m2_value : m2 = 15) (a_value : a = 4) (μ_value : μ = 0.6) :
  m1 * a = 12 :=
by
  sorry

end frictional_force_is_12N_l1016_101640


namespace balance_after_transactions_l1016_101617

variable (x : ℝ)

def monday_spent : ℝ := 0.525 * x
def tuesday_spent (remaining : ℝ) : ℝ := 0.106875 * remaining
def wednesday_spent (remaining : ℝ) : ℝ := 0.131297917 * remaining
def thursday_spent (remaining : ℝ) : ℝ := 0.040260605 * remaining

def final_balance (x : ℝ) : ℝ :=
  let after_monday := x - monday_spent x
  let after_tuesday := after_monday - tuesday_spent after_monday
  let after_wednesday := after_tuesday - wednesday_spent after_tuesday
  after_wednesday - thursday_spent after_wednesday

theorem balance_after_transactions (x : ℝ) :
  final_balance x = 0.196566478 * x :=
by
  sorry

end balance_after_transactions_l1016_101617


namespace find_other_integer_l1016_101686

theorem find_other_integer (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : y = 14 ∨ x = 14 :=
  sorry

end find_other_integer_l1016_101686


namespace quadratic_inequality_k_range_l1016_101657

variable (k : ℝ)

theorem quadratic_inequality_k_range (h : ∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) :
  -1 < k ∧ k < 0 := by
sorry

end quadratic_inequality_k_range_l1016_101657


namespace rectangle_shaded_area_equal_l1016_101681

theorem rectangle_shaded_area_equal {x : ℝ} :
  let total_area := 72
  let shaded_area := 24 + 6*x
  let non_shaded_area := total_area / 2
  shaded_area = non_shaded_area → x = 2 := 
by 
  intros h
  sorry

end rectangle_shaded_area_equal_l1016_101681


namespace die_top_face_after_path_l1016_101695

def opposite_face (n : ℕ) : ℕ :=
  7 - n

def roll_die (start : ℕ) (sequence : List String) : ℕ :=
  sequence.foldl
    (λ top movement =>
      match movement with
      | "left" => opposite_face (7 - top) -- simplified assumption for movements
      | "forward" => opposite_face (top - 1)
      | "right" => opposite_face (7 - top + 1)
      | "back" => opposite_face (top + 1)
      | _ => top) start

theorem die_top_face_after_path : roll_die 3 ["left", "forward", "right", "back", "forward", "back"] = 4 :=
  by
  sorry

end die_top_face_after_path_l1016_101695


namespace rabbit_turtle_travel_distance_l1016_101694

-- Define the initial conditions and their values
def rabbit_velocity : ℕ := 40 -- meters per minute when jumping
def rabbit_jump_time : ℕ := 3 -- minutes of jumping
def rabbit_rest_time : ℕ := 2 -- minutes of resting
def rabbit_start_time : ℕ := 9 * 60 -- 9:00 AM in minutes from midnight

def turtle_velocity : ℕ := 10 -- meters per minute
def turtle_start_time : ℕ := 6 * 60 + 40 -- 6:40 AM in minutes from midnight
def lead_time : ℕ := 15 -- turtle leads the rabbit by 15 seconds at the end

-- Define the final distance the turtle traveled by the time rabbit arrives
def distance_traveled_by_turtle (total_time : ℕ) : ℕ :=
  total_time * turtle_velocity

-- Define time intervals for periodic calculations (in minutes)
def time_interval : ℕ := 5

-- Define the total distance rabbit covers in one periodic interval
def rabbit_distance_in_interval : ℕ :=
  rabbit_velocity * rabbit_jump_time

-- Calculate total time taken by the rabbit to close the gap before starting actual run
def initial_time_to_close_gap (gap : ℕ) : ℕ := 
  gap * time_interval / rabbit_distance_in_interval

-- Define the total time the rabbit travels
def total_travel_time : ℕ :=
  initial_time_to_close_gap ((rabbit_start_time - turtle_start_time) * turtle_velocity) + 97

-- Define the total distance condition to be proved as 2370 meters
theorem rabbit_turtle_travel_distance :
  distance_traveled_by_turtle (total_travel_time + lead_time) = 2370 :=
  by sorry

end rabbit_turtle_travel_distance_l1016_101694


namespace solution_a_eq_2_solution_a_in_real_l1016_101627

-- Define the polynomial inequality for the given conditions
def inequality (x : ℝ) (a : ℝ) : Prop := 12 * x ^ 2 - a * x > a ^ 2

-- Proof statement for when a = 2
theorem solution_a_eq_2 :
  ∀ x : ℝ, inequality x 2 ↔ (x < - (1 : ℝ) / 2) ∨ (x > (2 : ℝ) / 3) :=
sorry

-- Proof statement for when a is in ℝ
theorem solution_a_in_real (a : ℝ) :
  ∀ x : ℝ, inequality x a ↔
    if h : 0 < a then (x < - a / 4) ∨ (x > a / 3)
    else if h : a = 0 then (x ≠ 0)
    else (x < a / 3) ∨ (x > - a / 4) :=
sorry

end solution_a_eq_2_solution_a_in_real_l1016_101627


namespace range_of_n_l1016_101643

theorem range_of_n (n : ℝ) (x : ℝ) (h1 : 180 - n > 0) (h2 : ∀ x, 180 - n != x ∧ 180 - n != x + 24 → 180 - n + x + x + 24 = 180 → 44 ≤ x ∧ x ≤ 52 → 112 ≤ n ∧ n ≤ 128)
  (h3 : ∀ n, 180 - n = max (180 - n) (180 - n) - 24 ∧ min (180 - n) (180 - n) = n - 24 → 104 ≤ n ∧ n ≤ 112)
  (h4 : ∀ n, 180 - n = min (180 - n) (180 - n) ∧ max (180 - n) (180 - n) = 180 - n + 24 → 128 ≤ n ∧ n ≤ 136) :
  104 ≤ n ∧ n ≤ 136 :=
by sorry

end range_of_n_l1016_101643


namespace bottom_rightmost_rectangle_is_E_l1016_101649

-- Definitions of the given conditions
structure Rectangle where
  w : ℕ
  y : ℕ

def A : Rectangle := { w := 5, y := 8 }
def B : Rectangle := { w := 2, y := 4 }
def C : Rectangle := { w := 4, y := 6 }
def D : Rectangle := { w := 8, y := 5 }
def E : Rectangle := { w := 10, y := 9 }

-- The theorem we need to prove
theorem bottom_rightmost_rectangle_is_E :
    (E.w = 10) ∧ (E.y = 9) :=
by
  -- Proof would go here
  sorry

end bottom_rightmost_rectangle_is_E_l1016_101649


namespace cafeteria_extra_apples_l1016_101661

-- Define the conditions from the problem
def red_apples : ℕ := 33
def green_apples : ℕ := 23
def students : ℕ := 21

-- Define the total apples and apples given out based on the conditions
def total_apples : ℕ := red_apples + green_apples
def apples_given : ℕ := students

-- Define the extra apples as the difference between total apples and apples given out
def extra_apples : ℕ := total_apples - apples_given

-- The theorem to prove that the number of extra apples is 35
theorem cafeteria_extra_apples : extra_apples = 35 :=
by
  -- The structure of the proof would go here, but is omitted
  sorry

end cafeteria_extra_apples_l1016_101661


namespace stock_market_value_l1016_101662

def face_value : ℝ := 100
def dividend_rate : ℝ := 0.05
def yield_rate : ℝ := 0.10

theorem stock_market_value :
  (dividend_rate * face_value / yield_rate = 50) :=
by
  sorry

end stock_market_value_l1016_101662


namespace joan_gave_27_apples_l1016_101618

theorem joan_gave_27_apples (total_apples : ℕ) (current_apples : ℕ)
  (h1 : total_apples = 43) 
  (h2 : current_apples = 16) : 
  total_apples - current_apples = 27 := 
by
  sorry

end joan_gave_27_apples_l1016_101618


namespace possible_area_l1016_101632

theorem possible_area (A : ℝ) (B : ℝ) (L : ℝ × ℝ) (H₁ : L.1 = 13) (H₂ : L.2 = 14) (area_needed : ℝ) (H₃ : area_needed = 200) : 
∃ x y : ℝ, x = 13 ∧ y = 16 ∧ x * y ≥ area_needed :=
by
  sorry

end possible_area_l1016_101632


namespace quadratic_has_real_root_l1016_101616

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l1016_101616


namespace Mel_weight_is_70_l1016_101628

-- Definitions and conditions
def MelWeight (M : ℕ) :=
  3 * M + 10

theorem Mel_weight_is_70 (M : ℕ) (h1 : 3 * M + 10 = 220) :
  M = 70 :=
by
  sorry

end Mel_weight_is_70_l1016_101628


namespace find_distance_between_posters_and_wall_l1016_101668

-- Definitions for given conditions
def poster_width : ℝ := 29.05
def num_posters : ℕ := 8
def wall_width : ℝ := 394.4

-- The proof statement: find the distance 'd' between posters and ends
theorem find_distance_between_posters_and_wall :
  ∃ d : ℝ, (wall_width - num_posters * poster_width) / (num_posters + 1) = d ∧ d = 18 := 
by {
  -- The proof would involve showing that this specific d meets the constraints.
  sorry
}

end find_distance_between_posters_and_wall_l1016_101668


namespace ratio_of_candy_bar_to_caramel_l1016_101645

noncomputable def price_of_caramel : ℝ := 3
noncomputable def price_of_candy_bar (k : ℝ) : ℝ := k * price_of_caramel
noncomputable def price_of_cotton_candy (C : ℝ) : ℝ := 2 * C 

theorem ratio_of_candy_bar_to_caramel (k : ℝ) (C CC : ℝ) :
  C = price_of_candy_bar k →
  CC = price_of_cotton_candy C →
  6 * C + 3 * price_of_caramel + CC = 57 →
  C / price_of_caramel = 2 :=
by
  sorry

end ratio_of_candy_bar_to_caramel_l1016_101645


namespace triangle_arithmetic_geometric_equilateral_l1016_101626

theorem triangle_arithmetic_geometric_equilateral :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ (∃ d, β = α + d ∧ γ = α + 2 * d) ∧ (∃ r, β = α * r ∧ γ = α * r^2) →
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by
  sorry

end triangle_arithmetic_geometric_equilateral_l1016_101626


namespace triangle_angle_ABC_l1016_101613

theorem triangle_angle_ABC
  (ABD CBD ABC : ℝ) 
  (h1 : ABD = 70)
  (h2 : ABD + CBD + ABC = 200)
  (h3 : CBD = 60) : ABC = 70 := 
sorry

end triangle_angle_ABC_l1016_101613


namespace distance_last_day_l1016_101680

theorem distance_last_day
  (total_distance : ℕ)
  (days : ℕ)
  (initial_distance : ℕ)
  (common_ratio : ℚ)
  (sum_geometric : initial_distance * (1 - common_ratio^days) / (1 - common_ratio) = total_distance) :
  total_distance = 378 → days = 6 → common_ratio = 1/2 → 
  initial_distance = 192 → initial_distance * common_ratio^(days - 1) = 6 := 
by
  intros h1 h2 h3 h4
  sorry

end distance_last_day_l1016_101680


namespace quadrilateral_diagonal_length_l1016_101642

theorem quadrilateral_diagonal_length (d : ℝ) 
  (h_offsets : true) 
  (area_quadrilateral : 195 = ((1 / 2) * d * 9) + ((1 / 2) * d * 6)) : 
  d = 26 :=
by 
  sorry

end quadrilateral_diagonal_length_l1016_101642


namespace problem_equiv_proof_l1016_101612

noncomputable def simplify_and_evaluate (a : ℝ) :=
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4))

theorem problem_equiv_proof :
  simplify_and_evaluate (Real.sqrt 2) = 1 := 
  sorry

end problem_equiv_proof_l1016_101612


namespace bhanu_income_l1016_101673

theorem bhanu_income (I P : ℝ) (h1 : (P / 100) * I = 300) (h2 : (20 / 100) * (I - 300) = 140) : P = 30 := by
  sorry

end bhanu_income_l1016_101673


namespace proposition_P_l1016_101621

theorem proposition_P (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := 
by 
  sorry

end proposition_P_l1016_101621


namespace factorize_x_squared_sub_xy_l1016_101678

theorem factorize_x_squared_sub_xy (x y : ℝ) : x^2 - x * y = x * (x - y) :=
sorry

end factorize_x_squared_sub_xy_l1016_101678


namespace latus_rectum_of_parabola_l1016_101604

theorem latus_rectum_of_parabola :
  (∃ p : ℝ, ∀ x y : ℝ, y = - (1 / 6) * x^2 → y = p ∧ p = 3 / 2) :=
sorry

end latus_rectum_of_parabola_l1016_101604


namespace chips_reach_end_l1016_101620

theorem chips_reach_end (n k : ℕ) (h : n > k * 2^k) : True := sorry

end chips_reach_end_l1016_101620


namespace range_of_expression_l1016_101636

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  1 ≤ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ∧ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ≤ 22 + 4 * Real.sqrt 5 :=
sorry

end range_of_expression_l1016_101636


namespace possible_values_f_l1016_101614

noncomputable def f (x y z : ℝ) : ℝ := (y / (y + x)) + (z / (z + y)) + (x / (x + z))

theorem possible_values_f (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x^2 + y^3 = z^4) : 
  1 < f x y z ∧ f x y z < 2 :=
sorry

end possible_values_f_l1016_101614


namespace number_of_4_letter_words_with_B_l1016_101692

-- Define the set of letters.
inductive Alphabet
| A | B | C | D | E

-- The number of 4-letter words with repetition allowed and must include 'B' at least once.
noncomputable def words_with_at_least_one_B : ℕ :=
  let total := 5 ^ 4 -- Total number of 4-letter words.
  let without_B := 4 ^ 4 -- Total number of 4-letter words without 'B'.
  total - without_B

-- The main theorem statement.
theorem number_of_4_letter_words_with_B : words_with_at_least_one_B = 369 :=
  by sorry

end number_of_4_letter_words_with_B_l1016_101692


namespace comic_books_l1016_101653

variables (x y : ℤ)

def condition1 (x y : ℤ) : Prop := y + 7 = 5 * (x - 7)
def condition2 (x y : ℤ) : Prop := y - 9 = 3 * (x + 9)

theorem comic_books (x y : ℤ) (h₁ : condition1 x y) (h₂ : condition2 x y) : x = 39 ∧ y = 153 :=
by
  sorry

end comic_books_l1016_101653


namespace ratatouille_cost_per_quart_l1016_101690

theorem ratatouille_cost_per_quart:
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let zucchini_pounds := 4
  let zucchini_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let total_quarts := 4
  let eggplants_cost := eggplants_pounds * eggplants_cost_per_pound
  let zucchini_cost := zucchini_pounds * zucchini_cost_per_pound
  let tomatoes_cost := tomatoes_pounds * tomatoes_cost_per_pound
  let onions_cost := onions_pounds * onions_cost_per_pound
  let basil_cost := basil_pounds * (basil_cost_per_half_pound / 0.5)
  let total_cost := eggplants_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost
  let cost_per_quart := total_cost / total_quarts
  cost_per_quart = 10.00 :=
  by
    sorry

end ratatouille_cost_per_quart_l1016_101690


namespace february_saving_l1016_101684

-- Definitions for the conditions
variable {F D : ℝ}

-- Condition 1: Saving in January
def january_saving : ℝ := 2

-- Condition 2: Saving in March
def march_saving : ℝ := 8

-- Condition 3: Total savings after 6 months
def total_savings : ℝ := 126

-- Condition 4: Savings increase by a fixed amount D each month
def fixed_increase : ℝ := D

-- Condition 5: Difference between savings in March and January
def difference_jan_mar : ℝ := 8 - 2

-- The main theorem to prove: Robi saved 50 in February
theorem february_saving : F = 50 :=
by
  -- The required proof is omitted
  sorry

end february_saving_l1016_101684


namespace shaded_area_proof_l1016_101675

noncomputable def shaded_area (side_length : ℝ) (radius_factor : ℝ) : ℝ :=
  let square_area := side_length * side_length
  let radius := radius_factor * side_length
  let circle_area := Real.pi * (radius * radius)
  square_area - circle_area

theorem shaded_area_proof : shaded_area 8 0.6 = 64 - 23.04 * Real.pi :=
by sorry

end shaded_area_proof_l1016_101675


namespace ratio_of_volumes_total_surface_area_smaller_cube_l1016_101608

-- Definitions using the conditions in (a)
def edge_length_smaller_cube := 4 -- in inches
def edge_length_larger_cube := 24 -- in inches (2 feet converted to inches)

-- Propositions based on the correct answers in (b)
theorem ratio_of_volumes : 
  (edge_length_smaller_cube ^ 3) / (edge_length_larger_cube ^ 3) = 1 / 216 := by
  sorry

theorem total_surface_area_smaller_cube : 
  6 * (edge_length_smaller_cube ^ 2) = 96 := by
  sorry

end ratio_of_volumes_total_surface_area_smaller_cube_l1016_101608


namespace amy_balloons_l1016_101647

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 232) (h2 : james_balloons = amy_balloons + 131) :
  amy_balloons = 101 :=
by
  sorry

end amy_balloons_l1016_101647


namespace question1_question2_l1016_101685

def setA : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

def setB (a : ℝ) : Set ℝ := {x : ℝ | abs (x - a) ≤ 1 }

def complementA : Set ℝ := {x : ℝ | x ≤ -1 ∨ x > 3}

theorem question1 : A = setA := sorry

theorem question2 (a : ℝ) : setB a ∩ complementA = setB a → a ∈ Set.union (Set.Iic (-2)) (Set.Ioi 4) := sorry

end question1_question2_l1016_101685


namespace simplify_tan_cot_fraction_l1016_101635

theorem simplify_tan_cot_fraction :
  let tan45 := 1
  let cot45 := 1
  (tan45^3 + cot45^3) / (tan45 + cot45) = 1 := by
    sorry

end simplify_tan_cot_fraction_l1016_101635


namespace eccentricity_of_ellipse_l1016_101607

theorem eccentricity_of_ellipse :
  (∃ θ : Real, (x = 3 * Real.cos θ) ∧ (y = 4 * Real.sin θ))
  → (∃ e : Real, e = Real.sqrt 7 / 4) := 
sorry

end eccentricity_of_ellipse_l1016_101607


namespace total_animals_peppersprayed_l1016_101623

-- Define the conditions
def number_of_raccoons : ℕ := 12
def squirrels_vs_raccoons : ℕ := 6
def number_of_squirrels (raccoons : ℕ) (factor : ℕ) : ℕ := raccoons * factor

-- Define the proof statement
theorem total_animals_peppersprayed : 
  number_of_squirrels number_of_raccoons squirrels_vs_raccoons + number_of_raccoons = 84 :=
by
  -- The proof would go here
  sorry

end total_animals_peppersprayed_l1016_101623


namespace households_subscribing_to_F_l1016_101656

theorem households_subscribing_to_F
  (x y : ℕ)
  (hx : x ≥ 1)
  (h_subscriptions : 1 + 4 + 2 + 2 + 2 + y = 2 + 2 + 4 + 3 + 5 + x)
  : y = 6 :=
sorry

end households_subscribing_to_F_l1016_101656


namespace coach_recommendation_l1016_101603

def shots_A : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def shots_B : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) (mean : ℚ) : ℚ :=
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

noncomputable def recommendation (shots_A shots_B : List ℕ) : String :=
  let avg_A := average shots_A
  let avg_B := average shots_B
  let var_A := variance shots_A avg_A
  let var_B := variance shots_B avg_B
  if avg_A = avg_B ∧ var_A > var_B then "player B" else "player A"

theorem coach_recommendation : recommendation shots_A shots_B = "player B" :=
  by
  sorry

end coach_recommendation_l1016_101603


namespace custom_operation_difference_correct_l1016_101602

def custom_operation (x y : ℕ) : ℕ := x * y + 2 * x

theorem custom_operation_difference_correct :
  custom_operation 5 3 - custom_operation 3 5 = 4 :=
by
  sorry

end custom_operation_difference_correct_l1016_101602


namespace solve_for_x_l1016_101622

theorem solve_for_x {x : ℤ} (h : x - 2 * x + 3 * x - 4 * x = 120) : x = -60 :=
sorry

end solve_for_x_l1016_101622


namespace probability_sum_10_l1016_101600

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l1016_101600


namespace tina_earned_more_l1016_101641

def candy_bar_problem_statement : Prop :=
  let type_a_price := 2
  let type_b_price := 3
  let marvin_type_a_sold := 20
  let marvin_type_b_sold := 15
  let tina_type_a_sold := 70
  let tina_type_b_sold := 35
  let marvin_discount_per_5_type_a := 1
  let tina_discount_per_10_type_b := 2
  let tina_returns_type_b := 2
  let marvin_total_earnings := 
    (marvin_type_a_sold * type_a_price) + 
    (marvin_type_b_sold * type_b_price) -
    (marvin_type_a_sold / 5 * marvin_discount_per_5_type_a)
  let tina_total_earnings := 
    (tina_type_a_sold * type_a_price) + 
    (tina_type_b_sold * type_b_price) -
    (tina_type_b_sold / 10 * tina_discount_per_10_type_b) -
    (tina_returns_type_b * type_b_price)
  let difference := tina_total_earnings - marvin_total_earnings
  difference = 152

theorem tina_earned_more :
  candy_bar_problem_statement :=
by
  sorry

end tina_earned_more_l1016_101641


namespace probability_red_next_ball_l1016_101630

-- Definitions of initial conditions
def initial_red_balls : ℕ := 50
def initial_blue_balls : ℕ := 50
def initial_yellow_balls : ℕ := 30
def total_pulled_balls : ℕ := 65

-- Condition that Calvin pulled out 5 more red balls than blue balls
def red_balls_pulled (blue_balls_pulled : ℕ) : ℕ := blue_balls_pulled + 5

-- Compute the remaining balls
def remaining_balls (blue_balls_pulled : ℕ) : Prop :=
  let remaining_red_balls := initial_red_balls - red_balls_pulled blue_balls_pulled
  let remaining_blue_balls := initial_blue_balls - blue_balls_pulled
  let remaining_yellow_balls := initial_yellow_balls - (total_pulled_balls - red_balls_pulled blue_balls_pulled - blue_balls_pulled)
  (remaining_red_balls + remaining_blue_balls + remaining_yellow_balls) = 15

-- Main theorem to be proven
theorem probability_red_next_ball (blue_balls_pulled : ℕ) (h : remaining_balls blue_balls_pulled) :
  (initial_red_balls - red_balls_pulled blue_balls_pulled) / 15 = 9 / 26 :=
sorry

end probability_red_next_ball_l1016_101630


namespace reciprocal_of_neg_one_sixth_is_neg_six_l1016_101697

theorem reciprocal_of_neg_one_sixth_is_neg_six : 1 / (- (1 / 6)) = -6 :=
by sorry

end reciprocal_of_neg_one_sixth_is_neg_six_l1016_101697


namespace grid_problem_l1016_101652

theorem grid_problem 
    (A B : ℕ)
    (H1 : 1 ≠ A)
    (H2 : 1 ≠ B)
    (H3 : 2 ≠ A)
    (H4 : 2 ≠ B)
    (H5 : 3 ≠ A)
    (H6 : 3 ≠ B)
    (H7 : A = 2)
    (H8 : B = 1)
    :
    A * B = 2 :=
by
  sorry

end grid_problem_l1016_101652


namespace cricket_run_rate_l1016_101651

theorem cricket_run_rate (initial_run_rate : ℝ) (initial_overs : ℕ) (target : ℕ) (remaining_overs : ℕ) 
    (run_rate_in_remaining_overs : ℝ)
    (h1 : initial_run_rate = 3.2)
    (h2 : initial_overs = 10)
    (h3 : target = 272)
    (h4 : remaining_overs = 40) :
    run_rate_in_remaining_overs = 6 :=
  sorry

end cricket_run_rate_l1016_101651


namespace cost_of_Roger_cookie_l1016_101629

theorem cost_of_Roger_cookie
  (art_cookie_length : ℕ := 4)
  (art_cookie_width : ℕ := 3)
  (art_cookie_count : ℕ := 10)
  (roger_cookie_side : ℕ := 3)
  (art_cookie_price : ℕ := 50)
  (same_dough_used : ℕ := art_cookie_count * art_cookie_length * art_cookie_width)
  (roger_cookie_area : ℕ := roger_cookie_side * roger_cookie_side)
  (roger_cookie_count : ℕ := same_dough_used / roger_cookie_area) :
  (500 / roger_cookie_count) = 38 := by
  sorry

end cost_of_Roger_cookie_l1016_101629


namespace algebra_minimum_value_l1016_101615

theorem algebra_minimum_value :
  ∀ x y : ℝ, ∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 + 6*x - 2*y + 12 ≥ m) ∧ m = 2 :=
by
  sorry

end algebra_minimum_value_l1016_101615


namespace ordered_concrete_weight_l1016_101669

def weight_of_materials : ℝ := 0.83
def weight_of_bricks : ℝ := 0.17
def weight_of_stone : ℝ := 0.5

theorem ordered_concrete_weight :
  weight_of_materials - (weight_of_bricks + weight_of_stone) = 0.16 := by
  sorry

end ordered_concrete_weight_l1016_101669


namespace smaller_cube_surface_area_l1016_101683

theorem smaller_cube_surface_area (edge_length : ℝ) (h : edge_length = 12) :
  let sphere_diameter := edge_length
  let smaller_cube_side := sphere_diameter / Real.sqrt 3
  let surface_area := 6 * smaller_cube_side ^ 2
  surface_area = 288 := by
  sorry

end smaller_cube_surface_area_l1016_101683


namespace jane_spent_more_on_ice_cream_l1016_101605

-- Definitions based on the conditions
def ice_cream_cone_cost : ℕ := 5
def pudding_cup_cost : ℕ := 2
def ice_cream_cones_bought : ℕ := 15
def pudding_cups_bought : ℕ := 5

-- The mathematically equivalent proof statement
theorem jane_spent_more_on_ice_cream : 
  (ice_cream_cones_bought * ice_cream_cone_cost - pudding_cups_bought * pudding_cup_cost) = 65 := 
by
  sorry

end jane_spent_more_on_ice_cream_l1016_101605


namespace trader_sold_bags_l1016_101638

-- Define the conditions as constants
def initial_bags : ℕ := 55
def restocked_bags : ℕ := 132
def current_bags : ℕ := 164

-- Define a function to calculate the number of bags sold
def bags_sold (initial restocked current : ℕ) : ℕ :=
  initial + restocked - current

-- Statement of the proof problem
theorem trader_sold_bags : bags_sold initial_bags restocked_bags current_bags = 23 :=
by
  -- Proof is omitted
  sorry

end trader_sold_bags_l1016_101638


namespace ellipse_foci_coordinates_l1016_101658

theorem ellipse_foci_coordinates :
  ∀ x y : ℝ,
  25 * x^2 + 16 * y^2 = 1 →
  (x, y) = (0, 3/20) ∨ (x, y) = (0, -3/20) :=
by
  intro x y h
  sorry

end ellipse_foci_coordinates_l1016_101658


namespace value_of_polynomial_l1016_101688

theorem value_of_polynomial (a b : ℝ) (h : a^2 - 2 * b - 1 = 0) : -2 * a^2 + 4 * b + 2025 = 2023 :=
by
  sorry

end value_of_polynomial_l1016_101688


namespace three_digit_numbers_left_l1016_101682

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def isABAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ n = 100 * A + 10 * B + A

def isAABOrBAAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ (n = 100 * A + 10 * A + B ∨ n = 100 * B + 10 * A + A)

def totalThreeDigitNumbers : ℕ := 900

def countABA : ℕ := 81

def countAABAndBAA : ℕ := 153

theorem three_digit_numbers_left : 
  (totalThreeDigitNumbers - countABA - countAABAndBAA) = 666 := 
by
   sorry

end three_digit_numbers_left_l1016_101682


namespace max_value_of_k_l1016_101639

theorem max_value_of_k (m : ℝ) (h₁ : 0 < m) (h₂ : m < 1/2) : 
  (1 / m + 2 / (1 - 2 * m)) ≥ 8 :=
sorry

end max_value_of_k_l1016_101639


namespace cube_skew_lines_l1016_101699

theorem cube_skew_lines (cube : Prop) (diagonal : Prop) (edges : Prop) :
  ( ∃ n : ℕ, n = 6 ) :=
by
  sorry

end cube_skew_lines_l1016_101699


namespace roger_gave_candies_l1016_101670

theorem roger_gave_candies :
  ∀ (original_candies : ℕ) (remaining_candies : ℕ) (given_candies : ℕ),
  original_candies = 95 → remaining_candies = 92 → given_candies = original_candies - remaining_candies → given_candies = 3 :=
by
  intros
  sorry

end roger_gave_candies_l1016_101670


namespace range_of_k_l1016_101637

noncomputable def equation (k x : ℝ) : ℝ := 4^x - k * 2^x + k + 3

theorem range_of_k {x : ℝ} (h : ∀ k, equation k x = 0 → ∃! x : ℝ, equation k x = 0) :
  ∃ k : ℝ, (k = 6 ∨ k < -3)∧ (∀ y, equation k y ≠ 0 → (y ≠ x)) :=
sorry

end range_of_k_l1016_101637


namespace friends_boat_crossing_impossible_l1016_101646

theorem friends_boat_crossing_impossible : 
  ∀ (friends : Finset ℕ) (boat_capacity : ℕ), friends.card = 5 → boat_capacity ≥ 5 → 
  ¬ (∀ group : Finset ℕ, group ⊆ friends → group ≠ ∅ → group.card ≤ boat_capacity → 
  ∃ crossing : ℕ, (crossing = group.card ∧ group ⊆ friends)) :=
by
  intro friends boat_capacity friends_card boat_capacity_cond goal
  sorry

end friends_boat_crossing_impossible_l1016_101646


namespace imo1987_q6_l1016_101679

theorem imo1987_q6 (m n : ℤ) (h : n = m + 2) :
  ⌊(n : ℝ) * Real.sqrt 2⌋ = 2 + ⌊(m : ℝ) * Real.sqrt 2⌋ := 
by
  sorry -- We skip the detailed proof steps here.

end imo1987_q6_l1016_101679


namespace no_positive_integer_n_satisfies_l1016_101624

theorem no_positive_integer_n_satisfies :
  ¬∃ (n : ℕ), (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) := by
  sorry

end no_positive_integer_n_satisfies_l1016_101624


namespace horner_multiplications_additions_l1016_101611

def f (x : ℝ) : ℝ := 6 * x^6 + 5

def x : ℝ := 2

theorem horner_multiplications_additions :
  (6 : ℕ) = 6 ∧ (6 : ℕ) = 6 := 
by 
  sorry

end horner_multiplications_additions_l1016_101611


namespace sum_of_squares_of_extremes_l1016_101687

theorem sum_of_squares_of_extremes
  (a b c : ℕ)
  (h1 : 2*b = 3*a)
  (h2 : 3*b = 4*c)
  (h3 : b = 9) :
  a^2 + c^2 = 180 :=
sorry

end sum_of_squares_of_extremes_l1016_101687


namespace ratio_of_areas_is_five_l1016_101660

-- Define a convex quadrilateral ABCD
structure Quadrilateral (α : Type) :=
  (A B C D : α)
  (convex : True)  -- We assume convexity

-- Define the additional points B1, C1, D1, A1
structure ExtendedPoints (α : Type) (q : Quadrilateral α) :=
  (B1 C1 D1 A1 : α)
  (BB1_eq_AB : True) -- we assume the conditions BB1 = AB
  (CC1_eq_BC : True) -- CC1 = BC
  (DD1_eq_CD : True) -- DD1 = CD
  (AA1_eq_DA : True) -- AA1 = DA

-- Define the areas of the quadrilaterals
noncomputable def area {α : Type} [MetricSpace α] (A B C D : α) : ℝ := sorry
noncomputable def ratio_of_areas {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) : ℝ :=
  (area p.A1 p.B1 p.C1 p.D1) / (area q.A q.B q.C q.D)

theorem ratio_of_areas_is_five {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) :
  ratio_of_areas q p = 5 := sorry

end ratio_of_areas_is_five_l1016_101660


namespace mn_value_l1016_101676

theorem mn_value (m n : ℝ) 
  (h1 : m^2 + 1 = 4)
  (h2 : 2 * m + n = 0) :
  m * n = -6 := 
sorry

end mn_value_l1016_101676


namespace part1_part2_l1016_101667

theorem part1 (x y : ℝ) (h1 : y = x + 30) (h2 : 2 * x + 3 * y = 340) : x = 50 ∧ y = 80 :=
by {
  -- Later, we can place the steps to prove x = 50 and y = 80 here.
  sorry
}

theorem part2 (m : ℕ) (h3 : 0 ≤ m ∧ m ≤ 50)
               (h4 : 54 * (50 - m) + 72 * m = 3060) : m = 20 :=
by {
  -- Later, we can place the steps to prove m = 20 here.
  sorry
}

end part1_part2_l1016_101667


namespace tiffany_uploaded_7_pics_from_her_phone_l1016_101654

theorem tiffany_uploaded_7_pics_from_her_phone
  (camera_pics : ℕ)
  (albums : ℕ)
  (pics_per_album : ℕ)
  (total_pics : ℕ)
  (h_camera_pics : camera_pics = 13)
  (h_albums : albums = 5)
  (h_pics_per_album : pics_per_album = 4)
  (h_total_pics : total_pics = albums * pics_per_album) :
  total_pics - camera_pics = 7 := by
  sorry

end tiffany_uploaded_7_pics_from_her_phone_l1016_101654


namespace gcd_max_value_l1016_101677

noncomputable def max_gcd (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else 1

theorem gcd_max_value :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → gcd (13 * m + 4) (7 * m + 2) ≤ max_gcd m) ∧
              (∀ m : ℕ, m > 0 → max_gcd m ≤ 2) :=
by {
  sorry
}

end gcd_max_value_l1016_101677


namespace minimal_guests_l1016_101655

-- Problem statement: For 120 chairs arranged in a circle,
-- determine the smallest number of guests (N) needed 
-- so that any additional guest must sit next to an already seated guest.

theorem minimal_guests (N : ℕ) : 
  (∀ (chairs : ℕ), chairs = 120 → 
    ∃ (N : ℕ), N = 20 ∧ 
      (∀ (new_guest : ℕ), new_guest + chairs = 120 → 
        new_guest ≤ N + 1 ∧ new_guest ≤ N - 1)) :=
by
  sorry

end minimal_guests_l1016_101655


namespace find_missing_ratio_l1016_101664

def compounded_ratio (x y : ℚ) : ℚ := (x / y) * (6 / 11) * (11 / 2)

theorem find_missing_ratio (x y : ℚ) (h : compounded_ratio x y = 2) :
  x / y = 2 / 3 :=
sorry

end find_missing_ratio_l1016_101664


namespace no_positive_integers_satisfy_l1016_101665

theorem no_positive_integers_satisfy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 + 1 ≠ (x + 2)^5 + (y - 3)^5 :=
sorry

end no_positive_integers_satisfy_l1016_101665


namespace s_point_condition_l1016_101691

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def g_prime (a : ℝ) (x : ℝ) : ℝ := 1 / x

theorem s_point_condition (a : ℝ) (x₀ : ℝ) (h_f_g : f a x₀ = g a x₀) (h_f'g' : f_prime a x₀ = g_prime a x₀) :
  a = 2 / Real.exp 1 :=
by
  sorry

end s_point_condition_l1016_101691


namespace max_value_expr_bound_l1016_101601

noncomputable def max_value_expr (x : ℝ) : ℝ := 
  x^6 / (x^10 + x^8 - 6 * x^6 + 27 * x^4 + 64)

theorem max_value_expr_bound : 
  ∃ x : ℝ, max_value_expr x ≤ 1 / 8.38 := sorry

end max_value_expr_bound_l1016_101601


namespace ratio_of_heights_l1016_101609

-- Define the height of the first rocket.
def H1 : ℝ := 500

-- Define the combined height of the two rockets.
def combined_height : ℝ := 1500

-- Define the height of the second rocket.
def H2 : ℝ := combined_height - H1

-- The statement to be proven.
theorem ratio_of_heights : H2 / H1 = 2 := by
  -- Proof goes here
  sorry

end ratio_of_heights_l1016_101609


namespace shobha_current_age_l1016_101648

theorem shobha_current_age (S B : ℕ) (h1 : S / B = 4 / 3) (h2 : S + 6 = 26) : B = 15 :=
by
  -- Here we would begin the proof
  sorry

end shobha_current_age_l1016_101648


namespace manager_decision_correct_l1016_101689

theorem manager_decision_correct (x : ℝ) (profit : ℝ) 
  (h_condition1 : ∀ (x : ℝ), profit = (2 * x + 20) * (40 - x)) 
  (h_condition2 : 0 ≤ x ∧ x ≤ 40)
  (h_price_reduction : x = 15) :
  profit = 1250 :=
by
  sorry

end manager_decision_correct_l1016_101689


namespace f_at_3_l1016_101674

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 5

theorem f_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 11 :=
by
  sorry

end f_at_3_l1016_101674


namespace sin_2alpha_val_l1016_101634

-- Define the conditions and the problem in Lean 4
theorem sin_2alpha_val (α : ℝ) (h1 : π < α ∨ α < 3 * π / 2)
  (h2 : 2 * (Real.tan α) ^ 2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5 * π / 4 → Real.sin (2 * α) = 4 / 5) ∧ 
  (5 * π / 4 < α ∧ α < 3 * π / 2 → Real.sin (2 * α) = 3 / 5) := 
sorry

end sin_2alpha_val_l1016_101634


namespace trajectory_of_circle_center_l1016_101631

theorem trajectory_of_circle_center :
  ∀ (M : ℝ × ℝ), (∃ r : ℝ, (M.1 + r = 1 ∧ M.1 - r = -1) ∧ (M.1 - 1)^2 + (M.2 - 0)^2 = r^2) → M.2^2 = 4 * M.1 :=
by
  intros M h
  sorry

end trajectory_of_circle_center_l1016_101631


namespace steve_commute_l1016_101619

theorem steve_commute :
  ∃ (D : ℝ), 
    (∃ (V : ℝ), 2 * V = 5 ∧ (D / V + D / (2 * V) = 6)) ∧ D = 10 :=
by
  sorry

end steve_commute_l1016_101619


namespace total_points_l1016_101698

def points_earned (goblins orcs dragons : ℕ): ℕ :=
  goblins * 3 + orcs * 5 + dragons * 10

theorem total_points :
  points_earned 10 7 1 = 75 :=
by
  sorry

end total_points_l1016_101698


namespace dan_minimum_speed_to_beat_cara_l1016_101693

theorem dan_minimum_speed_to_beat_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 120 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (dan_speed : ℕ), dan_speed > 40 :=
by
  sorry

end dan_minimum_speed_to_beat_cara_l1016_101693


namespace determinant_expression_l1016_101666

theorem determinant_expression (a b c d p q r : ℝ)
  (h1: (∃ x: ℝ, x^4 + p*x^2 + q*x + r = 0) → (x = a ∨ x = b ∨ x = c ∨ x = d))
  (h2: a*b + a*c + a*d + b*c + b*d + c*d = p)
  (h3: a*b*c + a*b*d + a*c*d + b*c*d = q)
  (h4: a*b*c*d = -r):
  (Matrix.det ![![1 + a, 1, 1, 1], ![1, 1 + b, 1, 1], ![1, 1, 1 + c, 1], ![1, 1, 1, 1 + d]]) 
  = r + q + p := 
sorry

end determinant_expression_l1016_101666


namespace interior_diagonal_length_l1016_101663

theorem interior_diagonal_length (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26)
  (h2 : 4 * (a + b + c) = 28) : 
  (a^2 + b^2 + c^2) = 23 :=
by
  sorry

end interior_diagonal_length_l1016_101663


namespace remainder_g10_div_g_l1016_101671

-- Conditions/Definitions
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1
def g10 (x : ℝ) : ℝ := (g (x^10))

-- Theorem/Question
theorem remainder_g10_div_g : (g10 x) % (g x) = 6 :=
by
  sorry

end remainder_g10_div_g_l1016_101671


namespace total_cases_after_three_days_l1016_101659

def initial_cases : ℕ := 2000
def increase_rate : ℝ := 0.20
def recovery_rate : ℝ := 0.02

def day_cases (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_cases
  | n + 1 => 
      let prev_cases := day_cases n
      let new_cases := increase_rate * prev_cases
      let recovered := recovery_rate * prev_cases
      prev_cases + new_cases - recovered

theorem total_cases_after_three_days : day_cases 3 = 3286 := by sorry

end total_cases_after_three_days_l1016_101659


namespace sufficient_condition_not_necessary_condition_l1016_101625

variable {a b : ℝ} 

theorem sufficient_condition (h : a < b ∧ b < 0) : a ^ 2 > b ^ 2 :=
sorry

theorem not_necessary_condition : ¬ (∀ {a b : ℝ}, a ^ 2 > b ^ 2 → a < b ∧ b < 0) :=
sorry

end sufficient_condition_not_necessary_condition_l1016_101625


namespace journey_speed_l1016_101644

theorem journey_speed 
  (total_time : ℝ)
  (total_distance : ℝ)
  (second_half_speed : ℝ)
  (first_half_speed : ℝ) :
  total_time = 30 ∧ total_distance = 400 ∧ second_half_speed = 10 ∧
  2 * (total_distance / 2 / second_half_speed) + total_distance / 2 / first_half_speed = total_time →
  first_half_speed = 20 :=
by
  intros hyp
  sorry

end journey_speed_l1016_101644


namespace planks_from_friends_l1016_101633

theorem planks_from_friends :
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  planks_from_friends = 20 :=
by
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  rfl

end planks_from_friends_l1016_101633


namespace chantel_bracelets_final_count_l1016_101696

-- Definitions for conditions
def bracelets_made_days (days : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  days * bracelets_per_day

def initial_bracelets (days1 : ℕ) (bracelets_per_day1 : ℕ) : ℕ :=
  bracelets_made_days days1 bracelets_per_day1

def after_giving_away1 (initial_count : ℕ) (given_away1 : ℕ) : ℕ :=
  initial_count - given_away1

def additional_bracelets (days2 : ℕ) (bracelets_per_day2 : ℕ) : ℕ :=
  bracelets_made_days days2 bracelets_per_day2

def final_count (remaining_after_giving1 : ℕ) (additional_made : ℕ) (given_away2 : ℕ) : ℕ :=
  remaining_after_giving1 + additional_made - given_away2

-- Main theorem statement
theorem chantel_bracelets_final_count :
  ∀ (days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 : ℕ),
  days1 = 5 →
  bracelets_per_day1 = 2 →
  given_away1 = 3 →
  days2 = 4 →
  bracelets_per_day2 = 3 →
  given_away2 = 6 →
  final_count (after_giving_away1 (initial_bracelets days1 bracelets_per_day1) given_away1)
              (additional_bracelets days2 bracelets_per_day2)
              given_away2 = 13 :=
by
  intros days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 hdays1 hbracelets_per_day1 hgiven_away1 hdays2 hbracelets_per_day2 hgiven_away2
  -- Proof is not required, so we use sorry
  sorry

end chantel_bracelets_final_count_l1016_101696


namespace range_of_k_l1016_101606

noncomputable def e := Real.exp 1

theorem range_of_k (k : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ e ^ (x1 - 1) = |k * x1| ∧ e ^ (x2 - 1) = |k * x2| ∧ e ^ (x3 - 1) = |k * x3|) : k^2 > 1 := sorry

end range_of_k_l1016_101606


namespace sum_of_interior_angles_of_regular_polygon_l1016_101672

theorem sum_of_interior_angles_of_regular_polygon :
  (∀ (n : ℕ), (n ≠ 0) ∧ ((360 / 45 = n) → (180 * (n - 2) = 1080))) := by sorry

end sum_of_interior_angles_of_regular_polygon_l1016_101672
