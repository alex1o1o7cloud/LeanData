import Mathlib

namespace NUMINAMATH_GPT_local_minimum_f_is_1_maximum_local_minimum_g_is_1_l1761_176199

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

def local_minimum_value_f := 1

theorem local_minimum_f_is_1 : 
  ∃ x0 : ℝ, x0 > 0 ∧ (∀ x > 0, f x0 ≤ f x) ∧ f x0 = local_minimum_value_f :=
sorry

noncomputable def g (a x : ℝ) : ℝ := f x - a * (x - 1)

def maximum_value_local_minimum_g := 1

theorem maximum_local_minimum_g_is_1 :
  ∃ a x0 : ℝ, a = 0 ∧ x0 > 0 ∧ (∀ x > 0, g a x0 ≤ g a x) ∧ g a x0 = maximum_value_local_minimum_g :=
sorry

end NUMINAMATH_GPT_local_minimum_f_is_1_maximum_local_minimum_g_is_1_l1761_176199


namespace NUMINAMATH_GPT_trains_meeting_distance_l1761_176138

theorem trains_meeting_distance :
  ∃ D T : ℕ, (D = 20 * T) ∧ (D + 60 = 25 * T) ∧ (2 * D + 60 = 540) :=
by
  sorry

end NUMINAMATH_GPT_trains_meeting_distance_l1761_176138


namespace NUMINAMATH_GPT_product_of_two_numbers_l1761_176191

-- State the conditions and the proof problem
theorem product_of_two_numbers (x y : ℤ) (h_sum : x + y = 30) (h_diff : x - y = 6) :
  x * y = 216 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1761_176191


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1761_176172

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1 / x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∀ a b : ℝ, a < b → (f b ≤ f a → b ≤ (1 : ℝ) / 2)) :=
by sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1761_176172


namespace NUMINAMATH_GPT_frosting_sugar_calc_l1761_176105

theorem frosting_sugar_calc (total_sugar cake_sugar : ℝ) (h1 : total_sugar = 0.8) (h2 : cake_sugar = 0.2) : 
  total_sugar - cake_sugar = 0.6 :=
by
  rw [h1, h2]
  sorry  -- Proof should go here

end NUMINAMATH_GPT_frosting_sugar_calc_l1761_176105


namespace NUMINAMATH_GPT_factorize_x2_minus_9_l1761_176142

theorem factorize_x2_minus_9 (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
sorry

end NUMINAMATH_GPT_factorize_x2_minus_9_l1761_176142


namespace NUMINAMATH_GPT_backpack_prices_purchasing_plans_backpacks_given_away_l1761_176185

-- Part 1: Prices of Type A and Type B backpacks
theorem backpack_prices (x y : ℝ) (h1 : x = 2 * y - 30) (h2 : 2 * x + 3 * y = 255) : x = 60 ∧ y = 45 :=
sorry

-- Part 2: Possible purchasing plans
theorem purchasing_plans (m : ℕ) (h1 : 8900 ≥ 50 * m + 40 * (200 - m)) (h2 : m > 87) : 
  m = 88 ∨ m = 89 ∨ m = 90 :=
sorry

-- Part 3: Number of backpacks given away
theorem backpacks_given_away (m n : ℕ) (total_A : ℕ := 89) (total_B : ℕ := 111) 
(h1 : m + n = 4) 
(h2 : 1250 = (total_A - if total_A > 10 then total_A / 10 else 0) * 60 + (total_B - if total_B > 10 then total_B / 10 else 0) * 45 - (50 * total_A + 40 * total_B)) :
m = 1 ∧ n = 3 := 
sorry

end NUMINAMATH_GPT_backpack_prices_purchasing_plans_backpacks_given_away_l1761_176185


namespace NUMINAMATH_GPT_monotonicity_and_inequality_l1761_176170

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem monotonicity_and_inequality (a : ℝ) (p q : ℝ) (hp : 0 < p ∧ p < 1) (hq : 0 < q ∧ q < 1)
  (h_distinct: p ≠ q) (h_a : a ≥ 10) : 
  (f a (p + 1) - f a (q + 1)) / (p - q) > 1 := by
  sorry

end NUMINAMATH_GPT_monotonicity_and_inequality_l1761_176170


namespace NUMINAMATH_GPT_circle_area_difference_l1761_176164

noncomputable def difference_of_circle_areas (C1 C2 : ℝ) : ℝ :=
  let π := Real.pi
  let r1 := C1 / (2 * π)
  let r2 := C2 / (2 * π)
  let A1 := π * r1 ^ 2
  let A2 := π * r2 ^ 2
  A2 - A1

theorem circle_area_difference :
  difference_of_circle_areas 396 704 = 26948.4 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_difference_l1761_176164


namespace NUMINAMATH_GPT_two_numbers_are_opposites_l1761_176139

theorem two_numbers_are_opposites (x y z : ℝ) (h : (1 / x) + (1 / y) + (1 / z) = 1 / (x + y + z)) :
  (x + y = 0) ∨ (x + z = 0) ∨ (y + z = 0) :=
by
  sorry

end NUMINAMATH_GPT_two_numbers_are_opposites_l1761_176139


namespace NUMINAMATH_GPT_game_completion_days_l1761_176132

theorem game_completion_days (initial_playtime hours_per_day : ℕ) (initial_days : ℕ) (completion_percentage : ℚ) (increased_playtime : ℕ) (remaining_days : ℕ) :
  initial_playtime = 4 →
  hours_per_day = 2 * 7 →
  completion_percentage = 0.4 →
  increased_playtime = 7 →
  ((initial_playtime * hours_per_day) / completion_percentage) - (initial_playtime * hours_per_day) = increased_playtime * remaining_days →
  remaining_days = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_game_completion_days_l1761_176132


namespace NUMINAMATH_GPT_triangle_to_square_difference_l1761_176154

noncomputable def number_of_balls_in_triangle (T : ℕ) : ℕ :=
  T * (T + 1) / 2

noncomputable def number_of_balls_in_square (S : ℕ) : ℕ :=
  S * S

theorem triangle_to_square_difference (T S : ℕ) 
  (h1 : number_of_balls_in_triangle T = 1176) 
  (h2 : number_of_balls_in_square S = 1600) :
  T - S = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_to_square_difference_l1761_176154


namespace NUMINAMATH_GPT_ratio_second_shop_to_shirt_l1761_176109

-- Define the initial conditions in Lean
def initial_amount : ℕ := 55
def spent_on_shirt : ℕ := 7
def final_amount : ℕ := 27

-- Define the amount spent in the second shop calculation
def spent_in_second_shop (i_amt s_shirt f_amt : ℕ) : ℕ :=
  (i_amt - s_shirt) - f_amt

-- Define the ratio calculation
def ratio (a b : ℕ) : ℕ := a / b

-- Lean 4 statement proving the ratio of amounts
theorem ratio_second_shop_to_shirt : 
  ratio (spent_in_second_shop initial_amount spent_on_shirt final_amount) spent_on_shirt = 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_second_shop_to_shirt_l1761_176109


namespace NUMINAMATH_GPT_range_of_a_l1761_176178

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_of_a :
  (∃ (a : ℝ), (a ≤ -2 ∨ a ≥ 0) ∧ (∃ (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4), f x ≤ a^2 + 2 * a)) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1761_176178


namespace NUMINAMATH_GPT_c_is_11_years_younger_than_a_l1761_176110

variable (A B C : ℕ) (h : A + B = B + C + 11)

theorem c_is_11_years_younger_than_a (A B C : ℕ) (h : A + B = B + C + 11) : C = A - 11 := by
  sorry

end NUMINAMATH_GPT_c_is_11_years_younger_than_a_l1761_176110


namespace NUMINAMATH_GPT_person_B_catches_up_after_meeting_point_on_return_l1761_176137
noncomputable def distance_A := 46
noncomputable def speed_A := 15
noncomputable def speed_B := 40
noncomputable def initial_gap_time := 1

-- Prove that Person B catches up to Person A after 3/5 hours.
theorem person_B_catches_up_after : 
  ∃ x : ℚ, 40 * x = 15 * (x + 1) ∧ x = 3 / 5 := 
by
  sorry

-- Prove that they meet 10 kilometers away from point B on the return journey.
theorem meeting_point_on_return : 
  ∃ y : ℚ, (46 - y) / 15 - (46 + y) / 40 = 1 ∧ y = 10 := 
by 
  sorry

end NUMINAMATH_GPT_person_B_catches_up_after_meeting_point_on_return_l1761_176137


namespace NUMINAMATH_GPT_sum_first_2018_terms_of_given_sequence_l1761_176116

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_first_2018_terms_of_given_sequence :
  let a := 1
  let d := -1 / 2017
  S_2018 = 1009 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_2018_terms_of_given_sequence_l1761_176116


namespace NUMINAMATH_GPT_negation_of_forall_exp_positive_l1761_176114

theorem negation_of_forall_exp_positive :
  ¬ (∀ x : ℝ, Real.exp x > 0) ↔ ∃ x : ℝ, Real.exp x ≤ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_forall_exp_positive_l1761_176114


namespace NUMINAMATH_GPT_negation_of_existential_proposition_l1761_176155

-- Define the propositions
def proposition (x : ℝ) := x^2 - 2 * x + 1 ≤ 0

-- Define the negation of the propositions
def negation_prop (x : ℝ) := x^2 - 2 * x + 1 > 0

-- Theorem to prove that the negation of the existential proposition is the universal proposition
theorem negation_of_existential_proposition
  (h : ¬ ∃ x : ℝ, proposition x) :
  ∀ x : ℝ, negation_prop x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_proposition_l1761_176155


namespace NUMINAMATH_GPT_times_faster_l1761_176171

theorem times_faster (A B : ℝ) (h1 : A + B = 1 / 12) (h2 : A = 1 / 16) : 
  A / B = 3 :=
by
  sorry

end NUMINAMATH_GPT_times_faster_l1761_176171


namespace NUMINAMATH_GPT_integral_part_odd_l1761_176165

theorem integral_part_odd (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, (⌊(3 + Real.sqrt 5)^n⌋ = 2 * m + 1) := 
by
  -- Sorry used since the proof steps are not required in the task
  sorry

end NUMINAMATH_GPT_integral_part_odd_l1761_176165


namespace NUMINAMATH_GPT_mixed_doubles_selection_l1761_176159

-- Given conditions
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- The statement to show the number of different ways to select two players is 20
theorem mixed_doubles_selection : (num_male_players * num_female_players) = 20 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_mixed_doubles_selection_l1761_176159


namespace NUMINAMATH_GPT_smaller_angle_measure_l1761_176193

theorem smaller_angle_measure (x : ℝ) (h₁ : 5 * x + 3 * x = 180) : 3 * x = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_measure_l1761_176193


namespace NUMINAMATH_GPT_prove_optionC_is_suitable_l1761_176180

def OptionA := "Understanding the height of students in Class 7(1)"
def OptionB := "Companies recruiting and interviewing job applicants"
def OptionC := "Investigating the impact resistance of a batch of cars"
def OptionD := "Selecting the fastest runner in our school to participate in the city-wide competition"

def is_suitable_for_sampling_survey (option : String) : Prop :=
  option = OptionC

theorem prove_optionC_is_suitable :
  is_suitable_for_sampling_survey OptionC :=
by
  sorry

end NUMINAMATH_GPT_prove_optionC_is_suitable_l1761_176180


namespace NUMINAMATH_GPT_represent_2021_as_squares_l1761_176163

theorem represent_2021_as_squares :
  ∃ n : ℕ, n = 505 → 2021 = (n + 1)^2 - (n - 1)^2 + 1^2 :=
by
  sorry

end NUMINAMATH_GPT_represent_2021_as_squares_l1761_176163


namespace NUMINAMATH_GPT_max_remainder_division_by_9_l1761_176188

theorem max_remainder_division_by_9 : ∀ (r : ℕ), r < 9 → r ≤ 8 :=
by sorry

end NUMINAMATH_GPT_max_remainder_division_by_9_l1761_176188


namespace NUMINAMATH_GPT_first_player_winning_strategy_l1761_176123

def game_strategy (S : ℕ) : Prop :=
  ∃ k, (1 ≤ k ∧ k ≤ 5 ∧ (S - k) % 6 = 1)

theorem first_player_winning_strategy : game_strategy 100 :=
sorry

end NUMINAMATH_GPT_first_player_winning_strategy_l1761_176123


namespace NUMINAMATH_GPT_jenna_less_than_bob_l1761_176140

def bob_amount : ℕ := 60
def phil_amount : ℕ := (1 / 3) * bob_amount
def jenna_amount : ℕ := 2 * phil_amount

theorem jenna_less_than_bob : bob_amount - jenna_amount = 20 := by
  sorry

end NUMINAMATH_GPT_jenna_less_than_bob_l1761_176140


namespace NUMINAMATH_GPT_probability_black_or_white_l1761_176147

-- Defining the probabilities of drawing red and white balls
def prob_red : ℝ := 0.45
def prob_white : ℝ := 0.25

-- Defining the total probability
def total_prob : ℝ := 1.0

-- Define the probability of drawing a black or white ball
def prob_black_or_white : ℝ := total_prob - prob_red

-- The theorem stating the required proof
theorem probability_black_or_white : 
  prob_black_or_white = 0.55 := by
    sorry

end NUMINAMATH_GPT_probability_black_or_white_l1761_176147


namespace NUMINAMATH_GPT_not_black_cows_count_l1761_176127

theorem not_black_cows_count (total_cows : ℕ) (black_cows : ℕ) (h1 : total_cows = 18) (h2 : black_cows = 5 + total_cows / 2) :
  total_cows - black_cows = 4 :=
by 
  -- Insert the actual proof here
  sorry

end NUMINAMATH_GPT_not_black_cows_count_l1761_176127


namespace NUMINAMATH_GPT_units_digit_of_quotient_l1761_176115

theorem units_digit_of_quotient : 
  let n := 1993
  let term1 := 4 ^ n
  let term2 := 6 ^ n
  (term1 + term2) % 5 = 0 →
  let quotient := (term1 + term2) / 5
  (quotient % 10 = 0) := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_quotient_l1761_176115


namespace NUMINAMATH_GPT_amy_created_albums_l1761_176189

theorem amy_created_albums (total_photos : ℕ) (photos_per_album : ℕ) 
  (h1 : total_photos = 180)
  (h2 : photos_per_album = 20) : 
  (total_photos / photos_per_album = 9) :=
by
  sorry

end NUMINAMATH_GPT_amy_created_albums_l1761_176189


namespace NUMINAMATH_GPT_ants_in_third_anthill_l1761_176158

-- Define the number of ants in the first anthill
def ants_first : ℕ := 100

-- Define the percentage reduction for each subsequent anthill
def percentage_reduction : ℕ := 20

-- Calculate the number of ants in the second anthill
def ants_second : ℕ := ants_first - (percentage_reduction * ants_first / 100)

-- Calculate the number of ants in the third anthill
def ants_third : ℕ := ants_second - (percentage_reduction * ants_second / 100)

-- Main theorem to prove that the number of ants in the third anthill is 64
theorem ants_in_third_anthill : ants_third = 64 := sorry

end NUMINAMATH_GPT_ants_in_third_anthill_l1761_176158


namespace NUMINAMATH_GPT_problem_solution_l1761_176118

-- Define the ellipse equation and foci positions.
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 2) = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the line equation
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)
variable (k : ℝ)

-- Define the points lie on the line and ellipse
def A_on_line := ∃ x y, A = (x, y) ∧ line x y k
def B_on_line := ∃ x y, B = (x, y) ∧ line x y k

-- Define the parallel and perpendicular conditions
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k, v1.1 = k * v2.1 ∧ v1.2 = k * v2.2
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Lean theorem for the conclusions of the problem
theorem problem_solution (A_cond : A_on_line A k ∧ ellipse A.1 A.2) 
                          (B_cond : B_on_line B k ∧ ellipse B.1 B.2) :

  -- Prove these two statements
  ¬ parallel (A.1 + 1, A.2) (B.1 - 1, B.2) ∧
  ¬ perpendicular (A.1 + 1, A.2) (A.1 - 1, A.2) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1761_176118


namespace NUMINAMATH_GPT_darnell_texts_l1761_176120

theorem darnell_texts (T : ℕ) (unlimited_plan_cost alternative_text_cost alternative_call_cost : ℕ) 
    (call_minutes : ℕ) (cost_difference : ℕ) :
    unlimited_plan_cost = 12 →
    alternative_text_cost = 1 →
    alternative_call_cost = 3 →
    call_minutes = 60 →
    cost_difference = 1 →
    (alternative_text_cost * T / 30 + alternative_call_cost * call_minutes / 20) = 
      unlimited_plan_cost - cost_difference →
    T = 60 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_darnell_texts_l1761_176120


namespace NUMINAMATH_GPT_Billie_has_2_caps_l1761_176143

-- Conditions as definitions in Lean
def Sammy_caps : ℕ := 8
def Janine_caps : ℕ := Sammy_caps - 2
def Billie_caps : ℕ := Janine_caps / 3

-- Problem statement to prove
theorem Billie_has_2_caps : Billie_caps = 2 := by
  sorry

end NUMINAMATH_GPT_Billie_has_2_caps_l1761_176143


namespace NUMINAMATH_GPT_snickers_bars_needed_l1761_176161

-- Definitions of the conditions
def points_needed : ℕ := 2000
def chocolate_bunny_points : ℕ := 100
def number_of_chocolate_bunnies : ℕ := 8
def snickers_points : ℕ := 25

-- Derived conditions
def points_from_bunnies : ℕ := number_of_chocolate_bunnies * chocolate_bunny_points
def remaining_points : ℕ := points_needed - points_from_bunnies

-- Statement to prove
theorem snickers_bars_needed : ∀ (n : ℕ), n = remaining_points / snickers_points → n = 48 :=
by 
  sorry

end NUMINAMATH_GPT_snickers_bars_needed_l1761_176161


namespace NUMINAMATH_GPT_total_paint_area_l1761_176169

structure Room where
  length : ℕ
  width : ℕ
  height : ℕ

def livingRoom : Room := { length := 40, width := 40, height := 10 }
def bedroom : Room := { length := 12, width := 10, height := 10 }

def wallArea (room : Room) (n_walls : ℕ) : ℕ :=
  let longWallsArea := 2 * (room.length * room.height)
  let shortWallsArea := 2 * (room.width * room.height)
  if n_walls <= 2 then
    longWallsArea * n_walls / 2
  else if n_walls <= 4 then
    longWallsArea + (shortWallsArea * (n_walls - 2) / 2)
  else
    0

def totalWallArea (livingRoom : Room) (bedroom : Room) (n_livingRoomWalls n_bedroomWalls : ℕ) : ℕ :=
  wallArea livingRoom n_livingRoomWalls + wallArea bedroom n_bedroomWalls

theorem total_paint_area : totalWallArea livingRoom bedroom 3 4 = 1640 := by
  sorry

end NUMINAMATH_GPT_total_paint_area_l1761_176169


namespace NUMINAMATH_GPT_cos_A_eq_l1761_176195

variable (A : Real) (A_interior_angle_tri_ABC : A > π / 2 ∧ A < π) (tan_A_eq_neg_two : Real.tan A = -2)

theorem cos_A_eq : Real.cos A = - (Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_GPT_cos_A_eq_l1761_176195


namespace NUMINAMATH_GPT_total_time_to_fill_tank_l1761_176152

noncomputable def pipe_filling_time : ℕ := 
  let tank_capacity := 2000
  let pipe_a_rate := 200
  let pipe_b_rate := 50
  let pipe_c_rate := 25
  let cycle_duration := 5
  let cycle_fill := (pipe_a_rate * 1 + pipe_b_rate * 2 - pipe_c_rate * 2)
  let num_cycles := tank_capacity / cycle_fill
  num_cycles * cycle_duration

theorem total_time_to_fill_tank : pipe_filling_time = 40 := 
by
  unfold pipe_filling_time
  sorry

end NUMINAMATH_GPT_total_time_to_fill_tank_l1761_176152


namespace NUMINAMATH_GPT_sale_savings_l1761_176151

theorem sale_savings (price_fox : ℝ) (price_pony : ℝ) 
(discount_fox : ℝ) (discount_pony : ℝ) 
(total_discount : ℝ) (num_fox : ℕ) (num_pony : ℕ) 
(price_saved_during_sale : ℝ) :
price_fox = 15 → 
price_pony = 18 → 
num_fox = 3 → 
num_pony = 2 → 
total_discount = 22 → 
discount_pony = 15 → 
discount_fox = total_discount - discount_pony → 
price_saved_during_sale = num_fox * price_fox * (discount_fox / 100) + num_pony * price_pony * (discount_pony / 100) →
price_saved_during_sale = 8.55 := 
by sorry

end NUMINAMATH_GPT_sale_savings_l1761_176151


namespace NUMINAMATH_GPT_max_x_value_l1761_176148

variables {x y : ℝ}
variables (data : list (ℝ × ℝ))
variables (linear_relation : ℝ → ℝ → Prop)

def max_y : ℝ := 10

-- Given conditions
axiom linear_data :
  (data = [(16, 11), (14, 9), (12, 8), (8, 5)]) ∧
  (∀ (p : ℝ × ℝ), p ∈ data → linear_relation p.1 p.2)

-- Prove the maximum value of x for which y ≤ max_y
theorem max_x_value (h : ∀ (x y : ℝ), linear_relation x y → y = 11 - (16 - x) / 3):
  ∀ (x : ℝ), (∃ y : ℝ, linear_relation x y) → y ≤ max_y → x ≤ 15 :=
sorry

end NUMINAMATH_GPT_max_x_value_l1761_176148


namespace NUMINAMATH_GPT_hiker_speed_third_day_l1761_176129

-- Define the conditions
def first_day_distance : ℕ := 18
def first_day_speed : ℕ := 3
def second_day_distance : ℕ :=
  let first_day_hours := first_day_distance / first_day_speed
  let second_day_hours := first_day_hours - 1
  let second_day_speed := first_day_speed + 1
  second_day_hours * second_day_speed
def total_distance : ℕ := 53
def third_day_hours : ℕ := 3

-- Define the speed on the third day based on given conditions
def speed_on_third_day : ℕ :=
  let third_day_distance := total_distance - first_day_distance - second_day_distance
  third_day_distance / third_day_hours

-- The theorem we need to prove
theorem hiker_speed_third_day : speed_on_third_day = 5 := by
  sorry

end NUMINAMATH_GPT_hiker_speed_third_day_l1761_176129


namespace NUMINAMATH_GPT_checkered_board_cut_l1761_176183

def can_cut_equal_squares (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n > 5

theorem checkered_board_cut (n : ℕ) (h : n % 5 = 0 ∧ n > 5) :
  ∃ m, n^2 = 5 * m :=
by
  sorry

end NUMINAMATH_GPT_checkered_board_cut_l1761_176183


namespace NUMINAMATH_GPT_alligator_population_at_end_of_year_l1761_176156

-- Define the conditions
def initial_population : ℕ := 4
def doubling_period_months : ℕ := 6
def total_months : ℕ := 12

-- Define the proof goal
theorem alligator_population_at_end_of_year (initial_population doubling_period_months total_months : ℕ)
  (h_init : initial_population = 4)
  (h_double : doubling_period_months = 6)
  (h_total : total_months = 12) :
  initial_population * (2 ^ (total_months / doubling_period_months)) = 16 := 
by
  sorry

end NUMINAMATH_GPT_alligator_population_at_end_of_year_l1761_176156


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1761_176149

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) * (2 - x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1761_176149


namespace NUMINAMATH_GPT_triangle_side_condition_angle_condition_l1761_176124

variable (a b c A B C : ℝ)

theorem triangle_side_condition (a_eq : a = 2) (b_eq : b = Real.sqrt 7) (h : a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B) :
  c = 3 :=
  sorry

theorem angle_condition (angle_eq : Real.sqrt 3 * Real.sin (2 * A - π / 6) - 2 * Real.sin (C - π / 12)^2 = 0) :
  A = π / 4 :=
  sorry

end NUMINAMATH_GPT_triangle_side_condition_angle_condition_l1761_176124


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l1761_176133

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l1761_176133


namespace NUMINAMATH_GPT_inequality_proof_l1761_176181

theorem inequality_proof (a b c : ℝ) (h : a ^ 2 + b ^ 2 + c ^ 2 = 3) :
  (a ^ 2) / (2 + b + c ^ 2) + (b ^ 2) / (2 + c + a ^ 2) + (c ^ 2) / (2 + a + b ^ 2) ≥ (a + b + c) ^ 2 / 12 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1761_176181


namespace NUMINAMATH_GPT_probability_grunters_win_all_5_games_l1761_176168

noncomputable def probability_grunters_win_game : ℚ := 4 / 5

theorem probability_grunters_win_all_5_games :
  (probability_grunters_win_game ^ 5) = 1024 / 3125 := 
  by 
  sorry

end NUMINAMATH_GPT_probability_grunters_win_all_5_games_l1761_176168


namespace NUMINAMATH_GPT_path_traveled_is_correct_l1761_176112

-- Define the original triangle and the circle.
def side_a : ℝ := 8
def side_b : ℝ := 10
def side_c : ℝ := 12.5
def radius : ℝ := 1.5

-- Define the condition that the circle is rolling inside the triangle.
def new_side (original_side : ℝ) (r : ℝ) : ℝ := original_side - 2 * r

-- Calculate the new sides of the smaller triangle path.
def new_side_a := new_side side_a radius
def new_side_b := new_side side_b radius
def new_side_c := new_side side_c radius

-- Calculate the perimeter of the path traced by the circle's center.
def path_perimeter := new_side_a + new_side_b + new_side_c

-- Prove that this perimeter equals 21.5 units under given conditions.
theorem path_traveled_is_correct : path_perimeter = 21.5 := by
  simp [new_side, new_side_a, new_side_b, new_side_c, path_perimeter]
  sorry

end NUMINAMATH_GPT_path_traveled_is_correct_l1761_176112


namespace NUMINAMATH_GPT_problem_proof_l1761_176196

def problem_statement : Prop :=
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64

theorem problem_proof : problem_statement := by
  sorry

end NUMINAMATH_GPT_problem_proof_l1761_176196


namespace NUMINAMATH_GPT_simplify_expression_l1761_176125

variable (b c d x y : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + dy * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (cx + dy) 
  = b^2 * x^3 + 3 * c^2 * xy^3 + c^3 * y^3 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1761_176125


namespace NUMINAMATH_GPT_expected_value_of_win_l1761_176198

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end NUMINAMATH_GPT_expected_value_of_win_l1761_176198


namespace NUMINAMATH_GPT_B_initial_investment_l1761_176100

-- Definitions for investments and conditions
def A_init_invest : Real := 3000
def A_later_invest := 2 * A_init_invest

def A_yearly_investment := (A_init_invest * 6) + (A_later_invest * 6)

-- The amount B needs to invest for the yearly investment to be equal in the profit ratio 1:1
def B_investment (x : Real) := x * 12 

-- Definition of the proof problem
theorem B_initial_investment (x : Real) : A_yearly_investment = B_investment x → x = 4500 := 
by 
  sorry

end NUMINAMATH_GPT_B_initial_investment_l1761_176100


namespace NUMINAMATH_GPT_find_k_solution_l1761_176102

noncomputable def vec1 : ℝ × ℝ := (3, -4)
noncomputable def vec2 : ℝ × ℝ := (5, 8)
noncomputable def target_norm : ℝ := 3 * Real.sqrt 10

theorem find_k_solution : ∃ k : ℝ, 0 ≤ k ∧ ‖(k * vec1.1 - vec2.1, k * vec1.2 - vec2.2)‖ = target_norm ∧ k = 0.0288 :=
by
  sorry

end NUMINAMATH_GPT_find_k_solution_l1761_176102


namespace NUMINAMATH_GPT_area_original_is_504_l1761_176122

-- Define the sides of the three rectangles
variable (a1 b1 a2 b2 a3 b3 : ℕ)

-- Define the perimeters of the three rectangles
def P1 := 2 * (a1 + b1)
def P2 := 2 * (a2 + b2)
def P3 := 2 * (a3 + b3)

-- Define the conditions given in the problem
axiom P1_equal_P2_plus_20 : P1 = P2 + 20
axiom P2_equal_P3_plus_16 : P2 = P3 + 16

-- Define the calculation for the area of the original rectangle
def area_original := a1 * b1

-- Proof goal: the area of the original rectangle is 504
theorem area_original_is_504 : area_original = 504 := 
sorry

end NUMINAMATH_GPT_area_original_is_504_l1761_176122


namespace NUMINAMATH_GPT_selling_price_with_increase_l1761_176177

variable (a : ℝ)

theorem selling_price_with_increase (h : a > 0) : 1.1 * a = a + 0.1 * a := by
  -- Here you will add the proof, which we skip with sorry
  sorry

end NUMINAMATH_GPT_selling_price_with_increase_l1761_176177


namespace NUMINAMATH_GPT_remainder_of_expression_l1761_176104

theorem remainder_of_expression (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_expression_l1761_176104


namespace NUMINAMATH_GPT_number_142857_has_property_l1761_176134

noncomputable def has_desired_property (n : ℕ) : Prop :=
∀ m ∈ [1, 2, 3, 4, 5, 6], ∀ d ∈ (Nat.digits 10 (n * m)), d ∈ (Nat.digits 10 n)

theorem number_142857_has_property : has_desired_property 142857 :=
sorry

end NUMINAMATH_GPT_number_142857_has_property_l1761_176134


namespace NUMINAMATH_GPT_find_A_l1761_176192

theorem find_A (A : ℕ) (h : 59 = (A * 6) + 5) : A = 9 :=
by sorry

end NUMINAMATH_GPT_find_A_l1761_176192


namespace NUMINAMATH_GPT_total_coins_last_month_l1761_176108

theorem total_coins_last_month (m s : ℝ) : 
  (100 = 1.25 * m) ∧ (100 = 0.80 * s) → m + s = 205 :=
by sorry

end NUMINAMATH_GPT_total_coins_last_month_l1761_176108


namespace NUMINAMATH_GPT_parallel_lines_l1761_176157

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = a - 7) → a = 3 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_l1761_176157


namespace NUMINAMATH_GPT_emily_age_l1761_176153

theorem emily_age (A B C D E : ℕ) (h1 : A = B - 4) (h2 : B = C + 5) (h3 : D = C + 2) (h4 : E = A + D - B) (h5 : B = 20) : E = 13 :=
by sorry

end NUMINAMATH_GPT_emily_age_l1761_176153


namespace NUMINAMATH_GPT_fraction_of_time_to_cover_distance_l1761_176126

-- Definitions for the given conditions
def distance : ℝ := 540
def initial_time : ℝ := 12
def new_speed : ℝ := 60

-- The statement we need to prove
theorem fraction_of_time_to_cover_distance :
  ∃ (x : ℝ), (x = 3 / 4) ∧ (distance / (initial_time * x) = new_speed) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_fraction_of_time_to_cover_distance_l1761_176126


namespace NUMINAMATH_GPT_part1_proof_part2_proof_l1761_176187

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - 1| - |x - m|

theorem part1_proof : ∀ x, f x 2 ≥ 1 ↔ x ≥ 2 :=
by 
  sorry

theorem part2_proof : (∀ x : ℝ, f x m ≤ 5) → (-4 ≤ m ∧ m ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_part1_proof_part2_proof_l1761_176187


namespace NUMINAMATH_GPT_total_goals_in_five_matches_is_4_l1761_176145

theorem total_goals_in_five_matches_is_4
    (A : ℚ) -- defining the average number of goals before the fifth match as rational
    (h1 : A * 4 + 2 = (A + 0.3) * 5) : -- condition representing total goals equation
    4 = (4 * A + 2) := -- statement that the total number of goals in 5 matches is 4
by
  sorry

end NUMINAMATH_GPT_total_goals_in_five_matches_is_4_l1761_176145


namespace NUMINAMATH_GPT_initial_money_given_l1761_176131

def bracelet_cost : ℕ := 15
def necklace_cost : ℕ := 10
def mug_cost : ℕ := 20
def num_bracelets : ℕ := 3
def num_necklaces : ℕ := 2
def num_mugs : ℕ := 1
def change_received : ℕ := 15

theorem initial_money_given : num_bracelets * bracelet_cost + num_necklaces * necklace_cost + num_mugs * mug_cost + change_received = 100 := 
sorry

end NUMINAMATH_GPT_initial_money_given_l1761_176131


namespace NUMINAMATH_GPT_expression_evaluation_l1761_176136

theorem expression_evaluation : 
  76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1761_176136


namespace NUMINAMATH_GPT_triangle_BDC_is_isosceles_l1761_176101

-- Define the given conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC BC AD DC : ℝ)
variables (a : ℝ)
variables (α : ℝ)

-- Given conditions
def is_isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) : Prop :=
AB = AC

def angle_BAC_120 (α : ℝ) : Prop :=
α = 120

def point_D_extension (AD AB : ℝ) : Prop :=
AD = 2 * AB

-- Let triangle ABC be isosceles with AB = AC and angle BAC = 120 degrees
axiom isosceles_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB AC : ℝ) : is_isosceles_triangle A B C AB AC

axiom angle_BAC (α : ℝ) : angle_BAC_120 α

axiom point_D (AD AB : ℝ) : point_D_extension AD AB

-- Prove that triangle BDC is isosceles
theorem triangle_BDC_is_isosceles 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC BC AD DC : ℝ) 
  (α : ℝ) 
  (h1 : is_isosceles_triangle A B C AB AC)
  (h2 : angle_BAC_120 α)
  (h3 : point_D_extension AD AB) :
  BC = DC :=
sorry

end NUMINAMATH_GPT_triangle_BDC_is_isosceles_l1761_176101


namespace NUMINAMATH_GPT_frustum_surface_area_l1761_176179

theorem frustum_surface_area (r r' l : ℝ) (h_r : r = 1) (h_r' : r' = 4) (h_l : l = 5) :
  π * r^2 + π * r'^2 + π * (r + r') * l = 42 * π :=
by
  rw [h_r, h_r', h_l]
  norm_num
  sorry

end NUMINAMATH_GPT_frustum_surface_area_l1761_176179


namespace NUMINAMATH_GPT_michael_remaining_money_l1761_176144

variables (m b n : ℝ) (h1 : (1 : ℝ) / 3 * m = 1 / 2 * n * b) (h2 : 5 = m / 15)

theorem michael_remaining_money : m - (2 / 3 * m + m / 15) = 4 / 15 * m :=
by
  have hb1 : 2 / 3 * m = (2 * m) / 3 := by ring
  have hb2 : m / 15 = (1 * m) / 15 := by ring
  rw [hb1, hb2]
  sorry

end NUMINAMATH_GPT_michael_remaining_money_l1761_176144


namespace NUMINAMATH_GPT_odometer_reading_before_trip_l1761_176176

-- Define the given conditions
def odometer_reading_lunch : ℝ := 372.0
def miles_traveled : ℝ := 159.7

-- Theorem to prove that the odometer reading before the trip was 212.3 miles
theorem odometer_reading_before_trip : odometer_reading_lunch - miles_traveled = 212.3 := by
  sorry

end NUMINAMATH_GPT_odometer_reading_before_trip_l1761_176176


namespace NUMINAMATH_GPT_no5_battery_mass_l1761_176194

theorem no5_battery_mass :
  ∃ (x y : ℝ), 2 * x + 2 * y = 72 ∧ 3 * x + 2 * y = 96 ∧ x = 24 :=
by
  sorry

end NUMINAMATH_GPT_no5_battery_mass_l1761_176194


namespace NUMINAMATH_GPT_gravitational_force_at_300000_l1761_176167

-- Definitions and premises
def gravitational_force (d : ℝ) : ℝ := sorry

axiom inverse_square_law (d : ℝ) (f : ℝ) (k : ℝ) : f * d^2 = k

axiom surface_force : gravitational_force 5000 = 800

-- Goal: Prove the gravitational force at 300,000 miles
theorem gravitational_force_at_300000 : gravitational_force 300000 = 1 / 45 := sorry

end NUMINAMATH_GPT_gravitational_force_at_300000_l1761_176167


namespace NUMINAMATH_GPT_james_received_stickers_l1761_176117

theorem james_received_stickers (initial_stickers given_away final_stickers received_stickers : ℕ) 
  (h_initial : initial_stickers = 269)
  (h_given : given_away = 48)
  (h_final : final_stickers = 423)
  (h_total_before_giving_away : initial_stickers + received_stickers = given_away + final_stickers) :
  received_stickers = 202 :=
by
  sorry

end NUMINAMATH_GPT_james_received_stickers_l1761_176117


namespace NUMINAMATH_GPT_floor_add_double_eq_15_4_l1761_176174

theorem floor_add_double_eq_15_4 (r : ℝ) (h : (⌊r⌋ : ℝ) + 2 * r = 15.4) : r = 5.2 := 
sorry

end NUMINAMATH_GPT_floor_add_double_eq_15_4_l1761_176174


namespace NUMINAMATH_GPT_ellipses_have_equal_focal_length_l1761_176197

-- Define ellipses and their focal lengths
def ellipse1_focal_length : ℝ := 8
def k_condition (k : ℝ) : Prop := 0 < k ∧ k < 9
def ellipse2_focal_length (k : ℝ) : ℝ := 8

-- The main statement
theorem ellipses_have_equal_focal_length (k : ℝ) (hk : k_condition k) :
  ellipse1_focal_length = ellipse2_focal_length k :=
sorry

end NUMINAMATH_GPT_ellipses_have_equal_focal_length_l1761_176197


namespace NUMINAMATH_GPT_vec_subtraction_l1761_176106

variables (a b : Prod ℝ ℝ)
def vec1 : Prod ℝ ℝ := (1, 2)
def vec2 : Prod ℝ ℝ := (3, 1)

theorem vec_subtraction : (2 * (vec1.fst, vec1.snd) - (vec2.fst, vec2.snd)) = (-1, 3) := by
  -- Proof here, skipped
  sorry

end NUMINAMATH_GPT_vec_subtraction_l1761_176106


namespace NUMINAMATH_GPT_number_of_six_digit_numbers_formable_by_1_2_3_4_l1761_176119

theorem number_of_six_digit_numbers_formable_by_1_2_3_4
  (digits : Finset ℕ := {1, 2, 3, 4})
  (pairs_count : ℕ := 2)
  (non_adjacent_pair : ℕ := 1)
  (adjacent_pair : ℕ := 1)
  (six_digit_numbers : ℕ := 432) :
  ∃ (n : ℕ), n = 432 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_number_of_six_digit_numbers_formable_by_1_2_3_4_l1761_176119


namespace NUMINAMATH_GPT_sum_and_product_of_three_numbers_l1761_176113

variables (a b c : ℝ)

-- Conditions
axiom h1 : a + b = 35
axiom h2 : b + c = 47
axiom h3 : c + a = 52

-- Prove the sum and product
theorem sum_and_product_of_three_numbers : a + b + c = 67 ∧ a * b * c = 9600 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_and_product_of_three_numbers_l1761_176113


namespace NUMINAMATH_GPT_range_of_f_l1761_176175

noncomputable def f (x : ℝ) : ℝ := 4^x + 2^(x + 1) + 1

theorem range_of_f : Set.range f = {y : ℝ | y > 1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1761_176175


namespace NUMINAMATH_GPT_divisor_is_four_l1761_176146

theorem divisor_is_four (d n : ℤ) (k j : ℤ) 
  (h1 : n % d = 3) 
  (h2 : 2 * n % d = 2): d = 4 :=
sorry

end NUMINAMATH_GPT_divisor_is_four_l1761_176146


namespace NUMINAMATH_GPT_total_bottles_per_day_l1761_176173

def num_cases_per_day : ℕ := 7200
def bottles_per_case : ℕ := 10

theorem total_bottles_per_day : num_cases_per_day * bottles_per_case = 72000 := by
  sorry

end NUMINAMATH_GPT_total_bottles_per_day_l1761_176173


namespace NUMINAMATH_GPT_gcd_n4_plus_27_n_plus_3_l1761_176103

theorem gcd_n4_plus_27_n_plus_3 (n : ℕ) (h_pos : n > 9) : 
  gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_n4_plus_27_n_plus_3_l1761_176103


namespace NUMINAMATH_GPT_work_done_in_one_day_l1761_176182

theorem work_done_in_one_day (A_time B_time : ℕ) (hA : A_time = 4) (hB : B_time = A_time / 2) : 
  (1 / A_time + 1 / B_time) = (3 / 4) :=
by
  -- Here we are setting up the conditions as per our identified steps
  rw [hA, hB]
  -- The remaining steps to prove will be omitted as per instructions
  sorry

end NUMINAMATH_GPT_work_done_in_one_day_l1761_176182


namespace NUMINAMATH_GPT_laura_garden_daisies_l1761_176186

/-
Laura's Garden Problem: Given the ratio of daisies to tulips is 3:4,
Laura currently has 32 tulips, and she plans to add 24 more tulips,
prove that Laura will have 42 daisies in total after the addition to
maintain the same ratio.
-/

theorem laura_garden_daisies (daisies tulips add_tulips : ℕ) (ratio_d : ℕ) (ratio_t : ℕ)
    (h1 : ratio_d = 3) (h2 : ratio_t = 4) (h3 : tulips = 32) (h4 : add_tulips = 24)
    (new_tulips : ℕ := tulips + add_tulips) :
  daisies = 42 :=
by
  sorry

end NUMINAMATH_GPT_laura_garden_daisies_l1761_176186


namespace NUMINAMATH_GPT_avg_weight_section_B_l1761_176150

theorem avg_weight_section_B 
  (W_B : ℝ) 
  (num_students_A : ℕ := 36) 
  (avg_weight_A : ℝ := 30) 
  (num_students_B : ℕ := 24) 
  (total_students : ℕ := 60) 
  (avg_weight_class : ℝ := 30) 
  (h1 : num_students_A * avg_weight_A + num_students_B * W_B = total_students * avg_weight_class) :
  W_B = 30 :=
sorry

end NUMINAMATH_GPT_avg_weight_section_B_l1761_176150


namespace NUMINAMATH_GPT_probability_of_event_l1761_176166

theorem probability_of_event (favorable unfavorable : ℕ) (h : favorable = 3) (h2 : unfavorable = 5) :
  (favorable / (favorable + unfavorable) : ℚ) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_event_l1761_176166


namespace NUMINAMATH_GPT_sum_of_positive_integers_eq_32_l1761_176160

noncomputable def sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : ℕ :=
  x + y

theorem sum_of_positive_integers_eq_32 (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : sum_of_integers x y h1 h2 = 32 :=
  sorry

end NUMINAMATH_GPT_sum_of_positive_integers_eq_32_l1761_176160


namespace NUMINAMATH_GPT_cos_alpha_in_second_quadrant_l1761_176135

variable (α : Real) -- Define the variable α as a Real number (angle in radians)
variable (h1 : α > π / 2 ∧ α < π) -- Condition that α is in the second quadrant
variable (h2 : Real.sin α = 2 / 3) -- Condition that sin(α) = 2/3

theorem cos_alpha_in_second_quadrant (α : Real) (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.sin α = 2 / 3) : Real.cos α = - Real.sqrt (1 - (2 / 3) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_in_second_quadrant_l1761_176135


namespace NUMINAMATH_GPT_cheaper_price_difference_is_75_cents_l1761_176128

noncomputable def list_price := 42.50
noncomputable def store_a_discount := 12.00
noncomputable def store_b_discount_percent := 0.30

noncomputable def store_a_price := list_price - store_a_discount
noncomputable def store_b_price := (1 - store_b_discount_percent) * list_price
noncomputable def price_difference_in_dollars := store_a_price - store_b_price
noncomputable def price_difference_in_cents := price_difference_in_dollars * 100

theorem cheaper_price_difference_is_75_cents :
  price_difference_in_cents = 75 := by
  sorry

end NUMINAMATH_GPT_cheaper_price_difference_is_75_cents_l1761_176128


namespace NUMINAMATH_GPT_three_x4_plus_two_x5_l1761_176190

theorem three_x4_plus_two_x5 (x1 x2 x3 x4 x5 : ℤ)
  (h1 : 2 * x1 + x2 + x3 + x4 + x5 = 6)
  (h2 : x1 + 2 * x2 + x3 + x4 + x5 = 12)
  (h3 : x1 + x2 + 2 * x3 + x4 + x5 = 24)
  (h4 : x1 + x2 + x3 + 2 * x4 + x5 = 48)
  (h5 : x1 + x2 + x3 + x4 + 2 * x5 = 96) : 
  3 * x4 + 2 * x5 = 181 := 
sorry

end NUMINAMATH_GPT_three_x4_plus_two_x5_l1761_176190


namespace NUMINAMATH_GPT_base15_mod_9_l1761_176184

noncomputable def base15_to_decimal : ℕ :=
  2 * 15^3 + 6 * 15^2 + 4 * 15^1 + 3 * 15^0

theorem base15_mod_9 (n : ℕ) (h : n = base15_to_decimal) : n % 9 = 0 :=
sorry

end NUMINAMATH_GPT_base15_mod_9_l1761_176184


namespace NUMINAMATH_GPT_coffee_equals_milk_l1761_176111

theorem coffee_equals_milk (S : ℝ) (h : 0 < S ∧ S < 1/2) :
  let initial_milk := 1 / 2
  let initial_coffee := 1 / 2
  let glass1_initial := initial_milk
  let glass2_initial := initial_coffee
  let glass2_after_first_transfer := glass2_initial + S
  let coffee_transferred_back := (S * initial_coffee) / (initial_coffee + S)
  let milk_transferred_back := (S^2) / (initial_coffee + S)
  let glass1_after_second_transfer := glass1_initial - S + milk_transferred_back
  let glass2_after_second_transfer := glass2_initial + S - coffee_transferred_back
  (glass1_initial - S + milk_transferred_back) = (glass2_initial + S - coffee_transferred_back) :=
sorry

end NUMINAMATH_GPT_coffee_equals_milk_l1761_176111


namespace NUMINAMATH_GPT_expand_binomials_l1761_176162

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end NUMINAMATH_GPT_expand_binomials_l1761_176162


namespace NUMINAMATH_GPT_factorize_quadratic_l1761_176121

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_factorize_quadratic_l1761_176121


namespace NUMINAMATH_GPT_fourth_root_equiv_l1761_176141

theorem fourth_root_equiv (x : ℝ) (hx : 0 < x) : (x * (x ^ (3 / 4))) ^ (1 / 4) = x ^ (7 / 16) :=
sorry

end NUMINAMATH_GPT_fourth_root_equiv_l1761_176141


namespace NUMINAMATH_GPT_quadratic_points_relation_l1761_176107

theorem quadratic_points_relation
  (k y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -((-1) - 1)^2 + k)
  (hB : y₂ = -(2 - 1)^2 + k)
  (hC : y₃ = -(4 - 1)^2 + k) : y₃ < y₁ ∧ y₁ < y₂ :=
by
  sorry

end NUMINAMATH_GPT_quadratic_points_relation_l1761_176107


namespace NUMINAMATH_GPT_Mitch_needs_to_keep_500_for_license_and_registration_l1761_176130

-- Define the constants and variables
def total_savings : ℕ := 20000
def cost_per_foot : ℕ := 1500
def longest_boat_length : ℕ := 12
def docking_fee_factor : ℕ := 3

-- Define the price of the longest boat
def cost_longest_boat : ℕ := longest_boat_length * cost_per_foot

-- Define the amount for license and registration
def license_and_registration (L : ℕ) : Prop :=
  total_savings - cost_longest_boat = L * (docking_fee_factor + 1)

-- The statement to be proved
theorem Mitch_needs_to_keep_500_for_license_and_registration :
  ∃ L : ℕ, license_and_registration L ∧ L = 500 :=
by
  -- Conditions and setup have already been defined, we now state the proof goal.
  sorry

end NUMINAMATH_GPT_Mitch_needs_to_keep_500_for_license_and_registration_l1761_176130
