import Mathlib

namespace equal_commissions_implies_list_price_l56_5695

theorem equal_commissions_implies_list_price (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  sorry

end equal_commissions_implies_list_price_l56_5695


namespace max_photo_area_correct_l56_5629

def frame_area : ℝ := 59.6
def num_photos : ℕ := 4
def max_photo_area : ℝ := 14.9

theorem max_photo_area_correct : frame_area / num_photos = max_photo_area :=
by sorry

end max_photo_area_correct_l56_5629


namespace intersection_points_count_l56_5638

theorem intersection_points_count (A : ℝ) (hA : A > 0) :
  ((A > 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y) ∧
                              (x ≠ 0 ∨ y ≠ 0)) ∧
  ((A ≤ 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y)) :=
by
  sorry

end intersection_points_count_l56_5638


namespace negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l56_5677

-- Definition of a triangle with a property on the angles.
def triangle (a b c : ℝ) : Prop := a + b + c = 180 ∧ 0 < a ∧ 0 < b ∧ 0 < c

-- Definition of an obtuse angle.
def obtuse (x : ℝ) : Prop := x > 90

-- Proposition: In a triangle, at most one angle is obtuse.
def at_most_one_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a → ¬ obtuse b ∧ ¬ obtuse c) ∧ (obtuse b → ¬ obtuse a ∧ ¬ obtuse c) ∧ (obtuse c → ¬ obtuse a ∧ ¬ obtuse b)

-- Negation: In a triangle, there are at least two obtuse angles.
def at_least_two_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a ∧ obtuse b) ∨ (obtuse a ∧ obtuse c) ∨ (obtuse b ∧ obtuse c)

-- Prove the negation equivalence
theorem negation_of_at_most_one_obtuse_is_at_least_two_obtuse (a b c : ℝ) :
  (¬ at_most_one_obtuse a b c) ↔ at_least_two_obtuse a b c :=
sorry

end negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l56_5677


namespace longer_diagonal_is_116_l56_5619

-- Given conditions
def side_length : ℕ := 65
def short_diagonal : ℕ := 60

-- Prove that the length of the longer diagonal in the rhombus is 116 units.
theorem longer_diagonal_is_116 : 
  let s := side_length
  let d1 := short_diagonal / 2
  let d2 := (s^2 - d1^2).sqrt
  (2 * d2) = 116 :=
by
  sorry

end longer_diagonal_is_116_l56_5619


namespace simplify_expression_l56_5657

theorem simplify_expression :
  (16 / 54) * (27 / 8) * (64 / 81) = 64 / 9 :=
by sorry

end simplify_expression_l56_5657


namespace number_of_solutions_l56_5625

noncomputable def g_n (n : ℕ) (x : ℝ) := (Real.sin x)^(2 * n) + (Real.cos x)^(2 * n)

theorem number_of_solutions : ∀ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi) -> 
  8 * g_n 3 x - 6 * g_n 2 x = 3 * g_n 1 x -> false :=
by sorry

end number_of_solutions_l56_5625


namespace fraction_not_simplifiable_l56_5685

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_not_simplifiable_l56_5685


namespace decimal_fraction_error_l56_5614

theorem decimal_fraction_error (A B C D E : ℕ) (hA : A < 100) 
    (h10B : 10 * B = A + C) (h10C : 10 * C = 6 * A + D) (h10D : 10 * D = 7 * A + E) 
    (hBCDE_lt_A : B < A ∧ C < A ∧ D < A ∧ E < A) : 
    false :=
sorry

end decimal_fraction_error_l56_5614


namespace price_of_orange_is_60_l56_5668

-- Definitions from the conditions
def price_of_apple : ℕ := 40 -- The price of each apple is 40 cents
def total_fruits : ℕ := 10 -- Mary selects a total of 10 apples and oranges
def avg_price_initial : ℕ := 48 -- The average price of the 10 pieces of fruit is 48 cents
def put_back_oranges : ℕ := 2 -- Mary puts back 2 oranges
def avg_price_remaining : ℕ := 45 -- The average price of the remaining fruits is 45 cents

-- Variable definition for the price of an orange which will be solved for
variable (price_of_orange : ℕ)

-- Theorem: proving the price of each orange is 60 cents given the conditions
theorem price_of_orange_is_60 : 
  (∀ a o : ℕ, a + o = total_fruits →
  40 * a + price_of_orange * o = total_fruits * avg_price_initial →
  40 * a + price_of_orange * (o - put_back_oranges) = (total_fruits - put_back_oranges) * avg_price_remaining)
  → price_of_orange = 60 :=
by
  -- Proof is omitted
  sorry

end price_of_orange_is_60_l56_5668


namespace fraction_simplification_l56_5626

def numerator : Int := 5^4 + 5^2 + 5
def denominator : Int := 5^3 - 2 * 5

theorem fraction_simplification :
  (numerator : ℚ) / (denominator : ℚ) = 27 + (14 / 23) := by
  sorry

end fraction_simplification_l56_5626


namespace reading_club_coordinator_selection_l56_5660

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem reading_club_coordinator_selection :
  let total_ways := choose 18 4
  let no_former_ways := choose 10 4
  total_ways - no_former_ways = 2850 := by
  sorry

end reading_club_coordinator_selection_l56_5660


namespace students_not_playing_games_l56_5644

theorem students_not_playing_games 
  (total_students : ℕ)
  (basketball_players : ℕ)
  (volleyball_players : ℕ)
  (both_players : ℕ)
  (h1 : total_students = 20)
  (h2 : basketball_players = (1 / 2) * total_students)
  (h3 : volleyball_players = (2 / 5) * total_students)
  (h4 : both_players = (1 / 10) * total_students) :
  total_students - ((basketball_players + volleyball_players) - both_players) = 4 :=
by
  sorry

end students_not_playing_games_l56_5644


namespace probability_white_ball_second_draw_l56_5688

noncomputable def probability_white_given_red (red_white_yellow_balls : Nat × Nat × Nat) : ℚ :=
  let (r, w, y) := red_white_yellow_balls
  let total_balls := r + w + y
  let p_A := (r : ℚ) / total_balls
  let p_AB := (r : ℚ) / total_balls * (w : ℚ) / (total_balls - 1)
  p_AB / p_A

theorem probability_white_ball_second_draw (r w y : Nat) (h_r : r = 2) (h_w : w = 3) (h_y : y = 1) :
  probability_white_given_red (r, w, y) = 3 / 5 :=
by
  rw [h_r, h_w, h_y]
  unfold probability_white_given_red
  simp
  sorry

end probability_white_ball_second_draw_l56_5688


namespace inequality_solution_l56_5601

theorem inequality_solution (x y : ℝ) (h1 : y ≥ x^2 + 1) :
    2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by
  sorry

end inequality_solution_l56_5601


namespace first_applicant_earnings_l56_5600

def first_applicant_salary : ℕ := 42000
def first_applicant_training_cost_per_month : ℕ := 1200
def first_applicant_training_months : ℕ := 3
def second_applicant_salary : ℕ := 45000
def second_applicant_bonus_percentage : ℕ := 1
def company_earnings_from_second_applicant : ℕ := 92000
def earnings_difference : ℕ := 850

theorem first_applicant_earnings 
  (salary1 : first_applicant_salary = 42000)
  (train_cost_per_month : first_applicant_training_cost_per_month = 1200)
  (train_months : first_applicant_training_months = 3)
  (salary2 : second_applicant_salary = 45000)
  (bonus_percentage : second_applicant_bonus_percentage = 1)
  (earnings2 : company_earnings_from_second_applicant = 92000)
  (earning_diff : earnings_difference = 850) :
  (company_earnings_from_second_applicant - (second_applicant_salary + (second_applicant_salary * second_applicant_bonus_percentage / 100)) - earnings_difference) = 45700 := 
by 
  sorry

end first_applicant_earnings_l56_5600


namespace convex_polygon_max_interior_angles_l56_5602

theorem convex_polygon_max_interior_angles (n : ℕ) (h1 : n ≥ 3) (h2 : n < 360) :
  ∃ x, x ≤ 4 ∧ ∀ k, k > 4 → False :=
by
  sorry

end convex_polygon_max_interior_angles_l56_5602


namespace perpendicular_vector_l56_5643

theorem perpendicular_vector {a : ℝ × ℝ} (h : a = (1, -2)) : ∃ (b : ℝ × ℝ), b = (2, 1) ∧ (a.1 * b.1 + a.2 * b.2 = 0) :=
by 
  sorry

end perpendicular_vector_l56_5643


namespace solve_for_n_l56_5628

theorem solve_for_n :
  ∃ n : ℤ, -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180) ∧ n = 30 :=
by
  sorry

end solve_for_n_l56_5628


namespace garden_roller_length_l56_5613

theorem garden_roller_length
  (diameter : ℝ)
  (total_area : ℝ)
  (revolutions : ℕ)
  (pi : ℝ)
  (circumference : ℝ)
  (area_per_revolution : ℝ)
  (length : ℝ)
  (h1 : diameter = 1.4)
  (h2 : total_area = 44)
  (h3 : revolutions = 5)
  (h4 : pi = (22 / 7))
  (h5 : circumference = pi * diameter)
  (h6 : area_per_revolution = total_area / (revolutions : ℝ))
  (h7 : area_per_revolution = circumference * length) :
  length = 7 := by
  sorry

end garden_roller_length_l56_5613


namespace distance_travelled_l56_5623

theorem distance_travelled (t : ℝ) (h : 15 * t = 10 * t + 20) : 10 * t = 40 :=
by
  have ht : t = 4 := by linarith
  rw [ht]
  norm_num

end distance_travelled_l56_5623


namespace remainder_cd_mod_40_l56_5647

theorem remainder_cd_mod_40 (c d : ℤ) (hc : c % 80 = 75) (hd : d % 120 = 117) : (c + d) % 40 = 32 :=
by
  sorry

end remainder_cd_mod_40_l56_5647


namespace num_boys_in_circle_l56_5608

theorem num_boys_in_circle (n : ℕ) 
  (h : ∃ k, n = 2 * k ∧ k = 40 - 10) : n = 60 :=
by
  sorry

end num_boys_in_circle_l56_5608


namespace not_possible_to_get_105_single_stone_piles_l56_5637

noncomputable def piles : List Nat := [51, 49, 5]
def combine (a b : Nat) : Nat := a + b
def split (a : Nat) : List Nat := if a % 2 = 0 then [a / 2, a / 2] else [a]

theorem not_possible_to_get_105_single_stone_piles 
  (initial_piles : List Nat := piles) 
  (combine : Nat → Nat → Nat := combine) 
  (split : Nat → List Nat := split) :
  ¬ ∃ (final_piles : List Nat), final_piles.length = 105 ∧ (∀ n ∈ final_piles, n = 1) :=
by
  sorry

end not_possible_to_get_105_single_stone_piles_l56_5637


namespace goldfish_feeding_l56_5675

theorem goldfish_feeding (g : ℕ) (h : g = 8) : 4 * g = 32 :=
by
  sorry

end goldfish_feeding_l56_5675


namespace target_hit_probability_l56_5615

-- Define the probabilities of Person A and Person B hitting the target
def prob_A_hits := 0.8
def prob_B_hits := 0.7

-- Define the probability that the target is hit when both shoot independently at the same time
def prob_target_hit := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

theorem target_hit_probability : prob_target_hit = 0.94 := 
by
  sorry

end target_hit_probability_l56_5615


namespace trapezoid_area_l56_5635

variables (R₁ R₂ : ℝ)

theorem trapezoid_area (h_eq : h = 4 * R₁ * R₂ / (R₁ + R₂)) (mn_eq : mn = 2 * Real.sqrt (R₁ * R₂)) :
  S_ABCD = 8 * R₁ * R₂ * Real.sqrt (R₁ * R₂) / (R₁ + R₂) :=
sorry

end trapezoid_area_l56_5635


namespace tan_condition_then_expression_value_l56_5652

theorem tan_condition_then_expression_value (θ : ℝ) (h : Real.tan θ = 2) :
  (2 * Real.sin θ) / (Real.sin θ + 2 * Real.cos θ) = 1 :=
sorry

end tan_condition_then_expression_value_l56_5652


namespace find_f6_l56_5665

variable {R : Type*} [AddGroup R] [Semiring R]

def functional_equation (f : R → R) :=
∀ x y : R, f (x + y) = f x + f y

theorem find_f6 (f : ℝ → ℝ) (h1 : functional_equation f) (h2 : f 4 = 10) : f 6 = 10 :=
sorry

end find_f6_l56_5665


namespace correct_chart_for_percentage_representation_l56_5684

def bar_chart_characteristic := "easily shows the quantity"
def line_chart_characteristic := "shows the quantity and reflects the changes in quantity"
def pie_chart_characteristic := "reflects the relationship between a part and the whole"

def representation_requirement := "represents the percentage of students in each grade level in the fifth grade's physical education test scores out of the total number of students in the grade"

theorem correct_chart_for_percentage_representation : 
  (representation_requirement = pie_chart_characteristic) := 
by 
   -- The proof follows from the prior definition of characteristics.
   sorry

end correct_chart_for_percentage_representation_l56_5684


namespace min_y_in_quadratic_l56_5631

theorem min_y_in_quadratic (x : ℝ) : ∃ y : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ y', (y' = x^2 + 16 * x + 20) → y ≤ y' := 
sorry

end min_y_in_quadratic_l56_5631


namespace solution_interval_l56_5634

-- Define the differentiable function f over the interval (-∞, 0)
variable {f : ℝ → ℝ}
variable (hf : ∀ x < 0, HasDerivAt f (f' x) x)
variable (hx_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2)

-- Proof statement to show the solution interval
theorem solution_interval :
  {x : ℝ | (x + 2018)^2 * f (x + 2018) - 4 * f (-2) > 0} = {x | x < -2020} :=
sorry

end solution_interval_l56_5634


namespace number_is_43_l56_5699

theorem number_is_43 (m : ℕ) : (m > 30 ∧ m < 50) ∧ Nat.Prime m ∧ m % 12 = 7 ↔ m = 43 :=
by
  sorry

end number_is_43_l56_5699


namespace total_amount_paid_l56_5621

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l56_5621


namespace train_actual_speed_l56_5661
-- Import necessary libraries

-- Define the given conditions and question
def departs_time := 6
def planned_speed := 100
def scheduled_arrival_time := 18
def actual_arrival_time := 16
def distance (t₁ t₂ : ℕ) (s : ℕ) : ℕ := s * (t₂ - t₁)
def actual_speed (d t₁ t₂ : ℕ) : ℕ := d / (t₂ - t₁)

-- Proof problem statement
theorem train_actual_speed:
  actual_speed (distance departs_time scheduled_arrival_time planned_speed) departs_time actual_arrival_time = 120 := by sorry

end train_actual_speed_l56_5661


namespace roses_remain_unchanged_l56_5663

variable (initial_roses : ℕ) (initial_orchids : ℕ) (final_orchids : ℕ)

def unchanged_roses (roses_now : ℕ) : Prop :=
  roses_now = initial_roses

theorem roses_remain_unchanged :
  initial_roses = 13 → 
  initial_orchids = 84 → 
  final_orchids = 91 →
  ∀ (roses_now : ℕ), unchanged_roses initial_roses roses_now :=
by
  intros _ _ _ _
  simp [unchanged_roses]
  sorry

end roses_remain_unchanged_l56_5663


namespace competition_results_l56_5646

namespace Competition

-- Define the probabilities for each game
def prob_win_game_A : ℚ := 2 / 3
def prob_win_game_B : ℚ := 1 / 2

-- Define the probability of winning each project (best of five format)
def prob_win_project_A : ℚ := (8 / 27) + (8 / 27) + (16 / 81)
def prob_win_project_B : ℚ := (1 / 8) + (3 / 16) + (3 / 16)

-- Define the distribution of the random variable X (number of projects won by player A)
def P_X_0 : ℚ := (17 / 81) * (1 / 2)
def P_X_2 : ℚ := (64 / 81) * (1 / 2)
def P_X_1 : ℚ := 1 - P_X_0 - P_X_2

-- Define the mathematical expectation of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

-- Theorem stating the results
theorem competition_results :
  prob_win_project_A = 64 / 81 ∧
  prob_win_project_B = 1 / 2 ∧
  P_X_0 = 17 / 162 ∧
  P_X_1 = 81 / 162 ∧
  P_X_2 = 64 / 162 ∧
  E_X = 209 / 162 :=
by sorry

end Competition

end competition_results_l56_5646


namespace litter_collection_total_weight_l56_5670

/-- Gina collected 8 bags of litter: 5 bags of glass bottles weighing 7 pounds each and 3 bags of plastic waste weighing 4 pounds each. The 25 neighbors together collected 120 times as much glass as Gina and 80 times as much plastic as Gina. Prove that the total weight of all the collected litter is 5207 pounds. -/
theorem litter_collection_total_weight
  (glass_bags_gina : ℕ)
  (glass_weight_per_bag : ℕ)
  (plastic_bags_gina : ℕ)
  (plastic_weight_per_bag : ℕ)
  (neighbors_glass_multiplier : ℕ)
  (neighbors_plastic_multiplier : ℕ)
  (total_weight : ℕ)
  (h1 : glass_bags_gina = 5)
  (h2 : glass_weight_per_bag = 7)
  (h3 : plastic_bags_gina = 3)
  (h4 : plastic_weight_per_bag = 4)
  (h5 : neighbors_glass_multiplier = 120)
  (h6 : neighbors_plastic_multiplier = 80)
  (h_total_weight : total_weight = 5207) : total_weight = 
  glass_bags_gina * glass_weight_per_bag + 
  plastic_bags_gina * plastic_weight_per_bag + 
  neighbors_glass_multiplier * (glass_bags_gina * glass_weight_per_bag) + 
  neighbors_plastic_multiplier * (plastic_bags_gina * plastic_weight_per_bag) := 
by {
  /- Proof omitted -/
  sorry
}

end litter_collection_total_weight_l56_5670


namespace recruits_total_l56_5642

theorem recruits_total (P N D : ℕ) (total_recruits : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170)
  (h4 : (∃ x y, (x = 50) ∧ (y = 100) ∧ (x = 4 * y))
        ∨ (∃ x z, (x = 50) ∧ (z = 170) ∧ (x = 4 * z))
        ∨ (∃ y z, (y = 100) ∧ (z = 170) ∧ (y = 4 * z))) : 
  total_recruits = 211 :=
by
  sorry

end recruits_total_l56_5642


namespace M_eq_N_l56_5627

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r}

theorem M_eq_N : M = N :=
by
  sorry

end M_eq_N_l56_5627


namespace mike_games_l56_5636

theorem mike_games (init_money spent_money game_cost : ℕ) (h1 : init_money = 42) (h2 : spent_money = 10) (h3 : game_cost = 8) :
  (init_money - spent_money) / game_cost = 4 :=
by
  sorry

end mike_games_l56_5636


namespace basketball_court_length_difference_l56_5606

theorem basketball_court_length_difference :
  ∃ (l w : ℕ), l = 31 ∧ w = 17 ∧ l - w = 14 := by
  sorry

end basketball_court_length_difference_l56_5606


namespace share_per_person_is_135k_l56_5605

noncomputable def calculate_share : ℝ :=
  (0.90 * (500000 * 1.20)) / 4

theorem share_per_person_is_135k : calculate_share = 135000 :=
by
  sorry

end share_per_person_is_135k_l56_5605


namespace y_coordinate_midpoint_l56_5639

theorem y_coordinate_midpoint : 
  let L : (ℝ → ℝ) := λ x => x - 1
  let P : (ℝ → ℝ) := λ y => 8 * (y^2)
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    P (L x₁) = y₁ ∧ P (L x₂) = y₂ ∧ 
    L x₁ = y₁ ∧ L x₂ = y₂ ∧ 
    x₁ + x₂ = 10 ∧ y₁ + y₂ = 8 ∧
    (y₁ + y₂) / 2 = 4 := sorry

end y_coordinate_midpoint_l56_5639


namespace quadratic_equation_solution_l56_5616

theorem quadratic_equation_solution : ∀ x : ℝ, x^2 - 9 = 0 ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end quadratic_equation_solution_l56_5616


namespace integer_pairs_satisfying_condition_l56_5659

theorem integer_pairs_satisfying_condition :
  { (m, n) : ℤ × ℤ | ∃ k : ℤ, (n^3 + 1) = k * (m * n - 1) } =
  { (1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2) } :=
sorry

end integer_pairs_satisfying_condition_l56_5659


namespace product_of_two_numbers_l56_5618

theorem product_of_two_numbers (a b : ℝ) 
  (h1 : a - b = 2 * k)
  (h2 : a + b = 8 * k)
  (h3 : 2 * a * b = 30 * k) : a * b = 15 :=
by
  sorry

end product_of_two_numbers_l56_5618


namespace distance_from_reflected_point_l56_5641

theorem distance_from_reflected_point
  (P : ℝ × ℝ) (P' : ℝ × ℝ)
  (hP : P = (3, 2))
  (hP' : P' = (3, -2))
  : dist P P' = 4 := sorry

end distance_from_reflected_point_l56_5641


namespace diagonals_of_hexadecagon_l56_5658

-- Define the function to calculate number of diagonals in a convex polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- State the theorem for the number of diagonals in a convex hexadecagon
theorem diagonals_of_hexadecagon : num_diagonals 16 = 104 := by
  -- sorry is used to indicate the proof is skipped
  sorry

end diagonals_of_hexadecagon_l56_5658


namespace eq_of_nonzero_real_x_l56_5691

theorem eq_of_nonzero_real_x (x : ℝ) (hx : x ≠ 0) (a b : ℝ) (ha : a = 9) (hb : b = 18) :
  ((a * x) ^ 10 = (b * x) ^ 5) → x = 2 / 9 :=
by
  sorry

end eq_of_nonzero_real_x_l56_5691


namespace dot_product_value_l56_5649

-- Define vectors a and b, and the condition of their linear combination
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def a : Vector2D := ⟨-1, 2⟩
def b (m : ℝ) : Vector2D := ⟨m, 1⟩

-- Define the condition that vector a + 2b is parallel to 2a - b
def parallel (v w : Vector2D) : Prop := ∃ k : ℝ, v.x = k * w.x ∧ v.y = k * w.y

def vector_add (v w : Vector2D) : Vector2D := ⟨v.x + w.x, v.y + w.y⟩
def scalar_mul (c : ℝ) (v : Vector2D) : Vector2D := ⟨c * v.x, c * v.y⟩

-- Dot product definition
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

-- The theorem to prove
theorem dot_product_value (m : ℝ)
  (h : parallel (vector_add a (scalar_mul 2 (b m))) (vector_add (scalar_mul 2 a) (scalar_mul (-1) (b m)))) :
  dot_product a (b m) = 5 / 2 :=
sorry

end dot_product_value_l56_5649


namespace math_problem_l56_5673

open Set

noncomputable def A : Set ℝ := { x | x < 1 }
noncomputable def B : Set ℝ := { x | x * (x - 1) > 6 }
noncomputable def C (m : ℝ) : Set ℝ := { x | -1 + m < x ∧ x < 2 * m }

theorem math_problem (m : ℝ) (m_range : C m ≠ ∅) :
  (A ∪ B = { x | x > 3 ∨ x < 1 }) ∧
  (A ∩ (compl B) = { x | -2 ≤ x ∧ x < 1 }) ∧
  (-1 < m ∧ m ≤ 0.5) :=
by
  sorry

end math_problem_l56_5673


namespace min_value_g_range_of_m_l56_5650

section
variable (x : ℝ)
noncomputable def g (x : ℝ) := Real.exp x - x

theorem min_value_g :
  (∀ x : ℝ, g x ≥ g 0) ∧ g 0 = 1 := 
by 
  sorry

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / g x > x) → m < Real.log 2 ^ 2 := 
by 
  sorry
end

end min_value_g_range_of_m_l56_5650


namespace original_grain_amount_l56_5674

def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

theorem original_grain_amount : grain_spilled + grain_remaining = 50870 :=
by
  sorry

end original_grain_amount_l56_5674


namespace group_C_both_axis_and_central_l56_5651

def is_axisymmetric (shape : Type) : Prop := sorry
def is_centrally_symmetric (shape : Type) : Prop := sorry

def square : Type := sorry
def rhombus : Type := sorry
def rectangle : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry

def group_A := [square, rhombus, rectangle, parallelogram]
def group_B := [equilateral_triangle, square, rhombus, rectangle]
def group_C := [square, rectangle, rhombus]
def group_D := [parallelogram, square, isosceles_triangle]

def all_axisymmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_axisymmetric shape

def all_centrally_symmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_centrally_symmetric shape

theorem group_C_both_axis_and_central :
  (all_axisymmetric group_C ∧ all_centrally_symmetric group_C) ∧
  (∀ (group : List Type), (all_axisymmetric group ∧ all_centrally_symmetric group) →
    group = group_C) :=
by sorry

end group_C_both_axis_and_central_l56_5651


namespace friend_reading_time_l56_5654

-- Define the conditions
def my_reading_time : ℝ := 1.5 * 60 -- 1.5 hours converted to minutes
def friend_speed_multiplier : ℝ := 5 -- Friend reads 5 times faster than I do
def distraction_time : ℝ := 15 -- Friend is distracted for 15 minutes

-- Define the time taken for my friend to read the book accounting for distraction
theorem friend_reading_time :
  (my_reading_time / friend_speed_multiplier) + distraction_time = 33 := by
  sorry

end friend_reading_time_l56_5654


namespace people_at_the_beach_l56_5679

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l56_5679


namespace Kenneth_money_left_l56_5687

def initial_amount : ℕ := 50
def number_of_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def number_of_bottles : ℕ := 2
def cost_per_bottle : ℕ := 1

-- This theorem states that Kenneth has $44 left after his purchases.
theorem Kenneth_money_left : initial_amount - (number_of_baguettes * cost_per_baguette + number_of_bottles * cost_per_bottle) = 44 := by
  sorry

end Kenneth_money_left_l56_5687


namespace compute_expression_l56_5640

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := 
by
  sorry

end compute_expression_l56_5640


namespace students_per_bus_l56_5666

def total_students : ℕ := 360
def number_of_buses : ℕ := 8

theorem students_per_bus : total_students / number_of_buses = 45 :=
by
  sorry

end students_per_bus_l56_5666


namespace a_b_work_days_l56_5694

-- Definitions:
def work_days_a_b_together := 40
def work_days_a_alone := 12
def remaining_work_days_with_a := 9

-- Statement to be proven:
theorem a_b_work_days (x : ℕ) 
  (h1 : ∀ W : ℕ, W / work_days_a_b_together + remaining_work_days_with_a * (W / work_days_a_alone) = W) :
  x = 10 :=
sorry

end a_b_work_days_l56_5694


namespace circles_C1_C2_intersect_C1_C2_l56_5655

noncomputable def center1 : (ℝ × ℝ) := (5, 3)
noncomputable def radius1 : ℝ := 3

noncomputable def center2 : (ℝ × ℝ) := (2, -1)
noncomputable def radius2 : ℝ := Real.sqrt 14

noncomputable def distance : ℝ := Real.sqrt ((5 - 2)^2 + (3 + 1)^2)

def circles_intersect : Prop :=
  radius2 - radius1 < distance ∧ distance < radius2 + radius1

theorem circles_C1_C2_intersect_C1_C2 : circles_intersect :=
by
  -- The proof of this theorem is to be worked out using the given conditions and steps.
  sorry

end circles_C1_C2_intersect_C1_C2_l56_5655


namespace plan_Y_cheaper_l56_5696

theorem plan_Y_cheaper (y : ℤ) :
  (15 * (y : ℚ) > 2500 + 8 * (y : ℚ)) ↔ y > 358 :=
by
  sorry

end plan_Y_cheaper_l56_5696


namespace complex_number_solution_l56_5676

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i * i = -1) (h : z + z * i = 1 + 5 * i) : z = 3 + 2 * i :=
sorry

end complex_number_solution_l56_5676


namespace bushes_needed_for_octagon_perimeter_l56_5620

theorem bushes_needed_for_octagon_perimeter
  (side_length : ℝ) (spacing : ℝ)
  (octagonal : ∀ (s : ℝ), s = 8 → 8 * s = 64)
  (spacing_condition : ∀ (p : ℝ), p = 64 → p / spacing = 32) :
  spacing = 2 → side_length = 8 → (64 / 2 = 32) := 
by
  sorry

end bushes_needed_for_octagon_perimeter_l56_5620


namespace tom_watching_days_l56_5611

noncomputable def total_watch_time : ℕ :=
  30 * 22 + 28 * 25 + 27 * 29 + 20 * 31 + 25 * 27 + 20 * 35

noncomputable def daily_watch_time : ℕ := 2 * 60

theorem tom_watching_days : ⌈(total_watch_time / daily_watch_time : ℚ)⌉ = 35 := by
  sorry

end tom_watching_days_l56_5611


namespace geometric_sequence_problem_l56_5630

theorem geometric_sequence_problem (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : ∀ n, a (n + 1) = r * a n) 
  (h_cond: a 4 + a 6 = 8) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
  sorry

end geometric_sequence_problem_l56_5630


namespace range_of_m_l56_5604

variable (m : ℝ)

def p : Prop := ∀ x : ℝ, 2 * x > m * (x^2 + 1)
def q : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 - m - 1 = 0

theorem range_of_m (hp : p m) (hq : q m) : -2 ≤ m ∧ m < -1 :=
sorry

end range_of_m_l56_5604


namespace total_bananas_in_collection_l56_5622

theorem total_bananas_in_collection (g b T : ℕ) (h₀ : g = 196) (h₁ : b = 2) (h₂ : T = 392) : g * b = T :=
by
  sorry

end total_bananas_in_collection_l56_5622


namespace mean_eq_median_of_set_l56_5624

theorem mean_eq_median_of_set (x : ℕ) (hx : 0 < x) :
  let s := [1, 2, 4, 5, x]
  let mean := (1 + 2 + 4 + 5 + x) / 5
  let median := if x ≤ 2 then 2 else if x ≤ 4 then x else 4
  mean = median → (x = 3 ∨ x = 8) :=
by {
  sorry
}

end mean_eq_median_of_set_l56_5624


namespace maximum_value_minimum_value_l56_5656

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def check_digits (N M : ℕ) (a b c d e f g h : ℕ) : Prop :=
  N = 1000 * a + 100 * b + 10 * c + d ∧
  M = 1000 * e + 100 * f + 10 * g + h ∧
  a ≠ e ∧
  b ≠ f ∧
  c ≠ g ∧
  d ≠ h ∧
  a ≠ f ∧
  a ≠ g ∧
  a ≠ h ∧
  b ≠ e ∧
  b ≠ g ∧
  b ≠ h ∧
  c ≠ e ∧
  c ≠ f ∧
  c ≠ h ∧
  d ≠ e ∧
  d ≠ f ∧
  d ≠ g

theorem maximum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 15000 :=
by
  intros
  sorry

theorem minimum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 4998 :=
by
  intros
  sorry

end maximum_value_minimum_value_l56_5656


namespace limit_calculation_l56_5693

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  (Real.exp (-1) * Real.exp 0 - Real.exp (-1) * Real.exp 0) / 0 = -3 / Real.exp 1 := by
  sorry

end limit_calculation_l56_5693


namespace period_of_sin3x_plus_cos3x_l56_5612

noncomputable def period_of_trig_sum (x : ℝ) : ℝ := 
  let y := (fun x => Real.sin (3 * x) + Real.cos (3 * x))
  (2 * Real.pi) / 3

theorem period_of_sin3x_plus_cos3x : (fun x => Real.sin (3 * x) + Real.cos (3 * x)) =
  (fun x => Real.sin (3 * (x + period_of_trig_sum x)) + Real.cos (3 * (x + period_of_trig_sum x))) :=
by
  sorry

end period_of_sin3x_plus_cos3x_l56_5612


namespace range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l56_5678

-- Problem (1)
theorem range_of_x_in_tight_sequence (a : ℕ → ℝ) (x : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = x ∧ a 4 = 4 → 2 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (2)
theorem arithmetic_tight_sequence (a : ℕ → ℝ) (a1 d : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  ∀ n : ℕ, a n = a1 + ↑n * d → 0 < d ∧ d ≤ a1 →
  ∀ n : ℕ, 1 / 2 ≤ (a (n + 1) / a n) ∧ (a (n + 1) / a n) ≤ 2 :=
sorry

-- Problem (3)
theorem geometric_tight_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (h_seq : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2)
(S : ℕ → ℝ) (h_sum_seq : ∀ n : ℕ, 1 / 2 ≤ S (n + 1) / S n ∧ S (n + 1) / S n ≤ 2) :
  (∀ n : ℕ, a n = a1 * q ^ n ∧ S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) → 
  1 / 2 ≤ q ∧ q ≤ 1 :=
sorry

end range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l56_5678


namespace whitney_greatest_sets_l56_5662

-- Define the conditions: Whitney has 4 T-shirts and 20 buttons.
def num_tshirts := 4
def num_buttons := 20

-- The problem statement: Prove that the greatest number of sets Whitney can make is 4.
theorem whitney_greatest_sets : Nat.gcd num_tshirts num_buttons = 4 := by
  sorry

end whitney_greatest_sets_l56_5662


namespace thomas_savings_years_l56_5683

def weekly_allowance : ℕ := 50
def weekly_coffee_shop_earning : ℕ := 9 * 30
def weekly_spending : ℕ := 35
def car_cost : ℕ := 15000
def additional_amount_needed : ℕ := 2000
def weeks_in_a_year : ℕ := 52

def first_year_savings : ℕ := weeks_in_a_year * (weekly_allowance - weekly_spending)
def second_year_savings : ℕ := weeks_in_a_year * (weekly_coffee_shop_earning - weekly_spending)

noncomputable def total_savings_needed : ℕ := car_cost - additional_amount_needed

theorem thomas_savings_years : 
  first_year_savings + second_year_savings = total_savings_needed → 2 = 2 :=
by
  sorry

end thomas_savings_years_l56_5683


namespace x7_value_l56_5689

theorem x7_value
  (x : ℕ → ℕ)
  (h1 : x 6 = 144)
  (h2 : ∀ n, 1 ≤ n ∧ n ≤ 4 → x (n + 3) = x (n + 2) * (x (n + 1) + x n))
  (h3 : ∀ m, m < 1 → 0 < x m) : x 7 = 3456 :=
by
  sorry

end x7_value_l56_5689


namespace non_shaded_area_l56_5698

theorem non_shaded_area (s : ℝ) (hex_area : ℝ) (tri_area : ℝ) (non_shaded_area : ℝ) :
  s = 12 →
  hex_area = (3 * Real.sqrt 3 / 2) * s^2 →
  tri_area = (Real.sqrt 3 / 4) * (2 * s)^2 →
  non_shaded_area = hex_area - tri_area →
  non_shaded_area = 288 * Real.sqrt 3 :=
by
  intros hs hhex htri hnon
  sorry

end non_shaded_area_l56_5698


namespace camilla_blueberry_jelly_beans_l56_5648

theorem camilla_blueberry_jelly_beans (b c : ℕ) 
  (h1 : b = 3 * c)
  (h2 : b - 20 = 2 * (c - 5)) : 
  b = 30 := 
sorry

end camilla_blueberry_jelly_beans_l56_5648


namespace intersection_A_B_l56_5645

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l56_5645


namespace least_possible_value_of_z_l56_5686

theorem least_possible_value_of_z (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : y - x > 5) 
  (h2 : z - x = 9) : 
  z = 11 := 
by
  sorry

end least_possible_value_of_z_l56_5686


namespace solution_of_two_quadratics_l56_5671

theorem solution_of_two_quadratics (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 := 
by 
  sorry

end solution_of_two_quadratics_l56_5671


namespace inequality_solution_l56_5632

theorem inequality_solution (x : ℝ) : 
  (3 / 20 + abs (2 * x - 5 / 40) < 9 / 40) → (1 / 40 < x ∧ x < 1 / 10) :=
by
  sorry

end inequality_solution_l56_5632


namespace minimal_storing_capacity_required_l56_5667

theorem minimal_storing_capacity_required (k : ℕ) (h1 : k > 0)
    (bins : ℕ → ℕ → ℕ → Prop)
    (h_initial : bins 0 0 0)
    (h_laundry_generated : ∀ n, bins (10 * n) (10 * n) (10 * n))
    (h_heaviest_bin_emptied : ∀ n r b g, (r + b + g = 10 * n) → max r (max b g) + 10 * n - max r (max b g) = 10 * n)
    : ∀ (capacity : ℕ), capacity = 25 :=
sorry

end minimal_storing_capacity_required_l56_5667


namespace volume_of_pyramid_l56_5669

-- Define the conditions
def pyramid_conditions : Prop :=
  ∃ (s h : ℝ),
  s^2 = 256 ∧
  ∃ (h_A h_C h_B : ℝ),
  ((∃ h_A, 128 = 1/2 * s * h_A) ∧
  (∃ h_C, 112 = 1/2 * s * h_C) ∧
  (∃ h_B, 96 = 1/2 * s * h_B)) ∧
  h^2 + (s/2)^2 = h_A^2 ∧
  h^2 = 256 - (s/2)^2 ∧
  h^2 + (s/4)^2 = h_B^2

-- Define the theorem
theorem volume_of_pyramid :
  pyramid_conditions → 
  ∃ V : ℝ, V = 682.67 * Real.sqrt 3 :=
sorry

end volume_of_pyramid_l56_5669


namespace movie_box_office_revenue_l56_5633

variable (x : ℝ)

theorem movie_box_office_revenue (h : 300 + 300 * (1 + x) + 300 * (1 + x)^2 = 1000) :
  3 + 3 * (1 + x) + 3 * (1 + x)^2 = 10 :=
by
  sorry

end movie_box_office_revenue_l56_5633


namespace relation_between_x_and_y_l56_5664

variable (t : ℝ)
variable (x : ℝ := t ^ (2 / (t - 1))) (y : ℝ := t ^ ((t + 1) / (t - 1)))

theorem relation_between_x_and_y (h1 : t > 0) (h2 : t ≠ 1) : y ^ (1 / x) = x ^ y :=
by sorry

end relation_between_x_and_y_l56_5664


namespace minimize_AB_l56_5672

-- Definition of the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 3 = 0

-- Definition of the point P
def P : ℝ × ℝ := (-1, 2)

-- Definition of the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- The goal is to prove that line_l is the line through P minimizing |AB|
theorem minimize_AB : 
  ∀ l : ℝ → ℝ → Prop, 
  (∀ x y, l x y → (∃ a b, circleC a b ∧ l a b ∧ circleC x y ∧ l x y ∧ (x ≠ a ∨ y ≠ b)) → False) 
  → l = line_l :=
by
  sorry

end minimize_AB_l56_5672


namespace smallest_y_value_l56_5682

-- Define the original equation
def original_eq (y : ℝ) := 3 * y^2 + 36 * y - 90 = y * (y + 18)

-- Define the problem statement
theorem smallest_y_value : ∃ (y : ℝ), original_eq y ∧ y = -15 :=
by
  sorry

end smallest_y_value_l56_5682


namespace f_2008_eq_zero_l56_5653

noncomputable def f : ℝ → ℝ := sorry

-- f is odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- f satisfies f(x + 2) = -f(x)
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x

theorem f_2008_eq_zero : f 2008 = 0 :=
by
  sorry

end f_2008_eq_zero_l56_5653


namespace percentage_of_first_pay_cut_l56_5681

theorem percentage_of_first_pay_cut
  (x : ℝ)
  (h1 : ∃ y z w : ℝ, y = 1 - x/100 ∧ z = 0.86 ∧ w = 0.82 ∧ y * z * w = 0.648784):
  x = 8.04 := by
-- The proof will be added here, this is just the statement
sorry

end percentage_of_first_pay_cut_l56_5681


namespace litter_patrol_total_pieces_l56_5609

theorem litter_patrol_total_pieces :
  let glass_bottles := 25
  let aluminum_cans := 18
  let plastic_bags := 12
  let paper_cups := 7
  let cigarette_packs := 5
  let discarded_face_masks := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + discarded_face_masks = 70 :=
by
  sorry

end litter_patrol_total_pieces_l56_5609


namespace comp_figure_perimeter_l56_5607

-- Given conditions
def side_length_square : ℕ := 2
def side_length_triangle : ℕ := 1
def number_of_squares : ℕ := 4
def number_of_triangles : ℕ := 3

-- Define the perimeter calculation
def perimeter_of_figure : ℕ :=
  let perimeter_squares := (2 * (number_of_squares - 2) + 2 * 2 + 2 * 1) * side_length_square
  let perimeter_triangles := number_of_triangles * side_length_triangle
  perimeter_squares + perimeter_triangles

-- Target theorem
theorem comp_figure_perimeter : perimeter_of_figure = 17 := by
  sorry

end comp_figure_perimeter_l56_5607


namespace cube_surface_area_sum_of_edges_l56_5690

noncomputable def edge_length (sum_of_edges : ℝ) (num_of_edges : ℝ) : ℝ :=
  sum_of_edges / num_of_edges

noncomputable def surface_area (edge_length : ℝ) : ℝ :=
  6 * edge_length ^ 2

theorem cube_surface_area_sum_of_edges (sum_of_edges : ℝ) (num_of_edges : ℝ) (expected_area : ℝ) :
  num_of_edges = 12 → sum_of_edges = 72 → surface_area (edge_length sum_of_edges num_of_edges) = expected_area :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cube_surface_area_sum_of_edges_l56_5690


namespace distance_between_P_and_Q_l56_5617

theorem distance_between_P_and_Q : 
  let initial_speed := 40  -- Speed in kmph
  let increment := 20      -- Speed increment in kmph after every 12 minutes
  let segment_duration := 12 / 60 -- Duration of each segment in hours (12 minutes in hours)
  let total_duration := 48 / 60    -- Total duration in hours (48 minutes in hours)
  let total_segments := total_duration / segment_duration -- Number of segments
  (total_segments = 4) ∧ 
  (∀ n : ℕ, n ≥ 0 → n < total_segments → 
    let speed := initial_speed + n * increment
    let distance := speed * segment_duration
    distance = speed * (12 / 60)) 
  → (40 * (12 / 60) + 60 * (12 / 60) + 80 * (12 / 60) + 100 * (12 / 60)) = 56 :=
by
  sorry

end distance_between_P_and_Q_l56_5617


namespace probability_visible_l56_5697

-- Definitions of the conditions
def lap_time_sarah : ℕ := 120
def lap_time_sam : ℕ := 100
def start_to_photo_min : ℕ := 15
def start_to_photo_max : ℕ := 16
def photo_fraction : ℚ := 1/3
def shadow_start_interval : ℕ := 45
def shadow_duration : ℕ := 15

-- The theorem to prove
theorem probability_visible :
  let total_time := 60
  let valid_overlap_time := 13.33
  valid_overlap_time / total_time = 1333 / 6000 :=
by {
  sorry
}

end probability_visible_l56_5697


namespace range_of_a_l56_5692

theorem range_of_a (a b : ℝ) (h1 : 0 ≤ a - b ∧ a - b ≤ 1) (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1 / 2 ≤ a ∧ a ≤ 5 / 2 := 
sorry

end range_of_a_l56_5692


namespace solution_set_f_l56_5610

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^(x - 1) - 2 else 2^(1 - x) - 2

theorem solution_set_f (x : ℝ) : 
  (1 ≤ x ∧ x ≤ 3) ↔ (f (x - 1) ≤ 0) :=
sorry

end solution_set_f_l56_5610


namespace chocolate_ratio_l56_5603

theorem chocolate_ratio (N A : ℕ) (h1 : N = 10) (h2 : A - 5 = N + 15) : A / N = 3 :=
by {
  sorry
}

end chocolate_ratio_l56_5603


namespace second_number_is_72_l56_5680

-- Define the necessary variables and conditions
variables (x y : ℕ)
variables (h_first_num : x = 48)
variables (h_ratio : 48 / 8 = x / y)
variables (h_LCM : Nat.lcm x y = 432)

-- State the problem as a theorem
theorem second_number_is_72 : y = 72 :=
by
  sorry

end second_number_is_72_l56_5680
