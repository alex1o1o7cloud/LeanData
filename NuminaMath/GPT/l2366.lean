import Mathlib

namespace slope_of_line_with_sine_of_angle_l2366_236636

theorem slope_of_line_with_sine_of_angle (α : ℝ) 
  (hα₁ : 0 ≤ α) (hα₂ : α < Real.pi) 
  (h_sin : Real.sin α = Real.sqrt 3 / 2) : 
  ∃ k : ℝ, k = Real.tan α ∧ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end slope_of_line_with_sine_of_angle_l2366_236636


namespace simplify_expression_1_simplify_expression_2_l2366_236658

-- Problem 1
theorem simplify_expression_1 (a b : ℤ) : a + 2 * b + 3 * a - 2 * b = 4 * a :=
by
  sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℤ) (h_m : m = 2) (h_n : n = 1) :
  (2 * m ^ 2 - 3 * m * n + 8) - (5 * m * n - 4 * m ^ 2 + 8) = 8 :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l2366_236658


namespace total_snow_volume_l2366_236677

-- Definitions and conditions set up from part (a)
def driveway_length : ℝ := 30
def driveway_width : ℝ := 3
def section1_length : ℝ := 10
def section1_depth : ℝ := 1
def section2_length : ℝ := driveway_length - section1_length
def section2_depth : ℝ := 0.5

-- The theorem corresponding to part (c)
theorem total_snow_volume : 
  (section1_length * driveway_width * section1_depth) +
  (section2_length * driveway_width * section2_depth) = 60 :=
by 
  -- Proof is omitted as required
  sorry

end total_snow_volume_l2366_236677


namespace simplify_fraction_expression_l2366_236644

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a^3 - b^3 = a - b)

theorem simplify_fraction_expression : (a / b) + (b / a) + (1 / (a * b)) = 2 := by
  sorry

end simplify_fraction_expression_l2366_236644


namespace geometric_sequence_frac_l2366_236633

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
variable (h_decreasing : ∀ n, a (n+1) < a n)
variable (h1 : a 2 * a 8 = 6)
variable (h2 : a 4 + a 6 = 5)

theorem geometric_sequence_frac (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
                                (h_decreasing : ∀ n, a (n+1) < a n)
                                (h1 : a 2 * a 8 = 6)
                                (h2 : a 4 + a 6 = 5) :
                                a 3 / a 7 = 9 / 4 :=
by sorry

end geometric_sequence_frac_l2366_236633


namespace action_figure_prices_l2366_236638

noncomputable def prices (x y z w : ℝ) : Prop :=
  12 * x + 8 * y + 5 * z + 10 * w = 220 ∧
  x / 4 = y / 3 ∧
  x / 4 = z / 2 ∧
  x / 4 = w / 1

theorem action_figure_prices :
  ∃ x y z w : ℝ, prices x y z w ∧
    x = 220 / 23 ∧
    y = (3 / 4) * (220 / 23) ∧
    z = (1 / 2) * (220 / 23) ∧
    w = (1 / 4) * (220 / 23) :=
  sorry

end action_figure_prices_l2366_236638


namespace sin_cos_ratio_l2366_236687

open Real

theorem sin_cos_ratio
  (θ : ℝ)
  (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) :
  sin θ * cos θ = 3 / 10 := 
by
  sorry

end sin_cos_ratio_l2366_236687


namespace compute_expression_l2366_236602

variable {R : Type*} [LinearOrderedField R]

theorem compute_expression (r s t : R)
  (h_eq_root: ∀ x, x^3 - 4 * x^2 + 4 * x - 6 = 0)
  (h1: r + s + t = 4)
  (h2: r * s + r * t + s * t = 4)
  (h3: r * s * t = 6) :
  r * s / t + s * t / r + t * r / s = -16 / 3 :=
sorry

end compute_expression_l2366_236602


namespace distinct_values_of_b_l2366_236699

theorem distinct_values_of_b : ∃ b_list : List ℝ, b_list.length = 8 ∧ ∀ b ∈ b_list, ∃ p q : ℤ, p + q = b ∧ p * q = 8 * b :=
by
  sorry

end distinct_values_of_b_l2366_236699


namespace nathan_banana_payment_l2366_236655

theorem nathan_banana_payment
  (bunches_8 : ℕ)
  (cost_per_bunch_8 : ℝ)
  (bunches_7 : ℕ)
  (cost_per_bunch_7 : ℝ)
  (discount : ℝ)
  (total_payment : ℝ) :
  bunches_8 = 6 →
  cost_per_bunch_8 = 2.5 →
  bunches_7 = 5 →
  cost_per_bunch_7 = 2.2 →
  discount = 0.10 →
  total_payment = 6 * 2.5 + 5 * 2.2 - 0.10 * (6 * 2.5 + 5 * 2.2) →
  total_payment = 23.40 :=
by
  intros
  sorry

end nathan_banana_payment_l2366_236655


namespace lcm_150_294_l2366_236616

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end lcm_150_294_l2366_236616


namespace probability_club_then_spade_l2366_236626

/--
   Two cards are dealt at random from a standard deck of 52 cards.
   Prove that the probability that the first card is a club (♣) and the second card is a spade (♠) is 13/204.
-/
theorem probability_club_then_spade :
  let total_cards := 52
  let clubs := 13
  let spades := 13
  let first_card_club_prob := (clubs : ℚ) / total_cards
  let second_card_spade_prob := (spades : ℚ) / (total_cards - 1)
  first_card_club_prob * second_card_spade_prob = 13 / 204 :=
by
  sorry

end probability_club_then_spade_l2366_236626


namespace remainder_of_122_div_20_l2366_236662

theorem remainder_of_122_div_20 :
  (∃ (q r : ℕ), 122 = 20 * q + r ∧ r < 20 ∧ q = 6) →
  r = 2 :=
by
  sorry

end remainder_of_122_div_20_l2366_236662


namespace max_sum_non_zero_nats_l2366_236683

theorem max_sum_non_zero_nats (O square : ℕ) (hO : O ≠ 0) (hsquare : square ≠ 0) :
  (O / 11 < 7 / square) ∧ (7 / square < 4 / 5) → O + square = 77 :=
by 
  sorry -- Proof omitted as requested

end max_sum_non_zero_nats_l2366_236683


namespace amount_of_sugar_l2366_236669

-- Let ratio_sugar_flour be the ratio of sugar to flour.
def ratio_sugar_flour : ℕ := 10

-- Let flour be the amount of flour used in ounces.
def flour : ℕ := 5

-- Let sugar be the amount of sugar used in ounces.
def sugar (ratio_sugar_flour : ℕ) (flour : ℕ) : ℕ := ratio_sugar_flour * flour

-- The proof goal: given the conditions, prove that the amount of sugar used is 50 ounces.
theorem amount_of_sugar (h_ratio : ratio_sugar_flour = 10) (h_flour : flour = 5) : sugar ratio_sugar_flour flour = 50 :=
by
  -- Proof omitted.
  sorry
 
end amount_of_sugar_l2366_236669


namespace absent_children_l2366_236630

theorem absent_children (A : ℕ) (h1 : 2 * 610 = (610 - A) * 4) : A = 305 := 
by sorry

end absent_children_l2366_236630


namespace total_dress_designs_l2366_236615

def num_colors := 5
def num_patterns := 6
def num_sizes := 3

theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 :=
by
  sorry

end total_dress_designs_l2366_236615


namespace cost_of_playing_cards_l2366_236667

theorem cost_of_playing_cards 
  (allowance_each : ℕ)
  (combined_allowance : ℕ)
  (sticker_box_cost : ℕ)
  (number_of_sticker_packs : ℕ)
  (number_of_packs_Dora_got : ℕ)
  (cost_of_playing_cards : ℕ)
  (h1 : allowance_each = 9)
  (h2 : combined_allowance = allowance_each * 2)
  (h3 : sticker_box_cost = 2)
  (h4 : number_of_packs_Dora_got = 2)
  (h5 : number_of_sticker_packs = number_of_packs_Dora_got * 2)
  (h6 : combined_allowance - number_of_sticker_packs * sticker_box_cost = cost_of_playing_cards) :
  cost_of_playing_cards = 10 :=
sorry

end cost_of_playing_cards_l2366_236667


namespace black_and_blue_lines_l2366_236666

-- Definition of given conditions
def grid_size : ℕ := 50
def total_points : ℕ := grid_size * grid_size
def blue_points : ℕ := 1510
def blue_edge_points : ℕ := 110
def red_segments : ℕ := 947
def corner_points : ℕ := 4

-- Calculations based on conditions
def red_points : ℕ := total_points - blue_points

def edge_points (size : ℕ) : ℕ := (size - 1) * 4
def non_corner_edge_points (edge : ℕ) : ℕ := edge - corner_points

-- Math translation
noncomputable def internal_red_points : ℕ := red_points - corner_points - (edge_points grid_size - blue_edge_points)
noncomputable def connections_from_red_points : ℕ :=
  corner_points * 2 + (non_corner_edge_points (edge_points grid_size) - blue_edge_points) * 3 + internal_red_points * 4

noncomputable def adjusted_red_lines : ℕ := red_segments * 2
noncomputable def black_lines : ℕ := connections_from_red_points - adjusted_red_lines

def total_lines (size : ℕ) : ℕ := (size - 1) * size + (size - 1) * size
noncomputable def blue_lines : ℕ := total_lines grid_size - red_segments - black_lines

-- The theorem to be proven
theorem black_and_blue_lines :
  (black_lines = 1972) ∧ (blue_lines = 1981) :=
by
  sorry

end black_and_blue_lines_l2366_236666


namespace simplify_and_evaluate_expr_l2366_236614

theorem simplify_and_evaluate_expr :
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 :=
by
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  sorry

end simplify_and_evaluate_expr_l2366_236614


namespace vision_statistics_l2366_236674

noncomputable def average (values : List ℝ) : ℝ := (List.sum values) / (List.length values)

noncomputable def variance (values : List ℝ) : ℝ :=
  let mean := average values
  (List.sum (values.map (λ x => (x - mean) ^ 2))) / (List.length values)

def classA_visions : List ℝ := [4.3, 5.1, 4.6, 4.1, 4.9]
def classB_visions : List ℝ := [5.1, 4.9, 4.0, 4.0, 4.5]

theorem vision_statistics :
  average classA_visions = 4.6 ∧
  average classB_visions = 4.5 ∧
  variance classA_visions = 0.136 ∧
  (let count := List.length classB_visions
   let total := count.choose 2
   let favorable := 3  -- (5.1, 4.5), (5.1, 4.9), (4.9, 4.5)
   7 / 10 = 1 - (favorable / total)) :=
by
  sorry

end vision_statistics_l2366_236674


namespace correct_operation_l2366_236682

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^4 ≠ a^6) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  ((a^2 * b)^3 = a^6 * b^3) ∧
  (a^6 / a^6 ≠ a) :=
by
  sorry

end correct_operation_l2366_236682


namespace combined_percentage_grade4_l2366_236622

-- Definitions based on the given conditions
def Pinegrove_total_students : ℕ := 120
def Maplewood_total_students : ℕ := 180

def Pinegrove_grade4_percentage : ℕ := 10
def Maplewood_grade4_percentage : ℕ := 20

theorem combined_percentage_grade4 :
  let combined_total_students := Pinegrove_total_students + Maplewood_total_students
  let Pinegrove_grade4_students := Pinegrove_grade4_percentage * Pinegrove_total_students / 100
  let Maplewood_grade4_students := Maplewood_grade4_percentage * Maplewood_total_students / 100 
  let combined_grade4_students := Pinegrove_grade4_students + Maplewood_grade4_students
  (combined_grade4_students * 100 / combined_total_students) = 16 := by
  sorry

end combined_percentage_grade4_l2366_236622


namespace clown_balloons_l2366_236661

theorem clown_balloons 
  (initial_balloons : ℕ := 123) 
  (additional_balloons : ℕ := 53) 
  (given_away_balloons : ℕ := 27) : 
  initial_balloons + additional_balloons - given_away_balloons = 149 := 
by 
  sorry

end clown_balloons_l2366_236661


namespace difference_in_pups_l2366_236648

theorem difference_in_pups :
  let huskies := 5
  let pitbulls := 2
  let golden_retrievers := 4
  let pups_per_husky := 3
  let pups_per_pitbull := 3
  let total_adults := huskies + pitbulls + golden_retrievers
  let total_pups := total_adults + 30
  let total_husky_pups := huskies * pups_per_husky
  let total_pitbull_pups := pitbulls * pups_per_pitbull
  let H := pups_per_husky
  let D := (total_pups - total_husky_pups - total_pitbull_pups - 3 * golden_retrievers) / golden_retrievers
  D = 2 := sorry

end difference_in_pups_l2366_236648


namespace remainder_1493824_div_4_l2366_236653

theorem remainder_1493824_div_4 : 1493824 % 4 = 0 :=
by
  sorry

end remainder_1493824_div_4_l2366_236653


namespace Alan_finish_time_third_task_l2366_236640

theorem Alan_finish_time_third_task :
  let start_time := 480 -- 8:00 AM in minutes from midnight
  let finish_time_second_task := 675 -- 11:15 AM in minutes from midnight
  let total_tasks_time := 195 -- Total time spent on first two tasks
  let first_task_time := 65 -- Time taken for the first task calculated as per the solution
  let second_task_time := 130 -- Time taken for the second task calculated as per the solution
  let third_task_time := 65 -- Time taken for the third task
  let finish_time_third_task := 740 -- 12:20 PM in minutes from midnight
  start_time + total_tasks_time + third_task_time = finish_time_third_task :=
by
  -- proof here
  sorry

end Alan_finish_time_third_task_l2366_236640


namespace length_of_each_part_l2366_236600

-- Definitions from the conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def number_of_parts : ℕ := 4

-- Proof statement
theorem length_of_each_part : total_length_in_inches / number_of_parts = 20 :=
by
  sorry

end length_of_each_part_l2366_236600


namespace option_A_is_quadratic_l2366_236668

def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

-- Given options
def option_A_equation (x : ℝ) : Prop :=
  x^2 - 2 = 0

def option_B_equation (x y : ℝ) : Prop :=
  x + 2 * y = 3

def option_C_equation (x : ℝ) : Prop :=
  x - 1/x = 1

def option_D_equation (x y : ℝ) : Prop :=
  x^2 + x = y + 1

-- Prove that option A is a quadratic equation
theorem option_A_is_quadratic (x : ℝ) : is_quadratic_equation 1 0 (-2) :=
by
  sorry

end option_A_is_quadratic_l2366_236668


namespace negation_of_universal_proposition_l2366_236631

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
sorry

end negation_of_universal_proposition_l2366_236631


namespace triangle_area_with_median_l2366_236697

theorem triangle_area_with_median (a b m : ℝ) (area : ℝ) 
  (h_a : a = 6) (h_b : b = 8) (h_m : m = 5) : 
  area = 24 :=
sorry

end triangle_area_with_median_l2366_236697


namespace least_n_factorial_6930_l2366_236639

theorem least_n_factorial_6930 (n : ℕ) (h : n! % 6930 = 0) : n ≥ 11 := by
  sorry

end least_n_factorial_6930_l2366_236639


namespace total_pencils_correct_l2366_236619

def initial_pencils : ℕ := 245
def added_pencils : ℕ := 758
def total_pencils : ℕ := initial_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 1003 := 
by
  sorry

end total_pencils_correct_l2366_236619


namespace find_numbers_l2366_236632

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l2366_236632


namespace necessary_but_not_sufficient_condition_l2366_236698

noncomputable def p (x : ℝ) : Prop := (1 - x^2 < 0 ∧ |x| - 2 > 0) ∨ (1 - x^2 > 0 ∧ |x| - 2 < 0)
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
sorry

end necessary_but_not_sufficient_condition_l2366_236698


namespace simplify_expression_l2366_236601

theorem simplify_expression (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 :=
by
  sorry

end simplify_expression_l2366_236601


namespace total_hike_time_l2366_236623

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l2366_236623


namespace Carl_chops_more_onions_than_Brittney_l2366_236696

theorem Carl_chops_more_onions_than_Brittney :
  let Brittney_rate := 15 / 5
  let Carl_rate := 20 / 5
  let Brittney_onions := Brittney_rate * 30
  let Carl_onions := Carl_rate * 30
  Carl_onions = Brittney_onions + 30 :=
by
  sorry

end Carl_chops_more_onions_than_Brittney_l2366_236696


namespace solve_quadratic_1_solve_quadratic_2_l2366_236621

-- 1. Prove that the solutions to the equation x^2 - 4x - 1 = 0 are x = 2 + sqrt(5) and x = 2 - sqrt(5)
theorem solve_quadratic_1 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

-- 2. Prove that the solutions to the equation 3(x - 1)^2 = 2(x - 1) are x = 1 and x = 5/3
theorem solve_quadratic_2 (x : ℝ) : 3 * (x - 1) ^ 2 = 2 * (x - 1) ↔ x = 1 ∨ x = 5 / 3 :=
sorry

end solve_quadratic_1_solve_quadratic_2_l2366_236621


namespace john_payment_and_hourly_rate_l2366_236693

variable (court_hours : ℕ) (prep_hours : ℕ) (upfront_fee : ℕ) 
variable (total_payment : ℕ) (brother_contribution_factor : ℕ)
variable (hourly_rate : ℚ) (john_payment : ℚ)

axiom condition1 : upfront_fee = 1000
axiom condition2 : court_hours = 50
axiom condition3 : prep_hours = 2 * court_hours
axiom condition4 : total_payment = 8000
axiom condition5 : brother_contribution_factor = 2

theorem john_payment_and_hourly_rate :
  (john_payment = total_payment / brother_contribution_factor + upfront_fee) ∧
  (hourly_rate = (total_payment - upfront_fee) / (court_hours + prep_hours)) :=
by
  sorry

end john_payment_and_hourly_rate_l2366_236693


namespace minimum_choir_members_l2366_236620

def choir_members_min (n : ℕ) : Prop :=
  (n % 8 = 0) ∧ 
  (n % 9 = 0) ∧ 
  (n % 10 = 0) ∧ 
  (n % 11 = 0)

theorem minimum_choir_members : ∃ n, choir_members_min n ∧ (∀ m, choir_members_min m → n ≤ m) :=
sorry

end minimum_choir_members_l2366_236620


namespace natural_number_property_l2366_236650

theorem natural_number_property (N k : ℕ) (hk : k > 0)
    (h1 : 10^(k-1) ≤ N) (h2 : N < 10^k) (h3 : N * 10^(k-1) ≤ N^2) (h4 : N^2 ≤ N * 10^k) :
    N = 10^(k-1) := 
sorry

end natural_number_property_l2366_236650


namespace geometric_sequence_sum_l2366_236656

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
axiom a1 : (a 1) = 1
axiom a2 : ∀ (n : ℕ), n ≥ 2 → 2 * a (n + 1) + 2 * a (n - 1) = 5 * a n
axiom increasing : ∀ (n m : ℕ), n < m → a n < a m

-- Target
theorem geometric_sequence_sum : S 5 = 31 := by
  sorry

end geometric_sequence_sum_l2366_236656


namespace big_sale_commission_l2366_236685

theorem big_sale_commission (avg_increase : ℝ) (new_avg : ℝ) (num_sales : ℕ) 
  (prev_avg := new_avg - avg_increase)
  (total_prev := prev_avg * (num_sales - 1))
  (total_new := new_avg * num_sales)
  (C := total_new - total_prev) :
  avg_increase = 150 → new_avg = 250 → num_sales = 6 → C = 1000 :=
by
  intros 
  sorry

end big_sale_commission_l2366_236685


namespace initial_amount_l2366_236670

-- Define the given conditions
def amount_spent : ℕ := 16
def amount_left : ℕ := 2

-- Define the statement that we want to prove
theorem initial_amount : amount_spent + amount_left = 18 :=
by
  sorry

end initial_amount_l2366_236670


namespace domain_f_2x_plus_1_eq_l2366_236609

-- Conditions
def domain_fx_plus_1 : Set ℝ := {x : ℝ | -2 < x ∧ x < -1}

-- Question and Correct Answer
theorem domain_f_2x_plus_1_eq :
  (∃ (x : ℝ), x ∈ domain_fx_plus_1) →
  {x : ℝ | -1 < x ∧ x < -1/2} = {x : ℝ | (2*x + 1 ∈ domain_fx_plus_1)} :=
by
  sorry

end domain_f_2x_plus_1_eq_l2366_236609


namespace bus_routes_theorem_l2366_236641

open Function

def bus_routes_exist : Prop :=
  ∃ (routes : Fin 10 → Set (Fin 10)), 
  (∀ (s : Finset (Fin 10)), (s.card = 8) → ∃ (stop : Fin 10), ∀ i ∈ s, stop ∉ routes i) ∧
  (∀ (s : Finset (Fin 10)), (s.card = 9) → ∀ (stop : Fin 10), ∃ i ∈ s, stop ∈ routes i)

theorem bus_routes_theorem : bus_routes_exist :=
sorry

end bus_routes_theorem_l2366_236641


namespace pens_sold_l2366_236681

variable (C S : ℝ)
variable (n : ℕ)

-- Define conditions
def condition1 : Prop := 10 * C = n * S
def condition2 : Prop := S = 1.5 * C

-- Define the statement to be proved
theorem pens_sold (h1 : condition1 C S n) (h2 : condition2 C S) : n = 6 := by
  -- leave the proof steps to be filled in
  sorry

end pens_sold_l2366_236681


namespace quadratic_roots_l2366_236679

theorem quadratic_roots : ∀ (x : ℝ), x^2 + 5 * x - 4 = 0 ↔ x = (-5 + Real.sqrt 41) / 2 ∨ x = (-5 - Real.sqrt 41) / 2 := 
by
  sorry

end quadratic_roots_l2366_236679


namespace winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l2366_236612

def game (n : ℕ) : Prop :=
  ∃ A_winning_strategy B_winning_strategy neither_winning_strategy,
    (n ≥ 8 → A_winning_strategy) ∧
    (n ≤ 5 → B_winning_strategy) ∧
    (n = 6 ∨ n = 7 → neither_winning_strategy)

theorem winning_strategy_for_A (n : ℕ) (h : n ≥ 8) :
  game n :=
sorry

theorem winning_strategy_for_B (n : ℕ) (h : n ≤ 5) :
  game n :=
sorry

theorem no_winning_strategy (n : ℕ) (h : n = 6 ∨ n = 7) :
  game n :=
sorry

end winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l2366_236612


namespace bob_total_investment_l2366_236673

variable (x : ℝ) -- the amount invested at 14%

noncomputable def total_investment_amount : ℝ :=
  let interest18 := 7000 * 0.18
  let interest14 := x * 0.14
  let total_interest := 3360
  let total_investment := 7000 + x
  total_investment

theorem bob_total_investment (h : 7000 * 0.18 + x * 0.14 = 3360) :
  total_investment_amount x = 22000 := by
  sorry

end bob_total_investment_l2366_236673


namespace combined_work_time_l2366_236649

-- Define the time taken by Paul and Rose to complete the work individually
def paul_days : ℕ := 80
def rose_days : ℕ := 120

-- Define the work rates of Paul and Rose
def paul_rate := 1 / (paul_days : ℚ)
def rose_rate := 1 / (rose_days : ℚ)

-- Define the combined work rate
def combined_rate := paul_rate + rose_rate

-- Statement to prove: Together they can complete the work in 48 days.
theorem combined_work_time : combined_rate = 1 / 48 := by 
  sorry

end combined_work_time_l2366_236649


namespace polygon_sides_of_interior_angle_l2366_236624

theorem polygon_sides_of_interior_angle (n : ℕ) (h : ∀ i : Fin n, (∃ (x : ℝ), x = (180 - 144) / 1) → (360 / (180 - 144)) = n) : n = 10 :=
sorry

end polygon_sides_of_interior_angle_l2366_236624


namespace soaking_time_l2366_236691

theorem soaking_time (time_per_grass_stain : ℕ) (time_per_marinara_stain : ℕ) 
    (number_of_grass_stains : ℕ) (number_of_marinara_stains : ℕ) : 
    time_per_grass_stain = 4 ∧ time_per_marinara_stain = 7 ∧ 
    number_of_grass_stains = 3 ∧ number_of_marinara_stains = 1 →
    (time_per_grass_stain * number_of_grass_stains + time_per_marinara_stain * number_of_marinara_stains) = 19 :=
by
  sorry

end soaking_time_l2366_236691


namespace chores_per_week_l2366_236663

theorem chores_per_week :
  ∀ (cookie_per_chore : ℕ) 
    (total_money : ℕ) 
    (cost_per_pack : ℕ) 
    (cookies_per_pack : ℕ) 
    (weeks : ℕ)
    (chores_per_week : ℕ),
  cookie_per_chore = 3 →
  total_money = 15 →
  cost_per_pack = 3 →
  cookies_per_pack = 24 →
  weeks = 10 →
  chores_per_week = (total_money / cost_per_pack * cookies_per_pack / weeks) / cookie_per_chore →
  chores_per_week = 4 :=
by
  intros cookie_per_chore total_money cost_per_pack cookies_per_pack weeks chores_per_week
  intros h1 h2 h3 h4 h5 h6
  sorry

end chores_per_week_l2366_236663


namespace find_fraction_l2366_236654

variable (F N : ℚ)

-- Defining the conditions
def condition1 : Prop := (1 / 3) * F * N = 18
def condition2 : Prop := (3 / 10) * N = 64.8

-- Proof statement
theorem find_fraction (h1 : condition1 F N) (h2 : condition2 N) : F = 1 / 4 := by 
  sorry

end find_fraction_l2366_236654


namespace triangle_side_length_difference_l2366_236605

theorem triangle_side_length_difference (x : ℤ) :
  (2 < x ∧ x < 16) → (∀ y : ℤ, (2 < y ∧ y < 16) → (3 ≤ y) ∧ (y ≤ 15)) →
  (∀ z : ℤ, (3 ≤ z ∨ z ≤ 15) → (15 - 3 = 12)) := by
  sorry

end triangle_side_length_difference_l2366_236605


namespace find_angle_A_find_area_l2366_236643

-- Definition for angle A
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
  (h_tria : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_A : 0 < A ∧ A < Real.pi) :
  A = Real.pi / 3 :=
by
  sorry

-- Definition for area of triangle ABC
theorem find_area (a b c : ℝ) (A : ℝ)
  (h_a : a = Real.sqrt 7) 
  (h_b : b = 2)
  (h_A : A = Real.pi / 3) 
  (h_c : c = 3) :
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end find_angle_A_find_area_l2366_236643


namespace tank_capacity_l2366_236659

theorem tank_capacity (C : ℝ) : 
  (0.5 * C = 0.9 * C - 45) → C = 112.5 :=
by
  intro h
  sorry

end tank_capacity_l2366_236659


namespace trapezoid_perimeter_l2366_236695

-- Define the problem conditions
variables (A B C D : Point) (BC AD : Line) (AB CD : Segment)

-- Conditions
def is_parallel (L1 L2 : Line) : Prop := sorry
def is_right_angle (A B C : Point) : Prop := sorry
def is_angle_150 (A B C : Point) : Prop := sorry

noncomputable def length (s : Segment) : ℝ := sorry

def trapezoid_conditions (A B C D : Point) (BC AD : Line) (AB CD : Segment) : Prop :=
  is_parallel BC AD ∧ is_angle_150 A B C ∧ is_right_angle C D B ∧
  length AB = 4 ∧ length BC = 3 - Real.sqrt 3

-- Perimeter calculation
noncomputable def perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) : ℝ :=
  length AB + length BC + length CD + length AD

-- Lean statement for the math proof problem
theorem trapezoid_perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) :
  trapezoid_conditions A B C D BC AD AB CD → perimeter A B C D BC AD AB CD = 12 :=
sorry

end trapezoid_perimeter_l2366_236695


namespace find_value_of_x2_plus_y2_l2366_236665

theorem find_value_of_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + y^2 - 4 * x * y + 24 ≤ 10 * x - 1) : x^2 + y^2 = 125 := 
sorry

end find_value_of_x2_plus_y2_l2366_236665


namespace area_enclosed_by_circle_l2366_236652

theorem area_enclosed_by_circle : Π (x y : ℝ), x^2 + y^2 + 8 * x - 6 * y = -9 → 
  ∃ A, A = 7 * Real.pi :=
by
  sorry

end area_enclosed_by_circle_l2366_236652


namespace volume_between_concentric_spheres_l2366_236627

theorem volume_between_concentric_spheres
  (r1 r2 : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 10) :
  (4 / 3 * Real.pi * r2^3 - 4 / 3 * Real.pi * r1^3) = (3500 / 3) * Real.pi :=
by
  rw [h_r1, h_r2]
  sorry

end volume_between_concentric_spheres_l2366_236627


namespace divide_milk_in_half_l2366_236642

theorem divide_milk_in_half (bucket : ℕ) (a : ℕ) (b : ℕ) (a_liters : a = 5) (b_liters : b = 7) (bucket_liters : bucket = 12) :
  ∃ x y : ℕ, x = 6 ∧ y = 6 ∧ x + y = bucket := by
  sorry

end divide_milk_in_half_l2366_236642


namespace total_people_l2366_236678

-- Define the conditions as constants
def B : ℕ := 50
def S : ℕ := 70
def B_inter_S : ℕ := 20

-- Total number of people in the group
theorem total_people : B + S - B_inter_S = 100 := by
  sorry

end total_people_l2366_236678


namespace all_faces_rhombuses_l2366_236651

variable {R : Type} [LinearOrderedCommRing R]

structure Parallelepiped (R : Type) :=
  (a b c : R)

def parallelogram_area {R : Type} [LinearOrderedCommRing R] (x y : R) : R :=
  x * y

def is_rhombus (x y : R) : Prop :=
  x = y

theorem all_faces_rhombuses (P : Parallelepiped R)
  (h1: parallelogram_area P.a P.b = parallelogram_area P.b P.c)
  (h2: parallelogram_area P.b P.c = parallelogram_area P.a P.c)
  (h3: parallelogram_area P.a P.b = parallelogram_area P.a P.c) :
  is_rhombus P.a P.b ∧ is_rhombus P.b P.c ∧ is_rhombus P.a P.c :=
  sorry

end all_faces_rhombuses_l2366_236651


namespace problem1_problem2_problem3_problem4_problem5_problem6_l2366_236606

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l2366_236606


namespace max_gcd_of_consecutive_terms_l2366_236617

-- Given conditions
def a (n : ℕ) : ℕ := 2 * (n.factorial) + n

-- Theorem statement
theorem max_gcd_of_consecutive_terms : ∃ (d : ℕ), ∀ n ≥ 0, d ≤ gcd (a n) (a (n + 1)) ∧ d = 1 := by sorry

end max_gcd_of_consecutive_terms_l2366_236617


namespace modulo_inverse_product_l2366_236684

open Int 

theorem modulo_inverse_product (n : ℕ) (a b c : ℤ) 
  (hn : 0 < n) 
  (ha : a * a.gcd n = 1) 
  (hb : b * b.gcd n = 1) 
  (hc : c * c.gcd n = 1) 
  (hab : (a * b) % n = 1) 
  (hac : (c * a) % n = 1) : 
  ((a * b) * c) % n = c % n :=
by
  sorry

end modulo_inverse_product_l2366_236684


namespace pure_gala_trees_l2366_236604

theorem pure_gala_trees (T F G : ℝ) (h1 : F + 0.10 * T = 221)
  (h2 : F = 0.75 * T) : G = T - F - 0.10 * T := 
by 
  -- We define G and show it equals 39
  have eq : T = F / 0.75 := by sorry
  have G_eq : G = T - F - 0.10 * T := by sorry 
  exact G_eq

end pure_gala_trees_l2366_236604


namespace min_editors_at_conference_l2366_236618

variable (x E : ℕ)

theorem min_editors_at_conference (h1 : x ≤ 26) 
    (h2 : 100 = 35 + E + x) 
    (h3 : 2 * x ≤ 100 - 35 - E + x) : 
    E ≥ 39 :=
by
  sorry

end min_editors_at_conference_l2366_236618


namespace exists_ten_positive_integers_l2366_236680

theorem exists_ten_positive_integers :
  ∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → ¬ (a i ∣ a j))
  ∧ (∀ i j, (a i)^2 ∣ a j) :=
sorry

end exists_ten_positive_integers_l2366_236680


namespace father_l2366_236671

-- Conditions definitions
def man's_current_age (F : ℕ) : ℕ := (2 / 5) * F
def man_after_5_years (M F : ℕ) : Prop := M + 5 = (1 / 2) * (F + 5)

-- Main statement to prove
theorem father's_age (F : ℕ) (h₁ : man's_current_age F = (2 / 5) * F)
  (h₂ : ∀ M, man_after_5_years M F → M = (2 / 5) * F + 5): F = 25 :=
sorry

end father_l2366_236671


namespace not_prime_for_some_n_l2366_236686

theorem not_prime_for_some_n (a : ℕ) (h : 1 < a) : ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := 
sorry

end not_prime_for_some_n_l2366_236686


namespace total_distance_proof_l2366_236664

-- Define the conditions
def amoli_speed : ℕ := 42      -- Amoli's speed in miles per hour
def amoli_time : ℕ := 3        -- Amoli's driving time in hours
def anayet_speed : ℕ := 61     -- Anayet's speed in miles per hour
def anayet_time : ℕ := 2       -- Anayet's driving time in hours
def remaining_distance : ℕ := 121  -- Remaining distance to be traveled in miles

-- Total distance calculation
def total_distance : ℕ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

-- The theorem to prove
theorem total_distance_proof : total_distance = 369 :=
by
  -- Proof goes here
  sorry

end total_distance_proof_l2366_236664


namespace Wendy_total_glasses_l2366_236647

theorem Wendy_total_glasses (small large : ℕ)
  (h1 : small = 50)
  (h2 : large = small + 10) :
  small + large = 110 :=
by
  sorry

end Wendy_total_glasses_l2366_236647


namespace vector_parallel_eq_l2366_236657

theorem vector_parallel_eq (k : ℝ) (a b : ℝ × ℝ) 
  (h_a : a = (k, 2)) (h_b : b = (1, 1)) (h_parallel : (∃ c : ℝ, a = (c * 1, c * 1))) : k = 2 := by
  sorry

end vector_parallel_eq_l2366_236657


namespace tom_ate_one_pound_of_carrots_l2366_236672

noncomputable def calories_from_carrots (C : ℝ) : ℝ := 51 * C
noncomputable def calories_from_broccoli (C : ℝ) : ℝ := (51 / 3) * (2 * C)
noncomputable def total_calories (C : ℝ) : ℝ :=
  calories_from_carrots C + calories_from_broccoli C

theorem tom_ate_one_pound_of_carrots :
  ∃ C : ℝ, total_calories C = 85 ∧ C = 1 :=
by
  use 1
  simp [total_calories, calories_from_carrots, calories_from_broccoli]
  sorry

end tom_ate_one_pound_of_carrots_l2366_236672


namespace correct_sampling_methods_l2366_236611

theorem correct_sampling_methods :
  (let num_balls := 1000
   let red_box := 500
   let blue_box := 200
   let yellow_box := 300
   let sample_balls := 100
   let num_students := 20
   let selected_students := 3
   let q1_method := "stratified"
   let q2_method := "simple_random"
   q1_method = "stratified" ∧ q2_method = "simple_random") := sorry

end correct_sampling_methods_l2366_236611


namespace charles_average_speed_l2366_236607

theorem charles_average_speed
  (total_distance : ℕ)
  (half_distance : ℕ)
  (second_half_speed : ℕ)
  (total_time : ℕ)
  (first_half_distance second_half_distance : ℕ)
  (time_for_second_half : ℕ)
  (time_for_first_half : ℕ)
  (first_half_speed : ℕ)
  (h1 : total_distance = 3600)
  (h2 : half_distance = total_distance / 2)
  (h3 : first_half_distance = half_distance)
  (h4 : second_half_distance = half_distance)
  (h5 : second_half_speed = 180)
  (h6 : total_time = 30)
  (h7 : time_for_second_half = second_half_distance / second_half_speed)
  (h8 : time_for_first_half = total_time - time_for_second_half)
  (h9 : first_half_speed = first_half_distance / time_for_first_half) :
  first_half_speed = 90 := by
  sorry

end charles_average_speed_l2366_236607


namespace chess_tournament_ratio_l2366_236629

theorem chess_tournament_ratio:
  ∃ n : ℕ, (n * (n - 1)) / 2 = 231 ∧ (n - 1) = 21 := 
sorry

end chess_tournament_ratio_l2366_236629


namespace rectangular_field_area_l2366_236676

theorem rectangular_field_area (w l A : ℝ) 
  (h1 : l = 3 * w)
  (h2 : 2 * (w + l) = 80) :
  A = w * l → A = 300 :=
by
  sorry

end rectangular_field_area_l2366_236676


namespace computation_result_l2366_236610

theorem computation_result :
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 :=
by
  sorry

end computation_result_l2366_236610


namespace most_likely_number_of_red_balls_l2366_236634

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l2366_236634


namespace quadratic_intersects_x_axis_l2366_236628

theorem quadratic_intersects_x_axis (a b : ℝ) (h : a ≠ 0) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 - (b^2 / (4 * a)) = 0 ∧ a * x2^2 + b * x2 - (b^2 / (4 * a)) = 0 := by
  sorry

end quadratic_intersects_x_axis_l2366_236628


namespace C_share_of_rent_l2366_236646

-- Define the given conditions
def A_ox_months : ℕ := 10 * 7
def B_ox_months : ℕ := 12 * 5
def C_ox_months : ℕ := 15 * 3
def total_rent : ℕ := 175
def total_ox_months : ℕ := A_ox_months + B_ox_months + C_ox_months
def cost_per_ox_month := total_rent / total_ox_months

-- The goal is to prove that C's share of the rent is Rs. 45
theorem C_share_of_rent : C_ox_months * cost_per_ox_month = 45 := by
  -- Adding sorry to skip the proof
  sorry

end C_share_of_rent_l2366_236646


namespace cone_angle_60_degrees_l2366_236694

theorem cone_angle_60_degrees (r : ℝ) (h : ℝ) (θ : ℝ) 
  (arc_len : θ = 60) 
  (slant_height : h = r) : θ = 60 :=
sorry

end cone_angle_60_degrees_l2366_236694


namespace range_of_a_l2366_236690

variables (a : ℝ) (x : ℝ) (x0 : ℝ)

def proposition_P (a : ℝ) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def proposition_Q (a : ℝ) : Prop :=
  ∃ x0, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_P a ∧ proposition_Q a) → a ∈ {a : ℝ | a ≤ -2} ∪ {a : ℝ | a = 1} :=
by {
  sorry -- Proof goes here.
}

end range_of_a_l2366_236690


namespace three_colored_flag_l2366_236608

theorem three_colored_flag (colors : Finset ℕ) (h : colors.card = 6) : 
  (∃ top middle bottom : ℕ, top ≠ middle ∧ top ≠ bottom ∧ middle ≠ bottom ∧ 
                            top ∈ colors ∧ middle ∈ colors ∧ bottom ∈ colors) → 
  colors.card * (colors.card - 1) * (colors.card - 2) = 120 :=
by 
  intro h_exists
  exact sorry

end three_colored_flag_l2366_236608


namespace sum_of_squares_of_rates_l2366_236613

variable (b j s : ℤ) -- rates in km/h
-- conditions
def ed_condition : Prop := 3 * b + 4 * j + 2 * s = 86
def sue_condition : Prop := 5 * b + 2 * j + 4 * s = 110

theorem sum_of_squares_of_rates (b j s : ℤ) (hEd : ed_condition b j s) (hSue : sue_condition b j s) : 
  b^2 + j^2 + s^2 = 3349 := 
sorry

end sum_of_squares_of_rates_l2366_236613


namespace pythagorean_theorem_sets_l2366_236635

theorem pythagorean_theorem_sets :
  ¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2) ∧
  (1 ^ 2 + (Real.sqrt 3) ^ 2 = 2 ^ 2) ∧
  ¬ (5 ^ 2 + 6 ^ 2 = 7 ^ 2) ∧
  ¬ (1 ^ 2 + (Real.sqrt 2) ^ 2 = 3 ^ 2) :=
by {
  sorry
}

end pythagorean_theorem_sets_l2366_236635


namespace mail_distribution_l2366_236660

theorem mail_distribution (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : total_mail / total_houses = 6 := by
  sorry

end mail_distribution_l2366_236660


namespace system_of_equations_solution_l2366_236625

theorem system_of_equations_solution :
  ∀ (a b : ℝ),
  (-2 * a + b^2 = Real.cos (π * a + b^2) - 1 ∧ b^2 = Real.cos (2 * π * a + b^2) - 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 0) :=
by
  intro a b
  sorry

end system_of_equations_solution_l2366_236625


namespace find_n_l2366_236675

theorem find_n : ∃ n : ℕ, 50^4 + 43^4 + 36^4 + 6^4 = n^4 := by
  sorry

end find_n_l2366_236675


namespace mindy_tax_rate_l2366_236689

variables (M : ℝ) -- Mork's income
variables (r : ℝ) -- Mindy's tax rate

-- Conditions
def Mork_tax_rate := 0.45 -- 45% tax rate
def Mindx_income := 4 * M -- Mindy earned 4 times as much as Mork
def combined_tax_rate := 0.21 -- Combined tax rate is 21%

-- Equation derived from the conditions
def combined_tax_rate_eq := (0.45 * M + 4 * M * r) / (M + 4 * M) = 0.21

theorem mindy_tax_rate : combined_tax_rate_eq M r → r = 0.15 :=
by
  intros conditional_eq
  sorry

end mindy_tax_rate_l2366_236689


namespace value_of_f_m_plus_one_depends_on_m_l2366_236637

def f (x a : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one_depends_on_m (m a : ℝ) (h : f (-m) a < 0) :
  (∃ m, f (m + 1) a < 0) ∧ (∃ m, f (m + 1) a > 0) :=
by
  sorry

end value_of_f_m_plus_one_depends_on_m_l2366_236637


namespace cannot_cover_completely_with_dominoes_l2366_236688

theorem cannot_cover_completely_with_dominoes :
  ¬ (∃ f : Fin 5 × Fin 3 → Fin 5 × Fin 3, 
      (∀ p q, f p = f q → p = q) ∧ 
      (∀ p, ∃ q, f q = p) ∧ 
      (∀ p, (f p).1 = p.1 + 1 ∨ (f p).2 = p.2 + 1)) := 
sorry

end cannot_cover_completely_with_dominoes_l2366_236688


namespace solomon_sale_price_l2366_236603

def original_price : ℝ := 500
def discount_rate : ℝ := 0.10
def sale_price := original_price * (1 - discount_rate)

theorem solomon_sale_price : sale_price = 450 := by
  sorry

end solomon_sale_price_l2366_236603


namespace calc_g_x_plus_2_minus_g_x_l2366_236645

def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

theorem calc_g_x_plus_2_minus_g_x (x : ℝ) : g (x + 2) - g x = 12 * x + 22 := 
by 
  sorry

end calc_g_x_plus_2_minus_g_x_l2366_236645


namespace vector_subtraction_proof_l2366_236692

theorem vector_subtraction_proof (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
    3 • b - a = (-3, -5) := by
  sorry

end vector_subtraction_proof_l2366_236692
