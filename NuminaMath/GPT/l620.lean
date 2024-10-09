import Mathlib

namespace present_worth_of_bill_l620_62036

theorem present_worth_of_bill (P : ℝ) (TD BD : ℝ) 
  (hTD : TD = 36) (hBD : BD = 37.62) 
  (hFormula : BD = (TD * (P + TD)) / P) : P = 800 :=
by
  sorry

end present_worth_of_bill_l620_62036


namespace trains_crossing_time_l620_62056

theorem trains_crossing_time (length : ℕ) (time1 time2 : ℕ) (h1 : length = 120) (h2 : time1 = 10) (h3 : time2 = 20) :
  (2 * length : ℚ) / (length / time1 + length / time2 : ℚ) = 13.33 :=
by
  sorry

end trains_crossing_time_l620_62056


namespace dealer_purchased_articles_l620_62052

/-
The dealer purchases some articles for Rs. 25 and sells 12 articles for Rs. 38. 
The dealer has a profit percentage of 90%. Prove that the number of articles 
purchased by the dealer is 14.
-/

theorem dealer_purchased_articles (x : ℕ) 
    (total_cost : ℝ) (group_selling_price : ℝ) (group_size : ℕ) (profit_percentage : ℝ) 
    (h1 : total_cost = 25)
    (h2 : group_selling_price = 38)
    (h3 : group_size = 12)
    (h4 : profit_percentage = 90 / 100) :
    x = 14 :=
by
  sorry

end dealer_purchased_articles_l620_62052


namespace drums_per_day_l620_62095

theorem drums_per_day (total_drums : Nat) (days : Nat) (total_drums_eq : total_drums = 6264) (days_eq : days = 58) :
  total_drums / days = 108 :=
by
  sorry

end drums_per_day_l620_62095


namespace compute_c_over_d_l620_62087

noncomputable def RootsResult (a b c d : ℝ) : Prop :=
  (3 * 4 + 4 * 5 + 5 * 3 = - c / a) ∧ (3 * 4 * 5 = - d / a)

theorem compute_c_over_d (a b c d : ℝ)
  (h1 : (a * 3 ^ 3 + b * 3 ^ 2 + c * 3 + d = 0))
  (h2 : (a * 4 ^ 3 + b * 4 ^ 2 + c * 4 + d = 0))
  (h3 : (a * 5 ^ 3 + b * 5 ^ 2 + c * 5 + d = 0)) 
  (hr : RootsResult a b c d) :
  c / d = 47 / 60 := 
by
  sorry

end compute_c_over_d_l620_62087


namespace general_formula_neg_seq_l620_62017

theorem general_formula_neg_seq (a : ℕ → ℝ) (h_neg : ∀ n, a n < 0)
  (h_recurrence : ∀ n, 2 * a n = 3 * a (n + 1))
  (h_product : a 2 * a 5 = 8 / 27) :
  ∀ n, a n = - ((2/3)^(n-2) : ℝ) :=
by
  sorry

end general_formula_neg_seq_l620_62017


namespace baskets_and_remainder_l620_62044

-- Define the initial conditions
def cucumbers : ℕ := 216
def basket_capacity : ℕ := 23

-- Define the expected calculations
def expected_baskets : ℕ := cucumbers / basket_capacity
def expected_remainder : ℕ := cucumbers % basket_capacity

-- Theorem to prove the output values
theorem baskets_and_remainder :
  expected_baskets = 9 ∧ expected_remainder = 9 := by
  sorry

end baskets_and_remainder_l620_62044


namespace participants_neither_coffee_nor_tea_l620_62096

-- Define the total number of participants
def total_participants : ℕ := 30

-- Define the number of participants who drank coffee
def coffee_drinkers : ℕ := 15

-- Define the number of participants who drank tea
def tea_drinkers : ℕ := 18

-- Define the number of participants who drank both coffee and tea
def both_drinkers : ℕ := 8

-- The proof statement for the number of participants who drank neither coffee nor tea
theorem participants_neither_coffee_nor_tea :
  total_participants - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
  sorry

end participants_neither_coffee_nor_tea_l620_62096


namespace intersection_domains_l620_62059

def domain_f : Set ℝ := {x : ℝ | x < 1}
def domain_g : Set ℝ := {x : ℝ | x > -1}

theorem intersection_domains : {x : ℝ | x < 1} ∩ {x : ℝ | x > -1} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end intersection_domains_l620_62059


namespace oakwood_team_count_l620_62071

theorem oakwood_team_count :
  let girls := 5
  let boys := 7
  let choose_3_girls := Nat.choose girls 3
  let choose_2_boys := Nat.choose boys 2
  choose_3_girls * choose_2_boys = 210 := by
sorry

end oakwood_team_count_l620_62071


namespace solve_problem_l620_62045

namespace Example

-- Definitions based on given conditions
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def condition_2 (f : ℝ → ℝ) : Prop := f 2 = -1

def condition_3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = -f (2 - x)

-- Main theorem statement
theorem solve_problem (f : ℝ → ℝ)
  (h1 : isEvenFunction f)
  (h2 : condition_2 f)
  (h3 : condition_3 f) : f 2016 = 1 :=
sorry

end Example

end solve_problem_l620_62045


namespace paige_scored_17_points_l620_62029

def paige_points (total_points : ℕ) (num_players : ℕ) (points_per_player_exclusive : ℕ) : ℕ :=
  total_points - ((num_players - 1) * points_per_player_exclusive)

theorem paige_scored_17_points :
  paige_points 41 5 6 = 17 :=
by
  sorry

end paige_scored_17_points_l620_62029


namespace find_positive_integer_solutions_l620_62079

def is_solution (x y : ℕ) : Prop :=
  4 * x^3 + 4 * x^2 * y - 15 * x * y^2 - 18 * y^3 - 12 * x^2 + 6 * x * y + 36 * y^2 + 5 * x - 10 * y = 0

theorem find_positive_integer_solutions :
  ∀ x y : ℕ, 0 < x ∧ 0 < y → (is_solution x y ↔ (x = 1 ∧ y = 1) ∨ (∃ y', y = y' ∧ x = 2 * y' ∧ 0 < y')) :=
by
  intros x y hxy
  sorry

end find_positive_integer_solutions_l620_62079


namespace lateral_surface_area_of_cone_l620_62021

-- Definitions from the conditions
def base_radius : ℝ := 6
def slant_height : ℝ := 15

-- Theorem statement to be proved
theorem lateral_surface_area_of_cone (r l : ℝ) (hr : r = base_radius) (hl : l = slant_height) : 
  (π * r * l) = 90 * π :=
by
  sorry

end lateral_surface_area_of_cone_l620_62021


namespace distinct_solutions_square_l620_62023

theorem distinct_solutions_square (α β : ℝ) (h₁ : α ≠ β)
    (h₂ : α^2 = 2 * α + 2 ∧ β^2 = 2 * β + 2) : (α - β) ^ 2 = 12 := by
  sorry

end distinct_solutions_square_l620_62023


namespace sum_of_three_numbers_l620_62062

theorem sum_of_three_numbers
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 252)
  (h2 : ab + bc + ca = 116) :
  a + b + c = 22 :=
by
  sorry

end sum_of_three_numbers_l620_62062


namespace minimize_abs_expression_l620_62002

theorem minimize_abs_expression {x : ℝ} : 
  ((|x - 2|) + 3) ≥ ((|2 - 2|) + 3) := 
sorry

end minimize_abs_expression_l620_62002


namespace sufficient_not_necessary_condition_l620_62009

variable (x a : ℝ)

def p := x ≤ -1
def q := a ≤ x ∧ x < a + 2

-- If q is sufficient but not necessary for p, then the range of a is (-∞, -3]
theorem sufficient_not_necessary_condition : 
  (∀ x, q x a → p x) ∧ ∃ x, p x ∧ ¬ q x a → a ≤ -3 :=
by
  sorry

end sufficient_not_necessary_condition_l620_62009


namespace triangle_ABC_properties_l620_62072

open Real

theorem triangle_ABC_properties
  (a b c : ℝ) 
  (A B C : ℝ) 
  (A_eq : A = π / 3) 
  (b_eq : b = sqrt 2) 
  (cond1 : b^2 + sqrt 2 * a * c = a^2 + c^2) 
  (cond2 : a * cos B = b * sin A) 
  (cond3 : sin B + cos B = sqrt 2) : 
  B = π / 4 ∧ (1 / 2) * a * b * sin (π - A - B) = (3 + sqrt 3) / 4 := 
by 
  sorry

end triangle_ABC_properties_l620_62072


namespace find_b_l620_62049

theorem find_b (a b : ℝ) (x : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^5) * x^5):
  b = 40 :=
  sorry

end find_b_l620_62049


namespace farmer_has_11_goats_l620_62055

theorem farmer_has_11_goats
  (pigs cows goats : ℕ)
  (h1 : pigs = 2 * cows)
  (h2 : cows = goats + 4)
  (h3 : goats + cows + pigs = 56) :
  goats = 11 := by
  sorry

end farmer_has_11_goats_l620_62055


namespace initial_persons_count_l620_62001

theorem initial_persons_count (P : ℕ) (H1 : 18 * P = 1) (H2 : 6 * P = 1/3) (H3 : 9 * (P + 4) = 2/3) : P = 12 :=
by
  sorry

end initial_persons_count_l620_62001


namespace triangle_ratio_l620_62028

noncomputable def triangle_problem (BC AC : ℝ) (angleC : ℝ) : ℝ :=
  let CD := AC / 2
  let BD := BC - CD
  let HD := BD / 2
  let AD := (3^(1/2)) * CD
  let AH := AD - HD
  (AH / HD)

theorem triangle_ratio (BC AC : ℝ) (angleC : ℝ) (h1 : BC = 6) (h2 : AC = 3 * Real.sqrt 3) (h3 : angleC = Real.pi / 6) :
  triangle_problem BC AC angleC = -2 - Real.sqrt 3 :=
by
  sorry  

end triangle_ratio_l620_62028


namespace sam_dimes_now_l620_62094

-- Define the initial number of dimes Sam had
def initial_dimes : ℕ := 9

-- Define the number of dimes Sam gave away
def dimes_given : ℕ := 7

-- State the theorem: The number of dimes Sam has now is 2
theorem sam_dimes_now : initial_dimes - dimes_given = 2 := by
  sorry

end sam_dimes_now_l620_62094


namespace valid_triples_count_l620_62046

def validTriple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 15 ∧ 
  1 ≤ b ∧ b ≤ 15 ∧ 
  1 ≤ c ∧ c ≤ 15 ∧ 
  (b % a = 0 ∨ (∃ k : ℕ, k ≤ 15 ∧ c % k = 0))

def countValidTriples : ℕ := 
  (15 + 7 + 5 + 3 + 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1) * 2 - 15

theorem valid_triples_count : countValidTriples = 75 :=
  by
  sorry

end valid_triples_count_l620_62046


namespace max_crate_weight_on_single_trip_l620_62014

-- Define the conditions
def trailer_capacity := {n | n = 3 ∨ n = 4 ∨ n = 5}
def min_crate_weight : ℤ := 1250

-- Define the maximum weight calculation
def max_weight (n : ℤ) (w : ℤ) : ℤ := n * w

-- Proof statement
theorem max_crate_weight_on_single_trip :
  ∃ w, (5 ∈ trailer_capacity) → max_weight 5 min_crate_weight = w ∧ w = 6250 := 
by
  sorry

end max_crate_weight_on_single_trip_l620_62014


namespace percent_increase_correct_l620_62000

-- Define the original and new visual ranges
def original_range : Float := 90
def new_range : Float := 150

-- Define the calculation for percent increase
def percent_increase : Float :=
  ((new_range - original_range) / original_range) * 100

-- Statement to prove
theorem percent_increase_correct : percent_increase = 66.67 :=
by
  -- To be proved
  sorry

end percent_increase_correct_l620_62000


namespace stretching_transformation_eq_curve_l620_62041

variable (x y x₁ y₁ : ℝ)

theorem stretching_transformation_eq_curve :
  (x₁ = 3 * x) →
  (y₁ = y) →
  (x₁^2 + 9 * y₁^2 = 9) →
  (x^2 + y^2 = 1) :=
by
  intros h1 h2 h3
  sorry

end stretching_transformation_eq_curve_l620_62041


namespace Deepak_age_l620_62048

theorem Deepak_age : ∃ (A D : ℕ), (A / D = 4 / 3) ∧ (A + 6 = 26) ∧ (D = 15) :=
by
  sorry

end Deepak_age_l620_62048


namespace number_of_n_for_prime_l620_62086

theorem number_of_n_for_prime (n : ℕ) : (n > 0) → ∃! n, Nat.Prime (n * (n + 2)) :=
by 
  sorry

end number_of_n_for_prime_l620_62086


namespace arithmetic_seq_sum_l620_62077

-- Definition of an arithmetic sequence using a common difference d
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Statement of the problem
theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (hs : arithmetic_sequence a d)
  (hmean : (a 3 + a 8) / 2 = 10) : 
  a 1 + a 10 = 20 :=
sorry

end arithmetic_seq_sum_l620_62077


namespace speed_of_stream_l620_62075

theorem speed_of_stream :
  ∃ (v : ℝ), (∀ (swim_speed : ℝ), swim_speed = 1.5 → 
    (∀ (time_upstream : ℝ) (time_downstream : ℝ), 
      time_upstream = 2 * time_downstream → 
      (1.5 + v) / (1.5 - v) = 2)) → v = 0.5 :=
sorry

end speed_of_stream_l620_62075


namespace correct_average_of_ten_numbers_l620_62024

theorem correct_average_of_ten_numbers :
  let incorrect_average := 20 
  let num_values := 10 
  let incorrect_number := 26
  let correct_number := 86 
  let incorrect_total_sum := incorrect_average * num_values
  let correct_total_sum := incorrect_total_sum - incorrect_number + correct_number 
  (correct_total_sum / num_values) = 26 := 
by
  sorry

end correct_average_of_ten_numbers_l620_62024


namespace base_angle_isosceles_l620_62047

-- Define an isosceles triangle with one angle being 100 degrees
def isosceles_triangle (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A) ∧ (angle_A + angle_B + angle_C = 180) ∧ (angle_A = 100)

-- The main theorem statement
theorem base_angle_isosceles {A B C : Type} (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) :
  isosceles_triangle A B C angle_A angle_B angle_C → (angle_B = 40 ∨ angle_C = 40) :=
  sorry

end base_angle_isosceles_l620_62047


namespace jenny_mother_age_l620_62093

theorem jenny_mother_age:
  (∀ x : ℕ, (50 + x = 2 * (10 + x)) → (2010 + x = 2040)) :=
by
  sorry

end jenny_mother_age_l620_62093


namespace average_marks_l620_62003

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l620_62003


namespace angle_B_value_l620_62005

theorem angle_B_value (a b c B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
    B = (Real.pi / 3) ∨ B = (2 * Real.pi / 3) :=
by
    sorry

end angle_B_value_l620_62005


namespace unique_function_satisfying_equation_l620_62011

theorem unique_function_satisfying_equation :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → ∀ x : ℝ, f x = x :=
by
  intro f h
  sorry

end unique_function_satisfying_equation_l620_62011


namespace tangent_line_eq_l620_62033

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Define the point at which we are evaluating the tangent
def point : ℝ × ℝ := (1, -1)

-- Define the derivative of the function f(x)
def f' (x : ℝ) : ℝ := 2 * x - 3

-- The desired theorem
theorem tangent_line_eq :
  ∀ x y : ℝ, (x, y) = point → (y = -x) :=
by sorry

end tangent_line_eq_l620_62033


namespace successfully_served_pizzas_l620_62078

-- Defining the conditions
def total_pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

-- Stating the theorem
theorem successfully_served_pizzas :
  total_pizzas_served - pizzas_returned = 3 :=
by
  -- Since this is only the statement, the proof is omitted using sorry
  sorry

end successfully_served_pizzas_l620_62078


namespace billion_in_scientific_notation_l620_62040

theorem billion_in_scientific_notation :
  (10^9 = 1 * 10^9) :=
by
  sorry

end billion_in_scientific_notation_l620_62040


namespace points_satisfy_diamond_eq_l620_62089

noncomputable def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_satisfy_diamond_eq (x y : ℝ) :
  (diamond x y = diamond y x) ↔ ((x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x = -y)) := 
by
  sorry

end points_satisfy_diamond_eq_l620_62089


namespace length_of_garden_l620_62018

-- Definitions based on conditions
def P : ℕ := 600
def b : ℕ := 200

-- Theorem statement
theorem length_of_garden : ∃ L : ℕ, 2 * (L + b) = P ∧ L = 100 :=
by
  existsi 100
  simp
  sorry

end length_of_garden_l620_62018


namespace fourth_number_in_12th_row_is_92_l620_62050

-- Define the number of elements per row and the row number
def elements_per_row := 8
def row_number := 12

-- Define the last number in a row function
def last_number_in_row (n : ℕ) := elements_per_row * n

-- Define the starting number in a row function
def starting_number_in_row (n : ℕ) := (elements_per_row * (n - 1)) + 1

-- Define the nth number in a specified row function
def nth_number_in_row (n : ℕ) (k : ℕ) := starting_number_in_row n + (k - 1)

-- Prove that the fourth number in the 12th row is 92
theorem fourth_number_in_12th_row_is_92 : nth_number_in_row 12 4 = 92 :=
by
  -- state the required equivalences
  sorry

end fourth_number_in_12th_row_is_92_l620_62050


namespace polygon_with_three_times_exterior_angle_sum_is_octagon_l620_62026

theorem polygon_with_three_times_exterior_angle_sum_is_octagon
  (n : ℕ)
  (h : (n - 2) * 180 = 3 * 360) : n = 8 := by
  sorry

end polygon_with_three_times_exterior_angle_sum_is_octagon_l620_62026


namespace find_interest_rate_l620_62057

noncomputable def compound_interest_rate (A P : ℝ) (t n : ℕ) : ℝ := sorry

theorem find_interest_rate :
  compound_interest_rate 676 625 2 1 = 0.04 := 
sorry

end find_interest_rate_l620_62057


namespace cally_pants_count_l620_62010

variable (cally_white_shirts : ℕ)
variable (cally_colored_shirts : ℕ)
variable (cally_shorts : ℕ)
variable (danny_white_shirts : ℕ)
variable (danny_colored_shirts : ℕ)
variable (danny_shorts : ℕ)
variable (danny_pants : ℕ)
variable (total_clothes_washed : ℕ)
variable (cally_pants : ℕ)

-- Given conditions
#check cally_white_shirts = 10
#check cally_colored_shirts = 5
#check cally_shorts = 7
#check danny_white_shirts = 6
#check danny_colored_shirts = 8
#check danny_shorts = 10
#check danny_pants = 6
#check total_clothes_washed = 58
#check cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed

-- Proof goal
theorem cally_pants_count (cally_white_shirts cally_colored_shirts cally_shorts danny_white_shirts danny_colored_shirts danny_shorts danny_pants cally_pants total_clothes_washed : ℕ) :
  cally_white_shirts = 10 →
  cally_colored_shirts = 5 →
  cally_shorts = 7 →
  danny_white_shirts = 6 →
  danny_colored_shirts = 8 →
  danny_shorts = 10 →
  danny_pants = 6 →
  total_clothes_washed = 58 →
  (cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed) →
  cally_pants = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end cally_pants_count_l620_62010


namespace bugs_meeting_time_l620_62090

/-- Two circles with radii 7 inches and 3 inches are tangent at a point P. 
Two bugs start crawling at the same time from point P, one along the larger circle 
at 4π inches per minute, and the other along the smaller circle at 3π inches per minute. 
Prove they will meet again after 14 minutes and determine how far each has traveled.

The bug on the larger circle will have traveled 28π inches.
The bug on the smaller circle will have traveled 42π inches.
-/
theorem bugs_meeting_time
  (r₁ r₂ : ℝ) (v₁ v₂ : ℝ)
  (h₁ : r₁ = 7) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) :
  ∃ t d₁ d₂, t = 14 ∧ d₁ = 28 * Real.pi ∧ d₂ = 42 * Real.pi := by
  sorry

end bugs_meeting_time_l620_62090


namespace greatest_sundays_in_49_days_l620_62038

theorem greatest_sundays_in_49_days : 
  ∀ (days : ℕ), 
    days = 49 → 
    ∀ (sundays_per_week : ℕ), 
      sundays_per_week = 1 → 
      ∀ (weeks : ℕ), 
        weeks = days / 7 → 
        weeks * sundays_per_week = 7 :=
by
  sorry

end greatest_sundays_in_49_days_l620_62038


namespace cos_alpha_minus_pi_over_4_l620_62065

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.tan α = 2) :
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_alpha_minus_pi_over_4_l620_62065


namespace decimal_15_to_binary_l620_62053

theorem decimal_15_to_binary : (15 : ℕ) = (4*1 + 2*1 + 1*1)*2^3 + (4*1 + 2*1 + 1*1)*2^2 + (4*1 + 2*1 + 1*1)*2 + 1 := by
  sorry

end decimal_15_to_binary_l620_62053


namespace each_boy_makes_14_l620_62088

/-- Proof that each boy makes 14 dollars given the initial conditions and sales scheme. -/
theorem each_boy_makes_14 (victor_shrimp : ℕ)
                          (austin_shrimp : ℕ)
                          (brian_shrimp : ℕ)
                          (total_shrimp : ℕ)
                          (sets_sold : ℕ)
                          (total_earnings : ℕ)
                          (individual_earnings : ℕ)
                          (h1 : victor_shrimp = 26)
                          (h2 : austin_shrimp = victor_shrimp - 8)
                          (h3 : brian_shrimp = (victor_shrimp + austin_shrimp) / 2)
                          (h4 : total_shrimp = victor_shrimp + austin_shrimp + brian_shrimp)
                          (h5 : sets_sold = total_shrimp / 11)
                          (h6 : total_earnings = sets_sold * 7)
                          (h7 : individual_earnings = total_earnings / 3):
  individual_earnings = 14 := 
by
  sorry

end each_boy_makes_14_l620_62088


namespace g_sum_even_l620_62008

def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5

theorem g_sum_even (a b c d : ℝ) (h : g 42 a b c d = 3) : g 42 a b c d + g (-42) a b c d = 6 := by
  sorry

end g_sum_even_l620_62008


namespace fourth_even_integer_l620_62074

theorem fourth_even_integer (n : ℤ) (h : (n-2) + (n+2) = 92) : n + 4 = 50 := by
  -- This will skip the proof steps and assume the correct answer
  sorry

end fourth_even_integer_l620_62074


namespace arithmetic_sequence_sum_proof_l620_62066

theorem arithmetic_sequence_sum_proof
  (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 17 = 170)
  (h2 : a 2000 = 2001)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a (n + 1) = a n + (a 2 - a 1)) :
  S 2008 = 2019044 :=
  sorry

end arithmetic_sequence_sum_proof_l620_62066


namespace repeatingDecimals_fraction_eq_l620_62081

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l620_62081


namespace aquarium_water_l620_62076

theorem aquarium_water (T1 T2 T3 T4 : ℕ) (g w : ℕ) (hT1 : T1 = 8) (hT2 : T2 = 8) (hT3 : T3 = 6) (hT4 : T4 = 6):
  (g = T1 + T2 + T3 + T4) → (w = g * 4) → w = 112 :=
by
  sorry

end aquarium_water_l620_62076


namespace log_sum_greater_than_two_l620_62060

variables {x y a m : ℝ}

theorem log_sum_greater_than_two
  (hx : 0 < x) (hxy : x < y) (hya : y < a) (ha1 : a < 1)
  (hm : m = Real.log x / Real.log a + Real.log y / Real.log a) : m > 2 :=
sorry

end log_sum_greater_than_two_l620_62060


namespace distance_between_trees_l620_62013

-- Definitions based on conditions
def yard_length : ℝ := 360
def number_of_trees : ℕ := 31
def number_of_gaps : ℕ := number_of_trees - 1

-- The proposition to prove
theorem distance_between_trees : yard_length / number_of_gaps = 12 := sorry

end distance_between_trees_l620_62013


namespace volume_and_surface_area_implies_sum_of_edges_l620_62085

-- Define the problem conditions and prove the required statement
theorem volume_and_surface_area_implies_sum_of_edges :
  ∃ (a r : ℝ), 
    (a / r) * a * (a * r) = 216 ∧ 
    2 * ((a^2 / r) + a^2 * r + a^2) = 288 →
    4 * ((a / r) + a * r + a) = 96 :=
by
  sorry

end volume_and_surface_area_implies_sum_of_edges_l620_62085


namespace find_x_l620_62099

def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (4, x)

theorem find_x (x : ℝ) (h : ∃k : ℝ, b x = (k * a.1, k * a.2)) : x = 6 := 
by 
  sorry

end find_x_l620_62099


namespace dimensions_increased_three_times_l620_62035

variables (L B H k : ℝ) (n : ℝ)
 
-- Given conditions
axiom cost_initial : 350 = k * 2 * (L + B) * H
axiom cost_increased : 3150 = k * 2 * n^2 * (L + B) * H

-- Proof statement
theorem dimensions_increased_three_times : n = 3 :=
by
  sorry

end dimensions_increased_three_times_l620_62035


namespace jonas_needs_35_pairs_of_socks_l620_62006

def JonasWardrobeItems (socks_pairs shoes_pairs pants_items tshirts : ℕ) : ℕ :=
  2 * socks_pairs + 2 * shoes_pairs + pants_items + tshirts

def itemsNeededToDouble (initial_items : ℕ) : ℕ :=
  2 * initial_items - initial_items

theorem jonas_needs_35_pairs_of_socks (socks_pairs : ℕ) 
                                      (shoes_pairs : ℕ) 
                                      (pants_items : ℕ) 
                                      (tshirts : ℕ) 
                                      (final_socks_pairs : ℕ) 
                                      (initial_items : ℕ := JonasWardrobeItems socks_pairs shoes_pairs pants_items tshirts) 
                                      (needed_items : ℕ := itemsNeededToDouble initial_items) 
                                      (needed_pairs_of_socks := needed_items / 2) : 
                                      final_socks_pairs = 35 :=
by
  sorry

end jonas_needs_35_pairs_of_socks_l620_62006


namespace determine_s_l620_62098

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem determine_s (s : ℝ) (h : g (-3) s = 0) : s = -192 :=
by
  sorry

end determine_s_l620_62098


namespace washington_goats_l620_62030

variables (W : ℕ) (P : ℕ) (total_goats : ℕ)

theorem washington_goats (W : ℕ) (h1 : P = W + 40) (h2 : total_goats = W + P) (h3 : total_goats = 320) : W = 140 :=
by
  sorry

end washington_goats_l620_62030


namespace correct_sampling_l620_62039

-- Let n be the total number of students
def total_students : ℕ := 60

-- Define the systematic sampling function
def systematic_sampling (n m : ℕ) (start : ℕ) : List ℕ :=
  List.map (λ k => start + k * m) (List.range n)

-- Prove that the sequence generated is equal to [3, 13, 23, 33, 43, 53]
theorem correct_sampling :
  systematic_sampling 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end correct_sampling_l620_62039


namespace loss_percentage_l620_62051

theorem loss_percentage (CP SP : ℝ) (hCP : CP = 1500) (hSP : SP = 1200) : 
  (CP - SP) / CP * 100 = 20 :=
by
  -- Proof would be provided here
  sorry

end loss_percentage_l620_62051


namespace valid_outfits_number_l620_62042

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end valid_outfits_number_l620_62042


namespace percentage_increase_third_year_l620_62063

theorem percentage_increase_third_year
  (initial_price : ℝ)
  (price_2007 : ℝ := initial_price * (1 + 20 / 100))
  (price_2008 : ℝ := price_2007 * (1 - 25 / 100))
  (price_end_third_year : ℝ := initial_price * (108 / 100)) :
  ((price_end_third_year - price_2008) / price_2008) * 100 = 20 :=
by
  sorry

end percentage_increase_third_year_l620_62063


namespace borrowed_sheets_l620_62022

-- Defining the page sum function
def sum_pages (n : ℕ) : ℕ := n * (n + 1)

-- Formulating the main theorem statement
theorem borrowed_sheets (b c : ℕ) (H : c + b ≤ 30) (H_avg : (sum_pages b + sum_pages (30 - b - c) - sum_pages (b + c)) * 2 = 25 * (60 - 2 * c)) :
  c = 10 :=
sorry

end borrowed_sheets_l620_62022


namespace travel_time_at_constant_speed_l620_62091

theorem travel_time_at_constant_speed
  (distance : ℝ) (speed : ℝ) : 
  distance = 100 → speed = 20 → distance / speed = 5 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end travel_time_at_constant_speed_l620_62091


namespace moles_of_HCl_is_one_l620_62034

def moles_of_HCl_combined 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : ℝ := 
by 
  sorry

theorem moles_of_HCl_is_one 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : moles_of_HCl_combined moles_NaHSO3 moles_H2O_formed reaction_completes one_mole_NaHSO3_used = 1 := 
by 
  sorry

end moles_of_HCl_is_one_l620_62034


namespace total_songs_in_june_l620_62027

-- Define the conditions
def Vivian_daily_songs : ℕ := 10
def Clara_daily_songs : ℕ := Vivian_daily_songs - 2
def Lucas_daily_songs : ℕ := Vivian_daily_songs + 5
def total_play_days_in_june : ℕ := 30 - 8 - 1

-- Total songs listened to in June
def total_songs_Vivian : ℕ := Vivian_daily_songs * total_play_days_in_june
def total_songs_Clara : ℕ := Clara_daily_songs * total_play_days_in_june
def total_songs_Lucas : ℕ := Lucas_daily_songs * total_play_days_in_june

-- The total number of songs listened to by all three
def total_songs_all_three : ℕ := total_songs_Vivian + total_songs_Clara + total_songs_Lucas

-- The proof problem
theorem total_songs_in_june : total_songs_all_three = 693 := by
  -- Placeholder for the proof
  sorry

end total_songs_in_june_l620_62027


namespace Jesse_read_pages_l620_62004

theorem Jesse_read_pages (total_pages : ℝ) (h : (2 / 3) * total_pages = 166) :
  (1 / 3) * total_pages = 83 :=
sorry

end Jesse_read_pages_l620_62004


namespace tradesman_gain_on_outlay_l620_62067

-- Define the percentage defrauded and the percentage gain in both buying and selling
def defraud_percent := 20
def original_value := 100
def buying_price := original_value * (1 - (defraud_percent / 100))
def selling_price := original_value * (1 + (defraud_percent / 100))
def gain := selling_price - buying_price
def gain_percent := (gain / buying_price) * 100

theorem tradesman_gain_on_outlay :
  gain_percent = 50 := 
sorry

end tradesman_gain_on_outlay_l620_62067


namespace cost_of_ice_cream_l620_62092

theorem cost_of_ice_cream (x : ℝ) (h1 : 10 * x = 40) : x = 4 :=
by sorry

end cost_of_ice_cream_l620_62092


namespace area_of_shaded_region_l620_62007

def side_length_of_square : ℝ := 12
def radius_of_quarter_circle : ℝ := 6

theorem area_of_shaded_region :
  let area_square := side_length_of_square ^ 2
  let area_full_circle := π * radius_of_quarter_circle ^ 2
  (area_square - area_full_circle) = 144 - 36 * π :=
by
  sorry

end area_of_shaded_region_l620_62007


namespace smallest_value_in_geometric_progression_l620_62084

open Real

theorem smallest_value_in_geometric_progression 
  (d : ℝ) : 
  (∀ a b c d : ℝ, 
    a = 5 ∧ b = 5 + d ∧ c = 5 + 2 * d ∧ d = 5 + 3 * d ∧ 
    ∀ a' b' c' d' : ℝ, 
      a' = 5 ∧ b' = 6 + d ∧ c' = 15 + 2 * d ∧ d' = 3 * d ∧ 
      (b' / a' = c' / b' ∧ c' / b' = d' / c')) → 
  (d = (-1 + 4 * sqrt 10) ∨ d = (-1 - 4 * sqrt 10)) → 
  (min (3 * (-1 + 4 * sqrt 10)) (3 * (-1 - 4 * sqrt 10)) = -3 - 12 * sqrt 10) :=
by
  intros ha hd
  sorry

end smallest_value_in_geometric_progression_l620_62084


namespace intersecting_lines_l620_62080

variable (a b m : ℝ)

-- Conditions
def condition1 : Prop := 8 = -m + a
def condition2 : Prop := 8 = m + b

-- Statement to prove
theorem intersecting_lines : condition1 a m  → condition2 b m  → a + b = 16 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l620_62080


namespace sara_height_correct_l620_62058

variable (Roy_height : ℕ)
variable (Joe_height : ℕ)
variable (Sara_height : ℕ)

def problem_conditions (Roy_height Joe_height Sara_height : ℕ) : Prop :=
  Roy_height = 36 ∧
  Joe_height = Roy_height + 3 ∧
  Sara_height = Joe_height + 6

theorem sara_height_correct (Roy_height Joe_height Sara_height : ℕ) :
  problem_conditions Roy_height Joe_height Sara_height → Sara_height = 45 := by
  sorry

end sara_height_correct_l620_62058


namespace smallest_n_square_average_l620_62082

theorem smallest_n_square_average (n : ℕ) (h : n > 1)
  (S : ℕ := (n * (n + 1) * (2 * n + 1)) / 6)
  (avg : ℕ := S / n) :
  (∃ k : ℕ, avg = k^2) → n = 337 := by
  sorry

end smallest_n_square_average_l620_62082


namespace greatest_number_of_dimes_l620_62043

-- Definitions according to the conditions in a)
def total_value_in_cents : ℤ := 485
def dime_value_in_cents : ℤ := 10
def nickel_value_in_cents : ℤ := 5

-- The proof problem in Lean 4
theorem greatest_number_of_dimes : 
  ∃ (d : ℤ), (dime_value_in_cents * d + nickel_value_in_cents * d = total_value_in_cents) ∧ d = 32 := 
by
  sorry

end greatest_number_of_dimes_l620_62043


namespace space_shuttle_speed_l620_62016

-- Define the conditions in Lean
def speed_kmph : ℕ := 43200 -- Speed in kilometers per hour
def seconds_per_hour : ℕ := 60 * 60 -- Number of seconds in an hour

-- Define the proof problem
theorem space_shuttle_speed :
  speed_kmph / seconds_per_hour = 12 := by
  sorry

end space_shuttle_speed_l620_62016


namespace digit_divisibility_by_7_l620_62054

theorem digit_divisibility_by_7 (d : ℕ) (h : d < 10) : (10000 + 100 * d + 10) % 7 = 0 ↔ d = 5 :=
by
  sorry

end digit_divisibility_by_7_l620_62054


namespace min_max_expression_l620_62015

variable (a b c d e : ℝ)

def expression (a b c d e : ℝ) : ℝ :=
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)

theorem min_max_expression :
  a + b + c + d + e = 10 →
  a^2 + b^2 + c^2 + d^2 + e^2 = 20 →
  expression a b c d e = 120 := by
  sorry

end min_max_expression_l620_62015


namespace sales_worth_l620_62064

variables (S : ℝ)
variables (old_scheme_remuneration new_scheme_remuneration : ℝ)

def old_scheme := 0.05 * S
def new_scheme := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_scheme S = old_scheme S + 600 →
  S = 24000 :=
by
  intro h
  sorry

end sales_worth_l620_62064


namespace no_single_x_for_doughnut_and_syrup_l620_62025

theorem no_single_x_for_doughnut_and_syrup :
  ¬ ∃ x : ℝ, (x^2 - 9 * x + 13 < 0) ∧ (x^2 + x - 5 < 0) :=
sorry

end no_single_x_for_doughnut_and_syrup_l620_62025


namespace lcm_two_numbers_l620_62069

theorem lcm_two_numbers (a b : ℕ) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) : Nat.lcm a b = 1485 := 
by
  sorry

end lcm_two_numbers_l620_62069


namespace min_value_of_z_ineq_l620_62070

noncomputable def z (x y : ℝ) : ℝ := 2 * x + 4 * y

theorem min_value_of_z_ineq (k : ℝ) :
  (∃ x y : ℝ, (3 * x + y ≥ 0) ∧ (4 * x + 3 * y ≥ k) ∧ (z x y = -6)) ↔ k = 0 :=
by
  sorry

end min_value_of_z_ineq_l620_62070


namespace problem_correct_l620_62061

def decimal_to_fraction_eq_80_5 : Prop :=
  ( (0.5 + 0.25 + 0.125) / (0.5 * 0.25 * 0.125) * ((7 / 18 * (9 / 2) + 1 / 6) / (13 + 1 / 3 - (15 / 4 * 16 / 5))) = 80.5 )

theorem problem_correct : decimal_to_fraction_eq_80_5 :=
  sorry

end problem_correct_l620_62061


namespace at_least_one_not_less_than_two_l620_62097

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a + (1 / b) ≥ 2 ∨ b + (1 / a) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l620_62097


namespace percentage_of_x_is_40_l620_62019

theorem percentage_of_x_is_40 
  (x p : ℝ)
  (h1 : (1 / 2) * x = 200)
  (h2 : p * x = 160) : 
  p * 100 = 40 := 
by 
  sorry

end percentage_of_x_is_40_l620_62019


namespace ratio_Lisa_Claire_l620_62068

-- Definitions
def Claire_photos : ℕ := 6
def Robert_photos : ℕ := Claire_photos + 12
def Lisa_photos : ℕ := Robert_photos

-- Theorem statement
theorem ratio_Lisa_Claire : (Lisa_photos : ℚ) / (Claire_photos : ℚ) = 3 / 1 :=
by
  sorry

end ratio_Lisa_Claire_l620_62068


namespace a6_is_32_l620_62037

namespace arithmetic_sequence

variables {a : ℕ → ℝ} -- {aₙ} is an arithmetic sequence with positive terms
variables (q : ℝ) -- Common ratio

-- Conditions as definitions
def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a1_is_1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2_times_a4_is_16 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 4 = 16

-- The ultimate goal is to prove a₆ = 32
theorem a6_is_32 (h_arith : is_arithmetic_sequence a q) 
  (h_a1 : a1_is_1 a) (h_product : a2_times_a4_is_16 a q) : 
  a 6 = 32 := 
sorry

end arithmetic_sequence

end a6_is_32_l620_62037


namespace johns_pants_cost_50_l620_62032

variable (P : ℝ)

theorem johns_pants_cost_50 (h1 : P + 1.60 * P = 130) : P = 50 := 
by
  sorry

end johns_pants_cost_50_l620_62032


namespace probability_within_circle_eq_pi_over_nine_l620_62020

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let circle_area := Real.pi * (2 ^ 2)
  let square_area := 6 * 6
  circle_area / square_area

theorem probability_within_circle_eq_pi_over_nine :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_circle_eq_pi_over_nine_l620_62020


namespace speed_of_second_car_l620_62073

theorem speed_of_second_car (s1 s2 s : ℕ) (v1 : ℝ) (h_s1 : s1 = 500) (h_s2 : s2 = 700) 
  (h_s : s = 100) (h_v1 : v1 = 10) : 
  (∃ v2 : ℝ, v2 = 12 ∨ v2 = 16) :=
by 
  sorry

end speed_of_second_car_l620_62073


namespace triangle_properties_l620_62012

variable (a b c A B C : ℝ)
variable (CD BD : ℝ)

-- triangle properties and given conditions
variable (b_squared_eq_ac : b ^ 2 = a * c)
variable (cos_A_minus_C : Real.cos (A - C) = Real.cos B + 1 / 2)

theorem triangle_properties :
  B = π / 3 ∧ 
  A = π / 3 ∧ 
  (CD = 6 → ∃ x, x > 0 ∧ x = 4 * Real.sqrt 3 + 6) ∧
  (BD = 6 → ∀ area, area ≠ 9 / 4) :=
  by
    sorry

end triangle_properties_l620_62012


namespace f_inequality_l620_62031

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f(x+3) = -1 / f(x)
axiom f_prop1 : ∀ x : ℝ, f (x + 3) = -1 / f x

-- Condition 2: ∀ 3 ≤ x_1 < x_2 ≤ 6, f(x_1) < f(x_2)
axiom f_prop2 : ∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 6 → f x1 < f x2

-- Condition 3: The graph of y = f(x + 3) is symmetric about the y-axis
axiom f_prop3 : ∀ x : ℝ, f (3 - x) = f (3 + x)

-- Theorem: f(3) < f(4.5) < f(7)
theorem f_inequality : f 3 < f 4.5 ∧ f 4.5 < f 7 := by
  sorry

end f_inequality_l620_62031


namespace actual_distance_traveled_l620_62083

theorem actual_distance_traveled 
  (D : ℝ)
  (h1 : ∃ (D : ℝ), D/12 = (D + 36)/20)
  : D = 54 :=
sorry

end actual_distance_traveled_l620_62083
