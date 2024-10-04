import Mathlib

namespace business_transaction_loss_l292_292568

theorem business_transaction_loss (cost_price : ℝ) (final_price : ℝ) (markup_percent : ℝ) (reduction_percent : ℝ) : 
  (final_price = 96) ∧ (markup_percent = 0.2) ∧ (reduction_percent = 0.2) ∧ (cost_price * (1 + markup_percent) * (1 - reduction_percent) = final_price) → 
  (cost_price - final_price = -4) :=
by
sorry

end business_transaction_loss_l292_292568


namespace second_candy_cost_l292_292086

theorem second_candy_cost 
  (C : ℝ) 
  (hp := 25 * 8 + 50 * C = 75 * 6) : 
  C = 5 := 
  sorry

end second_candy_cost_l292_292086


namespace proof_l292_292322

noncomputable def question := ∀ x : ℝ, (0.12 * x = 36) → (0.5 * (0.4 * 0.3 * x) = 18) 

theorem proof : question :=
by
  intro x
  intro h
  sorry

end proof_l292_292322


namespace proof_problem_l292_292139

theorem proof_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) :
  (a < 1 → b > 2) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y - 1 / (2 * x) - 2 / y = 3 / 2 → x + y ≥ 3) :=
by
  sorry

end proof_problem_l292_292139


namespace find_fraction_l292_292478

theorem find_fraction 
  (f : ℚ) (t k : ℚ)
  (h1 : t = f * (k - 32)) 
  (h2 : t = 75)
  (h3 : k = 167) : 
  f = 5 / 9 :=
by
  sorry

end find_fraction_l292_292478


namespace price_each_puppy_l292_292033

def puppies_initial : ℕ := 8
def puppies_given_away : ℕ := puppies_initial / 2
def puppies_remaining_after_giveaway : ℕ := puppies_initial - puppies_given_away
def puppies_kept : ℕ := 1
def puppies_to_sell : ℕ := puppies_remaining_after_giveaway - puppies_kept
def stud_fee : ℕ := 300
def profit : ℕ := 1500
def total_amount_made : ℕ := profit + stud_fee
def price_per_puppy : ℕ := total_amount_made / puppies_to_sell

theorem price_each_puppy :
  price_per_puppy = 600 :=
sorry

end price_each_puppy_l292_292033


namespace projectile_height_reaches_49_l292_292064

theorem projectile_height_reaches_49 (t : ℝ) :
  (∃ t : ℝ, 49 = -20 * t^2 + 100 * t) → t = 0.7 :=
by
  sorry

end projectile_height_reaches_49_l292_292064


namespace even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l292_292661

open Int
open Nat

theorem even_n_square_mod_8 (n : ℤ) (h : n % 2 = 0) : (n^2 % 8 = 0) ∨ (n^2 % 8 = 4) := sorry

theorem odd_n_square_mod_8 (n : ℤ) (h : n % 2 = 1) : n^2 % 8 = 1 := sorry

theorem odd_n_fourth_mod_8 (n : ℤ) (h : n % 2 = 1) : n^4 % 8 = 1 := sorry

end even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l292_292661


namespace miles_hiked_first_day_l292_292252

theorem miles_hiked_first_day (total_distance remaining_distance : ℕ)
  (h1 : total_distance = 36)
  (h2 : remaining_distance = 27) :
  total_distance - remaining_distance = 9 :=
by
  sorry

end miles_hiked_first_day_l292_292252


namespace find_n_from_lcms_l292_292000

theorem find_n_from_lcms (n : ℕ) (h_pos : n > 0) (h_lcm1 : Nat.lcm 40 n = 200) (h_lcm2 : Nat.lcm n 45 = 180) : n = 100 := 
by
  sorry

end find_n_from_lcms_l292_292000


namespace log_product_max_l292_292453

open Real

theorem log_product_max (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : log x + log y = 4) : log x * log y ≤ 4 := 
by
  sorry

end log_product_max_l292_292453


namespace cube_root_simplification_l292_292392

theorem cube_root_simplification : (∛(4^6 + 4^6 + 4^6 + 4^6) = 16 * ∛4) :=
by {
  -- Proof goes here
  sorry
}

end cube_root_simplification_l292_292392


namespace max_quadratic_value_l292_292909

def quadratic (x : ℝ) : ℝ :=
  -2 * x^2 + 4 * x + 3

theorem max_quadratic_value : ∃ x : ℝ, ∀ y : ℝ, quadratic x = y → y ≤ 5 ∧ (∀ z : ℝ, quadratic z ≤ y) := 
by
  sorry

end max_quadratic_value_l292_292909


namespace james_out_of_pocket_l292_292814

theorem james_out_of_pocket :
  let initial_expenditure := 3000
  let tv_return := 700
  let bike_return := 500
  let second_bike_cost := bike_return + 0.20 * bike_return
  let second_bike_sell := 0.80 * second_bike_cost
  let toaster_cost := 100
  initial_expenditure - tv_return - bike_return - second_bike_sell + toaster_cost = 1420 :=
by
  -- Definitions
  let initial_expenditure := 3000
  let tv_return := 700
  let bike_return := 500
  let second_bike_cost := bike_return + 0.20 * bike_return
  let second_bike_sell := 0.80 * second_bike_cost
  let toaster_cost := 100
  -- Goal
  have h : initial_expenditure - tv_return - bike_return - second_bike_sell + toaster_cost = 1420
  simp [initial_expenditure, tv_return, bike_return, second_bike_cost, second_bike_sell, toaster_cost]
  exact h

end james_out_of_pocket_l292_292814


namespace minimum_colors_needed_l292_292387

def paint_fence_colors (B : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, B i ≠ B (i + 2)) ∧
  (∀ i : ℕ, B i ≠ B (i + 3)) ∧
  (∀ i : ℕ, B i ≠ B (i + 5))

theorem minimum_colors_needed : ∃ (c : ℕ), 
  (∀ B : ℕ → ℕ, paint_fence_colors B → c ≥ 3) ∧
  (∃ B : ℕ → ℕ, paint_fence_colors B ∧ c = 3) :=
sorry

end minimum_colors_needed_l292_292387


namespace greatest_GCD_of_product_7200_l292_292543

theorem greatest_GCD_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ ∀ d, (d ∣ a ∧ d ∣ b) → d ≤ 60 :=
by
  sorry

end greatest_GCD_of_product_7200_l292_292543


namespace smallest_t_for_temperature_104_l292_292670

theorem smallest_t_for_temperature_104 : 
  ∃ t : ℝ, (-t^2 + 16*t + 40 = 104) ∧ (t > 0) ∧ (∀ s : ℝ, (-s^2 + 16*s + 40 = 104) ∧ (s > 0) → t ≤ s) :=
sorry

end smallest_t_for_temperature_104_l292_292670


namespace cos_diff_identity_l292_292458

theorem cos_diff_identity 
  (α β : ℝ)
  (h1 : sin α - sin β = 1 - (sqrt 3) / 2)
  (h2 : cos α - cos β = 1 / 2) :
  cos (α - β) = (sqrt 3) / 2 :=
by
  sorry

end cos_diff_identity_l292_292458


namespace problem_l292_292801

theorem problem (y : ℝ) (hy : 5 = y^2 + 4 / y^2) : y + 2 / y = 3 ∨ y + 2 / y = -3 :=
by
  sorry

end problem_l292_292801


namespace number_of_trees_is_correct_l292_292444

-- Define the conditions
def length_of_plot := 120
def width_of_plot := 70
def distance_between_trees := 5

-- Define the calculated number of intervals along each side
def intervals_along_length := length_of_plot / distance_between_trees
def intervals_along_width := width_of_plot / distance_between_trees

-- Define the number of trees along each side including the boundaries
def trees_along_length := intervals_along_length + 1
def trees_along_width := intervals_along_width + 1

-- Define the total number of trees
def total_number_of_trees := trees_along_length * trees_along_width

-- The theorem we want to prove
theorem number_of_trees_is_correct : total_number_of_trees = 375 :=
by sorry

end number_of_trees_is_correct_l292_292444


namespace plan_b_more_cost_effective_l292_292990

noncomputable def fare (x : ℝ) : ℝ :=
if x < 3 then 5
else if x <= 10 then 1.2 * x + 1.4
else 1.8 * x - 4.6

theorem plan_b_more_cost_effective :
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  plan_a > plan_b :=
by
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  sorry

end plan_b_more_cost_effective_l292_292990


namespace solve_for_m_l292_292136

noncomputable def operation (a b c x y : ℝ) := a * x + b * y + c * x * y

theorem solve_for_m (a b c : ℝ) (h1 : operation a b c 1 2 = 3)
                              (h2 : operation a b c 2 3 = 4) 
                              (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), operation a b c x m = x) :
  ∃ (m : ℝ), m = 4 :=
sorry

end solve_for_m_l292_292136


namespace max_subset_card_l292_292747

open Finset

theorem max_subset_card (S : Finset ℕ) (h₁ : ∀ x ∈ S, x ≤ 100) (h₂ : ∀ x y ∈ S, x = 3 * y → false) :
  card S ≤ 76 :=
sorry

end max_subset_card_l292_292747


namespace find_m_n_l292_292792

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := x^3 + m * x^2 + n * x + 1

theorem find_m_n (m n : ℝ) (x : ℝ) (hx : x ≠ 0 ∧ f x m n = 1 ∧ (3 * x^2 + 2 * m * x + n = 0) ∧ (∀ y, f y m n ≥ -31 ∧ f (-2) m n = -31)) :
  m = 12 ∧ n = 36 :=
sorry

end find_m_n_l292_292792


namespace students_before_intersection_equal_l292_292707

-- Define the conditions
def students_after_stop : Nat := 58
def percentage : Real := 0.40
def percentage_students_entered : Real := 12

-- Define the target number of students before stopping
def students_before_stop (total_after : Nat) (entered : Nat) : Nat :=
  total_after - entered

-- State the proof problem
theorem students_before_intersection_equal :
  ∃ (x : Nat), 
  percentage * (x : Real) = percentage_students_entered ∧ 
  students_before_stop students_after_stop x = 28 :=
by
  sorry

end students_before_intersection_equal_l292_292707


namespace min_pq_value_l292_292905

theorem min_pq_value : 
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 98 * p = q ^ 3 ∧ (∀ p' q' : ℕ, p' > 0 ∧ q' > 0 ∧ 98 * p' = q' ^ 3 → p' + q' ≥ p + q) ∧ p + q = 42 :=
sorry

end min_pq_value_l292_292905


namespace David_marks_in_Chemistry_l292_292428

theorem David_marks_in_Chemistry (e m p b avg c : ℕ) 
  (h1 : e = 91) 
  (h2 : m = 65) 
  (h3 : p = 82) 
  (h4 : b = 85) 
  (h5 : avg = 78) 
  (h6 : avg * 5 = e + m + p + b + c) :
  c = 67 := 
sorry

end David_marks_in_Chemistry_l292_292428


namespace cows_now_l292_292564

-- Defining all conditions
def initial_cows : ℕ := 39
def cows_died : ℕ := 25
def cows_sold : ℕ := 6
def cows_increase : ℕ := 24
def cows_bought : ℕ := 43
def cows_gift : ℕ := 8

-- Lean statement for the equivalent proof problem
theorem cows_now :
  let cows_left := initial_cows - cows_died
  let cows_after_selling := cows_left - cows_sold
  let cows_this_year_increased := cows_after_selling + cows_increase
  let cows_with_purchase := cows_this_year_increased + cows_bought
  let total_cows := cows_with_purchase + cows_gift
  total_cows = 83 :=
by
  sorry

end cows_now_l292_292564


namespace sum_mod_30_l292_292073

theorem sum_mod_30 (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 7) 
  (h3 : c % 30 = 18) : 
  (a + 2 * b + c) % 30 = 17 := 
by
  sorry

end sum_mod_30_l292_292073


namespace supervisors_per_bus_l292_292693

theorem supervisors_per_bus (total_supervisors : ℕ) (total_buses : ℕ) (H1 : total_supervisors = 21) (H2 : total_buses = 7) : (total_supervisors / total_buses = 3) :=
by
  sorry

end supervisors_per_bus_l292_292693


namespace side_length_of_square_l292_292979

theorem side_length_of_square 
  (x : ℝ) 
  (h₁ : 4 * x = 2 * (x * x)) :
  x = 2 :=
by 
  sorry

end side_length_of_square_l292_292979


namespace triangle_identity_l292_292140

theorem triangle_identity (a b c : ℝ) (B: ℝ) (hB: B = 120) :
    a^2 + a * c + c^2 - b^2 = 0 :=
by
  sorry

end triangle_identity_l292_292140


namespace extra_profit_is_60000_l292_292373

theorem extra_profit_is_60000 (base_house_cost special_house_cost base_house_price special_house_price : ℝ) :
  (special_house_cost = base_house_cost + 100000) →
  (special_house_price = 1.5 * base_house_price) →
  (base_house_price = 320000) →
  (special_house_price - base_house_price - 100000 = 60000) :=
by
  -- Definitions and conditions
  intro h1 h2 h3
  -- Placeholder for the eventual proof
  sorry

end extra_profit_is_60000_l292_292373


namespace triangle_longest_side_l292_292846

theorem triangle_longest_side (y : ℝ) (h₁ : 8 + (y + 5) + (3 * y + 2) = 45) : 
  ∃ s1 s2 s3, s1 = 8 ∧ s2 = y + 5 ∧ s3 = 3 * y + 2 ∧ (s1 + s2 + s3 = 45) ∧ (s3 = 24.5) := 
by
  sorry

end triangle_longest_side_l292_292846


namespace routes_from_A_to_B_in_4_by_3_grid_l292_292582

-- Problem: Given a 4 by 3 rectangular grid, and movement allowing only right (R) or down (D),
-- prove that the number of different routes from point A to point B is 35.
def routes_4_by_3 : ℕ :=
  let n_moves := 3 + 4  -- Total moves required are 3 Rs and 4 Ds
  let r_moves := 3      -- Number of Right moves (R)
  Nat.choose (n_moves) (r_moves) -- Number of ways to choose 3 Rs from 7 moves

theorem routes_from_A_to_B_in_4_by_3_grid : routes_4_by_3 = 35 := by {
  sorry -- Proof omitted
}

end routes_from_A_to_B_in_4_by_3_grid_l292_292582


namespace ice_cream_volume_l292_292742

theorem ice_cream_volume (r_cone h_cone r_hemisphere : ℝ) (h1 : r_cone = 3) (h2 : h_cone = 10) (h3 : r_hemisphere = 5) :
  (1 / 3 * π * r_cone^2 * h_cone + 2 / 3 * π * r_hemisphere^3) = (520 / 3) * π :=
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end ice_cream_volume_l292_292742


namespace find_x_l292_292518

theorem find_x (x : ℝ) (h : x^2 + 75 = (x - 20)^2) : x = 8.125 :=
by
  sorry

end find_x_l292_292518


namespace count_strictly_increasing_digits_l292_292594

theorem count_strictly_increasing_digits : 
  (∑ k in Finset.range 9, Nat.choose 9 k.succ) = 502 :=
by
  sorry

end count_strictly_increasing_digits_l292_292594


namespace min_a_plus_b_l292_292461

-- Given conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Equation of line L passing through point (4,1) with intercepts a and b
def line_eq (a b : ℝ) : Prop := (4 / a) + (1 / b) = 1

-- Proof statement
theorem min_a_plus_b (h : line_eq a b) : a + b ≥ 9 :=
sorry

end min_a_plus_b_l292_292461


namespace is_minimum_value_l292_292311

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem is_minimum_value (h : ∀ x > 0, f x ≥ 0) : ∃ (a : ℝ) (h : a > 0), f a = 0 :=
by {
  sorry
}

end is_minimum_value_l292_292311


namespace find_number_l292_292307

theorem find_number (x : ℕ) (h : 5 + 2 * (8 - x) = 15) : x = 3 :=
sorry

end find_number_l292_292307


namespace inverse_proportion_quadrants_l292_292692

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  ∀ (x y : ℝ), y = k^2 / x → (x > 0 → y > 0) ∧ (x < 0 → y < 0) :=
by
  sorry

end inverse_proportion_quadrants_l292_292692


namespace total_bill_amount_l292_292276

theorem total_bill_amount (n : ℕ) (cost_per_meal : ℕ) (gratuity_rate : ℚ) (total_bill_with_gratuity : ℚ)
  (h1 : n = 7) (h2 : cost_per_meal = 100) (h3 : gratuity_rate = 20 / 100) :
  total_bill_with_gratuity = (n * cost_per_meal : ℕ) * (1 + gratuity_rate) :=
sorry

end total_bill_amount_l292_292276


namespace colorings_count_l292_292989

theorem colorings_count :
  let grid_size := 7
  let valid_colorings := binomial (2 * grid_size) grid_size
  valid_colorings = 3432 :=
by
  let grid_size := 7
  let valid_colorings := binomial (2 * grid_size) grid_size
  have h : valid_colorings = 3432 := by {
    simp [valid_colorings],
    norm_num
  }
  exact h

end colorings_count_l292_292989


namespace find_x_pos_integer_l292_292663

theorem find_x_pos_integer (x : ℕ) (h : 0 < x) (n d : ℕ)
    (h1 : n = x^2 + 4 * x + 29)
    (h2 : d = 4 * x + 9)
    (h3 : n = d * x + 13) : 
    x = 2 := 
sorry

end find_x_pos_integer_l292_292663


namespace opposite_neg_fraction_l292_292227

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l292_292227


namespace gift_exchange_equation_l292_292884

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l292_292884


namespace women_at_dance_event_l292_292423

theorem women_at_dance_event (men women : ℕ)
  (each_man_dances_with : ℕ)
  (each_woman_dances_with : ℕ)
  (total_men : men = 18)
  (dances_per_man : each_man_dances_with = 4)
  (dances_per_woman : each_woman_dances_with = 3)
  (total_dance_pairs : men * each_man_dances_with = 72) :
  women = 24 := 
  by {
    sorry
  }

end women_at_dance_event_l292_292423


namespace sally_picked_peaches_l292_292830

-- Definitions from the conditions
def originalPeaches : ℕ := 13
def totalPeaches : ℕ := 55

-- The proof statement
theorem sally_picked_peaches : totalPeaches - originalPeaches = 42 := by
  sorry

end sally_picked_peaches_l292_292830


namespace percentage_employees_four_years_or_more_l292_292991

theorem percentage_employees_four_years_or_more 
  (x : ℝ) 
  (less_than_one_year : ℝ := 6 * x)
  (one_to_two_years : ℝ := 4 * x)
  (two_to_three_years : ℝ := 7 * x)
  (three_to_four_years : ℝ := 3 * x)
  (four_to_five_years : ℝ := 3 * x)
  (five_to_six_years : ℝ := 1 * x)
  (six_to_seven_years : ℝ := 1 * x)
  (seven_to_eight_years : ℝ := 2 * x)
  (total_employees : ℝ := 27 * x)
  (employees_four_years_or_more : ℝ := 7 * x) : 
  (employees_four_years_or_more / total_employees) * 100 = 25.93 := 
by
  sorry

end percentage_employees_four_years_or_more_l292_292991


namespace find_digit_A_l292_292301

def sum_of_digits_divisible_by_3 (A : ℕ) : Prop :=
  (2 + A + 3) % 3 = 0

theorem find_digit_A (A : ℕ) (hA : sum_of_digits_divisible_by_3 A) : A = 1 ∨ A = 4 :=
  sorry

end find_digit_A_l292_292301


namespace gcd_of_A_B_l292_292021

noncomputable def A (k : ℕ) := 2 * k
noncomputable def B (k : ℕ) := 5 * k

theorem gcd_of_A_B (k : ℕ) (h_lcm : Nat.lcm (A k) (B k) = 180) : Nat.gcd (A k) (B k) = 18 :=
by
  sorry

end gcd_of_A_B_l292_292021


namespace number_of_students_like_photography_l292_292289

variable (n_dislike n_like n_neutral : ℕ)

theorem number_of_students_like_photography :
  (3 * n_dislike = n_dislike + 12) →
  (5 * n_dislike = n_like) →
  n_like = 30 :=
by
  sorry

end number_of_students_like_photography_l292_292289


namespace complement_intersection_l292_292944

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def A_def : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B_def : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem complement_intersection :
  (U = univ ∧ A = A_def ∧ B = B_def) →
  (compl (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) :=
by
  sorry

end complement_intersection_l292_292944


namespace sector_area_l292_292787

theorem sector_area (α : ℝ) (l : ℝ) (S : ℝ) (hα : α = 60 * Real.pi / 180) (hl : l = 6 * Real.pi) : S = 54 * Real.pi :=
sorry

end sector_area_l292_292787


namespace friend_reading_time_l292_292946

theorem friend_reading_time (S : ℝ) (H1 : S > 0) (H2 : 3 = 2 * (3 / 2)) : 
  (1.5 / (5 * S)) = 0.3 :=
by 
  sorry

end friend_reading_time_l292_292946


namespace cosine_sum_formula_l292_292146

theorem cosine_sum_formula
  (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 4 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end cosine_sum_formula_l292_292146


namespace base_is_16_l292_292477

noncomputable def base_y_eq : Prop := ∃ base : ℕ, base ^ 8 = 4 ^ 16

theorem base_is_16 (base : ℕ) (h₁ : base ^ 8 = 4 ^ 16) : base = 16 :=
by
  sorry  -- Proof goes here

end base_is_16_l292_292477


namespace least_number_to_add_1055_to_div_by_23_l292_292553

theorem least_number_to_add_1055_to_div_by_23 : ∃ k : ℕ, (1055 + k) % 23 = 0 ∧ k = 3 :=
by
  sorry

end least_number_to_add_1055_to_div_by_23_l292_292553


namespace correct_equation_for_gift_exchanges_l292_292894

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l292_292894


namespace sum_of_digits_2_1989_and_5_1989_l292_292268

theorem sum_of_digits_2_1989_and_5_1989 
  (m n : ℕ) 
  (h1 : 10^(m-1) < 2^1989 ∧ 2^1989 < 10^m) 
  (h2 : 10^(n-1) < 5^1989 ∧ 5^1989 < 10^n) 
  (h3 : 2^1989 * 5^1989 = 10^1989) : 
  m + n = 1990 := 
sorry

end sum_of_digits_2_1989_and_5_1989_l292_292268


namespace cos_double_angle_l292_292016

-- Define the hypothesis
def cos_alpha (α : ℝ) : Prop := Real.cos α = 1 / 2

-- State the theorem
theorem cos_double_angle (α : ℝ) (h : cos_alpha α) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l292_292016


namespace sector_area_eq_l292_292786

theorem sector_area_eq (α : ℝ) (l : ℝ) (h1 : α = 60 * Real.pi / 180) (h2 : l = 6 * Real.pi) : 
  1 / 2 * l * (l * 3 / Real.pi) = 54 * Real.pi :=
by
  have r_eq : l / α = l * 3 / Real.pi := by
    calc
      l / α = l / (60 * Real.pi / 180) : by { rw [h1] }
      ... = l * (180 / 60) / Real.pi  : by { field_simp, ring }
      ... = l * 3 / Real.pi           : by { norm_num }
  rw [r_eq, h2]
  sorry

end sector_area_eq_l292_292786


namespace simplify_expression_l292_292520

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := 
by
  sorry

end simplify_expression_l292_292520


namespace bonnie_egg_count_indeterminable_l292_292911

theorem bonnie_egg_count_indeterminable
    (eggs_Kevin : ℕ)
    (eggs_George : ℕ)
    (eggs_Cheryl : ℕ)
    (diff_Cheryl_combined : ℕ)
    (c1 : eggs_Kevin = 5)
    (c2 : eggs_George = 9)
    (c3 : eggs_Cheryl = 56)
    (c4 : diff_Cheryl_combined = 29)
    (h₁ : eggs_Cheryl = diff_Cheryl_combined + (eggs_Kevin + eggs_George + some_children)) :
    ∀ (eggs_Bonnie : ℕ), ∃ some_children : ℕ, eggs_Bonnie = eggs_Bonnie :=
by
  -- The proof is omitted here
  sorry

end bonnie_egg_count_indeterminable_l292_292911


namespace land_to_water_time_ratio_l292_292059

-- Define the conditions
def distance_water : ℕ := 50
def distance_land : ℕ := 300
def speed_ratio : ℕ := 3

-- Define the Lean theorem statement
theorem land_to_water_time_ratio (x : ℝ) (hx : x > 0) : 
  (distance_land / (speed_ratio * x)) / (distance_water / x) = 2 := by
  sorry

end land_to_water_time_ratio_l292_292059


namespace correct_equation_l292_292666

-- Condition 1: Machine B transports 60 kg more per hour than Machine A
def machine_B_transports_more (x : ℝ) : Prop := 
  x + 60

-- Condition 2: Time to transport 500 kg by Machine A equals time 
-- to transport 800 kg by Machine B.
def transportation_time_eq (x : ℝ) : Prop :=
  500 / x = 800 / (x + 60)

-- Theorem statement: Prove the correct equation for given conditions
theorem correct_equation (x : ℝ) (h1 : machine_B_transports_more x) (h2 : transportation_time_eq x) : 
  500 / x = 800 / (x + 60) :=
  by
    sorry

end correct_equation_l292_292666


namespace find_highest_score_l292_292525

-- Define the conditions for the proof
section
  variable {runs_innings : ℕ → ℕ}

  -- Total runs scored in 46 innings
  def total_runs (average num_innings : ℕ) : ℕ := average * num_innings
  def total_runs_46_innings := total_runs 60 46
  def total_runs_excluding_H_L := total_runs 58 44

  -- Evaluated difference and sum of scores
  def diff_H_and_L : ℕ := 180
  def sum_H_and_L : ℕ := total_runs_46_innings - total_runs_excluding_H_L

  -- Define the proof goal
  theorem find_highest_score (H L : ℕ)
    (h1 : H - L = diff_H_and_L)
    (h2 : H + L = sum_H_and_L) :
    H = 194 :=
  by
    sorry

end

end find_highest_score_l292_292525


namespace expression_X_l292_292951

variable {a b X : ℝ}

theorem expression_X (h1 : a / b = 4 / 3) (h2 : (3 * a + 2 * b) / X = 3) : X = 2 * b := 
sorry

end expression_X_l292_292951


namespace intersection_point_is_neg3_l292_292180

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 9 * x + 15

theorem intersection_point_is_neg3 :
  ∃ a b : ℝ, (f a = b) ∧ (f b = a) ∧ (a, b) = (-3, -3) := sorry

end intersection_point_is_neg3_l292_292180


namespace exists_two_digit_pair_product_l292_292067

theorem exists_two_digit_pair_product (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (hprod : a * b = 8670) : a * b = 8670 :=
by
  exact hprod

end exists_two_digit_pair_product_l292_292067


namespace tom_needs_more_boxes_l292_292999

theorem tom_needs_more_boxes
    (living_room_length : ℕ)
    (living_room_width : ℕ)
    (box_coverage : ℕ)
    (already_installed : ℕ) :
    living_room_length = 16 →
    living_room_width = 20 →
    box_coverage = 10 →
    already_installed = 250 →
    (living_room_length * living_room_width - already_installed) / box_coverage = 7 :=
by
    intros h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    sorry

end tom_needs_more_boxes_l292_292999


namespace simplify_expression_l292_292519

variable (x y z : ℝ)

-- Statement of the problem to be proved.
theorem simplify_expression :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 
  (30 * x - 10 * z) :=
by
  -- Placeholder for the actual proof
  sorry

end simplify_expression_l292_292519


namespace arithmetic_mean_of_arithmetic_progression_l292_292048

variable (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ)

/-- General term of an arithmetic progression -/
def arithmetic_progression (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem arithmetic_mean_of_arithmetic_progression (k p : ℕ) (hk : 1 < k) :
  a k = (a (k - p) + a (k + p)) / 2 := by
  sorry

end arithmetic_mean_of_arithmetic_progression_l292_292048


namespace tens_digit_13_pow_2023_tens_digit_of_13_pow_2023_l292_292765

theorem tens_digit_13_pow_2023 :
  (13 ^ 2023) % 100 = 97 :=
sorry

theorem tens_digit_of_13_pow_2023 :
  ((13 ^ 2023) % 100) / 10 % 10 = 9 :=
by
  have h := tens_digit_13_pow_2023
  rw h
  norm_num
sorry

end tens_digit_13_pow_2023_tens_digit_of_13_pow_2023_l292_292765


namespace complement_set_l292_292332

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x | x < -2 ∨ x > 2}

-- The mathematical proof to be stated
theorem complement_set :
  U \ M = complement_M_in_U := sorry

end complement_set_l292_292332


namespace work_done_on_gas_in_process_1_2_l292_292812

variables (V₁ V₂ V₃ V₄ A₁₂ A₃₄ T n R : ℝ)

-- Both processes 1-2 and 3-4 are isothermal.
def is_isothermal_process := true -- Placeholder

-- Volumes relationship: for any given pressure, the volume in process 1-2 is exactly twice the volume in process 3-4.
def volumes_relation (V₁ V₂ V₃ V₄ : ℝ) : Prop :=
  V₁ = 2 * V₃ ∧ V₂ = 2 * V₄

-- Work done on a gas during an isothermal process can be represented as: A = 2 * A₃₄
def work_relation (A₁₂ A₃₄ : ℝ) : Prop :=
  A₁₂ = 2 * A₃₄

theorem work_done_on_gas_in_process_1_2
  (h_iso : is_isothermal_process)
  (h_vol : volumes_relation V₁ V₂ V₃ V₄)
  (h_work : work_relation A₁₂ A₃₄) :
  A₁₂ = 2 * A₃₄ :=
by 
  sorry

end work_done_on_gas_in_process_1_2_l292_292812


namespace f_3_minus_f_4_l292_292608

noncomputable def f : ℝ → ℝ := sorry
axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom initial_condition : f 1 = 1

theorem f_3_minus_f_4 : f 3 - f 4 = -1 :=
by
  sorry

end f_3_minus_f_4_l292_292608


namespace volunteers_distribution_l292_292316

theorem volunteers_distribution:
  let num_volunteers := 5
  let group_distribution := (2, 2, 1)
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end volunteers_distribution_l292_292316


namespace bicycle_wheels_l292_292993

theorem bicycle_wheels :
  ∀ (b : ℕ),
  let bicycles := 24
  let tricycles := 14
  let wheels_per_tricycle := 3
  let total_wheels := 90
  ((bicycles * b) + (tricycles * wheels_per_tricycle) = total_wheels) → b = 2 :=
by {
  sorry
}

end bicycle_wheels_l292_292993


namespace length_of_AB_l292_292938

def parabola_eq (y : ℝ) : Prop := y^2 = 8 * y

def directrix_x : ℝ := 2

def dist_to_y_axis (E : ℝ × ℝ) : ℝ := E.1

theorem length_of_AB (A B F E : ℝ × ℝ)
  (p : parabola_eq A.2) (q : parabola_eq B.2) 
  (F_focus : F.1 = 2 ∧ F.2 = 0) 
  (midpoint_E : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (E_distance_from_y_axis : dist_to_y_axis E = 3) : 
  (abs (A.1 - B.1) + abs (A.2 - B.2)) = 10 := 
sorry

end length_of_AB_l292_292938


namespace probability_of_six_and_queen_l292_292075

variable {deck : Finset (ℕ × String)}
variable (sixes : Finset (ℕ × String))
variable (queens : Finset (ℕ × String))

def standard_deck : Finset (ℕ × String) := sorry

-- Condition: the deck contains 52 cards (13 hearts, 13 clubs, 13 spades, 13 diamonds)
-- and it has 4 sixes and 4 Queens.
axiom h_deck_size : standard_deck.card = 52
axiom h_sixes : ∀ c ∈ standard_deck, c.1 = 6 → c ∈ sixes
axiom h_queens : ∀ c ∈ standard_deck, c.1 = 12 → c ∈ queens

-- Define the probability function for dealing cards
noncomputable def prob_first_six_and_second_queen : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_six_and_queen :
  prob_first_six_and_second_queen = 4 / 663 :=
by
  sorry

end probability_of_six_and_queen_l292_292075


namespace tree_planting_total_l292_292415

theorem tree_planting_total (t4 t5 t6 : ℕ) 
  (h1 : t4 = 30)
  (h2 : t5 = 2 * t4)
  (h3 : t6 = 3 * t5 - 30) : 
  t4 + t5 + t6 = 240 := 
by 
  sorry

end tree_planting_total_l292_292415


namespace equation_of_curve_t_circle_through_fixed_point_l292_292029

noncomputable def problem (x y : ℝ) : Prop :=
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -1)
  let O : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (x, y)
  let N : ℝ × ℝ := (0, y)
  (x + 1) * (x - 1) + y * y = y * (y + 1)

noncomputable def curve_t_equation (x : ℝ) : ℝ :=
  x^2 - 1

theorem equation_of_curve_t (x y : ℝ) 
  (h : problem x y) :
  y = curve_t_equation x := 
sorry

noncomputable def passing_through_fixed_point (x y : ℝ) : Prop :=
  let y := x^2 - 1
  let y' := 2 * x
  let P : ℝ × ℝ := (x, y)
  let Q_x := (4 * x^2 - 1) / (8 * x)
  let Q : ℝ × ℝ := (Q_x, -5 / 4)
  let H : ℝ × ℝ := (0, -3 / 4)
  (x * Q_x + (-3 / 4 - y) * ( -3 / 4 + 5 / 4)) = 0

theorem circle_through_fixed_point (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y = curve_t_equation x)
  (h : passing_through_fixed_point x y) :
  ∃ t : ℝ, passing_through_fixed_point x t ∧ t = -3 / 4 :=
sorry

end equation_of_curve_t_circle_through_fixed_point_l292_292029


namespace age_of_youngest_child_l292_292534

theorem age_of_youngest_child (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 55) → x = 5 :=
by
  intro h
  sorry

end age_of_youngest_child_l292_292534


namespace both_buyers_correct_l292_292865

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_mix_buyers : ℕ := 50

-- Define the number of buyers who purchase muffin mix
def muffin_mix_buyers : ℕ := 40

-- Define the number of buyers who purchase neither cake mix nor muffin mix
def neither_buyers : ℕ := 29

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- The assertion to be proved
theorem both_buyers_correct :
  neither_buyers = total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_buyers) :=
sorry

end both_buyers_correct_l292_292865


namespace hyperbola_product_slopes_constant_l292_292181

theorem hyperbola_product_slopes_constant (a b x0 y0 : ℝ) (h_a : a > 0) (h_b : b > 0) (hP : (x0 / a) ^ 2 - (y0 / b) ^ 2 = 1) (h_diff_a1_a2 : x0 ≠ a ∧ x0 ≠ -a) :
  (y0 / (x0 + a)) * (y0 / (x0 - a)) = b^2 / a^2 :=
by sorry

end hyperbola_product_slopes_constant_l292_292181


namespace remainder_of_2_pow_87_plus_3_mod_7_l292_292713

theorem remainder_of_2_pow_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end remainder_of_2_pow_87_plus_3_mod_7_l292_292713


namespace sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l292_292962

noncomputable def b : ℕ → ℚ
| 0     => 2
| 1     => 3
| (n+2) => 2 * b (n+1) + 3 * b n

theorem sum_bn_over_3_pow_n_plus_1_eq_2_over_5 :
  (∑' n : ℕ, (b n) / (3 ^ (n + 1))) = (2 / 5) :=
by
  sorry

end sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l292_292962


namespace cows_count_l292_292566

theorem cows_count (initial_cows last_year_deaths last_year_sales this_year_increase purchases gifts : ℕ)
  (h1 : initial_cows = 39)
  (h2 : last_year_deaths = 25)
  (h3 : last_year_sales = 6)
  (h4 : this_year_increase = 24)
  (h5 : purchases = 43)
  (h6 : gifts = 8) : 
  initial_cows - last_year_deaths - last_year_sales + this_year_increase + purchases + gifts = 83 := by
  sorry

end cows_count_l292_292566


namespace transformed_coords_of_point_l292_292151

noncomputable def polar_to_rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def transformed_coordinates (r θ : ℝ) : ℝ × ℝ :=
  let new_r := r ^ 3
  let new_θ := (3 * Real.pi / 2) * θ
  polar_to_rectangular_coordinates new_r new_θ

theorem transformed_coords_of_point (r θ : ℝ)
  (h_r : r = Real.sqrt (8^2 + 6^2))
  (h_cosθ : Real.cos θ = 8 / 10)
  (h_sinθ : Real.sin θ = 6 / 10)
  (coords_match : polar_to_rectangular_coordinates r θ = (8, 6)) :
  transformed_coordinates r θ = (-600, -800) :=
by
  -- The proof goes here
  sorry

end transformed_coords_of_point_l292_292151


namespace jenna_age_l292_292177

theorem jenna_age (D J : ℕ) (h1 : J = D + 5) (h2 : J + D = 21) (h3 : D = 8) : J = 13 :=
by
  sorry

end jenna_age_l292_292177


namespace simple_interest_rate_l292_292284

theorem simple_interest_rate :
  ∀ (P R : ℝ), 
  (R * 25 / 100 = 1) → 
  R = 4 := 
by
  intros P R h
  sorry

end simple_interest_rate_l292_292284


namespace num_pairs_of_nat_numbers_satisfying_eq_l292_292624

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l292_292624


namespace equal_roots_quadratic_l292_292152

theorem equal_roots_quadratic (k : ℝ) : (∃ (x : ℝ), x*(x + 2) + k = 0 ∧ ∀ y z, (y, z) = (x, x)) → k = 1 :=
sorry

end equal_roots_quadratic_l292_292152


namespace geoff_additional_votes_needed_l292_292490

-- Define the given conditions
def totalVotes : ℕ := 6000
def geoffPercentage : ℕ := 5 -- Represent 0.5% as 5 out of 1000 for better integer computation
def requiredPercentage : ℕ := 505 -- Represent 50.5% as 505 out of 1000 for better integer computation

-- Define the expressions for the number of votes received by Geoff and the votes required to win
def geoffVotes := (geoffPercentage * totalVotes) / 1000
def requiredVotes := (requiredPercentage * totalVotes) / 1000 + 1

-- The proposition to prove the additional number of votes needed for Geoff to win
theorem geoff_additional_votes_needed : requiredVotes - geoffVotes = 3001 := by sorry

end geoff_additional_votes_needed_l292_292490


namespace quadrant_iv_l292_292857

theorem quadrant_iv (x y : ℚ) (h1 : x = 1) (h2 : x - y = 12 / 5) (h3 : 6 * x + 5 * y = -1) :
  x = 1 ∧ y = -7 / 5 ∧ (12 / 5 > 0 ∧ -7 / 5 < 0) :=
by
  sorry

end quadrant_iv_l292_292857


namespace final_balance_is_60_million_l292_292093

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end final_balance_is_60_million_l292_292093


namespace monomial_exponent_match_l292_292633

theorem monomial_exponent_match (m : ℤ) (x y : ℂ) : (-x^(2*m) * y^3 = 2 * x^6 * y^3) → m = 3 := 
by 
  sorry

end monomial_exponent_match_l292_292633


namespace simplify_and_evaluate_l292_292211

variable (x y : ℝ)

theorem simplify_and_evaluate (h : x / y = 3) : 
  (1 + y^2 / (x^2 - y^2)) * (x - y) / x = 3 / 4 :=
by
  sorry

end simplify_and_evaluate_l292_292211


namespace eighteen_gon_vertex_number_l292_292575

theorem eighteen_gon_vertex_number (a b : ℕ) (P : ℕ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : P = a + b) : P = 38 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end eighteen_gon_vertex_number_l292_292575


namespace find_a_for_chord_length_l292_292479

theorem find_a_for_chord_length :
  ∀ a : ℝ, ((∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 ∧ (2 * x - y + a = 0)) 
  → ((2 * 1 - 1 + a = 0) → a = -1)) :=
by
  sorry

end find_a_for_chord_length_l292_292479


namespace asymptote_of_hyperbola_l292_292065

theorem asymptote_of_hyperbola :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) → y = x / 2 ∨ y = - x / 2 :=
sorry

end asymptote_of_hyperbola_l292_292065


namespace triangle_perimeter_l292_292571

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 19)
  (ineq1 : a + b > c) (ineq2 : a + c > b) (ineq3 : b + c > a) : a + b + c = 44 :=
by
  -- Proof omitted
  sorry

end triangle_perimeter_l292_292571


namespace clock_rings_in_a_day_l292_292287

theorem clock_rings_in_a_day (intervals : ℕ) (hours_in_a_day : ℕ) (time_between_rings : ℕ) : 
  intervals = hours_in_a_day / time_between_rings + 1 → intervals = 7 :=
sorry

end clock_rings_in_a_day_l292_292287


namespace metal_relative_atomic_mass_is_24_l292_292556

noncomputable def relative_atomic_mass (metal_mass : ℝ) (hcl_mass_percent : ℝ) (hcl_total_mass : ℝ) (mol_mass_hcl : ℝ) : ℝ :=
  let moles_hcl := (hcl_total_mass * hcl_mass_percent / 100) / mol_mass_hcl
  let maximum_molar_mass := metal_mass / (moles_hcl / 2)
  let minimum_molar_mass := metal_mass / (moles_hcl / 2)
  if 20 < maximum_molar_mass ∧ maximum_molar_mass < 28 then
    24
  else
    0

theorem metal_relative_atomic_mass_is_24
  (metal_mass_1 : ℝ)
  (metal_mass_2 : ℝ)
  (hcl_mass_percent : ℝ)
  (hcl_total_mass : ℝ)
  (mol_mass_hcl : ℝ)
  (moles_used_1 : ℝ)
  (moles_used_2 : ℝ)
  (excess : Bool)
  (complete : Bool) :
  relative_atomic_mass 3.5 18.25 50 36.5 = 24 :=
by
  sorry

end metal_relative_atomic_mass_is_24_l292_292556


namespace value_of_x_l292_292438

theorem value_of_x (a b x : ℝ) (h : x^2 + 4 * b^2 = (2 * a - x)^2) : 
  x = (a^2 - b^2) / a :=
by
  sorry

end value_of_x_l292_292438


namespace find_x_values_l292_292918

theorem find_x_values (x : ℝ) :
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ - (12 / 7 : ℝ) ≤ x ∧ x < - (3 / 4 : ℝ) :=
by
  sorry

end find_x_values_l292_292918


namespace circle_center_l292_292431

theorem circle_center (x y : ℝ) : (x^2 - 6 * x + y^2 + 2 * y = 20) → (x,y) = (3,-1) :=
by {
  sorry
}

end circle_center_l292_292431


namespace min_additional_coins_needed_l292_292746

/--
Alex has 15 friends, 90 coins, and needs to give each friend at least one coin with no two friends receiving the same number of coins.
Prove that the minimum number of additional coins he needs is 30.
-/
theorem min_additional_coins_needed (
  friends : ℕ := 15
  coins : ℕ := 90
) (h1 : friends = 15)
  (h2 : coins = 90) : 
  let total_required := (friends * (friends + 1)) / 2 in
  total_required - coins = 30 :=
by {
  have total_required_eq : total_required = (15 * (15 + 1)) / 2, from by simp [friends, h1],
  have total_required_eval : total_required = 120, from calc
    total_required = (15 * 16) / 2 : by rw total_required_eq
                 ... = 120        : by norm_num,
  calc
    total_required - coins = 120 - 90 : by rw [total_required_eval, h2]
                 ... = 30             : by norm_num
}

end min_additional_coins_needed_l292_292746


namespace find_amplitude_l292_292901

theorem find_amplitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : ∀ x, a * Real.cos (b * x - c) ≤ 3) 
  (h5 : ∀ x, abs (a * Real.cos (b * x - c) - a * Real.cos (b * (x + 2 * π / b) - c)) = 0) :
  a = 3 := 
sorry

end find_amplitude_l292_292901


namespace probability_white_grid_after_process_l292_292402

-- Definitions related to the problem's conditions:
def is_white : Prop := sorry -- Define this as a unit square being white
def is_black : Prop := sorry -- Define this as a unit square being black
def probability_white : ℚ := 1/2 -- Each unit square has a probability of being white

-- Using the fact that conditions of independence and equal likelihood are given:
def independent_colors (squares : List Prop) : Prop := sorry -- To describe independence of color choice

-- Defining the conditional transformation process:
def rotate_180 (grid : Array (Array Prop)) : Array (Array Prop) := sorry -- Rotates the grid 180 degrees

-- The proof problem statement:
theorem probability_white_grid_after_process :
  -- Given: probability of each unit square being white is 1/2
  (∀ sq : Prop, sq = is_white ∨ sq = is_black) →
  -- Given: Each square's color is chosen independently
  (independent_colors [is_white, is_black]) →
  -- Our goal to prove:
  -- The probability that the grid is entirely white after applying the described process (rotated and paint change) is 1/512.
  probability_white * (1/4) * (1/4) * (1/4) * (1/4) = 1/512 :=
sorry

end probability_white_grid_after_process_l292_292402


namespace factorize_x4_plus_81_l292_292436

theorem factorize_x4_plus_81 : 
  ∀ x : ℝ, 
    (x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9)) :=
by
  intro x
  sorry

end factorize_x4_plus_81_l292_292436


namespace equipment_unit_prices_purchasing_scenarios_l292_292809

theorem equipment_unit_prices
  (x : ℝ)
  (price_A_eq_price_B_minus_10 : ∀ y, ∃ z, z = y + 10)
  (eq_purchases_equal_cost_A : ∀ n : ℕ, 300 / x = n)
  (eq_purchases_equal_cost_B : ∀ n : ℕ, 360 / (x + 10) = n) :
  x = 50 ∧ (x + 10) = 60 :=
by
  sorry

theorem purchasing_scenarios
  (m n : ℕ)
  (price_A : ℝ := 50)
  (price_B : ℝ := 60)
  (budget : ℝ := 1000)
  (purchase_eq_budget : 50 * m + 60 * n = 1000)
  (pos_integers : m > 0 ∧ n > 0) :
  (m = 14 ∧ n = 5) ∨ (m = 8 ∧ n = 10) ∨ (m = 2 ∧ n = 15) :=
by
  sorry

end equipment_unit_prices_purchasing_scenarios_l292_292809


namespace girls_doctors_percentage_l292_292488

-- Define the total number of students in the class
variables (total_students : ℕ)

-- Define the proportions given in the problem
def proportion_boys : ℚ := 3 / 5
def proportion_boys_who_want_to_be_doctors : ℚ := 1 / 3
def proportion_doctors_who_are_boys : ℚ := 2 / 5

-- Compute the proportion of boys in the class who want to be doctors
def proportion_boys_as_doctors := proportion_boys * proportion_boys_who_want_to_be_doctors

-- Compute the proportion of girls in the class
def proportion_girls := 1 - proportion_boys

-- Compute the number of girls who want to be doctors compared to boys
def proportion_girls_as_doctors := (1 - proportion_doctors_who_are_boys) / proportion_doctors_who_are_boys * proportion_boys_as_doctors

-- Compute the proportion of girls who want to be doctors
def proportion_girls_who_want_to_be_doctors := proportion_girls_as_doctors / proportion_girls

-- Define the expected percentage of girls who want to be doctors
def expected_percentage_girls_who_want_to_be_doctors : ℚ := 75 / 100

-- The theorem we need to prove
theorem girls_doctors_percentage : proportion_girls_who_want_to_be_doctors * 100 = expected_percentage_girls_who_want_to_be_doctors :=
sorry

end girls_doctors_percentage_l292_292488


namespace elvis_ralph_matchsticks_l292_292125

/-- 
   Elvis and Ralph are making square shapes with matchsticks from a box containing 
   50 matchsticks. Elvis makes 4-matchstick squares and Ralph makes 8-matchstick 
   squares. If Elvis makes 5 squares and Ralph makes 3, prove the number of matchsticks 
   left in the box is 6. 
-/
def matchsticks_left_in_box
  (initial_matchsticks : ℕ)
  (elvis_squares : ℕ)
  (elvis_matchsticks : ℕ)
  (ralph_squares : ℕ)
  (ralph_matchsticks : ℕ)
  (elvis_squares_count : ℕ)
  (ralph_squares_count : ℕ) : ℕ :=
  initial_matchsticks - (elvis_squares_count * elvis_matchsticks + ralph_squares_count * ralph_matchsticks)

theorem elvis_ralph_matchsticks : matchsticks_left_in_box 50 4 5 8 3 = 6 := 
  sorry

end elvis_ralph_matchsticks_l292_292125


namespace children_neither_happy_nor_sad_l292_292511

theorem children_neither_happy_nor_sad (total_children happy_children sad_children : ℕ)
  (total_boys total_girls happy_boys sad_girls boys_neither_happy_nor_sad : ℕ)
  (h₀ : total_children = 60)
  (h₁ : happy_children = 30)
  (h₂ : sad_children = 10)
  (h₃ : total_boys = 19)
  (h₄ : total_girls = 41)
  (h₅ : happy_boys = 6)
  (h₆ : sad_girls = 4)
  (h₇ : boys_neither_happy_nor_sad = 7) :
  total_children - happy_children - sad_children = 20 :=
by
  sorry

end children_neither_happy_nor_sad_l292_292511


namespace common_point_exists_l292_292674

theorem common_point_exists (a b c : ℝ) :
  ∃ x y : ℝ, y = a * x ^ 2 - b * x + c ∧ y = b * x ^ 2 - c * x + a ∧ y = c * x ^ 2 - a * x + b :=
  sorry

end common_point_exists_l292_292674


namespace average_of_three_numbers_l292_292383

theorem average_of_three_numbers (a b c : ℝ)
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76) :
  (a + b + c) / 3 = 35 := 
sorry

end average_of_three_numbers_l292_292383


namespace matrix_calculation_l292_292186

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l292_292186


namespace snack_eaters_remaining_l292_292277

theorem snack_eaters_remaining 
  (initial_population : ℕ)
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (first_half_leave : ℕ)
  (new_outsiders_2 : ℕ)
  (second_leave : ℕ)
  (final_half_leave : ℕ) 
  (h_initial_population : initial_population = 200)
  (h_initial_snackers : initial_snackers = 100)
  (h_new_outsiders_1 : new_outsiders_1 = 20)
  (h_first_half_leave : first_half_leave = (initial_snackers + new_outsiders_1) / 2)
  (h_new_outsiders_2 : new_outsiders_2 = 10)
  (h_second_leave : second_leave = 30)
  (h_final_half_leave : final_half_leave = (first_half_leave + new_outsiders_2 - second_leave) / 2) : 
  final_half_leave = 20 := 
sorry

end snack_eaters_remaining_l292_292277


namespace joan_change_received_l292_292653

theorem joan_change_received :
  let cat_toy_cost := 8.77
  let cage_cost := 10.97
  let payment := 20.00
  let total_cost := cat_toy_cost + cage_cost
  let change_received := payment - total_cost
  change_received = 0.26 :=
by
  sorry

end joan_change_received_l292_292653


namespace min_ab_value_l292_292784

theorem min_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / a) + (4 / b) = 1) : ab ≥ 16 :=
by
  sorry

end min_ab_value_l292_292784


namespace math_proof_l292_292003

noncomputable def f (ω x : ℝ) := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem math_proof (h1 : ∀ x, f ω x = f ω (x + π)) (h2 : 0 < ω) :
  (ω = 2) ∧ (f 2 (-5 * Real.pi / 6) = 0) ∧ ¬∀ x : ℝ, x ∈ Set.Ioo (Real.pi / 3) (11 * Real.pi / 12) → 
  (∃ x₁ x₂ : ℝ, f 2 x₁ < f 2 x₂) ∧ (∀ x : ℝ, f 2 (x - Real.pi / 3) ≠ Real.cos (2 * x - Real.pi / 6)) := 
by
  sorry

end math_proof_l292_292003


namespace true_proposition_p_and_q_l292_292939

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Define the proposition q
def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

-- Statement to prove the conjunction p ∧ q
theorem true_proposition_p_and_q : p ∧ q := 
by 
    sorry

end true_proposition_p_and_q_l292_292939


namespace number_of_eighth_graders_l292_292954

theorem number_of_eighth_graders (x y : ℕ) :
  (x > 0) ∧ (y > 0) ∧ (8 + x * y = (x * (x + 3) - 14) / 2) →
  x = 7 ∨ x = 14 :=
by
  sorry

end number_of_eighth_graders_l292_292954


namespace speech_competition_sequences_l292_292498

theorem speech_competition_sequences
    (contestants : Fin 5 → Prop)
    (girls boys : Fin 5 → Prop)
    (girl_A : Fin 5)
    (not_girl_A_first : ¬contestants 0)
    (no_consecutive_boys : ∀ i, boys i → ¬boys (i + 1))
    (count_girls : ∀ x, girls x → x = girl_A ∨ (contestants x ∧ ¬boys x))
    (count_boys : ∀ x, (boys x) → contestants x)
    (total_count : Fin 5 → Fin 5 → ℕ)
    (correct_answer : total_count = 276) : 
    ∃ seq_count, seq_count = 276 := 
sorry

end speech_competition_sequences_l292_292498


namespace probability_three_draws_one_white_l292_292704

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_white_balls + num_black_balls

def probability_one_white_three_draws : ℚ := 
  (num_white_balls / total_balls) * 
  ((num_black_balls - 1) / (total_balls - 1)) * 
  ((num_black_balls - 2) / (total_balls - 2)) * 3

theorem probability_three_draws_one_white :
  probability_one_white_three_draws = 12 / 35 := by sorry

end probability_three_draws_one_white_l292_292704


namespace intersection_of_lines_l292_292442

theorem intersection_of_lines :
  ∃ x y : ℚ, 3 * y = -2 * x + 6 ∧ 2 * y = 6 * x - 4 ∧ x = 12 / 11 ∧ y = 14 / 11 := by
  sorry

end intersection_of_lines_l292_292442


namespace number_of_arrangements_l292_292493

theorem number_of_arrangements (P : Fin 5 → Type) (youngest : Fin 5) 
  (h_in_not_first_last : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4 → i ≠ youngest) : 
  ∃ n, n = 72 := 
by
  sorry

end number_of_arrangements_l292_292493


namespace opposite_of_neg_frac_l292_292233

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l292_292233


namespace fuel_cost_per_liter_l292_292489

def service_cost_per_vehicle : ℝ := 2.20
def num_minivans : ℕ := 3
def num_trucks : ℕ := 2
def total_cost : ℝ := 347.7
def mini_van_tank_capacity : ℝ := 65
def truck_tank_increase : ℝ := 1.2
def truck_tank_capacity : ℝ := mini_van_tank_capacity * (1 + truck_tank_increase)

theorem fuel_cost_per_liter : 
  let total_service_cost := (num_minivans + num_trucks) * service_cost_per_vehicle
  let total_capacity_minivans := num_minivans * mini_van_tank_capacity
  let total_capacity_trucks := num_trucks * truck_tank_capacity
  let total_fuel_capacity := total_capacity_minivans + total_capacity_trucks
  let fuel_cost := total_cost - total_service_cost
  let cost_per_liter := fuel_cost / total_fuel_capacity
  cost_per_liter = 0.70 := 
  sorry

end fuel_cost_per_liter_l292_292489


namespace find_a6_a7_l292_292495

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Given Conditions
axiom cond1 : arithmetic_sequence a d
axiom cond2 : a 2 + a 4 + a 9 + a 11 = 32

-- Proof Problem
theorem find_a6_a7 : a 6 + a 7 = 16 :=
  sorry

end find_a6_a7_l292_292495


namespace ratio_pq_equilateral_triangle_l292_292078

theorem ratio_pq_equilateral_triangle (p q : ℝ) (a : ℝ) 
  (h : 0 < p ∧ 0 < q ∧ 0 < a) 
  (area_relation : (19 / 64) * a^2 = p^2 + q^2 - p * q) : 
  p / q = 5 / 3 ∨ p / q = 3 / 5 := 
sorry

end ratio_pq_equilateral_triangle_l292_292078


namespace fill_in_the_blanks_l292_292437

theorem fill_in_the_blanks :
  (9 / 18 = 0.5) ∧
  (27 / 54 = 0.5) ∧
  (50 / 100 = 0.5) ∧
  (10 / 20 = 0.5) ∧
  (5 / 10 = 0.5) :=
by
  sorry

end fill_in_the_blanks_l292_292437


namespace possible_values_l292_292309

theorem possible_values (m n : ℕ) (h1 : 10 ≥ m) (h2 : m > n) (h3 : n ≥ 4) (h4 : (m - n) ^ 2 = m + n) :
    (m, n) = (10, 6) :=
sorry

end possible_values_l292_292309


namespace propositionA_necessary_for_propositionB_l292_292115

variable {α : Type*} [TopologicalSpace α] [NormedGroup α] [NormedSpace ℝ α]
variable {f : ℝ → α}
variable {x : ℝ}

-- Proposition A: f'(x) = 0
def propA (f : ℝ → α) (x : ℝ) : Prop :=
  deriv f x = 0

-- Proposition B: f(x) has an extremum at x = c
def has_extremum (f : ℝ → α) (c : ℝ) : Prop :=
  ∃ x, (((∀ y, x < y → f y ≥ f x) ∨ (∀ y, y < x → f y ≥ f x)) ∨
        ((∀ y, x < y → f y ≤ f x) ∨ (∀ y, y < x → f y ≤ f x))) 

theorem propositionA_necessary_for_propositionB (f : ℝ → α) (c : ℝ) :
  has_extremum f c → propA f c :=
by
  intros h,
  sorry

end propositionA_necessary_for_propositionB_l292_292115


namespace twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l292_292297

theorem twelve_plus_four_times_five_minus_five_cubed_equals_twelve :
  12 + 4 * (5 - 10 / 2) ^ 3 = 12 := by
  sorry

end twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l292_292297


namespace difference_is_correct_l292_292727

-- Define the digits
def digits : List ℕ := [9, 2, 1, 5]

-- Define the largest number that can be formed by these digits
def largestNumber : ℕ :=
  1000 * 9 + 100 * 5 + 10 * 2 + 1 * 1

-- Define the smallest number that can be formed by these digits
def smallestNumber : ℕ :=
  1000 * 1 + 100 * 2 + 10 * 5 + 1 * 9

-- Define the correct difference
def difference : ℕ :=
  largestNumber - smallestNumber

-- Theorem statement
theorem difference_is_correct : difference = 8262 :=
by
  sorry

end difference_is_correct_l292_292727


namespace value_of_a_l292_292723

theorem value_of_a 
  (a : ℝ) 
  (h : 0.005 * a = 0.85) : 
  a = 170 :=
sorry

end value_of_a_l292_292723


namespace numPeopleToLeftOfKolya_l292_292980

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l292_292980


namespace field_width_calculation_l292_292695

theorem field_width_calculation (w : ℝ) (h_length : length = 24) (h_length_width_relation : length = 2 * w - 3) : w = 13.5 :=
by 
  sorry

end field_width_calculation_l292_292695


namespace foxes_wolves_bears_num_l292_292599

-- Definitions and theorem statement
def num_hunters := 45
def num_rabbits := 2008
def rabbits_per_fox := 59
def rabbits_per_wolf := 41
def rabbits_per_bear := 40

theorem foxes_wolves_bears_num (x y z : ℤ) : 
  x + y + z = num_hunters → 
  rabbits_per_wolf * x + rabbits_per_fox * y + rabbits_per_bear * z = num_rabbits → 
  x = 18 ∧ y = 10 ∧ z = 17 :=
by 
  intro h1 h2 
  sorry

end foxes_wolves_bears_num_l292_292599


namespace num_pairs_of_nat_numbers_satisfying_eq_l292_292625

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l292_292625


namespace Jeff_has_20_trucks_l292_292176

theorem Jeff_has_20_trucks
  (T C : ℕ)
  (h1 : C = 2 * T)
  (h2 : T + C = 60) :
  T = 20 :=
sorry

end Jeff_has_20_trucks_l292_292176


namespace value_of_x_l292_292703

theorem value_of_x (z : ℕ) (y : ℕ) (x : ℕ) 
  (h₁ : y = z / 5)
  (h₂ : x = y / 2)
  (h₃ : z = 60) : 
  x = 6 :=
by
  sorry

end value_of_x_l292_292703


namespace velvet_needed_for_box_l292_292971

theorem velvet_needed_for_box :
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  long_side_area + short_side_area + top_and_bottom_area = 236 := by
{
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  sorry
}

end velvet_needed_for_box_l292_292971


namespace amy_books_l292_292825

theorem amy_books (maddie_books : ℕ) (luisa_books : ℕ) (amy_luisa_more_than_maddie : ℕ) (h1 : maddie_books = 15) (h2 : luisa_books = 18) (h3 : amy_luisa_more_than_maddie = maddie_books + 9) : ∃ (amy_books : ℕ), amy_books = amy_luisa_more_than_maddie - luisa_books ∧ amy_books = 6 :=
by
  have total_books := 24
  sorry

end amy_books_l292_292825


namespace prob_exactly_M_laws_expected_laws_included_l292_292340

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l292_292340


namespace tree_planting_activity_l292_292418

noncomputable def total_trees (grade4: ℕ) (grade5: ℕ) (grade6: ℕ) :=
  grade4 + grade5 + grade6

theorem tree_planting_activity:
  let grade4 := 30 in
  let grade5 := 2 * grade4 in
  let grade6 := (3 * grade5) - 30 in
  total_trees grade4 grade5 grade6 = 240 :=
by
  let grade4 := 30
  let grade5 := 2 * grade4
  let grade6 := (3 * grade5) - 30
  show total_trees grade4 grade5 grade6 = 240
  -- step-by-step calculations omitted
  sorry

end tree_planting_activity_l292_292418


namespace complex_number_in_third_quadrant_l292_292645

open Complex

noncomputable def complex_number : ℂ := (1 - 3 * I) / (1 + 2 * I)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem complex_number_in_third_quadrant : in_third_quadrant complex_number :=
sorry

end complex_number_in_third_quadrant_l292_292645


namespace Z_3_5_value_l292_292953

def Z (a b : ℕ) : ℕ :=
  b + 12 * a - a ^ 2

theorem Z_3_5_value : Z 3 5 = 32 := by
  sorry

end Z_3_5_value_l292_292953


namespace discount_problem_l292_292379

theorem discount_problem (x : ℝ) (h : 560 * (1 - x / 100) * 0.70 = 313.6) : x = 20 := 
by
  sorry

end discount_problem_l292_292379


namespace wire_length_after_cuts_l292_292634

-- Given conditions as parameters
def initial_length_cm : ℝ := 23.3
def first_cut_mm : ℝ := 105
def second_cut_cm : ℝ := 4.6

-- Final statement to be proved
theorem wire_length_after_cuts (ell : ℝ) (c1 : ℝ) (c2 : ℝ) : (ell = 23.3) → (c1 = 105) → (c2 = 4.6) → 
  (ell * 10 - c1 - c2 * 10 = 82) := sorry

end wire_length_after_cuts_l292_292634


namespace sqrt_expression_l292_292758

theorem sqrt_expression : Real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end sqrt_expression_l292_292758


namespace total_trees_planted_l292_292416

theorem total_trees_planted :
  let fourth_graders := 30
  let fifth_graders := 2 * fourth_graders
  let sixth_graders := 3 * fifth_graders - 30
  fourth_graders + fifth_graders + sixth_graders = 240 :=
by
  sorry

end total_trees_planted_l292_292416


namespace find_x_l292_292824

def operation_star (a b c d : ℤ) : ℤ × ℤ :=
  (a + c, b - 2 * d)

theorem find_x (x y : ℤ) (h : operation_star (x+1) (y-1) 1 3 = (2, -4)) : x = 0 :=
by 
  sorry

end find_x_l292_292824


namespace real_values_of_x_l292_292917

theorem real_values_of_x (x : ℝ) (h : x ≠ 4) :
  (x * (x + 1) / (x - 4)^2 ≥ 15) ↔ (x ≤ 3 ∨ (40/7 < x ∧ x < 4) ∨ x > 4) :=
by sorry

end real_values_of_x_l292_292917


namespace determine_p_l292_292117

theorem determine_p (p : ℝ) (h : (2 * p - 1) * (-1)^2 + 2 * (1 - p) * (-1) + 3 * p = 0) : p = 3 / 7 := by
  sorry

end determine_p_l292_292117


namespace norm_of_w_l292_292450

variable (u v : EuclideanSpace ℝ (Fin 2)) 
variable (hu : ‖u‖ = 3) (hv : ‖v‖ = 5) 
variable (h_orthogonal : inner u v = 0)

theorem norm_of_w :
  ‖4 • u - 2 • v‖ = 2 * Real.sqrt 61 := by
  sorry

end norm_of_w_l292_292450


namespace petroleum_crude_oil_problem_l292_292411

variables (x y : ℝ)

theorem petroleum_crude_oil_problem (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 27.5) : y = 30 :=
by
  -- Proof would go here
  sorry

end petroleum_crude_oil_problem_l292_292411


namespace part_a_l292_292267

theorem part_a (x α : ℝ) (hα : 0 < α ∧ α < 1) (hx : x ≥ 0) : x^α - α * x ≤ 1 - α :=
sorry

end part_a_l292_292267


namespace polynomial_roots_l292_292776

theorem polynomial_roots :
  Polynomial.roots (3 * X^4 + 11 * X^3 - 28 * X^2 + 10 * X) = {0, 1/3, 2, -5} :=
sorry

end polynomial_roots_l292_292776


namespace minimum_a_for_f_leq_one_range_of_a_for_max_value_l292_292153

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * log x - (1 / 3) * a * x^3 + 2 * x

theorem minimum_a_for_f_leq_one :
  ∀ {a : ℝ}, (a > 0) → (∀ x : ℝ, f a x ≤ 1) → (a ≥ 3) :=
sorry

theorem range_of_a_for_max_value :
  ∀ {a : ℝ}, (a > 0) → (∃ B : ℝ, ∀ x : ℝ, f a x ≤ B) ↔ (0 < a ∧ a ≤ (3 / 2) * exp 3) :=
sorry

end minimum_a_for_f_leq_one_range_of_a_for_max_value_l292_292153


namespace problem1_problem2_problem3_problem4_l292_292903

theorem problem1 : -20 + (-14) - (-18) - 13 = -29 := by
  sorry

theorem problem2 : (-2) * 3 + (-5) - 4 / (-1/2) = -3 := by
  sorry

theorem problem3 : (-3/8 - 1/6 + 3/4) * (-24) = -5 := by
  sorry

theorem problem4 : -81 / (9/4) * abs (-4/9) - (-3)^3 / 27 = -15 := by
  sorry

end problem1_problem2_problem3_problem4_l292_292903


namespace statement_A_statement_A_statement_C_statement_D_l292_292326

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem statement_A (x : ℝ) (hx : x > 1) : f x > 0 := sorry

theorem statement_A' (x : ℝ) (hx : 0 < x ∧ x < 1) : f x < 0 := sorry

theorem statement_C : Set.range f = Set.Ici (-1 / (2 * Real.exp 1)) := sorry

theorem statement_D (x : ℝ) : f x ≥ x - 1 := sorry

end statement_A_statement_A_statement_C_statement_D_l292_292326


namespace angle_equiv_470_110_l292_292881

theorem angle_equiv_470_110 : ∃ (k : ℤ), 470 = k * 360 + 110 :=
by
  use 1
  exact rfl

end angle_equiv_470_110_l292_292881


namespace opposite_of_neg2_is_2_l292_292842

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_neg2_is_2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_is_2_l292_292842


namespace candidate_X_votes_l292_292345

theorem candidate_X_votes (Z : ℕ) (Y : ℕ) (X : ℕ) (hZ : Z = 25000) 
                          (hY : Y = Z - (2 / 5) * Z) 
                          (hX : X = Y + (1 / 2) * Y) : 
                          X = 22500 :=
by
  sorry

end candidate_X_votes_l292_292345


namespace bicycle_total_distance_l292_292218

noncomputable def front_wheel_circumference : ℚ := 4/3
noncomputable def rear_wheel_circumference : ℚ := 3/2
noncomputable def extra_revolutions : ℕ := 25

theorem bicycle_total_distance :
  (front_wheel_circumference * extra_revolutions + (rear_wheel_circumference * 
  ((front_wheel_circumference * extra_revolutions) / (rear_wheel_circumference - front_wheel_circumference))) = 300) := sorry

end bicycle_total_distance_l292_292218


namespace evaluate_expression_l292_292772

theorem evaluate_expression : 64 ^ (-1/3 : ℤ) + 81 ^ (-1/4 : ℤ) = (7/12 : ℚ) :=
by
  -- Given conditions
  have h1 : (64 : ℝ) = (2 ^ 6 : ℝ) := by norm_num,
  have h2 : (81 : ℝ) = (3 ^ 4 : ℝ) := by norm_num,
  -- Definitions based on given conditions
  have expr1 : (64 : ℝ) ^ (-1 / 3 : ℝ) = (2 ^ 6 : ℝ) ^ (-1 / 3 : ℝ) := by rw h1,
  have expr2 : (81 : ℝ) ^ (-1 / 4 : ℝ) = (3 ^ 4 : ℝ) ^ (-1 / 4 : ℝ) := by rw h2,
  -- Simplify expressions (details omitted, handled by sorry)
  sorry

end evaluate_expression_l292_292772


namespace gathering_gift_exchange_l292_292897

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l292_292897


namespace cement_used_tess_street_l292_292207

-- Define the given conditions
def cement_used_lexi_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Define the statement to prove the amount of cement used for Tess's street
theorem cement_used_tess_street : total_cement_used - cement_used_lexi_street = 5.1 :=
by
  sorry

end cement_used_tess_street_l292_292207


namespace expression_equivalence_l292_292840

theorem expression_equivalence (a b : ℝ) :
  let P := a + b
  let Q := a - b
  (P + Q)^2 / (P - Q)^2 - (P - Q)^2 / (P + Q)^2 = (a^2 + b^2) * (a^2 - b^2) / (a^2 * b^2) :=
by
  sorry

end expression_equivalence_l292_292840


namespace remainder_of_polynomial_l292_292598

noncomputable def P (x : ℝ) := 3 * x^5 - 2 * x^3 + 5 * x^2 - 8
noncomputable def D (x : ℝ) := x^2 + 3 * x + 2
noncomputable def R (x : ℝ) := 64 * x + 60

theorem remainder_of_polynomial :
  ∀ x : ℝ, P x % D x = R x :=
sorry

end remainder_of_polynomial_l292_292598


namespace probability_of_green_ball_l292_292643

theorem probability_of_green_ball
  (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = 10)
  (h2 : red_balls = 3)
  (h3 : blue_balls = 2)
  (h4 : green_balls = total_balls - (red_balls + blue_balls)) :
  (green_balls : ℚ) / total_balls = 1 / 2 :=
sorry

end probability_of_green_ball_l292_292643


namespace white_tshirts_per_package_l292_292081

theorem white_tshirts_per_package (p t : ℕ) (h1 : p = 28) (h2 : t = 56) :
  t / p = 2 :=
by 
  sorry

end white_tshirts_per_package_l292_292081


namespace union_complements_eq_l292_292659

-- Definitions as per conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define complements
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof statement
theorem union_complements_eq :
  (C_UA ∪ C_UB) = {0, 1, 4} :=
by
  sorry

end union_complements_eq_l292_292659


namespace attendees_gift_exchange_l292_292889

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l292_292889


namespace system_solution_l292_292145

theorem system_solution (x y : ℝ) (h1 : 4 * x - y = 3) (h2 : x + 6 * y = 17) : x + y = 4 :=
by
  sorry

end system_solution_l292_292145


namespace velvet_needed_for_box_l292_292970

theorem velvet_needed_for_box :
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  long_side_area + short_side_area + top_and_bottom_area = 236 := by
{
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  sorry
}

end velvet_needed_for_box_l292_292970


namespace compute_expression_l292_292756

theorem compute_expression : (3 + 9)^3 + (3^3 + 9^3) = 2484 := by
  sorry

end compute_expression_l292_292756


namespace count_valid_numbers_l292_292797

-- Define conditions
def is_multiple_of_10 (n : ℕ) : Prop := n % 10 = 0
def is_positive (n : ℕ) : Prop := n > 0
def less_than_200 (n : ℕ) : Prop := n < 200

-- Define the set of numbers we are interested in
def valid_numbers (n : ℕ) : Prop := is_multiple_of_10 n ∧ is_positive n ∧ less_than_200 n

-- Statement to be proven
theorem count_valid_numbers : ∃ (count : ℕ), count = 20 ∧ (∀ n, valid_numbers n ↔ n ∈ finset.range(200) ∧ n % 10 = 0 ∧ n > 0) := 
by
  sorry

end count_valid_numbers_l292_292797


namespace distance_to_destination_l292_292848

theorem distance_to_destination :
  ∀ (D : ℝ) (T : ℝ),
    (15:ℝ) = T →
    (30:ℝ) = T / 2 →
    T - (T / 2) = 3 →
    D = 15 * T → D = 90 :=
by
  intros D T Theon_speed Yara_speed time_difference distance_calc
  sorry

end distance_to_destination_l292_292848


namespace john_candies_correct_l292_292648

variable (Bob_candies : ℕ) (Mary_candies : ℕ)
          (Sue_candies : ℕ) (Sam_candies : ℕ)
          (Total_candies : ℕ) (John_candies : ℕ)

axiom bob_has : Bob_candies = 10
axiom mary_has : Mary_candies = 5
axiom sue_has : Sue_candies = 20
axiom sam_has : Sam_candies = 10
axiom total_has : Total_candies = 50

theorem john_candies_correct : 
  Bob_candies + Mary_candies + Sue_candies + Sam_candies + John_candies = Total_candies → John_candies = 5 := by
sorry

end john_candies_correct_l292_292648


namespace machine_transport_equation_l292_292665

theorem machine_transport_equation (x : ℝ) :
  (∀ (rateA rateB : ℝ), rateB = rateA + 60 → (500 / rateA = 800 / rateB) → rateA = x → rateB = x + 60) :=
by
  sorry

end machine_transport_equation_l292_292665


namespace solve_for_x_and_y_l292_292335

theorem solve_for_x_and_y : 
  (∃ x y : ℝ, 0.65 * 900 = 0.40 * x ∧ 0.35 * 1200 = 0.25 * y) → 
  ∃ x y : ℝ, x + y = 3142.5 :=
by
  sorry

end solve_for_x_and_y_l292_292335


namespace tile_arrangement_probability_l292_292680

theorem tile_arrangement_probability :
  let X := 4  -- Number of tiles marked X
  let O := 2  -- Number of tiles marked O
  let total := 6  -- Total number of tiles
  let arrangement := [true, true, false, true, false, true]  -- XXOXOX represented as [X, X, O, X, O, X]
  (↑(X / total) * ↑((X - 1) / (total - 1)) * ↑((O / (total - 2))) * ↑((X - 2) / (total - 3)) * ↑((O - 1) / (total - 4)) * 1 : ℚ) = 1 / 15 :=
sorry

end tile_arrangement_probability_l292_292680


namespace power_function_solution_l292_292164

theorem power_function_solution (f : ℝ → ℝ) (alpha : ℝ)
  (h₀ : ∀ x, f x = x ^ alpha)
  (h₁ : f (1 / 8) = 2) :
  f (-1 / 8) = -2 :=
sorry

end power_function_solution_l292_292164


namespace find_pairs_l292_292916

def is_prime (p : ℕ) : Prop := (p ≥ 2) ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_pairs (a p : ℕ) (h_pos_a : a > 0) (h_prime_p : is_prime p) :
  (∀ m n : ℕ, 0 < m → 0 < n → (a ^ (2 ^ n) % p ^ n = a ^ (2 ^ m) % p ^ m ∧ a ^ (2 ^ n) % p ^ n ≠ 0))
  ↔ (∃ k : ℕ, a = 2 * k + 1 ∧ p = 2) :=
sorry

end find_pairs_l292_292916


namespace average_wx_l292_292858

theorem average_wx (w x a b : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h1 : 6 / w + 6 / x = 6 / (a + b * i))
  (h2 : w * x = a + b * i) :
  (w + x) / 2 = 1 / 2 :=
by
  sorry

end average_wx_l292_292858


namespace find_x_l292_292160

theorem find_x (x : ℝ) (hx : x > 0) (h : Real.sqrt (12*x) * Real.sqrt (5*x) * Real.sqrt (7*x) * Real.sqrt (21*x) = 21) : 
  x = 21 / 97 :=
by
  sorry

end find_x_l292_292160


namespace minimum_cable_length_l292_292381

def station_positions : List ℝ := [0, 3, 7, 11, 14]

def total_cable_length (x : ℝ) : ℝ :=
  abs x + abs (x - 3) + abs (x - 7) + abs (x - 11) + abs (x - 14)

theorem minimum_cable_length :
  (∀ x : ℝ, total_cable_length x ≥ 22) ∧ total_cable_length 7 = 22 :=
by
  sorry

end minimum_cable_length_l292_292381


namespace amount_distributed_l292_292854

theorem amount_distributed (A : ℝ) (h : A / 20 = A / 25 + 120) : A = 12000 :=
by
  sorry

end amount_distributed_l292_292854


namespace solution_set_l292_292682

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

theorem solution_set :
  ∀ x y z : ℝ, system_of_equations x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = -1 ∧ z = -3) ∨ 
  (x = -2 ∧ y = 1 ∧ z = -3) ∨ (x = -2 ∧ y = -1 ∧ z = 3) :=
by
  sorry

end solution_set_l292_292682


namespace find_y_l292_292783

variable (α : ℝ) (y : ℝ)
axiom sin_alpha_neg_half : Real.sin α = -1 / 2
axiom point_on_terminal_side : 2^2 + y^2 = (Real.sin α)^2 + (Real.cos α)^2

theorem find_y : y = -2 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end find_y_l292_292783


namespace pizzas_ordered_l292_292778

variable (m : ℕ) (x : ℕ)

theorem pizzas_ordered (h1 : m * 2 * x = 14) (h2 : x = 1 / 2 * m) (h3 : m > 13) : 
  14 + 13 * x = 15 := 
sorry

end pizzas_ordered_l292_292778


namespace task_completion_time_l292_292285

noncomputable def work_time (A B C : ℝ) : ℝ := 1 / (A + B + C)

theorem task_completion_time (x y z : ℝ) (h1 : 8 * (x + y) = 1) (h2 : 6 * (x + z) = 1) (h3 : 4.8 * (y + z) = 1) :
    work_time x y z = 4 :=
by
  sorry

end task_completion_time_l292_292285


namespace distilled_water_required_l292_292636

theorem distilled_water_required :
  ∀ (nutrient_concentrate distilled_water : ℝ) (total_solution prep_solution : ℝ), 
    nutrient_concentrate = 0.05 →
    distilled_water = 0.025 →
    total_solution = 0.075 → 
    prep_solution = 0.6 →
    (prep_solution * (distilled_water / total_solution)) = 0.2 :=
by
  intros nutrient_concentrate distilled_water total_solution prep_solution
  sorry

end distilled_water_required_l292_292636


namespace sufficient_not_necessary_condition_l292_292039

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

theorem sufficient_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) :=
by
  sorry

end sufficient_not_necessary_condition_l292_292039


namespace matrix_calculation_l292_292188

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l292_292188


namespace find_number_with_21_multiples_of_4_l292_292537

theorem find_number_with_21_multiples_of_4 (n : ℕ) (h₁ : ∀ k : ℕ, n + k * 4 ≤ 92 → k < 21) : n = 80 :=
sorry

end find_number_with_21_multiples_of_4_l292_292537


namespace prime_divisor_of_ones_l292_292440

theorem prime_divisor_of_ones (p : ℕ) (hp : Nat.Prime p ∧ p ≠ 2 ∧ p ≠ 5) :
  ∃ k : ℕ, p ∣ (10^k - 1) / 9 :=
by
  sorry

end prime_divisor_of_ones_l292_292440


namespace opposite_of_neg_frac_l292_292231

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l292_292231


namespace valid_number_count_l292_292426

def is_valid_digit (d: Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def are_adjacent (d1 d2: Nat) : Bool :=
  (d1 = 1 ∧ d2 = 2) ∨ (d1 = 2 ∧ d2 = 1) ∨
  (d1 = 5 ∧ (d2 = 1 ∨ d2 = 2)) ∨ 
  (d2 = 5 ∧ (d1 = 1 ∨ d1 = 2))

def count_valid_numbers : Nat :=
  sorry -- expression to count numbers according to given conditions.

theorem valid_number_count : count_valid_numbers = 36 :=
  sorry

end valid_number_count_l292_292426


namespace num_increasing_digits_l292_292593

theorem num_increasing_digits :
  let C := λ (n k : ℕ), Nat.choose n k in
  ∑ k in Finset.range 8, C 9 (k + 2) = 502 :=
by
  sorry

end num_increasing_digits_l292_292593


namespace ninety_times_ninety_l292_292299

theorem ninety_times_ninety : (90 * 90) = 8100 := by
  let a := 100
  let b := 10
  have h1 : (90 * 90) = (a - b) * (a - b) := by decide
  have h2 : (a - b) * (a - b) = a^2 - 2 * a * b + b^2 := by decide
  have h3 : a = 100 := rfl
  have h4 : b = 10 := rfl
  have h5 : 100^2 - 2 * 100 * 10 + 10^2 = 8100 := by decide
  sorry

end ninety_times_ninety_l292_292299


namespace interval_of_decrease_for_f_l292_292585

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3)

def decreasing_interval (s : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

theorem interval_of_decrease_for_f :
  decreasing_interval {x : ℝ | x < -1} f :=
by
  sorry

end interval_of_decrease_for_f_l292_292585


namespace rental_lower_amount_eq_50_l292_292826

theorem rental_lower_amount_eq_50 (L : ℝ) (total_rent : ℝ) (reduction : ℝ) (rooms_changed : ℕ) (diff_per_room : ℝ)
  (h1 : total_rent = 400)
  (h2 : reduction = 0.25 * total_rent)
  (h3 : rooms_changed = 10)
  (h4 : diff_per_room = reduction / ↑rooms_changed)
  (h5 : 60 - L = diff_per_room) :
  L = 50 :=
  sorry

end rental_lower_amount_eq_50_l292_292826


namespace compute_value_of_expression_l292_292963

theorem compute_value_of_expression (p q : ℝ) (h1 : 3 * p^2 - 7 * p + 1 = 0) (h2 : 3 * q^2 - 7 * q + 1 = 0) :
  (9 * p^3 - 9 * q^3) / (p - q) = 46 :=
sorry

end compute_value_of_expression_l292_292963


namespace prob_exactly_M_laws_included_expected_laws_included_l292_292338

variables (K N M : ℕ) (p : ℝ)

-- Definition of the probabilities as given in the conditions and answers
def prob_no_minister_knows_law : ℝ := (1 - p) ^ N
def prob_law_included : ℝ := 1 - prob_no_minister_knows_law p N

-- Part (a)
theorem prob_exactly_M_laws_included :
  (nat.choose K M) * (prob_law_included p N) ^ M * (prob_no_minister_knows_law p N) ^ (K - M) = 
  (nat.choose K M) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by
  sorry

-- Part (b)
theorem expected_laws_included :
  K * (prob_law_included p N) = K * (1 - (1 - p) ^ N) :=
by
  sorry

end prob_exactly_M_laws_included_expected_laws_included_l292_292338


namespace determine_omega_phi_l292_292936

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem determine_omega_phi :
  ∃ (ω φ : ℝ), ω > 0 ∧ |φ| < Real.pi ∧ 
  f (Real.pi * 5 / 8) ω φ = 2 ∧ 
  f (Real.pi * 11 / 8) ω φ = 0 ∧ 
  (∃ T > 2 * Real.pi, ∀ (x : ℝ), f (x + T) ω φ = f x ω φ) ∧ 
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end determine_omega_phi_l292_292936


namespace value_of_g_at_3_l292_292473

def g (x : ℝ) := x^2 + 1

theorem value_of_g_at_3 : g 3 = 10 := by
  sorry

end value_of_g_at_3_l292_292473


namespace coffee_shop_brewed_cups_in_week_l292_292087

theorem coffee_shop_brewed_cups_in_week 
    (weekday_rate : ℕ) (weekend_rate : ℕ)
    (weekday_hours : ℕ) (saturday_hours : ℕ) (sunday_hours : ℕ)
    (num_weekdays : ℕ) (num_saturdays : ℕ) (num_sundays : ℕ)
    (h1 : weekday_rate = 10)
    (h2 : weekend_rate = 15)
    (h3 : weekday_hours = 5)
    (h4 : saturday_hours = 6)
    (h5 : sunday_hours = 4)
    (h6 : num_weekdays = 5)
    (h7 : num_saturdays = 1)
    (h8 : num_sundays = 1) :
    (weekday_rate * weekday_hours * num_weekdays) + 
    (weekend_rate * saturday_hours * num_saturdays) + 
    (weekend_rate * sunday_hours * num_sundays) = 400 := 
by
  sorry

end coffee_shop_brewed_cups_in_week_l292_292087


namespace probability_red_ball_10th_draw_l292_292171

-- Definitions for conditions in the problem
def total_balls : ℕ := 10
def red_balls : ℕ := 2

-- Probability calculation function
def probability_of_red_ball (total : ℕ) (red : ℕ) : ℚ :=
  red / total

-- Theorem statement: Given the conditions, the probability of drawing a red ball on the 10th attempt is 1/5
theorem probability_red_ball_10th_draw :
  probability_of_red_ball total_balls red_balls = 1 / 5 :=
by
  sorry

end probability_red_ball_10th_draw_l292_292171


namespace completing_the_square_x_squared_plus_4x_plus_3_eq_0_l292_292548

theorem completing_the_square_x_squared_plus_4x_plus_3_eq_0 :
  (x : ℝ) → x^2 + 4 * x + 3 = 0 → (x + 2)^2 = 1 :=
by
  intros x h
  -- The actual proof will be provided here
  sorry

end completing_the_square_x_squared_plus_4x_plus_3_eq_0_l292_292548


namespace pizza_cost_is_correct_l292_292972

noncomputable def total_pizza_cost : ℝ :=
  let triple_cheese_pizza_cost := (3 * 10) + (6 * 2 * 2.5)
  let meat_lovers_pizza_cost := (3 * 8) + (4 * 3 * 2.5)
  let veggie_delight_pizza_cost := (6 * 5) + (10 * 1 * 2.5)
  triple_cheese_pizza_cost + meat_lovers_pizza_cost + veggie_delight_pizza_cost

theorem pizza_cost_is_correct : total_pizza_cost = 169 := by
  sorry

end pizza_cost_is_correct_l292_292972


namespace polygon_sides_l292_292413

theorem polygon_sides {R : ℝ} (hR : R > 0) : 
  (∃ n : ℕ, n > 2 ∧ (1/2) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) → 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end polygon_sides_l292_292413


namespace solution_set_inequality_l292_292531

theorem solution_set_inequality (x : ℝ) : (| 2 * x - 1 | - | x - 2 | < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end solution_set_inequality_l292_292531


namespace infinite_set_k_l292_292032

theorem infinite_set_k (C : ℝ) : ∃ᶠ k : ℤ in at_top, (k : ℝ) * Real.sin k > C :=
sorry

end infinite_set_k_l292_292032


namespace maxProfitAchievable_l292_292275

namespace BarrelProduction

structure ProductionPlan where
  barrelsA : ℕ
  barrelsB : ℕ

def profit (plan : ProductionPlan) : ℕ :=
  300 * plan.barrelsA + 400 * plan.barrelsB

def materialAUsage (plan : ProductionPlan) : ℕ :=
  plan.barrelsA + 2 * plan.barrelsB

def materialBUsage (plan : ProductionPlan) : ℕ :=
  2 * plan.barrelsA + plan.barrelsB

def isValidPlan (plan : ProductionPlan) : Prop :=
  materialAUsage plan ≤ 12 ∧ materialBUsage plan ≤ 12

def maximumProfit : ℕ :=
  2800

theorem maxProfitAchievable : 
  ∃ (plan : ProductionPlan), isValidPlan plan ∧ profit plan = maximumProfit :=
sorry

end BarrelProduction

end maxProfitAchievable_l292_292275


namespace Ann_trip_takes_longer_l292_292668

theorem Ann_trip_takes_longer (mary_distance : ℕ) (mary_speed : ℕ)
                              (ann_distance : ℕ) (ann_speed : ℕ)
                              (mary_time : ℕ) (ann_time : ℕ) :
  mary_distance = 630 →
  mary_speed = 90 →
  ann_distance = 800 →
  ann_speed = 40 →
  mary_time = mary_distance / mary_speed →
  ann_time = ann_distance / ann_speed →
  (ann_time - mary_time) = 13 :=
by
  intros
  calculate!
  sorry

end Ann_trip_takes_longer_l292_292668


namespace members_not_in_A_nor_B_l292_292992

variable (U A B : Finset ℕ) -- We define the sets as finite sets of natural numbers.
variable (hU_size : U.card = 190) -- Size of set U is 190.
variable (hB_size : (U ∩ B).card = 49) -- 49 items are in set B.
variable (hAB_size : (A ∩ U ∩ B).card = 23) -- 23 items are in both A and B.
variable (hA_size : (U ∩ A).card = 105) -- 105 items are in set A.

theorem members_not_in_A_nor_B :
  (U \ (A ∪ B)).card = 59 := sorry

end members_not_in_A_nor_B_l292_292992


namespace find_point_A_l292_292205

theorem find_point_A :
  (∃ A : ℤ, A + 2 = -2) ∨ (∃ A : ℤ, A - 2 = -2) → (∃ A : ℤ, A = 0 ∨ A = -4) :=
by
  sorry

end find_point_A_l292_292205


namespace sufficient_but_not_necessary_l292_292779

theorem sufficient_but_not_necessary (a b : ℝ) : (a > |b|) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > |b|)) := 
sorry

end sufficient_but_not_necessary_l292_292779


namespace matrix_calculation_l292_292187

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l292_292187


namespace largest_c_value_l292_292774

theorem largest_c_value (c : ℝ) (h : -2 * c^2 + 8 * c - 6 ≥ 0) : c ≤ 3 := 
sorry

end largest_c_value_l292_292774


namespace totalBottleCaps_l292_292380

-- Variables for the conditions
def bottleCapsPerBox : ℝ := 35.0
def numberOfBoxes : ℝ := 7.0

-- Theorem stating the equivalent proof problem
theorem totalBottleCaps : bottleCapsPerBox * numberOfBoxes = 245.0 := by
  sorry

end totalBottleCaps_l292_292380


namespace peter_read_more_books_l292_292513

/-
Given conditions:
  Peter has 20 books.
  Peter has read 40% of them.
  Peter's brother has read 10% of them.
We aim to prove that Peter has read 6 more books than his brother.
-/

def total_books : ℕ := 20
def peter_read_fraction : ℚ := 0.4
def brother_read_fraction : ℚ := 0.1

def books_read_by_peter := total_books * peter_read_fraction
def books_read_by_brother := total_books * brother_read_fraction

theorem peter_read_more_books :
  books_read_by_peter - books_read_by_brother = 6 := by
  sorry

end peter_read_more_books_l292_292513


namespace option_a_option_b_l292_292856

theorem option_a (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  -- Proof goes here
  sorry

theorem option_b (a b : ℝ) (ha : a > 0) (hb : b > 0) : a * b ≤ (a + b)^2 / 4 :=
by
  -- Proof goes here
  sorry

end option_a_option_b_l292_292856


namespace find_d_l292_292038

noncomputable def polynomial_roots_neg_int (g : Polynomial ℝ) : Prop :=
  ∃ s1 s2 s3 s4 : ℝ, s1 > 0 ∧ s2 > 0 ∧ s3 > 0 ∧ s4 > 0 ∧
    g = Polynomial.C (s1 * s2 * s3 * s4) * (Polynomial.X + (-s1)) *
          (Polynomial.X + (-s2)) * (Polynomial.X + (-s3)) * (Polynomial.X + (-s4))

theorem find_d 
  (g : Polynomial ℝ) 
  (a b c d : ℝ)
  (h_g : g = Polynomial.mk [d, c, b, a, 1]) 
  (h_roots : polynomial_roots_neg_int g)
  (h_sum : a + b + c + d = 2003) :
  d = 1992 :=
  sorry

end find_d_l292_292038


namespace jane_babysitting_start_l292_292649

-- Definitions based on the problem conditions
def jane_current_age := 32
def years_since_babysitting := 10
def oldest_current_child_age := 24

-- Definition for the starting babysitting age
def starting_babysitting_age : ℕ := 8

-- Theorem statement to prove
theorem jane_babysitting_start (h1 : jane_current_age - years_since_babysitting = 22)
  (h2 : oldest_current_child_age - years_since_babysitting = 14)
  (h3 : ∀ (age_jane age_child : ℕ), age_child ≤ age_jane / 2) :
  starting_babysitting_age = 8 :=
by
  sorry

end jane_babysitting_start_l292_292649


namespace ellipse_equation_minimum_distance_l292_292143

-- Define the conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2) / (a^2) + (y^2) / (b^2) = 1)

def eccentricity (a c : ℝ) : Prop :=
  c = a / 2

def focal_distance (c : ℝ) : Prop :=
  2 * c = 4

def foci_parallel (F1 A B C D : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := F1;
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (yA - y1) / (xA - x1) = (yC - y1) / (xC - x1) ∧ 
  (yB - y1) / (xB - x1) = (yD - y1) / (xD - x1)

def orthogonal_vectors (A C B D : ℝ × ℝ) : Prop :=
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (xC - xA) * (xD - xB) + (yC - yA) * (yD - yB) = 0

-- Prove equation of ellipse E
theorem ellipse_equation (a b : ℝ) (x y : ℝ) (c : ℝ)
  (h1 : ellipse a b x y)
  (h2 : eccentricity a c)
  (h3 : focal_distance c) :
  (a = 4) ∧ (b^2 = 12) ∧ (x^2 / 16 + y^2 / 12 = 1) :=
sorry

-- Prove minimum value of |AC| + |BD|
theorem minimum_distance (A B C D : ℝ × ℝ)
  (F1 : ℝ × ℝ)
  (h1 : foci_parallel F1 A B C D)
  (h2 : orthogonal_vectors A C B D) :
  |(AC : ℝ)| + |(BD : ℝ)| = 96 / 7 :=
sorry

end ellipse_equation_minimum_distance_l292_292143


namespace BD_is_diameter_of_circle_l292_292657

variables {A B C D X Y : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D] [MetricSpace X] [MetricSpace Y]

-- Assume these four points lie on a circle with certain ordering
variables (circ : Circle A B C D)

-- Given conditions
variables (h1 : circ.AB < circ.AD)
variables (h2 : circ.BC > circ.CD)

-- Points X and Y are where angle bisectors meet the circle again
variables (h3 : circ.bisects_angle_BAD_at X)
variables (h4 : circ.bisects_angle_BCD_at Y)

-- Hexagon sides with four equal lengths
variables (hex_equal : circ.hexagon_sides_equal_length A B X C D Y)

-- Prove that BD is a diameter
theorem BD_is_diameter_of_circle : circ.is_diameter BD := 
by
  sorry

end BD_is_diameter_of_circle_l292_292657


namespace average_score_of_juniors_l292_292486

theorem average_score_of_juniors :
  ∀ (N : ℕ) (junior_percent senior_percent overall_avg senior_avg : ℚ),
  junior_percent = 0.20 →
  senior_percent = 0.80 →
  overall_avg = 86 →
  senior_avg = 85 →
  (N * overall_avg - (N * senior_percent * senior_avg)) / (N * junior_percent) = 90 := 
by
  intros N junior_percent senior_percent overall_avg senior_avg
  intros h1 h2 h3 h4
  sorry

end average_score_of_juniors_l292_292486


namespace A_eq_B_l292_292966

namespace SetsEquality

open Set

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4 * a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4 * b^2 + 4 * b + 2}

theorem A_eq_B : A = B := by
  sorry

end SetsEquality

end A_eq_B_l292_292966


namespace solution_set_of_inequality_l292_292988

theorem solution_set_of_inequality (x : ℝ) : (x - 1) * (2 - x) > 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l292_292988


namespace find_middle_number_l292_292420

theorem find_middle_number (a : Fin 11 → ℝ)
  (h1 : ∀ i : Fin 9, a i + a (⟨i.1 + 1, by linarith [i.2]⟩) + a (⟨i.1 + 2, by linarith [i.2]⟩) = 18)
  (h2 : (Finset.univ.sum a) = 64) :
  a 5 = 8 := 
by
  sorry

end find_middle_number_l292_292420


namespace gift_exchange_equation_l292_292883

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l292_292883


namespace gecko_cricket_eating_l292_292866

theorem gecko_cricket_eating :
  ∀ (total_crickets : ℕ) (first_day_percent : ℚ) (second_day_less : ℕ),
    total_crickets = 70 →
    first_day_percent = 0.3 →
    second_day_less = 6 →
    let first_day_crickets := total_crickets * first_day_percent
    let second_day_crickets := first_day_crickets - second_day_less
    total_crickets - first_day_crickets - second_day_crickets = 34 :=
by
  intros total_crickets first_day_percent second_day_less h_total h_percent h_less
  let first_day_crickets := total_crickets * first_day_percent
  let second_day_crickets := first_day_crickets - second_day_less
  have : total_crickets - first_day_crickets - second_day_crickets = 34 := sorry
  exact this

end gecko_cricket_eating_l292_292866


namespace average_speed_is_35_l292_292581

-- Given constants
def distance : ℕ := 210
def speed_difference : ℕ := 5
def time_difference : ℕ := 1

-- Definition of time for planned speed and actual speed
def planned_time (x : ℕ) : ℚ := distance / (x - speed_difference)
def actual_time (x : ℕ) : ℚ := distance / x

-- Main theorem to be proved
theorem average_speed_is_35 (x : ℕ) (h : (planned_time x - actual_time x) = time_difference) : x = 35 :=
sorry

end average_speed_is_35_l292_292581


namespace focus_of_parabola_l292_292919

-- Define the given parabola equation
def given_parabola (x : ℝ) : ℝ := 4 * x^2

-- Define what it means to be the focus of this parabola
def is_focus (focus : ℝ × ℝ) : Prop :=
  focus = (0, 1 / 16)

-- The theorem to prove
theorem focus_of_parabola : ∃ focus : ℝ × ℝ, is_focus focus :=
  by 
    use (0, 1 / 16)
    exact sorry

end focus_of_parabola_l292_292919


namespace red_marbles_count_l292_292074

noncomputable def total_marbles (R : ℕ) : ℕ := R + 16

noncomputable def P_blue (R : ℕ) : ℚ := 10 / (total_marbles R)

noncomputable def P_neither_blue (R : ℕ) : ℚ := (1 - P_blue R) * (1 - P_blue R)

noncomputable def P_either_blue (R : ℕ) : ℚ := 1 - P_neither_blue R

theorem red_marbles_count
  (R : ℕ) 
  (h1 : P_either_blue R = 0.75) :
  R = 4 :=
by
  sorry

end red_marbles_count_l292_292074


namespace percent_increase_in_maintenance_time_l292_292720

theorem percent_increase_in_maintenance_time (original_time new_time : ℝ) (h1 : original_time = 25) (h2 : new_time = 30) : 
  ((new_time - original_time) / original_time) * 100 = 20 :=
by
  sorry

end percent_increase_in_maintenance_time_l292_292720


namespace quadratic_root_l292_292476

theorem quadratic_root (k : ℝ) (h : ∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) : k = -1 :=
sorry

end quadratic_root_l292_292476


namespace frac_e_a_l292_292802

variable (a b c d e : ℚ)

theorem frac_e_a (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) (h4 : d / e = 1 / 2) :
  e / a = 8 / 35 :=
sorry

end frac_e_a_l292_292802


namespace velocity_at_2_l292_292328

variable (t : ℝ) (s : ℝ)

noncomputable def displacement (t : ℝ) : ℝ := t^2 + 3 / t

noncomputable def velocity (t : ℝ) : ℝ := (deriv displacement) t

theorem velocity_at_2 : velocity t = 2 * 2 - (3 / 4) := by
  sorry

end velocity_at_2_l292_292328


namespace percentage_error_in_calculated_area_l292_292699

theorem percentage_error_in_calculated_area :
  let initial_length_error := 0.03 -- 3%
  let initial_width_error := -0.02 -- 2% deficit
  let temperature_change := 15 -- °C
  let humidity_increase := 20 -- %
  let length_error_temp_increase := (temperature_change / 5) * 0.01
  let width_error_humidity_increase := (humidity_increase / 10) * 0.005
  let total_length_error := initial_length_error + length_error_temp_increase
  let total_width_error := initial_width_error + width_error_humidity_increase
  let total_percentage_error := total_length_error + total_width_error
  total_percentage_error * 100 = 3 -- 3%
:= by
  sorry

end percentage_error_in_calculated_area_l292_292699


namespace solve_coin_problem_l292_292170

def coin_problem : Prop :=
  ∃ (x y z : ℕ), 
  1 * x + 2 * y + 5 * z = 71 ∧ 
  x = y ∧ 
  x + y + z = 31 ∧ 
  x = 12 ∧ 
  y = 12 ∧ 
  z = 7

theorem solve_coin_problem : coin_problem :=
  sorry

end solve_coin_problem_l292_292170


namespace oak_trees_cut_down_l292_292705

-- Define the conditions
def initial_oak_trees : ℕ := 9
def final_oak_trees : ℕ := 7

-- Prove that the number of oak trees cut down is 2
theorem oak_trees_cut_down : (initial_oak_trees - final_oak_trees) = 2 :=
by
  -- Proof is omitted
  sorry

end oak_trees_cut_down_l292_292705


namespace sue_driving_days_l292_292214

-- Define the conditions as constants or variables
def total_cost : ℕ := 2100
def sue_payment : ℕ := 900
def sister_days : ℕ := 4
def total_days_in_week : ℕ := 7

-- Prove that the number of days Sue drives the car (x) equals 3
theorem sue_driving_days : ∃ x : ℕ, x = 3 ∧ sue_payment * sister_days = x * (total_cost - sue_payment) := 
by
  sorry

end sue_driving_days_l292_292214


namespace boat_speed_l292_292808

theorem boat_speed (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by
  sorry

end boat_speed_l292_292808


namespace negation_of_proposition_true_l292_292433

theorem negation_of_proposition_true (a b : ℝ) : 
  (¬ ((a > b) → (∀ c : ℝ, c ^ 2 ≠ 0 → a * c ^ 2 > b * c ^ 2)) = true) :=
by
  sorry

end negation_of_proposition_true_l292_292433


namespace total_children_count_l292_292251

theorem total_children_count (boys girls : ℕ) (hb : boys = 40) (hg : girls = 77) : boys + girls = 117 := by
  sorry

end total_children_count_l292_292251


namespace total_animals_to_spay_l292_292425

theorem total_animals_to_spay : 
  ∀ (c d : ℕ), c = 7 → d = 2 * c → c + d = 21 :=
by
  intros c d h1 h2
  sorry

end total_animals_to_spay_l292_292425


namespace Jose_got_5_questions_wrong_l292_292026

def Jose_questions_wrong (M J A : ℕ) : Prop :=
  M = J - 20 ∧
  J = A + 40 ∧
  M + J + A = 210 ∧
  (50 * 2 = 100) ∧
  (100 - J) / 2 = 5

theorem Jose_got_5_questions_wrong (M J A : ℕ) (h1 : M = J - 20) (h2 : J = A + 40) (h3 : M + J + A = 210) : 
  Jose_questions_wrong M J A :=
by
  sorry

end Jose_got_5_questions_wrong_l292_292026


namespace current_speed_l292_292880

theorem current_speed (r w : ℝ) 
  (h1 : 21 / (r + w) + 3 = 21 / (r - w))
  (h2 : 21 / (1.5 * r + w) + 0.75 = 21 / (1.5 * r - w)) 
  : w = 9.8 :=
by
  sorry

end current_speed_l292_292880


namespace intersection_points_count_l292_292336

def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

theorem intersection_points_count :
  ∃ p1 p2 p3 : ℝ × ℝ,
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (line1 p3.1 p3.2 ∧ line3 p3.1 p3.2) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
  sorry

end intersection_points_count_l292_292336


namespace vertex_of_parabola_l292_292061

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

-- Define the vertex point
def vertex : ℝ × ℝ := (-2, -1)

-- The theorem we need to prove
theorem vertex_of_parabola : ∀ x : ℝ, parabola x = (x + 2)^2 - 1 → vertex = (-2, -1) := 
by
  sorry

end vertex_of_parabola_l292_292061


namespace equal_numbers_possible_l292_292508

noncomputable def circle_operations (n : ℕ) (α : ℝ) : Prop :=
  (n ≥ 3) ∧ (∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n))

-- Statement of the theorem
theorem equal_numbers_possible (n : ℕ) (α : ℝ) (h1 : n ≥ 3) (h2 : α > 0) :
  circle_operations n α ↔ ∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n) :=
sorry

end equal_numbers_possible_l292_292508


namespace custom_op_4_3_equals_37_l292_292762

def custom_op (a b : ℕ) : ℕ := a^2 + a*b + b^2

theorem custom_op_4_3_equals_37 : custom_op 4 3 = 37 := by
  sorry

end custom_op_4_3_equals_37_l292_292762


namespace remainder_of_5032_div_28_l292_292255

theorem remainder_of_5032_div_28 : 5032 % 28 = 20 :=
by
  sorry

end remainder_of_5032_div_28_l292_292255


namespace option_C_correct_l292_292475

theorem option_C_correct (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := 
by
  sorry

end option_C_correct_l292_292475


namespace solve_for_x_l292_292681

theorem solve_for_x : ∀ x : ℝ, (x - 27) / 3 = (3 * x + 6) / 8 → x = -234 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l292_292681


namespace Diane_age_when_conditions_met_l292_292261

variable (Diane_current : ℕ) (Alex_current : ℕ) (Allison_current : ℕ)
variable (D : ℕ)

axiom Diane_current_age : Diane_current = 16
axiom Alex_Allison_sum : Alex_current + Allison_current = 47
axiom Diane_half_Alex : D = (Alex_current + (D - 16)) / 2
axiom Diane_twice_Allison : D = 2 * (Allison_current + (D - 16))

theorem Diane_age_when_conditions_met : D = 78 :=
by
  sorry

end Diane_age_when_conditions_met_l292_292261


namespace purple_balls_correct_l292_292406

-- Define the total number of balls and individual counts
def total_balls : ℕ := 100
def white_balls : ℕ := 20
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def red_balls : ℕ := 37

-- Probability that a ball chosen is neither red nor purple
def prob_neither_red_nor_purple : ℚ := 0.6

-- The number of purple balls to be proven
def purple_balls : ℕ := 3

-- The condition used for the proof
def condition : Prop := prob_neither_red_nor_purple = (white_balls + green_balls + yellow_balls) / total_balls

-- The proof problem statement
theorem purple_balls_correct (h : condition) : 
  ∃ P : ℕ, P = purple_balls ∧ P + red_balls = total_balls - (white_balls + green_balls + yellow_balls) :=
by
  have P := total_balls - (white_balls + green_balls + yellow_balls + red_balls)
  existsi P
  sorry

end purple_balls_correct_l292_292406


namespace prob_even_sum_l292_292832

structure Spinner :=
  (outcomes : list ℕ)

def S : Spinner := ⟨[1, 2, 4]⟩
def T : Spinner := ⟨[3, 3, 6]⟩
def U : Spinner := ⟨[2, 4, 6]⟩

def even (n : ℕ) : Prop := n % 2 = 0

def event_prob (spinner : Spinner) (p : ℕ → Prop) : ℝ :=
  (spinner.outcomes.filter p).length.toReal / spinner.outcomes.length.toReal

def sum_even_event (s : Spinner) (t : Spinner) (u : Spinner) : ℝ :=
  (event_prob S (λ n, ¬ even n)) * (event_prob T (λ n, ¬ even n)) * (event_prob U even) +
  (event_prob S even) * (event_prob T even) * (event_prob U even)

theorem prob_even_sum : sum_even_event S T U = 5 / 9 := sorry

end prob_even_sum_l292_292832


namespace attendees_gift_exchange_l292_292888

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l292_292888


namespace Allison_greater_probability_l292_292572

open ProbabilityTheory

/-- The probability problem -/
theorem Allison_greater_probability :
  let AllisonRoll := 6
  let CharlieRoll := {1, 1, 2, 2, 3, 3}
  let EmmaRoll := {3, 3, 3, 3, 5, 5}
  (1 : ℚ) * (4 / 6 : ℚ) = (2 / 3 : ℚ) :=
by
  let AllisonRoll := 6
  let CharlieRoll := {1, 1, 2, 2, 3, 3}
  let EmmaRoll := {3, 3, 3, 3, 5, 5}
  have hCharlie : (1 : ℚ) = 1 := by sorry
  have hEmma : (4 / 6 : ℚ) = (2 / 3 : ℚ) := by sorry
  show (1 : ℚ) * (4 / 6 : ℚ) = (2 / 3 : ℚ) from 
    by rw [hCharlie, hEmma]


end Allison_greater_probability_l292_292572


namespace range_of_x_l292_292788

-- Define the even and increasing properties of the function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- The main theorem to be proven
theorem range_of_x (f : ℝ → ℝ) (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) 
  (h_cond : ∀ x : ℝ, f (x - 1) < f (2 - x)) :
  ∀ x : ℝ, x < 3 / 2 :=
by
  sorry

end range_of_x_l292_292788


namespace gcd_problem_l292_292432

open Int -- Open the integer namespace to use gcd.

theorem gcd_problem : Int.gcd (Int.gcd 188094 244122) 395646 = 6 :=
by
  -- provide the proof here
  sorry

end gcd_problem_l292_292432


namespace electricity_price_per_kWh_l292_292063

theorem electricity_price_per_kWh (consumption_rate : ℝ) (hours_used : ℝ) (total_cost : ℝ) :
  consumption_rate = 2.4 → hours_used = 25 → total_cost = 6 →
  total_cost / (consumption_rate * hours_used) = 0.10 :=
by
  intros hc hh ht
  have h_energy : consumption_rate * hours_used = 60 :=
    by rw [hc, hh]; norm_num
  rw [ht, h_energy]
  norm_num

end electricity_price_per_kWh_l292_292063


namespace original_distance_between_Stacy_and_Heather_l292_292058

theorem original_distance_between_Stacy_and_Heather
  (H_speed : ℝ := 5)  -- Heather's speed in miles per hour
  (S_speed : ℝ := 6)  -- Stacy's speed in miles per hour
  (delay : ℝ := 0.4)  -- Heather's start delay in hours
  (H_distance : ℝ := 1.1818181818181817)  -- Distance Heather walked when they meet
  : H_speed * (H_distance / H_speed) + S_speed * ((H_distance / H_speed) + delay) = 5 := by
  sorry

end original_distance_between_Stacy_and_Heather_l292_292058


namespace markers_per_box_l292_292510

theorem markers_per_box (original_markers new_boxes total_markers : ℕ) 
    (h1 : original_markers = 32) (h2 : new_boxes = 6) (h3 : total_markers = 86) : 
    total_markers - original_markers = new_boxes * 9 :=
by sorry

end markers_per_box_l292_292510


namespace product_of_numbers_eq_zero_l292_292132

theorem product_of_numbers_eq_zero (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := 
by
  sorry

end product_of_numbers_eq_zero_l292_292132


namespace park_area_approx_l292_292957

noncomputable def area_of_park (r L B : ℝ) (x : ℝ) : ℝ :=
  L * B

theorem park_area_approx :
  ∀ (L B : ℝ)
    (h1 : ∃ (x : ℝ), L = 3 * x ∧ B = 5 * x)
    (h2 : 15 * 1000 / 60 * 12 = 3.5 * 2 * (L + B)),
  area_of_park 43057.60 L B ≈ 43057.60 :=
by
  sorry

end park_area_approx_l292_292957


namespace katy_books_l292_292179

theorem katy_books (june july aug : ℕ) (h1 : june = 8) (h2 : july = 2 * june) (h3 : june + july + aug = 37) :
  july - aug = 3 :=
by sorry

end katy_books_l292_292179


namespace ratio_a7_b7_l292_292318

variable (a b : ℕ → ℝ)
variable (S T : ℕ → ℝ)

-- Given conditions
axiom sum_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * a 2) -- Formula for sum of arithmetic series
axiom sum_T : ∀ n, T n = (n / 2) * (2 * b 1 + (n - 1) * b 2) -- Formula for sum of arithmetic series
axiom ratio_ST : ∀ n, S n / T n = (2 * n + 1) / (n + 3)

-- Prove the ratio of seventh terms
theorem ratio_a7_b7 : a 7 / b 7 = 27 / 16 :=
by
  sorry

end ratio_a7_b7_l292_292318


namespace opposite_of_neg_one_div_2023_l292_292235

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l292_292235


namespace ann_trip_longer_than_mary_l292_292667

-- Define constants for conditions
def mary_hill_length : ℕ := 630
def mary_speed : ℕ := 90
def ann_hill_length : ℕ := 800
def ann_speed : ℕ := 40

-- Define a theorem to express the question and correct answer
theorem ann_trip_longer_than_mary : 
  (ann_hill_length / ann_speed - mary_hill_length / mary_speed) = 13 :=
by
  -- Now insert sorry to leave the proof unfinished
  sorry

end ann_trip_longer_than_mary_l292_292667


namespace sum_of_base4_numbers_is_correct_l292_292923

-- Define the four base numbers
def n1 : ℕ := 2 * 4^2 + 1 * 4^1 + 2 * 4^0
def n2 : ℕ := 1 * 4^2 + 0 * 4^1 + 3 * 4^0
def n3 : ℕ := 3 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Define the expected sum in base 4 interpreted as a natural number
def expected_sum : ℕ := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

-- State the theorem
theorem sum_of_base4_numbers_is_correct : n1 + n2 + n3 = expected_sum := by
  sorry

end sum_of_base4_numbers_is_correct_l292_292923


namespace gecko_third_day_crickets_l292_292869

def total_crickets : ℕ := 70
def first_day_percentage : ℝ := 0.30
def first_day_crickets : ℝ := first_day_percentage * total_crickets
def second_day_crickets : ℝ := first_day_crickets - 6
def third_day_crickets : ℝ := total_crickets - (first_day_crickets + second_day_crickets)

theorem gecko_third_day_crickets :
  third_day_crickets = 34 :=
by
  sorry

end gecko_third_day_crickets_l292_292869


namespace min_cos_y_plus_sin_x_l292_292225

theorem min_cos_y_plus_sin_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.cos x = Real.sin (3 * x))
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) - Real.cos (2 * x)) :
  ∃ (v : ℝ), v = -1 - Real.sqrt (2 + Real.sqrt 2) / 2 :=
sorry

end min_cos_y_plus_sin_x_l292_292225


namespace cone_lateral_surface_area_l292_292607

theorem cone_lateral_surface_area (r l : ℝ) (h1 : r = 2) (h2 : l = 5) : 
    0.5 * (2 * Real.pi * r * l) = 10 * Real.pi := by
    sorry

end cone_lateral_surface_area_l292_292607


namespace rectangular_prism_diagonals_l292_292282

theorem rectangular_prism_diagonals
  (num_vertices : ℕ) (num_edges : ℕ)
  (h1 : num_vertices = 12) (h2 : num_edges = 18) :
  (total_diagonals : ℕ) → total_diagonals = 20 :=
by
  sorry

end rectangular_prism_diagonals_l292_292282


namespace ellipse_properties_l292_292323

theorem ellipse_properties :
  (∀ x y: ℝ, (x^2)/100 + (y^2)/36 = 1) →
  ∃ a b c e : ℝ, 
  a = 10 ∧ 
  b = 6 ∧ 
  c = 8 ∧ 
  2 * a = 20 ∧ 
  e = 4 / 5 :=
by
  intros
  sorry

end ellipse_properties_l292_292323


namespace solve_equation_l292_292120

theorem solve_equation (x : ℝ) :
  (x^2 + 2*x + 1 = abs (3*x - 2)) ↔ 
  (x = (-7 + Real.sqrt 37) / 2) ∨ 
  (x = (-7 - Real.sqrt 37) / 2) :=
by
  sorry

end solve_equation_l292_292120


namespace correct_equation_for_gift_exchanges_l292_292891

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l292_292891


namespace trapezoid_area_l292_292702

-- Definitions of the problem's conditions
def a : ℕ := 4
def b : ℕ := 8
def h : ℕ := 3

-- Lean statement to prove the area of the trapezoid is 18 square centimeters
theorem trapezoid_area : (a + b) * h / 2 = 18 := by
  sorry

end trapezoid_area_l292_292702


namespace middle_tree_distance_l292_292740

theorem middle_tree_distance (d : ℕ) (b : ℕ) (c : ℕ) 
  (h_b : b = 84) (h_c : c = 91) 
  (h_right_triangle : d^2 + b^2 = c^2) : 
  d = 35 :=
by
  sorry

end middle_tree_distance_l292_292740


namespace inequality_holds_l292_292422

variable {a b c : ℝ}

theorem inequality_holds (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) : (a - b) * c ^ 2 ≤ 0 :=
sorry

end inequality_holds_l292_292422


namespace value_of_f_at_5_l292_292691

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_f_at_5 : f 5 = 15 := 
by {
  sorry
}

end value_of_f_at_5_l292_292691


namespace total_money_shared_l292_292109

theorem total_money_shared 
  (A B C D total : ℕ) 
  (h1 : A = 3 * 15)
  (h2 : B = 5 * 15)
  (h3 : C = 6 * 15)
  (h4 : D = 8 * 15)
  (h5 : A = 45) :
  total = A + B + C + D → total = 330 :=
by
  sorry

end total_money_shared_l292_292109


namespace square_area_l292_292841

theorem square_area 
  (s r l : ℝ)
  (h_r_s : r = s)
  (h_l_r : l = (2/5) * r)
  (h_area_rect : l * 10 = 120) : 
  s^2 = 900 := by
  -- Proof will go here
  sorry

end square_area_l292_292841


namespace dilation_at_origin_neg3_l292_292443

-- Define the dilation matrix centered at the origin with scale factor -3
def dilation_matrix (scale_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor, 0], ![0, scale_factor]]

-- The theorem stating that a dilation with scale factor -3 results in the specified matrix
theorem dilation_at_origin_neg3 :
  dilation_matrix (-3) = ![![(-3 : ℝ), 0], ![0, -3]] :=
sorry

end dilation_at_origin_neg3_l292_292443


namespace coeff_x_squared_is_16_l292_292372

open Nat

noncomputable def coeff_x_squared := (∑ k in finset.range 5, (binom 4 k * (pow (-2) k) * (k = 1)) * (1 : ℚ))

theorem coeff_x_squared_is_16 :
  coeff_x_squared = 16 :=
by
  -- Coefficient of x^2 term calculation.
  sorry

end coeff_x_squared_is_16_l292_292372


namespace physical_fitness_test_l292_292168

theorem physical_fitness_test (x : ℝ) (hx : x > 0) :
  (1000 / x - 1000 / (1.25 * x) = 30) :=
sorry

end physical_fitness_test_l292_292168


namespace total_turnips_l292_292035

-- Conditions
def turnips_keith : ℕ := 6
def turnips_alyssa : ℕ := 9

-- Statement to be proved
theorem total_turnips : turnips_keith + turnips_alyssa = 15 := by
  -- Proof is not required for this prompt, so we use sorry
  sorry

end total_turnips_l292_292035


namespace train_length_l292_292106

theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (speed_mps : ℝ) (length_train : ℝ) : 
  speed_kmph = 90 → 
  time_seconds = 6 → 
  speed_mps = (speed_kmph * 1000 / 3600) →
  length_train = (speed_mps * time_seconds) → 
  length_train = 150 :=
by
  intros h_speed h_time h_speed_mps h_length
  sorry

end train_length_l292_292106


namespace find_x_l292_292800

theorem find_x {x y : ℝ} (h1 : 3 * x - 2 * y = 7) (h2 : x^2 + 3 * y = 17) : x = 3.5 :=
sorry

end find_x_l292_292800


namespace count_pairs_satisfying_condition_l292_292619

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l292_292619


namespace inequality_proof_l292_292270

theorem inequality_proof
  {a b c d e f : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (h_abs : |sqrt (a * d) - sqrt (b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ 
    (a^2 * f^2 - (b^2) / (f^2)) * ((d^2) / (f^2) - c^2 * f^2) :=
by
  sorry

end inequality_proof_l292_292270


namespace total_amount_paid_l292_292377

/-- The owner's markup percentage and the cost price are given. 
We need to find out the total amount paid by the customer, which is equivalent to proving the total cost. -/
theorem total_amount_paid (markup_percentage : ℝ) (cost_price : ℝ) (markup : ℝ) (total_paid : ℝ) 
    (h1 : markup_percentage = 0.24) 
    (h2 : cost_price = 6425) 
    (h3 : markup = markup_percentage * cost_price) 
    (h4 : total_paid = cost_price + markup) : 
    total_paid = 7967 := 
sorry

end total_amount_paid_l292_292377


namespace number_of_people_to_the_left_of_Kolya_l292_292985

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l292_292985


namespace roots_cubic_identity_l292_292017

theorem roots_cubic_identity (p q r s : ℝ) (h1 : r + s = p) (h2 : r * s = -q) (h3 : ∀ x : ℝ, x^2 - p*x - q = 0 → (x = r ∨ x = s)) :
  r^3 + s^3 = p^3 + 3*p*q := by
  sorry

end roots_cubic_identity_l292_292017


namespace smallest_lambda_inequality_l292_292305

theorem smallest_lambda_inequality 
  (a b c d : ℝ) (h_pos : ∀ x ∈ [a, b, c, d], 0 < x) (h_sum : a + b + c + d = 4) :
  5 * (a*b + a*c + a*d + b*c + b*d + c*d) ≤ 8 * (a*b*c*d) + 12 :=
sorry

end smallest_lambda_inequality_l292_292305


namespace polynomial_simplification_l292_292055

theorem polynomial_simplification (y : ℤ) : 
  (2 * y - 1) * (4 * y ^ 10 + 2 * y ^ 9 + 4 * y ^ 8 + 2 * y ^ 7) = 8 * y ^ 11 + 6 * y ^ 9 - 2 * y ^ 7 :=
by 
  sorry

end polynomial_simplification_l292_292055


namespace work_problem_l292_292539

theorem work_problem (W : ℕ) (h1: ∀ w, w = W → (24 * w + 1 = 73)) : W = 3 :=
by {
  -- Insert proof here
  sorry
}

end work_problem_l292_292539


namespace ellipse_range_l292_292466

theorem ellipse_range (t : ℝ) (x y : ℝ) :
  (10 - t > 0) → (t - 4 > 0) → (10 - t ≠ t - 4) →
  (t ∈ (Set.Ioo 4 7 ∪ Set.Ioo 7 10)) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_l292_292466


namespace rachels_milk_consumption_l292_292434

theorem rachels_milk_consumption :
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  rachel_ratio * total_milk = (15 / 32 : ℚ) :=
by
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  -- proof placeholder
  sorry

end rachels_milk_consumption_l292_292434


namespace fraction_of_brilliant_integers_divisible_by_18_l292_292761

def is_brilliant (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n > 20 ∧ n < 200 ∧ (n.digits 10).sum = 11

def is_divisible_by_18 (n : ℕ) : Prop :=
  n % 18 = 0

theorem fraction_of_brilliant_integers_divisible_by_18 :
  let brilliant_integers := { n : ℕ | is_brilliant n }
  let divisible_brilliant_integers := { n : ℕ | is_brilliant n ∧ is_divisible_by_18 n }
  brilliant_integers.nonempty →
  (divisible_brilliant_integers.card / brilliant_integers.card : ℚ) = 2 / 7 :=
  by
  sorry

end fraction_of_brilliant_integers_divisible_by_18_l292_292761


namespace slope_y_intercept_product_l292_292223

theorem slope_y_intercept_product (m b : ℝ) (hm : m = -1/2) (hb : b = 4/5) : -1 < m * b ∧ m * b < 0 :=
by
  sorry

end slope_y_intercept_product_l292_292223


namespace average_salary_rest_l292_292641

variable (totalWorkers : ℕ)
variable (averageSalaryAll : ℕ)
variable (numTechnicians : ℕ)
variable (averageSalaryTechnicians : ℕ)

theorem average_salary_rest (h1 : totalWorkers = 28) 
                           (h2 : averageSalaryAll = 8000)
                           (h3 : numTechnicians = 7)
                           (h4 : averageSalaryTechnicians = 14000) : 
                           (averageSalaryAll * totalWorkers - averageSalaryTechnicians * numTechnicians) / (totalWorkers - numTechnicians) = 6000 :=
begin
  -- The proof will be provided here
  sorry
end

end average_salary_rest_l292_292641


namespace brandon_textbooks_weight_l292_292034

-- Define the weights of Jon's textbooks
def jon_textbooks : List ℕ := [2, 8, 5, 9]

-- Define the weight ratio between Jon's and Brandon's textbooks
def weight_ratio : ℕ := 3

-- Define the total weight of Jon's textbooks
def weight_jon : ℕ := jon_textbooks.sum

-- Define the weight of Brandon's textbooks to be proven
def weight_brandon : ℕ := weight_jon / weight_ratio

-- The theorem to be proven
theorem brandon_textbooks_weight : weight_brandon = 8 :=
by sorry

end brandon_textbooks_weight_l292_292034


namespace sum_of_consecutive_integers_l292_292244

theorem sum_of_consecutive_integers (x : ℤ) (h1 : x * (x + 1) + x + (x + 1) = 156) (h2 : x + 1 < 20) : x + (x + 1) = 23 :=
by
  sorry

end sum_of_consecutive_integers_l292_292244


namespace linear_function_no_third_quadrant_l292_292789

theorem linear_function_no_third_quadrant (m : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ -2 * x + 1 - m) : 
  m ≤ 1 :=
by
  sorry

end linear_function_no_third_quadrant_l292_292789


namespace alpha_beta_sum_two_l292_292007

theorem alpha_beta_sum_two (α β : ℝ) 
  (hα : α^3 - 3 * α^2 + 5 * α - 17 = 0)
  (hβ : β^3 - 3 * β^2 + 5 * β + 11 = 0) : 
  α + β = 2 :=
by
  sorry

end alpha_beta_sum_two_l292_292007


namespace find_consecutive_numbers_l292_292526

theorem find_consecutive_numbers (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c)
    (h_lcm : Nat.lcm a (Nat.lcm b c) = 660) : a = 10 ∧ b = 11 ∧ c = 12 := 
    sorry

end find_consecutive_numbers_l292_292526


namespace count_numbers_without_digit_1_l292_292013

theorem count_numbers_without_digit_1 :
  let count := Finset.filter (λ n : ℕ, 
        n < 1001 ∧ (n.toString.toList.all (λ d, d ≠ '1'))
      ) (Finset.range 1001)
  in count.card = 728 :=
by
  sorry

end count_numbers_without_digit_1_l292_292013


namespace fraction_meaningful_range_l292_292480

variable (x : ℝ)

theorem fraction_meaningful_range (h : x - 2 ≠ 0) : x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l292_292480


namespace equal_candies_l292_292904

theorem equal_candies
  (sweet_math_per_box : ℕ := 12)
  (geometry_nuts_per_box : ℕ := 15)
  (sweet_math_boxes : ℕ := 5)
  (geometry_nuts_boxes : ℕ := 4) :
  sweet_math_boxes * sweet_math_per_box = geometry_nuts_boxes * geometry_nuts_per_box := 
  by
  sorry

end equal_candies_l292_292904


namespace solve_for_a_l292_292319

def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 6

theorem solve_for_a (a : ℝ) (h : a > 0) (h1 : f (g a) = 18) : a = Real.sqrt (2 * Real.sqrt 2 + 6) :=
by
  sorry

end solve_for_a_l292_292319


namespace find_n_l292_292749

noncomputable def r1 : ℚ := 6 / 15
noncomputable def S1 : ℚ := 15 / (1 - r1)
noncomputable def r2 (n : ℚ) : ℚ := (6 + n) / 15
noncomputable def S2 (n : ℚ) : ℚ := 15 / (1 - r2 n)

theorem find_n : ∃ (n : ℚ), S2 n = 3 * S1 ∧ n = 6 :=
by
  use 6
  sorry

end find_n_l292_292749


namespace value_of_expression_l292_292485

theorem value_of_expression (x y : ℤ) (h1 : x = -6) (h2 : y = -3) : 4 * (x - y) ^ 2 - x * y = 18 :=
by sorry

end value_of_expression_l292_292485


namespace matchsticks_left_l292_292124

theorem matchsticks_left (total_matchsticks elvis_match_per_square ralph_match_per_square elvis_squares ralph_squares : ℕ)
  (h1 : total_matchsticks = 50)
  (h2 : elvis_match_per_square = 4)
  (h3 : ralph_match_per_square = 8)
  (h4 : elvis_squares = 5)
  (h5 : ralph_squares = 3) :
  total_matchsticks - (elvis_match_per_square * elvis_squares + ralph_match_per_square * ralph_squares) = 6 := 
by
  sorry

end matchsticks_left_l292_292124


namespace ellipse_foci_coordinates_l292_292689

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (y^2 / 3 + x^2 / 2 = 1) → (x, y) = (0, -1) ∨ (x, y) = (0, 1) :=
by
  sorry

end ellipse_foci_coordinates_l292_292689


namespace tony_initial_amount_l292_292541

-- Define the initial amount P
variable (P : ℝ)

-- Define the conditions
def initial_amount := P
def after_first_year := 1.20 * P
def after_half_taken := 0.60 * P
def after_second_year := 0.69 * P
def final_amount : ℝ := 690

-- State the theorem to prove
theorem tony_initial_amount : 
  (after_second_year P = final_amount) → (initial_amount P = 1000) :=
by 
  intro h
  sorry

end tony_initial_amount_l292_292541


namespace original_price_of_petrol_l292_292876

theorem original_price_of_petrol (P : ℝ) :
  (∃ P, 
    ∀ (GA GB GC : ℝ),
    0.8 * P = 0.8 * P ∧
    GA = 200 / P ∧
    GB = 300 / P ∧
    GC = 400 / P ∧
    200 = (GA + 8) * 0.8 * P ∧
    300 = (GB + 15) * 0.8 * P ∧
    400 = (GC + 22) * 0.8 * P) → 
  P = 6.25 :=
by
  sorry

end original_price_of_petrol_l292_292876


namespace rational_powers_imply_integers_l292_292460

theorem rational_powers_imply_integers (a b : ℚ) (h_distinct : a ≠ b)
  (h_infinitely_many_n : ∃ᶠ (n : ℕ) in Filter.atTop, (n * (a^n - b^n) : ℚ).den = 1) :
  ∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int := 
sorry

end rational_powers_imply_integers_l292_292460


namespace minimum_filtrations_needed_l292_292409

theorem minimum_filtrations_needed (I₀ I_n : ℝ) (n : ℕ) (h1 : I₀ = 0.02) (h2 : I_n ≤ 0.001) (h3 : I_n = I₀ * 0.5 ^ n) :
  n = 8 := by
sorry

end minimum_filtrations_needed_l292_292409


namespace does_not_pass_through_third_quadrant_l292_292222

noncomputable def f (a b x : ℝ) : ℝ := a^x + b - 1

theorem does_not_pass_through_third_quadrant (a b : ℝ) (h_a : 0 < a ∧ a < 1) (h_b : 0 < b ∧ b < 1) :
  ¬ ∃ x, f a b x < 0 ∧ x < 0 := sorry

end does_not_pass_through_third_quadrant_l292_292222


namespace solve_inequality_l292_292129

theorem solve_inequality (x : ℝ) : 
  1 / (x^2 + 2) > 4 / x + 21 / 10 ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end solve_inequality_l292_292129


namespace max_distance_convoy_l292_292560

structure Vehicle :=
  (mpg : ℝ) (min_gallons : ℝ)

def SUV : Vehicle := ⟨12.2, 10⟩
def Sedan : Vehicle := ⟨52, 5⟩
def Motorcycle : Vehicle := ⟨70, 2⟩

def total_gallons : ℝ := 21

def total_distance (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) : ℝ :=
  SUV.mpg * SUV_gallons + Sedan.mpg * Sedan_gallons + Motorcycle.mpg * Motorcycle_gallons

theorem max_distance_convoy (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) :
  SUV_gallons + Sedan_gallons + Motorcycle_gallons = total_gallons →
  SUV_gallons >= SUV.min_gallons →
  Sedan_gallons >= Sedan.min_gallons →
  Motorcycle_gallons >= Motorcycle.min_gallons →
  total_distance SUV_gallons Sedan_gallons Motorcycle_gallons = 802 :=
sorry

end max_distance_convoy_l292_292560


namespace minimum_pieces_for_K_1997_l292_292408

-- Definitions provided by the conditions in the problem.
def is_cube_shaped (n : ℕ) := ∃ (a : ℕ), n = a^3

def has_chocolate_coating (surface_area : ℕ) (n : ℕ) := 
  surface_area = 6 * n^2

def min_pieces (n K : ℕ) := n^3 / K

-- Expressing the proof problem in Lean 4.
theorem minimum_pieces_for_K_1997 {n : ℕ} (h_n : n = 1997) (H : ∀ (K : ℕ), K = 1997 ∧ K > 0) 
  (h_cube : is_cube_shaped n) (h_chocolate : has_chocolate_coating 6 n) :
  min_pieces 1997 1997 = 1997^3 :=
by
  sorry

end minimum_pieces_for_K_1997_l292_292408


namespace max_area_of_triangle_AMN_l292_292249

noncomputable def parabola (y : ℝ) : ℝ := (y^2) / 4

def line (x b : ℝ) : ℝ := x + b

def triangle_area (b : ℝ) : ℝ := 
  2 * abs(5 + b) * real.sqrt(1 - b)

theorem max_area_of_triangle_AMN : 
  ∃ b : ℝ, (∀ x b : ℝ, y = line x b) ∧ 
           (y^2 = 4 * x) ∧ 
           b = -1 → 
           triangle_area b = 8 * real.sqrt 2 := 
begin
  sorry
end

end max_area_of_triangle_AMN_l292_292249


namespace num_pairs_of_regular_polygons_l292_292586

def num_pairs : Nat := 
  let pairs := [(7, 42), (6, 18), (5, 10), (4, 6)]
  pairs.length

theorem num_pairs_of_regular_polygons : num_pairs = 4 := 
  sorry

end num_pairs_of_regular_polygons_l292_292586


namespace min_additional_coins_needed_l292_292745

/--
Alex has 15 friends, 90 coins, and needs to give each friend at least one coin with no two friends receiving the same number of coins.
Prove that the minimum number of additional coins he needs is 30.
-/
theorem min_additional_coins_needed (
  friends : ℕ := 15
  coins : ℕ := 90
) (h1 : friends = 15)
  (h2 : coins = 90) : 
  let total_required := (friends * (friends + 1)) / 2 in
  total_required - coins = 30 :=
by {
  have total_required_eq : total_required = (15 * (15 + 1)) / 2, from by simp [friends, h1],
  have total_required_eval : total_required = 120, from calc
    total_required = (15 * 16) / 2 : by rw total_required_eq
                 ... = 120        : by norm_num,
  calc
    total_required - coins = 120 - 90 : by rw [total_required_eval, h2]
                 ... = 30             : by norm_num
}

end min_additional_coins_needed_l292_292745


namespace gift_exchange_equation_l292_292886

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l292_292886


namespace smallest_positive_integer_23n_mod_5678_mod_11_l292_292259

theorem smallest_positive_integer_23n_mod_5678_mod_11 :
  ∃ n : ℕ, 0 < n ∧ 23 * n % 11 = 5678 % 11 ∧ ∀ m : ℕ, 0 < m ∧ 23 * m % 11 = 5678 % 11 → n ≤ m :=
by
  sorry

end smallest_positive_integer_23n_mod_5678_mod_11_l292_292259


namespace triangle_isosceles_if_equal_bisectors_l292_292365

theorem triangle_isosceles_if_equal_bisectors
  (A B C : ℝ)
  (a b c l_a l_b : ℝ)
  (ha : l_a = l_b)
  (h1 : l_a = 2 * b * c * Real.cos (A / 2) / (b + c))
  (h2 : l_b = 2 * a * c * Real.cos (B / 2) / (a + c)) :
  a = b :=
by
  sorry

end triangle_isosceles_if_equal_bisectors_l292_292365


namespace gopi_salary_turbans_l292_292010

-- Define the question and conditions as statements
def total_salary (turbans : ℕ) : ℕ := 90 + 30 * turbans
def servant_receives : ℕ := 60 + 30
def fraction_annual_salary : ℚ := 3 / 4

-- The theorem statement capturing the equivalent proof problem
theorem gopi_salary_turbans (T : ℕ) 
  (salary_eq : total_salary T = 90 + 30 * T)
  (servant_eq : servant_receives = 60 + 30)
  (fraction_eq : fraction_annual_salary = 3 / 4)
  (received_after_9_months : ℚ) :
  fraction_annual_salary * (90 + 30 * T : ℚ) = received_after_9_months → 
  received_after_9_months = 90 →
  T = 1 :=
sorry

end gopi_salary_turbans_l292_292010


namespace total_pages_allowed_l292_292816

noncomputable def words_total := 48000
noncomputable def words_per_page_large := 1800
noncomputable def words_per_page_small := 2400
noncomputable def pages_large := 4
noncomputable def total_pages : ℕ := 21

theorem total_pages_allowed :
  pages_large * words_per_page_large + (total_pages - pages_large) * words_per_page_small = words_total :=
  by sorry

end total_pages_allowed_l292_292816


namespace Bryan_deposited_312_l292_292043

-- Definitions based on conditions
def MarkDeposit : ℕ := 88
def TotalDeposit : ℕ := 400
def MaxBryanDeposit (MarkDeposit : ℕ) : ℕ := 5 * MarkDeposit 

def BryanDeposit (B : ℕ) : Prop := B < MaxBryanDeposit MarkDeposit ∧ MarkDeposit + B = TotalDeposit

theorem Bryan_deposited_312 : BryanDeposit 312 :=
by
   -- Proof steps go here
   sorry

end Bryan_deposited_312_l292_292043


namespace proposition_range_l292_292930

theorem proposition_range (m : ℝ) : 
  (m < 1/2 ∧ m ≠ 1/3) ∨ (m = 3) ↔ m ∈ Set.Iio (1/3:ℝ) ∪ Set.Ioo (1/3:ℝ) (1/2:ℝ) ∪ {3} :=
sorry

end proposition_range_l292_292930


namespace no_conf_of_7_points_and_7_lines_l292_292827

theorem no_conf_of_7_points_and_7_lines (points : Fin 7 → Prop) (lines : Fin 7 → (Fin 7 → Prop)) :
  (∀ p : Fin 7, ∃ l₁ l₂ l₃ : Fin 7, lines l₁ p ∧ lines l₂ p ∧ lines l₃ p ∧ l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
  (∀ l : Fin 7, ∃ p₁ p₂ p₃ : Fin 7, lines l p₁ ∧ lines l p₂ ∧ lines l p₃ ∧ p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃) 
  → false :=
by
  sorry

end no_conf_of_7_points_and_7_lines_l292_292827


namespace mario_haircut_price_l292_292294

theorem mario_haircut_price (P : ℝ) 
  (weekend_multiplier : ℝ := 1.50)
  (sunday_price : ℝ := 27) 
  (weekend_price_eq : sunday_price = P * weekend_multiplier) : 
  P = 18 := 
by
  sorry

end mario_haircut_price_l292_292294


namespace winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l292_292683

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

end winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l292_292683


namespace man_rowing_upstream_speed_l292_292735

theorem man_rowing_upstream_speed (V_down V_m V_up V_s : ℕ) 
  (h1 : V_down = 41)
  (h2 : V_m = 33)
  (h3 : V_down = V_m + V_s)
  (h4 : V_up = V_m - V_s) 
  : V_up = 25 := 
by
  sorry

end man_rowing_upstream_speed_l292_292735


namespace matrix_power_minus_l292_292185

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end matrix_power_minus_l292_292185


namespace garden_ratio_l292_292412

theorem garden_ratio 
  (P : ℕ) (L : ℕ) (W : ℕ) 
  (h1 : P = 900) 
  (h2 : L = 300) 
  (h3 : P = 2 * (L + W)) : 
  L / W = 2 :=
by 
  sorry

end garden_ratio_l292_292412


namespace car_cost_l292_292108

def initial_savings : ℕ := 14500
def charge_per_trip : ℚ := 1.5
def percentage_groceries_earnings : ℚ := 0.05
def number_of_trips : ℕ := 40
def total_value_of_groceries : ℕ := 800

theorem car_cost (initial_savings charge_per_trip percentage_groceries_earnings number_of_trips total_value_of_groceries : ℚ) :
  initial_savings + (charge_per_trip * number_of_trips) + (percentage_groceries_earnings * total_value_of_groceries) = 14600 := 
by
  sorry

end car_cost_l292_292108


namespace find_roots_l292_292134

theorem find_roots (a b c d x : ℝ) (h₁ : a + d = 2015) (h₂ : b + c = 2015) (h₃ : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 0 := 
sorry

end find_roots_l292_292134


namespace football_club_balance_l292_292091

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end football_club_balance_l292_292091


namespace union_M_N_eq_l292_292155

def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N : Set ℝ := {0, 4}

theorem union_M_N_eq : M ∪ N = Set.Icc 0 4 := 
  by
    sorry

end union_M_N_eq_l292_292155


namespace present_age_of_B_l292_292166

theorem present_age_of_B :
  ∃ (A B : ℕ), (A + 20 = 2 * (B - 20)) ∧ (A = B + 10) ∧ (B = 70) :=
by
  sorry

end present_age_of_B_l292_292166


namespace smallest_model_length_l292_292677

theorem smallest_model_length (full_size : ℕ) (mid_size_factor smallest_size_factor : ℚ) :
  full_size = 240 →
  mid_size_factor = 1 / 10 →
  smallest_size_factor = 1 / 2 →
  (full_size * mid_size_factor) * smallest_size_factor = 12 :=
by
  intros h_full_size h_mid_size_factor h_smallest_size_factor
  sorry

end smallest_model_length_l292_292677


namespace invitation_methods_l292_292879

-- Definitions
def num_ways_invite_6_out_of_10 : ℕ := Nat.choose 10 6
def num_ways_both_A_and_B : ℕ := Nat.choose 8 4

-- Theorem statement
theorem invitation_methods : num_ways_invite_6_out_of_10 - num_ways_both_A_and_B = 140 :=
by
  -- Proof should be provided here
  sorry

end invitation_methods_l292_292879


namespace solve_number_l292_292684

noncomputable def find_number : Prop :=
  ∃ x : ℝ, (3/4 * x - 25) / 7 + 50 = 100 ∧ x = 500

theorem solve_number : find_number :=
  sorry

end solve_number_l292_292684


namespace chocolate_difference_l292_292199

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l292_292199


namespace stored_bales_correct_l292_292995

theorem stored_bales_correct :
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  stored_bales = 26 :=
by
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  show stored_bales = 26
  sorry

end stored_bales_correct_l292_292995


namespace max_value_fraction_l292_292308

theorem max_value_fraction (x : ℝ) : x ≠ 0 → 1 / (x^4 + 4*x^2 + 2 + 8/x^2 + 16/x^4) ≤ 1 / 31 :=
by sorry

end max_value_fraction_l292_292308


namespace elena_subtracts_99_to_compute_49_squared_l292_292710

noncomputable def difference_between_squares_50_49 : ℕ := 99

theorem elena_subtracts_99_to_compute_49_squared :
  ∀ (n : ℕ), n = 50 → (n - 1)^2 = n^2 - difference_between_squares_50_49 :=
by
  intro n
  sorry

end elena_subtracts_99_to_compute_49_squared_l292_292710


namespace sign_of_c_l292_292334

theorem sign_of_c (a b c : ℝ) (h1 : (a * b / c) < 0) (h2 : (a * b) < 0) : c > 0 :=
sorry

end sign_of_c_l292_292334


namespace certain_number_of_tenths_l292_292333

theorem certain_number_of_tenths (n : ℝ) (h : n = 375 * (1/10)) : n = 37.5 :=
by
  sorry

end certain_number_of_tenths_l292_292333


namespace correct_definition_of_regression_independence_l292_292264

-- Definitions
def regression_analysis (X Y : Type) := ∃ r : X → Y, true -- Placeholder, ideal definition studies correlation
def independence_test (X Y : Type) := ∃ rel : X → Y → Prop, true -- Placeholder, ideal definition examines relationship

-- Theorem statement
theorem correct_definition_of_regression_independence (X Y : Type) :
  (∃ r : X → Y, true) ∧ (∃ rel : X → Y → Prop, true)
  → "Regression analysis studies the correlation between two variables, and independence tests examine whether there is some kind of relationship between two variables" = "C" :=
sorry

end correct_definition_of_regression_independence_l292_292264


namespace map_length_l292_292286

theorem map_length 
  (width : ℝ) (area : ℝ) 
  (h_width : width = 10) (h_area : area = 20) : 
  ∃ length : ℝ, area = width * length ∧ length = 2 :=
by 
  sorry

end map_length_l292_292286


namespace transformed_polynomial_roots_l292_292821

variable (a b c d : ℝ)
variable (h1 : ∀ x : ℝ, x^4 - b * x^2 - 6 = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

theorem transformed_polynomial_roots :
  ∃ (f : ℝ → ℝ), (∀ y : ℝ, f y = 6 * y^2 + b * y + 1) ∧
  (f (-1 / a^2) = 0 ∧ f (-1 / b^2) = 0 ∧ f (-1 / c^2) = 0 ∧ f (-1 / d^2) = 0) :=
sorry

end transformed_polynomial_roots_l292_292821


namespace race_distance_l292_292956

theorem race_distance (D : ℝ)
  (A_speed : ℝ := D / 20)
  (B_speed : ℝ := D / 25)
  (A_beats_B_by : ℝ := 18)
  (h1 : A_speed * 25 = D + A_beats_B_by)
  : D = 72 := 
by
  sorry

end race_distance_l292_292956


namespace symmetric_colors_different_at_8281_div_2_l292_292113

def is_red (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ n = 81 * x + 100 * y

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n

theorem symmetric_colors_different_at_8281_div_2 :
  ∃ n : ℕ, (is_red n ∧ is_blue (8281 - n)) ∨ (is_blue n ∧ is_red (8281 - n)) ∧ 2 * n = 8281 :=
by
  sorry

end symmetric_colors_different_at_8281_div_2_l292_292113


namespace ratio_of_wages_l292_292655

def hours_per_day_josh : ℕ := 8
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def wage_per_hour_josh : ℕ := 9
def monthly_total_payment : ℚ := 1980

def hours_per_day_carl : ℕ := hours_per_day_josh - 2

def monthly_hours_josh : ℕ := hours_per_day_josh * days_per_week * weeks_per_month
def monthly_hours_carl : ℕ := hours_per_day_carl * days_per_week * weeks_per_month

def monthly_earnings_josh : ℚ := wage_per_hour_josh * monthly_hours_josh
def monthly_earnings_carl : ℚ := monthly_total_payment - monthly_earnings_josh

def hourly_wage_carl : ℚ := monthly_earnings_carl / monthly_hours_carl

theorem ratio_of_wages : hourly_wage_carl / wage_per_hour_josh = 1 / 2 := by
  sorry

end ratio_of_wages_l292_292655


namespace number_of_pairs_count_number_of_pairs_l292_292626

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l292_292626


namespace bridesmaids_count_l292_292050

theorem bridesmaids_count
  (hours_per_dress : ℕ)
  (hours_per_week : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (dresses : ℕ) :
  hours_per_dress = 12 →
  hours_per_week = 4 →
  weeks = 15 →
  total_hours = hours_per_week * weeks →
  dresses = total_hours / hours_per_dress →
  dresses = 5 := by
  sorry

end bridesmaids_count_l292_292050


namespace cost_per_item_l292_292815

theorem cost_per_item (total_cost : ℝ) (num_items : ℕ) (cost_per_item : ℝ) 
                      (h1 : total_cost = 26) (h2 : num_items = 8) : 
                      cost_per_item = total_cost / num_items := 
by
  sorry

end cost_per_item_l292_292815


namespace simplify_expression_l292_292974

theorem simplify_expression (x : ℝ) :
  (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 :=
by
  sorry

end simplify_expression_l292_292974


namespace three_Z_five_l292_292483

def Z (a b : ℤ) : ℤ := b + 10 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = 8 := sorry

end three_Z_five_l292_292483


namespace all_integers_equal_l292_292929

theorem all_integers_equal (k : ℕ) (a : Fin (2 * k + 1) → ℤ)
(h : ∀ b : Fin (2 * k + 1) → ℤ,
  (∀ i : Fin (2 * k + 1), b i = (a ((i : ℕ) % (2 * k + 1)) + a ((i + 1) % (2 * k + 1))) / 2) →
  ∀ i : Fin (2 * k + 1), ↑(b i) % 2 = 0) :
∀ i j : Fin (2 * k + 1), a i = a j :=
by
  sorry

end all_integers_equal_l292_292929


namespace six_people_with_A_not_on_ends_l292_292555

-- Define the conditions and the problem statement
def standing_arrangements (n : ℕ) (A : Type) :=
  {l : List A // l.length = n}

theorem six_people_with_A_not_on_ends : 
  (arr : standing_arrangements 6 ℕ) → 
  (∀ a ∈ arr.val, a ≠ 0 ∧ a ≠ 5) → 
  ∃! (total_arrangements : ℕ), total_arrangements = 480 :=
  by
    sorry

end six_people_with_A_not_on_ends_l292_292555


namespace problem_statement_l292_292927

noncomputable def f : ℕ+ → ℝ := sorry

theorem problem_statement (x : ℕ+) :
  (f 1 = 1) →
  (∀ x, f (x + 1) = (2 * f x) / (f x + 2)) →
  f x = 2 / (x + 1) := 
sorry

end problem_statement_l292_292927


namespace polynomial_roots_fraction_sum_l292_292116

theorem polynomial_roots_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 12) 
  (h2 : ab + ac + bc = 20) 
  (h3 : abc = 3) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 328 / 9 := 
by 
  sorry

end polynomial_roots_fraction_sum_l292_292116


namespace opposite_neg_fraction_l292_292226

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l292_292226


namespace div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l292_292768

-- Define the values provided in the problem
def div_1 := (8 : ℚ) / (8 / 17 : ℚ)
def div_2 := (6 / 11 : ℚ) / 3
def mul_1 := (5 / 4 : ℚ) * (1 / 5 : ℚ)

-- Prove the equivalences
theorem div_1_eq_17 : div_1 = 17 := by
  sorry

theorem div_2_eq_2_11 : div_2 = 2 / 11 := by
  sorry

theorem mul_1_eq_1_4 : mul_1 = 1 / 4 := by
  sorry

end div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l292_292768


namespace oak_total_after_planting_l292_292382

-- Let oak_current represent the current number of oak trees in the park.
def oak_current : ℕ := 9

-- Let oak_new represent the number of new oak trees being planted.
def oak_new : ℕ := 2

-- The problem is to prove the total number of oak trees after planting equals 11
theorem oak_total_after_planting : oak_current + oak_new = 11 :=
by
  sorry

end oak_total_after_planting_l292_292382


namespace largest_sum_of_distinct_factors_l292_292637

theorem largest_sum_of_distinct_factors (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) (h_product : A * B * C = 3003) :
  A + B + C ≤ 105 :=
sorry  -- Proof is not required, just the statement.

end largest_sum_of_distinct_factors_l292_292637


namespace riverside_theme_parks_adjustment_plans_l292_292172

/-
In my city, we are building the happiest city with a plan to construct 7 riverside theme parks along the Wei River.
To enhance the city's quality and upgrade the park functions, it is proposed to reduce the number of riverside theme parks by 2.
The theme parks at both ends of the river are not to be adjusted, and two adjacent riverside theme parks cannot be adjusted simultaneously.
The number of possible adjustment plans is 6.
-/

theorem riverside_theme_parks_adjustment_plans :
  let total_parks := 7
  let end_parks := 2
  let remaining_parks := total_parks - end_parks
  let adjustments_needed := 2
  let adjacent_pairs := 4
  nat.choose remaining_parks adjustments_needed - adjacent_pairs = 6 := 
by {
  sorry
}

end riverside_theme_parks_adjustment_plans_l292_292172


namespace speed_of_man_rowing_upstream_l292_292410

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream : ℝ) 
  (H1 : V_m = 60) 
  (H2 : V_downstream = 65) 
  (H3 : V_upstream = V_m - (V_downstream - V_m)) : 
  V_upstream = 55 := 
by 
  subst H1 
  subst H2 
  rw [H3] 
  norm_num

end speed_of_man_rowing_upstream_l292_292410


namespace derivative_at_2_l292_292324

noncomputable def f (x : ℝ) : ℝ := x

theorem derivative_at_2 : (deriv f 2) = 1 :=
by
  -- sorry, proof not included
  sorry

end derivative_at_2_l292_292324


namespace distance_between_centers_same_side_distance_between_centers_opposite_side_l292_292839

open Real

noncomputable def distance_centers_same_side (r : ℝ) : ℝ := (r * (sqrt 6 + sqrt 2)) / 2

noncomputable def distance_centers_opposite_side (r : ℝ) : ℝ := (r * (sqrt 6 - sqrt 2)) / 2

theorem distance_between_centers_same_side (r : ℝ):
  ∃ dist, dist = distance_centers_same_side r :=
sorry

theorem distance_between_centers_opposite_side (r : ℝ):
  ∃ dist, dist = distance_centers_opposite_side r :=
sorry

end distance_between_centers_same_side_distance_between_centers_opposite_side_l292_292839


namespace canadian_olympiad_2008_inequality_l292_292780

variable (a b c : ℝ)
variables (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
variable (sum_abc : a + b + c = 1)

theorem canadian_olympiad_2008_inequality :
  (ab / ((b + c) * (c + a))) + (bc / ((c + a) * (a + b))) + (ca / ((a + b) * (b + c))) ≥ 3 / 4 :=
sorry

end canadian_olympiad_2008_inequality_l292_292780


namespace geometric_sequence_sum_l292_292027

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n)
    (h1 : a 0 + a 1 = 324) (h2 : a 2 + a 3 = 36) : a 4 + a 5 = 4 :=
by
  sorry

end geometric_sequence_sum_l292_292027


namespace min_gumballs_to_ensure_four_same_color_l292_292098

/-- A structure to represent the number of gumballs of each color. -/
structure Gumballs :=
(red : ℕ)
(white : ℕ)
(blue : ℕ)
(green : ℕ)

def gumball_machine : Gumballs := { red := 10, white := 9, blue := 8, green := 6 }

/-- Theorem to state the minimum number of gumballs required to ensure at least four of any color. -/
theorem min_gumballs_to_ensure_four_same_color 
  (g : Gumballs) 
  (h1 : g.red = 10)
  (h2 : g.white = 9)
  (h3 : g.blue = 8)
  (h4 : g.green = 6) : 
  ∃ n, n = 13 := 
sorry

end min_gumballs_to_ensure_four_same_color_l292_292098


namespace peter_reads_more_books_l292_292515

-- Definitions and conditions
def total_books : ℕ := 20
def peter_percentage_read : ℕ := 40
def brother_percentage_read : ℕ := 10

def percentage_to_count (percentage : ℕ) (total : ℕ) : ℕ := (percentage * total) / 100

-- Main statement to prove
theorem peter_reads_more_books :
  percentage_to_count peter_percentage_read total_books - percentage_to_count brother_percentage_read total_books = 6 :=
by
  sorry

end peter_reads_more_books_l292_292515


namespace chemical_reaction_produces_l292_292613

def balanced_equation : Prop :=
  ∀ {CaCO3 HCl CaCl2 CO2 H2O : ℕ},
    (CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O)

def calculate_final_products (initial_CaCO3 initial_HCl final_CaCl2 final_CO2 final_H2O remaining_HCl : ℕ) : Prop :=
  balanced_equation ∧
  initial_CaCO3 = 3 ∧
  initial_HCl = 8 ∧
  final_CaCl2 = 3 ∧
  final_CO2 = 3 ∧
  final_H2O = 3 ∧
  remaining_HCl = 2

theorem chemical_reaction_produces :
  calculate_final_products 3 8 3 3 3 2 :=
by sorry

end chemical_reaction_produces_l292_292613


namespace matrix_power_minus_l292_292184

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end matrix_power_minus_l292_292184


namespace trigonometric_expression_evaluation_l292_292114

theorem trigonometric_expression_evaluation :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 3 / Real.cos (70 * Real.pi / 180) = -4 :=
by
  sorry

end trigonometric_expression_evaluation_l292_292114


namespace min_value_expression_l292_292823

theorem min_value_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end min_value_expression_l292_292823


namespace common_solution_ys_l292_292588

theorem common_solution_ys : 
  {y : ℝ | ∃ x : ℝ, x^2 + y^2 = 9 ∧ x^2 + 2*y = 7} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
sorry

end common_solution_ys_l292_292588


namespace sandy_total_sums_l292_292517

theorem sandy_total_sums (C I : ℕ) (h1 : C = 22) (h2 : 3 * C - 2 * I = 50) :
  C + I = 30 :=
sorry

end sandy_total_sums_l292_292517


namespace driving_time_to_beach_l292_292175

theorem driving_time_to_beach (total_trip_time : ℝ) (k : ℝ) (x : ℝ)
  (h1 : total_trip_time = 14)
  (h2 : k = 2.5)
  (h3 : total_trip_time = (2 * x) + (k * (2 * x))) :
  x = 2 := by 
  sorry

end driving_time_to_beach_l292_292175


namespace solve_fractional_equation_l292_292533

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 4) : 
  (3 - x) / (x - 4) + 1 / (4 - x) = 1 → x = 3 :=
by {
  sorry
}

end solve_fractional_equation_l292_292533


namespace solve_for_x_l292_292463

theorem solve_for_x (x : ℝ) (h : 3 * x + 1 = -(5 - 2 * x)) : x = -6 :=
by
  sorry

end solve_for_x_l292_292463


namespace add_congruence_mul_congruence_l292_292049

namespace ModularArithmetic

-- Define the congruence relation mod m
def is_congruent_mod (a b m : ℤ) : Prop := ∃ k : ℤ, a - b = k * m

-- Part (a): Proving a + c ≡ b + d (mod m)
theorem add_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a + c) (b + d) m :=
  sorry

-- Part (b): Proving a ⋅ c ≡ b ⋅ d (mod m)
theorem mul_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a * c) (b * d) m :=
  sorry

end ModularArithmetic

end add_congruence_mul_congruence_l292_292049


namespace inequality_solution_l292_292130

theorem inequality_solution :
  {x : ℝ | (x^2 + 5 * x) / ((x - 3) ^ 2) ≥ 0} = {x | x < -5} ∪ {x | 0 ≤ x ∧ x < 3} ∪ {x | x > 3} :=
by
  sorry

end inequality_solution_l292_292130


namespace maximize_sequence_l292_292154

theorem maximize_sequence (n : ℕ) (an : ℕ → ℝ) (h : ∀ n, an n = (10/11)^n * (3 * n + 13)) : 
  (∃ n_max, (∀ m, an m ≤ an n_max) ∧ n_max = 6) :=
by
  sorry

end maximize_sequence_l292_292154


namespace simplify_fraction_l292_292522

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end simplify_fraction_l292_292522


namespace d_share_l292_292721

theorem d_share (x : ℝ) (d c : ℝ)
  (h1 : c = 3 * x + 500)
  (h2 : d = 3 * x)
  (h3 : c = 4 * x) :
  d = 1500 := 
by 
  sorry

end d_share_l292_292721


namespace smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l292_292258

theorem smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits : 
  ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1 ∧ d % 2 = 0) ∧ 
    (n % 11 = 0)) ∧ n = 1056 :=
by
  sorry

end smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l292_292258


namespace quadratic_has_real_roots_l292_292924

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, x^2 + x - 4 * m = 0) ↔ m ≥ -1 / 16 :=
by
  sorry

end quadratic_has_real_roots_l292_292924


namespace final_balance_is_60_million_l292_292094

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end final_balance_is_60_million_l292_292094


namespace negation_of_universal_proposition_l292_292978

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l292_292978


namespace largest_integer_a_can_be_less_than_l292_292395

theorem largest_integer_a_can_be_less_than (a b : ℕ) (h1 : 9 < a) (h2 : 19 < b) (h3 : b < 31) (h4 : a / b = 2 / 3) :
  a < 21 :=
sorry

end largest_integer_a_can_be_less_than_l292_292395


namespace profit_difference_l292_292374

-- Setting up the conditions
def construction_cost_others (C : ℝ) : ℝ := C

def construction_cost_certain (C : ℝ) : ℝ := C + 100000

def selling_price_others : ℝ := 320000

def selling_price_certain : ℝ := 1.5 * 320000

def profit_certain (C : ℝ) : ℝ := selling_price_certain - construction_cost_certain C

def profit_others (C : ℝ) : ℝ := selling_price_others - construction_cost_others C

-- Proving the difference in profit
theorem profit_difference (C : ℝ) : profit_certain C - profit_others C = 60000 :=
by
    simp [profit_certain, profit_others, selling_price_certain, selling_price_others, construction_cost_certain, construction_cost_others]
    ring
    sorry

end profit_difference_l292_292374


namespace courtyard_is_25_meters_long_l292_292088

noncomputable def courtyard_length (width : ℕ) (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) : ℝ :=
  let brick_area := brick_length * brick_width
  let total_area := num_bricks * brick_area
  total_area / width

theorem courtyard_is_25_meters_long (h_width : 16 = 16)
  (h_brick_length : 0.20 = 0.20)
  (h_brick_width: 0.10 = 0.10)
  (h_num_bricks: 20_000 = 20_000)
  (h_total_area: 20_000 * (0.20 * 0.10) = 400) :
  courtyard_length 16 0.20 0.10 20_000 = 25 := by
        sorry

end courtyard_is_25_meters_long_l292_292088


namespace part1_part2_l292_292870

-- Let m be the cost price this year
-- Let x be the selling price per bottle
-- Assuming:
-- 1. The cost price per bottle increased by 4 yuan this year compared to last year.
-- 2. The quantity of detergent purchased for 1440 yuan this year equals to the quantity purchased for 1200 yuan last year.
-- 3. The selling price per bottle is 36 yuan with 600 bottles sold per week.
-- 4. Weekly sales increase by 100 bottles for every 1 yuan reduction in price.
-- 5. The selling price cannot be lower than the cost price.

-- Definition for improved readability:
def costPriceLastYear (m : ℕ) : ℕ := m - 4

-- Quantity equations
def quantityPurchasedThisYear (m : ℕ) : ℕ := 1440 / m
def quantityPurchasedLastYear (m : ℕ) : ℕ := 1200 / (costPriceLastYear m)

-- Profit Function
def profitFunction (m x : ℝ) : ℝ :=
  (x - m) * (600 + 100 * (36 - x))

-- Maximum Profit and Best Selling Price
def maxProfit : ℝ := 8100
def bestSellingPrice : ℝ := 33

theorem part1 (m : ℕ) (h₁ : 1440 / m = 1200 / costPriceLastYear m) : m = 24 := by
  sorry  -- Will be proved later

theorem part2 (m : ℝ) (x : ℝ)
    (h₀ : m = 24)
    (hx : 600 + 100 * (36 - x) > 0)
    (hx₁ : x ≥ m)
    : profitFunction m x ≤ maxProfit ∧ (∃! (y : ℝ), y = bestSellingPrice ∧ profitFunction m y = maxProfit) := by
  sorry  -- Will be proved later

end part1_part2_l292_292870


namespace find_f_105_5_l292_292934

noncomputable def f : ℝ → ℝ :=
sorry -- Definition of f

-- Hypotheses
axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (x + 2) = -f x
axiom function_values (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 3) : f x = x

-- Goal
theorem find_f_105_5 : f 105.5 = 2.5 :=
sorry

end find_f_105_5_l292_292934


namespace min_value_fraction_l292_292949

theorem min_value_fraction (x : ℝ) (hx : x < 2) : ∃ y : ℝ, y = (5 - 4 * x + x^2) / (2 - x) ∧ y = 2 :=
by sorry

end min_value_fraction_l292_292949


namespace arrange_COMMUNICATION_l292_292118

theorem arrange_COMMUNICATION : 
  let n := 12
  let o_count := 2
  let i_count := 2
  let n_count := 2
  let m_count := 2
  let total_repeats := o_count * i_count * n_count * m_count
  n.factorial / (o_count.factorial * i_count.factorial * n_count.factorial * m_count.factorial) = 29937600 :=
by sorry

end arrange_COMMUNICATION_l292_292118


namespace smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits_l292_292257

theorem smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 11 = 0) ∧ (num_even_digits n = 2) ∧ (num_odd_digits n = 2) ∧ (n = 1469) :=
by
  sorry

-- Auxiliary definitions for counting even and odd digits can be provided
def num_even_digits (n : ℕ) : ℕ :=
  (to_digits n).count (λ d, d % 2 = 0)

def num_odd_digits (n : ℕ) : ℕ :=
  (to_digits n).count (λ d, d % 2 = 1)

def to_digits (n : ℕ) : list ℕ :=
  if n < 10 then [n] else to_digits (n / 10) ++ [n % 10]

end smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits_l292_292257


namespace find_f_l292_292150

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f (x : ℝ) :
  (∀ t : ℝ, t = (1 - x) / (1 + x) → f t = (1 - x^2) / (1 + x^2)) →
  f x = (2 * x) / (1 + x^2) :=
by
  intros h
  specialize h ((1 - x) / (1 + x))
  specialize h rfl
  exact sorry

end find_f_l292_292150


namespace expression_evaluation_l292_292948

theorem expression_evaluation (x y : ℝ) (h₁ : x > y) (h₂ : y > 0) : 
    (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x / y)^(y - x) :=
by
  sorry

end expression_evaluation_l292_292948


namespace total_sales_15_days_l292_292298

def edgar_sales (n : ℕ) : ℕ := 3 * n - 1

def clara_sales (n : ℕ) : ℕ := 4 * n

def edgar_total_sales (d : ℕ) : ℕ := (d * (2 + (d * 3 - 1))) / 2

def clara_total_sales (d : ℕ) : ℕ := (d * (4 + (d * 4))) / 2

def total_sales (d : ℕ) : ℕ := edgar_total_sales d + clara_total_sales d

theorem total_sales_15_days : total_sales 15 = 810 :=
by
  sorry

end total_sales_15_days_l292_292298


namespace added_number_after_doubling_l292_292873

theorem added_number_after_doubling (original_number : ℕ) (result : ℕ) (added_number : ℕ) 
  (h1 : original_number = 7)
  (h2 : 3 * (2 * original_number + added_number) = result)
  (h3 : result = 69) :
  added_number = 9 :=
by
  sorry

end added_number_after_doubling_l292_292873


namespace range_of_m_l292_292313

noncomputable def problem (x m : ℝ) (p q : Prop) : Prop :=
  (¬ p → ¬ q) ∧ (¬ q → ¬ p → False) ∧ (p ↔ |1 - (x - 1) / 3| ≤ 2) ∧ 
  (q ↔ x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0)

theorem range_of_m (m : ℝ) (x : ℝ) (p q : Prop) 
  (h : problem x m p q) : m ≥ 9 :=
sorry

end range_of_m_l292_292313


namespace payment_for_30_kilograms_l292_292864

-- Define the price calculation based on quantity x
def payment_amount (x : ℕ) : ℕ :=
  if x ≤ 10 then 20 * x
  else 16 * x + 40

-- Prove that for x = 30, the payment amount y equals 520
theorem payment_for_30_kilograms : payment_amount 30 = 520 := by
  sorry

end payment_for_30_kilograms_l292_292864


namespace math_problem_l292_292002

-- Condition 1: The solution set of the inequality \(\frac{x-2}{ax+b} > 0\) is \((-1,2)\)
def solution_set_condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x > -1 ∧ x < 2) ↔ ((x - 2) * (a * x + b) > 0)

-- Condition 2: \(m\) is the geometric mean of \(a\) and \(b\)
def geometric_mean_condition (a b m : ℝ) : Prop :=
  a * b = m^2

-- The mathematical statement to prove: \(\frac{3m^{2}a}{a^{3}+2b^{3}} = 1\)
theorem math_problem (a b m : ℝ) (h1 : solution_set_condition a b) (h2 : geometric_mean_condition a b m) :
  3 * m^2 * a / (a^3 + 2 * b^3) = 1 :=
sorry

end math_problem_l292_292002


namespace simplify_fraction_l292_292521

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end simplify_fraction_l292_292521


namespace expected_value_bound_l292_292191

noncomputable theory

variables {R : ℝ × ℝ → ℝ} {X Y : ℝ → ℝ} [measure_space ℝ]

/-- R is a symmetric non-negative definite function on ℝ² -/
def symmetric_nonneg_definite (R : ℝ × ℝ → ℝ) : Prop :=
∀ x y : ℝ, R (x, y) = R (y, x) ∧ R (x, x) ≥ 0

/-- X is a random variable such that E[√R(X, X)] < ∞ -/
def random_variable_cond (R : ℝ × ℝ → ℝ) (X : ℝ → ℝ) [measure_space ℝ] : Prop :=
𝔼 (λ ω, (R ((X ω), (X ω))).sqrt) < ∞

/-- Y is an independent copy of the variable X -/
def independent_copy (X Y : ℝ → ℝ) [measure_space ℝ] : Prop :=
independent X Y

/-- Main Theorem -/
theorem expected_value_bound (R : ℝ × ℝ → ℝ) (X Y : ℝ → ℝ) [measure_space ℝ] 
  (h1 : symmetric_nonneg_definite R) 
  (h2 : random_variable_cond R X) 
  (h3 : independent_copy X Y) : 
  0 ≤ 𝔼 (λ ω, R ((X ω), (Y ω))) ∧ 𝔼 (λ ω, R ((X ω), (Y ω))) < ∞ := 
sorry

end expected_value_bound_l292_292191


namespace anne_cleans_in_12_hours_l292_292754

theorem anne_cleans_in_12_hours (B A C : ℝ) (h1 : B + A + C = 1/4)
    (h2 : B + 2 * A + 3 * C = 1/3) (h3 : B + C = 1/6) : 1 / A = 12 :=
by
    sorry

end anne_cleans_in_12_hours_l292_292754


namespace find_A_l292_292069

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 :=
by
  sorry

end find_A_l292_292069


namespace probability_two_white_balls_sequential_l292_292863

theorem probability_two_white_balls_sequential :
  let total_balls := 15 in
  let white_balls := 7 in
  let black_balls := 8 in
  let first_white_prob := (white_balls:ℝ) / (total_balls:ℝ) in
  let second_white_prob_given_first_white := (white_balls - 1)/(total_balls - 1:ℝ) in
  first_white_prob * second_white_prob_given_first_white = 1 / 5 :=
by
  sorry

end probability_two_white_balls_sequential_l292_292863


namespace alix_has_15_more_chocolates_than_nick_l292_292197

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l292_292197


namespace people_to_left_of_kolya_l292_292982

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l292_292982


namespace investment_Y_l292_292265

theorem investment_Y
  (X_investment : ℝ)
  (Y_investment : ℝ)
  (Z_investment : ℝ)
  (X_months : ℝ)
  (Y_months : ℝ)
  (Z_months : ℝ)
  (total_profit : ℝ)
  (Z_profit_share : ℝ)
  (h1 : X_investment = 36000)
  (h2 : Z_investment = 48000)
  (h3 : X_months = 12)
  (h4 : Y_months = 12)
  (h5 : Z_months = 8)
  (h6 : total_profit = 13970)
  (h7 : Z_profit_share = 4064) :
  Y_investment = 75000 := by
  -- Proof omitted
  sorry

end investment_Y_l292_292265


namespace sqrt_expression_equality_l292_292757

theorem sqrt_expression_equality : real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end sqrt_expression_equality_l292_292757


namespace count_valid_pairs_l292_292614

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l292_292614


namespace sequence_contains_30_l292_292940

theorem sequence_contains_30 :
  ∃ n : ℕ, n * (n + 1) = 30 :=
sorry

end sequence_contains_30_l292_292940


namespace yellow_flower_count_l292_292807

theorem yellow_flower_count :
  ∀ (total_flower_count green_flower_count : ℕ)
    (red_flower_factor blue_flower_percentage : ℕ),
  total_flower_count = 96 →
  green_flower_count = 9 →
  red_flower_factor = 3 →
  blue_flower_percentage = 50 →
  let red_flower_count := red_flower_factor * green_flower_count in
  let blue_flower_count := (blue_flower_percentage * total_flower_count) / 100 in
  let yellow_flower_count := total_flower_count - blue_flower_count - red_flower_count - green_flower_count in
  yellow_flower_count = 12 :=
by
  intros total_flower_count green_flower_count red_flower_factor blue_flower_percentage
  assume h1 h2 h3 h4
  let red_flower_count := red_flower_factor * green_flower_count
  let blue_flower_count := (blue_flower_percentage * total_flower_count) / 100
  let yellow_flower_count := total_flower_count - blue_flower_count - red_flower_count - green_flower_count
  show yellow_flower_count = 12 from sorry

end yellow_flower_count_l292_292807


namespace range_of_a_minus_b_l292_292799

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 2) (h₃ : -2 < b) (h₄ : b < 1) :
  -2 < a - b ∧ a - b < 4 :=
by
  sorry

end range_of_a_minus_b_l292_292799


namespace largest_n_with_100_trailing_zeros_l292_292304

def trailing_zeros_factorial (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + trailing_zeros_factorial (n / 5)

theorem largest_n_with_100_trailing_zeros :
  ∃ (n : ℕ), trailing_zeros_factorial n = 100 ∧ ∀ (m : ℕ), (trailing_zeros_factorial m = 100 → m ≤ 409) :=
by
  sorry

end largest_n_with_100_trailing_zeros_l292_292304


namespace Z_is_1_5_decades_younger_l292_292242

theorem Z_is_1_5_decades_younger (X Y Z : ℝ) (h : X + Y = Y + Z + 15) : (X - Z) / 10 = 1.5 :=
by
  sorry

end Z_is_1_5_decades_younger_l292_292242


namespace simplify_fraction_l292_292367

theorem simplify_fraction :
  (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end simplify_fraction_l292_292367


namespace mia_socks_problem_l292_292509

theorem mia_socks_problem (x y z w : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hw : 1 ≤ w)
  (h1 : x + y + z + w = 16) (h2 : x + 2*y + 3*z + 4*w = 36) : x = 3 :=
sorry

end mia_socks_problem_l292_292509


namespace original_number_l292_292346

/-- Proof that the original three-digit number abc equals 118 under the given conditions. -/
theorem original_number (N : ℕ) (hN : N = 4332) (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 118) :
  100 * a + 10 * b + c = 118 :=
by
  sorry

end original_number_l292_292346


namespace number_of_students_l292_292845

-- Define the conditions as hypotheses
def ordered_apples : ℕ := 6 + 15   -- 21 apples ordered
def extra_apples : ℕ := 16         -- 16 extra apples after distribution

-- Define the main theorem statement to prove S = 21
theorem number_of_students (S : ℕ) (H1 : ordered_apples = 21) (H2 : extra_apples = 16) : S = 21 := 
by
  sorry

end number_of_students_l292_292845


namespace minimally_intersecting_triples_modulo_1000_eq_344_l292_292959

def minimally_intersecting_triples_count_modulo : ℕ :=
  let total_count := 57344
  total_count % 1000

theorem minimally_intersecting_triples_modulo_1000_eq_344 :
  minimally_intersecting_triples_count_modulo = 344 := by
  sorry

end minimally_intersecting_triples_modulo_1000_eq_344_l292_292959


namespace nest_building_twig_count_l292_292274

theorem nest_building_twig_count
    (total_twigs_to_weave : ℕ)
    (found_twigs : ℕ)
    (remaining_twigs : ℕ)
    (n : ℕ)
    (x : ℕ)
    (h1 : total_twigs_to_weave = 12 * x)
    (h2 : found_twigs = (total_twigs_to_weave) / 3)
    (h3 : remaining_twigs = 48)
    (h4 : found_twigs + remaining_twigs = total_twigs_to_weave) :
    x = 18 := 
by
  sorry

end nest_building_twig_count_l292_292274


namespace sequences_properties_l292_292331

-- Definition of sequences and their properties
variable {n : ℕ}

noncomputable def S (n : ℕ) : ℕ := n^2 - n
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else 2 * n - 2
noncomputable def b (n : ℕ) : ℕ := 3^(n-1)
noncomputable def c (n : ℕ) : ℕ := (2 * (n - 1)) / 3^(n - 1)
noncomputable def T (n : ℕ) : ℕ := 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))

-- Main theorem
theorem sequences_properties (n : ℕ) (hn : n > 0) :
  S n = n^2 - n ∧
  (∀ n, a n = if n = 1 then 0 else 2 * n - 2) ∧
  (∀ n, b n = 3^(n-1)) ∧
  (∀ n, T n = 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))) :=
by sorry

end sequences_properties_l292_292331


namespace lower_upper_bound_f_l292_292052

-- definition of the function f(n, d) as given in the problem
def func_f (n : ℕ) (d : ℕ) : ℕ :=
  -- placeholder definition; actual definition would rely on the described properties
  sorry

theorem lower_upper_bound_f (n d : ℕ) (hn : 0 < n) (hd : 0 < d) :
  (n-1) * 2^d + 1 ≤ func_f n d ∧ func_f n d ≤ (n-1) * n^d + 1 :=
by
  sorry

end lower_upper_bound_f_l292_292052


namespace num_pairs_nat_nums_eq_l292_292622

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l292_292622


namespace square_area_l292_292752

/- Given: 
    1. The area of the isosceles right triangle ΔAEF is 1 cm².
    2. The area of the rectangle EFGH is 10 cm².
- To prove: 
    The area of the square ABCD is 24.5 cm².
-/

theorem square_area
  (h1 : ∃ a : ℝ, (0 < a) ∧ (a * a / 2 = 1))  -- Area of isosceles right triangle ΔAEF is 1 cm²
  (h2 : ∃ w l : ℝ, (w = 2) ∧ (l * w = 10))  -- Area of rectangle EFGH is 10 cm²
  : ∃ s : ℝ, (s * s = 24.5) := -- Area of the square ABCD is 24.5 cm²
sorry

end square_area_l292_292752


namespace pet_food_cost_is_correct_l292_292349

-- Define the given conditions
def rabbit_toy_cost := 6.51
def cage_cost := 12.51
def total_cost := 24.81
def found_dollar := 1.00

-- Define the cost of pet food
def pet_food_cost := total_cost - (rabbit_toy_cost + cage_cost) + found_dollar

-- The statement to prove
theorem pet_food_cost_is_correct : pet_food_cost = 6.79 :=
by
  -- proof steps here
  sorry

end pet_food_cost_is_correct_l292_292349


namespace multiple_statements_l292_292215

theorem multiple_statements (c d : ℤ)
  (hc4 : ∃ k : ℤ, c = 4 * k)
  (hd8 : ∃ k : ℤ, d = 8 * k) :
  (∃ k : ℤ, d = 4 * k) ∧
  (∃ k : ℤ, c + d = 4 * k) ∧
  (∃ k : ℤ, c + d = 2 * k) :=
by
  sorry

end multiple_statements_l292_292215


namespace opposite_of_neg_one_div_2023_l292_292236

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l292_292236


namespace park_area_l292_292698

-- Definitions for the conditions
def length (breadth : ℕ) : ℕ := 4 * breadth
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the proof problem
theorem park_area (breadth : ℕ) (h1 : perimeter (length breadth) breadth = 1600) : 
  let len := length breadth
  len * breadth = 102400 := 
by 
  sorry

end park_area_l292_292698


namespace relationship_between_A_B_C_l292_292928

-- Definitions based on the problem conditions
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- Proof statement: Prove the specified relationship
theorem relationship_between_A_B_C : B ∪ C = C := by
  sorry

end relationship_between_A_B_C_l292_292928


namespace find_the_number_l292_292834

-- Define the variables and conditions
variable (x z : ℝ)
variable (the_number : ℝ)

-- Condition: given that x = 1
axiom h1 : x = 1

-- Condition: given the equation
axiom h2 : 14 * (-x + z) + 18 = -14 * (x - z) - the_number

-- The theorem to prove
theorem find_the_number : the_number = -4 :=
by
  sorry

end find_the_number_l292_292834


namespace bus_stops_per_hour_l292_292722

theorem bus_stops_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h₁ : speed_without_stoppages = 50)
  (h₂ : speed_with_stoppages = 40) :
  ∃ (minutes_stopped : ℝ), minutes_stopped = 12 :=
by
  sorry

end bus_stops_per_hour_l292_292722


namespace equal_triples_l292_292947

theorem equal_triples (a b c x : ℝ) (h_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : (xb + (1 - x) * c) / a = (x * c + (1 - x) * a) / b ∧ 
          (x * c + (1 - x) * a) / b = (x * a + (1 - x) * b) / c) : a = b ∧ b = c := by
  sorry

end equal_triples_l292_292947


namespace opposite_of_neg_frac_l292_292230

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l292_292230


namespace brinley_animals_count_l292_292121

theorem brinley_animals_count :
  let snakes := 100
  let arctic_foxes := 80
  let leopards := 20
  let bee_eaters := 10 * ((snakes / 2) + (2 * leopards))
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 481340 := by
  sorry

end brinley_animals_count_l292_292121


namespace aquatic_reserve_total_fishes_l292_292882

-- Define the number of bodies of water
def bodies_of_water : ℕ := 6

-- Define the number of fishes per body of water
def fishes_per_body : ℕ := 175

-- Define the total number of fishes
def total_fishes : ℕ := bodies_of_water * fishes_per_body

theorem aquatic_reserve_total_fishes : bodies_of_water * fishes_per_body = 1050 := by
  -- The proof is omitted.
  sorry

end aquatic_reserve_total_fishes_l292_292882


namespace max_value_of_3x_plus_4y_l292_292459

theorem max_value_of_3x_plus_4y (x y : ℝ) (h : x^2 + y^2 = 10) : 
  ∃ z, z = 5 * Real.sqrt 10 ∧ z = 3 * x + 4 * y :=
by
  sorry

end max_value_of_3x_plus_4y_l292_292459


namespace maximize_rectangle_area_l292_292356

theorem maximize_rectangle_area (l w : ℝ) (h : l + w ≥ 40) : l * w ≤ 400 :=
by sorry

end maximize_rectangle_area_l292_292356


namespace pascal_row_12_sum_pascal_row_12_middle_l292_292635

open Nat

/-- Definition of the sum of all numbers in a given row of Pascal's Triangle -/
def pascal_sum (n : ℕ) : ℕ :=
  2^n

/-- Definition of the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Pascal Triangle Row 12 sum -/
theorem pascal_row_12_sum : pascal_sum 12 = 4096 :=
by
  sorry

/-- Pascal Triangle Row 12 middle number -/
theorem pascal_row_12_middle : binomial 12 6 = 924 :=
by
  sorry

end pascal_row_12_sum_pascal_row_12_middle_l292_292635


namespace prob_at_least_one_red_l292_292844

-- Definitions for conditions
def probRedA : ℚ := 1/3
def probRedB : ℚ := 1/2
def probNotRedA : ℚ := 1 - probRedA
def probNotRedB : ℚ := 1 - probRedB

-- Theorem statement for the proof problem
theorem prob_at_least_one_red : 
  (1 - (probNotRedA * probNotRedB)) = 2/3 :=
by
  sorry

end prob_at_least_one_red_l292_292844


namespace toothpick_250_stage_l292_292221

-- Define the arithmetic sequence for number of toothpicks at each stage
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

-- The proof statement for the 250th stage
theorem toothpick_250_stage : toothpicks 250 = 1001 :=
  by
  sorry

end toothpick_250_stage_l292_292221


namespace ab_eq_one_l292_292935

theorem ab_eq_one (a b : ℝ) (h1 : a ≠ b) (h2 : abs (Real.log a) = abs (Real.log b)) : a * b = 1 := sorry

end ab_eq_one_l292_292935


namespace cabin_charges_per_night_l292_292122

theorem cabin_charges_per_night 
  (total_lodging_cost : ℕ)
  (hostel_cost_per_night : ℕ)
  (hostel_days : ℕ)
  (total_cabin_days : ℕ)
  (friends_sharing_expenses : ℕ)
  (jimmy_lodging_expense : ℕ) 
  (total_cost_paid_by_jimmy : ℕ) :
  total_lodging_cost = total_cost_paid_by_jimmy →
  hostel_cost_per_night = 15 →
  hostel_days = 3 →
  total_cabin_days = 2 →
  friends_sharing_expenses = 3 →
  jimmy_lodging_expense = 75 →
  ∃ cabin_cost_per_night, cabin_cost_per_night = 45 :=
by
  sorry

end cabin_charges_per_night_l292_292122


namespace correct_calculation_result_l292_292471

theorem correct_calculation_result (n : ℤ) (h1 : n - 59 = 43) : n - 46 = 56 :=
by {
  sorry -- Proof is omitted
}

end correct_calculation_result_l292_292471


namespace totalCost_l292_292359
-- Importing the necessary library

-- Defining the conditions
def numberOfHotDogs : Nat := 6
def costPerHotDog : Nat := 50

-- Proving the total cost
theorem totalCost : numberOfHotDogs * costPerHotDog = 300 := by
  sorry

end totalCost_l292_292359


namespace adam_earnings_l292_292728

theorem adam_earnings
  (earn_per_lawn : ℕ) (total_lawns : ℕ) (forgot_lawns : ℕ)
  (h1 : earn_per_lawn = 9) (h2 : total_lawns = 12) (h3 : forgot_lawns = 8) :
  (total_lawns - forgot_lawns) * earn_per_lawn = 36 :=
by
  sorry

end adam_earnings_l292_292728


namespace binary_to_decimal_l292_292862

theorem binary_to_decimal :
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 + 1 * 2^6 + 0 * 2^7 + 1 * 2^8) = 379 := 
by
  sorry

end binary_to_decimal_l292_292862


namespace intersection_set_l292_292008

def M : Set ℤ := {1, 2, 3, 5, 7}
def N : Set ℤ := {x | ∃ k ∈ M, x = 2 * k - 1}
def I : Set ℤ := {1, 3, 5}

theorem intersection_set :
  M ∩ N = I :=
by sorry

end intersection_set_l292_292008


namespace field_width_l292_292694

variable width : ℚ -- Define a variable width of type rational

-- Define the conditions
def length_eq_24 : Prop := 24 = 2 * width - 3

-- State the theorem to prove the width is 13.5 meters
theorem field_width :
  length_eq_24 → width = 13.5 :=
by
  intro h,
  -- Proof can be filled out here. For now, we use sorry to skip it
  sorry

end field_width_l292_292694


namespace find_p_plus_s_l292_292190

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem find_p_plus_s (p q r s : ℝ) (h : p * q * r * s ≠ 0) 
  (hg : ∀ x : ℝ, g p q r s (g p q r s x) = x) : p + s = 0 := 
by 
  sorry

end find_p_plus_s_l292_292190


namespace truncated_pyramid_volume_ratio_l292_292640

/-
Statement: Given a truncated triangular pyramid with a plane drawn through a side of the upper base parallel to the opposite lateral edge,
and the corresponding sides of the bases in the ratio 1:2, prove that the volume of the truncated pyramid is divided in the ratio 3:4.
-/

theorem truncated_pyramid_volume_ratio (S1 S2 h : ℝ) 
  (h_ratio : S1 = 4 * S2) :
  (h * S2) / ((7 * h * S2) / 3 - h * S2) = 3 / 4 :=
by
  sorry

end truncated_pyramid_volume_ratio_l292_292640


namespace power_sum_result_l292_292770

theorem power_sum_result : (64 ^ (-1/3 : ℝ)) + (81 ^ (-1/4 : ℝ)) = (7 / 12 : ℝ) :=
by
  have h64 : (64 : ℝ) = 2 ^ 6 := by norm_num
  have h81 : (81 : ℝ) = 3 ^ 4 := by norm_num
  sorry

end power_sum_result_l292_292770


namespace value_of_y_l292_292024

theorem value_of_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) : y = 1 / 2 :=
by
  sorry

end value_of_y_l292_292024


namespace abs_c_five_l292_292835

theorem abs_c_five (a b c : ℤ) (h_coprime : Int.gcd a (Int.gcd b c) = 1) 
  (h1 : a = 2 * (b + c)) 
  (h2 : b = 3 * (a + c)) : 
  |c| = 5 :=
by
  sorry

end abs_c_five_l292_292835


namespace number_of_diagonals_l292_292281

-- Define a rectangular prism with its properties
structure RectangularPrism :=
  (vertices : Finset (Fin 12))
  (edges : Finset (Fin 18))

-- Define what it means for a segment to be diagonal
def is_diagonal (prism : RectangularPrism) (seg : (Fin 12) × (Fin 12)) : Prop :=
  ¬ prism.edges.contain seg ∧ 
  seg.1 ≠ seg.2

-- Define face and space diagonals separately
def face_diagonals (prism : RectangularPrism) : Nat :=
  6 * 2

def space_diagonals (prism : RectangularPrism) : Nat :=
  (12 * 2) // 2

-- Prove the total number of diagonals in a rectangular prism is 24
theorem number_of_diagonals (prism : RectangularPrism) : 
  face_diagonals prism + space_diagonals prism = 24 :=
by 
  sorry

end number_of_diagonals_l292_292281


namespace tenth_number_in_sixteenth_group_is_257_l292_292468

-- Define the general term of the sequence a_n = 2n - 3.
def a_n (n : ℕ) : ℕ := 2 * n - 3

-- Define the first number of the n-th group.
def first_number_of_group (n : ℕ) : ℕ := n^2 - n - 1

-- Define the m-th number in the n-th group.
def group_n_m (n m : ℕ) : ℕ := first_number_of_group n + (m - 1) * 2

theorem tenth_number_in_sixteenth_group_is_257 : group_n_m 16 10 = 257 := by
  sorry

end tenth_number_in_sixteenth_group_is_257_l292_292468


namespace servings_in_bottle_l292_292403

theorem servings_in_bottle (total_revenue : ℕ) (price_per_serving : ℕ) (h1 : total_revenue = 98) (h2 : price_per_serving = 8) : Nat.floor (total_revenue / price_per_serving) = 12 :=
by
  sorry

end servings_in_bottle_l292_292403


namespace find_q_l292_292603

noncomputable def Sn (n : ℕ) (d : ℚ) : ℚ :=
  d^2 * (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def Tn (n : ℕ) (d : ℚ) (q : ℚ) : ℚ :=
  d^2 * (1 - q^n) / (1 - q)

theorem find_q (d : ℚ) (q : ℚ) (hd : d ≠ 0) (hq : 0 < q ∧ q < 1) :
  Sn 3 d / Tn 3 d q = 14 → q = 1 / 2 :=
by
  sorry

end find_q_l292_292603


namespace robert_elizabeth_age_difference_l292_292362

theorem robert_elizabeth_age_difference 
  (patrick_age_1_5_times_robert : ∀ (robert_age : ℝ), ∃ (patrick_age : ℝ), patrick_age = 1.5 * robert_age)
  (elizabeth_born_after_richard : ∀ (richard_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = richard_age - 7 / 12)
  (elizabeth_younger_by_4_5_years : ∀ (patrick_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = patrick_age - 4.5)
  (robert_will_be_30_3_after_2_5_years : ∃ (robert_age_current : ℝ), robert_age_current = 30.3 - 2.5) :
  ∃ (years : ℤ) (months : ℤ), years = 9 ∧ months = 4 := by
  sorry

end robert_elizabeth_age_difference_l292_292362


namespace custom_op_difference_l292_292950

def custom_op (x y : ℕ) : ℕ := x * y - (x + y)

theorem custom_op_difference : custom_op 7 4 - custom_op 4 7 = 0 :=
by
  sorry

end custom_op_difference_l292_292950


namespace norm_squared_sum_l292_292820

variables (p q : ℝ × ℝ)
def n : ℝ × ℝ := (4, -2)
variables (h_midpoint : n = ((p.1 + q.1) / 2, (p.2 + q.2) / 2))
variables (h_dot_product : p.1 * q.1 + p.2 * q.2 = 12)

theorem norm_squared_sum : (p.1 ^ 2 + p.2 ^ 2) + (q.1 ^ 2 + q.2 ^ 2) = 56 :=
by
  sorry

end norm_squared_sum_l292_292820


namespace painting_methods_correct_l292_292769

noncomputable def num_painting_methods : ℕ :=
  sorry 

theorem painting_methods_correct :
  num_painting_methods = 24 :=
by
  -- proof would go here
  sorry

end painting_methods_correct_l292_292769


namespace find_number_l292_292630

def x : ℝ := 33.75

theorem find_number (x: ℝ) :
  (0.30 * x = 0.25 * 45) → x = 33.75 :=
by
  sorry

end find_number_l292_292630


namespace line_equation_l292_292457

noncomputable def P (A B C x y : ℝ) := A * x + B * y + C

theorem line_equation {A B C x₁ y₁ x₂ y₂ : ℝ} (h1 : P A B C x₁ y₁ = 0) (h2 : P A B C x₂ y₂ ≠ 0) :
    ∀ (x y : ℝ), P A B C x y - P A B C x₁ y₁ - P A B C x₂ y₂ = 0 ↔ P A B 0 x y = -P A B 0 x₂ y₂ := by
  sorry

end line_equation_l292_292457


namespace grants_test_score_l292_292011

theorem grants_test_score :
  ∀ (hunter_score : ℕ) (john_score : ℕ) (grant_score : ℕ), hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 :=
by
  intro hunter_score john_score grant_score
  intro hunter_eq john_eq grant_eq
  rw [hunter_eq, john_eq, grant_eq]
  sorry

end grants_test_score_l292_292011


namespace average_salary_rest_l292_292642

theorem average_salary_rest (number_of_workers : ℕ) 
                            (avg_salary_all : ℝ) 
                            (number_of_technicians : ℕ) 
                            (avg_salary_technicians : ℝ) 
                            (rest_workers : ℕ) 
                            (total_salary_all : ℝ) 
                            (total_salary_technicians : ℝ) 
                            (total_salary_rest : ℝ) 
                            (avg_salary_rest : ℝ) 
                            (h1 : number_of_workers = 28)
                            (h2 : avg_salary_all = 8000)
                            (h3 : number_of_technicians = 7)
                            (h4 : avg_salary_technicians = 14000)
                            (h5 : rest_workers = number_of_workers - number_of_technicians)
                            (h6 : total_salary_all = number_of_workers * avg_salary_all)
                            (h7 : total_salary_technicians = number_of_technicians * avg_salary_technicians)
                            (h8 : total_salary_rest = total_salary_all - total_salary_technicians)
                            (h9 : avg_salary_rest = total_salary_rest / rest_workers) :
  avg_salary_rest = 6000 :=
by {
  -- the proof would go here
  sorry
}

end average_salary_rest_l292_292642


namespace inequality_proof_l292_292269

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l292_292269


namespace smallest_integral_area_of_circle_l292_292217

noncomputable def A (r : ℝ) : ℝ := π * r^2

noncomputable def C (r : ℝ) : ℝ := 2 * π * r

lemma integral_area_greater_than_circumference (r : ℝ) : (π * r^2 > 2 * π * r) → r > 2 :=
begin
  intro h,
  have h1 : r * (r - 2) > 0,
  { rw mul_comm at h,
    exact (lt_div_iff (pi_pos)).mp ((div_lt_iff (pi_pos)).mpr h) },
  exact gt_of_not_le (@not_or_distrib ℝ r 0 2).mpr (not_le_of_gt h1)
end

theorem smallest_integral_area_of_circle : ∃ r : ℝ, r > 2 ∧ (28 < π * r^2 ∧ π * r^2 < 30) :=
begin
  use 3,
  split,
  { linarith },
  split,
  { linarith [pi_pos],
    norm_num,
    linarith [real.pi_pos] },
  { norm_num,
    linarith }
end

end smallest_integral_area_of_circle_l292_292217


namespace timber_logging_years_l292_292290

theorem timber_logging_years 
  (V0 : ℝ) (r : ℝ) (V : ℝ) (t : ℝ)
  (hV0 : V0 = 100000)
  (hr : r = 0.08)
  (hV : V = 400000)
  (hformula : V = V0 * (1 + r)^t)
  : t = (Real.log 4 / Real.log 1.08) :=
by
  sorry

end timber_logging_years_l292_292290


namespace polynomial_divisible_by_3_l292_292662

/--
Given q and p are integers where q is divisible by 3 and p+1 is divisible by 3,
prove that the polynomial Q(x) = x^3 - x + (p+1)x + q is divisible by 3 for any integer x.
-/
theorem polynomial_divisible_by_3 (q p : ℤ) (hq : 3 ∣ q) (hp1 : 3 ∣ (p + 1)) :
  ∀ x : ℤ, 3 ∣ (x^3 - x + (p+1) * x + q) :=
by {
  sorry
}

end polynomial_divisible_by_3_l292_292662


namespace inequality_proof_l292_292272

theorem inequality_proof (a b c d e f : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
    (hcond : abs (sqrt (a * d) - sqrt (b * c)) ≤ 1) :
    (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := 
  sorry

end inequality_proof_l292_292272


namespace find_n_l292_292714

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 15 ∧ n = 4 :=
by {
    -- Let's assume there exists an integer n such that the given condition holds
    use 4,
    -- We now prove the condition and the conclusion
    split,
    -- This simplifies the left-hand side of the condition to 15, achieving the goal
    calc 
        4 + (4 + 1) + (4 + 2) = 4 + 5 + 6 : by rfl
        ... = 15 : by norm_num,
    -- The conclusion directly follows
    rfl
}

end find_n_l292_292714


namespace license_plate_count_l292_292945

theorem license_plate_count : 
  let vowels := 5
  let consonants := 21
  let digits := 10
  21 * 21 * 5 * 5 * 10 = 110250 := 
by 
  sorry

end license_plate_count_l292_292945


namespace find_x_l292_292612

structure Vector2D where
  x : ℝ
  y : ℝ

def vecAdd (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

def vecScale (c : ℝ) (v : Vector2D) : Vector2D :=
  ⟨c * v.x, c * v.y⟩

def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem find_x (x : ℝ)
  (a : Vector2D := ⟨1, 2⟩)
  (b : Vector2D := ⟨x, 1⟩)
  (h : areParallel (vecAdd a (vecScale 2 b)) (vecAdd (vecScale 2 a) (vecScale (-2) b))) :
  x = 1 / 2 :=
by
  sorry

end find_x_l292_292612


namespace condition_necessary_but_not_sufficient_l292_292932

-- Definitions based on given conditions
variables {a b c : ℝ}

-- The condition that needs to be qualified
def condition (a b c : ℝ) := a > 0 ∧ b^2 - 4 * a * c < 0

-- The statement to be verified
def statement (a b c : ℝ) := ∀ x : ℝ, a * x^2 + b * x + c > 0

-- Prove that the condition is a necessary but not sufficient condition for the statement
theorem condition_necessary_but_not_sufficient :
  condition a b c → (¬ (condition a b c ↔ statement a b c)) :=
by
  sorry

end condition_necessary_but_not_sufficient_l292_292932


namespace find_third_number_l292_292688

theorem find_third_number (x : ℝ) 
  (h : (20 + 40 + x) / 3 = (10 + 50 + 45) / 3 + 5) : x = 60 :=
sorry

end find_third_number_l292_292688


namespace percentage_calculation_l292_292717

variable (x : Real)
variable (hx : x > 0)

theorem percentage_calculation : 
  ∃ p : Real, p = (0.18 * x) / (x + 20) * 100 :=
sorry

end percentage_calculation_l292_292717


namespace min_value_of_g_function_l292_292920

noncomputable def g (x : Real) := x + (x + 1) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem min_value_of_g_function : ∀ x : ℝ, x > 0 → g x ≥ 3 := sorry

end min_value_of_g_function_l292_292920


namespace g_g_x_has_exactly_4_distinct_real_roots_l292_292961

noncomputable def g (d x : ℝ) : ℝ := x^2 + 8*x + d

theorem g_g_x_has_exactly_4_distinct_real_roots (d : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ g d (g d x1) = 0 ∧ g d (g d x2) = 0 ∧ g d (g d x3) = 0 ∧ g d (g d x4) = 0) ↔ d < 4 := by {
  sorry
}

end g_g_x_has_exactly_4_distinct_real_roots_l292_292961


namespace total_weight_of_peppers_l292_292157

def green_peppers := 0.3333333333333333
def red_peppers := 0.4444444444444444
def yellow_peppers := 0.2222222222222222
def orange_peppers := 0.7777777777777778

theorem total_weight_of_peppers :
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 :=
by
  sorry

end total_weight_of_peppers_l292_292157


namespace annie_total_blocks_l292_292750

-- Definitions of the blocks traveled in each leg of Annie's journey
def walk_to_bus_stop := 5
def ride_bus_to_train_station := 7
def train_to_friends_house := 10
def walk_to_coffee_shop := 4
def walk_back_to_friends_house := walk_to_coffee_shop

-- The total blocks considering the round trip and additional walk to/from coffee shop
def total_blocks_traveled :=
  2 * (walk_to_bus_stop + ride_bus_to_train_station + train_to_friends_house) +
  walk_to_coffee_shop + walk_back_to_friends_house

-- Statement to prove
theorem annie_total_blocks : total_blocks_traveled = 52 :=
by
  sorry

end annie_total_blocks_l292_292750


namespace range_of_m_l292_292430

-- Define the function g as an even function on the interval [-2, 2] 
-- and monotonically decreasing on [0, 2]

variable {g : ℝ → ℝ}

axiom even_g : ∀ x, g x = g (-x)
axiom mono_dec_g : ∀ {x y}, 0 ≤ x → x ≤ y → g y ≤ g x
axiom domain_g : ∀ x, -2 ≤ x ∧ x ≤ 2

theorem range_of_m (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) (h : g (1 - m) < g m) : -1 ≤ m ∧ m < 1 / 2 :=
sorry

end range_of_m_l292_292430


namespace gathering_gift_exchange_l292_292895

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l292_292895


namespace remainder_when_doubling_l292_292860

theorem remainder_when_doubling:
  ∀ (n k : ℤ), n = 30 * k + 16 → (2 * n) % 15 = 2 :=
by
  intros n k h
  sorry

end remainder_when_doubling_l292_292860


namespace sin_cos_identity_l292_292925

variable {α : ℝ}

/-- Given 1 / sin(α) + 1 / cos(α) = √3, then sin(α) * cos(α) = -1 / 3 -/
theorem sin_cos_identity (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) : 
  Real.sin α * Real.cos α = -1 / 3 := 
sorry

end sin_cos_identity_l292_292925


namespace age_sum_proof_l292_292708

noncomputable def leilei_age : ℝ := 30 -- Age of Leilei this year
noncomputable def feifei_age (R : ℝ) : ℝ := 1 / 2 * R + 12 -- Age of Feifei this year defined in terms of R

theorem age_sum_proof (R F : ℝ)
  (h1 : F = 1 / 2 * R + 12)
  (h2 : F + 1 = 2 * (R + 1) - 34) :
  R + F = 57 :=
by 
  -- Proof steps would go here
  sorry

end age_sum_proof_l292_292708


namespace simplify_fraction_l292_292054

theorem simplify_fraction : (270 / 18) * (7 / 140) * (9 / 4) = 27 / 16 :=
by sorry

end simplify_fraction_l292_292054


namespace smallest_b_value_l292_292366

noncomputable def smallest_possible_value_of_b : ℝ :=
  (3 + Real.sqrt 5) / 2

theorem smallest_b_value
  (a b : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : (1/b) + (1/a) ≤ 1) :
  b = smallest_possible_value_of_b :=
sorry

end smallest_b_value_l292_292366


namespace correct_equation_for_gift_exchanges_l292_292892

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l292_292892


namespace m_range_l292_292610

theorem m_range (m : ℝ) :
  (∀ x : ℝ, 1 < x → 2 * x + m + 2 / (x - 1) > 0) ↔ m > -6 :=
by
  -- The proof will be provided later
  sorry

end m_range_l292_292610


namespace evaluate_expression_l292_292913

theorem evaluate_expression : 12 * ((1/3 : ℚ) + (1/4) + (1/6))⁻¹ = 16 := 
by 
  sorry

end evaluate_expression_l292_292913


namespace total_marks_of_all_candidates_l292_292060

theorem total_marks_of_all_candidates 
  (average_marks : ℕ) 
  (num_candidates : ℕ) 
  (average : average_marks = 35) 
  (candidates : num_candidates = 120) : 
  average_marks * num_candidates = 4200 :=
by
  -- The proof will be written here
  sorry

end total_marks_of_all_candidates_l292_292060


namespace product_implication_l292_292470

theorem product_implication (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a * b > 1) : a > 1 ∨ b > 1 :=
sorry

end product_implication_l292_292470


namespace train_pass_jogger_in_41_seconds_l292_292552

-- Definitions based on conditions
def jogger_speed_kmh := 9 -- in km/hr
def train_speed_kmh := 45 -- in km/hr
def initial_distance_jogger := 200 -- in meters
def train_length := 210 -- in meters

-- Converting speeds from km/hr to m/s
def kmh_to_ms (kmh: ℕ) : ℕ := (kmh * 1000) / 3600

def jogger_speed_ms := kmh_to_ms jogger_speed_kmh -- in m/s
def train_speed_ms := kmh_to_ms train_speed_kmh -- in m/s

-- Relative speed of the train with respect to the jogger
def relative_speed := train_speed_ms - jogger_speed_ms -- in m/s

-- Total distance to be covered by the train to pass the jogger
def total_distance := initial_distance_jogger + train_length -- in meters

-- Time taken to pass the jogger
def time_to_pass (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem train_pass_jogger_in_41_seconds : time_to_pass total_distance relative_speed = 41 :=
by
  sorry

end train_pass_jogger_in_41_seconds_l292_292552


namespace vanessa_savings_weeks_l292_292077

-- Define the conditions as constants
def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def weekly_arcade_spending : ℕ := 15
def weekly_snack_spending : ℕ := 5

-- The theorem statement based on the problem
theorem vanessa_savings_weeks : 
  ∃ (n : ℕ), (n * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings) ≥ dress_cost ∧ 
             (n - 1) * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings < dress_cost := by
  sorry

end vanessa_savings_weeks_l292_292077


namespace simplify_expression_l292_292833

variable {x : ℝ}

theorem simplify_expression : 8 * x - 3 + 2 * x - 7 + 4 * x + 15 = 14 * x + 5 :=
by
  sorry

end simplify_expression_l292_292833


namespace projection_onto_plane_l292_292921

def projection (v n : Matrix (Fin 3) (Fin 1) ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  v - ((v.dot_product n) / (n.dot_product n)) • n

theorem projection_onto_plane :
  let v := ![4, -1, 2]
  let n := ![1, 2, -3]
  let p := ![30/7, -3/7, 8/7]
  projection v n = p :=
  by
    sorry

end projection_onto_plane_l292_292921


namespace find_a_plus_b_l292_292068

theorem find_a_plus_b (a b : ℤ) (h1 : 2 * a = 0) (h2 : a^2 - b = 25) : a + b = -25 :=
by 
  sorry

end find_a_plus_b_l292_292068


namespace number_of_increasing_digits_l292_292596

theorem number_of_increasing_digits : 
  (∑ k in finset.range 10, if 2 ≤ k then nat.choose 9 k else 0) = 502 :=
by
  sorry

end number_of_increasing_digits_l292_292596


namespace somu_present_age_l292_292524

def Somu_Age_Problem (S F : ℕ) : Prop := 
  S = F / 3 ∧ S - 6 = (F - 6) / 5

theorem somu_present_age (S F : ℕ) 
  (h : Somu_Age_Problem S F) : S = 12 := 
by
  sorry

end somu_present_age_l292_292524


namespace car_speed_l292_292730

theorem car_speed 
  (d : ℝ) (t : ℝ) 
  (hd : d = 520) (ht : t = 8) : 
  d / t = 65 := 
by 
  sorry

end car_speed_l292_292730


namespace supreme_sports_package_channels_l292_292505

theorem supreme_sports_package_channels (c_start : ℕ) (c_removed1 : ℕ) (c_added1 : ℕ)
                                         (c_removed2 : ℕ) (c_added2 : ℕ)
                                         (c_final : ℕ)
                                         (net1 : ℕ) (net2 : ℕ) (c_mid : ℕ) :
  c_start = 150 →
  c_removed1 = 20 →
  c_added1 = 12 →
  c_removed2 = 10 →
  c_added2 = 8 →
  c_final = 147 →
  net1 = c_removed1 - c_added1 →
  net2 = c_removed2 - c_added2 →
  c_mid = c_start - net1 - net2 →
  c_final - c_mid = 7 :=
by
  intros
  sorry

end supreme_sports_package_channels_l292_292505


namespace largest_of_five_consecutive_non_primes_under_40_l292_292131

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n 

theorem largest_of_five_consecutive_non_primes_under_40 :
  ∃ x, (x > 9) ∧ (x + 4 < 40) ∧ 
       (¬ is_prime x) ∧
       (¬ is_prime (x + 1)) ∧
       (¬ is_prime (x + 2)) ∧
       (¬ is_prime (x + 3)) ∧
       (¬ is_prime (x + 4)) ∧
       (x + 4 = 36) :=
sorry

end largest_of_five_consecutive_non_primes_under_40_l292_292131


namespace cyclic_quadrilateral_XF_XG_l292_292829

/-- 
Given:
- A cyclic quadrilateral ABCD inscribed in a circle O,
- Side lengths: AB = 4, BC = 3, CD = 7, DA = 9,
- Points X and Y such that DX/BD = 1/3 and BY/BD = 1/4,
- E is the intersection of line AX and the line through Y parallel to BC,
- F is the intersection of line CX and the line through E parallel to AB,
- G is the other intersection of line CX with circle O,
Prove:
- XF * XG = 36.5.
-/
theorem cyclic_quadrilateral_XF_XG (AB BC CD DA DX BD BY : ℝ) 
  (h_AB : AB = 4) (h_BC : BC = 3) (h_CD : CD = 7) (h_DA : DA = 9)
  (h_ratio1 : DX / BD = 1 / 3) (h_ratio2 : BY / BD = 1 / 4)
  (BD := Real.sqrt 73) :
  ∃ (XF XG : ℝ), XF * XG = 36.5 :=
by
  sorry

end cyclic_quadrilateral_XF_XG_l292_292829


namespace coefficient_of_x9_in_polynomial_is_240_l292_292219

-- Define the polynomial (1 + 3x - 2x^2)^5
noncomputable def polynomial : ℕ → ℝ := (fun x => (1 + 3*x - 2*x^2)^5)

-- Define the term we are interested in (x^9)
def term := 9

-- The coefficient we want to prove
def coefficient := 240

-- The goal is to prove that the coefficient of x^9 in the expansion of (1 + 3x - 2x^2)^5 is 240
theorem coefficient_of_x9_in_polynomial_is_240 : polynomial 9 = coefficient := sorry

end coefficient_of_x9_in_polynomial_is_240_l292_292219


namespace person_a_work_days_l292_292266

theorem person_a_work_days (x : ℝ) (h1 : 1 / 6 + 1 / x = 1 / 3.75) : x = 10 := 
sorry

end person_a_work_days_l292_292266


namespace find_n_l292_292715

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 15 ∧ n = 4 :=
by {
    -- Let's assume there exists an integer n such that the given condition holds
    use 4,
    -- We now prove the condition and the conclusion
    split,
    -- This simplifies the left-hand side of the condition to 15, achieving the goal
    calc 
        4 + (4 + 1) + (4 + 2) = 4 + 5 + 6 : by rfl
        ... = 15 : by norm_num,
    -- The conclusion directly follows
    rfl
}

end find_n_l292_292715


namespace minimum_a_l292_292192

noncomputable def f (x a : ℝ) := Real.exp x * (x^3 - 3 * x + 3) - a * Real.exp x - x

theorem minimum_a (a : ℝ) : (∃ x, x ≥ -2 ∧ f x a ≤ 0) ↔ a ≥ 1 - 1 / Real.exp 1 :=
by
  sorry

end minimum_a_l292_292192


namespace quadratic_b_value_l292_292874
open Real

theorem quadratic_b_value (b n : ℝ) 
  (h1: b < 0) 
  (h2: ∀ x, x^2 + b * x + (1 / 4) = (x + n)^2 + (1 / 16)) :
  b = - (sqrt 3 / 2) :=
by
  -- sorry is used to skip the proof
  sorry

end quadratic_b_value_l292_292874


namespace Mr_Kishore_saved_10_percent_l292_292421

-- Define the costs and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 6100
def savings : ℕ := 2400

-- Define the total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage saved
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- The statement to prove
theorem Mr_Kishore_saved_10_percent : percentage_saved = 10 := by
  sorry

end Mr_Kishore_saved_10_percent_l292_292421


namespace average_percentage_decrease_l292_292523

theorem average_percentage_decrease :
  ∃ (x : ℝ), (5000 * (1 - x / 100)^3 = 2560) ∧ x = 20 :=
by
  sorry

end average_percentage_decrease_l292_292523


namespace games_attended_l292_292501

theorem games_attended (games_this_month games_last_month games_next_month total_games : ℕ) 
  (h1 : games_this_month = 11) 
  (h2 : games_last_month = 17) 
  (h3 : games_next_month = 16) : 
  total_games = games_this_month + games_last_month + games_next_month → 
  total_games = 44 :=
by
  sorry

end games_attended_l292_292501


namespace angle_between_sum_is_pi_over_6_l292_292156

open Real EuclideanSpace

noncomputable def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_u := sqrt (u.1^2 + u.2^2)
  let norm_v := sqrt (v.1^2 + v.2^2)
  arccos (dot_product / (norm_u * norm_v))

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (1/2 * cos (π / 3), 1/2 * sin (π / 3))

theorem angle_between_sum_is_pi_over_6 :
  angle_between_vectors (a.1 + 2 * b.1, a.2 + 2 * b.2) b = π / 6 :=
by
  sorry

end angle_between_sum_is_pi_over_6_l292_292156


namespace average_of_hidden_primes_l292_292580

theorem average_of_hidden_primes (p₁ p₂ : ℕ) (h₁ : Nat.Prime p₁) (h₂ : Nat.Prime p₂) (h₃ : p₁ + 37 = p₂ + 53) : 
  (p₁ + p₂) / 2 = 11 := 
by
  sorry

end average_of_hidden_primes_l292_292580


namespace marcus_dropped_8_pies_l292_292357

-- Step d): Rewrite as a Lean 4 statement
-- Define all conditions from the problem
def total_pies (pies_per_batch : ℕ) (batches : ℕ) : ℕ :=
  pies_per_batch * batches

def pies_dropped (total_pies : ℕ) (remaining_pies : ℕ) : ℕ :=
  total_pies - remaining_pies

-- Prove that Marcus dropped 8 pies
theorem marcus_dropped_8_pies : 
  total_pies 5 7 - 27 = 8 := by
  sorry

end marcus_dropped_8_pies_l292_292357


namespace prize_distribution_l292_292733

theorem prize_distribution (x y z : ℕ) (h₁ : 15000 * x + 10000 * y + 5000 * z = 1000000) (h₂ : 93 ≤ z - x) (h₃ : z - x < 96) :
  x + y + z = 147 :=
sorry

end prize_distribution_l292_292733


namespace bottles_left_l292_292429

theorem bottles_left (total_bottles : ℕ) (bottles_per_day : ℕ) (days : ℕ)
  (h_total : total_bottles = 264)
  (h_bottles_per_day : bottles_per_day = 15)
  (h_days : days = 11) :
  total_bottles - bottles_per_day * days = 99 :=
by
  sorry

end bottles_left_l292_292429


namespace range_of_a_l292_292781

-- Given conditions
def p (x : ℝ) : Prop := abs (4 - x) ≤ 6
def q (x : ℝ) (a : ℝ) : Prop := (x - 1)^2 - a^2 ≥ 0

-- The statement to prove
theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : ∀ x, ¬p x → q x a) : 
  0 < a ∧ a ≤ 3 :=
by
  sorry -- Proof placeholder

end range_of_a_l292_292781


namespace betty_honey_oats_problem_l292_292902

theorem betty_honey_oats_problem
  (o h : ℝ)
  (h_condition1 : o ≥ 8 + h / 3)
  (h_condition2 : o ≤ 3 * h) :
  h ≥ 3 :=
sorry

end betty_honey_oats_problem_l292_292902


namespace sam_time_to_cover_distance_l292_292209

/-- Define the total distance between points A and B as the sum of distances from A to C and C to B -/
def distance_A_to_C : ℕ := 600
def distance_C_to_B : ℕ := 400
def speed_sam : ℕ := 50
def distance_A_to_B : ℕ := distance_A_to_C + distance_C_to_B

theorem sam_time_to_cover_distance :
  let time := distance_A_to_B / speed_sam
  time = 20 := 
by
  sorry

end sam_time_to_cover_distance_l292_292209


namespace arithmetic_sequence_value_l292_292958

theorem arithmetic_sequence_value 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  (1 / 5 * a 4 = 1) := 
by
  sorry

end arithmetic_sequence_value_l292_292958


namespace find_a_and_b_l292_292481

noncomputable def find_ab (a b : ℝ) : Prop :=
  (3 - 2 * a + b = 0) ∧
  (27 + 6 * a + b = 0)

theorem find_a_and_b :
  ∃ (a b : ℝ), (find_ab a b) ∧ (a = -3) ∧ (b = -9) :=
by
  sorry

end find_a_and_b_l292_292481


namespace books_needed_to_buy_clarinet_l292_292685

def cost_of_clarinet : ℕ := 90
def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def halfway_loss : ℕ := (cost_of_clarinet - initial_savings) / 2

theorem books_needed_to_buy_clarinet 
    (cost_of_clarinet initial_savings price_per_book halfway_loss : ℕ)
    (initial_savings_lost : halfway_loss = (cost_of_clarinet - initial_savings) / 2) : 
    ((cost_of_clarinet - initial_savings + halfway_loss) / price_per_book) = 24 := 
sorry

end books_needed_to_buy_clarinet_l292_292685


namespace simplify_expr_l292_292053

theorem simplify_expr : ((256 : ℝ) ^ (1 / 4)) * ((144 : ℝ) ^ (1 / 2)) = 48 := by
  have h1 : (256 : ℝ) = 2^8 := by
    norm_num,
  have h2 : (144 : ℝ) = 12^2 := by
    norm_num,
  have h3 : (2^8 : ℝ) ^ (1 / 4) = 4 := by
    norm_num,
  have h4 : (12^2 : ℝ) ^ (1 / 2) = 12 := by
    norm_num,
  sorry

end simplify_expr_l292_292053


namespace fraction_of_girls_is_half_l292_292638

variables (T G B : ℝ)
def fraction_x_of_girls (x : ℝ) : Prop :=
  x * G = (1/5) * T ∧ B / G = 1.5 ∧ T = B + G

theorem fraction_of_girls_is_half (x : ℝ) (h : fraction_x_of_girls T G B x) : x = 0.5 :=
sorry

end fraction_of_girls_is_half_l292_292638


namespace number_of_people_to_the_left_of_Kolya_l292_292984

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l292_292984


namespace football_preference_related_to_gender_stratified_selection_expected_value_E_X_l292_292836

noncomputable def chi_squared (a b c d : ℕ) : ℝ :=
  let n := a + b + c + d in
  (n * ((a * d - b * c)^2 : ℝ)) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem football_preference_related_to_gender (a b c d : ℕ)
  (h_a : a = 80) (h_b : b = 40) (h_c : c = 60) (h_d : d = 60)
  (alpha : ℝ) (critical_value : ℝ) (h_alpha : alpha = 0.01) (h_cv : critical_value = 6.635) :
  chi_squared a b c d > critical_value := by
  sorry

def stratified_sample (total_likes males_likes females_likes : ℕ) (sample_size : ℕ)
  : ℕ × ℕ :=
  let total := males_likes + females_likes in
  let males_selected := (males_likes * sample_size) / total in
  let females_selected := (females_likes * sample_size) / total in
  (males_selected, females_selected)

theorem stratified_selection : stratified_sample 140 80 60 7 = (4, 3) := by
  sorry

noncomputable def E_X : ℝ :=
  (1 * (4 / 35 : ℝ)) + (2 * (18 / 35 : ℝ)) + (3 * (12 / 35 : ℝ)) + (4 * (1 / 35 : ℝ))

theorem expected_value_E_X : E_X = (16 / 7 : ℝ) := by
  sorry

end football_preference_related_to_gender_stratified_selection_expected_value_E_X_l292_292836


namespace blanket_rate_l292_292736

/-- 
A man purchased 4 blankets at Rs. 100 each, 
5 blankets at Rs. 150 each, 
and two blankets at an unknown rate x. 
If the average price of the blankets was Rs. 150, 
prove that the unknown rate x is 250. 
-/
theorem blanket_rate (x : ℝ) 
  (h1 : 4 * 100 + 5 * 150 + 2 * x = 11 * 150) : 
  x = 250 := 
sorry

end blanket_rate_l292_292736


namespace inequality_am_gm_l292_292148

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
by sorry

end inequality_am_gm_l292_292148


namespace cost_of_book_first_sold_at_loss_l292_292014

theorem cost_of_book_first_sold_at_loss (C1 C2 C3 : ℝ) (h1 : C1 + C2 + C3 = 810)
    (h2 : 0.88 * C1 = 1.18 * C2) (h3 : 0.88 * C1 = 1.27 * C3) : 
    C1 = 333.9 := 
by
  -- Conditions given
  have h4 : C2 = 0.88 * C1 / 1.18 := by sorry
  have h5 : C3 = 0.88 * C1 / 1.27 := by sorry

  -- Substituting back into the total cost equation
  have h6 : C1 + 0.88 * C1 / 1.18 + 0.88 * C1 / 1.27 = 810 := by sorry

  -- Simplifying and solving for C1
  have h7 : C1 = 333.9 := by sorry

  -- Conclusion
  exact h7

end cost_of_book_first_sold_at_loss_l292_292014


namespace peter_reads_more_books_l292_292514

-- Definitions and conditions
def total_books : ℕ := 20
def peter_percentage_read : ℕ := 40
def brother_percentage_read : ℕ := 10

def percentage_to_count (percentage : ℕ) (total : ℕ) : ℕ := (percentage * total) / 100

-- Main statement to prove
theorem peter_reads_more_books :
  percentage_to_count peter_percentage_read total_books - percentage_to_count brother_percentage_read total_books = 6 :=
by
  sorry

end peter_reads_more_books_l292_292514


namespace smallest_n_l292_292922

theorem smallest_n (n : ℕ) (h : 5 * n ≡ 850 [MOD 26]) : n = 14 :=
by
  sorry

end smallest_n_l292_292922


namespace closest_integer_to_cube_root_of_1728_l292_292263

theorem closest_integer_to_cube_root_of_1728: 
  ∃ n : ℕ, n^3 = 1728 ∧ (∀ m : ℤ, m^3 < 1728 → m < n) ∧ (∀ p : ℤ, p^3 > 1728 → p > n) :=
by
  sorry

end closest_integer_to_cube_root_of_1728_l292_292263


namespace problem_inequality_l292_292047

theorem problem_inequality 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_le_a : a ≤ 1)
  (h_pos_b : 0 < b) (h_le_b : b ≤ 1)
  (h_pos_c : 0 < c) (h_le_c : c ≤ 1)
  (h_pos_d : 0 < d) (h_le_d : d ≤ 1) :
  (1 / (a^2 + b^2 + c^2 + d^2)) ≥ (1 / 4) + (1 - a) * (1 - b) * (1 - c) * (1 - d) :=
by
  sorry

end problem_inequality_l292_292047


namespace solution_set_for_inequality_l292_292158

theorem solution_set_for_inequality (f : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_decreasing : ∀ ⦃x y⦄, 0 < x → x < y → f y < f x)
  (h_f_neg3 : f (-3) = 1) :
  { x | f x < 1 } = { x | x < -3 ∨ 3 < x } := 
by
  -- TODO: Prove this theorem
  sorry

end solution_set_for_inequality_l292_292158


namespace molecular_weight_is_171_35_l292_292546

def atomic_weight_ba : ℝ := 137.33
def atomic_weight_o : ℝ := 16.00
def atomic_weight_h : ℝ := 1.01

def molecular_weight : ℝ :=
  (1 * atomic_weight_ba) + (2 * atomic_weight_o) + (2 * atomic_weight_h)

-- The goal is to prove that the molecular weight is 171.35
theorem molecular_weight_is_171_35 : molecular_weight = 171.35 :=
by
  sorry

end molecular_weight_is_171_35_l292_292546


namespace sum_of_f_is_negative_l292_292142

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_of_f_is_negative (x₁ x₂ x₃ : ℝ)
  (h1: x₁ + x₂ < 0)
  (h2: x₂ + x₃ < 0) 
  (h3: x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := 
sorry

end sum_of_f_is_negative_l292_292142


namespace remainder_3_pow_500_mod_17_l292_292547

theorem remainder_3_pow_500_mod_17 : (3^500) % 17 = 13 := 
by
  sorry

end remainder_3_pow_500_mod_17_l292_292547


namespace ratio_of_areas_l292_292254

theorem ratio_of_areas (side_length : ℝ) (h : side_length = 6) :
  let area_triangle := (side_length^2 * Real.sqrt 3) / 4
  let area_square := side_length^2
  (area_triangle / area_square) = Real.sqrt 3 / 4 :=
by
  sorry

end ratio_of_areas_l292_292254


namespace problem_statement_l292_292584

variable {f : ℝ → ℝ}

-- Condition 1: The function f satisfies (x - 1)f'(x) ≤ 0
def cond1 (f : ℝ → ℝ) : Prop := ∀ x, (x - 1) * (deriv f x) ≤ 0

-- Condition 2: The function f satisfies f(-x) = f(2 + x)
def cond2 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f (2 + x)

theorem problem_statement (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h_cond1 : cond1 f)
  (h_cond2 : cond2 f)
  (h_dist : abs (x₁ - 1) < abs (x₂ - 1)) :
  f (2 - x₁) > f (2 - x₂) :=
sorry

end problem_statement_l292_292584


namespace sum_of_squares_l292_292837

theorem sum_of_squares (a b : ℕ) (h_side_lengths : 20^2 = a^2 + b^2) : a + b = 28 :=
sorry

end sum_of_squares_l292_292837


namespace sum_of_integers_mod_59_l292_292709

theorem sum_of_integers_mod_59 (a b c : ℕ) (h1 : a % 59 = 29) (h2 : b % 59 = 31) (h3 : c % 59 = 7)
  (h4 : a^2 % 59 = 29) (h5 : b^2 % 59 = 31) (h6 : c^2 % 59 = 7) :
  (a + b + c) % 59 = 8 :=
by
  sorry

end sum_of_integers_mod_59_l292_292709


namespace intersect_horizontal_asymptote_l292_292587

theorem intersect_horizontal_asymptote (x : ℚ) :
  let g := (3 * x^2 - 8 * x + 4) / (x^2 - 5 * x + 6) in
  g = 3 ↔ x = 2 :=
by
  sorry

end intersect_horizontal_asymptote_l292_292587


namespace polynomial_divisibility_l292_292138

theorem polynomial_divisibility (a : ℤ) : 
  (∀x : ℤ, x^2 - x + a ∣ x^13 + x + 94) → a = 2 := 
by 
  sorry

end polynomial_divisibility_l292_292138


namespace Nina_money_before_tax_l292_292044

theorem Nina_money_before_tax :
  ∃ (M P : ℝ), M = 6 * P ∧ M = 8 * 0.9 * P ∧ M = 5 :=
by 
  sorry

end Nina_money_before_tax_l292_292044


namespace roots_in_intervals_l292_292310

theorem roots_in_intervals {a b c : ℝ} (h₁ : a < b) (h₂ : b < c) :
  let f (x : ℝ) := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)
  -- statement that the roots are in the intervals (a, b) and (b, c)
  ∃ r₁ r₂, (a < r₁ ∧ r₁ < b) ∧ (b < r₂ ∧ r₂ < c) ∧ f r₁ = 0 ∧ f r₂ = 0 := 
sorry

end roots_in_intervals_l292_292310


namespace circles_tangent_l292_292843

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

theorem circles_tangent :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end circles_tangent_l292_292843


namespace abs_add_gt_abs_sub_l292_292144

variables {a b : ℝ}

theorem abs_add_gt_abs_sub (h : a * b > 0) : |a + b| > |a - b| :=
sorry

end abs_add_gt_abs_sub_l292_292144


namespace proof_problem_l292_292327

noncomputable def f (x : ℝ) : ℝ := x^2 * log x

-- Proof definitions
lemma statement_A (x : ℝ) (hx1 : x > 1) : f x > 0 :=
sorry

lemma statement_A' (x : ℝ) (hx2 : 0 < x ∧ x < 1) : f x < 0 :=
sorry

lemma statement_C : set.range f = { y | -1 / (2 * real.exp 1) <= y } :=
sorry

lemma statement_D (x : ℝ) : f x >= x - 1 :=
sorry

-- Combined statement to match the final problem request
theorem proof_problem :
  (∀ x > 1, f x > 0) ∧ (∀ x, 0 < x ∧ x < 1 → f x < 0) ∧ 
  (set.range f = { y | -1 / (2 * real.exp 1) <= y }) ∧ 
  (∀ x, f x >= x - 1) :=
by
  exact ⟨statement_A, statement_A', statement_C, statement_D⟩

end proof_problem_l292_292327


namespace Grant_score_is_100_l292_292012

/-- Definition of scores --/
def Hunter_score : ℕ := 45

def John_score (H : ℕ) : ℕ := 2 * H

def Grant_score (J : ℕ) : ℕ := J + 10

/-- Theorem to prove Grant's score --/
theorem Grant_score_is_100 : Grant_score (John_score Hunter_score) = 100 := 
  sorry

end Grant_score_is_100_l292_292012


namespace donny_paid_l292_292589

variable (total_capacity initial_fuel price_per_liter change : ℕ)

theorem donny_paid (h1 : total_capacity = 150) 
                   (h2 : initial_fuel = 38) 
                   (h3 : price_per_liter = 3) 
                   (h4 : change = 14) : 
                   (total_capacity - initial_fuel) * price_per_liter + change = 350 := 
by
  sorry

end donny_paid_l292_292589


namespace opposite_of_neg_one_over_2023_l292_292239

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l292_292239


namespace min_value_xyz_l292_292141

theorem min_value_xyz (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + y^2 + z^2 ≥ 1 / 14 := 
by
  sorry

end min_value_xyz_l292_292141


namespace alix_has_15_more_chocolates_than_nick_l292_292195

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l292_292195


namespace proposition_A_iff_proposition_B_l292_292664

-- Define propositions
def Proposition_A (A B C : ℕ) : Prop := (A = 60 ∨ B = 60 ∨ C = 60)
def Proposition_B (A B C : ℕ) : Prop :=
  (A + B + C = 180) ∧ 
  (2 * B = A + C)

-- The theorem stating the relationship between Proposition_A and Proposition_B
theorem proposition_A_iff_proposition_B (A B C : ℕ) :
  Proposition_A A B C ↔ Proposition_B A B C :=
sorry

end proposition_A_iff_proposition_B_l292_292664


namespace total_invested_amount_l292_292076

theorem total_invested_amount :
  ∃ (A B : ℝ), (A = 3000 ∧ B = 5000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000)
  ∨ 
  (A = 5000 ∧ B = 3000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000) :=
sorry

end total_invested_amount_l292_292076


namespace exist_three_primes_sum_to_30_l292_292551

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def less_than_twenty (n : ℕ) : Prop := n < 20

theorem exist_three_primes_sum_to_30 : 
  ∃ A B C : ℕ, is_prime A ∧ is_prime B ∧ is_prime C ∧ 
  less_than_twenty A ∧ less_than_twenty B ∧ less_than_twenty C ∧ 
  A + B + C = 30 :=
by 
  -- assume A = 2, prime and less than 20
  -- find B, C such that B and C are primes less than 20 and A + B + C = 30
  sorry

end exist_three_primes_sum_to_30_l292_292551


namespace range_of_a_l292_292849

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l292_292849


namespace baseball_card_count_l292_292554

-- Define initial conditions
def initial_cards := 15

-- Maria takes half of one more than the number of initial cards
def maria_takes := (initial_cards + 1) / 2

-- Remaining cards after Maria takes her share
def remaining_after_maria := initial_cards - maria_takes

-- You give Peter 1 card
def remaining_after_peter := remaining_after_maria - 1

-- Paul triples the remaining cards
def final_cards := remaining_after_peter * 3

-- Theorem statement to prove
theorem baseball_card_count :
  final_cards = 18 := by
sorry

end baseball_card_count_l292_292554


namespace required_earnings_correct_l292_292967

-- Definitions of the given conditions
def retail_price : ℝ := 600
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def amount_saved : ℝ := 120
def amount_given_by_mother : ℝ := 250
def additional_costs : ℝ := 50

-- Required amount Maria must earn
def required_earnings : ℝ := 247

-- Lean 4 theorem statement
theorem required_earnings_correct :
  let discount_amount := discount_rate * retail_price
  let discounted_price := retail_price - discount_amount
  let sales_tax_amount := sales_tax_rate * discounted_price
  let total_bike_cost := discounted_price + sales_tax_amount
  let total_cost := total_bike_cost + additional_costs
  let total_have := amount_saved + amount_given_by_mother
  required_earnings = total_cost - total_have :=
by
  sorry

end required_earnings_correct_l292_292967


namespace required_additional_coins_l292_292743

-- Summing up to the first 15 natural numbers
def sum_first_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Given: Alex has 15 friends and 90 coins
def number_of_friends := 15
def initial_coins := 90

-- The total number of coins required
def total_coins_required := sum_first_natural_numbers number_of_friends

-- Calculate the additional coins needed
theorem required_additional_coins : total_coins_required - initial_coins = 30 :=
by
  -- Placeholder for proof
  sorry

end required_additional_coins_l292_292743


namespace triangle_area_16_l292_292390

theorem triangle_area_16 : 
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 8)
  let base := (B.1 - A.1)
  let height := (C.2 - A.2)
  (base * height) / 2 = 16 := by
  sorry

end triangle_area_16_l292_292390


namespace arithmetic_sequence_product_l292_292352

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_condition : b 5 * b 6 = 14) :
  (b 4 * b 7 = -324) ∨ (b 4 * b 7 = -36) :=
sorry

end arithmetic_sequence_product_l292_292352


namespace total_shingles_for_all_roofs_l292_292503

def roof_A_length : ℕ := 20
def roof_A_width : ℕ := 40
def roof_A_shingles_per_sqft : ℕ := 8

def roof_B_length : ℕ := 25
def roof_B_width : ℕ := 35
def roof_B_shingles_per_sqft : ℕ := 10

def roof_C_length : ℕ := 30
def roof_C_width : ℕ := 30
def roof_C_shingles_per_sqft : ℕ := 12

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def total_area (length : ℕ) (width : ℕ) : ℕ :=
  2 * area length width

def total_shingles_needed (length : ℕ) (width : ℕ) (shingles_per_sqft : ℕ) : ℕ :=
  total_area length width * shingles_per_sqft

theorem total_shingles_for_all_roofs :
  total_shingles_needed roof_A_length roof_A_width roof_A_shingles_per_sqft +
  total_shingles_needed roof_B_length roof_B_width roof_B_shingles_per_sqft +
  total_shingles_needed roof_C_length roof_C_width roof_C_shingles_per_sqft = 51900 :=
by
  sorry

end total_shingles_for_all_roofs_l292_292503


namespace lincoln_one_way_fare_l292_292042

-- Define the given conditions as assumptions
variables (x : ℝ) (days : ℝ) (total_cost : ℝ) (trips_per_day : ℝ)

-- State the conditions
axiom condition1 : days = 9
axiom condition2 : total_cost = 288
axiom condition3 : trips_per_day = 2

-- The theorem we want to prove based on the conditions
theorem lincoln_one_way_fare (h1 : total_cost = days * trips_per_day * x) : x = 16 :=
by
  -- We skip the proof for the sake of this exercise
  sorry

end lincoln_one_way_fare_l292_292042


namespace sum_of_roots_of_cubic_l292_292819

noncomputable def P (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_cubic (a b c d : ℝ) (h : ∀ x : ℝ, P a b c d (x^2 + x) ≥ P a b c d (x + 1)) :
  (-b / a) = (P a b c d 0) :=
sorry

end sum_of_roots_of_cubic_l292_292819


namespace football_team_practice_hours_l292_292563

-- Definitions for each day's practice adjusted for weather events
def monday_hours : ℕ := 4
def tuesday_hours : ℕ := 5 - 1
def wednesday_hours : ℕ := 0
def thursday_hours : ℕ := 5
def friday_hours : ℕ := 3 + 2
def saturday_hours : ℕ := 4
def sunday_hours : ℕ := 0

-- Total practice hours calculation
def total_practice_hours : ℕ := 
  monday_hours + tuesday_hours + wednesday_hours + 
  thursday_hours + friday_hours + saturday_hours + 
  sunday_hours

-- Statement to prove
theorem football_team_practice_hours : total_practice_hours = 22 := by
  sorry

end football_team_practice_hours_l292_292563


namespace remainder_division_P_by_D_l292_292256

def P (x : ℝ) := 8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15
def D (x : ℝ) := 4 * x - 8

theorem remainder_division_P_by_D :
  let remainder := P 2 % D 2
  remainder = 31 :=
by
  -- Proof will be inserted here, but currently skipped
  sorry

end remainder_division_P_by_D_l292_292256


namespace problem_inequality_solution_l292_292302

theorem problem_inequality_solution (x : ℝ) :
  5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10 ↔ (69 / 29) < x ∧ x ≤ (17 / 7) :=
by sorry

end problem_inequality_solution_l292_292302


namespace tenth_pair_in_twentieth_row_l292_292943

def nth_pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if h : n > 0 ∧ k > 0 ∧ n >= k then (k, n + 1 - k)
  else (0, 0) -- define (0,0) as a default for invalid inputs

theorem tenth_pair_in_twentieth_row : nth_pair_in_row 20 10 = (10, 11) :=
by sorry

end tenth_pair_in_twentieth_row_l292_292943


namespace inequality_proof_l292_292271

noncomputable theory
open real

theorem inequality_proof {a b c d e f : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f) (h_ineq : |sqrt(a * d) - sqrt(b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) :=
sorry

end inequality_proof_l292_292271


namespace initial_tiger_sharks_l292_292907

open Nat

theorem initial_tiger_sharks (initial_guppies : ℕ) (initial_angelfish : ℕ) (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ) (sold_angelfish : ℕ) (sold_tiger_sharks : ℕ) (sold_oscar_fish : ℕ)
  (remaining_fish : ℕ) (initial_total_fish : ℕ) (total_guppies_angelfish_oscar : ℕ) (initial_tiger_sharks : ℕ) :
  initial_guppies = 94 → initial_angelfish = 76 → initial_oscar_fish = 58 →
  sold_guppies = 30 → sold_angelfish = 48 → sold_tiger_sharks = 17 → sold_oscar_fish = 24 →
  remaining_fish = 198 →
  initial_total_fish = (sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish + remaining_fish) →
  total_guppies_angelfish_oscar = (initial_guppies + initial_angelfish + initial_oscar_fish) →
  initial_tiger_sharks = (initial_total_fish - total_guppies_angelfish_oscar) →
  initial_tiger_sharks = 89 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end initial_tiger_sharks_l292_292907


namespace evaluate_expression_l292_292300

theorem evaluate_expression : 2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end evaluate_expression_l292_292300


namespace sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l292_292497

def recurrence_relation (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x (n + 1) = (2 * x n ^ 2 - x n) / (3 * (x n - 2))

-- For the first problem
theorem sequence_increasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : 4 < x 0 ∧ x 0 < 5) : ∀ n, x n < x (n + 1) ∧ x (n + 1) < 5 :=
by
  sorry

-- For the second problem
theorem sequence_decreasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : x 0 > 5) : ∀ n, 5 < x (n + 1) ∧ x (n + 1) < x n :=
by
  sorry

end sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l292_292497


namespace probability_of_heads_or_five_tails_is_one_eighth_l292_292975

namespace coin_flip

def num_heads_or_at_least_five_tails : ℕ :=
1 + 6 + 1

def total_outcomes : ℕ :=
2^6

def probability_heads_or_five_tails : ℚ :=
num_heads_or_at_least_five_tails / total_outcomes

theorem probability_of_heads_or_five_tails_is_one_eighth :
  probability_heads_or_five_tails = 1 / 8 := by
  sorry

end coin_flip

end probability_of_heads_or_five_tails_is_one_eighth_l292_292975


namespace triangle_base_length_l292_292632

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 24) (h_height : height = 8) (h_area_formula : area = (base * height) / 2) :
  base = 6 :=
by
  sorry

end triangle_base_length_l292_292632


namespace loss_equals_cost_price_of_balls_l292_292360

variable (selling_price : ℕ) (cost_price_ball : ℕ)
variable (number_of_balls : ℕ) (loss_incurred : ℕ) (x : ℕ)

-- Conditions
def condition1 : selling_price = 720 := sorry -- Selling price of 11 balls is Rs. 720
def condition2 : cost_price_ball = 120 := sorry -- Cost price of one ball is Rs. 120
def condition3 : number_of_balls = 11 := sorry -- Number of balls is 11

-- Cost price of 11 balls
def cost_price (n : ℕ) (cp_ball : ℕ): ℕ := n * cp_ball

-- Loss incurred on selling 11 balls
def loss (cp : ℕ) (sp : ℕ): ℕ := cp - sp

-- Equation for number of balls the loss equates to
def loss_equation (l : ℕ) (cp_ball : ℕ): ℕ := l / cp_ball

theorem loss_equals_cost_price_of_balls : 
  ∀ (n sp cp_ball cp l: ℕ), 
  sp = 720 ∧ cp_ball = 120 ∧ n = 11 ∧ 
  cp = cost_price n cp_ball ∧ 
  l = loss cp sp →
  loss_equation l cp_ball = 5 := sorry

end loss_equals_cost_price_of_balls_l292_292360


namespace calculate_sample_std_dev_l292_292102

-- Define the sample data
def sample_weights : List ℝ := [125, 124, 121, 123, 127]

-- Define the mean calculation
def sample_mean (weights : List ℝ) : ℝ :=
  (weights.foldl (+) 0) / weights.length

-- Define the variance calculation
def sample_variance (weights : List ℝ) (mean : ℝ) : ℝ :=
  (weights.foldl (λ acc x => acc + (x - mean) ^ 2) 0) / weights.length

-- Define the standard deviation calculation
def sample_std_dev (variance : ℝ) : ℝ :=
  Real.sqrt variance

-- The main theorem we want to prove
theorem calculate_sample_std_dev :
  let weights := sample_weights in
  let mean := sample_mean weights in
  let variance := sample_variance weights mean in
  sample_std_dev variance = 2 :=
by
  sorry

end calculate_sample_std_dev_l292_292102


namespace other_root_l292_292206

theorem other_root (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + m * x - 5 = 0 → (x = 1 ∨ x = -5 / 3)) :=
by {
  sorry
}

end other_root_l292_292206


namespace find_number_l292_292224

theorem find_number (n : ℕ) (h1 : n % 5 = 0) (h2 : 70 ≤ n ∧ n ≤ 90) (h3 : Nat.Prime n) : n = 85 := 
sorry

end find_number_l292_292224


namespace find_number_l292_292085

def single_digit (n : ℕ) : Prop := n < 10
def greater_than_zero (n : ℕ) : Prop := n > 0
def less_than_two (n : ℕ) : Prop := n < 2

theorem find_number (n : ℕ) : 
  single_digit n ∧ greater_than_zero n ∧ less_than_two n → n = 1 :=
by
  sorry

end find_number_l292_292085


namespace find_higher_percentage_l292_292558

-- Definitions based on conditions
def principal : ℕ := 8400
def time : ℕ := 2
def rate_0 : ℕ := 10
def delta_interest : ℕ := 840

-- The proof statement
theorem find_higher_percentage (r : ℕ) :
  (principal * rate_0 * time / 100 + delta_interest = principal * r * time / 100) →
  r = 15 :=
by sorry

end find_higher_percentage_l292_292558


namespace opposite_neg_fraction_l292_292228

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l292_292228


namespace painting_area_l292_292718

theorem painting_area (c t A : ℕ) (h1 : c = 15) (h2 : t = 840) (h3 : c * A = t) : A = 56 := 
by
  sorry -- proof to demonstrate A = 56

end painting_area_l292_292718


namespace prime_intersect_even_l292_292182

-- Definitions for prime numbers and even numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Sets P and Q
def P : Set ℕ := { n | is_prime n }
def Q : Set ℕ := { n | is_even n }

-- Proof statement
theorem prime_intersect_even : P ∩ Q = {2} :=
by
  sorry

end prime_intersect_even_l292_292182


namespace angleC_equals_40_of_angleA_40_l292_292173

-- Define an arbitrary quadrilateral type and its angle A and angle C
structure Quadrilateral :=
  (angleA : ℝ)  -- angleA is in degrees
  (angleC : ℝ)  -- angleC is in degrees

-- Given condition in the problem
def quadrilateral_with_A_40 : Quadrilateral :=
  { angleA := 40, angleC := 0 } -- Initialize angleC as a placeholder

-- Theorem stating the problem's claim
theorem angleC_equals_40_of_angleA_40 :
  quadrilateral_with_A_40.angleA = 40 → quadrilateral_with_A_40.angleC = 40 :=
by
  sorry  -- Proof is omitted for brevity

end angleC_equals_40_of_angleA_40_l292_292173


namespace train_speed_l292_292397

noncomputable def trainLength : ℕ := 400
noncomputable def timeToCrossPole : ℕ := 20

theorem train_speed : (trainLength / timeToCrossPole) = 20 := by
  sorry

end train_speed_l292_292397


namespace min_ab_l292_292147

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a + b + 3 = a * b) : 9 ≤ a * b :=
sorry

end min_ab_l292_292147


namespace tim_initial_balls_correct_l292_292208

-- Defining the initial number of balls Robert had
def robert_initial_balls : ℕ := 25

-- Defining the final number of balls Robert had
def robert_final_balls : ℕ := 45

-- Defining the number of balls Tim had initially
def tim_initial_balls := 40

-- Now, we state the proof problem:
theorem tim_initial_balls_correct :
  robert_initial_balls + (tim_initial_balls / 2) = robert_final_balls :=
by
  -- This is the part where you typically write the proof.
  -- However, we put sorry here because the task does not require the proof itself.
  sorry

end tim_initial_balls_correct_l292_292208


namespace composite_polynomial_l292_292051

-- Definition that checks whether a number is composite
def is_composite (a : ℕ) : Prop := ∃ (b c : ℕ), b > 1 ∧ c > 1 ∧ a = b * c

-- Problem translated into a Lean 4 statement
theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  is_composite (n ^ (5 * n - 1) + n ^ (5 * n - 2) + n ^ (5 * n - 3) + n + 1) :=
sorry

end composite_polynomial_l292_292051


namespace even_function_has_zero_coefficient_l292_292019

theorem even_function_has_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (x^2 + a*x) = (x^2 + a*(-x))) → a = 0 :=
by
  intro h
  -- the proof part is omitted as requested
  sorry

end even_function_has_zero_coefficient_l292_292019


namespace solve_inequality_l292_292447

theorem solve_inequality : {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {(-1 : ℝ) / 3} :=
by
  sorry

end solve_inequality_l292_292447


namespace value_of_a_l292_292165

-- Definition of the function and the point
def graph_function (x : ℝ) : ℝ := -x^2
def point_lies_on_graph (a : ℝ) : Prop := (a, -9) ∈ {p : ℝ × ℝ | p.2 = graph_function p.1}

-- The theorem stating that if the point (a, -9) lies on the graph of y = -x^2, then a = ±3
theorem value_of_a (a : ℝ) (h : point_lies_on_graph a) : a = 3 ∨ a = -3 :=
by 
  sorry

end value_of_a_l292_292165


namespace number_of_pairs_count_number_of_pairs_l292_292627

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l292_292627


namespace handshakes_at_convention_l292_292996

theorem handshakes_at_convention :
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  handshakes_among_gremlins + handshakes_between_imps_gremlins = 660 :=
by
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  show handshakes_among_gremlins + handshakes_between_imps_gremlins = 660
  sorry

end handshakes_at_convention_l292_292996


namespace total_output_equal_at_20_l292_292562

noncomputable def total_output_A (x : ℕ) : ℕ :=
  200 + 20 * x

noncomputable def total_output_B (x : ℕ) : ℕ :=
  30 * x

theorem total_output_equal_at_20 :
  total_output_A 20 = total_output_B 20 :=
by
  sorry

end total_output_equal_at_20_l292_292562


namespace gcd_of_lcm_ratio_l292_292022

theorem gcd_of_lcm_ratio {A B k : ℕ} (h1 : Nat.lcm A B = 180) (h2 : A * 5 = B * 2) :
  Nat.gcd A B = 18 := 
by
  sorry

end gcd_of_lcm_ratio_l292_292022


namespace weight_of_new_person_l292_292399

-- Definitions based on conditions
def average_weight_increase : ℝ := 2.5
def number_of_persons : ℕ := 8
def old_weight : ℝ := 65
def total_weight_increase : ℝ := number_of_persons * average_weight_increase

-- Proposition to prove
theorem weight_of_new_person : (old_weight + total_weight_increase) = 85 := by
  -- add the actual proof here
  sorry

end weight_of_new_person_l292_292399


namespace find_sum_l292_292570

noncomputable def sumPutAtSimpleInterest (R: ℚ) (P: ℚ) := 
  let I := P * R * 5 / 100
  I + 90 = P * (R + 6) * 5 / 100 → P = 300

theorem find_sum (R: ℚ) (P: ℚ) : sumPutAtSimpleInterest R P := by
  sorry

end find_sum_l292_292570


namespace certain_number_divisibility_l292_292484

theorem certain_number_divisibility (n : ℕ) (p : ℕ) (h : p = 1) (h2 : 4864 * 9 * n % 12 = 0) : n = 43776 :=
by {
  sorry
}

end certain_number_divisibility_l292_292484


namespace chocolate_difference_l292_292200

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l292_292200


namespace Rhett_rent_expense_l292_292676

-- Define the problem statement using given conditions
theorem Rhett_rent_expense
  (late_payments : ℕ := 2)
  (no_late_fees : Bool := true)
  (fraction_of_salary : ℝ := 3 / 5)
  (monthly_salary : ℝ := 5000)
  (tax_rate : ℝ := 0.1) :
  let salary_after_taxes := monthly_salary * (1 - tax_rate)
  let total_late_rent := fraction_of_salary * salary_after_taxes
  let monthly_rent_expense := total_late_rent / late_payments
  monthly_rent_expense = 1350 := by
  sorry

end Rhett_rent_expense_l292_292676


namespace football_club_balance_l292_292096

/-- A football club has a balance of $100 million. The club then sells 2 of its players at $10 million each, and buys 4 more at $15 million each. Prove that the final balance is $60 million. -/
theorem football_club_balance :
  let initial_balance := 100
  let income_from_sales := 10 * 2
  let expenditure_on_purchases := 15 * 4
  let final_balance := initial_balance + income_from_sales - expenditure_on_purchases
  final_balance = 60 :=
by
  simp only [initial_balance, income_from_sales, expenditure_on_purchases, final_balance]
  sorry

end football_club_balance_l292_292096


namespace dune_buggy_speed_l292_292427

theorem dune_buggy_speed (S : ℝ) :
  (1/3 * S + 1/3 * (S + 12) + 1/3 * (S - 18) = 58) → S = 60 :=
by
  sorry

end dune_buggy_speed_l292_292427


namespace point_D_coordinates_l292_292496

theorem point_D_coordinates 
  (F : (ℕ × ℕ)) 
  (coords_F : F = (5,5)) 
  (D : (ℕ × ℕ)) 
  (coords_D : D = (2,4)) :
  (D = (2,4)) :=
by 
  sorry

end point_D_coordinates_l292_292496


namespace symmetric_point_y_axis_l292_292810

theorem symmetric_point_y_axis (A B : ℝ × ℝ) (hA : A = (2, 5)) (h_symm : B = (-A.1, A.2)) :
  B = (-2, 5) :=
sorry

end symmetric_point_y_axis_l292_292810


namespace helpers_cakes_l292_292753

theorem helpers_cakes (S : ℕ) (helpers large_cakes small_cakes : ℕ)
  (h1 : helpers = 10)
  (h2 : large_cakes = 2)
  (h3 : small_cakes = 700)
  (h4 : 1 * helpers * large_cakes = 20)
  (h5 : 2 * helpers * S = small_cakes) :
  S = 35 :=
by
  sorry

end helpers_cakes_l292_292753


namespace books_to_sell_to_reach_goal_l292_292686

-- Definitions for conditions
def initial_savings : Nat := 10
def clarinet_cost : Nat := 90
def book_price : Nat := 5
def halfway_goal : Nat := clarinet_cost / 2

-- The primary theorem to prove
theorem books_to_sell_to_reach_goal : 
  initial_savings + (initial_savings = 0 → clarinet_cost) / book_price = 25 :=
by
  -- Proof steps (skipped in the statement)
  sorry

end books_to_sell_to_reach_goal_l292_292686


namespace alix_more_chocolates_than_nick_l292_292202

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l292_292202


namespace evaluate_cube_root_fraction_l292_292912

theorem evaluate_cube_root_fraction (h : 18.75 = 75 / 4) : (Real.cbrt (6 / 18.75)) = 2 / (Real.cbrt 25) :=
  sorry

end evaluate_cube_root_fraction_l292_292912


namespace div_eq_of_scaled_div_eq_l292_292159

theorem div_eq_of_scaled_div_eq (h : 29.94 / 1.45 = 17.7) : 2994 / 14.5 = 17.7 := 
by
  sorry

end div_eq_of_scaled_div_eq_l292_292159


namespace max_view_angle_dist_l292_292544

theorem max_view_angle_dist (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : ∃ (x : ℝ), x = Real.sqrt (b * (a + b)) := by
  sorry

end max_view_angle_dist_l292_292544


namespace must_be_odd_l292_292292

theorem must_be_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
sorry

end must_be_odd_l292_292292


namespace avg_one_sixth_one_fourth_l292_292527

theorem avg_one_sixth_one_fourth : (1 / 6 + 1 / 4) / 2 = 5 / 24 := by
  sorry

end avg_one_sixth_one_fourth_l292_292527


namespace nat_pair_count_eq_five_l292_292616

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l292_292616


namespace two_a7_minus_a8_l292_292347

variable (a : ℕ → ℝ) -- Assuming the arithmetic sequence {a_n} is a sequence of real numbers

-- Definitions and conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

axiom a1_plus_3a6_plus_a11 : a 1 + 3 * (a 6) + a 11 = 120

-- The theorem to be proved
theorem two_a7_minus_a8 (h : is_arithmetic_sequence a) : 2 * a 7 - a 8 = 24 := 
sorry

end two_a7_minus_a8_l292_292347


namespace area_of_rectangle_l292_292644

noncomputable def leanProblem : Prop :=
  let E := 8
  let F := 2.67
  let BE := E -- length from B to E on AB
  let AF := F -- length from A to F on AD
  let BC := E * (Real.sqrt 3) -- from triangle properties CB is BE * sqrt(3)
  let FD := BC - F -- length from F to D on AD
  let CD := FD * (Real.sqrt 3) -- applying the triangle properties again
  (BC * CD = 192 * (Real.sqrt 3) - 64.08)

theorem area_of_rectangle (E : ℝ) (F : ℝ) 
  (hE : E = 8) 
  (hF : F = 2.67) 
  (BC : ℝ) (CD : ℝ) :
  leanProblem :=
by 
  sorry

end area_of_rectangle_l292_292644


namespace cows_now_l292_292565

-- Defining all conditions
def initial_cows : ℕ := 39
def cows_died : ℕ := 25
def cows_sold : ℕ := 6
def cows_increase : ℕ := 24
def cows_bought : ℕ := 43
def cows_gift : ℕ := 8

-- Lean statement for the equivalent proof problem
theorem cows_now :
  let cows_left := initial_cows - cows_died
  let cows_after_selling := cows_left - cows_sold
  let cows_this_year_increased := cows_after_selling + cows_increase
  let cows_with_purchase := cows_this_year_increased + cows_bought
  let total_cows := cows_with_purchase + cows_gift
  total_cows = 83 :=
by
  sorry

end cows_now_l292_292565


namespace max_sum_of_factors_l292_292388

theorem max_sum_of_factors (x y : ℕ) (h1 : x * y = 48) (h2 : x ≠ y) : x + y ≤ 49 :=
by
  sorry

end max_sum_of_factors_l292_292388


namespace clerk_daily_salary_l292_292639

theorem clerk_daily_salary (manager_salary : ℝ) (num_managers num_clerks : ℕ) (total_salary : ℝ) (clerk_salary : ℝ)
  (h1 : manager_salary = 5)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16) :
  clerk_salary = 2 :=
by
  sorry

end clerk_daily_salary_l292_292639


namespace count_pairs_satisfying_condition_l292_292620

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l292_292620


namespace possible_values_of_m_l292_292941

open Set

variable (A B : Set ℤ)
variable (m : ℤ)

theorem possible_values_of_m (h₁ : A = {1, 2, m * m}) (h₂ : B = {1, m}) (h₃ : B ⊆ A) :
  m = 0 ∨ m = 2 :=
  sorry

end possible_values_of_m_l292_292941


namespace cannot_determine_right_triangle_l292_292080

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def two_angles_complementary (α β : ℝ) : Prop :=
  α + β = 90

def exterior_angle_is_right (γ : ℝ) : Prop :=
  γ = 90

theorem cannot_determine_right_triangle :
  ¬ (∃ (a b c : ℝ), a = 1 ∧ b = 1 ∧ c = 2 ∧ is_right_triangle a b c) :=
by sorry

end cannot_determine_right_triangle_l292_292080


namespace find_a_minus_b_l292_292375

-- Given definitions for conditions
variables (a b : ℤ)

-- Given conditions as hypotheses
def condition1 := a + 2 * b = 5
def condition2 := a * b = -12

theorem find_a_minus_b (h1 : condition1 a b) (h2 : condition2 a b) : a - b = -7 :=
sorry

end find_a_minus_b_l292_292375


namespace intersection_of_asymptotes_l292_292446

theorem intersection_of_asymptotes :
  ∃ x y : ℝ, (y = 1) ∧ (x = 3) ∧ (y = (x^2 - 6*x + 8) / (x^2 - 6*x + 9)) := 
by {
  sorry
}

end intersection_of_asymptotes_l292_292446


namespace find_g2_l292_292376

theorem find_g2
  (g : ℝ → ℝ)
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2) :
  g 2 = 19 / 16 := 
sorry

end find_g2_l292_292376


namespace shortest_distance_curve_to_line_l292_292246

open Real

def curve (x : ℝ) : ℝ := log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem shortest_distance_curve_to_line :
  ∃ (d_min : ℝ), ∀ x : ℝ, x > 0.5 → 
  ∃ y : ℝ, y = curve x ∧ 
  (line x y → abs (2 * x - log (2 * x - 1) + 3) / sqrt 5 = d_min) :=
by
  sorry

end shortest_distance_curve_to_line_l292_292246


namespace relationship_between_a_and_b_l292_292785

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- Given conditions
variables (b a : ℝ)
variables (hx : 0 < b) (ha : 0 < a)
variables (x : ℝ) (hb : |x - 1| < b) (hf : |f x - 4| < a)

-- The theorem statement
theorem relationship_between_a_and_b
  (hf_x : ∀ x : ℝ, |x - 1| < b -> |f x - 4| < a) :
  a - 3 * b ≥ 0 :=
sorry

end relationship_between_a_and_b_l292_292785


namespace average_annual_growth_rate_equation_l292_292028

variable (x : ℝ)
axiom seventh_to_ninth_reading_increase : (1 : ℝ) * (1 + x) * (1 + x) = 1.21

theorem average_annual_growth_rate_equation :
  100 * (1 + x) ^ 2 = 121 :=
by
  have h : (1 : ℝ) * (1 + x) * (1 + x) = 1.21 := seventh_to_ninth_reading_increase x
  sorry

end average_annual_growth_rate_equation_l292_292028


namespace minimum_norm_of_v_l292_292965

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end minimum_norm_of_v_l292_292965


namespace train_speed_is_correct_l292_292877

noncomputable def speed_of_train (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_is_correct :
  speed_of_train 200 19.99840012798976 = 36.00287976960864 :=
by
  sorry

end train_speed_is_correct_l292_292877


namespace time_no_traffic_is_4_hours_l292_292502

-- Definitions and conditions
def distance : ℕ := 200
def time_traffic : ℕ := 5

axiom traffic_speed_relation : ∃ (speed_traffic : ℕ), distance = speed_traffic * time_traffic
axiom speed_difference : ∀ (speed_traffic speed_no_traffic : ℕ), speed_no_traffic = speed_traffic + 10

-- Prove that the time when there's no traffic is 4 hours
theorem time_no_traffic_is_4_hours : ∀ (speed_traffic speed_no_traffic : ℕ), 
  distance = speed_no_traffic * (distance / speed_no_traffic) -> (distance / speed_no_traffic) = 4 :=
by
  intros speed_traffic speed_no_traffic h
  sorry

end time_no_traffic_is_4_hours_l292_292502


namespace airsickness_related_to_gender_l292_292169

def a : ℕ := 28
def b : ℕ := 28
def c : ℕ := 28
def d : ℕ := 56
def n : ℕ := 140

def contingency_relation (a b c d n K2 : ℕ) : Prop := 
  let numerator := n * (a * d - b * c)^2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  K2 > 3841 / 1000

-- Goal statement for the proof
theorem airsickness_related_to_gender :
  contingency_relation a b c d n 3888 :=
  sorry

end airsickness_related_to_gender_l292_292169


namespace part1_part2_l292_292005

-- Definition of the function f
def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

-- Part 1: For m = 1, the solution set of f(x) >= 6
theorem part1 (x : ℝ) : f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := 
by 
  sorry

-- Part 2: If the inequality f(x) ≤ 2m - 5 has a solution with respect to x, then m ≥ 8
theorem part2 (m : ℝ) (h : ∃ x, f x m ≤ 2 * m - 5) : m ≥ 8 :=
by
  sorry

end part1_part2_l292_292005


namespace radish_patch_area_l292_292739

-- Definitions from the conditions
variables (R P : ℕ) -- R: area of radish patch, P: area of pea patch
variable (h1 : P = 2 * R) -- The pea patch is twice as large as the radish patch
variable (h2 : P / 6 = 5) -- One-sixth of the pea patch is 5 square feet

-- Goal statement
theorem radish_patch_area : R = 15 :=
by
  sorry

end radish_patch_area_l292_292739


namespace wendy_initial_flowers_l292_292389

theorem wendy_initial_flowers (wilted: ℕ) (bouquets_made: ℕ) (flowers_per_bouquet: ℕ) (flowers_initially_picked: ℕ):
  wilted = 35 →
  bouquets_made = 2 →
  flowers_per_bouquet = 5 →
  flowers_initially_picked = wilted + bouquets_made * flowers_per_bouquet →
  flowers_initially_picked = 45 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_initial_flowers_l292_292389


namespace symmetrical_point_wrt_x_axis_l292_292030

theorem symmetrical_point_wrt_x_axis (x y : ℝ) (P_symmetrical : (ℝ × ℝ)) (hx : x = -1) (hy : y = 2) : 
  P_symmetrical = (x, -y) → P_symmetrical = (-1, -2) :=
by
  intros h
  rw [hx, hy] at h
  exact h

end symmetrical_point_wrt_x_axis_l292_292030


namespace f_f_3_eq_651_over_260_l292_292149

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (2 + x⁻¹))

/-- Prove that f(f(3)) = 651/260 -/
theorem f_f_3_eq_651_over_260 : f (f (3)) = 651 / 260 := 
sorry

end f_f_3_eq_651_over_260_l292_292149


namespace glove_probability_correct_l292_292167

noncomputable def glove_probability : ℚ :=
  let red_pair := ("r1", "r2") -- pair of red gloves
  let black_pair := ("b1", "b2") -- pair of black gloves
  let white_pair := ("w1", "w2") -- pair of white gloves
  let all_pairs := [
    (red_pair.1, red_pair.2), 
    (black_pair.1, black_pair.2), 
    (white_pair.1, white_pair.2),
    (red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
    (red_pair.2, black_pair.1), (red_pair.2, white_pair.1),
    (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)
  ]
  let valid_pairs := [(red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
                      (red_pair.2, black_pair.1), (red_pair.2, white_pair.1), 
                      (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)]
  (valid_pairs.length : ℚ) / (all_pairs.length : ℚ)

theorem glove_probability_correct :
  glove_probability = 2 / 5 := 
by
  sorry

end glove_probability_correct_l292_292167


namespace reversed_number_increase_l292_292101

theorem reversed_number_increase (a b c : ℕ) 
  (h1 : a + b + c = 10) 
  (h2 : b = a + c)
  (h3 : a = 2 ∧ b = 5 ∧ c = 3) :
  (c * 100 + b * 10 + a) - (a * 100 + b * 10 + c) = 99 :=
by
  sorry

end reversed_number_increase_l292_292101


namespace required_additional_coins_l292_292744

-- Summing up to the first 15 natural numbers
def sum_first_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Given: Alex has 15 friends and 90 coins
def number_of_friends := 15
def initial_coins := 90

-- The total number of coins required
def total_coins_required := sum_first_natural_numbers number_of_friends

-- Calculate the additional coins needed
theorem required_additional_coins : total_coins_required - initial_coins = 30 :=
by
  -- Placeholder for proof
  sorry

end required_additional_coins_l292_292744


namespace value_of_M_l292_292119

theorem value_of_M (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 :=
by
  sorry

end value_of_M_l292_292119


namespace axis_of_symmetry_l292_292937

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + (Real.pi / 2))) * (Real.cos (x + (Real.pi / 4)))

theorem axis_of_symmetry : 
  ∃ (a : ℝ), a = 5 * Real.pi / 8 ∧ ∀ x : ℝ, f (2 * a - x) = f x := 
by
  sorry

end axis_of_symmetry_l292_292937


namespace opposite_of_neg_one_over_2023_l292_292241

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l292_292241


namespace count_pairs_satisfying_condition_l292_292618

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l292_292618


namespace min_arg_z_l292_292465

noncomputable def z (x y : ℝ) := x + y * Complex.I

def satisfies_condition (x y : ℝ) : Prop :=
  Complex.abs (z x y + 3 - Real.sqrt 3 * Complex.I) = Real.sqrt 3

theorem min_arg_z (x y : ℝ) (h : satisfies_condition x y) :
  Complex.arg (z x y) = 5 * Real.pi / 6 := 
sorry

end min_arg_z_l292_292465


namespace numPeopleToLeftOfKolya_l292_292981

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l292_292981


namespace wrapping_paper_area_l292_292559

variable {l w h : ℝ}

theorem wrapping_paper_area (hl : 0 < l) (hw : 0 < w) (hh : 0 < h) :
  (4 * l * h + 2 * l * h + 2 * w * h) = 6 * l * h + 2 * w * h :=
  sorry

end wrapping_paper_area_l292_292559


namespace attendees_gift_exchange_l292_292887

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l292_292887


namespace interest_rate_l292_292872

theorem interest_rate (SI P T : ℕ) (h1 : SI = 2000) (h2 : P = 5000) (h3 : T = 10) :
  (SI = (P * R * T) / 100) -> R = 4 :=
by
  sorry

end interest_rate_l292_292872


namespace simplify_expression_l292_292296

theorem simplify_expression (y : ℝ) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^2) = (9 / 2) * y^3 :=
by sorry

end simplify_expression_l292_292296


namespace matrix_power_minus_l292_292183

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end matrix_power_minus_l292_292183


namespace infinite_solutions_distinct_natural_numbers_l292_292363

theorem infinite_solutions_distinct_natural_numbers :
  ∃ (x y z : ℕ), (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) ∧ (x ^ 2015 + y ^ 2015 = z ^ 2016) :=
by
  sorry

end infinite_solutions_distinct_natural_numbers_l292_292363


namespace friend_spent_more_l292_292550

/-- Given that the total amount spent for lunch is $15 and your friend spent $8 on their lunch,
we need to prove that your friend spent $1 more than you did. -/
theorem friend_spent_more (total_spent friend_spent : ℤ) (h1 : total_spent = 15) (h2 : friend_spent = 8) :
  friend_spent - (total_spent - friend_spent) = 1 :=
by
  sorry

end friend_spent_more_l292_292550


namespace jason_probability_reroll_two_dice_l292_292499

-- Defining the problem conditions
def rolls : Type := list ℕ -- a representation of the three dice rolls, values between 1 and 6

-- Function to count favorable outcomes based on Jason's strategy
def count_favorable_outcomes (dice : rolls) : ℕ := sorry -- Detailed implementation omitted

-- Function to calculate probability (number of favorable outcomes / total outcomes)
def probability_of_rerolling_two_dice (dice : rolls) : ℚ := 
  (count_favorable_outcomes dice) / 216

-- Theorem: The probability that Jason chooses to reroll exactly two of the dice is 7/36
theorem jason_probability_reroll_two_dice :
  probability_of_rerolling_two_dice [1, 2, 3] = 7 / 36 :=
sorry

end jason_probability_reroll_two_dice_l292_292499


namespace elvis_ralph_matchsticks_l292_292126

/-- 
   Elvis and Ralph are making square shapes with matchsticks from a box containing 
   50 matchsticks. Elvis makes 4-matchstick squares and Ralph makes 8-matchstick 
   squares. If Elvis makes 5 squares and Ralph makes 3, prove the number of matchsticks 
   left in the box is 6. 
-/
def matchsticks_left_in_box
  (initial_matchsticks : ℕ)
  (elvis_squares : ℕ)
  (elvis_matchsticks : ℕ)
  (ralph_squares : ℕ)
  (ralph_matchsticks : ℕ)
  (elvis_squares_count : ℕ)
  (ralph_squares_count : ℕ) : ℕ :=
  initial_matchsticks - (elvis_squares_count * elvis_matchsticks + ralph_squares_count * ralph_matchsticks)

theorem elvis_ralph_matchsticks : matchsticks_left_in_box 50 4 5 8 3 = 6 := 
  sorry

end elvis_ralph_matchsticks_l292_292126


namespace solution_set_l292_292004

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.sin x + Real.cos x + x^2

theorem solution_set (x : ℝ) :
  f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 ↔ (1 / Real.exp 1 < x ∧ x < Real.exp 1) :=
by {
  sorry
}

end solution_set_l292_292004


namespace count_increasing_numbers_l292_292597

-- Define the set of digits we are concerned with
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a natural number type representing numbers with increasing digits
def increasing_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → n.digits i < n.digits j

-- Define the set of natural numbers with increasing digits and at least two digits
def increasing_numbers : set ℕ :=
  {n | increasing_digits n ∧ 10 ≤ n ∧ n ≤ 987654321}

-- Define the theorem to be proved
theorem count_increasing_numbers : set.card increasing_numbers = 502 :=
by sorry

end count_increasing_numbers_l292_292597


namespace west_1000_move_l292_292161

def eastMovement (d : Int) := d  -- east movement positive
def westMovement (d : Int) := -d -- west movement negative

theorem west_1000_move : westMovement 1000 = -1000 :=
  by
    sorry

end west_1000_move_l292_292161


namespace Heather_delay_l292_292213

noncomputable def find_start_time : ℝ :=
  let d := 15 -- Initial distance between Stacy and Heather in miles
  let H := 5 -- Heather's speed in miles/hour
  let S := H + 1 -- Stacy's speed in miles/hour
  let d_H := 5.7272727272727275 -- Distance Heather walked when they meet
  let t_H := d_H / H -- Time Heather walked till they meet in hours
  let d_S := S * t_H -- Distance Stacy walked till they meet in miles
  let total_distance := d_H + d_S -- Total distance covered when they meet in miles
  let remaining_distance := d - total_distance -- Remaining distance Stacy covers alone before Heather starts in miles
  let t_S := remaining_distance / S -- Time Stacy walked alone in hours
  let minutes := t_S * 60 -- Convert time Stacy walked alone to minutes
  minutes -- Result in minutes

theorem Heather_delay : find_start_time = 24 := by
  sorry -- Proof of the theorem

end Heather_delay_l292_292213


namespace parallel_lines_b_value_l292_292260

-- Define the first line equation in slope-intercept form.
def line1_slope (b : ℝ) : ℝ :=
  3

-- Define the second line equation in slope-intercept form.
def line2_slope (b : ℝ) : ℝ :=
  b + 10

-- Theorem stating that if the lines are parallel, the value of b is -7.
theorem parallel_lines_b_value :
  ∀ b : ℝ, line1_slope b = line2_slope b → b = -7 :=
by
  intro b
  intro h
  sorry

end parallel_lines_b_value_l292_292260


namespace coordinates_of_P_l292_292977

-- Define the point P with given coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P(3, 5)
def P : Point := ⟨3, 5⟩

-- Define a theorem stating that the coordinates of P are (3, 5)
theorem coordinates_of_P : P = ⟨3, 5⟩ :=
  sorry

end coordinates_of_P_l292_292977


namespace machine_A_produces_1_sprockets_per_hour_l292_292724

namespace SprocketsProduction

variable {A T : ℝ} -- A: sprockets per hour of machine A, T: hours it takes for machine Q to produce 110 sprockets

-- Given conditions
axiom machine_Q_production_rate : 110 / T = 1.10 * A
axiom machine_P_production_rate : 110 / (T + 10) = A

-- The target theorem to prove
theorem machine_A_produces_1_sprockets_per_hour (h1 : 110 / T = 1.10 * A) (h2 : 110 / (T + 10) = A) : A = 1 :=
by sorry

end SprocketsProduction

end machine_A_produces_1_sprockets_per_hour_l292_292724


namespace dogwood_trees_after_5_years_l292_292697

theorem dogwood_trees_after_5_years :
  let current_trees := 39
  let trees_planted_today := 41
  let growth_rate_today := 2 -- trees per year
  let trees_planted_tomorrow := 20
  let growth_rate_tomorrow := 4 -- trees per year
  let years := 5
  let total_planted_trees := trees_planted_today + trees_planted_tomorrow
  let total_initial_trees := current_trees + total_planted_trees
  let total_growth_today := growth_rate_today * years
  let total_growth_tomorrow := growth_rate_tomorrow * years
  let total_growth := total_growth_today + total_growth_tomorrow
  let final_tree_count := total_initial_trees + total_growth
  final_tree_count = 130 := by
  sorry

end dogwood_trees_after_5_years_l292_292697


namespace sum_adjacent_to_49_l292_292084

noncomputable def sum_of_adjacent_divisors : ℕ :=
  let divisors := [5, 7, 35, 49, 245]
  -- We assume an arrangement such that adjacent pairs to 49 are {35, 245}
  35 + 245

theorem sum_adjacent_to_49 : sum_of_adjacent_divisors = 280 := by
  sorry

end sum_adjacent_to_49_l292_292084


namespace gcd_of_lcm_ratio_l292_292023

theorem gcd_of_lcm_ratio {A B k : ℕ} (h1 : Nat.lcm A B = 180) (h2 : A * 5 = B * 2) :
  Nat.gcd A B = 18 := 
by
  sorry

end gcd_of_lcm_ratio_l292_292023


namespace number_of_girls_in_first_year_l292_292557

theorem number_of_girls_in_first_year
  (total_students : ℕ)
  (sample_size : ℕ)
  (boys_in_sample : ℕ)
  (girls_in_first_year : ℕ) :
  total_students = 2400 →
  sample_size = 80 →
  boys_in_sample = 42 →
  girls_in_first_year = total_students * (sample_size - boys_in_sample) / sample_size →
  girls_in_first_year = 1140 :=
by 
  intros h1 h2 h3 h4
  sorry

end number_of_girls_in_first_year_l292_292557


namespace product_of_numbers_l292_292701

theorem product_of_numbers :
  ∃ (x y z : ℚ), (x + y + z = 30) ∧ (x = 3 * (y + z)) ∧ (y = 5 * z) ∧ (x * y * z = 175.78125) :=
by
  sorry

end product_of_numbers_l292_292701


namespace gcd_of_A_B_l292_292020

noncomputable def A (k : ℕ) := 2 * k
noncomputable def B (k : ℕ) := 5 * k

theorem gcd_of_A_B (k : ℕ) (h_lcm : Nat.lcm (A k) (B k) = 180) : Nat.gcd (A k) (B k) = 18 :=
by
  sorry

end gcd_of_A_B_l292_292020


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l292_292344

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l292_292344


namespace Sally_quarters_l292_292831

theorem Sally_quarters : 760 + 418 - 152 = 1026 := 
by norm_num

end Sally_quarters_l292_292831


namespace hockey_season_games_l292_292070

theorem hockey_season_games (n_teams : ℕ) (n_faces : ℕ) (h1 : n_teams = 18) (h2 : n_faces = 10) :
  let total_games := (n_teams * (n_teams - 1) / 2) * n_faces
  total_games = 1530 :=
by
  sorry

end hockey_season_games_l292_292070


namespace max_ad_minus_bc_l292_292452

theorem max_ad_minus_bc (a b c d : ℤ) (ha : a ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hb : b ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hc : c ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hd : d ∈ Set.image (fun x => x) {(-1), 1, 2}) :
  ad - bc ≤ 6 :=
sorry

end max_ad_minus_bc_l292_292452


namespace find_g2_l292_292353

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g x

theorem find_g2 {f g : ℝ → ℝ}
  (h1 : odd_function f)
  (h2 : even_function g)
  (h3 : ∀ x : ℝ, f x + g x = 2^x) :
  g 2 = 17 / 8 :=
sorry

end find_g2_l292_292353


namespace proof_problem_l292_292455

variables (p q : Prop)

-- Assuming p is true and q is false
axiom p_is_true : p
axiom q_is_false : ¬ q

-- Proving that (¬p) ∨ (¬q) is true
theorem proof_problem : (¬p) ∨ (¬q) :=
by {
  sorry
}

end proof_problem_l292_292455


namespace solve_fractional_equation_l292_292532

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 4) : 
  (3 - x) / (x - 4) + 1 / (4 - x) = 1 → x = 3 :=
by {
  sorry
}

end solve_fractional_equation_l292_292532


namespace one_cow_one_bag_l292_292861

theorem one_cow_one_bag {days_per_bag : ℕ} (h : 50 * days_per_bag = 50 * 50) : days_per_bag = 50 :=
by
  sorry

end one_cow_one_bag_l292_292861


namespace number_of_multiples_of_10_lt_200_l292_292796

theorem number_of_multiples_of_10_lt_200 : 
  ∃ n, (∀ k, (1 ≤ k) → (k < 20) → k * 10 < 200) ∧ n = 19 := 
by
  sorry

end number_of_multiples_of_10_lt_200_l292_292796


namespace abs_inequality_solution_set_l292_292530

theorem abs_inequality_solution_set (x : ℝ) : -1 < x ∧ x < 1 ↔ |2*x - 1| - |x - 2| < 0 := by
  sorry

end abs_inequality_solution_set_l292_292530


namespace second_smallest_is_3_probability_l292_292368

noncomputable def probability_of_second_smallest_is_3 : ℚ := 
  let total_ways := Nat.choose 10 6
  let favorable_ways := 2 * Nat.choose 7 4
  favorable_ways / total_ways

theorem second_smallest_is_3_probability : probability_of_second_smallest_is_3 = 1 / 3 := sorry

end second_smallest_is_3_probability_l292_292368


namespace change_received_l292_292651

theorem change_received (cost_cat_toy : ℝ) (cost_cage : ℝ) (total_paid : ℝ) (change : ℝ) :
  cost_cat_toy = 8.77 →
  cost_cage = 10.97 →
  total_paid = 20.00 →
  change = 0.26 →
  total_paid - (cost_cat_toy + cost_cage) = change := by
sorry

end change_received_l292_292651


namespace opposite_of_neg_one_div_2023_l292_292234

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l292_292234


namespace find_a5_l292_292454

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions
variable (a : ℕ → ℝ)
variable (h_arith : arithmetic_sequence a)
variable (h_a1 : a 0 = 2)
variable (h_sum : a 1 + a 3 = 8)

-- The target question
theorem find_a5 : a 4 = 6 :=
by
  sorry

end find_a5_l292_292454


namespace calculate_earths_atmosphere_mass_l292_292764

noncomputable def mass_of_earths_atmosphere (R p0 g : ℝ) : ℝ :=
  (4 * Real.pi * R^2 * p0) / g

theorem calculate_earths_atmosphere_mass (R p0 g : ℝ) (h : 0 < g) : 
  mass_of_earths_atmosphere R p0 g = 5 * 10^18 := 
sorry

end calculate_earths_atmosphere_mass_l292_292764


namespace total_maggots_served_l292_292424

-- Define the conditions in Lean
def maggots_first_attempt : ℕ := 10
def maggots_second_attempt : ℕ := 10

-- Define the statement to prove
theorem total_maggots_served : maggots_first_attempt + maggots_second_attempt = 20 :=
by 
  sorry

end total_maggots_served_l292_292424


namespace average_class_is_45_6_l292_292082

noncomputable def average_class_score (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) 
  (zero_scorers : ℕ) (remaining_students_avg : ℕ) : ℚ :=
  let total_top_score := top_scorers * top_score
  let total_zero_score := zero_scorers * 0
  let remaining_students := total_students - top_scorers - zero_scorers
  let total_remaining_score := remaining_students * remaining_students_avg
  let total_score := total_top_score + total_zero_score + total_remaining_score
  total_score / total_students

theorem average_class_is_45_6 : average_class_score 25 3 95 3 45 = 45.6 := 
by
  -- sorry is used here to skip the proof. Lean will expect a proof here.
  sorry

end average_class_is_45_6_l292_292082


namespace avg_problem_l292_292371

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Formulate the proof problem statement
theorem avg_problem : avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end avg_problem_l292_292371


namespace sum_of_areas_B_D_l292_292574

theorem sum_of_areas_B_D (area_large_square : ℝ) (area_small_square : ℝ) (B D : ℝ) 
  (h1 : area_large_square = 9) 
  (h2 : area_small_square = 1)
  (h3 : B + D = 4) : 
  B + D = 4 := 
by
  sorry

end sum_of_areas_B_D_l292_292574


namespace four_digit_composite_l292_292097

theorem four_digit_composite (abcd : ℕ) (h : 1000 ≤ abcd ∧ abcd < 10000) :
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≥ 2 ∧ m * n = (abcd * 10001) :=
by
  sorry

end four_digit_composite_l292_292097


namespace initial_number_of_friends_l292_292706

theorem initial_number_of_friends (X : ℕ) (H : 3 * (X - 3) = 15) : X = 8 :=
by
  sorry

end initial_number_of_friends_l292_292706


namespace alcohol_quantity_l292_292100

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 2 / 5) (h2 : A / (W + 10) = 2 / 7) : A = 10 :=
by
  sorry

end alcohol_quantity_l292_292100


namespace number_of_moles_H2SO4_formed_l292_292775

-- Define the moles of reactants
def initial_moles_SO2 : ℕ := 1
def initial_moles_H2O2 : ℕ := 1

-- Given the balanced chemical reaction
-- SO2 + H2O2 → H2SO4
def balanced_reaction := (1, 1) -- Representing the reactant coefficients for SO2 and H2O2

-- Define the number of moles of product formed
def moles_H2SO4 (moles_SO2 moles_H2O2 : ℕ) : ℕ :=
moles_SO2 -- Since according to balanced equation, 1 mole of each reactant produces 1 mole of product

theorem number_of_moles_H2SO4_formed :
  moles_H2SO4 initial_moles_SO2 initial_moles_H2O2 = 1 := by
  sorry

end number_of_moles_H2SO4_formed_l292_292775


namespace max_of_three_numbers_l292_292540

theorem max_of_three_numbers : ∀ (a b c : ℕ), a = 10 → b = 11 → c = 12 → max (max a b) c = 12 :=
by
  intros a b c h1 h2 h3
  rw [h1, h2, h3]
  sorry

end max_of_three_numbers_l292_292540


namespace domain_of_function_l292_292690

theorem domain_of_function :
  {x : ℝ | x ≥ -1} \ {0} = {x : ℝ | (x ≥ -1 ∧ x < 0) ∨ x > 0} :=
by
  sorry

end domain_of_function_l292_292690


namespace sum_valid_m_l292_292851

noncomputable def median (s : Finset ℝ) : ℝ := sorry

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / s.card

lemma median_eq_mean {m : ℝ}
  (h : m ≠ 4 ∧ m ≠ 7 ∧ m ≠ 11 ∧ m ≠ 13)
  (s : Finset ℝ := {4, 7, 11, 13}.insert m)
  (hm : median s = mean s) : m = 20 := sorry

theorem sum_valid_m : (∑ (m : ℝ) in {4, 7, 11, 13, 20}, if m ≠ 4 ∧ m ≠ 7 ∧ m ≠ 11 ∧ m ≠ 13 ∧ median {4, 7, 11, 13}.insert m = mean {4, 7, 11, 13}.insert m then m else 0) = 20 :=
by
  sorry

end sum_valid_m_l292_292851


namespace angle_same_terminal_side_l292_292015

theorem angle_same_terminal_side (α θ : ℝ) (hα : α = 1690) (hθ : 0 < θ) (hθ2 : θ < 360) (h_terminal_side : ∃ k : ℤ, α = k * 360 + θ) : θ = 250 :=
by
  sorry

end angle_same_terminal_side_l292_292015


namespace prob_exactly_M_laws_in_concept_l292_292342

theorem prob_exactly_M_laws_in_concept 
  (K N M : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N in
  (nat.choose K M) * q^M * (1 - q)^(K - M) = 
  (nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) :=
by {
  let q := 1 - (1 - p)^N,
  have hq_def : q = 1 - (1 - p)^N := rfl,
  rw [hq_def],
  sorry
}

end prob_exactly_M_laws_in_concept_l292_292342


namespace alix_has_15_more_chocolates_than_nick_l292_292196

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l292_292196


namespace correct_equation_for_gift_exchanges_l292_292893

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l292_292893


namespace nat_pair_count_eq_five_l292_292617

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l292_292617


namespace range_of_a_l292_292329

variable (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0

theorem range_of_a (hpq : p a ∨ q a) (hpnq : ¬p a ∧ ¬q a) : 
  (-1 ≤ a ∧ a ≤ 1) ∨ (a > 3) :=
sorry

end range_of_a_l292_292329


namespace socks_pairing_l292_292798

noncomputable def number_of_ways_to_choose_socks_same_color : Nat :=
  (nat.choose 5 2) + (nat.choose 5 2) + (nat.choose 2 2)

theorem socks_pairing :
  number_of_ways_to_choose_socks_same_color = 21 :=
by
  sorry

end socks_pairing_l292_292798


namespace material_for_7_quilts_l292_292178

theorem material_for_7_quilts (x : ℕ) (h1 : ∀ y : ℕ, y = 7 * x) (h2 : 36 = 12 * x) : 7 * x = 21 := 
by 
  sorry

end material_for_7_quilts_l292_292178


namespace max_elements_in_set_l292_292193

theorem max_elements_in_set (S : Finset ℕ) (hS : ∀ (a b : ℕ), a ≠ b → a ∈ S → b ∈ S → 
  ∃ (k : ℕ) (c d : ℕ), c < d ∧ c ∈ S ∧ d ∈ S ∧ a + b = c^k * d) :
  S.card ≤ 48 :=
sorry

end max_elements_in_set_l292_292193


namespace find_k_l292_292009

-- Define the lines as given in the problem
def line1 (k : ℝ) (x y : ℝ) : Prop := k * x + (1 - k) * y - 3 = 0
def line2 (k : ℝ) (x y : ℝ) : Prop := (k - 1) * x + (2 * k + 3) * y - 2 = 0

-- Define the condition for perpendicular lines
def perpendicular (k : ℝ) : Prop :=
  let slope1 := -k / (1 - k)
  let slope2 := -(k - 1) / (2 * k + 3)
  slope1 * slope2 = -1

-- Problem statement: Prove that the lines are perpendicular implies k == 1 or k == -3
theorem find_k (k : ℝ) : perpendicular k → (k = 1 ∨ k = -3) :=
sorry

end find_k_l292_292009


namespace union_of_sets_l292_292037

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (hA : A = {x, y}) (hB : B = {x + 1, 5}) (h_inter : A ∩ B = {2}) :
  A ∪ B = {1, 2, 5} :=
by
  sorry

end union_of_sets_l292_292037


namespace ratio_eq_two_l292_292163

theorem ratio_eq_two (a b c d : ℤ) (h1 : b * c + a * d = 1) (h2 : a * c + 2 * b * d = 1) : 
  (a^2 + c^2 : ℚ) / (b^2 + d^2) = 2 :=
sorry

end ratio_eq_two_l292_292163


namespace velvet_needed_for_box_l292_292968

theorem velvet_needed_for_box : 
  let area_long_side := 8 * 6
  let area_short_side := 5 * 6
  let area_top_bottom := 40
  let total_area := (2 * area_long_side) + (2 * area_short_side) + (2 * area_top_bottom)
  total_area = 236 :=
by
  sorry

end velvet_needed_for_box_l292_292968


namespace binom_1000_1000_and_999_l292_292759

theorem binom_1000_1000_and_999 :
  (Nat.choose 1000 1000 = 1) ∧ (Nat.choose 1000 999 = 1000) :=
by
  sorry

end binom_1000_1000_and_999_l292_292759


namespace coloring_ways_l292_292435

-- Define the vertices and edges of the graph
def vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}

def edges : Finset (ℕ × ℕ) :=
  { (0, 1), (1, 2), (2, 0),  -- First triangle
    (3, 4), (4, 5), (5, 3),  -- Middle triangle
    (6, 7), (7, 8), (8, 6),  -- Third triangle
    (2, 5),   -- Connecting top horizontal edge
    (1, 7) }  -- Connecting bottom horizontal edge

-- Define the number of colors available
def colors := 4

-- Define a function to count the valid colorings given the vertices and edges
noncomputable def countValidColorings (vertices : Finset ℕ) (edges : Finset (ℕ × ℕ)) (colors : ℕ) : ℕ := sorry

-- The theorem statement
theorem coloring_ways : countValidColorings vertices edges colors = 3456 := 
sorry

end coloring_ways_l292_292435


namespace geometric_proportion_exists_l292_292852

theorem geometric_proportion_exists (x y : ℝ) (h1 : x + (24 - x) = 24) 
  (h2 : y + (16 - y) = 16) (h3 : x^2 + y^2 + (16 - y)^2 + (24 - x)^2 = 580) : 
  (21 / 7 = 9 / 3) :=
  sorry

end geometric_proportion_exists_l292_292852


namespace multiple_of_6_is_multiple_of_3_l292_292245

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) : (∃ k : ℕ, n = 6 * k) → (∃ m : ℕ, n = 3 * m) :=
by
  sorry

end multiple_of_6_is_multiple_of_3_l292_292245


namespace at_least_two_inequalities_hold_l292_292354

variable {a b c : ℝ}

theorem at_least_two_inequalities_hold (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) :=
  sorry

end at_least_two_inequalities_hold_l292_292354


namespace sum_of_squares_of_four_integers_equals_175_l292_292696

theorem sum_of_squares_of_four_integers_equals_175 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a^2 + b^2 + c^2 + d^2 = 175 ∧ a + b + c + d = 23 :=
sorry

end sum_of_squares_of_four_integers_equals_175_l292_292696


namespace smallest_square_l292_292243

theorem smallest_square 
  (a b : ℕ) 
  (h1 : 15 * a + 16 * b = m ^ 2) 
  (h2 : 16 * a - 15 * b = n ^ 2)
  (hm : m > 0) 
  (hn : n > 0) : 
  min (15 * a + 16 * b) (16 * a - 15 * b) = 481 ^ 2 := 
sorry

end smallest_square_l292_292243


namespace calculated_area_error_l292_292700

def percentage_error_area (initial_length_error : ℝ) (initial_width_error : ℝ) 
(temperature_change : ℝ) (humidity_change : ℝ) 
(length_error_per_temp : ℝ) (width_error_per_humidity : ℝ) : ℝ :=
let total_length_error := initial_length_error + (temperature_change / 5) * length_error_per_temp in
let total_width_error := initial_width_error + (humidity_change / 10) * width_error_per_humidity in
total_length_error - total_width_error

theorem calculated_area_error :
  percentage_error_area 3 2 15 20 1 0.5 = 3 :=
sorry

end calculated_area_error_l292_292700


namespace people_to_left_of_kolya_l292_292983

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l292_292983


namespace line_equation_l292_292871

theorem line_equation (b r S : ℝ) (h : ℝ) (m : ℝ) (eq_one : S = 1/2 * b * h) (eq_two : h = 2*S / b) (eq_three : |m| = r / b) 
  (eq_four : m = r / b) : 
  (∀ x y : ℝ, y = m * (x - b) → b > 0 → r > 0 → S > 0 → rx - bry - rb = 0) := 
sorry

end line_equation_l292_292871


namespace sums_equal_l292_292751

theorem sums_equal (A B C : Type) (a b c : ℕ) :
  (a + b + c) = (a + (b + c)) ∧
  (a + b + c) = (b + (c + a)) ∧
  (a + b + c) = (c + (a + b)) :=
by 
  sorry

end sums_equal_l292_292751


namespace total_employees_in_company_l292_292407

-- Given facts and conditions
def ratio_A_B_C : Nat × Nat × Nat := (5, 4, 1)
def sample_size : Nat := 20
def prob_sel_A_B_from_C : ℚ := 1 / 45

-- Number of group C individuals, calculated from probability constraint
def num_persons_group_C := 10

theorem total_employees_in_company (x : Nat) :
  x = 10 * (5 + 4 + 1) :=
by
  -- Since the sample size is 20, and the ratio of sampling must be consistent with the population ratio,
  -- it can be derived that the total number of employees in the company must be 100.
  -- Adding sorry to skip the actual detailed proof.
  sorry

end total_employees_in_company_l292_292407


namespace tangent_line_eq_l292_292791

noncomputable section

open Function

def f (x : ℝ) := x * Real.log x

theorem tangent_line_eq :
  (∃ (x_0 : ℝ) (y_0 : ℝ), y_0 = f x_0 ∧ TangentLineOf f x_0 (0, -1) = (1, -1, 1)) :=
sorry

end tangent_line_eq_l292_292791


namespace gathering_gift_exchange_l292_292896

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l292_292896


namespace sqrt_mul_neg_eq_l292_292579

theorem sqrt_mul_neg_eq : - (Real.sqrt 2) * (Real.sqrt 7) = - (Real.sqrt 14) := sorry

end sqrt_mul_neg_eq_l292_292579


namespace max_value_of_expr_l292_292216

open Classical
open Real

theorem max_value_of_expr 
  (x y : ℝ) 
  (h₁ : 0 < x) 
  (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  ∃ a b c d : ℝ, 
    (x^2 + 2 * x * y + 3 * y^2 = 20 + 10 * sqrt 3) ∧ 
    (a = 20) ∧ 
    (b = 10) ∧ 
    (c = 3) ∧ 
    (d = 2) := 
sorry

end max_value_of_expr_l292_292216


namespace part1_f_inequality_part2_a_range_l292_292507

open Real

-- Proof Problem 1
theorem part1_f_inequality (x : ℝ) : 
    (|x - 1| + |x + 1| ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5) :=
sorry

-- Proof Problem 2
theorem part2_a_range (a : ℝ) : 
    (∀ x : ℝ, |x - 1| + |x - a| ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end part1_f_inequality_part2_a_range_l292_292507


namespace min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l292_292315

noncomputable def line_equation (A B C x y : ℝ) : Prop := A * x + B * y + C = 0

noncomputable def point_on_line (x y A B C : ℝ) : Prop := line_equation A B C x y

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  (|C2 - C1|) / (Real.sqrt (A^2 + B^2))

theorem min_distance_between_parallel_lines :
  ∀ (A B C1 C2 x y : ℝ),
  point_on_line x y A B C1 ∧ point_on_line x y A B C2 →
  distance_between_parallel_lines A B C1 C2 = 3 :=
by
  intros A B C1 C2 x y h
  sorry

theorem distance_when_line_parallel_to_x_axis :
  ∀ (x1 x2 y k A B C1 C2 : ℝ),
  k = 3 →
  point_on_line x1 k A B C1 →
  point_on_line x2 k A B C2 →
  |x2 - x1| = 5 :=
by
  intros x1 x2 y k A B C1 C2 hk h1 h2
  sorry

end min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l292_292315


namespace sophie_marbles_probability_l292_292057

theorem sophie_marbles_probability :
  let blue_marbles := 10
  let red_marbles := 5
  let total_marbles := blue_marbles + red_marbles
  let withdraws := 8
  let exact_four_blue : ℝ :=
    (nat.choose withdraws 4).to_real *
    ((blue_marbles / total_marbles) ^ 4 * (red_marbles / total_marbles) ^ 4)
  let no_red : ℝ := (blue_marbles / total_marbles) ^ withdraws
  ((exact_four_blue * 1000.0).round / 1000.0) - ((no_red * 1000.0).round / 1000.0) = 0.131 :=
by
  sorry

end sophie_marbles_probability_l292_292057


namespace shaded_grid_percentage_l292_292811

theorem shaded_grid_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 64) (h2 : shaded_squares = 48) : 
  ((shaded_squares : ℚ) / (total_squares : ℚ)) * 100 = 75 :=
by
  rw [h1, h2]
  norm_num

end shaded_grid_percentage_l292_292811


namespace terminating_decimals_count_l292_292137

theorem terminating_decimals_count :
  (∃ count : ℕ, count = 166 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → (∃ m : ℕ, n = 3 * m)) :=
sorry

end terminating_decimals_count_l292_292137


namespace same_exponent_for_all_bases_l292_292716

theorem same_exponent_for_all_bases {a : Type} [LinearOrderedField a] {C : a} (ha : ∀ (a : a), a ≠ 0 → a^0 = C) : C = 1 :=
by
  sorry

end same_exponent_for_all_bases_l292_292716


namespace towers_remainder_l292_292732

noncomputable def count_towers (k : ℕ) : ℕ := sorry

theorem towers_remainder : (count_towers 9) % 1000 = 768 := sorry

end towers_remainder_l292_292732


namespace arithmetic_value_l292_292248

theorem arithmetic_value : (8 * 4) + 3 = 35 := by
  sorry

end arithmetic_value_l292_292248


namespace geometry_problems_l292_292997

-- Given conditions
structure TriangleConfig where
  A B C D E F : Point
  AB BC : ℝ
  h1 : circle_through A B tangent BC
  h2 : circle_through B C tangent AB
  h3 : chord BD intersect AC at E
  h4 : chord AD intersects_other_circle_at F
  h5 : AB = 5
  h6 : BC = 9

-- Lean 4 statement for proving the ratio of AE to EC
def ratio_AE_EC (cfg : TriangleConfig) : Prop :=
  (segment_ratio cfg.AE cfg.EC) = (25 / 81)

-- Lean 4 statement for proving the areas of triangles are equal
def areas_equal (cfg : TriangleConfig) : Prop :=
  (area_triangle cfg.A cfg.B cfg.C) = (area_triangle cfg.A cfg.B cfg.F)

theorem geometry_problems (cfg : TriangleConfig) : ratio_AE_EC cfg ∧ areas_equal cfg :=
by
  sorry

end geometry_problems_l292_292997


namespace cube_weight_doubled_side_length_l292_292734

-- Theorem: Prove that the weight of a new cube with sides twice as long as the original cube is 40 pounds, given the conditions.
theorem cube_weight_doubled_side_length (s : ℝ) (h₁ : s > 0) (h₂ : (s^3 : ℝ) > 0) (w : ℝ) (h₃ : w = 5) : 
  8 * w = 40 :=
by
  sorry

end cube_weight_doubled_side_length_l292_292734


namespace clownfish_ratio_l292_292291

theorem clownfish_ratio (C B : ℕ) (h₁ : C = B) (h₂ : C + B = 100) (h₃ : C = B) : 
  (let B := 50; 
  let initially_clownfish := B - 26; -- Number of clownfish that initially joined display tank
  let swam_back := (B - 26) - 16; -- Number of clownfish that swam back
  initially_clownfish > 0 → 
  swam_back > 0 → 
  (swam_back : ℚ) / (initially_clownfish : ℚ) = 1 / 3) :=
by 
  sorry

end clownfish_ratio_l292_292291


namespace handshake_count_l292_292805

theorem handshake_count : 
  let n := 5  -- number of representatives per company
  let c := 5  -- number of companies
  let total_people := n * c  -- total number of people
  let handshakes_per_person := total_people - n  -- each person shakes hands with 20 others
  (total_people * handshakes_per_person) / 2 = 250 := 
by
  sorry

end handshake_count_l292_292805


namespace prob_exactly_M_laws_in_concept_l292_292341

theorem prob_exactly_M_laws_in_concept 
  (K N M : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N in
  (nat.choose K M) * q^M * (1 - q)^(K - M) = 
  (nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) :=
by {
  let q := 1 - (1 - p)^N,
  have hq_def : q = 1 - (1 - p)^N := rfl,
  rw [hq_def],
  sorry
}

end prob_exactly_M_laws_in_concept_l292_292341


namespace simplify_fraction_l292_292678

variable (x y : ℕ)

theorem simplify_fraction (hx : x = 3) (hy : y = 2) :
  (12 * x^2 * y^3) / (9 * x * y^2) = 8 :=
by
  sorry

end simplify_fraction_l292_292678


namespace Mike_age_l292_292358

-- We define the ages of Mike and Barbara
variables (M B : ℕ)

-- Conditions extracted from the problem
axiom h1 : B = M / 2
axiom h2 : M - B = 8

-- The theorem to prove
theorem Mike_age : M = 16 :=
by sorry

end Mike_age_l292_292358


namespace remainder_product_l292_292041

theorem remainder_product (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := 
by 
  sorry

end remainder_product_l292_292041


namespace max_c_friendly_value_l292_292817

def is_c_friendly (c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| ≤ c * |x - y|

theorem max_c_friendly_value (c : ℝ) (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  c > 1 → is_c_friendly c f → |f x - f y| ≤ (c + 1) / 2 :=
sorry

end max_c_friendly_value_l292_292817


namespace cost_increase_percentage_l292_292491

theorem cost_increase_percentage 
  (C S : ℝ) (X : ℝ)
  (h_proft : S = 2.6 * C)
  (h_new_profit : 1.6 * C - (X / 100) * C = 0.5692307692307692 * S) :
  X = 12 := 
by
  sorry

end cost_increase_percentage_l292_292491


namespace strictly_increasing_seqs_count_l292_292595

theorem strictly_increasing_seqs_count : 
  ∑ k in (finset.range 9).filter (λ k, k ≥ 2), nat.choose 9 k = 502 := by
  sorry

end strictly_increasing_seqs_count_l292_292595


namespace rachelle_meat_needed_l292_292516

-- Define the ratio of meat per hamburger
def meat_per_hamburger (pounds : ℕ) (hamburgers : ℕ) : ℚ :=
  pounds / hamburgers

-- Define the total meat needed for a given number of hamburgers
def total_meat (meat_per_hamburger : ℚ) (hamburgers : ℕ) : ℚ :=
  meat_per_hamburger * hamburgers

-- Prove that Rachelle needs 15 pounds of meat to make 36 hamburgers
theorem rachelle_meat_needed : total_meat (meat_per_hamburger 5 12) 36 = 15 := by
  sorry

end rachelle_meat_needed_l292_292516


namespace fixed_points_bound_l292_292960

-- Define the polynomial P and the degree condition
variable {R : Type*} [CommRing R]
variable (P : R[X])

-- Assume that P has integer coefficients and a degree of at least 2
noncomputable def polynomial_has_integer_coefficients_and_degree_n (P : ℤ[X]) (n : ℕ) : Prop :=
  P.degree = n ∧ n ≥ 2

-- Q is defined as P applied k times
def Q (k : ℕ) (P : R[X]) : R[X] :=
  Nat.iterate P.eval k

-- Main statement
theorem fixed_points_bound (P : ℤ[X]) (n k : ℕ) (h : polynomial_has_integer_coefficients_and_degree_n P n) :
  ∃ m ≤ n, ∀ a, Q k P a = a → ∑ i in Finset.range m, 1 = n :=
by
  sorry

end fixed_points_bound_l292_292960


namespace circle_with_diameter_AB_tangent_parabola_l292_292910

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def tangent_line (x1 y1 x y : ℝ) : Prop := x1 * x - 2 * y - 2 * y1 = 0

noncomputable def tangent_through_H (x1 y1 : ℝ) : Prop := let H := (1 : ℝ, -1 : ℝ)
  in x1 * H.1 - 2 * H.2 - 2 * y1 = 2

noncomputable def line_AB (x y : ℝ) : Prop := x - 2 * y + 2 = 0

noncomputable def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 3/2)^2 = 25/4

theorem circle_with_diameter_AB_tangent_parabola :
  ∃ x1 y1 x2 y2 : ℝ, parabola x1 y1 ∧ parabola x2 y2 ∧ tangent_through_H x1 y1 ∧
                  tangent_through_H x2 y2 ∧ line_AB x1 y1 ∧ line_AB x2 y2 ∧ circle x1 y1 ∧ circle x2 y2 :=
by {
  sorry
}

end circle_with_diameter_AB_tangent_parabola_l292_292910


namespace relationship_roots_geometric_progression_l292_292803

theorem relationship_roots_geometric_progression 
  (x y z p q r : ℝ)
  (h1 : x^2 ≠ y^2 ∧ y^2 ≠ z^2 ∧ x^2 ≠ z^2) -- Distinct non-zero numbers
  (h2 : y^2 = x^2 * r)
  (h3 : z^2 = y^2 * r)
  (h4 : x + y + z = p)
  (h5 : x * y + y * z + z * x = q)
  (h6 : x * y * z = r) : r^2 = 1 := sorry

end relationship_roots_geometric_progression_l292_292803


namespace find_s_l292_292303

theorem find_s : ∃ s : ℚ, (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) ∧ s = -95 / 9 := sorry

end find_s_l292_292303


namespace crayons_end_of_school_year_l292_292045

-- Definitions based on conditions
def crayons_after_birthday : Float := 479.0
def total_crayons_now : Float := 613.0

-- The mathematically equivalent proof problem statement
theorem crayons_end_of_school_year : (total_crayons_now - crayons_after_birthday = 134.0) :=
by
  sorry

end crayons_end_of_school_year_l292_292045


namespace ratio_AR_AU_l292_292031

-- Define the conditions in the problem as variables and constraints
variables (A B C P Q U R : Type)
variables (AP PB AQ QC : ℝ)
variables (angle_bisector_AU : A -> U)
variables (intersect_AU_PQ_at_R : A -> U -> P -> Q -> R)

-- Assuming the given distances
def conditions (AP PB AQ QC : ℝ) : Prop :=
  AP = 2 ∧ PB = 6 ∧ AQ = 4 ∧ QC = 5

-- The statement to prove
theorem ratio_AR_AU (h : conditions AP PB AQ QC) : 
  (AR / AU) = 108 / 289 :=
sorry

end ratio_AR_AU_l292_292031


namespace quotient_of_powers_l292_292853

theorem quotient_of_powers:
  (50 : ℕ) = 2 * 5^2 →
  (25 : ℕ) = 5^2 →
  (50^50 / 25^25 : ℕ) = 100^25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end quotient_of_powers_l292_292853


namespace angle_ACB_33_l292_292822

noncomputable def triangle_ABC : Type := sorry  -- Define the triangle ABC
noncomputable def ω : Type := sorry  -- Define the circumcircle of ABC
noncomputable def M : Type := sorry  -- Define the midpoint of arc BC not containing A
noncomputable def D : Type := sorry  -- Define the point D such that DM is tangent to ω
def AM_eq_AC : Prop := sorry  -- Define the equality AM = AC
def angle_DMC := (38 : ℝ)  -- Define angle DMC = 38 degrees

theorem angle_ACB_33 (h1 : triangle_ABC) 
                      (h2 : ω) 
                      (h3 : M) 
                      (h4 : D) 
                      (h5 : AM_eq_AC)
                      (h6 : angle_DMC = 38) : ∃ θ, (θ = 33) ∧ (angle_ACB = θ) :=
sorry  -- Proof goes here

end angle_ACB_33_l292_292822


namespace total_number_of_girls_in_school_l292_292561

theorem total_number_of_girls_in_school 
  (students_sampled : ℕ) 
  (students_total : ℕ) 
  (sample_girls : ℕ) 
  (sample_boys : ℕ)
  (h_sample_size : students_sampled = 200)
  (h_total_students : students_total = 2000)
  (h_diff_girls_boys : sample_boys = sample_girls + 6)
  (h_stratified_sampling : students_sampled / students_total = 200 / 2000) :
  sample_girls * (students_total / students_sampled) = 970 :=
by
  sorry

end total_number_of_girls_in_school_l292_292561


namespace two_dollar_coin_is_toonie_l292_292293

/-- We define the $2 coin in Canada -/
def two_dollar_coin_name : String := "toonie"

/-- Antonella's wallet problem setup -/
def Antonella_has_ten_coins := 10
def loonies_value := 1
def toonies_value := 2
def coins_after_purchase := 11
def purchase_amount := 3
def initial_toonies := 4

/-- Proving that the $2 coin is called a "toonie" -/
theorem two_dollar_coin_is_toonie :
  two_dollar_coin_name = "toonie" :=
by
  -- Here, we place the logical steps to derive that two_dollar_coin_name = "toonie"
  sorry

end two_dollar_coin_is_toonie_l292_292293


namespace birds_on_fence_l292_292370

theorem birds_on_fence (B S : ℕ): 
  S = 3 →
  S + 6 = B + 5 →
  B = 4 :=
by
  intros h1 h2
  sorry

end birds_on_fence_l292_292370


namespace abs_inequality_solution_l292_292369

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ -9 / 2 < x ∧ x < 7 / 2 :=
by
  sorry

end abs_inequality_solution_l292_292369


namespace max_ab_l292_292793

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 6) : ab ≤ 9 / 2 :=
by
  sorry

end max_ab_l292_292793


namespace gift_exchange_equation_l292_292885

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l292_292885


namespace max_probability_of_winning_is_correct_l292_292828

noncomputable def max_probability_of_winning : ℚ :=
  sorry

theorem max_probability_of_winning_is_correct :
  max_probability_of_winning = 17 / 32 :=
sorry

end max_probability_of_winning_is_correct_l292_292828


namespace remainder_of_sum_l292_292133

theorem remainder_of_sum (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = 7145) (h2 : n2 = 7146)
  (h3 : n3 = 7147) (h4 : n4 = 7148) (h5 : n5 = 7149) :
  ((n1 + n2 + n3 + n4 + n5) % 8) = 7 :=
by sorry

end remainder_of_sum_l292_292133


namespace minimum_arg_z_l292_292464

open Complex Real

noncomputable def z_cond (z : ℂ) := abs (z + 3 - (complex.I * sqrt 3)) = sqrt 3

theorem minimum_arg_z : ∀ z : ℂ, z_cond z → arg z = 5 / 6 * π :=
by
  intros
  sorry

end minimum_arg_z_l292_292464


namespace distance_between_A_and_B_l292_292110

theorem distance_between_A_and_B :
  let A := (0, 0)
  let B := (-10, 24)
  dist A B = 26 :=
by
  sorry

end distance_between_A_and_B_l292_292110


namespace triangle_shortest_side_l292_292378

theorem triangle_shortest_side (x y z : ℝ) (h : x / y = 1 / 2) (h1 : x / z = 1 / 3) (hyp : x = 6) : z = 3 :=
sorry

end triangle_shortest_side_l292_292378


namespace opposite_of_neg_one_over_2023_l292_292240

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l292_292240


namespace fraction_evaluation_l292_292474

def h (x : ℤ) : ℤ := 3 * x + 4
def k (x : ℤ) : ℤ := 4 * x - 3

theorem fraction_evaluation :
  (h (k (h 3))) / (k (h (k 3))) = 151 / 121 :=
by sorry

end fraction_evaluation_l292_292474


namespace fraction_product_simplified_l292_292577

theorem fraction_product_simplified :
  (1 / 3) * (4 / 7) * (9 / 11) = 12 / 77 :=
by
  -- Here, we add the proof steps
  sorry

end fraction_product_simplified_l292_292577


namespace depth_of_second_hole_l292_292401

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let man_hours1 := workers1 * hours1 -- 360 man-hours
  let workers2 := 45 + 35 -- 80 workers
  let hours2 := 6
  let man_hours2 := workers2 * hours2 -- 480 man-hours
  let depth2 := (man_hours2 * depth1) / man_hours1 -- value to solve for
  depth2 = 40 :=
by
  sorry

end depth_of_second_hole_l292_292401


namespace solve_system_eq_l292_292767

theorem solve_system_eq (x y : ℚ) 
  (h1 : 3 * x - 7 * y = 31) 
  (h2 : 5 * x + 2 * y = -10) : 
  x = -336 / 205 := 
sorry

end solve_system_eq_l292_292767


namespace speed_in_still_water_l292_292396

def upstream_speed : ℝ := 35
def downstream_speed : ℝ := 45

theorem speed_in_still_water:
  (upstream_speed + downstream_speed) / 2 = 40 := 
by
  sorry

end speed_in_still_water_l292_292396


namespace quadratic_root_eq_l292_292449

theorem quadratic_root_eq {b : ℝ} (h : (2 : ℝ)^2 + b * 2 - 6 = 0) : b = 1 :=
by
  sorry

end quadratic_root_eq_l292_292449


namespace function_identity_l292_292364

theorem function_identity
    (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x ≤ x)
    (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
    ∀ x : ℝ, f x = x :=
by
    sorry

end function_identity_l292_292364


namespace find_a_minus_b_l292_292320

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 3 * a * x + 4

-- Define the condition for the function being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the function f(x) with given parameters
theorem find_a_minus_b (a b : ℝ) (h_dom_range : ∀ x : ℝ, b - 3 ≤ x → x ≤ 2 * b) (h_even_f : is_even (f a)) :
  a - b = -1 :=
  sorry

end find_a_minus_b_l292_292320


namespace add_to_37_eq_52_l292_292107

theorem add_to_37_eq_52 (x : ℕ) (h : 37 + x = 52) : x = 15 := by
  sorry

end add_to_37_eq_52_l292_292107


namespace alix_more_chocolates_than_nick_l292_292201

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l292_292201


namespace employee_B_paid_l292_292850

variable (A B : ℝ)

/-- Two employees A and B are paid a total of Rs. 550 per week by their employer. 
A is paid 120 percent of the sum paid to B. -/
theorem employee_B_paid (h₁ : A + B = 550) (h₂ : A = 1.2 * B) : B = 250 := by
  -- Proof will go here
  sorry

end employee_B_paid_l292_292850


namespace velvet_needed_for_box_l292_292969

theorem velvet_needed_for_box : 
  let area_long_side := 8 * 6
  let area_short_side := 5 * 6
  let area_top_bottom := 40
  let total_area := (2 * area_long_side) + (2 * area_short_side) + (2 * area_top_bottom)
  total_area = 236 :=
by
  sorry

end velvet_needed_for_box_l292_292969


namespace rent_expense_l292_292675

theorem rent_expense (salary gross: ℕ) (tax_percentage: ℕ) (rent_months: ℕ) :
  gross = 5000 → tax_percentage = 10 → rent_months = 2 → salary = gross * (100 - tax_percentage) / 100 → 
  (3 * salary / 5) / rent_months = 1350 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end rent_expense_l292_292675


namespace tens_digit_of_13_pow_2023_l292_292766

theorem tens_digit_of_13_pow_2023 :
  ∀ (n : ℕ), (13 ^ (2023 % 20) ≡ 13 ^ n [MOD 100]) ∧ (13 ^ n ≡ 97 [MOD 100]) → (13 ^ 2023) % 100 / 10 % 10 = 9 :=
by
sorry

end tens_digit_of_13_pow_2023_l292_292766


namespace range_of_b_l292_292629

theorem range_of_b (b : ℝ) (hb : b > 0) : (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) :=
by
  sorry

end range_of_b_l292_292629


namespace not_forall_abs_ge_zero_l292_292330

theorem not_forall_abs_ge_zero : (¬(∀ x : ℝ, |x + 1| ≥ 0)) ↔ (∃ x : ℝ, |x + 1| < 0) :=
by
  sorry

end not_forall_abs_ge_zero_l292_292330


namespace rhombus_longer_diagonal_l292_292220

theorem rhombus_longer_diagonal (d1 d2 : ℝ) (h_d1 : d1 = 11) (h_area : (d1 * d2) / 2 = 110) : d2 = 20 :=
by
  sorry

end rhombus_longer_diagonal_l292_292220


namespace arithmetic_sequence_a2_a4_a9_eq_18_l292_292790

theorem arithmetic_sequence_a2_a4_a9_eq_18 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : S 9 = 54) 
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 2 + a 4 + a 9 = 18 :=
sorry

end arithmetic_sequence_a2_a4_a9_eq_18_l292_292790


namespace tree_planting_total_l292_292414

theorem tree_planting_total (t4 t5 t6 : ℕ) 
  (h1 : t4 = 30)
  (h2 : t5 = 2 * t4)
  (h3 : t6 = 3 * t5 - 30) : 
  t4 + t5 + t6 = 240 := 
by 
  sorry

end tree_planting_total_l292_292414


namespace gecko_cricket_eating_l292_292867

theorem gecko_cricket_eating :
  ∀ (total_crickets : ℕ) (first_day_percent : ℚ) (second_day_less : ℕ),
    total_crickets = 70 →
    first_day_percent = 0.3 →
    second_day_less = 6 →
    let first_day_crickets := total_crickets * first_day_percent
    let second_day_crickets := first_day_crickets - second_day_less
    total_crickets - first_day_crickets - second_day_crickets = 34 :=
by
  intros total_crickets first_day_percent second_day_less h_total h_percent h_less
  let first_day_crickets := total_crickets * first_day_percent
  let second_day_crickets := first_day_crickets - second_day_less
  have : total_crickets - first_day_crickets - second_day_crickets = 34 := sorry
  exact this

end gecko_cricket_eating_l292_292867


namespace number_of_proper_subsets_of_P_l292_292040

theorem number_of_proper_subsets_of_P (P : Set ℝ) (hP : P = {x | x^2 = 1}) : 
  (∃ n, n = 2 ∧ ∃ k, k = 2 ^ n - 1 ∧ k = 3) :=
by
  sorry

end number_of_proper_subsets_of_P_l292_292040


namespace find_x0_and_m_l292_292006

theorem find_x0_and_m (x : ℝ) (m : ℝ) (x0 : ℝ) :
  (abs (x + 3) - 2 * x - 1 < 0 ↔ x > 2) ∧ 
  (∃ x, abs (x - m) + abs (x + 1 / m) - 2 = 0) → 
  (x0 = 2 ∧ m = 1) := 
by
  sorry

end find_x0_and_m_l292_292006


namespace chord_property_l292_292602

noncomputable def chord_length (R r k : ℝ) : Prop :=
  k = 2 * Real.sqrt (R^2 - r^2)

theorem chord_property (P O : Point) (R k : ℝ) (hR : 0 < R) (hk : 0 < k) :
  ∃ r, r = Real.sqrt (R^2 - k^2 / 4) ∧ chord_length R r k :=
sorry

end chord_property_l292_292602


namespace probability_two_painted_and_none_painted_l292_292760

theorem probability_two_painted_and_none_painted : 
  let total_cubes := 27
  let cubes_two_painted_faces := 4
  let cubes_no_painted_faces := 9
  let total_ways := Nat.choose total_cubes 2
  let favorable_outcomes := cubes_two_painted_faces * cubes_no_painted_faces
  (favorable_outcomes : ℚ) / (total_ways : ℚ) = 4 / 39 := 
by
  -- Definitions
  let total_cubes := 27
  let cubes_two_painted_faces := 4
  let cubes_no_painted_faces := 9
  let total_ways := Nat.choose total_cubes 2
  let favorable_outcomes := cubes_two_painted_faces * cubes_no_painted_faces
  
  -- Probability Calculation
  have h1 : (favorable_outcomes : ℚ) / (total_ways : ℚ) = (cubes_two_painted_faces * cubes_no_painted_faces : ℚ) / (Nat.choose total_cubes 2 : ℚ),
    from sorry, -- Placeholder for the proof step equivalent to the calculation in the solution.

  have h2 : (favorable_outcomes : ℚ) / (total_ways : ℚ) = 4 / 39 := sorry, -- Placeholder for simplifying the fraction.
  exact h2 

end probability_two_painted_and_none_painted_l292_292760


namespace find_P_l292_292899

variable (P : ℕ) 

-- Conditions
def cost_samosas : ℕ := 3 * 2
def cost_mango_lassi : ℕ := 2
def cost_per_pakora : ℕ := 3
def total_cost : ℕ := 25
def tip_rate : ℚ := 0.25

-- Total cost before tip
def total_cost_before_tip (P : ℕ) : ℕ := cost_samosas + cost_mango_lassi + cost_per_pakora * P

-- Total cost with tip
def total_cost_with_tip (P : ℕ) : ℚ := 
  (total_cost_before_tip P : ℚ) + (tip_rate * total_cost_before_tip P : ℚ)

-- Proof Goal
theorem find_P (h : total_cost_with_tip P = total_cost) : P = 4 :=
by
  sorry

end find_P_l292_292899


namespace at_most_one_negative_l292_292253

theorem at_most_one_negative (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (a < 0 ∧ b >= 0 ∧ c >= 0) ∨ (a >= 0 ∧ b < 0 ∧ c >= 0) ∨ (a >= 0 ∧ b >= 0 ∧ c < 0) ∨ 
  (a >= 0 ∧ b >= 0 ∧ c >= 0) :=
sorry

end at_most_one_negative_l292_292253


namespace gathering_gift_exchange_l292_292898

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l292_292898


namespace peter_read_more_books_l292_292512

/-
Given conditions:
  Peter has 20 books.
  Peter has read 40% of them.
  Peter's brother has read 10% of them.
We aim to prove that Peter has read 6 more books than his brother.
-/

def total_books : ℕ := 20
def peter_read_fraction : ℚ := 0.4
def brother_read_fraction : ℚ := 0.1

def books_read_by_peter := total_books * peter_read_fraction
def books_read_by_brother := total_books * brother_read_fraction

theorem peter_read_more_books :
  books_read_by_peter - books_read_by_brother = 6 := by
  sorry

end peter_read_more_books_l292_292512


namespace gecko_third_day_crickets_l292_292868

def total_crickets : ℕ := 70
def first_day_percentage : ℝ := 0.30
def first_day_crickets : ℝ := first_day_percentage * total_crickets
def second_day_crickets : ℝ := first_day_crickets - 6
def third_day_crickets : ℝ := total_crickets - (first_day_crickets + second_day_crickets)

theorem gecko_third_day_crickets :
  third_day_crickets = 34 :=
by
  sorry

end gecko_third_day_crickets_l292_292868


namespace molecular_weight_CuCO3_8_moles_l292_292712

-- Definitions for atomic weights
def atomic_weight_Cu : ℝ := 63.55
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Definition for the molecular formula of CuCO3
def molecular_weight_CuCO3 :=
  atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O

-- Number of moles
def moles : ℝ := 8

-- Total weight of 8 moles of CuCO3
def total_weight := moles * molecular_weight_CuCO3

-- Proof statement
theorem molecular_weight_CuCO3_8_moles :
  total_weight = 988.48 :=
  by
  sorry

end molecular_weight_CuCO3_8_moles_l292_292712


namespace find_non_negative_integer_pairs_l292_292592

theorem find_non_negative_integer_pairs (m n : ℕ) :
  3 * 2^m + 1 = n^2 ↔ (m = 0 ∧ n = 2) ∨ (m = 3 ∧ n = 5) ∨ (m = 4 ∧ n = 7) := by
  sorry

end find_non_negative_integer_pairs_l292_292592


namespace courtyard_length_l292_292089

theorem courtyard_length (width_of_courtyard : ℝ) (brick_length_cm brick_width_cm : ℝ) (total_bricks : ℕ) (H1 : width_of_courtyard = 16) (H2 : brick_length_cm = 20) (H3 : brick_width_cm = 10) (H4 : total_bricks = 20000) :
  ∃ length_of_courtyard : ℝ, length_of_courtyard = 25 := 
by
  -- variables and hypotheses
  let brick_length_m := brick_length_cm / 100
  let brick_width_m := brick_width_cm / 100
  let area_one_brick := brick_length_m * brick_width_m
  let total_area := total_bricks * area_one_brick
  have width_of_courtyard_val : width_of_courtyard = 16 := H1
  have brick_length_cm_val : brick_length_cm = 20 := H2
  have brick_width_cm_val : brick_width_cm = 10 := H3
  have total_bricks_val : total_bricks = 20000 := H4
  let length_of_courtyard := total_area / width_of_courtyard
  have length_courtyard_val : length_of_courtyard = 25 := sorry
  use length_of_courtyard,
  exact length_courtyard_val sorry

end courtyard_length_l292_292089


namespace remainder_difference_l292_292628

theorem remainder_difference :
  ∃ (d r: ℤ), (1 < d) ∧ (1250 % d = r) ∧ (1890 % d = r) ∧ (2500 % d = r) ∧ (d - r = 10) :=
sorry

end remainder_difference_l292_292628


namespace exists_integers_gcd_eq_one_addition_l292_292351

theorem exists_integers_gcd_eq_one_addition 
  (n k : ℕ) 
  (hnk_pos : n > 0 ∧ k > 0) 
  (hn_even_or_nk_even : (¬ n % 2 = 0) ∨ (n % 2 = 0 ∧ k % 2 = 0)) :
  ∃ a b : ℤ, Int.gcd a ↑n = 1 ∧ Int.gcd b ↑n = 1 ∧ k = a + b :=
by
  sorry

end exists_integers_gcd_eq_one_addition_l292_292351


namespace game_A_probability_greater_than_B_l292_292729

-- Defining the probabilities of heads and tails for the biased coin
def prob_heads : ℚ := 2 / 3
def prob_tails : ℚ := 1 / 3

-- Defining the winning probabilities for Game A
def prob_winning_A : ℚ := (prob_heads^4) + (prob_tails^4)

-- Defining the winning probabilities for Game B
def prob_winning_B : ℚ := (prob_heads^3 * prob_tails) + (prob_tails^3 * prob_heads)

-- The statement we want to prove
theorem game_A_probability_greater_than_B : prob_winning_A - prob_winning_B = 7 / 81 := by
  sorry

end game_A_probability_greater_than_B_l292_292729


namespace find_range_of_a_l292_292908

-- Define the operation ⊗ on ℝ: x ⊗ y = x(1 - y)
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the inequality condition for all real numbers x
def inequality_condition (a : ℝ) : Prop :=
  ∀ (x : ℝ), tensor (x - a) (x + 1) < 1

-- State the theorem to prove the range of a
theorem find_range_of_a (a : ℝ) (h : inequality_condition a) : -2 < a ∧ a < 2 :=
  sorry

end find_range_of_a_l292_292908


namespace alix_more_chocolates_than_nick_l292_292203

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l292_292203


namespace closest_integer_to_cbrt_1728_l292_292262

theorem closest_integer_to_cbrt_1728 : 
  let x := 1728 in closest_integer (real.cbrt x) = 12 :=
by
  sorry

end closest_integer_to_cbrt_1728_l292_292262


namespace attendees_gift_exchange_l292_292890

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l292_292890


namespace emily_lives_l292_292590

theorem emily_lives :
  ∃ (lives_gained : ℕ), 
    let initial_lives := 42
    let lives_lost := 25
    let lives_after_loss := initial_lives - lives_lost
    let final_lives := 41
    lives_after_loss + lives_gained = final_lives :=
sorry

end emily_lives_l292_292590


namespace inequality_holds_l292_292658

variable {a b c : ℝ}

theorem inequality_holds (h : a > 0) (h' : b > 0) (h'' : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end inequality_holds_l292_292658


namespace smallest_distance_AB_ge_2_l292_292036

noncomputable def A (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9
noncomputable def B (x y : ℝ) : Prop := y^2 = -8 * x

theorem smallest_distance_AB_ge_2 :
  ∀ (x1 y1 x2 y2 : ℝ), A x1 y1 → B x2 y2 → dist (x1, y1) (x2, y2) ≥ 2 := by
  sorry

end smallest_distance_AB_ge_2_l292_292036


namespace intersection_M_N_complement_N_U_l292_292355

-- Definitions for the sets and the universal set
def U := Set ℝ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def N : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) } -- Simplified domain interpretation for N

-- Intersection and complement calculations
theorem intersection_M_N (x : ℝ) : x ∈ M ∧ x ∈ N ↔ x ∈ { x | -2 ≤ x ∧ x ≤ 1 } := by sorry

theorem complement_N_U (x : ℝ) : x ∉ N ↔ x ∈ { x | x > 1 } := by sorry

end intersection_M_N_complement_N_U_l292_292355


namespace velocity_equal_distance_l292_292900

theorem velocity_equal_distance (v t : ℝ) (h : v * t = t) (ht : t ≠ 0) : v = 1 :=
by sorry

end velocity_equal_distance_l292_292900


namespace cube_volume_from_surface_area_l292_292247

theorem cube_volume_from_surface_area (s : ℝ) (h : 6 * s^2 = 54) : s^3 = 27 :=
sorry

end cube_volume_from_surface_area_l292_292247


namespace valentines_left_l292_292669

def initial_valentines : ℕ := 60
def valentines_given_away : ℕ := 16
def valentines_received : ℕ := 5

theorem valentines_left : (initial_valentines - valentines_given_away + valentines_received) = 49 :=
by sorry

end valentines_left_l292_292669


namespace find_xy_sum_l292_292606

open Nat

theorem find_xy_sum (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + x * y = 8) 
  (h2 : y + z + y * z = 15) 
  (h3 : z + x + z * x = 35) : 
  x + y + z + x * y = 15 := 
sorry

end find_xy_sum_l292_292606


namespace total_money_l292_292295

def Billy_money (S : ℕ) := 3 * S - 150
def Lila_money (B S : ℕ) := B - S

theorem total_money (S B L : ℕ) (h1 : B = Billy_money S) (h2 : S = 200) (h3 : L = Lila_money B S) : 
  S + B + L = 900 :=
by
  -- The proof would go here.
  sorry

end total_money_l292_292295


namespace simplify_expression_l292_292679

theorem simplify_expression (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 5) * (2 * x - 1) - 
  (2 * x - 1) * (x^2 + 2 * x - 8) + 
  (x^2 - 2 * x + 3) * (2 * x - 1) * (x - 2) = 
  8 * x^4 - 2 * x^3 - 5 * x^2 + 32 * x - 15 := 
  sorry

end simplify_expression_l292_292679


namespace fare_from_midpoint_C_to_B_l292_292535

noncomputable def taxi_fare (d : ℝ) : ℝ :=
  if d <= 5 then 10.8 else 10.8 + 1.2 * (d - 5)

theorem fare_from_midpoint_C_to_B (x : ℝ) (h1 : taxi_fare x = 24)
    (h2 : taxi_fare (x - 0.46) = 24) :
    taxi_fare (x / 2) = 14.4 :=
by
  sorry

end fare_from_midpoint_C_to_B_l292_292535


namespace vertex_of_parabola_l292_292062

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

-- Define the vertex point
def vertex : ℝ × ℝ := (-2, -1)

-- The theorem we need to prove
theorem vertex_of_parabola : ∀ x : ℝ, parabola x = (x + 2)^2 - 1 → vertex = (-2, -1) := 
by
  sorry

end vertex_of_parabola_l292_292062


namespace opposite_of_neg_one_over_2023_l292_292238

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l292_292238


namespace tree_planting_activity_l292_292419

noncomputable def total_trees (grade4: ℕ) (grade5: ℕ) (grade6: ℕ) :=
  grade4 + grade5 + grade6

theorem tree_planting_activity:
  let grade4 := 30 in
  let grade5 := 2 * grade4 in
  let grade6 := (3 * grade5) - 30 in
  total_trees grade4 grade5 grade6 = 240 :=
by
  let grade4 := 30
  let grade5 := 2 * grade4
  let grade6 := (3 * grade5) - 30
  show total_trees grade4 grade5 grade6 = 240
  -- step-by-step calculations omitted
  sorry

end tree_planting_activity_l292_292419


namespace other_root_of_quadratic_l292_292448

theorem other_root_of_quadratic (m : ℝ) (x2 : ℝ) : (x^2 + m * x + 6 = 0) → (x + 2) * (x + x2) = 0 → x2 = -3 :=
by
  sorry

end other_root_of_quadratic_l292_292448


namespace smallest_k_l292_292964

-- Define the set S
def S (m : ℕ) : Finset ℕ :=
  (Finset.range (30 * m)).filter (λ n => n % 2 = 1 ∧ n % 5 ≠ 0)

-- Theorem statement
theorem smallest_k (m : ℕ) (k : ℕ) : 
  (∀ (A : Finset ℕ), A ⊆ S m → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (x ∣ y ∨ y ∣ x)) ↔ k ≥ 8 * m + 1 :=
sorry

end smallest_k_l292_292964


namespace consecutive_numbers_perfect_square_l292_292673

theorem consecutive_numbers_perfect_square (a : ℕ) (h : a ≥ 1) : 
  (a * (a + 1) * (a + 2) * (a + 3) + 1) = (a^2 + 3 * a + 1)^2 :=
by sorry

end consecutive_numbers_perfect_square_l292_292673


namespace snack_eaters_left_l292_292279

theorem snack_eaters_left (initial_participants : ℕ)
    (snack_initial : ℕ)
    (new_outsiders1 : ℕ)
    (half_left1 : ℕ)
    (new_outsiders2 : ℕ)
    (left2 : ℕ)
    (half_left2 : ℕ)
    (h1 : initial_participants = 200)
    (h2 : snack_initial = 100)
    (h3 : new_outsiders1 = 20)
    (h4 : half_left1 = (snack_initial + new_outsiders1) / 2)
    (h5 : new_outsiders2 = 10)
    (h6 : left2 = 30)
    (h7 : half_left2 = (half_left1 + new_outsiders2 - left2) / 2) :
    half_left2 = 20 := 
  sorry

end snack_eaters_left_l292_292279


namespace opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l292_292528

theorem opposite_of_neg23_eq_23 : -(-23) = 23 := 
by sorry

theorem reciprocal_of_neg23_eq_neg_1_div_23 : (1 : ℚ) / (-23) = -(1 / 23 : ℚ) :=
by sorry

theorem abs_value_of_neg23_eq_23 : abs (-23) = 23 :=
by sorry

end opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l292_292528


namespace minimum_trucks_needed_l292_292210

theorem minimum_trucks_needed (total_weight : ℝ) (box_weight : ℕ → ℝ) 
  (n : ℕ) (H_total_weight : total_weight = 10) 
  (H_box_weight : ∀ i, box_weight i ≤ 1) 
  (truck_capacity : ℝ) 
  (H_truck_capacity : truck_capacity = 3) : 
  n = 5 :=
by {
  sorry
}

end minimum_trucks_needed_l292_292210


namespace recommended_cups_l292_292998

theorem recommended_cups (current_cups : ℕ) (R : ℕ) : 
  current_cups = 20 →
  R = current_cups + (6 / 10) * current_cups →
  R = 32 :=
by
  intros h1 h2
  sorry

end recommended_cups_l292_292998


namespace geometric_series_sum_l292_292578

theorem geometric_series_sum : 
  ∀ (a r l : ℕ), 
    a = 2 ∧ r = 3 ∧ l = 4374 → 
    ∃ n S, 
      a * r ^ (n - 1) = l ∧ 
      S = a * (r^n - 1) / (r - 1) ∧ 
      S = 6560 :=
by 
  intros a r l h
  sorry

end geometric_series_sum_l292_292578


namespace divisibility_condition_of_exponents_l292_292439

theorem divisibility_condition_of_exponents (n : ℕ) (h : n ≥ 1) :
  (∀ a b : ℕ, (11 ∣ a^n + b^n) → (11 ∣ a ∧ 11 ∣ b)) ↔ (n % 2 = 0) :=
sorry

end divisibility_condition_of_exponents_l292_292439


namespace cone_lateral_surface_area_l292_292952

theorem cone_lateral_surface_area (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) 
  (h₁ : r = 3)
  (h₂ : V = 12 * Real.pi)
  (h₃ : V = (1 / 3) * Real.pi * r^2 * h)
  (h₄ : l = Real.sqrt (r^2 + h^2)) : 
  ∃ A : ℝ, A = Real.pi * r * l ∧ A = 15 * Real.pi := 
by
  use Real.pi * r * l
  have hr : r = 3 := by exact h₁
  have hV : V = 12 * Real.pi := by exact h₂
  have volume_formula : V = (1 / 3) * Real.pi * r^2 * h := by exact h₃
  have slant_height : l = Real.sqrt (r^2 + h^2) := by exact h₄
  sorry

end cone_lateral_surface_area_l292_292952


namespace christmas_tree_seller_l292_292273

theorem christmas_tree_seller 
  (cost_spruce : ℕ := 220) 
  (cost_pine : ℕ := 250) 
  (cost_fir : ℕ := 330) 
  (total_revenue : ℕ := 36000) 
  (equal_trees: ℕ) 
  (h_costs : cost_spruce + cost_pine + cost_fir = 800) 
  (h_revenue : equal_trees * 800 = total_revenue):
  3 * equal_trees = 135 :=
sorry

end christmas_tree_seller_l292_292273


namespace prime_divides_sum_diff_l292_292506

theorem prime_divides_sum_diff
  (a b c p : ℕ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hp : p.Prime) 
  (h1 : p ∣ (100 * a + 10 * b + c)) 
  (h2 : p ∣ (100 * c + 10 * b + a)) 
  : p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) :=
by
  sorry

end prime_divides_sum_diff_l292_292506


namespace area_above_line_l292_292545

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 - 10*x + y^2 - 16*y + 56 = 0

-- Define the line above which we need to find the area
def line (y : ℝ) : Prop := y = 4

theorem area_above_line : 
  ∃ A : ℝ, (∃ O r : ℝ, (O = (5, 8) ∧ r = sqrt 33) ∧ ∃ f : ℝ → ℝ → ℝ, f = (λ x y, x^2 - 10*x + y^2 - 16*y + 56)) → A = 99 * (π / 4) :=
by
  sorry

end area_above_line_l292_292545


namespace problem_solution_l292_292394

theorem problem_solution :
  ∃ a b c d : ℚ, 
  4 * a + 2 * b + 5 * c + 8 * d = 67 ∧ 
  4 * (d + c) = b ∧ 
  2 * b + 3 * c = a ∧ 
  c + 1 = d ∧ 
  a * b * c * d = (1201 * 572 * 19 * 124) / (105 ^ 4) :=
sorry

end problem_solution_l292_292394


namespace average_of_three_numbers_l292_292385

theorem average_of_three_numbers
  (a b c : ℕ)
  (h1 : 2 * a + b + c = 130)
  (h2 : a + 2 * b + c = 138)
  (h3 : a + b + 2 * c = 152) :
  (a + b + c) / 3 = 35 :=
by
  sorry

end average_of_three_numbers_l292_292385


namespace geom_seq_sum_a3_a4_a5_l292_292813

-- Define the geometric sequence terms and sum condition
def geometric_seq (a1 q : ℕ) (n : ℕ) : ℕ :=
  a1 * q^(n - 1)

def sum_first_three (a1 q : ℕ) : ℕ :=
  a1 + a1 * q + a1 * q^2

-- Given conditions
def a1 : ℕ := 3
def S3 : ℕ := 21

-- Define the problem statement
theorem geom_seq_sum_a3_a4_a5 (q : ℕ) (h : sum_first_three a1 q = S3) (h_pos : ∀ n, geometric_seq a1 q n > 0) :
  geometric_seq a1 q 3 + geometric_seq a1 q 4 + geometric_seq a1 q 5 = 84 :=
by sorry

end geom_seq_sum_a3_a4_a5_l292_292813


namespace yellow_flower_count_l292_292806

-- Define the number of flowers of each color and total flowers based on given conditions
def total_flowers : Nat := 96
def green_flowers : Nat := 9
def red_flowers : Nat := 3 * green_flowers
def blue_flowers : Nat := total_flowers / 2

-- Define the number of yellow flowers
def yellow_flowers : Nat := total_flowers - (green_flowers + red_flowers + blue_flowers)

-- The theorem we aim to prove
theorem yellow_flower_count : yellow_flowers = 12 := by
  sorry

end yellow_flower_count_l292_292806


namespace symmetrical_parabola_eq_l292_292687

/-- 
  Given a parabola y = (x-1)^2 + 3, prove that its symmetrical parabola 
  about the x-axis is y = -(x-1)^2 - 3.
-/
theorem symmetrical_parabola_eq (x : ℝ) : 
  (x-1)^2 + 3 = -(x-1)^2 - 3 ↔ y = -(x-1)^2 - 3 := 
sorry

end symmetrical_parabola_eq_l292_292687


namespace range_of_b_l292_292066

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

theorem range_of_b (b : ℝ) : 
  (∃ (x1 x2 x3 : ℝ), f x1 = -b ∧ f x2 = -b ∧ f x3 = -b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ (-1 < b ∧ b < 0) :=
by
  sorry

end range_of_b_l292_292066


namespace average_steps_per_day_l292_292741

theorem average_steps_per_day (total_steps : ℕ) (h : total_steps = 56392) : 
  (total_steps / 7 : ℚ) = 8056.00 :=
by
  sorry

end average_steps_per_day_l292_292741


namespace card_d_total_percent_change_l292_292135

noncomputable def card_d_initial_value : ℝ := 250
noncomputable def card_d_percent_changes : List ℝ := [0.05, -0.15, 0.30, -0.10, 0.20]

noncomputable def final_value (initial_value : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_value

theorem card_d_total_percent_change :
  let final_val := final_value card_d_initial_value card_d_percent_changes
  let total_percent_change := ((final_val - card_d_initial_value) / card_d_initial_value) * 100
  total_percent_change = 25.307 := by
  sorry

end card_d_total_percent_change_l292_292135


namespace div_neg_21_by_3_l292_292755

theorem div_neg_21_by_3 : (-21 : ℤ) / 3 = -7 :=
by sorry

end div_neg_21_by_3_l292_292755


namespace find_value_l292_292601

variable (a : ℝ) (h : a + 1/a = 7)

theorem find_value :
  a^2 + 1/a^2 = 47 :=
sorry

end find_value_l292_292601


namespace cows_count_l292_292567

theorem cows_count (initial_cows last_year_deaths last_year_sales this_year_increase purchases gifts : ℕ)
  (h1 : initial_cows = 39)
  (h2 : last_year_deaths = 25)
  (h3 : last_year_sales = 6)
  (h4 : this_year_increase = 24)
  (h5 : purchases = 43)
  (h6 : gifts = 8) : 
  initial_cows - last_year_deaths - last_year_sales + this_year_increase + purchases + gifts = 83 := by
  sorry

end cows_count_l292_292567


namespace initial_floor_l292_292748

theorem initial_floor (x y z : ℤ)
  (h1 : y = x - 7)
  (h2 : z = y + 3)
  (h3 : 13 = z + 8) :
  x = 9 :=
sorry

end initial_floor_l292_292748


namespace fuel_consumption_new_model_l292_292738

variable (d_old : ℝ) (d_new : ℝ) (c_old : ℝ) (c_new : ℝ)

theorem fuel_consumption_new_model :
  (d_new = d_old + 4.4) →
  (c_new = c_old - 2) →
  (c_old = 100 / d_old) →
  d_old = 12.79 →
  c_new = 5.82 :=
by
  intro h1 h2 h3 h4
  sorry

end fuel_consumption_new_model_l292_292738


namespace eccentricity_of_ellipse_l292_292001

variables {E F1 F2 P Q : Type}
variables (a c : ℝ) 

-- Define the foci and intersection conditions
def is_right_foci (F1 F2 : Type) (E : Type) : Prop := sorry
def line_intersects_ellipse (E : Type) (P Q : Type) (slope : ℝ) : Prop := sorry
def is_right_triangle (P F2 : Type) : Prop := sorry

-- Prove the eccentricity condition
theorem eccentricity_of_ellipse
  (h_foci : is_right_foci F1 F2 E)
  (h_line : line_intersects_ellipse E P Q (4 / 3))
  (h_triangle : is_right_triangle P F2) :
  (c / a) = (5 / 7) :=
sorry

end eccentricity_of_ellipse_l292_292001


namespace average_of_three_numbers_l292_292384

theorem average_of_three_numbers (a b c : ℝ)
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76) :
  (a + b + c) / 3 = 35 := 
sorry

end average_of_three_numbers_l292_292384


namespace polynomial_value_l292_292025

theorem polynomial_value (y : ℝ) (h : 4 * y^2 - 2 * y + 5 = 7) : 2 * y^2 - y + 1 = 2 :=
by
  sorry

end polynomial_value_l292_292025


namespace prob_exactly_M_laws_included_expected_laws_included_l292_292337

variables (K N M : ℕ) (p : ℝ)

-- Definition of the probabilities as given in the conditions and answers
def prob_no_minister_knows_law : ℝ := (1 - p) ^ N
def prob_law_included : ℝ := 1 - prob_no_minister_knows_law p N

-- Part (a)
theorem prob_exactly_M_laws_included :
  (nat.choose K M) * (prob_law_included p N) ^ M * (prob_no_minister_knows_law p N) ^ (K - M) = 
  (nat.choose K M) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by
  sorry

-- Part (b)
theorem expected_laws_included :
  K * (prob_law_included p N) = K * (1 - (1 - p) ^ N) :=
by
  sorry

end prob_exactly_M_laws_included_expected_laws_included_l292_292337


namespace total_cups_for_8_batches_l292_292671

def cups_of_flour (batches : ℕ) : ℝ := 4 * batches
def cups_of_sugar (batches : ℕ) : ℝ := 1.5 * batches
def total_cups (batches : ℕ) : ℝ := cups_of_flour batches + cups_of_sugar batches

theorem total_cups_for_8_batches : total_cups 8 = 44 := 
by
  -- This is where the proof would go
  sorry

end total_cups_for_8_batches_l292_292671


namespace contrapositive_roots_l292_292549

theorem contrapositive_roots {a b c : ℝ} (h : a ≠ 0) (hac : a * c ≤ 0) :
  ¬ (∀ x : ℝ, (a * x^2 - b * x + c = 0) → x > 0) :=
sorry

end contrapositive_roots_l292_292549


namespace find_n_l292_292915

theorem find_n : ∃ n : ℕ, 2^7 * 3^3 * 5 * n = Nat.factorial 12 ∧ n = 27720 :=
by
  use 27720
  have h1 : 2^7 * 3^3 * 5 * 27720 = Nat.factorial 12 :=
  sorry -- This will be the place to prove the given equation eventually.
  exact ⟨h1, rfl⟩

end find_n_l292_292915


namespace range_of_k_l292_292605

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 3| + |x - 1| > k) ↔ k < 4 :=
by sorry

end range_of_k_l292_292605


namespace sum_first_n_terms_of_geometric_seq_l292_292482

variable {α : Type*} [LinearOrderedField α] (a r : α) (n : ℕ)

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

def sum_geometric_sequence (a r : α) (n : ℕ) : α :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_first_n_terms_of_geometric_seq (h₁ : a * r + a * r^3 = 20) 
    (h₂ : a * r^2 + a * r^4 = 40) :
  sum_geometric_sequence a r n = 2^(n + 1) - 2 := 
sorry

end sum_first_n_terms_of_geometric_seq_l292_292482


namespace train_length_proof_l292_292878

-- Define the conditions
def time_to_cross := 12 -- Time in seconds
def speed_km_per_h := 75 -- Speed in km/h

-- Convert the speed to m/s
def speed_m_per_s := speed_km_per_h * (5 / 18 : ℚ)

-- The length of the train using the formula: length = speed * time
def length_of_train := speed_m_per_s * (time_to_cross : ℚ)

-- The theorem to prove
theorem train_length_proof : length_of_train = 250 := by
  sorry

end train_length_proof_l292_292878


namespace find_pairs_l292_292591

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 1 + 5^a = 6^b → (a, b) = (1, 1) := by
  sorry

end find_pairs_l292_292591


namespace quadratic_roots_shifted_l292_292660

theorem quadratic_roots_shifted (a b c : ℝ) (r s : ℝ) 
  (h1 : 4 * r ^ 2 + 2 * r - 9 = 0) 
  (h2 : 4 * s ^ 2 + 2 * s - 9 = 0) :
  c = 51 / 4 := by
  sorry

end quadratic_roots_shifted_l292_292660


namespace average_price_per_book_l292_292725

-- Definitions of the conditions
def books_shop1 := 65
def cost_shop1 := 1480
def books_shop2 := 55
def cost_shop2 := 920

-- Definition of total values
def total_books := books_shop1 + books_shop2
def total_cost := cost_shop1 + cost_shop2

-- Proof statement
theorem average_price_per_book : (total_cost / total_books) = 20 := by
  sorry

end average_price_per_book_l292_292725


namespace factorial_mod_5_l292_292914

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_5 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
   factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) % 5 = 3 :=
by
  sorry

end factorial_mod_5_l292_292914


namespace matchsticks_left_l292_292123

theorem matchsticks_left (total_matchsticks elvis_match_per_square ralph_match_per_square elvis_squares ralph_squares : ℕ)
  (h1 : total_matchsticks = 50)
  (h2 : elvis_match_per_square = 4)
  (h3 : ralph_match_per_square = 8)
  (h4 : elvis_squares = 5)
  (h5 : ralph_squares = 3) :
  total_matchsticks - (elvis_match_per_square * elvis_squares + ralph_match_per_square * ralph_squares) = 6 := 
by
  sorry

end matchsticks_left_l292_292123


namespace snack_eaters_remaining_l292_292278

theorem snack_eaters_remaining 
  (initial_population : ℕ)
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (first_half_leave : ℕ)
  (new_outsiders_2 : ℕ)
  (second_leave : ℕ)
  (final_half_leave : ℕ) 
  (h_initial_population : initial_population = 200)
  (h_initial_snackers : initial_snackers = 100)
  (h_new_outsiders_1 : new_outsiders_1 = 20)
  (h_first_half_leave : first_half_leave = (initial_snackers + new_outsiders_1) / 2)
  (h_new_outsiders_2 : new_outsiders_2 = 10)
  (h_second_leave : second_leave = 30)
  (h_final_half_leave : final_half_leave = (first_half_leave + new_outsiders_2 - second_leave) / 2) : 
  final_half_leave = 20 := 
sorry

end snack_eaters_remaining_l292_292278


namespace relationship_among_a_b_c_l292_292314

-- Defining the properties and conditions of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Defining the function f based on the condition
noncomputable def f (x m : ℝ) : ℝ := 2 ^ |x - m| - 1

-- Defining the constants a, b, c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5) 0
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2) 0
noncomputable def c : ℝ := f 0 0

-- The theorem stating the relationship among a, b, and c
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l292_292314


namespace parallel_lines_m_values_l292_292018

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (m-2) * x - y - 1 = 0) ∧ (∀ x y : ℝ, 3 * x - m * y = 0) → 
  (m = -1 ∨ m = 3) :=
by
  sorry

end parallel_lines_m_values_l292_292018


namespace simplify_expression_l292_292127

noncomputable def sqrt' (x : ℝ) : ℝ := Real.sqrt x

theorem simplify_expression :
  (3 * sqrt' 8 / (sqrt' 2 + sqrt' 3 + sqrt' 7)) = (sqrt' 2 + sqrt' 3 - sqrt' 7) := 
by
  sorry

end simplify_expression_l292_292127


namespace probability_of_selecting_cubes_l292_292090

/-- Define the probabilities and conditions for unit cubes in the larger cube -/
def unitCubes : ℕ := 125
def doublePainted : ℕ := 8
def unpainted : ℕ := 83
def totalWays : ℕ := (unitCubes * (unitCubes - 1)) / 2
def successfulOutcomes : ℕ := doublePainted * unpainted
def probability := rat.mk successfulOutcomes totalWays

/-- Probability that one of two selected unit cubes has exactly two painted faces while
 the other unit cube has no painted faces -/
theorem probability_of_selecting_cubes :
  probability = rat.ofInt 332 / rat.ofInt 3875 :=
sorry

end probability_of_selecting_cubes_l292_292090


namespace problem_3_equals_answer_l292_292112

variable (a : ℝ)

theorem problem_3_equals_answer :
  (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 :=
by
  sorry

end problem_3_equals_answer_l292_292112


namespace real_part_of_diff_times_i_l292_292162

open Complex

def z1 : ℂ := (4 : ℂ) + (29 : ℂ) * I
def z2 : ℂ := (6 : ℂ) + (9 : ℂ) * I

theorem real_part_of_diff_times_i :
  re ((z1 - z2) * I) = -20 := 
sorry

end real_part_of_diff_times_i_l292_292162


namespace problem_l292_292933

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

variable (f g : ℝ → ℝ)
variable (h₁ : is_odd f)
variable (h₂ : is_even g)
variable (h₃ : ∀ x, f x - g x = 2 * x^3 + x^2 + 3)

theorem problem : f 2 + g 2 = 9 :=
by sorry

end problem_l292_292933


namespace largest_n_exists_l292_292763

theorem largest_n_exists (n x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6 → 
  n ≤ 8 :=
sorry

end largest_n_exists_l292_292763


namespace crayons_in_drawer_before_l292_292994

theorem crayons_in_drawer_before (m c : ℕ) (h1 : m = 3) (h2 : c = 10) : c - m = 7 := 
  sorry

end crayons_in_drawer_before_l292_292994


namespace cube_root_of_sum_of_powers_l292_292391

theorem cube_root_of_sum_of_powers :
  ∃ (x : ℝ), x = 16 * (4 ^ (1 / 3)) ∧ x = (4^6 + 4^6 + 4^6 + 4^6) ^ (1 / 3) :=
by
  sorry

end cube_root_of_sum_of_powers_l292_292391


namespace range_of_m_l292_292795

open Real

def vector_a (m : ℝ) : ℝ × ℝ := (m, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (-2 * m, m)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def not_parallel (m : ℝ) : Prop :=
  m^2 + 2 * m ≠ 0

theorem range_of_m (m : ℝ) (h1 : dot_product (vector_a m) (vector_b m) < 0) (h2 : not_parallel m) :
  m < 0 ∨ (m > (1 / 2) ∧ m ≠ -2) :=
sorry

end range_of_m_l292_292795


namespace people_left_of_Kolya_l292_292987

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l292_292987


namespace student_rank_from_right_l292_292105

theorem student_rank_from_right (n m : ℕ) (h1 : n = 8) (h2 : m = 20) : m - (n - 1) = 13 :=
by
  sorry

end student_rank_from_right_l292_292105


namespace normal_operation_probability_l292_292777

def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_binomial (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def probability_of_normal_operation : ℝ :=
  probability_binomial 10 8 0.9 + probability_binomial 10 9 0.9 + probability_binomial 10 10 0.9

theorem normal_operation_probability :
  probability_of_normal_operation ≈ 0.9298 := by
sorry

end normal_operation_probability_l292_292777


namespace arrangement_of_chairs_and_stools_l292_292072

theorem arrangement_of_chairs_and_stools :
  (Nat.choose 10 3) = 120 :=
by
  -- Proof goes here
  sorry

end arrangement_of_chairs_and_stools_l292_292072


namespace compare_expr_l292_292451

theorem compare_expr (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) :=
sorry

end compare_expr_l292_292451


namespace solve_for_x_l292_292631

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h_eq : x^4 = 6561) : x = 9 :=
sorry

end solve_for_x_l292_292631


namespace count_valid_pairs_l292_292615

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l292_292615


namespace opposite_of_neg_one_div_2023_l292_292237

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l292_292237


namespace income_day_3_is_750_l292_292405

-- Define the given incomes for the specific days
def income_day_1 : ℝ := 250
def income_day_2 : ℝ := 400
def income_day_4 : ℝ := 400
def income_day_5 : ℝ := 500

-- Define the total number of days and the average income over these days
def total_days : ℝ := 5
def average_income : ℝ := 460

-- Define the total income based on the average
def total_income : ℝ := total_days * average_income

-- Define the income on the third day
def income_day_3 : ℝ := total_income - (income_day_1 + income_day_2 + income_day_4 + income_day_5)

-- Claim: The income on the third day is $750
theorem income_day_3_is_750 : income_day_3 = 750 := by
  sorry

end income_day_3_is_750_l292_292405


namespace value_of_xy_l292_292321

noncomputable def distinct_nonzero_reals (x y : ℝ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y

theorem value_of_xy (x y : ℝ) (h : distinct_nonzero_reals x y) (h_eq : x + 4 / x = y + 4 / y) :
  x * y = 4 :=
sorry

end value_of_xy_l292_292321


namespace oliver_siblings_l292_292847

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)

def oliver := Child.mk "Oliver" "Gray" "Brown"
def charles := Child.mk "Charles" "Gray" "Red"
def diana := Child.mk "Diana" "Green" "Brown"
def olivia := Child.mk "Olivia" "Green" "Red"
def ethan := Child.mk "Ethan" "Green" "Red"
def fiona := Child.mk "Fiona" "Green" "Brown"

def sharesCharacteristic (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

def sameFamily (c1 c2 c3 : Child) : Prop :=
  sharesCharacteristic c1 c2 ∧
  sharesCharacteristic c2 c3 ∧
  sharesCharacteristic c3 c1

theorem oliver_siblings : 
  sameFamily oliver charles diana :=
by
  -- proof skipped
  sorry

end oliver_siblings_l292_292847


namespace highest_degree_has_asymptote_l292_292906

noncomputable def highest_degree_of_px (denom : ℕ → ℕ) (n : ℕ) : ℕ :=
  let deg := denom n
  deg

theorem highest_degree_has_asymptote (p : ℕ → ℕ) (denom : ℕ → ℕ) (n : ℕ)
  (h_denom : denom n = 6) :
  highest_degree_of_px denom n = 6 := by
  sorry

end highest_degree_has_asymptote_l292_292906


namespace jerry_initial_action_figures_l292_292348

theorem jerry_initial_action_figures 
(A : ℕ) 
(h1 : ∀ A, A + 7 = 9 + 3)
: A = 5 :=
by
  sorry

end jerry_initial_action_figures_l292_292348


namespace base12_mod_9_remainder_l292_292719

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 7 * 12^2 + 3 * 12^1 + 2 * 12^0

theorem base12_mod_9_remainder : (base12_to_base10 1732) % 9 = 2 := by
  sorry

end base12_mod_9_remainder_l292_292719


namespace part1_part2_l292_292325

open Real

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - 1

-- Define the function g for the second part
def g (x : ℝ) : ℝ := |x - 2| + |x + 3|

theorem part1 (m : ℝ) : (∀ x, f x m ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 :=
  by sorry

theorem part2 (t x: ℝ) (h: ∀ x: ℝ, f x 2 + f (x + 5) 2 ≥ t - 2) : t ≤ 5 :=
  by sorry

end part1_part2_l292_292325


namespace jungkook_biggest_l292_292445

noncomputable def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem jungkook_biggest :
  jungkook_number > yoongi_number ∧ jungkook_number > yuna_number :=
by
  unfold jungkook_number yoongi_number yuna_number
  sorry

end jungkook_biggest_l292_292445


namespace perpendicular_planes_l292_292189

variables (b c : Line) (α β : Plane)
axiom line_in_plane (b : Line) (α : Plane) : Prop -- b ⊆ α
axiom line_parallel_plane (c : Line) (α : Plane) : Prop -- c ∥ α
axiom lines_are_skew (b c : Line) : Prop -- b and c could be skew
axiom planes_are_perpendicular (α β : Plane) : Prop -- α ⊥ β
axiom line_perpendicular_plane (c : Line) (β : Plane) : Prop -- c ⊥ β

theorem perpendicular_planes (hcα : line_in_plane c α) (hcβ : line_perpendicular_plane c β) : planes_are_perpendicular α β := 
sorry

end perpendicular_planes_l292_292189


namespace evaluate_power_sum_l292_292771

theorem evaluate_power_sum : (64:ℝ)^(-1/3) + (81:ℝ)^(-1/4) = 7 / 12 := 
by
  sorry

end evaluate_power_sum_l292_292771


namespace initial_peanuts_l292_292647

-- Definitions based on conditions
def peanuts_added := 8
def total_peanuts_now := 12

-- Statement to prove
theorem initial_peanuts (initial_peanuts : ℕ) (h : initial_peanuts + peanuts_added = total_peanuts_now) : initial_peanuts = 4 :=
sorry

end initial_peanuts_l292_292647


namespace solve_system_l292_292600

theorem solve_system : ∀ (x y : ℤ), 2 * x + y = 5 → x + 2 * y = 6 → x - y = -1 :=
by
  intros x y h1 h2
  sorry

end solve_system_l292_292600


namespace area_of_figure1_values_of_a_for_three_solutions_l292_292611

noncomputable def figure1 (x y : ℝ) : Prop :=
  |3 * x| + |4 * y| + |48 - 3 * x - 4 * y| = 48

noncomputable def figure2 (x y a : ℝ) : Prop :=
  (x - 8)^2 + (y + 6 * Real.cos (a * Real.pi / 2))^2 = (a + 4)^2

theorem area_of_figure1 :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ), figure1 a b ∧ figure1 a c ∧ figure1 b c ∧ (triangle_area a b c = 96) :=
sorry

theorem values_of_a_for_three_solutions :
  ∀ (a : ℝ), (∀ (x y : ℝ), figure1 x y ∧ figure2 x y a) ↔ (a = 6 ∨ a = -14) :=
sorry

end area_of_figure1_values_of_a_for_three_solutions_l292_292611


namespace value_of_R_l292_292472

theorem value_of_R (R : ℝ) (hR_pos : 0 < R)
  (h_line : ∀ x y : ℝ, x + y = 2 * R)
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = R) :
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 :=
by
  sorry

end value_of_R_l292_292472


namespace change_received_l292_292650

theorem change_received (cost_cat_toy : ℝ) (cost_cage : ℝ) (total_paid : ℝ) (change : ℝ) :
  cost_cat_toy = 8.77 →
  cost_cage = 10.97 →
  total_paid = 20.00 →
  change = 0.26 →
  total_paid - (cost_cat_toy + cost_cage) = change := by
sorry

end change_received_l292_292650


namespace equilateral_triangle_side_length_l292_292672

theorem equilateral_triangle_side_length 
  (x1 y1 : ℝ) 
  (hx1y1 : y1 = - (1 / 4) * x1^2)
  (h_eq_tri: ∃ (x2 y2 : ℝ), x2 = -x1 ∧ y2 = y1 ∧ (x2, y2) ≠ (x1, y1) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = x1^2 + y1^2 ∧ (x1 - 0)^2 + (y1 - 0)^2 = (x1 - x2)^2 + (y1 - y2)^2)):
  2 * x1 = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l292_292672


namespace new_ratio_l292_292398

def milk_to_water_initial_ratio (M W : ℕ) : Prop := 4 * W = M

def total_volume (V M W : ℕ) : Prop := V = M + W

def new_water_volume (W_new W A : ℕ) : Prop := W_new = W + A

theorem new_ratio (V M W W_new A : ℕ) 
  (h1: milk_to_water_initial_ratio M W) 
  (h2: total_volume V M W) 
  (h3: A = 23) 
  (h4: new_water_volume W_new W A) 
  (h5: V = 45) 
  : 9 * W_new = 8 * M :=
by 
  sorry

end new_ratio_l292_292398


namespace football_club_balance_l292_292092

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end football_club_balance_l292_292092


namespace carrot_servings_l292_292504

theorem carrot_servings (C : ℕ) 
  (H1 : ∀ (corn_servings : ℕ), corn_servings = 5 * C)
  (H2 : ∀ (green_bean_servings : ℕ) (corn_servings : ℕ), green_bean_servings = corn_servings / 2)
  (H3 : ∀ (plot_plants : ℕ), plot_plants = 9)
  (H4 : ∀ (total_servings : ℕ) 
         (carrot_servings : ℕ)
         (corn_servings : ℕ)
         (green_bean_servings : ℕ), 
         total_servings = carrot_servings + corn_servings + green_bean_servings ∧
         total_servings = 306) : 
  C = 4 := 
    sorry

end carrot_servings_l292_292504


namespace smallest_a_plus_b_l292_292317

theorem smallest_a_plus_b 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : 2^10 * 3^5 = a^b) : a + b = 248833 :=
sorry

end smallest_a_plus_b_l292_292317


namespace sin_alpha_beta_value_l292_292456

theorem sin_alpha_beta_value (α β : ℝ) (h1 : 13 * Real.sin α + 5 * Real.cos β = 9) (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 :=
by
  sorry

end sin_alpha_beta_value_l292_292456


namespace snack_eaters_left_l292_292280

theorem snack_eaters_left (initial_participants : ℕ)
    (snack_initial : ℕ)
    (new_outsiders1 : ℕ)
    (half_left1 : ℕ)
    (new_outsiders2 : ℕ)
    (left2 : ℕ)
    (half_left2 : ℕ)
    (h1 : initial_participants = 200)
    (h2 : snack_initial = 100)
    (h3 : new_outsiders1 = 20)
    (h4 : half_left1 = (snack_initial + new_outsiders1) / 2)
    (h5 : new_outsiders2 = 10)
    (h6 : left2 = 30)
    (h7 : half_left2 = (half_left1 + new_outsiders2 - left2) / 2) :
    half_left2 = 20 := 
  sorry

end snack_eaters_left_l292_292280


namespace math_proof_problem_l292_292393

-- Definitions
def PropA : Prop := ¬ (∀ n : ℤ, (3 ∣ n → ¬ (n % 2 = 1)))
def PropB : Prop := ¬ (¬ (∃ x : ℝ, x^2 + x + 1 ≥ 0))
def PropC : Prop := ∀ (α β : ℝ) (k : ℤ), α = k * Real.pi + β ↔ Real.tan α = Real.tan β
def PropD : Prop := ∀ (a b : ℝ), a ≠ 0 → a * b ≠ 0 → b ≠ 0

def correct_options : Prop := PropA ∧ PropC ∧ ¬PropB ∧ PropD

-- The theorem to be proven
theorem math_proof_problem : correct_options :=
by
  sorry

end math_proof_problem_l292_292393


namespace length_cut_XY_l292_292103

theorem length_cut_XY (a x : ℝ) (h1 : 4 * a = 100) (h2 : a + a + 2 * x = 56) : x = 3 :=
by { sorry }

end length_cut_XY_l292_292103


namespace probability_two_spoons_one_knife_l292_292955

open Finset

theorem probability_two_spoons_one_knife (forks spoons knives total removed : ℕ) 
  (hf : forks = 4) (hs : spoons = 8) (hk : knives = 6) (ht : total = 18) 
  (hr : removed = 3) :
  ((choose spoons 2 * choose knives 1).to_rat / choose total removed).to_rat = (7 / 34) :=
by
  have h_total_eq : total = forks + spoons + knives := by
    rw [hf, hs, hk]
    norm_num
  have h_choose_total : choose total removed = choose 18 3 := by
    rw ht
  have h_choose_favored : choose spoons 2 * choose knives 1 = 28 * 6 := by
    rw [hs, hk]
    norm_num
  have h_total_ways : choose 18 3 = 816 := by
    norm_num
  have h_favored_outcomes : 28 * 6 = 168 := by
    norm_num
  have probability_eq : (168 : ℚ) / 816 = 7 / 34 := by
    norm_num
  sorry

end probability_two_spoons_one_knife_l292_292955


namespace num_pairs_nat_nums_eq_l292_292623

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l292_292623


namespace percentage_of_students_who_speak_lies_l292_292487

theorem percentage_of_students_who_speak_lies
  (T : ℝ)    -- percentage of students who speak the truth
  (I : ℝ)    -- percentage of students who speak both truth and lies
  (U : ℝ)    -- probability of a randomly selected student speaking the truth or lies
  (H_T : T = 0.3)
  (H_I : I = 0.1)
  (H_U : U = 0.4) :
  ∃ (L : ℝ), L = 0.2 :=
by
  sorry

end percentage_of_students_who_speak_lies_l292_292487


namespace sum_of_coordinates_B_l292_292046

theorem sum_of_coordinates_B 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hM_def : M = (-3, 2))
  (hA_def : A = (-8, 5))
  (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  B.1 + B.2 = 1 := 
sorry

end sum_of_coordinates_B_l292_292046


namespace shopping_center_expense_l292_292654

theorem shopping_center_expense
    (films_count : ℕ := 9)
    (films_original_price : ℝ := 7)
    (film_discount : ℝ := 2)
    (books_full_price : ℝ := 10)
    (books_count : ℕ := 5)
    (books_discount_rate : ℝ := 0.25)
    (cd_price : ℝ := 4.50)
    (cd_count : ℕ := 6)
    (tax_rate : ℝ := 0.06)
    (total_amount_spent : ℝ := 109.18) :
    let films_total := films_count * (films_original_price - film_discount)
    let remaining_books := books_count - 1
    let discounted_books_total := remaining_books * (books_full_price * (1 - books_discount_rate))
    let books_total := books_full_price + discounted_books_total
    let cds_paid_count := cd_count - (cd_count / 3)
    let cds_total := cds_paid_count * cd_price
    let total_before_tax := films_total + books_total + cds_total
    let tax := total_before_tax * tax_rate
    let total_with_tax := total_before_tax + tax
    total_with_tax = total_amount_spent :=
by
  sorry

end shopping_center_expense_l292_292654


namespace people_left_of_Kolya_l292_292986

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l292_292986


namespace quadratic_inequality_l292_292794

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * a * x + a > 0) : 0 < a ∧ a < 1 :=
sorry

end quadratic_inequality_l292_292794


namespace count_pairs_satisfying_condition_l292_292621

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l292_292621


namespace max_bricks_truck_can_carry_l292_292250

-- Define the truck's capacity in terms of bags of sand and bricks
def max_sand_bags := 50
def max_bricks := 400
def sand_to_bricks_ratio := 8

-- Define the current number of sand bags already on the truck
def current_sand_bags := 32

-- Define the number of bricks equivalent to a given number of sand bags
def equivalent_bricks (sand_bags: ℕ) := sand_bags * sand_to_bricks_ratio

-- Define the remaining capacity in terms of bags of sand
def remaining_sand_bags := max_sand_bags - current_sand_bags

-- Define the maximum number of additional bricks the truck can carry
def max_additional_bricks := equivalent_bricks remaining_sand_bags

-- Prove the number of additional bricks the truck can carry is 144
theorem max_bricks_truck_can_carry : max_additional_bricks = 144 := by
  sorry

end max_bricks_truck_can_carry_l292_292250


namespace addition_belongs_to_Q_l292_292818

def P : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def R : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}

theorem addition_belongs_to_Q (a b : ℤ) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end addition_belongs_to_Q_l292_292818


namespace systematic_sampling_eighth_group_l292_292099

theorem systematic_sampling_eighth_group (total_students : ℕ) (groups : ℕ) (group_size : ℕ)
(start_number : ℕ) (group_number : ℕ)
(h1 : total_students = 480)
(h2 : groups = 30)
(h3 : group_size = 16)
(h4 : start_number = 5)
(h5 : group_number = 8) :
  (group_number - 1) * group_size + start_number = 117 := by
  sorry

end systematic_sampling_eighth_group_l292_292099


namespace football_club_balance_l292_292095

/-- A football club has a balance of $100 million. The club then sells 2 of its players at $10 million each, and buys 4 more at $15 million each. Prove that the final balance is $60 million. -/
theorem football_club_balance :
  let initial_balance := 100
  let income_from_sales := 10 * 2
  let expenditure_on_purchases := 15 * 4
  let final_balance := initial_balance + income_from_sales - expenditure_on_purchases
  final_balance = 60 :=
by
  simp only [initial_balance, income_from_sales, expenditure_on_purchases, final_balance]
  sorry

end football_club_balance_l292_292095


namespace complete_square_l292_292079

theorem complete_square (x : ℝ) : (x^2 + 4*x - 1 = 0) → ((x + 2)^2 = 5) :=
by
  intro h
  sorry

end complete_square_l292_292079


namespace equation_of_parallel_line_l292_292441

theorem equation_of_parallel_line : 
  ∃ l : ℝ, (∀ x y : ℝ, 2 * x - 3 * y + 8 = 0 ↔ l = 2 * x - 3 * y + 8) :=
sorry

end equation_of_parallel_line_l292_292441


namespace opposite_neg_fraction_l292_292229

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l292_292229


namespace determine_x_l292_292492

variable (A B C x : ℝ)
variable (hA : A = x)
variable (hB : B = 2 * x)
variable (hC : C = 45)
variable (hSum : A + B + C = 180)

theorem determine_x : x = 45 := 
by
  -- proof steps would go here
  sorry

end determine_x_l292_292492


namespace koi_fish_multiple_l292_292731

theorem koi_fish_multiple (n m : ℕ) (h1 : n = 39) (h2 : m * n - 64 < n) : m * n = 78 :=
by
  sorry

end koi_fish_multiple_l292_292731


namespace cos_cofunction_identity_l292_292604

theorem cos_cofunction_identity (α : ℝ) (h : Real.sin (30 * Real.pi / 180 + α) = Real.sqrt 3 / 2) :
  Real.cos (60 * Real.pi / 180 - α) = Real.sqrt 3 / 2 := by
  sorry

end cos_cofunction_identity_l292_292604


namespace expected_rank_of_winner_l292_292536

noncomputable def higher_rank_wins_probability := (3/5 : ℝ)

-- Function to determine the probability of a player winning in a given round
def win_probability (rank : ℕ) : ℝ :=
  if rank % 2 = 0 then higher_rank_wins_probability else 1 - higher_rank_wins_probability

-- Recursive function to compute the probability of winning all rounds
def total_win_probability (rank : ℕ) (rounds : ℕ) : ℝ :=
  nat.rec_on rounds
    1 -- Base case: probability of reaching the first round is 1
    (λ n prob, prob * win_probability rank) -- Recurrence: multiply by the win probability each round

-- Function to calculate the expected value of the winner's rank
def expected_winner_rank : ℝ :=
  let ranks := (list.range 256).map (λ r, r + 1) in
  let probabilities := ranks.map (λ r, total_win_probability r 8) in
  (ranks.zip probabilities).map (λ p, p.1 * p.2).sum

-- Theorem to assert the expected value is 103
theorem expected_rank_of_winner : expected_winner_rank = 103 := sorry

end expected_rank_of_winner_l292_292536


namespace intersection_P_Q_l292_292942

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x < 1}

-- The proof statement
theorem intersection_P_Q : P ∩ Q = {-1, 0} :=
by
  sorry

end intersection_P_Q_l292_292942


namespace find_values_of_cubes_l292_292350

def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b], ![c, b, a], ![b, a, c]]

theorem find_values_of_cubes (a b c : ℂ) (h1 : (N a b c) ^ 2 = 1) (h2 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 :=
by
  sorry

end find_values_of_cubes_l292_292350


namespace water_consumption_per_hour_l292_292656

theorem water_consumption_per_hour 
  (W : ℝ) 
  (initial_water : ℝ := 20) 
  (initial_food : ℝ := 10) 
  (initial_gear : ℝ := 20) 
  (food_consumption_rate : ℝ := 1 / 3) 
  (hours : ℝ := 6) 
  (remaining_weight : ℝ := 34)
  (initial_weight := initial_water + initial_food + initial_gear)
  (consumed_water := W * hours)
  (consumed_food := food_consumption_rate * W * hours)
  (consumed_weight := consumed_water + consumed_food)
  (final_equation := initial_weight - consumed_weight)
  (correct_answer := 2) :
  final_equation = remaining_weight → W = correct_answer := 
by 
  sorry

end water_consumption_per_hour_l292_292656


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l292_292343

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l292_292343


namespace workers_time_l292_292542

variables (x y: ℝ)

theorem workers_time (h1 : (x > 0) ∧ (y > 0)) 
                     (h2 : (3/x + 2/y = 11/20)) 
                     (h3 : (1/x + 1/y = 1/2)) :
                     (x = 10 ∧ y = 8) := 
by
  sorry

end workers_time_l292_292542


namespace set_complement_intersection_l292_292469

theorem set_complement_intersection
  (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)
  (hU : U = {0, 1, 2, 3, 4})
  (hM : M = {0, 1, 2})
  (hN : N = {2, 3}) :
  ((U \ M) ∩ N) = {3} :=
  by sorry

end set_complement_intersection_l292_292469


namespace third_racer_sent_time_l292_292875

theorem third_racer_sent_time (a : ℝ) (t t1 : ℝ) :
  t1 = 1.5 * t → 
  (1.25 * a) * (t1 - (1 / 2)) = 1.5 * a * t → 
  t = 5 / 3 → 
  (t1 - t) * 60 = 50 :=
by 
  intro h_t1_eq h_second_eq h_t_value
  rw [h_t1_eq] at h_second_eq
  have t_correct : t = 5 / 3 := h_t_value
  sorry

end third_racer_sent_time_l292_292875


namespace meeting_time_coincides_l292_292726

variables (distance_ab : ℕ) (speed_train_a : ℕ) (start_time_train_a : ℕ) (distance_at_9am : ℕ) (speed_train_b : ℕ) (start_time_train_b : ℕ)

def total_distance_ab := 465
def train_a_speed := 60
def train_b_speed := 75
def start_time_a := 8
def start_time_b := 9
def distance_train_a_by_9am := train_a_speed * (start_time_b - start_time_a)
def remaining_distance := total_distance_ab - distance_train_a_by_9am
def relative_speed := train_a_speed + train_b_speed
def time_to_meet := remaining_distance / relative_speed

theorem meeting_time_coincides :
  time_to_meet = 3 → (start_time_b + time_to_meet = 12) :=
by
  sorry

end meeting_time_coincides_l292_292726


namespace general_term_a_sum_Tn_l292_292462

section sequence_problem

variables {n : ℕ} (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Problem 1: General term formula for {a_n}
axiom Sn_def : ∀ n, S n = 1/4 * (a n + 1)^2
axiom a1_def : a 1 = 1
axiom an_diff : ∀ n, a (n+1) - a n = 2

theorem general_term_a : a n = 2 * n - 1 := sorry

-- Problem 2: Sum of the first n terms of sequence {b_n}
axiom an_formula : ∀ n, a n = 2 * n - 1
axiom bn_def : ∀ n, b n = 1 / (a n * a (n+1))

theorem sum_Tn : T n = n / (2 * n + 1) := sorry

end sequence_problem

end general_term_a_sum_Tn_l292_292462


namespace find_a3_a4_a5_l292_292174

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 2 * a n

noncomputable def sum_first_three (a : ℕ → ℝ) : Prop :=
a 0 + a 1 + a 2 = 21

theorem find_a3_a4_a5 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : sum_first_three a) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end find_a3_a4_a5_l292_292174


namespace average_of_three_numbers_l292_292386

theorem average_of_three_numbers
  (a b c : ℕ)
  (h1 : 2 * a + b + c = 130)
  (h2 : a + 2 * b + c = 138)
  (h3 : a + b + 2 * c = 152) :
  (a + b + c) / 3 = 35 :=
by
  sorry

end average_of_three_numbers_l292_292386


namespace fractions_addition_l292_292711

theorem fractions_addition :
  (1 / 3) * (3 / 4) * (1 / 5) + (1 / 6) = 13 / 60 :=
by 
  sorry

end fractions_addition_l292_292711


namespace equation_represents_single_point_l292_292976

theorem equation_represents_single_point (d : ℝ) :
  (∀ x y : ℝ, 3*x^2 + 4*y^2 + 6*x - 8*y + d = 0 ↔ (x = -1 ∧ y = 1)) → d = 7 :=
sorry

end equation_represents_single_point_l292_292976


namespace brenda_initial_peaches_l292_292576

variable (P : ℕ)

def brenda_conditions (P : ℕ) : Prop :=
  let fresh_peaches := P - 15
  (P > 15) ∧ (fresh_peaches * 60 = 100 * 150)

theorem brenda_initial_peaches : ∃ (P : ℕ), brenda_conditions P ∧ P = 250 :=
by
  sorry

end brenda_initial_peaches_l292_292576


namespace radius_inner_circle_l292_292104

theorem radius_inner_circle (s : ℝ) (n : ℕ) (d : ℝ) (r : ℝ) :
  s = 4 ∧ n = 16 ∧ d = s / 4 ∧ ∀ k, k = d / 2 → r = (Real.sqrt (s^2 / 4 + k^2) - k) / 2 
  → r = Real.sqrt 4.25 / 2 :=
by
  sorry

end radius_inner_circle_l292_292104


namespace find_range_of_m_l292_292312

noncomputable def p (m : ℝ) : Prop := 1 - Real.sqrt 2 < m ∧ m < 1 + Real.sqrt 2
noncomputable def q (m : ℝ) : Prop := 0 < m ∧ m < 4

theorem find_range_of_m (m : ℝ) (hpq : p m ∨ q m) (hnp : ¬ p m) : 1 + Real.sqrt 2 ≤ m ∧ m < 4 :=
sorry

end find_range_of_m_l292_292312


namespace probability_of_rerolling_two_dice_l292_292500

/-- Jason rolls three fair six-sided dice. Then he looks at the rolls and chooses a subset of the 
dice (possibly empty, possibly all three dice) to reroll. After rerolling, he wins if and only 
if the sum of the numbers face up on the three dice is exactly 7. Jason always plays to optimize 
his chances of winning. Prove that the probability he chooses to reroll exactly two of the dice 
is 7/36. -/
theorem probability_of_rerolling_two_dice :
  let win (dice : Fin 3 → ℕ) := (∑ i, dice i = 7) 
  let F := (Finset.finRange 7).val
  let ⦃optimize_strategy⦄ : Prop := sorry
  in (probability (reroll_exactly_two_and_win win F) = 7 / 36) :=
sorry

end probability_of_rerolling_two_dice_l292_292500


namespace sandy_hours_per_day_l292_292973

theorem sandy_hours_per_day (total_hours : ℕ) (days : ℕ) (H : total_hours = 45 ∧ days = 5) : total_hours / days = 9 :=
by
  sorry

end sandy_hours_per_day_l292_292973


namespace sin_2_alpha_plus_pi_by_3_l292_292782

-- Define the statement to be proved
theorem sin_2_alpha_plus_pi_by_3 (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hcos : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (2 * α + π / 3) = 24 / 25 := sorry

end sin_2_alpha_plus_pi_by_3_l292_292782


namespace divisor_problem_l292_292859

theorem divisor_problem :
  ∃ D : ℕ, 12401 = D * 76 + 13 ∧ D = 163 := 
by
  sorry

end divisor_problem_l292_292859


namespace joan_change_received_l292_292652

theorem joan_change_received :
  let cat_toy_cost := 8.77
  let cage_cost := 10.97
  let payment := 20.00
  let total_cost := cat_toy_cost + cage_cost
  let change_received := payment - total_cost
  change_received = 0.26 :=
by
  sorry

end joan_change_received_l292_292652


namespace find_points_PQ_l292_292204

-- Define the points A, B, M, and E in 3D space
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨10, 0, 0⟩
def M : Point := ⟨5, 5, 0⟩
def E : Point := ⟨0, 0, 10⟩

-- Define the lines AB and EM
def line_AB (t : ℝ) : Point := ⟨10 * t, 0, 0⟩
def line_EM (s : ℝ) : Point := ⟨5 * s, 5 * s, 10 - 10 * s⟩

-- Define the points P and Q
def P (t : ℝ) : Point := line_AB t
def Q (s : ℝ) : Point := line_EM s

-- Define the distance function in 3D space
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

-- The main theorem
theorem find_points_PQ (t s : ℝ) (h1 : t = 0.4) (h2 : s = 0.8) :
  (P t = ⟨4, 0, 0⟩) ∧ (Q s = ⟨4, 4, 2⟩) ∧
  (distance (P t) (Q s) = distance (line_AB 0.4) (line_EM 0.8)) :=
by
  sorry

end find_points_PQ_l292_292204


namespace chocolate_difference_l292_292198

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l292_292198


namespace percentage_water_mixture_l292_292361

theorem percentage_water_mixture 
  (volume_A : ℝ) (volume_B : ℝ) (volume_C : ℝ)
  (ratio_A : ℝ := 5) (ratio_B : ℝ := 3) (ratio_C : ℝ := 2)
  (percentage_water_A : ℝ := 0.20) (percentage_water_B : ℝ := 0.35) (percentage_water_C : ℝ := 0.50) :
  (volume_A = ratio_A) → (volume_B = ratio_B) → (volume_C = ratio_C) → 
  ((percentage_water_A * volume_A + percentage_water_B * volume_B + percentage_water_C * volume_C) /
   (ratio_A + ratio_B + ratio_C)) * 100 = 30.5 := 
by 
  intros hA hB hC
  -- Proof steps would go here
  sorry

end percentage_water_mixture_l292_292361


namespace opposite_of_neg_frac_l292_292232

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l292_292232


namespace probability_neither_event_l292_292400

open ProbabilityTheory

variables (Ω : Type)
variables (P : ProbabilityMassFunction Ω)

variables (A B : Event Ω)
variables (hA : P A = 0.15)
variables (hB : P B = 0.40)
variables (hAB : P (A ∩ B) = 0.15)

theorem probability_neither_event : P (Aᶜ ∩ Bᶜ) = 0.60 := by
  have hAorB : P (A ∪ B) = P A + P B - P (A ∩ B) := 
    ProbabilityMassFunction.prob_union_add_inter A B
  rw [hA, hB, hAB] at hAorB
  have hAorB_value : P (A ∪ B) = 0.40 := by linarith
  have hComplement : P (Aᶜ ∩ Bᶜ) = 1 - P (A ∪ B) := 
    ProbabilityMassFunction.prob_compl_union_eq A B
  rw [hAorB_value] at hComplement
  linarith

end probability_neither_event_l292_292400


namespace volume_of_pyramid_l292_292569

noncomputable def volume_pyramid : ℝ :=
  let a := 9
  let b := 12
  let s := 15
  let base_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let half_diagonal := diagonal / 2
  let height := Real.sqrt (s^2 - half_diagonal^2)
  (1 / 3) * base_area * height

theorem volume_of_pyramid :
  volume_pyramid = 36 * Real.sqrt 168.75 := by
  sorry

end volume_of_pyramid_l292_292569


namespace rhombus_longest_diagonal_l292_292283

theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (h_area : area = 192) (h_ratio : ratio = 4 / 3) :
  ∃ d1 d2 : ℝ, d1 / d2 = 4 / 3 ∧ (d1 * d2) / 2 = 192 ∧ d1 = 16 * Real.sqrt 2 :=
by
  sorry

end rhombus_longest_diagonal_l292_292283


namespace two_real_roots_opposite_signs_l292_292773

theorem two_real_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ (x * y < 0)) ↔ (a < 0) :=
by
  sorry

end two_real_roots_opposite_signs_l292_292773


namespace oil_amount_to_add_l292_292071

variable (a b : ℝ)
variable (h1 : a = 0.16666666666666666)
variable (h2 : b = 0.8333333333333334)

theorem oil_amount_to_add (a b : ℝ) (h1 : a = 0.16666666666666666) (h2 : b = 0.8333333333333334) : 
  b - a = 0.6666666666666667 := by
  rw [h1, h2]
  norm_num
  sorry

end oil_amount_to_add_l292_292071


namespace increasing_interval_of_f_l292_292609

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 15 * x ^ 2 + 36 * x - 24

theorem increasing_interval_of_f : (∀ x : ℝ, x = 2 → deriv f x = 0) → ∀ x : ℝ, 3 < x → 0 < deriv f x :=
by
  intro h x hx
  -- We know that the function has an extreme value at x = 2
  have : deriv f 2 = 0 := h 2 rfl
  -- Require to prove the function is increasing in interval (3, +∞)
  sorry

end increasing_interval_of_f_l292_292609


namespace even_function_iff_b_zero_l292_292926

theorem even_function_iff_b_zero (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c) = ((-x)^2 + b * (-x) + c)) ↔ b = 0 :=
by
  sorry

end even_function_iff_b_zero_l292_292926


namespace find_c_for_given_radius_l292_292306

theorem find_c_for_given_radius (c : ℝ) : (∃ x y : ℝ, (x^2 - 2 * x + y^2 + 6 * y + c = 0) ∧ ((x - 1)^2 + (y + 3)^2 = 25)) → c = -15 :=
by
  sorry

end find_c_for_given_radius_l292_292306


namespace prob_exactly_M_laws_expected_laws_included_l292_292339

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l292_292339


namespace comparison_17_pow_14_31_pow_11_l292_292855

theorem comparison_17_pow_14_31_pow_11 : 17^14 > 31^11 :=
by
  sorry

end comparison_17_pow_14_31_pow_11_l292_292855


namespace scrabble_champions_l292_292646

theorem scrabble_champions :
  let total_champions := 10
  let men_percentage := 0.40
  let men_champions := total_champions * men_percentage
  let bearded_percentage := 0.40
  let non_bearded_percentage := 0.60

  let bearded_men_champions := men_champions * bearded_percentage
  let non_bearded_men_champions := men_champions * non_bearded_percentage

  let bearded_bald_percentage := 0.60
  let bearded_with_hair_percentage := 0.40
  let non_bearded_bald_percentage := 0.30
  let non_bearded_with_hair_percentage := 0.70

  (bearded_men_champions * bearded_bald_percentage).round = 2 ∧
  (bearded_men_champions * bearded_with_hair_percentage).round = 1 ∧
  (non_bearded_men_champions * non_bearded_bald_percentage).round = 2 ∧
  (non_bearded_men_champions * non_bearded_with_hair_percentage).round = 4 :=
by 
sorry

end scrabble_champions_l292_292646


namespace evaluate_trig_expression_l292_292128

theorem evaluate_trig_expression :
  (Real.tan (π / 18) - Real.sqrt 3) * Real.sin (2 * π / 9) = -1 :=
by
  sorry

end evaluate_trig_expression_l292_292128


namespace basketball_team_initial_players_l292_292804

theorem basketball_team_initial_players
  (n : ℕ)
  (h_average_initial : Real := 190)
  (height_nikolai : Real := 197)
  (height_peter : Real := 181)
  (h_average_new : Real := 188)
  (total_height_initial : Real := h_average_initial * n)
  (total_height_new : Real := total_height_initial - (height_nikolai - height_peter))
  (avg_height_new_calculated : Real := total_height_new / n) :
  n = 8 :=
by
  sorry

end basketball_team_initial_players_l292_292804


namespace maximize_S_n_at_24_l292_292494

noncomputable def a_n (n : ℕ) : ℝ := 142 + (n - 1) * (-2)
noncomputable def b_n (n : ℕ) : ℝ := 142 + (n - 1) * (-6)
noncomputable def S_n (n : ℕ) : ℝ := (n / 2.0) * (2 * 142 + (n - 1) * (-6))

theorem maximize_S_n_at_24 : ∀ (n : ℕ), S_n n ≤ S_n 24 :=
by sorry

end maximize_S_n_at_24_l292_292494


namespace speed_of_train_l292_292083

-- Define the given conditions
def length_of_bridge : ℝ := 200
def length_of_train : ℝ := 100
def time_to_cross_bridge : ℝ := 60

-- Define the speed conversion factor
def m_per_s_to_km_per_h : ℝ := 3.6

-- Prove that the speed of the train is 18 km/h
theorem speed_of_train :
  (length_of_bridge + length_of_train) / time_to_cross_bridge * m_per_s_to_km_per_h = 18 :=
by
  sorry

end speed_of_train_l292_292083


namespace total_trees_planted_l292_292417

theorem total_trees_planted :
  let fourth_graders := 30
  let fifth_graders := 2 * fourth_graders
  let sixth_graders := 3 * fifth_graders - 30
  fourth_graders + fifth_graders + sixth_graders = 240 :=
by
  sorry

end total_trees_planted_l292_292417


namespace work_completion_time_l292_292404

theorem work_completion_time (A B C D : Type) 
  (work_rate_A : ℚ := 1 / 10) 
  (work_rate_AB : ℚ := 1 / 5)
  (work_rate_C : ℚ := 1 / 15) 
  (work_rate_D : ℚ := 1 / 20) 
  (combined_work_rate_AB : work_rate_A + (work_rate_AB - work_rate_A) = 1 / 10) : 
  (1 / (work_rate_A + (work_rate_AB - work_rate_A) + work_rate_C + work_rate_D)) = 60 / 19 := 
sorry

end work_completion_time_l292_292404


namespace parallelogram_area_l292_292838

def base := 12 -- in meters
def height := 6 -- in meters

theorem parallelogram_area : base * height = 72 := by
  sorry

end parallelogram_area_l292_292838


namespace bekahs_reading_l292_292111

def pages_per_day (total_pages read_pages days_left : ℕ) : ℕ :=
  (total_pages - read_pages) / days_left

theorem bekahs_reading :
  pages_per_day 408 113 5 = 59 := by
  sorry

end bekahs_reading_l292_292111


namespace value_of_expression_l292_292529

theorem value_of_expression (x : ℝ) (h : x^2 - 5 * x + 6 < 0) : x^2 - 5 * x + 10 = 4 :=
sorry

end value_of_expression_l292_292529


namespace train_speed_with_coaches_l292_292573

theorem train_speed_with_coaches (V₀ : ℝ) (V₉ V₁₆ : ℝ) (k : ℝ) :
  V₀ = 30 → V₁₆ = 14 → V₉ = 30 - k * (9: ℝ) ^ (1/2: ℝ) ∧ V₁₆ = 30 - k * (16: ℝ) ^ (1/2: ℝ) →
  V₉ = 18 :=
by sorry

end train_speed_with_coaches_l292_292573


namespace range_m_n_l292_292467

noncomputable def f (m n x: ℝ) : ℝ := m * Real.exp x + x^2 + n * x

theorem range_m_n (m n: ℝ) :
  (∃ x, f m n x = 0) ∧ (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_m_n_l292_292467


namespace find_a_for_inverse_proportion_l292_292931

theorem find_a_for_inverse_proportion (a : ℝ)
  (h_A : ∃ k : ℝ, 4 = k / (-1))
  (h_B : ∃ k : ℝ, 2 = k / a) :
  a = -2 :=
sorry

end find_a_for_inverse_proportion_l292_292931


namespace solve_quadratic_eq_l292_292056

theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - (x^2 - 2 * x + 1) = 0 ↔ x = 1 / 3 ∨ x = -1 := by
  sorry

end solve_quadratic_eq_l292_292056


namespace weekly_tax_percentage_is_zero_l292_292194

variables (daily_expense : ℕ) (daily_revenue_fries : ℕ) (daily_revenue_poutine : ℕ) (weekly_net_income : ℕ)

def weekly_expense := daily_expense * 7
def weekly_revenue := daily_revenue_fries * 7 + daily_revenue_poutine * 7
def weekly_total_income := weekly_net_income + weekly_expense
def weekly_tax := weekly_total_income - weekly_revenue

theorem weekly_tax_percentage_is_zero
  (h1 : daily_expense = 10)
  (h2 : daily_revenue_fries = 12)
  (h3 : daily_revenue_poutine = 8)
  (h4 : weekly_net_income = 56) :
  weekly_tax = 0 :=
by sorry

end weekly_tax_percentage_is_zero_l292_292194


namespace motorcyclist_average_speed_BC_l292_292737

theorem motorcyclist_average_speed_BC :
  ∀ (d_AB : ℝ) (theta : ℝ) (d_BC_half_d_AB : ℝ) (avg_speed_trip : ℝ)
    (time_ratio_AB_BC : ℝ) (total_speed : ℝ) (t_AB : ℝ) (t_BC : ℝ),
    d_AB = 120 →
    theta = 10 →
    d_BC_half_d_AB = 1 / 2 →
    avg_speed_trip = 30 →
    time_ratio_AB_BC = 3 →
    t_AB = 4.5 →
    t_BC = 1.5 →
    t_AB = time_ratio_AB_BC * t_BC →
    avg_speed_trip = total_speed →
    total_speed = (d_AB + (d_AB * d_BC_half_d_AB)) / (t_AB + t_BC) →
    t_AB / 3 = t_BC →
    ((d_AB * d_BC_half_d_AB) / t_BC = 40) :=
by
  intros d_AB theta d_BC_half_d_AB avg_speed_trip time_ratio_AB_BC total_speed
        t_AB t_BC h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end motorcyclist_average_speed_BC_l292_292737


namespace asymptotes_equation_l292_292583

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  x^2 / 64 - y^2 / 36 = 1

theorem asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y → (y = (3/4) * x ∨ y = - (3/4) * x) :=
by
  intro x y
  intro h
  sorry

end asymptotes_equation_l292_292583


namespace wizard_elixir_combinations_l292_292288

def roots : ℕ := 4
def minerals : ℕ := 5
def incompatible_pairs : ℕ := 3
def total_combinations : ℕ := roots * minerals
def valid_combinations : ℕ := total_combinations - incompatible_pairs

theorem wizard_elixir_combinations : valid_combinations = 17 := by
  sorry

end wizard_elixir_combinations_l292_292288


namespace dan_took_pencils_l292_292538

theorem dan_took_pencils (initial_pencils remaining_pencils : ℕ) (h_initial : initial_pencils = 34) (h_remaining : remaining_pencils = 12) : (initial_pencils - remaining_pencils) = 22 := 
by
  sorry

end dan_took_pencils_l292_292538


namespace solve_inequality_l292_292212

theorem solve_inequality (x : ℝ) : 
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 ↔ 
  3 < x ∧ x < 17 / 3 :=
by
  sorry

end solve_inequality_l292_292212
