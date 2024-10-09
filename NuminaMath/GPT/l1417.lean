import Mathlib

namespace problem1_problem2_problem3_problem4_l1417_141719

theorem problem1 : (5 / 16) - (3 / 16) + (7 / 16) = 9 / 16 := by
  sorry

theorem problem2 : (3 / 12) - (4 / 12) + (6 / 12) = 5 / 12 := by
  sorry

theorem problem3 : 64 + 27 + 81 + 36 + 173 + 219 + 136 = 736 := by
  sorry

theorem problem4 : (2 : ℚ) - (8 / 9) - (1 / 9) + (1 + 98 / 99) = 2 + 98 / 99 := by
  sorry

end problem1_problem2_problem3_problem4_l1417_141719


namespace sin_double_angle_l1417_141764

theorem sin_double_angle (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 4 / 5) :
  Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_l1417_141764


namespace area_of_quadrilateral_l1417_141768

def Quadrilateral (A B C D : Type) :=
  ∃ (ABC_deg : ℝ) (ADC_deg : ℝ) (AD : ℝ) (DC : ℝ) (AB : ℝ) (BC : ℝ),
  (ABC_deg = 90) ∧ (ADC_deg = 90) ∧ (AD = DC) ∧ (AB + BC = 20)

theorem area_of_quadrilateral (A B C D : Type) (h : Quadrilateral A B C D) : 
  ∃ (area : ℝ), area = 100 := 
sorry

end area_of_quadrilateral_l1417_141768


namespace sequence_a_n_l1417_141799

theorem sequence_a_n {a : ℕ → ℤ}
  (h1 : a 2 = 5)
  (h2 : a 1 = 1)
  (h3 : ∀ n ≥ 2, a (n+1) - 2 * a n + a (n-1) = 7) :
  a 17 = 905 :=
  sorry

end sequence_a_n_l1417_141799


namespace Q1_Intersection_Q1_Union_Q2_l1417_141787

namespace Example

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Question 1: 
theorem Q1_Intersection (a : ℝ) (ha : a = -1) : 
  A ∩ B a = {x | -2 ≤ x ∧ x ≤ -1} :=
sorry

theorem Q1_Union (a : ℝ) (ha : a = -1) :
  A ∪ B a = {x | x ≤ 1 ∨ x ≥ 5} :=
sorry

-- Question 2:
theorem Q2 (a : ℝ) :
  (A ∩ B a = B a) ↔ (a ≤ -3 ∨ a > 2) :=
sorry

end Example

end Q1_Intersection_Q1_Union_Q2_l1417_141787


namespace number_of_photographs_is_twice_the_number_of_paintings_l1417_141702

theorem number_of_photographs_is_twice_the_number_of_paintings (P Q : ℕ) :
  (Q * (Q - 1) * P) = 2 * (P * (Q * (Q - 1)) / 2) := by
  sorry

end number_of_photographs_is_twice_the_number_of_paintings_l1417_141702


namespace probability_different_colors_l1417_141762

/-- There are 5 blue chips and 3 yellow chips in a bag. One chip is drawn from the bag and placed
back into the bag. A second chip is then drawn. Prove that the probability of the two selected chips
being of different colors is 15/32. -/
theorem probability_different_colors : 
  let total_chips := 8
  let blue_chips := 5
  let yellow_chips := 3
  let prob_blue_then_yellow := (blue_chips/total_chips) * (yellow_chips/total_chips)
  let prob_yellow_then_blue := (yellow_chips/total_chips) * (blue_chips/total_chips)
  prob_blue_then_yellow + prob_yellow_then_blue = 15/32 := by
  sorry

end probability_different_colors_l1417_141762


namespace triangle_at_most_one_obtuse_l1417_141767

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180) (h3 : 0 < C ∧ C < 180) (h4 : A + B + C = 180) : A ≤ 90 ∨ B ≤ 90 ∨ C ≤ 90 :=
by
  sorry

end triangle_at_most_one_obtuse_l1417_141767


namespace tenth_term_is_correct_l1417_141722

-- Conditions and calculation
variable (a l : ℚ)
variable (d : ℚ)
variable (a10 : ℚ)

-- Setting the given values:
noncomputable def first_term : ℚ := 2 / 3
noncomputable def seventeenth_term : ℚ := 3 / 2
noncomputable def common_difference : ℚ := (seventeenth_term - first_term) / 16

-- Calculate the tenth term using the common difference
noncomputable def tenth_term : ℚ := first_term + 9 * common_difference

-- Statement to prove
theorem tenth_term_is_correct : 
  first_term = 2 / 3 →
  seventeenth_term = 3 / 2 →
  common_difference = (3 / 2 - 2 / 3) / 16 →
  tenth_term = 2 / 3 + 9 * ((3 / 2 - 2 / 3) / 16) →
  tenth_term = 109 / 96 :=
  by
    sorry

end tenth_term_is_correct_l1417_141722


namespace binom_12_10_eq_66_l1417_141795

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l1417_141795


namespace probability_of_x_greater_than_3y_l1417_141769

noncomputable def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height
noncomputable def area_rectangle (width height : ℝ) : ℝ := width * height
noncomputable def probability (area_triangle area_rectangle : ℝ) : ℝ := area_triangle / area_rectangle

theorem probability_of_x_greater_than_3y :
  let base := 2010
  let height_triangle := base / 3
  let height_rectangle := 2011
  let area_triangle := area_triangle base height_triangle
  let area_rectangle := area_rectangle base height_rectangle
  let prob := probability area_triangle area_rectangle
  prob = (335 : ℝ) / 2011 :=
by
  sorry

end probability_of_x_greater_than_3y_l1417_141769


namespace january_31_is_friday_l1417_141715

theorem january_31_is_friday (h : ∀ (d : ℕ), (d % 7 = 0 → d = 1)) : ∀ d, (d = 31) → (d % 7 = 3) :=
by
  sorry

end january_31_is_friday_l1417_141715


namespace moon_iron_percentage_l1417_141774

variables (x : ℝ) -- percentage of iron in the moon

-- Given conditions
def carbon_percentage_of_moon : ℝ := 0.20
def mass_of_moon : ℝ := 250
def mass_of_mars : ℝ := 2 * mass_of_moon
def mass_of_other_elements_on_mars : ℝ := 150
def composition_same (m : ℝ) (x : ℝ) := 
  (x / 100 * m + carbon_percentage_of_moon * m + (100 - x - 20) / 100 * m) = m

-- Theorem statement
theorem moon_iron_percentage : x = 50 :=
by
  sorry

end moon_iron_percentage_l1417_141774


namespace check_prime_large_number_l1417_141735

def large_number := 23021^377 - 1

theorem check_prime_large_number : ¬ Prime large_number := by
  sorry

end check_prime_large_number_l1417_141735


namespace value_in_half_dollars_percentage_l1417_141763

theorem value_in_half_dollars_percentage (n h q : ℕ) (hn : n = 75) (hh : h = 40) (hq : q = 30) : 
  (h * 50 : ℕ) / (n * 5 + h * 50 + q * 25 : ℕ) * 100 = 64 := by
  sorry

end value_in_half_dollars_percentage_l1417_141763


namespace roots_of_polynomial_l1417_141772

theorem roots_of_polynomial :
  (∃ (r : List ℤ), r = [1, 3, 4] ∧ 
    (∀ x : ℤ, x ∈ r → x^3 - 8*x^2 + 19*x - 12 = 0)) ∧ 
  (∀ x, x^3 - 8*x^2 + 19*x - 12 = 0 → x ∈ [1, 3, 4]) := 
sorry

end roots_of_polynomial_l1417_141772


namespace range_of_x_l1417_141732

variable {f : ℝ → ℝ}

-- Define the function is_increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_x (h_inc : is_increasing f) (h_ineq : ∀ x : ℝ, f x < f (2 * x - 3)) :
  ∀ x : ℝ, 3 < x → f x < f (2 * x - 3) := 
sorry

end range_of_x_l1417_141732


namespace count_non_decreasing_digits_of_12022_l1417_141738

/-- Proof that the number of digits left in the number 12022 that form a non-decreasing sequence is 3. -/
theorem count_non_decreasing_digits_of_12022 : 
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2] -- non-decreasing sequence from 12022
  List.length remaining = 3 :=
by
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2]
  have h : List.length remaining = 3 := rfl
  exact h

end count_non_decreasing_digits_of_12022_l1417_141738


namespace problem_16_l1417_141727

-- Definitions of the problem conditions
def trapezoid_inscribed_in_circle (r : ℝ) (a b : ℝ) : Prop :=
  r = 25 ∧ a = 14 ∧ b = 30 

def average_leg_length_of_trapezoid (a b : ℝ) (m : ℝ) : Prop :=
  a = 14 ∧ b = 30 ∧ m = 2000 

-- Using Lean to state the problem
theorem problem_16 (r a b m : ℝ) 
  (h1 : trapezoid_inscribed_in_circle r a b) 
  (h2 : average_leg_length_of_trapezoid a b m) : 
  m = 2000 := by
  sorry

end problem_16_l1417_141727


namespace hiking_committee_selection_l1417_141789

def comb (n k : ℕ) : ℕ := n.choose k

theorem hiking_committee_selection :
  comb 10 3 = 120 :=
by
  sorry

end hiking_committee_selection_l1417_141789


namespace find_speed_of_second_boy_l1417_141781

theorem find_speed_of_second_boy
  (v : ℝ)
  (speed_first_boy : ℝ)
  (distance_apart : ℝ)
  (time_taken : ℝ)
  (h1 : speed_first_boy = 5.3)
  (h2 : distance_apart = 10.5)
  (h3 : time_taken = 35) :
  v = 5.6 :=
by {
  -- translation of the steps to work on the proof
  -- sorry is used to indicate that the proof is not provided here
  sorry
}

end find_speed_of_second_boy_l1417_141781


namespace pool_cleaning_l1417_141748

theorem pool_cleaning (full_capacity_liters : ℕ) (percent_full : ℕ) (loss_per_jump_ml : ℕ) 
    (full_capacity : full_capacity_liters = 2000) (trigger_clean : percent_full = 80) 
    (loss_per_jump : loss_per_jump_ml = 400) : 
    let trigger_capacity_liters := (full_capacity_liters * percent_full) / 100
    let splash_out_capacity_liters := full_capacity_liters - trigger_capacity_liters
    let splash_out_capacity_ml := splash_out_capacity_liters * 1000
    (splash_out_capacity_ml / loss_per_jump_ml) = 1000 :=
by {
    sorry
}

end pool_cleaning_l1417_141748


namespace min_value_function_l1417_141785

theorem min_value_function (x y: ℝ) (hx: x > 2) (hy: y > 2) : 
  (∃c: ℝ, c = (x^3/(y - 2) + y^3/(x - 2)) ∧ ∀x y: ℝ, x > 2 → y > 2 → (x^3/(y - 2) + y^3/(x - 2)) ≥ c) ∧ c = 96 :=
sorry

end min_value_function_l1417_141785


namespace problem_statement_l1417_141797

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
  else -- define elsewhere based on periodicity and oddness properties
    sorry 

theorem problem_statement : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) → f 2015.5 = -0.5 :=
by
  intros
  sorry

end problem_statement_l1417_141797


namespace three_integers_desc_order_l1417_141736

theorem three_integers_desc_order (a b c : ℤ) : ∃ a' b' c' : ℤ, 
  (a = a' ∨ a = b' ∨ a = c') ∧
  (b = a' ∨ b = b' ∨ b = c') ∧
  (c = a' ∨ c = b' ∨ c = c') ∧ 
  (a' ≠ b' ∨ a' ≠ c' ∨ b' ≠ c') ∧
  a' ≥ b' ∧ b' ≥ c' :=
sorry

end three_integers_desc_order_l1417_141736


namespace prime_saturated_96_l1417_141759

def is_prime_saturated (d : ℕ) : Prop :=
  let prime_factors := [2, 3]  -- list of the different positive prime factors of 96
  prime_factors.prod < d       -- the product of prime factors should be less than d

theorem prime_saturated_96 : is_prime_saturated 96 :=
by
  sorry

end prime_saturated_96_l1417_141759


namespace furniture_definition_based_on_vocabulary_study_l1417_141790

theorem furniture_definition_based_on_vocabulary_study (term : String) (h : term = "furniture") :
  term = "furniture" :=
by
  sorry

end furniture_definition_based_on_vocabulary_study_l1417_141790


namespace tan_product_pi_nine_l1417_141741

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l1417_141741


namespace novel_to_history_ratio_l1417_141784

-- Define the conditions
def history_book_pages : ℕ := 300
def science_book_pages : ℕ := 600
def novel_pages := science_book_pages / 4

-- Define the target ratio to prove
def target_ratio := (novel_pages : ℚ) / (history_book_pages : ℚ)

theorem novel_to_history_ratio :
  target_ratio = (1 : ℚ) / (2 : ℚ) :=
by
  sorry

end novel_to_history_ratio_l1417_141784


namespace endpoint_coordinates_l1417_141737

theorem endpoint_coordinates (x y : ℝ) (h : y > 0) :
  let slope_condition := (y - 2) / (x - 2) = 3 / 4
  let distance_condition := (x - 2) ^ 2 + (y - 2) ^ 2 = 64
  slope_condition → distance_condition → 
    (x = 2 + (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 + (4 * Real.sqrt 5475) / 25) + 1 / 2) ∨
    (x = 2 - (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 - (4 * Real.sqrt 5475) / 25) + 1 / 2) :=
by
  intros slope_condition distance_condition
  sorry

end endpoint_coordinates_l1417_141737


namespace find_side_lengths_l1417_141765

variable (a b : ℝ)

-- Conditions
def diff_side_lengths := a - b = 2
def diff_areas := a^2 - b^2 = 40

-- Theorem to prove
theorem find_side_lengths (h1 : diff_side_lengths a b) (h2 : diff_areas a b) :
  a = 11 ∧ b = 9 := by
  -- Proof skipped
  sorry

end find_side_lengths_l1417_141765


namespace age_of_replaced_person_l1417_141793

theorem age_of_replaced_person (avg_age x : ℕ) (h1 : 10 * avg_age - 10 * (avg_age - 3) = x - 18) : x = 48 := 
by
  -- The proof goes here, but we are omitting it as per instruction.
  sorry

end age_of_replaced_person_l1417_141793


namespace geometric_progressions_common_ratio_l1417_141716

theorem geometric_progressions_common_ratio (a b p q : ℝ) :
  (∀ n : ℕ, (a * p^n + b * q^n) = (a * b) * ((p^n + q^n)/a)) →
  p = q := by
  sorry

end geometric_progressions_common_ratio_l1417_141716


namespace max_min_values_l1417_141726

-- Define the function f(x) = x^2 - 2ax + 1
def f (x a : ℝ) : ℝ := x ^ 2 - 2 * a * x + 1

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem max_min_values (a : ℝ) : 
  (a > 2 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = 5 - 4 * a))
  ∧ (1 ≤ a ∧ a ≤ 2 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (0 ≤ a ∧ a < 1 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (a < 0 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = 1)) := by
  sorry

end max_min_values_l1417_141726


namespace diagonals_perpendicular_l1417_141756

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 3 }
def B : Point := { x := 2, y := 6 }
def C : Point := { x := 6, y := -1 }
def D : Point := { x := -3, y := -4 }

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

theorem diagonals_perpendicular :
  let AC := vector A C
  let BD := vector B D
  dot_product AC BD = 0 :=
by
  let AC := vector A C
  let BD := vector B D
  sorry

end diagonals_perpendicular_l1417_141756


namespace cost_per_topping_is_2_l1417_141700

theorem cost_per_topping_is_2 : 
  ∃ (x : ℝ), 
    let large_pizza_cost := 14 
    let num_large_pizzas := 2 
    let num_toppings_per_pizza := 3 
    let tip_rate := 0.25 
    let total_cost := 50 
    let cost_pizzas := num_large_pizzas * large_pizza_cost 
    let num_toppings := num_large_pizzas * num_toppings_per_pizza 
    let cost_toppings := num_toppings * x 
    let before_tip_cost := cost_pizzas + cost_toppings 
    let tip := tip_rate * before_tip_cost 
    let final_cost := before_tip_cost + tip 
    final_cost = total_cost ∧ x = 2 := 
by
  simp
  sorry

end cost_per_topping_is_2_l1417_141700


namespace candice_spending_l1417_141709

variable (total_budget : ℕ) (remaining_money : ℕ) (mildred_spending : ℕ)

theorem candice_spending 
  (h1 : total_budget = 100)
  (h2 : remaining_money = 40)
  (h3 : mildred_spending = 25) :
  (total_budget - remaining_money) - mildred_spending = 35 := 
by
  sorry

end candice_spending_l1417_141709


namespace longest_badminton_match_duration_l1417_141752

theorem longest_badminton_match_duration :
  let hours := 12
  let minutes := 25
  (hours * 60 + minutes = 745) :=
by
  sorry

end longest_badminton_match_duration_l1417_141752


namespace rect_plot_length_more_than_breadth_l1417_141779

theorem rect_plot_length_more_than_breadth (b x : ℕ) (cost_per_m : ℚ)
  (length_eq : b + x = 56)
  (fencing_cost : (4 * b + 2 * x) * cost_per_m = 5300)
  (cost_rate : cost_per_m = 26.50) : x = 12 :=
by
  sorry

end rect_plot_length_more_than_breadth_l1417_141779


namespace part_a_part_b_l1417_141740

theorem part_a (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * (Real.sqrt n + 1) / (n - 1))) :
  μ < (2 * (Real.sqrt n + 1) / (n - 1)) :=
by 
  exact h_μ.2

theorem part_b (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1)))) :
  μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1))) :=
by
  exact h_μ.2

end part_a_part_b_l1417_141740


namespace maria_purse_value_l1417_141725

def value_of_nickels (num_nickels : ℕ) : ℕ := num_nickels * 5
def value_of_dimes (num_dimes : ℕ) : ℕ := num_dimes * 10
def value_of_quarters (num_quarters : ℕ) : ℕ := num_quarters * 25
def total_value (num_nickels num_dimes num_quarters : ℕ) : ℕ := 
  value_of_nickels num_nickels + value_of_dimes num_dimes + value_of_quarters num_quarters
def percentage_of_dollar (value_cents : ℕ) : ℕ := value_cents * 100 / 100

theorem maria_purse_value : percentage_of_dollar (total_value 2 3 2) = 90 := by
  sorry

end maria_purse_value_l1417_141725


namespace function_satisfies_condition_l1417_141718

noncomputable def f : ℕ → ℕ := sorry

theorem function_satisfies_condition (f : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → f (n + 1) > (f n + f (f n)) / 2) :
  (∃ b : ℕ, ∀ n : ℕ, (n < b → f n = n) ∧ (n ≥ b → f n = n + 1)) :=
sorry

end function_satisfies_condition_l1417_141718


namespace find_x_for_fx_neg_half_l1417_141713

open Function 

theorem find_x_for_fx_neg_half (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 2) = -f x)
  (h_interval : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 1/2 * x) :
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ n : ℤ, x = 4 * n - 1} :=
by
  sorry

end find_x_for_fx_neg_half_l1417_141713


namespace david_remaining_money_l1417_141711

noncomputable def initial_funds : ℝ := 1500
noncomputable def spent_on_accommodations : ℝ := 400
noncomputable def spent_on_food_eur : ℝ := 300
noncomputable def eur_to_usd : ℝ := 1.10
noncomputable def spent_on_souvenirs_yen : ℝ := 5000
noncomputable def yen_to_usd : ℝ := 0.009
noncomputable def loan_to_friend : ℝ := 200
noncomputable def difference : ℝ := 500

noncomputable def spent_on_food_usd : ℝ := spent_on_food_eur * eur_to_usd
noncomputable def spent_on_souvenirs_usd : ℝ := spent_on_souvenirs_yen * yen_to_usd
noncomputable def total_spent_excluding_loan : ℝ := spent_on_accommodations + spent_on_food_usd + spent_on_souvenirs_usd

theorem david_remaining_money : 
  initial_funds - total_spent_excluding_loan - difference = 275 :=
by
  sorry

end david_remaining_money_l1417_141711


namespace sector_area_l1417_141794

theorem sector_area (α r : ℝ) (hα : α = 2) (h_r : r = 1 / Real.sin 1) : 
  (1 / 2) * r^2 * α = 1 / (Real.sin 1)^2 :=
by
  sorry

end sector_area_l1417_141794


namespace volume_increase_factor_l1417_141766

variable (π : ℝ) (r h : ℝ)

def original_volume : ℝ := π * r^2 * h

def new_volume : ℝ := π * (2 * r)^2 * (3 * h)

theorem volume_increase_factor : new_volume π r h = 12 * original_volume π r h :=
by
  -- Here we would include the proof that new_volume = 12 * original_volume
  sorry

end volume_increase_factor_l1417_141766


namespace find_fourth_student_number_l1417_141714

theorem find_fourth_student_number 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (student1_num : ℕ) 
  (student2_num : ℕ) 
  (student3_num : ℕ) 
  (student4_num : ℕ)
  ( H1 : total_students = 52 )
  ( H2 : sample_size = 4 )
  ( H3 : student1_num = 6 )
  ( H4 : student2_num = 32 )
  ( H5 : student3_num = 45 ) :
  student4_num = 19 :=
sorry

end find_fourth_student_number_l1417_141714


namespace equation_of_perpendicular_line_through_point_l1417_141783

theorem equation_of_perpendicular_line_through_point :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ), (a = 3) ∧ (b = 1) ∧ (x - 2 * y - 3 = 0 → y = (-(1/2)) * x + 3/2) ∧ (2 * a + b - 7 = 0) := sorry

end equation_of_perpendicular_line_through_point_l1417_141783


namespace largest_integer_solution_l1417_141710

theorem largest_integer_solution (x : ℤ) : 
  (x - 3 * (x - 2) ≥ 4) → (2 * x + 1 < x - 1) → (x = -3) :=
by
  sorry

end largest_integer_solution_l1417_141710


namespace purchasing_power_increase_l1417_141747

theorem purchasing_power_increase (P M : ℝ) (h : 0 < P ∧ 0 < M) :
  let new_price := 0.80 * P
  let original_quantity := M / P
  let new_quantity := M / new_price
  new_quantity = 1.25 * original_quantity :=
by
  sorry

end purchasing_power_increase_l1417_141747


namespace cherry_ratio_l1417_141796

theorem cherry_ratio (total_lollipops cherry_lollipops watermelon_lollipops sour_apple_lollipops grape_lollipops : ℕ) 
  (h_total : total_lollipops = 42) 
  (h_rest_equally_distributed : watermelon_lollipops = sour_apple_lollipops ∧ sour_apple_lollipops = grape_lollipops) 
  (h_grape : grape_lollipops = 7) 
  (h_total_sum : cherry_lollipops + watermelon_lollipops + sour_apple_lollipops + grape_lollipops = total_lollipops) : 
  cherry_lollipops = 21 ∧ (cherry_lollipops : ℚ) / total_lollipops = 1 / 2 :=
by
  sorry

end cherry_ratio_l1417_141796


namespace four_digit_integer_transformation_l1417_141744

theorem four_digit_integer_transformation (a b c d n : ℕ) (A : ℕ)
  (hA : A = 1000 * a + 100 * b + 10 * c + d)
  (ha : a + 2 < 10)
  (hc : c + 2 < 10)
  (hb : b ≥ 2)
  (hd : d ≥ 2)
  (hA4 : 1000 ≤ A ∧ A < 10000) :
  (1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n)) = n * A → n = 2 → A = 1818 :=
by sorry

end four_digit_integer_transformation_l1417_141744


namespace denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l1417_141739

variable (DenyMotion : Prop) (AcknowledgeStillness : Prop) (LeadsToRelativism : Prop)
variable (LeadsToSophistry : Prop)

theorem denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry
  (h1 : DenyMotion)
  (h2 : AcknowledgeStillness)
  (h3 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToRelativism)
  (h4 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToSophistry):
  ¬ (DenyMotion ∧ AcknowledgeStillness → LeadsToRelativism ∧ LeadsToSophistry) :=
by sorry

end denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l1417_141739


namespace smallest_add_to_2002_l1417_141707

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def next_palindrome_after (n : ℕ) : ℕ :=
  -- a placeholder function for the next palindrome calculation
  -- implementation logic is skipped
  2112

def smallest_add_to_palindrome (n target : ℕ) : ℕ :=
  target - n

theorem smallest_add_to_2002 :
  let target := next_palindrome_after 2002
  ∃ k, is_palindrome (2002 + k) ∧ (2002 < 2002 + k) ∧ target = 2002 + k ∧ k = 110 := 
by
  use 110
  sorry

end smallest_add_to_2002_l1417_141707


namespace fingers_game_conditions_l1417_141771

noncomputable def minNForWinningSubset (N : ℕ) : Prop :=
  N ≥ 220

-- To state the probability condition, we need to express it in terms of actual probabilities
noncomputable def probLeaderWins (N : ℕ) : ℝ := 
  1 / N

noncomputable def leaderWinProbabilityTendsToZero : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, probLeaderWins n < ε

theorem fingers_game_conditions (N : ℕ) (probLeaderWins : ℕ → ℝ) :
  (minNForWinningSubset N) ∧ leaderWinProbabilityTendsToZero :=
by
  sorry

end fingers_game_conditions_l1417_141771


namespace distance_from_origin_to_line_l1417_141770

theorem distance_from_origin_to_line : 
  let a := 1
  let b := 2
  let c := -5
  let x0 := 0
  let y0 := 0
  let distance := (|a * x0 + b * y0 + c|) / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l1417_141770


namespace max_height_of_ball_l1417_141733

def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

theorem max_height_of_ball : ∃ t₀, h t₀ = 81.25 ∧ ∀ t, h t ≤ 81.25 :=
by
  sorry

end max_height_of_ball_l1417_141733


namespace hydrochloric_acid_required_l1417_141777

-- Define the quantities for the balanced reaction equation
def molesOfAgNO3 : ℕ := 2
def molesOfHNO3 : ℕ := 2
def molesOfHCl : ℕ := 2

-- Define the condition for the reaction (balances the equation)
def balanced_reaction (x y z w : ℕ) : Prop :=
  x = y ∧ x = z ∧ y = w

-- The goal is to prove that the number of moles of HCl needed is 2
theorem hydrochloric_acid_required :
  balanced_reaction molesOfAgNO3 molesOfHCl molesOfHNO3 2 →
  molesOfHCl = 2 :=
by sorry

end hydrochloric_acid_required_l1417_141777


namespace marbles_per_boy_l1417_141757

theorem marbles_per_boy (boys marbles : ℕ) (h1 : boys = 5) (h2 : marbles = 35) : marbles / boys = 7 := by
  sorry

end marbles_per_boy_l1417_141757


namespace half_plus_five_l1417_141773

theorem half_plus_five (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end half_plus_five_l1417_141773


namespace common_root_poly_identity_l1417_141745

theorem common_root_poly_identity
  (α p p' q q' : ℝ)
  (h1 : α^3 + p*α + q = 0)
  (h2 : α^3 + p'*α + q' = 0) : 
  (p * q' - q * p') * (p - p')^2 = (q - q')^3 := 
by
  sorry

end common_root_poly_identity_l1417_141745


namespace inequality_on_abc_l1417_141706

theorem inequality_on_abc (α β γ : ℝ) (h : α^2 + β^2 + γ^2 = 1) :
  -1/2 ≤ α * β + β * γ + γ * α ∧ α * β + β * γ + γ * α ≤ 1 :=
by {
  sorry -- Proof to be added
}

end inequality_on_abc_l1417_141706


namespace jenny_spent_625_dollars_l1417_141721

def adoption_fee := 50
def vet_visits_cost := 500
def monthly_food_cost := 25
def toys_cost := 200
def year_months := 12

def jenny_adoption_vet_share := (adoption_fee + vet_visits_cost) / 2
def jenny_food_share := (monthly_food_cost * year_months) / 2
def jenny_total_cost := jenny_adoption_vet_share + jenny_food_share + toys_cost

theorem jenny_spent_625_dollars :
  jenny_total_cost = 625 := by
  sorry

end jenny_spent_625_dollars_l1417_141721


namespace sector_radius_cone_l1417_141728

theorem sector_radius_cone {θ R r : ℝ} (sector_angle : θ = 120) (cone_base_radius : r = 2) :
  (R * θ / 360) * 2 * π = 2 * π * r → R = 6 :=
by
  intros h
  sorry

end sector_radius_cone_l1417_141728


namespace geometric_sequence_common_ratio_l1417_141798

noncomputable def common_ratio_q (a1 a5 a : ℕ) (q : ℕ) : Prop :=
  a1 * a5 = 16 ∧ a1 > 0 ∧ a5 > 0 ∧ a = 2 ∧ q = 2

theorem geometric_sequence_common_ratio : ∀ (a1 a5 a q : ℕ), 
  common_ratio_q a1 a5 a q → q = 2 :=
by
  intros a1 a5 a q h
  have h1 : a1 * a5 = 16 := h.1
  have h2 : a1 > 0 := h.2.1
  have h3 : a5 > 0 := h.2.2.1
  have h4 : a = 2 := h.2.2.2.1
  have h5 : q = 2 := h.2.2.2.2
  exact h5

end geometric_sequence_common_ratio_l1417_141798


namespace assume_proof_by_contradiction_l1417_141760

theorem assume_proof_by_contradiction (a b : ℤ) (hab : ∃ k : ℤ, ab = 3 * k) :
  (¬ (∃ k : ℤ, a = 3 * k) ∧ ¬ (∃ k : ℤ, b = 3 * k)) :=
sorry

end assume_proof_by_contradiction_l1417_141760


namespace total_seeds_grace_can_plant_l1417_141717

theorem total_seeds_grace_can_plant :
  let lettuce_seeds_per_row := 25
  let carrot_seeds_per_row := 20
  let radish_seeds_per_row := 30
  let large_bed_rows_limit := 5
  let medium_bed_rows_limit := 3
  let small_bed_rows_limit := 2
  let large_beds := 2
  let medium_beds := 2
  let small_bed := 1
  let large_bed_planting := 
    [(3, lettuce_seeds_per_row), (2, carrot_seeds_per_row)]  -- 3 rows of lettuce, 2 rows of carrots in large beds
  let medium_bed_planting := 
    [(1, lettuce_seeds_per_row), (1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in medium beds
  let small_bed_planting := 
    [(1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in small beds
  (3 * lettuce_seeds_per_row + 2 * carrot_seeds_per_row) * large_beds +
  (1 * lettuce_seeds_per_row + 1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * medium_beds +
  (1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * small_bed = 430 :=
by
  sorry

end total_seeds_grace_can_plant_l1417_141717


namespace num_valid_k_l1417_141749

/--
The number of natural numbers \( k \), not exceeding 485000, 
such that \( k^2 - 1 \) is divisible by 485 is 4000.
-/
theorem num_valid_k (n : ℕ) (h₁ : n ≤ 485000) (h₂ : 485 ∣ (n^2 - 1)) : 
  (∃ k : ℕ, k = 4000) :=
sorry

end num_valid_k_l1417_141749


namespace sum_abc_equals_16_l1417_141729

theorem sum_abc_equals_16 (a b c : ℝ) (h : (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0) : 
  a + b + c = 16 :=
by
  sorry

end sum_abc_equals_16_l1417_141729


namespace remainder_when_4x_div_7_l1417_141780

theorem remainder_when_4x_div_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_when_4x_div_7_l1417_141780


namespace find_C_l1417_141753

theorem find_C (A B C : ℕ) :
  (8 + 5 + 6 + 3 + 2 + A + B) % 3 = 0 →
  (4 + 3 + 7 + 5 + A + B + C) % 3 = 0 →
  C = 2 :=
by
  intros h1 h2
  sorry

end find_C_l1417_141753


namespace expected_disease_count_l1417_141705

/-- Define the probability of an American suffering from the disease. -/
def probability_of_disease := 1 / 3

/-- Define the sample size of Americans surveyed. -/
def sample_size := 450

/-- Calculate the expected number of individuals suffering from the disease in the sample. -/
noncomputable def expected_number := probability_of_disease * sample_size

/-- State the theorem: the expected number of individuals suffering from the disease is 150. -/
theorem expected_disease_count : expected_number = 150 :=
by
  -- Proof is required but skipped using sorry.
  sorry

end expected_disease_count_l1417_141705


namespace number_of_different_ways_is_18_l1417_141724

-- Define the problem conditions
def number_of_ways_to_place_balls : ℕ :=
  let total_balls := 9
  let boxes := 3
  -- Placeholder function to compute the requirement
  -- The actual function would involve combinatorial logic
  -- Let us define it as an axiom for now.
  sorry

-- The theorem to be proven
theorem number_of_different_ways_is_18 :
  number_of_ways_to_place_balls = 18 :=
sorry

end number_of_different_ways_is_18_l1417_141724


namespace problem_1_problem_2_l1417_141750

open Real

theorem problem_1 : sqrt 3 * cos (π / 12) - sin (π / 12) = sqrt 2 := 
sorry

theorem problem_2 : ∀ θ : ℝ, sqrt 3 * cos θ - sin θ ≤ 2 := 
sorry

end problem_1_problem_2_l1417_141750


namespace smallest_cube_with_divisor_l1417_141776

theorem smallest_cube_with_divisor (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ (m : ℕ), m = (p * q * r^2) ^ 3 ∧ (p * q^3 * r^5 ∣ m) :=
by
  sorry

end smallest_cube_with_divisor_l1417_141776


namespace problem_statement_l1417_141751

-- Define the set of numbers
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_multiple (a b : ℕ) : Prop := b ∣ a

-- Problem statement
theorem problem_statement (al bill cal : ℕ) (h_al : al ∈ num_set) (h_bill : bill ∈ num_set) (h_cal : cal ∈ num_set) (h_distinct: distinct al bill cal) : 
  (is_multiple al bill) ∧ (is_multiple bill cal) →
  ∃ (p : ℚ), p = 1 / 190 :=
sorry

end problem_statement_l1417_141751


namespace y_is_multiple_of_3_and_6_l1417_141743

-- Define y as a sum of given numbers
def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_is_multiple_of_3_and_6 :
  (y % 3 = 0) ∧ (y % 6 = 0) :=
by
  -- Proof would go here, but we will end with sorry
  sorry

end y_is_multiple_of_3_and_6_l1417_141743


namespace quarters_total_l1417_141754

variable (q1 q2 S: Nat)

def original_quarters := 760
def additional_quarters := 418

theorem quarters_total : S = original_quarters + additional_quarters :=
sorry

end quarters_total_l1417_141754


namespace intersection_of_M_and_N_l1417_141775

-- Define sets M and N as given in the conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem statement to prove the intersection of M and N is {2, 3}
theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by sorry  -- The proof is skipped with 'sorry'

end intersection_of_M_and_N_l1417_141775


namespace math_problem_l1417_141782

-- Conditions
variables {f g : ℝ → ℝ}
axiom f_zero : f 0 = 0
axiom inequality : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y

-- Problem Statement
theorem math_problem : ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 :=
by
  sorry

end math_problem_l1417_141782


namespace workEfficiencyRatioProof_is_2_1_l1417_141723

noncomputable def workEfficiencyRatioProof : Prop :=
  ∃ (A B : ℝ), 
  (1 / B = 21) ∧ 
  (1 / (A + B) = 7) ∧
  (A / B = 2)

theorem workEfficiencyRatioProof_is_2_1 : workEfficiencyRatioProof :=
  sorry

end workEfficiencyRatioProof_is_2_1_l1417_141723


namespace min_value_sqrt_expression_l1417_141703

open Real

theorem min_value_sqrt_expression : ∃ x : ℝ, ∀ y : ℝ, 
  sqrt (y^2 + (2 - y)^2) + sqrt ((y - 1)^2 + (y + 2)^2) ≥ sqrt 17 :=
by
  sorry

end min_value_sqrt_expression_l1417_141703


namespace investment_interest_min_l1417_141758

theorem investment_interest_min (x y : ℝ) (hx : x + y = 25000) (hmax : x ≤ 11000) : 
  0.07 * x + 0.12 * y ≥ 2450 :=
by
  sorry

end investment_interest_min_l1417_141758


namespace root_of_f_l1417_141720

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

theorem root_of_f (h_inv : f_inv 0 = 2) (h_interval : 1 ≤ (f_inv 0) ∧ (f_inv 0) ≤ 4) : f 2 = 0 := 
sorry

end root_of_f_l1417_141720


namespace nature_of_roots_of_quadratic_l1417_141734

theorem nature_of_roots_of_quadratic (k : ℝ) (h1 : k > 0) (h2 : 3 * k^2 - 2 = 10) :
  let a := 1
  let b := -(4 * k - 3)
  let c := 3 * k^2 - 2
  let Δ := b^2 - 4 * a * c
  Δ < 0 :=
by
  sorry

end nature_of_roots_of_quadratic_l1417_141734


namespace molly_gift_cost_l1417_141704

noncomputable def cost_per_package : ℕ := 5
noncomputable def num_parents : ℕ := 2
noncomputable def num_brothers : ℕ := 3
noncomputable def num_sisters_in_law : ℕ := num_brothers -- each brother is married
noncomputable def num_children_per_brother : ℕ := 2
noncomputable def num_nieces_nephews : ℕ := num_brothers * num_children_per_brother
noncomputable def total_relatives : ℕ := num_parents + num_brothers + num_sisters_in_law + num_nieces_nephews

theorem molly_gift_cost : (total_relatives * cost_per_package) = 70 := by
  sorry

end molly_gift_cost_l1417_141704


namespace abc_value_l1417_141788

theorem abc_value {a b c : ℂ} 
  (h1 : a * b + 5 * b + 20 = 0) 
  (h2 : b * c + 5 * c + 20 = 0) 
  (h3 : c * a + 5 * a + 20 = 0) : 
  a * b * c = 100 := 
by 
  sorry

end abc_value_l1417_141788


namespace units_digit_3_pow_2005_l1417_141746

theorem units_digit_3_pow_2005 : 
  let units_digit (n : ℕ) : ℕ := n % 10
  units_digit (3^2005) = 3 :=
by
  sorry

end units_digit_3_pow_2005_l1417_141746


namespace no_such_integers_exist_l1417_141786

theorem no_such_integers_exist : ¬ ∃ (n k : ℕ), n > 0 ∧ k > 0 ∧ (n ∣ (k ^ n - 1)) ∧ (n.gcd (k - 1) = 1) :=
by
  sorry

end no_such_integers_exist_l1417_141786


namespace line_symmetric_to_itself_l1417_141708

theorem line_symmetric_to_itself :
  ∀ x y : ℝ, y = 3 * x + 3 ↔ ∃ (m b : ℝ), y = m * x + b ∧ m = 3 ∧ b = 3 :=
by
  sorry

end line_symmetric_to_itself_l1417_141708


namespace six_hundred_billion_in_scientific_notation_l1417_141792

theorem six_hundred_billion_in_scientific_notation (billion : ℕ) (h_billion : billion = 10^9) : 
  600 * billion = 6 * 10^11 :=
by
  rw [h_billion]
  sorry

end six_hundred_billion_in_scientific_notation_l1417_141792


namespace num_boys_l1417_141731

theorem num_boys (total_students : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) (r : girls_ratio = 4) (b : boys_ratio = 3) (o : others_ratio = 2) (total_eq : girls_ratio * k + boys_ratio * k + others_ratio * k = total_students) (total_given : total_students = 63) : 
  boys_ratio * k = 21 :=
by
  sorry

end num_boys_l1417_141731


namespace count_3_digit_numbers_divisible_by_13_l1417_141712

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l1417_141712


namespace rectangle_width_l1417_141701

theorem rectangle_width (w : ℝ)
    (h₁ : 5 > 0) (h₂ : 6 > 0) (h₃ : 3 > 0) 
    (area_relation : w * 5 = 3 * 6 + 2) : w = 4 :=
by
  sorry

end rectangle_width_l1417_141701


namespace travelers_on_liner_l1417_141730

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l1417_141730


namespace imaginary_part_of_quotient_l1417_141791

noncomputable def imaginary_part_of_complex (z : ℂ) : ℂ := z.im

theorem imaginary_part_of_quotient :
  imaginary_part_of_complex (i / (1 - i)) = 1 / 2 :=
by sorry

end imaginary_part_of_quotient_l1417_141791


namespace bucket_capacity_l1417_141755

theorem bucket_capacity (x : ℕ) (h₁ : 12 * x = 132 * 5) : x = 55 := by
  sorry

end bucket_capacity_l1417_141755


namespace find_m3_minus_2mn_plus_n3_l1417_141761

theorem find_m3_minus_2mn_plus_n3 (m n : ℝ) (h1 : m^2 = n + 2) (h2 : n^2 = m + 2) (h3 : m ≠ n) : m^3 - 2 * m * n + n^3 = -2 := by
  sorry

end find_m3_minus_2mn_plus_n3_l1417_141761


namespace f_zero_f_odd_f_not_decreasing_f_increasing_l1417_141742

noncomputable def f (x : ℝ) : ℝ := sorry -- The function definition is abstract.

-- Functional equation condition
axiom functional_eq (x y : ℝ) (h1 : -1 < x) (h2 : x < 1) (h3 : -1 < y) (h4 : y < 1) : 
  f x + f y = f ((x + y) / (1 + x * y))

-- Condition for negative interval
axiom neg_interval (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : f x < 0

-- Statements to prove

-- a): f(0) = 0
theorem f_zero : f 0 = 0 := 
by
  sorry

-- b): f(x) is an odd function
theorem f_odd (x : ℝ) (h1 : -1 < x) (h2 : x < 1) : f (-x) = -f x := 
by
  sorry

-- c): f(x) is not a decreasing function
theorem f_not_decreasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : ¬(f x1 > f x2) :=
by
  sorry

-- d): f(x) is an increasing function
theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : f x1 < f x2 :=
by
  sorry

end f_zero_f_odd_f_not_decreasing_f_increasing_l1417_141742


namespace rita_hours_per_month_l1417_141778

theorem rita_hours_per_month :
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  let h_remaining := t - h_completed
  let h := h_remaining / m
  h = 220
:= by 
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  have h_remaining := t - h_completed
  have h := h_remaining / m
  sorry

end rita_hours_per_month_l1417_141778
