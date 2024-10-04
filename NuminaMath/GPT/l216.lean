import Mathlib

namespace xy_equals_nine_l216_216735

theorem xy_equals_nine (x y : ℝ) (h : (|x + 3| > 0 ∧ (y - 2)^2 = 0) ∨ (|x + 3| = 0 ∧ (y - 2)^2 > 0)) : x^y = 9 :=
sorry

end xy_equals_nine_l216_216735


namespace minimum_distance_after_9_minutes_l216_216223

-- Define the initial conditions and movement rules of the robot
structure RobotMovement :=
  (minutes : ℕ)
  (movesStraight : Bool) -- Did the robot move straight in the first minute
  (speed : ℕ)          -- The speed, which is 10 meters/minute
  (turns : Fin (minutes + 1) → ℤ) -- Turns in degrees (-90 for left, 0 for straight, 90 for right)

-- Define the distance function for the robot movement after given minutes
def distanceFromOrigin (rm : RobotMovement) : ℕ :=
  -- This function calculates the minimum distance from the origin where the details are abstracted
  sorry

-- Define the specific conditions of our problem
def robotMovementExample : RobotMovement :=
  { minutes := 9, movesStraight := true, speed := 10,
    turns := λ i => if i = 0 then 0 else -- no turn in the first minute
                      if i % 2 == 0 then 90 else -90 -- Example turning pattern
  }

-- Statement of the proof
theorem minimum_distance_after_9_minutes :
  distanceFromOrigin robotMovementExample = 10 :=
sorry

end minimum_distance_after_9_minutes_l216_216223


namespace jerry_age_is_13_l216_216590

variable (M J : ℕ)

theorem jerry_age_is_13 (h1 : M = 2 * J - 6) (h2 : M = 20) : J = 13 := by
  sorry

end jerry_age_is_13_l216_216590


namespace mushroom_distribution_l216_216773

-- Define the total number of mushrooms
def total_mushrooms : ℕ := 120

-- Define the number of girls
def number_of_girls : ℕ := 5

-- Auxiliary function to represent each girl receiving pattern
def mushrooms_received (n :ℕ) (total : ℕ) : ℝ :=
  (n + 20) + 0.04 * (total - (n + 20))

-- Define the equality function to check distribution condition
def equal_distribution (girls : ℕ) (total : ℕ) : Prop :=
  ∀ i j : ℕ, i < girls → j < girls → mushrooms_received i total = mushrooms_received j total

-- Main proof statement about the total mushrooms and number of girls following the distribution
theorem mushroom_distribution :
  total_mushrooms = 120 ∧ number_of_girls = 5 ∧ equal_distribution number_of_girls total_mushrooms := 
by 
  sorry

end mushroom_distribution_l216_216773


namespace frank_bakes_for_5_days_l216_216862

variable (d : ℕ) -- The number of days Frank bakes cookies

def cookies_baked_per_day : ℕ := 2 * 12
def cookies_eaten_per_day : ℕ := 1

-- Total cookies baked over d days minus the cookies Frank eats each day
def cookies_remaining_before_ted (d : ℕ) : ℕ :=
  d * (cookies_baked_per_day - cookies_eaten_per_day)

-- Ted eats 4 cookies on the last day, so we add that back to get total before Ted ate
def total_cookies_before_ted (d : ℕ) : ℕ :=
  cookies_remaining_before_ted d + 4

-- After Ted's visit, there are 134 cookies left
axiom ted_leaves_134_cookies : total_cookies_before_ted d = 138

-- Prove that Frank bakes cookies for 5 days
theorem frank_bakes_for_5_days : d = 5 := by
  sorry

end frank_bakes_for_5_days_l216_216862


namespace double_neg_eq_pos_l216_216641

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l216_216641


namespace triangle_is_obtuse_l216_216341

-- Definitions based on given conditions
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  if a ≥ b ∧ a ≥ c then a^2 > b^2 + c^2
  else if b ≥ a ∧ b ≥ c then b^2 > a^2 + c^2
  else c^2 > a^2 + b^2

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove
theorem triangle_is_obtuse : is_triangle 4 6 8 ∧ is_obtuse_triangle 4 6 8 :=
by
  sorry

end triangle_is_obtuse_l216_216341


namespace bakery_doughnuts_given_away_l216_216821

theorem bakery_doughnuts_given_away :
  (∀ (boxes_doughnuts : ℕ) (total_doughnuts : ℕ) (boxes_sold : ℕ), 
    boxes_doughnuts = 10 →
    total_doughnuts = 300 →
    boxes_sold = 27 →
    ∃ (doughnuts_given_away : ℕ),
    doughnuts_given_away = (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts ∧
    doughnuts_given_away = 30) :=
by
  intros boxes_doughnuts total_doughnuts boxes_sold h1 h2 h3
  use (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts
  split
  · rw [h1, h2, h3]
    sorry
  · sorry

end bakery_doughnuts_given_away_l216_216821


namespace factor_roots_l216_216113

theorem factor_roots (t : ℝ) : (x - t) ∣ (8 * x^2 + 18 * x - 5) ↔ (t = 1 / 4 ∨ t = -5) :=
by
  sorry

end factor_roots_l216_216113


namespace yogurt_combinations_l216_216510

-- Definitions: Given conditions from the problem
def num_flavors : ℕ := 5
def num_toppings : ℕ := 7

-- Function to calculate binomial coefficient
def nCr (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: The problem translated into Lean
theorem yogurt_combinations : 
  (num_flavors * nCr num_toppings 2) = 105 := by
  sorry

end yogurt_combinations_l216_216510


namespace percent_increase_output_l216_216013

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l216_216013


namespace problem_statement_l216_216268

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 8) * x

theorem problem_statement
  (f : ℝ → ℝ → ℝ)
  (sol_set : Set ℝ)
  (cond1 : ∀ a : ℝ, sol_set = {x : ℝ | -1 ≤ x ∧ x ≤ 5} → ∀ x : ℝ, f x a ≤ 5 ↔ x ∈ sol_set)
  (cond2 : ∀ x : ℝ, ∀ m : ℝ, f x 2 ≥ m^2 - 4 * m - 9) :
  (∃ a : ℝ, a = 2) ∧ (∀ m : ℝ, -1 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_statement_l216_216268


namespace find_a_range_l216_216717

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * a * x^2 + 2 * x + 1
def f' (x a : ℝ) : ℝ := x^2 - a * x + 2

theorem find_a_range (a : ℝ) :
  (0 < x1) ∧ (x1 < 1) ∧ (1 < x2) ∧ (x2 < 3) ∧
  (f' 0 a > 0) ∧ (f' 1 a < 0) ∧ (f' 3 a > 0) →
  3 < a ∧ a < 11 / 3 :=
by
  sorry

end find_a_range_l216_216717


namespace find_a_given_difference_l216_216878

theorem find_a_given_difference (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : |a - a^2| = 6) : a = 3 :=
sorry

end find_a_given_difference_l216_216878


namespace inequality_geq_8_l216_216792

theorem inequality_geq_8 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 :=
by
  sorry

end inequality_geq_8_l216_216792


namespace even_product_probability_l216_216805

def number_on_first_spinner := [3, 6, 5, 10, 15]
def number_on_second_spinner := [7, 6, 11, 12, 13, 14]

noncomputable def probability_even_product : ℚ :=
  1 - (3 / 5) * (3 / 6)

theorem even_product_probability :
  probability_even_product = 7 / 10 :=
by
  sorry

end even_product_probability_l216_216805


namespace arithmetic_geo_sequence_sum_l216_216291

theorem arithmetic_geo_sequence_sum (a : ℕ → ℝ) 
  (h1 : a 2 = 3)
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ a 1 ≠ 0 ∧ a 3 = r * a 1 ∧ a 7 = r^2 * a 1)
  (h3 : ∀ n, a n = 2 + (n - 1) * 1)
  : ∀ (T : ℕ → ℝ) n, 
  (∀ m, T m = ∑ i in finset.range m, (9 / (2 * ∑ j in finset.range (3 * (i + 1)), a (j + 1)))) → 
  (T n = n / (n + 1)) :=
begin
  sorry
end

end arithmetic_geo_sequence_sum_l216_216291


namespace sandwiches_per_person_l216_216651

open Nat

theorem sandwiches_per_person (total_sandwiches : ℕ) (total_people : ℕ) (h1 : total_sandwiches = 657) (h2 : total_people = 219) : 
(total_sandwiches / total_people) = 3 :=
by
  -- a proof would go here
  sorry

end sandwiches_per_person_l216_216651


namespace isosceles_triangle_congruent_side_length_l216_216464

theorem isosceles_triangle_congruent_side_length
  (B : ℕ) (A : ℕ) (P : ℕ) (L : ℕ)
  (h₁ : B = 36) (h₂ : A = 108) (h₃ : P = 84) :
  L = 24 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_congruent_side_length_l216_216464


namespace n_cubed_minus_n_plus_one_is_square_l216_216989

theorem n_cubed_minus_n_plus_one_is_square (n : ℕ) (h : (n^5 + n^4 + 1).divisors.card = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
sorry

end n_cubed_minus_n_plus_one_is_square_l216_216989


namespace total_men_employed_l216_216509

/--
A work which could be finished in 11 days was finished 3 days earlier 
after 10 more men joined. Prove that the total number of men employed 
to finish the work earlier is 37.
-/
theorem total_men_employed (x : ℕ) (h1 : 11 * x = 8 * (x + 10)) : x = 27 ∧ 27 + 10 = 37 := by
  sorry

end total_men_employed_l216_216509


namespace sum_of_edges_l216_216182

-- Define the properties of the rectangular solid
variables (a b c : ℝ)
variables (V : ℝ) (S : ℝ)

-- Set the conditions
def geometric_progression := (a * b * c = V) ∧ (2 * (a * b + b * c + c * a) = S) ∧ (∃ k : ℝ, k ≠ 0 ∧ a = b / k ∧ c = b * k)

-- Define the main proof statement
theorem sum_of_edges (hV : V = 1000) (hS : S = 600) (hg : geometric_progression a b c V S) : 
  4 * (a + b + c) = 120 :=
sorry

end sum_of_edges_l216_216182


namespace hyperbola_slope_of_asymptote_positive_value_l216_216467

noncomputable def hyperbola_slope_of_asymptote (x y : ℝ) : ℝ :=
  if h : (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4)
  then (Real.sqrt 5) / 2
  else 0

-- Statement of the mathematically equivalent proof problem
theorem hyperbola_slope_of_asymptote_positive_value :
  ∃ x y : ℝ, (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) ∧
  hyperbola_slope_of_asymptote x y = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_slope_of_asymptote_positive_value_l216_216467


namespace hyperbola_asymptote_slope_proof_l216_216469

noncomputable def hyperbola_asymptote_slope : ℝ :=
  let foci_distance := Real.sqrt ((8 - 2)^2 + (3 - 3)^2)
  let c := foci_distance / 2
  let a := 2  -- Given that 2a = 4
  let b := Real.sqrt (c^2 - a^2)
  b / a

theorem hyperbola_asymptote_slope_proof :
  ∀ x y : ℝ, 
  (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) →
  hyperbola_asymptote_slope = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_asymptote_slope_proof_l216_216469


namespace infinitely_many_n_divide_b_pow_n_plus_1_l216_216410

theorem infinitely_many_n_divide_b_pow_n_plus_1 (b : ℕ) (h1 : b > 2) :
  (∃ᶠ n in at_top, n^2 ∣ b^n + 1) ↔ ¬ ∃ k : ℕ, b + 1 = 2^k :=
sorry

end infinitely_many_n_divide_b_pow_n_plus_1_l216_216410


namespace quadratic_function_points_relationship_l216_216305

theorem quadratic_function_points_relationship (c y1 y2 y3 : ℝ) 
  (h₁ : y1 = -((-1) ^ 2) + 2 * (-1) + c)
  (h₂ : y2 = -(2 ^ 2) + 2 * 2 + c)
  (h₃ : y3 = -(5 ^ 2) + 2 * 5 + c) :
  y2 > y1 ∧ y1 > y3 :=
by
  sorry

end quadratic_function_points_relationship_l216_216305


namespace area_of_red_flowers_is_54_l216_216000

noncomputable def total_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def red_yellow_area (total : ℝ) : ℝ :=
  total / 2

noncomputable def red_area (red_yellow : ℝ) : ℝ :=
  red_yellow / 2

theorem area_of_red_flowers_is_54 :
  total_area 18 12 / 2 / 2 = 54 := 
  by
    sorry

end area_of_red_flowers_is_54_l216_216000


namespace cylinder_base_radii_l216_216433

theorem cylinder_base_radii {l w : ℝ} (hl : l = 3 * Real.pi) (hw : w = Real.pi) :
  (∃ r : ℝ, l = 2 * Real.pi * r ∧ r = 3 / 2) ∨ (∃ r : ℝ, w = 2 * Real.pi * r ∧ r = 1 / 2) :=
sorry

end cylinder_base_radii_l216_216433


namespace sum_of_geometric_numbers_l216_216389

def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ∃ r : ℕ, r > 0 ∧ 
  (d2 = d1 * r) ∧ 
  (d3 = d2 * r) ∧ 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

theorem sum_of_geometric_numbers : 
  (∃ smallest largest : ℕ,
    (smallest = 124) ∧ 
    (largest = 972) ∧ 
    is_geometric (smallest) ∧ 
    is_geometric (largest)
  ) →
  124 + 972 = 1096 :=
by
  sorry

end sum_of_geometric_numbers_l216_216389


namespace average_weight_of_all_girls_l216_216606

theorem average_weight_of_all_girls (avg1 : ℝ) (n1 : ℕ) (avg2 : ℝ) (n2 : ℕ) :
  avg1 = 50.25 → n1 = 16 → avg2 = 45.15 → n2 = 8 → 
  ((n1 * avg1 + n2 * avg2) / (n1 + n2)) = 48.55 := 
by
  intros h1 h2 h3 h4
  sorry

end average_weight_of_all_girls_l216_216606


namespace algebraic_expression_l216_216874

theorem algebraic_expression (a b : Real) 
  (h : a * b = 2 * (a^2 + b^2)) : 2 * a * b - (a^2 + b^2) = 0 :=
by
  sorry

end algebraic_expression_l216_216874


namespace find_number_l216_216254

theorem find_number (x : ℝ) (h : x * 9999 = 824777405) : x = 82482.5 :=
by
  sorry

end find_number_l216_216254


namespace find_notebook_price_l216_216942

noncomputable def notebook_and_pencil_prices : Prop :=
  ∃ (x y : ℝ),
    5 * x + 4 * y = 16.5 ∧
    2 * x + 2 * y = 7 ∧
    x = 2.5

theorem find_notebook_price : notebook_and_pencil_prices :=
  sorry

end find_notebook_price_l216_216942


namespace employees_in_january_l216_216203

theorem employees_in_january (E : ℝ) (h : 500 = 1.15 * E) : E = 500 / 1.15 :=
by
  sorry

end employees_in_january_l216_216203


namespace number_of_trees_l216_216538

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l216_216538


namespace sequence_type_l216_216867

-- Definitions based on the conditions
def Sn (a : ℝ) (n : ℕ) : ℝ := a^n - 1

def sequence_an (a : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a - 1 else (Sn a n - Sn a (n - 1))

-- Proving the mathematical statement
theorem sequence_type (a : ℝ) (h : a ≠ 0) : 
  (∀ n > 1, (sequence_an a n = sequence_an a 1 + (n - 1) * (sequence_an a 2 - sequence_an a 1)) ∨
  (∀ n > 2, sequence_an a n / sequence_an a (n-1) = a)) :=
sorry

end sequence_type_l216_216867


namespace weight_of_b_l216_216463

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45) 
  (h2 : (a + b) / 2 = 41) 
  (h3 : (b + c) / 2 = 43) 
  : b = 33 :=
by
  sorry

end weight_of_b_l216_216463


namespace car_speed_l216_216953

variable (Distance : ℕ) (Time : ℕ)
variable (h1 : Distance = 495)
variable (h2 : Time = 5)

theorem car_speed (Distance Time : ℕ) (h1 : Distance = 495) (h2 : Time = 5) : 
  Distance / Time = 99 :=
by
  sorry

end car_speed_l216_216953


namespace prove_total_weekly_allowance_l216_216951

noncomputable def total_weekly_allowance : ℕ :=
  let students := 200
  let group1 := students * 45 / 100
  let group2 := students * 30 / 100
  let group3 := students * 15 / 100
  let group4 := students - group1 - group2 - group3  -- Remaining students
  let daily_allowance := group1 * 6 + group2 * 4 + group3 * 7 + group4 * 10
  daily_allowance * 7

theorem prove_total_weekly_allowance :
  total_weekly_allowance = 8330 := by
  sorry

end prove_total_weekly_allowance_l216_216951


namespace parabola_point_distance_to_focus_l216_216050

theorem parabola_point_distance_to_focus :
  ∀ (x y : ℝ), (y^2 = 12 * x) → (∃ (xf : ℝ), xf = 3 ∧ 0 ≤ y) → (∃ (d : ℝ), d = 7) → x = 4 :=
by
  intros x y parabola_focus distance_to_focus distance
  sorry

end parabola_point_distance_to_focus_l216_216050


namespace commission_percentage_l216_216827

-- Define the conditions
def cost_of_item := 18.0
def observed_price := 27.0
def profit_percentage := 0.20
def desired_selling_price := cost_of_item + profit_percentage * cost_of_item
def commission_amount := observed_price - desired_selling_price

-- Prove the commission percentage taken by the online store
theorem commission_percentage : (commission_amount / desired_selling_price) * 100 = 25 :=
by
  -- Here the proof would normally be implemented
  sorry

end commission_percentage_l216_216827


namespace remaining_money_l216_216665

-- Defining the conditions
def orchids_qty := 20
def price_per_orchid := 50
def chinese_money_plants_qty := 15
def price_per_plant := 25
def worker_qty := 2
def salary_per_worker := 40
def cost_of_pots := 150

-- Earnings and expenses calculations
def earnings_from_orchids := orchids_qty * price_per_orchid
def earnings_from_plants := chinese_money_plants_qty * price_per_plant
def total_earnings := earnings_from_orchids + earnings_from_plants

def worker_expenses := worker_qty * salary_per_worker
def total_expenses := worker_expenses + cost_of_pots

-- The proof problem
theorem remaining_money : total_earnings - total_expenses = 1145 :=
by
  sorry

end remaining_money_l216_216665


namespace bike_tire_fixing_charge_l216_216157

theorem bike_tire_fixing_charge (total_profit rent_profit retail_profit: ℝ) (cost_per_tire_parts charge_per_complex_parts charge_per_complex: ℝ) (complex_repairs tire_repairs: ℕ) (charge_per_tire: ℝ) :
  total_profit  = 3000 → rent_profit = 4000 → retail_profit = 2000 →
  cost_per_tire_parts = 5 → charge_per_complex_parts = 50 → charge_per_complex = 300 →
  complex_repairs = 2 → tire_repairs = 300 →
  total_profit = (tire_repairs * charge_per_tire + complex_repairs * charge_per_complex + retail_profit - tire_repairs * cost_per_tire_parts - complex_repairs * charge_per_complex_parts - rent_profit) →
  charge_per_tire = 20 :=
by 
  sorry

end bike_tire_fixing_charge_l216_216157


namespace find_sum_x_y_l216_216027

theorem find_sum_x_y (x y : ℝ) 
  (h1 : x^3 - 3 * x^2 + 2026 * x = 2023)
  (h2 : y^3 + 6 * y^2 + 2035 * y = -4053) : 
  x + y = -1 := 
sorry

end find_sum_x_y_l216_216027


namespace age_difference_l216_216143

def age1 : ℕ := 10
def age2 : ℕ := age1 - 2
def age3 : ℕ := age2 + 4
def age4 : ℕ := age3 / 2
def age5 : ℕ := age4 + 20
def avg : ℕ := (age1 + age5) / 2

theorem age_difference :
  (age3 - age2) = 4 ∧ avg = 18 := by
  sorry

end age_difference_l216_216143


namespace largest_reciprocal_l216_216363

-- Definitions of the given numbers
def num1 := 1 / 6
def num2 := 2 / 7
def num3 := (2 : ℝ)
def num4 := (8 : ℝ)
def num5 := (1000 : ℝ)

-- The main problem: prove that the reciprocal of 1/6 is the largest
theorem largest_reciprocal :
  (1 / num1 > 1 / num2) ∧ (1 / num1 > 1 / num3) ∧ (1 / num1 > 1 / num4) ∧ (1 / num1 > 1 / num5) :=
by
  sorry

end largest_reciprocal_l216_216363


namespace value_of_x_y_squared_l216_216258

theorem value_of_x_y_squared (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : (x - y)^2 = 16 :=
by
  sorry

end value_of_x_y_squared_l216_216258


namespace order_of_abc_l216_216418

section
variables {a b c : ℝ}

def a_def : a = (1/2) * Real.log 2 := by sorry
def b_def : b = (1/4) * Real.log 16 := by sorry
def c_def : c = (1/6) * Real.log 27 := by sorry

theorem order_of_abc : a < c ∧ c < b :=
by
  have ha : a = (1/2) * Real.log 2 := by sorry
  have hb : b = (1/2) * Real.log 4 := by sorry
  have hc : c = (1/2) * Real.log 3 := by sorry
  sorry
end

end order_of_abc_l216_216418


namespace original_deck_size_l216_216236

/-- 
Aubrey adds 2 additional cards to a deck and then splits the deck evenly among herself and 
two other players, each player having 18 cards. 
We want to prove that the original number of cards in the deck was 52. 
-/
theorem original_deck_size :
  ∃ (n : ℕ), (n + 2) / 3 = 18 ∧ n = 52 :=
by
  sorry

end original_deck_size_l216_216236


namespace parallel_lines_slope_l216_216727

theorem parallel_lines_slope (n : ℝ) :
  (∀ x y : ℝ, 2 * x + 2 * y - 5 = 0 → 4 * x + n * y + 1 = 0 → -1 = - (4 / n)) →
  n = 4 :=
by sorry

end parallel_lines_slope_l216_216727


namespace find_a_l216_216873

theorem find_a (a : ℝ) :
  (∀ x y, x + y = a → x^2 + y^2 = 4) →
  (∀ A B : ℝ × ℝ, (A.1 + A.2 = a ∧ B.1 + B.2 = a ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4) →
      ‖(A.1, A.2) + (B.1, B.2)‖ = ‖(A.1, A.2) - (B.1, B.2)‖) →
  a = 2 ∨ a = -2 :=
by
  intros line_circle_intersect vector_eq_magnitude
  sorry

end find_a_l216_216873


namespace find_unknown_number_l216_216440

-- Define the problem conditions and required proof
theorem find_unknown_number (a b : ℕ) (h1 : 2 * a = 3 + b) (h2 : (a - 6)^2 = 3 * b) : b = 3 ∨ b = 27 :=
sorry

end find_unknown_number_l216_216440


namespace distance_between_trees_l216_216814

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 255) (h2 : num_trees = 18) : yard_length / (num_trees - 1) = 15 := by
  sorry

end distance_between_trees_l216_216814


namespace inscribed_circle_radius_in_quarter_circle_l216_216602

theorem inscribed_circle_radius_in_quarter_circle (R r : ℝ) (hR : R = 4) :
  (r + r * Real.sqrt 2 = R) ↔ r = 4 * Real.sqrt 2 - 4 := by
  sorry

end inscribed_circle_radius_in_quarter_circle_l216_216602


namespace eval_expression_l216_216404

theorem eval_expression : (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ)) = 0 := by
  sorry

end eval_expression_l216_216404


namespace tournament_min_cost_l216_216159

variables (k : ℕ) (m : ℕ) (S E : ℕ → ℕ)

noncomputable def min_cost (k : ℕ) : ℕ :=
  k * (4 * k^2 + k - 1) / 2

theorem tournament_min_cost (k_pos : 0 < k) (players : m = 2 * k)
  (each_plays_once 
      : ∀ i j, i ≠ j → ∃ d, S d = i ∧ E d = j) -- every two players play once, matches have days
  (one_match_per_day : ∀ d, ∃! i j, i ≠ j ∧ S d = i ∧ E d = j) -- exactly one match per day
  : min_cost k = k * (4 * k^2 + k - 1) / 2 := 
sorry

end tournament_min_cost_l216_216159


namespace necessary_but_not_sufficient_condition_l216_216515

theorem necessary_but_not_sufficient_condition :
  (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) ∧ 
  ¬ (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) :=
sorry

end necessary_but_not_sufficient_condition_l216_216515


namespace total_volume_of_five_cubes_l216_216196

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end total_volume_of_five_cubes_l216_216196


namespace negation_of_square_positivity_l216_216343

theorem negation_of_square_positivity :
  (¬ ∀ n : ℕ, n * n > 0) ↔ (∃ n : ℕ, n * n ≤ 0) :=
  sorry

end negation_of_square_positivity_l216_216343


namespace proof_problem_l216_216512

noncomputable def a : ℚ := 2 / 3
noncomputable def b : ℚ := - 3 / 2
noncomputable def n : ℕ := 2023

theorem proof_problem :
  (a ^ n) * (b ^ n) = -1 :=
by
  sorry

end proof_problem_l216_216512


namespace Yasmin_children_count_l216_216756

theorem Yasmin_children_count (Y : ℕ) (h1 : 2 * Y + Y = 6) : Y = 2 :=
by
  sorry

end Yasmin_children_count_l216_216756


namespace valid_b_values_count_l216_216183

theorem valid_b_values_count : 
  (∃! b : ℤ, ∃ x1 x2 x3 : ℤ, 
    (∀ x : ℤ, x^2 + b * x + 5 ≤ 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧ 
    (20 ≤ b^2 ∧ b^2 < 29)) :=
sorry

end valid_b_values_count_l216_216183


namespace molecular_weight_is_correct_l216_216983

-- Define the masses of the individual isotopes
def H1 : ℕ := 1
def H2 : ℕ := 2
def O : ℕ := 16
def C : ℕ := 13
def N : ℕ := 15
def S : ℕ := 33

-- Define the molecular weight calculation
def molecular_weight : ℕ := (2 * H1) + H2 + O + C + N + S

-- The goal is to prove that the calculated molecular weight is 81
theorem molecular_weight_is_correct : molecular_weight = 81 :=
by 
  sorry

end molecular_weight_is_correct_l216_216983


namespace fir_trees_alley_l216_216541

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l216_216541


namespace packs_in_each_set_l216_216885

variable (cost_per_set cost_per_pack total_savings : ℝ)
variable (x : ℕ)

-- Objecting conditions
axiom cost_set : cost_per_set = 2.5
axiom cost_pack : cost_per_pack = 1.3
axiom savings : total_savings = 1

-- Main proof problem
theorem packs_in_each_set :
  10 * x * cost_per_pack = 10 * cost_per_set + total_savings → x = 2 :=
by
  -- sorry is a placeholder for the proof
  sorry

end packs_in_each_set_l216_216885


namespace min_students_participating_l216_216234

def ratio_9th_to_10th (n9 n10 : ℕ) : Prop := n9 * 4 = n10 * 3
def ratio_10th_to_11th (n10 n11 : ℕ) : Prop := n10 * 6 = n11 * 5

theorem min_students_participating (n9 n10 n11 : ℕ) 
    (h1 : ratio_9th_to_10th n9 n10) 
    (h2 : ratio_10th_to_11th n10 n11) : 
    n9 + n10 + n11 = 59 :=
sorry

end min_students_participating_l216_216234


namespace minimum_value_of_f_l216_216164

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - a - 15|

theorem minimum_value_of_f {a : ℝ} (h0 : 0 < a) (h1 : a < 15) : ∃ Q, (∀ x, a ≤ x ∧ x ≤ 15 → f x a ≥ Q) ∧ Q = 15 := by
  sorry

end minimum_value_of_f_l216_216164


namespace find_g_50_l216_216466

theorem find_g_50 (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - y * g x = g (x / y) + x - y) :
  g 50 = -24.5 :=
sorry

end find_g_50_l216_216466


namespace cubic_identity_l216_216428

theorem cubic_identity (x : ℝ) (h : x + 1/x = -6) : x^3 + 1/x^3 = -198 := 
by
  sorry

end cubic_identity_l216_216428


namespace jimmy_irene_total_payment_l216_216892

def cost_jimmy_shorts : ℝ := 3 * 15
def cost_irene_shirts : ℝ := 5 * 17
def total_cost_before_discount : ℝ := cost_jimmy_shorts + cost_irene_shirts
def discount : ℝ := total_cost_before_discount * 0.10
def total_paid : ℝ := total_cost_before_discount - discount

theorem jimmy_irene_total_payment : total_paid = 117 := by
  sorry

end jimmy_irene_total_payment_l216_216892


namespace solution_set_quadratic_inequality_l216_216876

theorem solution_set_quadratic_inequality (a b : ℝ) (h1 : a < 0)
    (h2 : ∀ x, ax^2 - bx - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
    ∀ x, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 := 
by
  sorry

end solution_set_quadratic_inequality_l216_216876


namespace machine_part_masses_l216_216511

theorem machine_part_masses :
  ∃ (x y : ℝ), (y - 2 * x = 100) ∧ (875 / x - 900 / y = 3) ∧ (x = 175) ∧ (y = 450) :=
by {
  sorry
}

end machine_part_masses_l216_216511


namespace lasagna_ground_mince_l216_216673

theorem lasagna_ground_mince (total_ground_mince : ℕ) (num_cottage_pies : ℕ) (ground_mince_per_cottage_pie : ℕ) 
  (num_lasagnas : ℕ) (L : ℕ) : 
  total_ground_mince = 500 ∧ num_cottage_pies = 100 ∧ ground_mince_per_cottage_pie = 3 
  ∧ num_lasagnas = 100 ∧ total_ground_mince - num_cottage_pies * ground_mince_per_cottage_pie = num_lasagnas * L 
  → L = 2 := 
by sorry

end lasagna_ground_mince_l216_216673


namespace work_problem_l216_216653

theorem work_problem (W : ℕ) (T_AB T_A T_B together_worked alone_worked remaining_work : ℕ)
  (h1 : T_AB = 30)
  (h2 : T_A = 60)
  (h3 : together_worked = 20)
  (h4 : T_B = 30)
  (h5 : remaining_work = W / 3)
  (h6 : alone_worked = 20)
  : alone_worked = 20 :=
by
  /- Proof is not required -/
  sorry

end work_problem_l216_216653


namespace runners_meet_again_l216_216804

-- Definitions based on the problem conditions
def track_length : ℝ := 500 
def speed_runner1 : ℝ := 4.4
def speed_runner2 : ℝ := 4.8
def speed_runner3 : ℝ := 5.0

-- The time at which runners meet again at the starting point
def time_when_runners_meet : ℝ := 2500

theorem runners_meet_again :
  ∀ t : ℝ, t = time_when_runners_meet → 
  (∀ n1 n2 n3 : ℤ, 
    ∃ k : ℤ, 
    speed_runner1 * t = n1 * track_length ∧ 
    speed_runner2 * t = n2 * track_length ∧ 
    speed_runner3 * t = n3 * track_length) :=
by 
  sorry

end runners_meet_again_l216_216804


namespace find_g_9_l216_216334

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l216_216334


namespace smallest_b_value_l216_216034

theorem smallest_b_value (a b c : ℕ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0)
  (h3 : (31 : ℚ) / 72 = (a : ℚ) / 8 + (b : ℚ) / 9 - c) :
  b = 5 :=
sorry

end smallest_b_value_l216_216034


namespace find_prime_p_l216_216554

def is_prime (p: ℕ) : Prop := Nat.Prime p

def is_product_of_three_distinct_primes (n: ℕ) : Prop :=
  ∃ (p1 p2 p3: ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3

theorem find_prime_p (p: ℕ) (hp: is_prime p) :
  (∃ x y z: ℕ, x^p + y^p + z^p - x - y - z = 30) ↔ (p = 2 ∨ p = 3 ∨ p = 5) := 
sorry

end find_prime_p_l216_216554


namespace distance_from_origin_l216_216748

theorem distance_from_origin (x y : ℝ) :
  (x, y) = (12, -5) →
  (0, 0) = (0, 0) →
  Real.sqrt ((x - 0)^2 + (y - 0)^2) = 13 :=
by
  -- Please note, the proof steps go here, but they are omitted as per instructions.
  -- Typically we'd use sorry to indicate the proof is missing.
  sorry

end distance_from_origin_l216_216748


namespace probability_of_choosing_A_l216_216503

def P (n : ℕ) : ℝ :=
  if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1)

theorem probability_of_choosing_A (n : ℕ) :
  P n = if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1) := 
by {
  sorry
}

end probability_of_choosing_A_l216_216503


namespace train_length_calculation_l216_216228

noncomputable def length_of_train (speed : ℝ) (time_in_sec : ℝ) : ℝ :=
  let time_in_hr := time_in_sec / 3600
  let distance_in_km := speed * time_in_hr
  distance_in_km * 1000

theorem train_length_calculation : 
  length_of_train 60 30 = 500 :=
by
  -- The proof would go here, but we provide a stub with sorry.
  sorry

end train_length_calculation_l216_216228


namespace average_multiples_of_10_l216_216059

theorem average_multiples_of_10 (a b : ℕ) (h1 : a = 10) (h2 : b = 100) :
  (a + b) / 2 = 55 :=
by
  rw [h1, h2]
  norm_num
  sorry

end average_multiples_of_10_l216_216059


namespace symmetry_y_axis_l216_216361

theorem symmetry_y_axis (A B C D : ℝ → ℝ → Prop) 
  (A_eq : ∀ x y : ℝ, A x y ↔ (x^2 - x + y^2 = 1))
  (B_eq : ∀ x y : ℝ, B x y ↔ (x^2 * y + x * y^2 = 1))
  (C_eq : ∀ x y : ℝ, C x y ↔ (x^2 - y^2 = 1))
  (D_eq : ∀ x y : ℝ, D x y ↔ (x - y = 1)) : 
  (∀ x y : ℝ, C x y ↔ C (-x) y) ∧ 
  ¬(∀ x y : ℝ, A x y ↔ A (-x) y) ∧ 
  ¬(∀ x y : ℝ, B x y ↔ B (-x) y) ∧ 
  ¬(∀ x y : ℝ, D x y ↔ D (-x) y) :=
by
  -- Proof goes here
  sorry

end symmetry_y_axis_l216_216361


namespace ratio_of_investments_l216_216201

theorem ratio_of_investments {A B C : ℝ} (x y z k : ℝ)
  (h1 : B - A = 100)
  (h2 : A + B + C = 2900)
  (h3 : A = 6 * k)
  (h4 : B = 5 * k)
  (h5 : C = 4 * k) : 
  (x / y = 6 / 5) ∧ (y / z = 5 / 4) ∧ (x / z = 6 / 4) :=
by
  sorry

end ratio_of_investments_l216_216201


namespace max_missed_questions_l216_216842

theorem max_missed_questions (total_questions : ℕ) (pass_percentage : ℝ) (student_score : ℝ) 
    (h1 : total_questions = 50) (h2 : pass_percentage = 0.85) (h3 : student_score = 0.15) :
    ⌊student_score * total_questions⌋ = 7 :=
by
    -- Given calculations
    rw [h1, h3]
    -- Calculation of missed questions
    norm_num
    -- Ceiling of missed questions
    rfl

end max_missed_questions_l216_216842


namespace josh_final_pencils_l216_216024

-- Define the conditions as Lean definitions
def josh_initial_pencils : ℕ := 142
def percent_given_to_dorothy : ℚ := 25 / 100
def pencils_given_to_dorothy : ℕ := floor (percent_given_to_dorothy * josh_initial_pencils)
def pencils_given_to_mark : ℕ := 10
def pencils_given_back_by_dorothy : ℕ := 6

-- Define the final computation as Lean definition
theorem josh_final_pencils : 
  josh_initial_pencils - pencils_given_to_dorothy - pencils_given_to_mark + pencils_given_back_by_dorothy = 103 := 
by
  sorry

end josh_final_pencils_l216_216024


namespace minimize_M_l216_216118

noncomputable def M (x y : ℝ) : ℝ := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9

theorem minimize_M : ∃ x y, M x y = 5 ∧ x = -3 ∧ y = -2 :=
by
  sorry

end minimize_M_l216_216118


namespace dot_product_eq_one_l216_216546

open Real

noncomputable def a : ℝ × ℝ := (2 * sin (35 * (π / 180)), 2 * cos (35 * (π / 180)))
noncomputable def b : ℝ × ℝ := (cos (5 * (π / 180)), -sin (5 * (π / 180)))

theorem dot_product_eq_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  let θ := 35 * (π / 180)
  let φ := 5 * (π / 180)
  have H : a = (2 * sin θ, 2 * cos θ) :=
    rfl
  have H' : b = (cos φ, -sin φ) :=
    rfl
  sorry

end dot_product_eq_one_l216_216546


namespace cucumbers_count_l216_216090

theorem cucumbers_count (C T : ℕ) 
  (h1 : C + T = 280)
  (h2 : T = 3 * C) : C = 70 :=
by sorry

end cucumbers_count_l216_216090


namespace directrix_of_parabola_l216_216609

-- Define the given condition
def parabola_eq (x y : ℝ) : Prop := y = -4 * x^2

-- The problem we need to prove
theorem directrix_of_parabola :
  ∃ y : ℝ, (∀ x : ℝ, parabola_eq x y) ↔ y = 1 / 16 :=
by
  sorry

end directrix_of_parabola_l216_216609


namespace original_number_l216_216144

theorem original_number (x : ℝ) (h : x - x / 3 = 36) : x = 54 :=
by
  sorry

end original_number_l216_216144


namespace probability_three_correct_out_of_five_l216_216708

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l216_216708


namespace solve_system_eq_l216_216598

theorem solve_system_eq (x y z t : ℕ) : 
  ((x^2 + t^2) * (z^2 + y^2) = 50) ↔
    (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
    (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
    (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
by 
  sorry

end solve_system_eq_l216_216598


namespace minimum_distance_after_9_minutes_l216_216224

-- Define the initial conditions and movement rules of the robot
structure RobotMovement :=
  (minutes : ℕ)
  (movesStraight : Bool) -- Did the robot move straight in the first minute
  (speed : ℕ)          -- The speed, which is 10 meters/minute
  (turns : Fin (minutes + 1) → ℤ) -- Turns in degrees (-90 for left, 0 for straight, 90 for right)

-- Define the distance function for the robot movement after given minutes
def distanceFromOrigin (rm : RobotMovement) : ℕ :=
  -- This function calculates the minimum distance from the origin where the details are abstracted
  sorry

-- Define the specific conditions of our problem
def robotMovementExample : RobotMovement :=
  { minutes := 9, movesStraight := true, speed := 10,
    turns := λ i => if i = 0 then 0 else -- no turn in the first minute
                      if i % 2 == 0 then 90 else -90 -- Example turning pattern
  }

-- Statement of the proof
theorem minimum_distance_after_9_minutes :
  distanceFromOrigin robotMovementExample = 10 :=
sorry

end minimum_distance_after_9_minutes_l216_216224


namespace arrangement_schemes_l216_216077

theorem arrangement_schemes (n m k: ℕ) (h1 : n = 5) (h2 : m = 2) (h3 : k = 4):
  (1 / 2) * (Nat.choose k (k/2)) * (Nat.perm n m) = 60 :=
by {
  sorry
}

end arrangement_schemes_l216_216077


namespace Glorys_favorite_number_l216_216593

variable (M G : ℝ)

theorem Glorys_favorite_number :
  (M = G / 3) →
  (M + G = 600) →
  (G = 450) :=
by
sorry

end Glorys_favorite_number_l216_216593


namespace sum_possible_values_l216_216613

theorem sum_possible_values (N : ℤ) (h : N * (N - 8) = -7) : 
  ∀ (N1 N2 : ℤ), (N1 * (N1 - 8) = -7) ∧ (N2 * (N2 - 8) = -7) → (N1 + N2 = 8) :=
by
  sorry

end sum_possible_values_l216_216613


namespace intersection_A_B_l216_216028

open Set

def U := ℝ
def A := { x : ℝ | (2 * x + 3) / (x - 2) > 0 }
def B := { x : ℝ | abs (x - 1) < 2 }

theorem intersection_A_B : (A ∩ B) = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l216_216028


namespace number_of_fir_trees_is_11_l216_216535

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l216_216535


namespace damaged_cartons_per_customer_l216_216758

theorem damaged_cartons_per_customer (total_cartons : ℕ) (num_customers : ℕ) (total_accepted : ℕ) 
    (h1 : total_cartons = 400) (h2 : num_customers = 4) (h3 : total_accepted = 160) 
    : (total_cartons - total_accepted) / num_customers = 60 :=
by
  sorry

end damaged_cartons_per_customer_l216_216758


namespace neg_neg_eq_l216_216644

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l216_216644


namespace number_of_trees_is_11_l216_216542

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l216_216542


namespace Sara_house_size_l216_216167

theorem Sara_house_size (nada_size : ℕ) (h1 : nada_size = 450) (h2 : Sara_size = 2 * nada_size + 100) : Sara_size = 1000 :=
by sorry

end Sara_house_size_l216_216167


namespace water_level_function_l216_216823

def water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : ℝ :=
  0.3 * x + 6

theorem water_level_function :
  ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5), water_level x h = 6 + 0.3 * x :=
by
  intros
  unfold water_level
  sorry -- Proof skipped

end water_level_function_l216_216823


namespace probability_three_correct_packages_l216_216702

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l216_216702


namespace total_travel_time_l216_216111

theorem total_travel_time (distance1 distance2 speed time1: ℕ) (h1 : distance1 = 100) (h2 : time1 = 1) (h3 : distance2 = 300) (h4 : speed = distance1 / time1) :
  (time1 + distance2 / speed) = 4 :=
by
  sorry

end total_travel_time_l216_216111


namespace cos_B_in_triangle_l216_216742

theorem cos_B_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = (Real.sqrt 5 / 2) * b)
  (h2 : A = 2 * B)
  (h_triangle: A + B + C = Real.pi) : 
  Real.cos B = Real.sqrt 5 / 4 :=
sorry

end cos_B_in_triangle_l216_216742


namespace find_a_b_c_sum_l216_216459

theorem find_a_b_c_sum (a b c : ℤ)
  (h_gcd : gcd (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x + 1)
  (h_lcm : lcm (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x ^ 3 - 4 * x ^ 2 + x + 6) :
  a + b + c = -6 := 
sorry

end find_a_b_c_sum_l216_216459


namespace smallest_missing_unit_digit_l216_216487

theorem smallest_missing_unit_digit :
  (∀ n, n ∈ [0, 1, 4, 5, 6, 9]) → ∃ smallest_digit, smallest_digit = 2 :=
by
  sorry

end smallest_missing_unit_digit_l216_216487


namespace mass_of_man_l216_216631

def density_of_water : ℝ := 1000  -- kg/m³
def boat_length : ℝ := 4  -- meters
def boat_breadth : ℝ := 2  -- meters
def sinking_depth : ℝ := 0.01  -- meters (1 cm)

theorem mass_of_man
  (V : ℝ := boat_length * boat_breadth * sinking_depth)
  (m : ℝ := V * density_of_water) :
  m = 80 :=
by
  sorry

end mass_of_man_l216_216631


namespace population_exceeds_l216_216290

theorem population_exceeds (n : ℕ) : (∃ n, 4 * 3^n > 200) ∧ ∀ m, m < n → 4 * 3^m ≤ 200 := by
  sorry

end population_exceeds_l216_216290


namespace sam_total_cents_l216_216911

def dimes_to_cents (dimes : ℕ) : ℕ := dimes * 10
def quarters_to_cents (quarters : ℕ) : ℕ := quarters * 25
def nickels_to_cents (nickels : ℕ) : ℕ := nickels * 5
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

noncomputable def total_cents (initial_dimes dad_dimes mom_dimes grandma_dollars sister_quarters_initial : ℕ)
                             (initial_quarters dad_quarters mom_quarters grandma_transform sister_quarters_donation : ℕ)
                             (initial_nickels dad_nickels mom_nickels grandma_conversion sister_nickels_donation : ℕ) : ℕ :=
  dimes_to_cents initial_dimes +
  quarters_to_cents initial_quarters +
  nickels_to_cents initial_nickels +
  dimes_to_cents dad_dimes +
  quarters_to_cents dad_quarters -
  nickels_to_cents mom_nickels -
  dimes_to_cents mom_dimes +
  dollars_to_cents grandma_dollars +
  quarters_to_cents sister_quarters_donation +
  nickels_to_cents sister_nickels_donation

theorem sam_total_cents :
  total_cents 9 7 2 3 4 5 2 0 0 3 2 1 = 735 := 
  by exact sorry

end sam_total_cents_l216_216911


namespace newspaper_target_l216_216317

theorem newspaper_target (total_collected_2_weeks : Nat) (needed_more : Nat) (sections : Nat) (kilos_per_section_2_weeks : Nat)
  (h1 : sections = 6)
  (h2 : kilos_per_section_2_weeks = 280)
  (h3 : total_collected_2_weeks = sections * kilos_per_section_2_weeks)
  (h4 : needed_more = 320)
  : total_collected_2_weeks + needed_more = 2000 :=
by
  sorry

end newspaper_target_l216_216317


namespace common_ratio_l216_216578

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n, a (n+1) = r * a n)
variable (h1 : a 5 * a 11 = 3)
variable (h2 : a 3 + a 13 = 4)

theorem common_ratio (h_geom : ∀ n, a (n+1) = r * a n) (h1 : a 5 * a 11 = 3) (h2 : a 3 + a 13 = 4) :
  (r = 3 ∨ r = -3) :=
by
  sorry

end common_ratio_l216_216578


namespace handshake_count_l216_216845

def gathering_handshakes (total_people : ℕ) (know_each_other : ℕ) (know_no_one : ℕ) : ℕ :=
  let group2_handshakes := know_no_one * (total_people - 1)
  group2_handshakes / 2

theorem handshake_count :
  gathering_handshakes 30 20 10 = 145 :=
by
  sorry

end handshake_count_l216_216845


namespace compare_sine_values_1_compare_sine_values_2_l216_216682

theorem compare_sine_values_1 (h1 : 0 < Real.pi / 10) (h2 : Real.pi / 10 < Real.pi / 8) (h3 : Real.pi / 8 < Real.pi / 2) :
  Real.sin (- Real.pi / 10) > Real.sin (- Real.pi / 8) :=
by
  sorry

theorem compare_sine_values_2 (h1 : 0 < Real.pi / 8) (h2 : Real.pi / 8 < 3 * Real.pi / 8) (h3 : 3 * Real.pi / 8 < Real.pi / 2) :
  Real.sin (7 * Real.pi / 8) < Real.sin (5 * Real.pi / 8) :=
by
  sorry

end compare_sine_values_1_compare_sine_values_2_l216_216682


namespace sqrt_1708249_eq_1307_l216_216974

theorem sqrt_1708249_eq_1307 :
  ∃ (n : ℕ), n * n = 1708249 ∧ n = 1307 :=
sorry

end sqrt_1708249_eq_1307_l216_216974


namespace find_second_sum_l216_216836

theorem find_second_sum (x : ℝ) (h_sum : x + (2678 - x) = 2678)
  (h_interest : x * (3 / 100) * 8 = (2678 - x) * (5 / 100) * 3) :
  (2678 - x) = 2401 :=
by
  sorry

end find_second_sum_l216_216836


namespace tram_speed_l216_216592

theorem tram_speed
  (L v : ℝ)
  (h1 : L = 2 * v)
  (h2 : 96 + L = 10 * v) :
  v = 12 := 
by sorry

end tram_speed_l216_216592


namespace remaining_slices_after_weekend_l216_216035

theorem remaining_slices_after_weekend 
  (initial_pies : ℕ) (slices_per_pie : ℕ) (rebecca_initial_slices : ℕ) 
  (family_fraction : ℚ) (sunday_evening_slices : ℕ) : 
  initial_pies = 2 → 
  slices_per_pie = 8 → 
  rebecca_initial_slices = 2 → 
  family_fraction = 0.5 → 
  sunday_evening_slices = 2 → 
  (initial_pies * slices_per_pie 
   - rebecca_initial_slices 
   - family_fraction * (initial_pies * slices_per_pie - rebecca_initial_slices) 
   - sunday_evening_slices) = 5 :=
by 
  intros initial_pies_eq slices_per_pie_eq rebecca_initial_slices_eq family_fraction_eq sunday_evening_slices_eq
  sorry

end remaining_slices_after_weekend_l216_216035


namespace sum_six_consecutive_integers_l216_216778

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l216_216778


namespace student_A_more_stable_l216_216357

-- Define the variances for students A and B
def variance_A : ℝ := 0.05
def variance_B : ℝ := 0.06

-- The theorem to prove that student A has more stable performance
theorem student_A_more_stable : variance_A < variance_B :=
by {
  -- proof goes here
  sorry
}

end student_A_more_stable_l216_216357


namespace geometric_progression_condition_l216_216119

theorem geometric_progression_condition (a b c : ℝ) (h_b_neg : b < 0) : 
  (b^2 = a * c) ↔ (∃ (r : ℝ), a = r * b ∧ b = r * c) :=
sorry

end geometric_progression_condition_l216_216119


namespace min_distance_from_circle_to_line_l216_216424

-- Define the circle and line conditions
def is_on_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- The theorem to prove
theorem min_distance_from_circle_to_line (x y : ℝ) (h : is_on_circle x y) : 
  ∃ m_dist : ℝ, m_dist = 2 :=
by
  -- Place holder proof
  sorry

end min_distance_from_circle_to_line_l216_216424


namespace verka_digit_sets_l216_216620

-- Define the main conditions as:
def is_three_digit_number (a b c : ℕ) : Prop :=
  let num1 := 100 * a + 10 * b + c
  let num2 := 100 * a + 10 * c + b
  let num3 := 100 * b + 10 * a + c
  let num4 := 100 * b + 10 * c + a
  let num5 := 100 * c + 10 * a + b
  let num6 := 100 * c + 10 * b + a
  num1 + num2 + num3 + num4 + num5 + num6 = 1221

-- Prove the main theorem
theorem verka_digit_sets :
  ∃ (a b c : ℕ), is_three_digit_number a a c ∧
                 ((a, c) = (1, 9) ∨ (a, c) = (2, 7) ∨ (a, c) = (3, 5) ∨ (a, c) = (4, 3) ∨ (a, c) = (5, 1)) :=
by sorry

end verka_digit_sets_l216_216620


namespace total_recruits_211_l216_216614

theorem total_recruits_211 (P N D : ℕ) (total : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170) 
  (h4 : ∃ (x y : ℕ), (x = 4 * y ∨ y = 4 * x) ∧ 
                      ((x, P) = (y, N) ∨ (x, N) = (y, D) ∨ (x, P) = (y, D))) :
  total = 211 :=
by
  sorry

end total_recruits_211_l216_216614


namespace largest_stores_visited_l216_216637

theorem largest_stores_visited 
  (stores : ℕ) (total_visits : ℕ) (shoppers : ℕ) 
  (two_store_visitors : ℕ) (min_visits_per_person : ℕ)
  (h1 : stores = 8)
  (h2 : total_visits = 22)
  (h3 : shoppers = 12)
  (h4 : two_store_visitors = 8)
  (h5 : min_visits_per_person = 1)
  : ∃ (max_stores : ℕ), max_stores = 3 := 
by 
  -- Define the exact details given in the conditions
  have h_total_two_store_visits : two_store_visitors * 2 = 16 := by sorry
  have h_remaining_visits : total_visits - 16 = 6 := by sorry
  have h_remaining_shoppers : shoppers - two_store_visitors = 4 := by sorry
  have h_each_remaining_one_visit : 4 * 1 = 4 := by sorry
  -- Prove the largest number of stores visited by any one person is 3
  have h_max_stores : 1 + 2 = 3 := by sorry
  exact ⟨3, h_max_stores⟩

end largest_stores_visited_l216_216637


namespace factor_expression_l216_216683

variable (x : ℝ)

def e : ℝ := (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5)

theorem factor_expression : e x = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by
  sorry

end factor_expression_l216_216683


namespace sum_x_y_is_9_l216_216285

-- Definitions of the conditions
variables (x y S : ℝ)
axiom h1 : x + y = S
axiom h2 : x - y = 3
axiom h3 : x^2 - y^2 = 27

-- The theorem to prove
theorem sum_x_y_is_9 : S = 9 :=
by
  -- Placeholder for the proof
  sorry

end sum_x_y_is_9_l216_216285


namespace whole_numbers_between_cubicroots_l216_216135

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l216_216135


namespace geometric_sequence_sum_l216_216866

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (S_n : ℝ) (S_3n : ℝ) (S_4n : ℝ)
    (h1 : S_n = 2) 
    (h2 : S_3n = 14) 
    (h3 : ∀ m : ℕ, S_m = a_n 1 * (1 - q^m) / (1 - q)) :
    S_4n = 30 :=
by
  sorry

end geometric_sequence_sum_l216_216866


namespace percent_of_x_is_y_l216_216065

variable {x y : ℝ}

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.4 * (x + y)) :
  y = (1 / 9) * x :=
sorry

end percent_of_x_is_y_l216_216065


namespace sum_of_squares_l216_216513

theorem sum_of_squares (a b : ℕ) (h₁ : a = 300000) (h₂ : b = 20000) : a^2 + b^2 = 9004000000 :=
by
  rw [h₁, h₂]
  sorry

end sum_of_squares_l216_216513


namespace B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l216_216618

def prob_A_solve : ℝ := 0.8
def prob_B_solve : ℝ := 0.75

-- Definitions for A and B scoring in rounds
def prob_B_score_1_point : ℝ := 
  prob_B_solve * (1 - prob_B_solve) + (1 - prob_B_solve) * prob_B_solve

-- Definitions for A winning without a tiebreaker
def prob_A_score_1_point : ℝ :=
  prob_A_solve * (1 - prob_A_solve) + (1 - prob_A_solve) * prob_A_solve

def prob_A_score_2_points : ℝ :=
  prob_A_solve * prob_A_solve

def prob_B_score_0_points : ℝ :=
  (1 - prob_B_solve) * (1 - prob_B_solve)

def prob_B_score_total : ℝ :=
  prob_B_score_1_point

def prob_A_wins_without_tiebreaker : ℝ :=
  prob_A_score_2_points * prob_B_score_1_point +
  prob_A_score_2_points * prob_B_score_0_points +
  prob_A_score_1_point * prob_B_score_0_points

theorem B_score_1_probability_correct :
  prob_B_score_1_point = 3 / 8 := 
by
  sorry

theorem A_wins_without_tiebreaker_probability_correct :
  prob_A_wins_without_tiebreaker = 3 / 10 := 
by 
  sorry

end B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l216_216618


namespace proof_rp_eq_rq_l216_216121

noncomputable def triangle (A B C : Type) [linear_ordered_field A] [euclidean_geometry B] (BC : B) (ABC : BC > 0) : Type :=
{AB BC : B // (AB > BC)}

variables {A B C : Type} [linear_ordered_field A] [euclidean_geometry B] (ABC : triangle A B C)
variables {Ω : circle A B} {M N K P Q R : point B} (H1 : R ∈ Ω.mid_arc A B C) (H2 : angle.eq R "ABC")

-- Let M and N lie on sides AB and BC respectively such that AM = CN
variables (H3 : ∃ M N : point B, M ∈ ABC.AB ∧ N ∈ ABC.BC ∧ segment.length (ABC.A M) = segment.length (ABC.C N))

-- Let K be the intersection of MN and AC
variables (H4 : ∃ K : point B, K ∈ (line MN ∩ line AC))

-- P is the incenter of ΔAMK and Q is the K-excenter of ΔCNK
variables (H5 : ∃ P : point B, is_incenter (triangle AM K) P)
variables (H6 : ∃ Q : point B, is_excenter (triangle CN K) Q)

-- R is the midpoint of the arc ABC of Ω
variables (H7 : ∃ R : point B, is_arc_midpoint (circumcircle ABC) R)

-- Prove RP = RQ
theorem proof_rp_eq_rq (H8 : segment.length (segment RP) = segment.length (segment RQ)) : segment.eq (segment RP) (segment RQ) :=
by { sorry }

end proof_rp_eq_rq_l216_216121


namespace second_person_avg_pages_per_day_l216_216103

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end second_person_avg_pages_per_day_l216_216103


namespace olivia_remaining_usd_l216_216301

def initial_usd : ℝ := 78
def initial_eur : ℝ := 50
def exchange_rate : ℝ := 1.20
def spent_usd_supermarket : ℝ := 15
def book_eur : ℝ := 10
def spent_usd_lunch : ℝ := 12

theorem olivia_remaining_usd :
  let total_usd := initial_usd + (initial_eur * exchange_rate)
  let remaining_after_supermarket := total_usd - spent_usd_supermarket
  let remaining_after_book := remaining_after_supermarket - (book_eur * exchange_rate)
  let final_remaining := remaining_after_book - spent_usd_lunch
  final_remaining = 99 :=
by
  sorry

end olivia_remaining_usd_l216_216301


namespace arithmetic_sequence_sum_l216_216267

variable {S : ℕ → ℕ}

theorem arithmetic_sequence_sum (h1 : S 3 = 15) (h2 : S 9 = 153) : S 6 = 66 :=
sorry

end arithmetic_sequence_sum_l216_216267


namespace quadratic_eq_standard_form_coefficients_l216_216516

-- Define initial quadratic equation
def initial_eq (x : ℝ) : Prop := (x + 5) * (x + 3) = 2 * x^2

-- Define the quadratic equation in standard form
def standard_form (x : ℝ) : Prop := x^2 - 8 * x - 15 = 0

-- Prove that given the initial equation, it can be converted to its standard form
theorem quadratic_eq_standard_form (x : ℝ) :
  initial_eq x → standard_form x := 
sorry

-- Verify the coefficients of the quadratic term, linear term, and constant term
theorem coefficients (x : ℝ) :
  initial_eq x → 
  (∀ a b c : ℝ, (a = 1) ∧ (b = -8) ∧ (c = -15) → standard_form x) :=
sorry

end quadratic_eq_standard_form_coefficients_l216_216516


namespace profit_percent_is_25_l216_216636

-- Define the cost price (CP) and selling price (SP) based on the given ratio.
def CP (x : ℝ) := 4 * x
def SP (x : ℝ) := 5 * x

-- Calculate the profit percent based on the given conditions.
noncomputable def profitPercent (x : ℝ) := ((SP x - CP x) / CP x) * 100

-- Prove that the profit percent is 25% given the ratio of CP to SP is 4:5.
theorem profit_percent_is_25 (x : ℝ) : profitPercent x = 25 := by
  sorry

end profit_percent_is_25_l216_216636


namespace probability_shorts_not_equal_jersey_l216_216173

-- Definitions based on conditions
def color := {black, gold, white}

def shorts_color : color := _
def jersey_color : color := _

-- Assuming independent and equally likely choices for shorts and jerseys
def total_combinations := 3 * 3
def different_color_combinations := 2 + 2 + 2

-- Lean statement to prove the probability
theorem probability_shorts_not_equal_jersey : 
  (different_color_combinations : ℚ) / total_combinations = 2 / 3 := 
by sorry

end probability_shorts_not_equal_jersey_l216_216173


namespace sandwich_cost_is_5_l216_216931

-- We define the variables and conditions first
def total_people := 4
def sandwiches := 4
def fruit_salads := 4
def sodas := 8
def snack_bags := 3

def fruit_salad_cost_per_unit := 3
def soda_cost_per_unit := 2
def snack_bag_cost_per_unit := 4
def total_cost := 60

-- We now define the calculations based on the given conditions
def total_fruit_salad_cost := fruit_salads * fruit_salad_cost_per_unit
def total_soda_cost := sodas * soda_cost_per_unit
def total_snack_bag_cost := snack_bags * snack_bag_cost_per_unit
def other_items_cost := total_fruit_salad_cost + total_soda_cost + total_snack_bag_cost
def remaining_budget := total_cost - other_items_cost
def sandwich_cost := remaining_budget / sandwiches

-- The final proof problem statement in Lean 4
theorem sandwich_cost_is_5 : sandwich_cost = 5 := by
  sorry

end sandwich_cost_is_5_l216_216931


namespace smallest_n_for_common_factor_l216_216623

theorem smallest_n_for_common_factor : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (11 * n - 3) (8 * n + 4) > 1) ∧ n = 42 := 
by
  sorry

end smallest_n_for_common_factor_l216_216623


namespace probability_exactly_three_correct_l216_216697

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l216_216697


namespace maxwell_walking_speed_l216_216589

theorem maxwell_walking_speed :
  ∃ v : ℝ, (8 * v + 6 * 7 = 74) ∧ v = 4 :=
by
  exists 4
  constructor
  { norm_num }
  rfl

end maxwell_walking_speed_l216_216589


namespace percentage_of_non_honda_red_cars_l216_216206

/-- 
Total car population in Chennai is 9000.
Honda cars in Chennai is 5000.
Out of every 100 Honda cars, 90 are red.
60% of the total car population is red.
Prove that the percentage of non-Honda cars that are red is 22.5%.
--/
theorem percentage_of_non_honda_red_cars 
  (total_cars : ℕ) (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) (total_red_ratio : ℚ) 
  (h : total_cars = 9000) 
  (h1 : honda_cars = 5000) 
  (h2 : red_honda_ratio = 90 / 100) 
  (h3 : total_red_ratio = 60 / 100) : 
  (900 / (9000 - 5000) * 100 = 22.5) := 
sorry

end percentage_of_non_honda_red_cars_l216_216206


namespace exists_person_who_knows_everyone_l216_216886

variable {Person : Type}
variable (knows : Person → Person → Prop)
variable (n : ℕ)

-- Condition: In a company of 2n + 1 people, for any n people, there is another person different from them who knows each of them.
axiom knows_condition : ∀ (company : Finset Person) (h : company.card = 2 * n + 1), 
  (∀ (subset : Finset Person) (hs : subset.card = n), ∃ (p : Person), p ∉ subset ∧ ∀ q ∈ subset, knows p q)

-- Statement to be proven:
theorem exists_person_who_knows_everyone (company : Finset Person) (hcompany : company.card = 2 * n + 1) :
  ∃ p, ∀ q ∈ company, knows p q :=
sorry

end exists_person_who_knows_everyone_l216_216886


namespace not_square_difference_formula_l216_216940

theorem not_square_difference_formula (x y : ℝ) : ¬ ∃ (a b : ℝ), (x - y) * (-x + y) = (a + b) * (a - b) := 
sorry

end not_square_difference_formula_l216_216940


namespace max_mn_sq_l216_216242

theorem max_mn_sq {m n : ℤ} (h1: 1 ≤ m ∧ m ≤ 2005) (h2: 1 ≤ n ∧ n ≤ 2005) 
(h3: (n^2 + 2*m*n - 2*m^2)^2 = 1): m^2 + n^2 ≤ 702036 :=
sorry

end max_mn_sq_l216_216242


namespace correct_conclusions_count_l216_216128

theorem correct_conclusions_count :
  (¬ (¬ p → (q ∨ r)) ↔ (¬ p → ¬ q ∧ ¬ r)) = false ∧
  ((¬ p → q) ↔ (p → ¬ q)) = false ∧
  (¬ ∃ n : ℕ, n > 0 ∧ (n ^ 2 + 3 * n) % 10 = 0 ∧ (∀ n : ℕ, n > 0 → (n ^ 2 + 3 * n) % 10 ≠ 0)) = true ∧
  (¬ ∀ x, x ^ 2 - 2 * x + 3 > 0 ∧ (∃ x, x ^ 2 - 2 * x + 3 < 0)) = false :=
by
  sorry

end correct_conclusions_count_l216_216128


namespace fraction_of_field_planted_l216_216112

theorem fraction_of_field_planted (a b : ℕ) (d : ℝ) :
  a = 5 → b = 12 → d = 3 →
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let side_square := (d * hypotenuse - d^2)/(a + b - 2 * d)
  let area_square := side_square^2
  let area_triangle : ℝ := 1/2 * a * b
  let planted_area := area_triangle - area_square
  let fraction_planted := planted_area / area_triangle
  fraction_planted = 9693/10140 := by
  sorry

end fraction_of_field_planted_l216_216112


namespace candy_ratio_l216_216169

theorem candy_ratio 
  (tabitha_candy : ℕ)
  (stan_candy : ℕ)
  (julie_candy : ℕ)
  (carlos_candy : ℕ)
  (total_candy : ℕ)
  (h1 : tabitha_candy = 22)
  (h2 : stan_candy = 13)
  (h3 : julie_candy = tabitha_candy / 2)
  (h4 : total_candy = 72)
  (h5 : tabitha_candy + stan_candy + julie_candy + carlos_candy = total_candy) :
  carlos_candy / stan_candy = 2 :=
by
  sorry

end candy_ratio_l216_216169


namespace quadratic_function_points_relationship_l216_216306

theorem quadratic_function_points_relationship (c y1 y2 y3 : ℝ) 
  (h₁ : y1 = -((-1) ^ 2) + 2 * (-1) + c)
  (h₂ : y2 = -(2 ^ 2) + 2 * 2 + c)
  (h₃ : y3 = -(5 ^ 2) + 2 * 5 + c) :
  y2 > y1 ∧ y1 > y3 :=
by
  sorry

end quadratic_function_points_relationship_l216_216306


namespace x_pow_y_equals_nine_l216_216733

theorem x_pow_y_equals_nine (x y : ℝ) (h : (|x + 3| * (y - 2)^2 < 0)) : x^y = 9 :=
sorry

end x_pow_y_equals_nine_l216_216733


namespace sum_smallest_largest_l216_216783

theorem sum_smallest_largest (z b : ℤ) (n : ℤ) (h_even_n : (n % 2 = 0)) (h_mean : z = (n * b + ((n - 1) * n) / 2) / n) : 
  (2 * (z - (n - 1) / 2) + n - 1) = 2 * z := by
  sorry

end sum_smallest_largest_l216_216783


namespace triangle_angle_bisectors_l216_216985

theorem triangle_angle_bisectors (α β γ : ℝ) 
  (h1 : α + β + γ = 180)
  (h2 : α = 100) 
  (h3 : β = 30) 
  (h4 : γ = 50) :
  ∃ α' β' γ', α' = 40 ∧ β' = 65 ∧ γ' = 75 :=
sorry

end triangle_angle_bisectors_l216_216985


namespace increase_in_output_with_assistant_l216_216010

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l216_216010


namespace magnitude_squared_complex_l216_216976

noncomputable def complex_number := Complex.mk 3 (-4)
noncomputable def squared_complex := complex_number * complex_number

theorem magnitude_squared_complex : Complex.abs squared_complex = 25 :=
by
  sorry

end magnitude_squared_complex_l216_216976


namespace negate_proposition_l216_216923

theorem negate_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
by
  sorry

end negate_proposition_l216_216923


namespace least_number_to_subtract_l216_216069

theorem least_number_to_subtract {x : ℕ} (h : x = 13604) : 
    ∃ n : ℕ, n = 32 ∧ (13604 - n) % 87 = 0 :=
by
  sorry

end least_number_to_subtract_l216_216069


namespace smallest_n_congruent_5n_eq_n5_mod_7_l216_216244

theorem smallest_n_congruent_5n_eq_n5_mod_7 : ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, 5^m % 7 ≠ m^5 % 7 → m ≥ n) :=
by
  use 6
  -- Proof steps here which are skipped
  sorry

end smallest_n_congruent_5n_eq_n5_mod_7_l216_216244


namespace average_remaining_two_numbers_l216_216634

theorem average_remaining_two_numbers 
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (h_avg_6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.80)
  (h_avg_2_1 : (a1 + a2) / 2 = 2.4)
  (h_avg_2_2 : (a3 + a4) / 2 = 2.3) :
  (a5 + a6) / 2 = 3.7 :=
by
  sorry

end average_remaining_two_numbers_l216_216634


namespace relationship_p_q_l216_216117

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem relationship_p_q (x a p q : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1)
  (hp : p = |log_a a (1 + x)|) (hq : q = |log_a a (1 - x)|) : p ≤ q :=
sorry

end relationship_p_q_l216_216117


namespace find_x_l216_216652

theorem find_x (x : ℝ) (h : 5020 - (x / 100.4) = 5015) : x = 502 :=
sorry

end find_x_l216_216652


namespace bean_seedlings_l216_216663

theorem bean_seedlings
  (beans_per_row : ℕ)
  (pumpkins : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (radishes_per_row : ℕ)
  (rows_per_bed : ℕ) (beds : ℕ)
  (H_beans_per_row : beans_per_row = 8)
  (H_pumpkins : pumpkins = 84)
  (H_pumpkins_per_row : pumpkins_per_row = 7)
  (H_radishes : radishes = 48)
  (H_radishes_per_row : radishes_per_row = 6)
  (H_rows_per_bed : rows_per_bed = 2)
  (H_beds : beds = 14) :
  (beans_per_row * ((beds * rows_per_bed) - (pumpkins / pumpkins_per_row) - (radishes / radishes_per_row)) = 64) :=
by
  sorry

end bean_seedlings_l216_216663


namespace race_time_l216_216745

theorem race_time (t : ℝ) (h1 : 100 / t = 66.66666666666667 / 45) : t = 67.5 :=
by
  sorry

end race_time_l216_216745


namespace jims_sum_divided_by_anas_sum_l216_216581

noncomputable def sum_of_squares_of_odds (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), if k % 2 = 1 then (k:ℝ)^2 else 0

noncomputable def sum_of_first_n_integers (n : ℕ) : ℝ :=
  n * (n + 1) / 2

theorem jims_sum_divided_by_anas_sum : 
  (sum_of_squares_of_odds 249) / (sum_of_first_n_integers 249) = 1001 / 6 := 
by
  sorry

end jims_sum_divided_by_anas_sum_l216_216581


namespace reservoir_water_level_l216_216826

theorem reservoir_water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 6 + 0.3 * x :=
by sorry

end reservoir_water_level_l216_216826


namespace number_b_smaller_than_number_a_l216_216916

theorem number_b_smaller_than_number_a (A B : ℝ)
  (h : A = B + 1/4) : (B + 1/4 = A) ∧ (B < A) → B = (4 * A - A) / 5 := by
  sorry

end number_b_smaller_than_number_a_l216_216916


namespace increase_in_output_with_assistant_l216_216008

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l216_216008


namespace abc_sum_eq_11sqrt6_l216_216275

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l216_216275


namespace reflected_ray_equation_l216_216671

-- Define the initial point
def point_of_emanation : (ℝ × ℝ) := (-1, 3)

-- Define the point after reflection which the ray passes through
def point_after_reflection : (ℝ × ℝ) := (4, 6)

-- Define the expected equation of the line in general form
def expected_line_equation (x y : ℝ) : Prop := 9 * x - 5 * y - 6 = 0

-- The theorem we need to prove
theorem reflected_ray_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) → expected_line_equation x y :=
sorry

end reflected_ray_equation_l216_216671


namespace hannah_bought_two_sets_of_measuring_spoons_l216_216272

-- Definitions of conditions
def number_of_cookies_sold : ℕ := 40
def price_per_cookie : ℝ := 0.8
def number_of_cupcakes_sold : ℕ := 30
def price_per_cupcake : ℝ := 2.0
def cost_per_measuring_spoon_set : ℝ := 6.5
def remaining_money : ℝ := 79

-- Definition of total money made from selling cookies and cupcakes
def total_money_made : ℝ := (number_of_cookies_sold * price_per_cookie) + (number_of_cupcakes_sold * price_per_cupcake)

-- Definition of money spent on measuring spoons
def money_spent_on_measuring_spoons : ℝ := total_money_made - remaining_money

-- Theorem statement
theorem hannah_bought_two_sets_of_measuring_spoons :
  (money_spent_on_measuring_spoons / cost_per_measuring_spoon_set) = 2 := by
  sorry

end hannah_bought_two_sets_of_measuring_spoons_l216_216272


namespace find_the_number_l216_216377

theorem find_the_number (x : ℝ) (h : 8 * x + 64 = 336) : x = 34 :=
by
  sorry

end find_the_number_l216_216377


namespace emmalyn_earnings_l216_216397

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l216_216397


namespace evaluate_sets_are_equal_l216_216679

theorem evaluate_sets_are_equal :
  (-3^5) = ((-3)^5) ∧
  ¬ ((-2^2) = ((-2)^2)) ∧
  ¬ ((-4 * 2^3) = (-4^2 * 3)) ∧
  ¬ ((- (-3)^2) = (- (-2)^3)) :=
by
  sorry

end evaluate_sets_are_equal_l216_216679


namespace algebraic_expression_value_l216_216419

theorem algebraic_expression_value (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m + 2021 = 2023 := 
sorry

end algebraic_expression_value_l216_216419


namespace total_hotdogs_sold_l216_216829

-- Define the number of small and large hotdogs
def small_hotdogs : ℕ := 58
def large_hotdogs : ℕ := 21

-- Define the total hotdogs
def total_hotdogs : ℕ := small_hotdogs + large_hotdogs

-- The Main Statement to prove the total number of hotdogs sold
theorem total_hotdogs_sold : total_hotdogs = 79 :=
by
  -- Proof is skipped using sorry
  sorry

end total_hotdogs_sold_l216_216829


namespace cucumbers_count_l216_216089

theorem cucumbers_count (C T : ℕ) 
  (h1 : C + T = 280)
  (h2 : T = 3 * C) : C = 70 :=
by sorry

end cucumbers_count_l216_216089


namespace all_suits_different_in_groups_of_four_l216_216287

-- Define the alternation pattern of the suits in the deck of 36 cards
def suits : List String := ["spades", "clubs", "hearts", "diamonds"]

-- Formalize the condition that each 4-card group in the deck contains all different suits
def suits_includes_all (cards : List String) : Prop :=
  ∀ i j, i < 4 → j < 4 → i ≠ j → cards.get? i ≠ cards.get? j

-- The main theorem statement
theorem all_suits_different_in_groups_of_four (L : List String)
  (hL : L.length = 36)
  (hA : ∀ n, n < 9 → L.get? (4*n) = some "spades" ∧ L.get? (4*n + 1) = some "clubs" ∧ L.get? (4*n + 2) = some "hearts" ∧ L.get? (4*n + 3) = some "diamonds"):
  ∀ cut reversed_deck, (@List.append String (List.reverse (List.take cut L)) (List.drop cut L) = reversed_deck)
  → ∀ n, n < 9 → suits_includes_all (List.drop (4*n) (List.take 4 reversed_deck)) := sorry

end all_suits_different_in_groups_of_four_l216_216287


namespace g_9_l216_216337

variable (g : ℝ → ℝ)

-- Conditions
axiom func_eq : ∀ x y : ℝ, g(x + y) = g(x) * g(y)
axiom g_3 : g 3 = 4

-- Theorem to prove
theorem g_9 : g 9 = 64 :=
by
  sorry

end g_9_l216_216337


namespace largest_circle_area_rounded_to_nearest_int_l216_216378

theorem largest_circle_area_rounded_to_nearest_int
  (x : Real)
  (hx : 3 * x^2 = 180) :
  let r := (16 * Real.sqrt 15) / (2 * Real.pi)
  let area_of_circle := Real.pi * r^2
  round (area_of_circle) = 306 :=
by
  sorry

end largest_circle_area_rounded_to_nearest_int_l216_216378


namespace range_of_x_l216_216284

variable (a x : ℝ)

theorem range_of_x :
  (∃ a ∈ Set.Icc 2 4, a * x ^ 2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 4) :=
by
  sorry

end range_of_x_l216_216284


namespace salad_cucumbers_l216_216087

theorem salad_cucumbers (c t : ℕ) 
  (h1 : c + t = 280)
  (h2 : t = 3 * c) : c = 70 :=
sorry

end salad_cucumbers_l216_216087


namespace find_x_l216_216519

theorem find_x (x y : ℝ) (h : y ≠ -5 * x) : (x - 5) / (5 * x + y) = 0 → x = 5 := by
  sorry

end find_x_l216_216519


namespace g_9_eq_64_l216_216330

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g(x + y) = g(x) * g(y)

axiom g_3_eq_4 : g(3) = 4

theorem g_9_eq_64 : g(9) = 64 := by
  sorry

end g_9_eq_64_l216_216330


namespace plant_supplier_earnings_l216_216668

theorem plant_supplier_earnings :
  let orchids_price := 50
  let orchids_sold := 20
  let money_plant_price := 25
  let money_plants_sold := 15
  let worker_wage := 40
  let workers := 2
  let pot_cost := 150
  let total_earnings := (orchids_price * orchids_sold) + (money_plant_price * money_plants_sold)
  let total_expense := (worker_wage * workers) + pot_cost
  total_earnings - total_expense = 1145 :=
by
  sorry

end plant_supplier_earnings_l216_216668


namespace largest_consecutive_multiple_of_3_l216_216930

theorem largest_consecutive_multiple_of_3 (n : ℕ) 
  (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 72) : 3 * (n + 2) = 27 :=
by 
  sorry

end largest_consecutive_multiple_of_3_l216_216930


namespace children_ticket_cost_is_8_l216_216904

-- Defining the costs of different tickets
def adult_ticket_cost : ℕ := 11
def senior_ticket_cost : ℕ := 9
def total_tickets_cost : ℕ := 64

-- Number of tickets needed
def number_of_adult_tickets : ℕ := 2
def number_of_senior_tickets : ℕ := 2
def number_of_children_tickets : ℕ := 3

-- Defining the total cost equation using the price of children's tickets (C)
def total_cost (children_ticket_cost : ℕ) : ℕ :=
  number_of_adult_tickets * adult_ticket_cost +
  number_of_senior_tickets * senior_ticket_cost +
  number_of_children_tickets * children_ticket_cost

-- Statement to prove that the children's ticket cost is $8
theorem children_ticket_cost_is_8 : (C : ℕ) → total_cost C = total_tickets_cost → C = 8 :=
by
  intro C h
  sorry

end children_ticket_cost_is_8_l216_216904


namespace increase_in_average_weight_l216_216887

theorem increase_in_average_weight :
  let initial_group_size := 6
  let initial_weight := 65
  let new_weight := 74
  let initial_avg_weight := A
  (new_weight - initial_weight) / initial_group_size = 1.5 := by
    sorry

end increase_in_average_weight_l216_216887


namespace A_equals_half_C_equals_half_l216_216492

noncomputable def A := 2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)
noncomputable def C := Real.sin (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - Real.cos (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180)

theorem A_equals_half : A = 1 / 2 := 
by
  sorry

theorem C_equals_half : C = 1 / 2 := 
by
  sorry

end A_equals_half_C_equals_half_l216_216492


namespace first_pump_rate_is_180_l216_216965

-- Define the known conditions
variables (R : ℕ) -- The rate of the first pump in gallons per hour
def second_pump_rate : ℕ := 250 -- The rate of the second pump in gallons per hour
def second_pump_time : ℕ := 35 / 10 -- 3.5 hours represented as a fraction
def total_pump_time : ℕ := 60 / 10 -- 6 hours represented as a fraction
def total_volume : ℕ := 1325 -- Total volume pumped by both pumps in gallons

-- Define derived conditions from the problem
def second_pump_volume : ℕ := second_pump_rate * second_pump_time -- Volume pumped by the second pump
def first_pump_volume : ℕ := total_volume - second_pump_volume -- Volume pumped by the first pump
def first_pump_time : ℕ := total_pump_time - second_pump_time -- Time the first pump was used

-- The main theorem to prove that the rate of the first pump is 180 gallons per hour
theorem first_pump_rate_is_180 : R = 180 :=
by
  -- The proof would go here
  sorry

end first_pump_rate_is_180_l216_216965


namespace max_value_2xy_sqrt6_8yz2_l216_216446

theorem max_value_2xy_sqrt6_8yz2 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
sorry

end max_value_2xy_sqrt6_8yz2_l216_216446


namespace solutions_to_equation_l216_216248

theorem solutions_to_equation :
  ∀ x : ℝ, 
  sqrt ((3 + sqrt 5) ^ x) + sqrt ((3 - sqrt 5) ^ x) = 6 ↔ (x = 2 ∨ x = -2) :=
by
  intros x
  sorry

end solutions_to_equation_l216_216248


namespace percent_increase_output_l216_216012

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l216_216012


namespace probability_of_exactly_three_correct_packages_l216_216704

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l216_216704


namespace negation_of_exists_l216_216179

theorem negation_of_exists : (¬ ∃ x : ℝ, x > 0 ∧ x^2 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 ≤ 0 :=
by sorry

end negation_of_exists_l216_216179


namespace number_is_square_l216_216990

theorem number_is_square (n : ℕ) (h : (n^5 + n^4 + 1).factors.length = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
begin
  sorry
end

end number_is_square_l216_216990


namespace wednesday_tips_value_l216_216880

-- Definitions for the conditions
def hourly_wage : ℕ := 10
def monday_hours : ℕ := 7
def tuesday_hours : ℕ := 5
def wednesday_hours : ℕ := 7
def monday_tips : ℕ := 18
def tuesday_tips : ℕ := 12
def total_earnings : ℕ := 240

-- Hourly earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def wednesday_earnings := wednesday_hours * hourly_wage

-- Total wage earnings
def total_wage_earnings := monday_earnings + tuesday_earnings + wednesday_earnings

-- Total earnings with known tips
def known_earnings := total_wage_earnings + monday_tips + tuesday_tips

-- Prove that Wednesday tips is $20
theorem wednesday_tips_value : (total_earnings - known_earnings) = 20 := by
  sorry

end wednesday_tips_value_l216_216880


namespace intersection_sets_l216_216414

theorem intersection_sets :
  let M := {x : ℝ | 0 < x} 
  let N := {y : ℝ | 1 ≤ y}
  M ∩ N = {z : ℝ | 1 ≤ z} :=
by
  -- Proof goes here
  sorry

end intersection_sets_l216_216414


namespace remainder_when_divided_by_11_l216_216243

theorem remainder_when_divided_by_11 :
  (7 * 10^20 + 2^20) % 11 = 8 := by
sorry

end remainder_when_divided_by_11_l216_216243


namespace even_function_A_value_l216_216738

-- Given function definition
def f (x : ℝ) (A : ℝ) : ℝ := (x + 1) * (x - A)

-- Statement to prove
theorem even_function_A_value (A : ℝ) (h : ∀ x : ℝ, f x A = f (-x) A) : A = 1 :=
by
  sorry

end even_function_A_value_l216_216738


namespace fraction_study_only_japanese_l216_216844

variable (J : ℕ)

def seniors := 2 * J
def sophomores := (3 / 4) * J

def seniors_study_japanese := (3 / 8) * seniors J
def juniors_study_japanese := (1 / 4) * J
def sophomores_study_japanese := (2 / 5) * sophomores J

def seniors_study_both := (1 / 6) * seniors J
def juniors_study_both := (1 / 12) * J
def sophomores_study_both := (1 / 10) * sophomores J

def seniors_study_only_japanese := seniors_study_japanese J - seniors_study_both J
def juniors_study_only_japanese := juniors_study_japanese J - juniors_study_both J
def sophomores_study_only_japanese := sophomores_study_japanese J - sophomores_study_both J

def total_study_only_japanese := seniors_study_only_japanese J + juniors_study_only_japanese J + sophomores_study_only_japanese J
def total_students := J + seniors J + sophomores J

theorem fraction_study_only_japanese :
  (total_study_only_japanese J) / (total_students J) = 97 / 450 :=
by sorry

end fraction_study_only_japanese_l216_216844


namespace g_9_eq_64_l216_216331

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g(x + y) = g(x) * g(y)

axiom g_3_eq_4 : g(3) = 4

theorem g_9_eq_64 : g(9) = 64 := by
  sorry

end g_9_eq_64_l216_216331


namespace triangle_inequality_sum_l216_216126

theorem triangle_inequality_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (c / (a + b)) + (a / (b + c)) + (b / (c + a)) > 1 :=
by
  sorry

end triangle_inequality_sum_l216_216126


namespace sum_of_six_consecutive_integers_l216_216780

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l216_216780


namespace square_implies_increasing_l216_216615

def seq (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n > 1, 
    ((a n - 2 > 0 ∧ ¬(∃ m < n, a m = a n - 2)) → a (n + 1) = a n - 2) ∧
    ((a n - 2 ≤ 0 ∨ ∃ m < n, a m = a n - 2) → a (n + 1) = a n + 3)

theorem square_implies_increasing (a : ℕ → ℤ) (n : ℕ) (h_seq : seq a) 
  (h_square : ∃ k, a n = k^2) (h_n_pos : n > 1) : 
  a n > a (n - 1) :=
sorry

end square_implies_increasing_l216_216615


namespace candidate_lost_by_votes_l216_216655

theorem candidate_lost_by_votes :
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  candidate_votes <= 6450 ∧ rival_votes <= 6450 ∧ rival_votes - candidate_votes = 2451 :=
by
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  have h1: candidate_votes <= 6450 := sorry
  have h2: rival_votes <= 6450 := sorry
  have h3: rival_votes - candidate_votes = 2451 := sorry
  exact ⟨h1, h2, h3⟩

end candidate_lost_by_votes_l216_216655


namespace Alyssa_next_year_games_l216_216968

theorem Alyssa_next_year_games 
  (games_this_year : ℕ) 
  (games_last_year : ℕ) 
  (total_games : ℕ) 
  (games_up_to_this_year : ℕ)
  (total_up_to_next_year : ℕ) 
  (H1 : games_this_year = 11)
  (H2 : games_last_year = 13)
  (H3 : total_up_to_next_year = 39)
  (H4 : games_up_to_this_year = games_this_year + games_last_year) :
  total_up_to_next_year - games_up_to_this_year = 15 :=
by
  sorry

end Alyssa_next_year_games_l216_216968


namespace max_time_digit_sum_l216_216215

-- Define the conditions
def is_valid_time (h m : ℕ) : Prop :=
  (0 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60)

-- Define the function to calculate the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  n % 10 + n / 10

-- Define the function to calculate the sum of digits in the time display
def time_digit_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

-- The theorem to prove
theorem max_time_digit_sum : ∀ (h m : ℕ),
  is_valid_time h m → time_digit_sum h m ≤ 24 :=
by {
  sorry
}

end max_time_digit_sum_l216_216215


namespace coke_to_sprite_ratio_l216_216441

theorem coke_to_sprite_ratio
  (x : ℕ) 
  (Coke Sprite MountainDew : ℕ)
  (total_volume : ℕ) 
  (coke_volume : ℕ)
  (ratio_condition : Coke : Sprite : MountainDew = x : 1 : 3)
  (coke_volume_condition : coke_volume = 6)
  (total_volume_condition : total_volume = 18) :
  (Coke = 2) :=
by
  sorry

end coke_to_sprite_ratio_l216_216441


namespace probability_male_is_2_5_l216_216954

variable (num_male_students num_female_students : ℕ)

def total_students (num_male_students num_female_students : ℕ) : ℕ :=
  num_male_students + num_female_students

def probability_of_male (num_male_students num_female_students : ℕ) : ℚ :=
  num_male_students / (total_students num_male_students num_female_students : ℚ)

theorem probability_male_is_2_5 :
  probability_of_male 2 3 = 2 / 5 := by
    sorry

end probability_male_is_2_5_l216_216954


namespace largest_constant_c_l216_216858

theorem largest_constant_c (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 1) : 
  x^6 + y^6 ≥ (1 / 2) * x * y :=
sorry

end largest_constant_c_l216_216858


namespace rectangle_area_error_percentage_l216_216204

theorem rectangle_area_error_percentage 
  (L W : ℝ)
  (measured_length : ℝ := L * 1.16)
  (measured_width : ℝ := W * 0.95)
  (actual_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width) :
  ((measured_area - actual_area) / actual_area) * 100 = 10.2 := 
by
  sorry

end rectangle_area_error_percentage_l216_216204


namespace cos_960_eq_neg_half_l216_216688

theorem cos_960_eq_neg_half (cos : ℝ → ℝ) (h1 : ∀ x, cos (x + 360) = cos x) 
  (h_even : ∀ x, cos (-x) = cos x) (h_cos120 : cos 120 = - cos 60)
  (h_cos60 : cos 60 = 1 / 2) : cos 960 = -(1 / 2) := by
  sorry

end cos_960_eq_neg_half_l216_216688


namespace toll_for_18_wheel_truck_l216_216800

noncomputable def toll (x : ℕ) : ℝ :=
  2.50 + 0.50 * (x - 2)

theorem toll_for_18_wheel_truck :
  let num_wheels := 18
  let wheels_on_front_axle := 2
  let wheels_per_other_axle := 4
  let num_other_axles := (num_wheels - wheels_on_front_axle) / wheels_per_other_axle
  let total_num_axles := num_other_axles + 1
  toll total_num_axles = 4.00 :=
by
  sorry

end toll_for_18_wheel_truck_l216_216800


namespace n_digit_numbers_modulo_3_l216_216548

def a (i : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then if i = 0 then 1 else 0 else 2 * a i (n - 1) + a ((i + 1) % 3) (n - 1) + a ((i + 2) % 3) (n - 1)

theorem n_digit_numbers_modulo_3 (n : ℕ) (h : 0 < n) : 
  (a 0 n) = (4^n + 2) / 3 :=
sorry

end n_digit_numbers_modulo_3_l216_216548


namespace present_condition_l216_216148

variable {α : Type} [Finite α]

-- We will represent children as members of a type α and assume there are precisely 3n children.
variable (n : ℕ) (h_odd : odd n) [h : Fintype α] (card_3n : Fintype.card α = 3 * n)

noncomputable def makes_present_to (A B : α) : α := sorry -- Create a function that maps pairs of children to exactly one child.

theorem present_condition : ∀ (A B C : α), makes_present_to A B = C → makes_present_to A C = B :=
sorry

end present_condition_l216_216148


namespace minimum_value_l216_216435

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 / x + 1 / y = 1) : 3 * x + 4 * y ≥ 25 :=
sorry

end minimum_value_l216_216435


namespace original_cost_l216_216497

theorem original_cost (P : ℝ) (h : 0.76 * P = 608) : P = 800 :=
by
  sorry

end original_cost_l216_216497


namespace students_not_in_any_activity_l216_216289

def total_students : ℕ := 1500
def students_chorus : ℕ := 420
def students_band : ℕ := 780
def students_chorus_and_band : ℕ := 150
def students_drama : ℕ := 300
def students_drama_and_other : ℕ := 50

theorem students_not_in_any_activity :
  total_students - ((students_chorus + students_band - students_chorus_and_band) + (students_drama - students_drama_and_other)) = 200 :=
by
  sorry

end students_not_in_any_activity_l216_216289


namespace length_AC_and_area_OAC_l216_216259

open Real EuclideanGeometry

def ellipse (x y : ℝ) : Prop :=
  x^2 + 2 * y^2 = 2

def line_1 (x y : ℝ) : Prop :=
  y = x + 1

def line_2 (B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  B.fst = 3 * P.fst ∧ B.snd = 3 * P.snd

theorem length_AC_and_area_OAC 
  (A C : ℝ × ℝ) 
  (B P : ℝ × ℝ) 
  (O : ℝ × ℝ := (0, 0)) 
  (h1 : ellipse A.fst A.snd) 
  (h2 : ellipse C.fst C.snd) 
  (h3 : line_1 A.fst A.snd) 
  (h4 : line_1 C.fst C.snd) 
  (h5 : line_2 B P) 
  (h6 : (P.fst = (A.fst + C.fst) / 2) ∧ (P.snd = (A.snd + C.snd) / 2)) : 
  |(dist A C)| = 4/3 * sqrt 2 ∧
  (1/2 * abs (A.fst * C.snd - C.fst * A.snd)) = 4/9 := sorry

end length_AC_and_area_OAC_l216_216259


namespace solve_diophantine_equations_l216_216114

theorem solve_diophantine_equations :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    a * b - 2 * c * d = 3 ∧
    a * c + b * d = 1 } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end solve_diophantine_equations_l216_216114


namespace find_g7_l216_216922

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_value : g 6 = 7

theorem find_g7 : g 7 = 49 / 6 := by
  sorry

end find_g7_l216_216922


namespace symmetric_linear_functions_l216_216132

theorem symmetric_linear_functions :
  (∃ (a b : ℝ), ∀ x y : ℝ, (y = a * x + 2 ∧ y = 3 * x - b) → a = 1 / 3 ∧ b = 6) :=
by
  sorry

end symmetric_linear_functions_l216_216132


namespace expected_value_of_smallest_seven_selected_from_sixty_three_l216_216453

noncomputable def expected_value_smallest_selected (n r : ℕ) : ℕ :=
  (n + 1) / (r + 1)

theorem expected_value_of_smallest_seven_selected_from_sixty_three :
  expected_value_smallest_selected 63 7 = 8 :=
by
  sorry -- Proof is omitted as per instructions

end expected_value_of_smallest_seven_selected_from_sixty_three_l216_216453


namespace prob_twins_street_l216_216945

variable (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

theorem prob_twins_street : p ≠ 1 → real := sorry

end prob_twins_street_l216_216945


namespace binom_7_2_eq_21_l216_216981

open Nat

theorem binom_7_2_eq_21 : binomial 7 2 = 21 := 
by sorry

end binom_7_2_eq_21_l216_216981


namespace scientific_notation_l216_216454

theorem scientific_notation (x : ℝ) (h : x = 70819) : x = 7.0819 * 10^4 :=
by 
  -- Proof goes here
  sorry

end scientific_notation_l216_216454


namespace abc_sum_l216_216373

def f (x : Int) (a b c : Nat) : Int :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b
  else b * x + c

theorem abc_sum :
  ∃ a b c : Nat, 
  f 3 a b c = 7 ∧ 
  f 0 a b c = 6 ∧ 
  f (-3) a b c = -15 ∧ 
  a + b + c = 10 :=
by
  sorry

end abc_sum_l216_216373


namespace largest_common_number_in_sequences_from_1_to_200_l216_216840

theorem largest_common_number_in_sequences_from_1_to_200 :
  ∃ a, a ≤ 200 ∧ a % 8 = 3 ∧ a % 9 = 5 ∧ ∀ b, (b ≤ 200 ∧ b % 8 = 3 ∧ b % 9 = 5) → b ≤ a :=
sorry

end largest_common_number_in_sequences_from_1_to_200_l216_216840


namespace fruit_basket_combinations_l216_216273

theorem fruit_basket_combinations (apples oranges : ℕ) (ha : apples = 6) (ho : oranges = 12) : 
  (∃ (baskets : ℕ), 
    (∀ a, 1 ≤ a ∧ a ≤ apples → ∃ b, 2 ≤ b ∧ b ≤ oranges ∧ baskets = a * b) ∧ baskets = 66) :=
by {
  sorry
}

end fruit_basket_combinations_l216_216273


namespace valid_two_digit_numbers_l216_216518

def is_valid_two_digit_number_pair (a b : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a > b ∧ (Nat.gcd (10 * a + b) (10 * b + a) = a^2 - b^2)

theorem valid_two_digit_numbers :
  (is_valid_two_digit_number_pair 2 1 ∨ is_valid_two_digit_number_pair 5 4) ∧
  ∀ a b, is_valid_two_digit_number_pair a b → (a = 2 ∧ b = 1 ∨ a = 5 ∧ b = 4) :=
by
  sorry

end valid_two_digit_numbers_l216_216518


namespace radishes_in_first_basket_l216_216092

theorem radishes_in_first_basket :
  ∃ x : ℕ, ∃ y : ℕ, x + y = 88 ∧ y = x + 14 ∧ x = 37 :=
by
  -- Proof goes here
  sorry

end radishes_in_first_basket_l216_216092


namespace sequence_distinct_l216_216346

theorem sequence_distinct (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) = f (n + 1) + f n) :
  ∀ i j : ℕ, i ≠ j → f i ≠ f j :=
by
  sorry

end sequence_distinct_l216_216346


namespace x_coordinate_of_q_l216_216293

theorem x_coordinate_of_q : 
  ∃ (Q : ℝ × ℝ), Q.1 = - (7 * Real.sqrt 2) / 10 ∧ 
  Q.2 < 0 ∧ 
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) ∧ 
  ∃ (P : ℝ × ℝ), P = (3 / 5, 4 / 5) ∧
  Real.angle (0, 0) P Q = 3 * Real.pi / 4 :=
by
  sorry

end x_coordinate_of_q_l216_216293


namespace find_number_exceeds_sixteen_percent_l216_216635

theorem find_number_exceeds_sixteen_percent (x : ℝ) (h : x - 0.16 * x = 63) : x = 75 :=
sorry

end find_number_exceeds_sixteen_percent_l216_216635


namespace gcd_2952_1386_l216_216806

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end gcd_2952_1386_l216_216806


namespace jane_output_increase_l216_216022

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l216_216022


namespace general_formula_for_sequence_a_l216_216726

noncomputable def S (n : ℕ) : ℕ := 3^n + 1

def a (n : ℕ) : ℕ :=
if n = 1 then 4 else 2 * 3^(n-1)

theorem general_formula_for_sequence_a (n : ℕ) :
  a n = if n = 1 then 4 else 2 * 3^(n-1) :=
by {
  sorry
}

end general_formula_for_sequence_a_l216_216726


namespace value_of_a_l216_216865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x > 0 ∧ x < 3 then real.log x - a * x else 0 -- f defined conditionally

theorem value_of_a (a : ℝ) : 
(∀ x : ℝ, f a (x + 3) = 3 * f a x) ∧ 
(∀ x : ℝ, (0 < x ∧ x < 3) → f a x = real.log x - a * x) ∧ 
(a > 1 / 3) ∧ 
(∀ y : ℝ, (-6 < y ∧ y < -3) → f a y ≤ -1 / 9 ∧ ((∃ c, (-6 < c ∧ c < -3) ∧ f a c = -1 / 9))) 
→ a = 1 :=
by sorry

end value_of_a_l216_216865


namespace combined_share_b_d_l216_216834

-- Definitions for the amounts shared between the children
def total_amount : ℝ := 15800
def share_a_plus_c : ℝ := 7022.222222222222

-- The goal is to prove that the combined share of B and D is 8777.777777777778
theorem combined_share_b_d :
  ∃ B D : ℝ, (B + D = total_amount - share_a_plus_c) :=
by
  sorry

end combined_share_b_d_l216_216834


namespace abs_diff_of_two_numbers_l216_216147

theorem abs_diff_of_two_numbers (x y : ℝ) (h_sum : x + y = 42) (h_prod : x * y = 437) : |x - y| = 4 :=
sorry

end abs_diff_of_two_numbers_l216_216147


namespace y_pow_one_div_x_neq_x_pow_y_l216_216998

theorem y_pow_one_div_x_neq_x_pow_y (t : ℝ) (ht : t > 1) : 
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  (y ^ (1 / x) ≠ x ^ y) :=
by
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  sorry

end y_pow_one_div_x_neq_x_pow_y_l216_216998


namespace orchestra_member_count_l216_216472

theorem orchestra_member_count :
  ∃ x : ℕ, 150 ≤ x ∧ x ≤ 250 ∧ 
           x % 4 = 2 ∧
           x % 5 = 3 ∧
           x % 8 = 4 ∧
           x % 9 = 5 :=
sorry

end orchestra_member_count_l216_216472


namespace hotel_charge_difference_l216_216174

variables (G P R : ℝ)

-- Assumptions based on the problem conditions
variables
  (hR : R = 2 * G) -- Charge for a single room at hotel R is 100% greater than at hotel G
  (hP : P = 0.9 * G) -- Charge for a single room at hotel P is 10% less than at hotel G

theorem hotel_charge_difference :
  ((R - P) / R) * 100 = 55 :=
by
  -- Calculation
  sorry

end hotel_charge_difference_l216_216174


namespace adult_ticket_cost_l216_216055

theorem adult_ticket_cost (A Tc : ℝ) (T C : ℕ) (M : ℝ) 
  (hTc : Tc = 3.50) 
  (hT : T = 21) 
  (hC : C = 16) 
  (hM : M = 83.50) 
  (h_eq : 16 * Tc + (↑(T - C)) * A = M) : 
  A = 5.50 :=
by sorry

end adult_ticket_cost_l216_216055


namespace intersection_of_sets_l216_216722

def setA : Set ℝ := {x | (x - 2) / x ≤ 0}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def setC : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets : setA ∩ setB = setC :=
by
  sorry

end intersection_of_sets_l216_216722


namespace inequality_holds_l216_216549

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
axiom symmetric_property : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom increasing_property : ∀ x y : ℝ, (1 ≤ x) → (x ≤ y) → f x ≤ f y

-- The statement of the theorem
theorem inequality_holds (m : ℝ) (h : m < 1 / 2) : f (1 - m) < f m :=
by sorry

end inequality_holds_l216_216549


namespace workers_not_worked_days_l216_216508

theorem workers_not_worked_days (W N : ℤ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 := 
by
  sorry

end workers_not_worked_days_l216_216508


namespace remainder_of_large_number_l216_216253

theorem remainder_of_large_number :
  (102938475610 % 12) = 10 :=
by
  have h1 : (102938475610 % 4) = 2 := sorry
  have h2 : (102938475610 % 3) = 1 := sorry
  sorry

end remainder_of_large_number_l216_216253


namespace octal_to_decimal_l216_216828

theorem octal_to_decimal : (1 * 8^3 + 7 * 8^2 + 4 * 8^1 + 3 * 8^0) = 995 :=
by
  sorry

end octal_to_decimal_l216_216828


namespace admission_price_for_adults_l216_216184

def total_people := 610
def num_adults := 350
def child_price := 1
def total_receipts := 960

theorem admission_price_for_adults (A : ℝ) (h1 : 350 * A + 260 = 960) : A = 2 :=
by {
  -- proof omitted
  sorry
}

end admission_price_for_adults_l216_216184


namespace exists_four_integers_multiple_1984_l216_216409

theorem exists_four_integers_multiple_1984 (a : Fin 97 → ℕ) (h_distinct : Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ k ≠ l ∧ 1984 ∣ (a i - a j) * (a k - a l) :=
sorry

end exists_four_integers_multiple_1984_l216_216409


namespace perpendicular_lines_condition_l216_216370

theorem perpendicular_lines_condition (m : ℝ) : (m = -1) ↔ ∀ (x y : ℝ), (x + y = 0) ∧ (x + m * y = 0) → 
  ((m ≠ 0) ∧ (-1) * (-1 / m) = 1) :=
by 
  sorry

end perpendicular_lines_condition_l216_216370


namespace largest_possible_perimeter_l216_216803

noncomputable def max_perimeter (a b c: ℕ) : ℕ := 2 * (a + b + c - 6)

theorem largest_possible_perimeter :
  ∃ (a b c : ℕ), (a = c) ∧ ((a - 2) * (b - 2) = 8) ∧ (max_perimeter a b c = 42) := by
  sorry

end largest_possible_perimeter_l216_216803


namespace g_at_9_l216_216332

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l216_216332


namespace sum_of_factors_636405_l216_216178

theorem sum_of_factors_636405 :
  ∃ (a b c : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 ∧
    a * b * c = 636405 ∧ a + b + c = 259 :=
sorry

end sum_of_factors_636405_l216_216178


namespace angle_YDA_eq_2_angle_YCA_l216_216584

theorem angle_YDA_eq_2_angle_YCA
  (A B C D P X Y : Point)
  (h_trapezoid : IsoscelesTrapezoid A B C D)
  (h_inter_AC_BD : P = line_intersection (line A C) (line B D))
  (h_circumcircle_intersect_BC : X ∈ circumcircle (triangle A B P) ∧ X ≠ B)
  (h_Y_on_AX : Y ∈ line A X)
  (h_parallel : parallel (line D Y) (line B C)) :
  angle Y D A = 2 * angle Y C A := 
sorry

end angle_YDA_eq_2_angle_YCA_l216_216584


namespace g_9_l216_216336

variable (g : ℝ → ℝ)

-- Conditions
axiom func_eq : ∀ x y : ℝ, g(x + y) = g(x) * g(y)
axiom g_3 : g 3 = 4

-- Theorem to prove
theorem g_9 : g 9 = 64 :=
by
  sorry

end g_9_l216_216336


namespace perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l216_216999

-- Mathematical definitions and theorems required for the problem
theorem perpendicular_lines_condition (m : ℝ) :
  3 * m + m * (2 * m - 1) = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

-- Translate the specific problem into Lean
theorem perpendicular_lines_sufficient_not_necessary (m : ℝ) (h : 3 * m + m * (2 * m - 1) = 0) :
  m = -1 ∨ (m ≠ -1 ∧ 3 * m + m * (2 * m - 1) = 0) :=
by sorry

end perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l216_216999


namespace probability_three_correct_deliveries_is_one_sixth_l216_216700

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l216_216700


namespace find_a_l216_216723

variable {x n : ℝ}

theorem find_a (hx : x > 0) (hn : n > 0) :
    (∀ n > 0, x + n^n / x^n ≥ n + 1) ↔ (∀ n > 0, a = n^n) :=
sorry

end find_a_l216_216723


namespace whole_numbers_between_l216_216139

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l216_216139


namespace subtract_3a_result_l216_216462

theorem subtract_3a_result (a : ℝ) : 
  (9 * a^2 - 3 * a + 8) + 3 * a = 9 * a^2 + 8 := 
sorry

end subtract_3a_result_l216_216462


namespace correct_calculation_l216_216064

theorem correct_calculation (a : ℝ) : a^3 / a^2 = a := by
  sorry

end correct_calculation_l216_216064


namespace geometric_series_sum_l216_216625

theorem geometric_series_sum :
  let b1 := (3 : ℚ) / 4 in
  let r := (3 : ℚ) / 4 in
  let n := 15 in
  let result := (∑ i in finset.range n, b1 * r^i) in
  result = 3177878751 / 1073741824 :=
by
  let b1 := (3 : ℚ) / 4
  let r := (3 : ℚ) / 4
  let n := 15
  let result := (∑ i in finset.range n, b1 * r^i)
  exact (∑ i in finset.range 15, (3 : ℚ) / 4 * ((3 : ℚ) / 4)^i) = 3177878751 / 1073741824
  sorry

end geometric_series_sum_l216_216625


namespace insulation_cost_l216_216812

def rectangular_prism_surface_area (l w h : ℕ) : ℕ :=
2 * l * w + 2 * l * h + 2 * w * h

theorem insulation_cost
  (l w h : ℕ) (cost_per_square_foot : ℕ)
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost : cost_per_square_foot = 20) :
  rectangular_prism_surface_area l w h * cost_per_square_foot = 1440 := 
sorry

end insulation_cost_l216_216812


namespace three_correct_deliveries_probability_l216_216699

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l216_216699


namespace landscape_length_l216_216046

theorem landscape_length (b length : ℕ) (A_playground : ℕ) (h1 : length = 4 * b) (h2 : A_playground = 1200) (h3 : A_playground = (1 / 3 : ℚ) * (length * b)) :
  length = 120 :=
by
  sorry

end landscape_length_l216_216046


namespace rhombus_has_perpendicular_diagonals_and_rectangle_not_l216_216230

-- Definitions based on conditions (a))
def rhombus (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_perpendicular : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_perpendicular

def rectangle (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_equal : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_equal

-- Theorem to prove (c))
theorem rhombus_has_perpendicular_diagonals_and_rectangle_not 
  (rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular : Prop)
  (rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal : Prop) :
  rhombus rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular → 
  rectangle rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal → 
  rhombus_diagonals_perpendicular ∧ ¬(rectangle (rectangle_sides_equal) (rectangle_diagonals_bisect) (rhombus_diagonals_perpendicular)) :=
sorry

end rhombus_has_perpendicular_diagonals_and_rectangle_not_l216_216230


namespace zoe_spent_amount_l216_216817

theorem zoe_spent_amount :
  (3 * (8 + 2) = 30) :=
by sorry

end zoe_spent_amount_l216_216817


namespace work_days_together_l216_216944

theorem work_days_together (d : ℕ) (h : d * (17 / 140) = 6 / 7) : d = 17 := by
  sorry

end work_days_together_l216_216944


namespace smallest_n_property_l216_216854

theorem smallest_n_property (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
 (hxy : x ∣ y^3) (hyz : y ∣ z^3) (hzx : z ∣ x^3) : 
  x * y * z ∣ (x + y + z) ^ 13 := 
by sorry

end smallest_n_property_l216_216854


namespace assistant_increases_output_by_100_percent_l216_216016

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l216_216016


namespace solution_set_of_f_inequality_l216_216043

variable {f : ℝ → ℝ}
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, f' x < 1/2)

theorem solution_set_of_f_inequality :
  {x : ℝ | f (x^2) < x^2 / 2 + 1 / 2} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end solution_set_of_f_inequality_l216_216043


namespace tens_place_of_8_pow_1234_l216_216484

theorem tens_place_of_8_pow_1234 : (8^1234 / 10) % 10 = 0 := by
  sorry

end tens_place_of_8_pow_1234_l216_216484


namespace no_integer_y_makes_Q_perfect_square_l216_216687

def Q (y : ℤ) : ℤ := y^4 + 8 * y^3 + 18 * y^2 + 10 * y + 41

theorem no_integer_y_makes_Q_perfect_square :
  ¬ ∃ y : ℤ, ∃ b : ℤ, Q y = b^2 :=
by
  intro h
  rcases h with ⟨y, b, hQ⟩
  sorry

end no_integer_y_makes_Q_perfect_square_l216_216687


namespace binom_7_2_eq_21_l216_216980

open Nat

theorem binom_7_2_eq_21 : binomial 7 2 = 21 := 
by sorry

end binom_7_2_eq_21_l216_216980


namespace value_of_smaller_denom_l216_216086

-- We are setting up the conditions given in the problem.
variables (x : ℕ) -- The value of the smaller denomination bill.

-- Condition 1: She has 4 bills of denomination x.
def value_smaller_denomination : ℕ := 4 * x

-- Condition 2: She has 8 bills of $10 denomination.
def value_ten_bills : ℕ := 8 * 10

-- Condition 3: The total value of the bills is $100.
def total_value : ℕ := 100

-- Prove that x = 5 using the given conditions.
theorem value_of_smaller_denom : value_smaller_denomination x + value_ten_bills = total_value → x = 5 :=
by
  intro h
  -- Proof steps would go here
  sorry

end value_of_smaller_denom_l216_216086


namespace abc_sum_l216_216279

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l216_216279


namespace abc_sum_l216_216277

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l216_216277


namespace sum_of_pairwise_products_does_not_end_in_2019_l216_216514

theorem sum_of_pairwise_products_does_not_end_in_2019 (n : ℤ) : ¬ (∃ (k : ℤ), 10000 ∣ (3 * n ^ 2 - 2020 + k * 10000)) := by
  sorry

end sum_of_pairwise_products_does_not_end_in_2019_l216_216514


namespace angle_D_measure_l216_216550

theorem angle_D_measure (A B C D : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 35) :
  D = 120 :=
  sorry

end angle_D_measure_l216_216550


namespace border_area_correct_l216_216080

-- Definition of the dimensions of the photograph
def photo_height := 8
def photo_width := 10
def frame_border := 3

-- Definition of the areas of the photograph and the framed area
def photo_area := photo_height * photo_width
def frame_height := photo_height + 2 * frame_border
def frame_width := photo_width + 2 * frame_border
def frame_area := frame_height * frame_width

-- Theorem stating that the area of the border is 144 square inches
theorem border_area_correct : (frame_area - photo_area) = 144 := 
by
  sorry

end border_area_correct_l216_216080


namespace yasmin_children_l216_216755

-- Definitions based on the conditions
def has_twice_children (children_john children_yasmin : ℕ) := children_john = 2 * children_yasmin
def total_grandkids (children_john children_yasmin : ℕ) (total : ℕ) := total = children_john + children_yasmin

-- Problem statement in Lean 4
theorem yasmin_children (children_john children_yasmin total_grandkids : ℕ) 
  (h1 : has_twice_children children_john children_yasmin) 
  (h2 : total_grandkids children_john children_yasmin 6) : 
  children_yasmin = 2 :=
by
  sorry

end yasmin_children_l216_216755


namespace not_perfect_cube_l216_216309

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℕ, k ^ 3 = 2 ^ (2 ^ n) + 1 :=
sorry

end not_perfect_cube_l216_216309


namespace inequality_condition_l216_216789

theorem inequality_condition {x : ℝ} (h : -1/2 ≤ x ∧ x < 1) : (2 * x + 1) / (1 - x) ≥ 0 :=
sorry

end inequality_condition_l216_216789


namespace solve_for_x_l216_216774

theorem solve_for_x (x : ℝ) :
  (x - 5)^4 = (1/16)⁻¹ → x = 7 :=
by
  sorry

end solve_for_x_l216_216774


namespace injective_func_identity_l216_216526

open Function

theorem injective_func_identity :
  (∀ f : ℕ → ℕ,
    injective f ∧ (∀ n : ℕ, f(f(n)) ≤ (n + f(n)) / 2) →
    (∀ n, f(n) = n)) :=
by
  intro f
  intro h
  cases h with hf hf'
  sorry

end injective_func_identity_l216_216526


namespace water_level_function_l216_216824

def water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : ℝ :=
  0.3 * x + 6

theorem water_level_function :
  ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5), water_level x h = 6 + 0.3 * x :=
by
  intros
  unfold water_level
  sorry -- Proof skipped

end water_level_function_l216_216824


namespace sqrt3_minus1_plus_inv3_pow_minus2_l216_216848

theorem sqrt3_minus1_plus_inv3_pow_minus2 :
  (Real.sqrt 3 - 1) + (1 / (1/3) ^ 2) = Real.sqrt 3 + 8 :=
by
  sorry

end sqrt3_minus1_plus_inv3_pow_minus2_l216_216848


namespace combined_tax_rate_l216_216202

theorem combined_tax_rate
  (Mork_income : ℝ)
  (Mindy_income : ℝ)
  (h1 : Mindy_income = 3 * Mork_income)
  (Mork_tax_rate : ℝ := 0.30)
  (Mindy_tax_rate : ℝ := 0.20) :
  (Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income) * 100 = 22.5 :=
by
  sorry

end combined_tax_rate_l216_216202


namespace problem_solution_l216_216288

namespace MathProof

-- Definitions of our events and probabilities

structure BallPocket := 
  (ball_count : ℕ)
  (red_balls : Finset ℕ)
  (blue_balls : Finset ℕ)

-- Define the specific setup of the problem
def pocket : BallPocket := {
  ball_count := 8,
  red_balls := {1, 2},
  blue_balls := {1, 2, 3, 4, 5, 6}
}

def event_A (p : BallPocket) : Set ℕ := p.red_balls
def event_B (p : BallPocket) : Set ℕ := {ball ∈ Finset.range (p.ball_count + 1) | ball % 2 = 0}
def event_C (p : BallPocket) : Set ℕ := {ball ∈ Finset.range (p.ball_count + 1) | ball % 3 = 0}

-- To be used for independence and mutual exclusivity
def probability (p : BallPocket) (event : Set ℕ) : ℚ := (Finset.card (event ∩ (p.red_balls ∪ p.blue_balls)).val : ℚ) / (p.ball_count : ℚ)

theorem problem_solution :
  (event_A pocket ∩ event_C pocket = ∅) ∧
  (probability pocket (event_A pocket ∩ event_B pocket) = probability pocket (event_A pocket) * probability pocket (event_B pocket)) ∧
  (probability pocket (event_B pocket ∩ event_C pocket) = probability pocket (event_B pocket) * probability pocket (event_C pocket)) := 
  sorry

end MathProof

end problem_solution_l216_216288


namespace jane_output_increase_l216_216019

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l216_216019


namespace find_g_9_l216_216335

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l216_216335


namespace solve_for_x_l216_216775

theorem solve_for_x (x : ℝ) :
  (x - 5)^4 = (1/16)⁻¹ → x = 7 :=
by
  sorry

end solve_for_x_l216_216775


namespace ant_climbing_floors_l216_216832

theorem ant_climbing_floors (time_per_floor : ℕ) (total_time : ℕ) (floors_climbed : ℕ) :
  time_per_floor = 15 →
  total_time = 105 →
  floors_climbed = total_time / time_per_floor + 1 →
  floors_climbed = 8 :=
by
  intros
  sorry

end ant_climbing_floors_l216_216832


namespace find_missing_edge_l216_216328

-- Define the known parameters
def volume : ℕ := 80
def edge1 : ℕ := 2
def edge3 : ℕ := 8

-- Define the missing edge
def missing_edge : ℕ := 5

-- State the problem
theorem find_missing_edge (volume : ℕ) (edge1 : ℕ) (edge3 : ℕ) (missing_edge : ℕ) :
  volume = edge1 * missing_edge * edge3 →
  missing_edge = 5 :=
by
  sorry

end find_missing_edge_l216_216328


namespace matt_total_vibrations_l216_216764

noncomputable def vibrations_lowest : ℕ := 1600
noncomputable def vibrations_highest : ℕ := vibrations_lowest + (6 * vibrations_lowest / 10)
noncomputable def time_seconds : ℕ := 300
noncomputable def total_vibrations : ℕ := vibrations_highest * time_seconds

theorem matt_total_vibrations :
  total_vibrations = 768000 := by
  sorry

end matt_total_vibrations_l216_216764


namespace necessary_but_not_sufficient_condition_l216_216869

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x ^ 2

theorem necessary_but_not_sufficient_condition :
  (∀ x, q x → p x) ∧ (¬ ∀ x, p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_condition_l216_216869


namespace bakery_gives_away_30_doughnuts_at_end_of_day_l216_216819

def boxes_per_day (total_doughnuts doughnuts_per_box : ℕ) : ℕ :=
  total_doughnuts / doughnuts_per_box

def leftover_boxes (total_boxes sold_boxes : ℕ) : ℕ :=
  total_boxes - sold_boxes

def doughnuts_given_away (leftover_boxes doughnuts_per_box : ℕ) : ℕ :=
  leftover_boxes * doughnuts_per_box

theorem bakery_gives_away_30_doughnuts_at_end_of_day 
  (total_doughnuts doughnuts_per_box sold_boxes : ℕ) 
  (H1 : total_doughnuts = 300) (H2 : doughnuts_per_box = 10) (H3 : sold_boxes = 27) : 
  doughnuts_given_away (leftover_boxes (boxes_per_day total_doughnuts doughnuts_per_box) sold_boxes) doughnuts_per_box = 30 :=
by
  sorry

end bakery_gives_away_30_doughnuts_at_end_of_day_l216_216819


namespace probability_ray_OA_within_angle_xOT_l216_216752

open MeasureTheory Set

-- Define the context: Cartesian coordinate system and angles
def angle (deg : ℝ) := deg / 360

-- Condition: angle xOT = 60 degrees
def angle_xOT : ℝ := 60

-- Question: What is the probability of the ray OA falling within angle xOT?
theorem probability_ray_OA_within_angle_xOT :
  (angle angle_xOT) = 1 / 6 :=
by
  sorry

end probability_ray_OA_within_angle_xOT_l216_216752


namespace num_whole_numbers_between_l216_216142

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l216_216142


namespace three_distinct_real_roots_l216_216899

theorem three_distinct_real_roots 
  (c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1*x1 + 6*x1 + c)*(x1*x1 + 6*x1 + c) = 0 ∧ 
    (x2*x2 + 6*x2 + c)*(x2*x2 + 6*x2 + c) = 0 ∧ 
    (x3*x3 + 6*x3 + c)*(x3*x3 + 6*x3 + c) = 0) 
  ↔ c = (11 - Real.sqrt 13) / 2 :=
by sorry

end three_distinct_real_roots_l216_216899


namespace necessary_but_not_sufficient_condition_l216_216231

theorem necessary_but_not_sufficient_condition (x : ℝ) : |x - 1| < 2 → -3 < x ∧ x < 3 :=
by
  sorry

end necessary_but_not_sufficient_condition_l216_216231


namespace rectangle_side_length_l216_216312

theorem rectangle_side_length (a c : ℝ) (h_ratio : a / c = 3 / 4) (hc : c = 4) : a = 3 :=
by
  sorry

end rectangle_side_length_l216_216312


namespace stating_area_trapezoid_AMBQ_is_18_l216_216392

/-- Definition of the 20-sided polygon configuration with 2 unit sides and right-angle turns. -/
structure Polygon20 where
  sides : ℕ → ℝ
  units : ∀ i, sides i = 2
  right_angles : ∀ i, (i + 1) % 20 ≠ i -- Right angles between consecutive sides

/-- Intersection point of AJ and DP, named M, under the given polygon configuration. -/
def intersection_point (p : Polygon20) : ℝ × ℝ :=
  (5 * p.sides 0, 5 * p.sides 1)  -- Assuming relevant distances for simplicity

/-- Area of the trapezoid AMBQ formed given the defined Polygon20. -/
noncomputable def area_trapezoid_AMBQ (p : Polygon20) : ℝ :=
  let base1 := 10 * p.sides 0
  let base2 := 8 * p.sides 0
  let height := p.sides 0
  (base1 + base2) * height / 2

/-- 
  Theorem stating the area of the trapezoid AMBQ in the given configuration.
  We prove that the area is 18 units.
-/
theorem area_trapezoid_AMBQ_is_18 (p : Polygon20) :
  area_trapezoid_AMBQ p = 18 :=
sorry -- Proof to be done

end stating_area_trapezoid_AMBQ_is_18_l216_216392


namespace no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l216_216070

noncomputable def system_discriminant (a b c : ℝ) : ℝ := (b - 1)^2 - 4 * a * c

theorem no_real_solutions_if_discriminant_neg (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c < 0) :
  ¬∃ (x₁ x₂ x₃ : ℝ), (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

theorem one_real_solution_if_discriminant_zero (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c = 0) :
  ∃ (x : ℝ), ∀ (x₁ x₂ x₃ : ℝ), (x₁ = x) ∧ (x₂ = x) ∧ (x₃ = x) ∧
                              (a * x₁^2 + b * x₁ + c = x₂) ∧
                              (a * x₂^2 + b * x₂ + c = x₃) ∧
                              (a * x₃^2 + b * x₃ + c = x₁)  :=
sorry

theorem more_than_one_real_solution_if_discriminant_pos (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c > 0) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

end no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l216_216070


namespace ratio_Jake_sister_l216_216569

theorem ratio_Jake_sister (Jake_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (expected_ratio : ℕ) :
  Jake_weight = 113 →
  total_weight = 153 →
  weight_loss = 33 →
  expected_ratio = 2 →
  (Jake_weight - weight_loss) / (total_weight - Jake_weight) = expected_ratio :=
by
  intros hJake hTotal hLoss hRatio
  sorry

end ratio_Jake_sister_l216_216569


namespace total_salaries_proof_l216_216796

def total_salaries (A_salary B_salary : ℝ) :=
  A_salary + B_salary

theorem total_salaries_proof : ∀ A_salary B_salary : ℝ,
  A_salary = 3000 →
  (0.05 * A_salary = 0.15 * B_salary) →
  total_salaries A_salary B_salary = 4000 :=
by
  intros A_salary B_salary h1 h2
  rw [h1] at h2
  sorry

end total_salaries_proof_l216_216796


namespace tens_digit_of_8_pow_1234_l216_216486

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end tens_digit_of_8_pow_1234_l216_216486


namespace base7_first_digit_l216_216936

noncomputable def first_base7_digit : ℕ := 625

theorem base7_first_digit (n : ℕ) (h : n = 625) : ∃ (d : ℕ), d = 12 ∧ (d * 49 ≤ n) ∧ (n < (d + 1) * 49) :=
by
  sorry

end base7_first_digit_l216_216936


namespace original_number_one_more_reciprocal_is_11_over_5_l216_216907

theorem original_number_one_more_reciprocal_is_11_over_5 (x : ℚ) (h : 1 + 1/x = 11/5) : x = 5/6 :=
by
  sorry

end original_number_one_more_reciprocal_is_11_over_5_l216_216907


namespace total_income_l216_216401

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l216_216401


namespace line_perpendicular_to_plane_l216_216864

-- Define a structure for vectors in 3D
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define line l with the given direction vector
def direction_vector_l : Vector3D := ⟨1, -1, -2⟩

-- Define plane α with the given normal vector
def normal_vector_alpha : Vector3D := ⟨2, -2, -4⟩

-- Prove that line l is perpendicular to plane α
theorem line_perpendicular_to_plane :
  let a := direction_vector_l
  let b := normal_vector_alpha
  (b.x = 2 * a.x) ∧ (b.y = 2 * a.y) ∧ (b.z = 2 * a.z) → 
  (a.x * b.x + a.y * b.y + a.z * b.z = 0) :=
by
  intro a b h
  sorry

end line_perpendicular_to_plane_l216_216864


namespace circle_reflection_l216_216465

/-- The reflection of a point over the line y = -x results in swapping the x and y coordinates 
and changing their signs. Given a circle with center (3, -7), the reflected center should be (7, -3). -/
theorem circle_reflection (x y : ℝ) (h : (x, y) = (3, -7)) : (y, -x) = (7, -3) :=
by
  -- since the problem is stated to skip the proof, we use sorry
  sorry

end circle_reflection_l216_216465


namespace t_of_polynomial_has_factor_l216_216122

theorem t_of_polynomial_has_factor (t : ℤ) :
  (∃ a b : ℤ, x ^ 3 - x ^ 2 - 7 * x + t = (x + 1) * (x ^ 2 + a * x + b)) → t = -5 :=
by
  sorry

end t_of_polynomial_has_factor_l216_216122


namespace pairs_of_participants_l216_216903

theorem pairs_of_participants (n : Nat) (h : n = 12) : (Nat.choose n 2) = 66 := by
  sorry

end pairs_of_participants_l216_216903


namespace smallest_arith_prog_term_l216_216694

-- Define the conditions of the problem as a structure
structure ArithProgCondition (a d : ℝ) :=
  (sum_of_squares : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (sum_of_cubes : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)

-- Define the theorem we want to prove
theorem smallest_arith_prog_term :
  ∃ (a d : ℝ), ArithProgCondition a d ∧ (a = 0 ∧ (d = sqrt 7 ∨ d = -sqrt 7) → -2 * sqrt 7) := 
sorry

end smallest_arith_prog_term_l216_216694


namespace probability_of_three_correct_packages_l216_216706

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l216_216706


namespace calc_square_difference_and_square_l216_216380

theorem calc_square_difference_and_square (a b : ℤ) (h1 : a = 7) (h2 : b = 3)
  (h3 : a^2 = 49) (h4 : b^2 = 9) : (a^2 - b^2)^2 = 1600 := by
  sorry

end calc_square_difference_and_square_l216_216380


namespace neg_neg_eq_pos_l216_216646

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l216_216646


namespace complex_quadrant_l216_216151

def z1 := Complex.mk 1 (-2)
def z2 := Complex.mk 2 1
def z := z1 * z2

theorem complex_quadrant : z = Complex.mk 4 (-3) ∧ z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l216_216151


namespace num_ordered_pairs_divisible_by_11_l216_216759

/- 
 Problem statement: 
 Prove that the number of ordered pairs (i, j) of divisors of 6^8 such that d_i - d_j is divisible by 11 is 665.
-/ 

theorem num_ordered_pairs_divisible_by_11 :
  let m := 6^8 in
  let divisors := {d : ℕ | d ∣ m} in
  let condition := λ (i j : ℕ), ((i ∈ divisors) ∧ (j ∈ divisors) ∧ ((i - j) % 11 = 0)) in
  finset.card ((finset.filter (λ p : ℕ × ℕ, condition p.1 p.2) 
                  (finset.product (finset.filter (λ d, d ∣ m) finset.univ) 
                                  (finset.filter (λ d, d ∣ m) finset.univ))) = 665

sorry

end num_ordered_pairs_divisible_by_11_l216_216759


namespace domain_log_function_min_value_h_l216_216371

open Real

-- Problem (1)
theorem domain_log_function (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2x + m > 0) ↔ m > 1 :=
sorry

-- Problem (2)
def h (a : ℝ) : ℝ :=
  if a < 1 / 3 then (28 - 6 * a) / 9
  else if 1 / 3 ≤ a ∧ a ≤ 3 then 3 - a^2
  else 12 - 6 * a

theorem min_value_h (a : ℝ) : ∀ x : ℝ, x ∈ Icc (-1 : ℝ) 1 → 
  let y = (1 / 3) ^ x in 
  (y^2 - 2 * a * y + 3) ≥ (h a) :=
sorry

end domain_log_function_min_value_h_l216_216371


namespace fill_time_two_pipes_l216_216369

variable (R : ℝ)
variable (c : ℝ)
variable (t1 : ℝ) (t2 : ℝ)

noncomputable def fill_time_with_pipes (num_pipes : ℝ) (time_per_tank : ℝ) : ℝ :=
  time_per_tank / num_pipes

theorem fill_time_two_pipes (h1 : fill_time_with_pipes 3 t1 = 12) 
                            (h2 : c = R)
                            : fill_time_with_pipes 2 (3 * R * t1) = 18 := 
by
  sorry

end fill_time_two_pipes_l216_216369


namespace speed_of_man_correct_l216_216677

noncomputable def speed_of_man_in_kmph (train_speed_kmph : ℝ) (train_length_m : ℝ) (time_pass_sec : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := (train_length_m / time_pass_sec)
  let man_speed_mps := relative_speed_mps - train_speed_mps
  man_speed_mps * 3600 / 1000

theorem speed_of_man_correct : 
  speed_of_man_in_kmph 77.993280537557 140 6 = 6.00871946444388 := 
by simp [speed_of_man_in_kmph]; sorry

end speed_of_man_correct_l216_216677


namespace girl_weaves_on_tenth_day_l216_216751

theorem girl_weaves_on_tenth_day 
  (a1 d : ℝ)
  (h1 : 7 * a1 + 21 * d = 28)
  (h2 : a1 + d + a1 + 4 * d + a1 + 7 * d = 15) :
  a1 + 9 * d = 10 :=
by sorry

end girl_weaves_on_tenth_day_l216_216751


namespace adam_chocolate_boxes_l216_216083

theorem adam_chocolate_boxes 
  (c : ℕ) -- number of chocolate boxes Adam bought
  (h1 : 4 * c + 4 * 5 = 28) : 
  c = 2 := 
by
  sorry

end adam_chocolate_boxes_l216_216083


namespace sides_of_triangle_l216_216504

-- Definitions from conditions
variables (a b c : ℕ) (r bk kc : ℕ)
def is_tangent_split : Prop := bk = 8 ∧ kc = 6
def inradius : Prop := r = 4

-- Main theorem statement
theorem sides_of_triangle (h1 : is_tangent_split bk kc) (h2 : inradius r) : a + 6 = 13 ∧ a + 8 = 15 ∧ b = 14 := by
  sorry

end sides_of_triangle_l216_216504


namespace max_real_solutions_l216_216241

noncomputable def max_number_of_real_solutions (n : ℕ) (y : ℝ) : ℕ :=
if (n + 1) % 2 = 1 then 1 else 0

theorem max_real_solutions (n : ℕ) (hn : 0 < n) (y : ℝ) :
  max_number_of_real_solutions n y = 1 :=
by
  sorry

end max_real_solutions_l216_216241


namespace ratio_a_to_c_l216_216066

theorem ratio_a_to_c {a b c : ℚ} (h1 : a / b = 4 / 3) (h2 : b / c = 1 / 5) :
  a / c = 4 / 5 := 
sorry

end ratio_a_to_c_l216_216066


namespace smallest_among_5_neg7_0_neg53_l216_216970

-- Define the rational numbers involved as constants
def a : ℚ := 5
def b : ℚ := -7
def c : ℚ := 0
def d : ℚ := -5 / 3

-- Define the conditions as separate lemmas
lemma positive_greater_than_zero (x : ℚ) (hx : x > 0) : x > c := by sorry
lemma zero_greater_than_negative (x : ℚ) (hx : x < 0) : c > x := by sorry
lemma compare_negative_by_absolute_value (x y : ℚ) (hx : x < 0) (hy : y < 0) (habs : |x| > |y|) : x < y := by sorry

-- Prove the main assertion
theorem smallest_among_5_neg7_0_neg53 : 
    b < a ∧ b < c ∧ b < d := by
    -- Here we apply the defined conditions to show b is the smallest
    sorry

end smallest_among_5_neg7_0_neg53_l216_216970


namespace Nara_is_1_69_meters_l216_216772

-- Define the heights of Sangheon, Chiho, and Nara
def Sangheon_height : ℝ := 1.56
def Chiho_height : ℝ := Sangheon_height - 0.14
def Nara_height : ℝ := Chiho_height + 0.27

-- The statement to be proven
theorem Nara_is_1_69_meters : Nara_height = 1.69 :=
by
  -- the proof goes here
  sorry

end Nara_is_1_69_meters_l216_216772


namespace water_consumption_l216_216450

theorem water_consumption (num_cows num_goats num_pigs num_sheep : ℕ)
  (water_per_cow water_per_goat water_per_pig water_per_sheep daily_total weekly_total : ℕ)
  (h1 : num_cows = 40)
  (h2 : num_goats = 25)
  (h3 : num_pigs = 30)
  (h4 : water_per_cow = 80)
  (h5 : water_per_goat = water_per_cow / 2)
  (h6 : water_per_pig = water_per_cow / 3)
  (h7 : num_sheep = 10 * num_cows)
  (h8 : water_per_sheep = water_per_cow / 4)
  (h9 : daily_total = num_cows * water_per_cow + num_goats * water_per_goat + num_pigs * water_per_pig + num_sheep * water_per_sheep)
  (h10 : weekly_total = daily_total * 7) :
  weekly_total = 91000 := by
  sorry

end water_consumption_l216_216450


namespace abc_sum_eq_11sqrt6_l216_216276

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l216_216276


namespace unique_12_tuple_l216_216859

theorem unique_12_tuple : 
  ∃! (x : Fin 12 → ℝ), 
    ((1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
    (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 +
    (x 7 - x 8)^2 + (x 8 - x 9)^2 + (x 9 - x 10)^2 + (x 10 - x 11)^2 + 
    (x 11)^2 = 1 / 13) ∧ (x 0 + x 11 = 1 / 2) :=
by
  sorry

end unique_12_tuple_l216_216859


namespace quadratic_inequality_condition_l216_216120

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) :=
sorry

end quadratic_inequality_condition_l216_216120


namespace part1_part2_part3_l216_216559

noncomputable def f (x : ℝ) : ℝ := 3 * x - Real.exp x + 1

theorem part1 :
  ∃ x0 > 0, f x0 = 0 :=
sorry

theorem part2 (x0 : ℝ) (h1 : f x0 = 0) :
  ∀ x, f x ≤ (3 - Real.exp x0) * (x - x0) :=
sorry

theorem part3 (m x1 x2 : ℝ) (h1 : m > 0) (h2 : x1 < x2) (h3 : f x1 = m) (h4 : f x2 = m):
  x2 - x1 < 2 - 3 * m / 4 :=
sorry

end part1_part2_part3_l216_216559


namespace quadratic_m_leq_9_l216_216884

-- Define the quadratic equation
def quadratic_eq_has_real_roots (a b c : ℝ) : Prop := 
  b^2 - 4*a*c ≥ 0

-- Define the specific property we need to prove
theorem quadratic_m_leq_9 (m : ℝ) : (quadratic_eq_has_real_roots 1 (-6) m) → (m ≤ 9) := 
by
  sorry

end quadratic_m_leq_9_l216_216884


namespace amy_tickets_initial_l216_216235

theorem amy_tickets_initial (x : ℕ) (h1 : x + 21 = 54) : x = 33 :=
by sorry

end amy_tickets_initial_l216_216235


namespace rectangle_perimeter_l216_216365

theorem rectangle_perimeter (l d : ℝ) (h_l : l = 8) (h_d : d = 17) :
  ∃ w : ℝ, (d^2 = l^2 + w^2) ∧ (2*l + 2*w = 46) :=
by
  sorry

end rectangle_perimeter_l216_216365


namespace second_person_avg_pages_per_day_l216_216105

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end second_person_avg_pages_per_day_l216_216105


namespace exact_value_range_l216_216473

theorem exact_value_range (a : ℝ) (h : |170 - a| < 0.5) : 169.5 ≤ a ∧ a < 170.5 :=
by
  sorry

end exact_value_range_l216_216473


namespace angela_more_marbles_l216_216381

/--
Albert has three times as many marbles as Angela.
Allison has 28 marbles.
Albert and Allison have 136 marbles together.
Prove that Angela has 8 more marbles than Allison.
-/
theorem angela_more_marbles 
  (albert_angela : ℕ) 
  (angela: ℕ) 
  (albert: ℕ) 
  (allison: ℕ) 
  (h_albert_is_three_times_angela : albert = 3 * angela) 
  (h_allison_is_28 : allison = 28) 
  (h_albert_allison_is_136 : albert + allison = 136) 
  : angela - allison = 8 := 
by
  sorry

end angela_more_marbles_l216_216381


namespace sarah_meals_count_l216_216315

theorem sarah_meals_count :
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  main_courses * sides * drinks * desserts = 48 := 
by
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  calc
    4 * 3 * 2 * 2 = 48 := sorry

end sarah_meals_count_l216_216315


namespace emmalyn_earnings_l216_216399

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l216_216399


namespace quadratic_increasing_l216_216753

theorem quadratic_increasing (x : ℝ) (hx : x > 1) : ∃ y : ℝ, y = (x-1)^2 + 1 ∧ ∀ (x₁ x₂ : ℝ), x₁ > x ∧ x₂ > x₁ → (x₁ - 1)^2 + 1 < (x₂ - 1)^2 + 1 := by
  sorry

end quadratic_increasing_l216_216753


namespace farmer_goats_l216_216216

theorem farmer_goats (cows sheep goats : ℕ) (extra_goats : ℕ) 
(hcows : cows = 7) (hsheep : sheep = 8) (hgoats : goats = 6) 
(h : (goats + extra_goats = (cows + sheep + goats + extra_goats) / 2)) : 
extra_goats = 9 := by
  sorry

end farmer_goats_l216_216216


namespace all_of_the_above_used_as_money_l216_216488

-- Definition to state that each item was used as money
def gold_used_as_money : Prop := true
def stones_used_as_money : Prop := true
def horses_used_as_money : Prop := true
def dried_fish_used_as_money : Prop := true
def mollusk_shells_used_as_money : Prop := true

-- Statement that all of the above items were used as money
theorem all_of_the_above_used_as_money : gold_used_as_money ∧ stones_used_as_money ∧ horses_used_as_money ∧ dried_fish_used_as_money ∧ mollusk_shells_used_as_money :=
by {
  split; -- Split conjunctions
  all_goals { exact true.intro }; -- Each assumption is true
}

end all_of_the_above_used_as_money_l216_216488


namespace billy_sleep_total_hours_l216_216882

theorem billy_sleep_total_hours : 
    let first_night := 6
    let second_night := 2 * first_night
    let third_night := second_night - 3
    let fourth_night := 3 * third_night
    first_night + second_night + third_night + fourth_night = 54
  := by
    sorry

end billy_sleep_total_hours_l216_216882


namespace double_neg_eq_pos_l216_216640

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l216_216640


namespace problem1_sol_l216_216345

noncomputable def problem1 :=
  let total_people := 200
  let avg_feelings_total := 70
  let female_total := 100
  let a := 30 -- derived from 2a + (70 - a) = 100
  let chi_square := 200 * (70 * 40 - 30 * 60) ^ 2 / (130 * 70 * 100 * 100)
  let k_95 := 3.841 -- critical value for 95% confidence
  let p_xi_2 := (1 / 3)
  let p_xi_3 := (1 / 2)
  let p_xi_4 := (1 / 6)
  let exi := (2 * (1 / 3)) + (3 * (1 / 2)) + (4 * (1 / 6))
  chi_square < k_95 ∧ exi = 17 / 6

theorem problem1_sol : problem1 :=
  by {
    sorry
  }

end problem1_sol_l216_216345


namespace square_side_length_l216_216209

/-- Define OPEN as a square and T a point on side NO
    such that the areas of triangles TOP and TEN are 
    respectively 62 and 10. Prove that the side length 
    of the square is 12. -/
theorem square_side_length (s x y : ℝ) (T : x + y = s)
    (h1 : 0 < s) (h2 : 0 < x) (h3 : 0 < y)
    (a1 : 1 / 2 * x * s = 62)
    (a2 : 1 / 2 * y * s = 10) :
    s = 12 :=
by
    sorry

end square_side_length_l216_216209


namespace alpha_in_second_quadrant_l216_216868

variable (α : ℝ)

-- Conditions that P(tan α, cos α) is in the third quadrant
def P_in_third_quadrant (α : ℝ) : Prop := (Real.tan α < 0) ∧ (Real.cos α < 0)

-- Theorem statement
theorem alpha_in_second_quadrant (h : P_in_third_quadrant α) : 
  π/2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l216_216868


namespace fir_trees_count_l216_216544

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l216_216544


namespace total_volume_of_five_cubes_l216_216195

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end total_volume_of_five_cubes_l216_216195


namespace euclidean_division_mod_l216_216847

theorem euclidean_division_mod (h1 : 2022 % 19 = 8)
                               (h2 : 8^6 % 19 = 1)
                               (h3 : 2023 % 6 = 1)
                               (h4 : 2023^2024 % 6 = 1) 
: 2022^(2023^2024) % 19 = 8 := 
by
  sorry

end euclidean_division_mod_l216_216847


namespace smallest_n_l216_216897

theorem smallest_n (n : ℕ) (k : ℕ) (a m : ℕ) 
  (h1 : 0 ≤ k)
  (h2 : k < n)
  (h3 : a ≡ k [MOD n])
  (h4 : m > 0) :
  (∀ a m, (∃ k, a = n * k + 5) -> (a^2 - 3*a + 1) ∣ (a^m + 3^m) → false) 
  → n = 11 := sorry

end smallest_n_l216_216897


namespace rectangle_ratio_l216_216495

-- Define the width of the rectangle
def width : ℕ := 7

-- Define the area of the rectangle
def area : ℕ := 196

-- Define that the length is a multiple of the width
def length_is_multiple_of_width (l w : ℕ) : Prop := ∃ k : ℕ, l = k * w

-- Define that the ratio of the length to the width is 4:1
def ratio_is_4_to_1 (l w : ℕ) : Prop := l / w = 4

theorem rectangle_ratio (l w : ℕ) (h1 : w = width) (h2 : area = l * w) (h3 : length_is_multiple_of_width l w) : ratio_is_4_to_1 l w :=
by
  sorry

end rectangle_ratio_l216_216495


namespace inclination_angle_of_line_l216_216045

theorem inclination_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 2 * x - y + 1 = 0 → m = 2) → θ = Real.arctan 2 :=
by
  sorry

end inclination_angle_of_line_l216_216045


namespace ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l216_216249

-- Define the conditions for the ellipse problem
def major_axis_length : ℝ := 10
def focal_length : ℝ := 4

-- Define the conditions for the parabola problem
def point_P : ℝ × ℝ := (-2, -4)

-- The equations to be proven
theorem ellipse_equation_x_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, x^2 / 25 + y^2 / 21 = 1) := sorry

theorem ellipse_equation_y_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, y^2 / 25 + x^2 / 21 = 1) := sorry

theorem parabola_equation_x_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, y^2 = -8 * x) := sorry

theorem parabola_equation_y_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, x^2 = -y) := sorry

end ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l216_216249


namespace relationship_among_three_numbers_l216_216475

theorem relationship_among_three_numbers :
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  b < a ∧ a < c :=
by
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  sorry

end relationship_among_three_numbers_l216_216475


namespace part1_part2_part3_part4_l216_216905

-- Part 1: Prove that 1/42 is equal to 1/6 - 1/7
theorem part1 : (1/42 : ℚ) = (1/6 : ℚ) - (1/7 : ℚ) := sorry

-- Part 2: Prove that 1/240 is equal to 1/15 - 1/16
theorem part2 : (1/240 : ℚ) = (1/15 : ℚ) - (1/16 : ℚ) := sorry

-- Part 3: Prove the general rule for all natural numbers m
theorem part3 (m : ℕ) (hm : m > 0) : (1 / (m * (m + 1)) : ℚ) = (1 / m : ℚ) - (1 / (m + 1) : ℚ) := sorry

-- Part 4: Prove the given expression evaluates to 0 for any x
theorem part4 (x : ℚ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) : 
  (1 / ((x - 2) * (x - 3)) : ℚ) - (2 / ((x - 1) * (x - 3)) : ℚ) + (1 / ((x - 1) * (x - 2)) : ℚ) = 0 := sorry

end part1_part2_part3_part4_l216_216905


namespace drinking_ratio_l216_216205

variable (t_mala t_usha : ℝ) (d_usha : ℝ)

theorem drinking_ratio :
  (t_mala = t_usha) → 
  (d_usha = 2 / 10) →
  (1 - d_usha = 8 / 10) →
  (4 * d_usha = 8) :=
by
  intros h1 h2 h3
  sorry

end drinking_ratio_l216_216205


namespace least_number_to_subtract_l216_216807

theorem least_number_to_subtract (x : ℕ) (h : 509 - x = 45 * n) : ∃ x, (509 - x) % 9 = 0 ∧ (509 - x) % 15 = 0 ∧ x = 14 := by
  sorry

end least_number_to_subtract_l216_216807


namespace number_of_teams_l216_216347

-- Define the conditions
def math_club_girls : ℕ := 4
def math_club_boys : ℕ := 7
def team_girls : ℕ := 3
def team_boys : ℕ := 3

-- Compute the number of ways to choose 3 girls from 4 girls
def choose_comb_girls : ℕ := Nat.choose math_club_girls team_girls

-- Compute the number of ways to choose 3 boys from 7 boys
def choose_comb_boys : ℕ := Nat.choose math_club_boys team_boys

-- Formulate the goal statement
theorem number_of_teams : choose_comb_girls * choose_comb_boys = 140 := by
  sorry

end number_of_teams_l216_216347


namespace probability_of_exactly_three_correct_packages_l216_216705

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l216_216705


namespace double_neg_eq_pos_l216_216638

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l216_216638


namespace inequality_abc_l216_216162

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_abc :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) :=
  sorry

end inequality_abc_l216_216162


namespace color_opposite_gold_is_yellow_l216_216959

-- Define the colors as a datatype for clarity
inductive Color
| B | Y | O | K | S | G

-- Define the type for each face's color
structure CubeFaces :=
(top front right back left bottom : Color)

-- Given conditions
def first_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.Y ∧ c.right = Color.O

def second_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.K ∧ c.right = Color.O

def third_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.S ∧ c.right = Color.O

-- Problem statement
theorem color_opposite_gold_is_yellow (c : CubeFaces) :
  first_view c → second_view c → third_view c → (c.back = Color.G) → (c.front = Color.Y) :=
by
  sorry

end color_opposite_gold_is_yellow_l216_216959


namespace number_of_boys_in_school_l216_216674

theorem number_of_boys_in_school (total_students : ℕ) (sample_size : ℕ) 
(number_diff : ℕ) (ratio_boys_sample_girls_sample : ℚ) : 
total_students = 1200 → sample_size = 200 → number_diff = 10 →
ratio_boys_sample_girls_sample = 105 / 95 →
∃ (boys_in_school : ℕ), boys_in_school = 630 := by 
  sorry

end number_of_boys_in_school_l216_216674


namespace range_of_a_l216_216565

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end range_of_a_l216_216565


namespace line_eq_form_l216_216097

def line_equation (x y : ℝ) : Prop :=
  ((3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 3) = 0)

theorem line_eq_form (x y : ℝ) (h : line_equation x y) :
  ∃ (m b : ℝ), y = m * x + b ∧ (m = 3/4 ∧ b = -9/2) :=
by
  sorry

end line_eq_form_l216_216097


namespace december_fraction_of_yearly_sales_l216_216633

theorem december_fraction_of_yearly_sales (A : ℝ) (h_sales : ∀ (x : ℝ), x = 6 * A) :
    let yearly_sales := 11 * A + 6 * A
    let december_sales := 6 * A
    december_sales / yearly_sales = 6 / 17 := by
  sorry

end december_fraction_of_yearly_sales_l216_216633


namespace number_of_trees_is_eleven_l216_216532

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l216_216532


namespace max_integer_solutions_l216_216221

noncomputable def semi_centered (p : ℕ → ℤ) :=
  ∃ k : ℕ, p k = k + 50 - 50 * 50

theorem max_integer_solutions (p : ℕ → ℤ) (h1 : semi_centered p) (h2 : ∀ x : ℕ, ∃ c : ℤ, p x = c * x^2) (h3 : p 50 = 50) :
  ∃ n ≤ 6, ∀ k : ℕ, (p k = k^2) → k ∈ Finset.range (n+1) :=
sorry

end max_integer_solutions_l216_216221


namespace find_x_l216_216547

theorem find_x (x : ℝ) (h : x ^ 2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
sorry

end find_x_l216_216547


namespace ratio_of_increase_to_original_l216_216669

noncomputable def ratio_increase_avg_marks (T : ℝ) : ℝ :=
  let original_avg := T / 40
  let new_total := T + 20
  let new_avg := new_total / 40
  let increase_avg := new_avg - original_avg
  increase_avg / original_avg

theorem ratio_of_increase_to_original (T : ℝ) (hT : T > 0) :
  ratio_increase_avg_marks T = 20 / T :=
by
  unfold ratio_increase_avg_marks
  sorry

end ratio_of_increase_to_original_l216_216669


namespace quadratic_root_relationship_l216_216048

noncomputable def roots_of_quadratic (a b c: ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : Prop :=
  b / c = 27

theorem quadratic_root_relationship (a b c : ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : 
  roots_of_quadratic a b c h_nonzero h_root_relation := 
by 
  sorry

end quadratic_root_relationship_l216_216048


namespace ratio_of_dividends_l216_216955

-- Definitions based on conditions
def expected_earnings : ℝ := 0.80
def actual_earnings : ℝ := 1.10
def additional_per_increment : ℝ := 0.04
def increment_size : ℝ := 0.10

-- Definition for the base dividend D which remains undetermined
variable (D : ℝ)

-- Stating the theorem
theorem ratio_of_dividends 
  (h1 : actual_earnings = 1.10)
  (h2 : expected_earnings = 0.80)
  (h3 : additional_per_increment = 0.04)
  (h4 : increment_size = 0.10) :
  let additional_earnings := actual_earnings - expected_earnings
  let increments := additional_earnings / increment_size
  let additional_dividend := increments * additional_per_increment
  let total_dividend := D + additional_dividend
  let ratio := total_dividend / actual_earnings
  ratio = (D + 0.12) / 1.10 :=
by
  sorry

end ratio_of_dividends_l216_216955


namespace contrapositive_statement_l216_216607

theorem contrapositive_statement {a b : ℤ} :
  (∀ a b : ℤ, (a % 2 = 1 ∧ b % 2 = 1) → (a + b) % 2 = 0) →
  (∀ a b : ℤ, ¬((a + b) % 2 = 0) → ¬(a % 2 = 1 ∧ b % 2 = 1)) :=
by 
  intros h a b
  sorry

end contrapositive_statement_l216_216607


namespace parabola_properties_l216_216266

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c : ℝ) (h₀ : a ≠ 0)
    (h₁ : parabola a b c (-1) = -1)
    (h₂ : parabola a b c 0 = 1)
    (h₃ : parabola a b c (-2) > 1) :
    (a * b * c > 0) ∧
    (∃ Δ : ℝ, Δ > 0 ∧ (Δ = b^2 - 4*a*c)) ∧
    (a + b + c > 7) :=
sorry

end parabola_properties_l216_216266


namespace increase_in_output_with_assistant_l216_216009

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l216_216009


namespace correct_choice_C_l216_216384

theorem correct_choice_C (x : ℝ) : x^2 ≥ x - 1 := 
sorry

end correct_choice_C_l216_216384


namespace trajectory_of_P_eqn_l216_216560

theorem trajectory_of_P_eqn :
  ∀ {x y : ℝ}, -- For all real numbers x and y
  (-(x + 2)^2 + (x - 1)^2 + y^2 = 3*((x - 1)^2 + y^2)) → -- Condition |PA| = 2|PB|
  (x^2 + y^2 - 4*x = 0) := -- Prove the trajectory equation
by
  intros x y h
  sorry -- Proof to be completed

end trajectory_of_P_eqn_l216_216560


namespace range_of_a_l216_216269

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) * (2 * x^2 + 1)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (3 : ℝ) 4, f (a * x + 1) ≤ f (x - 2)) ↔ -2 / 3 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l216_216269


namespace boy_completes_work_in_nine_days_l216_216632

theorem boy_completes_work_in_nine_days :
  let M := (1 : ℝ) / 6
  let W := (1 : ℝ) / 18
  let B := (1 / 3 : ℝ) - M - W
  B = (1 : ℝ) / 9 := by
    sorry

end boy_completes_work_in_nine_days_l216_216632


namespace total_value_of_item_l216_216627

theorem total_value_of_item (V : ℝ) 
  (h1 : 0.07 * (V - 1000) = 109.20) : 
  V = 2560 :=
sorry

end total_value_of_item_l216_216627


namespace remainders_sum_l216_216062

theorem remainders_sum (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 20) 
  (h3 : c % 30 = 10) : 
  (a + b + c) % 30 = 15 := 
by
  sorry

end remainders_sum_l216_216062


namespace combined_collectors_edition_dolls_l216_216521

-- Definitions based on given conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def luna_dolls : ℕ := ivy_dolls - 10

-- Additional constraints based on the problem statement
def total_dolls : ℕ := dina_dolls + ivy_dolls + luna_dolls
def ivy_collectors_edition_dolls : ℕ := 2/3 * ivy_dolls
def luna_collectors_edition_dolls : ℕ := 1/2 * luna_dolls

-- Proof statement
theorem combined_collectors_edition_dolls :
  ivy_collectors_edition_dolls + luna_collectors_edition_dolls = 30 :=
sorry

end combined_collectors_edition_dolls_l216_216521


namespace smallest_possible_value_abs_sum_l216_216359

theorem smallest_possible_value_abs_sum :
  ∃ x : ℝ, (∀ y : ℝ, abs (y + 3) + abs (y + 5) + abs (y + 7) ≥ abs (x + 3) + abs (x + 5) + abs (x + 7))
  ∧ (abs (x + 3) + abs (x + 5) + abs (x + 7) = 4) := by
  sorry

end smallest_possible_value_abs_sum_l216_216359


namespace max_area_quadrilateral_sum_opposite_angles_l216_216816

theorem max_area_quadrilateral (a b c d : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) :
  ∃ (area : ℝ), area = 12 :=
by {
  sorry
}

theorem sum_opposite_angles (a b c d : ℝ) (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) 
  (h_area : ∃ (area : ℝ), area = 12) 
  (h_opposite1 : θ₁ + θ₃ = 180) (h_opposite2 : θ₂ + θ₄ = 180) :
  ∃ θ, θ = 180 :=
by {
  sorry
}

end max_area_quadrilateral_sum_opposite_angles_l216_216816


namespace salad_cucumbers_l216_216088

theorem salad_cucumbers (c t : ℕ) 
  (h1 : c + t = 280)
  (h2 : t = 3 * c) : c = 70 :=
sorry

end salad_cucumbers_l216_216088


namespace Iris_pairs_of_pants_l216_216002

theorem Iris_pairs_of_pants (jacket_cost short_cost pant_cost total_spent n_jackets n_shorts n_pants : ℕ) :
  (jacket_cost = 10) →
  (short_cost = 6) →
  (pant_cost = 12) →
  (total_spent = 90) →
  (n_jackets = 3) →
  (n_shorts = 2) →
  (n_jackets * jacket_cost + n_shorts * short_cost + n_pants * pant_cost = total_spent) →
  (n_pants = 4) := 
by
  intros h_jacket_cost h_short_cost h_pant_cost h_total_spent h_n_jackets h_n_shorts h_eq
  sorry

end Iris_pairs_of_pants_l216_216002


namespace number_of_students_l216_216379

theorem number_of_students (pencils: ℕ) (pencils_per_student: ℕ) (total_students: ℕ) 
  (h1: pencils = 195) (h2: pencils_per_student = 3) (h3: total_students = pencils / pencils_per_student) :
  total_students = 65 := by
  -- proof would go here, but we skip it with sorry for now
  sorry

end number_of_students_l216_216379


namespace jacqueline_erasers_l216_216297

def num_boxes : ℕ := 4
def erasers_per_box : ℕ := 10
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 := by
  sorry

end jacqueline_erasers_l216_216297


namespace inequality_proof_l216_216909

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x / Real.sqrt y + y / Real.sqrt x) ≥ (Real.sqrt x + Real.sqrt y) := 
sorry

end inequality_proof_l216_216909


namespace reservoir_water_level_l216_216825

theorem reservoir_water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 6 + 0.3 * x :=
by sorry

end reservoir_water_level_l216_216825


namespace inequality_am_gm_l216_216872

theorem inequality_am_gm (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (h : a^2 + b^2 + c^2 = 12) :
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := 
by
  sorry

end inequality_am_gm_l216_216872


namespace system1_solution_l216_216168

theorem system1_solution (x y : ℝ) (h1 : 2 * x - y = 1) (h2 : 7 * x - 3 * y = 4) : x = 1 ∧ y = 1 :=
by sorry

end system1_solution_l216_216168


namespace breadth_of_rectangle_l216_216368

theorem breadth_of_rectangle (b l : ℝ) (h1 : l * b = 24 * b) (h2 : l - b = 10) : b = 14 :=
by
  sorry

end breadth_of_rectangle_l216_216368


namespace solvable_eq_l216_216247

theorem solvable_eq (x : ℝ) :
    Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6 →
    (x = 2 ∨ x = -2) :=
by
  sorry

end solvable_eq_l216_216247


namespace maximal_correlation_sin_nU_sin_mU_eq_zero_l216_216715

noncomputable def maximal_correlation_of_sine (n m : ℕ) : ℝ :=
  let U := uniform_of_real ([0, 2 * real.pi])
  let sin_nU := fun (u : ℝ) => real.sin(n * u)
  let sin_mU := fun (u : ℝ) => real.sin(m * u)
  let correlation (X Y : ℝ → ℝ) := 
    (E[X U * Y U] - E[X U] * E[Y U]) / (sqrt (E[(X U)^2] - (E[X U])^2) * sqrt (E[(Y U)^2] - (E[Y U])^2))
  sup {f g : ℝ → ℝ | measurable f ∧ measurable g ∧ ∀ u, is_finite (f (U u)) ∧ is_finite (g (U u))} 
    (correlation f g)

theorem maximal_correlation_sin_nU_sin_mU_eq_zero (n m : ℕ) (h_n : n > 0) (h_m : m > 0) :
  maximal_correlation_of_sine n m = 0 := 
sorry

end maximal_correlation_sin_nU_sin_mU_eq_zero_l216_216715


namespace minimum_workers_needed_l216_216100

theorem minimum_workers_needed 
  (total_days : ℕ)
  (completed_days : ℕ)
  (initial_workers : ℕ)
  (fraction_completed : ℚ)
  (remaining_fraction : ℚ)
  (remaining_days : ℕ)
  (rate_completed_per_day : ℚ)
  (required_rate_per_day : ℚ)
  (equal_productivity : Prop) 
  : initial_workers = 10 :=
by
  -- Definitions
  let total_days := 40
  let completed_days := 10
  let initial_workers := 10
  let fraction_completed := 1 / 4
  let remaining_fraction := 1 - fraction_completed
  let remaining_days := total_days - completed_days
  let rate_completed_per_day := fraction_completed / completed_days
  let required_rate_per_day := remaining_fraction / remaining_days
  let equal_productivity := true

  -- Sorry is used to skip the proof
  sorry

end minimum_workers_needed_l216_216100


namespace problem_probability_ao_drawn_second_l216_216353

def is_ao_drawn_second (pair : ℕ × ℕ) : Bool :=
  pair.snd = 3

def random_pairs : List (ℕ × ℕ) := [
  (1, 3), (2, 4), (1, 2), (3, 2), (4, 3), (1, 4), (2, 4), (3, 2), (3, 1), (2, 1), 
  (2, 3), (1, 3), (3, 2), (2, 1), (2, 4), (4, 2), (1, 3), (3, 2), (2, 1), (3, 4)
]

def count_ao_drawn_second : ℕ :=
  (random_pairs.filter is_ao_drawn_second).length

def probability_ao_drawn_second : ℚ :=
  count_ao_drawn_second / random_pairs.length

theorem problem_probability_ao_drawn_second :
  probability_ao_drawn_second = 1 / 4 :=
by
  sorry

end problem_probability_ao_drawn_second_l216_216353


namespace Todd_time_correct_l216_216237

theorem Todd_time_correct :
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  Todd_time = 88 :=
by
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  sorry

end Todd_time_correct_l216_216237


namespace total_games_played_in_league_l216_216822

theorem total_games_played_in_league (n : ℕ) (k : ℕ) (games_per_team : ℕ) 
  (h1 : n = 10) 
  (h2 : k = 4) 
  (h3 : games_per_team = n - 1) 
  : (k * (n * games_per_team) / 2) = 180 :=
by
  -- Definitions and transformations go here
  sorry

end total_games_played_in_league_l216_216822


namespace alice_meets_bob_at_25_km_l216_216382

-- Define variables for times, speeds, and distances
variables (t : ℕ) (d : ℕ)

-- Conditions
def distance_between_homes := 41
def alice_speed := 5
def bob_speed := 4
def alice_start_time := 1

-- Relating the distances covered by Alice and Bob when they meet
def alice_walk_distance := alice_speed * (t + alice_start_time)
def bob_walk_distance := bob_speed * t
def total_walk_distance := alice_walk_distance + bob_walk_distance

-- Alexander walks 25 kilometers before meeting Bob
theorem alice_meets_bob_at_25_km :
  total_walk_distance = distance_between_homes → alice_walk_distance = 25 :=
by
  sorry

end alice_meets_bob_at_25_km_l216_216382


namespace same_side_interior_not_complementary_l216_216601

-- Defining the concept of same-side interior angles and complementary angles
def same_side_interior (α β : ℝ) : Prop := 
  α + β = 180 

def complementary (α β : ℝ) : Prop :=
  α + β = 90

-- To state the proposition that should be proven false
theorem same_side_interior_not_complementary (α β : ℝ) (h : same_side_interior α β) : ¬ complementary α β :=
by
  -- We state the observable contradiction here, and since the proof is not required we use sorry
  sorry

end same_side_interior_not_complementary_l216_216601


namespace clock_angle_at_7_oclock_l216_216358

theorem clock_angle_at_7_oclock : 
  ∀ (hour_angle minute_angle : ℝ), 
    (12 : ℝ) * (30 : ℝ) = 360 →
    (7 : ℝ) * (30 : ℝ) = 210 →
    (210 : ℝ) > 180 →
    (360 : ℝ) - (210 : ℝ) = 150 →
    hour_angle = 7 * 30 →
    minute_angle = 0 →
    min (abs (hour_angle - minute_angle)) (abs ((360 - hour_angle) - minute_angle)) = 150 := by
  sorry

end clock_angle_at_7_oclock_l216_216358


namespace ratio_josh_to_doug_l216_216757

theorem ratio_josh_to_doug (J D B : ℕ) (h1 : J + D + B = 68) (h2 : J = 2 * B) (h3 : D = 32) : J / D = 3 / 4 := 
by
  sorry

end ratio_josh_to_doug_l216_216757


namespace speed_first_32_miles_l216_216523

theorem speed_first_32_miles (x : ℝ) (y : ℝ) : 
  (100 / x + 0.52 * 100 / x = 32 / y + 68 / (x / 2)) → 
  y = 2 * x :=
by
  sorry

end speed_first_32_miles_l216_216523


namespace hot_water_bottles_sold_l216_216191

theorem hot_water_bottles_sold (T H : ℕ) (h1 : 2 * T + 6 * H = 1200) (h2 : T = 7 * H) : H = 60 := 
by 
  sorry

end hot_water_bottles_sold_l216_216191


namespace yellow_dandelions_day_before_yesterday_l216_216961

-- Define the problem in terms of conditions and conclusion
theorem yellow_dandelions_day_before_yesterday
  (yellow_yesterday : ℕ) (white_yesterday : ℕ)
  (yellow_today : ℕ) (white_today : ℕ) :
  yellow_yesterday = 20 → white_yesterday = 14 →
  yellow_today = 15 → white_today = 11 →
  (let yellow_day_before_yesterday := white_yesterday + white_today
  in yellow_day_before_yesterday = 25) :=
by
  intros h1 h2 h3 h4
  let yellow_day_before_yesterday := white_yesterday + white_today
  show yellow_day_before_yesterday = 25
  exact (by sorry)

end yellow_dandelions_day_before_yesterday_l216_216961


namespace member_sum_or_double_exists_l216_216233

theorem member_sum_or_double_exists (n : ℕ) (k : ℕ) (P: ℕ → ℕ) (m: ℕ) 
  (h_mem : n = 1978)
  (h_countries : m = 6) : 
  ∃ k, (∃ i j, P i + P j = k ∧ P i = P j)
    ∨ (∃ i, 2 * P i = k) :=
sorry

end member_sum_or_double_exists_l216_216233


namespace solution_set_of_inequality_l216_216564

theorem solution_set_of_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (a - x) * (x - 1 / a) > 0} = {x : ℝ | a < x ∧ x < 1 / a} :=
sorry

end solution_set_of_inequality_l216_216564


namespace count_whole_numbers_between_roots_l216_216134

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l216_216134


namespace doughnuts_given_away_l216_216820

def doughnuts_left (total_doughnuts : Nat) (doughnuts_per_box : Nat) (boxes_sold : Nat) : Nat :=
  total_doughnuts - (doughnuts_per_box * boxes_sold)

theorem doughnuts_given_away :
  doughnuts_left 300 10 27 = 30 :=
by
  rw [doughnuts_left]
  simp
  sorry

end doughnuts_given_away_l216_216820


namespace max_cos_a_l216_216300

theorem max_cos_a (a b c : ℝ) 
  (h1 : Real.sin a = Real.cos b) 
  (h2 : Real.sin b = Real.cos c) 
  (h3 : Real.sin c = Real.cos a) : 
  Real.cos a = Real.sqrt 2 / 2 := by
sorry

end max_cos_a_l216_216300


namespace sequence_general_formula_l216_216725

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  3^n + 1

theorem sequence_general_formula (a : ℕ → ℕ) (n : ℕ) (h₁ : sequence_sum n = ∑ i in Finset.range n, a i) :
  a n = if n = 1 then 4 else 2 * 3^(n - 1) := sorry

end sequence_general_formula_l216_216725


namespace total_vegetables_correct_l216_216846

def cucumbers : ℕ := 70
def tomatoes : ℕ := 3 * cucumbers
def total_vegetables : ℕ := cucumbers + tomatoes

theorem total_vegetables_correct : total_vegetables = 280 :=
by
  sorry

end total_vegetables_correct_l216_216846


namespace root_polynomial_l216_216761

noncomputable def roots : ℂ := sorry

theorem root_polynomial (roots: ℂ) :
    (root.polynomial (x^4 - 4 * x^3 + 8 * x^2 - 7 * x + 3)) = roots ->
    (root.sum (roots) = 4) ->
    (root.sum_products (roots) 2 = 8) ->
  \[
  \frac{roots[1]^2}{roots[2]^2 + roots[3]^2 + roots[4]^2} + 
  \frac{roots[2]^2}{roots[1]^2 + roots[3]^2 + roots{4]^2} + 
  \frac{roots[3]^2}{roots[1]^2 + roots[2]^2 + + roots[4]^2} + 
  \frac{roots[4]^2}{roots[1]^2 + roots[2]^2 + + roots[3]^2} == -4
  sorry

end root_polynomial_l216_216761


namespace simplify_140_210_l216_216914

noncomputable def simplify_fraction (num den : Nat) : Nat × Nat :=
  let d := Nat.gcd num den
  (num / d, den / d)

theorem simplify_140_210 :
  simplify_fraction 140 210 = (2, 3) :=
by
  have p140 : 140 = 2^2 * 5 * 7 := by rfl
  have p210 : 210 = 2 * 3 * 5 * 7 := by rfl
  sorry

end simplify_140_210_l216_216914


namespace gary_asparagus_l216_216994

/-- Formalization of the problem -/
theorem gary_asparagus (A : ℝ) (ha : 700 * 0.50 = 350) (hg : 40 * 2.50 = 100) (hw : 630 = 3 * A + 350 + 100) : A = 60 :=
by
  sorry

end gary_asparagus_l216_216994


namespace trapezoid_base_count_l216_216323

theorem trapezoid_base_count (A h : ℕ) (multiple : ℕ) (bases_sum pairs_count : ℕ) : 
  A = 1800 ∧ h = 60 ∧ multiple = 10 ∧ pairs_count = 4 ∧ 
  bases_sum = (A / (1/2 * h)) / multiple → pairs_count > 3 := 
by 
  sorry

end trapezoid_base_count_l216_216323


namespace approximation_range_l216_216474

theorem approximation_range (a : ℝ) (h : ∃ a_approx : ℝ, 170 = a_approx ∧ a = real_floor (a_approx) + 0.5) :
  169.5 ≤ a ∧ a < 170.5 :=
sorry

end approximation_range_l216_216474


namespace log2_monotone_l216_216901

theorem log2_monotone (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (Real.log a / Real.log 2 > Real.log b / Real.log 2) :=
sorry

end log2_monotone_l216_216901


namespace calculate_sum_and_difference_l216_216975

theorem calculate_sum_and_difference : 0.5 - 0.03 + 0.007 = 0.477 := sorry

end calculate_sum_and_difference_l216_216975


namespace gray_region_area_l216_216749

theorem gray_region_area (r : ℝ) (h1 : r > 0) (h2 : 3 * r - r = 3) : 
  (π * (3 * r) * (3 * r) - π * r * r) = 18 * π := by
  sorry

end gray_region_area_l216_216749


namespace solve_for_x_l216_216852

def star (a b : ℤ) := a * b + 3 * b - a

theorem solve_for_x : ∃ x : ℤ, star 4 x = 46 := by
  sorry

end solve_for_x_l216_216852


namespace hannah_sweatshirts_l216_216271

theorem hannah_sweatshirts (S : ℕ) (h1 : 15 * S + 2 * 10 = 65) : S = 3 := 
by
  sorry

end hannah_sweatshirts_l216_216271


namespace evaluate_expression_l216_216522

noncomputable def expression (a b : ℕ) := (a + b)^2 - (a - b)^2

theorem evaluate_expression:
  expression (5^500) (6^501) = 24 * 30^500 := by
sorry

end evaluate_expression_l216_216522


namespace find_a_and_other_root_l216_216870

-- Define the quadratic equation with a
def quadratic_eq (a x : ℝ) : ℝ := (a + 1) * x^2 + x - 1

-- Define the conditions where -1 is a root
def condition (a : ℝ) : Prop := quadratic_eq a (-1) = 0

theorem find_a_and_other_root (a : ℝ) :
  condition a → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ quadratic_eq 1 x = 0 ∧ x = 1 / 2) :=
by
  intro h
  sorry

end find_a_and_other_root_l216_216870


namespace circle_locus_l216_216189

theorem circle_locus (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  13 * a^2 + 49 * b^2 - 12 * a - 1 = 0 := 
sorry

end circle_locus_l216_216189


namespace medium_as_decoy_and_rational_choice_l216_216033

/-- 
  Define the prices and sizes of the popcorn containers:
  Small: 50g for 200 rubles.
  Medium: 70g for 400 rubles.
  Large: 130g for 500 rubles.
-/
structure PopcornContainer where
  size : ℕ -- in grams
  price : ℕ -- in rubles

def small := PopcornContainer.mk 50 200
def medium := PopcornContainer.mk 70 400
def large := PopcornContainer.mk 130 500

/-- 
  The medium-sized popcorn container can be considered a decoy
  in the context of asymmetric dominance.
  Additionally, under certain budget constraints and preferences, 
  rational economic agents may find the medium-sized container optimal.
-/
theorem medium_as_decoy_and_rational_choice :
  (medium.price = 400 ∧ medium.size = 70) ∧ 
  (∃ (budget : ℕ) (pref : ℕ → ℕ → Prop), (budget ≥ medium.price ∧ 
    pref medium.size (budget - medium.price))) :=
by
  sorry

end medium_as_decoy_and_rational_choice_l216_216033


namespace incorrect_option_c_l216_216383

theorem incorrect_option_c (a b c d : ℝ)
  (h1 : a + b + c ≥ d)
  (h2 : a + b + d ≥ c)
  (h3 : a + c + d ≥ b)
  (h4 : b + c + d ≥ a) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) :=
by sorry

end incorrect_option_c_l216_216383


namespace max_integer_solutions_l216_216222

theorem max_integer_solutions (p : ℤ[X]) (h₀ : p.coeffs ∈ (set.range (coe : ℤ → ℤ))) 
(h₁ : p.eval 50 = 50) : 
  ∃ k₁ k₂ k₃ k₄ k₅ k₆ : ℤ, 
    p.eval k₁ = k₁ ^ 2 ∧ 
    p.eval k₂ = k₂ ^ 2 ∧ 
    p.eval k₃ = k₃ ^ 2 ∧ 
    p.eval k₄ = k₄ ^ 2 ∧ 
    p.eval k₅ = k₅ ^ 2 ∧ 
    p.eval k₆ = k₆ ^ 2 ∧
    ((set.to_finset {k₁, k₂, k₃, k₄, k₅, k₆}).card ≤ 6) := 
sorry

end max_integer_solutions_l216_216222


namespace cannot_use_square_diff_formula_l216_216939

theorem cannot_use_square_diff_formula :
  ¬ (∃ a b, (x - y) * (-x + y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (-x - y) * (x - y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (-x + y) * (-x - y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (x + y) * (-x + y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) :=
sorry

end cannot_use_square_diff_formula_l216_216939


namespace part_a_impossible_part_b_possible_l216_216099

-- Part (a)
theorem part_a_impossible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ¬ ∀ (x : ℝ), (1 < x ∧ x < a) ∧ (a < 2*x ∧ 2*x < a^2) :=
sorry

-- Part (b)
theorem part_b_possible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ∃ (x : ℝ), (a < 2*x ∧ 2*x < a^2) ∧ ¬ (1 < x ∧ x < a) :=
sorry

end part_a_impossible_part_b_possible_l216_216099


namespace total_income_l216_216402

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l216_216402


namespace baked_by_brier_correct_l216_216031

def baked_by_macadams : ℕ := 20
def baked_by_flannery : ℕ := 17
def total_baked : ℕ := 55

def baked_by_brier : ℕ := total_baked - (baked_by_macadams + baked_by_flannery)

-- Theorem statement
theorem baked_by_brier_correct : baked_by_brier = 18 := 
by
  -- proof will go here 
  sorry

end baked_by_brier_correct_l216_216031


namespace xanthia_hot_dogs_l216_216811

theorem xanthia_hot_dogs (a b : ℕ) (h₁ : a = 5) (h₂ : b = 7) :
  ∃ n m : ℕ, n * a = m * b ∧ n = 7 := by 
sorry

end xanthia_hot_dogs_l216_216811


namespace jimmy_irene_total_payment_l216_216891

def cost_jimmy_shorts : ℝ := 3 * 15
def cost_irene_shirts : ℝ := 5 * 17
def total_cost_before_discount : ℝ := cost_jimmy_shorts + cost_irene_shirts
def discount : ℝ := total_cost_before_discount * 0.10
def total_paid : ℝ := total_cost_before_discount - discount

theorem jimmy_irene_total_payment : total_paid = 117 := by
  sorry

end jimmy_irene_total_payment_l216_216891


namespace problem_statement_l216_216924

noncomputable def x : ℝ := sorry -- Let x be a real number satisfying the condition

theorem problem_statement (x_real_cond : x + 1/x = 3) : 
  (x^12 - 7*x^8 + 2*x^4) = 44387*x - 15088 :=
sorry

end problem_statement_l216_216924


namespace car_speed_proof_l216_216929

noncomputable def car_speed_second_hour 
  (speed_first_hour: ℕ) (average_speed: ℕ) (total_time: ℕ) 
  (speed_second_hour: ℕ) : Prop :=
  (speed_first_hour = 80) ∧ (average_speed = 70) ∧ (total_time = 2) → speed_second_hour = 60

theorem car_speed_proof : 
  car_speed_second_hour 80 70 2 60 := by
  sorry

end car_speed_proof_l216_216929


namespace max_f_of_sin_bounded_l216_216611

theorem max_f_of_sin_bounded (x : ℝ) : (∀ y, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1) → ∃ m, (∀ z, (1 + 2 * Real.sin z) ≤ m) ∧ (∀ n, (∀ z, (1 + 2 * Real.sin z) ≤ n) → m ≤ n) :=
by
  sorry

end max_f_of_sin_bounded_l216_216611


namespace range_of_a_l216_216650

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l216_216650


namespace sample_variance_is_two_l216_216932

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : 
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
sorry

end sample_variance_is_two_l216_216932


namespace abs_value_identity_l216_216146

theorem abs_value_identity (a : ℝ) (h : a + |a| = 0) : a - |2 * a| = 3 * a :=
by
  sorry

end abs_value_identity_l216_216146


namespace find_p_from_circle_and_parabola_tangency_l216_216421

theorem find_p_from_circle_and_parabola_tangency :
  (∃ x y : ℝ, (x^2 + y^2 - 6*x - 7 = 0) ∧ (y^2 = 2*p * x) ∧ p > 0) →
  p = 2 :=
by {
  sorry
}

end find_p_from_circle_and_parabola_tangency_l216_216421


namespace five_pow_10000_mod_1000_l216_216908

theorem five_pow_10000_mod_1000 (h : 5^500 ≡ 1 [MOD 1000]) : 5^10000 ≡ 1 [MOD 1000] := sorry

end five_pow_10000_mod_1000_l216_216908


namespace man_l216_216833

theorem man's_rate_in_still_water (speed_with_stream speed_against_stream : ℝ) (h1 : speed_with_stream = 26) (h2 : speed_against_stream = 12) : 
  (speed_with_stream + speed_against_stream) / 2 = 19 := 
by
  rw [h1, h2]
  norm_num

end man_l216_216833


namespace even_poly_iff_a_zero_l216_216283

theorem even_poly_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 3) = (x^2 - a*x + 3)) → a = 0 :=
by
  sorry

end even_poly_iff_a_zero_l216_216283


namespace number_of_math_fun_books_l216_216355

def intelligence_challenge_cost := 18
def math_fun_cost := 8
def total_spent := 92

theorem number_of_math_fun_books (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : intelligence_challenge_cost * x + math_fun_cost * y = total_spent) : y = 7 := 
by
  sorry

end number_of_math_fun_books_l216_216355


namespace polygon_diagonals_l216_216993

-- Lean statement of the problem

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 2018) : n = 2021 :=
  by sorry

end polygon_diagonals_l216_216993


namespace typing_time_in_hours_l216_216156

def words_per_minute := 32
def word_count := 7125
def break_interval := 25
def break_time := 5
def mistake_interval := 100
def correction_time_per_mistake := 1

theorem typing_time_in_hours :
  let typing_time := (word_count + words_per_minute - 1) / words_per_minute
  let breaks := typing_time / break_interval
  let total_break_time := breaks * break_time
  let mistakes := (word_count + mistake_interval - 1) / mistake_interval
  let total_correction_time := mistakes * correction_time_per_mistake
  let total_time := typing_time + total_break_time + total_correction_time
  let total_hours := (total_time + 60 - 1) / 60
  total_hours = 6 :=
by
  sorry

end typing_time_in_hours_l216_216156


namespace calculate_z_l216_216571

-- Given conditions
def equally_spaced : Prop := true -- assume equally spaced markings do exist
def total_distance : ℕ := 35
def number_of_steps : ℕ := 7
def step_length : ℕ := total_distance / number_of_steps
def starting_point : ℕ := 10
def steps_forward : ℕ := 4

-- Theorem to prove
theorem calculate_z (h1 : equally_spaced)
(h2 : step_length = 5)
: starting_point + (steps_forward * step_length) = 30 :=
by sorry

end calculate_z_l216_216571


namespace assistant_increases_output_by_100_percent_l216_216018

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l216_216018


namespace total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l216_216298

def keith_pears : ℕ := 6
def keith_apples : ℕ := 4
def jason_pears : ℕ := 9
def jason_apples : ℕ := 8
def joan_pears : ℕ := 4
def joan_apples : ℕ := 12

def total_pears : ℕ := keith_pears + jason_pears + joan_pears
def total_apples : ℕ := keith_apples + jason_apples + joan_apples
def total_fruits : ℕ := total_pears + total_apples
def apple_to_pear_ratio : ℚ := total_apples / total_pears

theorem total_fruits_is_43 : total_fruits = 43 := by
  sorry

theorem apple_to_pear_ratio_is_24_to_19 : apple_to_pear_ratio = 24/19 := by
  sorry

end total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l216_216298


namespace total_matchsticks_l216_216525

theorem total_matchsticks (boxes : ℕ) (matchboxes_per_box : ℕ) (sticks_per_matchbox : ℕ) 
  (h1 : boxes = 4) (h2 : matchboxes_per_box = 20) (h3 : sticks_per_matchbox = 300) :
  boxes * matchboxes_per_box * sticks_per_matchbox = 24000 :=
by 
  rw [h1, h2, h3];
  norm_num

end total_matchsticks_l216_216525


namespace parabola_and_line_sum_l216_216270

theorem parabola_and_line_sum (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y : ℝ, (y^2 = 4 * x) ↔ (x, y) = A ∨ (x, y) = B)
  (h_line : ∀ x y : ℝ, (2 * x + y - 4 = 0) ↔ (x, y) = A ∨ (x, y) = B)
  (h_focus : F = (1, 0))
  : |F - A| + |F - B| = 7 := 
sorry

end parabola_and_line_sum_l216_216270


namespace reciprocal_of_one_fifth_l216_216047

theorem reciprocal_of_one_fifth : (∃ x : ℚ, (1/5) * x = 1 ∧ x = 5) :=
by
  -- The proof goes here, for now we assume it with sorry
  sorry

end reciprocal_of_one_fifth_l216_216047


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l216_216140

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l216_216140


namespace converse_not_true_without_negatives_l216_216091

theorem converse_not_true_without_negatives (a b c d : ℕ) (h : a + d = b + c) : ¬(a - c = b - d) :=
by
  sorry

end converse_not_true_without_negatives_l216_216091


namespace fraction_of_budget_is_31_percent_l216_216442

def coffee_pastry_cost (B : ℝ) (c : ℝ) (p : ℝ) :=
  c = 0.25 * (B - p) ∧ p = 0.10 * (B - c)

theorem fraction_of_budget_is_31_percent (B c p : ℝ) (h : coffee_pastry_cost B c p) :
  c + p = 0.31 * B :=
sorry

end fraction_of_budget_is_31_percent_l216_216442


namespace slopes_angle_l216_216798

theorem slopes_angle (k_1 k_2 : ℝ) (θ : ℝ) 
  (h1 : 6 * k_1^2 + k_1 - 1 = 0)
  (h2 : 6 * k_2^2 + k_2 - 1 = 0) :
  θ = π / 4 ∨ θ = 3 * π / 4 := 
by sorry

end slopes_angle_l216_216798


namespace relationship_between_c_and_d_l216_216280

noncomputable def c : ℝ := Real.log 400 / Real.log 4
noncomputable def d : ℝ := Real.log 20 / Real.log 2

theorem relationship_between_c_and_d : c = d := by
  sorry

end relationship_between_c_and_d_l216_216280


namespace brick_height_l216_216407

def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem brick_height 
  (l : ℝ) (w : ℝ) (SA : ℝ) (h : ℝ) 
  (surface_area_eq : surface_area l w h = SA)
  (length_eq : l = 10)
  (width_eq : w = 4)
  (surface_area_given : SA = 164) :
  h = 3 :=
by
  sorry

end brick_height_l216_216407


namespace correct_solutions_l216_216600

theorem correct_solutions (x y z t : ℕ) : 
  (x^2 + t^2) * (z^2 + y^2) = 50 → 
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨ 
  (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨ 
  (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
sorry

end correct_solutions_l216_216600


namespace volume_is_750_sqrt2_l216_216919

noncomputable def volume_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : ℝ :=
a * b * c

theorem volume_is_750_sqrt2 (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : volume_of_prism a b c h1 h2 h3 = 750 * real.sqrt 2 :=
by sorry

end volume_is_750_sqrt2_l216_216919


namespace dryer_sheets_per_load_l216_216776

theorem dryer_sheets_per_load (loads_per_week : ℕ) (cost_of_box : ℝ) (sheets_per_box : ℕ)
  (annual_savings : ℝ) (weeks_in_year : ℕ) (x : ℕ)
  (h1 : loads_per_week = 4)
  (h2 : cost_of_box = 5.50)
  (h3 : sheets_per_box = 104)
  (h4 : annual_savings = 11)
  (h5 : weeks_in_year = 52)
  (h6 : annual_savings = 2 * cost_of_box)
  (h7 : sheets_per_box * 2 = weeks_in_year * (loads_per_week * x)):
  x = 1 :=
by
  sorry

end dryer_sheets_per_load_l216_216776


namespace arithmetic_sequence_terms_sum_l216_216875

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)

-- Definitions based on given problem conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) := 
  ∀ n, S n = n * (a 1 + a n) / 2

axiom Sn_2017 : S_n 2017 = 4034

-- Goal: a_3 + a_1009 + a_2015 = 6
theorem arithmetic_sequence_terms_sum :
  arithmetic_sequence a_n →
  sum_first_n_terms S_n a_n →
  S_n 2017 = 4034 → 
  a_n 3 + a_n 1009 + a_n 2015 = 6 :=
by
  intros
  sorry

end arithmetic_sequence_terms_sum_l216_216875


namespace problem_proof_l216_216432

def M : ℚ := 28
def N : ℚ := 147

theorem problem_proof : (M - N) = -119 := by
  -- Given conditions
  have hM : (4 : ℚ) / 7 = M / 49 := by
    rw [M]
    norm_num
  have hN : (4 : ℚ) / 7 = 84 / N := by
    rw [N]
    norm_num
  -- Prove the required result
  calc
    M - N = 28 - 147 := by rw [M, N]
    ... = -119 := by norm_num

end problem_proof_l216_216432


namespace probability_sum_seven_two_dice_l216_216360

theorem probability_sum_seven_two_dice :
  let outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} }
  let favorable := { (x, y) ∈ outcomes | x + y = 7 }
  ∑ x in favorable, 1 / ∑ xy in outcomes, 1 = 1 / 6 := 
by
  let outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} }
  let favorable := { (x, y) ∈ outcomes | x + y = 7 }
  have favorable_count : favorable.card = 6 := sorry
  have outcomes_count : outcomes.card = 36 := sorry
  have probability_eq : favorable_count / outcomes_count = 1 / 6 := by 
    rw [favorable_count, outcomes_count]
    norm_num
  show probability_eq = 1 / 6 from probability_eq

end probability_sum_seven_two_dice_l216_216360


namespace equation_solution_l216_216566

theorem equation_solution (x : ℝ) (h : 8^(Real.log 5 / Real.log 8) = 10 * x + 3) : x = 1 / 5 :=
sorry

end equation_solution_l216_216566


namespace find_m_and_n_l216_216900

theorem find_m_and_n (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
                    (h2 : a + b + c + d = m^2) 
                    (h3 : max a (max b (max c d)) = n^2) : 
                    m = 9 ∧ n = 6 := 
sorry

end find_m_and_n_l216_216900


namespace range_of_m_decreasing_l216_216256

theorem range_of_m_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m - 3) * x₁ + 5 > (m - 3) * x₂ + 5) ↔ m < 3 :=
by
  sorry

end range_of_m_decreasing_l216_216256


namespace solve_system_eq_l216_216597

theorem solve_system_eq (x y z t : ℕ) : 
  ((x^2 + t^2) * (z^2 + y^2) = 50) ↔
    (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
    (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
    (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
by 
  sorry

end solve_system_eq_l216_216597


namespace part1_part2_l216_216997

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := 
by
  sorry

theorem part2 (h : Real.tan α = 2) : Real.sin α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
by
  sorry

end part1_part2_l216_216997


namespace find_triangle_sides_l216_216603

noncomputable def triangle_sides (x : ℝ) : Prop :=
  let a := x - 2
  let b := x
  let c := x + 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a + 2 = b ∧ b + 2 = c ∧ area = 6 ∧
  a = 2 * Real.sqrt 6 - 2 ∧
  b = 2 * Real.sqrt 6 ∧
  c = 2 * Real.sqrt 6 + 2

theorem find_triangle_sides :
  ∃ x : ℝ, triangle_sides x := by
  sorry

end find_triangle_sides_l216_216603


namespace subtraction_example_l216_216482

theorem subtraction_example :
  145.23 - 0.07 = 145.16 :=
sorry

end subtraction_example_l216_216482


namespace smallest_value_l216_216299

open Matrix

noncomputable def is_solution (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (⟦3, 0; 0, 2⟧ ⬝ ⟦a, b; c, d⟧ = ⟦a, b; c, d⟧ ⬝ ⟦18, 12; -20, -13⟧)

theorem smallest_value :
  ∃ a b c d : ℕ, is_solution a b c d ∧ (a + b + c + d = 16) :=
sorry

end smallest_value_l216_216299


namespace proportion_equation_correct_l216_216411

theorem proportion_equation_correct (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) : 
  x / 3 = y / 2 := 
  sorry

end proportion_equation_correct_l216_216411


namespace sqrt_plus_inv_sqrt_eq_l216_216445

noncomputable def sqrt_plus_inv_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x + 1 / Real.sqrt x

theorem sqrt_plus_inv_sqrt_eq (x : ℝ) (h₁ : 0 < x) (h₂ : x + 1 / x = 50) :
  sqrt_plus_inv_sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_plus_inv_sqrt_eq_l216_216445


namespace sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l216_216349

theorem sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^4 + t^2) = |t| * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l216_216349


namespace difference_of_numbers_is_21938_l216_216616

theorem difference_of_numbers_is_21938 
  (x y : ℕ) 
  (h1 : x + y = 26832) 
  (h2 : x % 10 = 0) 
  (h3 : y = x / 10 + 4) 
  : x - y = 21938 :=
sorry

end difference_of_numbers_is_21938_l216_216616


namespace largest_angle_of_obtuse_isosceles_triangle_l216_216188

def triangleXYZ : Type := { XYZ : Type // XYZ = Triangle }
def isosceles_triangle (T : triangleXYZ) : Prop := Isosceles T.val
def obtuse_triangle (T : triangleXYZ) : Prop := Obtuse T.val
def angle_X_30_degrees (T : triangleXYZ) : Prop := Angle T.val X = 30

def largest_angle_measure (T : triangleXYZ) : ℕ := 120

theorem largest_angle_of_obtuse_isosceles_triangle (T : triangleXYZ) 
  (h1 : isosceles_triangle T) 
  (h2 : obtuse_triangle T) 
  (h3 : angle_X_30_degrees T) : 
  Angle T.val (largest_interior_angle T.val) = largest_angle_measure T :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l216_216188


namespace tens_digit_of_8_pow_1234_l216_216485

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end tens_digit_of_8_pow_1234_l216_216485


namespace hybrid_monotonous_count_l216_216853

open Nat

/--
A positive integer is defined as "hybrid-monotonous" if it is a one-digit number or 
its digits, when read from left to right, form a strictly increasing or a strictly 
decreasing sequence and must include at least one odd and one even digit.
Using digits 0 through 9, the number of hybrid-monotonous positive integers is 1902.
-/
theorem hybrid_monotonous_count : 
  let is_hybrid_monotonous (n : Nat) : Prop :=
    (n < 10) ∨ (strictlyIncreasingDigits n ∨ strictlyDecreasingDigits n) ∧ includesOddAndEvenDigits n
  in count (fun n => is_hybrid_monotonous n) (Finset.range 10000) = 1902 :=
  sorry

end hybrid_monotonous_count_l216_216853


namespace cashier_amount_l216_216889

def amount_to_cashier (discount : ℝ) (shorts_count : ℕ) (shorts_price : ℕ) (shirts_count : ℕ) (shirts_price : ℕ) : ℝ :=
  let total_cost := (shorts_count * shorts_price) + (shirts_count * shirts_price)
  let discount_amount := discount * total_cost
  total_cost - discount_amount

theorem cashier_amount : amount_to_cashier 0.1 3 15 5 17 = 117 := 
by
  sorry

end cashier_amount_l216_216889


namespace skateboard_weight_is_18_l216_216322

def weight_of_canoe : Nat := 45
def weight_of_four_canoes := 4 * weight_of_canoe
def weight_of_ten_skateboards := weight_of_four_canoes
def weight_of_one_skateboard := weight_of_ten_skateboards / 10

theorem skateboard_weight_is_18 : weight_of_one_skateboard = 18 := by
  sorry

end skateboard_weight_is_18_l216_216322


namespace A_is_5_years_older_than_B_l216_216491

-- Given conditions
variables (A B : ℕ) -- A and B are the current ages
variables (x y : ℕ) -- x is the current age of A, y is the current age of B
variables 
  (A_was_B_age : A = y)
  (B_was_10_when_A_was_B_age : B = 10)
  (B_will_be_A_age : B = x)
  (A_will_be_25_when_B_will_be_A_age : A = 25)

-- Define the theorem to prove that A is 5 years older than B: A = B + 5
theorem A_is_5_years_older_than_B (x y : ℕ) (A B : ℕ) 
  (A_was_B_age : x = y) 
  (B_was_10_when_A_was_B_age : y = 10) 
  (B_will_be_A_age : y = x) 
  (A_will_be_25_when_B_will_be_A_age : x = 25): 
  x - y = 5 := 
by sorry

end A_is_5_years_older_than_B_l216_216491


namespace train_crossing_time_correct_l216_216190

noncomputable def train_crossing_time (speed_kmph : ℕ) (length_m : ℕ) (train_dir_opposite : Bool) : ℕ :=
  if train_dir_opposite then
    let speed_mps := speed_kmph * 1000 / 3600
    let relative_speed := speed_mps + speed_mps
    let total_distance := length_m + length_m
    total_distance / relative_speed
  else 0

theorem train_crossing_time_correct :
  train_crossing_time 54 120 true = 8 :=
by
  sorry

end train_crossing_time_correct_l216_216190


namespace number_of_fir_trees_is_11_l216_216534

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l216_216534


namespace original_faculty_members_correct_l216_216303

noncomputable def original_faculty_members : ℝ := 282

theorem original_faculty_members_correct:
  ∃ F : ℝ, (0.6375 * F = 180) ∧ (F = original_faculty_members) :=
by
  sorry

end original_faculty_members_correct_l216_216303


namespace min_value_inverse_sum_l216_216447

theorem min_value_inverse_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) :
  (1 / x + 1 / y + 1 / z) ≥ 9 :=
  sorry

end min_value_inverse_sum_l216_216447


namespace abs_div_one_add_i_by_i_l216_216262

noncomputable def imaginary_unit : ℂ := Complex.I

/-- The absolute value of the complex number (1 + i)/i is √2. -/
theorem abs_div_one_add_i_by_i : Complex.abs ((1 + imaginary_unit) / imaginary_unit) = Real.sqrt 2 := by
  sorry

end abs_div_one_add_i_by_i_l216_216262


namespace sum_of_six_terms_l216_216412

variable {a : ℕ → ℝ} {q : ℝ}

/-- Given conditions:
* a is a decreasing geometric sequence with ratio q
-/
def is_decreasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem sum_of_six_terms
  (h_geo : is_decreasing_geometric_sequence a q)
  (h_decreasing : 0 < q ∧ q < 1)
  (h_a1 : 0 < a 1)
  (h_a1a3 : a 1 * a 3 = 1)
  (h_a2a4 : a 2 + a 4 = 5 / 4) :
  (a 1 * (1 - q^6) / (1 - q)) = 63 / 16 := by
  sorry

end sum_of_six_terms_l216_216412


namespace x_pow_y_equals_nine_l216_216732

theorem x_pow_y_equals_nine (x y : ℝ) (h : (|x + 3| * (y - 2)^2 < 0)) : x^y = 9 :=
sorry

end x_pow_y_equals_nine_l216_216732


namespace digit_equation_l216_216686

-- Definitions for digits and the equation components
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x ≤ 9

def three_digit_number (A B C : ℤ) : ℤ := 100 * A + 10 * B + C
def two_digit_number (A D : ℤ) : ℤ := 10 * A + D
def four_digit_number (A D C : ℤ) : ℤ := 1000 * A + 100 * D + 10 * D + C

-- Statement of the theorem
theorem digit_equation (A B C D : ℤ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D) :
  three_digit_number A B C * two_digit_number A D = four_digit_number A D C :=
sorry

end digit_equation_l216_216686


namespace second_person_avg_pages_per_day_l216_216104

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end second_person_avg_pages_per_day_l216_216104


namespace fraction_evaluation_l216_216408

theorem fraction_evaluation (x z : ℚ) (hx : x = 4/7) (hz : z = 8/11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end fraction_evaluation_l216_216408


namespace carton_weight_l216_216966

theorem carton_weight :
  ∀ (x : ℝ),
  (12 * 4 + 16 * x = 96) → 
  x = 3 :=
by
  intros x h
  sorry

end carton_weight_l216_216966


namespace intersection_A_B_l216_216721

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_A_B :
  A ∩ B = {-3, 1} := by
  sorry

end intersection_A_B_l216_216721


namespace problem_inequality_l216_216794

theorem problem_inequality 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b)
  (c_pos : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := 
sorry

end problem_inequality_l216_216794


namespace heat_more_games_than_bulls_l216_216744

theorem heat_more_games_than_bulls (H : ℕ) 
(h1 : 70 + H = 145) :
H - 70 = 5 :=
sorry

end heat_more_games_than_bulls_l216_216744


namespace number_of_trees_is_11_l216_216543

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l216_216543


namespace Megan_pays_correct_amount_l216_216030

def original_price : ℝ := 22
def discount : ℝ := 6
def amount_paid := original_price - discount

theorem Megan_pays_correct_amount : amount_paid = 16 := by
  sorry

end Megan_pays_correct_amount_l216_216030


namespace seq_general_term_l216_216154

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 0 then 1/2
  else if n = 1 then 1/2
  else seq (n - 1) * 3 / (seq (n - 1) + 3)

theorem seq_general_term : ∀ n : ℕ, seq (n + 1) = 3 / (n + 6) :=
by
  intro n
  induction n with
  | zero => sorry
  | succ k ih => sorry

end seq_general_term_l216_216154


namespace largest_is_C_l216_216494

def A : ℝ := 0.978
def B : ℝ := 0.9719
def C : ℝ := 0.9781
def D : ℝ := 0.917
def E : ℝ := 0.9189

theorem largest_is_C : 
  (C > A) ∧ 
  (C > B) ∧ 
  (C > D) ∧ 
  (C > E) := by
  sorry

end largest_is_C_l216_216494


namespace reversed_digits_sum_l216_216444

theorem reversed_digits_sum (a b n : ℕ) (x y : ℕ) (ha : a < 10) (hb : b < 10) 
(hx : x = 10 * a + b) (hy : y = 10 * b + a) (hsq : x^2 + y^2 = n^2) : 
  x + y + n = 264 :=
sorry

end reversed_digits_sum_l216_216444


namespace min_omega_value_l216_216423

theorem min_omega_value (ω : ℝ) (hω : ω > 0)
  (f : ℝ → ℝ)
  (hf_def : ∀ x, f x = Real.cos (ω * x - (Real.pi / 6))) :
  (∀ x, f x ≤ f (Real.pi / 4)) → ω = 2 / 3 :=
by
  sorry

end min_omega_value_l216_216423


namespace certain_number_l216_216431

theorem certain_number (p q x : ℝ) (h1 : 3 / p = x) (h2 : 3 / q = 15) (h3 : p - q = 0.3) : x = 6 :=
sorry

end certain_number_l216_216431


namespace minimum_color_bound_l216_216860

theorem minimum_color_bound (n : ℕ) (h : n > 0) :
  ∃ (χ : ℕ), (∀ (tournament : Fin n → Fin n → ℕ),
    (∀ u v w : Fin n, u ≠ v → v ≠ w → w ≠ u → tournament u v ≠ tournament v w) →
    ∀ c : ℕ, c ≥ log 2 n →
       χ = c) :=
sorry

end minimum_color_bound_l216_216860


namespace typist_original_salary_l216_216180

theorem typist_original_salary (x : ℝ) (h : (x * 1.10 * 0.95 = 4180)) : x = 4000 :=
by sorry

end typist_original_salary_l216_216180


namespace average_letters_per_day_l216_216367

theorem average_letters_per_day (letters_tuesday : Nat) (letters_wednesday : Nat) (total_days : Nat) 
  (h_tuesday : letters_tuesday = 7) (h_wednesday : letters_wednesday = 3) (h_days : total_days = 2) : 
  (letters_tuesday + letters_wednesday) / total_days = 5 :=
by 
  sorry

end average_letters_per_day_l216_216367


namespace calculate_expression_l216_216681

theorem calculate_expression : (Real.pi - 2023)^0 - |1 - Real.sqrt 2| + 2 * Real.cos (Real.pi / 4) - (1 / 2)⁻¹ = 0 :=
by
  sorry

end calculate_expression_l216_216681


namespace function_symmetry_origin_l216_216044

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x

theorem function_symmetry_origin : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end function_symmetry_origin_l216_216044


namespace angle_bw_vectors_correct_l216_216417

open Real

variables {a b : ℝ^3} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : ∥ a + b ∥ = ∥ a - 2 • b ∥)

noncomputable def angle_between_vectors : ℝ :=
  Real.arccos ((∥ b ∥) / (2 * ∥ a ∥))

theorem angle_bw_vectors_correct : 
  angle_between_vectors h1 h2 h3 = Real.arccos ((∥ b ∥) / (2 * ∥ a ∥)) :=
sorry

end angle_bw_vectors_correct_l216_216417


namespace unique_solution_f_l216_216026

theorem unique_solution_f (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + f y) ≥ f (f x + y))
  (h2 : f 0 = 0) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_l216_216026


namespace total_vehicle_wheels_in_parking_lot_l216_216855

def vehicles_wheels := (1 * 4) + (1 * 4) + (8 * 4) + (4 * 2) + (3 * 6) + (2 * 4) + (1 * 8) + (2 * 3)

theorem total_vehicle_wheels_in_parking_lot : vehicles_wheels = 88 :=
by {
    sorry
}

end total_vehicle_wheels_in_parking_lot_l216_216855


namespace students_contribution_l216_216217

theorem students_contribution (n x : ℕ) 
  (h₁ : ∃ (k : ℕ), k * 9 = 22725)
  (h₂ : n * x = k / 9)
  : (n = 5 ∧ x = 505) ∨ (n = 25 ∧ x = 101) :=
sorry

end students_contribution_l216_216217


namespace red_shirts_count_l216_216995

theorem red_shirts_count :
  ∀ (total blue_fraction green_fraction : ℕ),
    total = 60 →
    blue_fraction = total / 3 →
    green_fraction = total / 4 →
    (total - (blue_fraction + green_fraction)) = 25 :=
by
  intros total blue_fraction green_fraction h_total h_blue h_green
  rw [h_total, h_blue, h_green]
  norm_num
  sorry

end red_shirts_count_l216_216995


namespace Scarlett_adds_correct_amount_l216_216052

-- Define the problem with given conditions
def currentOilAmount : ℝ := 0.17
def desiredOilAmount : ℝ := 0.84

-- Prove that the amount of oil Scarlett needs to add is 0.67 cup
theorem Scarlett_adds_correct_amount : (desiredOilAmount - currentOilAmount) = 0.67 := by
  sorry

end Scarlett_adds_correct_amount_l216_216052


namespace rachel_brought_16_brownies_l216_216310

def total_brownies : ℕ := 40
def brownies_left_at_home : ℕ := 24

def brownies_brought_to_school : ℕ :=
  total_brownies - brownies_left_at_home

theorem rachel_brought_16_brownies :
  brownies_brought_to_school = 16 :=
by
  sorry

end rachel_brought_16_brownies_l216_216310


namespace turquoise_beads_count_l216_216211

-- Define the conditions
def num_beads_total : ℕ := 40
def num_amethyst : ℕ := 7
def num_amber : ℕ := 2 * num_amethyst

-- Define the main theorem to prove
theorem turquoise_beads_count :
  num_beads_total - (num_amethyst + num_amber) = 19 :=
by
  sorry

end turquoise_beads_count_l216_216211


namespace no_nat_nums_satisfying_l216_216595

theorem no_nat_nums_satisfying (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k :=
by
  sorry

end no_nat_nums_satisfying_l216_216595


namespace base_number_l216_216737

theorem base_number (a x : ℕ) (h1 : a ^ x - a ^ (x - 2) = 3 * 2 ^ 11) (h2 : x = 13) : a = 2 :=
by
  sorry

end base_number_l216_216737


namespace average_price_of_pig_l216_216946

theorem average_price_of_pig :
  ∀ (total_cost total_cost_hens total_cost_pigs : ℕ) (num_hens num_pigs avg_price_hen avg_price_pig : ℕ),
  num_hens = 10 →
  num_pigs = 3 →
  total_cost = 1200 →
  avg_price_hen = 30 →
  total_cost_hens = num_hens * avg_price_hen →
  total_cost_pigs = total_cost - total_cost_hens →
  avg_price_pig = total_cost_pigs / num_pigs →
  avg_price_pig = 300 :=
by
  intros total_cost total_cost_hens total_cost_pigs num_hens num_pigs avg_price_hen avg_price_pig h_num_hens h_num_pigs h_total_cost h_avg_price_hen h_total_cost_hens h_total_cost_pigs h_avg_price_pig
  sorry

end average_price_of_pig_l216_216946


namespace double_inequality_l216_216207

variable (a b c : ℝ)

theorem double_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a * b - b * c - c * a ∧ 
  a + b + c - a * b - b * c - c * a ≤ 1 / 2 * (1 + a^2 + b^2 + c^2) := 
sorry

end double_inequality_l216_216207


namespace circle_possible_values_l216_216192

theorem circle_possible_values (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 + a - 1 = 0 → -2 < a ∧ a < 2/3) := sorry

end circle_possible_values_l216_216192


namespace michael_passes_donovan_l216_216366

noncomputable def track_length : ℕ := 600
noncomputable def donovan_lap_time : ℕ := 45
noncomputable def michael_lap_time : ℕ := 40

theorem michael_passes_donovan :
  ∃ n : ℕ, michael_lap_time * n > donovan_lap_time * (n - 1) ∧ n = 9 :=
by
  sorry

end michael_passes_donovan_l216_216366


namespace beef_stew_duration_l216_216364

noncomputable def original_portions : ℕ := 14
noncomputable def your_portion : ℕ := 1
noncomputable def roommate_portion : ℕ := 3
noncomputable def guest_portion : ℕ := 4
noncomputable def total_daily_consumption : ℕ := your_portion + roommate_portion + guest_portion
noncomputable def days_stew_lasts : ℕ := original_portions / total_daily_consumption

theorem beef_stew_duration : days_stew_lasts = 2 :=
by
  sorry

end beef_stew_duration_l216_216364


namespace relationship_of_abc_l216_216123

theorem relationship_of_abc (a b c : ℝ) 
  (h1 : b + c = 6 - 4 * a + 3 * a^2) 
  (h2 : c - b = 4 - 4 * a + a^2) : 
  a < b ∧ b ≤ c := 
sorry

end relationship_of_abc_l216_216123


namespace chandler_tickets_total_cost_l216_216977

theorem chandler_tickets_total_cost :
  let movie_ticket_cost := 30
  let num_movie_tickets := 8
  let num_football_tickets := 5
  let num_concert_tickets := 3
  let num_theater_tickets := 4
  let theater_ticket_cost := 40
  let discount := 0.10
  let total_movie_cost := num_movie_tickets * movie_ticket_cost
  let football_ticket_cost := total_movie_cost / 2
  let total_football_cost := num_football_tickets * football_ticket_cost
  let concert_ticket_cost := football_ticket_cost - 10
  let total_concert_cost := num_concert_tickets * concert_ticket_cost
  let discounted_theater_ticket_cost := theater_ticket_cost * (1 - discount)
  let total_theater_cost := num_theater_tickets * discounted_theater_ticket_cost
  let total_cost := total_movie_cost + total_football_cost + total_concert_cost + total_theater_cost
  total_cost = 1314 := by
  sorry

end chandler_tickets_total_cost_l216_216977


namespace probability_exactly_three_correct_l216_216696

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l216_216696


namespace yellow_jelly_bean_probability_l216_216079

theorem yellow_jelly_bean_probability :
  ∀ (p_red p_orange p_green p_total p_yellow : ℝ),
    p_red = 0.15 →
    p_orange = 0.35 →
    p_green = 0.25 →
    p_total = 1 →
    p_red + p_orange + p_green + p_yellow = p_total →
    p_yellow = 0.25 :=
by
  intros p_red p_orange p_green p_total p_yellow h_red h_orange h_green h_total h_sum
  sorry

end yellow_jelly_bean_probability_l216_216079


namespace calculate_expression_l216_216072

theorem calculate_expression :
  107 * 107 + 93 * 93 = 20098 := by
  sorry

end calculate_expression_l216_216072


namespace tan_periodic_mod_l216_216250

theorem tan_periodic_mod (m : ℤ) (h1 : -180 < m) (h2 : m < 180) : 
  (m : ℤ) = 10 := by
  sorry

end tan_periodic_mod_l216_216250


namespace minimum_restoration_time_l216_216661

structure Handicraft :=
  (shaping: ℕ)
  (painting: ℕ)

def handicraft_A : Handicraft := ⟨9, 15⟩
def handicraft_B : Handicraft := ⟨16, 8⟩
def handicraft_C : Handicraft := ⟨10, 14⟩

def total_restoration_time (order: List Handicraft) : ℕ :=
  let rec aux (remaining: List Handicraft) (A_time: ℕ) (B_time: ℕ) (acc: ℕ) : ℕ :=
    match remaining with
    | [] => acc
    | h :: t =>
      let A_next := A_time + h.shaping
      let B_next := max A_next B_time + h.painting
      aux t A_next B_next B_next
  aux order 0 0 0

theorem minimum_restoration_time :
  total_restoration_time [handicraft_A, handicraft_C, handicraft_B] = 46 :=
by
  simp [total_restoration_time, handicraft_A, handicraft_B, handicraft_C]
  sorry

end minimum_restoration_time_l216_216661


namespace jo_reading_hours_l216_216894

theorem jo_reading_hours :
  ∀ (total_pages current_page previous_page pages_per_hour remaining_pages : ℕ),
    total_pages = 210 →
    current_page = 90 →
    previous_page = 60 →
    pages_per_hour = current_page - previous_page →
    remaining_pages = total_pages - current_page →
    remaining_pages / pages_per_hour = 4 :=
by
  intros total_pages current_page previous_page pages_per_hour remaining_pages
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  dsimp at *
  sorry

end jo_reading_hours_l216_216894


namespace final_jacket_price_is_correct_l216_216675

-- Define the initial price, the discounts, and the tax rate
def initial_price : ℝ := 120
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def sales_tax : ℝ := 0.05

-- Calculate the final price using the given conditions
noncomputable def price_after_first_discount := initial_price * (1 - first_discount)
noncomputable def price_after_second_discount := price_after_first_discount * (1 - second_discount)
noncomputable def final_price := price_after_second_discount * (1 + sales_tax)

-- The theorem to prove
theorem final_jacket_price_is_correct : final_price = 75.60 := by
  -- The proof is omitted
  sorry

end final_jacket_price_is_correct_l216_216675


namespace inequality_problem_l216_216175

-- Define a and the condition that expresses the given problem as an inequality
variable (a : ℝ)

-- The inequality to prove
theorem inequality_problem : a - 5 > 2 * a := sorry

end inequality_problem_l216_216175


namespace total_trips_correct_l216_216934

-- Define Timothy's movie trips in 2009
def timothy_2009_trips : ℕ := 24

-- Define Timothy's movie trips in 2010
def timothy_2010_trips : ℕ := timothy_2009_trips + 7

-- Define Theresa's movie trips in 2009
def theresa_2009_trips : ℕ := timothy_2009_trips / 2

-- Define Theresa's movie trips in 2010
def theresa_2010_trips : ℕ := timothy_2010_trips * 2

-- Define the total number of trips for Timothy and Theresa in 2009 and 2010
def total_trips : ℕ := (timothy_2009_trips + timothy_2010_trips) + (theresa_2009_trips + theresa_2010_trips)

-- Prove the total number of trips is 129
theorem total_trips_correct : total_trips = 129 :=
by
  sorry

end total_trips_correct_l216_216934


namespace files_deleted_l216_216851

theorem files_deleted 
  (orig_files : ℕ) (final_files : ℕ) (deleted_files : ℕ) 
  (h_orig : orig_files = 24) 
  (h_final : final_files = 21) : 
  deleted_files = orig_files - final_files :=
by
  rw [h_orig, h_final]
  sorry

end files_deleted_l216_216851


namespace probability_three_correct_packages_l216_216703

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l216_216703


namespace find_value_l216_216264

-- Define the theorem with the given conditions and the expected result
theorem find_value (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + a * c^2 + a + b + c = 2 * (a * b + b * c + a * c)) :
  c^2017 / (a^2016 + b^2018) = 1 / 2 :=
sorry

end find_value_l216_216264


namespace incorrect_statement_C_l216_216557

theorem incorrect_statement_C 
  (x y : ℝ)
  (n : ℕ)
  (data : Fin n → (ℝ × ℝ))
  (h : ∀ (i : Fin n), (x, y) = data i)
  (reg_eq : ∀ (x : ℝ), 0.85 * x - 85.71 = y) :
  ¬ (forall (x : ℝ), x = 160 → ∀ (y : ℝ), y = 50.29) := 
sorry

end incorrect_statement_C_l216_216557


namespace distance_from_origin_to_point_P_l216_216747

def origin : ℝ × ℝ := (0, 0)
def point_P : ℝ × ℝ := (12, -5)

theorem distance_from_origin_to_point_P :
  (Real.sqrt ((12 - 0)^2 + (-5 - 0)^2)) = 13 := by
  simp [Real.sqrt, *]
  sorry

end distance_from_origin_to_point_P_l216_216747


namespace min_bills_required_l216_216354

-- Conditions
def ten_dollar_bills := 13
def five_dollar_bills := 11
def one_dollar_bills := 17
def total_amount := 128

-- Prove that Tim can pay exactly $128 with the minimum number of bills being 16
theorem min_bills_required : (∃ ten five one : ℕ, 
    ten ≤ ten_dollar_bills ∧
    five ≤ five_dollar_bills ∧
    one ≤ one_dollar_bills ∧
    ten * 10 + five * 5 + one = total_amount ∧
    ten + five + one = 16) :=
by
  -- We will skip the proof for now
  sorry

end min_bills_required_l216_216354


namespace emmalyn_earnings_l216_216396

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l216_216396


namespace solve_expression_hundreds_digit_l216_216857

def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

def div_mod (a b m : ℕ) : ℕ :=
  (a / b) % m

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem solve_expression_hundreds_digit :
  hundreds_digit (div_mod (factorial 17) 5 1000 - div_mod (factorial 10) 2 1000) = 8 :=
by
  sorry

end solve_expression_hundreds_digit_l216_216857


namespace carolyn_fewer_stickers_l216_216972

theorem carolyn_fewer_stickers :
  let belle_stickers := 97
  let carolyn_stickers := 79
  carolyn_stickers < belle_stickers →
  belle_stickers - carolyn_stickers = 18 :=
by
  intros
  sorry

end carolyn_fewer_stickers_l216_216972


namespace completion_time_workshop_3_l216_216040

-- Define the times for workshops
def time_in_workshop_3 : ℝ := 8
def time_in_workshop_1 : ℝ := time_in_workshop_3 + 10
def time_in_workshop_2 : ℝ := (time_in_workshop_3 + 10) - 3.6

-- Define the combined work equation
def combined_work_eq := (1 / time_in_workshop_1) + (1 / time_in_workshop_2) = (1 / time_in_workshop_3)

-- Final theorem statement
theorem completion_time_workshop_3 (h : combined_work_eq) : time_in_workshop_3 - 7 = 1 :=
by
  sorry

end completion_time_workshop_3_l216_216040


namespace sequence_b_l216_216161

theorem sequence_b (b : ℕ → ℕ) 
  (h1 : b 1 = 2) 
  (h2 : ∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) : 
  b 10 = 110 :=
sorry

end sequence_b_l216_216161


namespace find_number_l216_216324

variable (x : ℕ)

theorem find_number (h : (10 + 20 + x) / 3 = ((10 + 40 + 25) / 3) + 5) : x = 60 :=
by
  sorry

end find_number_l216_216324


namespace smallest_sum_is_381_l216_216938

def is_valid_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def uses_digits_once (n m : ℕ) : Prop :=
  (∀ d ∈ [1, 2, 3, 4, 5, 6], (d ∈ n.digits 10 ∨ d ∈ m.digits 10)) ∧
  (∀ d, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ m.digits 10 → d ∈ [1, 2, 3, 4, 5, 6])

theorem smallest_sum_is_381 :
  ∃ (n m : ℕ), is_valid_3_digit_number n ∧ is_valid_3_digit_number m ∧
  uses_digits_once n m ∧ n + m = 381 :=
sorry

end smallest_sum_is_381_l216_216938


namespace exists_perfect_square_subtraction_l216_216913

theorem exists_perfect_square_subtraction {k : ℕ} (hk : k > 0) : 
  ∃ (n : ℕ), n > 0 ∧ ∃ m : ℕ, n * 2^k - 7 = m^2 := 
  sorry

end exists_perfect_square_subtraction_l216_216913


namespace least_three_digit_divisible_by_2_3_5_7_l216_216809

theorem least_three_digit_divisible_by_2_3_5_7 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ k, 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → n ≤ k) ∧
  (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) ∧ n = 210 :=
by sorry

end least_three_digit_divisible_by_2_3_5_7_l216_216809


namespace No_response_percentage_l216_216456

theorem No_response_percentage (total_guests : ℕ) (yes_percentage : ℕ) (non_respondents : ℕ) (yes_guests := total_guests * yes_percentage / 100) (no_guests := total_guests - yes_guests - non_respondents) (no_percentage := no_guests * 100 / total_guests) :
  total_guests = 200 → yes_percentage = 83 → non_respondents = 16 → no_percentage = 9 :=
by
  sorry

end No_response_percentage_l216_216456


namespace chandra_pairings_l216_216095

theorem chandra_pairings : 
  let bowls := 5
  let glasses := 6
  (bowls * glasses) = 30 :=
by
  sorry

end chandra_pairings_l216_216095


namespace first_student_time_l216_216947

theorem first_student_time
  (n : ℕ)
  (avg_last_three avg_all : ℕ)
  (h_n : n = 4)
  (h_avg_last_three : avg_last_three = 35)
  (h_avg_all : avg_all = 30) :
  let total_time_all := n * avg_all in
  let total_time_last_three := 3 * avg_last_three in
  (total_time_all - total_time_last_three) = 15 :=
by
  let total_time_all := 4 * 30
  let total_time_last_three := 3 * 35
  show total_time_all - total_time_last_three = 15
  sorry

end first_student_time_l216_216947


namespace range_m_l216_216499

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_m (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ (-1 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_m_l216_216499


namespace intersection_of_M_and_N_l216_216902

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l216_216902


namespace number_of_scoops_l216_216304

/-- Pierre gets 3 scoops of ice cream given the conditions described -/
theorem number_of_scoops (P : ℕ) (cost_per_scoop total_bill : ℝ) (mom_scoops : ℕ)
  (h1 : cost_per_scoop = 2) 
  (h2 : mom_scoops = 4) 
  (h3 : total_bill = 14) 
  (h4 : cost_per_scoop * P + cost_per_scoop * mom_scoops = total_bill) :
  P = 3 :=
by
  sorry

end number_of_scoops_l216_216304


namespace change_positions_of_three_out_of_eight_l216_216372

theorem change_positions_of_three_out_of_eight :
  (Nat.choose 8 3) * (Nat.factorial 3) = (Nat.choose 8 3) * 6 :=
by
  sorry

end change_positions_of_three_out_of_eight_l216_216372


namespace everything_used_as_money_l216_216490

theorem everything_used_as_money :
  (used_as_money gold) ∧
  (used_as_money stones) ∧
  (used_as_money horses) ∧
  (used_as_money dried_fish) ∧
  (used_as_money mollusk_shells) →
  (∀ x ∈ {gold, stones, horses, dried_fish, mollusk_shells}, used_as_money x) :=
by
  intro h
  cases h with
  | intro h_gold h_stones =>
    cases h_stones with
    | intro h_stones h_horses =>
      cases h_horses with
      | intro h_horses h_dried_fish =>
        cases h_dried_fish with
        | intro h_dried_fish h_mollusk_shells =>
          intro x h_x
          cases Set.mem_def.mpr h_x with
          | or.inl h => exact h_gold
          | or.inr h_x1 => cases Set.mem_def.mpr h_x1 with
            | or.inl h => exact h_stones
            | or.inr h_x2 => cases Set.mem_def.mpr h_x2 with
              | or.inl h => exact h_horses
              | or.inr h_x3 => cases Set.mem_def.mpr h_x3 with
                | or.inl h => exact h_dried_fish
                | or.inr h_x4 => exact h_mollusk_shells

end everything_used_as_money_l216_216490


namespace smallest_b_factors_l216_216528

theorem smallest_b_factors (b : ℕ) (m n : ℤ) (h : m * n = 2023 ∧ m + n = b) : b = 136 :=
sorry

end smallest_b_factors_l216_216528


namespace factorial_divide_l216_216391

theorem factorial_divide {n m k : ℕ} (h : n = 11) (h1 : m = 8) (h2 : k = 3) : (nat.factorial n / (nat.factorial m * nat.factorial k)) = 165 := 
by {
  rw [h, h1, h2],
  sorry
}

end factorial_divide_l216_216391


namespace harriet_current_age_l216_216576

theorem harriet_current_age (peter_age harriet_age : ℕ) (mother_age : ℕ := 60) (h₁ : peter_age = mother_age / 2) 
  (h₂ : peter_age + 4 = 2 * (harriet_age + 4)) : harriet_age = 13 :=
by
  sorry

end harriet_current_age_l216_216576


namespace chocolate_milk_container_size_l216_216425

/-- Holly's chocolate milk consumption conditions and container size -/
theorem chocolate_milk_container_size
  (morning_initial: ℝ)  -- Initial amount in the morning
  (morning_drink: ℝ)    -- Amount drank in the morning with breakfast
  (lunch_drink: ℝ)      -- Amount drank at lunch
  (dinner_drink: ℝ)     -- Amount drank with dinner
  (end_of_day: ℝ)       -- Amount she ends the day with
  (lunch_container_size: ℝ) -- Size of the container bought at lunch
  (C: ℝ)                -- Container size she bought at lunch
  (h_initial: morning_initial = 16)
  (h_morning_drink: morning_drink = 8)
  (h_lunch_drink: lunch_drink = 8)
  (h_dinner_drink: dinner_drink = 8)
  (h_end_of_day: end_of_day = 56) :
  (morning_initial - morning_drink) + C - lunch_drink - dinner_drink = end_of_day → 
  lunch_container_size = 64 :=
by
  sorry

end chocolate_milk_container_size_l216_216425


namespace nth_sum_eq_square_l216_216553

theorem nth_sum_eq_square (n : ℕ) : 
  (∑ k in finset.range (3 * n - n + 1), k + n) = (2 * n - 1) ^ 2 :=
by
  sorry

end nth_sum_eq_square_l216_216553


namespace permutations_with_exactly_four_not_in_original_positions_l216_216252

theorem permutations_with_exactly_four_not_in_original_positions :
  ∃ (S : Finset (Set (Fin 8))) (card_S : S.card = 1) (T : Finset (Set (Fin 8))) (card_T : T.card = 70),
  (∃ (d : Derangements (Fin 8)) (hd : d.card = 9), S.card * T.card * d.card = 630) :=
sorry

end permutations_with_exactly_four_not_in_original_positions_l216_216252


namespace magic_box_problem_l216_216588

theorem magic_box_problem (m : ℝ) :
  (m^2 - 2*m - 1 = 2) → (m = 3 ∨ m = -1) :=
by
  intro h
  sorry

end magic_box_problem_l216_216588


namespace find_x_when_y_is_20_l216_216127

variable (x y k : ℝ)

axiom constant_ratio : (5 * 4 - 6) / (5 + 20) = k

theorem find_x_when_y_is_20 (h : (5 * x - 6) / (y + 20) = k) (hy : y = 20) : x = 5.68 := by
  sorry

end find_x_when_y_is_20_l216_216127


namespace group_is_abelian_l216_216443

variable {G : Type} [Group G]
variable (e : G)
variable (h : ∀ x : G, x * x = e)

theorem group_is_abelian (a b : G) : a * b = b * a :=
sorry

end group_is_abelian_l216_216443


namespace trajectory_eq_ellipse_l216_216551

theorem trajectory_eq_ellipse :
  (∀ M : ℝ × ℝ, (∀ r : ℝ, (M.1 - 4)^2 + M.2^2 = r^2 ∧ (M.1 + 4)^2 + M.2^2 = (10 - r)^2) → false) →
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) :=
by
  sorry

end trajectory_eq_ellipse_l216_216551


namespace angle_B_in_right_triangle_in_degrees_l216_216437

def angleSum (A B C: ℝ) : Prop := A + B + C = 180

theorem angle_B_in_right_triangle_in_degrees (A B C : ℝ) (h1 : C = 90) (h2 : A = 35.5) (h3 : angleSum A B C) : B = 54.5 := 
by
  sorry

end angle_B_in_right_triangle_in_degrees_l216_216437


namespace jo_reading_time_l216_216893

structure Book :=
  (totalPages : Nat)
  (currentPage : Nat)
  (pageOneHourAgo : Nat)

def readingTime (b : Book) : Nat :=
  let pagesRead := b.currentPage - b.pageOneHourAgo
  let pagesLeft := b.totalPages - b.currentPage
  pagesLeft / pagesRead

theorem jo_reading_time :
  ∀ (b : Book), b.totalPages = 210 → b.currentPage = 90 → b.pageOneHourAgo = 60 → readingTime b = 4 :=
by
  intro b h1 h2 h3
  sorry

end jo_reading_time_l216_216893


namespace Farrah_total_match_sticks_l216_216524

def boxes := 4
def matchboxes_per_box := 20
def sticks_per_matchbox := 300

def total_matchboxes : Nat :=
  boxes * matchboxes_per_box

def total_match_sticks : Nat :=
  total_matchboxes * sticks_per_matchbox

theorem Farrah_total_match_sticks : total_match_sticks = 24000 := sorry

end Farrah_total_match_sticks_l216_216524


namespace initial_birds_count_l216_216818

theorem initial_birds_count (B : ℕ) (h1 : 6 = B + 3 + 1) : B = 2 :=
by
  -- Placeholder for the proof, we are not required to provide it here.
  sorry

end initial_birds_count_l216_216818


namespace range_of_m_l216_216741

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = -3) (h3 : x + y > 0) : m > 2 :=
by
  sorry

end range_of_m_l216_216741


namespace min_restoration_time_l216_216659

/-- Prove the minimum time required to complete the restoration work of three handicrafts. -/

def shaping_time_A : Nat := 9
def shaping_time_B : Nat := 16
def shaping_time_C : Nat := 10

def painting_time_A : Nat := 15
def painting_time_B : Nat := 8
def painting_time_C : Nat := 14

theorem min_restoration_time : 
  (shaping_time_A + painting_time_A + painting_time_C + painting_time_B) = 46 := by
  sorry

end min_restoration_time_l216_216659


namespace assistant_increases_output_by_100_percent_l216_216017

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l216_216017


namespace exists_perpendicular_line_l216_216630

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure DirectionVector :=
  (dx : ℝ)
  (dy : ℝ)
  (dz : ℝ)

noncomputable def parametric_line_through_point 
  (P : Point3D) 
  (d : DirectionVector) : Prop :=
  ∀ t : ℝ, ∃ x y z : ℝ, 
  x = P.x + d.dx * t ∧
  y = P.y + d.dy * t ∧
  z = P.z + d.dz * t

theorem exists_perpendicular_line : 
  ∃ d : DirectionVector, 
    (d.dx * 2 + d.dy * 3 - d.dz = 0) ∧ 
    (d.dx * 4 - d.dy * -1 + d.dz * 3 = 0) ∧ 
    parametric_line_through_point 
      ⟨3, -2, 1⟩ d :=
  sorry

end exists_perpendicular_line_l216_216630


namespace problem_statement_l216_216799

theorem problem_statement (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 4 / 5) : y - x = 500 / 9 := 
by
  sorry

end problem_statement_l216_216799


namespace sum_six_consecutive_integers_l216_216777

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l216_216777


namespace hyperbola_slope_of_asymptote_positive_value_l216_216468

noncomputable def hyperbola_slope_of_asymptote (x y : ℝ) : ℝ :=
  if h : (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4)
  then (Real.sqrt 5) / 2
  else 0

-- Statement of the mathematically equivalent proof problem
theorem hyperbola_slope_of_asymptote_positive_value :
  ∃ x y : ℝ, (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) ∧
  hyperbola_slope_of_asymptote x y = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_slope_of_asymptote_positive_value_l216_216468


namespace prism_volume_l216_216920

/-- The volume of a rectangular prism given the areas of three of its faces. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 :=
by
  sorry

end prism_volume_l216_216920


namespace min_number_of_squares_l216_216053

theorem min_number_of_squares (length width : ℕ) (h_length : length = 10) (h_width : width = 9) : 
  ∃ n, n = 10 :=
by
  sorry

end min_number_of_squares_l216_216053


namespace remainder_of_6_pow_1234_mod_13_l216_216984

theorem remainder_of_6_pow_1234_mod_13 : 6 ^ 1234 % 13 = 10 := 
by 
  sorry

end remainder_of_6_pow_1234_mod_13_l216_216984


namespace age_ratio_4_years_hence_4_years_ago_l216_216612

-- Definitions based on the conditions
def current_age_ratio (A B : ℕ) := 5 * B = 3 * A
def age_ratio_4_years_ago_4_years_hence (A B : ℕ) := A - 4 = B + 4

-- The main theorem to prove
theorem age_ratio_4_years_hence_4_years_ago (A B : ℕ) 
  (h1 : current_age_ratio A B) 
  (h2 : age_ratio_4_years_ago_4_years_hence A B) : 
  A + 4 = 3 * (B - 4) := 
sorry

end age_ratio_4_years_hence_4_years_ago_l216_216612


namespace May4th_Sunday_l216_216218

theorem May4th_Sunday (x : ℕ) (h_sum : x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 80) : 
  (4 % 7) = 0 :=
by
  sorry

end May4th_Sunday_l216_216218


namespace probability_three_correct_deliveries_l216_216711

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l216_216711


namespace average_snack_sales_per_ticket_l216_216348

theorem average_snack_sales_per_ticket :
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  (total_sales / movie_tickets = 2.79) :=
by
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  show total_sales / movie_tickets = 2.79
  sorry

end average_snack_sales_per_ticket_l216_216348


namespace area_of_square_efgh_proof_l216_216950

noncomputable def area_of_square_efgh : ℝ :=
  let original_square_side_length := 3
  let radius_of_circles := (3 * Real.sqrt 2) / 2
  let efgh_side_length := original_square_side_length + 2 * radius_of_circles 
  efgh_side_length ^ 2

theorem area_of_square_efgh_proof :
  area_of_square_efgh = 27 + 18 * Real.sqrt 2 :=
by
  sorry

end area_of_square_efgh_proof_l216_216950


namespace simon_students_l216_216604

theorem simon_students (S L : ℕ) (h1 : S = 4 * L) (h2 : S + L = 2500) : S = 2000 :=
by {
  sorry
}

end simon_students_l216_216604


namespace functional_equation_continuous_function_l216_216375

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_continuous_function (f : ℝ → ℝ) (x₀ : ℝ) (h1 : Continuous f) (h2 : f x₀ ≠ 0) 
  (h3 : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x : ℝ, f x = a ^ x := 
by
  sorry

end functional_equation_continuous_function_l216_216375


namespace neg_neg_eq_pos_l216_216647

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l216_216647


namespace scrap_cookie_radius_is_correct_l216_216451

noncomputable def radius_of_scrap_cookie (large_radius small_radius : ℝ) (number_of_cookies : ℕ) : ℝ :=
  have large_area : ℝ := Real.pi * large_radius^2
  have small_area : ℝ := Real.pi * small_radius^2
  have total_small_area : ℝ := small_area * number_of_cookies
  have scrap_area : ℝ := large_area - total_small_area
  Real.sqrt (scrap_area / Real.pi)

theorem scrap_cookie_radius_is_correct :
  radius_of_scrap_cookie 8 2 9 = 2 * Real.sqrt 7 :=
sorry

end scrap_cookie_radius_is_correct_l216_216451


namespace find_de_l216_216438

def magic_square (f : ℕ × ℕ → ℕ) : Prop :=
  (f (0, 0) = 30) ∧ (f (0, 1) = 20) ∧ (f (0, 2) = f (0, 2)) ∧
  (f (1, 0) = f (1, 0)) ∧ (f (1, 1) = f (1, 1)) ∧ (f (1, 2) = f (1, 2)) ∧
  (f (2, 0) = 24) ∧ (f (2, 1) = 32) ∧ (f (2, 2) = f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (1, 0) + f (1, 1) + f (1, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (2, 0) + f (2, 1) + f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (0, 0) + f (1, 0) + f (2, 0)) ∧
  (f (0, 0) + f (1, 1) + f (2, 2) = f (0, 2) + f (1, 1) + f (2, 0)) 

theorem find_de (f : ℕ × ℕ → ℕ) (h : magic_square f) : 
  (f (1, 0) + f (1, 1) = 54) :=
sorry

end find_de_l216_216438


namespace probability_of_three_correct_packages_l216_216707

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l216_216707


namespace polynomial_real_roots_l216_216692

theorem polynomial_real_roots :
  (∃ x : ℝ, x^4 - 3*x^3 - 2*x^2 + 6*x + 9 = 0) ↔ (x = 1 ∨ x = 3) := 
by
  sorry

end polynomial_real_roots_l216_216692


namespace how_many_buckets_did_Eden_carry_l216_216856

variable (E : ℕ) -- Natural number representing buckets Eden carried
variable (M : ℕ) -- Natural number representing buckets Mary carried
variable (I : ℕ) -- Natural number representing buckets Iris carried

-- Conditions based on the problem
axiom Mary_Carry_More : M = E + 3
axiom Iris_Carry_Less : I = M - 1
axiom Total_Buckets : E + M + I = 34

theorem how_many_buckets_did_Eden_carry (h1 : M = E + 3) (h2 : I = M - 1) (h3 : E + M + I = 34) :
  E = 29 / 3 := by
  sorry

end how_many_buckets_did_Eden_carry_l216_216856


namespace index_card_area_l216_216917

theorem index_card_area (a b : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : (a - 2) * b = 21) : (a * (b - 1)) = 30 := by
  sorry

end index_card_area_l216_216917


namespace value_of_a_l216_216240

def star (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem value_of_a (a : ℝ) (h : star a 3 = 15) : a = 11 := 
by
  sorry

end value_of_a_l216_216240


namespace percent_increase_output_l216_216011

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l216_216011


namespace neg_neg_eq_pos_l216_216649

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l216_216649


namespace g_f2_minus_f_g2_eq_zero_l216_216760

def f (x : ℝ) : ℝ := x^2 + 3 * x + 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem g_f2_minus_f_g2_eq_zero : g (f 2) - f (g 2) = 0 := by
  sorry

end g_f2_minus_f_g2_eq_zero_l216_216760


namespace systematic_sampling_interval_l216_216185

-- Definitions based on the conditions in part a)
def total_students : ℕ := 1500
def sample_size : ℕ := 30

-- The goal is to prove that the interval k in systematic sampling equals 50
theorem systematic_sampling_interval :
  (total_students / sample_size = 50) :=
by
  sorry

end systematic_sampling_interval_l216_216185


namespace lines_passing_through_neg1_0_l216_216786

theorem lines_passing_through_neg1_0 (k : ℝ) :
  ∀ x y : ℝ, (y = k * (x + 1)) ↔ (x = -1 → y = 0 ∧ k ≠ 0) :=
by
  sorry

end lines_passing_through_neg1_0_l216_216786


namespace arithmetic_seq_50th_term_l216_216685

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_50th_term : 
  arithmetic_seq_nth_term 3 7 50 = 346 :=
by
  -- Intentionally left as sorry
  sorry

end arithmetic_seq_50th_term_l216_216685


namespace mia_has_largest_final_value_l216_216102

def daniel_final : ℕ := (12 * 2 - 3 + 5)
def mia_final : ℕ := ((15 - 2) * 2 + 3)
def carlos_final : ℕ := (13 * 2 - 4 + 6)

theorem mia_has_largest_final_value : mia_final > daniel_final ∧ mia_final > carlos_final := by
  sorry

end mia_has_largest_final_value_l216_216102


namespace expected_value_is_correct_l216_216313

noncomputable def expected_value_max_two_rolls : ℝ :=
  let p_max_1 := (1/6) * (1/6)
  let p_max_2 := (2/6) * (2/6) - (1/6) * (1/6)
  let p_max_3 := (3/6) * (3/6) - (2/6) * (2/6)
  let p_max_4 := (4/6) * (4/6) - (3/6) * (3/6)
  let p_max_5 := (5/6) * (5/6) - (4/6) * (4/6)
  let p_max_6 := 1 - (5/6) * (5/6)
  1 * p_max_1 + 2 * p_max_2 + 3 * p_max_3 + 4 * p_max_4 + 5 * p_max_5 + 6 * p_max_6

theorem expected_value_is_correct :
  expected_value_max_two_rolls = 4.5 :=
sorry

end expected_value_is_correct_l216_216313


namespace math_problem_l216_216877

open Nat

-- Given conditions
def S (n : ℕ) : ℕ := n * (n + 1)

-- Definitions for the terms a_n, b_n, c_n, and the sum T_n
def a_n (n : ℕ) (h : n ≠ 0) : ℕ := if n = 1 then 2 else 2 * n
def b_n (n : ℕ) (h : n ≠ 0) : ℕ := 2 * (3^n + 1)
def c_n (n : ℕ) (h : n ≠ 0) : ℕ := a_n n h * b_n n h / 4
def T (n : ℕ) (h : 0 < n) : ℕ := 
  (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2

-- Main theorem to establish the solution
theorem math_problem (n : ℕ) (h : n ≠ 0) : 
  S n = n * (n + 1) →
  a_n n h = 2 * n ∧ 
  b_n n h = 2 * (3^n + 1) ∧ 
  T n (Nat.pos_of_ne_zero h) = (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2 := 
by
  intros hS
  sorry

end math_problem_l216_216877


namespace mat_pow_four_eq_l216_216096

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, -2; 1, 1]

def mat_fourth_power : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-14, -6; 3, -17]

theorem mat_pow_four_eq :
  mat ^ 4 = mat_fourth_power :=
by
  sorry

end mat_pow_four_eq_l216_216096


namespace gcd_888_1147_l216_216610

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l216_216610


namespace students_tried_out_l216_216617

theorem students_tried_out (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ)
  (h1 : not_picked = 36) (h2 : groups = 4) (h3 : students_per_group = 7) :
  not_picked + groups * students_per_group = 64 :=
by
  sorry

end students_tried_out_l216_216617


namespace cost_per_minute_l216_216530

theorem cost_per_minute (monthly_fee cost total_bill : ℝ) (minutes : ℕ) :
  monthly_fee = 2 ∧ total_bill = 23.36 ∧ minutes = 178 → 
  cost = (total_bill - monthly_fee) / minutes → 
  cost = 0.12 :=
by
  intros h1 h2
  sorry

end cost_per_minute_l216_216530


namespace problem1_problem2_problem3_l216_216388

theorem problem1 : (-3) - (-5) - 6 + (-4) = -8 := by sorry

theorem problem2 : ((1 / 9) + (1 / 6) - (1 / 2)) / (-1 / 18) = 4 := by sorry

theorem problem3 : -1^4 + abs (3 - 6) - 2 * (-2) ^ 2 = -6 := by sorry

end problem1_problem2_problem3_l216_216388


namespace point_in_third_quadrant_l216_216739

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 1 + 2 * m < 0) : m < -1 / 2 := 
by 
  sorry

end point_in_third_quadrant_l216_216739


namespace smallest_n_which_contains_643_l216_216327

theorem smallest_n_which_contains_643 (m n : ℕ) (h_rel_prime : Nat.coprime m n) (h_cond : m < n) (h_contains_643 : ∃ a b c, (6::4::3::a) ∈ (n::b::c)) :
  n = 358 :=
sorry

end smallest_n_which_contains_643_l216_216327


namespace not_perfect_square_4n_squared_plus_4n_plus_4_l216_216493

theorem not_perfect_square_4n_squared_plus_4n_plus_4 :
  ¬ ∃ m n : ℕ, m^2 = 4 * n^2 + 4 * n + 4 := 
by
  sorry

end not_perfect_square_4n_squared_plus_4n_plus_4_l216_216493


namespace mean_home_runs_correct_l216_216338

def mean_home_runs (players: List ℕ) (home_runs: List ℕ) : ℚ :=
  let total_runs := (List.zipWith (· * ·) players home_runs).sum
  let total_players := players.sum
  total_runs / total_players

theorem mean_home_runs_correct :
  mean_home_runs [6, 4, 3, 1, 1, 1] [6, 7, 8, 10, 11, 12] = 121 / 16 :=
by
  -- The proof should go here
  sorry

end mean_home_runs_correct_l216_216338


namespace fraction_equality_l216_216520

theorem fraction_equality : 
  (3 ^ 8 + 3 ^ 6) / (3 ^ 8 - 3 ^ 6) = 5 / 4 :=
by
  -- Expression rewrite and manipulation inside parenthesis can be ommited
  sorry

end fraction_equality_l216_216520


namespace ellipse_foci_y_axis_l216_216042

theorem ellipse_foci_y_axis (k : ℝ) :
  (∃ a b : ℝ, a = 15 - k ∧ b = k - 9 ∧ a > 0 ∧ b > 0) ↔ (12 < k ∧ k < 15) :=
by
  sorry

end ellipse_foci_y_axis_l216_216042


namespace number_of_whole_numbers_between_cubicroots_l216_216138

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l216_216138


namespace square_field_side_length_l216_216952

theorem square_field_side_length (t : ℕ) (v : ℕ) 
  (run_time : t = 56) 
  (run_speed : v = 9) : 
  ∃ l : ℝ, l = 35 := 
sorry

end square_field_side_length_l216_216952


namespace b_days_to_complete_work_l216_216813

theorem b_days_to_complete_work (x : ℕ) 
  (A : ℝ := 1 / 30) 
  (B : ℝ := 1 / x) 
  (C : ℝ := 1 / 40)
  (work_eq : 8 * (A + B + C) + 4 * (A + B) = 1) 
  (x_ne_0 : x ≠ 0) : 
  x = 30 := 
by
  sorry

end b_days_to_complete_work_l216_216813


namespace length_of_train_l216_216200

theorem length_of_train 
  (L V : ℝ) 
  (h1 : L = V * 8) 
  (h2 : L + 279 = V * 20) : 
  L = 186 :=
by
  -- solve using the given conditions
  sorry

end length_of_train_l216_216200


namespace polynomial_average_k_l216_216125

theorem polynomial_average_k (h : ∀ x : ℕ, x * (36 / x) = 36 → (x + (36 / x) = 37 ∨ x + (36 / x) = 20 ∨ x + (36 / x) = 15 ∨ x + (36 / x) = 13 ∨ x + (36 / x) = 12)) :
  (37 + 20 + 15 + 13 + 12) / 5 = 19.4 := by
sorry

end polynomial_average_k_l216_216125


namespace increase_in_output_with_assistant_l216_216007

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l216_216007


namespace circle_condition_l216_216570

theorem circle_condition (k : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + 5 * k = 0) ↔ k < 1 := 
sorry

end circle_condition_l216_216570


namespace root_exponent_equiv_l216_216176

theorem root_exponent_equiv :
  (7 ^ (1 / 2)) / (7 ^ (1 / 4)) = 7 ^ (1 / 4) := by
  sorry

end root_exponent_equiv_l216_216176


namespace product_of_ninth_and_tenth_l216_216436

def scores_first_8 := [7, 4, 3, 6, 8, 3, 1, 5]
def total_points_first_8 := scores_first_8.sum

def condition1 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  ninth_game_points < 10 ∧ tenth_game_points < 10

def condition2 (ninth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points) % 9 = 0

def condition3 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points + tenth_game_points) % 10 = 0

theorem product_of_ninth_and_tenth (ninth_game_points : ℕ) (tenth_game_points : ℕ) 
  (h1 : condition1 ninth_game_points tenth_game_points)
  (h2 : condition2 ninth_game_points)
  (h3 : condition3 ninth_game_points tenth_game_points) : 
  ninth_game_points * tenth_game_points = 40 :=
sorry

end product_of_ninth_and_tenth_l216_216436


namespace greatest_value_of_b_l216_216690

theorem greatest_value_of_b : ∃ b, (∀ a, (-a^2 + 7 * a - 10 ≥ 0) → (a ≤ b)) ∧ b = 5 :=
by
  sorry

end greatest_value_of_b_l216_216690


namespace cannot_determine_red_marbles_l216_216155

variable (Jason_blue : ℕ) (Tom_blue : ℕ) (Total_blue : ℕ)

-- Conditions
axiom Jason_has_44_blue : Jason_blue = 44
axiom Tom_has_24_blue : Tom_blue = 24
axiom Together_have_68_blue : Total_blue = 68

theorem cannot_determine_red_marbles (Jason_blue Tom_blue Total_blue : ℕ) : ¬ ∃ (Jason_red : ℕ), True := by
  sorry

end cannot_determine_red_marbles_l216_216155


namespace option_b_not_zero_option_a_is_zero_option_c_is_zero_option_d_is_zero_l216_216969

variables {V : Type*} [AddCommGroup V] [Vector V]

-- Define vectors
variables (A B C D O N Q P M : V) 

--1  The expression sums
def OptionA := A + B + C
def OptionB := O + C + B + O
def OptionC := A - B + D - C
def OptionD := N + Q + P - M

--2  Prove that OptionB is not necessarily 0
theorem option_b_not_zero (v₁ v₂ v₃ v₄ : V) : ¬ (v₁ + v₂ + v₃ + v₄ = 0) :=
begin
  sorry
end

--3  Prove that the other expressions add up to zero
theorem option_a_is_zero (v₁ v₂ v₃ : V) : v₁ + v₂ + v₃ = 0 :=
begin
  sorry
end

theorem option_c_is_zero (v₁ v₂ v₃ v₄ : V) : v₁ - v₂ + v₃ - v₄ = 0 :=
begin
  sorry
end

theorem option_d_is_zero (v₁ v₂ v₃ v₄ : V) : v₁ + v₂ + v₃ - v₄ = 0 :=
begin
  sorry
end

end option_b_not_zero_option_a_is_zero_option_c_is_zero_option_d_is_zero_l216_216969


namespace cost_per_minute_l216_216531

theorem cost_per_minute (monthly_fee total_bill billed_minutes : ℝ) (h_monthly_fee : monthly_fee = 2) (h_total_bill : total_bill = 23.36) (h_billed_minutes : billed_minutes = 178) : 
  (total_bill - monthly_fee) / billed_minutes = 0.12 :=
by
  rw [h_monthly_fee, h_total_bill, h_billed_minutes]
  norm_num
  sorry

end cost_per_minute_l216_216531


namespace sub_eight_l216_216933

theorem sub_eight (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end sub_eight_l216_216933


namespace recycling_target_l216_216319

/-- Six Grade 4 sections launched a recycling drive where they collect old newspapers to recycle.
Each section collected 280 kilos in two weeks. After the third week, they found that they need 320 kilos more to reach their target.
  How many kilos of the newspaper is their target? -/
theorem recycling_target (sections : ℕ) (kilos_collected_2_weeks : ℕ) (additional_kilos : ℕ) : 
  sections = 6 ∧ kilos_collected_2_weeks = 280 ∧ additional_kilos = 320 → 
  (sections * (kilos_collected_2_weeks / 2) * 3 + additional_kilos) = 2840 :=
by
  sorry

end recycling_target_l216_216319


namespace common_roots_product_sum_l216_216329

theorem common_roots_product_sum (C D u v w t p q r : ℝ) (huvw : u^3 + C * u - 20 = 0) (hvw : v^3 + C * v - 20 = 0)
  (hw: w^3 + C * w - 20 = 0) (hut: t^3 + D * t^2 - 40 = 0) (hvw: v^3 + D * v^2 - 40 = 0) 
  (hu: u^3 + D * u^2 - 40 = 0) (h1: u + v + w = 0) (h2: u * v * w = 20) 
  (h3: u * v + u * t + v * t = 0) (h4: u * v * t = 40) :
  p = 4 → q = 3 → r = 5 → p + q + r = 12 :=
by sorry

end common_roots_product_sum_l216_216329


namespace min_distance_from_start_after_9_minutes_l216_216226

noncomputable def robot_min_distance : ℝ :=
  let movement_per_minute := 10
  sorry

theorem min_distance_from_start_after_9_minutes :
  robot_min_distance = 10 :=
sorry

end min_distance_from_start_after_9_minutes_l216_216226


namespace individual_max_food_l216_216339

/-- Given a minimum number of guests and a total amount of food consumed,
    we want to find the maximum amount of food an individual guest could have consumed. -/
def total_food : ℝ := 319
def min_guests : ℝ := 160
def max_food_per_guest : ℝ := 1.99

theorem individual_max_food :
  total_food / min_guests <= max_food_per_guest := by
  sorry

end individual_max_food_l216_216339


namespace ratio_a_to_c_l216_216762

-- Define the variables and ratios
variables (x y z a b c d : ℝ)

-- Define the conditions as given ratios
variables (h1 : a / b = 2 * x / (3 * y))
variables (h2 : b / c = z / (5 * z))
variables (h3 : a / d = 4 * x / (7 * y))
variables (h4 : d / c = 7 * y / (3 * z))

-- Statement to prove the ratio of a to c
theorem ratio_a_to_c (x y z a b c d : ℝ) 
  (h1 : a / b = 2 * x / (3 * y)) 
  (h2 : b / c = z / (5 * z)) 
  (h3 : a / d = 4 * x / (7 * y)) 
  (h4 : d / c = 7 * y / (3 * z)) : a / c = 2 * x / (15 * y) :=
sorry

end ratio_a_to_c_l216_216762


namespace total_worksheets_l216_216676

theorem total_worksheets (worksheets_graded : ℕ) (problems_per_worksheet : ℕ) (problems_remaining : ℕ)
  (h1 : worksheets_graded = 7)
  (h2 : problems_per_worksheet = 2)
  (h3 : problems_remaining = 14): 
  worksheets_graded + (problems_remaining / problems_per_worksheet) = 14 := 
by 
  sorry

end total_worksheets_l216_216676


namespace g_eq_l216_216896

noncomputable def g (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem g_eq (n : ℕ) : g (n + 2) - g (n - 2) = 3 * g n := by
  sorry

end g_eq_l216_216896


namespace sum_of_six_consecutive_integers_l216_216782

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l216_216782


namespace triangle_third_side_length_l216_216054

theorem triangle_third_side_length 
  (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 3) :
  (4 < c ∧ c < 18) → c ≠ 3 :=
by
  sorry

end triangle_third_side_length_l216_216054


namespace hcf_of_two_numbers_l216_216281

theorem hcf_of_two_numbers (A B : ℕ) (h1 : Nat.lcm A B = 750) (h2 : A * B = 18750) : Nat.gcd A B = 25 :=
by
  sorry

end hcf_of_two_numbers_l216_216281


namespace sum_of_six_consecutive_integers_l216_216781

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l216_216781


namespace simplify_expression_l216_216458

open Complex

theorem simplify_expression :
  ((4 + 6 * I) / (4 - 6 * I) * (4 - 6 * I) / (4 + 6 * I) + (4 - 6 * I) / (4 + 6 * I) * (4 + 6 * I) / (4 - 6 * I)) = 2 :=
by
  sorry

end simplify_expression_l216_216458


namespace tom_needs_noodle_packages_l216_216619

def beef_pounds : ℕ := 10
def noodle_multiplier : ℕ := 2
def initial_noodles : ℕ := 4
def package_weight : ℕ := 2

theorem tom_needs_noodle_packages :
  (noodle_multiplier * beef_pounds - initial_noodles) / package_weight = 8 := 
by 
  -- Faithfully skipping the solution steps
  sorry

end tom_needs_noodle_packages_l216_216619


namespace book_arrangement_l216_216133

theorem book_arrangement (math_books : ℕ) (english_books : ℕ) (science_books : ℕ)
  (math_different : math_books = 4) 
  (english_different : english_books = 5) 
  (science_different : science_books = 2) :
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books) = 34560 := 
by
  sorry

end book_arrangement_l216_216133


namespace optimal_ticket_price_l216_216502

noncomputable def revenue (x : ℕ) : ℤ :=
  if x < 6 then -5750
  else if x ≤ 10 then 1000 * (x : ℤ) - 5750
  else if x ≤ 38 then -30 * (x : ℤ)^2 + 1300 * (x : ℤ) - 5750
  else -5750

theorem optimal_ticket_price :
  revenue 22 = 8330 :=
by
  sorry

end optimal_ticket_price_l216_216502


namespace factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l216_216986

theorem factorize_3x_squared_minus_7x_minus_6 (x : ℝ) :
  3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2) :=
sorry

theorem factorize_6x_squared_minus_7x_minus_5 (x : ℝ) :
  6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5) :=
sorry

end factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l216_216986


namespace circle_equation_l216_216555

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := x + y + 2 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def line2 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def is_solution (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 6 * y - 16 = 0

-- Problem statement in Lean
theorem circle_equation : ∃ x y : ℝ, 
  (line1 x y ∧ circle1 x y ∧ line2 (x / 2) (x / 2)) → is_solution x y :=
sorry

end circle_equation_l216_216555


namespace expected_difference_is_correct_l216_216084

noncomputable def expected_days_jogging : ℚ :=
  4 / 7 * 365

noncomputable def expected_days_yoga : ℚ :=
  3 / 7 * 365

noncomputable def expected_difference_days : ℚ :=
  expected_days_jogging - expected_days_yoga

theorem expected_difference_is_correct :
  expected_difference_days ≈ 52.71 := 
sorry

end expected_difference_is_correct_l216_216084


namespace petya_passwords_l216_216594

theorem petya_passwords : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 8, 9} in
  let is_password (password : Fin 4 → ℕ) : Prop := ∀ i, password i ∈ digits in
  let has_at_least_two_identical (password : Fin 4 → ℕ) : Prop := ∃ i j, i ≠ j ∧ password i = password j in
  (Fin 4 → ℕ) → 
  (∀ password, is_password password → has_at_least_two_identical password) = 3537 :=
by sorry

end petya_passwords_l216_216594


namespace neg_neg_eq_pos_l216_216648

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l216_216648


namespace sin_zero_necessary_not_sufficient_l216_216208

theorem sin_zero_necessary_not_sufficient:
  (∀ α : ℝ, (∃ k : ℤ, α = 2 * k * Real.pi) → (Real.sin α = 0)) ∧
  ¬ (∀ α : ℝ, (Real.sin α = 0) → (∃ k : ℤ, α = 2 * k * Real.pi)) :=
by
  sorry

end sin_zero_necessary_not_sufficient_l216_216208


namespace g_at_9_l216_216333

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l216_216333


namespace initial_kittens_l216_216220

theorem initial_kittens (x : ℕ) (h : x + 3 = 9) : x = 6 :=
by {
  sorry
}

end initial_kittens_l216_216220


namespace zhang_qiu_jian_problem_l216_216170

-- Define the arithmetic sequence
def arithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

-- Sum of first n terms of an arithmetic sequence
def sumArithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem zhang_qiu_jian_problem :
  sumArithmeticSequence 5 (16 / 29) 30 = 390 := 
by 
  sorry

end zhang_qiu_jian_problem_l216_216170


namespace solution_l216_216149

open Real

variables (a b c A B C : ℝ)

-- Condition: In ΔABC, the sides opposite to angles A, B, and C are a, b, and c respectively
-- Condition: Given equation relating sides and angles in ΔABC
axiom eq1 : a * sin C / (1 - cos A) = sqrt 3 * c
-- Condition: b + c = 10
axiom eq2 : b + c = 10
-- Condition: Area of ΔABC
axiom eq3 : (1 / 2) * b * c * sin A = 4 * sqrt 3

-- The final statement to prove
theorem solution :
    (A = π / 3) ∧ (a = 2 * sqrt 13) :=
by
    sorry

end solution_l216_216149


namespace compute_x_over_w_l216_216390

theorem compute_x_over_w (w x y z : ℚ) (hw : w ≠ 0)
  (h1 : (x + 6 * y - 3 * z) / (-3 * x + 4 * w) = 2 / 3)
  (h2 : (-2 * y + z) / (x - w) = 2 / 3) :
  x / w = 2 / 3 :=
sorry

end compute_x_over_w_l216_216390


namespace hcf_of_two_numbers_l216_216282

theorem hcf_of_two_numbers (A B : ℕ) (h1 : Nat.lcm A B = 750) (h2 : A * B = 18750) : Nat.gcd A B = 25 :=
by
  sorry

end hcf_of_two_numbers_l216_216282


namespace total_distance_traveled_l216_216716

theorem total_distance_traveled (x : ℕ) (d_1 d_2 d_3 d_4 d_5 d_6 : ℕ) 
  (h1 : d_1 = 60 / x) 
  (h2 : d_2 = 60 / (x + 3)) 
  (h3 : d_3 = 60 / (x + 6)) 
  (h4 : d_4 = 60 / (x + 9)) 
  (h5 : d_5 = 60 / (x + 12)) 
  (h6 : d_6 = 60 / (x + 15)) 
  (hx1 : x ∣ 60) 
  (hx2 : (x + 3) ∣ 60) 
  (hx3 : (x + 6) ∣ 60) 
  (hx4 : (x + 9) ∣ 60) 
  (hx5 : (x + 12) ∣ 60) 
  (hx6 : (x + 15) ∣ 60) :
  d_1 + d_2 + d_3 + d_4 + d_5 + d_6 = 39 := 
sorry

end total_distance_traveled_l216_216716


namespace biology_physics_ratio_l216_216478

theorem biology_physics_ratio (boys_bio : ℕ) (girls_bio : ℕ) (total_bio : ℕ) (total_phys : ℕ) 
  (h1 : boys_bio = 25) 
  (h2 : girls_bio = 3 * boys_bio) 
  (h3 : total_bio = boys_bio + girls_bio) 
  (h4 : total_phys = 200) : 
  total_bio / total_phys = 1 / 2 :=
by
  sorry

end biology_physics_ratio_l216_216478


namespace second_number_is_sixty_l216_216068

theorem second_number_is_sixty (x : ℕ) (h_sum : 2 * x + x + (2 / 3) * x = 220) : x = 60 :=
by
  sorry

end second_number_is_sixty_l216_216068


namespace one_eighth_of_N_l216_216501

theorem one_eighth_of_N
  (N : ℝ)
  (h : (6 / 11) * N = 48) : (1 / 8) * N = 11 :=
sorry

end one_eighth_of_N_l216_216501


namespace smallest_n_for_common_factor_l216_216624

theorem smallest_n_for_common_factor : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (11 * n - 3) (8 * n + 4) > 1) ∧ n = 42 := 
by
  sorry

end smallest_n_for_common_factor_l216_216624


namespace painting_area_l216_216448

def wall_height : ℝ := 10
def wall_length : ℝ := 15
def door_height : ℝ := 3
def door_length : ℝ := 5

noncomputable def area_of_wall : ℝ :=
  wall_height * wall_length

noncomputable def area_of_door : ℝ :=
  door_height * door_length

noncomputable def area_to_paint : ℝ :=
  area_of_wall - area_of_door

theorem painting_area :
  area_to_paint = 135 := by
  sorry

end painting_area_l216_216448


namespace number_of_trees_l216_216539

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l216_216539


namespace complex_purely_imaginary_condition_l216_216731

theorem complex_purely_imaginary_condition (a : ℝ) :
  (a = 1 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) ∧
  ¬(a = 1 ∧ ¬a = -2 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) :=
  sorry

end complex_purely_imaginary_condition_l216_216731


namespace total_share_amount_l216_216199

theorem total_share_amount (x y z : ℝ) (hx : y = 0.45 * x) (hz : z = 0.30 * x) (hy_share : y = 63) : x + y + z = 245 := by
  sorry

end total_share_amount_l216_216199


namespace second_person_average_pages_per_day_l216_216107

-- Define the given conditions.
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def deshaun_pages_per_book : ℕ := 320
def second_person_percentage : ℝ := 0.75

-- Calculate the total number of pages DeShaun read.
def deshaun_total_pages : ℕ := deshaun_books * deshaun_pages_per_book

-- Calculate the total number of pages the second person read.
def second_person_total_pages : ℕ := (second_person_percentage * deshaun_total_pages).toNat

-- Prove the average number of pages the second person read per day.
def average_pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

theorem second_person_average_pages_per_day :
  average_pages_per_day second_person_total_pages summer_days = 180 :=
by
  sorry

end second_person_average_pages_per_day_l216_216107


namespace prime_condition_l216_216883

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_condition (p : ℕ) (h1 : is_prime p) (h2 : is_prime (8 * p^2 + 1)) : 
  p = 3 ∧ is_prime (8 * p^2 - p + 2) :=
by
  sorry

end prime_condition_l216_216883


namespace smallest_arith_prog_l216_216695

theorem smallest_arith_prog (a d : ℝ) 
  (h1 : (a - 2 * d) < (a - d) ∧ (a - d) < a ∧ a < (a + d) ∧ (a + d) < (a + 2 * d))
  (h2 : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (h3 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
  : (a - 2 * d) = -2 * Real.sqrt 7 :=
sorry

end smallest_arith_prog_l216_216695


namespace eval_expression_l216_216403

theorem eval_expression : (⌈(7: ℚ) / 3⌉ + ⌊ -((7: ℚ) / 3)⌋) = 0 :=
begin
  sorry
end

end eval_expression_l216_216403


namespace value_of_g_l216_216567

-- Defining the function g and its property
def g (x : ℝ) : ℝ := 5

-- Theorem to prove g(x - 3) = 5 for any real number x
theorem value_of_g (x : ℝ) : g (x - 3) = 5 := by
  sorry

end value_of_g_l216_216567


namespace total_volume_of_cubes_l216_216198

theorem total_volume_of_cubes {n : ℕ} (h_n : n = 5) (s : ℕ) (h_s : s = 5) :
  n * (s^3) = 625 :=
by {
  rw [h_n, h_s],
  norm_num,
  sorry
}

end total_volume_of_cubes_l216_216198


namespace size_of_each_group_l216_216927

theorem size_of_each_group 
  (boys : ℕ) (girls : ℕ) (groups : ℕ)
  (total_students : boys + girls = 63)
  (num_groups : groups = 7) :
  63 / 7 = 9 :=
by
  sorry

end size_of_each_group_l216_216927


namespace solve_equation_l216_216838

theorem solve_equation (x : ℝ) : 
  x^2 + 2 * x + 4 * Real.sqrt (x^2 + 2 * x) - 5 = 0 →
  (x = Real.sqrt 2 - 1) ∨ (x = - (Real.sqrt 2 + 1)) :=
by 
  sorry

end solve_equation_l216_216838


namespace number_of_fir_trees_l216_216536

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l216_216536


namespace initial_girls_l216_216480

theorem initial_girls (G : ℕ) 
  (h1 : G + 7 + (15 - 4) = 36) : G = 18 :=
by
  sorry

end initial_girls_l216_216480


namespace percent_increase_output_per_hour_l216_216004

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l216_216004


namespace number_of_books_l216_216352

-- Define the given conditions as variables
def movies_in_series : Nat := 62
def books_read : Nat := 4
def books_yet_to_read : Nat := 15

-- State the proposition we need to prove
theorem number_of_books : (books_read + books_yet_to_read) = 19 :=
by
  sorry

end number_of_books_l216_216352


namespace compounding_frequency_l216_216037

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compounding_frequency (P A r t n : ℝ) 
  (principal : P = 6000) 
  (amount : A = 6615)
  (rate : r = 0.10)
  (time : t = 1) 
  (comp_freq : n = 2) :
  compound_interest P r n t = A := 
by 
  simp [compound_interest, principal, rate, time, comp_freq, amount]
  -- calculations and proof omitted
  sorry

end compounding_frequency_l216_216037


namespace assistant_increases_output_by_100_percent_l216_216015

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l216_216015


namespace symmetric_point_proof_l216_216577

noncomputable def point_symmetric_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_proof :
  point_symmetric_to_x_axis (-2, 3) = (-2, -3) :=
by
  sorry

end symmetric_point_proof_l216_216577


namespace stretching_transformation_eq_curve_l216_216294

variable (x y x₁ y₁ : ℝ)

theorem stretching_transformation_eq_curve :
  (x₁ = 3 * x) →
  (y₁ = y) →
  (x₁^2 + 9 * y₁^2 = 9) →
  (x^2 + y^2 = 1) :=
by
  intros h1 h2 h3
  sorry

end stretching_transformation_eq_curve_l216_216294


namespace fraction_ordering_l216_216193

theorem fraction_ordering :
  (4 / 13) < (12 / 37) ∧ (12 / 37) < (15 / 31) ∧ (4 / 13) < (15 / 31) :=
by sorry

end fraction_ordering_l216_216193


namespace abc_sum_eq_11sqrt6_l216_216274

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l216_216274


namespace decoy_effect_rational_medium_purchase_l216_216032

structure PopcornOption where
  grams : ℕ
  price : ℕ

def small : PopcornOption := ⟨50, 200⟩
def medium : PopcornOption := ⟨70, 400⟩
def large : PopcornOption := ⟨130, 500⟩

-- Hypothesis: Small, Medium and Large options as described.
def options : List PopcornOption := [small, medium, large]

-- We need a theorem that states the usage of the decoy effect.
theorem decoy_effect (o : List PopcornOption) :
  (o = options) →
  (medium.grams < large.grams ∧ medium.price < large.price ∧ 
   (small.price < medium.price ∧ small.grams < medium.grams)) →
  (∃ d : PopcornOption, d = medium ∧ d ≠ small ∧ d ≠ large) →
  (∃ better_option : PopcornOption, better_option = large ∧
    better_option.price - medium.price ≤ 100 ∧
    better_option.grams - medium.grams ≥ 60) :=
begin
  intros hopts hcomp hdc,
  sorry
end

-- Rationality of buying medium-sized popcorn under certain conditions.
theorem rational_medium_purchase (o : List PopcornOption) :
  (o = options) →
  (∃ budget : ℕ, budget = 500 ∧ ∃ drink_price, drink_price = 100 ∧ 
   (medium.price + drink_price ≤ budget) ∧ (large.price > budget ∨ 
   small.grams < medium.grams)) →
  rational_choice : (PopcornOption → ℕ) (d :=
    if medium.price + drink_price ≤ budget then medium else if small.price ≤ budget then small else large) :=
begin
  intros hopts hbudget,
  sorry
end

end decoy_effect_rational_medium_purchase_l216_216032


namespace product_in_fourth_quadrant_l216_216152

def complex_number_1 := (1 : ℂ) - (2 : ℂ) * I
def complex_number_2 := (2 : ℂ) + I

def product := complex_number_1 * complex_number_2

theorem product_in_fourth_quadrant :
  product.re > 0 ∧ product.im < 0 :=
by
  -- the proof is omitted
  sorry

end product_in_fourth_quadrant_l216_216152


namespace relationship_y1_y2_y3_l216_216308

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3_l216_216308


namespace mean_cars_sold_l216_216075

open Rat

theorem mean_cars_sold :
  let monday := 8
  let tuesday := 3
  let wednesday := 10
  let thursday := 4
  let friday := 4
  let saturday := 4
  let total_days := 6
  let total_cars := monday + tuesday + wednesday + thursday + friday + saturday
  let mean := total_cars / total_days
  mean = 33 / 6 :=
by
  let monday := 8
  let tuesday := 3
  let wednesday := 10
  let thursday := 4
  let friday := 4
  let saturday := 4
  let total_days := 6
  let total_cars := monday + tuesday + wednesday + thursday + friday + saturday
  have h1 : total_cars = 8 + 3 + 10 + 4 + 4 + 4 := rfl
  have h2 : total_cars = 33 := by norm_num
  let mean := total_cars / total_days
  have h3 : mean = 33 / 6 := by norm_num
  exact h3

end mean_cars_sold_l216_216075


namespace probability_three_correct_out_of_five_l216_216709

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l216_216709


namespace valerie_needs_21_stamps_l216_216481

def thank_you_cards : ℕ := 3
def bills : ℕ := 2
def mail_in_rebates : ℕ := bills + 3
def job_applications : ℕ := 2 * mail_in_rebates
def water_bill_stamps : ℕ := 1
def electric_bill_stamps : ℕ := 2

def stamps_for_thank_you_cards : ℕ := thank_you_cards * 1
def stamps_for_bills : ℕ := 1 * water_bill_stamps + 1 * electric_bill_stamps
def stamps_for_rebates : ℕ := mail_in_rebates * 1
def stamps_for_job_applications : ℕ := job_applications * 1

def total_stamps : ℕ :=
  stamps_for_thank_you_cards +
  stamps_for_bills +
  stamps_for_rebates +
  stamps_for_job_applications

theorem valerie_needs_21_stamps : total_stamps = 21 := by
  sorry

end valerie_needs_21_stamps_l216_216481


namespace parametric_to_standard_l216_216850

theorem parametric_to_standard (theta : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = 1 + 2 * Real.cos theta)
  (h2 : y = -2 + 2 * Real.sin theta) :
  (x - 1)^2 + (y + 2)^2 = 4 :=
sorry

end parametric_to_standard_l216_216850


namespace calculate_fraction_l216_216094

theorem calculate_fraction : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end calculate_fraction_l216_216094


namespace distance_between_places_l216_216213

theorem distance_between_places
  (d : ℝ) -- let d be the distance between A and B
  (v : ℝ) -- let v be the original speed
  (h1 : v * 4 = d) -- initially, speed * time = distance
  (h2 : (v + 20) * 3 = d) -- after speed increase, speed * new time = distance
  : d = 240 :=
sorry

end distance_between_places_l216_216213


namespace number_of_fir_trees_l216_216537

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l216_216537


namespace find_p_of_tangency_l216_216422

-- condition definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def circle_center : (ℝ × ℝ) := (3, 0)
def circle_radius : ℝ := 4
def directrix (p : ℝ) : ℝ := -p / 2

-- theorem definition
theorem find_p_of_tangency (p : ℝ) :
  (∀ x y : ℝ, circle_eq x y) →
  (∀ x y : ℝ, parabola_eq p x y) →
  dist (circle_center.fst) (directrix p / 0) = circle_radius →
  p = 2 :=
sorry

end find_p_of_tangency_l216_216422


namespace angle_A_size_max_area_triangle_l216_216286

open Real

variable {A B C a b c : ℝ}

-- Part 1: Prove the size of angle A given the conditions
theorem angle_A_size (h1 : (2 * c - b) / a = cos B / cos A) :
  A = π / 3 :=
sorry

-- Part 2: Prove the maximum area of triangle ABC
theorem max_area_triangle (h2 : a = 2 * sqrt 5) :
  ∃ (S : ℝ), S = 5 * sqrt 3 ∧ ∀ (b c : ℝ), S ≤ 1/2 * b * c * sin (π / 3) :=
sorry

end angle_A_size_max_area_triangle_l216_216286


namespace find_k_l216_216344

theorem find_k (k : ℤ) : 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    (x1, y1) = (2, 9) ∧ (x2, y2) = (5, 18) ∧ (x3, y3) = (8, 27) ∧ 
    ∃ m b : ℤ, y1 = m * x1 + b ∧ y2 = m * x2 + b ∧ y3 = m * x3 + b) 
  ∧ ∃ m b : ℤ, k = m * 42 + b
  → k = 129 :=
sorry

end find_k_l216_216344


namespace g_zero_g_one_l216_216507

variable (g : ℤ → ℤ)

axiom condition1 (x : ℤ) : g (x + 5) - g x = 10 * x + 30
axiom condition2 (x : ℤ) : g (x^2 - 2) = (g x - x)^2 + x^2 - 4

theorem g_zero_g_one : (g 0, g 1) = (-4, 1) := 
by 
  sorry

end g_zero_g_one_l216_216507


namespace train_overtake_distance_l216_216943

theorem train_overtake_distance (speed_a speed_b hours_late time_to_overtake distance_a distance_b : ℝ) 
  (h1 : speed_a = 30)
  (h2 : speed_b = 38)
  (h3 : hours_late = 2) 
  (h4 : distance_a = speed_a * hours_late) 
  (h5 : distance_b = speed_b * time_to_overtake) 
  (h6 : time_to_overtake = distance_a / (speed_b - speed_a)) : 
  distance_b = 285 := sorry

end train_overtake_distance_l216_216943


namespace renovation_cost_distribution_l216_216958

/-- A mathematical proof that if Team A works alone for 3 weeks, followed by both Team A and Team B working together, and the total renovation cost is 4000 yuan, then the payment should be distributed equally between Team A and Team B, each receiving 2000 yuan. -/
theorem renovation_cost_distribution :
  let time_A := 18
  let time_B := 12
  let initial_time_A := 3
  let total_cost := 4000
  ∃ x, (1 / time_A * (x + initial_time_A) + 1 / time_B * x = 1) ∧
       let work_A := 1 / time_A * (x + initial_time_A)
       let work_B := 1 / time_B * x
       work_A = work_B ∧
       total_cost / 2 = 2000 :=
by
  sorry

end renovation_cost_distribution_l216_216958


namespace math_problem_solution_l216_216057

theorem math_problem_solution : 8 / 4 - 3 - 9 + 3 * 9 = 17 := 
by 
  sorry

end math_problem_solution_l216_216057


namespace second_person_average_pages_per_day_l216_216108

-- Define the given conditions.
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def deshaun_pages_per_book : ℕ := 320
def second_person_percentage : ℝ := 0.75

-- Calculate the total number of pages DeShaun read.
def deshaun_total_pages : ℕ := deshaun_books * deshaun_pages_per_book

-- Calculate the total number of pages the second person read.
def second_person_total_pages : ℕ := (second_person_percentage * deshaun_total_pages).toNat

-- Prove the average number of pages the second person read per day.
def average_pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

theorem second_person_average_pages_per_day :
  average_pages_per_day second_person_total_pages summer_days = 180 :=
by
  sorry

end second_person_average_pages_per_day_l216_216108


namespace polynomial_A_l216_216729

variables {a b : ℝ} (A : ℝ)
variables (h1 : 2 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem polynomial_A (h : A / (2 * a * b) = 1 - 4 * a ^ 2) : 
  A = 2 * a * b - 8 * a ^ 3 * b :=
by
  sorry

end polynomial_A_l216_216729


namespace total_income_l216_216400

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l216_216400


namespace angle_Z_is_120_l216_216586

-- Define angles and lines
variables {p q : Prop} {X Y Z : ℝ}
variables (h_parallel : p ∧ q)
variables (hX : X = 100)
variables (hY : Y = 140)

-- Proof statement: Given the angles X and Y, we prove that angle Z is 120 degrees.
theorem angle_Z_is_120 (h_parallel : p ∧ q) (hX : X = 100) (hY : Y = 140) : Z = 120 := by 
  -- Here we would add the proof steps
  sorry

end angle_Z_is_120_l216_216586


namespace total_people_correct_current_people_correct_l216_216153

-- Define the conditions as constants
def morning_people : ℕ := 473
def noon_left : ℕ := 179
def afternoon_people : ℕ := 268

-- Define the total number of people
def total_people : ℕ := morning_people + afternoon_people

-- Define the current number of people in the amusement park
def current_people : ℕ := morning_people - noon_left + afternoon_people

-- Theorem proofs
theorem total_people_correct : total_people = 741 := by sorry
theorem current_people_correct : current_people = 562 := by sorry

end total_people_correct_current_people_correct_l216_216153


namespace totalCups_l216_216076

-- Let's state our definitions based on the conditions:
def servingsPerBox : ℕ := 9
def cupsPerServing : ℕ := 2

-- Our goal is to prove the following statement.
theorem totalCups (hServings: servingsPerBox = 9) (hCups: cupsPerServing = 2) : servingsPerBox * cupsPerServing = 18 := by
  -- The detailed proof will go here.
  sorry

end totalCups_l216_216076


namespace count_whole_numbers_between_cubes_l216_216137

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l216_216137


namespace max_sum_x_y_l216_216251

theorem max_sum_x_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x^3 + y^3 + (x + y)^3 + 36 * x * y = 3456) : x + y ≤ 12 :=
sorry

end max_sum_x_y_l216_216251


namespace exp4_is_odd_l216_216629

-- Define the domain for n to be integers and the expressions used in the conditions
variable (n : ℤ)

-- Define the expressions
def exp1 := (n + 1) ^ 2
def exp2 := (n + 1) ^ 2 - (n - 1)
def exp3 := (n + 1) ^ 3
def exp4 := (n + 1) ^ 3 - n ^ 3

-- Prove that exp4 is always odd
theorem exp4_is_odd : ∀ n : ℤ, exp4 n % 2 = 1 := by {
  -- Lean code does not require a proof here, we'll put sorry to skip the proof
  sorry
}

end exp4_is_odd_l216_216629


namespace corrected_mean_of_observations_l216_216471

theorem corrected_mean_of_observations (mean : ℝ) (n : ℕ) (incorrect_observation : ℝ) (correct_observation : ℝ) 
  (h_mean : mean = 41) (h_n : n = 50) (h_incorrect_observation : incorrect_observation = 23) (h_correct_observation : correct_observation = 48) 
  (h_sum_incorrect : mean * n = 2050) : 
  (mean * n - incorrect_observation + correct_observation) / n = 41.5 :=
by
  sorry

end corrected_mean_of_observations_l216_216471


namespace mirasol_initial_amount_l216_216165

/-- 
Mirasol had some money in her account. She spent $10 on coffee beans and $30 on a tumbler. She has $10 left in her account.
Prove that the initial amount of money Mirasol had in her account is $50.
-/
theorem mirasol_initial_amount (spent_coffee : ℕ) (spent_tumbler : ℕ) (left_in_account : ℕ) :
  spent_coffee = 10 → spent_tumbler = 30 → left_in_account = 10 → 
  spent_coffee + spent_tumbler + left_in_account = 50 := 
by
  sorry

end mirasol_initial_amount_l216_216165


namespace cylindrical_coordinates_of_point_l216_216101

noncomputable def cylindrical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x = -r then Real.pi else 0 -- From the step if cos θ = -1
  (r, θ, z)

theorem cylindrical_coordinates_of_point :
  cylindrical_coordinates (-5) 0 (-8) = (5, Real.pi, -8) :=
by
  -- placeholder for the actual proof
  sorry

end cylindrical_coordinates_of_point_l216_216101


namespace folder_cost_l216_216023

theorem folder_cost (cost_pens : ℕ) (cost_notebooks : ℕ) (total_spent : ℕ) (folders : ℕ) :
  cost_pens = 3 → cost_notebooks = 12 → total_spent = 25 → folders = 2 →
  ∃ (cost_per_folder : ℕ), cost_per_folder = 5 :=
by
  intros
  sorry

end folder_cost_l216_216023


namespace white_squares_95th_figure_l216_216246

theorem white_squares_95th_figure : ∀ (T : ℕ → ℕ),
  T 1 = 8 →
  (∀ n ≥ 1, T (n + 1) = T n + 5) →
  T 95 = 478 :=
by
  intros T hT1 hTrec
  -- Skipping the proof
  sorry

end white_squares_95th_figure_l216_216246


namespace sally_paid_peaches_l216_216910

def total_spent : ℝ := 23.86
def amount_spent_on_cherries : ℝ := 11.54
def amount_spent_on_peaches_after_coupon : ℝ := total_spent - amount_spent_on_cherries

theorem sally_paid_peaches : amount_spent_on_peaches_after_coupon = 12.32 :=
by 
  -- The actual proof will involve concrete calculation here.
  -- For now, we skip it with sorry.
  sorry

end sally_paid_peaches_l216_216910


namespace other_number_is_300_l216_216918

theorem other_number_is_300 (A B : ℕ) (h1 : A = 231) (h2 : lcm A B = 2310) (h3 : gcd A B = 30) : B = 300 := by
  sorry

end other_number_is_300_l216_216918


namespace half_product_unique_l216_216051

theorem half_product_unique (x : ℕ) (n k : ℕ) 
  (hn : x = n * (n + 1) / 2) (hk : x = k * (k + 1) / 2) : 
  n = k := 
sorry

end half_product_unique_l216_216051


namespace square_of_square_root_l216_216194

theorem square_of_square_root (x : ℝ) (hx : (Real.sqrt x)^2 = 49) : x = 49 :=
by 
  sorry

end square_of_square_root_l216_216194


namespace correct_options_l216_216362

theorem correct_options :
  (1 + Real.tan 1) * (1 + Real.tan 44) = 2 ∧
  ¬((1 / Real.sin 10) - (Real.sqrt 3 / Real.cos 10) = 2) ∧
  (3 - Real.sin 70) / (2 - (Real.cos 10) ^ 2) = 2 ∧
  ¬(Real.tan 70 * Real.cos 10 * (Real.sqrt 3 * Real.tan 20 - 1) = 2) :=
sorry

end correct_options_l216_216362


namespace find_w_l216_216255

variables {x y : ℚ}

def w : ℚ × ℚ := (-48433 / 975, 2058 / 325)

def vec1 : ℚ × ℚ := (3, 2)
def vec2 : ℚ × ℚ := (3, 4)

def proj (u v : ℚ × ℚ) : ℚ × ℚ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

def p1 : ℚ × ℚ := (47 / 13, 31 / 13)
def p2 : ℚ × ℚ := (85 / 25, 113 / 25)

theorem find_w (hw : w = (x, y)) :
  proj ⟨x, y⟩ vec1 = p1 ∧
  proj ⟨x, y⟩ vec2 = p2 :=
sorry

end find_w_l216_216255


namespace xy_equals_nine_l216_216734

theorem xy_equals_nine (x y : ℝ) (h : (|x + 3| > 0 ∧ (y - 2)^2 = 0) ∨ (|x + 3| = 0 ∧ (y - 2)^2 > 0)) : x^y = 9 :=
sorry

end xy_equals_nine_l216_216734


namespace expected_profit_may_is_3456_l216_216449

-- Given conditions as definitions
def february_profit : ℝ := 2000
def april_profit : ℝ := 2880
def growth_rate (x : ℝ) : Prop := (2000 * (1 + x)^2 = 2880)

-- The expected profit in May
def expected_may_profit (x : ℝ) : ℝ := april_profit * (1 + x)

-- The theorem to be proved based on the given conditions
theorem expected_profit_may_is_3456 (x : ℝ) (h : growth_rate x) (h_pos : x = (1:ℝ)/5) : 
    expected_may_profit x = 3456 :=
by sorry

end expected_profit_may_is_3456_l216_216449


namespace find_matrix_A_l216_216115

-- Let A be a 2x2 matrix such that 
def A (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

theorem find_matrix_A :
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ,
  (A.mulVec ![4, 1] = ![8, 14]) ∧ (A.mulVec ![2, -3] = ![-2, 11]) ∧
  A = ![![2, 1/2], ![-1, -13/3]] :=
by
  sorry

end find_matrix_A_l216_216115


namespace smallest_positive_n_common_factor_l216_216622

open Int

theorem smallest_positive_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ gcd (11 * n - 3) (8 * n + 4) > 1 ∧ n = 5 :=
by
  sorry

end smallest_positive_n_common_factor_l216_216622


namespace probability_spinner_lands_in_shaded_region_l216_216078

theorem probability_spinner_lands_in_shaded_region :
  let total_regions := 4
  let shaded_regions := 3
  (shaded_regions: ℝ) / total_regions = 3 / 4 :=
by
  let total_regions := 4
  let shaded_regions := 3
  sorry

end probability_spinner_lands_in_shaded_region_l216_216078


namespace problem_inequality_l216_216795

theorem problem_inequality 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b)
  (c_pos : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := 
sorry

end problem_inequality_l216_216795


namespace total_area_correct_l216_216956

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def rect_area : ℝ := length * width
noncomputable def square_side : ℝ := radius * Real.sqrt 2
noncomputable def square_area : ℝ := square_side ^ 2
noncomputable def total_area : ℝ := rect_area + square_area

theorem total_area_correct : total_area = 686 := 
by
  -- Definitions provided above represent the problem's conditions
  -- The value calculated manually is 686
  -- Proof steps skipped for initial statement creation
  sorry

end total_area_correct_l216_216956


namespace renu_suma_work_together_l216_216455

-- Define the time it takes for Renu to do the work by herself
def renu_days : ℕ := 6

-- Define the time it takes for Suma to do the work by herself
def suma_days : ℕ := 12

-- Define the work rate for Renu
def renu_work_rate : ℚ := 1 / renu_days

-- Define the work rate for Suma
def suma_work_rate : ℚ := 1 / suma_days

-- Define the combined work rate
def combined_work_rate : ℚ := renu_work_rate + suma_work_rate

-- Define the days it takes for both Renu and Suma to complete the work together
def days_to_complete_together : ℚ := 1 / combined_work_rate

-- The theorem stating that Renu and Suma can complete the work together in 4 days
theorem renu_suma_work_together : days_to_complete_together = 4 :=
by
  have h1 : renu_days = 6 := rfl
  have h2 : suma_days = 12 := rfl
  have h3 : renu_work_rate = 1 / 6 := by simp [renu_work_rate, h1]
  have h4 : suma_work_rate = 1 / 12 := by simp [suma_work_rate, h2]
  have h5 : combined_work_rate = 1 / 6 + 1 / 12 := by simp [combined_work_rate, h3, h4]
  have h6 : combined_work_rate = 1 / 4 := by norm_num [h5]
  have h7 : days_to_complete_together = 1 / (1 / 4) := by simp [days_to_complete_together, h6]
  have h8 : days_to_complete_together = 4 := by norm_num [h7]
  exact h8

end renu_suma_work_together_l216_216455


namespace fir_trees_alley_l216_216540

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l216_216540


namespace problem_statement_l216_216093

theorem problem_statement : 15 * 30 + 45 * 15 + 15 * 15 = 1350 :=
by
  sorry

end problem_statement_l216_216093


namespace relationship_between_m_and_n_l216_216427

variable (x : ℝ)

def m := x^2 + 2*x + 3
def n := 2

theorem relationship_between_m_and_n :
  m x ≥ n := by
  sorry

end relationship_between_m_and_n_l216_216427


namespace solve_inequality_l216_216320

theorem solve_inequality (x : ℝ) :
  (x^2 - 4 * x - 12) / (x - 3) < 0 ↔ (-2 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) := by
  sorry

end solve_inequality_l216_216320


namespace identical_prob_of_painted_cubes_l216_216356

/-
  Given:
  - Each face of a cube can be painted in one of 3 colors.
  - Each cube has 6 faces.
  - The total possible ways to paint both cubes is 531441.
  - The total ways to paint them such that they are identical after rotation is 66.

  Prove:
  - The probability of two painted cubes being identical after rotation is 2/16101.
-/
theorem identical_prob_of_painted_cubes :
  let total_ways := 531441
  let identical_ways := 66
  (identical_ways : ℚ) / total_ways = 2 / 16101 := by
  sorry

end identical_prob_of_painted_cubes_l216_216356


namespace find_f_at_six_l216_216962

theorem find_f_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2) : f 6 = 3.75 :=
by
  sorry

end find_f_at_six_l216_216962


namespace pencils_multiple_of_40_l216_216214

theorem pencils_multiple_of_40 :
  ∃ n : ℕ, 640 % n = 0 ∧ n ≤ 40 → ∃ m : ℕ, 40 * m = 40 * n :=
by
  sorry

end pencils_multiple_of_40_l216_216214


namespace minimum_w_coincide_after_translation_l216_216718

noncomputable def period_of_cosine (w : ℝ) : ℝ := (2 * Real.pi) / w

theorem minimum_w_coincide_after_translation
  (w : ℝ) (h_w_pos : 0 < w) :
  period_of_cosine w = (4 * Real.pi) / 3 → w = 3 / 2 :=
by
  sorry

end minimum_w_coincide_after_translation_l216_216718


namespace empty_set_subset_zero_set_l216_216810

-- Define the sets
def zero_set : Set ℕ := {0}
def empty_set : Set ℕ := ∅

-- State the problem
theorem empty_set_subset_zero_set : empty_set ⊂ zero_set :=
sorry

end empty_set_subset_zero_set_l216_216810


namespace neg_neg_eq_l216_216642

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l216_216642


namespace min_value_of_a_l216_216556

theorem min_value_of_a
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (a : ℝ)
  (h_cond : f (Real.logb 2 a) + f (Real.logb (1/2) a) ≤ 2 * f 1) :
  a = 1/2 := sorry

end min_value_of_a_l216_216556


namespace claudia_coins_l216_216979

theorem claudia_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 29 - x = 26) :
  y = 12 :=
by
  sorry

end claudia_coins_l216_216979


namespace isabella_hair_length_end_of_year_l216_216295

/--
Isabella's initial hair length.
-/
def initial_hair_length : ℕ := 18

/--
Isabella's hair growth over the year.
-/
def hair_growth : ℕ := 6

/--
Prove that Isabella's hair length at the end of the year is 24 inches.
-/
theorem isabella_hair_length_end_of_year : initial_hair_length + hair_growth = 24 := by
  sorry

end isabella_hair_length_end_of_year_l216_216295


namespace total_volume_of_removed_pyramids_l216_216837

noncomputable def volume_of_removed_pyramids (edge_length : ℝ) : ℝ :=
  8 * (1 / 3 * (1 / 2 * (edge_length / 4) * (edge_length / 4)) * (edge_length / 4) / 6)

theorem total_volume_of_removed_pyramids :
  volume_of_removed_pyramids 1 = 1 / 48 :=
by
  sorry

end total_volume_of_removed_pyramids_l216_216837


namespace johns_total_animals_l216_216583

variable (Snakes Monkeys Lions Pandas Dogs : ℕ)

theorem johns_total_animals :
  Snakes = 15 →
  Monkeys = 2 * Snakes →
  Lions = Monkeys - 5 →
  Pandas = Lions + 8 →
  Dogs = Pandas / 3 →
  Snakes + Monkeys + Lions + Pandas + Dogs = 114 :=
by
  intros hSnakes hMonkeys hLions hPandas hDogs
  rw [hSnakes] at hMonkeys
  rw [hMonkeys] at hLions
  rw [hLions] at hPandas
  rw [hPandas] at hDogs
  sorry

end johns_total_animals_l216_216583


namespace product_mod_7_l216_216926

theorem product_mod_7 (a b c : ℕ) (ha : a % 7 = 3) (hb : b % 7 = 4) (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 4 :=
sorry

end product_mod_7_l216_216926


namespace smallest_number_of_oranges_l216_216496

theorem smallest_number_of_oranges (n : ℕ) (total_oranges : ℕ) :
  (total_oranges > 200) ∧ total_oranges = 15 * n - 6 ∧ n ≥ 14 → total_oranges = 204 :=
by
  sorry

end smallest_number_of_oranges_l216_216496


namespace min_distance_from_start_after_9_minutes_l216_216225

noncomputable def robot_min_distance : ℝ :=
  let movement_per_minute := 10
  sorry

theorem min_distance_from_start_after_9_minutes :
  robot_min_distance = 10 :=
sorry

end min_distance_from_start_after_9_minutes_l216_216225


namespace prime_factor_condition_l216_216879

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 1
  | n + 2 => seq (n + 1) + seq n

theorem prime_factor_condition (p k : ℕ) (hp : Nat.Prime p) (h : p ∣ seq (2 * k) - 2) :
  p ∣ seq (2 * k - 1) - 1 :=
sorry

end prime_factor_condition_l216_216879


namespace average_infection_rate_infected_computers_exceed_700_l216_216657

theorem average_infection_rate (h : (1 + x) ^ 2 = 81) : x = 8 := by
  sorry

theorem infected_computers_exceed_700 (h_infection_rate : 8 = 8) : (1 + 8) ^ 3 > 700 := by
  sorry

end average_infection_rate_infected_computers_exceed_700_l216_216657


namespace average_distance_run_l216_216591

theorem average_distance_run :
  let mickey_lap := 250
  let johnny_lap := 300
  let alex_lap := 275
  let lea_lap := 280
  let johnny_times := 8
  let lea_times := 5
  let mickey_times := johnny_times / 2
  let alex_times := mickey_times + 1 + 2 * lea_times
  let total_distance := johnny_times * johnny_lap + mickey_times * mickey_lap + lea_times * lea_lap + alex_times * alex_lap
  let number_of_participants := 4
  let avg_distance := total_distance / number_of_participants
  avg_distance = 2231.25 := by
  sorry

end average_distance_run_l216_216591


namespace fraction_equality_l216_216730

theorem fraction_equality (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := 
by
  -- Use the hypthesis to derive that a = 2k, b = 3k, c = 4k and show the equality.
  sorry

end fraction_equality_l216_216730


namespace raft_downstream_time_l216_216608

variables {s v_s v_c : ℝ}

-- Distance covered by the motor ship downstream in 5 hours
def downstream_time (s : ℝ) (v_s v_c : ℝ) : Prop := s / (v_s + v_c) = 5

-- Distance covered by the motor ship upstream in 6 hours
def upstream_time (s : ℝ) (v_s v_c : ℝ) : Prop := s / (v_s - v_c) = 6

-- Time it takes for a raft to float downstream over this distance
theorem raft_downstream_time : 
  ∀ (s v_s v_c : ℝ), 
  downstream_time s v_s v_c ∧ upstream_time s v_s v_c → s / v_c = 60 :=
by
  sorry

end raft_downstream_time_l216_216608


namespace length_of_BE_l216_216001

theorem length_of_BE (A B C E : ℝ)
  (h₀ : 0 < A) 
  (h₁ : 0 < B) 
  (h₂ : 0 < C) 
  (h₃ : 0 < E)
  (h4 : B = 5)
  (h5 : C = 12)
  (h6 : A = 13)
  (h7 : E = sqrt 2)
  (h8: ((A^2) + (B^2)) = (C^2)) :
  E = (65 / 18) * sqrt 2 := 
by
  sorry

end length_of_BE_l216_216001


namespace number_of_trees_is_eleven_l216_216533

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l216_216533


namespace charge_R_12_5_percent_more_l216_216325

-- Let R be the charge for a single room at hotel R.
-- Let G be the charge for a single room at hotel G.
-- Let P be the charge for a single room at hotel P.

def charge_R (R : ℝ) : Prop := true
def charge_G (G : ℝ) : Prop := true
def charge_P (P : ℝ) : Prop := true

axiom hotel_P_20_less_R (R P : ℝ) : charge_R R → charge_P P → P = 0.80 * R
axiom hotel_P_10_less_G (G P : ℝ) : charge_G G → charge_P P → P = 0.90 * G

theorem charge_R_12_5_percent_more (R G : ℝ) :
  charge_R R → charge_G G → (∃ P, charge_P P ∧ P = 0.80 * R ∧ P = 0.90 * G) → R = 1.125 * G :=
by sorry

end charge_R_12_5_percent_more_l216_216325


namespace car_distances_equal_600_l216_216181

-- Define the variables
def time_R (t : ℝ) := t
def speed_R := 50
def time_P (t : ℝ) := t - 2
def speed_P := speed_R + 10
def distance (t : ℝ) := speed_R * time_R t

-- The Lean theorem statement
theorem car_distances_equal_600 (t : ℝ) (h : time_R t = t) (h1 : speed_R = 50) 
  (h2 : time_P t = t - 2) (h3 : speed_P = speed_R + 10) :
  distance t = 600 :=
by
  -- We would provide the proof here, but for now we use sorry to indicate the proof is omitted.
  sorry

end car_distances_equal_600_l216_216181


namespace find_a_l216_216124

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

end find_a_l216_216124


namespace three_correct_deliveries_probability_l216_216698

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l216_216698


namespace all_of_the_above_were_used_as_money_l216_216489

-- Defining the conditions that each type was used as money
def gold_used_as_money : Prop := True
def stones_used_as_money : Prop := True
def horses_used_as_money : Prop := True
def dried_fish_used_as_money : Prop := True
def mollusk_shells_used_as_money : Prop := True

-- The statement to prove
theorem all_of_the_above_were_used_as_money :
  gold_used_as_money ∧
  stones_used_as_money ∧
  horses_used_as_money ∧
  dried_fish_used_as_money ∧
  mollusk_shells_used_as_money ↔
  (∀ x, (x = "gold" ∨ x = "stones" ∨ x = "horses" ∨ x = "dried fish" ∨ x = "mollusk shells") → 
  (x = "gold" ∧ gold_used_as_money) ∨ 
  (x = "stones" ∧ stones_used_as_money) ∨ 
  (x = "horses" ∧ horses_used_as_money) ∨ 
  (x = "dried fish" ∧ dried_fish_used_as_money) ∨ 
  (x = "mollusk shells" ∧ mollusk_shells_used_as_money)) :=
by
  sorry

end all_of_the_above_were_used_as_money_l216_216489


namespace emmalyn_earnings_l216_216395

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l216_216395


namespace find_x_value_l216_216568

theorem find_x_value (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := 
by 
  sorry

end find_x_value_l216_216568


namespace remaining_money_l216_216666

-- Defining the conditions
def orchids_qty := 20
def price_per_orchid := 50
def chinese_money_plants_qty := 15
def price_per_plant := 25
def worker_qty := 2
def salary_per_worker := 40
def cost_of_pots := 150

-- Earnings and expenses calculations
def earnings_from_orchids := orchids_qty * price_per_orchid
def earnings_from_plants := chinese_money_plants_qty * price_per_plant
def total_earnings := earnings_from_orchids + earnings_from_plants

def worker_expenses := worker_qty * salary_per_worker
def total_expenses := worker_expenses + cost_of_pots

-- The proof problem
theorem remaining_money : total_earnings - total_expenses = 1145 :=
by
  sorry

end remaining_money_l216_216666


namespace sin_maximum_value_l216_216973

theorem sin_maximum_value (c : ℝ) :
  (∀ x : ℝ, x = -π/4 → 3 * Real.sin (2 * x + c) = 3) → c = π :=
by
 sorry

end sin_maximum_value_l216_216973


namespace inequality_solution_set_range_of_a_l216_216130

noncomputable def f (x : ℝ) := abs (2 * x - 1) - abs (x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x > 0 } = { x : ℝ | x < -1 / 3 ∨ x > 3 } :=
sorry

theorem range_of_a (x0 : ℝ) (h : f x0 + 2 * a ^ 2 < 4 * a) :
  -1 / 2 < a ∧ a < 5 / 2 :=
sorry

end inequality_solution_set_range_of_a_l216_216130


namespace sum_six_consecutive_integers_l216_216779

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l216_216779


namespace car_mileage_before_modification_l216_216074

theorem car_mileage_before_modification (miles_per_gallon_before : ℝ) 
  (fuel_efficiency_modifier : ℝ := 0.75) (tank_capacity : ℝ := 12) 
  (extra_miles_after_modification : ℝ := 96) :
  (1 / fuel_efficiency_modifier) * miles_per_gallon_before * (tank_capacity - 1) = 24 :=
by
  sorry

end car_mileage_before_modification_l216_216074


namespace percent_increase_output_l216_216014

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l216_216014


namespace min_xy_min_x_add_y_l216_216413

open Real

theorem min_xy (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : xy ≥ 9 := sorry

theorem min_x_add_y (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : x + y ≥ 6 := sorry

end min_xy_min_x_add_y_l216_216413


namespace arithmetic_mean_of_normal_distribution_l216_216172

theorem arithmetic_mean_of_normal_distribution
  (σ : ℝ) (hσ : σ = 1.5)
  (value : ℝ) (hvalue : value = 11.5)
  (hsd : value = μ - 2 * σ) :
  μ = 14.5 :=
by
  sorry

end arithmetic_mean_of_normal_distribution_l216_216172


namespace miles_left_to_drive_l216_216498

theorem miles_left_to_drive 
  (total_distance : ℕ) 
  (distance_covered : ℕ) 
  (remaining_distance : ℕ) 
  (h1 : total_distance = 78) 
  (h2 : distance_covered = 32) 
  : remaining_distance = total_distance - distance_covered -> remaining_distance = 46 :=
by
  sorry

end miles_left_to_drive_l216_216498


namespace volume_Q3_l216_216552

noncomputable def sequence_of_polyhedra (n : ℕ) : ℚ :=
match n with
| 0     => 1
| 1     => 3 / 2
| 2     => 45 / 32
| 3     => 585 / 128
| _     => 0 -- for n > 3 not defined

theorem volume_Q3 : sequence_of_polyhedra 3 = 585 / 128 :=
by
  -- Placeholder for the theorem proof
  sorry

end volume_Q3_l216_216552


namespace inverse_proportion_neg_k_l216_216831

theorem inverse_proportion_neg_k (x1 x2 y1 y2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 > y2) :
  ∃ k : ℝ, k < 0 ∧ (∀ x, (x = x1 → y1 = k / x) ∧ (x = x2 → y2 = k / x)) := by
  use -1
  sorry

end inverse_proportion_neg_k_l216_216831


namespace alpha_in_second_quadrant_l216_216996

theorem alpha_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 > 0 :=
by
  -- Given conditions
  have : Real.sin α > 0 := h1
  have : Real.cos α < 0 := h2
  sorry

end alpha_in_second_quadrant_l216_216996


namespace path_bound_l216_216098

/-- Definition of P_k: the number of non-intersecting paths of length k starting from point O on a grid 
    where each cell has side length 1. -/
def P_k (k : ℕ) : ℕ := sorry  -- This would normally be defined through some combinatorial method

/-- The main theorem stating the required proof statement. -/
theorem path_bound (k : ℕ) : (P_k k : ℝ) / (3^k : ℝ) < 2 := sorry

end path_bound_l216_216098


namespace recycling_target_l216_216318

/-- Six Grade 4 sections launched a recycling drive where they collect old newspapers to recycle.
Each section collected 280 kilos in two weeks. After the third week, they found that they need 320 kilos more to reach their target.
  How many kilos of the newspaper is their target? -/
theorem recycling_target (sections : ℕ) (kilos_collected_2_weeks : ℕ) (additional_kilos : ℕ) : 
  sections = 6 ∧ kilos_collected_2_weeks = 280 ∧ additional_kilos = 320 → 
  (sections * (kilos_collected_2_weeks / 2) * 3 + additional_kilos) = 2840 :=
by
  sorry

end recycling_target_l216_216318


namespace sum_2_75_0_003_0_158_l216_216067

theorem sum_2_75_0_003_0_158 : 2.75 + 0.003 + 0.158 = 2.911 :=
by
  -- Lean proof goes here  
  sorry

end sum_2_75_0_003_0_158_l216_216067


namespace largest_five_digit_number_divisible_by_6_l216_216937

theorem largest_five_digit_number_divisible_by_6 : 
  ∃ n : ℕ, n < 100000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 100000 ∧ m % 6 = 0 → m ≤ n :=
by
  sorry

end largest_five_digit_number_divisible_by_6_l216_216937


namespace ice_cream_flavors_l216_216881

theorem ice_cream_flavors : (Nat.choose (4 + 4 - 1) (4 - 1) = 35) :=
by
  sorry

end ice_cream_flavors_l216_216881


namespace equivalent_expression_l216_216971

variable (x y : ℝ)

def is_positive_real (r : ℝ) : Prop := r > 0

theorem equivalent_expression 
  (hx : is_positive_real x) 
  (hy : is_positive_real y) : 
  (Real.sqrt (Real.sqrt (x ^ 2 * Real.sqrt (y ^ 3)))) = x ^ (1 / 2) * y ^ (1 / 12) :=
by
  sorry

end equivalent_expression_l216_216971


namespace predicted_height_at_age_10_l216_216219

-- Define the regression model as a function
def regression_model (x : ℝ) : ℝ := 7.19 * x + 73.93

-- Assert the predicted height at age 10
theorem predicted_height_at_age_10 : abs (regression_model 10 - 145.83) < 0.01 := 
by
  -- Here, we would prove the calculation steps
  sorry

end predicted_height_at_age_10_l216_216219


namespace otimes_property_l216_216109

def otimes (a b : ℚ) : ℚ := (a^3) / b

theorem otimes_property : otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = 80 / 27 := by
  sorry

end otimes_property_l216_216109


namespace find_angle_A_find_area_l216_216439

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

end find_angle_A_find_area_l216_216439


namespace smallest_constant_inequality_l216_216406

theorem smallest_constant_inequality :
  ∀ (x y : ℝ), 1 + (x + y)^2 ≤ (4 / 3) * (1 + x^2) * (1 + y^2) :=
by
  intro x y
  sorry

end smallest_constant_inequality_l216_216406


namespace dogs_for_sale_l216_216835

variable (D : ℕ)
def number_of_cats := D / 2
def number_of_birds := 2 * D
def number_of_fish := 3 * D
def total_animals := D + number_of_cats D + number_of_birds D + number_of_fish D

theorem dogs_for_sale (h : total_animals D = 39) : D = 6 :=
by
  sorry

end dogs_for_sale_l216_216835


namespace rectangle_area_increase_l216_216689

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let A_original := L * W
  let A_new := (2 * L) * (2 * W)
  (A_new - A_original) / A_original * 100 = 300 := by
  sorry

end rectangle_area_increase_l216_216689


namespace abs_eq_condition_l216_216430

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 :=
by sorry

end abs_eq_condition_l216_216430


namespace rachel_baked_brownies_l216_216771

theorem rachel_baked_brownies (b : ℕ) (h : 3 * b / 5 = 18) : b = 30 :=
by
  sorry

end rachel_baked_brownies_l216_216771


namespace smallest_n_l216_216967

theorem smallest_n (n : ℕ) (h1: n ≥ 100) (h2: n ≤ 999) 
  (h3: (n + 5) % 8 = 0) (h4: (n - 8) % 5 = 0) : 
  n = 123 :=
sorry

end smallest_n_l216_216967


namespace hyperbola_asymptote_slope_proof_l216_216470

noncomputable def hyperbola_asymptote_slope : ℝ :=
  let foci_distance := Real.sqrt ((8 - 2)^2 + (3 - 3)^2)
  let c := foci_distance / 2
  let a := 2  -- Given that 2a = 4
  let b := Real.sqrt (c^2 - a^2)
  b / a

theorem hyperbola_asymptote_slope_proof :
  ∀ x y : ℝ, 
  (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) →
  hyperbola_asymptote_slope = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_asymptote_slope_proof_l216_216470


namespace fir_trees_count_l216_216545

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l216_216545


namespace hyperbola_focus_to_asymptote_distance_l216_216041

theorem hyperbola_focus_to_asymptote_distance :
  ∀ (x y : ℝ), (x ^ 2 - y ^ 2 = 1) →
  ∃ c : ℝ, (c = 1) :=
by
  sorry

end hyperbola_focus_to_asymptote_distance_l216_216041


namespace desired_ellipse_properties_l216_216987

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2)/(a^2) + (x^2)/(b^2) = 1

def ellipse_has_foci (a b : ℝ) (c : ℝ) : Prop :=
  c^2 = a^2 - b^2

def desired_ellipse_passes_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  is_ellipse a b P.1 P.2

def foci_of_ellipse (a b : ℝ) (c : ℝ) : Prop :=
  ellipse_has_foci a b c

axiom given_ellipse_foci : foci_of_ellipse 3 2 (Real.sqrt 5)

theorem desired_ellipse_properties :
  desired_ellipse_passes_through_point 4 (Real.sqrt 11) (0, 4) ∧
  foci_of_ellipse 4 (Real.sqrt 11) (Real.sqrt 5) :=
by
  sorry

end desired_ellipse_properties_l216_216987


namespace exists_horizontal_chord_l216_216991

theorem exists_horizontal_chord (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_eq : f 0 = f 1) : ∃ n : ℕ, n ≥ 1 ∧ ∃ x : ℝ, 0 ≤ x ∧ x + 1/n ≤ 1 ∧ f x = f (x + 1/n) :=
by
  sorry

end exists_horizontal_chord_l216_216991


namespace beads_necklace_l216_216212

theorem beads_necklace :
  ∀ (total amethyst amber turquoise : ℕ),
  total = 40 →
  amethyst = 7 →
  amber = 2 * amethyst →
  turquoise = total - amethyst - amber →
  turquoise = 19 :=
by
  intros total amethyst amber turquoise h_total h_amethyst h_amber h_turquoise
  rw [h_total, h_amethyst, h_amber] at h_turquoise
  exact h_turquoise

end beads_necklace_l216_216212


namespace relationship_y1_y2_y3_l216_216307

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3_l216_216307


namespace double_neg_eq_pos_l216_216639

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l216_216639


namespace trigonometric_identity_l216_216039

theorem trigonometric_identity (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (Real.cos (α / 2) ^ 2) = 4 * Real.sin α :=
by
  sorry

end trigonometric_identity_l216_216039


namespace quadratic_root_in_interval_l216_216784

theorem quadratic_root_in_interval 
  (a b c : ℝ) 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_root_in_interval_l216_216784


namespace largest_angle_of_obtuse_isosceles_triangle_l216_216187

variables (X Y Z : ℝ)

def is_triangle (X Y Z : ℝ) : Prop := X + Y + Z = 180
def is_isosceles_triangle (X Y : ℝ) : Prop := X = Y
def is_obtuse_triangle (X Y Z : ℝ) : Prop := X > 90 ∨ Y > 90 ∨ Z > 90

theorem largest_angle_of_obtuse_isosceles_triangle
  (X Y Z : ℝ)
  (h1 : is_triangle X Y Z)
  (h2 : is_isosceles_triangle X Y)
  (h3 : X = 30)
  (h4 : is_obtuse_triangle X Y Z) :
  Z = 120 :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l216_216187


namespace average_infection_rate_infected_computers_exceed_700_l216_216656

theorem average_infection_rate (h : (1 + x) ^ 2 = 81) : x = 8 := by
  sorry

theorem infected_computers_exceed_700 (h_infection_rate : 8 = 8) : (1 + 8) ^ 3 > 700 := by
  sorry

end average_infection_rate_infected_computers_exceed_700_l216_216656


namespace jane_output_increase_l216_216021

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l216_216021


namespace least_multiple_of_24_gt_500_l216_216060

theorem least_multiple_of_24_gt_500 : ∃ x : ℕ, (x % 24 = 0) ∧ (x > 500) ∧ (∀ y : ℕ, (y % 24 = 0) ∧ (y > 500) → y ≥ x) ∧ (x = 504) := by
  sorry

end least_multiple_of_24_gt_500_l216_216060


namespace solution_set_of_inequality_l216_216684

def f (x : ℝ) : ℝ := sorry
def f_prime (x : ℝ) : ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x > 0, x^2 * f_prime x + 1 > 0) → 
  f 1 = 5 →
  { x : ℝ | 0 < x ∧ x < 1 } = { x : ℝ | 0 < x ∧ f x < 1 / x + 4 } :=
by 
  intros h1 h2 
  sorry

end solution_set_of_inequality_l216_216684


namespace blocks_probability_l216_216386

-- Definitions for problem conditions
def ang_blocks : list (String × ℕ) := [("red", 1), ("blue", 1), ("yellow", 1), ("white", 1), ("green", 1), ("purple", 1)]
def ben_blocks : list (String × ℕ) := [("red", 1), ("blue", 1), ("yellow", 1), ("white", 1), ("green", 1), ("purple", 1)]
def jasmin_blocks : list (String × ℕ) := [("red", 1), ("blue", 1), ("yellow", 1), ("white", 1), ("green", 1), ("purple", 1)]
def boxes : ℕ := 6

-- The probability calculation is part of the following problem representation
theorem blocks_probability :
  ∃ (m n : ℕ), Nat.rel_prime m n ∧ (m + n = 14471) ∧
  (( ∏ x : fin 6, if ( ∏ x_1 : fin 6, ang_blocks.nth x_1) = ( ∏ x_2 : fin 6, ben_blocks.nth x_2) = ( ∏ x_3 : fin 6, jasmin_blocks.nth x_3) then 1 else 0) /
  ( 6 * 6 * 6 ) = (m / n)) :=
sorry

-- Definitions and calculations that are required should be here

end blocks_probability_l216_216386


namespace f_def_pos_l216_216145

-- Define f to be an odd function
variable (f : ℝ → ℝ)
-- Define f as an odd function
axiom odd_f (x : ℝ) : f (-x) = -f x

-- Define f when x < 0
axiom f_def_neg (x : ℝ) (h : x < 0) : f x = (Real.cos (3 * x)) + (Real.sin (2 * x))

-- State the theorem to be proven:
theorem f_def_pos (x : ℝ) (h : 0 < x) : f x = - (Real.cos (3 * x)) + (Real.sin (2 * x)) :=
sorry

end f_def_pos_l216_216145


namespace paper_length_l216_216785

theorem paper_length :
  ∃ (L : ℝ), (2 * (11 * L) = 2 * (8.5 * 11) + 100 ∧ L = 287 / 22) :=
sorry

end paper_length_l216_216785


namespace Angle_CNB_20_l216_216186

theorem Angle_CNB_20 :
  ∀ (A B C N : Type) 
    (AC BC : Prop) 
    (angle_ACB : ℕ)
    (angle_NAC : ℕ)
    (angle_NCA : ℕ), 
    (AC ↔ BC) →
    angle_ACB = 98 →
    angle_NAC = 15 →
    angle_NCA = 21 →
    ∃ angle_CNB, angle_CNB = 20 :=
by
  sorry

end Angle_CNB_20_l216_216186


namespace minimum_restoration_time_l216_216660

structure Handicraft :=
  (shaping: ℕ)
  (painting: ℕ)

def handicraft_A : Handicraft := ⟨9, 15⟩
def handicraft_B : Handicraft := ⟨16, 8⟩
def handicraft_C : Handicraft := ⟨10, 14⟩

def total_restoration_time (order: List Handicraft) : ℕ :=
  let rec aux (remaining: List Handicraft) (A_time: ℕ) (B_time: ℕ) (acc: ℕ) : ℕ :=
    match remaining with
    | [] => acc
    | h :: t =>
      let A_next := A_time + h.shaping
      let B_next := max A_next B_time + h.painting
      aux t A_next B_next B_next
  aux order 0 0 0

theorem minimum_restoration_time :
  total_restoration_time [handicraft_A, handicraft_C, handicraft_B] = 46 :=
by
  simp [total_restoration_time, handicraft_A, handicraft_B, handicraft_C]
  sorry

end minimum_restoration_time_l216_216660


namespace simplify_complex_expr_l216_216457

theorem simplify_complex_expr : 
  ∀ (i : ℂ), i^2 = -1 → ( (2 + 4 * i) / (2 - 4 * i) - (2 - 4 * i) / (2 + 4 * i) )
  = -8/5 + (16/5 : ℂ) * i :=
by
  intro i h_i_squared
  sorry

end simplify_complex_expr_l216_216457


namespace sequence_periodicity_l216_216561

variable {a b : ℕ → ℤ}

theorem sequence_periodicity (h : ∀ n ≥ 3, 
    (a n - a (n - 1)) * (a n - a (n - 2)) + 
    (b n - b (n - 1)) * (b n - b (n - 2)) = 0) : 
    ∃ k > 0, a k + b k = a (k + 2018) + b (k + 2018) := 
    by
    sorry

end sequence_periodicity_l216_216561


namespace leaves_count_l216_216654

theorem leaves_count {m n L : ℕ} (h1 : m + n = 10) (h2 : L = 5 * m + 2 * n) :
  ¬(L = 45 ∨ L = 39 ∨ L = 37 ∨ L = 31) :=
by
  sorry

end leaves_count_l216_216654


namespace algebraic_expression_value_l216_216429

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 :=
by
  sorry

end algebraic_expression_value_l216_216429


namespace triangle_perimeter_l216_216574

-- Definitions and given conditions
def side_length_a (a : ℝ) : Prop := a = 6
def inradius (r : ℝ) : Prop := r = 2
def circumradius (R : ℝ) : Prop := R = 5

-- The final proof statement to be proven
theorem triangle_perimeter (a r R : ℝ) (b c P : ℝ) 
  (h1 : side_length_a a)
  (h2 : inradius r)
  (h3 : circumradius R)
  (h4 : P = 2 * ((a + b + c) / 2)) :
  P = 24 :=
sorry

end triangle_perimeter_l216_216574


namespace total_blue_balloons_l216_216582

def joan_blue_balloons : ℕ := 60
def melanie_blue_balloons : ℕ := 85
def alex_blue_balloons : ℕ := 37
def gary_blue_balloons : ℕ := 48

theorem total_blue_balloons :
  joan_blue_balloons + melanie_blue_balloons + alex_blue_balloons + gary_blue_balloons = 230 :=
by simp [joan_blue_balloons, melanie_blue_balloons, alex_blue_balloons, gary_blue_balloons]

end total_blue_balloons_l216_216582


namespace eating_ways_eq_6720_l216_216767

-- Define the problem context
def is_valid_chocolate_bar (R : ℕ) (C : ℕ) := R = 2 ∧ C = 4

def valid_eating_condition (R C : ℕ) (grid : Matrix (Fin R) (Fin C) Bool) :=
  ∀ (r c : Fin R) (hrc : grid[r, c] = true), (neighboring_unvisited r c grid ≤ 2)

-- Function that computes number of neighbor cells that are unvisited
def neighboring_unvisited (r c : Fin 2) (grid : Matrix (Fin 2) (Fin 4) Bool) : ℕ :=
  let directions := [(1, 0), (0, 1), (-1, 0), (0, -1)]  -- down, right, up, left
  directions.foldl (λ acc (dx, dy), 
    let nr := r + dx,
        nc := c + dy in 
    if nr < 2 ∧ nr >= 0 ∧ nc < 4 ∧ nc >= 0 ∧ grid[nr, nc] = true then
      acc + 1
    else
      acc) 0

noncomputable def number_of_ways_to_eat_chocolate : ℕ :=
  6720

-- The theorem statement
theorem eating_ways_eq_6720 :
  ∀ (R C : ℕ), is_valid_chocolate_bar R C → ∃ k : ℕ, valid_eating_condition R C grid → k = 6720 :=
begin
  intros R C hRC,
  existsi number_of_ways_to_eat_chocolate,
  intro h,
  sorry -- The proof steps would go here
end

end eating_ways_eq_6720_l216_216767


namespace whole_numbers_count_between_cubic_roots_l216_216136

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l216_216136


namespace winner_won_by_288_votes_l216_216801

theorem winner_won_by_288_votes (V : ℝ) (votes_won : ℝ) (perc_won : ℝ) 
(h1 : perc_won = 0.60)
(h2 : votes_won = 864)
(h3 : votes_won = perc_won * V) : 
votes_won - (1 - perc_won) * V = 288 := 
sorry

end winner_won_by_288_votes_l216_216801


namespace plant_supplier_earnings_l216_216667

theorem plant_supplier_earnings :
  let orchids_price := 50
  let orchids_sold := 20
  let money_plant_price := 25
  let money_plants_sold := 15
  let worker_wage := 40
  let workers := 2
  let pot_cost := 150
  let total_earnings := (orchids_price * orchids_sold) + (money_plant_price * money_plants_sold)
  let total_expense := (worker_wage * workers) + pot_cost
  total_earnings - total_expense = 1145 :=
by
  sorry

end plant_supplier_earnings_l216_216667


namespace problem_l216_216736

theorem problem (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x ^ 3 / y ^ 2) + (y ^ 3 / x ^ 2) + y = 440 := by
  sorry

end problem_l216_216736


namespace present_age_of_son_l216_216664

/-- A man is 46 years older than his son and in two years, the man's age will be twice the age of his son. Prove that the present age of the son is 44. -/
theorem present_age_of_son (M S : ℕ) (h1 : M = S + 46) (h2 : M + 2 = 2 * (S + 2)) : S = 44 :=
by {
  sorry
}

end present_age_of_son_l216_216664


namespace inequality_geq_8_l216_216793

theorem inequality_geq_8 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 :=
by
  sorry

end inequality_geq_8_l216_216793


namespace cashier_amount_l216_216890

def amount_to_cashier (discount : ℝ) (shorts_count : ℕ) (shorts_price : ℕ) (shirts_count : ℕ) (shirts_price : ℕ) : ℝ :=
  let total_cost := (shorts_count * shorts_price) + (shirts_count * shirts_price)
  let discount_amount := discount * total_cost
  total_cost - discount_amount

theorem cashier_amount : amount_to_cashier 0.1 3 15 5 17 = 117 := 
by
  sorry

end cashier_amount_l216_216890


namespace division_remainder_l216_216746

theorem division_remainder (q d D R : ℕ) (h_q : q = 40) (h_d : d = 72) (h_D : D = 2944) (h_div : D = d * q + R) : R = 64 :=
by sorry

end division_remainder_l216_216746


namespace slope_of_line_l216_216740

theorem slope_of_line (θ : ℝ) (h : θ = 30) :
  ∃ k, k = Real.tan (60 * (π / 180)) ∨ k = Real.tan (120 * (π / 180)) := by
    sorry

end slope_of_line_l216_216740


namespace num_ordered_pairs_l216_216393

theorem num_ordered_pairs :
  ∃ (m n : ℤ), (m * n ≥ 0) ∧ (m^3 + n^3 + 99 * m * n = 33^3) ∧ (35 = 35) :=
by
  sorry

end num_ordered_pairs_l216_216393


namespace number_of_games_X_l216_216461

variable (x : ℕ) -- Total number of games played by team X
variable (y : ℕ) -- Wins by team Y
variable (ly : ℕ) -- Losses by team Y
variable (dy : ℕ) -- Draws by team Y
variable (wx : ℕ) -- Wins by team X
variable (lx : ℕ) -- Losses by team X
variable (dx : ℕ) -- Draws by team X

axiom wins_ratio_X : wx = 3 * x / 4
axiom wins_ratio_Y : y = 2 * (x + 12) / 3
axiom wins_difference : y = wx + 4
axiom losses_difference : ly = lx + 5
axiom draws_difference : dy = dx + 3
axiom eq_losses_draws : lx + dx = (x - wx)

theorem number_of_games_X : x = 48 :=
by
  sorry

end number_of_games_X_l216_216461


namespace problem_statement_l216_216460

theorem problem_statement (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (∃ φn : ℕ, φn = Nat.totient n ∧ p ∣ φn ∧ (∀ a : ℕ, Nat.gcd a n = 1 → n ∣ a ^ (φn / p) - 1)) ↔ 
  (∃ q1 q2 : ℕ, q1 ≠ q2 ∧ Nat.Prime q1 ∧ Nat.Prime q2 ∧ q1 ≡ 1 [MOD p] ∧ q2 ≡ 1 [MOD p] ∧ q1 ∣ n ∧ q2 ∣ n ∨ 
  (∃ q : ℕ, Nat.Prime q ∧ q ≡ 1 [MOD p] ∧ q ∣ n ∧ p ^ 2 ∣ n)) :=
by {
  sorry
}

end problem_statement_l216_216460


namespace second_person_avg_pages_per_day_l216_216106

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end second_person_avg_pages_per_day_l216_216106


namespace sum_of_solutions_eq_zero_l216_216988

noncomputable def f (x : ℝ) : ℝ := 2 ^ |x| + 5 * |x|

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : f x = 28) :
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l216_216988


namespace lights_on_top_layer_l216_216292

theorem lights_on_top_layer
  (x : ℕ)
  (H1 : x + 2 * x + 4 * x + 8 * x + 16 * x + 32 * x + 64 * x = 381) :
  x = 3 :=
  sorry

end lights_on_top_layer_l216_216292


namespace price_difference_eq_l216_216238

-- Define the problem conditions
variable (P : ℝ) -- Original price
variable (H1 : P - 0.15 * P = 61.2) -- Condition 1: 15% discount results in $61.2
variable (H2 : P * (1 - 0.15) = 61.2) -- Another way to represent Condition 1 (if needed)
variable (H3 : 61.2 * 1.25 = 76.5) -- Condition 4: Price raises by 25% after the 15% discount
variable (H4 : 76.5 * 0.9 = 68.85) -- Condition 5: Additional 10% discount after raise
variable (H5 : P = 72) -- Calculated original price

-- Define the theorem to prove
theorem price_difference_eq :
  (P - 68.85 = 3.15) := 
by
  sorry

end price_difference_eq_l216_216238


namespace percent_area_square_in_rectangle_l216_216082

theorem percent_area_square_in_rectangle 
  (s : ℝ) (rect_width : ℝ) (rect_length : ℝ) (h1 : rect_width = 2 * s) (h2 : rect_length = 2 * rect_width) : 
  (s^2 / (rect_length * rect_width)) * 100 = 12.5 :=
by
  sorry

end percent_area_square_in_rectangle_l216_216082


namespace dan_initial_amount_l216_216239

variables (initial_amount spent_amount remaining_amount : ℝ)

theorem dan_initial_amount (h1 : spent_amount = 1) (h2 : remaining_amount = 2) : initial_amount = spent_amount + remaining_amount := by
  sorry

end dan_initial_amount_l216_216239


namespace min_restoration_time_l216_216658

/-- Prove the minimum time required to complete the restoration work of three handicrafts. -/

def shaping_time_A : Nat := 9
def shaping_time_B : Nat := 16
def shaping_time_C : Nat := 10

def painting_time_A : Nat := 15
def painting_time_B : Nat := 8
def painting_time_C : Nat := 14

theorem min_restoration_time : 
  (shaping_time_A + painting_time_A + painting_time_C + painting_time_B) = 46 := by
  sorry

end min_restoration_time_l216_216658


namespace probability_at_least_one_girl_l216_216505

theorem probability_at_least_one_girl 
  (boys girls : ℕ) 
  (total : boys + girls = 7) 
  (combinations_total : ℕ := Nat.choose 7 2) 
  (combinations_boys : ℕ := Nat.choose 4 2) 
  (prob_no_girls : ℚ := combinations_boys / combinations_total) 
  (prob_at_least_one_girl : ℚ := 1 - prob_no_girls) :
  boys = 4 ∧ girls = 3 → prob_at_least_one_girl = 5 / 7 := 
by
  intro h
  cases h
  sorry

end probability_at_least_one_girl_l216_216505


namespace problem_integer_solution_l216_216691

def satisfies_condition (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem problem_integer_solution :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 20200 ∧ satisfies_condition n :=
sorry

end problem_integer_solution_l216_216691


namespace total_barking_dogs_eq_l216_216210

-- Definitions
def initial_barking_dogs : ℕ := 30
def additional_barking_dogs : ℕ := 10

-- Theorem to prove the total number of barking dogs
theorem total_barking_dogs_eq :
  initial_barking_dogs + additional_barking_dogs = 40 :=
by
  sorry

end total_barking_dogs_eq_l216_216210


namespace correct_operator_is_subtraction_l216_216517

theorem correct_operator_is_subtraction :
  (8 - 2) + 5 * (3 - 2) = 11 :=
by
  sorry

end correct_operator_is_subtraction_l216_216517


namespace sammy_remaining_problems_l216_216314

variable (total_problems : Nat)
variable (fraction_problems : Nat) (decimal_problems : Nat) (multiplication_problems : Nat) (division_problems : Nat)
variable (completed_fraction_problems : Nat) (completed_decimal_problems : Nat)
variable (completed_multiplication_problems : Nat) (completed_division_problems : Nat)
variable (remaining_problems : Nat)

theorem sammy_remaining_problems
  (h₁ : total_problems = 115)
  (h₂ : fraction_problems = 35)
  (h₃ : decimal_problems = 40)
  (h₄ : multiplication_problems = 20)
  (h₅ : division_problems = 20)
  (h₆ : completed_fraction_problems = 11)
  (h₇ : completed_decimal_problems = 17)
  (h₈ : completed_multiplication_problems = 9)
  (h₉ : completed_division_problems = 5)
  (h₁₀ : remaining_problems =
    fraction_problems - completed_fraction_problems +
    decimal_problems - completed_decimal_problems +
    multiplication_problems - completed_multiplication_problems +
    division_problems - completed_division_problems) :
  remaining_problems = 73 :=
  by
    -- proof to be written
    sorry

end sammy_remaining_problems_l216_216314


namespace systematic_sampling_l216_216670

-- Define the conditions
def total_products : ℕ := 100
def selected_products (n : ℕ) : ℕ := 3 + 10 * n
def is_systematic (f : ℕ → ℕ) : Prop :=
  ∃ k b, ∀ n, f n = b + k * n

-- Theorem to prove that the selection method is systematic sampling
theorem systematic_sampling : is_systematic selected_products :=
  sorry

end systematic_sampling_l216_216670


namespace parallel_lines_coefficient_l216_216265

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 2 * y + 2 = 0) → (3 * x - y - 2 = 0)) → a = -6 :=
  by
    sorry

end parallel_lines_coefficient_l216_216265


namespace area_of_figure_l216_216110

noncomputable def area_enclosed : ℝ :=
  ∫ x in (0 : ℝ)..(2 * Real.pi / 3), 2 * Real.sin x

theorem area_of_figure :
  area_enclosed = 3 := by
  sorry

end area_of_figure_l216_216110


namespace find_m_l216_216116

theorem find_m (m : ℤ) (h1 : -180 ≤ m ∧ m ≤ 180) (h2 : Real.sin (m * Real.pi / 180) = Real.cos (810 * Real.pi / 180)) :
  m = 0 ∨ m = 180 :=
sorry

end find_m_l216_216116


namespace merchant_discount_l216_216964

-- Definitions based on conditions
def original_price : ℝ := 1
def increased_price : ℝ := original_price * 1.2
def final_price : ℝ := increased_price * 0.8
def actual_discount : ℝ := original_price - final_price

-- The theorem to be proved
theorem merchant_discount : actual_discount = 0.04 :=
by
  -- Proof goes here
  sorry

end merchant_discount_l216_216964


namespace jane_output_increase_l216_216020

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l216_216020


namespace fg_diff_zero_l216_216898

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x + 3

theorem fg_diff_zero (x : ℝ) : f (g x) - g (f x) = 0 :=
by
  sorry

end fg_diff_zero_l216_216898


namespace afternoon_shells_eq_l216_216587

def morning_shells : ℕ := 292
def total_shells : ℕ := 616

theorem afternoon_shells_eq :
  total_shells - morning_shells = 324 := by
  sorry

end afternoon_shells_eq_l216_216587


namespace yellow_dandelions_day_before_yesterday_l216_216960

theorem yellow_dandelions_day_before_yesterday :
  ∀ (yellow_yesterday white_yesterday yellow_today white_today : ℕ),
    yellow_yesterday = 20 →
    white_yesterday = 14 →
    yellow_today = 15 →
    white_today = 11 →
    ∃ yellow_day_before_yesterday : ℕ,
      yellow_day_before_yesterday = white_yesterday + white_today :=
by sorry

end yellow_dandelions_day_before_yesterday_l216_216960


namespace smallest_positive_n_common_factor_l216_216621

open Int

theorem smallest_positive_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ gcd (11 * n - 3) (8 * n + 4) > 1 ∧ n = 5 :=
by
  sorry

end smallest_positive_n_common_factor_l216_216621


namespace total_volume_of_cubes_l216_216197

theorem total_volume_of_cubes {n : ℕ} (h_n : n = 5) (s : ℕ) (h_s : s = 5) :
  n * (s^3) = 625 :=
by {
  rw [h_n, h_s],
  norm_num,
  sorry
}

end total_volume_of_cubes_l216_216197


namespace track_and_field_analysis_l216_216227

theorem track_and_field_analysis :
  let male_athletes := 12
  let female_athletes := 8
  let tallest_height := 190
  let shortest_height := 160
  let avg_male_height := 175
  let avg_female_height := 165
  let total_athletes := male_athletes + female_athletes
  let sample_size := 10
  let prob_selected := 1 / 2
  let prop_male := male_athletes / total_athletes * sample_size
  let prop_female := female_athletes / total_athletes * sample_size
  let overall_avg_height := (male_athletes / total_athletes) * avg_male_height + (female_athletes / total_athletes) * avg_female_height
  (tallest_height - shortest_height = 30) ∧
  (sample_size / total_athletes = prob_selected) ∧
  (prop_male = 6 ∧ prop_female = 4) ∧
  (overall_avg_height = 171) →
  (A = true ∧ B = true ∧ C = false ∧ D = true) :=
by
  sorry

end track_and_field_analysis_l216_216227


namespace unique_arrangements_of_BANANA_l216_216387

-- Define the conditions as separate definitions in Lean 4
def word := "BANANA"
def total_letters := 6
def count_A := 3
def count_N := 2
def count_B := 1

-- State the theorem to be proven
theorem unique_arrangements_of_BANANA : 
  (total_letters.factorial) / (count_A.factorial * count_N.factorial * count_B.factorial) = 60 := 
by
  sorry

end unique_arrangements_of_BANANA_l216_216387


namespace sequence_tuple_l216_216754

/-- Prove the unique solution to the system of equations derived from the sequence pattern. -/
theorem sequence_tuple (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 7) : (x, y) = (8, 1) :=
by
  sorry

end sequence_tuple_l216_216754


namespace solve_eq_l216_216928

theorem solve_eq {x : ℝ} (h : x * (x - 1) = x) : x = 0 ∨ x = 2 := 
by {
    sorry
}

end solve_eq_l216_216928


namespace andrew_daily_work_hours_l216_216841

theorem andrew_daily_work_hours (total_hours : ℝ) (days : ℝ) (h1 : total_hours = 7.5) (h2 : days = 3) : total_hours / days = 2.5 :=
by
  rw [h1, h2]
  norm_num

end andrew_daily_work_hours_l216_216841


namespace percentage_correct_l216_216063

noncomputable def part : ℝ := 172.8
noncomputable def whole : ℝ := 450.0
noncomputable def percentage (part whole : ℝ) := (part / whole) * 100

theorem percentage_correct : percentage part whole = 38.4 := by
  sorry

end percentage_correct_l216_216063


namespace difference_students_l216_216766

variables {A B AB : ℕ}

theorem difference_students (h1 : A + AB + B = 800)
  (h2 : AB = 20 * (A + AB) / 100)
  (h3 : AB = 25 * (B + AB) / 100) :
  A - B = 100 :=
sorry

end difference_students_l216_216766


namespace number_of_pizza_varieties_l216_216678

-- Definitions for the problem conditions
def number_of_flavors : Nat := 8
def toppings : List String := ["C", "M", "O", "J", "L"]

-- Function to count valid combinations of toppings
def valid_combinations (n : Nat) : Nat :=
  match n with
  | 1 => 5
  | 2 => 10 - 1 -- Subtracting the invalid combination (O, J)
  | 3 => 10 - 3 -- Subtracting the 3 invalid combinations containing (O, J)
  | _ => 0

def total_topping_combinations : Nat :=
  valid_combinations 1 + valid_combinations 2 + valid_combinations 3

-- The final proof stating the number of pizza varieties
theorem number_of_pizza_varieties : total_topping_combinations * number_of_flavors = 168 := by
  -- Calculation steps can be inserted here, we use sorry for now
  sorry

end number_of_pizza_varieties_l216_216678


namespace smallest_possible_value_of_n_l216_216628

theorem smallest_possible_value_of_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 45) : n = 1080 :=
by
  sorry

end smallest_possible_value_of_n_l216_216628


namespace circle_center_radius_1_circle_center_coordinates_radius_1_l216_216500

theorem circle_center_radius_1 (x y : ℝ) : 
  x^2 + y^2 + 2*x - 4*y - 3 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 8 :=
sorry

theorem circle_center_coordinates_radius_1 : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y - 3 = 0 ∧ (x, y) = (-1, 2)) ∧ 
  (∃ r : ℝ, r = 2*Real.sqrt 2) :=
sorry

end circle_center_radius_1_circle_center_coordinates_radius_1_l216_216500


namespace emmalyn_earnings_l216_216398

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l216_216398


namespace price_reduction_l216_216229

theorem price_reduction (x : ℝ) : 
  188 * (1 - x) ^ 2 = 108 :=
sorry

end price_reduction_l216_216229


namespace negation_proof_l216_216790

theorem negation_proof : ¬ (∃ x : ℝ, (x ≤ -1) ∨ (x ≥ 2)) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 := 
by 
  -- proof skipped
  sorry

end negation_proof_l216_216790


namespace solve_for_x_l216_216714

-- Definition of the operation
def otimes (a b : ℝ) : ℝ := a^2 + b^2 - a * b

-- The mathematical statement to be proved
theorem solve_for_x (x : ℝ) (h : otimes x (x - 1) = 3) : x = 2 ∨ x = -1 := 
by 
  sorry

end solve_for_x_l216_216714


namespace Nell_initial_cards_l216_216166

theorem Nell_initial_cards 
  (cards_given : ℕ)
  (cards_left : ℕ)
  (cards_given_eq : cards_given = 301)
  (cards_left_eq : cards_left = 154) :
  cards_given + cards_left = 455 := by
sorry

end Nell_initial_cards_l216_216166


namespace percentage_of_invalid_votes_l216_216150

theorem percentage_of_invalid_votes:
  ∃ (A B V I VV : ℕ), 
    V = 5720 ∧
    B = 1859 ∧
    A = B + 15 / 100 * V ∧
    VV = A + B ∧
    V = VV + I ∧
    (I: ℚ) / V * 100 = 20 :=
by
  sorry

end percentage_of_invalid_votes_l216_216150


namespace abc_sum_l216_216278

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l216_216278


namespace cooler_capacity_l216_216830

theorem cooler_capacity (linemen: ℕ) (linemen_drink: ℕ) 
                        (skill_position: ℕ) (skill_position_drink: ℕ) 
                        (linemen_count: ℕ) (skill_position_count: ℕ) 
                        (skill_wait: ℕ) 
                        (h1: linemen_count = 12) 
                        (h2: linemen_drink = 8) 
                        (h3: skill_position_count = 10) 
                        (h4: skill_position_drink = 6) 
                        (h5: skill_wait = 5):
 linemen_count * linemen_drink + skill_wait * skill_position_drink = 126 :=
by
  sorry

end cooler_capacity_l216_216830


namespace problem1_solution_l216_216839

theorem problem1_solution (x : ℝ) :
  x^2 + 2 * x + 4 * real.sqrt (x^2 + 2 * x) - 5 = 0 →
  x = real.sqrt 2 - 1 ∨ x = -real.sqrt 2 - 1 :=
sorry

end problem1_solution_l216_216839


namespace integer_root_of_polynomial_l216_216791

/-- Prove that -6 is a root of the polynomial equation x^3 + bx + c = 0,
    where b and c are rational numbers and 3 - sqrt(5) is a root
 -/
theorem integer_root_of_polynomial (b c : ℚ)
  (h : ∀ x : ℝ, (x^3 + (b : ℝ)*x + (c : ℝ) = 0) → x = (3 - Real.sqrt 5) ∨ x = (3 + Real.sqrt 5) ∨ x = -6) :
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6 :=
by
  sorry

end integer_root_of_polynomial_l216_216791


namespace neg_neg_eq_l216_216643

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l216_216643


namespace product_mod_9_l216_216925

theorem product_mod_9 (a b c : ℕ) (h1 : a % 6 = 2) (h2 : b % 7 = 3) (h3 : c % 8 = 4) : (a * b * c) % 9 = 6 :=
by
  sorry

end product_mod_9_l216_216925


namespace four_digit_numbers_with_3_or_7_l216_216728

theorem four_digit_numbers_with_3_or_7 : 
  let total_four_digit_numbers := 9000
  let numbers_without_3_or_7 := 3584
  total_four_digit_numbers - numbers_without_3_or_7 = 5416 :=
by
  trivial

end four_digit_numbers_with_3_or_7_l216_216728


namespace committee_problem_solution_l216_216957

def committee_problem : Prop :=
  let total_committees := Nat.choose 15 5
  let zero_profs_committees := Nat.choose 8 5
  let one_prof_committees := (Nat.choose 7 1) * (Nat.choose 8 4)
  let undesirable_committees := zero_profs_committees + one_prof_committees
  let desired_committees := total_committees - undesirable_committees
  desired_committees = 2457

theorem committee_problem_solution : committee_problem :=
by
  sorry

end committee_problem_solution_l216_216957


namespace geom_seq_product_l216_216888

theorem geom_seq_product (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_prod : a 5 * a 14 = 5) :
  a 8 * a 9 * a 10 * a 11 = 10 := 
sorry

end geom_seq_product_l216_216888


namespace volume_region_inequality_l216_216529

theorem volume_region_inequality : 
  ∃ (V : ℝ), V = (20 / 3) ∧ 
    ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 4 
    → x^2 + y^2 + z^2 ≤ V :=
sorry

end volume_region_inequality_l216_216529


namespace total_cost_of_commodities_l216_216326

theorem total_cost_of_commodities (a b : ℕ) (h₁ : a = 477) (h₂ : a - b = 127) : a + b = 827 :=
by
  sorry

end total_cost_of_commodities_l216_216326


namespace max_a_for_f_l216_216257

theorem max_a_for_f :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |a * x^2 - a * x + 1| ≤ 1) → a ≤ 8 :=
sorry

end max_a_for_f_l216_216257


namespace Kim_drink_amount_l216_216056

namespace MathProof

-- Define the conditions
variable (milk_initial t_drinks k_drinks : ℚ)
variable (H1 : milk_initial = 3/4)
variable (H2 : t_drinks = 1/3 * milk_initial)
variable (H3 : k_drinks = 1/2 * (milk_initial - t_drinks))

-- Theorem statement
theorem Kim_drink_amount : k_drinks = 1/4 :=
by
  sorry -- Proof steps would go here, but we're just setting up the statement

end MathProof

end Kim_drink_amount_l216_216056


namespace find_angle_A_triangle_area_l216_216743

variable {A B C : Real} -- Angles
variable {a b c : Real} -- Sides

-- Problem A
theorem find_angle_A (h1 : (a - b) / c = (Real.sin B + Real.sin C) / (Real.sin B + Real.sin A)) :
  A = 2 * Real.pi / 3 :=
sorry

-- Problem B
theorem triangle_area (h1 : a = Real.sqrt 7)
                     (h2 : b = 2 * c)
                     (h3 : A = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end find_angle_A_triangle_area_l216_216743


namespace find_x_l216_216861

-- Let x be a real number such that x > 0 and the area of the given triangle is 180.
theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : 3 * x^2 = 180) : x = 2 * Real.sqrt 15 :=
by
  -- Placeholder for the actual proof
  sorry

end find_x_l216_216861


namespace parallelogram_base_length_l216_216527

theorem parallelogram_base_length :
  ∀ (A H : ℝ), (A = 480) → (H = 15) → (A = Base * H) → (Base = 32) := 
by 
  intros A H hA hH hArea 
  sorry

end parallelogram_base_length_l216_216527


namespace installation_cost_l216_216311

theorem installation_cost (P I : ℝ) (h₁ : 0.80 * P = 12500)
  (h₂ : 18400 = 1.15 * (12500 + 125 + I)) :
  I = 3375 :=
by
  sorry

end installation_cost_l216_216311


namespace price_after_9_years_l216_216662

-- Assume the initial conditions
def initial_price : ℝ := 640
def decrease_factor : ℝ := 0.75
def years : ℕ := 9
def period : ℕ := 3

-- Define the function to calculate the price after a certain number of years, given the period and decrease factor
def price_after_years (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_price * (decrease_factor ^ (years / period))

-- State the theorem that we intend to prove
theorem price_after_9_years : price_after_years initial_price decrease_factor 9 period = 270 := by
  sorry

end price_after_9_years_l216_216662


namespace average_age_of_three_l216_216158

theorem average_age_of_three (Tonya_age John_age Mary_age : ℕ)
  (h1 : John_age = 2 * Mary_age)
  (h2 : Tonya_age = 2 * John_age)
  (h3 : Tonya_age = 60) :
  (Tonya_age + John_age + Mary_age) / 3 = 35 := by
  sorry

end average_age_of_three_l216_216158


namespace veronica_pitting_time_is_2_hours_l216_216935

def veronica_cherries_pitting_time (pounds : ℕ) (cherries_per_pound : ℕ) (minutes_per_20_cherries : ℕ) :=
  let cherries := pounds * cherries_per_pound
  let sets := cherries / 20
  let total_minutes := sets * minutes_per_20_cherries
  total_minutes / 60

theorem veronica_pitting_time_is_2_hours : 
  veronica_cherries_pitting_time 3 80 10 = 2 :=
  by
    sorry

end veronica_pitting_time_is_2_hours_l216_216935


namespace max_value_of_f_in_interval_l216_216129

noncomputable def f (x m : ℝ) : ℝ := -x^3 + 3 * x^2 + m

theorem max_value_of_f_in_interval (m : ℝ) (h₁ : ∀ x ∈ [-2, 2], - x^3 + 3 * x^2 + m ≥ 1) : 
  ∃ x ∈ [-2, 2], f x m = 21 :=
by
  sorry

end max_value_of_f_in_interval_l216_216129


namespace emmalyn_earnings_l216_216394

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l216_216394


namespace no_other_distinct_prime_products_l216_216061

theorem no_other_distinct_prime_products :
  ∀ (q1 q2 q3 : Nat), 
  Prime q1 ∧ Prime q2 ∧ Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ q1 * q2 * q3 ≠ 17 * 11 * 23 → 
  q1 + q2 + q3 ≠ 51 :=
by
  intros q1 q2 q3 h
  sorry

end no_other_distinct_prime_products_l216_216061


namespace minimal_flights_in_complete_graph_l216_216906

-- Definitions based on the conditions
variables {n : ℕ} (K_n : Graph ℕ)

-- Given condition statements
def complete_graph (G : Graph ℕ) : Prop :=
  ∀ (v w : G.V), v ≠ w → G.adj v w

-- Proof problem statement
theorem minimal_flights_in_complete_graph (n : ℕ)
  (G : Graph ℕ) (h_complete : complete_graph G)
  (h_graph : G = (graph_complete n)) :
  minimal_flights G = 0 :=
sorry

end minimal_flights_in_complete_graph_l216_216906


namespace distance_between_points_l216_216724

theorem distance_between_points {A B : ℝ}
  (hA : abs A = 3)
  (hB : abs B = 9) :
  abs (A - B) = 6 ∨ abs (A - B) = 12 :=
sorry

end distance_between_points_l216_216724


namespace at_least_one_nonnegative_l216_216562

theorem at_least_one_nonnegative (x : ℝ) (a b : ℝ) (h1 : a = x^2 - 1) (h2 : b = 4 * x + 5) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end at_least_one_nonnegative_l216_216562


namespace max_value_output_l216_216558

theorem max_value_output (a b c : ℝ) (h_a : a = 3) (h_b : b = 7) (h_c : c = 2) : max (max a b) c = 7 := 
by
  sorry

end max_value_output_l216_216558


namespace multiplier_of_difference_l216_216385

variable (x y : ℕ)
variable (h : x + y = 49) (h1 : x > y)

theorem multiplier_of_difference (h2 : x^2 - y^2 = k * (x - y)) : k = 49 :=
by sorry

end multiplier_of_difference_l216_216385


namespace cos_beta_value_l216_216415

-- Definitions and assumptions
variable (α β : ℝ)
variable (h_alpha_acute : 0 < α ∧ α < π / 2)
variable (h_beta_acute : 0 < β ∧ β < π / 2)
variable (h_cos_sum : Real.cos (α + β) = 3 / 5)
variable (h_sin_alpha : Real.sin α = 5 / 13)

-- Statement of the theorem
theorem cos_beta_value :
  Real.cos β = 56 / 65 :=
  sorry

end cos_beta_value_l216_216415


namespace probability_log_condition_correct_l216_216038

noncomputable def probability_log_condition
  (x y : ℕ)
  (hx : x ∈ {1, 2, 3, 4, 5, 6})
  (hy : y ∈ {1, 2, 3, 4, 5, 6})
  (h : (∀ x y, log (2 * x) y = 1 → y = 2 * x)) : ℚ :=
let possible_outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} } in
let favorable_outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ log (2 * x) y = 1 } in
favorable_outcomes.card / possible_outcomes.card

theorem probability_log_condition_correct
  : probability_log_condition = 1 / 12 :=
sorry

end probability_log_condition_correct_l216_216038


namespace whole_numbers_between_cuberoots_l216_216141

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l216_216141


namespace acute_angle_sine_diff_l216_216263

theorem acute_angle_sine_diff (α β : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : Real.sin α = (Real.sqrt 5) / 5) (h₃ : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 :=
sorry

end acute_angle_sine_diff_l216_216263


namespace number_of_ways_to_distribute_balls_l216_216770

theorem number_of_ways_to_distribute_balls : 
  ∃ n : ℕ, n = 81 ∧ n = 3^4 := 
by sorry

end number_of_ways_to_distribute_balls_l216_216770


namespace product_of_roots_is_12_l216_216058

theorem product_of_roots_is_12 :
  (81 ^ (1 / 4) * 8 ^ (1 / 3) * 4 ^ (1 / 2)) = 12 := by
  sorry

end product_of_roots_is_12_l216_216058


namespace container_volume_ratio_l216_216085

theorem container_volume_ratio
  (A B : ℚ)
  (H1 : 3/5 * A + 1/4 * B = 4/5 * B)
  (H2 : 3/5 * A = (4/5 * B - 1/4 * B)) :
  A / B = 11 / 12 :=
by
  sorry

end container_volume_ratio_l216_216085


namespace probability_three_correct_deliveries_l216_216713

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l216_216713


namespace circle_symmetry_l216_216787

def initial_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 1 = 0

def standard_form_eq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

def symmetric_circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

theorem circle_symmetry :
  (∀ x y : ℝ, initial_circle_eq x y ↔ standard_form_eq x y) →
  (∀ x y : ℝ, standard_form_eq x y → symmetric_circle_eq (-x) (-y)) →
  ∀ x y : ℝ, initial_circle_eq x y → symmetric_circle_eq x y :=
by
  intros h1 h2 x y hxy
  sorry

end circle_symmetry_l216_216787


namespace find_first_number_l216_216921

theorem find_first_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 60 + 35) / 3 + 5 → 
  x = 10 := 
by 
  sorry

end find_first_number_l216_216921


namespace percent_increase_output_per_hour_l216_216005

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l216_216005


namespace harriet_current_age_l216_216575

-- Definitions from the conditions in a)
def mother_age : ℕ := 60
def peter_current_age : ℕ := mother_age / 2
def peter_age_in_four_years : ℕ := peter_current_age + 4
def harriet_age_in_four_years : ℕ := peter_age_in_four_years / 2

-- Proof statement
theorem harriet_current_age : harriet_age_in_four_years - 4 = 13 :=
by
  -- from the given conditions and the solution steps
  let h_current_age := harriet_age_in_four_years - 4
  have : h_current_age = (peter_age_in_four_years / 2) - 4 := by sorry
  have : peter_age_in_four_years = 34 := by sorry
  have : harriet_age_in_four_years = 17 := by sorry
  show 17 - 4 = 13 from sorry

end harriet_current_age_l216_216575


namespace masha_nonnegative_l216_216768

theorem masha_nonnegative (a b c d : ℝ) (h1 : a + b = c * d) (h2 : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := 
by
  -- Proof is omitted
  sorry

end masha_nonnegative_l216_216768


namespace sum_of_acute_angles_l216_216416

theorem sum_of_acute_angles (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : β > 0 ∧ β < π / 2) (h3: γ > 0 ∧ γ < π / 2) (h4 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) :
  (3 * π / 4) < α + β + γ ∧ α + β + γ < π :=
by
  sorry

end sum_of_acute_angles_l216_216416


namespace regular_polygon_sides_l216_216573

theorem regular_polygon_sides (n : ℕ) (h : 108 = 180 * (n - 2) / n) : n = 5 := 
sorry

end regular_polygon_sides_l216_216573


namespace range_of_a1_l216_216719

theorem range_of_a1 {a : ℕ → ℝ} (h_seq : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h_a1_positive : a 1 > 0) :
  (0 < a 1) ∧ (a 1 < 1) ↔ ∀ m n : ℕ, m < n → a m < a n := by
  sorry

end range_of_a1_l216_216719


namespace lumberjack_question_l216_216963

def logs_per_tree (total_firewood : ℕ) (firewood_per_log : ℕ) (trees_chopped : ℕ) : ℕ :=
  total_firewood / firewood_per_log / trees_chopped

theorem lumberjack_question : logs_per_tree 500 5 25 = 4 := by
  sorry

end lumberjack_question_l216_216963


namespace prime_product_sum_91_l216_216049

theorem prime_product_sum_91 (p1 p2 : ℕ) (h1 : Nat.Prime p1) (h2 : Nat.Prime p2) (h3 : p1 + p2 = 91) : p1 * p2 = 178 :=
sorry

end prime_product_sum_91_l216_216049


namespace solve_eq1_solve_eq2_l216_216605

theorem solve_eq1 (x : ℝ) : 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : x^2 + 4 * x + 3 = 0 ↔ x = -3 ∨ x = -1 :=
by
  sorry

end solve_eq1_solve_eq2_l216_216605


namespace larger_of_two_numbers_l216_216476

open Real

theorem larger_of_two_numbers : ∃ x y : ℝ, x + y = 60 ∧ x * y = 882 ∧ x > y ∧ x = 30 + 3 * sqrt 2 :=
begin
  sorry
end

end larger_of_two_numbers_l216_216476


namespace find_t_l216_216426

variables (V V₀ g a S t : ℝ)

-- Conditions
axiom eq1 : V = 3 * g * t + V₀
axiom eq2 : S = (3 / 2) * g * t^2 + V₀ * t + (1 / 2) * a * t^2

-- Theorem to prove
theorem find_t : t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by
  sorry

end find_t_l216_216426


namespace tens_place_of_8_pow_1234_l216_216483

theorem tens_place_of_8_pow_1234 : (8^1234 / 10) % 10 = 0 := by
  sorry

end tens_place_of_8_pow_1234_l216_216483


namespace volume_of_sphere_in_cone_l216_216672

/-- The volume of a sphere inscribed in a right circular cone with
a base diameter of 16 inches and a cross-section with a vertex angle of 45 degrees
is 4096 * sqrt 2 * π / 3 cubic inches. -/
theorem volume_of_sphere_in_cone :
  let d := 16 -- the diameter of the base of the cone in inches
  let angle := 45 -- the vertex angle of the cross-section triangle in degrees
  let r := 8 * Real.sqrt 2 -- the radius of the sphere in inches
  let V := 4 / 3 * Real.pi * r^3 -- the volume of the sphere in cubic inches
  V = 4096 * Real.sqrt 2 * Real.pi / 3 :=
by
  simp only [Real.sqrt]
  sorry -- proof goes here

end volume_of_sphere_in_cone_l216_216672


namespace initial_population_l216_216750

-- Define the initial population
variable (P : ℝ)

-- Define the conditions
theorem initial_population
  (h1 : P * 1.25 * 0.8 * 1.1 * 0.85 * 1.3 + 150 = 25000) :
  P = 24850 :=
by
  sorry

end initial_population_l216_216750


namespace surface_area_of_solid_l216_216245

theorem surface_area_of_solid (num_unit_cubes : ℕ) (top_layer_cubes : ℕ) 
(bottom_layer_cubes : ℕ) (side_layer_cubes : ℕ) 
(front_and_back_cubes : ℕ) (left_and_right_cubes : ℕ) :
  num_unit_cubes = 15 →
  top_layer_cubes = 5 →
  bottom_layer_cubes = 5 →
  side_layer_cubes = 3 →
  front_and_back_cubes = 5 →
  left_and_right_cubes = 3 →
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  total_surface = 26 :=
by
  intros h_n h_t h_b h_s h_f h_lr
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  sorry

end surface_area_of_solid_l216_216245


namespace sum_of_coefficients_l216_216177

noncomputable def polynomial (x : ℝ) : ℝ := x^3 + 3*x^2 - 4*x - 12
noncomputable def simplified_polynomial (x : ℝ) (A B C : ℝ) : ℝ := A*x^2 + B*x + C

theorem sum_of_coefficients : 
  ∃ (A B C D : ℝ), 
    (∀ x ≠ D, simplified_polynomial x A B C = (polynomial x) / (x + 3)) ∧ 
    (A + B + C + D = -6) :=
by
  sorry

end sum_of_coefficients_l216_216177


namespace correct_solutions_l216_216599

theorem correct_solutions (x y z t : ℕ) : 
  (x^2 + t^2) * (z^2 + y^2) = 50 → 
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨ 
  (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨ 
  (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
sorry

end correct_solutions_l216_216599


namespace initial_red_balls_l216_216580

-- Define all the conditions as given in part (a)
variables (R : ℕ)  -- Initial number of red balls
variables (B : ℕ)  -- Number of blue balls
variables (Y : ℕ)  -- Number of yellow balls

-- The conditions
def conditions (R B Y total : ℕ) : Prop :=
  B = 2 * R ∧
  Y = 32 ∧
  total = (R - 6) + B + Y

-- The target statement proving R = 16 given the conditions
theorem initial_red_balls (R: ℕ) (B: ℕ) (Y: ℕ) (total: ℕ) 
  (h : conditions R B Y total): 
  total = 74 → R = 16 :=
by 
  sorry

end initial_red_balls_l216_216580


namespace system_of_equations_solution_l216_216261

theorem system_of_equations_solution (a b x y : ℝ) 
  (h1 : x = 1) 
  (h2 : y = 2)
  (h3 : a * x + y = -1)
  (h4 : 2 * x - b * y = 0) : 
  a + b = -2 := 
sorry

end system_of_equations_solution_l216_216261


namespace probability_three_correct_deliveries_l216_216710

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l216_216710


namespace ratio_of_weights_l216_216025

def initial_weight : ℝ := 2
def weight_after_brownies (w : ℝ) : ℝ := w * 3
def weight_after_more_jelly_beans (w : ℝ) : ℝ := w + 2
def final_weight : ℝ := 16
def weight_before_adding_gummy_worms : ℝ := weight_after_more_jelly_beans (weight_after_brownies initial_weight)

theorem ratio_of_weights :
  final_weight / weight_before_adding_gummy_worms = 2 := 
by
  sorry

end ratio_of_weights_l216_216025


namespace regular_polygon_sides_l216_216434

theorem regular_polygon_sides (n : ℕ) (h₁ : n > 2) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n → True) (h₃ : (360 / n : ℝ) = 30) : n = 12 := by
  sorry

end regular_polygon_sides_l216_216434


namespace cement_percentage_first_concrete_correct_l216_216506

open Real

noncomputable def cement_percentage_of_first_concrete := 
  let total_weight := 4500 
  let cement_percentage := 10.8 / 100
  let weight_each_type := 1125
  let total_cement_weight := cement_percentage * total_weight
  let x := 2.0 / 100
  let y := 21.6 / 100 - x
  (weight_each_type * x + weight_each_type * y = total_cement_weight) →
  (x = 2.0 / 100)

theorem cement_percentage_first_concrete_correct :
  cement_percentage_of_first_concrete := sorry

end cement_percentage_first_concrete_correct_l216_216506


namespace largest_among_five_numbers_l216_216941

theorem largest_among_five_numbers :
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  sorry

end largest_among_five_numbers_l216_216941


namespace function_decreasing_interval_l216_216340

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem function_decreasing_interval :
  ∃ I : Set ℝ, I = (Set.Ioo 0 2) ∧ ∀ x ∈ I, deriv f x < 0 :=
by
  sorry

end function_decreasing_interval_l216_216340


namespace percent_increase_l216_216374

variable (E : ℝ)

-- Given conditions
def enrollment_1992 := 1.20 * E
def enrollment_1993 := 1.26 * E

-- Theorem to prove
theorem percent_increase :
  ((enrollment_1993 E - enrollment_1992 E) / enrollment_1992 E) * 100 = 5 := by
  sorry

end percent_increase_l216_216374


namespace probability_three_correct_deliveries_is_one_sixth_l216_216701

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l216_216701


namespace angle_bisectors_concurrence_l216_216788

noncomputable def acute_triangle (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : Prop :=
  ∡ABC < 90 ∧ ∡BAC < 90 ∧ ∡ACB < 90

theorem angle_bisectors_concurrence 
  {A B C : Type} [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] 
  (ha : acute_triangle A B C)
  (hA1 : ∃ O : Type, is_circumcircle O A B C ∧ O ∈ circle_center A B C)
  (hB1 : ∃ O : Type, is_circumcircle O A B C ∧ O ∈ circle_center B C A)
  (hC1 : ∃ O : Type, is_circumcircle O A B C ∧ O ∈ circle_center C A B) :
  ∃ line : Type, is_bisector line A1 B1 C1 :=
sorry

end angle_bisectors_concurrence_l216_216788


namespace ice_skating_rinks_and_ski_resorts_2019_l216_216321

theorem ice_skating_rinks_and_ski_resorts_2019 (x y : ℕ) :
  x + y = 1230 →
  2 * x + 212 + y + 288 = 2560 →
  x = 830 ∧ y = 400 :=
by {
  sorry
}

end ice_skating_rinks_and_ski_resorts_2019_l216_216321


namespace area_covered_by_AP_l216_216720

noncomputable def area_swept_by_segment (A : ℝ × ℝ) (P : ℝ → ℝ × ℝ) (t1 t2 : ℝ) : ℝ :=
sorry

def A : ℝ × ℝ := (2, 0)

def P (t : ℝ) : ℝ × ℝ :=
  (Real.sin (2 * t - Real.pi / 3), Real.cos (2 * t - Real.pi / 3))

def t1 : ℝ := Real.pi / 12 -- 15 degrees in radians
def t2 : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem area_covered_by_AP : area_swept_by_segment A P t1 t2 = sorry :=
by 
  -- Using t1 and t2 as the bounds
  have h1 : P t1 = (Real.sin ((2 * Real.pi / 12) - (Real.pi / 3)), Real.cos ((2 * Real.pi / 12) - (Real.pi / 3))) := sorry,
  have h2 : P t2 = (Real.sin ((2 * Real.pi / 4) - (Real.pi / 3)), Real.cos ((2 * Real.pi / 4) - (Real.pi / 3))) := sorry,
  sorry

end area_covered_by_AP_l216_216720


namespace clock_confusion_times_l216_216563

-- Conditions translated into Lean definitions
def h_move : ℝ := 0.5  -- hour hand moves at 0.5 degrees per minute
def m_move : ℝ := 6.0  -- minute hand moves at 6 degrees per minute

-- Overlap condition formulated
def overlap_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 10 ∧ 11 * (n : ℝ) = k * 360

-- The final theorem statement in Lean 4
theorem clock_confusion_times : 
  ∃ (count : ℕ), count = 132 ∧ 
    (∀ n < 144, (overlap_condition n → false)) :=
by
  -- Proof to be inserted here
  sorry

end clock_confusion_times_l216_216563


namespace probability_two_red_two_blue_one_green_l216_216693

def total_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_choose_red (r : ℕ) : ℕ := total_ways_to_choose 4 r
def ways_to_choose_blue (b : ℕ) : ℕ := total_ways_to_choose 3 b
def ways_to_choose_green (g : ℕ) : ℕ := total_ways_to_choose 2 g

def successful_outcomes (r b g : ℕ) : ℕ :=
  ways_to_choose_red r * ways_to_choose_blue b * ways_to_choose_green g

def total_outcomes : ℕ := total_ways_to_choose 9 5

def probability_of_selection (r b g : ℕ) : ℚ :=
  (successful_outcomes r b g : ℚ) / (total_outcomes : ℚ)

theorem probability_two_red_two_blue_one_green :
  probability_of_selection 2 2 1 = 2 / 7 := by
  sorry

end probability_two_red_two_blue_one_green_l216_216693


namespace percent_increase_output_per_hour_l216_216006

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l216_216006


namespace evaluate_expression_l216_216849

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-2) = 85 :=
by
  sorry

end evaluate_expression_l216_216849


namespace find_m_n_l216_216160

theorem find_m_n (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_sol : (m + Real.sqrt n)^2 - 10 * (m + Real.sqrt n) + 1 = Real.sqrt (m + Real.sqrt n) * (m + Real.sqrt n + 1)) : m + n = 55 :=
sorry

end find_m_n_l216_216160


namespace scale_model_height_l216_216232

theorem scale_model_height :
  let scale_ratio : ℚ := 1 / 25
  let actual_height : ℚ := 151
  let model_height : ℚ := actual_height * scale_ratio
  round model_height = 6 :=
by
  sorry

end scale_model_height_l216_216232


namespace books_brought_back_l216_216351

def initial_books : ℕ := 235
def taken_out_tuesday : ℕ := 227
def taken_out_friday : ℕ := 35
def books_remaining : ℕ := 29

theorem books_brought_back (B : ℕ) :
  B = 56 ↔ (initial_books - taken_out_tuesday + B - taken_out_friday = books_remaining) :=
by
  -- proof steps would go here
  sorry

end books_brought_back_l216_216351


namespace correct_technology_used_l216_216171

-- Define the condition that the program title is "Back to the Dinosaur Era"
def program_title : String := "Back to the Dinosaur Era"

-- Define the condition that the program vividly recreated various dinosaurs and their living environments
def recreated_living_environments : Bool := true

-- Define the options for digital Earth technologies
inductive DigitalEarthTechnology
| InformationSuperhighway
| HighResolutionSatelliteTechnology
| SpatialInformationTechnology
| VisualizationAndVirtualRealityTechnology

-- Define the correct answer
def correct_technology := DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology

-- The proof problem: Prove that given the conditions, the technology used is the correct one
theorem correct_technology_used
  (title : program_title = "Back to the Dinosaur Era")
  (recreated : recreated_living_environments) :
  correct_technology = DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology :=
by
  sorry

end correct_technology_used_l216_216171


namespace anne_age_ratio_l216_216680

-- Define the given conditions and prove the final ratio
theorem anne_age_ratio (A M : ℕ) (h1 : A = 4 * (A - 4 * M) + M) 
(h2 : A - M = 3 * (A - 4 * M)) : (A : ℚ) / (M : ℚ) = 5.5 := 
sorry

end anne_age_ratio_l216_216680


namespace find_valid_pairs_l216_216420

theorem find_valid_pairs :
  ∃ (a b c : ℕ), 
    (a = 33 ∧ b = 22 ∧ c = 1111) ∨
    (a = 66 ∧ b = 88 ∧ c = 4444) ∨
    (a = 88 ∧ b = 33 ∧ c = 7777) ∧
    (11 ≤ a ∧ a ≤ 99) ∧ (11 ≤ b ∧ b ≤ 99) ∧ (1111 ≤ c ∧ c ≤ 9999) ∧
    (a % 11 = 0) ∧ (b % 11 = 0) ∧ (c % 1111 = 0) ∧
    (a * a + b = c) := sorry

end find_valid_pairs_l216_216420


namespace find_side_c_l216_216579

noncomputable def triangle_side_c (A b S : ℝ) (c : ℝ) : Prop :=
  S = 0.5 * b * c * Real.sin A

theorem find_side_c :
  ∀ (c : ℝ), triangle_side_c (Real.pi / 3) 16 (64 * Real.sqrt 3) c → c = 16 :=
by
  sorry

end find_side_c_l216_216579


namespace average_of_consecutive_integers_l216_216912

variable (c : ℕ)
variable (d : ℕ)

-- Given condition: d == (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7
def condition1 : Prop := d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7

-- The theorem to prove
theorem average_of_consecutive_integers : condition1 c d → 
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7 + d + 8 + d + 9) / 10 = c + 9 :=
sorry

end average_of_consecutive_integers_l216_216912


namespace neg_neg_eq_l216_216645

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l216_216645


namespace system_of_equations_has_integer_solutions_l216_216769

theorem system_of_equations_has_integer_solutions (a b : ℤ) :
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_of_equations_has_integer_solutions_l216_216769


namespace cos2_a_plus_sin2_b_eq_one_l216_216585

variable {a b c : ℝ}

theorem cos2_a_plus_sin2_b_eq_one
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  Real.cos a ^ 2 + Real.sin b ^ 2 = 1 := 
  sorry

end cos2_a_plus_sin2_b_eq_one_l216_216585


namespace intersection_A_B_at_1_range_of_a_l216_216131

-- Problem definitions
def set_A (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def set_B (x a : ℝ) : Prop := x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0

-- Question (I) If a = 1, find A ∩ B
theorem intersection_A_B_at_1 : (∀ x : ℝ, set_A x ∧ set_B x 1 ↔ (1 < x ∧ x ≤ 1 + Real.sqrt 2)) := sorry

-- Question (II) If A ∩ B contains exactly one integer, find the range of a.
theorem range_of_a (h : ∃ x : ℤ, set_A x ∧ set_B x 2) : 3 / 4 ≤ 2 ∧ 2 < 4 / 3 := sorry

end intersection_A_B_at_1_range_of_a_l216_216131


namespace no_rational_satisfies_l216_216596

theorem no_rational_satisfies (a b c d : ℚ) : ¬ ((a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 = 1 + Real.sqrt 3) :=
sorry

end no_rational_satisfies_l216_216596


namespace hypotenuse_length_l216_216342

def triangle_hypotenuse (x : ℝ) (h : ℝ) : Prop :=
  (3 * x - 3)^2 + x^2 = h^2 ∧
  (1 / 2) * x * (3 * x - 3) = 72

theorem hypotenuse_length :
  ∃ (x h : ℝ), triangle_hypotenuse x h ∧ h = Real.sqrt 505 :=
by
  sorry

end hypotenuse_length_l216_216342


namespace coffee_cost_l216_216029

theorem coffee_cost :
  ∃ y : ℕ, 
  (∃ x : ℕ, 3 * x + 2 * y = 630 ∧ 2 * x + 3 * y = 690) → y = 162 :=
by
  sorry

end coffee_cost_l216_216029


namespace hours_per_day_in_deliberation_l216_216895

noncomputable def jury_selection_days : ℕ := 2
noncomputable def trial_days : ℕ := 4 * jury_selection_days
noncomputable def total_deliberation_hours : ℕ := 6 * 24
noncomputable def total_days_on_jury_duty : ℕ := 19

theorem hours_per_day_in_deliberation :
  (total_deliberation_hours / (total_days_on_jury_duty - (jury_selection_days + trial_days))) = 16 :=
by
  sorry

end hours_per_day_in_deliberation_l216_216895


namespace zero_of_f_l216_216350

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ↔ x = 1 :=
by
  sorry

end zero_of_f_l216_216350


namespace max_value_ineq_l216_216260

theorem max_value_ineq (x y : ℝ) (h : x^2 + y^2 = 20) : xy + 8*x + y ≤ 42 := by
  sorry

end max_value_ineq_l216_216260


namespace probability_three_correct_deliveries_l216_216712

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l216_216712


namespace inequality_example_l216_216081

theorem inequality_example (a b m : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_m_pos : 0 < m) (h_ba : b = 2) (h_aa : a = 1) :
  (b + m) / (a + m) < b / a :=
sorry

end inequality_example_l216_216081


namespace max_possible_percentage_l216_216949

theorem max_possible_percentage (p_wi : ℝ) (p_fs : ℝ) (h_wi : p_wi = 0.4) (h_fs : p_fs = 0.7) :
  ∃ p_both : ℝ, p_both = min p_wi p_fs ∧ p_both = 0.4 :=
by
  sorry

end max_possible_percentage_l216_216949


namespace trader_loss_percentage_l216_216815

theorem trader_loss_percentage :
  let SP := 325475
  let gain := 14 / 100
  let loss := 14 / 100
  let CP1 := SP / (1 + gain)
  let CP2 := SP / (1 - loss)
  let TCP := CP1 + CP2
  let TSP := SP + SP
  let profit_or_loss := TSP - TCP
  let profit_or_loss_percentage := (profit_or_loss / TCP) * 100
  profit_or_loss_percentage = -1.958 :=
by
  sorry

end trader_loss_percentage_l216_216815


namespace max_hours_worked_l216_216763

theorem max_hours_worked
  (r : ℝ := 8)  -- Regular hourly rate
  (h_r : ℝ := 20)  -- Hours at regular rate
  (r_o : ℝ := r + 0.25 * r)  -- Overtime hourly rate
  (E : ℝ := 410)  -- Total weekly earnings
  : (h_r + (E - r * h_r) / r_o) = 45 :=
by
  sorry

end max_hours_worked_l216_216763


namespace least_three_digit_with_factors_2_3_5_7_l216_216808

theorem least_three_digit_with_factors_2_3_5_7 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ [2, 3, 5, 7], d ∣ n) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < n → ∀ d ∈ [2, 3, 5, 7], ¬ d ∣ m :=
  ⟨210, by
     apply And.intro _,
     apply And.intro _,
     apply And.intro _,
     iterate 4 { split },
sorry⟩

end least_three_digit_with_factors_2_3_5_7_l216_216808


namespace percent_increase_output_per_hour_l216_216003

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l216_216003


namespace right_triangles_product_hypotenuses_square_l216_216036

/-- 
Given two right triangles T₁ and T₂ with areas 2 and 8 respectively. 
The hypotenuse of T₁ is congruent to one leg of T₂.
The shorter leg of T₁ is congruent to the hypotenuse of T₂.
Prove that the square of the product of the lengths of their hypotenuses is 4624.
-/
theorem right_triangles_product_hypotenuses_square :
  ∃ x y z u : ℝ, 
    (1 / 2) * x * y = 2 ∧
    (1 / 2) * y * u = 8 ∧
    x^2 + y^2 = z^2 ∧
    y^2 + (16 / y)^2 = z^2 ∧ 
    (z^2)^2 = 4624 := 
sorry

end right_triangles_product_hypotenuses_square_l216_216036


namespace central_cell_value_l216_216992

theorem central_cell_value (a1 a2 a3 a4 a5 a6 a7 a8 C : ℕ) 
  (h1 : a1 + a3 + C = 13) (h2 : a2 + a4 + C = 13)
  (h3 : a5 + a7 + C = 13) (h4 : a6 + a8 + C = 13)
  (h5 : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 40) : 
  C = 3 := 
sorry

end central_cell_value_l216_216992


namespace seeds_per_flowerbed_l216_216452

theorem seeds_per_flowerbed (total_seeds : ℕ) (flowerbeds : ℕ) (seeds_per_bed : ℕ) 
  (h1 : total_seeds = 45) (h2 : flowerbeds = 9) 
  (h3 : total_seeds = flowerbeds * seeds_per_bed) : seeds_per_bed = 5 :=
by sorry

end seeds_per_flowerbed_l216_216452


namespace simplify_expression_l216_216915

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a :=
by
  sorry

end simplify_expression_l216_216915


namespace number_of_boys_l216_216477

-- Definitions based on conditions
def students_in_class : ℕ := 30
def cups_brought_total : ℕ := 90
def cups_per_boy : ℕ := 5

-- Definition of boys and girls, with a constraint from the conditions
variable (B : ℕ)
def girls_in_class (B : ℕ) : ℕ := 2 * B

-- Properties from the conditions
axiom h1 : B + girls_in_class B = students_in_class
axiom h2 : B * cups_per_boy = cups_brought_total - (students_in_class - B) * 0 -- Assume no girl brought any cup

-- We state the question as a theorem to be proved
theorem number_of_boys (B : ℕ) : B = 10 :=
by
  sorry

end number_of_boys_l216_216477


namespace expression_equivalence_l216_216982

theorem expression_equivalence :
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 :=
by
  sorry

end expression_equivalence_l216_216982


namespace sum_of_abcd_l216_216163

theorem sum_of_abcd (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : ∀ x, x^2 - 8*a*x - 9*b = 0 → x = c ∨ x = d)
  (h2 : ∀ x, x^2 - 8*c*x - 9*d = 0 → x = a ∨ x = b) :
  a + b + c + d = 648 := sorry

end sum_of_abcd_l216_216163


namespace clare_bought_loaves_l216_216978

-- Define the given conditions
def initial_amount : ℕ := 47
def remaining_amount : ℕ := 35
def cost_per_loaf : ℕ := 2
def cost_per_carton : ℕ := 2
def number_of_cartons : ℕ := 2

-- Required to prove the number of loaves of bread bought by Clare
theorem clare_bought_loaves (initial_amount remaining_amount cost_per_loaf cost_per_carton number_of_cartons : ℕ) 
    (h1 : initial_amount = 47) 
    (h2 : remaining_amount = 35) 
    (h3 : cost_per_loaf = 2) 
    (h4 : cost_per_carton = 2) 
    (h5 : number_of_cartons = 2) : 
    (initial_amount - remaining_amount - cost_per_carton * number_of_cartons) / cost_per_loaf = 4 :=
by sorry

end clare_bought_loaves_l216_216978


namespace correct_calculation_result_l216_216073

theorem correct_calculation_result (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end correct_calculation_result_l216_216073


namespace female_students_in_first_class_l216_216802

theorem female_students_in_first_class
  (females_in_second_class : ℕ)
  (males_in_first_class : ℕ)
  (males_in_second_class : ℕ)
  (males_in_third_class : ℕ)
  (females_in_third_class : ℕ)
  (extra_students : ℕ)
  (total_students_need_partners : ℕ)
  (total_males : ℕ := males_in_first_class + males_in_second_class + males_in_third_class)
  (total_females : ℕ := females_in_second_class + females_in_third_class)
  (females_in_first_class : ℕ)
  (females : ℕ := females_in_first_class + total_females) :
  (females_in_second_class = 18) →
  (males_in_first_class = 17) →
  (males_in_second_class = 14) →
  (males_in_third_class = 15) →
  (females_in_third_class = 17) →
  (extra_students = 2) →
  (total_students_need_partners = total_males - extra_students) →
  females = total_students_need_partners →
  females_in_first_class = 9 :=
by
  intros
  sorry

end female_students_in_first_class_l216_216802


namespace difference_of_interests_l216_216765

def investment_in_funds (X Y : ℝ) (total_investment : ℝ) : ℝ := X + Y
def interest_earned (investment_rate : ℝ) (amount : ℝ) : ℝ := investment_rate * amount

variable (X : ℝ) (Y : ℝ)
variable (total_investment : ℝ) (rate_X : ℝ) (rate_Y : ℝ)
variable (investment_X : ℝ) 

axiom h1 : total_investment = 100000
axiom h2 : rate_X = 0.23
axiom h3 : rate_Y = 0.17
axiom h4 : investment_X = 42000
axiom h5 : investment_in_funds X Y total_investment = total_investment - investment_X

-- We need to show the difference in interest is 200
theorem difference_of_interests : 
  let interest_X := interest_earned rate_X investment_X
  let investment_Y := total_investment - investment_X
  let interest_Y := interest_earned rate_Y investment_Y
  interest_Y - interest_X = 200 :=
by
  sorry

end difference_of_interests_l216_216765


namespace cos_theta_is_correct_l216_216376

def vector_1 : ℝ × ℝ := (4, 5)
def vector_2 : ℝ × ℝ := (2, 7)

noncomputable def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1 * v1.1 + v1.2 * v1.2) * Real.sqrt (v2.1 * v2.1 + v2.2 * v2.2))

theorem cos_theta_is_correct :
  cos_theta vector_1 vector_2 = 43 / (Real.sqrt 41 * Real.sqrt 53) :=
by
  -- proof goes here
  sorry

end cos_theta_is_correct_l216_216376


namespace luncheon_cost_l216_216843

variable (s c p : ℝ)
variable (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
variable (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
variable (eq3 : 4 * s + 8 * c + p = 5.20)

theorem luncheon_cost :
  s + c + p = 1.30 :=
by
  sorry

end luncheon_cost_l216_216843


namespace sequence_general_term_l216_216797

-- Define the sequence
def a : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * a n + 1

-- State the theorem
theorem sequence_general_term (n : ℕ) : a n = 2^n - 1 :=
sorry

end sequence_general_term_l216_216797


namespace find_solutions_l216_216871

theorem find_solutions (k : ℤ) (x y : ℤ) (h : x^2 - 2*y^2 = k) :
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ (t = x + 2*y ∨ t = x - 2*y) ∧ (u = x + y ∨ u = x - y) :=
sorry

end find_solutions_l216_216871


namespace newspaper_target_l216_216316

theorem newspaper_target (total_collected_2_weeks : Nat) (needed_more : Nat) (sections : Nat) (kilos_per_section_2_weeks : Nat)
  (h1 : sections = 6)
  (h2 : kilos_per_section_2_weeks = 280)
  (h3 : total_collected_2_weeks = sections * kilos_per_section_2_weeks)
  (h4 : needed_more = 320)
  : total_collected_2_weeks + needed_more = 2000 :=
by
  sorry

end newspaper_target_l216_216316


namespace matthews_annual_income_l216_216572

noncomputable def annual_income (q : ℝ) (I : ℝ) (T : ℝ) : Prop :=
  T = 0.01 * q * 50000 + 0.01 * (q + 3) * (I - 50000) ∧
  T = 0.01 * (q + 0.5) * I → I = 60000

-- Statement of the math proof
theorem matthews_annual_income (q : ℝ) (T : ℝ) :
  ∃ I : ℝ, I = 60000 ∧ annual_income q I T :=
sorry

end matthews_annual_income_l216_216572


namespace first_student_time_l216_216948

-- Define the conditions
def num_students := 4
def avg_last_three := 35
def avg_all := 30
def total_time_all := num_students * avg_all
def total_time_last_three := (num_students - 1) * avg_last_three

-- State the theorem
theorem first_student_time : (total_time_all - total_time_last_three) = 15 :=
by
  -- Proof is skipped
  sorry

end first_student_time_l216_216948


namespace distance_greater_than_two_l216_216302

theorem distance_greater_than_two (x : ℝ) (h : |x| > 2) : x > 2 ∨ x < -2 :=
sorry

end distance_greater_than_two_l216_216302


namespace xy_relationship_l216_216405

theorem xy_relationship (x y : ℝ) (h : y = 2 * x - 1 - Real.sqrt (y^2 - 2 * x * y + 3 * x - 2)) :
  (x ≠ 1 → y = 2 * x - 1.5) ∧ (x = 1 → y ≤ 1) :=
by
  sorry

end xy_relationship_l216_216405


namespace geom_series_sum_l216_216626

def geom_sum (b1 : ℚ) (r : ℚ) (n : ℕ) : ℚ := 
  b1 * (1 - r^n) / (1 - r)

def b1 : ℚ := 3 / 4
def r : ℚ := 3 / 4
def n : ℕ := 15

theorem geom_series_sum :
  geom_sum b1 r n = 3177884751 / 1073741824 :=
by sorry

end geom_series_sum_l216_216626


namespace isabella_hair_length_l216_216296

def initial_length : ℕ := 18
def growth : ℕ := 6
def final_length : ℕ := initial_length + growth

theorem isabella_hair_length : final_length = 24 :=
by
  simp [final_length, initial_length, growth]
  sorry

end isabella_hair_length_l216_216296


namespace part1_part2_l216_216863

-- Definitions for the sets A and B
def A := {x : ℝ | x^2 - 2 * x - 8 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + a * x + a^2 - 12 = 0}

-- Proof statements
theorem part1 (a : ℝ) : (A ∩ B a = A) → a = -2 :=
by
  sorry

theorem part2 (a : ℝ) : (A ∪ B a = A) → (a ≥ 4 ∨ a < -4 ∨ a = -2) :=
by
  sorry

end part1_part2_l216_216863


namespace expected_profit_correct_l216_216479

-- Define the conditions
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit calculation
def expected_profit : ℝ := (winning_probability * prize) - ticket_cost

-- The theorem we want to prove
theorem expected_profit_correct : expected_profit = -1.5 := by
  sorry

end expected_profit_correct_l216_216479


namespace find_number_l216_216071

theorem find_number (x : ℝ) : (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 :=
  by 
  sorry

end find_number_l216_216071
