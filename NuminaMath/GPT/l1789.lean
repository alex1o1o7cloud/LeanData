import Mathlib

namespace root_and_value_of_a_equation_has_real_roots_l1789_178993

theorem root_and_value_of_a (a : ℝ) (other_root : ℝ) :
  (∃ x : ℝ, x^2 + a * x + a - 1 = 0 ∧ x = 2) → a = -1 ∧ other_root = -1 :=
by sorry

theorem equation_has_real_roots (a : ℝ) :
  ∃ x : ℝ, x^2 + a * x + a - 1 = 0 :=
by sorry

end root_and_value_of_a_equation_has_real_roots_l1789_178993


namespace common_ratio_of_geometric_sequence_l1789_178923

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a n < a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_diff : a 4 - a 3 = 4) : 
  q = 2 := 
sorry

end common_ratio_of_geometric_sequence_l1789_178923


namespace eccentricity_of_hyperbola_l1789_178927

noncomputable def hyperbola_eccentricity : Prop :=
  ∀ (a b : ℝ), a > 0 → b > 0 → (∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  → (∀ (c : ℝ), c^2 = a^2 + b^2) → b = 3 * a → ∃ e : ℝ, e = Real.sqrt 10

-- Statement of the problem without proof (includes the conditions)
theorem eccentricity_of_hyperbola (a b : ℝ) (h : a > 0) (h2 : b > 0) 
  (h3 : ∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  (h4 : ∀ (c : ℝ), c^2 = a^2 + b^2) : hyperbola_eccentricity := 
  sorry

end eccentricity_of_hyperbola_l1789_178927


namespace range_of_a_l1789_178999

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, 2^(2 * x) + 2^x * a + a + 1 = 0) : a ≤ 2 - 2 * Real.sqrt 2 :=
sorry

end range_of_a_l1789_178999


namespace max_value_expression_correct_l1789_178982

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_correct :
  ∃ a b c d : ℝ, a ∈ Set.Icc (-13.5) 13.5 ∧ b ∈ Set.Icc (-13.5) 13.5 ∧ 
                  c ∈ Set.Icc (-13.5) 13.5 ∧ d ∈ Set.Icc (-13.5) 13.5 ∧ 
                  max_value_expression a b c d = 756 := 
sorry

end max_value_expression_correct_l1789_178982


namespace slope_of_parallel_line_l1789_178932

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l1789_178932


namespace find_ab_l1789_178915

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry -- Proof to be provided

end find_ab_l1789_178915


namespace calculate_result_l1789_178961

def multiply (a b : ℕ) : ℕ := a * b
def subtract (a b : ℕ) : ℕ := a - b
def three_fifths (a : ℕ) : ℕ := 3 * a / 5

theorem calculate_result :
  let result := three_fifths (subtract (multiply 12 10) 20)
  result = 60 :=
by
  sorry

end calculate_result_l1789_178961


namespace canoe_row_probability_l1789_178916

-- Definitions based on conditions
def prob_left_works : ℚ := 3 / 5
def prob_right_works : ℚ := 3 / 5

-- The probability that you can still row the canoe
def prob_can_row : ℚ := 
  prob_left_works * prob_right_works +  -- both oars work
  prob_left_works * (1 - prob_right_works) +  -- left works, right breaks
  (1 - prob_left_works) * prob_right_works  -- left breaks, right works
  
theorem canoe_row_probability : prob_can_row = 21 / 25 := by
  -- Skip proof for now
  sorry

end canoe_row_probability_l1789_178916


namespace pure_imaginary_complex_number_solution_l1789_178910

theorem pure_imaginary_complex_number_solution (m : ℝ) :
  (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) → m = 2 :=
by
  sorry

end pure_imaginary_complex_number_solution_l1789_178910


namespace find_b_l1789_178925

/-- Given the distance between the parallel lines l₁ : x - y = 0
  and l₂ : x - y + b = 0 is √2, prove that b = 2 or b = -2. --/
theorem find_b (b : ℝ) (h : ∀ (x y : ℝ), (x - y = 0) → ∀ (x' y' : ℝ), (x' - y' + b = 0) → (|b| / Real.sqrt 2 = Real.sqrt 2)) :
  b = 2 ∨ b = -2 :=
sorry

end find_b_l1789_178925


namespace find_m_l1789_178901

theorem find_m (m : ℝ) (A : Set ℝ) (B : Set ℝ) (hA : A = { -1, 2, 2 * m - 1 }) (hB : B = { 2, m^2 }) (hSubset : B ⊆ A) : m = 1 := by
  sorry
 
end find_m_l1789_178901


namespace find_x4_y4_z4_l1789_178947

theorem find_x4_y4_z4
  (x y z : ℝ)
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59 / 3 :=
by
  sorry

end find_x4_y4_z4_l1789_178947


namespace first_dimension_length_l1789_178976

-- Definitions for conditions
def tank_surface_area (x : ℝ) : ℝ := 14 * x + 20
def cost_per_sqft : ℝ := 20
def total_cost (x : ℝ) : ℝ := (tank_surface_area x) * cost_per_sqft

-- The theorem we need to prove
theorem first_dimension_length : ∃ x : ℝ, total_cost x = 1520 ∧ x = 4 := by 
  sorry

end first_dimension_length_l1789_178976


namespace Murtha_pebbles_problem_l1789_178913

theorem Murtha_pebbles_problem : 
  let a := 3
  let d := 3
  let n := 18
  let a_n := a + (n - 1) * d
  let S_n := n / 2 * (a + a_n)
  S_n = 513 :=
by
  sorry

end Murtha_pebbles_problem_l1789_178913


namespace closest_perfect_square_to_350_l1789_178905

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l1789_178905


namespace alice_reeboks_sold_l1789_178980

theorem alice_reeboks_sold
  (quota : ℝ)
  (price_adidas : ℝ)
  (price_nike : ℝ)
  (price_reeboks : ℝ)
  (num_nike : ℕ)
  (num_adidas : ℕ)
  (excess : ℝ)
  (total_sales_goal : ℝ)
  (total_sales : ℝ)
  (sales_nikes_adidas : ℝ)
  (sales_reeboks : ℝ)
  (num_reeboks : ℕ) :
  quota = 1000 →
  price_adidas = 45 →
  price_nike = 60 →
  price_reeboks = 35 →
  num_nike = 8 →
  num_adidas = 6 →
  excess = 65 →
  total_sales_goal = quota + excess →
  total_sales = 1065 →
  sales_nikes_adidas = price_nike * num_nike + price_adidas * num_adidas →
  sales_reeboks = total_sales - sales_nikes_adidas →
  num_reeboks = sales_reeboks / price_reeboks →
  num_reeboks = 9 :=
by
  intros
  sorry

end alice_reeboks_sold_l1789_178980


namespace average_sales_l1789_178938

-- Define the cost calculation for each special weekend
noncomputable def valentines_day_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20)) / 10

noncomputable def st_patricks_day_sales_per_ticket : Real :=
  ((3 * 2.00) + 6.25 + (8 * 1.00)) / 8

noncomputable def christmas_sales_per_ticket : Real :=
  ((6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 9

-- Define the combined average snack sales
noncomputable def combined_average_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20) + (3 * 2.00) + 6.25 + (8 * 1.00) + (6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 27

-- Proof problem as a Lean theorem
theorem average_sales : 
  valentines_day_sales_per_ticket = 2.62 ∧ 
  st_patricks_day_sales_per_ticket = 2.53 ∧ 
  christmas_sales_per_ticket = 3.16 ∧ 
  combined_average_sales_per_ticket = 2.78 :=
by 
  sorry

end average_sales_l1789_178938


namespace selected_number_in_first_group_is_7_l1789_178965

def N : ℕ := 800
def k : ℕ := 50
def interval : ℕ := N / k
def selected_number : ℕ := 39
def second_group_start : ℕ := 33
def second_group_end : ℕ := 48

theorem selected_number_in_first_group_is_7 
  (h1 : interval = 16)
  (h2 : selected_number ≥ second_group_start ∧ selected_number ≤ second_group_end)
  (h3 : ∃ n, selected_number = second_group_start + interval * n - 1) :
  selected_number % interval = 7 :=
sorry

end selected_number_in_first_group_is_7_l1789_178965


namespace mixed_number_division_l1789_178948

theorem mixed_number_division : 
  let a := 9 / 4
  let b := 3 / 5
  a / b = 15 / 4 :=
by
  sorry

end mixed_number_division_l1789_178948


namespace vector_subtraction_magnitude_l1789_178978

theorem vector_subtraction_magnitude (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 :=
sorry

end vector_subtraction_magnitude_l1789_178978


namespace function_behavior_on_negative_interval_l1789_178940

-- Define the necessary conditions and function properties
variables {f : ℝ → ℝ}

-- Conditions: f is even, increasing on [0, 7], and f(7) = 6
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def f7_eq_6 (f : ℝ → ℝ) : Prop := f 7 = 6

-- The theorem to prove
theorem function_behavior_on_negative_interval (h1 : even_function f) (h2 : increasing_on_interval f 0 7) (h3 : f7_eq_6 f) : 
  (∀ x y, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
sorry

end function_behavior_on_negative_interval_l1789_178940


namespace every_integer_appears_exactly_once_l1789_178946

-- Define the sequence of integers
variable (a : ℕ → ℤ)

-- Define the conditions
axiom infinite_positives : ∀ n : ℕ, ∃ i > n, a i > 0
axiom infinite_negatives : ∀ n : ℕ, ∃ i > n, a i < 0
axiom distinct_remainders : ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → (a i % n) ≠ (a j % n)

-- The proof statement
theorem every_integer_appears_exactly_once :
  ∀ x : ℤ, ∃! i : ℕ, a i = x :=
sorry

end every_integer_appears_exactly_once_l1789_178946


namespace tan_of_cos_neg_five_thirteenth_l1789_178919

variable {α : Real}

theorem tan_of_cos_neg_five_thirteenth (hcos : Real.cos α = -5/13) (hα : π < α ∧ α < 3 * π / 2) : 
  Real.tan α = 12 / 5 := 
sorry

end tan_of_cos_neg_five_thirteenth_l1789_178919


namespace intersection_complement_l1789_178990

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l1789_178990


namespace neg_exists_eq_forall_ne_l1789_178984

variable (x : ℝ)

theorem neg_exists_eq_forall_ne : (¬ ∃ x : ℝ, x^2 - 2 * x = 0) ↔ ∀ x : ℝ, x^2 - 2 * x ≠ 0 := by
  sorry

end neg_exists_eq_forall_ne_l1789_178984


namespace duck_travel_days_l1789_178956

theorem duck_travel_days (x : ℕ) (h1 : 40 + 2 * 40 + x = 180) : x = 60 := by
  sorry

end duck_travel_days_l1789_178956


namespace exam_passing_marks_l1789_178994

theorem exam_passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.40 * T + 10 = P) 
  (h3 : 0.50 * T - 5 = P + 40) : 
  P = 210 := 
sorry

end exam_passing_marks_l1789_178994


namespace sum_of_integers_l1789_178977

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 144) : x + y = 24 :=
sorry

end sum_of_integers_l1789_178977


namespace foma_should_give_ierema_55_coins_l1789_178951

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l1789_178951


namespace find_xyz_l1789_178995

variables (x y z s : ℝ)

theorem find_xyz (h₁ : (x + y + z) * (x * y + x * z + y * z) = 12)
    (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
    (hs : x + y + z = s) : xyz = -8 :=
by
  sorry

end find_xyz_l1789_178995


namespace kishore_expenses_l1789_178918

noncomputable def total_salary (savings : ℕ) (percent : ℝ) : ℝ :=
savings / percent

noncomputable def total_expenses (rent milk groceries education petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + petrol

noncomputable def miscellaneous_expenses (total_salary : ℝ) (total_expenses : ℕ) (savings : ℕ) : ℝ :=
  total_salary - (total_expenses + savings)

theorem kishore_expenses :
  total_salary 2160 0.1 - (total_expenses 5000 1500 4500 2500 2000 + 2160) = 3940 := by
  sorry

end kishore_expenses_l1789_178918


namespace average_speed_correct_l1789_178985

-- Define the speeds for each hour
def speed_hour1 := 90 -- km/h
def speed_hour2 := 40 -- km/h
def speed_hour3 := 60 -- km/h
def speed_hour4 := 80 -- km/h
def speed_hour5 := 50 -- km/h

-- Define the total time of the journey
def total_time := 5 -- hours

-- Calculate the sum of distances
def total_distance := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4 + speed_hour5

-- Define the average speed calculation
def average_speed := total_distance / total_time

-- The proof problem: average speed is 64 km/h
theorem average_speed_correct : average_speed = 64 := by
  sorry

end average_speed_correct_l1789_178985


namespace tens_digit_of_2013_squared_minus_2013_l1789_178974

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 / 10 = 5 := by
  sorry

end tens_digit_of_2013_squared_minus_2013_l1789_178974


namespace cost_to_fill_sandbox_l1789_178991

-- Definitions for conditions
def side_length : ℝ := 3
def volume_per_bag : ℝ := 3
def cost_per_bag : ℝ := 4

-- Theorem statement
theorem cost_to_fill_sandbox : (side_length ^ 3 / volume_per_bag * cost_per_bag) = 36 := by
  sorry

end cost_to_fill_sandbox_l1789_178991


namespace value_of_a_l1789_178926

variable (x y a : ℝ)

-- Conditions
def condition1 : Prop := (x = 1)
def condition2 : Prop := (y = 2)
def condition3 : Prop := (3 * x - a * y = 1)

-- Theorem stating the equivalence between the conditions and the value of 'a'
theorem value_of_a (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 x y a) : a = 1 :=
by
  -- Insert proof here
  sorry

end value_of_a_l1789_178926


namespace interest_percentage_face_value_l1789_178908

def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_percentage_selling_price : ℝ := 0.065

def interest_amount : ℝ := interest_percentage_selling_price * selling_price

theorem interest_percentage_face_value :
  (interest_amount / face_value) * 100 = 8 :=
by
  sorry

end interest_percentage_face_value_l1789_178908


namespace never_prime_l1789_178909

theorem never_prime (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 105) := sorry

end never_prime_l1789_178909


namespace hexagon_perimeter_l1789_178955

-- Define the length of one side of the hexagon
def side_length : ℕ := 5

-- Define the number of sides of a hexagon
def num_sides : ℕ := 6

-- Problem statement: Prove the perimeter of a regular hexagon with the given side length
theorem hexagon_perimeter (s : ℕ) (n : ℕ) : s = side_length ∧ n = num_sides → n * s = 30 :=
by sorry

end hexagon_perimeter_l1789_178955


namespace home_electronics_budget_l1789_178936

theorem home_electronics_budget (deg_ba: ℝ) (b_deg: ℝ) (perc_me: ℝ) (perc_fa: ℝ) (perc_gm: ℝ) (perc_il: ℝ) : 
  deg_ba = 43.2 → 
  b_deg = 360 → 
  perc_me = 12 →
  perc_fa = 15 →
  perc_gm = 29 →
  perc_il = 8 →
  (b_deg / 360 * 100 = 12) → 
  perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100) = 76 →
  100 - (perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100)) = 24 :=
by
  intro h_deg_ba h_b_deg h_perc_me h_perc_fa h_perc_gm h_perc_il h_ba_12perc h_total_76perc
  sorry

end home_electronics_budget_l1789_178936


namespace find_a_for_no_x2_term_l1789_178979

theorem find_a_for_no_x2_term :
  ∀ a : ℝ, (∀ x : ℝ, (3 * x^2 + 2 * a * x + 1) * (-3 * x) - 4 * x^2 = -9 * x^3 + (-6 * a - 4) * x^2 - 3 * x) →
  (¬ ∃ x : ℝ, (-6 * a - 4) * x^2 ≠ 0) →
  a = -2 / 3 :=
by
  intros a h1 h2
  sorry

end find_a_for_no_x2_term_l1789_178979


namespace value_of_a_plus_b_l1789_178902

theorem value_of_a_plus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 1) (h3 : a - b < 0) :
  a + b = -6 ∨ a + b = -4 :=
by
  sorry

end value_of_a_plus_b_l1789_178902


namespace watermelon_seeds_l1789_178939

variable (G Y B : ℕ)

theorem watermelon_seeds (h1 : Y = 3 * G) (h2 : G > B) (h3 : B = 300) (h4 : G + Y + B = 1660) : G = 340 := by
  sorry

end watermelon_seeds_l1789_178939


namespace expression_value_l1789_178963

theorem expression_value : 2 + 3 * 5 + 2 = 19 := by
  sorry

end expression_value_l1789_178963


namespace scale_of_map_l1789_178942

theorem scale_of_map 
  (map_distance : ℝ)
  (travel_time : ℝ)
  (average_speed : ℝ)
  (actual_distance : ℝ)
  (scale : ℝ)
  (h1 : map_distance = 5)
  (h2 : travel_time = 6.5)
  (h3 : average_speed = 60)
  (h4 : actual_distance = average_speed * travel_time)
  (h5 : scale = map_distance / actual_distance) :
  scale = 0.01282 :=
by
  sorry

end scale_of_map_l1789_178942


namespace product_of_two_numbers_l1789_178964

-- Define the conditions
def two_numbers (x y : ℝ) : Prop :=
  x + y = 27 ∧ x - y = 7

-- Define the product function
def product_two_numbers (x y : ℝ) : ℝ := x * y

-- State the theorem
theorem product_of_two_numbers : ∃ x y : ℝ, two_numbers x y ∧ product_two_numbers x y = 170 := by
  sorry

end product_of_two_numbers_l1789_178964


namespace selling_price_of_bracelet_l1789_178934

theorem selling_price_of_bracelet (x : ℝ) 
  (cost_per_bracelet : ℝ) 
  (num_bracelets : ℕ) 
  (box_of_cookies_cost : ℝ) 
  (money_left_after_buying_cookies : ℝ) 
  (total_revenue : ℝ) 
  (total_cost_of_supplies : ℝ) :
  cost_per_bracelet = 1 →
  num_bracelets = 12 →
  box_of_cookies_cost = 3 →
  money_left_after_buying_cookies = 3 →
  total_cost_of_supplies = cost_per_bracelet * num_bracelets →
  total_revenue = 9 →
  x = total_revenue / num_bracelets :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Placeholder for the actual proof
  sorry

end selling_price_of_bracelet_l1789_178934


namespace simplify_fraction_l1789_178958

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by
  sorry

end simplify_fraction_l1789_178958


namespace exchange_rate_change_2014_l1789_178954

theorem exchange_rate_change_2014 :
  let init_rate := 32.6587
  let final_rate := 56.2584
  let change := final_rate - init_rate
  let rounded_change := Float.round change
  rounded_change = 24 :=
by
  sorry

end exchange_rate_change_2014_l1789_178954


namespace intersection_M_N_eq_l1789_178900

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N based on the given inequality
def N : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

-- The statement we want to prove
theorem intersection_M_N_eq {M N: Set ℝ} (hm: M = {0, 1, 2}) 
  (hn: N = {x | x^2 - 3 * x + 2 ≤ 0}) : 
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_eq_l1789_178900


namespace box_inscribed_in_sphere_l1789_178931

theorem box_inscribed_in_sphere (x y z r : ℝ) (surface_area : ℝ)
  (edge_sum : ℝ) (given_x : x = 8) 
  (given_surface_area : surface_area = 432) 
  (given_edge_sum : edge_sum = 104) 
  (surface_area_eq : 2 * (x * y + y * z + z * x) = surface_area)
  (edge_sum_eq : 4 * (x + y + z) = edge_sum) : 
  r = 7 :=
by
  sorry

end box_inscribed_in_sphere_l1789_178931


namespace expression_value_l1789_178930

-- Proving the value of the expression using the factorial and sum formulas
theorem expression_value :
  (Nat.factorial 10) / (10 * 11 / 2) = 66069 := 
sorry

end expression_value_l1789_178930


namespace parallel_if_perp_to_plane_l1789_178971

variable {α m n : Type}

variables (plane : α) (line_m line_n : m)

-- Define what it means for lines to be perpendicular to a plane
def perpendicular_to_plane (line : m) (pl : α) : Prop := sorry

-- Define what it means for lines to be parallel
def parallel (line1 line2 : m) : Prop := sorry

-- The conditions
axiom perp_1 : perpendicular_to_plane line_m plane
axiom perp_2 : perpendicular_to_plane line_n plane

-- The theorem to prove
theorem parallel_if_perp_to_plane : parallel line_m line_n := sorry

end parallel_if_perp_to_plane_l1789_178971


namespace max_k_value_l1789_178968

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∃ k : ℝ, (∀ m, 0 < m → m < 1/2 → (1/m + 2/(1-2*m) ≥ k)) ∧ k = 8) := 
sorry

end max_k_value_l1789_178968


namespace value_of_2a_plus_b_l1789_178992

theorem value_of_2a_plus_b (a b : ℤ) (h1 : |a - 1| = 4) (h2 : |b| = 7) (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := sorry

end value_of_2a_plus_b_l1789_178992


namespace zero_in_interval_l1789_178973

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval (h_mono : ∀ x y, 0 < x → x < y → f x < f y) (h_f2 : f 2 < 0) (h_f3 : 0 < f 3) :
  ∃ x₀ ∈ (Set.Ioo 2 3), f x₀ = 0 :=
by
  sorry

end zero_in_interval_l1789_178973


namespace variance_of_data_is_0_02_l1789_178929

def data : List ℝ := [10.1, 9.8, 10, 9.8, 10.2]

theorem variance_of_data_is_0_02 (h : (10.1 + 9.8 + 10 + 9.8 + 10.2) / 5 = 10) : 
  (1 / 5) * ((10.1 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10.2 - 10) ^ 2) = 0.02 :=
by
  sorry

end variance_of_data_is_0_02_l1789_178929


namespace morning_snowfall_l1789_178997

theorem morning_snowfall (afternoon_snowfall total_snowfall : ℝ) (h₀ : afternoon_snowfall = 0.5) (h₁ : total_snowfall = 0.63):
  total_snowfall - afternoon_snowfall = 0.13 :=
by 
  sorry

end morning_snowfall_l1789_178997


namespace marla_colors_red_squares_l1789_178928

-- Conditions
def total_rows : Nat := 10
def squares_per_row : Nat := 15
def total_squares : Nat := total_rows * squares_per_row

def blue_rows_top : Nat := 2
def blue_rows_bottom : Nat := 2
def total_blue_rows : Nat := blue_rows_top + blue_rows_bottom
def total_blue_squares : Nat := total_blue_rows * squares_per_row

def green_squares : Nat := 66
def red_rows : Nat := 4

-- Theorem to prove 
theorem marla_colors_red_squares : 
  total_squares - total_blue_squares - green_squares = red_rows * 6 :=
by
  sorry -- This skips the proof

end marla_colors_red_squares_l1789_178928


namespace find_smallest_denominator_difference_l1789_178911

theorem find_smallest_denominator_difference :
  ∃ (r s : ℕ), 
    r > 0 ∧ s > 0 ∧ 
    (5 : ℚ) / 11 < r / s ∧ r / s < (4 : ℚ) / 9 ∧ 
    ¬ ∃ t : ℕ, t < s ∧ (5 : ℚ) / 11 < r / t ∧ r / t < (4 : ℚ) / 9 ∧ 
    s - r = 11 := 
sorry

end find_smallest_denominator_difference_l1789_178911


namespace pairs_a_eq_b_l1789_178907

theorem pairs_a_eq_b 
  (n : ℕ) (h_n : ¬ ∃ k : ℕ, k^2 = n) (a b : ℕ) 
  (r : ℝ) (h_r_pos : 0 < r) (h_ra_rational : ∃ q₁ : ℚ, r^a + (n:ℝ)^(1/2) = q₁) 
  (h_rb_rational : ∃ q₂ : ℚ, r^b + (n:ℝ)^(1/2) = q₂) : 
  a = b :=
sorry

end pairs_a_eq_b_l1789_178907


namespace optimal_perimeter_proof_l1789_178944

-- Definition of conditions
def fencing_length : Nat := 400
def min_width : Nat := 50
def area : Nat := 8000

-- Definition of the perimeter to be proven as optimal
def optimal_perimeter : Nat := 360

-- Theorem statement to be proven
theorem optimal_perimeter_proof (l w : Nat) (h1 : l * w = area) (h2 : 2 * l + 2 * w <= fencing_length) (h3 : w >= min_width) :
  2 * l + 2 * w = optimal_perimeter :=
sorry

end optimal_perimeter_proof_l1789_178944


namespace sin_alpha_in_second_quadrant_l1789_178967

theorem sin_alpha_in_second_quadrant
  (α : ℝ)
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.tan α = - (8 / 15)) :
  Real.sin α = 8 / 17 :=
sorry

end sin_alpha_in_second_quadrant_l1789_178967


namespace find_number_l1789_178904

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 105) : x = 15 :=
by
  sorry

end find_number_l1789_178904


namespace collinear_points_eq_sum_l1789_178950

theorem collinear_points_eq_sum (a b : ℝ) :
  -- Collinearity conditions in ℝ³
  (∃ t1 t2 t3 t4 : ℝ,
    (2, a, b) = (a + t1 * (a - 2), 3 + t1 * (b - 3), b + t1 * (4 - b)) ∧
    (a, 3, b) = (a + t2 * (a - 2), 3 + t2 * (b - 3), b + t2 * (4 - b)) ∧
    (a, b, 4) = (a + t3 * (a - 2), 3 + t3 * (b - 3), b + t3 * (4 - b)) ∧
    (5, b, a) = (a + t4 * (a - 2), 3 + t4 * (b - 3), b + t4 * (4 - b))) →
  a + b = 9 :=
by
  sorry

end collinear_points_eq_sum_l1789_178950


namespace cube_root_equation_l1789_178970

theorem cube_root_equation (x : ℝ) (h : (2 * x - 14)^(1/3) = -2) : 2 * x + 3 = 9 := by
  sorry

end cube_root_equation_l1789_178970


namespace percentage_of_girls_l1789_178981

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 900) (h2 : B = 90) :
  (G / (B + G) : ℚ) * 100 = 90 :=
  by
  sorry

end percentage_of_girls_l1789_178981


namespace quadratic_has_real_roots_l1789_178952

open Real

theorem quadratic_has_real_roots (k : ℝ) (h : k ≠ 0) :
    ∃ x : ℝ, x^2 + k * x + k^2 - 1 = 0 ↔
    -2 / sqrt 3 ≤ k ∧ k ≤ 2 / sqrt 3 :=
by
  sorry

end quadratic_has_real_roots_l1789_178952


namespace part1_solution_set_part2_range_of_a_l1789_178962

-- Define the function f
def f (a x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Problem 1: When a = 1, solve the inequality f(x) ≤ 5
theorem part1_solution_set : 
  { x : ℝ | f 1 x ≤ 5 } = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 4} := 
  by 
  sorry

-- Problem 2: Determine the range of a for which f(x) has a minimum
theorem part2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x < 1/3 → f a x ≤ f a 1/3) → 
           (∀ x : ℝ, x ≥ 1/3 → f a x ≥ f a 1/3) ↔ 
           (-3 ≤ a ∧ a ≤ 3) := 
  by
  sorry

end part1_solution_set_part2_range_of_a_l1789_178962


namespace ratio_of_sheep_to_cow_l1789_178988

noncomputable def sheep_to_cow_ratio 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : ℕ × ℕ := 
if h3 : 12 = 0 then (0, 0) else (2, 1)

theorem ratio_of_sheep_to_cow 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : sheep_to_cow_ratio S h1 h2 = (2, 1) := 
sorry

end ratio_of_sheep_to_cow_l1789_178988


namespace find_marks_in_mathematics_l1789_178933

theorem find_marks_in_mathematics
  (english : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (subjects : ℕ)
  (marks_math : ℕ) :
  english = 96 →
  physics = 82 →
  chemistry = 97 →
  biology = 95 →
  average = 93 →
  subjects = 5 →
  (average * subjects = english + marks_math + physics + chemistry + biology) →
  marks_math = 95 :=
  by
    intros h_eng h_phy h_chem h_bio h_avg h_sub h_eq
    rw [h_eng, h_phy, h_chem, h_bio, h_avg, h_sub] at h_eq
    sorry

end find_marks_in_mathematics_l1789_178933


namespace sequence_divisibility_l1789_178912

theorem sequence_divisibility (g : ℕ → ℕ) (h₁ : g 1 = 1) 
(h₂ : ∀ n : ℕ, g (n + 1) = g n ^ 2 + g n + 1) 
(n : ℕ) : g n ^ 2 + 1 ∣ g (n + 1) ^ 2 + 1 :=
sorry

end sequence_divisibility_l1789_178912


namespace triangle_area_inradius_l1789_178903

theorem triangle_area_inradius
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 35)
  (h2 : inradius = 4.5)
  (h3 : area = inradius * (perimeter / 2)) :
  area = 78.75 := by
  sorry

end triangle_area_inradius_l1789_178903


namespace closest_integer_to_a2013_l1789_178921

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 100 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) + (1 / a (n + 1))

theorem closest_integer_to_a2013 (a : ℕ → ℝ) (h : seq a) : abs (a 2013 - 118) < 0.5 :=
sorry

end closest_integer_to_a2013_l1789_178921


namespace intersection_with_x_axis_l1789_178922

noncomputable def f (x : ℝ) : ℝ := 
  (3 * x - 1) * (Real.sqrt (9 * x^2 - 6 * x + 5) + 1) + 
  (2 * x - 3) * (Real.sqrt (4 * x^2 - 12 * x + 13)) + 1

theorem intersection_with_x_axis :
  ∃ x : ℝ, f x = 0 ∧ x = 4 / 5 :=
by
  sorry

end intersection_with_x_axis_l1789_178922


namespace size_relationship_l1789_178959

theorem size_relationship (a b : ℝ) (h₀ : a + b > 0) :
  a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b :=
by
  sorry

end size_relationship_l1789_178959


namespace James_leftover_money_l1789_178987

variable (W : ℝ)
variable (M : ℝ)

theorem James_leftover_money 
  (h1 : M = (W / 2 - 2))
  (h2 : M + 114 = W) : 
  M = 110 := sorry

end James_leftover_money_l1789_178987


namespace max_value_trig_expression_l1789_178957

variable (a b φ θ : ℝ)

theorem max_value_trig_expression :
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + 2 * a * b * Real.sin φ + b^2) := sorry

end max_value_trig_expression_l1789_178957


namespace bus_stoppage_time_l1789_178949

theorem bus_stoppage_time (speed_excl_stoppages speed_incl_stoppages : ℕ) (h1 : speed_excl_stoppages = 54) (h2 : speed_incl_stoppages = 45) : 
  ∃ (t : ℕ), t = 10 := by
  sorry

end bus_stoppage_time_l1789_178949


namespace constructed_expression_equals_original_l1789_178953

variable (a : ℝ)

theorem constructed_expression_equals_original : 
  a ≠ 0 → 
  ((1/a) / ((1/a) * (1/a)) - (1/a)) / (1/a) = (a + 1) * (a - 1) :=
by
  intro h
  sorry

end constructed_expression_equals_original_l1789_178953


namespace eval_g_at_3_l1789_178969

def g (x : ℤ) : ℤ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem eval_g_at_3 : g 3 = 183 := by
  sorry

end eval_g_at_3_l1789_178969


namespace min_value_four_l1789_178986

noncomputable def min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : y > 2 * x) : ℝ :=
  (y^2 - 2 * x * y + x^2) / (x * y - 2 * x^2)

theorem min_value_four (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hy_gt_2x : y > 2 * x) :
  min_value x y hx_pos hy_pos hy_gt_2x = 4 := 
sorry

end min_value_four_l1789_178986


namespace complement_of_M_l1789_178914

open Set

def U : Set ℝ := univ

def M : Set ℝ := { x | x^2 - x ≥ 0 }

theorem complement_of_M :
  compl M = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end complement_of_M_l1789_178914


namespace solve_compound_inequality_l1789_178937

noncomputable def compound_inequality_solution (x : ℝ) : Prop :=
  (3 - (1 / (3 * x + 4)) < 5) ∧ (2 * x + 1 > 0)

theorem solve_compound_inequality (x : ℝ) :
  compound_inequality_solution x ↔ (x > -1/2) :=
by
  sorry

end solve_compound_inequality_l1789_178937


namespace fruit_basket_l1789_178941

theorem fruit_basket :
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  oranges + apples + bananas + peaches = 28 :=
by
  intros oranges apples bananas peaches h_oranges h_apples h_bananas h_peaches
  rw [h_oranges, h_apples, h_bananas, h_peaches]
  sorry

end fruit_basket_l1789_178941


namespace initial_stock_decaf_percentage_l1789_178998

variable (x : ℝ)
variable (initialStock newStock totalStock initialDecaf newDecaf totalDecaf: ℝ)

theorem initial_stock_decaf_percentage :
  initialStock = 400 ->
  newStock = 100 ->
  totalStock = 500 ->
  initialDecaf = initialStock * x / 100 ->
  newDecaf = newStock * 60 / 100 ->
  totalDecaf = 180 ->
  initialDecaf + newDecaf = totalDecaf ->
  x = 30 := by
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇
  sorry

end initial_stock_decaf_percentage_l1789_178998


namespace first_term_of_arithmetic_progression_l1789_178966

theorem first_term_of_arithmetic_progression 
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (nth_term_eq : a + (n - 1) * d = 26)
  (common_diff : d = 2)
  (term_num : n = 10) : 
  a = 8 := 
by 
  sorry

end first_term_of_arithmetic_progression_l1789_178966


namespace isosceles_triangle_base_l1789_178924

theorem isosceles_triangle_base (h_perimeter : 2 * 1.5 + x = 3.74) : x = 0.74 :=
by
  sorry

end isosceles_triangle_base_l1789_178924


namespace eval_g_five_l1789_178960

def g (x : ℝ) : ℝ := 4 * x - 2

theorem eval_g_five : g 5 = 18 := by
  sorry

end eval_g_five_l1789_178960


namespace cube_cut_off_edges_l1789_178943

theorem cube_cut_off_edges :
  let original_edges := 12
  let new_edges_per_vertex := 3
  let vertices := 8
  let new_edges := new_edges_per_vertex * vertices
  (original_edges + new_edges) = 36 :=
by
  sorry

end cube_cut_off_edges_l1789_178943


namespace absolute_value_expression_l1789_178917

theorem absolute_value_expression : 
  (abs ((-abs (-1 + 2))^2 - 1) = 0) :=
sorry

end absolute_value_expression_l1789_178917


namespace original_number_correct_l1789_178906

-- Definitions for the problem conditions
/-
Let N be the original number.
X is the number to be subtracted.
We are given that X = 8.
We need to show that (N - 8) mod 5 = 4, (N - 8) mod 7 = 4, and (N - 8) mod 9 = 4.
-/

-- Declaration of variables
variable (N : ℕ) (X : ℕ)

-- Given conditions
def conditions := (N - X) % 5 = 4 ∧ (N - X) % 7 = 4 ∧ (N - X) % 9 = 4

-- Given the subtracted number X is 8.
def X_val : ℕ := 8

-- Prove that N = 326 meets the conditions
theorem original_number_correct (h : X = X_val) : ∃ N, conditions N X ∧ N = 326 := by
  sorry

end original_number_correct_l1789_178906


namespace option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l1789_178972

noncomputable def triangle (A B C : ℝ) := A + B + C = 180

-- Define the conditions for options A, B, C, and D
def option_a := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = 3 * C
def option_b := ∀ A B C : ℝ, triangle A B C → A + B = C
def option_c := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = (1/2) * C
def option_d := ∀ A B C : ℝ, triangle A B C → ∃ x : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x

-- Define that option A does not form a right triangle
theorem option_a_not_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_a → A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 :=
sorry

-- Check that options B, C, and D do form right triangles
theorem option_b_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_b → C = 90 :=
sorry

theorem option_c_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_c → C = 90 :=
sorry

theorem option_d_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_d → C = 90 :=
sorry

end option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l1789_178972


namespace not_possible_consecutive_results_l1789_178920

theorem not_possible_consecutive_results 
  (dot_counts : ℕ → ℕ)
  (h_identical_conditions : ∀ (i : ℕ), dot_counts i = 1 ∨ dot_counts i = 2 ∨ dot_counts i = 3) 
  (h_correct_dot_distribution : ∀ (i j : ℕ), (i ≠ j → dot_counts i ≠ dot_counts j))
  : ¬ (∃ (consecutive : ℕ → ℕ), 
        (∀ (k : ℕ), k < 6 → consecutive k = dot_counts (4 * k) + dot_counts (4 * k + 1) 
                         + dot_counts (4 * k + 2) + dot_counts (4 * k + 3))
        ∧ (∀ (k : ℕ), k < 5 → consecutive (k + 1) = consecutive k + 1)) := sorry

end not_possible_consecutive_results_l1789_178920


namespace money_lent_to_B_l1789_178989

theorem money_lent_to_B (total_money : ℕ) (interest_A_rate : ℚ) (interest_B_rate : ℚ) (interest_difference : ℚ) (years : ℕ) 
  (x y : ℚ) 
  (h1 : total_money = 10000)
  (h2 : interest_A_rate = 0.15)
  (h3 : interest_B_rate = 0.18)
  (h4 : interest_difference = 360)
  (h5 : years = 2)
  (h6 : y = total_money - x)
  (h7 : ((x * interest_A_rate * years) = ((y * interest_B_rate * years) + interest_difference))) : 
  y = 4000 := 
sorry

end money_lent_to_B_l1789_178989


namespace kevin_total_hops_l1789_178996

/-- Define the hop function for Kevin -/
def hop (remaining_distance : ℚ) : ℚ :=
  remaining_distance / 4

/-- Summing the series for five hops -/
def total_hops (start_distance : ℚ) (hops : ℕ) : ℚ :=
  let h0 := hop start_distance
  let h1 := hop (start_distance - h0)
  let h2 := hop (start_distance - h0 - h1)
  let h3 := hop (start_distance - h0 - h1 - h2)
  let h4 := hop (start_distance - h0 - h1 - h2 - h3)
  h0 + h1 + h2 + h3 + h4

/-- Final proof statement: after five hops from starting distance of 2, total distance hopped should be 1031769/2359296 -/
theorem kevin_total_hops :
  total_hops 2 5 = 1031769 / 2359296 :=
sorry

end kevin_total_hops_l1789_178996


namespace ratio_of_arithmetic_sequence_sums_l1789_178983

-- Definitions of the arithmetic sequences based on the conditions
def numerator_seq (n : ℕ) : ℕ := 3 + (n - 1) * 3
def denominator_seq (m : ℕ) : ℕ := 4 + (m - 1) * 4

-- Definitions of the number of terms based on the conditions
def num_terms_num : ℕ := 32
def num_terms_den : ℕ := 16

-- Definitions of the sums based on the sequences
def sum_numerator_seq : ℕ := (num_terms_num / 2) * (3 + 96)
def sum_denominator_seq : ℕ := (num_terms_den / 2) * (4 + 64)

-- Calculate the ratio of the sums
def ratio_of_sums : ℚ := sum_numerator_seq / sum_denominator_seq

-- Proof statement
theorem ratio_of_arithmetic_sequence_sums : ratio_of_sums = 99 / 34 := by
  sorry

end ratio_of_arithmetic_sequence_sums_l1789_178983


namespace carpenter_material_cost_l1789_178975

theorem carpenter_material_cost (total_estimate hourly_rate num_hours : ℝ) 
    (h1 : total_estimate = 980)
    (h2 : hourly_rate = 28)
    (h3 : num_hours = 15) : 
    total_estimate - hourly_rate * num_hours = 560 := 
by
  sorry

end carpenter_material_cost_l1789_178975


namespace total_expense_l1789_178935

theorem total_expense (tanya_face_cost : ℕ) (tanya_face_qty : ℕ) (tanya_body_cost : ℕ) (tanya_body_qty : ℕ) 
  (tanya_total_expense : ℕ) (christy_multiplier : ℕ) (christy_total_expense : ℕ) (total_expense : ℕ) :
  tanya_face_cost = 50 →
  tanya_face_qty = 2 →
  tanya_body_cost = 60 →
  tanya_body_qty = 4 →
  tanya_total_expense = tanya_face_qty * tanya_face_cost + tanya_body_qty * tanya_body_cost →
  christy_multiplier = 2 →
  christy_total_expense = christy_multiplier * tanya_total_expense →
  total_expense = christy_total_expense + tanya_total_expense →
  total_expense = 1020 :=
by
  intros
  sorry

end total_expense_l1789_178935


namespace rectangle_area_l1789_178945

theorem rectangle_area (AB AC : ℝ) (H1 : AB = 15) (H2 : AC = 17) : 
  ∃ (BC : ℝ), (AB * BC = 120) :=
by
  sorry

end rectangle_area_l1789_178945
