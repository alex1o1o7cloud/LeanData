import Mathlib

namespace cost_difference_l2333_233325

theorem cost_difference (joy_pencils : ℕ) (colleen_pencils : ℕ) 
  (price_per_pencil_joy : ℝ) (price_per_pencil_colleen : ℝ) :
  joy_pencils = 30 →
  colleen_pencils = 50 →
  price_per_pencil_joy = 4 →
  price_per_pencil_colleen = 3.5 →
  (colleen_pencils * price_per_pencil_colleen - joy_pencils * price_per_pencil_joy) = 55 :=
by
  intros h_joy_pencils h_colleen_pencils h_price_joy h_price_colleen
  rw [h_joy_pencils, h_colleen_pencils, h_price_joy, h_price_colleen]
  norm_num
  repeat { sorry }

end cost_difference_l2333_233325


namespace number_of_cyclic_sets_l2333_233331

-- Definition of conditions: number of teams and wins/losses
def num_teams : ℕ := 21
def wins (team : ℕ) : ℕ := 12
def losses (team : ℕ) : ℕ := 8
def played_everyone_once (team1 team2 : ℕ) : Prop := (team1 ≠ team2)

-- Proposition to prove:
theorem number_of_cyclic_sets (h_teams: ∀ t, wins t = 12 ∧ losses t = 8)
  (h_played_once: ∀ t1 t2, played_everyone_once t1 t2) : 
  ∃ n, n = 144 :=
sorry

end number_of_cyclic_sets_l2333_233331


namespace negation_proposition_l2333_233344

variable {f : ℝ → ℝ}

theorem negation_proposition : ¬ (∀ x : ℝ, f x > 0) ↔ ∃ x : ℝ, f x ≤ 0 := by
  sorry

end negation_proposition_l2333_233344


namespace problem_proof_l2333_233342

variable (a b c : ℝ)
noncomputable def a_def : ℝ := Real.exp 0.2
noncomputable def b_def : ℝ := Real.sin 1.2
noncomputable def c_def : ℝ := 1 + Real.log 1.2

theorem problem_proof (ha : a = a_def) (hb : b = b_def) (hc : c = c_def) : b < c ∧ c < a :=
by
  have ha_val : a = Real.exp 0.2 := ha
  have hb_val : b = Real.sin 1.2 := hb
  have hc_val : c = 1 + Real.log 1.2 := hc
  sorry

end problem_proof_l2333_233342


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l2333_233306

def is_sum_of_squares_of_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

def T (x : ℤ) : Prop :=
  ∃ n : ℤ, x = is_sum_of_squares_of_consecutive_integers n

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_5 :
  (∀ x, T x → ¬ (9 ∣ x)) ∧ (∃ y, T y ∧ (5 ∣ y)) :=
by
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l2333_233306


namespace discriminant_square_eq_l2333_233385

variable {a b c x : ℝ}

-- Condition: a ≠ 0
axiom h_a : a ≠ 0

-- Condition: x is a root of the quadratic equation ax^2 + bx + c = 0
axiom h_root : a * x^2 + b * x + c = 0

theorem discriminant_square_eq (h_a : a ≠ 0) (h_root : a * x^2 + b * x + c = 0) :
  (2 * a * x + b)^2 = b^2 - 4 * a * c :=
by 
  sorry

end discriminant_square_eq_l2333_233385


namespace average_distance_one_hour_l2333_233367

theorem average_distance_one_hour (d : ℝ) (t : ℝ) (h1 : d = 100) (h2 : t = 5 / 4) : (d / t) = 80 :=
by
  sorry

end average_distance_one_hour_l2333_233367


namespace people_per_car_l2333_233368

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 3) : total_people / num_cars = 21 :=
by
  sorry

end people_per_car_l2333_233368


namespace right_triangle_area_l2333_233383

-- Define the initial lengths and the area calculation function.
def area_right_triangle (base height : ℕ) : ℕ :=
  (1 / 2) * base * height

theorem right_triangle_area
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a = 18)
  (h2 : b = 24)
  (h3 : c = 30)  -- Derived from the solution steps
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) :
  area_right_triangle a b = 216 :=
sorry

end right_triangle_area_l2333_233383


namespace max_area_2017_2018_l2333_233322

noncomputable def max_area_of_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem max_area_2017_2018 :
  max_area_of_triangle 2017 2018 = 2035133 := by
  sorry

end max_area_2017_2018_l2333_233322


namespace triangle_area_right_angle_l2333_233334

noncomputable def area_of_triangle (AB BC : ℝ) : ℝ :=
  1 / 2 * AB * BC

theorem triangle_area_right_angle (AB BC : ℝ) (hAB : AB = 12) (hBC : BC = 9) :
  area_of_triangle AB BC = 54 := by
  rw [hAB, hBC]
  norm_num
  sorry

end triangle_area_right_angle_l2333_233334


namespace no_solutions_eq_l2333_233375

theorem no_solutions_eq (x y : ℝ) : (x + y)^2 ≠ x^2 + y^2 + 1 :=
by sorry

end no_solutions_eq_l2333_233375


namespace geometric_sequence_term_eq_l2333_233352

theorem geometric_sequence_term_eq (a₁ q : ℝ) (n : ℕ) :
  a₁ = 1 / 2 → q = 1 / 2 → a₁ * q ^ (n - 1) = 1 / 32 → n = 5 :=
by
  intros ha₁ hq han
  sorry

end geometric_sequence_term_eq_l2333_233352


namespace emmy_rosa_ipods_l2333_233354

theorem emmy_rosa_ipods :
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  Emmy_left + Rosa_ipods = 12 :=
by
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  sorry

end emmy_rosa_ipods_l2333_233354


namespace linear_function_does_not_pass_fourth_quadrant_l2333_233396

theorem linear_function_does_not_pass_fourth_quadrant :
  ∀ x, (2 * x + 1 ≥ 0) :=
by sorry

end linear_function_does_not_pass_fourth_quadrant_l2333_233396


namespace seq_increasing_l2333_233378

theorem seq_increasing (n : ℕ) (h : n > 0) : (↑n / (↑n + 2): ℝ) < (↑n + 1) / (↑n + 3) :=
by 
-- Converting ℕ to ℝ to make definitions correct
let an := (↑n / (↑n + 2): ℝ)
let an1 := (↑n + 1) / (↑n + 3)
-- Proof would go here
sorry

end seq_increasing_l2333_233378


namespace range_a_sufficient_not_necessary_l2333_233394

theorem range_a_sufficient_not_necessary (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, (x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0) → (|x - 3| > 1)) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2 / 3) :=
sorry

end range_a_sufficient_not_necessary_l2333_233394


namespace Lei_Lei_sheep_count_l2333_233332

-- Define the initial average price and number of sheep as parameters
variables (a : ℝ) (x : ℕ)

-- Conditions as hypotheses
def condition1 : Prop := ∀ a x: ℝ,
  60 * x + 2 * (a + 60) = 90 * x + 2 * (a - 90)

-- The main problem stated as a theorem to be proved
theorem Lei_Lei_sheep_count (h : condition1) : x = 10 :=
sorry


end Lei_Lei_sheep_count_l2333_233332


namespace seven_pow_k_minus_k_pow_seven_l2333_233397

theorem seven_pow_k_minus_k_pow_seven (k : ℕ) (h : 21^k ∣ 435961) : 7^k - k^7 = 1 :=
sorry

end seven_pow_k_minus_k_pow_seven_l2333_233397


namespace fraction_to_decimal_l2333_233348

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l2333_233348


namespace find_x_l2333_233305

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the proof goal
theorem find_x (x : ℝ) : 2 * f x - 19 = f (x - 4) → x = 4 :=
by
  sorry

end find_x_l2333_233305


namespace min_fence_posts_l2333_233345

theorem min_fence_posts (length width wall_length interval : ℕ) (h_dim : length = 80) (w_dim : width = 50) (h_wall : wall_length = 150) (h_interval : interval = 10) : 
  length/interval + 1 + 2 * (width/interval - 1) = 17 :=
by
  sorry

end min_fence_posts_l2333_233345


namespace exists_divisible_sk_l2333_233376

noncomputable def sequence_of_integers (c : ℕ) (a : ℕ → ℕ) :=
  ∀ n : ℕ, 0 < n → a n < a (n + 1) ∧ a (n + 1) < a n + c

noncomputable def infinite_string (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (10 ^ n) * (a (n + 1)) + a n

noncomputable def sk (s : ℕ) (k : ℕ) : ℕ :=
  (s % (10 ^ k))

theorem exists_divisible_sk (a : ℕ → ℕ) (c m : ℕ)
  (h : sequence_of_integers c a) :
  ∀ m : ℕ, ∃ k : ℕ, m > 0 → (sk (infinite_string a k) k) % m = 0 := by
  sorry

end exists_divisible_sk_l2333_233376


namespace total_time_hover_layover_two_days_l2333_233318

theorem total_time_hover_layover_two_days 
    (hover_pacific_day1 : ℝ)
    (hover_mountain_day1 : ℝ)
    (hover_central_day1 : ℝ)
    (hover_eastern_day1 : ℝ)
    (layover_time : ℝ)
    (speed_increase : ℝ)
    (time_decrease : ℝ) :
    hover_pacific_day1 = 2 →
    hover_mountain_day1 = 3 →
    hover_central_day1 = 4 →
    hover_eastern_day1 = 3 →
    layover_time = 1.5 →
    speed_increase = 0.2 →
    time_decrease = 1.6 →
    hover_pacific_day1 + hover_mountain_day1 + hover_central_day1 + hover_eastern_day1 + 4 * layover_time 
      + (hover_eastern_day1 - (speed_increase * hover_eastern_day1) + hover_central_day1 - (speed_increase * hover_central_day1) 
         + hover_mountain_day1 - (speed_increase * hover_mountain_day1) + hover_pacific_day1 - (speed_increase * hover_pacific_day1)) 
      + 4 * layover_time = 33.6 := 
by
  intros
  sorry

end total_time_hover_layover_two_days_l2333_233318


namespace initial_num_families_eq_41_l2333_233321

-- Definitions based on the given conditions
def num_families_flew_away : ℕ := 27
def num_families_left : ℕ := 14

-- Statement to prove
theorem initial_num_families_eq_41 : num_families_flew_away + num_families_left = 41 := by
  sorry

end initial_num_families_eq_41_l2333_233321


namespace find_y_l2333_233384

theorem find_y (x y : ℝ) (h₁ : x^2 - 2 * x + 5 = y + 3) (h₂ : x = 5) : y = 17 :=
by
  sorry

end find_y_l2333_233384


namespace modified_triangle_array_sum_100_l2333_233346

def triangle_array_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem modified_triangle_array_sum_100 :
  triangle_array_sum 100 = 2^100 - 2 :=
sorry

end modified_triangle_array_sum_100_l2333_233346


namespace find_polynomials_satisfy_piecewise_l2333_233379

def f (x : ℝ) : ℝ := 0
def g (x : ℝ) : ℝ := -x
def h (x : ℝ) : ℝ := -x + 2

theorem find_polynomials_satisfy_piecewise :
  ∀ x : ℝ, abs (f x) - abs (g x) + h x = 
    if x < -1 then -1
    else if x <= 0 then 2
    else -2 * x + 2 :=
by
  sorry

end find_polynomials_satisfy_piecewise_l2333_233379


namespace J_speed_is_4_l2333_233313

noncomputable def J_speed := 4
variable (v_J v_P : ℝ)

axiom condition1 : v_J > v_P
axiom condition2 : v_J + v_P = 7
axiom condition3 : (24 / v_J) + (24 / v_P) = 14

theorem J_speed_is_4 : v_J = J_speed :=
by
  sorry

end J_speed_is_4_l2333_233313


namespace sum_of_favorite_numbers_l2333_233392

def Glory_favorite_number : ℕ := 450
def Misty_favorite_number : ℕ := Glory_favorite_number / 3

theorem sum_of_favorite_numbers : Misty_favorite_number + Glory_favorite_number = 600 :=
by
  sorry

end sum_of_favorite_numbers_l2333_233392


namespace find_x_value_l2333_233312

theorem find_x_value (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : xy / (x + y) = a) (h5 : xz / (x + z) = b) (h6 : yz / (y + z) = c)
  (h7 : x + y + z = abc) : 
  x = (2 * a * b * c) / (a * b + b * c + a * c) :=
sorry

end find_x_value_l2333_233312


namespace divide_sum_eq_100_l2333_233351

theorem divide_sum_eq_100 (x : ℕ) (h1 : 100 = 2 * x + (100 - 2 * x)) (h2 : (300 - 6 * x) + x = 100) : x = 40 :=
by
  sorry

end divide_sum_eq_100_l2333_233351


namespace largest_four_digit_number_l2333_233301

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l2333_233301


namespace min_value_reciprocal_sum_l2333_233395

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

end min_value_reciprocal_sum_l2333_233395


namespace range_of_m_l2333_233353

theorem range_of_m (m n : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |m * x| - |x - n|) 
  (h_n_pos : 0 < n) (h_n_m : n < 1 + m) 
  (h_integer_sol : ∃ xs : Finset ℤ, xs.card = 3 ∧ ∀ x ∈ xs, f x < 0) : 
  1 < m ∧ m < 3 := 
sorry

end range_of_m_l2333_233353


namespace Angela_insect_count_l2333_233303

variables (Angela Jacob Dean : ℕ)
-- Conditions
def condition1 : Prop := Angela = Jacob / 2
def condition2 : Prop := Jacob = 5 * Dean
def condition3 : Prop := Dean = 30

-- Theorem statement proving Angela's insect count
theorem Angela_insect_count (h1 : condition1 Angela Jacob) (h2 : condition2 Jacob Dean) (h3 : condition3 Dean) : Angela = 75 :=
by
  sorry

end Angela_insect_count_l2333_233303


namespace hyperbola_s_squared_zero_l2333_233330

open Real

theorem hyperbola_s_squared_zero :
  ∃ s : ℝ, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), 
  ((x, y) = (-2, 3) ∨ (x, y) = (0, -1) ∨ (x, y) = (s, 1)) → (y^2 / a^2 - x^2 / b^2 = 1))
  ) → s ^ 2 = 0 :=
by
  sorry

end hyperbola_s_squared_zero_l2333_233330


namespace merchant_profit_percentage_l2333_233327

noncomputable def cost_price : ℝ := 100
noncomputable def marked_up_price : ℝ := cost_price + (0.75 * cost_price)
noncomputable def discount : ℝ := 0.30 * marked_up_price
noncomputable def selling_price : ℝ := marked_up_price - discount
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem merchant_profit_percentage :
  profit_percentage = 22.5 :=
by
  sorry

end merchant_profit_percentage_l2333_233327


namespace hydrangea_cost_l2333_233326

def cost_of_each_plant : ℕ :=
  let total_years := 2021 - 1989
  let total_amount_spent := 640
  total_amount_spent / total_years

theorem hydrangea_cost :
  cost_of_each_plant = 20 :=
by
  -- skipping the proof for Lean statement
  sorry

end hydrangea_cost_l2333_233326


namespace percentage_water_in_fresh_grapes_is_65_l2333_233372

noncomputable def percentage_water_in_fresh_grapes 
  (weight_fresh : ℝ) (weight_dried : ℝ) (percentage_water_dried : ℝ) : ℝ :=
  100 - ((weight_dried / weight_fresh) - percentage_water_dried / 100 * weight_dried / weight_fresh) * 100

theorem percentage_water_in_fresh_grapes_is_65 :
  percentage_water_in_fresh_grapes 400 155.56 10 = 65 := 
by
  sorry

end percentage_water_in_fresh_grapes_is_65_l2333_233372


namespace union_of_sets_l2333_233308

theorem union_of_sets (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2 * a - 1 | a ∈ M}) :
  M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_l2333_233308


namespace smallest_Q_value_l2333_233374

noncomputable def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 + 4*x + 6

theorem smallest_Q_value :
  min (Q (-1)) (min (6) (min (1 + 2 - 1 + 4 + 6) (sorry))) = Q (-1) :=
by
  sorry

end smallest_Q_value_l2333_233374


namespace lattice_points_count_l2333_233316

-- Definition of a lattice point
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Given endpoints of the line segment
def point1 : LatticePoint := ⟨5, 13⟩
def point2 : LatticePoint := ⟨38, 214⟩

-- Function to count lattice points on the line segment given the endpoints
def countLatticePoints (p1 p2 : LatticePoint) : ℕ := sorry

-- The proof statement
theorem lattice_points_count :
  countLatticePoints point1 point2 = 4 := sorry

end lattice_points_count_l2333_233316


namespace box_area_ratio_l2333_233300

theorem box_area_ratio 
  (l w h : ℝ)
  (V : l * w * h = 5184)
  (A1 : w * h = (1/2) * l * w)
  (A2 : l * h = 288):
  (l * w) / (l * h) = 3 / 2 := 
by
  sorry

end box_area_ratio_l2333_233300


namespace problem_l2333_233314

variable {a b c x y z : ℝ}

theorem problem 
  (h1 : 5 * x + b * y + c * z = 0)
  (h2 : a * x + 7 * y + c * z = 0)
  (h3 : a * x + b * y + 9 * z = 0)
  (h4 : a ≠ 5)
  (h5 : x ≠ 0) :
  (a / (a - 5)) + (b / (b - 7)) + (c / (c - 9)) = 1 :=
by
  sorry

end problem_l2333_233314


namespace muffins_apples_l2333_233398

def apples_left_for_muffins (total_apples : ℕ) (pie_apples : ℕ) (refrigerator_apples : ℕ) : ℕ :=
  total_apples - (pie_apples + refrigerator_apples)

theorem muffins_apples (total_apples pie_apples refrigerator_apples : ℕ) (h_total : total_apples = 62) (h_pie : pie_apples = total_apples / 2) (h_refrigerator : refrigerator_apples = 25) : apples_left_for_muffins total_apples pie_apples refrigerator_apples = 6 := 
by 
  sorry

end muffins_apples_l2333_233398


namespace average_percentage_decrease_is_10_l2333_233356

noncomputable def average_percentage_decrease (original_cost final_cost : ℝ) (n : ℕ) : ℝ :=
  1 - (final_cost / original_cost)^(1 / n)

theorem average_percentage_decrease_is_10
  (original_cost current_cost : ℝ)
  (n : ℕ)
  (h_original_cost : original_cost = 100)
  (h_current_cost : current_cost = 81)
  (h_n : n = 2) :
  average_percentage_decrease original_cost current_cost n = 0.1 :=
by
  -- The proof would go here if it were needed.
  sorry

end average_percentage_decrease_is_10_l2333_233356


namespace garden_path_width_l2333_233309

theorem garden_path_width (R r : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 20 * Real.pi) : R - r = 10 :=
by
  sorry

end garden_path_width_l2333_233309


namespace slices_left_l2333_233365

variable (total_pieces: ℕ) (joe_fraction: ℚ) (darcy_fraction: ℚ)
variable (carl_fraction: ℚ) (emily_fraction: ℚ)

theorem slices_left 
  (h1 : total_pieces = 24)
  (h2 : joe_fraction = 1/3)
  (h3 : darcy_fraction = 1/4)
  (h4 : carl_fraction = 1/6)
  (h5 : emily_fraction = 1/8) :
  total_pieces - (total_pieces * joe_fraction + total_pieces * darcy_fraction + total_pieces * carl_fraction + total_pieces * emily_fraction) = 3 := 
  by 
  sorry

end slices_left_l2333_233365


namespace total_cost_of_cloth_l2333_233370

/-- Define the length of the cloth in meters --/
def length_of_cloth : ℝ := 9.25

/-- Define the cost per meter in dollars --/
def cost_per_meter : ℝ := 46

/-- Theorem stating that the total cost is $425.50 given the length and cost per meter --/
theorem total_cost_of_cloth : length_of_cloth * cost_per_meter = 425.50 := by
  sorry

end total_cost_of_cloth_l2333_233370


namespace opposite_of_neg_one_third_l2333_233381

theorem opposite_of_neg_one_third : (-(-1/3)) = (1/3) := by
  sorry

end opposite_of_neg_one_third_l2333_233381


namespace cost_comparison_l2333_233349

def full_ticket_price : ℝ := 240

def cost_agency_A (x : ℕ) : ℝ :=
  full_ticket_price + 0.5 * full_ticket_price * x

def cost_agency_B (x : ℕ) : ℝ :=
  0.6 * full_ticket_price * (x + 1)

theorem cost_comparison (x : ℕ) :
  (x = 4 → cost_agency_A x = cost_agency_B x) ∧
  (x > 4 → cost_agency_A x < cost_agency_B x) ∧
  (x < 4 → cost_agency_A x > cost_agency_B x) :=
by
  sorry

end cost_comparison_l2333_233349


namespace jerry_needs_money_l2333_233340

theorem jerry_needs_money (has : ℕ) (total : ℕ) (cost_per_action_figure : ℕ) 
  (h1 : has = 7) (h2 : total = 16) (h3 : cost_per_action_figure = 8) : 
  (total - has) * cost_per_action_figure = 72 := by
  -- Proof goes here
  sorry

end jerry_needs_money_l2333_233340


namespace intersection_A_B_l2333_233361

noncomputable def domain_ln_1_minus_x : Set ℝ := {x : ℝ | x < 1}
def range_x_squared : Set ℝ := {y : ℝ | 0 ≤ y}
def intersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

theorem intersection_A_B :
  (domain_ln_1_minus_x ∩ range_x_squared) = intersection :=
by sorry

end intersection_A_B_l2333_233361


namespace length_of_CD_l2333_233369

theorem length_of_CD (x y : ℝ) (h1 : x / (3 + y) = 3 / 5) (h2 : (x + 3) / y = 4 / 7) (h3 : x + 3 + y = 273.6) : 3 + y = 273.6 :=
by
  sorry

end length_of_CD_l2333_233369


namespace quadratic_inequality_l2333_233357

theorem quadratic_inequality (a b c : ℝ)
  (h1 : ∀ x : ℝ, x = -2 → y = 8)
  (h2 : ∀ x : ℝ, x = -1 → y = 3)
  (h3 : ∀ x : ℝ, x = 0 → y = 0)
  (h4 : ∀ x : ℝ, x = 1 → y = -1)
  (h5 : ∀ x : ℝ, x = 2 → y = 0)
  (h6 : ∀ x : ℝ, x = 3 → y = 3)
  : ∀ x : ℝ, (y - 3 > 0) ↔ x < -1 ∨ x > 3 :=
sorry

end quadratic_inequality_l2333_233357


namespace sum_of_squares_l2333_233324

theorem sum_of_squares (n : ℕ) : ∃ k : ℤ, (∃ a b : ℤ, k = a^2 + b^2) ∧ (∃ d : ℕ, d ≥ n) :=
by
  sorry

end sum_of_squares_l2333_233324


namespace percentage_of_sikh_boys_is_10_l2333_233343

theorem percentage_of_sikh_boys_is_10 (total_boys : ℕ)
  (perc_muslim : ℝ) (perc_hindu : ℝ) (other_comm_boys : ℕ)
  (H_total_boys : total_boys = 850)
  (H_perc_muslim : perc_muslim = 0.40)
  (H_perc_hindu : perc_hindu = 0.28)
  (H_other_comm_boys : other_comm_boys = 187) :
  ((total_boys - ( (perc_muslim * total_boys) + (perc_hindu * total_boys) + other_comm_boys)) / total_boys) * 100 = 10 :=
by
  sorry

end percentage_of_sikh_boys_is_10_l2333_233343


namespace find_B_l2333_233336

-- Define the translation function for points in ℝ × ℝ.
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

-- Given conditions
def A : ℝ × ℝ := (2, 2)
def A' : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-1, 1)

-- The vector v representing the translation from A to A'
def v : ℝ × ℝ := (A'.1 - A.1, A'.2 - A.2)

-- Proving the coordinates of B' after applying the same translation vector v to B
theorem find_B' : translate B v = (-5, -3) :=
by
  -- translation function needs to be instantiated with the correct values.
  -- Since this is just a Lean 4 statement, we'll not include the proof here and leave it as a sorry.
  sorry

end find_B_l2333_233336


namespace initial_worth_of_wears_l2333_233311

theorem initial_worth_of_wears (W : ℝ) 
  (h1 : W + 2/5 * W = 1.4 * W)
  (h2 : 0.85 * (W + 2/5 * W) = W + 95) : 
  W = 500 := 
by 
  sorry

end initial_worth_of_wears_l2333_233311


namespace min_internal_fence_length_l2333_233350

-- Setup the given conditions in Lean 4
def total_land_area (length width : ℕ) : ℕ := length * width

def sotkas_to_m2 (sotkas : ℕ) : ℕ := sotkas * 100

-- Assume a father had three sons and left them an inheritance of land
def land_inheritance := 9 -- in sotkas

-- The dimensions of the land
def length := 25 
def width := 36

-- Prove that:
theorem min_internal_fence_length :
  ∃ (ways : ℕ) (min_length : ℕ),
    total_land_area length width = sotkas_to_m2 land_inheritance ∧
    (∀ (l1 l2 l3 w1 w2 w3 : ℕ),
      l1 * w1 = sotkas_to_m2 3 ∧ l2 * w2 = sotkas_to_m2 3 ∧ l3 * w3 = sotkas_to_m2 3 →
      ways = 4 ∧ min_length = 49) :=
by
  sorry

end min_internal_fence_length_l2333_233350


namespace roots_of_quadratic_l2333_233333

theorem roots_of_quadratic {a b c : ℝ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x, (x = a ∨ x = b ∨ x = c) ↔ x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 :=
by
  sorry

end roots_of_quadratic_l2333_233333


namespace rectangular_box_diagonals_l2333_233391

noncomputable def interior_diagonals_sum (a b c : ℝ) : ℝ := 4 * Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 150)
  (h2 : 4 * (a + b + c) = 60)
  (h3 : a * b * c = 216) :
  interior_diagonals_sum a b c = 20 * Real.sqrt 3 :=
by
  sorry

end rectangular_box_diagonals_l2333_233391


namespace range_of_b_l2333_233329

/-- Let A = {x | -1 < x < 1} and B = {x | b - 1 < x < b + 1}.
    We need to show that if A ∩ B ≠ ∅, then b is within the interval (-2, 2). -/
theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ b - 1 < x ∧ x < b + 1) →
  -2 < b ∧ b < 2 :=
sorry

end range_of_b_l2333_233329


namespace range_f_when_a_1_range_of_a_values_l2333_233302

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

theorem range_f_when_a_1 : 
  (∀ x : ℝ, f x 1 ≥ 5) :=
sorry

theorem range_of_a_values :
  (∀ x, f x a ≥ 1) → (a ∈ Set.union (Set.Iic (-5)) (Set.Ici (-3))) :=
sorry

end range_f_when_a_1_range_of_a_values_l2333_233302


namespace no_net_profit_or_loss_l2333_233363

theorem no_net_profit_or_loss (C : ℝ) : 
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  net_profit_loss = 0 :=
by
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  sorry

end no_net_profit_or_loss_l2333_233363


namespace pow_of_729_l2333_233335

theorem pow_of_729 : (729 : ℝ) ^ (2 / 3) = 81 :=
by sorry

end pow_of_729_l2333_233335


namespace vectors_parallel_iff_m_eq_neg_1_l2333_233399

-- Given vectors a and b
def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, m)
def vector_b : ℝ × ℝ := (3, 1)

-- Definition of vectors being parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- The statement to be proven
theorem vectors_parallel_iff_m_eq_neg_1 (m : ℝ) : 
  parallel (vector_a m) vector_b ↔ m = -1 :=
by 
  sorry

end vectors_parallel_iff_m_eq_neg_1_l2333_233399


namespace problem_statement_l2333_233371

theorem problem_statement 
  (a b c : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 :=
sorry

end problem_statement_l2333_233371


namespace projectile_height_l2333_233337

theorem projectile_height (t : ℝ) : 
  (∃ t : ℝ, (-4.9 * t^2 + 30.4 * t = 35)) → 
  (0 < t ∧ t ≤ 5) → 
  t = 10 / 7 :=
by
  sorry

end projectile_height_l2333_233337


namespace relationship_between_a_and_b_l2333_233387

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
  sorry

end relationship_between_a_and_b_l2333_233387


namespace matrix_arithmetic_sequence_sum_l2333_233373

theorem matrix_arithmetic_sequence_sum (a : ℕ → ℕ → ℕ)
  (h_row1 : ∀ i, 2 * a 4 2 = a 4 (i - 1) + a 4 (i + 1))
  (h_row2 : ∀ i, 2 * a 5 2 = a 5 (i - 1) + a 5 (i + 1))
  (h_row3 : ∀ i, 2 * a 6 2 = a 6 (i - 1) + a 6 (i + 1))
  (h_col1 : ∀ i, 2 * a 5 2 = a (i - 1) 2 + a (i + 1) 2)
  (h_sum : a 4 1 + a 4 2 + a 4 3 + a 5 1 + a 5 2 + a 5 3 + a 6 1 + a 6 2 + a 6 3 = 63)
  : a 5 2 = 7 := sorry

end matrix_arithmetic_sequence_sum_l2333_233373


namespace prime_sum_value_l2333_233390

theorem prime_sum_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 2019) : 
  (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 :=
by
  sorry

end prime_sum_value_l2333_233390


namespace smallest_n_rel_prime_to_300_l2333_233358

theorem smallest_n_rel_prime_to_300 : ∃ n : ℕ, n > 1 ∧ Nat.gcd n 300 = 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → Nat.gcd m 300 ≠ 1 :=
by
  sorry

end smallest_n_rel_prime_to_300_l2333_233358


namespace problem1_problem2_l2333_233364

-- Definitions for the problem

/-- Definition of point P in Cartesian coordinate system -/
def P (x : ℝ) : ℝ × ℝ :=
  (x - 2, x)

-- First proof problem statement
theorem problem1 (x : ℝ) (h : (x - 2) * x < 0) : x = 1 :=
sorry

-- Second proof problem statement
theorem problem2 (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end problem1_problem2_l2333_233364


namespace solve_eqn_l2333_233317

noncomputable def a : ℝ := 5 + 2 * Real.sqrt 6
noncomputable def b : ℝ := 5 - 2 * Real.sqrt 6

theorem solve_eqn (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 10) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_eqn_l2333_233317


namespace number_of_students_absent_l2333_233328

def classes := 18
def students_per_class := 28
def students_present := 496
def students_absent := (classes * students_per_class) - students_present

theorem number_of_students_absent : students_absent = 8 := 
by
  sorry

end number_of_students_absent_l2333_233328


namespace find_alpha_l2333_233355

variable {α p₀ p_new : ℝ}
def Q_d (p : ℝ) : ℝ := 150 - p
def Q_s (p : ℝ) : ℝ := 3 * p - 10
def Q_d_new (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem find_alpha 
  (h_eq_initial : Q_d p₀ = Q_s p₀)
  (h_eq_increase : p_new = 1.25 * p₀)
  (h_eq_new : Q_s p_new = Q_d_new α p_new) :
  α = 1.4 :=
by
  sorry

end find_alpha_l2333_233355


namespace conclusion_1_conclusion_2_l2333_233359

open Function

-- Conclusion ①
theorem conclusion_1 {f : ℝ → ℝ} (h : StrictMono f) :
  ∀ {x1 x2 : ℝ}, f x1 ≤ f x2 ↔ x1 ≤ x2 := 
by
  intros x1 x2
  exact h.le_iff_le

-- Conclusion ②
theorem conclusion_2 {f : ℝ → ℝ} (h : ∀ x, f x ^ 2 = f (-x) ^ 2) :
  ¬ (∀ x, f (-x) = f x ∨ f (-x) = -f x) :=
by
  sorry

end conclusion_1_conclusion_2_l2333_233359


namespace M_union_N_eq_l2333_233393

open Set

def M : Set ℝ := { x | x^2 - 4 * x < 0 }
def N : Set ℝ := { x | abs x ≤ 2 }

theorem M_union_N_eq : M ∪ N = Ico (-2 : ℝ) 4 := by
  sorry

end M_union_N_eq_l2333_233393


namespace trapezoid_leg_length_proof_l2333_233341

noncomputable def circumscribed_trapezoid_leg_length 
  (area : ℝ) (acute_angle_base : ℝ) : ℝ :=
  -- Hypothesis: Given conditions of the problem
  if h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3 then
    -- The length of the trapezoid's leg
    8
  else
    0

-- Statement of the proof problem
theorem trapezoid_leg_length_proof 
  (area : ℝ) (acute_angle_base : ℝ)
  (h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3) :
  circumscribed_trapezoid_leg_length area acute_angle_base = 8 := 
by {
  -- skipping actual proof
  sorry
}

end trapezoid_leg_length_proof_l2333_233341


namespace height_of_pole_l2333_233339

-- Defining the constants according to the problem statement
def AC := 5.0 -- meters
def AD := 4.0 -- meters
def DE := 1.7 -- meters

-- We need to prove that the height of the pole AB is 8.5 meters
theorem height_of_pole (AB : ℝ) (hAC : AC = 5) (hAD : AD = 4) (hDE : DE = 1.7) :
  AB = 8.5 := by
  sorry

end height_of_pole_l2333_233339


namespace factorial_square_gt_power_l2333_233307

theorem factorial_square_gt_power (n : ℕ) (h : n > 2) : (n!)^2 > n^n := by
  sorry

end factorial_square_gt_power_l2333_233307


namespace Jillian_collected_29_l2333_233386

variable (Savannah_shells Clayton_shells total_friends friend_shells : ℕ)

def Jillian_shells : ℕ :=
  let total_shells := friend_shells * total_friends
  let others_shells := Savannah_shells + Clayton_shells
  total_shells - others_shells

theorem Jillian_collected_29 (h_savannah : Savannah_shells = 17) 
                             (h_clayton : Clayton_shells = 8) 
                             (h_friends : total_friends = 2) 
                             (h_friend_shells : friend_shells = 27) : 
  Jillian_shells Savannah_shells Clayton_shells total_friends friend_shells = 29 :=
by
  sorry

end Jillian_collected_29_l2333_233386


namespace sum_of_c_and_d_l2333_233388

def digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem sum_of_c_and_d 
  (c d : ℕ)
  (hcd : digit c)
  (hdd : digit d)
  (h1: (4*c) * 5 % 10 = 5)
  (h2: 215 = (10 * (4*(d*5) + c*5)) + d*10 + 5) :
  c + d = 5 := 
  sorry

end sum_of_c_and_d_l2333_233388


namespace value_of_expression_l2333_233377

-- Definitions for the conditions
variables (a b : ℝ)

-- Theorem statement
theorem value_of_expression : (a - 3 * b = 3) → (a + 2 * b - (2 * a - b)) = -3 :=
by
  intro h
  sorry

end value_of_expression_l2333_233377


namespace walking_time_l2333_233319

theorem walking_time (intervals_time : ℕ) (poles_12_time : ℕ) (speed_constant : Prop) : 
  intervals_time = 2 → poles_12_time = 22 → speed_constant → 39 * intervals_time = 78 :=
by
  sorry

end walking_time_l2333_233319


namespace infinite_rational_points_in_region_l2333_233338

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 6) ∧ S.Infinite :=
sorry

end infinite_rational_points_in_region_l2333_233338


namespace number_of_bad_carrots_l2333_233304

-- Definitions for conditions
def olivia_picked : ℕ := 20
def mother_picked : ℕ := 14
def good_carrots : ℕ := 19

-- Sum of total carrots picked
def total_carrots : ℕ := olivia_picked + mother_picked

-- Theorem stating the number of bad carrots
theorem number_of_bad_carrots : total_carrots - good_carrots = 15 :=
by
  sorry

end number_of_bad_carrots_l2333_233304


namespace min_value_arith_prog_sum_l2333_233362

noncomputable def arithmetic_progression_sum (x y : ℝ) (n : ℕ) : ℝ :=
  (x + 2 * y + 1) * 3^n + (x - y - 4)

theorem min_value_arith_prog_sum (x y : ℝ)
  (hx : x > 0) (hy : y > 0)
  (h_sum : ∀ n, arithmetic_progression_sum x y n = (x + 2 * y + 1) * 3^n + (x - y - 4)) :
  (∀ x y, 2 * x + y = 3 → 1/x + 2/y ≥ 8/3) :=
by sorry

end min_value_arith_prog_sum_l2333_233362


namespace particle_paths_count_l2333_233380

-- Definitions for the movement in the Cartesian plane
def valid_moves (a b : ℕ) : List (ℕ × ℕ) := [(a + 2, b), (a, b + 2), (a + 1, b + 1)]

-- The condition to count unique paths from (0,0) to (6,6)
def count_paths (start target : ℕ × ℕ) : ℕ :=
  sorry -- The exact implementation to count paths is omitted here

theorem particle_paths_count :
  count_paths (0, 0) (6, 6) = 58 :=
sorry

end particle_paths_count_l2333_233380


namespace product_of_sequence_l2333_233323

theorem product_of_sequence : 
  (∃ (a : ℕ → ℚ), (a 1 * a 2 * a 3 * a 4 * a 5 = -32) ∧ 
  ((∀ n : ℕ, 3 * a (n + 1) + a n = 0) ∧ a 2 = 6)) :=
sorry

end product_of_sequence_l2333_233323


namespace points_on_opposite_sides_of_line_l2333_233360

theorem points_on_opposite_sides_of_line (m : ℝ) (h1 : 2 - 1 + m > 0) (h2 : 1 - 3 + m < 0) : -1 < m ∧ m < 2 :=
by
  have h : (m + 1) * (m - 2) < 0 := sorry
  exact sorry

end points_on_opposite_sides_of_line_l2333_233360


namespace sum_of_digits_of_x_l2333_233382

def two_digit_palindrome (x : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99) ∧ (x = (x % 10) * 10 + (x % 10))

def three_digit_palindrome (y : ℕ) : Prop :=
  (100 ≤ y ∧ y ≤ 999) ∧ (y = (y % 10) * 101 + (y % 10))

theorem sum_of_digits_of_x (x : ℕ) (h1 : two_digit_palindrome x) (h2 : three_digit_palindrome (x + 10)) : 
  (x % 10 + x / 10) = 10 :=
by
  sorry

end sum_of_digits_of_x_l2333_233382


namespace sarah_can_make_max_servings_l2333_233320

-- Definitions based on the conditions of the problem
def servings_from_bananas (bananas : ℕ) : ℕ := (bananas * 8) / 3
def servings_from_strawberries (cups_strawberries : ℕ) : ℕ := (cups_strawberries * 8) / 2
def servings_from_yogurt (cups_yogurt : ℕ) : ℕ := cups_yogurt * 8
def servings_from_milk (cups_milk : ℕ) : ℕ := (cups_milk * 8) / 4

-- Given Sarah's stock
def sarahs_bananas : ℕ := 10
def sarahs_strawberries : ℕ := 5
def sarahs_yogurt : ℕ := 3
def sarahs_milk : ℕ := 10

-- The maximum servings calculation
def max_servings : ℕ := 
  min (servings_from_bananas sarahs_bananas)
      (min (servings_from_strawberries sarahs_strawberries)
           (min (servings_from_yogurt sarahs_yogurt)
                (servings_from_milk sarahs_milk)))

-- The theorem to be proved
theorem sarah_can_make_max_servings : max_servings = 20 :=
by
  sorry

end sarah_can_make_max_servings_l2333_233320


namespace convert_base8_to_base10_l2333_233389

def base8_to_base10 (n : Nat) : Nat := 
  -- Assuming a specific function that converts from base 8 to base 10
  sorry 

theorem convert_base8_to_base10 :
  base8_to_base10 5624 = 2964 :=
by
  sorry

end convert_base8_to_base10_l2333_233389


namespace intersection_point_sum_l2333_233347

theorem intersection_point_sum {h j : ℝ → ℝ} 
    (h3: h 3 = 3) (j3: j 3 = 3) 
    (h6: h 6 = 9) (j6: j 6 = 9)
    (h9: h 9 = 18) (j9: j 9 = 18)
    (h12: h 12 = 18) (j12: j 12 = 18) :
    ∃ a b, (h (3 * a) = 3 * j a ∧ a + b = 22) := 
sorry

end intersection_point_sum_l2333_233347


namespace find_m_l2333_233315

-- Definitions for the given vectors
def vector_a : ℝ × ℝ := (-2, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (4, m)

-- The condition that (vector_a + 2 * vector_b) is parallel to (vector_a - vector_b)
def parallel_condition (m : ℝ) : Prop :=
  let left_vec := (vector_a.1 + 2 * 4, vector_a.2 + 2 * m)
  let right_vec := (vector_a.1 - 4, vector_a.2 - m)
  left_vec.1 * right_vec.2 - right_vec.1 * left_vec.2 = 0

-- The main theorem to prove
theorem find_m : ∃ m : ℝ, parallel_condition m ∧ m = -6 := 
sorry

end find_m_l2333_233315


namespace range_of_t_in_region_l2333_233310

theorem range_of_t_in_region : (t : ℝ) → ((1 - t + 1 > 0) → t < 2) :=
by
  intro t
  intro h
  sorry

end range_of_t_in_region_l2333_233310


namespace other_juice_cost_l2333_233366

theorem other_juice_cost (total_spent : ℕ := 94)
    (mango_cost_per_glass : ℕ := 5)
    (other_total_spent : ℕ := 54)
    (total_people : ℕ := 17) : 
  other_total_spent / (total_people - (total_spent - other_total_spent) / mango_cost_per_glass) = 6 := 
sorry

end other_juice_cost_l2333_233366
