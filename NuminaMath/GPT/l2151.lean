import Mathlib

namespace line_intersects_y_axis_at_5_l2151_215188

theorem line_intersects_y_axis_at_5 :
  ∃ (b : ℝ), ∀ (x y : ℝ), (x - 2 = 0 ∧ y - 9 = 0) ∨ (x - 4 = 0 ∧ y - 13 = 0) →
  (y = 2 * x + b) ∧ (b = 5) :=
by
  sorry

end line_intersects_y_axis_at_5_l2151_215188


namespace polar_to_cartesian_l2151_215155

-- Define the conditions
def polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the goal as a theorem
theorem polar_to_cartesian : ∀ (x y : ℝ), 
  (∃ θ : ℝ, polar_eq (Real.sqrt (x^2 + y^2)) θ ∧ x = (Real.sqrt (x^2 + y^2)) * Real.cos θ 
  ∧ y = (Real.sqrt (x^2 + y^2)) * Real.sin θ) → (x-1)^2 + y^2 = 1 :=
by
  intro x y
  intro h
  sorry

end polar_to_cartesian_l2151_215155


namespace selling_price_ratio_l2151_215166

theorem selling_price_ratio (CP SP1 SP2 : ℝ) (h1 : SP1 = CP + 0.5 * CP) (h2 : SP2 = CP + 3 * CP) :
  SP2 / SP1 = 8 / 3 :=
by
  sorry

end selling_price_ratio_l2151_215166


namespace solve_for_alpha_l2151_215175

variables (α β γ δ : ℝ)

theorem solve_for_alpha (h : α + β + γ + δ = 360) : α = 360 - β - γ - δ :=
by sorry

end solve_for_alpha_l2151_215175


namespace unit_prices_min_selling_price_l2151_215133

-- Problem 1: Unit price determination
theorem unit_prices (x y : ℕ) (hx : 3600 / x * 2 = 5400 / y) (hy : y = x - 5) : x = 20 ∧ y = 15 := 
by 
  sorry

-- Problem 2: Minimum selling price for 50% profit margin
theorem min_selling_price (a : ℕ) (hx : 3600 / 20 = 180) (hy : 180 * 2 = 360) (hz : 540 * a ≥ 13500) : a ≥ 25 := 
by 
  sorry

end unit_prices_min_selling_price_l2151_215133


namespace find_value_a2_b2_c2_l2151_215182

variable (a b c p q r : ℝ)
variable (h1 : a * b = p)
variable (h2 : b * c = q)
variable (h3 : c * a = r)
variable (h4 : p ≠ 0)
variable (h5 : q ≠ 0)
variable (h6 : r ≠ 0)

theorem find_value_a2_b2_c2 : a^2 + b^2 + c^2 = 1 :=
by sorry

end find_value_a2_b2_c2_l2151_215182


namespace k_value_l2151_215119

noncomputable def find_k (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : ℝ :=
  12 / 7

theorem k_value (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : 
  find_k AB BC AC BD h_AB h_BC h_AC h_BD = 12 / 7 :=
by
  sorry

end k_value_l2151_215119


namespace evaluate_expression_l2151_215146

theorem evaluate_expression :
  abs ((4^2 - 8 * (3^2 - 12))^2) - abs (Real.sin (5 * Real.pi / 6) - Real.cos (11 * Real.pi / 3)) = 1600 :=
by
  sorry

end evaluate_expression_l2151_215146


namespace given_sequence_find_a_and_b_l2151_215140

-- Define the general pattern of the sequence
def sequence_pattern (n a b : ℕ) : Prop :=
  n + (b / a : ℚ) = (n^2 : ℚ) * (b / a : ℚ)

-- State the specific case for n = 9
def sequence_case_for_9 (a b : ℕ) : Prop :=
  sequence_pattern 9 a b ∧ a + b = 89

-- Now, structure this as a theorem to be proven in Lean
theorem given_sequence_find_a_and_b :
  ∃ (a b : ℕ), sequence_case_for_9 a b :=
sorry

end given_sequence_find_a_and_b_l2151_215140


namespace tables_needed_l2151_215162

-- Conditions
def n_invited : ℕ := 18
def n_no_show : ℕ := 12
def capacity_per_table : ℕ := 3

-- Calculation of attendees
def n_attendees : ℕ := n_invited - n_no_show

-- Proof for the number of tables needed
theorem tables_needed : (n_attendees / capacity_per_table) = 2 := by
  -- Sorry will be here to show it's incomplete
  sorry

end tables_needed_l2151_215162


namespace total_campers_went_rowing_l2151_215108

-- Definitions based on given conditions
def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def evening_campers : ℕ := 49

-- Theorem statement to be proven
theorem total_campers_went_rowing : morning_campers + afternoon_campers + evening_campers = 98 :=
by sorry

end total_campers_went_rowing_l2151_215108


namespace first_term_geometric_sequence_b_n_bounded_l2151_215158

-- Definition: S_n = 3a_n - 5n for any n in ℕ*
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 3 * a n - 5 * n

-- The sequence a_n is given such that
-- Proving the first term a_1
theorem first_term (a : ℕ → ℝ) (h : ∀ n, S (n + 1) a = S n a + a n + 1 - 5) : 
  a 1 = 5 / 2 :=
sorry

-- Prove that {a_n + 5} is a geometric sequence with common ratio 3/2
theorem geometric_sequence (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 5 * n) : 
  ∃ r, (∀ n, a (n + 1) + 5 = r * (a n + 5)) ∧ r = 3 / 2 :=
sorry

-- Prove that there exists m such that b_n < m always holds for b_n = (9n + 4) / (a_n + 5)
theorem b_n_bounded (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, b n = (9 * ↑n + 4) / (a n + 5)) 
  (h2 : ∀ n, a n = (15 / 2) * (3 / 2)^(n-1) - 5) :
  ∃ m, ∀ n, b n < m ∧ m = 88 / 45 :=
sorry

end first_term_geometric_sequence_b_n_bounded_l2151_215158


namespace find_Natisfy_condition_l2151_215195

-- Define the original number
def N : Nat := 2173913043478260869565

-- Define the function to move the first digit of a number to the end
def move_first_digit_to_end (n : Nat) : Nat := sorry

-- The proof statement
theorem find_Natisfy_condition : 
  let new_num1 := N * 4
  let new_num2 := new_num1 / 5
  move_first_digit_to_end N = new_num2 
:=
  sorry

end find_Natisfy_condition_l2151_215195


namespace stratified_sampling_sophomores_selected_l2151_215116

theorem stratified_sampling_sophomores_selected 
  (total_freshmen : ℕ) (total_sophomores : ℕ) (total_seniors : ℕ) 
  (freshmen_selected : ℕ) (selection_ratio : ℕ) :
  total_freshmen = 210 →
  total_sophomores = 270 →
  total_seniors = 300 →
  freshmen_selected = 7 →
  selection_ratio = total_freshmen / freshmen_selected →
  selection_ratio = 30 →
  total_sophomores / selection_ratio = 9 :=
by sorry

end stratified_sampling_sophomores_selected_l2151_215116


namespace find_sample_size_l2151_215150

-- Definitions based on conditions
def ratio_students : ℕ := 2 + 3 + 5
def grade12_ratio : ℚ := 5 / ratio_students
def sample_grade12_students : ℕ := 150

-- The goal is to find n such that the proportion is maintained
theorem find_sample_size (n : ℕ) (h : grade12_ratio = sample_grade12_students / ↑n) : n = 300 :=
by sorry


end find_sample_size_l2151_215150


namespace find_g_3_l2151_215137

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 3 = 21 :=
by
  sorry

end find_g_3_l2151_215137


namespace remaining_leaves_l2151_215177

def initial_leaves := 1000
def first_week_shed := (2 / 5 : ℚ) * initial_leaves
def leaves_after_first_week := initial_leaves - first_week_shed
def second_week_shed := (40 / 100 : ℚ) * leaves_after_first_week
def leaves_after_second_week := leaves_after_first_week - second_week_shed
def third_week_shed := (3 / 4 : ℚ) * second_week_shed
def leaves_after_third_week := leaves_after_second_week - third_week_shed

theorem remaining_leaves (initial_leaves first_week_shed leaves_after_first_week second_week_shed leaves_after_second_week third_week_shed leaves_after_third_week: ℚ) : 
  leaves_after_third_week = 180 := by
  sorry

end remaining_leaves_l2151_215177


namespace smallest_divisor_l2151_215143

theorem smallest_divisor (n : ℕ) (h1 : n = 999) :
  ∃ d : ℕ, 2.45 ≤ (999 : ℝ) / d ∧ (999 : ℝ) / d < 2.55 ∧ d = 392 :=
by
  sorry

end smallest_divisor_l2151_215143


namespace profit_per_meter_is_20_l2151_215127

-- Define given conditions
def selling_price_total (n : ℕ) (price : ℕ) : ℕ := n * price
def cost_price_per_meter : ℕ := 85
def selling_price_total_85_meters : ℕ := 8925

-- Define the expected profit per meter
def expected_profit_per_meter : ℕ := 20

-- Rewrite the problem statement: Prove that with given conditions the profit per meter is Rs. 20
theorem profit_per_meter_is_20 
  (n : ℕ := 85)
  (sp : ℕ := selling_price_total_85_meters)
  (cp_pm : ℕ := cost_price_per_meter) 
  (expected_profit : ℕ := expected_profit_per_meter) :
  (sp - n * cp_pm) / n = expected_profit :=
by
  sorry

end profit_per_meter_is_20_l2151_215127


namespace unattainable_y_ne_l2151_215128

theorem unattainable_y_ne : ∀ x : ℝ, x ≠ -5/4 → y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3/4 :=
by
  sorry

end unattainable_y_ne_l2151_215128


namespace Danielle_rooms_is_6_l2151_215142

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l2151_215142


namespace find_cost_price_per_meter_l2151_215101

noncomputable def cost_price_per_meter
  (total_cloth : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_cloth) / total_cloth

theorem find_cost_price_per_meter :
  cost_price_per_meter 75 4950 15 = 51 :=
by
  unfold cost_price_per_meter
  sorry

end find_cost_price_per_meter_l2151_215101


namespace honey_harvest_this_year_l2151_215111

def last_year_harvest : ℕ := 2479
def increase_this_year : ℕ := 6085

theorem honey_harvest_this_year : last_year_harvest + increase_this_year = 8564 :=
by {
  sorry
}

end honey_harvest_this_year_l2151_215111


namespace flavors_remaining_to_try_l2151_215190

def total_flavors : ℕ := 100
def flavors_tried_two_years_ago (total_flavors : ℕ) : ℕ := total_flavors / 4
def flavors_tried_last_year (flavors_tried_two_years_ago : ℕ) : ℕ := 2 * flavors_tried_two_years_ago

theorem flavors_remaining_to_try
  (total_flavors : ℕ)
  (flavors_tried_two_years_ago : ℕ)
  (flavors_tried_last_year : ℕ) :
  flavors_tried_two_years_ago = total_flavors / 4 →
  flavors_tried_last_year = 2 * flavors_tried_two_years_ago →
  total_flavors - (flavors_tried_two_years_ago + flavors_tried_last_year) = 25 :=
by
  sorry

end flavors_remaining_to_try_l2151_215190


namespace cindy_total_time_to_travel_one_mile_l2151_215172

-- Definitions for the conditions
def run_speed : ℝ := 3 -- Cindy's running speed in miles per hour.
def walk_speed : ℝ := 1 -- Cindy's walking speed in miles per hour.
def run_distance : ℝ := 0.5 -- Distance run by Cindy in miles.
def walk_distance : ℝ := 0.5 -- Distance walked by Cindy in miles.

-- Theorem statement
theorem cindy_total_time_to_travel_one_mile : 
  ((run_distance / run_speed) + (walk_distance / walk_speed)) * 60 = 40 := 
by
  sorry

end cindy_total_time_to_travel_one_mile_l2151_215172


namespace yunas_math_score_l2151_215196

theorem yunas_math_score (K E M : ℕ) 
  (h1 : (K + E) / 2 = 92) 
  (h2 : (K + E + M) / 3 = 94) : 
  M = 98 :=
sorry

end yunas_math_score_l2151_215196


namespace initial_cupcakes_l2151_215173

   theorem initial_cupcakes (X : ℕ) (condition : X - 20 + 20 = 26) : X = 26 :=
   by
     sorry
   
end initial_cupcakes_l2151_215173


namespace length_of_DE_in_triangle_l2151_215126

noncomputable def triangle_length_DE (BC : ℝ) (C_deg: ℝ) (DE : ℝ) : Prop :=
  BC = 24 * Real.sqrt 2 ∧ C_deg = 45 ∧ DE = 12 * Real.sqrt 2

theorem length_of_DE_in_triangle :
  ∀ (BC : ℝ) (C_deg: ℝ) (DE : ℝ), (BC = 24 * Real.sqrt 2 ∧ C_deg = 45) → DE = 12 * Real.sqrt 2 :=
by
  intros BC C_deg DE h_cond
  have h_length := h_cond.2
  sorry

end length_of_DE_in_triangle_l2151_215126


namespace volume_of_cube_surface_area_times_l2151_215184

theorem volume_of_cube_surface_area_times (V1 : ℝ) (hV1 : V1 = 8) : 
  ∃ V2, V2 = 24 * Real.sqrt 3 :=
sorry

end volume_of_cube_surface_area_times_l2151_215184


namespace sophomores_selected_correct_l2151_215164

-- Define the number of students in each grade and the total spots for the event
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300
def totalSpots : ℕ := 40

-- Calculate the total number of students
def totalStudents : ℕ := freshmen + sophomores + juniors

-- The correct answer we want to prove
def numberOfSophomoresSelected : ℕ := (sophomores * totalSpots) / totalStudents

-- Statement to be proved
theorem sophomores_selected_correct : numberOfSophomoresSelected = 26 := by
  -- Proof is omitted
  sorry

end sophomores_selected_correct_l2151_215164


namespace find_integer_l2151_215129

theorem find_integer (a b c d : ℕ) (h1 : a + b + c + d = 18) 
  (h2 : b + c = 11) (h3 : a - d = 3) (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  10^3 * a + 10^2 * b + 10 * c + d = 5262 ∨ 10^3 * a + 10^2 * b + 10 * c + d = 5622 := 
by
  sorry

end find_integer_l2151_215129


namespace distinct_roots_condition_l2151_215197

noncomputable def f (x c : ℝ) : ℝ := x^2 + 6*x + c

theorem distinct_roots_condition (c : ℝ) :
  (∀x : ℝ, f (f x c) = 0 → ∃ a b : ℝ, (a ≠ b) ∧ f x c = a * (x - b) * (x - c) ) →
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end distinct_roots_condition_l2151_215197


namespace min_value_expr_l2151_215118

noncomputable def find_min_value (a b c d : ℝ) (x y : ℝ) : ℝ :=
  x / c^2 + y^2 / d^2

theorem min_value_expr (a b c d : ℝ) (h : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) :
  ∃ x y : ℝ, find_min_value a b c d x y = -abs a / c^2 := 
sorry

end min_value_expr_l2151_215118


namespace largest_number_systematic_sampling_l2151_215153

theorem largest_number_systematic_sampling (n k a1 a2: ℕ) (h1: n = 60) (h2: a1 = 3) (h3: a2 = 9) (h4: k = a2 - a1):
  ∃ largest, largest = a1 + k * (n / k - 1) := by
  sorry

end largest_number_systematic_sampling_l2151_215153


namespace system1_solution_system2_solution_l2151_215183

-- Statement for the Part 1 Equivalent Problem.
theorem system1_solution :
  ∀ (x y : ℤ),
    (x - 3 * y = -10) ∧ (x + y = 6) → (x = 2 ∧ y = 4) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

-- Statement for the Part 2 Equivalent Problem.
theorem system2_solution :
  ∀ (x y : ℚ),
    (x / 2 - (y - 1) / 3 = 1) ∧ (4 * x - y = 8) → (x = 12 / 5 ∧ y = 8 / 5) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

end system1_solution_system2_solution_l2151_215183


namespace find_ordered_pair_l2151_215102

theorem find_ordered_pair (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  18 * m * n = 72 - 9 * m - 4 * n ↔ (m = 8 ∧ n = 36) := 
by 
  sorry

end find_ordered_pair_l2151_215102


namespace find_f_m_eq_neg_one_l2151_215131

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^(2 - m)

theorem find_f_m_eq_neg_one (m : ℝ)
  (h1 : ∀ x : ℝ, f x m = - f (-x) m) (h2 : m^2 - m = 3 + m) :
  f m m = -1 :=
by
  sorry

end find_f_m_eq_neg_one_l2151_215131


namespace percent_of_number_l2151_215148

theorem percent_of_number (N : ℝ) (h : (4 / 5) * (3 / 8) * N = 24) : 2.5 * N = 200 :=
by
  sorry

end percent_of_number_l2151_215148


namespace parabola_y_relation_l2151_215112

-- Conditions of the problem
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- The proof problem statement
theorem parabola_y_relation (c y1 y2 y3 : ℝ) :
  parabola (-4) c = y1 →
  parabola (-2) c = y2 →
  parabola (1 / 2) c = y3 →
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_y_relation_l2151_215112


namespace kayla_less_than_vika_l2151_215191

variable (S K V : ℕ)
variable (h1 : S = 216)
variable (h2 : S = 4 * K)
variable (h3 : V = 84)

theorem kayla_less_than_vika (S K V : ℕ) (h1 : S = 216) (h2 : S = 4 * K) (h3 : V = 84) : V - K = 30 :=
by
  sorry

end kayla_less_than_vika_l2151_215191


namespace largest_prime_factor_of_1729_is_19_l2151_215171

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) := is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_1729_is_19 : largest_prime_factor 1729 19 :=
by
  sorry

end largest_prime_factor_of_1729_is_19_l2151_215171


namespace double_root_equation_correct_statements_l2151_215194

theorem double_root_equation_correct_statements
  (a b c : ℝ) (r₁ r₂ : ℝ)
  (h1 : a ≠ 0)
  (h2 : r₁ = 2 * r₂)
  (h3 : r₁ ≠ r₂)
  (h4 : a * r₁ ^ 2 + b * r₁ + c = 0)
  (h5 : a * r₂ ^ 2 + b * r₂ + c = 0) :
  (∀ (m n : ℝ), (∀ (r : ℝ), r = 2 → (x - r) * (m * x + n) = 0 → 4 * m ^ 2 + 5 * m * n + n ^ 2 = 0)) ∧
  (∀ (p q : ℝ), p * q = 2 → ∃ x, p * x ^ 2 + 3 * x + q = 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ = -1 / p ∧ x₂ = -q ∧ x₁ = 2 * x₂)) ∧
  (2 * b ^ 2 = 9 * a * c) :=
by
  sorry

end double_root_equation_correct_statements_l2151_215194


namespace lewis_speed_l2151_215100

theorem lewis_speed
  (v : ℕ)
  (john_speed : ℕ := 40)
  (distance_AB : ℕ := 240)
  (meeting_distance : ℕ := 160)
  (time_john_to_meeting : ℕ := meeting_distance / john_speed)
  (distance_lewis_traveled : ℕ := distance_AB + (distance_AB - meeting_distance))
  (v_eq : v = distance_lewis_traveled / time_john_to_meeting) :
  v = 80 :=
by
  sorry

end lewis_speed_l2151_215100


namespace sum_of_numbers_l2151_215113

theorem sum_of_numbers (a b c : ℕ) 
  (h1 : a ≤ b ∧ b ≤ c) 
  (h2 : b = 10) 
  (h3 : (a + b + c) / 3 = a + 15) 
  (h4 : (a + b + c) / 3 = c - 20) 
  (h5 : c = 2 * a)
  : a + b + c = 115 := by
  sorry

end sum_of_numbers_l2151_215113


namespace blueberries_in_each_blue_box_l2151_215159

theorem blueberries_in_each_blue_box (S B : ℕ) (h1 : S - B = 12) (h2 : 2 * S = 76) : B = 26 := by
  sorry

end blueberries_in_each_blue_box_l2151_215159


namespace range_of_x_l2151_215187

theorem range_of_x (f : ℝ → ℝ) (h_increasing : ∀ x y, x ≤ y → f x ≤ f y) (h_defined : ∀ x, -1 ≤ x ∧ x ≤ 1)
  (h_condition : ∀ x, f (x-2) < f (1-x)) : ∀ x, 1 ≤ x ∧ x < 3/2 :=
by
  sorry

end range_of_x_l2151_215187


namespace minimum_value_l2151_215169

theorem minimum_value (n : ℝ) (h : n > 0) : n + 32 / n^2 ≥ 6 := 
sorry

end minimum_value_l2151_215169


namespace find_initial_music_files_l2151_215103

-- Define the initial state before any deletion
def initial_files (music_files : ℕ) (video_files : ℕ) : ℕ := music_files + video_files

-- Define the state after deleting files
def files_after_deletion (initial_files : ℕ) (deleted_files : ℕ) : ℕ := initial_files - deleted_files

-- Theorem to prove that the initial number of music files was 13
theorem find_initial_music_files 
  (video_files : ℕ) (deleted_files : ℕ) (remaining_files : ℕ) 
  (h_videos : video_files = 30) (h_deleted : deleted_files = 10) (h_remaining : remaining_files = 33) : 
  ∃ (music_files : ℕ), initial_files music_files video_files - deleted_files = remaining_files ∧ music_files = 13 :=
by {
  sorry
}

end find_initial_music_files_l2151_215103


namespace yulia_profit_l2151_215120

-- Assuming the necessary definitions in the problem
def lemonade_revenue : ℕ := 47
def babysitting_revenue : ℕ := 31
def expenses : ℕ := 34
def profit : ℕ := lemonade_revenue + babysitting_revenue - expenses

-- The proof statement to prove Yulia's profit
theorem yulia_profit : profit = 44 := by
  sorry -- Proof is skipped

end yulia_profit_l2151_215120


namespace jan_discount_percentage_l2151_215181

theorem jan_discount_percentage :
  ∃ percent_discount : ℝ,
    ∀ (roses_bought dozen : ℕ) (rose_cost amount_paid : ℝ),
      roses_bought = 5 * dozen → dozen = 12 →
      rose_cost = 6 →
      amount_paid = 288 →
      (roses_bought * rose_cost - amount_paid) / (roses_bought * rose_cost) * 100 = percent_discount →
      percent_discount = 20 :=
by
  sorry

end jan_discount_percentage_l2151_215181


namespace find_pos_integers_A_B_l2151_215167

noncomputable def concat (A B : ℕ) : ℕ :=
  let b := Nat.log 10 B + 1
  A * 10 ^ b + B

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def satisfiesConditions (A B : ℕ) : Prop :=
  isPerfectSquare (concat A B) ∧ concat A B = 2 * A * B

theorem find_pos_integers_A_B :
  ∃ (A B : ℕ), A = (5 ^ b + 1) / 2 ∧ B = 2 ^ b * A * 100 ^ m ∧ b % 2 = 1 ∧ ∀ m : ℕ, satisfiesConditions A B :=
sorry

end find_pos_integers_A_B_l2151_215167


namespace per_minute_charge_after_6_minutes_l2151_215121

noncomputable def cost_plan_a (x : ℝ) (t : ℝ) : ℝ :=
  if t <= 6 then 0.60 else 0.60 + (t - 6) * x

noncomputable def cost_plan_b (t : ℝ) : ℝ :=
  t * 0.08

theorem per_minute_charge_after_6_minutes :
  ∃ (x : ℝ), cost_plan_a x 12 = cost_plan_b 12 ∧ x = 0.06 :=
by
  use 0.06
  simp [cost_plan_a, cost_plan_b]
  sorry

end per_minute_charge_after_6_minutes_l2151_215121


namespace seq_a_n_value_l2151_215132

theorem seq_a_n_value (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1)) :
  a 10 = 19 :=
sorry

end seq_a_n_value_l2151_215132


namespace remainder_of_polynomial_division_l2151_215170

-- Definitions based on conditions in the problem
def polynomial (x : ℝ) : ℝ := 8 * x^4 - 22 * x^3 + 9 * x^2 + 10 * x - 45

def divisor (x : ℝ) : ℝ := 4 * x - 8

-- Proof statement as per the problem equivalence
theorem remainder_of_polynomial_division : polynomial 2 = -37 := by
  sorry

end remainder_of_polynomial_division_l2151_215170


namespace task1_task2_l2151_215149

-- Define the conditions and the probabilities to be proven

def total_pens := 6
def first_class_pens := 3
def second_class_pens := 2
def third_class_pens := 1

def total_combinations := Nat.choose total_pens 2

def combinations_with_exactly_one_first_class : Nat :=
  (first_class_pens * (total_pens - first_class_pens))

def probability_one_first_class_pen : ℚ :=
  combinations_with_exactly_one_first_class / total_combinations

def combinations_without_any_third_class : Nat :=
  Nat.choose (first_class_pens + second_class_pens) 2

def probability_no_third_class_pen : ℚ :=
  combinations_without_any_third_class / total_combinations

theorem task1 : probability_one_first_class_pen = 3 / 5 := 
  sorry

theorem task2 : probability_no_third_class_pen = 2 / 3 := 
  sorry

end task1_task2_l2151_215149


namespace division_result_l2151_215136

theorem division_result : 203515 / 2015 = 101 := 
by sorry

end division_result_l2151_215136


namespace tenth_term_arithmetic_seq_l2151_215105

theorem tenth_term_arithmetic_seq :
  let a₁ : ℚ := 1 / 2
  let a₂ : ℚ := 5 / 6
  let d : ℚ := a₂ - a₁
  let a₁₀ : ℚ := a₁ + 9 * d
  a₁₀ = 7 / 2 :=
by
  sorry

end tenth_term_arithmetic_seq_l2151_215105


namespace final_reduced_price_l2151_215160

noncomputable def original_price (P : ℝ) (Q : ℝ) : ℝ := 800 / Q

noncomputable def price_after_first_week (P : ℝ) : ℝ := 0.90 * P
noncomputable def price_after_second_week (price1 : ℝ) : ℝ := 0.85 * price1
noncomputable def price_after_third_week (price2 : ℝ) : ℝ := 0.80 * price2

noncomputable def reduced_price (P : ℝ) : ℝ :=
  let price1 := price_after_first_week P
  let price2 := price_after_second_week price1
  price_after_third_week price2

theorem final_reduced_price :
  ∃ P Q : ℝ, 
    800 = Q * P ∧
    800 = (Q + 5) * reduced_price P ∧
    abs (reduced_price P - 62.06) < 0.01 :=
by
  sorry

end final_reduced_price_l2151_215160


namespace absolute_value_inequality_l2151_215157

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by
  sorry

end absolute_value_inequality_l2151_215157


namespace first_two_digits_of_1666_l2151_215114

/-- Lean 4 statement for the given problem -/
theorem first_two_digits_of_1666 (y k : ℕ) (H_nonzero_k : k ≠ 0) (H_nonzero_y : y ≠ 0) (H_y_six : y = 6) :
  (1666 / 100) = 16 := by
  sorry

end first_two_digits_of_1666_l2151_215114


namespace find_c_plus_one_over_b_l2151_215134

theorem find_c_plus_one_over_b 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (h1 : a * b * c = 1) 
  (h2 : a + 1 / c = 8) 
  (h3 : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := 
sorry

end find_c_plus_one_over_b_l2151_215134


namespace jenny_ate_more_than_thrice_mike_l2151_215130

theorem jenny_ate_more_than_thrice_mike :
  let mike_ate := 20
  let jenny_ate := 65
  jenny_ate - 3 * mike_ate = 5 :=
by
  let mike_ate := 20
  let jenny_ate := 65
  have : jenny_ate - 3 * mike_ate = 5 := by
    sorry
  exact this

end jenny_ate_more_than_thrice_mike_l2151_215130


namespace distance_vancouver_calgary_l2151_215141

theorem distance_vancouver_calgary : 
  ∀ (map_distance : ℝ) (scale : ℝ) (terrain_factor : ℝ), 
    map_distance = 12 →
    scale = 35 →
    terrain_factor = 1.1 →
    map_distance * scale * terrain_factor = 462 := by
  intros map_distance scale terrain_factor 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end distance_vancouver_calgary_l2151_215141


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l2151_215109

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l2151_215109


namespace find_principal_l2151_215152

theorem find_principal
  (P R : ℝ)
  (h : (P * (R + 2) * 7) / 100 = (P * R * 7) / 100 + 140) :
  P = 1000 := by
sorry

end find_principal_l2151_215152


namespace part_I_solution_part_II_solution_l2151_215147

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

theorem part_I_solution :
  ∀ x : ℝ, f x 3 ≥ 1 ↔ 0 ≤ x ∧ x ≤ (4 / 3) := by
  sorry

theorem part_II_solution :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x a - |2*x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) := by
  sorry

end part_I_solution_part_II_solution_l2151_215147


namespace farmer_total_profit_l2151_215154

theorem farmer_total_profit :
  let group1_revenue := 3 * 375
  let group1_cost := (8 * 13 + 3 * 15) * 3
  let group1_profit := group1_revenue - group1_cost

  let group2_revenue := 4 * 425
  let group2_cost := (5 * 14 + 9 * 16) * 4
  let group2_profit := group2_revenue - group2_cost

  let group3_revenue := 2 * 475
  let group3_cost := (10 * 15 + 8 * 18) * 2
  let group3_profit := group3_revenue - group3_cost

  let group4_revenue := 1 * 550
  let group4_cost := 20 * 20 * 1
  let group4_profit := group4_revenue - group4_cost

  let total_profit := group1_profit + group2_profit + group3_profit + group4_profit
  total_profit = 2034 :=
by
  sorry

end farmer_total_profit_l2151_215154


namespace chocolates_divisible_l2151_215179

theorem chocolates_divisible (n m : ℕ) (h1 : n > 0) (h2 : m > 0) : 
  (n ≤ m) ∨ (m % (n - m) = 0) :=
sorry

end chocolates_divisible_l2151_215179


namespace transformation_identity_l2151_215189

theorem transformation_identity (n : Nat) (h : 2 ≤ n) : 
  n * Real.sqrt (n / (n ^ 2 - 1)) = Real.sqrt (n + n / (n ^ 2 - 1)) := 
sorry

end transformation_identity_l2151_215189


namespace amount_C_l2151_215138

theorem amount_C (A B C : ℕ) 
  (h₁ : A + B + C = 900) 
  (h₂ : A + C = 400) 
  (h₃ : B + C = 750) : 
  C = 250 :=
sorry

end amount_C_l2151_215138


namespace example_one_example_two_l2151_215163

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l2151_215163


namespace amanda_car_round_trip_time_l2151_215174

theorem amanda_car_round_trip_time :
  (bus_time = 40) ∧ (car_time = bus_time - 5) → (round_trip_time = car_time * 2) → round_trip_time = 70 :=
by
  sorry

end amanda_car_round_trip_time_l2151_215174


namespace circle_ratio_increase_l2151_215135

theorem circle_ratio_increase (r : ℝ) (h : r + 2 ≠ 0) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by
  sorry

end circle_ratio_increase_l2151_215135


namespace factorization_result_l2151_215178

theorem factorization_result :
  ∃ (c d : ℕ), (c > d) ∧ ((x^2 - 20 * x + 91) = (x - c) * (x - d)) ∧ (2 * d - c = 1) :=
by
  -- Using the conditions and proving the given equation
  sorry

end factorization_result_l2151_215178


namespace geo_seq_bn_plus_2_general_formula_an_l2151_215161

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 2 = 4
axiom h3 : ∀ n, b n = a (n + 1) - a n
axiom h4 : ∀ n, b (n + 1) = 2 * b n + 2

-- Proof goals
theorem geo_seq_bn_plus_2 : (∀ n, ∃ r : ℕ, b n + 2 = 4 * 2^n) :=
  sorry

theorem general_formula_an : (∀ n, a n = 2^(n + 1) - 2 * n) :=
  sorry

end geo_seq_bn_plus_2_general_formula_an_l2151_215161


namespace money_given_by_school_correct_l2151_215144

-- Definitions from the problem conditions
def cost_per_book : ℕ := 12
def number_of_students : ℕ := 30
def out_of_pocket : ℕ := 40

-- Derived definition from these conditions
def total_cost : ℕ := cost_per_book * number_of_students
def money_given_by_school : ℕ := total_cost - out_of_pocket

-- The theorem stating that the amount given by the school is $320
theorem money_given_by_school_correct : money_given_by_school = 320 :=
by
  sorry -- Proof placeholder

end money_given_by_school_correct_l2151_215144


namespace perfect_square_trinomial_l2151_215151

theorem perfect_square_trinomial (k : ℝ) : (∃ a b : ℝ, (a * x + b) ^ 2 = x^2 - k * x + 4) → (k = 4 ∨ k = -4) :=
by
  sorry

end perfect_square_trinomial_l2151_215151


namespace value_of_x_l2151_215156

theorem value_of_x
  (x : ℝ)
  (h1 : x = 0)
  (h2 : x^2 - 1 ≠ 0) :
  (x = 0) ↔ (x ^ 2 - 1 ≠ 0) :=
by
  sorry

end value_of_x_l2151_215156


namespace part_I_part_II_l2151_215193

-- Definition of the sequence a_n with given conditions
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else (n^2 + n) / 2

-- Define the sum of the first n terms S_n
def S_n (n : ℕ) : ℕ :=
  (n + 2) / 3 * a_n n

-- Define the sequence b_n in terms of a_n
def b_n (n : ℕ) : ℚ := 1 / a_n n

-- Define the sum of the first n terms of b_n
def T_n (n : ℕ) : ℚ :=
  2 * (1 - 1 / (n + 1))

-- Theorem statement for part (I)
theorem part_I (n : ℕ) : 
  a_n 2 = 3 ∧ a_n 3 = 6 ∧ (∀ (n : ℕ), n ≥ 2 → a_n n = (n^2 + n) / 2) := sorry

-- Theorem statement for part (II)
theorem part_II (n : ℕ) : 
  T_n n = 2 * (1 - 1 / (n + 1)) := sorry

end part_I_part_II_l2151_215193


namespace seven_books_cost_l2151_215123

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l2151_215123


namespace find_b_l2151_215192

theorem find_b (b : ℚ) (m : ℚ) 
  (h1 : x^2 + b*x + 1/6 = (x + m)^2 + 1/18) 
  (h2 : b < 0) : 
  b = -2/3 := 
sorry

end find_b_l2151_215192


namespace tan_five_pi_over_four_l2151_215117

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l2151_215117


namespace eval_polynomial_correct_l2151_215186

theorem eval_polynomial_correct (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) (hy_pos : 0 < y) :
  y^3 - 3 * y^2 - 9 * y + 3 = 3 :=
sorry

end eval_polynomial_correct_l2151_215186


namespace total_distance_trip_l2151_215110

-- Defining conditions
def time_paved := 2 -- hours
def time_dirt := 3 -- hours
def speed_dirt := 32 -- mph
def speed_paved := speed_dirt + 20 -- mph

-- Defining distances
def distance_dirt := speed_dirt * time_dirt -- miles
def distance_paved := speed_paved * time_paved -- miles

-- Proving total distance
theorem total_distance_trip : distance_dirt + distance_paved = 200 := by
  sorry

end total_distance_trip_l2151_215110


namespace no_real_solutions_l2151_215199

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
    (a = 0) ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a ^ 2 > 0) :=
by
  sorry

end no_real_solutions_l2151_215199


namespace log_expression_value_l2151_215165

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log_expression_value :
  (log2 8 * (log2 2 / log2 8)) + log2 4 = 3 :=
by
  sorry

end log_expression_value_l2151_215165


namespace six_identities_l2151_215125

theorem six_identities :
    (∀ x, (2 * x - 1) * (x - 3) = 2 * x^2 - 7 * x + 3) ∧
    (∀ x, (2 * x + 1) * (x + 3) = 2 * x^2 + 7 * x + 3) ∧
    (∀ x, (2 - x) * (1 - 3 * x) = 2 - 7 * x + 3 * x^2) ∧
    (∀ x, (2 + x) * (1 + 3 * x) = 2 + 7 * x + 3 * x^2) ∧
    (∀ x y, (2 * x - y) * (x - 3 * y) = 2 * x^2 - 7 * x * y + 3 * y^2) ∧
    (∀ x y, (2 * x + y) * (x + 3 * y) = 2 * x^2 + 7 * x * y + 3 * y^2) →
    6 = 6 :=
by
  intros
  sorry

end six_identities_l2151_215125


namespace probability_three_digit_multiple_5_remainder_3_div_7_l2151_215198

theorem probability_three_digit_multiple_5_remainder_3_div_7 :
  (∃ (P : ℝ), P = (26 / 900)) := 
by sorry

end probability_three_digit_multiple_5_remainder_3_div_7_l2151_215198


namespace probability_single_trial_l2151_215180

theorem probability_single_trial (p : ℚ) (h₁ : (1 - p)^4 = 16 / 81) : p = 1 / 3 :=
sorry

end probability_single_trial_l2151_215180


namespace E_plays_2_games_l2151_215145

-- Definitions for the students and the number of games they played
def students := ["A", "B", "C", "D", "E"]
def games_played_by (S : String) : Nat :=
  if S = "A" then 4 else
  if S = "B" then 3 else
  if S = "C" then 2 else 
  if S = "D" then 1 else
  2  -- this is the number of games we need to prove for student E 

-- Theorem stating the number of games played by E
theorem E_plays_2_games : games_played_by "E" = 2 :=
  sorry

end E_plays_2_games_l2151_215145


namespace heather_total_distance_l2151_215106

theorem heather_total_distance :
  let d1 := 0.3333333333333333
  let d2 := 0.3333333333333333
  let d3 := 0.08333333333333333
  d1 + d2 + d3 = 0.75 :=
by
  sorry

end heather_total_distance_l2151_215106


namespace b_is_nth_power_l2151_215168

theorem b_is_nth_power (b n : ℕ) (h1 : b > 1) (h2 : n > 1) 
    (h3 : ∀ k > 1, ∃ a_k : ℕ, k ∣ (b - a_k^n)) : 
    ∃ A : ℕ, b = A^n :=
sorry

end b_is_nth_power_l2151_215168


namespace radius_smaller_circle_l2151_215185

theorem radius_smaller_circle (A₁ A₂ A₃ : ℝ) (s : ℝ)
  (h1 : A₁ + A₂ = 12 * Real.pi)
  (h2 : A₃ = (Real.sqrt 3 / 4) * s^2)
  (h3 : 2 * A₂ = A₁ + A₁ + A₂ + A₃) :
  ∃ r : ℝ, r = Real.sqrt (6 - (Real.sqrt 3 / 8) * s^2) := by
  sorry

end radius_smaller_circle_l2151_215185


namespace sum_of_two_squares_l2151_215107

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 :=
by sorry

end sum_of_two_squares_l2151_215107


namespace least_number_of_plates_needed_l2151_215124

theorem least_number_of_plates_needed
  (cubes : ℕ)
  (cube_dim : ℕ)
  (temp_limit : ℕ)
  (plates_exist : ∀ (n : ℕ), n > temp_limit → ∃ (p : ℕ), p = 21) :
  cubes = 512 ∧ cube_dim = 8 → temp_limit > 0 → 21 = 7 + 7 + 7 :=
by {
  sorry
}

end least_number_of_plates_needed_l2151_215124


namespace total_savings_correct_l2151_215115

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end total_savings_correct_l2151_215115


namespace certain_number_is_7000_l2151_215104

theorem certain_number_is_7000 (x : ℕ) (h1 : 1 / 10 * (1 / 100 * x) = x / 1000)
    (h2 : 1 / 10 * x = x / 10)
    (h3 : x / 10 - x / 1000 = 693) : 
  x = 7000 := 
sorry

end certain_number_is_7000_l2151_215104


namespace gcd_459_357_l2151_215122

/-- Prove that the greatest common divisor of 459 and 357 is 51. -/
theorem gcd_459_357 : gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l2151_215122


namespace Jame_tears_30_cards_at_a_time_l2151_215176

theorem Jame_tears_30_cards_at_a_time
    (cards_per_deck : ℕ)
    (times_per_week : ℕ)
    (decks : ℕ)
    (weeks : ℕ)
    (total_cards : ℕ := decks * cards_per_deck)
    (total_times : ℕ := weeks * times_per_week)
    (cards_at_a_time : ℕ := total_cards / total_times)
    (h1 : cards_per_deck = 55)
    (h2 : times_per_week = 3)
    (h3 : decks = 18)
    (h4 : weeks = 11) :
    cards_at_a_time = 30 := by
  -- Proof can be added here
  sorry

end Jame_tears_30_cards_at_a_time_l2151_215176


namespace find_percent_defective_l2151_215139

def percent_defective (D : ℝ) : Prop :=
  (0.04 * D = 0.32)

theorem find_percent_defective : ∃ D, percent_defective D ∧ D = 8 := by
  sorry

end find_percent_defective_l2151_215139
