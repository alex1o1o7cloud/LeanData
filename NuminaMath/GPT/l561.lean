import Mathlib

namespace shorter_leg_in_right_triangle_l561_56103

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l561_56103


namespace find_f_neg2014_l561_56130

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem find_f_neg2014 (a b : ℝ) (h : f 2014 a b = 3) : f (-2014) a b = -7 :=
by sorry

end find_f_neg2014_l561_56130


namespace cost_comparison_l561_56150

-- Definitions based on the given conditions
def suit_price : ℕ := 200
def tie_price : ℕ := 40
def num_suits : ℕ := 20
def discount_rate : ℚ := 0.9

-- Define cost expressions for the two options
def option1_cost (x : ℕ) : ℕ :=
  (suit_price * num_suits) + (tie_price * (x - num_suits))

def option2_cost (x : ℕ) : ℚ :=
  ((suit_price * num_suits + tie_price * x) * discount_rate : ℚ)

-- Main theorem to prove the given answers
theorem cost_comparison (x : ℕ) (hx : x > 20) :
  option1_cost x = 40 * x + 3200 ∧
  option2_cost x = 3600 + 36 * x ∧
  (x = 30 → option1_cost 30 < option2_cost 30) :=
by
  sorry

end cost_comparison_l561_56150


namespace altitudes_not_form_triangle_l561_56111

theorem altitudes_not_form_triangle (h₁ h₂ h₃ : ℝ) :
  ¬(h₁ = 5 ∧ h₂ = 12 ∧ h₃ = 13 ∧ ∃ a b c : ℝ, a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ ∧
    a < b + c ∧ b < a + c ∧ c < a + b) :=
by sorry

end altitudes_not_form_triangle_l561_56111


namespace find_m_l561_56101

-- Definitions based on conditions
def is_eccentricity (a b c : ℝ) (e : ℝ) : Prop :=
  e = c / a

def ellipse_relation (a b m : ℝ) : Prop :=
  a ^ 2 = 3 ∧ b ^ 2 = m

def eccentricity_square_relation (c a : ℝ) : Prop :=
  (c / a) ^ 2 = 1 / 4

-- Main theorem statement
theorem find_m (m : ℝ) :
  (∀ (a b c : ℝ), ellipse_relation a b m → is_eccentricity a b c (1 / 2) → eccentricity_square_relation c a)
  → (m = 9 / 4 ∨ m = 4) := sorry

end find_m_l561_56101


namespace distance_between_stations_l561_56132

theorem distance_between_stations 
  (distance_P_to_meeting : ℝ)
  (distance_Q_to_meeting : ℝ)
  (h1 : distance_P_to_meeting = 20 * 3)
  (h2 : distance_Q_to_meeting = 25 * 2)
  (h3 : distance_P_to_meeting + distance_Q_to_meeting = D) :
  D = 110 :=
by
  sorry

end distance_between_stations_l561_56132


namespace find_numer_denom_n_l561_56137

theorem find_numer_denom_n (n : ℕ) 
    (h : (2 + n) / (7 + n) = (3 : ℤ) / 4) : n = 13 := sorry

end find_numer_denom_n_l561_56137


namespace tiling_impossible_2003x2003_l561_56165

theorem tiling_impossible_2003x2003 :
  ¬ (∃ (f : Fin 2003 × Fin 2003 → ℕ),
  (∀ p : Fin 2003 × Fin 2003, f p = 1 ∨ f p = 2) ∧
  (∀ p : Fin 2003, (f (p, 0) + f (p, 1)) % 3 = 0) ∧
  (∀ p : Fin 2003, (f (0, p) + f (1, p) + f (2, p)) % 3 = 0)) := 
sorry

end tiling_impossible_2003x2003_l561_56165


namespace simplify_expression_l561_56138

theorem simplify_expression (x y : ℝ) : (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3 * y + x * y^3)⁻¹ * (x + y) :=
by sorry

end simplify_expression_l561_56138


namespace find_t_l561_56128

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2 : ℝ)^(n-1)

noncomputable def S_3n (n : ℕ) : ℝ := (1 - (2 : ℝ)^(3 * n)) / (1 - 2)

noncomputable def a_n_cubed (n : ℕ) : ℝ := (a_n n)^3

noncomputable def T_n (n : ℕ) : ℝ := (1 - (a_n_cubed 2)^n) / (1 - (a_n_cubed 2))

theorem find_t (n : ℕ) : S_3n n = 7 * T_n n :=
by
  sorry

end find_t_l561_56128


namespace value_of_y_l561_56199

theorem value_of_y (y : ℚ) : |4 * y - 6| = 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_l561_56199


namespace percentage_temporary_employees_is_correct_l561_56160

noncomputable def percentage_temporary_employees
    (technicians_percentage : ℝ) (skilled_laborers_percentage : ℝ) (unskilled_laborers_percentage : ℝ)
    (permanent_technicians_percentage : ℝ) (permanent_skilled_laborers_percentage : ℝ)
    (permanent_unskilled_laborers_percentage : ℝ) : ℝ :=
  let total_workers : ℝ := 100
  let total_temporary_technicians := technicians_percentage * (1 - permanent_technicians_percentage / 100)
  let total_temporary_skilled_laborers := skilled_laborers_percentage * (1 - permanent_skilled_laborers_percentage / 100)
  let total_temporary_unskilled_laborers := unskilled_laborers_percentage * (1 - permanent_unskilled_laborers_percentage / 100)
  let total_temporary_workers := total_temporary_technicians + total_temporary_skilled_laborers + total_temporary_unskilled_laborers
  (total_temporary_workers / total_workers) * 100

theorem percentage_temporary_employees_is_correct :
  percentage_temporary_employees 40 35 25 60 45 35 = 51.5 :=
by
  sorry

end percentage_temporary_employees_is_correct_l561_56160


namespace factor_expression_l561_56142

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l561_56142


namespace g_triple_3_eq_31_l561_56109

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 1 else 2 * n - 3

theorem g_triple_3_eq_31 : g (g (g 3)) = 31 := by
  sorry

end g_triple_3_eq_31_l561_56109


namespace intersection_point_l561_56192

noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 10

noncomputable def slope_perp : ℝ := -1/3

noncomputable def line_perp (x : ℝ) : ℝ := slope_perp * x + (2 - slope_perp * 3)

theorem intersection_point : 
  ∃ (x y : ℝ), y = line1 x ∧ y = line_perp x ∧ x = -21 / 10 ∧ y = 37 / 10 :=
by
  sorry

end intersection_point_l561_56192


namespace brooke_butter_price_l561_56116

variables (price_per_gallon_of_milk : ℝ)
variables (gallons_to_butter_conversion : ℝ)
variables (number_of_cows : ℕ)
variables (milk_per_cow : ℝ)
variables (number_of_customers : ℕ)
variables (milk_demand_per_customer : ℝ)
variables (total_earnings : ℝ)

theorem brooke_butter_price :
    price_per_gallon_of_milk = 3 →
    gallons_to_butter_conversion = 2 →
    number_of_cows = 12 →
    milk_per_cow = 4 →
    number_of_customers = 6 →
    milk_demand_per_customer = 6 →
    total_earnings = 144 →
    (total_earnings - number_of_customers * milk_demand_per_customer * price_per_gallon_of_milk) /
    (number_of_cows * milk_per_cow - number_of_customers * milk_demand_per_customer) *
    gallons_to_butter_conversion = 1.50 :=
by { sorry }

end brooke_butter_price_l561_56116


namespace range_of_a_intersection_nonempty_range_of_a_intersection_A_l561_56148

noncomputable def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a_intersection_nonempty (a : ℝ) : (A a ∩ B ≠ ∅) ↔ (a < -1 ∨ a > 2) :=
sorry

theorem range_of_a_intersection_A (a : ℝ) : (A a ∩ B = A a) ↔ (a < -4 ∨ a > 5) :=
sorry

end range_of_a_intersection_nonempty_range_of_a_intersection_A_l561_56148


namespace tree_sidewalk_space_l561_56139

theorem tree_sidewalk_space (num_trees : ℕ) (tree_distance: ℝ) (total_road_length: ℝ): 
  num_trees = 13 → 
  tree_distance = 12 → 
  total_road_length = 157 → 
  (total_road_length - tree_distance * (num_trees - 1)) / num_trees = 1 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end tree_sidewalk_space_l561_56139


namespace grandma_olga_daughters_l561_56198

theorem grandma_olga_daughters :
  ∃ (D : ℕ), ∃ (S : ℕ),
  S = 3 ∧
  (∃ (total_grandchildren : ℕ), total_grandchildren = 33) ∧
  (∀ D', 6 * D' + 5 * S = 33 → D = D')
:=
sorry

end grandma_olga_daughters_l561_56198


namespace work_rate_problem_l561_56102

theorem work_rate_problem 
  (W : ℝ)
  (rate_ab : ℝ)
  (rate_c : ℝ)
  (rate_abc : ℝ)
  (cond1 : rate_c = W / 2)
  (cond2 : rate_abc = W / 1)
  (cond3 : rate_ab = (W / 1) - rate_c) :
  rate_ab = W / 2 :=
by 
  -- We can add the solution steps here, but we skip that part following the guidelines
  sorry

end work_rate_problem_l561_56102


namespace range_of_expr_l561_56154

noncomputable def expr (x y : ℝ) : ℝ := (x + 2 * y + 3) / (x + 1)

theorem range_of_expr : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ x → 4 * x + 3 * y ≤ 12 → 3 ≤ expr x y ∧ expr x y ≤ 11) :=
by
  sorry

end range_of_expr_l561_56154


namespace regular_hexagon_area_decrease_l561_56121

noncomputable def area_decrease (original_area : ℝ) (side_decrease : ℝ) : ℝ :=
  let s := (2 * original_area) / (3 * Real.sqrt 3)
  let new_side := s - side_decrease
  let new_area := (3 * Real.sqrt 3 / 2) * new_side ^ 2
  original_area - new_area

theorem regular_hexagon_area_decrease :
  area_decrease (150 * Real.sqrt 3) 3 = 76.5 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_decrease_l561_56121


namespace perpendicular_line_through_point_l561_56183

theorem perpendicular_line_through_point (m t : ℝ) (h : 2 * m^2 + m + t = 0) :
  m = 1 → t = -3 → (∀ x y : ℝ, m^2 * x + m * y + t = 0 ↔ x + y - 3 = 0) :=
by
  intros hm ht
  subst hm
  subst ht
  sorry

end perpendicular_line_through_point_l561_56183


namespace interval_where_f_increasing_l561_56140

noncomputable def f (x : ℝ) : ℝ := Real.log (4 * x - x^2) / Real.log (1 / 2)

theorem interval_where_f_increasing : ∀ x : ℝ, 2 ≤ x ∧ x < 4 → f x < f (x + 1) :=
by 
  sorry

end interval_where_f_increasing_l561_56140


namespace problem_solution_l561_56147

theorem problem_solution (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 :=
by
  sorry

end problem_solution_l561_56147


namespace range_of_b_over_a_l561_56188

noncomputable def f (a b x : ℝ) : ℝ := (x - a)^3 * (x - b)
noncomputable def g_k (a b k x : ℝ) : ℝ := (f a b x - f a b k) / (x - k)

theorem range_of_b_over_a (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1)
    (hk_inc : ∀ k : ℤ, ∀ x : ℝ, k < x → g_k a b k x ≥ g_k a b k (k + 1)) :
  1 < b / a ∧ b / a ≤ 3 :=
by
  sorry


end range_of_b_over_a_l561_56188


namespace daniel_biked_more_l561_56110

def miles_biked_after_4_hours_more (speed_plain_daniel : ℕ) (speed_plain_elsa : ℕ) (time_plain : ℕ) 
(speed_hilly_daniel : ℕ) (speed_hilly_elsa : ℕ) (time_hilly : ℕ) : ℕ :=
(speed_plain_daniel * time_plain + speed_hilly_daniel * time_hilly) - 
(speed_plain_elsa * time_plain + speed_hilly_elsa * time_hilly)

theorem daniel_biked_more : miles_biked_after_4_hours_more 20 18 3 16 15 1 = 7 :=
by
  sorry

end daniel_biked_more_l561_56110


namespace remove_terms_sum_l561_56118

theorem remove_terms_sum :
  let s := (1/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13 + 1/15 : ℚ)
  s = 16339/15015 →
  (1/13 + 1/15 = 2061/5005) →
  s - (1/13 + 1/15) = 3/2 :=
by
  intros s hs hremove
  have hrem : (s - (1/13 + 1/15 : ℚ) = 3/2) ↔ (16339/15015 - 2061/5005 = 3/2) := sorry
  exact hrem.mpr sorry

end remove_terms_sum_l561_56118


namespace age_ratio_holds_l561_56100

variables (e s : ℕ)

-- Conditions based on the problem statement
def condition_1 : Prop := e - 3 = 2 * (s - 3)
def condition_2 : Prop := e - 5 = 3 * (s - 5)

-- Proposition to prove that in 1 year, the age ratio will be 3:2
def age_ratio_in_one_year : Prop := (e + 1) * 2 = (s + 1) * 3

theorem age_ratio_holds (h1 : condition_1 e s) (h2 : condition_2 e s) : age_ratio_in_one_year e s :=
by {
  sorry
}

end age_ratio_holds_l561_56100


namespace concave_side_probability_l561_56186

theorem concave_side_probability (tosses : ℕ) (frequency_convex : ℝ) (htosses : tosses = 1000) (hfrequency : frequency_convex = 0.44) :
  ∀ probability_concave : ℝ, probability_concave = 1 - frequency_convex → probability_concave = 0.56 :=
by
  intros probability_concave h
  rw [hfrequency] at h
  rw [h]
  norm_num
  done

end concave_side_probability_l561_56186


namespace convert_234_base5_to_binary_l561_56149

def base5_to_decimal (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 4 * 5^0

def decimal_to_binary (n : Nat) : List Nat :=
  let rec to_binary_aux (n : Nat) (accum : List Nat) : List Nat :=
    if n = 0 then accum
    else to_binary_aux (n / 2) ((n % 2) :: accum)
  to_binary_aux n []

theorem convert_234_base5_to_binary :
  (base5_to_decimal 234 = 69) ∧ (decimal_to_binary 69 = [1,0,0,0,1,0,1]) :=
by
  sorry

end convert_234_base5_to_binary_l561_56149


namespace min_AP_l561_56178

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B' : ℝ × ℝ := (8, 6)
def parabola (P' : ℝ × ℝ) : Prop := P'.2^2 = 8 * P'.1

theorem min_AP'_plus_BP' : 
  ∃ P' : ℝ × ℝ, parabola P' ∧ (dist A P' + dist B' P' = 12) := 
sorry

end min_AP_l561_56178


namespace op_identity_l561_56184

-- Define the operation ⊕ as given by the table
def op (x y : ℕ) : ℕ :=
  match (x, y) with
  | (1, 1) => 4
  | (1, 2) => 1
  | (1, 3) => 2
  | (1, 4) => 3
  | (2, 1) => 1
  | (2, 2) => 3
  | (2, 3) => 4
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 4
  | (3, 3) => 1
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 3
  | (4, 4) => 4
  | _ => 0  -- default case for completeness

-- State the theorem
theorem op_identity : op (op 4 1) (op 2 3) = 3 := by
  sorry

end op_identity_l561_56184


namespace find_n_l561_56194

-- Define the variables d, Q, r, m, and n
variables (d Q r m n : ℝ)

-- Define the conditions Q = d / ((1 + r)^n - m) and m < (1 + r)^n
def conditions (d Q r m n : ℝ) : Prop :=
  Q = d / ((1 + r)^n - m) ∧ m < (1 + r)^n

theorem find_n (d Q r m : ℝ) (h : conditions d Q r m n) : 
  n = (Real.log (d / Q + m)) / (Real.log (1 + r)) :=
sorry

end find_n_l561_56194


namespace find_a_l561_56174

theorem find_a (P : ℝ) (hP : P ≠ 0) (S : ℕ → ℝ) (a_n : ℕ → ℝ)
  (hSn : ∀ n, S n = 3^n + a)
  (ha_n : ∀ n, a_n (n + 1) = P * a_n n)
  (hS1 : S 1 = a_n 1)
  (hS2 : S 2 = S 1 + a_n 2 - a_n 1)
  (hS3 : S 3 = S 2 + a_n 3 - a_n 2) :
  a = -1 := sorry

end find_a_l561_56174


namespace unique_solution_is_2_or_minus_2_l561_56181

theorem unique_solution_is_2_or_minus_2 (a : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, (y^2 + a * y + 1 = 0 ↔ y = x)) → (a = 2 ∨ a = -2) :=
by sorry

end unique_solution_is_2_or_minus_2_l561_56181


namespace triangle_inequality_l561_56136

variables {a b c : ℝ} {α : ℝ}

-- Assuming a, b, c are sides of a triangle
def triangle_sides (a b c : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Cosine rule definition
noncomputable def cos_alpha (a b c : ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)

theorem triangle_inequality (h_sides: triangle_sides a b c) (h_cos : α = cos_alpha a b c) :
  (2 * b * c * (cos_alpha a b c)) / (b + c) < b + c - a
  ∧ b + c - a < 2 * b * c / a :=
by
  sorry

end triangle_inequality_l561_56136


namespace sequence_a_n_a31_l561_56166

theorem sequence_a_n_a31 (a : ℕ → ℤ) 
  (h_initial : a 1 = 2)
  (h_recurrence : ∀ n : ℕ, a n + a (n + 1) + n^2 = 0) :
  a 31 = -463 :=
sorry

end sequence_a_n_a31_l561_56166


namespace remainder_9876543210_mod_101_l561_56119

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l561_56119


namespace coord_relationship_M_l561_56133

theorem coord_relationship_M (x y z : ℝ) (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, -1)) (hB : B = (2, 0, 2))
  (hM : ∃ M : ℝ × ℝ × ℝ, M = (x, y, z) ∧ y = 0 ∧ |(1 - x)^2 + 2^2 + (-1 - z)^2| = |(2 - x)^2 + (0 - z)^2|) :
  x + 3 * z - 1 = 0 ∧ y = 0 := 
sorry

end coord_relationship_M_l561_56133


namespace find_integer_n_l561_56187

theorem find_integer_n (n : ℕ) (hn1 : 0 ≤ n) (hn2 : n < 102) (hmod : 99 * n % 102 = 73) : n = 97 :=
  sorry

end find_integer_n_l561_56187


namespace alcohol_added_l561_56197

-- Definitions from conditions
def initial_volume : ℝ := 40
def initial_alcohol_concentration : ℝ := 0.05
def initial_alcohol_amount : ℝ := initial_volume * initial_alcohol_concentration
def added_water_volume : ℝ := 3.5
def final_alcohol_concentration : ℝ := 0.17

-- The problem to be proven
theorem alcohol_added :
  ∃ x : ℝ,
    x = (final_alcohol_concentration * (initial_volume + x + added_water_volume) - initial_alcohol_amount) :=
by
  sorry

end alcohol_added_l561_56197


namespace books_sold_over_summer_l561_56115

theorem books_sold_over_summer (n l t : ℕ) (h1 : n = 37835) (h2 : l = 143) (h3 : t = 271) : 
  t - l = 128 :=
by
  sorry

end books_sold_over_summer_l561_56115


namespace area_triangle_AMC_l561_56167

open Real

-- Definitions: Define the points A, B, C, D such that they form a rectangle
-- Define midpoint M of \overline{AD}

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def A : Point := {x := 0, y := 0}
noncomputable def B : Point := {x := 6, y := 0}
noncomputable def D : Point := {x := 0, y := 8}
noncomputable def C : Point := {x := 6, y := 8}
noncomputable def M : Point := {x := 0, y := 4} -- midpoint of AD

-- Function to compute the area of triangle AMC
noncomputable def triangle_area (A M C : Point) : ℝ :=
  (1 / 2 : ℝ) * abs ((A.x - C.x) * (M.y - A.y) - (A.x - M.x) * (C.y - A.y))

-- The theorem to prove
theorem area_triangle_AMC : triangle_area A M C = 12 :=
by
  sorry

end area_triangle_AMC_l561_56167


namespace n_mod_5_division_of_grid_l561_56189

theorem n_mod_5_division_of_grid (n : ℕ) :
  (∃ m : ℕ, n^2 = 4 + 5 * m) ↔ n % 5 = 2 :=
by
  sorry

end n_mod_5_division_of_grid_l561_56189


namespace four_distinct_sum_equal_l561_56161

theorem four_distinct_sum_equal (S : Finset ℕ) (hS : S.card = 10) (hS_subset : S ⊆ Finset.range 38) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
by
  sorry

end four_distinct_sum_equal_l561_56161


namespace perimeter_of_intersection_triangle_l561_56171

theorem perimeter_of_intersection_triangle :
  ∀ (P Q R : Type) (dist : P → Q → ℝ) (length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR : ℝ),
  (length_PQ = 150) →
  (length_QR = 250) →
  (length_PR = 200) →
  (seg_ellP = 75) →
  (seg_ellQ = 50) →
  (seg_ellR = 25) →
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  TU + US + ST = 266.67 :=
by
  intros P Q R dist length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR hPQ hQR hPR hP hQ hR
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  have : TU + US + ST = 266.67 := sorry
  exact this

end perimeter_of_intersection_triangle_l561_56171


namespace find_t_l561_56177

variables (c o u n t s : ℕ)

theorem find_t (h1 : c + o = u) 
               (h2 : u + n = t)
               (h3 : t + c = s)
               (h4 : o + n + s = 18)
               (hz : c > 0) (ho : o > 0) (hu : u > 0) (hn : n > 0) (ht : t > 0) (hs : s > 0) : 
               t = 9 := 
by
  sorry

end find_t_l561_56177


namespace at_most_n_diameters_l561_56193

theorem at_most_n_diameters {n : ℕ} (h : n ≥ 3) (points : Fin n → ℝ × ℝ) (d : ℝ) 
  (hd : ∀ i j, dist (points i) (points j) ≤ d) :
  ∃ (diameters : Fin n → Fin n), 
    (∀ i, dist (points i) (points (diameters i)) = d) ∧
    (∀ i j, (dist (points i) (points j) = d) → 
      (∃ k, k = i ∨ k = j → diameters k = if k = i then j else i)) :=
sorry

end at_most_n_diameters_l561_56193


namespace largest_unorderable_dumplings_l561_56185

theorem largest_unorderable_dumplings : 
  ∀ (a b c : ℕ), 43 ≠ 6 * a + 9 * b + 20 * c :=
by sorry

end largest_unorderable_dumplings_l561_56185


namespace tournament_min_cost_l561_56108

variables (k : ℕ) (m : ℕ) (S E : ℕ → ℕ)

noncomputable def min_cost (k : ℕ) : ℕ :=
  k * (4 * k^2 + k - 1) / 2

theorem tournament_min_cost (k_pos : 0 < k) (players : m = 2 * k)
  (each_plays_once 
      : ∀ i j, i ≠ j → ∃ d, S d = i ∧ E d = j) -- every two players play once, matches have days
  (one_match_per_day : ∀ d, ∃! i j, i ≠ j ∧ S d = i ∧ E d = j) -- exactly one match per day
  : min_cost k = k * (4 * k^2 + k - 1) / 2 := 
sorry

end tournament_min_cost_l561_56108


namespace difference_one_third_0_333_l561_56141

theorem difference_one_third_0_333 :
  let one_third : ℚ := 1 / 3
  let three_hundred_thirty_three_thousandth : ℚ := 333 / 1000
  one_third - three_hundred_thirty_three_thousandth = 1 / 3000 :=
by
  sorry

end difference_one_third_0_333_l561_56141


namespace sum_areas_of_circles_l561_56176

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l561_56176


namespace gcd_polynomials_l561_56113

-- State the problem in Lean 4.
theorem gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * 2 * k) : 
  Int.gcd (7 * b^2 + 55 * b + 125) (3 * b + 10) = 10 :=
by
  sorry

end gcd_polynomials_l561_56113


namespace x4_plus_inverse_x4_l561_56179

theorem x4_plus_inverse_x4 (x : ℝ) (hx : x ^ 2 + 1 / x ^ 2 = 2) : x ^ 4 + 1 / x ^ 4 = 2 := 
sorry

end x4_plus_inverse_x4_l561_56179


namespace total_cookies_needed_l561_56195

-- Define the conditions
def cookies_per_person : ℝ := 24.0
def number_of_people : ℝ := 6.0

-- Define the goal
theorem total_cookies_needed : cookies_per_person * number_of_people = 144.0 :=
by
  sorry

end total_cookies_needed_l561_56195


namespace center_of_circle_sum_eq_seven_l561_56105

theorem center_of_circle_sum_eq_seven 
  (h k : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = 6 * x + 8 * y - 15 → (x - h)^2 + (y - k)^2 = 10) :
  h + k = 7 := 
sorry

end center_of_circle_sum_eq_seven_l561_56105


namespace cookies_sum_l561_56157

theorem cookies_sum (C : ℕ) (h1 : C % 6 = 5) (h2 : C % 9 = 7) (h3 : C < 80) :
  C = 29 :=
by sorry

end cookies_sum_l561_56157


namespace clear_queue_with_three_windows_l561_56196

def time_to_clear_queue_one_window (a x y : ℕ) : Prop := a / (x - y) = 40

def time_to_clear_queue_two_windows (a x y : ℕ) : Prop := a / (2 * x - y) = 16

theorem clear_queue_with_three_windows (a x y : ℕ) 
  (h1 : time_to_clear_queue_one_window a x y) 
  (h2 : time_to_clear_queue_two_windows a x y) : 
  a / (3 * x - y) = 10 :=
by
  sorry

end clear_queue_with_three_windows_l561_56196


namespace incorrect_transformation_l561_56151

theorem incorrect_transformation (a b : ℤ) : ¬ (a / b = (a + 1) / (b + 1)) :=
sorry

end incorrect_transformation_l561_56151


namespace total_weight_l561_56129

variable (a b c d : ℝ)

-- Conditions
axiom h1 : a + b = 250
axiom h2 : b + c = 235
axiom h3 : c + d = 260
axiom h4 : a + d = 275

-- Proving the total weight
theorem total_weight : a + b + c + d = 510 := by
  sorry

end total_weight_l561_56129


namespace line_through_P_perpendicular_l561_56169

theorem line_through_P_perpendicular 
  (P : ℝ × ℝ) (a b c : ℝ) (hP : P = (-1, 3)) (hline : a = 1 ∧ b = -2 ∧ c = 3) :
  ∃ (a' b' c' : ℝ), (a' * P.1 + b' * P.2 + c' = 0) ∧ (a = b' ∧ b = -a') ∧ (a' = 2 ∧ b' = 1 ∧ c' = -1) := 
by
  use 2, 1, -1
  sorry

end line_through_P_perpendicular_l561_56169


namespace total_canoes_built_l561_56135

-- Definition of the conditions as suggested by the problem
def num_canoes_in_february : Nat := 5
def growth_rate : Nat := 3
def number_of_months : Nat := 5

-- Final statement to prove
theorem total_canoes_built : (num_canoes_in_february * (growth_rate^number_of_months - 1)) / (growth_rate - 1) = 605 := 
by sorry

end total_canoes_built_l561_56135


namespace square_ratio_l561_56144

def area (side_length : ℝ) : ℝ := side_length^2

theorem square_ratio (x : ℝ) (x_pos : 0 < x) :
  let A := area x
  let B := area (3*x)
  let C := area (2*x)
  A / (B + C) = 1 / 13 :=
by
  sorry

end square_ratio_l561_56144


namespace mixture_replacement_l561_56104

theorem mixture_replacement
  (A B : ℕ)
  (hA : A = 48)
  (h_ratio1 : A / B = 4)
  (x : ℕ)
  (h_ratio2 : A / (B + x) = 2 / 3) :
  x = 60 :=
by
  sorry

end mixture_replacement_l561_56104


namespace tim_surprises_combinations_l561_56120

theorem tim_surprises_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 120 :=
by
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  sorry

end tim_surprises_combinations_l561_56120


namespace negation_of_existence_is_universal_l561_56162

theorem negation_of_existence_is_universal (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
sorry

end negation_of_existence_is_universal_l561_56162


namespace projection_of_a_on_b_l561_56191

theorem projection_of_a_on_b (a b : ℝ) (θ : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 1)
  (hθ : θ = 60) : 
  (|a| * Real.cos (θ * Real.pi / 180)) = 1 := 
sorry

end projection_of_a_on_b_l561_56191


namespace three_digit_log3_eq_whole_and_log3_log9_eq_whole_l561_56159

noncomputable def logBase (b : ℝ) (x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem three_digit_log3_eq_whole_and_log3_log9_eq_whole (n : ℕ) (hn : 100 ≤ n ∧ n ≤ 999) (hlog3 : ∃ x : ℤ, logBase 3 n = x) (hlog3log9 : ∃ k : ℤ, logBase 3 n + logBase 9 n = k) :
  n = 729 := sorry

end three_digit_log3_eq_whole_and_log3_log9_eq_whole_l561_56159


namespace compute_a_d_sum_l561_56131

variables {a1 a2 a3 d1 d2 d3 : ℝ}

theorem compute_a_d_sum
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a_d_sum_l561_56131


namespace solution_set_of_inequality_cauchy_schwarz_application_l561_56126

theorem solution_set_of_inequality (c : ℝ) (h1 : c > 0) (h2 : ∀ x : ℝ, x + |x - 2 * c| ≥ 2) : 
  c ≥ 1 :=
by
  sorry

theorem cauchy_schwarz_application (m p q r : ℝ) (h1 : m ≥ 1) (h2 : 0 < p ∧ 0 < q ∧ 0 < r) (h3 : p + q + r = 3 * m) : 
  p^2 + q^2 + r^2 ≥ 3 :=
by
  sorry

end solution_set_of_inequality_cauchy_schwarz_application_l561_56126


namespace price_per_piece_l561_56180

variable (y : ℝ)

theorem price_per_piece (h : (20 + y - 12) * (240 - 40 * y) = 1980) :
  20 + y = 21 ∨ 20 + y = 23 :=
sorry

end price_per_piece_l561_56180


namespace area_of_CDE_in_isosceles_triangle_l561_56106

noncomputable def isosceles_triangle_area (b : ℝ) (s : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * b * s

noncomputable def cot (α : ℝ) : ℝ := 1 / Real.tan α

noncomputable def isosceles_triangle_vertex_angle (b : ℝ) (area : ℝ) (θ : ℝ) : Prop :=
  area = (b^2 / 4) * cot (θ / 2)

theorem area_of_CDE_in_isosceles_triangle (b θ area : ℝ) (hb : b = 3 * (2 * b / 3)) (hθ : θ = 100) (ha : area = 30) :
  ∃ CDE_area, CDE_area = area / 9 ∧ CDE_area = 10 / 3 :=
by
  sorry

end area_of_CDE_in_isosceles_triangle_l561_56106


namespace worker_idle_days_l561_56107

variable (x y : ℤ)

theorem worker_idle_days :
  (30 * x - 5 * y = 500) ∧ (x + y = 60) → y = 38 :=
by
  intros h
  have h1 : 30 * x - 5 * y = 500 := h.left
  have h2 : x + y = 60 := h.right
  sorry

end worker_idle_days_l561_56107


namespace jake_work_hours_l561_56163

-- Definitions for the conditions
def initial_debt : ℝ := 100
def amount_paid : ℝ := 40
def work_rate : ℝ := 15

-- The main theorem stating the number of hours Jake needs to work
theorem jake_work_hours : ∃ h : ℝ, initial_debt - amount_paid = h * work_rate ∧ h = 4 :=
by 
  -- sorry placeholder indicating the proof is not required
  sorry

end jake_work_hours_l561_56163


namespace max_shortest_part_duration_l561_56182

theorem max_shortest_part_duration (film_duration : ℕ) (part1 part2 part3 part4 : ℕ)
  (h_total : part1 + part2 + part3 + part4 = 192)
  (h_diff1 : part2 ≥ part1 + 6)
  (h_diff2 : part3 ≥ part2 + 6)
  (h_diff3 : part4 ≥ part3 + 6) :
  part1 ≤ 39 := 
sorry

end max_shortest_part_duration_l561_56182


namespace engineers_meeting_probability_l561_56170

theorem engineers_meeting_probability :
  ∀ (x y z : ℝ), 
    (0 ≤ x ∧ x ≤ 2) → 
    (0 ≤ y ∧ y ≤ 2) → 
    (0 ≤ z ∧ z ≤ 2) → 
    (abs (x - y) ≤ 0.5) → 
    (abs (y - z) ≤ 0.5) → 
    (abs (z - x) ≤ 0.5) → 
    Π (volume_region : ℝ) (total_volume : ℝ),
    (volume_region = 1.5 * 1.5 * 1.5) → 
    (total_volume = 2 * 2 * 2) → 
    (volume_region / total_volume = 0.421875) :=
by
  intros x y z hx hy hz hxy hyz hzx volume_region total_volume hr ht
  sorry

end engineers_meeting_probability_l561_56170


namespace Polyas_probability_relation_l561_56145

variable (Z : ℕ → ℤ → ℝ)

theorem Polyas_probability_relation (n : ℕ) (k : ℤ) :
  Z n k = (1/2) * (Z (n-1) (k-1) + Z (n-1) (k+1)) :=
by
  sorry

end Polyas_probability_relation_l561_56145


namespace ratio_of_areas_l561_56134

theorem ratio_of_areas (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : r₁ = (Real.sqrt 2) / 4)
  (h₂ : A₁ = π * r₁^2) (h₃ : r₂ = (Real.sqrt 2) * r₁) (h₄ : A₂ = π * r₂^2) :
  A₂ / A₁ = 2 :=
by
  sorry

end ratio_of_areas_l561_56134


namespace thirty_divides_p_squared_minus_one_iff_p_eq_five_l561_56123

theorem thirty_divides_p_squared_minus_one_iff_p_eq_five (p : ℕ) (hp : Nat.Prime p) (h_ge : p ≥ 5) : 30 ∣ (p^2 - 1) ↔ p = 5 :=
by
  sorry

end thirty_divides_p_squared_minus_one_iff_p_eq_five_l561_56123


namespace pinedale_mall_distance_l561_56168

theorem pinedale_mall_distance 
  (speed : ℝ) (time_between_stops : ℝ) (num_stops : ℕ) (distance : ℝ) 
  (h_speed : speed = 60) 
  (h_time_between_stops : time_between_stops = 5 / 60) 
  (h_num_stops : ↑num_stops = 5) :
  distance = 25 :=
by
  sorry

end pinedale_mall_distance_l561_56168


namespace correct_relationship_in_triangle_l561_56175

theorem correct_relationship_in_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A + B) = Real.sin C :=
sorry

end correct_relationship_in_triangle_l561_56175


namespace tom_read_in_five_months_l561_56112

def books_in_may : ℕ := 2
def books_in_june : ℕ := 6
def books_in_july : ℕ := 12
def books_in_august : ℕ := 20
def books_in_september : ℕ := 30

theorem tom_read_in_five_months : 
  books_in_may + books_in_june + books_in_july + books_in_august + books_in_september = 70 := by
  sorry

end tom_read_in_five_months_l561_56112


namespace block_fraction_visible_above_water_l561_56173

-- Defining constants
def weight_of_block : ℝ := 30 -- N
def buoyant_force_submerged : ℝ := 50 -- N

-- Defining the proof problem
theorem block_fraction_visible_above_water (W Fb : ℝ) (hW : W = weight_of_block) (hFb : Fb = buoyant_force_submerged) :
  (1 - W / Fb) = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end block_fraction_visible_above_water_l561_56173


namespace total_animals_correct_l561_56117

-- Define the number of aquariums and the number of animals per aquarium.
def num_aquariums : ℕ := 26
def animals_per_aquarium : ℕ := 2

-- Define the total number of saltwater animals.
def total_animals : ℕ := num_aquariums * animals_per_aquarium

-- The statement we want to prove.
theorem total_animals_correct : total_animals = 52 := by
  -- Proof is omitted.
  sorry

end total_animals_correct_l561_56117


namespace integer_solutions_count_l561_56125

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l561_56125


namespace inequality_solution_set_l561_56146

open Set

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 5 * a) * (x + a) > 0} = {x | x < 5 * a ∨ x > -a} :=
sorry

end inequality_solution_set_l561_56146


namespace inequality_x_y_z_l561_56164

open Real

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
    (x ^ 3) / ((1 + y) * (1 + z)) + (y ^ 3) / ((1 + z) * (1 + x)) + (z ^ 3) / ((1 + x) * (1 + y)) ≥ 3 / 4 :=
by
  sorry

end inequality_x_y_z_l561_56164


namespace geometric_sequence_sum_l561_56172

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ}

theorem geometric_sequence_sum (h1 : is_geometric_sequence a) (h2 : a 1 * a 2 = 8 * a 0)
  (h3 : (a 3 + 2 * a 4) / 2 = 20) :
  (a 0 * (2^5 - 1)) = 31 :=
by
  sorry

end geometric_sequence_sum_l561_56172


namespace quadratic_inequality_l561_56124

theorem quadratic_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end quadratic_inequality_l561_56124


namespace lily_received_books_l561_56114

def mike_books : ℕ := 45
def corey_books : ℕ := 2 * mike_books
def mike_gave_lily : ℕ := 10
def corey_gave_lily : ℕ := mike_gave_lily + 15
def lily_books_received : ℕ := mike_gave_lily + corey_gave_lily

theorem lily_received_books : lily_books_received = 35 := by
  sorry

end lily_received_books_l561_56114


namespace sale_in_first_month_l561_56143

theorem sale_in_first_month 
  (sale_2 : ℝ) (sale_3 : ℝ) (sale_4 : ℝ) (sale_5 : ℝ) (sale_6 : ℝ) (avg_sale : ℝ)
  (h_sale_2 : sale_2 = 5366) (h_sale_3 : sale_3 = 5808) 
  (h_sale_4 : sale_4 = 5399) (h_sale_5 : sale_5 = 6124) 
  (h_sale_6 : sale_6 = 4579) (h_avg_sale : avg_sale = 5400) :
  ∃ (sale_1 : ℝ), sale_1 = 5124 :=
by
  let total_sales := avg_sale * 6
  let known_sales := sale_2 + sale_3 + sale_4 + sale_5 + sale_6
  have h_total_sales : total_sales = 32400 := by sorry
  have h_known_sales : known_sales = 27276 := by sorry
  let sale_1 := total_sales - known_sales
  use sale_1
  have h_sale_1 : sale_1 = 5124 := by sorry
  exact h_sale_1

end sale_in_first_month_l561_56143


namespace largest_consecutive_even_integer_l561_56158

theorem largest_consecutive_even_integer (n : ℕ) (h : 5 * n - 20 = 2 * 15 * 16 / 2) : n = 52 :=
sorry

end largest_consecutive_even_integer_l561_56158


namespace length_PQ_calc_l561_56152

noncomputable def length_PQ 
  (F : ℝ × ℝ) 
  (P Q : ℝ × ℝ) 
  (hF : F = (1, 0)) 
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1) 
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1) 
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1) 
  (hx1x2 : P.1 + Q.1 = 9) : ℝ :=
|P.1 - Q.1|

theorem length_PQ_calc : ∀ F P Q
  (hF : F = (1, 0))
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1)
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1)
  (hx1x2 : P.1 + Q.1 = 9),
  length_PQ F P Q hF hP_on_parabola hQ_on_parabola hLine_through_focus hx1x2 = 11 := 
by
  sorry

end length_PQ_calc_l561_56152


namespace good_students_l561_56127

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l561_56127


namespace fg_difference_l561_56156

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference : f (g 3) - g (f 3) = 59 :=
by
  sorry

end fg_difference_l561_56156


namespace homework_problems_l561_56155

theorem homework_problems (p t : ℕ) (h1 : p >= 10) (h2 : pt = (2 * p + 2) * (t + 1)) : p * t = 60 :=
by
  sorry

end homework_problems_l561_56155


namespace largest_number_4597_l561_56153

def swap_adjacent_digits (n : ℕ) : ℕ :=
  sorry

def max_number_after_two_swaps_subtract_100 (n : ℕ) : ℕ :=
  -- logic to perform up to two adjacent digit swaps and subtract 100
  sorry

theorem largest_number_4597 : max_number_after_two_swaps_subtract_100 4597 = 4659 :=
  sorry

end largest_number_4597_l561_56153


namespace general_form_of_line_l561_56190

theorem general_form_of_line (x y : ℝ) 
  (passes_through_A : ∃ y, 2 = y)          -- Condition 1: passes through A(-2, 2)
  (same_y_intercept : ∃ y, 6 = y)          -- Condition 2: same y-intercept as y = x + 6
  : 2 * x - y + 6 = 0 := 
sorry

end general_form_of_line_l561_56190


namespace distance_relationship_l561_56122

noncomputable def plane_parallel (α β : Type) : Prop := sorry
noncomputable def line_in_plane (m : Type) (α : Type) : Prop := sorry
noncomputable def point_on_line (A : Type) (m : Type) : Prop := sorry
noncomputable def distance (A B : Type) : ℝ := sorry
noncomputable def distance_point_to_line (A : Type) (n : Type) : ℝ := sorry
noncomputable def distance_between_lines (m n : Type) : ℝ := sorry

variables (α β m n A B : Type)
variables (a b c : ℝ)

axiom plane_parallel_condition : plane_parallel α β
axiom line_m_in_alpha : line_in_plane m α
axiom line_n_in_beta : line_in_plane n β
axiom point_A_on_m : point_on_line A m
axiom point_B_on_n : point_on_line B n
axiom distance_a : a = distance A B
axiom distance_b : b = distance_point_to_line A n
axiom distance_c : c = distance_between_lines m n

theorem distance_relationship : c ≤ b ∧ b ≤ a := by
  sorry

end distance_relationship_l561_56122
