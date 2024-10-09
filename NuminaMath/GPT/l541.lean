import Mathlib

namespace joes_mean_score_is_88_83_l541_54166

def joesQuizScores : List ℕ := [88, 92, 95, 81, 90, 87]

noncomputable def mean (lst : List ℕ) : ℝ := (lst.sum : ℝ) / lst.length

theorem joes_mean_score_is_88_83 :
  mean joesQuizScores = 88.83 := 
sorry

end joes_mean_score_is_88_83_l541_54166


namespace ice_cream_children_count_ice_cream_girls_count_l541_54181

-- Proof Problem for part (a)
theorem ice_cream_children_count (n : ℕ) (h : 3 * n = 24) : n = 8 := sorry

-- Proof Problem for part (b)
theorem ice_cream_girls_count (x y : ℕ) (h : x + y = 8) 
  (hx_even : x % 2 = 0) (hy_even : y % 2 = 0) (hx_pos : x > 0) (hxy : x < y) : y = 6 := sorry

end ice_cream_children_count_ice_cream_girls_count_l541_54181


namespace set_intersection_l541_54103

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}
def B_complement : Set ℝ := {x | x ≥ 2}

theorem set_intersection :
  A ∩ B_complement = {x | 2 ≤ x ∧ x < 5} :=
by 
  sorry

end set_intersection_l541_54103


namespace mom_age_when_jayson_born_l541_54155

theorem mom_age_when_jayson_born (jayson_age dad_age mom_age : ℕ) 
  (h1 : jayson_age = 10) 
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 :=
by
  sorry

end mom_age_when_jayson_born_l541_54155


namespace min_eccentricity_sum_l541_54157

def circle_O1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16
def circle_O2 (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 0 < r ∧ r < 2

def moving_circle_tangent (e1 e2 : ℝ) (r : ℝ) : Prop :=
  e1 = 2 / (4 - r) ∧ e2 = 2 / (4 + r)

theorem min_eccentricity_sum : ∃ (e1 e2 : ℝ) (r : ℝ), 
  circle_O1 x y ∧ circle_O2 x y r ∧ moving_circle_tangent e1 e2 r ∧
    e1 > e2 ∧ (e1 + 2 * e2) = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_eccentricity_sum_l541_54157


namespace path_area_and_cost_correct_l541_54130

-- Define the given conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 7

-- Calculate new dimensions including the path
def length_including_path : ℝ := length_field + 2 * path_width
def width_including_path : ℝ := width_field + 2 * path_width

-- Calculate areas
def area_entire_field : ℝ := length_including_path * width_including_path
def area_grass_field : ℝ := length_field * width_field
def area_path : ℝ := area_entire_field - area_grass_field

-- Calculate cost
def cost_of_path : ℝ := area_path * cost_per_sq_meter

theorem path_area_and_cost_correct : 
  area_path = 675 ∧ cost_of_path = 4725 :=
by
  sorry

end path_area_and_cost_correct_l541_54130


namespace paint_gallons_l541_54111

theorem paint_gallons (W B : ℕ) (h1 : 5 * B = 8 * W) (h2 : W + B = 6689) : B = 4116 :=
by
  sorry

end paint_gallons_l541_54111


namespace solve_xy_l541_54153

theorem solve_xy : ∃ (x y : ℝ), x = 1 / 3 ∧ y = 2 / 3 ∧ x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 :=
by
  use 1 / 3, 2 / 3
  sorry

end solve_xy_l541_54153


namespace divide_into_parts_l541_54169

theorem divide_into_parts (x y : ℚ) (h_sum : x + y = 10) (h_diff : y - x = 5) : 
  x = 5 / 2 ∧ y = 15 / 2 := 
sorry

end divide_into_parts_l541_54169


namespace revenue_change_l541_54161

theorem revenue_change (x : ℝ) 
  (increase_in_1996 : ∀ R : ℝ, R * (1 + x/100) > R) 
  (decrease_in_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) < R * (1 + x/100)) 
  (decrease_from_1995_to_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) = R * 0.96): 
  x = 20 :=
by
  sorry

end revenue_change_l541_54161


namespace entrance_exit_ways_equal_49_l541_54197

-- Define the number of gates on each side
def south_gates : ℕ := 4
def north_gates : ℕ := 3

-- Define the total number of gates
def total_gates : ℕ := south_gates + north_gates

-- State the theorem and provide the expected proof structure
theorem entrance_exit_ways_equal_49 : (total_gates * total_gates) = 49 := 
by {
  sorry
}

end entrance_exit_ways_equal_49_l541_54197


namespace cost_of_bananas_l541_54187

theorem cost_of_bananas (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + B = 5) : B = 3 :=
by
  sorry

end cost_of_bananas_l541_54187


namespace train_cable_car_distance_and_speeds_l541_54189
-- Import necessary libraries

-- Defining the variables and conditions
variables (s v1 v2 : ℝ)
variables (half_hour_sym_dist additional_distance quarter_hour_meet : ℝ)

-- Defining the conditions
def conditions :=
  (half_hour_sym_dist = v1 * (1 / 2) + v2 * (1 / 2)) ∧
  (additional_distance = 2 / v2) ∧
  (quarter_hour_meet = 1 / 4) ∧
  (v1 + v2 = 2 * s) ∧
  (v2 * (additional_distance + half_hour_sym_dist) = (v1 * (additional_distance + half_hour_sym_dist) - s)) ∧
  ((v1 + v2) * (half_hour_sym_dist + additional_distance + quarter_hour_meet) = 2 * s)

-- Proving the statement
theorem train_cable_car_distance_and_speeds
  (h : conditions s v1 v2 half_hour_sym_dist additional_distance quarter_hour_meet) :
  s = 24 ∧ v1 = 40 ∧ v2 = 8 := sorry

end train_cable_car_distance_and_speeds_l541_54189


namespace store_price_reduction_l541_54163

theorem store_price_reduction 
    (initial_price : ℝ) (initial_sales : ℕ) (price_reduction : ℝ)
    (sales_increase_factor : ℝ) (target_profit : ℝ)
    (x : ℝ) : (initial_price, initial_price - price_reduction, x) = (80, 50, 12) →
    sales_increase_factor = 20 →
    target_profit = 7920 →
    (30 - x) * (200 + sales_increase_factor * x / 2) = 7920 →
    x = 12 ∧ (initial_price - x) = 68 :=
by 
    intros h₁ h₂ h₃ h₄
    sorry

end store_price_reduction_l541_54163


namespace perfect_square_n_l541_54137

theorem perfect_square_n (n : ℤ) (h1 : n > 0) (h2 : ∃ k : ℤ, n^2 + 19 * n + 48 = k^2) : n = 33 :=
sorry

end perfect_square_n_l541_54137


namespace one_of_a_b_c_is_zero_l541_54135

theorem one_of_a_b_c_is_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 :=
by
  sorry

end one_of_a_b_c_is_zero_l541_54135


namespace quad_eq_double_root_m_value_l541_54108

theorem quad_eq_double_root_m_value (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6 * x + m = 0) → m = 9 := 
by 
  sorry

end quad_eq_double_root_m_value_l541_54108


namespace second_projection_at_given_distance_l541_54146

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Line :=
  (point : Point)
  (direction : Point) -- Assume direction is given as a vector

def is_parallel (line1 line2 : Line) : Prop :=
  -- Function to check if two lines are parallel
  sorry

def distance (point1 point2 : Point) : ℝ := 
  -- Function to compute the distance between two points
  sorry

def first_projection_exists (M : Point) (a : Line) : Prop :=
  -- Check the projection outside the line a
  sorry

noncomputable def second_projection
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  Point :=
  sorry

theorem second_projection_at_given_distance
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  distance (second_projection M a d h_parallel h_projection) a.point = d :=
  sorry

end second_projection_at_given_distance_l541_54146


namespace band_and_chorus_but_not_orchestra_l541_54127

theorem band_and_chorus_but_not_orchestra (B C O : Finset ℕ)
  (hB : B.card = 100) 
  (hC : C.card = 120) 
  (hO : O.card = 60)
  (hUnion : (B ∪ C ∪ O).card = 200)
  (hIntersection : (B ∩ C ∩ O).card = 10) : 
  ((B ∩ C).card - (B ∩ C ∩ O).card = 30) :=
by sorry

end band_and_chorus_but_not_orchestra_l541_54127


namespace sam_original_seashells_count_l541_54122

-- Definitions representing the conditions
def seashells_given_to_joan : ℕ := 18
def seashells_sam_has_now : ℕ := 17

-- The question and the answer translated to a proof problem
theorem sam_original_seashells_count :
  seashells_given_to_joan + seashells_sam_has_now = 35 :=
by
  sorry

end sam_original_seashells_count_l541_54122


namespace cube_side_length_l541_54175

theorem cube_side_length (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
sorry

end cube_side_length_l541_54175


namespace total_rats_l541_54151

theorem total_rats (Elodie_rats Hunter_rats Kenia_rats : ℕ) 
  (h1 : Elodie_rats = 30) 
  (h2 : Elodie_rats = Hunter_rats + 10)
  (h3 : Kenia_rats = 3 * (Elodie_rats + Hunter_rats)) :
  Elodie_rats + Hunter_rats + Kenia_rats = 200 :=
by
  sorry

end total_rats_l541_54151


namespace cherry_sodas_correct_l541_54178

/-
A cooler is filled with 24 cans of cherry soda and orange pop. 
There are twice as many cans of orange pop as there are of cherry soda. 
Prove that the number of cherry sodas is 8.
-/
def num_cherry_sodas (C O : ℕ) : Prop :=
  O = 2 * C ∧ C + O = 24 → C = 8

theorem cherry_sodas_correct (C O : ℕ) (h : O = 2 * C ∧ C + O = 24) : C = 8 :=
by
  sorry

end cherry_sodas_correct_l541_54178


namespace math_problem_l541_54196

variables {A B : Type} [Fintype A] [Fintype B]
          (p1 p2 : ℝ) (h1 : 1/2 < p1) (h2 : p1 < p2) (h3 : p2 < 1)
          (nA : ℕ) (hA : nA = 3) (nB : ℕ) (hB : nB = 3)

noncomputable def E_X : ℝ := nA * p1
noncomputable def E_Y : ℝ := nB * p2

noncomputable def D_X : ℝ := nA * p1 * (1 - p1)
noncomputable def D_Y : ℝ := nB * p2 * (1 - p2)

theorem math_problem :
  E_X p1 nA = 3 * p1 →
  E_Y p2 nB = 3 * p2 →
  D_X p1 nA = 3 * p1 * (1 - p1) →
  D_Y p2 nB = 3 * p2 * (1 - p2) →
  E_X p1 nA < E_Y p2 nB ∧ D_X p1 nA > D_Y p2 nB :=
by
  sorry

end math_problem_l541_54196


namespace inequality_always_negative_l541_54192

theorem inequality_always_negative (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (-3 < k ∧ k ≤ 0) :=
by
  -- Proof omitted
  sorry

end inequality_always_negative_l541_54192


namespace coordinates_of_P_l541_54133

-- Define the conditions and the question as a Lean theorem
theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) (h1 : P = (m + 3, m + 1)) (h2 : P.2 = 0) :
  P = (2, 0) := 
sorry

end coordinates_of_P_l541_54133


namespace ed_pets_count_l541_54158

theorem ed_pets_count : 
  let dogs := 2 
  let cats := 3 
  let fish := 2 * (cats + dogs) 
  let birds := dogs * cats 
  dogs + cats + fish + birds = 21 := 
by
  sorry

end ed_pets_count_l541_54158


namespace triangular_weight_l541_54185

theorem triangular_weight (c t : ℝ) (h1 : c + t = 3 * c) (h2 : 4 * c + t = t + c + 90) : t = 60 := 
by sorry

end triangular_weight_l541_54185


namespace geometric_sequence_tenth_term_l541_54112

theorem geometric_sequence_tenth_term :
  let a := 5
  let r := 3 / 2
  let a_n (n : ℕ) := a * r ^ (n - 1)
  a_n 10 = 98415 / 512 :=
by
  sorry

end geometric_sequence_tenth_term_l541_54112


namespace square_side_4_FP_length_l541_54164

theorem square_side_4_FP_length (EF GH EP FP GP : ℝ) :
  EF = 4 ∧ GH = 4 ∧ EP = 4 ∧ GP = 4 ∧
  (1 / 2) * EP * 2 = 4 → FP = 2 * Real.sqrt 5 :=
by
  intro h
  sorry

end square_side_4_FP_length_l541_54164


namespace triple_composition_l541_54124

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition :
  g (g (g 3)) = 107 :=
by
  sorry

end triple_composition_l541_54124


namespace solution_l541_54129

-- Define M and N according to the given conditions
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x ≥ 1}

-- Define the complement of M in Real numbers
def complementM : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the union of the complement of M and N
def problem_statement : Set ℝ := complementM ∪ N

-- State the theorem
theorem solution :
  problem_statement = { x | x ≥ 0 } :=
by
  sorry

end solution_l541_54129


namespace tenth_term_is_19_over_4_l541_54104

def nth_term_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

theorem tenth_term_is_19_over_4 :
  nth_term_arithmetic_sequence (1/4) (1/2) 10 = 19/4 :=
by
  sorry

end tenth_term_is_19_over_4_l541_54104


namespace water_added_l541_54170

theorem water_added (initial_fullness : ℚ) (final_fullness : ℚ) (capacity : ℚ)
  (h1 : initial_fullness = 0.40) (h2 : final_fullness = 3 / 4) (h3 : capacity = 80) :
  (final_fullness * capacity - initial_fullness * capacity) = 28 := by
  sorry

end water_added_l541_54170


namespace largest_fraction_among_fractions_l541_54121

theorem largest_fraction_among_fractions :
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  (A < E) ∧ (B < E) ∧ (C < E) ∧ (D < E) :=
by
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  sorry

end largest_fraction_among_fractions_l541_54121


namespace max_x5_l541_54154

theorem max_x5 (x1 x2 x3 x4 x5 : ℕ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) : x5 ≤ 5 :=
  sorry

end max_x5_l541_54154


namespace points_on_x_axis_circles_intersect_l541_54193

theorem points_on_x_axis_circles_intersect (a b : ℤ)
  (h1 : 3 * a - b = 9)
  (h2 : 2 * a + 3 * b = -5) : (a : ℝ)^b = 1/8 :=
by
  sorry

end points_on_x_axis_circles_intersect_l541_54193


namespace sum_mnp_is_405_l541_54176

theorem sum_mnp_is_405 :
  let C1_radius := 4
  let C2_radius := 10
  let C3_radius := C1_radius + C2_radius
  let chord_length := (8 * Real.sqrt 390) / 7
  ∃ m n p : ℕ,
    m * Real.sqrt n / p = chord_length ∧
    m.gcd p = 1 ∧
    (∀ k : ℕ, k^2 ∣ n → k = 1) ∧
    m + n + p = 405 :=
by
  sorry

end sum_mnp_is_405_l541_54176


namespace sum_of_primes_less_than_20_l541_54182

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l541_54182


namespace sequences_meet_at_2017_l541_54140

-- Define the sequences for Paul and Penny
def paul_sequence (n : ℕ) : ℕ := 3 * n - 2
def penny_sequence (m : ℕ) : ℕ := 2022 - 5 * m

-- Statement to be proven
theorem sequences_meet_at_2017 : ∃ n m : ℕ, paul_sequence n = 2017 ∧ penny_sequence m = 2017 := by
  sorry

end sequences_meet_at_2017_l541_54140


namespace speed_of_man_in_still_water_l541_54147

theorem speed_of_man_in_still_water 
  (V_m V_s : ℝ)
  (h1 : 6 = V_m + V_s)
  (h2 : 4 = V_m - V_s) : 
  V_m = 5 := 
by 
  sorry

end speed_of_man_in_still_water_l541_54147


namespace find_sample_size_l541_54165

theorem find_sample_size
  (teachers : ℕ := 200)
  (male_students : ℕ := 1200)
  (female_students : ℕ := 1000)
  (sampled_females : ℕ := 80)
  (total_people := teachers + male_students + female_students)
  (ratio : sampled_females / female_students = n / total_people)
  : n = 192 := 
by
  sorry

end find_sample_size_l541_54165


namespace stormi_needs_more_money_l541_54172

theorem stormi_needs_more_money
  (cars_washed : ℕ) (price_per_car : ℕ)
  (lawns_mowed : ℕ) (price_per_lawn : ℕ)
  (bike_cost : ℕ)
  (h1 : cars_washed = 3)
  (h2 : price_per_car = 10)
  (h3 : lawns_mowed = 2)
  (h4 : price_per_lawn = 13)
  (h5 : bike_cost = 80) : 
  bike_cost - (cars_washed * price_per_car + lawns_mowed * price_per_lawn) = 24 := by
  sorry

end stormi_needs_more_money_l541_54172


namespace infinite_power_tower_equation_l541_54195

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  x ^ x ^ x ^ x ^ x -- continues infinitely

theorem infinite_power_tower_equation (x : ℝ) (h_pos : 0 < x) (h_eq : infinite_power_tower x = 2) : x = Real.sqrt 2 :=
  sorry

end infinite_power_tower_equation_l541_54195


namespace deepak_present_age_l541_54179

-- Let R be Rahul's current age and D be Deepak's current age
variables (R D : ℕ)

-- Given conditions
def ratio_condition : Prop := (4 : ℚ) / 3 = (R : ℚ) / D
def rahul_future_age_condition : Prop := R + 6 = 50

-- Prove Deepak's present age D is 33 years
theorem deepak_present_age : ratio_condition R D ∧ rahul_future_age_condition R → D = 33 := 
sorry

end deepak_present_age_l541_54179


namespace power_function_constant_l541_54156

theorem power_function_constant (k α : ℝ)
  (h : (1 / 2 : ℝ) ^ α * k = (Real.sqrt 2 / 2)) : k + α = 3 / 2 := by
  sorry

end power_function_constant_l541_54156


namespace find_s_t_l541_54171

theorem find_s_t 
  (FG GH EH : ℝ)
  (angleE angleF : ℝ)
  (h1 : FG = 10)
  (h2 : GH = 15)
  (h3 : EH = 12)
  (h4 : angleE = 45)
  (h5 : angleF = 45)
  (s t : ℕ)
  (h6 : 12 + 7.5 * Real.sqrt 2 = s + Real.sqrt t) :
  s + t = 5637 :=
sorry

end find_s_t_l541_54171


namespace find_k_l541_54194

open Real

noncomputable def k_value (θ : ℝ) : ℝ :=
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 - 2 * (tan θ ^ 2 + 1 / tan θ ^ 2) 

theorem find_k (θ : ℝ) (h : θ ≠ 0 ∧ θ ≠ π / 2 ∧ θ ≠ π) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k_value θ → k_value θ = 6 :=
by
  sorry

end find_k_l541_54194


namespace uniformColorGridPossible_l541_54125

noncomputable def canPaintUniformColor (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) : Prop :=
  ∀ (row : Fin n), ∃ (c : Fin (n - 1)), ∀ (col : Fin n), G row col = c

theorem uniformColorGridPossible (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) :
  (∀ r : Fin n, ∃ c₁ c₂ : Fin n, c₁ ≠ c₂ ∧ G r c₁ = G r c₂) ∧
  (∀ c : Fin n, ∃ r₁ r₂ : Fin n, r₁ ≠ r₂ ∧ G r₁ c = G r₂ c) →
  ∃ c : Fin (n - 1), ∀ (row col : Fin n), G row col = c := by
  sorry

end uniformColorGridPossible_l541_54125


namespace allocation_ways_l541_54134

theorem allocation_ways (programs : Finset ℕ) (grades : Finset ℕ) (h_programs : programs.card = 6) (h_grades : grades.card = 4) : 
  ∃ ways : ℕ, ways = 1080 := 
by 
  sorry

end allocation_ways_l541_54134


namespace fraction_of_track_in_forest_l541_54117

theorem fraction_of_track_in_forest (n : ℕ) (l : ℝ) (A B C : ℝ) :
  (∃ x, x = 2*l/3 ∨ x = l/3) → (∃ f, 0 < f ∧ f ≤ 1 ∧ (f = 2/3 ∨ f = 1/3)) :=
by
  -- sorry, the proof will go here
  sorry

end fraction_of_track_in_forest_l541_54117


namespace neg_eight_degrees_celsius_meaning_l541_54126

-- Define the temperature in degrees Celsius
def temp_in_degrees_celsius (t : Int) : String :=
  if t >= 0 then toString t ++ "°C above zero"
  else toString (abs t) ++ "°C below zero"

-- Define the proof statement
theorem neg_eight_degrees_celsius_meaning :
  temp_in_degrees_celsius (-8) = "8°C below zero" :=
sorry

end neg_eight_degrees_celsius_meaning_l541_54126


namespace longest_side_range_l541_54199

-- Definitions and conditions
def is_triangle (x y z : ℝ) : Prop := 
  x + y > z ∧ x + z > y ∧ y + z > x

-- Problem statement
theorem longest_side_range (l x y z : ℝ) 
  (h_triangle: is_triangle x y z) 
  (h_perimeter: x + y + z = l / 2) 
  (h_longest: x ≥ y ∧ x ≥ z) : 
  l / 6 ≤ x ∧ x < l / 4 :=
by
  sorry

end longest_side_range_l541_54199


namespace bricks_per_course_l541_54188

theorem bricks_per_course : 
  ∃ B : ℕ, (let initial_courses := 3
            let additional_courses := 2
            let total_courses := initial_courses + additional_courses
            let last_course_half_removed := B / 2
            let total_bricks := B * total_courses - last_course_half_removed
            total_bricks = 1800) ↔ B = 400 :=
by {sorry}

end bricks_per_course_l541_54188


namespace frogs_climbed_onto_logs_l541_54198

-- Definitions of the conditions
def f_lily : ℕ := 5
def f_rock : ℕ := 24
def f_total : ℕ := 32

-- The final statement we want to prove
theorem frogs_climbed_onto_logs : f_total - (f_lily + f_rock) = 3 :=
by
  sorry

end frogs_climbed_onto_logs_l541_54198


namespace cone_radius_l541_54139

open Real

theorem cone_radius
  (l : ℝ) (L : ℝ) (h_l : l = 5) (h_L : L = 15 * π) :
  ∃ r : ℝ, L = π * r * l ∧ r = 3 :=
by
  sorry

end cone_radius_l541_54139


namespace distinct_right_angles_l541_54149

theorem distinct_right_angles (n : ℕ) (h : n > 0) : 
  ∃ (a b c d : ℕ), (a + b + c + d ≥ 4 * (Int.sqrt n)) ∧ (a * c ≥ n) ∧ (b * d ≥ n) :=
by sorry

end distinct_right_angles_l541_54149


namespace min_friend_pairs_l541_54115

-- Define conditions
def n : ℕ := 2000
def invitations_per_person : ℕ := 1000
def total_invitations : ℕ := n * invitations_per_person

-- Mathematical problem statement
theorem min_friend_pairs : (total_invitations / 2) = 1000000 := 
by sorry

end min_friend_pairs_l541_54115


namespace largest_multiple_of_15_less_than_500_l541_54150

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l541_54150


namespace calc_value_l541_54106

theorem calc_value (a b x : ℤ) (h₁ : a = 153) (h₂ : b = 147) (h₃ : x = 900) : x^2 / (a^2 - b^2) = 450 :=
by
  rw [h₁, h₂, h₃]
  -- Proof follows from the calculation in the provided steps
  sorry

end calc_value_l541_54106


namespace transport_cost_l541_54105

-- Define the conditions
def cost_per_kg : ℕ := 15000
def grams_per_kg : ℕ := 1000
def weight_in_grams : ℕ := 500

-- Define the main theorem stating the proof problem
theorem transport_cost
  (c : ℕ := cost_per_kg)
  (gpk : ℕ := grams_per_kg)
  (w : ℕ := weight_in_grams)
  : c * w / gpk = 7500 :=
by
  -- Since we are not required to provide the proof, adding sorry here
  sorry

end transport_cost_l541_54105


namespace perimeter_of_regular_polygon_l541_54145

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l541_54145


namespace value_of_f_at_5_l541_54186

def f (x : ℤ) : ℤ := x^3 - x^2 + x

theorem value_of_f_at_5 : f 5 = 105 := by
  sorry

end value_of_f_at_5_l541_54186


namespace compare_neg_fractions_l541_54128

theorem compare_neg_fractions : - (4 / 3 : ℚ) < - (5 / 4 : ℚ) := 
by sorry

end compare_neg_fractions_l541_54128


namespace min_ab_is_2sqrt6_l541_54110

noncomputable def min_ab (a b : ℝ) : ℝ :=
  if h : (a > 0) ∧ (b > 0) ∧ ((2 / a) + (3 / b) = Real.sqrt (a * b)) then
      2 * Real.sqrt 6
  else
      0 -- or any other value, since this case should not occur in the context

theorem min_ab_is_2sqrt6 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : (2 / a) + (3 / b) = Real.sqrt (a * b)) :
  min_ab a b = 2 * Real.sqrt 6 := 
by
  sorry

end min_ab_is_2sqrt6_l541_54110


namespace correct_number_of_statements_l541_54107

-- Define the conditions as invalidity of the given statements
def statement_1_invalid : Prop := ¬ (true) -- INPUT a,b,c should use commas
def statement_2_invalid : Prop := ¬ (true) -- INPUT x=, 3 correct format
def statement_3_invalid : Prop := ¬ (true) -- 3=B , left side should be a variable name
def statement_4_invalid : Prop := ¬ (true) -- A=B=2, continuous assignment not allowed

-- Combine conditions
def all_statements_invalid : Prop := statement_1_invalid ∧ statement_2_invalid ∧ statement_3_invalid ∧ statement_4_invalid

-- State the theorem to prove
theorem correct_number_of_statements : all_statements_invalid → 0 = 0 := 
by sorry

end correct_number_of_statements_l541_54107


namespace ab_not_divisible_by_5_then_neither_divisible_l541_54173

theorem ab_not_divisible_by_5_then_neither_divisible (a b : ℕ) : ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) → ¬(5 ∣ (a * b)) :=
by
  -- Mathematical statement for proof by contradiction:
  have H1: ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := sorry
  -- Rest of the proof would go here  
  sorry

end ab_not_divisible_by_5_then_neither_divisible_l541_54173


namespace no_primes_in_sequence_l541_54142

def P : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61

theorem no_primes_in_sequence :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 59 → ¬ Nat.Prime (P + n) :=
by
  sorry

end no_primes_in_sequence_l541_54142


namespace largest_x_fraction_l541_54190

theorem largest_x_fraction (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 := by
  sorry

end largest_x_fraction_l541_54190


namespace carla_needs_30_leaves_l541_54159

-- Definitions of the conditions
def items_per_day : Nat := 5
def total_days : Nat := 10
def total_bugs : Nat := 20

-- Maths problem to be proved
theorem carla_needs_30_leaves :
  let total_items := items_per_day * total_days
  let required_leaves := total_items - total_bugs
  required_leaves = 30 :=
by
  sorry

end carla_needs_30_leaves_l541_54159


namespace problem_proof_l541_54143

variable {a1 a2 b1 b2 b3 : ℝ}

theorem problem_proof 
  (h1 : ∃ d, -7 + d = a1 ∧ a1 + d = a2 ∧ a2 + d = -1)
  (h2 : ∃ r, -4 * r = b1 ∧ b1 * r = b2 ∧ b2 * r = b3 ∧ b3 * r = -1)
  (ha : a2 - a1 = 2)
  (hb : b2 = -2) :
  (a2 - a1) / b2 = -1 :=
by
  sorry

end problem_proof_l541_54143


namespace remainder_of_polynomial_division_l541_54102

noncomputable def evaluate_polynomial (x : ℂ) : ℂ :=
  x^100 + x^75 + x^50 + x^25 + 1

noncomputable def divisor_polynomial (x : ℂ) : ℂ :=
  x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_polynomial_division : 
  ∀ β : ℂ, divisor_polynomial β = 0 → evaluate_polynomial β = -1 :=
by
  intros β hβ
  sorry

end remainder_of_polynomial_division_l541_54102


namespace sum_of_cubes_of_consecutive_even_integers_l541_54131

theorem sum_of_cubes_of_consecutive_even_integers 
    (x y z : ℕ) 
    (h1 : x % 2 = 0) 
    (h2 : y % 2 = 0) 
    (h3 : z % 2 = 0) 
    (h4 : y = x + 2) 
    (h5 : z = y + 2) 
    (h6 : x * y * z = 12 * (x + y + z)) : 
  x^3 + y^3 + z^3 = 8568 := 
by
  -- Proof goes here
  sorry

end sum_of_cubes_of_consecutive_even_integers_l541_54131


namespace range_of_y_l541_54162

theorem range_of_y (y : ℝ) (hy : y < 0) (h : ⌈y⌉ * ⌊y⌋ = 132) : -12 < y ∧ y < -11 := 
by 
  sorry

end range_of_y_l541_54162


namespace geometric_sequence_k_squared_l541_54160

theorem geometric_sequence_k_squared (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) (h5 : a 5 * a 8 * a 11 = k) : 
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 := by
  sorry

end geometric_sequence_k_squared_l541_54160


namespace simplify_exponent_expression_l541_54109

theorem simplify_exponent_expression : 2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end simplify_exponent_expression_l541_54109


namespace equilateral_cannot_be_obtuse_l541_54174

-- Additional definitions for clarity and mathematical rigor.
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c ∧ c = a
def is_obtuse (A B C : ℝ) : Prop := 
    (A > 90 ∧ B < 90 ∧ C < 90) ∨ 
    (B > 90 ∧ A < 90 ∧ C < 90) ∨
    (C > 90 ∧ A < 90 ∧ B < 90)

-- Theorem statement
theorem equilateral_cannot_be_obtuse (a b c : ℝ) (A B C : ℝ) :
  is_equilateral a b c → 
  (A + B + C = 180) → 
  (A = B ∧ B = C) → 
  ¬ is_obtuse A B C :=
by { sorry } -- Proof is not necessary as per instruction.

end equilateral_cannot_be_obtuse_l541_54174


namespace fuchsia_to_mauve_l541_54118

def fuchsia_to_mauve_amount (F : ℝ) : Prop :=
  let blue_in_fuchsia := (3 / 8) * F
  let red_in_fuchsia := (5 / 8) * F
  blue_in_fuchsia + 14 = 2 * red_in_fuchsia

theorem fuchsia_to_mauve (F : ℝ) (h : fuchsia_to_mauve_amount F) : F = 16 :=
by
  sorry

end fuchsia_to_mauve_l541_54118


namespace solution_existence_l541_54184

theorem solution_existence (m : ℤ) :
  (∀ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ↔
  (m = -3 ∨ m = 3 → 
    (m = -3 → ∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ∧
    (m = 3 → ¬∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3)) := by
  sorry

end solution_existence_l541_54184


namespace number_of_solutions_l541_54132

theorem number_of_solutions (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 + 4 * n ∧ (∃ (x y : ℤ), x ^ 2 + 2016 * y ^ 2 = 2017 ^ n) :=
by
  sorry

end number_of_solutions_l541_54132


namespace ducks_percentage_non_heron_birds_l541_54120

theorem ducks_percentage_non_heron_birds
  (total_birds : ℕ)
  (geese_percent pelicans_percent herons_percent ducks_percent : ℝ)
  (H_geese : geese_percent = 20 / 100)
  (H_pelicans: pelicans_percent = 40 / 100)
  (H_herons : herons_percent = 15 / 100)
  (H_ducks : ducks_percent = 25 / 100)
  (hnz : total_birds ≠ 0) :
  (ducks_percent / (1 - herons_percent)) * 100 = 30 :=
by
  sorry

end ducks_percentage_non_heron_birds_l541_54120


namespace solve_arithmetic_sequence_l541_54167

theorem solve_arithmetic_sequence (x : ℝ) 
  (term1 term2 term3 : ℝ)
  (h1 : term1 = 3 / 4)
  (h2 : term2 = 2 * x - 3)
  (h3 : term3 = 7 * x) 
  (h_arith : term2 - term1 = term3 - term2) :
  x = -9 / 4 :=
by
  sorry

end solve_arithmetic_sequence_l541_54167


namespace system_of_equations_xy_l541_54101

theorem system_of_equations_xy (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = 5) :
  x - y = 2 := sorry

end system_of_equations_xy_l541_54101


namespace find_number_l541_54177

theorem find_number : ∀ (x : ℝ), (0.15 * 0.30 * 0.50 * x = 99) → (x = 4400) :=
by
  intro x
  intro h
  sorry

end find_number_l541_54177


namespace inequality_proof_l541_54100

noncomputable def a := (1.01: ℝ) ^ (0.5: ℝ)
noncomputable def b := (1.01: ℝ) ^ (0.6: ℝ)
noncomputable def c := (0.6: ℝ) ^ (0.5: ℝ)

theorem inequality_proof : b > a ∧ a > c := 
by
  sorry

end inequality_proof_l541_54100


namespace intersecting_lines_value_l541_54113

theorem intersecting_lines_value (m b : ℚ)
  (h₁ : 10 = m * 7 + 5)
  (h₂ : 10 = 2 * 7 + b) :
  b + m = - (23 : ℚ) / 7 := 
sorry

end intersecting_lines_value_l541_54113


namespace last_digit_2019_digit_number_l541_54141

theorem last_digit_2019_digit_number :
  ∃ n : ℕ → ℕ,  
    (∀ k, 0 ≤ k → k < 2018 → (n k * 10 + n (k + 1)) % 13 = 0) ∧ 
    n 0 = 6 ∧ 
    n 2018 = 2 :=
sorry

end last_digit_2019_digit_number_l541_54141


namespace least_number_of_tiles_l541_54136

-- Definitions for classroom dimensions
def classroom_length : ℕ := 624 -- in cm
def classroom_width : ℕ := 432 -- in cm

-- Definitions for tile dimensions
def rectangular_tile_length : ℕ := 60
def rectangular_tile_width : ℕ := 80
def triangular_tile_base : ℕ := 40
def triangular_tile_height : ℕ := 40

-- Definition for the area calculation
def area (length width : ℕ) : ℕ := length * width
def area_triangular_tile (base height : ℕ) : ℕ := (base * height) / 2

-- Define the area of the classroom and tiles
def classroom_area : ℕ := area classroom_length classroom_width
def rectangular_tile_area : ℕ := area rectangular_tile_length rectangular_tile_width
def triangular_tile_area : ℕ := area_triangular_tile triangular_tile_base triangular_tile_height

-- Define the number of tiles required
def number_of_rectangular_tiles : ℕ := (classroom_area + rectangular_tile_area - 1) / rectangular_tile_area -- ceiling division in lean
def number_of_triangular_tiles : ℕ := (classroom_area + triangular_tile_area - 1) / triangular_tile_area -- ceiling division in lean

-- Define the minimum number of tiles required
def minimum_number_of_tiles : ℕ := min number_of_rectangular_tiles number_of_triangular_tiles

-- The main theorem establishing the least number of tiles required
theorem least_number_of_tiles : minimum_number_of_tiles = 57 := by
    sorry

end least_number_of_tiles_l541_54136


namespace remainder_783245_div_7_l541_54191

theorem remainder_783245_div_7 :
  783245 % 7 = 1 :=
sorry

end remainder_783245_div_7_l541_54191


namespace intersection_AB_l541_54114

variable {x : ℝ}

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_AB : A ∩ B = {x | 0 < x ∧ x < 2} :=
by sorry

end intersection_AB_l541_54114


namespace chris_pennies_count_l541_54168

theorem chris_pennies_count (a c : ℤ) 
  (h1 : c + 2 = 4 * (a - 2)) 
  (h2 : c - 2 = 3 * (a + 2)) : 
  c = 62 := 
by 
  -- The actual proof is omitted
  sorry

end chris_pennies_count_l541_54168


namespace star_five_seven_l541_54123

def star (a b : ℕ) : ℕ := (a + b + 3) ^ 2

theorem star_five_seven : star 5 7 = 225 := by
  sorry

end star_five_seven_l541_54123


namespace largest_consecutive_positive_elements_l541_54180

theorem largest_consecutive_positive_elements (a : ℕ → ℝ)
  (h₁ : ∀ n ≥ 2, a n = a (n-1) + a (n+2)) :
  ∃ m, m = 5 ∧ ∀ k < m, a k > 0 :=
sorry

end largest_consecutive_positive_elements_l541_54180


namespace largest_value_l541_54119

-- Define the five expressions as given in the conditions
def exprA : ℕ := 3 + 1 + 2 + 8
def exprB : ℕ := 3 * 1 + 2 + 8
def exprC : ℕ := 3 + 1 * 2 + 8
def exprD : ℕ := 3 + 1 + 2 * 8
def exprE : ℕ := 3 * 1 * 2 * 8

-- Define the theorem stating that exprE is the largest value
theorem largest_value : exprE = 48 ∧ exprE > exprA ∧ exprE > exprB ∧ exprE > exprC ∧ exprE > exprD := by
  sorry

end largest_value_l541_54119


namespace largest_three_digit_number_divisible_by_8_l541_54148

-- Define the properties of a number being a three-digit number
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of a number being divisible by 8
def isDivisibleBy8 (n : ℕ) : Prop := n % 8 = 0

-- The theorem we want to prove: the largest three-digit number divisible by 8 is 992
theorem largest_three_digit_number_divisible_by_8 : ∃ n, isThreeDigitNumber n ∧ isDivisibleBy8 n ∧ (∀ m, isThreeDigitNumber m ∧ isDivisibleBy8 m → m ≤ 992) :=
  sorry

end largest_three_digit_number_divisible_by_8_l541_54148


namespace length_of_base_AD_l541_54144

-- Definitions based on the conditions
def isosceles_trapezoid (A B C D : Type) : Prop := sorry -- Implementation of an isosceles trapezoid
def length_of_lateral_side (A B C D : Type) : ℝ := 40 -- The lateral side is 40 cm
def angle_BAC (A B C D : Type) : ℝ := 45 -- The angle ∠BAC is 45 degrees
def bisector_O_center (O A B D M : Type) : Prop := sorry -- Implementation that O is the center of circumscribed circle and lies on bisector

-- Main theorem based on the derived problem statement
theorem length_of_base_AD (A B C D O M : Type) 
  (h_iso_trapezoid : isosceles_trapezoid A B C D)
  (h_length_lateral : length_of_lateral_side A B C D = 40)
  (h_angle_BAC : angle_BAC A B C D = 45)
  (h_O_center_bisector : bisector_O_center O A B D M)
  : ℝ :=
  20 * (Real.sqrt 6 + Real.sqrt 2)

end length_of_base_AD_l541_54144


namespace tetrahedron_circumsphere_surface_area_eq_five_pi_l541_54116

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

noncomputable def circumscribed_sphere_radius (a b : ℝ) : ℝ :=
  rectangle_diagonal a b / 2

noncomputable def circumscribed_sphere_surface_area (a b : ℝ) : ℝ :=
  4 * Real.pi * (circumscribed_sphere_radius a b)^2

theorem tetrahedron_circumsphere_surface_area_eq_five_pi :
  circumscribed_sphere_surface_area 2 1 = 5 * Real.pi := by
  sorry

end tetrahedron_circumsphere_surface_area_eq_five_pi_l541_54116


namespace num_pigs_on_farm_l541_54138

variables (P : ℕ)
def cows := 2 * P - 3
def goats := (2 * P - 3) + 6
def total_animals := P + cows P + goats P

theorem num_pigs_on_farm (h : total_animals P = 50) : P = 10 :=
sorry

end num_pigs_on_farm_l541_54138


namespace germination_percentage_l541_54152

theorem germination_percentage :
  ∀ (seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 : ℝ),
    seeds_plot1 = 300 →
    seeds_plot2 = 200 →
    germination_rate1 = 0.30 →
    germination_rate2 = 0.35 →
    ((germination_rate1 * seeds_plot1 + germination_rate2 * seeds_plot2) / (seeds_plot1 + seeds_plot2)) * 100 = 32 :=
by
  intros seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 h1 h2 h3 h4
  sorry

end germination_percentage_l541_54152


namespace possible_values_of_N_l541_54183

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end possible_values_of_N_l541_54183
