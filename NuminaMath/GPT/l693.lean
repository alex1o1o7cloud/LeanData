import Mathlib

namespace NUMINAMATH_GPT_smallest_positive_real_l693_69340

theorem smallest_positive_real (x : ℝ) (h₁ : ∃ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 4) : x = 29 / 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_real_l693_69340


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l693_69377

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l693_69377


namespace NUMINAMATH_GPT_sin_double_angle_fourth_quadrant_l693_69307

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_fourth_quadrant_l693_69307


namespace NUMINAMATH_GPT_remainder_of_70_div_17_l693_69378

theorem remainder_of_70_div_17 : 70 % 17 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_70_div_17_l693_69378


namespace NUMINAMATH_GPT_even_goals_more_likely_l693_69384

theorem even_goals_more_likely (p₁ : ℝ) (q₁ : ℝ) 
  (h₁ : q₁ = 1 - p₁)
  (independent_halves : (p₁ * p₁ + q₁ * q₁) > (2 * p₁ * q₁)) :
  (p₁ * p₁ + q₁ * q₁) > (1 - (p₁ * p₁ + q₁ * q₁)) :=
by
  sorry

end NUMINAMATH_GPT_even_goals_more_likely_l693_69384


namespace NUMINAMATH_GPT_find_percentage_l693_69343

-- conditions
def N : ℕ := 160
def expected_percentage : ℕ := 35

-- statement to prove
theorem find_percentage (P : ℕ) (h : P / 100 * N = 50 / 100 * N - 24) : P = expected_percentage :=
sorry

end NUMINAMATH_GPT_find_percentage_l693_69343


namespace NUMINAMATH_GPT_cara_sitting_pairs_l693_69361

theorem cara_sitting_pairs : ∀ (n : ℕ), n = 7 → ∃ (pairs : ℕ), pairs = 6 :=
by
  intros n hn
  have h : n - 1 = 6 := sorry
  exact ⟨n - 1, h⟩

end NUMINAMATH_GPT_cara_sitting_pairs_l693_69361


namespace NUMINAMATH_GPT_prob_return_to_freezer_l693_69354

-- Define the probabilities of picking two pops of each flavor
def probability_same_flavor (total: ℕ) (pop1: ℕ) (pop2: ℕ) : ℚ :=
  (pop1 * pop2) / (total * (total - 1))

-- Definitions according to the problem conditions
def cherry_pops : ℕ := 4
def orange_pops : ℕ := 3
def lemon_lime_pops : ℕ := 4
def total_pops : ℕ := cherry_pops + orange_pops + lemon_lime_pops

-- Calculate the probability of picking two ice pops of the same flavor
def prob_cherry : ℚ := probability_same_flavor total_pops cherry_pops (cherry_pops - 1)
def prob_orange : ℚ := probability_same_flavor total_pops orange_pops (orange_pops - 1)
def prob_lemon_lime : ℚ := probability_same_flavor total_pops lemon_lime_pops (lemon_lime_pops - 1)

def prob_same_flavor : ℚ := prob_cherry + prob_orange + prob_lemon_lime
def prob_diff_flavor : ℚ := 1 - prob_same_flavor

-- Theorem stating the probability of needing to return to the freezer
theorem prob_return_to_freezer : prob_diff_flavor = 8 / 11 := by
  sorry

end NUMINAMATH_GPT_prob_return_to_freezer_l693_69354


namespace NUMINAMATH_GPT_find_square_side_l693_69308

theorem find_square_side (a b x : ℕ) (h_triangle : a^2 + x^2 = b^2)
  (h_trapezoid : 2 * a + 2 * b + 2 * x = 60)
  (h_rectangle : 4 * a + 2 * x = 58) :
  a = 12 := by
  sorry

end NUMINAMATH_GPT_find_square_side_l693_69308


namespace NUMINAMATH_GPT_max_students_distributed_equally_l693_69318

theorem max_students_distributed_equally (pens pencils : ℕ) (h1 : pens = 3528) (h2 : pencils = 3920) : 
  Nat.gcd pens pencils = 392 := 
by 
  sorry

end NUMINAMATH_GPT_max_students_distributed_equally_l693_69318


namespace NUMINAMATH_GPT_time_for_train_to_pass_jogger_l693_69331

noncomputable def time_to_pass (s_jogger s_train : ℝ) (d_headstart l_train : ℝ) : ℝ :=
  let speed_jogger := s_jogger * (1000 / 3600)
  let speed_train := s_train * (1000 / 3600)
  let relative_speed := speed_train - speed_jogger
  let total_distance := d_headstart + l_train
  total_distance / relative_speed

theorem time_for_train_to_pass_jogger :
  time_to_pass 12 60 360 180 = 40.48 :=
by
  sorry

end NUMINAMATH_GPT_time_for_train_to_pass_jogger_l693_69331


namespace NUMINAMATH_GPT_arthur_walked_distance_in_miles_l693_69368

def blocks_west : ℕ := 8
def blocks_south : ℕ := 10
def block_length_in_miles : ℚ := 1 / 4

theorem arthur_walked_distance_in_miles : 
  (blocks_west + blocks_south) * block_length_in_miles = 4.5 := by
sorry

end NUMINAMATH_GPT_arthur_walked_distance_in_miles_l693_69368


namespace NUMINAMATH_GPT_employee_B_payment_l693_69367

theorem employee_B_payment (total_payment A_payment B_payment : ℝ) 
    (h1 : total_payment = 450) 
    (h2 : A_payment = 1.5 * B_payment) 
    (h3 : total_payment = A_payment + B_payment) : 
    B_payment = 180 := 
by
  sorry

end NUMINAMATH_GPT_employee_B_payment_l693_69367


namespace NUMINAMATH_GPT_dilation_image_l693_69304

theorem dilation_image (z : ℂ) (c : ℂ) (k : ℝ) (w : ℂ) (h₁ : c = 0 + 5 * I) 
  (h₂ : k = 3) (h₃ : w = 3 + 2 * I) : z = 9 - 4 * I :=
by
  -- Given conditions
  have hc : c = 0 + 5 * I := h₁
  have hk : k = 3 := h₂
  have hw : w = 3 + 2 * I := h₃

  -- Dilation formula
  let formula := (w - c) * k + c

  -- Prove the result
  -- sorry for now, the proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_dilation_image_l693_69304


namespace NUMINAMATH_GPT_find_x_from_w_condition_l693_69359

theorem find_x_from_w_condition :
  ∀ (x u y z w : ℕ), 
  (x = u + 7) → 
  (u = y + 5) → 
  (y = z + 12) → 
  (z = w + 25) → 
  (w = 100) → 
  x = 149 :=
by intros x u y z w h1 h2 h3 h4 h5
   sorry

end NUMINAMATH_GPT_find_x_from_w_condition_l693_69359


namespace NUMINAMATH_GPT_pension_equality_l693_69306

theorem pension_equality (x c d r s: ℝ) (h₁ : d ≠ c) 
    (h₂ : x > 0) (h₃ : 2 * x * (d - c) + d^2 - c^2 ≠ 0)
    (h₄ : ∀ k:ℝ, k * (x + c)^2 - k * x^2 = r)
    (h₅ : ∀ k:ℝ, k * (x + d)^2 - k * x^2 = s) 
    : ∃ k : ℝ, k = (s - r) / (2 * x * (d - c) + d^2 - c^2) 
    → k * x^2 = (s - r) * x^2 / (2 * x * (d - c) + d^2 - c^2) :=
by {
    sorry
}

end NUMINAMATH_GPT_pension_equality_l693_69306


namespace NUMINAMATH_GPT_average_speed_is_20_mph_l693_69301

-- Defining the conditions
def distance1 := 40 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Calculating total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1 -- hours
def time2 := distance2 / speed2 -- hours
def time3 := distance3 / speed3 -- hours
def total_time := time1 + time2 + time3

-- Theorem statement
theorem average_speed_is_20_mph : (total_distance / total_time) = 20 := by
  sorry

end NUMINAMATH_GPT_average_speed_is_20_mph_l693_69301


namespace NUMINAMATH_GPT_Sarah_total_weeds_l693_69390

noncomputable def Tuesday_weeds : ℕ := 25
noncomputable def Wednesday_weeds : ℕ := 3 * Tuesday_weeds
noncomputable def Thursday_weeds : ℕ := (1 / 5) * Tuesday_weeds
noncomputable def Friday_weeds : ℕ := (3 / 4) * Tuesday_weeds - 10

noncomputable def Total_weeds : ℕ := Tuesday_weeds + Wednesday_weeds + Thursday_weeds + Friday_weeds

theorem Sarah_total_weeds : Total_weeds = 113 := by
  sorry

end NUMINAMATH_GPT_Sarah_total_weeds_l693_69390


namespace NUMINAMATH_GPT_number_of_classmates_l693_69311

theorem number_of_classmates (total_apples : ℕ) (apples_per_classmate : ℕ) (people_in_class : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) (h3 : people_in_class = total_apples / apples_per_classmate) : 
  people_in_class = 3 :=
by sorry

end NUMINAMATH_GPT_number_of_classmates_l693_69311


namespace NUMINAMATH_GPT_sin_2_alpha_plus_pi_by_3_l693_69352

-- Define the statement to be proved
theorem sin_2_alpha_plus_pi_by_3 (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hcos : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (2 * α + π / 3) = 24 / 25 := sorry

end NUMINAMATH_GPT_sin_2_alpha_plus_pi_by_3_l693_69352


namespace NUMINAMATH_GPT_negation_of_proposition_l693_69339

theorem negation_of_proposition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a * x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l693_69339


namespace NUMINAMATH_GPT_profit_percentage_is_12_36_l693_69350

noncomputable def calc_profit_percentage (SP CP : ℝ) : ℝ :=
  let Profit := SP - CP
  (Profit / CP) * 100

theorem profit_percentage_is_12_36
  (SP : ℝ) (h1 : SP = 100)
  (CP : ℝ) (h2 : CP = 0.89 * SP) :
  calc_profit_percentage SP CP = 12.36 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_12_36_l693_69350


namespace NUMINAMATH_GPT_range_of_a_l693_69314

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x ^ 2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h₁ : p a) (h₂ : q a) : a ≤ -2 ∨ a = 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l693_69314


namespace NUMINAMATH_GPT_g_half_l693_69397

noncomputable def g : ℝ → ℝ := sorry

axiom g0 : g 0 = 0
axiom g1 : g 1 = 1
axiom g_non_decreasing : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom g_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom g_fraction : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

theorem g_half : g (1 / 2) = 1 / 2 := sorry

end NUMINAMATH_GPT_g_half_l693_69397


namespace NUMINAMATH_GPT_car_speed_l693_69303

-- Define the given conditions
def distance := 800 -- in kilometers
def time := 5 -- in hours

-- Define the speed calculation
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- State the theorem to be proved
theorem car_speed : speed distance time = 160 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_car_speed_l693_69303


namespace NUMINAMATH_GPT_sum_c_d_eq_30_l693_69376

noncomputable def c_d_sum : ℕ :=
  let c : ℕ := 28
  let d : ℕ := 2
  c + d

theorem sum_c_d_eq_30 : c_d_sum = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_c_d_eq_30_l693_69376


namespace NUMINAMATH_GPT_students_taking_neither_l693_69398

theorem students_taking_neither (total_students : ℕ)
    (students_math : ℕ) (students_physics : ℕ) (students_chemistry : ℕ)
    (students_math_physics : ℕ) (students_physics_chemistry : ℕ) (students_math_chemistry : ℕ)
    (students_all_three : ℕ) :
    total_students = 60 →
    students_math = 40 →
    students_physics = 30 →
    students_chemistry = 25 →
    students_math_physics = 18 →
    students_physics_chemistry = 10 →
    students_math_chemistry = 12 →
    students_all_three = 5 →
    (total_students - (students_math + students_physics + students_chemistry - students_math_physics - students_physics_chemistry - students_math_chemistry + students_all_three)) = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_taking_neither_l693_69398


namespace NUMINAMATH_GPT_x_equals_1_over_16_l693_69342

-- Given conditions
def distance_center_to_tangents_intersection : ℚ := 3 / 8
def radius_of_circle : ℚ := 3 / 16
def distance_center_to_CD : ℚ := 1 / 2

-- Calculated total distance
def total_distance_center_to_C : ℚ := distance_center_to_tangents_intersection + radius_of_circle

-- Problem statement
theorem x_equals_1_over_16 (x : ℚ) 
    (h : total_distance_center_to_C = x + distance_center_to_CD) : 
    x = 1 / 16 := 
by
  -- Proof is omitted, based on the provided solution steps
  sorry

end NUMINAMATH_GPT_x_equals_1_over_16_l693_69342


namespace NUMINAMATH_GPT_odd_function_extended_l693_69387

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≥ 0 then 
    x * Real.log (x + 1)
  else 
    x * Real.log (-x + 1)

theorem odd_function_extended : (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x ≥ 0 → f x = x * Real.log (x + 1)) →
  (∀ x : ℝ, x < 0 → f x = x * Real.log (-x + 1)) :=
by
  intros h_odd h_def_neg
  sorry

end NUMINAMATH_GPT_odd_function_extended_l693_69387


namespace NUMINAMATH_GPT_hexagons_after_cuts_l693_69357

theorem hexagons_after_cuts (rectangles_initial : ℕ) (cuts : ℕ) (sheets_total : ℕ)
  (initial_sides : ℕ) (additional_sides : ℕ) 
  (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (final_sides : ℕ) (number_of_hexagons : ℕ) :
  rectangles_initial = 15 →
  cuts = 60 →
  sheets_total = rectangles_initial + cuts →
  initial_sides = rectangles_initial * 4 →
  additional_sides = cuts * 4 →
  final_sides = initial_sides + additional_sides →
  triangle_sides = 3 →
  hexagon_sides = 6 →
  (sheets_total * 4 = final_sides) →
  number_of_hexagons = (final_sides - 225) / 3 →
  number_of_hexagons = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hexagons_after_cuts_l693_69357


namespace NUMINAMATH_GPT_calculate_width_of_vessel_base_l693_69316

noncomputable def cube_edge : ℝ := 17
noncomputable def base_length : ℝ := 20
noncomputable def water_rise : ℝ := 16.376666666666665
noncomputable def cube_volume : ℝ := cube_edge ^ 3
noncomputable def base_area (W : ℝ) : ℝ := base_length * W
noncomputable def displaced_volume (W : ℝ) : ℝ := base_area W * water_rise

theorem calculate_width_of_vessel_base :
  ∃ W : ℝ, displaced_volume W = cube_volume ∧ W = 15 := by
  sorry

end NUMINAMATH_GPT_calculate_width_of_vessel_base_l693_69316


namespace NUMINAMATH_GPT_maisy_earns_more_l693_69362

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end NUMINAMATH_GPT_maisy_earns_more_l693_69362


namespace NUMINAMATH_GPT_exists_digit_combination_l693_69326

theorem exists_digit_combination (d1 d2 d3 d4 : ℕ) (H1 : 42 * (d1 * 10 + 8) = 2 * 1000 + d2 * 100 + d3 * 10 + d4) (H2: ∃ n: ℕ, n = 2 + d2 + d3 + d4 ∧ n % 2 = 1):
  d1 = 4 ∧ 42 * 48 = 2016 ∨ d1 = 6 ∧ 42 * 68 = 2856 :=
sorry

end NUMINAMATH_GPT_exists_digit_combination_l693_69326


namespace NUMINAMATH_GPT_functional_equation_zero_l693_69353

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (hx : ∀ x y : ℝ, f (x + y) = f x + f y) : f 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_zero_l693_69353


namespace NUMINAMATH_GPT_value_of_x_squared_y_plus_xy_squared_l693_69358

variable {R : Type} [CommRing R] (x y : R)

-- Given conditions
def cond1 : Prop := x + y = 3
def cond2 : Prop := x * y = 2

-- The main theorem to prove
theorem value_of_x_squared_y_plus_xy_squared (h1 : cond1 x y) (h2 : cond2 x y) : x^2 * y + x * y^2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_y_plus_xy_squared_l693_69358


namespace NUMINAMATH_GPT_inscribed_square_area_l693_69345

-- Define the conditions and the problem
theorem inscribed_square_area
  (side_length : ℝ)
  (square_area : ℝ) :
  side_length = 24 →
  square_area = 576 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_square_area_l693_69345


namespace NUMINAMATH_GPT_select_test_point_l693_69324

theorem select_test_point (x1 x2 : ℝ) (h1 : x1 = 2 + 0.618 * (4 - 2)) (h2 : x2 = 2 + 4 - x1) :
  (x1 > x2 → x3 = 4 - 0.618 * (4 - x1)) ∨ (x1 < x2 → x3 = 6 - x3) :=
  sorry

end NUMINAMATH_GPT_select_test_point_l693_69324


namespace NUMINAMATH_GPT_patio_rows_before_rearrangement_l693_69341

theorem patio_rows_before_rearrangement (r c : ℕ) 
  (h1 : r * c = 160) 
  (h2 : (r + 4) * (c - 2) = 160)
  (h3 : ∃ k : ℕ, 5 * k = r)
  (h4 : ∃ l : ℕ, 5 * l = c) :
  r = 16 :=
by
  sorry

end NUMINAMATH_GPT_patio_rows_before_rearrangement_l693_69341


namespace NUMINAMATH_GPT_converse_proposition_false_l693_69360

theorem converse_proposition_false (a b c : ℝ) : ¬(∀ a b c : ℝ, (a > b) → (a * c^2 > b * c^2)) :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_converse_proposition_false_l693_69360


namespace NUMINAMATH_GPT_geometric_sequence_n_l693_69348

theorem geometric_sequence_n (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 * a 2 * a 3 = 4) 
  (h2 : a 4 * a 5 * a 6 = 12) 
  (h3 : a (n-1) * a n * a (n+1) = 324) : 
  n = 14 := 
  sorry

end NUMINAMATH_GPT_geometric_sequence_n_l693_69348


namespace NUMINAMATH_GPT_solution_interval_l693_69344

theorem solution_interval (X₀ : ℝ) (h₀ : Real.log (X₀ + 1) = 2 / X₀) : 1 < X₀ ∧ X₀ < 2 :=
by
  admit -- to be proved

end NUMINAMATH_GPT_solution_interval_l693_69344


namespace NUMINAMATH_GPT_middle_number_is_9_point_5_l693_69349

theorem middle_number_is_9_point_5 (x y z : ℝ) 
  (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 22) : y = 9.5 := 
by {
  sorry
}

end NUMINAMATH_GPT_middle_number_is_9_point_5_l693_69349


namespace NUMINAMATH_GPT_problem1_l693_69330

variable (m : ℤ)

theorem problem1 : m * (m - 3) + 3 * (3 - m) = (m - 3) ^ 2 := by
  sorry

end NUMINAMATH_GPT_problem1_l693_69330


namespace NUMINAMATH_GPT_triangle_area_division_l693_69347

theorem triangle_area_division (T T_1 T_2 T_3 : ℝ) 
  (hT1_pos : 0 < T_1) (hT2_pos : 0 < T_2) (hT3_pos : 0 < T_3) (hT : T = T_1 + T_2 + T_3) :
  T = (Real.sqrt T_1 + Real.sqrt T_2 + Real.sqrt T_3) ^ 2 :=
sorry

end NUMINAMATH_GPT_triangle_area_division_l693_69347


namespace NUMINAMATH_GPT_bukvinsk_acquaintances_l693_69380

theorem bukvinsk_acquaintances (Martin Klim Inna Tamara Kamilla : Type) 
  (acquaints : Type → Type → Prop)
  (exists_same_letters : ∀ (x y : Type), acquaints x y ↔ ∃ S, (x = S ∧ y = S)) :
  (∃ (count_Martin : ℕ), count_Martin = 20) →
  (∃ (count_Klim : ℕ), count_Klim = 15) →
  (∃ (count_Inna : ℕ), count_Inna = 12) →
  (∃ (count_Tamara : ℕ), count_Tamara = 12) →
  (∃ (count_Kamilla : ℕ), count_Kamilla = 15) := by
  sorry

end NUMINAMATH_GPT_bukvinsk_acquaintances_l693_69380


namespace NUMINAMATH_GPT_factorization_problem_l693_69399

theorem factorization_problem (a b c : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 7 * x + 12 = (x + a) * (x + b))
  (h2 : ∀ x : ℝ, x^2 - 8 * x - 20 = (x - b) * (x - c)) :
  a - b + c = -9 :=
sorry

end NUMINAMATH_GPT_factorization_problem_l693_69399


namespace NUMINAMATH_GPT_solve_quadratic_problem_l693_69370

theorem solve_quadratic_problem :
  ∀ x : ℝ, (x^2 + 6 * x + 8 = -(x + 4) * (x + 7)) ↔ (x = -4 ∨ x = -4.5) := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_problem_l693_69370


namespace NUMINAMATH_GPT_intersection_points_l693_69365

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

theorem intersection_points :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ ¬(x = x ∧ y = y) → 0 = 1 :=
sorry

end NUMINAMATH_GPT_intersection_points_l693_69365


namespace NUMINAMATH_GPT_evaluate_expression_l693_69309

theorem evaluate_expression :
  (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l693_69309


namespace NUMINAMATH_GPT_correct_calculation_for_A_l693_69315

theorem correct_calculation_for_A (x : ℝ) : (-2 * x) ^ 3 = -8 * x ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_for_A_l693_69315


namespace NUMINAMATH_GPT_total_students_l693_69363

-- Definition of the conditions given in the problem
def num5 : ℕ := 12
def num6 : ℕ := 6 * num5

-- The theorem representing the mathematically equivalent proof problem
theorem total_students : num5 + num6 = 84 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l693_69363


namespace NUMINAMATH_GPT_tadpoles_more_than_fish_l693_69371

def fish_initial : ℕ := 100
def tadpoles_initial := 4 * fish_initial
def snails_initial : ℕ := 150
def fish_caught : ℕ := 12
def tadpoles_to_frogs := (2 * tadpoles_initial) / 3
def snails_crawled_away : ℕ := 20

theorem tadpoles_more_than_fish :
  let fish_now : ℕ := fish_initial - fish_caught
  let tadpoles_now : ℕ := tadpoles_initial - tadpoles_to_frogs
  fish_now < tadpoles_now ∧ tadpoles_now - fish_now = 46 :=
by
  sorry

end NUMINAMATH_GPT_tadpoles_more_than_fish_l693_69371


namespace NUMINAMATH_GPT_finding_value_of_expression_l693_69325

open Real

theorem finding_value_of_expression
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : 1/a - 1/b - 1/(a + b) = 0) :
  (b/a + a/b)^2 = 5 :=
sorry

end NUMINAMATH_GPT_finding_value_of_expression_l693_69325


namespace NUMINAMATH_GPT_same_terminal_side_angle_exists_l693_69388

theorem same_terminal_side_angle_exists :
  ∃ k : ℤ, -5 * π / 8 + 2 * k * π = 11 * π / 8 := 
by
  sorry

end NUMINAMATH_GPT_same_terminal_side_angle_exists_l693_69388


namespace NUMINAMATH_GPT_min_value_x_plus_3y_min_value_xy_l693_69338

variable {x y : ℝ}

theorem min_value_x_plus_3y (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x + 3 * y ≥ 16 :=
sorry

theorem min_value_xy (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x * y ≥ 12 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_3y_min_value_xy_l693_69338


namespace NUMINAMATH_GPT_find_dimensions_l693_69373

-- Define the conditions
def perimeter (x y : ℕ) : Prop := (2 * (x + y) = 3996)
def divisible_parts (x y k : ℕ) : Prop := (x * y = 1998 * k) ∧ ∃ (k : ℕ), (k * 1998 = x * y) ∧ k ≠ 0

-- State the theorem
theorem find_dimensions (x y : ℕ) (k : ℕ) : perimeter x y ∧ divisible_parts x y k → (x = 1332 ∧ y = 666) ∨ (x = 666 ∧ y = 1332) :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_find_dimensions_l693_69373


namespace NUMINAMATH_GPT_sum_of_two_numbers_l693_69364

theorem sum_of_two_numbers (a b : ℕ) (h1 : a - b = 10) (h2 : a = 22) : a + b = 34 :=
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l693_69364


namespace NUMINAMATH_GPT_exists_acute_triangle_side_lengths_l693_69319

-- Define the real numbers d_1, d_2, ..., d_12 in the interval (1, 12).
noncomputable def real_numbers_in_interval (d : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → 1 < d n ∧ d n < 12

-- Define the condition for d_i, d_j, d_k to form an acute triangle
def forms_acuse_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- The main theorem statement
theorem exists_acute_triangle_side_lengths (d : ℕ → ℝ) (h : real_numbers_in_interval d) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ forms_acuse_triangle (d i) (d j) (d k) :=
sorry

end NUMINAMATH_GPT_exists_acute_triangle_side_lengths_l693_69319


namespace NUMINAMATH_GPT_restaurant_sales_decrease_l693_69396

-- Conditions
variable (Sales_August : ℝ := 42000)
variable (Sales_October : ℝ := 27000)
variable (a : ℝ) -- monthly average decrease rate as a decimal

-- Theorem statement
theorem restaurant_sales_decrease :
  42 * (1 - a)^2 = 27 := sorry

end NUMINAMATH_GPT_restaurant_sales_decrease_l693_69396


namespace NUMINAMATH_GPT_system_of_equations_l693_69302

-- Given conditions: Total number of fruits and total cost of the fruits purchased
def total_fruits := 1000
def total_cost := 999
def cost_of_sweet_fruit := (11 : ℚ) / 9
def cost_of_bitter_fruit := (4 : ℚ) / 7

-- Variables representing the number of sweet and bitter fruits
variables (x y : ℚ)

-- Problem statement in Lean 4
theorem system_of_equations :
  (x + y = total_fruits) ∧ (cost_of_sweet_fruit * x + cost_of_bitter_fruit * y = total_cost) ↔
  ((x + y = 1000) ∧ (11 / 9 * x + 4 / 7 * y = 999)) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_l693_69302


namespace NUMINAMATH_GPT_least_number_subtracted_l693_69393

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_pos : 0 < x) (h_init : n = 427398) (h_div : ∃ k : ℕ, (n - x) = 14 * k) : x = 6 :=
sorry

end NUMINAMATH_GPT_least_number_subtracted_l693_69393


namespace NUMINAMATH_GPT_find_cost_price_per_item_min_items_type_A_l693_69317

-- Definitions based on the conditions
def cost_A (x : ℝ) (y : ℝ) : Prop := 4 * x + 10 = 5 * y
def cost_B (x : ℝ) (y : ℝ) : Prop := 20 * x + 10 * y = 160

-- Proving the cost price per item of goods A and B
theorem find_cost_price_per_item : ∃ x y : ℝ, cost_A x y ∧ cost_B x y ∧ x = 5 ∧ y = 6 :=
by
  -- This is where the proof would go
  sorry

-- Additional conditions for part (2)
def profit_condition (a : ℕ) : Prop :=
  10 * (a - 30) + 8 * (200 - (a - 30)) - 5 * a - 6 * (200 - a) ≥ 640

-- Proving the minimum number of items of type A purchased
theorem min_items_type_A : ∃ a : ℕ, profit_condition a ∧ a ≥ 100 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_find_cost_price_per_item_min_items_type_A_l693_69317


namespace NUMINAMATH_GPT_hyperbola_properties_l693_69394

def hyperbola (x y : ℝ) : Prop := x^2 - 4 * y^2 = 1

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → (x + 2 * y = 0 ∨ x - 2 * y = 0)) ∧
  (2 * (1 / 2) = 1) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_properties_l693_69394


namespace NUMINAMATH_GPT_triangle_area_l693_69355

theorem triangle_area :
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  area = 3 := by
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  sorry

end NUMINAMATH_GPT_triangle_area_l693_69355


namespace NUMINAMATH_GPT_expected_worth_coin_flip_l693_69369

noncomputable def expected_worth : ℝ := 
  (1 / 3) * 6 + (2 / 3) * (-2) - 1

theorem expected_worth_coin_flip : expected_worth = -0.33 := 
by 
  unfold expected_worth
  norm_num
  sorry

end NUMINAMATH_GPT_expected_worth_coin_flip_l693_69369


namespace NUMINAMATH_GPT_remainder_3_mod_6_l693_69356

theorem remainder_3_mod_6 (n : ℕ) (h : n % 18 = 3) : n % 6 = 3 :=
by
    sorry

end NUMINAMATH_GPT_remainder_3_mod_6_l693_69356


namespace NUMINAMATH_GPT_first_platform_length_l693_69328

noncomputable def length_of_first_platform (t1 t2 l_train l_plat2 time1 time2 : ℕ) : ℕ :=
  let s1 := (l_train + t1) / time1
  let s2 := (l_train + l_plat2) / time2
  if s1 = s2 then t1 else 0

theorem first_platform_length:
  ∀ (time1 time2 : ℕ) (l_train l_plat2 : ℕ), time1 = 15 → time2 = 20 → l_train = 350 → l_plat2 = 250 → length_of_first_platform 100 l_plat2 l_train l_plat2 time1 time2 = 100 :=
by
  intros time1 time2 l_train l_plat2 ht1 ht2 ht3 ht4
  rw [ht1, ht2, ht3, ht4]
  dsimp [length_of_first_platform]
  rfl

end NUMINAMATH_GPT_first_platform_length_l693_69328


namespace NUMINAMATH_GPT_extreme_points_range_of_a_l693_69305

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2

-- Problem 1: Extreme points
theorem extreme_points (a : ℝ) : 
  (a ≤ 0 → ∃! x, ∀ y, f y a ≤ f x a) ∧
  (0 < a ∧ a < 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) ∧
  (a = 1/2 → ∀ x y, f y a ≤ f x a → x = y) ∧
  (a > 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) :=
sorry

-- Problem 2: Range of values for 'a'
theorem range_of_a (a : ℝ) : 
  (∀ x, f x a + Real.exp x ≥ x^3 + x) ↔ (a ≤ Real.exp 1 - 2) :=
sorry

end NUMINAMATH_GPT_extreme_points_range_of_a_l693_69305


namespace NUMINAMATH_GPT_adult_meal_cost_l693_69335

theorem adult_meal_cost (x : ℝ) 
  (total_people : ℕ) (kids : ℕ) (total_cost : ℝ)  
  (h_total_people : total_people = 11) 
  (h_kids : kids = 2) 
  (h_total_cost : total_cost = 72)
  (h_adult_meals : (total_people - kids : ℕ) • x = total_cost) : 
  x = 8 := 
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_adult_meal_cost_l693_69335


namespace NUMINAMATH_GPT_A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l693_69392

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Statement for (1)
theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a ∈ Set.Ioi 0 :=
sorry

-- Statement for (2)
theorem A_single_element_iff_and_value (a : ℝ) : 
  (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) ∧ A a = {2 / 3} :=
sorry

-- Statement for (3)
theorem A_at_most_one_element_iff (a : ℝ) : 
  (∃ x, A a = {x} ∨ A a = ∅) ↔ (a = 0 ∨ a ∈ Set.Ici (9 / 8)) :=
sorry

end NUMINAMATH_GPT_A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l693_69392


namespace NUMINAMATH_GPT_range_of_x_range_of_a_l693_69320

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_of_x (h1 : a = 1) (h2 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem range_of_a (h : ∀ x, p x a → q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_x_range_of_a_l693_69320


namespace NUMINAMATH_GPT_find_length_l693_69332

-- Define the perimeter and breadth as constants
def P : ℕ := 950
def B : ℕ := 100

-- State the theorem
theorem find_length (L : ℕ) (H : 2 * (L + B) = P) : L = 375 :=
by sorry

end NUMINAMATH_GPT_find_length_l693_69332


namespace NUMINAMATH_GPT_sum_square_geq_one_third_l693_69379

variable (a b c : ℝ)

theorem sum_square_geq_one_third (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end NUMINAMATH_GPT_sum_square_geq_one_third_l693_69379


namespace NUMINAMATH_GPT_walter_chore_days_l693_69323

-- Definitions for the conditions
variables (b w : ℕ)  -- b: days regular, w: days exceptionally well

-- Conditions
def days_eq : Prop := b + w = 15
def earnings_eq : Prop := 3 * b + 4 * w = 47

-- The theorem stating the proof problem
theorem walter_chore_days (hb : days_eq b w) (he : earnings_eq b w) : w = 2 :=
by
  -- We only need to state the theorem; the proof is omitted.
  sorry

end NUMINAMATH_GPT_walter_chore_days_l693_69323


namespace NUMINAMATH_GPT_hydrochloric_acid_solution_l693_69312

variable (V : ℝ) (pure_acid_added : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ)

theorem hydrochloric_acid_solution :
  initial_concentration = 0.10 → 
  final_concentration = 0.15 → 
  pure_acid_added = 3.52941176471 → 
  0.10 * V + 3.52941176471 = 0.15 * (V + 3.52941176471) → 
  V = 60 :=
by
  intros h_initial h_final h_pure h_equation
  sorry

end NUMINAMATH_GPT_hydrochloric_acid_solution_l693_69312


namespace NUMINAMATH_GPT_range_of_H_l693_69346

def H (x : ℝ) : ℝ := 2 * |2 * x + 2| - 3 * |2 * x - 2|

theorem range_of_H : Set.range H = Set.Ici 8 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_H_l693_69346


namespace NUMINAMATH_GPT_find_D_l693_69395

theorem find_D (A B D : ℕ) (h1 : (100 * A + 10 * B + D) * (A + B + D) = 1323) (h2 : A ≥ B) : D = 1 :=
sorry

end NUMINAMATH_GPT_find_D_l693_69395


namespace NUMINAMATH_GPT_selling_price_per_pound_is_correct_l693_69337

noncomputable def cost_of_40_lbs : ℝ := 40 * 0.38
noncomputable def cost_of_8_lbs : ℝ := 8 * 0.50
noncomputable def total_cost : ℝ := cost_of_40_lbs + cost_of_8_lbs
noncomputable def total_weight : ℝ := 40 + 8
noncomputable def profit : ℝ := total_cost * 0.20
noncomputable def total_selling_price : ℝ := total_cost + profit
noncomputable def selling_price_per_pound : ℝ := total_selling_price / total_weight

theorem selling_price_per_pound_is_correct :
  selling_price_per_pound = 0.48 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_per_pound_is_correct_l693_69337


namespace NUMINAMATH_GPT_range_of_a_minus_b_l693_69386

theorem range_of_a_minus_b {a b : ℝ} (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) : -6 < a - b ∧ a - b < 1 :=
by
  sorry -- The proof is skipped as per the instructions.

end NUMINAMATH_GPT_range_of_a_minus_b_l693_69386


namespace NUMINAMATH_GPT_angle_C_measure_l693_69391

theorem angle_C_measure
  (D C : ℝ)
  (h1 : C + D = 90)
  (h2 : C = 3 * D) :
  C = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_measure_l693_69391


namespace NUMINAMATH_GPT_sin_alpha_two_alpha_plus_beta_l693_69313

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < π / 2)
variable (h₂ : 0 < β ∧ β < π / 2)
variable (h₃ : Real.tan (α / 2) = 1 / 3)
variable (h₄ : Real.cos (α - β) = -4 / 5)

theorem sin_alpha (h₁ : 0 < α ∧ α < π / 2)
                  (h₃ : Real.tan (α / 2) = 1 / 3) :
                  Real.sin α = 3 / 5 :=
by
  sorry

theorem two_alpha_plus_beta (h₁ : 0 < α ∧ α < π / 2)
                            (h₂ : 0 < β ∧ β < π / 2)
                            (h₄ : Real.cos (α - β) = -4 / 5) :
                            2 * α + β = π :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_two_alpha_plus_beta_l693_69313


namespace NUMINAMATH_GPT_train_length_l693_69336

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 52) (h2 : time_sec = 9) (h3 : length_m = 129.96) : 
  length_m = (speed_km_hr * 1000 / 3600) * time_sec := 
sorry

end NUMINAMATH_GPT_train_length_l693_69336


namespace NUMINAMATH_GPT_hallie_read_pages_third_day_more_than_second_day_l693_69334

theorem hallie_read_pages_third_day_more_than_second_day :
  ∀ (d1 d2 d3 d4 : ℕ),
  d1 = 63 →
  d2 = 2 * d1 →
  d4 = 29 →
  d1 + d2 + d3 + d4 = 354 →
  (d3 - d2) = 10 :=
by
  intros d1 d2 d3 d4 h1 h2 h4 h_sum
  sorry

end NUMINAMATH_GPT_hallie_read_pages_third_day_more_than_second_day_l693_69334


namespace NUMINAMATH_GPT_find_f_neg_5_l693_69375

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_domain : ∀ x : ℝ, true)
variable (h_positive : ∀ x : ℝ, x > 0 → f x = log 5 x + 1)

theorem find_f_neg_5 : f (-5) = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_5_l693_69375


namespace NUMINAMATH_GPT_bowl_capacity_percentage_l693_69382

theorem bowl_capacity_percentage
    (initial_half_full : ℕ)
    (added_water : ℕ)
    (total_water : ℕ)
    (full_capacity : ℕ)
    (percentage_filled : ℚ) :
    initial_half_full * 2 = full_capacity →
    initial_half_full + added_water = total_water →
    added_water = 4 →
    total_water = 14 →
    percentage_filled = (total_water * 100) / full_capacity →
    percentage_filled = 70 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end NUMINAMATH_GPT_bowl_capacity_percentage_l693_69382


namespace NUMINAMATH_GPT_sum_a1_to_a5_l693_69329

noncomputable def f (x : ℝ) : ℝ := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5
noncomputable def g (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : ℝ := a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5

theorem sum_a1_to_a5 (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, f x = g x a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 1 = g 1 a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 0 = g 0 a_0 a_1 a_2 a_3 a_4 a_5) →
  a_0 = 62 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -57 :=
by
  intro hf1 hf2 hf3 ha0 hsum
  sorry

end NUMINAMATH_GPT_sum_a1_to_a5_l693_69329


namespace NUMINAMATH_GPT_relay_team_orderings_l693_69383

theorem relay_team_orderings (Jordan Mike Friend1 Friend2 Friend3 : Type) :
  ∃ n : ℕ, n = 12 :=
by
  -- Define the team members
  let team : List Type := [Jordan, Mike, Friend1, Friend2, Friend3]
  
  -- Define the number of ways to choose the 4th and 5th runners
  let ways_choose_45 := 2
  
  -- Define the number of ways to order the first 3 runners
  let ways_order_123 := Nat.factorial 3
  
  -- Calculate the total number of ways
  let total_ways := ways_choose_45 * ways_order_123
  
  -- The total ways should be 12
  use total_ways
  have h : total_ways = 12
  sorry
  exact h

end NUMINAMATH_GPT_relay_team_orderings_l693_69383


namespace NUMINAMATH_GPT_total_nap_duration_l693_69389

def nap1 : ℚ := 1 / 5
def nap2 : ℚ := 1 / 4
def nap3 : ℚ := 1 / 6
def hour_to_minutes : ℚ := 60

theorem total_nap_duration :
  (nap1 + nap2 + nap3) * hour_to_minutes = 37 := by
  sorry

end NUMINAMATH_GPT_total_nap_duration_l693_69389


namespace NUMINAMATH_GPT_scientific_notation_of_220_billion_l693_69351

theorem scientific_notation_of_220_billion :
  220000000000 = 2.2 * 10^11 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_220_billion_l693_69351


namespace NUMINAMATH_GPT_time_to_cover_escalator_l693_69322

def escalator_speed := 11 -- ft/sec
def escalator_length := 126 -- feet
def person_speed := 3 -- ft/sec

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 9 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l693_69322


namespace NUMINAMATH_GPT_initial_percentage_of_alcohol_l693_69374

variable (P : ℝ)
variables (x y : ℝ) (initial_percent replacement_percent replaced_quantity final_percent : ℝ)

def whisky_problem :=
  initial_percent = P ∧
  replacement_percent = 0.19 ∧
  replaced_quantity = 2/3 ∧
  final_percent = 0.26 ∧
  (P * (1 - replaced_quantity) + replacement_percent * replaced_quantity = final_percent)

theorem initial_percentage_of_alcohol :
  whisky_problem P 0.40 0.19 (2/3) 0.26 := sorry

end NUMINAMATH_GPT_initial_percentage_of_alcohol_l693_69374


namespace NUMINAMATH_GPT_distance_from_center_l693_69385

-- Define the circle equation as a predicate
def isCircle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x - 4 * y + 8

-- Define the center of the circle
def circleCenter : ℝ × ℝ := (1, -2)

-- Define the point in question
def point : ℝ × ℝ := (-3, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the proof problem
theorem distance_from_center :
  ∀ (x y : ℝ), isCircle x y → distance circleCenter point = 2 * Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_center_l693_69385


namespace NUMINAMATH_GPT_expected_non_allergic_l693_69381

theorem expected_non_allergic (p : ℝ) (n : ℕ) (h : p = 1 / 4) (hn : n = 300) : n * p = 75 :=
by sorry

end NUMINAMATH_GPT_expected_non_allergic_l693_69381


namespace NUMINAMATH_GPT_sandbox_width_l693_69366

theorem sandbox_width (P : ℕ) (W L : ℕ) (h1 : P = 30) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : W = 5 := 
sorry

end NUMINAMATH_GPT_sandbox_width_l693_69366


namespace NUMINAMATH_GPT_front_view_correct_l693_69372

section stack_problem

def column1 : List ℕ := [3, 2]
def column2 : List ℕ := [1, 4, 2]
def column3 : List ℕ := [5]
def column4 : List ℕ := [2, 1]

def tallest (l : List ℕ) : ℕ := l.foldr max 0

theorem front_view_correct :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 4, 5, 2] :=
sorry

end stack_problem

end NUMINAMATH_GPT_front_view_correct_l693_69372


namespace NUMINAMATH_GPT_bakery_storage_l693_69321

theorem bakery_storage (S F B : ℕ) (h1 : S * 8 = 3 * F) (h2 : F * 1 = 10 * B) (h3 : F * 1 = 8 * (B + 60)) : S = 900 :=
by
  -- We would normally put the proof steps here, but since it's specified to include only the statement
  sorry

end NUMINAMATH_GPT_bakery_storage_l693_69321


namespace NUMINAMATH_GPT_jessica_blueberry_pies_l693_69333

theorem jessica_blueberry_pies 
  (total_pies : ℕ)
  (ratio_apple : ℕ)
  (ratio_blueberry : ℕ)
  (ratio_cherry : ℕ)
  (h_total : total_pies = 36)
  (h_ratios : ratio_apple = 2)
  (h_ratios_b : ratio_blueberry = 5)
  (h_ratios_c : ratio_cherry = 3) : 
  total_pies * ratio_blueberry / (ratio_apple + ratio_blueberry + ratio_cherry) = 18 := 
by
  sorry

end NUMINAMATH_GPT_jessica_blueberry_pies_l693_69333


namespace NUMINAMATH_GPT_alice_unanswered_questions_l693_69300

theorem alice_unanswered_questions :
  ∃ (c w u : ℕ), (5 * c - 2 * w = 54) ∧ (2 * c + u = 36) ∧ (c + w + u = 30) ∧ (u = 8) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_alice_unanswered_questions_l693_69300


namespace NUMINAMATH_GPT_exists_person_with_girls_as_neighbors_l693_69327

theorem exists_person_with_girls_as_neighbors (boys girls : Nat) (sitting : Nat) 
  (h_boys : boys = 25) (h_girls : girls = 25) (h_sitting : sitting = boys + girls) :
  ∃ p : Nat, p < sitting ∧ (p % 2 = 1 → p.succ % sitting % 2 = 0) := 
by
  sorry

end NUMINAMATH_GPT_exists_person_with_girls_as_neighbors_l693_69327


namespace NUMINAMATH_GPT_john_brown_bags_l693_69310

theorem john_brown_bags :
  (∃ b : ℕ, 
     let total_macaroons := 12
     let weight_per_macaroon := 5
     let total_weight := total_macaroons * weight_per_macaroon
     let remaining_weight := 45
     let bag_weight := total_weight - remaining_weight
     let macaroons_per_bag := bag_weight / weight_per_macaroon
     total_macaroons / macaroons_per_bag = b
  ) → b = 4 :=
by
  sorry

end NUMINAMATH_GPT_john_brown_bags_l693_69310
