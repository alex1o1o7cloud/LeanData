import Mathlib

namespace johns_share_l240_240449

theorem johns_share (total_amount : ℕ) (r1 r2 r3 : ℕ) (h : total_amount = 6000) (hr1 : r1 = 2) (hr2 : r2 = 4) (hr3 : r3 = 6) :
  let total_ratio := r1 + r2 + r3
  let johns_ratio := r1
  let johns_share := (johns_ratio * total_amount) / total_ratio
  johns_share = 1000 :=
by
  sorry

end johns_share_l240_240449


namespace cost_per_ton_ice_correct_l240_240123

variables {a p n s : ℝ}

-- Define the cost per ton of ice received by enterprise A
noncomputable def cost_per_ton_ice_received (a p n s : ℝ) : ℝ :=
  (2.5 * a + p * s) * 1000 / (2000 - n * s)

-- The statement of the theorem
theorem cost_per_ton_ice_correct :
  ∀ a p n s : ℝ,
  2000 - n * s ≠ 0 →
  cost_per_ton_ice_received a p n s = (2.5 * a + p * s) * 1000 / (2000 - n * s) := by
  intros a p n s h
  unfold cost_per_ton_ice_received
  sorry

end cost_per_ton_ice_correct_l240_240123


namespace negation_of_existence_lt_zero_l240_240774

theorem negation_of_existence_lt_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by sorry

end negation_of_existence_lt_zero_l240_240774


namespace correct_options_l240_240604

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + 1

def option_2 : Prop := ∃ x : ℝ, f (x) = 0 ∧ x = Real.pi / 3
def option_3 : Prop := ∀ T > 0, (∀ x : ℝ, f (x) = f (x + T)) → T = Real.pi
def option_5 : Prop := ∀ x : ℝ, f (x - Real.pi / 6) = f (-(x - Real.pi / 6))

theorem correct_options :
  option_2 ∧ option_3 ∧ option_5 :=
by
  sorry

end correct_options_l240_240604


namespace movies_in_series_l240_240423

theorem movies_in_series :
  -- conditions 
  let number_books := 10
  let books_read := 14
  let book_read_vs_movies_extra := 5
  (∀ number_movies : ℕ, 
  (books_read = number_movies + book_read_vs_movies_extra) →
  -- question
  number_movies = 9) := sorry

end movies_in_series_l240_240423


namespace boat_travel_time_downstream_l240_240070

-- Define the given conditions and statement to prove
theorem boat_travel_time_downstream (B : ℝ) (C : ℝ) (Us : ℝ) (Ds : ℝ) :
  (C = B / 4) ∧ (Us = B - C) ∧ (Ds = B + C) ∧ (Us = 3) ∧ (15 / Us = 5) ∧ (15 / Ds = 3) :=
by
  -- Provide the proof here; currently using sorry to skip the proof
  sorry

end boat_travel_time_downstream_l240_240070


namespace red_team_score_l240_240448

theorem red_team_score (C R : ℕ) (h1 : C = 95) (h2 : C - R = 19) : R = 76 :=
by
  sorry

end red_team_score_l240_240448


namespace speed_of_man_in_still_water_l240_240598

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 4 = 48 →
  (v_m - v_s) * 6 = 24 →
  v_m = 8 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end speed_of_man_in_still_water_l240_240598


namespace characterization_of_points_l240_240178

def satisfies_eq (x : ℝ) (y : ℝ) : Prop :=
  max x (x^2) + min y (y^2) = 1

theorem characterization_of_points :
  ∀ x y : ℝ,
  satisfies_eq x y ↔
  ((x < 0 ∨ x > 1) ∧ (y < 0 ∨ y > 1) ∧ y ≤ 0 ∧ y = 1 - x^2) ∨
  ((x < 0 ∨ x > 1) ∧ (0 < y ∧ y < 1) ∧ x^2 + y^2 = 1 ∧ x ≤ -1 ∧ x > 0) ∨
  ((0 < x ∧ x < 1) ∧ (y < 0 ∨ y > 1) ∧ false) ∨
  ((0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y^2 = 1 - x) :=
sorry

end characterization_of_points_l240_240178


namespace f_increasing_intervals_g_range_l240_240772

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1 + Real.sin x) * f x

theorem f_increasing_intervals : 
  (∀ x, 0 ≤ x → x ≤ Real.pi / 2 → 0 ≤ Real.cos x) ∧ (∀ x, 3 * Real.pi / 2 ≤ x → x ≤ 2 * Real.pi → 0 ≤ Real.cos x) :=
sorry

theorem g_range : 
  ∀ x, 0 ≤ x → x ≤ 2 * Real.pi → -1 / 2 ≤ g x ∧ g x ≤ 4 :=
sorry

end f_increasing_intervals_g_range_l240_240772


namespace base8_arithmetic_l240_240611

def base8_to_base10 (n : Nat) : Nat :=
  sorry -- Placeholder for base 8 to base 10 conversion

def base10_to_base8 (n : Nat) : Nat :=
  sorry -- Placeholder for base 10 to base 8 conversion

theorem base8_arithmetic (n m : Nat) (h1 : base8_to_base10 45 = n) (h2 : base8_to_base10 76 = m) :
  base10_to_base8 ((n * 2) - m) = 14 :=
by
  sorry

end base8_arithmetic_l240_240611


namespace minimum_value_func_minimum_value_attained_l240_240775

noncomputable def func (x : ℝ) : ℝ := (4 / (x - 1)) + x

theorem minimum_value_func : ∀ (x : ℝ), x > 1 → func x ≥ 5 :=
by
  intros x hx
  -- proof goes here
  sorry

theorem minimum_value_attained : func 3 = 5 :=
by
  -- proof goes here
  sorry

end minimum_value_func_minimum_value_attained_l240_240775


namespace y_intercept_of_parallel_line_l240_240963

theorem y_intercept_of_parallel_line (m x1 y1 : ℝ) (h_slope : m = -3) (h_point : (x1, y1) = (3, -1))
  (b : ℝ) (h_line_parallel : ∀ x, b = y1 + m * (x - x1)) :
  b = 8 :=
by
  sorry

end y_intercept_of_parallel_line_l240_240963


namespace fish_remaining_correct_l240_240850

def guppies := 225
def angelfish := 175
def tiger_sharks := 200
def oscar_fish := 140
def discus_fish := 120

def guppies_sold := 3/5 * guppies
def angelfish_sold := 3/7 * angelfish
def tiger_sharks_sold := 1/4 * tiger_sharks
def oscar_fish_sold := 1/2 * oscar_fish
def discus_fish_sold := 2/3 * discus_fish

def guppies_remaining := guppies - guppies_sold
def angelfish_remaining := angelfish - angelfish_sold
def tiger_sharks_remaining := tiger_sharks - tiger_sharks_sold
def oscar_fish_remaining := oscar_fish - oscar_fish_sold
def discus_fish_remaining := discus_fish - discus_fish_sold

def total_remaining_fish := guppies_remaining + angelfish_remaining + tiger_sharks_remaining + oscar_fish_remaining + discus_fish_remaining

theorem fish_remaining_correct : total_remaining_fish = 450 := 
by 
  -- insert the necessary steps of the proof here
  sorry

end fish_remaining_correct_l240_240850


namespace min_value_expression_l240_240293

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (c : ℝ), c = 216 ∧
    ∀ (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
      ( (a^2 + 3*a + 2) * (b^2 + 3*b + 2) * (c^2 + 3*c + 2) / (a * b * c) ) ≥ 216 := 
sorry

end min_value_expression_l240_240293


namespace al_bill_cal_probability_l240_240956

-- Let's define the conditions and problem setup
def al_bill_cal_prob : ℚ :=
  let total_ways := 12 * 11 * 10
  let valid_ways := 12 -- This represent the summed valid cases as calculated
  valid_ways / total_ways

theorem al_bill_cal_probability :
  al_bill_cal_prob = 1 / 110 :=
  by
  -- Placeholder for calculation and proof
  sorry

end al_bill_cal_probability_l240_240956


namespace common_area_of_rectangle_and_circle_l240_240094

theorem common_area_of_rectangle_and_circle :
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  ∃ (common_area : ℝ), common_area = 9 * Real.pi :=
by
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  have common_area := 9 * Real.pi
  use common_area
  sorry

end common_area_of_rectangle_and_circle_l240_240094


namespace joshua_total_payment_is_correct_l240_240327

noncomputable def total_cost : ℝ := 
  let t_shirt_price := 8
  let sweater_price := 18
  let jacket_price := 80
  let jeans_price := 35
  let shoes_price := 60
  let jacket_discount := 0.10
  let shoes_discount := 0.15
  let clothing_tax_rate := 0.05
  let shoes_tax_rate := 0.08

  let t_shirt_count := 6
  let sweater_count := 4
  let jacket_count := 5
  let jeans_count := 3
  let shoes_count := 2

  let t_shirts_subtotal := t_shirt_price * t_shirt_count
  let sweaters_subtotal := sweater_price * sweater_count
  let jackets_subtotal := jacket_price * jacket_count
  let jeans_subtotal := jeans_price * jeans_count
  let shoes_subtotal := shoes_price * shoes_count

  let jackets_discounted := jackets_subtotal * (1 - jacket_discount)
  let shoes_discounted := shoes_subtotal * (1 - shoes_discount)

  let total_before_tax := t_shirts_subtotal + sweaters_subtotal + jackets_discounted + jeans_subtotal + shoes_discounted

  let t_shirts_tax := t_shirts_subtotal * clothing_tax_rate
  let sweaters_tax := sweaters_subtotal * clothing_tax_rate
  let jackets_tax := jackets_discounted * clothing_tax_rate
  let jeans_tax := jeans_subtotal * clothing_tax_rate
  let shoes_tax := shoes_discounted * shoes_tax_rate

  total_before_tax + t_shirts_tax + sweaters_tax + jackets_tax + jeans_tax + shoes_tax

theorem joshua_total_payment_is_correct : total_cost = 724.41 := by
  sorry

end joshua_total_payment_is_correct_l240_240327


namespace max_sum_numbered_cells_max_zero_number_cell_l240_240399

-- Part 1
theorem max_sum_numbered_cells (n : ℕ) (grid : Matrix (Fin (2*n+1)) (Fin (2*n+1)) Cell) (mines : Finset (Fin (2*n+1) × Fin (2*n+1))) 
  (h1 : mines.card = n^2 + 1) :
  ∃ sum : ℕ, sum = 8 * n^2 + 4 := sorry

-- Part 2
theorem max_zero_number_cell (n k : ℕ) (grid : Matrix (Fin n) (Fin n) Cell) (mines : Finset (Fin n × Fin n)) 
  (h1 : mines.card = k) :
  ∃ (k_max : ℕ), k_max = (Nat.floor ((n + 2) / 3) ^ 2) - 1 := sorry

end max_sum_numbered_cells_max_zero_number_cell_l240_240399


namespace quadratic_roots_sign_l240_240797

theorem quadratic_roots_sign (p q : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x * y = q ∧ x + y = -p) ↔ q < 0 :=
sorry

end quadratic_roots_sign_l240_240797


namespace angle_parallel_result_l240_240456

theorem angle_parallel_result (A B : ℝ) (h1 : A = 60) (h2 : (A = B ∨ A + B = 180)) : (B = 60 ∨ B = 120) :=
by
  sorry

end angle_parallel_result_l240_240456


namespace cone_lateral_surface_area_l240_240435

noncomputable def lateralSurfaceArea (r l : ℝ) : ℝ := Real.pi * r * l

theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 5 → lateralSurfaceArea r l = 10 * Real.pi :=
by 
  intros r l hr hl
  rw [hr, hl]
  unfold lateralSurfaceArea
  norm_num
  sorry

end cone_lateral_surface_area_l240_240435


namespace parabola_hyperbola_tangent_l240_240016

open Real

theorem parabola_hyperbola_tangent (n : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 6 → y^2 - n * x^2 = 4 → y ≥ 6) ↔ (n = 12 + 4 * sqrt 7 ∨ n = 12 - 4 * sqrt 7) :=
by
  sorry

end parabola_hyperbola_tangent_l240_240016


namespace number_of_initials_sets_l240_240718

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l240_240718


namespace blocks_before_jess_turn_l240_240679

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end blocks_before_jess_turn_l240_240679


namespace prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l240_240393

noncomputable def P_A := 4 / 5
noncomputable def P_B := 3 / 4

def independent (P_X P_Y : ℚ) := P_X * P_Y

theorem prob_both_shoot_in_one_round : independent P_A P_B = 3 / 5 := by
  sorry

noncomputable def P_A_1 := 2 * (4 / 5) * (1 / 5)
noncomputable def P_A_2 := (4 / 5) * (4 / 5)
noncomputable def P_B_1 := 2 * (3 / 4) * (1 / 4)
noncomputable def P_B_2 := (3 / 4) * (3 / 4)

def event_A (P_A_1 P_A_2 P_B_1 P_B_2 : ℚ) := (P_A_1 * P_B_2) + (P_A_2 * P_B_1)

theorem prob_specified_shots_in_two_rounds : event_A P_A_1 P_A_2 P_B_1 P_B_2 = 3 / 10 := by
  sorry

end prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l240_240393


namespace find_a_find_b_plus_c_l240_240843

variable (a b c : ℝ)
variable (A B C : ℝ) -- Angles in radians

-- Condition: Given that 2a / cos A = (3c - 2b) / cos B
axiom condition1 : 2 * a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B)

-- Condition 1: b = sqrt(5) * sin B
axiom condition2 : b = Real.sqrt 5 * (Real.sin B)

-- Proof problem for finding a
theorem find_a : a = 5 / 3 := by
  sorry

-- Condition 2: a = sqrt(6) and the area is sqrt(5) / 2
axiom condition3 : a = Real.sqrt 6
axiom condition4 : 1 / 2 * b * c * (Real.sin A) = Real.sqrt 5 / 2

-- Proof problem for finding b + c
theorem find_b_plus_c : b + c = 4 := by
  sorry

end find_a_find_b_plus_c_l240_240843


namespace intersection_points_area_l240_240425

noncomputable def C (x : ℝ) : ℝ := (Real.log x)^2

noncomputable def L (α : ℝ) (x : ℝ) : ℝ :=
  (2 * Real.log α / α) * x - (Real.log α)^2

noncomputable def n (α : ℝ) : ℕ :=
  if α < 1 then 0 else if α = 1 then 1 else 2

noncomputable def S (α : ℝ) : ℝ :=
  2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α

theorem intersection_points (α : ℝ) (h : 0 < α) : n α = if α < 1 then 0 else if α = 1 then 1 else 2 := by
  sorry

theorem area (α : ℝ) (h : 0 < α ∧ α < 1) : S α = 2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α := by
  sorry

end intersection_points_area_l240_240425


namespace kirsty_initial_models_l240_240552

theorem kirsty_initial_models 
  (x : ℕ)
  (initial_price : ℝ)
  (increased_price : ℝ)
  (models_bought : ℕ)
  (h_initial_price : initial_price = 0.45)
  (h_increased_price : increased_price = 0.5)
  (h_models_bought : models_bought = 27) 
  (h_total_saved : x * initial_price = models_bought * increased_price) :
  x = 30 :=
by 
  sorry

end kirsty_initial_models_l240_240552


namespace arc_length_one_radian_l240_240203

-- Given definitions and conditions
def radius : ℝ := 6370
def angle : ℝ := 1

-- Arc length formula
def arc_length (R α : ℝ) : ℝ := R * α

-- Statement to prove
theorem arc_length_one_radian : arc_length radius angle = 6370 := 
by 
  -- Proof goes here
  sorry

end arc_length_one_radian_l240_240203


namespace eval_product_eq_1093_l240_240407

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 7)

theorem eval_product_eq_1093 : (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end eval_product_eq_1093_l240_240407


namespace trigonometric_equation_solution_l240_240095

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) :
  5.14 * (Real.sin (3 * x)) + Real.sin (5 * x) = 2 * (Real.cos (2 * x)) ^ 2 - 2 * (Real.sin (3 * x)) ^ 2 →
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨ (∃ k : ℤ, x = (π / 18) * (4 * k + 1)) :=
  by
  intro h
  sorry

end trigonometric_equation_solution_l240_240095


namespace find_value_of_f_l240_240305

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem find_value_of_f (ω φ : ℝ) (h_symmetry : ∀ x : ℝ, f ω φ (π/4 + x) = f ω φ (π/4 - x)) :
  f ω φ (π/4) = 2 ∨ f ω φ (π/4) = -2 := 
sorry

end find_value_of_f_l240_240305


namespace lines_are_coplanar_l240_240282

-- Define the first line
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, 2 - m * t, 6 + t)

-- Define the second line
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (4 + m * u, 5 + 3 * u, 8 + 2 * u)

-- Define the vector connecting points on the lines when t=0 and u=0
def connecting_vector : ℝ × ℝ × ℝ :=
  (1, 3, 2)

-- Define the cross product of the direction vectors
def cross_product (m : ℝ) : ℝ × ℝ × ℝ :=
  ((-2 * m - 3), (m + 2), (6 + 2 * m))

-- Prove that lines are coplanar when m = -9/4
theorem lines_are_coplanar : ∃ k : ℝ, ∀ m : ℝ,
  cross_product m = (k * 1, k * 3, k * 2) → m = -9/4 :=
by
  sorry

end lines_are_coplanar_l240_240282


namespace min_solutions_f_eq_zero_l240_240361

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 3) = f x)
variable (h_zero_at_2 : f 2 = 0)

theorem min_solutions_f_eq_zero : ∃ S : Finset ℝ, (∀ x ∈ S, f x = 0) ∧ 7 ≤ S.card ∧ (∀ x ∈ S, x > 0 ∧ x < 6) := 
sorry

end min_solutions_f_eq_zero_l240_240361


namespace even_iff_a_zero_monotonous_iff_a_range_max_value_l240_240838

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 2

-- (I) Prove that f(x) is even on [-5, 5] if and only if a = 0
theorem even_iff_a_zero (a : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a = f (-x) a) ↔ a = 0 := sorry

-- (II) Prove that f(x) is monotonous on [-5, 5] if and only if a ≥ 10 or a ≤ -10
theorem monotonous_iff_a_range (a : ℝ) : (∀ x y : ℝ, -5 ≤ x ∧ x ≤ y ∧ y ≤ 5 → f x a ≤ f y a) ↔ (a ≥ 10 ∨ a ≤ -10) := sorry

-- (III) Prove the maximum value of f(x) in the interval [-5, 5]
theorem max_value (a : ℝ) : (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ (∀ y : ℝ, -5 ≤ y ∧ y ≤ 5 → f y a ≤ f x a)) ∧  
                           ((a ≥ 0 → f 5 a = 27 + 5 * a) ∧ (a < 0 → f (-5) a = 27 - 5 * a)) := sorry

end even_iff_a_zero_monotonous_iff_a_range_max_value_l240_240838


namespace combined_tax_rate_l240_240694

-- Definitions and conditions
def tax_rate_mork : ℝ := 0.45
def tax_rate_mindy : ℝ := 0.20
def income_ratio_mindy_to_mork : ℝ := 4

-- Theorem statement
theorem combined_tax_rate :
  ∀ (M : ℝ), (tax_rate_mork * M + tax_rate_mindy * (income_ratio_mindy_to_mork * M)) / (M + income_ratio_mindy_to_mork * M) = 0.25 :=
by
  intros M
  sorry

end combined_tax_rate_l240_240694


namespace combined_work_time_l240_240198

def ajay_completion_time : ℕ := 8
def vijay_completion_time : ℕ := 24

theorem combined_work_time (T_A T_V : ℕ) (h1 : T_A = ajay_completion_time) (h2 : T_V = vijay_completion_time) :
  1 / (1 / (T_A : ℝ) + 1 / (T_V : ℝ)) = 6 :=
by
  rw [h1, h2]
  sorry

end combined_work_time_l240_240198


namespace fg_of_3_is_83_l240_240812

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2
theorem fg_of_3_is_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_is_83_l240_240812


namespace total_distance_correct_l240_240429

-- Given conditions
def fuel_efficiency_city : Float := 15
def fuel_efficiency_highway : Float := 25
def fuel_efficiency_gravel : Float := 18

def gallons_used_city : Float := 2.5
def gallons_used_highway : Float := 3.8
def gallons_used_gravel : Float := 1.7

-- Define distances
def distance_city := fuel_efficiency_city * gallons_used_city
def distance_highway := fuel_efficiency_highway * gallons_used_highway
def distance_gravel := fuel_efficiency_gravel * gallons_used_gravel

-- Define total distance
def total_distance := distance_city + distance_highway + distance_gravel

-- Prove the total distance traveled is 163.1 miles
theorem total_distance_correct : total_distance = 163.1 := by
  -- Proof to be filled in
  sorry

end total_distance_correct_l240_240429


namespace david_total_course_hours_l240_240316

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end david_total_course_hours_l240_240316


namespace forty_percent_of_number_l240_240977

variables {N : ℝ}

theorem forty_percent_of_number (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) : 0.40 * N = 120 :=
by sorry

end forty_percent_of_number_l240_240977


namespace average_salary_all_workers_l240_240088

-- Definitions based on the conditions
def num_technicians : ℕ := 7
def num_other_workers : ℕ := 7
def avg_salary_technicians : ℕ := 12000
def avg_salary_other_workers : ℕ := 8000
def total_workers : ℕ := 14

-- Total salary calculations based on the conditions
def total_salary_technicians : ℕ := num_technicians * avg_salary_technicians
def total_salary_other_workers : ℕ := num_other_workers * avg_salary_other_workers
def total_salary_all_workers : ℕ := total_salary_technicians + total_salary_other_workers

-- The statement to be proved
theorem average_salary_all_workers : total_salary_all_workers / total_workers = 10000 :=
by
  -- proof will be added here
  sorry

end average_salary_all_workers_l240_240088


namespace rectangle_perimeter_inscribed_l240_240985

noncomputable def circle_area : ℝ := 32 * Real.pi
noncomputable def rectangle_area : ℝ := 34
noncomputable def rectangle_perimeter : ℝ := 28

theorem rectangle_perimeter_inscribed (area_circle : ℝ := 32 * Real.pi)
  (area_rectangle : ℝ := 34) : ∃ (P : ℝ), P = 28 :=
by
  use rectangle_perimeter
  sorry

end rectangle_perimeter_inscribed_l240_240985


namespace fraction_meaningful_l240_240134

theorem fraction_meaningful (a : ℝ) : (∃ x, x = 2 / (a + 1)) ↔ a ≠ -1 :=
by
  sorry

end fraction_meaningful_l240_240134


namespace december_sales_fraction_l240_240428

theorem december_sales_fraction (A : ℚ) : 
  let sales_jan_to_nov := 11 * A
  let sales_dec := 5 * A
  let total_sales := sales_jan_to_nov + sales_dec
  (sales_dec / total_sales) = 5 / 16 :=
by
  sorry

end december_sales_fraction_l240_240428


namespace units_digit_square_l240_240947

theorem units_digit_square (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) (h2 : (n % 10 = 2) ∨ (n % 10 = 7)) :
  ∀ (d : ℕ), (d = 2 ∨ d = 6 ∨ d = 3) → (n^2 % 10 ≠ d) :=
by
  sorry

end units_digit_square_l240_240947


namespace steve_assignments_fraction_l240_240553

theorem steve_assignments_fraction (h_sleep: ℝ) (h_school: ℝ) (h_family: ℝ) (total_hours: ℝ) : 
  h_sleep = 1/3 ∧ 
  h_school = 1/6 ∧ 
  h_family = 10 ∧ 
  total_hours = 24 → 
  (2 / total_hours = 1 / 12) :=
by
  intros h
  sorry

end steve_assignments_fraction_l240_240553


namespace alpha_in_third_quadrant_l240_240059

theorem alpha_in_third_quadrant (α : ℝ)
 (h₁ : Real.tan (α - 3 * Real.pi) > 0)
 (h₂ : Real.sin (-α + Real.pi) < 0) :
 (0 < α % (2 * Real.pi) ∧ α % (2 * Real.pi) < Real.pi) := 
sorry

end alpha_in_third_quadrant_l240_240059


namespace repair_cost_l240_240819

theorem repair_cost (purchase_price transport_charges selling_price profit_percentage R : ℝ)
  (h1 : purchase_price = 10000)
  (h2 : transport_charges = 1000)
  (h3 : selling_price = 24000)
  (h4 : profit_percentage = 0.5)
  (h5 : selling_price = (1 + profit_percentage) * (purchase_price + R + transport_charges)) :
  R = 5000 :=
by
  sorry

end repair_cost_l240_240819


namespace eval_expression_l240_240044

open Real

noncomputable def e : ℝ := 2.71828

theorem eval_expression : abs (5 * e - 15) = 1.4086 := by
  sorry

end eval_expression_l240_240044


namespace third_candle_remaining_fraction_l240_240074

theorem third_candle_remaining_fraction (t : ℝ) 
  (h1 : 0 < t)
  (second_candle_fraction_remaining : ℝ := 2/5)
  (third_candle_fraction_remaining : ℝ := 3/7)
  (second_candle_burned_fraction : ℝ := 3/5)
  (third_candle_burned_fraction : ℝ := 4/7)
  (second_candle_burn_rate : ℝ := 3 / (5 * t))
  (third_candle_burn_rate : ℝ := 4 / (7 * t))
  (remaining_burn_time_second : ℝ := (2 * t) / 3)
  (third_candle_burned_in_remaining_time : ℝ := (2 * t * 4) / (3 * 7 * t))
  (common_denominator_third : ℝ := 21)
  (converted_third_candle_fraction_remaining : ℝ := 9 / 21)
  (third_candle_fraction_subtracted : ℝ := 8 / 21) :
  (converted_third_candle_fraction_remaining - third_candle_fraction_subtracted) = 1 / 21 := by
  sorry

end third_candle_remaining_fraction_l240_240074


namespace b_charges_l240_240716

theorem b_charges (total_cost : ℕ) (a_hours b_hours c_hours : ℕ)
  (h_total_cost : total_cost = 720)
  (h_a_hours : a_hours = 9)
  (h_b_hours : b_hours = 10)
  (h_c_hours : c_hours = 13) :
  (total_cost * b_hours / (a_hours + b_hours + c_hours)) = 225 :=
by
  sorry

end b_charges_l240_240716


namespace constant_term_zero_quadratic_l240_240032

theorem constant_term_zero_quadratic (m : ℝ) :
  (-m^2 + 1 = 0) → m = -1 :=
by
  intro h
  sorry

end constant_term_zero_quadratic_l240_240032


namespace distance_from_house_to_work_l240_240526

-- Definitions for the conditions
variables (D : ℝ) (speed_to_work speed_back_work : ℝ) (time_to_work time_back_work total_time : ℝ)

-- Specific conditions in the problem
noncomputable def conditions : Prop :=
  (speed_back_work = 20) ∧
  (speed_to_work = speed_back_work / 2) ∧
  (time_to_work = D / speed_to_work) ∧
  (time_back_work = D / speed_back_work) ∧
  (total_time = 6) ∧
  (time_to_work + time_back_work = total_time)

-- The statement to prove the distance D is 40 km given the conditions
theorem distance_from_house_to_work (h : conditions D speed_to_work speed_back_work time_to_work time_back_work total_time) : D = 40 :=
sorry

end distance_from_house_to_work_l240_240526


namespace parabola_vertex_value_of_a_l240_240296

-- Define the conditions as given in the math problem
variables (a b c : ℤ)
def quadratic_fun (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Given conditions about the vertex and a point on the parabola
def vertex_condition : Prop := (quadratic_fun a b c 2 = 3)
def point_condition : Prop := (quadratic_fun a b c 1 = 0)

-- Statement to prove
theorem parabola_vertex_value_of_a : vertex_condition a b c ∧ point_condition a b c → a = -3 :=
sorry

end parabola_vertex_value_of_a_l240_240296


namespace maximize_product_minimize_product_l240_240439

-- Define the numbers that need to be arranged
def numbers : List ℕ := [2, 4, 6, 8]

-- Prove that 82 * 64 is the maximum product arrangement
theorem maximize_product : ∃ a b c d : ℕ, (a = 8) ∧ (b = 2) ∧ (c = 6) ∧ (d = 4) ∧ 
  (a * 10 + b) * (c * 10 + d) = 5248 :=
by
  existsi 8, 2, 6, 4
  constructor; constructor
  repeat {assumption}
  sorry

-- Prove that 28 * 46 is the minimum product arrangement
theorem minimize_product : ∃ a b c d : ℕ, (a = 2) ∧ (b = 8) ∧ (c = 4) ∧ (d = 6) ∧ 
  (a * 10 + b) * (c * 10 + d) = 1288 :=
by
  existsi 2, 8, 4, 6
  constructor; constructor
  repeat {assumption}
  sorry

end maximize_product_minimize_product_l240_240439


namespace find_second_number_l240_240858

theorem find_second_number (A B : ℝ) (h1 : A = 3200) (h2 : 0.10 * A = 0.20 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l240_240858


namespace min_surface_area_l240_240620

/-- Defining the conditions and the problem statement -/
def solid (volume : ℝ) (face1 face2 : ℝ) : Prop := 
  ∃ x y z, x * y * z = volume ∧ (x * y = face1 ∨ y * z = face1 ∨ z * x = face1)
                      ∧ (x * y = face2 ∨ y * z = face2 ∨ z * x = face2)

def juan_solids (face1 face2 face3 face4 face5 face6 : ℝ) : Prop :=
  solid 128 4 32 ∧ solid 128 64 16 ∧ solid 128 8 32

theorem min_surface_area {volume : ℝ} {face1 face2 face3 face4 face5 face6 : ℝ} 
  (h : juan_solids 4 32 64 16 8 32) : 
  ∃ area : ℝ, area = 688 :=
sorry

end min_surface_area_l240_240620


namespace range_of_b_l240_240607

theorem range_of_b (a b x : ℝ) (ha : 0 < a ∧ a ≤ 5 / 4) (hb : 0 < b) :
  (∀ x, |x - a| < b → |x - a^2| < 1 / 2) ↔ 0 < b ∧ b ≤ 3 / 16 :=
by
  sorry

end range_of_b_l240_240607


namespace primes_p_q_divisibility_l240_240723

theorem primes_p_q_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hq_eq : q = p + 2) :
  (p + q) ∣ (p ^ q + q ^ p) := 
sorry

end primes_p_q_divisibility_l240_240723


namespace marie_daily_rent_l240_240083

noncomputable def daily_revenue (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ) : ℝ :=
  bread_loaves * bread_price + cakes * cake_price

noncomputable def total_profit (daily_revenue : ℝ) (days : ℕ) (cash_register_cost : ℝ) : ℝ :=
  cash_register_cost

noncomputable def daily_profit (total_profit : ℝ) (days : ℕ) : ℝ :=
  total_profit / days

noncomputable def daily_profit_after_electricity (daily_profit : ℝ) (electricity_cost : ℝ) : ℝ :=
  daily_profit - electricity_cost

noncomputable def daily_rent (daily_revenue : ℝ) (daily_profit_after_electricity : ℝ) : ℝ :=
  daily_revenue - daily_profit_after_electricity

theorem marie_daily_rent
  (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ)
  (days : ℕ) (cash_register_cost : ℝ) (electricity_cost : ℝ) :
  bread_loaves = 40 → bread_price = 2 → cakes = 6 → cake_price = 12 →
  days = 8 → cash_register_cost = 1040 → electricity_cost = 2 →
  daily_rent (daily_revenue bread_loaves bread_price cakes cake_price)
             (daily_profit_after_electricity (daily_profit (total_profit (daily_revenue bread_loaves bread_price cakes cake_price) days cash_register_cost) days) electricity_cost) = 24 :=
by
  intros h0 h1 h2 h3 h4 h5 h6
  sorry

end marie_daily_rent_l240_240083


namespace expansion_l240_240077

variable (x : ℝ)

noncomputable def expr : ℝ := (3 / 4) * (8 / (x^2) + 5 * x - 6)

theorem expansion :
  expr x = (6 / (x^2)) + (15 * x / 4) - 4.5 :=
by
  sorry

end expansion_l240_240077


namespace graph_of_equation_is_two_intersecting_lines_l240_240501

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ (x y : ℝ), (x - y)^2 = 3 * x^2 - y^2 ↔ 
  (x = (1 - Real.sqrt 5) / 2 * y) ∨ (x = (1 + Real.sqrt 5) / 2 * y) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l240_240501


namespace vacation_cost_split_l240_240939

theorem vacation_cost_split 
  (john_paid mary_paid lisa_paid : ℕ) 
  (total_amount : ℕ) 
  (share : ℕ)
  (j m : ℤ)
  (h1 : john_paid = 150)
  (h2 : mary_paid = 90)
  (h3 : lisa_paid = 210)
  (h4 : total_amount = 450)
  (h5 : share = total_amount / 3) 
  (h6 : john_paid - share = j) 
  (h7 : mary_paid - share = m) 
  : j - m = -60 :=
by
  sorry

end vacation_cost_split_l240_240939


namespace sufficient_but_not_necessary_l240_240646

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > b + 1) → (a > b) ∧ (¬(a > b) → ¬(a > b + 1)) :=
by
  sorry

end sufficient_but_not_necessary_l240_240646


namespace extremum_point_of_f_l240_240510

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem extremum_point_of_f : ∃ x, x = 1 ∧ (∀ y ≠ 1, f y ≥ f x) := 
sorry

end extremum_point_of_f_l240_240510


namespace suji_age_problem_l240_240328

theorem suji_age_problem (x : ℕ) 
  (h1 : 5 * x + 6 = 13 * (4 * x + 6) / 11)
  (h2 : 11 * (4 * x + 6) = 9 * (3 * x + 6)) :
  4 * x = 16 :=
by
  sorry

end suji_age_problem_l240_240328


namespace polygon_sum_of_sides_l240_240469

-- Define the problem conditions and statement
theorem polygon_sum_of_sides :
  ∀ (A B C D E F : ℝ)
    (area_polygon : ℝ)
    (AB BC FA DE horizontal_distance_DF : ℝ),
    area_polygon = 75 →
    AB = 7 →
    BC = 10 →
    FA = 6 →
    DE = AB →
    horizontal_distance_DF = 8 →
    (DE + (2 * area_polygon - AB * BC) / (2 * horizontal_distance_DF) = 8.25) := 
by
  intro A B C D E F area_polygon AB BC FA DE horizontal_distance_DF
  intro h_area_polygon h_AB h_BC h_FA h_DE h_horizontal_distance_DF
  sorry

end polygon_sum_of_sides_l240_240469


namespace true_proposition_l240_240564

theorem true_proposition : 
  (∃ x0 : ℝ, x0 > 0 ∧ 3^x0 + x0 = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, abs x - a * x = abs (-x) - a * (-x)) := by
  sorry

end true_proposition_l240_240564


namespace blue_notebook_cost_l240_240179

theorem blue_notebook_cost
    (total_spent : ℕ)
    (total_notebooks : ℕ)
    (red_notebooks : ℕ) (red_cost : ℕ)
    (green_notebooks : ℕ) (green_cost : ℕ)
    (blue_notebooks : ℕ) (blue_total_cost : ℕ) 
    (blue_cost : ℕ) :
    total_spent = 37 →
    total_notebooks = 12 →
    red_notebooks = 3 →
    red_cost = 4 →
    green_notebooks = 2 →
    green_cost = 2 →
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks →
    blue_total_cost = total_spent - red_notebooks * red_cost - green_notebooks * green_cost →
    blue_cost = blue_total_cost / blue_notebooks →
    blue_cost = 3 := 
    by sorry

end blue_notebook_cost_l240_240179


namespace negation_equiv_l240_240138

theorem negation_equiv (x : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) := 
by 
  sorry

end negation_equiv_l240_240138


namespace tap_emptying_time_l240_240455

theorem tap_emptying_time
  (F : ℝ := 1 / 3)
  (T_combined : ℝ := 7.5):
  ∃ x : ℝ, x = 5 ∧ (F - (1 / x) = 1 / T_combined) := 
sorry

end tap_emptying_time_l240_240455


namespace interest_rate_per_annum_l240_240895

noncomputable def principal : ℝ := 933.3333333333334
noncomputable def amount : ℝ := 1120
noncomputable def time : ℝ := 4

theorem interest_rate_per_annum (P A T : ℝ) (hP : P = principal) (hA : A = amount) (hT : T = time) :
  ∃ R : ℝ, R = 1.25 :=
sorry

end interest_rate_per_annum_l240_240895


namespace negation_relation_l240_240612

def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

def not_p (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 1
def not_q (x : ℝ) : Prop := x ≥ -2 ∧ x ≤ 1

theorem negation_relation : (∀ x, not_p x → not_q x) ∧ ¬ (∀ x, not_q x → not_p x) :=
by 
  sorry

end negation_relation_l240_240612


namespace selling_price_l240_240147

-- Definitions for conditions
variables (CP SP_loss SP_profit : ℝ)
variable (h1 : SP_loss = 0.8 * CP)
variable (h2 : SP_profit = 1.05 * CP)
variable (h3 : SP_profit = 11.8125)

-- Theorem statement to prove
theorem selling_price (h1 : SP_loss = 0.8 * CP) (h2 : SP_profit = 1.05 * CP) (h3 : SP_profit = 11.8125) :
  SP_loss = 9 := 
sorry

end selling_price_l240_240147


namespace sqrt_sequence_convergence_l240_240600

theorem sqrt_sequence_convergence :
  ∃ x : ℝ, (x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2) :=
sorry

end sqrt_sequence_convergence_l240_240600


namespace hotdogs_remainder_zero_l240_240486

theorem hotdogs_remainder_zero :
  25197624 % 6 = 0 :=
by
  sorry -- Proof not required

end hotdogs_remainder_zero_l240_240486


namespace exponent_proof_l240_240319

theorem exponent_proof (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 :=
by
  -- Proof steps
  sorry

end exponent_proof_l240_240319


namespace prime_eq_solution_l240_240948

theorem prime_eq_solution (a b : ℕ) (h1 : Nat.Prime a) (h2 : b > 0)
  (h3 : 9 * (2 * a + b) ^ 2 = 509 * (4 * a + 511 * b)) : 
  (a = 251 ∧ b = 7) :=
sorry

end prime_eq_solution_l240_240948


namespace ellipse_foci_coordinates_l240_240342

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) → ∃ c : ℝ, (c = 4) ∧ (x = c ∨ x = -c) ∧ (y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l240_240342


namespace evaluate_expression_l240_240771

theorem evaluate_expression : 2 + (3 / (4 + (5 / 6))) = 76 / 29 := 
by
  sorry

end evaluate_expression_l240_240771


namespace jodi_third_week_miles_l240_240186

theorem jodi_third_week_miles (total_miles : ℕ) (first_week : ℕ) (second_week : ℕ) (fourth_week : ℕ) (days_per_week : ℕ) (third_week_miles_per_day : ℕ) 
  (H1 : first_week * days_per_week + second_week * days_per_week + third_week_miles_per_day * days_per_week + fourth_week * days_per_week = total_miles)
  (H2 : first_week = 1) 
  (H3 : second_week = 2) 
  (H4 : fourth_week = 4)
  (H5 : total_miles = 60)
  (H6 : days_per_week = 6) :
    third_week_miles_per_day = 3 :=
by sorry

end jodi_third_week_miles_l240_240186


namespace sandy_spent_percentage_l240_240193

theorem sandy_spent_percentage (I R : ℝ) (hI : I = 200) (hR : R = 140) : 
  ((I - R) / I) * 100 = 30 :=
by
  sorry

end sandy_spent_percentage_l240_240193


namespace inequality_condition_l240_240445

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the main theorem to be proven
theorem inequality_condition (a b : ℝ) (h_a : a > 11 / 4) (h_b : b > 3 / 2) :
  (∀ x : ℝ, |x + 1| < b → |f x + 3| < a) :=
by
  -- We state the required proof without providing the steps
  sorry

end inequality_condition_l240_240445


namespace point_coordinates_correct_l240_240790

def point_coordinates : (ℕ × ℕ) :=
(11, 9)

theorem point_coordinates_correct :
  point_coordinates = (11, 9) :=
by
  sorry

end point_coordinates_correct_l240_240790


namespace range_of_a_l240_240159

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a + 3

theorem range_of_a :
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ g x2 a = 0 ∧ |x1 - x2| ≤ 1) ↔ (a ∈ Set.Icc 2 3) := sorry

end range_of_a_l240_240159


namespace problem_statement_l240_240866

theorem problem_statement 
  (x1 y1 x2 y2 x3 y3 x4 y4 a b c : ℝ)
  (h1 : x1 > 0) (h2 : y1 > 0)
  (h3 : x2 < 0) (h4 : y2 > 0)
  (h5 : x3 < 0) (h6 : y3 < 0)
  (h7 : x4 > 0) (h8 : y4 < 0)
  (h9 : (x1 - a)^2 + (y1 - b)^2 ≤ c^2)
  (h10 : (x2 - a)^2 + (y2 - b)^2 ≤ c^2)
  (h11 : (x3 - a)^2 + (y3 - b)^2 ≤ c^2)
  (h12 : (x4 - a)^2 + (y4 - b)^2 ≤ c^2) : a^2 + b^2 < c^2 :=
by sorry

end problem_statement_l240_240866


namespace arithmetic_sequence_inequality_l240_240368

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

-- All terms are positive
def all_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem arithmetic_sequence_inequality
  (h_arith_seq : is_arithmetic_sequence a a1 d)
  (h_non_zero_diff : d ≠ 0)
  (h_positive : all_positive a) :
  (a 1) * (a 8) < (a 4) * (a 5) :=
by
  sorry

end arithmetic_sequence_inequality_l240_240368


namespace sugar_needed_l240_240962

variable (a b c d : ℝ)
variable (H1 : a = 2)
variable (H2 : b = 1)
variable (H3 : d = 5)

theorem sugar_needed (c : ℝ) : c = 2.5 :=
by
  have H : 2 / 1 = 5 / c := by {
    sorry
  }
  sorry

end sugar_needed_l240_240962


namespace f1_min_max_f2_min_max_l240_240651

-- Define the first function and assert its max and min values
def f1 (x : ℝ) : ℝ := x^3 + 2 * x

theorem f1_min_max : ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  (∃ x_min x_max, x_min = -1 ∧ x_max = 1 ∧ f1 x_min = -3 ∧ f1 x_max = 3) := by
  sorry

-- Define the second function and assert its max and min values
def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem f2_min_max : ∀ x ∈ Set.Icc (0 : ℝ) 3,
  (∃ x_min x_max, x_min = 0 ∧ x_max = 3 ∧ (f2 x_min = -4) ∧ f2 x_max = 2) := by
  sorry

end f1_min_max_f2_min_max_l240_240651


namespace necklaces_caught_l240_240734

noncomputable def total_necklaces_caught (boudreaux rhonda latch cecilia : ℕ) : ℕ :=
  boudreaux + rhonda + latch + cecilia

theorem necklaces_caught :
  ∃ (boudreaux rhonda latch cecilia : ℕ), 
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 ∧
    cecilia = latch + 3 ∧
    total_necklaces_caught boudreaux rhonda latch cecilia = 49 ∧
    (total_necklaces_caught boudreaux rhonda latch cecilia) % 7 = 0 :=
by
  sorry

end necklaces_caught_l240_240734


namespace fraction_inequality_l240_240369

theorem fraction_inequality (a : ℝ) (h : a ≠ 2) : (1 / (a^2 - 4 * a + 4) > 2 / (a^3 - 8)) :=
by sorry

end fraction_inequality_l240_240369


namespace zachary_pushups_l240_240745

theorem zachary_pushups (david_pushups : ℕ) (h1 : david_pushups = 44) (h2 : ∀ z : ℕ, z = david_pushups + 7) : z = 51 :=
by
  sorry

end zachary_pushups_l240_240745


namespace solution_set_f_l240_240539

def f (x a b : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_f (a b : ℝ) (h1 : b = 2 * a) (h2 : 0 < a) :
  {x | f (2 - x) a b > 0} = {x | x < 0 ∨ 4 < x} :=
by
  sorry

end solution_set_f_l240_240539


namespace fixed_point_line_l240_240398

theorem fixed_point_line (m x y : ℝ) (h : (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0) :
  x = 3 ∧ y = 1 :=
sorry

end fixed_point_line_l240_240398


namespace logarithm_simplification_l240_240136

theorem logarithm_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 7 / Real.log 9 + 1)) =
  1 - (Real.log 7 / Real.log 1008) :=
sorry

end logarithm_simplification_l240_240136


namespace inverse_proposition_l240_240477

theorem inverse_proposition (a b : ℝ) (h : ab = 0) : (a = 0 → ab = 0) :=
by
  sorry

end inverse_proposition_l240_240477


namespace number_of_customers_l240_240264

theorem number_of_customers 
  (offices sandwiches_per_office total_sandwiches group_sandwiches_per_customer half_group_sandwiches : ℕ)
  (h1 : offices = 3)
  (h2 : sandwiches_per_office = 10)
  (h3 : total_sandwiches = 54)
  (h4 : group_sandwiches_per_customer = 4)
  (h5 : half_group_sandwiches = 54 - (3 * 10))
  : half_group_sandwiches = 24 → 2 * 12 = 24 :=
by
  sorry

end number_of_customers_l240_240264


namespace doves_count_l240_240873

theorem doves_count 
  (num_doves : ℕ)
  (num_eggs_per_dove : ℕ)
  (hatch_rate : ℚ)
  (initial_doves : num_doves = 50)
  (eggs_per_dove : num_eggs_per_dove = 5)
  (hatch_fraction : hatch_rate = 7/9) :
  (num_doves + Int.toNat ((hatch_rate * num_doves * num_eggs_per_dove).floor)) = 244 :=
by
  sorry

end doves_count_l240_240873


namespace solve_equation_l240_240531

-- Define the conditions
def satisfies_equation (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ n^5 + n^4 = 7^m - 1

-- Theorem statement
theorem solve_equation : ∀ n m : ℕ, satisfies_equation n m ↔ (n = 2 ∧ m = 2) := 
by { sorry }

end solve_equation_l240_240531


namespace floor_sqrt_72_l240_240156

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 :=
by
  -- Proof required here
  sorry

end floor_sqrt_72_l240_240156


namespace value_of_T_l240_240408

variables {A M T E H : ℕ}

theorem value_of_T (H : ℕ) (MATH : ℕ) (MEET : ℕ) (TEAM : ℕ) (H_eq : H = 8) (MATH_eq : MATH = 47) (MEET_eq : MEET = 62) (TEAM_eq : TEAM = 58) :
  T = 9 :=
by
  sorry

end value_of_T_l240_240408


namespace num_valid_pairs_l240_240576

theorem num_valid_pairs (a b : ℕ) (h1 : b > a) (h2 : a > 4) (h3 : b > 4)
(h4 : a * b = 3 * (a - 4) * (b - 4)) : 
    (1 + (a - 6) = 1 ∧ 72 = b - 6) ∨
    (2 + (a - 6) = 2 ∧ 36 = b - 6) ∨
    (3 + (a - 6) = 3 ∧ 24 = b - 6) ∨
    (4 + (a - 6) = 4 ∧ 18 = b - 6) :=
sorry

end num_valid_pairs_l240_240576


namespace intersection_M_N_l240_240602

noncomputable def M : Set ℝ := {x | x^2 + x - 6 < 0}
noncomputable def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x } = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l240_240602


namespace otimes_2_5_l240_240309

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l240_240309


namespace constant_term_in_expansion_l240_240383

theorem constant_term_in_expansion (n k : ℕ) (x : ℝ) (choose : ℕ → ℕ → ℕ):
  (choose 12 3) * (6 ^ 3) = 47520 :=
by
  sorry

end constant_term_in_expansion_l240_240383


namespace animal_costs_l240_240349

theorem animal_costs :
  ∃ (C G S P : ℕ),
      C + G + S + P = 1325 ∧
      G + S + P = 425 ∧
      C + S + P = 1225 ∧
      G + P = 275 ∧
      C = 900 ∧
      G = 100 ∧
      S = 150 ∧
      P = 175 :=
by
  sorry

end animal_costs_l240_240349


namespace simplify_fraction_l240_240257

def a : ℕ := 2016
def b : ℕ := 2017

theorem simplify_fraction :
  (a^4 - 2 * a^3 * b + 3 * a^2 * b^2 - a * b^3 + 1) / (a^2 * b^2) = 1 - 1 / b^2 :=
by
  sorry

end simplify_fraction_l240_240257


namespace simple_annual_interest_rate_l240_240066

noncomputable def monthly_interest_payment : ℝ := 216
noncomputable def principal_amount : ℝ := 28800
noncomputable def number_of_months_in_a_year : ℕ := 12

theorem simple_annual_interest_rate :
  ((monthly_interest_payment * number_of_months_in_a_year) / principal_amount) * 100 = 9 := by
sorry

end simple_annual_interest_rate_l240_240066


namespace inversely_proportional_y_ratio_l240_240412

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end inversely_proportional_y_ratio_l240_240412


namespace find_other_number_l240_240849

variable (A B : ℕ)
variable (LCM : ℕ → ℕ → ℕ)
variable (HCF : ℕ → ℕ → ℕ)

theorem find_other_number (h1 : LCM A B = 2310) 
  (h2 : HCF A B = 30) (h3 : A = 210) : B = 330 := by
  sorry

end find_other_number_l240_240849


namespace num_sets_l240_240910

theorem num_sets {A : Set ℕ} :
  {1} ⊆ A ∧ A ⊆ {1, 2, 3, 4, 5} → ∃ n, n = 16 := 
by
  sorry

end num_sets_l240_240910


namespace units_digit_of_N_is_8_l240_240859

def product_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens * units

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens + units

theorem units_digit_of_N_is_8 (N : ℕ) (hN_range : 10 ≤ N ∧ N < 100)
    (hN_eq : N = product_of_digits N * sum_of_digits N) : N % 10 = 8 :=
sorry

end units_digit_of_N_is_8_l240_240859


namespace gcd_180_270_eq_90_l240_240046

theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := sorry

end gcd_180_270_eq_90_l240_240046


namespace winning_candidate_percentage_l240_240730

theorem winning_candidate_percentage (P : ℕ) (majority : ℕ) (total_votes : ℕ) (h1 : majority = 188) (h2 : total_votes = 470) (h3 : 2 * majority = (2 * P - 100) * total_votes) : 
  P = 70 := 
sorry

end winning_candidate_percentage_l240_240730


namespace solve_inequality_l240_240176

-- Defining the inequality
def inequality (x : ℝ) : Prop := 1 / (x - 1) ≤ 1

-- Stating the theorem
theorem solve_inequality :
  { x : ℝ | inequality x } = { x : ℝ | x < 1 } ∪ { x : ℝ | 2 ≤ x } :=
by
  sorry

end solve_inequality_l240_240176


namespace functional_ineq_solution_l240_240836

theorem functional_ineq_solution (n : ℕ) (h : n > 0) :
  (∀ x : ℝ, n = 1 → (x^n + (1 - x)^n ≤ 1)) ∧
  (∀ x : ℝ, n > 1 → ((x < 0 ∨ x > 1) → (x^n + (1 - x)^n > 1))) :=
by
  intros
  sorry

end functional_ineq_solution_l240_240836


namespace no_A_with_integer_roots_l240_240786

theorem no_A_with_integer_roots 
  (A : ℕ) 
  (h1 : A > 0) 
  (h2 : A < 10) 
  : ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ p + q = 10 + A ∧ p * q = 10 * A + A :=
by sorry

end no_A_with_integer_roots_l240_240786


namespace ratio_of_expenditures_l240_240504

-- Let us define the conditions and rewrite the proof problem statement.
theorem ratio_of_expenditures
  (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℝ)
  (H1 : income_P1 / income_P2 = 5 / 4)
  (H2 : income_P1 = 5000)
  (H3 : income_P1 - expenditure_P1 = 2000)
  (H4 : income_P2 - expenditure_P2 = 2000) :
  expenditure_P1 / expenditure_P2 = 3 / 2 :=
sorry

end ratio_of_expenditures_l240_240504


namespace extreme_value_of_f_range_of_a_l240_240225

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem extreme_value_of_f (a : ℝ) (ha : 0 < a) : ∃ x, f x a = a - a * Real.log a - 1 :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 a = f x2 a ∧ abs (x1 - x2) ≥ 1 ) →
  (e - 1 < a ∧ a < Real.exp 2 - Real.exp 1) :=
sorry

end extreme_value_of_f_range_of_a_l240_240225


namespace perimeter_correct_l240_240040

-- Definitions based on the conditions
def large_rectangle_area : ℕ := 12 * 12
def shaded_rectangle_area : ℕ := 6 * 4
def non_shaded_area : ℕ := large_rectangle_area - shaded_rectangle_area
def perimeter_of_non_shaded_region : ℕ := 2 * ((12 - 6) + (12 - 4))

-- The theorem to prove
theorem perimeter_correct (large_rectangle_area_eq : large_rectangle_area = 144) :
  perimeter_of_non_shaded_region = 28 :=
by
  sorry

end perimeter_correct_l240_240040


namespace fred_balloons_l240_240372

variable (initial_balloons : ℕ := 709)
variable (balloons_given : ℕ := 221)
variable (remaining_balloons : ℕ := 488)

theorem fred_balloons :
  initial_balloons - balloons_given = remaining_balloons :=
  by
    sorry

end fred_balloons_l240_240372


namespace gain_percentage_l240_240365

theorem gain_percentage (selling_price gain : ℕ) (h_sp : selling_price = 110) (h_gain : gain = 10) :
  (gain * 100) / (selling_price - gain) = 10 :=
by
  sorry

end gain_percentage_l240_240365


namespace not_diff_of_squares_2022_l240_240356

theorem not_diff_of_squares_2022 :
  ¬ ∃ a b : ℤ, a^2 - b^2 = 2022 :=
by
  sorry

end not_diff_of_squares_2022_l240_240356


namespace tom_savings_l240_240353

theorem tom_savings :
  let insurance_cost_per_month := 20
  let total_months := 24
  let procedure_cost := 5000
  let insurance_coverage := 0.80
  let total_insurance_cost := total_months * insurance_cost_per_month
  let insurance_cover_amount := procedure_cost * insurance_coverage
  let out_of_pocket_cost := procedure_cost - insurance_cover_amount
  let savings := procedure_cost - total_insurance_cost - out_of_pocket_cost
  savings = 3520 :=
by
  sorry

end tom_savings_l240_240353


namespace dot_product_a_b_l240_240681

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (1, -2)

theorem dot_product_a_b :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = -3 :=
by
  sorry

end dot_product_a_b_l240_240681


namespace paths_A_to_C_l240_240037

theorem paths_A_to_C :
  let paths_AB := 2
  let paths_BD := 3
  let paths_DC := 3
  let paths_AC_direct := 1
  paths_AB * paths_BD * paths_DC + paths_AC_direct = 19 :=
by
  sorry

end paths_A_to_C_l240_240037


namespace three_students_received_A_l240_240491

variables (A B C E D : Prop)
variables (h1 : A → B) (h2 : B → C) (h3 : C → E) (h4 : E → D)

theorem three_students_received_A :
  (A ∨ ¬A) ∧ (B ∨ ¬B) ∧ (C ∨ ¬C) ∧ (E ∨ ¬E) ∧ (D ∨ ¬D) ∧ (¬A ∧ ¬B) → (C ∧ E ∧ D) ∧ ¬A ∧ ¬B :=
by sorry

end three_students_received_A_l240_240491


namespace cost_of_bananas_and_cantaloupe_l240_240748

-- Let a, b, c, and d be real numbers representing the prices of apples, bananas, cantaloupe, and dates respectively.
variables (a b c d : ℝ)

-- Conditions given in the problem
axiom h1 : a + b + c + d = 40
axiom h2 : d = 3 * a
axiom h3 : c = (a + b) / 2

-- Goal is to prove that the sum of the prices of bananas and cantaloupe is 8 dollars.
theorem cost_of_bananas_and_cantaloupe : b + c = 8 :=
by
  sorry

end cost_of_bananas_and_cantaloupe_l240_240748


namespace pizza_slices_l240_240183

theorem pizza_slices (P T S : ℕ) (h1 : P = 2) (h2 : T = 16) : S = 8 :=
by
  -- to be filled in
  sorry

end pizza_slices_l240_240183


namespace initial_pollykawgs_computation_l240_240834

noncomputable def initial_pollykawgs_in_pond (daily_rate_matured : ℕ) (daily_rate_caught : ℕ)
  (total_days : ℕ) (catch_days : ℕ) : ℕ :=
let first_phase := (daily_rate_matured + daily_rate_caught) * catch_days
let second_phase := daily_rate_matured * (total_days - catch_days)
first_phase + second_phase

theorem initial_pollykawgs_computation :
  initial_pollykawgs_in_pond 50 10 44 20 = 2400 :=
by sorry

end initial_pollykawgs_computation_l240_240834


namespace johns_age_in_8_years_l240_240555

theorem johns_age_in_8_years :
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  current_age + 8 = twice_age_five_years_ago :=
by
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  sorry

end johns_age_in_8_years_l240_240555


namespace find_a_of_parabola_and_hyperbola_intersection_l240_240768

theorem find_a_of_parabola_and_hyperbola_intersection
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_asymptotes_intersect_directrix_distance : ∀ (x_A x_B : ℝ),
    -1 / (4 * a) = (1 / 2) * x_A ∧ -1 / (4 * a) = -(1 / 2) * x_B →
    |x_B - x_A| = 4) : a = 1 / 4 := by
  sorry

end find_a_of_parabola_and_hyperbola_intersection_l240_240768


namespace complement_union_l240_240081

open Set

namespace ProofFormalization

/-- Declaration of the universal set U, and sets A and B -/
def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

def complement {α : Type*} (s t : Set α) : Set α := t \ s

/-- Theorem statement that proves the complement of A ∪ B with respect to U is {5} -/
theorem complement_union :
  complement (A ∪ B) U = {5} :=
by
  sorry

end ProofFormalization

end complement_union_l240_240081


namespace sum_of_coordinates_l240_240214

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l240_240214


namespace simplify_expression_l240_240934

theorem simplify_expression (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 :=
by
  sorry

end simplify_expression_l240_240934


namespace find_d_l240_240465

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) : d = 2 :=
by
  sorry

end find_d_l240_240465


namespace solution_set_of_absolute_value_inequality_l240_240261

theorem solution_set_of_absolute_value_inequality :
  { x : ℝ | |x + 1| - |x - 2| > 1 } = { x : ℝ | 1 < x } :=
by 
  sorry

end solution_set_of_absolute_value_inequality_l240_240261


namespace total_cups_needed_l240_240747

def servings : Float := 18.0
def cups_per_serving : Float := 2.0

theorem total_cups_needed : servings * cups_per_serving = 36.0 :=
by
  sorry

end total_cups_needed_l240_240747


namespace find_k_for_given_prime_l240_240740

theorem find_k_for_given_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (k : ℕ) 
  (h : ∃ a : ℕ, k^2 - p * k = a^2) : 
  k = (p + 1)^2 / 4 :=
sorry

end find_k_for_given_prime_l240_240740


namespace circle_diameter_tangents_l240_240100

open Real

theorem circle_diameter_tangents {x y : ℝ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) :
  ∃ d : ℝ, d = sqrt (x * y) :=
by
  sorry

end circle_diameter_tangents_l240_240100


namespace total_population_l240_240035

variables (b g t : ℕ)

-- Conditions
def cond1 := b = 4 * g
def cond2 := g = 2 * t

-- Theorem statement
theorem total_population (h1 : cond1 b g) (h2 : cond2 g t) : b + g + t = 11 * b / 8 :=
by sorry

end total_population_l240_240035


namespace total_guppies_correct_l240_240375

-- Define the initial conditions as variables
def initial_guppies : ℕ := 7
def baby_guppies_1 : ℕ := 3 * 12
def baby_guppies_2 : ℕ := 9

-- Define the total number of guppies
def total_guppies : ℕ := initial_guppies + baby_guppies_1 + baby_guppies_2

-- Theorem: Proving the total number of guppies is 52
theorem total_guppies_correct : total_guppies = 52 :=
by
  sorry

end total_guppies_correct_l240_240375


namespace negation_of_proposition_l240_240799

-- Definitions from the problem conditions
def proposition (x : ℝ) := ∃ x < 1, x^2 ≤ 1

-- Reformulated proof problem
theorem negation_of_proposition : 
  ¬ (∃ x < 1, x^2 ≤ 1) ↔ ∀ x < 1, x^2 > 1 :=
by
  sorry

end negation_of_proposition_l240_240799


namespace milk_leftover_l240_240390

def milk (milkshake_num : ℕ) := 4 * milkshake_num
def ice_cream (milkshake_num : ℕ) := 12 * milkshake_num
def possible_milkshakes (ice_cream_amount : ℕ) := ice_cream_amount / 12

theorem milk_leftover (total_milk total_ice_cream : ℕ) (h1 : total_milk = 72) (h2 : total_ice_cream = 192) :
  total_milk - milk (possible_milkshakes total_ice_cream) = 8 :=
by
  sorry

end milk_leftover_l240_240390


namespace sin_cos_tan_l240_240829

theorem sin_cos_tan (α : ℝ) (h1 : Real.tan α = 3) : Real.sin α * Real.cos α = 3 / 10 := 
sorry

end sin_cos_tan_l240_240829


namespace no_solution_eqn_l240_240277

theorem no_solution_eqn (m : ℝ) :
  ¬ ∃ x : ℝ, (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) = -1 ↔ m = 1 :=
by
  sorry

end no_solution_eqn_l240_240277


namespace initial_peanuts_count_l240_240973

def peanuts_initial (P : ℕ) : Prop :=
  P - (1 / 4 : ℝ) * P - 29 = 82

theorem initial_peanuts_count (P : ℕ) (h : peanuts_initial P) : P = 148 :=
by
  -- The complete proof can be constructed here.
  sorry

end initial_peanuts_count_l240_240973


namespace eggs_from_Martha_is_2_l240_240800

def eggs_from_Gertrude : ℕ := 4
def eggs_from_Blanche : ℕ := 3
def eggs_from_Nancy : ℕ := 2
def total_eggs_left : ℕ := 9
def eggs_dropped : ℕ := 2

def total_eggs_before_dropping (eggs_from_Martha : ℕ) :=
  eggs_from_Gertrude + eggs_from_Blanche + eggs_from_Nancy + eggs_from_Martha - eggs_dropped = total_eggs_left

-- The theorem stating the eggs collected from Martha.
theorem eggs_from_Martha_is_2 : ∃ (m : ℕ), total_eggs_before_dropping m ∧ m = 2 :=
by
  use 2
  sorry

end eggs_from_Martha_is_2_l240_240800


namespace intersection_points_and_verification_l240_240905

theorem intersection_points_and_verification :
  (∃ x y : ℝ, y = -3 * x ∧ y + 3 = 9 * x ∧ x = 1 / 4 ∧ y = -3 / 4) ∧
  ¬ (y = 2 * (1 / 4) - 1 ∧ (2 * (1 / 4) - 1 = -3 / 4)) :=
by
  sorry

end intersection_points_and_verification_l240_240905


namespace monthly_rent_of_shop_l240_240265

theorem monthly_rent_of_shop
  (length width : ℕ) (rent_per_sqft : ℕ)
  (h_length : length = 20) (h_width : width = 18) (h_rent : rent_per_sqft = 48) :
  (length * width * rent_per_sqft) / 12 = 1440 := 
by
  sorry

end monthly_rent_of_shop_l240_240265


namespace ratio_arms_martians_to_aliens_l240_240835

def arms_of_aliens : ℕ := 3
def legs_of_aliens : ℕ := 8
def legs_of_martians := legs_of_aliens / 2

def limbs_of_5_aliens := 5 * (arms_of_aliens + legs_of_aliens)
def limbs_of_5_martians (arms_of_martians : ℕ) := 5 * (arms_of_martians + legs_of_martians)

theorem ratio_arms_martians_to_aliens (A_m : ℕ) (h1 : limbs_of_5_aliens = limbs_of_5_martians A_m + 5) :
  (A_m : ℚ) / arms_of_aliens = 2 :=
sorry

end ratio_arms_martians_to_aliens_l240_240835


namespace max_n_for_factorable_polynomial_l240_240551

theorem max_n_for_factorable_polynomial : 
  ∃ n : ℤ, (∀ A B : ℤ, AB = 108 → n = 6 * B + A) ∧ n = 649 :=
by
  sorry

end max_n_for_factorable_polynomial_l240_240551


namespace non_zero_real_x_solution_l240_240102

noncomputable section

variables {x : ℝ} (hx : x ≠ 0)

theorem non_zero_real_x_solution 
  (h : (3 * x)^5 = (9 * x)^4) : 
  x = 27 := by
  sorry

end non_zero_real_x_solution_l240_240102


namespace amplitude_of_cosine_function_is_3_l240_240713

variable (a b : ℝ)
variable (h_a : a > 0)
variable (h_b : b > 0)
variable (h_max : ∀ x : ℝ, a * Real.cos (b * x) ≤ 3)
variable (h_cycle : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (∀ x : ℝ, a * Real.cos (b * (x + 2 * Real.pi)) = a * Real.cos (b * x)))

theorem amplitude_of_cosine_function_is_3 :
  a = 3 :=
sorry

end amplitude_of_cosine_function_is_3_l240_240713


namespace speed_of_truck_l240_240143

theorem speed_of_truck
  (v : ℝ)                         -- Let \( v \) be the speed of the truck.
  (car_speed : ℝ := 55)           -- Car speed is 55 mph.
  (start_delay : ℝ := 1)          -- Truck starts 1 hour later.
  (catchup_time : ℝ := 6.5)       -- Truck takes 6.5 hours to pass the car.
  (additional_distance_car : ℝ := car_speed * catchup_time)  -- Additional distance covered by the car in 6.5 hours.
  (total_distance_truck : ℝ := car_speed * start_delay + additional_distance_car)  -- Total distance truck must cover to pass the car.
  (truck_distance_eq : v * catchup_time = total_distance_truck)  -- Distance equation for the truck.
  : v = 63.46 :=                -- Prove the truck's speed is 63.46 mph.
by
  -- Original problem solution confirms truck's speed as 63.46 mph. 
  sorry

end speed_of_truck_l240_240143


namespace sum_of_prime_factors_eq_22_l240_240807

-- Conditions: n is defined as 3^6 - 1
def n : ℕ := 3^6 - 1

-- Statement: The sum of the prime factors of n is 22
theorem sum_of_prime_factors_eq_22 : 
  (∀ p : ℕ, p ∣ n → Prime p → p = 2 ∨ p = 7 ∨ p = 13) → 
  (2 + 7 + 13 = 22) :=
by sorry

end sum_of_prime_factors_eq_22_l240_240807


namespace transformation_result_l240_240267

theorem transformation_result (a b : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (a, b))
  (h2 : ∃ Q : ℝ × ℝ, Q = (b, a))
  (h3 : ∃ R : ℝ × ℝ, R = (2 - b, 10 - a))
  (h4 : (2 - b, 10 - a) = (-8, 2)) : 
  a - b = -2 := 
by 
  sorry

end transformation_result_l240_240267


namespace z_gets_amount_per_unit_l240_240642

-- Define the known conditions
variables (x y z : ℝ)
variables (x_share : ℝ)
variables (y_share : ℝ)
variables (z_share : ℝ)
variables (total : ℝ)

-- Assume the conditions given in the problem
axiom h1 : y_share = 54
axiom h2 : total = 234
axiom h3 : (y / x) = 0.45
axiom h4 : total = x_share + y_share + z_share

-- Prove the target statement
theorem z_gets_amount_per_unit : ((z_share / x_share) = 0.50) :=
by
  sorry

end z_gets_amount_per_unit_l240_240642


namespace find_xyz_l240_240889

theorem find_xyz
  (a b c x y z : ℂ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : a = (2 * b + 3 * c) / (x - 3))
  (h2 : b = (3 * a + 2 * c) / (y - 3))
  (h3 : c = (2 * a + 2 * b) / (z - 3))
  (h4 : x * y + x * z + y * z = -1)
  (h5 : x + y + z = 1) :
  x * y * z = 1 :=
sorry

end find_xyz_l240_240889


namespace no_non_similar_triangles_with_geometric_angles_l240_240527

theorem no_non_similar_triangles_with_geometric_angles :
  ¬∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ a ≠ a * r ∧ a ≠ a * r * r ∧ a * r ≠ a * r * r ∧
  a + a * r + a * r * r = 180 :=
by
  sorry

end no_non_similar_triangles_with_geometric_angles_l240_240527


namespace theo_selling_price_l240_240472

theorem theo_selling_price:
  ∀ (maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell: ℕ),
    maddox_price = 20 → 
    theo_cost = 20 → 
    maddox_sell = 28 →
    maddox_profit = (maddox_sell - maddox_price) * 3 →
    (theo_sell - theo_cost) * 3 = (maddox_profit - 15) →
    theo_sell = 23 := by
  intros maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell
  intros maddox_price_eq theo_cost_eq maddox_sell_eq maddox_profit_eq theo_profit_eq

  -- Use given assumptions
  rw [maddox_price_eq, theo_cost_eq, maddox_sell_eq] at *
  simp at *

  -- Final goal
  sorry

end theo_selling_price_l240_240472


namespace solve_for_y_l240_240943

theorem solve_for_y (y : ℝ) (h : y + 49 / y = 14) : y = 7 :=
sorry

end solve_for_y_l240_240943


namespace find_some_number_l240_240238

theorem find_some_number :
  ∃ (some_number : ℝ), (0.0077 * 3.6) / (some_number * 0.1 * 0.007) = 990.0000000000001 ∧ some_number = 0.04 :=
  sorry

end find_some_number_l240_240238


namespace right_triangle_perimeter_l240_240279

theorem right_triangle_perimeter
  (a b c : ℝ)
  (h_right: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c) :
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
sorry

end right_triangle_perimeter_l240_240279


namespace sarah_wide_reflections_l240_240318

variables (tall_mirrors_sarah : ℕ) (tall_mirrors_ellie : ℕ) 
          (wide_mirrors_ellie : ℕ) (tall_count : ℕ) (wide_count : ℕ)
          (total_reflections : ℕ) (S : ℕ)

def reflections_in_tall_mirrors_sarah := 10 * tall_count
def reflections_in_tall_mirrors_ellie := 6 * tall_count
def reflections_in_wide_mirrors_ellie := 3 * wide_count
def total_reflections_no_wide_sarah := reflections_in_tall_mirrors_sarah + reflections_in_tall_mirrors_ellie + reflections_in_wide_mirrors_ellie

theorem sarah_wide_reflections :
  reflections_in_tall_mirrors_sarah = 30 →
  reflections_in_tall_mirrors_ellie = 18 →
  reflections_in_wide_mirrors_ellie = 15 →
  tall_count = 3 →
  wide_count = 5 →
  total_reflections = 88 →
  total_reflections = total_reflections_no_wide_sarah + 5 * S →
  S = 5 :=
sorry

end sarah_wide_reflections_l240_240318


namespace speed_of_man_in_still_water_l240_240312

-- Define the parameters and conditions
def speed_in_still_water (v_m : ℝ) (v_s : ℝ) : Prop :=
    (v_m + v_s = 5) ∧  -- downstream condition
    (v_m - v_s = 7)    -- upstream condition

-- The theorem statement
theorem speed_of_man_in_still_water : 
  ∃ v_m v_s : ℝ, speed_in_still_water v_m v_s ∧ v_m = 6 := 
by
  sorry

end speed_of_man_in_still_water_l240_240312


namespace units_digit_of_3_pow_y_l240_240710

theorem units_digit_of_3_pow_y
    (x : ℕ)
    (h1 : (2^3)^x = 4096)
    (y : ℕ)
    (h2 : y = x^3) :
    (3^y) % 10 = 1 :=
by
  sorry

end units_digit_of_3_pow_y_l240_240710


namespace sum_of_coefficients_l240_240142

-- Given polynomial
def polynomial (x : ℝ) : ℝ := (3 * x - 1) ^ 7

-- Statement
theorem sum_of_coefficients :
  (polynomial 1) = 128 := 
sorry

end sum_of_coefficients_l240_240142


namespace find_b_value_l240_240581

def perfect_square_trinomial (a b c : ℕ) : Prop :=
  ∃ d, a = d^2 ∧ c = d^2 ∧ b = 2 * d * d

theorem find_b_value (b : ℝ) :
    (∀ x : ℝ, 16 * x^2 - b * x + 9 = (4 * x - 3) * (4 * x - 3) ∨ 16 * x^2 - b * x + 9 = (4 * x + 3) * (4 * x + 3)) -> 
    b = 24 ∨ b = -24 := 
by
  sorry

end find_b_value_l240_240581


namespace max_nested_fraction_value_l240_240641

-- Define the problem conditions
def numbers := (List.range 100).map (λ n => n + 1)

-- Define the nested fraction function
noncomputable def nested_fraction (l : List ℕ) : ℚ :=
  l.foldr (λ x acc => x / acc) 1

-- Prove that the maximum value of the nested fraction from 1 to 100 is 100! / 4
theorem max_nested_fraction_value :
  nested_fraction numbers = (Nat.factorial 100) / 4 :=
sorry

end max_nested_fraction_value_l240_240641


namespace initial_average_weight_l240_240693

theorem initial_average_weight
  (A : ℝ)
  (h : 30 * 27.4 - 10 = 29 * A) : 
  A = 28 := 
by
  sorry

end initial_average_weight_l240_240693


namespace domain_of_f_l240_240506

noncomputable def f (x : ℝ) : ℝ := (Real.log (x^2 - 1)) / (Real.sqrt (x^2 - x - 2))

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0 ∧ x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end domain_of_f_l240_240506


namespace gcd_ab_a2b2_l240_240117

theorem gcd_ab_a2b2 (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_l240_240117


namespace frac_sum_diff_l240_240867

theorem frac_sum_diff (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : (a + b) / (a - b) = -1001 :=
sorry

end frac_sum_diff_l240_240867


namespace angle_between_vectors_is_90_degrees_l240_240041

noncomputable def vec_angle (v₁ v₂ : ℝ × ℝ) : ℝ :=
sorry -- This would be the implementation that calculates the angle between two vectors

theorem angle_between_vectors_is_90_degrees
  (A B C O : ℝ × ℝ)
  (h1 : dist O A = dist O B)
  (h2 : dist O A = dist O C)
  (h3 : dist O B = dist O C)
  (h4 : 2 • (A - O) = (B - O) + (C - O)) :
  vec_angle (B - A) (C - A) = 90 :=
sorry

end angle_between_vectors_is_90_degrees_l240_240041


namespace original_commercial_length_l240_240640

theorem original_commercial_length (x : ℝ) (h : 0.70 * x = 21) : x = 30 := sorry

end original_commercial_length_l240_240640


namespace tan_alpha_ratio_expression_l240_240949

variable (α : Real)
variable (h1 : Real.sin α = 3/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : Real.tan α = -3/4 := sorry

theorem ratio_expression {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : (2 * Real.sin α + 3 * Real.cos α) / (Real.cos α - Real.sin α) = 6/7 := sorry

end tan_alpha_ratio_expression_l240_240949


namespace unique_prime_sum_8_l240_240842
-- Import all necessary mathematical libraries

-- Prime number definition
def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Function definition for f(y), number of unique ways to sum primes to form y
def f (y : Nat) : Nat :=
  if y = 8 then 2 else sorry -- We're assuming the correct answer to state the theorem; in a real proof, we would define this correctly.

theorem unique_prime_sum_8 :
  f 8 = 2 :=
by
  -- The proof goes here, but for now, we leave it as a placeholder.
  sorry

end unique_prime_sum_8_l240_240842


namespace problem_ab_cd_eq_l240_240048

theorem problem_ab_cd_eq (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 14) :
  ab + cd = 45 := 
by
  sorry

end problem_ab_cd_eq_l240_240048


namespace width_of_room_l240_240712

theorem width_of_room (length : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (width : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_rate = 800)
  (h3 : total_cost = 16500)
  (h4 : width = total_cost / cost_rate / length) : width = 3.75 :=
by
  sorry

end width_of_room_l240_240712


namespace sean_bought_3_sodas_l240_240239

def soda_cost (S : ℕ) : ℕ := S * 1
def soup_cost (S : ℕ) (C : ℕ) : Prop := C = S
def sandwich_cost (C : ℕ) (X : ℕ) : Prop := X = 3 * C
def total_cost (S C X : ℕ) : Prop := S + 2 * C + X = 18

theorem sean_bought_3_sodas (S C X : ℕ) (h1 : soup_cost S C) (h2 : sandwich_cost C X) (h3 : total_cost S C X) : S = 3 :=
by
  sorry

end sean_bought_3_sodas_l240_240239


namespace xyz_value_l240_240623

theorem xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 30) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : x * y * z = 7 :=
by
  sorry

end xyz_value_l240_240623


namespace second_smallest_palindromic_prime_l240_240222

-- Three digit number definition
def three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Palindromic number definition
def is_palindromic (n : ℕ) : Prop := 
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds = ones 

-- Prime number definition
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Second-smallest three-digit palindromic prime
theorem second_smallest_palindromic_prime :
  ∃ n : ℕ, three_digit_number n ∧ is_palindromic n ∧ is_prime n ∧ 
  ∃ m : ℕ, three_digit_number m ∧ is_palindromic m ∧ is_prime m ∧ m > 101 ∧ m < n ∧ 
  n = 131 := 
by
  sorry

end second_smallest_palindromic_prime_l240_240222


namespace infinite_solutions_a_l240_240987

theorem infinite_solutions_a (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 :=
by
  sorry

end infinite_solutions_a_l240_240987


namespace Nicki_runs_30_miles_per_week_in_second_half_l240_240259

/-
  Nicki ran 20 miles per week for the first half of the year.
  There are 26 weeks in each half of the year.
  She ran a total of 1300 miles for the year.
  Prove that Nicki ran 30 miles per week in the second half of the year.
-/

theorem Nicki_runs_30_miles_per_week_in_second_half (weekly_first_half : ℕ) (weeks_per_half : ℕ) (total_miles : ℕ) :
  weekly_first_half = 20 → weeks_per_half = 26 → total_miles = 1300 → 
  (total_miles - (weekly_first_half * weeks_per_half)) / weeks_per_half = 30 :=
by
  intros h1 h2 h3
  sorry

end Nicki_runs_30_miles_per_week_in_second_half_l240_240259


namespace makeup_exam_probability_l240_240659

theorem makeup_exam_probability (total_students : ℕ) (students_in_makeup_exam : ℕ)
  (h1 : total_students = 42) (h2 : students_in_makeup_exam = 3) :
  (students_in_makeup_exam : ℚ) / total_students = 1 / 14 := by
  sorry

end makeup_exam_probability_l240_240659


namespace determine_true_proposition_l240_240023

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x > 1

def proposition_q : Prop :=
  let focus_distance := 3/4 -- Distance from the focus to the directrix in y = (1/3)x^2
  focus_distance = 1/6

def true_proposition : Prop :=
  proposition_p ∧ ¬proposition_q

theorem determine_true_proposition :
  (proposition_p ∧ ¬proposition_q) = true_proposition :=
by
  sorry -- Proof will go here

end determine_true_proposition_l240_240023


namespace cost_of_dozen_pens_l240_240822

variable (x : ℝ) (pen_cost pencil_cost : ℝ)
variable (h1 : 3 * pen_cost + 5 * pencil_cost = 260)
variable (h2 : pen_cost / pencil_cost = 5)

theorem cost_of_dozen_pens (x_pos : 0 < x) 
    (pen_cost_def : pen_cost = 5 * x) 
    (pencil_cost_def : pencil_cost = x) :
    12 * pen_cost = 780 := by
  sorry

end cost_of_dozen_pens_l240_240822


namespace max_distance_on_curve_and_ellipse_l240_240233

noncomputable def max_distance_between_P_and_Q : ℝ :=
  6 * Real.sqrt 2

theorem max_distance_on_curve_and_ellipse :
  ∃ P Q, (P ∈ { p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2 }) ∧ 
         (Q ∈ { q : ℝ × ℝ | q.1^2 / 10 + q.2^2 = 1 }) ∧ 
         (dist P Q = max_distance_between_P_and_Q) := 
sorry

end max_distance_on_curve_and_ellipse_l240_240233


namespace negative_real_root_range_l240_240701

theorem negative_real_root_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (1 / Real.pi) ^ x = (1 + a) / (1 - a)) ↔ 0 < a ∧ a < 1 :=
by
  sorry

end negative_real_root_range_l240_240701


namespace pentagon_PT_value_l240_240078

-- Given conditions
def length_QR := 3
def length_RS := 3
def length_ST := 3
def angle_T := 90
def angle_P := 120
def angle_Q := 120
def angle_R := 120

-- The target statement to prove
theorem pentagon_PT_value (a b : ℝ) (h : PT = a + 3 * Real.sqrt b) : a + b = 6 :=
sorry

end pentagon_PT_value_l240_240078


namespace number_of_girls_at_camp_l240_240213

theorem number_of_girls_at_camp (total_people : ℕ) (difference_boys_girls : ℕ) (nb_girls : ℕ) :
  total_people = 133 ∧ difference_boys_girls = 33 ∧ 2 * nb_girls + 33 = total_people → nb_girls = 50 := 
by
  intros
  sorry

end number_of_girls_at_camp_l240_240213


namespace distance_between_pathway_lines_is_5_l240_240260

-- Define the conditions
def parallel_lines_distance (distance : ℤ) : Prop :=
  distance = 30

def pathway_length_between_lines (length : ℤ) : Prop :=
  length = 10

def pathway_line_length (length : ℤ) : Prop :=
  length = 60

-- Main proof problem
theorem distance_between_pathway_lines_is_5:
  ∀ (d : ℤ), parallel_lines_distance 30 → 
  pathway_length_between_lines 10 → 
  pathway_line_length 60 → 
  d = 5 := 
by
  sorry

end distance_between_pathway_lines_is_5_l240_240260


namespace intersect_at_two_points_l240_240629

theorem intersect_at_two_points (a : ℝ) :
  (∃ p q : ℝ × ℝ, 
    (p.1 - p.2 + 1 = 0) ∧ (2 * p.1 + p.2 - 4 = 0) ∧ (a * p.1 - p.2 + 2 = 0) ∧
    (q.1 - q.2 + 1 = 0) ∧ (2 * q.1 + q.2 - 4 = 0) ∧ (a * q.1 - q.2 + 2 = 0) ∧ p ≠ q) →
  (a = 1 ∨ a = -2) :=
by 
  sorry

end intersect_at_two_points_l240_240629


namespace find_n_l240_240777

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (1 + n)

theorem find_n (k : ℕ) (h : k = 3) (hn : ∃ k, n = k^2)
  (hs : sum_first_n_even_numbers n = 90) : n = 9 :=
by
  sorry

end find_n_l240_240777


namespace arithmetic_sequence_a4_l240_240175

theorem arithmetic_sequence_a4 (a : ℕ → ℤ) (a2 a4 a3 : ℤ) (S5 : ℤ)
  (h₁ : S5 = 25)
  (h₂ : a 2 = 3)
  (h₃ : S5 = a 1 + a 2 + a 3 + a 4 + a 5)
  (h₄ : a 3 = (a 1 + a 5) / 2)
  (h₅ : ∀ n : ℕ, (a (n+1) - a n) = (a 2 - a 1)) :
  a 4 = 7 := by
  sorry

end arithmetic_sequence_a4_l240_240175


namespace median_length_YN_perimeter_triangle_XYZ_l240_240229

-- Definitions for conditions
noncomputable def length_XY : ℝ := 5
noncomputable def length_XZ : ℝ := 12
noncomputable def is_right_angle_XYZ : Prop := true
noncomputable def midpoint_N : ℝ := length_XZ / 2

-- Theorem statement for the length of the median YN
theorem median_length_YN (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (13 / 2) = 6.5 := by
  sorry

-- Theorem statement for the perimeter of triangle XYZ
theorem perimeter_triangle_XYZ (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (XY + XZ + 13) = 30 := by
  sorry

end median_length_YN_perimeter_triangle_XYZ_l240_240229


namespace g_inv_f_five_l240_240234

-- Declare the existence of functions f and g and their inverses
variables (f g : ℝ → ℝ)

-- Given condition from the problem
axiom inv_cond : ∀ x, f⁻¹ (g x) = 4 * x - 1

-- Define the specific problem to solve
theorem g_inv_f_five : g⁻¹ (f 5) = 3 / 2 :=
by
  sorry

end g_inv_f_five_l240_240234


namespace smallest_positive_debt_pigs_goats_l240_240753

theorem smallest_positive_debt_pigs_goats :
  ∃ p g : ℤ, 350 * p + 240 * g = 10 :=
by
  sorry

end smallest_positive_debt_pigs_goats_l240_240753


namespace initial_kids_l240_240908

theorem initial_kids {N : ℕ} (h1 : 1 / 2 * N = N / 2) (h2 : 1 / 2 * (N / 2) = N / 4) (h3 : N / 4 = 5) : N = 20 :=
by
  sorry

end initial_kids_l240_240908


namespace integer_conditions_satisfy_eq_l240_240479

theorem integer_conditions_satisfy_eq (
  a b c : ℤ 
) : (a > b ∧ b = c → (a * (a - b) + b * (b - c) + c * (c - a) = 2)) ∧
    (¬(a = b - 1 ∧ b = c - 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c + 1 ∧ b = a + 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c ∧ b - 2 = c) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a + b + c = 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) :=
by
sorry

end integer_conditions_satisfy_eq_l240_240479


namespace average_marks_two_classes_correct_l240_240030

axiom average_marks_first_class : ℕ → ℕ → ℕ
axiom average_marks_second_class : ℕ → ℕ → ℕ
axiom combined_average_marks_correct : ℕ → ℕ → Prop

theorem average_marks_two_classes_correct :
  average_marks_first_class 39 45 = 39 * 45 →
  average_marks_second_class 35 70 = 35 * 70 →
  combined_average_marks_correct (average_marks_first_class 39 45) (average_marks_second_class 35 70) :=
by
  intros h1 h2
  sorry

end average_marks_two_classes_correct_l240_240030


namespace required_words_to_learn_l240_240212

def total_words : ℕ := 500
def required_percentage : ℕ := 85

theorem required_words_to_learn (x : ℕ) :
  (x : ℚ) / total_words ≥ (required_percentage : ℚ) / 100 ↔ x ≥ 425 := 
sorry

end required_words_to_learn_l240_240212


namespace exists_n_divides_2022n_minus_n_l240_240137

theorem exists_n_divides_2022n_minus_n (p : ℕ) [hp : Fact (Nat.Prime p)] :
  ∃ n : ℕ, p ∣ (2022^n - n) :=
sorry

end exists_n_divides_2022n_minus_n_l240_240137


namespace probability_yellow_ball_l240_240047

-- Definitions of the conditions
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3
def total_balls : ℕ := white_balls + yellow_balls

-- Theorem statement
theorem probability_yellow_ball : (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by
  -- Using tactics to facilitate the proof
  simp [yellow_balls, total_balls]
  sorry

end probability_yellow_ball_l240_240047


namespace determine_function_l240_240571

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (1/2) * (x^2 + (1/x)) else 0

theorem determine_function (f: ℝ → ℝ) (h : ∀ x ≠ 0, (1/x) * f (-x) + f (1/x) = x ) :
  ∀ x ≠ 0, f x = (1/2) * (x^2 + (1/x)) :=
by
  sorry

end determine_function_l240_240571


namespace students_still_in_school_l240_240643

def total_students := 5000
def students_to_beach := total_students / 2
def remaining_after_beach := total_students - students_to_beach
def students_to_art_museum := remaining_after_beach / 3
def remaining_after_art_museum := remaining_after_beach - students_to_art_museum
def students_to_science_fair := remaining_after_art_museum / 4
def remaining_after_science_fair := remaining_after_art_museum - students_to_science_fair
def students_to_music_workshop := 200
def remaining_students := remaining_after_science_fair - students_to_music_workshop

theorem students_still_in_school : remaining_students = 1051 := by
  sorry

end students_still_in_school_l240_240643


namespace first_nonzero_digit_one_div_139_l240_240543

theorem first_nonzero_digit_one_div_139 :
  ∀ n : ℕ, (n > 0 → (∀ m : ℕ, (m > 0 → (m * 10^n) ∣ (10^n * 1 - 1) ∧ n ∣ (139 * 10 ^ (n + 1)) ∧ 10^(n+1 - 1) * 1 - 1 < 10^n))) :=
sorry

end first_nonzero_digit_one_div_139_l240_240543


namespace combined_distance_l240_240913

theorem combined_distance (second_lady_distance : ℕ) (first_lady_distance : ℕ) 
  (h1 : second_lady_distance = 4) 
  (h2 : first_lady_distance = 2 * second_lady_distance) : 
  first_lady_distance + second_lady_distance = 12 :=
by 
  sorry

end combined_distance_l240_240913


namespace solve_for_x_l240_240971

theorem solve_for_x (a r s x : ℝ) (h1 : s > r) (h2 : r * (x + a) = s * (x - a)) :
  x = a * (s + r) / (s - r) :=
sorry

end solve_for_x_l240_240971


namespace factorize_expression_l240_240705

theorem factorize_expression (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := 
  sorry

end factorize_expression_l240_240705


namespace n_congruence_mod_9_l240_240650

def n : ℕ := 2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666

theorem n_congruence_mod_9 : n % 9 = 4 :=
by
  sorry

end n_congruence_mod_9_l240_240650


namespace good_function_count_l240_240636

noncomputable def num_good_functions (n : ℕ) : ℕ :=
  if n < 2 then 0 else
    n * Nat.totient n

theorem good_function_count (n : ℕ) (h : n ≥ 2) :
  ∃ (f : ℤ → Fin (n + 1)), 
  (∀ k, 1 ≤ k ∧ k ≤ n - 1 → ∃ j, ∀ m, (f (m + j) : ℤ) ≡ (f (m + k) - f m : ℤ) [ZMOD (n + 1)]) → 
  num_good_functions n = n * Nat.totient n :=
sorry

end good_function_count_l240_240636


namespace max_integer_k_l240_240297

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x - 1)) / (x - 2)

theorem max_integer_k (x : ℝ) (k : ℕ) (hx : x > 2) :
  (∀ x, x > 2 → f x > (k : ℝ) / (x - 1)) ↔ k ≤ 3 :=
sorry

end max_integer_k_l240_240297


namespace union_M_N_eq_l240_240736

open Set

-- Define set M and set N according to the problem conditions
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

-- The theorem we need to prove
theorem union_M_N_eq : M ∪ N = {0, 1, 2, 4} :=
by
  -- Just assert the theorem without proving it
  sorry

end union_M_N_eq_l240_240736


namespace area_of_f2_equals_7_l240_240794

def f0 (x : ℝ) : ℝ := abs x
def f1 (x : ℝ) : ℝ := abs (f0 x - 1)
def f2 (x : ℝ) : ℝ := abs (f1 x - 2)

theorem area_of_f2_equals_7 : 
  (∫ x in (-3 : ℝ)..3, f2 x) = 7 :=
by
  sorry

end area_of_f2_equals_7_l240_240794


namespace scientific_notation_eq_l240_240485

-- Define the number 82,600,000
def num : ℝ := 82600000

-- Define the scientific notation representation
def sci_not : ℝ := 8.26 * 10^7

-- The theorem to prove that the number is equal to its scientific notation
theorem scientific_notation_eq : num = sci_not :=
by 
  sorry

end scientific_notation_eq_l240_240485


namespace percent_time_in_meetings_l240_240519

-- Define the conditions
def work_day_minutes : ℕ := 10 * 60  -- Total minutes in a 10-hour work day is 600 minutes
def first_meeting_minutes : ℕ := 60  -- The first meeting took 60 minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes  -- The second meeting took three times as long as the first meeting

-- Total time spent in meetings
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes  -- 60 + 180 = 240 minutes

-- The task is to prove that Makarla spent 40% of her work day in meetings.
theorem percent_time_in_meetings : (total_meeting_minutes / work_day_minutes : ℚ) * 100 = 40 := by
  sorry

end percent_time_in_meetings_l240_240519


namespace twenty_five_percent_greater_l240_240816

theorem twenty_five_percent_greater (x : ℕ) (h : x = (88 + (88 * 25) / 100)) : x = 110 :=
sorry

end twenty_five_percent_greater_l240_240816


namespace inequality_proof_l240_240200

variable (x y z : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)
variable (hz : 0 < z)

theorem inequality_proof :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
sorry

end inequality_proof_l240_240200


namespace proof_F_2_f_3_l240_240663

def f (a : ℕ) : ℕ := a ^ 2 - 1

def F (a : ℕ) (b : ℕ) : ℕ := 3 * b ^ 2 + 2 * a

theorem proof_F_2_f_3 : F 2 (f 3) = 196 := by
  have h1 : f 3 = 3 ^ 2 - 1 := rfl
  rw [h1]
  have h2 : 3 ^ 2 - 1 = 8 := by norm_num
  rw [h2]
  exact rfl

end proof_F_2_f_3_l240_240663


namespace find_p_of_five_l240_240366

-- Define the cubic polynomial and the conditions
def cubic_poly (p : ℝ → ℝ) :=
  ∀ x, ∃ a b c d, p x = a * x^3 + b * x^2 + c * x + d

def satisfies_conditions (p : ℝ → ℝ) :=
  p 1 = 1 ^ 2 ∧
  p 2 = 2 ^ 2 ∧
  p 3 = 3 ^ 2 ∧
  p 4 = 4 ^ 2

-- Theorem statement to be proved
theorem find_p_of_five (p : ℝ → ℝ) (hcubic : cubic_poly p) (hconditions : satisfies_conditions p) : p 5 = 25 :=
by
  sorry

end find_p_of_five_l240_240366


namespace sum_xy_l240_240331

theorem sum_xy (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 10) : x + y = 14 ∨ x + y = -2 :=
sorry

end sum_xy_l240_240331


namespace original_fraction_eq_2_5_l240_240426

theorem original_fraction_eq_2_5 (a b : ℤ) (h : (a + 4) * b = a * (b + 10)) : (a / b) = (2 / 5) := by
  sorry

end original_fraction_eq_2_5_l240_240426


namespace find_a_l240_240009

theorem find_a (x y a : ℝ) (hx_pos_even : x > 0 ∧ ∃ n : ℕ, x = 2 * n) (hx_le_y : x ≤ y) 
  (h_eq_zero : |3 * y - 18| + |a * x - y| = 0) : 
  a = 3 ∨ a = 3 / 2 ∨ a = 1 :=
sorry

end find_a_l240_240009


namespace find_science_books_l240_240172

theorem find_science_books
  (S : ℕ)
  (h1 : 2 * 3 + 3 * 2 + 3 * S = 30) :
  S = 6 :=
by
  sorry

end find_science_books_l240_240172


namespace expansion_eq_coeff_sum_l240_240811

theorem expansion_eq_coeff_sum (a : ℕ → ℤ) (m : ℤ) 
  (h : (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7)
  (h_coeff : a 4 = -35) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1 ∧ a 1 + a 3 + a 5 + a 7 = 26 := 
by 
  sorry

end expansion_eq_coeff_sum_l240_240811


namespace find_sets_l240_240995

theorem find_sets (A B : Set ℕ) :
  A ∩ B = {1, 2, 3} ∧ A ∪ B = {1, 2, 3, 4, 5} →
    (A = {1, 2, 3} ∧ B = {1, 2, 3, 4, 5}) ∨
    (A = {1, 2, 3, 4, 5} ∧ B = {1, 2, 3}) ∨
    (A = {1, 2, 3, 4} ∧ B = {1, 2, 3, 5}) ∨
    (A = {1, 2, 3, 5} ∧ B = {1, 2, 3, 4}) :=
by
  sorry

end find_sets_l240_240995


namespace cos_diff_l240_240658

theorem cos_diff (α : ℝ) (h1 : Real.cos α = (Real.sqrt 2) / 10) (h2 : α > -π ∧ α < 0) :
  Real.cos (α - π / 4) = -3 / 5 :=
sorry

end cos_diff_l240_240658


namespace ratio_pentagon_area_l240_240538

noncomputable def square_side_length := 1
noncomputable def square_area := (square_side_length : ℝ)^2
noncomputable def total_area := 3 * square_area
noncomputable def area_triangle (base height : ℝ) := 0.5 * base * height
noncomputable def GC := 2 / 3 * square_side_length
noncomputable def HD := 2 / 3 * square_side_length
noncomputable def area_GJC := area_triangle GC square_side_length
noncomputable def area_HDJ := area_triangle HD square_side_length
noncomputable def area_AJKCB := square_area - (area_GJC + area_HDJ)

theorem ratio_pentagon_area :
  (area_AJKCB / total_area) = 1 / 9 := 
sorry

end ratio_pentagon_area_l240_240538


namespace range_of_a_l240_240497

noncomputable def satisfiesInequality (a : ℝ) (x : ℝ) : Prop :=
  x > 1 → a * Real.log x > 1 - 1/x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 1 → satisfiesInequality a x) ↔ a ∈ Set.Ici 1 := 
sorry

end range_of_a_l240_240497


namespace infinitely_many_H_points_l240_240392

-- Define the curve C as (x^2 / 4) + y^2 = 1
def is_on_curve (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

-- Define point P on curve C
def is_H_point (P : ℝ × ℝ) : Prop :=
  is_on_curve P.1 P.2 ∧
  ∃ (A B : ℝ × ℝ), is_on_curve A.1 A.2 ∧ B.1 = 4 ∧
  (dist (P.1, P.2) (A.1, A.2) = dist (P.1, P.2) (B.1, B.2) ∨
   dist (P.1, P.2) (A.1, A.2) = dist (A.1, A.2) (B.1, B.2))

-- Theorem to prove the existence of infinitely many H points
theorem infinitely_many_H_points : ∃ (P : ℝ × ℝ), is_H_point P ∧ ∀ (Q : ℝ × ℝ), Q ≠ P → is_H_point Q :=
sorry


end infinitely_many_H_points_l240_240392


namespace ellipse_abs_sum_max_min_l240_240631

theorem ellipse_abs_sum_max_min (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) :
  2 ≤ |x| + |y| ∧ |x| + |y| ≤ 3 :=
sorry

end ellipse_abs_sum_max_min_l240_240631


namespace minimal_dominoes_needed_l240_240965

-- Variables representing the number of dominoes and tetraminoes
variables (d t : ℕ)

-- Definitions related to the problem
def area_rectangle : ℕ := 2008 * 2010 -- Total area of the rectangle
def area_domino : ℕ := 1 * 2 -- Area of a single domino
def area_tetramino : ℕ := 2 * 3 - 2 -- Area of a single tetramino
def total_area_covered : ℕ := 2 * d + 4 * t -- Total area covered by dominoes and tetraminoes

-- The theorem we want to prove
theorem minimal_dominoes_needed :
  total_area_covered d t = area_rectangle → d = 0 :=
sorry

end minimal_dominoes_needed_l240_240965


namespace ab_bc_ca_value_a4_b4_c4_value_l240_240827

theorem ab_bc_ca_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ab + bc + ca = -1/2 :=
sorry

theorem a4_b4_c4_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1/2 :=
sorry

end ab_bc_ca_value_a4_b4_c4_value_l240_240827


namespace hanoi_tower_l240_240274

noncomputable def move_all_disks (n : ℕ) : Prop := 
  ∀ (A B C : Type), 
  (∃ (move : A → B), move = sorry) ∧ -- Only one disk can be moved
  (∃ (can_place : A → A → Prop), can_place = sorry) -- A disk cannot be placed on top of a smaller disk 
  → ∃ (u_n : ℕ), u_n = 2^n - 1 -- Formula for minimum number of steps

theorem hanoi_tower : ∀ n : ℕ, move_all_disks n :=
by sorry

end hanoi_tower_l240_240274


namespace sin_triangle_sides_l240_240396

theorem sin_triangle_sides (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b + c ≤ 2 * Real.pi) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  ∃ x y z : ℝ, x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ z + x > y := 
by
  sorry

end sin_triangle_sides_l240_240396


namespace paintings_in_four_weeks_l240_240291

def weekly_hours := 30
def hours_per_painting := 3
def weeks := 4

theorem paintings_in_four_weeks (w_hours : ℕ) (h_per_painting : ℕ) (n_weeks : ℕ) (result : ℕ) :
  w_hours = weekly_hours →
  h_per_painting = hours_per_painting →
  n_weeks = weeks →
  result = (w_hours / h_per_painting) * n_weeks →
  result = 40 :=
by
  intros
  sorry

end paintings_in_four_weeks_l240_240291


namespace problem_l240_240082

def otimes (x y : ℝ) : ℝ := x^3 + 5 * x * y - y

theorem problem (a : ℝ) : 
  otimes a (otimes a a) = 5 * a^4 + 24 * a^3 - 10 * a^2 + a :=
by
  sorry

end problem_l240_240082


namespace ordered_pair_sol_l240_240954

noncomputable def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![5, d]]

noncomputable def is_inverse_scalar_mul (d k : ℝ) : Prop :=
  (A d)⁻¹ = k • (A d)

theorem ordered_pair_sol (d k : ℝ) :
  is_inverse_scalar_mul d k → (d = -2 ∧ k = 1 / 19) :=
by
  intros h
  sorry

end ordered_pair_sol_l240_240954


namespace floor_sum_eq_55_l240_240752

noncomputable def x : ℝ := 9.42

theorem floor_sum_eq_55 : ∀ (x : ℝ), x = 9.42 → (⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋) = 55 := by
  intros
  sorry

end floor_sum_eq_55_l240_240752


namespace recurring_decimal_sum_l240_240463

theorem recurring_decimal_sum (x y : ℚ) (hx : x = 4/9) (hy : y = 7/9) :
  x + y = 11/9 :=
by
  rw [hx, hy]
  exact sorry

end recurring_decimal_sum_l240_240463


namespace ten_times_product_is_2010_l240_240310

theorem ten_times_product_is_2010 (n : ℕ) (hn : 10 ≤ n ∧ n < 100) : 
  (∃ k : ℤ, 4.02 * (n : ℝ) = k) → (10 * k = 2010) :=
by
  sorry

end ten_times_product_is_2010_l240_240310


namespace tan_half_sum_l240_240191

variable (p q : ℝ)

-- Given conditions
def cos_condition : Prop := (Real.cos p + Real.cos q = 1 / 3)
def sin_condition : Prop := (Real.sin p + Real.sin q = 4 / 9)

-- Prove the target expression
theorem tan_half_sum (h1 : cos_condition p q) (h2 : sin_condition p q) : 
  Real.tan ((p + q) / 2) = 4 / 3 :=
sorry

-- For better readability, I included variable declarations and definitions separately

end tan_half_sum_l240_240191


namespace factory_material_equation_correct_l240_240878

variable (a b x : ℝ)
variable (h_a : a = 180)
variable (h_b : b = 120)
variable (h_condition : (a - 2 * x) - (b + x) = 30)

theorem factory_material_equation_correct : (180 - 2 * x) - (120 + x) = 30 := by
  rw [←h_a, ←h_b]
  exact h_condition

end factory_material_equation_correct_l240_240878


namespace problem_value_l240_240363

theorem problem_value :
  (1 / 3 * 9 * 1 / 27 * 81 * 1 / 243 * 729 * 1 / 2187 * 6561 * 1 / 19683 * 59049) = 243 := 
sorry

end problem_value_l240_240363


namespace problem_l240_240687

-- Definitions based on the provided conditions
def frequency_varies (freq : Real) : Prop := true -- Placeholder definition
def probability_is_stable (prob : Real) : Prop := true -- Placeholder definition
def is_random_event (event : Type) : Prop := true -- Placeholder definition
def is_random_experiment (experiment : Type) : Prop := true -- Placeholder definition
def is_sum_of_events (event1 event2 : Prop) : Prop := event1 ∨ event2 -- Definition of sum of events
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B) -- Definition of mutually exclusive events
def complementary_events (A B : Prop) : Prop := A ↔ ¬B -- Definition of complementary events
def equally_likely_events (events : List Prop) : Prop := true -- Placeholder definition

-- Translation of the questions and correct answers
theorem problem (freq prob : Real) (event experiment : Type) (A B : Prop) (events : List Prop) :
  (¬(frequency_varies freq = probability_is_stable prob)) ∧ -- 1
  ((is_random_event event) ≠ (is_random_experiment experiment)) ∧ -- 2
  (probability_is_stable prob) ∧ -- 3
  (is_sum_of_events A B) ∧ -- 4
  (mutually_exclusive A B → ¬(probability_is_stable (1 - prob))) ∧ -- 5
  (¬(equally_likely_events events)) :=  -- 6
by
  sorry

end problem_l240_240687


namespace expression_varies_l240_240532

noncomputable def expr (x : ℝ) : ℝ := (3 * x^2 - 2 * x - 5) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3))

theorem expression_varies (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) : ∃ y : ℝ, expr x = y ∧ ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → expr x₁ ≠ expr x₂ :=
by
  sorry

end expression_varies_l240_240532


namespace destroyed_cakes_l240_240397

theorem destroyed_cakes (initial_cakes : ℕ) (half_falls : ℕ) (half_saved : ℕ)
  (h1 : initial_cakes = 12)
  (h2 : half_falls = initial_cakes / 2)
  (h3 : half_saved = half_falls / 2) :
  initial_cakes - half_falls / 2 = 3 :=
by
  sorry

end destroyed_cakes_l240_240397


namespace mod_3_power_87_plus_5_l240_240682

theorem mod_3_power_87_plus_5 :
  (3 ^ 87 + 5) % 11 = 3 := 
by
  sorry

end mod_3_power_87_plus_5_l240_240682


namespace sandy_position_l240_240845

structure Position :=
  (x : ℤ)
  (y : ℤ)

def initial_position : Position := { x := 0, y := 0 }
def after_south : Position := { x := 0, y := -20 }
def after_east : Position := { x := 20, y := -20 }
def after_north : Position := { x := 20, y := 0 }
def final_position : Position := { x := 30, y := 0 }

theorem sandy_position :
  final_position.x - initial_position.x = 10 ∧ final_position.y - initial_position.y = 0 :=
by
  sorry

end sandy_position_l240_240845


namespace find_number_l240_240482

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end find_number_l240_240482


namespace increasing_function_when_a_eq_2_range_of_a_for_solution_set_l240_240825

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x - a * (x - 1) / (x + 1)

theorem increasing_function_when_a_eq_2 :
  ∀ ⦃x⦄, x > 0 → (f 2 x - f 2 1) * (x - 1) > 0 := sorry

theorem range_of_a_for_solution_set :
  ∀ ⦃a x⦄, f a x ≥ 0 ↔ (x ≥ 1) → a ≤ 1 := sorry

end increasing_function_when_a_eq_2_range_of_a_for_solution_set_l240_240825


namespace perfect_square_A_plus_2B_plus_4_l240_240824

theorem perfect_square_A_plus_2B_plus_4 (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9 : ℚ) * (10 ^ (2 * n) - 1)
  let B := (8 / 9 : ℚ) * (10 ^ n - 1)
  ∃ k : ℚ, A + 2 * B + 4 = k^2 := 
by {
  sorry
}

end perfect_square_A_plus_2B_plus_4_l240_240824


namespace compare_A_B_l240_240577

noncomputable def A (x : ℝ) := x / (x^2 - x + 1)
noncomputable def B (y : ℝ) := y / (y^2 - y + 1)

theorem compare_A_B (x y : ℝ) (hx : x > y) (hx_val : x = 2.00 * 10^1998 + 4) (hy_val : y = 2.00 * 10^1998 + 2) : 
  A x < B y := 
by 
  sorry

end compare_A_B_l240_240577


namespace length_of_second_train_is_correct_l240_240584

noncomputable def convert_kmph_to_mps (speed_kmph: ℕ) : ℝ :=
  speed_kmph * (1000 / 3600)

def train_lengths_and_time
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℕ)
  (speed_second_train_kmph : ℕ)
  (time_to_cross : ℝ)
  (length_second_train : ℝ) : Prop :=
  let speed_first_train_mps := convert_kmph_to_mps speed_first_train_kmph
  let speed_second_train_mps := convert_kmph_to_mps speed_second_train_kmph
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_to_cross
  total_distance = length_first_train + length_second_train

theorem length_of_second_train_is_correct :
  train_lengths_and_time 260 120 80 9 239.95 :=
by
  sorry

end length_of_second_train_is_correct_l240_240584


namespace cos_sq_plus_two_sin_double_l240_240024

theorem cos_sq_plus_two_sin_double (α : ℝ) (h : Real.tan α = 3 / 4) : Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_sq_plus_two_sin_double_l240_240024


namespace first_donor_amount_l240_240150

theorem first_donor_amount
  (x second third fourth : ℝ)
  (h1 : second = 2 * x)
  (h2 : third = 3 * second)
  (h3 : fourth = 4 * third)
  (h4 : x + second + third + fourth = 132)
  : x = 4 := 
by 
  -- Simply add this line to make the theorem complete without proof.
  sorry

end first_donor_amount_l240_240150


namespace student_question_choices_l240_240882

-- Definitions based on conditions
def partA_questions := 10
def partB_questions := 10
def choose_from_partA := 8
def choose_from_partB := 5

-- The proof problem statement
theorem student_question_choices :
  (Nat.choose partA_questions choose_from_partA) * (Nat.choose partB_questions choose_from_partB) = 11340 :=
by
  sorry

end student_question_choices_l240_240882


namespace tangent_line_equation_l240_240026

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := 2 * x^2 - x
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Statement of the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (b = 1 - 3 * 1) ∧ 
  (m = 3) ∧ 
  ∀ (x y : ℝ), y = m * x + b → 3 * x - y - 2 = 0 :=
by
  sorry

end tangent_line_equation_l240_240026


namespace total_dots_not_visible_proof_l240_240008

def total_dots_on_one_die : ℕ := 21

def total_dots_on_five_dice : ℕ := 5 * total_dots_on_one_die

def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

def sum_visible_numbers : ℕ := visible_numbers.sum

def total_dots_not_visible (total : ℕ) (visible_sum : ℕ) : ℕ :=
  total - visible_sum

theorem total_dots_not_visible_proof :
  total_dots_not_visible total_dots_on_five_dice sum_visible_numbers = 81 :=
by
  sorry

end total_dots_not_visible_proof_l240_240008


namespace height_of_brick_l240_240940

-- Definitions of given conditions
def length_brick : ℝ := 125
def width_brick : ℝ := 11.25
def length_wall : ℝ := 800
def height_wall : ℝ := 600
def width_wall : ℝ := 22.5
def number_bricks : ℝ := 1280

-- Prove that the height of each brick is 6.01 cm
theorem height_of_brick :
  ∃ H : ℝ,
    H = 6.01 ∧
    (number_bricks * (length_brick * width_brick * H) = length_wall * height_wall * width_wall) :=
by
  sorry

end height_of_brick_l240_240940


namespace find_A_l240_240287

noncomputable def f (A B x : ℝ) : ℝ := A * x - 3 * B ^ 2
def g (B x : ℝ) : ℝ := B * x
variable (B : ℝ) (hB : B ≠ 0)

theorem find_A (h : f (A := A) B (g B 2) = 0) : A = 3 * B / 2 := by
  sorry

end find_A_l240_240287


namespace count_four_digit_integers_with_conditions_l240_240804

def is_four_digit_integer (n : Nat) : Prop := 1000 ≤ n ∧ n < 10000

def thousands_digit_is_seven (n : Nat) : Prop := 
  (n / 1000) % 10 = 7

def hundreds_digit_is_odd (n : Nat) : Prop := 
  let hd := (n / 100) % 10
  hd % 2 = 1

theorem count_four_digit_integers_with_conditions : 
  (Nat.card {n : Nat // is_four_digit_integer n ∧ thousands_digit_is_seven n ∧ hundreds_digit_is_odd n}) = 500 :=
by
  sorry

end count_four_digit_integers_with_conditions_l240_240804


namespace no_positive_integer_solutions_l240_240902

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2017 - 1 ≠ (x - 1) * (y^2015 - 1) :=
by sorry

end no_positive_integer_solutions_l240_240902


namespace jamie_cherry_pies_l240_240354

theorem jamie_cherry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36) (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) : 
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := 
by {
  sorry
}

end jamie_cherry_pies_l240_240354


namespace length_OC_l240_240014

theorem length_OC (a b : ℝ) (h_perpendicular : ∀ x, x^2 + a * x + b = 0 → x = 1 ∨ x = b) : 
  1 = 1 :=
by 
  sorry

end length_OC_l240_240014


namespace find_m_l240_240487

section
variables {R : Type*} [CommRing R]

def f (x : R) : R := 4 * x^2 - 3 * x + 5
def g (x : R) (m : R) : R := x^2 - m * x - 8

theorem find_m (m : ℚ) : 
  f (5 : ℚ) - g (5 : ℚ) m = 20 → m = -53 / 5 :=
by {
  sorry
}

end

end find_m_l240_240487


namespace Cara_possible_pairs_l240_240108

-- Define the conditions and the final goal.
theorem Cara_possible_pairs : ∃ p : Nat, p = Nat.choose 7 2 ∧ p = 21 :=
by
  sorry

end Cara_possible_pairs_l240_240108


namespace rectangle_area_l240_240791

variables (y w : ℝ)

-- Definitions from conditions
def is_width_of_rectangle : Prop := w = y / Real.sqrt 10
def is_length_of_rectangle : Prop := 3 * w = y / Real.sqrt 10

-- Theorem to be proved
theorem rectangle_area (h1 : is_width_of_rectangle y w) (h2 : is_length_of_rectangle y w) : 
  3 * (w^2) = 3 * (y^2 / 10) :=
by sorry

end rectangle_area_l240_240791


namespace infinite_k_lcm_gt_ck_l240_240931

theorem infinite_k_lcm_gt_ck 
  (a : ℕ → ℕ) 
  (distinct_pos : ∀ n m : ℕ, n ≠ m → a n ≠ a m) 
  (pos : ∀ n, 0 < a n) 
  (c : ℝ) 
  (c_pos : 0 < c) 
  (c_lt : c < 1.5) : 
  ∃ᶠ k in at_top, (Nat.lcm (a k) (a (k + 1)) : ℝ) > c * k :=
sorry

end infinite_k_lcm_gt_ck_l240_240931


namespace no_integer_solutions_l240_240927

theorem no_integer_solutions
  (x y : ℤ) :
  3 * x^2 = 16 * y^2 + 8 * y + 5 → false :=
by
  sorry

end no_integer_solutions_l240_240927


namespace abcd_zero_l240_240075

theorem abcd_zero (a b c d : ℝ) (h1 : a + b + c + d = 0) (h2 : ab + ac + bc + bd + ad + cd = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end abcd_zero_l240_240075


namespace plane_hover_central_time_l240_240547

theorem plane_hover_central_time (x : ℕ) (h1 : 3 + x + 2 + 5 + (x + 2) + 4 = 24) : x = 4 := by
  sorry

end plane_hover_central_time_l240_240547


namespace circle_area_l240_240084

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l240_240084


namespace cannot_be_expressed_as_x_squared_plus_y_fifth_l240_240862

theorem cannot_be_expressed_as_x_squared_plus_y_fifth :
  ¬ ∃ x y : ℤ, 59121 = x^2 + y^5 :=
by sorry

end cannot_be_expressed_as_x_squared_plus_y_fifth_l240_240862


namespace A_inter_B_l240_240464

open Set Real

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { y | ∃ x, y = exp x }

theorem A_inter_B :
  A ∩ B = { z | 0 < z ∧ z < 3 } :=
by
  sorry

end A_inter_B_l240_240464


namespace power_function_value_at_half_l240_240691

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_value_at_half (a : ℝ) (α : ℝ) 
  (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1 / 4) (h4 : g α 2 = 1 / 4) : 
  g α (1/2) = 4 := 
by
  sorry

end power_function_value_at_half_l240_240691


namespace sum_of_solutions_of_quadratic_l240_240991

theorem sum_of_solutions_of_quadratic :
  ∀ a b c x₁ x₂ : ℝ, a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (∃ s : ℝ, s = x₁ + x₂ ∧ -b / a = s) :=
by
  sorry

end sum_of_solutions_of_quadratic_l240_240991


namespace profit_percentage_l240_240523

-- Define the selling price
def selling_price : ℝ := 900

-- Define the profit
def profit : ℝ := 100

-- Define the cost price as selling price minus profit
def cost_price : ℝ := selling_price - profit

-- Statement of the profit percentage calculation
theorem profit_percentage : (profit / cost_price) * 100 = 12.5 := by
  sorry

end profit_percentage_l240_240523


namespace exists_matrices_B_C_not_exists_matrices_commute_l240_240352

-- Equivalent proof statement for part (a)
theorem exists_matrices_B_C (A : Matrix (Fin 2) (Fin 2) ℝ): 
  ∃ (B C : Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 :=
by
  sorry

-- Equivalent proof statement for part (b)
theorem not_exists_matrices_commute (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = ![![0, 1], ![1, 0]]) :
  ¬∃ (B C: Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 ∧ B * C = C * B :=
by
  sorry

end exists_matrices_B_C_not_exists_matrices_commute_l240_240352


namespace expression_for_neg_x_l240_240645

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem expression_for_neg_x (f : ℝ → ℝ) (h_odd : odd_function f) (h_nonneg : ∀ (x : ℝ), 0 ≤ x → f x = x^2 - 2 * x) :
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2 * x :=
by 
  intros x hx 
  have hx_pos : -x > 0 := by linarith 
  have h_fx_neg : f (-x) = -f x := h_odd x
  rw [h_nonneg (-x) (by linarith)] at h_fx_neg
  linarith

end expression_for_neg_x_l240_240645


namespace solve_for_k_l240_240359

theorem solve_for_k (k : ℤ) : (∃ x : ℤ, x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 :=
by
  sorry

end solve_for_k_l240_240359


namespace convert_20121_base3_to_base10_l240_240427

/- Define the base conversion function for base 3 to base 10 -/
def base3_to_base10 (d4 d3 d2 d1 d0 : ℕ) :=
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0

/- Define the specific number in base 3 -/
def num20121_base3 := (2, 0, 1, 2, 1)

/- The theorem stating the equivalence of the base 3 number 20121_3 to its base 10 equivalent -/
theorem convert_20121_base3_to_base10 :
  base3_to_base10 2 0 1 2 1 = 178 :=
by
  sorry

end convert_20121_base3_to_base10_l240_240427


namespace solution_is_63_l240_240738

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)
def last_digit (n : ℕ) : ℕ := n % 10
def rhyming_primes_around (r : ℕ) : Prop :=
  r >= 1 ∧ r <= 100 ∧
  ¬ is_prime r ∧
  ∃ ps : List ℕ, (∀ p ∈ ps, is_prime p ∧ last_digit p = last_digit r) ∧
  (∀ q : ℕ, is_prime q ∧ last_digit q = last_digit r → q ∈ ps) ∧
  List.length ps = 4

theorem solution_is_63 : ∃ r : ℕ, rhyming_primes_around r ∧ r = 63 :=
by sorry

end solution_is_63_l240_240738


namespace triplet_A_sums_to_2_triplet_B_sums_to_2_triplet_C_sums_to_2_l240_240021

theorem triplet_A_sums_to_2 : (1/4 + 1/4 + 3/2 = 2) := by
  sorry

theorem triplet_B_sums_to_2 : (3 + -1 + 0 = 2) := by
  sorry

theorem triplet_C_sums_to_2 : (0.2 + 0.7 + 1.1 = 2) := by
  sorry

end triplet_A_sums_to_2_triplet_B_sums_to_2_triplet_C_sums_to_2_l240_240021


namespace european_confidence_95_european_teams_not_face_l240_240630

-- Definitions for the conditions
def european_teams_round_of_16 := 44
def european_teams_not_round_of_16 := 22
def other_regions_round_of_16 := 36
def other_regions_not_round_of_16 := 58
def total_teams := 160

-- Formula for K^2 calculation
def k_value : ℚ := 3.841
def k_squared (n a_d_diff b_c_diff a b c d : ℚ) : ℚ :=
  n * ((a_d_diff - b_c_diff)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Definitions and calculation of K^2
def n1 := (european_teams_round_of_16 + other_regions_round_of_16 : ℚ)
def a_d_diff1 := (european_teams_round_of_16 * other_regions_not_round_of_16 : ℚ)
def b_c_diff1 := (european_teams_not_round_of_16 * other_regions_round_of_16 : ℚ)
def k_squared_result := k_squared n1 a_d_diff1 b_c_diff1
                                 (european_teams_round_of_16 + european_teams_not_round_of_16)
                                 (other_regions_round_of_16 + other_regions_not_round_of_16)
                                 total_teams total_teams

-- Theorem for 95% confidence derived
theorem european_confidence_95 :
  k_squared_result > k_value := sorry

-- Probability calculation setup
def total_ways_to_pair_teams : ℚ := 15
def ways_european_teams_not_face : ℚ := 6
def probability_european_teams_not_face := ways_european_teams_not_face / total_ways_to_pair_teams

-- Theorem for probability
theorem european_teams_not_face :
  probability_european_teams_not_face = 2 / 5 := sorry

end european_confidence_95_european_teams_not_face_l240_240630


namespace sara_bought_cards_l240_240495

-- Definition of the given conditions
def initial_cards : ℕ := 39
def torn_cards : ℕ := 9
def remaining_cards_after_sale : ℕ := 15

-- Derived definition: Number of good cards before selling to Sara
def good_cards_before_selling : ℕ := initial_cards - torn_cards

-- The statement we need to prove
theorem sara_bought_cards : good_cards_before_selling - remaining_cards_after_sale = 15 :=
by
  sorry

end sara_bought_cards_l240_240495


namespace derivative_of_constant_function_l240_240090

-- Define the constant function
def f (x : ℝ) : ℝ := 0

-- State the theorem
theorem derivative_of_constant_function : deriv f 0 = 0 := by
  -- Proof will go here, but we use sorry to skip it
  sorry

end derivative_of_constant_function_l240_240090


namespace solution_set_of_inequality_l240_240038

theorem solution_set_of_inequality (x : ℝ) : x > 1 ∨ (-1 < x ∧ x < 0) ↔ x > 1 ∨ (-1 < x ∧ x < 0) :=
by sorry

end solution_set_of_inequality_l240_240038


namespace solve_for_d_l240_240451

variable (n c b d : ℚ)  -- Alternatively, specify the types if they are required to be specific
variable (H : n = d * c * b / (c - d))

theorem solve_for_d :
  d = n * c / (c * b + n) :=
by
  sorry

end solve_for_d_l240_240451


namespace number_of_paintings_per_new_gallery_l240_240767

-- Define all the conditions as variables/constants
def pictures_original : Nat := 9
def new_galleries : Nat := 5
def pencils_per_picture : Nat := 4
def pencils_per_exhibition : Nat := 2
def total_pencils : Nat := 88

-- Define the proof problem in Lean
theorem number_of_paintings_per_new_gallery (pictures_original new_galleries pencils_per_picture pencils_per_exhibition total_pencils : Nat) :
(pictures_original = 9) → (new_galleries = 5) → (pencils_per_picture = 4) → (pencils_per_exhibition = 2) → (total_pencils = 88) → 
∃ (pictures_per_gallery : Nat), pictures_per_gallery = 2 :=
by
  intros
  sorry

end number_of_paintings_per_new_gallery_l240_240767


namespace Karlee_initial_grapes_l240_240120

theorem Karlee_initial_grapes (G S Remaining_Fruits : ℕ)
  (h1 : S = (3 * G) / 5)
  (h2 : Remaining_Fruits = 96)
  (h3 : Remaining_Fruits = (3 * G) / 5 + (9 * G) / 25) :
  G = 100 := by
  -- add proof here
  sorry

end Karlee_initial_grapes_l240_240120


namespace initial_mean_of_observations_l240_240585

theorem initial_mean_of_observations (M : ℝ) (h1 : 50 * M + 30 = 50 * 40.66) : M = 40.06 := 
sorry

end initial_mean_of_observations_l240_240585


namespace relay_race_l240_240864

theorem relay_race (n : ℕ) (H1 : 2004 % n = 0) (H2 : n ≤ 168) (H3 : n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6 ∧ n ≠ 12): n = 167 :=
by
  sorry

end relay_race_l240_240864


namespace checkerboard_probability_l240_240770

def checkerboard_size : ℕ := 10

def total_squares (n : ℕ) : ℕ := n * n

def perimeter_squares (n : ℕ) : ℕ := 4 * n - 4

def inner_squares (n : ℕ) : ℕ := total_squares n - perimeter_squares n

def probability_not_touching_edge (n : ℕ) : ℚ := inner_squares n / total_squares n

theorem checkerboard_probability :
  probability_not_touching_edge checkerboard_size = 16 / 25 := by
  sorry

end checkerboard_probability_l240_240770


namespace total_chips_eaten_l240_240080

theorem total_chips_eaten (dinner_chips after_dinner_chips : ℕ) (h1 : dinner_chips = 1) (h2 : after_dinner_chips = 2 * dinner_chips) : dinner_chips + after_dinner_chips = 3 := by
  sorry

end total_chips_eaten_l240_240080


namespace mows_in_summer_l240_240662

theorem mows_in_summer (S : ℕ) (h1 : 8 - S = 3) : S = 5 :=
sorry

end mows_in_summer_l240_240662


namespace linear_function_max_value_l240_240728

theorem linear_function_max_value (m x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (y : ℝ) 
  (hl : y = m * x - 2 * m) (hy : y = 6) : m = -2 ∨ m = 6 := 
by 
  sorry

end linear_function_max_value_l240_240728


namespace female_democrats_count_l240_240999

theorem female_democrats_count 
  (F M D : ℕ)
  (total_participants : F + M = 660)
  (total_democrats : F / 2 + M / 4 = 660 / 3)
  (female_democrats : D = F / 2) : 
  D = 110 := 
by
  sorry

end female_democrats_count_l240_240999


namespace minimum_value_a5_a6_l240_240321

-- Defining the arithmetic geometric sequence relational conditions.
def arithmetic_geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6) ∧ (∀ n, a n > 0)

-- The mathematical problem to prove:
theorem minimum_value_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h : arithmetic_geometric_sequence_condition a q) :
  a 5 + a 6 = 48 :=
sorry

end minimum_value_a5_a6_l240_240321


namespace sequence_term_1000_l240_240113

open Nat

theorem sequence_term_1000 :
  (∃ b : ℕ → ℤ,
    b 1 = 3010 ∧
    b 2 = 3011 ∧
    (∀ n, 1 ≤ n → b n + b (n + 1) + b (n + 2) = n + 4) ∧
    b 1000 = 3343) :=
sorry

end sequence_term_1000_l240_240113


namespace staircase_ways_four_steps_l240_240055

theorem staircase_ways_four_steps : 
  let one_step := 1
  let two_steps := 2
  let three_steps := 3
  let four_steps := 4
  1           -- one step at a time
  + 3         -- combination of one and two steps
  + 2         -- combination of one and three steps
  + 1         -- two steps at a time
  + 1 = 8     -- all four steps in one stride
:= by
  sorry

end staircase_ways_four_steps_l240_240055


namespace increase_result_l240_240711

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l240_240711


namespace answered_both_questions_correctly_l240_240983

theorem answered_both_questions_correctly (P_A P_B P_A_prime_inter_B_prime : ℝ)
  (h1 : P_A = 70 / 100) (h2 : P_B = 55 / 100) (h3 : P_A_prime_inter_B_prime = 20 / 100) :
  P_A + P_B - (1 - P_A_prime_inter_B_prime) = 45 / 100 := 
by
  sorry

end answered_both_questions_correctly_l240_240983


namespace regular_octagon_diagonal_l240_240062

variable {a b c : ℝ}

-- Define a function to check for a regular octagon where a, b, c are respective side, shortest diagonal, and longest diagonal
def is_regular_octagon (a b c : ℝ) : Prop :=
  -- Here, we assume the standard geometric properties of a regular octagon.
  -- In a real formalization, we might model the octagon directly.

  -- longest diagonal c of a regular octagon (spans 4 sides)
  c = 2 * a

theorem regular_octagon_diagonal (a b c : ℝ) (h : is_regular_octagon a b c) : c = 2 * a :=
by
  exact h

end regular_octagon_diagonal_l240_240062


namespace wine_cost_today_l240_240202

theorem wine_cost_today (C : ℝ) (h1 : ∀ (new_tariff : ℝ), new_tariff = 0.25) (h2 : ∀ (total_increase : ℝ), total_increase = 25) (h3 : C = 20) : 5 * (1.25 * C - C) = 25 :=
by
  sorry

end wine_cost_today_l240_240202


namespace probability_two_white_balls_l240_240664

-- Definitions based on the conditions provided
def total_balls := 17        -- 8 white + 9 black
def white_balls := 8
def drawn_without_replacement := true

-- Proposition: Probability of drawing two white balls successively
theorem probability_two_white_balls:
  drawn_without_replacement → 
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 7 / 34 :=
by
  intros
  sorry

end probability_two_white_balls_l240_240664


namespace equality_of_coefficients_l240_240168

open Real

theorem equality_of_coefficients (a b c x : ℝ)
  (h1 : a * x^2 - b * x - c = b * x^2 - c * x - a)
  (h2 : b * x^2 - c * x - a = c * x^2 - a * x - b)
  (h3 : c * x^2 - a * x - b = a * x^2 - b * x - c):
  a = b ∧ b = c :=
sorry

end equality_of_coefficients_l240_240168


namespace solve_system_of_equations_l240_240968

theorem solve_system_of_equations (x y : ℝ) : 
  (x + y = x^2 + 2 * x * y + y^2) ∧ (x - y = x^2 - 2 * x * y + y^2) ↔ 
  (x = 0 ∧ y = 0) ∨ 
  (x = 1/2 ∧ y = 1/2) ∨ 
  (x = 1/2 ∧ y = -1/2) ∨ 
  (x = 1 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l240_240968


namespace first_competitor_hotdogs_l240_240249

theorem first_competitor_hotdogs (x y z : ℕ) (h1 : y = 3 * x) (h2 : z = 2 * y) (h3 : z * 5 = 300) : x = 10 :=
sorry

end first_competitor_hotdogs_l240_240249


namespace total_boys_in_camp_l240_240192

theorem total_boys_in_camp (T : ℝ) 
  (h1 : 0.20 * T = number_of_boys_from_school_A)
  (h2 : 0.30 * number_of_boys_from_school_A = number_of_boys_study_science_from_school_A)
  (h3 : number_of_boys_from_school_A - number_of_boys_study_science_from_school_A = 42) :
  T = 300 := 
sorry

end total_boys_in_camp_l240_240192


namespace number_of_solutions_eq_l240_240671

open Nat

theorem number_of_solutions_eq (n : ℕ) : 
  ∃ N, (∀ (x : ℝ), 1 ≤ x ∧ x ≤ n → x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2) → N = n^2 - n + 1 :=
by sorry

end number_of_solutions_eq_l240_240671


namespace value_of_v_l240_240756

theorem value_of_v (n : ℝ) (v : ℝ) (h1 : 10 * n = v - 2 * n) (h2 : n = -4.5) : v = -9 := by
  sorry

end value_of_v_l240_240756


namespace simplify_fraction_l240_240131

theorem simplify_fraction (m : ℝ) (h₁: m ≠ 0) (h₂: m ≠ 1): (m - 1) / m / ((m - 1) / (m * m)) = m := by
  sorry

end simplify_fraction_l240_240131


namespace total_chairs_calc_l240_240286

-- Defining the condition of having 27 rows
def rows : ℕ := 27

-- Defining the condition of having 16 chairs per row
def chairs_per_row : ℕ := 16

-- Stating the theorem that the total number of chairs is 432
theorem total_chairs_calc : rows * chairs_per_row = 432 :=
by
  sorry

end total_chairs_calc_l240_240286


namespace package_cheaper_than_per_person_l240_240754

theorem package_cheaper_than_per_person (x : ℕ) :
  (90 * 6 + 10 * x < 54 * x + 8 * 3 * x) ↔ x ≥ 8 :=
by
  sorry

end package_cheaper_than_per_person_l240_240754


namespace weight_of_new_person_l240_240588

-- Definitions based on conditions
def average_weight_increase : ℝ := 2.5
def number_of_persons : ℕ := 8
def old_weight : ℝ := 65
def total_weight_increase : ℝ := number_of_persons * average_weight_increase

-- Proposition to prove
theorem weight_of_new_person : (old_weight + total_weight_increase) = 85 := by
  -- add the actual proof here
  sorry

end weight_of_new_person_l240_240588


namespace boys_to_girls_ratio_l240_240689

theorem boys_to_girls_ratio (T G : ℕ) (h : (1 / 2) * G = (1 / 6) * T) : (T - G) = 2 * G := by
  sorry

end boys_to_girls_ratio_l240_240689


namespace problem_1_l240_240087

theorem problem_1 :
  (-7/4) - (19/3) - 9/4 + 10/3 = -7 := by
  sorry

end problem_1_l240_240087


namespace arrange_leopards_correct_l240_240603

-- Definitions for conditions
def num_shortest : ℕ := 3
def total_leopards : ℕ := 9
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Calculation of total ways to arrange given conditions
def arrange_leopards (num_shortest : ℕ) (total_leopards : ℕ) : ℕ :=
  let choose2short := (num_shortest * (num_shortest - 1)) / 2
  let arrange2short := 2 * factorial (total_leopards - num_shortest)
  choose2short * arrange2short * factorial (total_leopards - num_shortest)

theorem arrange_leopards_correct :
  arrange_leopards num_shortest total_leopards = 30240 := by
  sorry

end arrange_leopards_correct_l240_240603


namespace value_of_x2_plus_4y2_l240_240574

theorem value_of_x2_plus_4y2 (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : x * y = -12) : x^2 + 4*y^2 = 84 := 
  sorry

end value_of_x2_plus_4y2_l240_240574


namespace hyperbola_foci_x_axis_range_l240_240881

theorem hyperbola_foci_x_axis_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1) →
  (1 < m) ↔ 
  (∀ x y : ℝ, (m + 2 > 0) ∧ (m - 1 > 0)) :=
sorry

end hyperbola_foci_x_axis_range_l240_240881


namespace min_value_l240_240364

theorem min_value (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_sum : a + b = 1) : 
  ∃ x : ℝ, (x = 25) ∧ x ≤ (4 / a + 9 / b) :=
by
  sorry

end min_value_l240_240364


namespace train_B_time_to_destination_l240_240236

-- Definitions (conditions)
def speed_train_A := 60  -- Train A travels at 60 kmph
def speed_train_B := 90  -- Train B travels at 90 kmph
def time_train_A_after_meeting := 9 -- Train A takes 9 hours after meeting train B

-- Theorem statement
theorem train_B_time_to_destination 
  (speed_A : ℝ)
  (speed_B : ℝ)
  (time_A_after_meeting : ℝ)
  (time_B_to_destination : ℝ) :
  speed_A = speed_train_A ∧
  speed_B = speed_train_B ∧
  time_A_after_meeting = time_train_A_after_meeting →
  time_B_to_destination = 4.5 :=
by
  sorry

end train_B_time_to_destination_l240_240236


namespace sum_of_reciprocals_l240_240201

theorem sum_of_reciprocals (a b : ℝ) (h_sum : a + b = 15) (h_prod : a * b = 225) :
  (1 / a) + (1 / b) = 1 / 15 :=
by 
  sorry

end sum_of_reciprocals_l240_240201


namespace squirrel_calories_l240_240935

def rabbits_caught_per_hour := 2
def rabbits_calories := 800
def squirrels_caught_per_hour := 6
def extra_calories_squirrels := 200

theorem squirrel_calories : 
  ∀ (S : ℕ), 
  (6 * S = (2 * 800) + 200) → S = 300 := by
  intros S h
  sorry

end squirrel_calories_l240_240935


namespace james_has_43_oreos_l240_240647

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos_l240_240647


namespace roundness_720_eq_7_l240_240235

def roundness (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := n.factorization
    factors.sum (λ _ k => k)
  else 0

theorem roundness_720_eq_7 : roundness 720 = 7 := by
  sorry

end roundness_720_eq_7_l240_240235


namespace value_of_expression_l240_240458

theorem value_of_expression : (3023 - 2990) ^ 2 / 121 = 9 := by
  sorry

end value_of_expression_l240_240458


namespace max_sum_is_1717_l240_240586

noncomputable def max_arithmetic_sum (a d : ℤ) : ℤ :=
  let n := 34
  let S : ℤ := n * (2*a + (n - 1)*d) / 2
  S

theorem max_sum_is_1717 (a d : ℤ) (h1 : a + 16 * d = 52) (h2 : a + 29 * d = 13) (hd : d = -3) (ha : a = 100) :
  max_arithmetic_sum a d = 1717 :=
by
  unfold max_arithmetic_sum
  rw [hd, ha]
  -- Add the necessary steps to prove max_arithmetic_sum 100 (-3) = 1717
  -- Sorry ensures the theorem can be checked syntactically without proof
  sorry

end max_sum_is_1717_l240_240586


namespace total_uniform_cost_l240_240411

theorem total_uniform_cost :
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  total_cost = 355 :=
by 
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  sorry

end total_uniform_cost_l240_240411


namespace line_connecting_centers_l240_240043

-- Define the first circle equation
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x + 6*y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_eq (x y : ℝ) := 3*x - y - 9 = 0

-- Prove that the line connecting the centers of the circles has the given equation
theorem line_connecting_centers :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y → line_eq x y := 
sorry

end line_connecting_centers_l240_240043


namespace minute_hand_distance_traveled_l240_240224

noncomputable def radius : ℝ := 8
noncomputable def minutes_in_one_revolution : ℝ := 60
noncomputable def total_minutes : ℝ := 45

theorem minute_hand_distance_traveled :
  (total_minutes / minutes_in_one_revolution) * (2 * Real.pi * radius) = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_traveled_l240_240224


namespace length_AF_l240_240453

def CE : ℝ := 40
def ED : ℝ := 50
def AE : ℝ := 120
def area_ABCD : ℝ := 7200

theorem length_AF (AF : ℝ) :
  CE = 40 → ED = 50 → AE = 120 → area_ABCD = 7200 →
  AF = 128 :=
by
  intros hCe hEd hAe hArea
  sorry

end length_AF_l240_240453


namespace det_B_eq_2_l240_240171

theorem det_B_eq_2 {x y : ℝ}
  (hB : ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), B = ![![x, 2], ![-3, y]])
  (h_eqn : ∃ (B_inv : Matrix (Fin 2) (Fin 2) ℝ),
    B_inv = (1 / (x * y + 6)) • ![![y, -2], ![3, x]] ∧
    ![![x, 2], ![-3, y]] + 2 • B_inv = 0) : 
  Matrix.det ![![x, 2], ![-3, y]] = 2 :=
by
  sorry

end det_B_eq_2_l240_240171


namespace lassis_from_mangoes_l240_240848

-- Define the given ratio
def lassis_per_mango := 15 / 3

-- Define the number of mangoes
def mangoes := 15

-- Define the expected number of lassis
def expected_lassis := 75

-- Prove that with 15 mangoes, 75 lassis can be made given the ratio
theorem lassis_from_mangoes (h : lassis_per_mango = 5) : mangoes * lassis_per_mango = expected_lassis :=
by
  sorry

end lassis_from_mangoes_l240_240848


namespace stratified_sampling_b_members_l240_240925

variable (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) (sampleSize : ℕ)

-- Conditions from the problem
def condition1 : groupA = 45 := by sorry
def condition2 : groupB = 45 := by sorry
def condition3 : groupC = 60 := by sorry
def condition4 : sampleSize = 10 := by sorry

-- The proof problem statement
theorem stratified_sampling_b_members : 
  (sampleSize * groupB) / (groupA + groupB + groupC) = 3 :=
by sorry

end stratified_sampling_b_members_l240_240925


namespace penniless_pete_dime_difference_l240_240391

theorem penniless_pete_dime_difference :
  ∃ a b c : ℕ, 
  (a + b + c = 100) ∧ 
  (5 * a + 10 * b + 50 * c = 1350) ∧ 
  (b = 170 ∨ b = 8) ∧ 
  (b - 8 = 162 ∨ 170 - b = 162) :=
sorry

end penniless_pete_dime_difference_l240_240391


namespace largest_three_digit_geometric_sequence_with_8_l240_240568

theorem largest_three_digit_geometric_sequence_with_8 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n = 842 ∧ (∃ (a b c : ℕ), n = 100*a + 10*b + c ∧ a = 8 ∧ (a * c = b^2) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) ) :=
by
  sorry

end largest_three_digit_geometric_sequence_with_8_l240_240568


namespace bottles_sold_tuesday_l240_240484

def initial_inventory : ℕ := 4500
def sold_monday : ℕ := 2445
def sold_days_wed_to_sun : ℕ := 50 * 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

theorem bottles_sold_tuesday : 
  initial_inventory + bottles_delivered_saturday - sold_monday - sold_days_wed_to_sun - final_inventory = 900 := 
by
  sorry

end bottles_sold_tuesday_l240_240484


namespace third_discount_l240_240792

noncomputable def find_discount (P S firstDiscount secondDiscount D3 : ℝ) : Prop :=
  S = P * (1 - firstDiscount / 100) * (1 - secondDiscount / 100) * (1 - D3 / 100)

theorem third_discount (P : ℝ) (S : ℝ) (firstDiscount : ℝ) (secondDiscount : ℝ) (D3 : ℝ) 
  (HP : P = 9649.12) (HS : S = 6600)
  (HfirstDiscount : firstDiscount = 20) (HsecondDiscount : secondDiscount = 10) : 
  find_discount P S firstDiscount secondDiscount 5.01 :=
  by
  rw [HP, HS, HfirstDiscount, HsecondDiscount]
  sorry

end third_discount_l240_240792


namespace sodas_to_take_back_l240_240950

def num_sodas_brought : ℕ := 50
def num_sodas_drank : ℕ := 38

theorem sodas_to_take_back : (num_sodas_brought - num_sodas_drank) = 12 := by
  sorry

end sodas_to_take_back_l240_240950


namespace tan_add_l240_240098

open Real

-- Define positive acute angles
def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

-- Theorem: Tangent addition formula
theorem tan_add (α β : ℝ) (hα : acute_angle α) (hβ : acute_angle β) :
  tan (α + β) = (tan α + tan β) / (1 - tan α * tan β) :=
  sorry

end tan_add_l240_240098


namespace vans_needed_l240_240676

-- Definitions of conditions
def students : Nat := 2
def adults : Nat := 6
def capacity_per_van : Nat := 4

-- Main theorem to prove
theorem vans_needed : (students + adults) / capacity_per_van = 2 := by
  sorry

end vans_needed_l240_240676


namespace all_points_on_single_quadratic_l240_240820

theorem all_points_on_single_quadratic (points : Fin 100 → (ℝ × ℝ)) :
  (∀ (p1 p2 p3 p4 : Fin 100),
    ∃ a b c : ℝ, 
      ∀ (i : Fin 100), 
        (i = p1 ∨ i = p2 ∨ i = p3 ∨ i = p4) →
          (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c) → 
  ∃ a b c : ℝ, ∀ i : Fin 100, (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c :=
by 
  sorry

end all_points_on_single_quadratic_l240_240820


namespace find_sum_l240_240535

-- Defining the conditions of the problem
variables (P r t : ℝ) 
theorem find_sum 
  (h1 : (P * r * t) / 100 = 88) 
  (h2 : (P * r * t) / (100 + (r * t)) = 80) 
  : P = 880 := 
sorry

end find_sum_l240_240535


namespace infinite_solutions_xyz_t_l240_240833

theorem infinite_solutions_xyz_t (x y z t : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : t ≠ 0) (h5 : gcd (gcd x y) (gcd z t) = 1) :
  ∃ (x y z t : ℕ), x^3 + y^3 + z^3 = t^4 ∧ gcd (gcd x y) (gcd z t) = 1 :=
sorry

end infinite_solutions_xyz_t_l240_240833


namespace fraction_inhabitable_earth_surface_l240_240341

theorem fraction_inhabitable_earth_surface 
  (total_land_fraction: ℚ) 
  (inhabitable_land_fraction: ℚ) 
  (h1: total_land_fraction = 1/3) 
  (h2: inhabitable_land_fraction = 2/3) 
  : (total_land_fraction * inhabitable_land_fraction) = 2/9 :=
by
  sorry

end fraction_inhabitable_earth_surface_l240_240341


namespace total_people_ride_l240_240462

theorem total_people_ride (people_per_carriage : ℕ) (num_carriages : ℕ) (h1 : people_per_carriage = 12) (h2 : num_carriages = 15) : 
    people_per_carriage * num_carriages = 180 := by
  sorry

end total_people_ride_l240_240462


namespace part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l240_240731

def is_continuous_representable (m : ℕ) (Q : List ℤ) : Prop :=
  ∀ n ∈ (List.range (m + 1)).tail, ∃ (sublist : List ℤ), sublist ≠ [] ∧ sublist ∈ Q.sublists' ∧ sublist.sum = n

theorem part_I_5_continuous :
  is_continuous_representable 5 [2, 1, 4] :=
sorry

theorem part_I_6_not_continuous :
  ¬is_continuous_representable 6 [2, 1, 4] :=
sorry

theorem part_II_min_k_for_8_continuous (Q : List ℤ) :
  is_continuous_representable 8 Q → Q.length ≥ 4 :=
sorry

theorem part_III_min_k_for_20_continuous (Q : List ℤ) 
  (h : is_continuous_representable 20 Q) (h_sum : Q.sum < 20) :
  Q.length ≥ 7 :=
sorry

end part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l240_240731


namespace problem_lean_l240_240739

noncomputable def a : ℕ+ → ℝ := sorry

theorem problem_lean :
  a 11 = 1 / 52 ∧ (∀ n : ℕ+, 1 / a (n + 1) - 1 / a n = 5) → a 1 = 1 / 2 :=
by
  sorry

end problem_lean_l240_240739


namespace Mabel_gave_away_daisies_l240_240362

-- Setting up the conditions
variables (d_total : ℕ) (p_per_daisy : ℕ) (p_remaining : ℕ)

-- stating the assumptions
def initial_petals (d_total p_per_daisy : ℕ) := d_total * p_per_daisy
def petals_given_away (d_total p_per_daisy p_remaining : ℕ) := initial_petals d_total p_per_daisy - p_remaining
def daisies_given_away (d_total p_per_daisy p_remaining : ℕ) := petals_given_away d_total p_per_daisy p_remaining / p_per_daisy

-- The main theorem
theorem Mabel_gave_away_daisies 
  (h1 : d_total = 5)
  (h2 : p_per_daisy = 8)
  (h3 : p_remaining = 24) :
  daisies_given_away d_total p_per_daisy p_remaining = 2 :=
sorry

end Mabel_gave_away_daisies_l240_240362


namespace correct_analogical_reasoning_l240_240565

-- Definitions of the statements in the problem
def statement_A : Prop := ∀ (a b : ℝ), a * 3 = b * 3 → a = b → a * 0 = b * 0 → a = b
def statement_B : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → (a * b) * c = a * c * b * c
def statement_C : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → c ≠ 0 → (a + b) / c = a / c + b / c
def statement_D : Prop := ∀ (a b : ℝ) (n : ℕ), (a * b)^n = a^n * b^n → (a + b)^n = a^n + b^n

-- The theorem stating that option C is the only correct analogical reasoning
theorem correct_analogical_reasoning : statement_C ∧ ¬statement_A ∧ ¬statement_B ∧ ¬statement_D := by
  sorry

end correct_analogical_reasoning_l240_240565


namespace probability_two_females_one_male_l240_240666

theorem probability_two_females_one_male
  (total_contestants : ℕ)
  (female_contestants : ℕ)
  (male_contestants : ℕ)
  (choose_count : ℕ)
  (total_combinations : ℕ)
  (female_combinations : ℕ)
  (male_combinations : ℕ)
  (favorable_outcomes : ℕ)
  (probability : ℚ)
  (h1 : total_contestants = 8)
  (h2 : female_contestants = 5)
  (h3 : male_contestants = 3)
  (h4 : choose_count = 3)
  (h5 : total_combinations = Nat.choose total_contestants choose_count)
  (h6 : female_combinations = Nat.choose female_contestants 2)
  (h7 : male_combinations = Nat.choose male_contestants 1)
  (h8 : favorable_outcomes = female_combinations * male_combinations)
  (h9 : probability = favorable_outcomes / total_combinations) :
  probability = 15 / 28 :=
by
  sorry

end probability_two_females_one_male_l240_240666


namespace oranges_weight_l240_240294

theorem oranges_weight (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := 
by 
  sorry

end oranges_weight_l240_240294


namespace tank_fill_time_l240_240441

theorem tank_fill_time (R L : ℝ) (h1 : (R - L) * 8 = 1) (h2 : L * 56 = 1) :
  (1 / R) = 7 :=
by
  sorry

end tank_fill_time_l240_240441


namespace increasing_range_of_a_l240_240470

noncomputable def f (x : ℝ) (a : ℝ) := 
  if x ≤ 1 then -x^2 + 4*a*x 
  else (2*a + 3)*x - 4*a + 5

theorem increasing_range_of_a :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
sorry

end increasing_range_of_a_l240_240470


namespace conservation_of_mass_l240_240857

def molecular_weight_C := 12.01
def molecular_weight_H := 1.008
def molecular_weight_O := 16.00
def molecular_weight_Na := 22.99

def molecular_weight_C9H8O4 := (9 * molecular_weight_C) + (8 * molecular_weight_H) + (4 * molecular_weight_O)
def molecular_weight_NaOH := molecular_weight_Na + molecular_weight_O + molecular_weight_H
def molecular_weight_C7H6O3 := (7 * molecular_weight_C) + (6 * molecular_weight_H) + (3 * molecular_weight_O)
def molecular_weight_CH3COONa := (2 * molecular_weight_C) + (3 * molecular_weight_H) + (2 * molecular_weight_O) + molecular_weight_Na

theorem conservation_of_mass :
  (molecular_weight_C9H8O4 + molecular_weight_NaOH) = (molecular_weight_C7H6O3 + molecular_weight_CH3COONa) := by
  sorry

end conservation_of_mass_l240_240857


namespace net_increase_correct_l240_240805

-- Definitions for the given conditions
def S1 : ℕ := 10
def B1 : ℕ := 15
def S2 : ℕ := 12
def B2 : ℕ := 8
def S3 : ℕ := 9
def B3 : ℕ := 11

def P1 : ℕ := 250
def P2 : ℕ := 275
def P3 : ℕ := 260
def C1 : ℕ := 100
def C2 : ℕ := 110
def C3 : ℕ := 120

def Sale_profit1 : ℕ := S1 * P1
def Sale_profit2 : ℕ := S2 * P2
def Sale_profit3 : ℕ := S3 * P3

def Repair_cost1 : ℕ := B1 * C1
def Repair_cost2 : ℕ := B2 * C2
def Repair_cost3 : ℕ := B3 * C3

def Net_profit1 : ℕ := Sale_profit1 - Repair_cost1
def Net_profit2 : ℕ := Sale_profit2 - Repair_cost2
def Net_profit3 : ℕ := Sale_profit3 - Repair_cost3

def Total_net_profit : ℕ := Net_profit1 + Net_profit2 + Net_profit3

def Net_Increase : ℕ := (B1 - S1) + (B2 - S2) + (B3 - S3)

-- The theorem to be proven
theorem net_increase_correct : Net_Increase = 3 := by
  sorry

end net_increase_correct_l240_240805


namespace domain_of_log_function_l240_240653

theorem domain_of_log_function :
  {x : ℝ | x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_function_l240_240653


namespace least_positive_integer_condition_l240_240677

theorem least_positive_integer_condition :
  ∃ (n : ℕ), n > 0 ∧ (n % 2 = 1) ∧ (n % 5 = 4) ∧ (n % 7 = 6) ∧ n = 69 :=
by
  sorry

end least_positive_integer_condition_l240_240677


namespace ratio_avg_speeds_l240_240779

-- Definitions based on the problem conditions
def distance_A_B := 600
def time_Eddy := 3
def distance_A_C := 460
def time_Freddy := 4

-- Definition of average speeds
def avg_speed_Eddy := distance_A_B / time_Eddy
def avg_speed_Freddy := distance_A_C / time_Freddy

-- Theorem statement
theorem ratio_avg_speeds : avg_speed_Eddy / avg_speed_Freddy = 40 / 23 := 
sorry

end ratio_avg_speeds_l240_240779


namespace evaluate_f_at_2_l240_240106

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem evaluate_f_at_2 :
  f 2 = 125 :=
by
  sorry

end evaluate_f_at_2_l240_240106


namespace project_time_for_A_l240_240831

/--
A can complete a project in some days and B can complete the same project in 30 days.
If A and B start working on the project together and A quits 5 days before the project is 
completed, the project will be completed in 15 days.
Prove that A can complete the project alone in 20 days.
-/
theorem project_time_for_A (x : ℕ) (h : 10 * (1 / x + 1 / 30) + 5 * (1 / 30) = 1) : x = 20 :=
sorry

end project_time_for_A_l240_240831


namespace relationship_between_abc_l240_240806

theorem relationship_between_abc (a b c k : ℝ) 
  (hA : -3 = - (k^2 + 1) / a)
  (hB : -2 = - (k^2 + 1) / b)
  (hC : 1 = - (k^2 + 1) / c)
  (hk : 0 < k^2 + 1) : c < a ∧ a < b :=
by
  sorry

end relationship_between_abc_l240_240806


namespace mallory_travel_expenses_l240_240004

theorem mallory_travel_expenses (fuel_tank_cost : ℕ) (fuel_tank_miles : ℕ) (total_miles : ℕ) (food_ratio : ℚ)
  (h_fuel_tank_cost : fuel_tank_cost = 45)
  (h_fuel_tank_miles : fuel_tank_miles = 500)
  (h_total_miles : total_miles = 2000)
  (h_food_ratio : food_ratio = 3/5) :
  ∃ total_cost : ℕ, total_cost = 288 :=
by
  sorry

end mallory_travel_expenses_l240_240004


namespace findTwoHeaviestStonesWith35Weighings_l240_240001

-- Define the problem with conditions
def canFindTwoHeaviestStones (stones : Fin 32 → ℝ) (weighings : ℕ) : Prop :=
  ∀ (balanceScale : (Fin 32 × Fin 32) → Bool), weighings ≤ 35 → 
  ∃ (heaviest : Fin 32) (secondHeaviest : Fin 32), 
  (heaviest ≠ secondHeaviest) ∧ 
  (∀ i : Fin 32, stones heaviest ≥ stones i) ∧ 
  (∀ j : Fin 32, j ≠ heaviest → stones secondHeaviest ≥ stones j)

-- Formally state the theorem
theorem findTwoHeaviestStonesWith35Weighings (stones : Fin 32 → ℝ) :
  canFindTwoHeaviestStones stones 35 :=
sorry -- Proof is omitted

end findTwoHeaviestStonesWith35Weighings_l240_240001


namespace prop_p_necessary_but_not_sufficient_for_prop_q_l240_240530

theorem prop_p_necessary_but_not_sufficient_for_prop_q (x y : ℕ) :
  (x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4) → ((x+y ≠ 4) → (x ≠ 1 ∨ y ≠ 3)) ∧ ¬ ((x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4)) :=
by
  sorry

end prop_p_necessary_but_not_sufficient_for_prop_q_l240_240530


namespace combined_weight_of_three_new_people_l240_240883

theorem combined_weight_of_three_new_people 
  (W : ℝ) 
  (h_avg_increase : (W + 80) / 20 = W / 20 + 4) 
  (h_replaced_weights : 60 + 75 + 85 = 220) : 
  220 + 80 = 300 :=
by
  sorry

end combined_weight_of_three_new_people_l240_240883


namespace pool_half_capacity_at_6_hours_l240_240057

noncomputable def double_volume_every_hour (t : ℕ) : ℕ := 2 ^ t

theorem pool_half_capacity_at_6_hours (V : ℕ) (h : ∀ t : ℕ, V = double_volume_every_hour 8) : double_volume_every_hour 6 = V / 2 := by
  sorry

end pool_half_capacity_at_6_hours_l240_240057


namespace least_subtracted_number_l240_240980

theorem least_subtracted_number (r : ℕ) : r = 10^1000 % 97 := 
sorry

end least_subtracted_number_l240_240980


namespace plane_equation_correct_l240_240174

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p q : Point3D) : Point3D :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def plane_eq (n : Point3D) (A : Point3D) : Point3D → ℝ :=
  fun P => n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

def is_perpendicular_plane (A B C : Point3D) (D : Point3D → ℝ) : Prop :=
  let BC := vector_sub C B
  D = plane_eq BC A

theorem plane_equation_correct :
  let A := { x := 7, y := -5, z := 1 }
  let B := { x := 5, y := -1, z := -3 }
  let C := { x := 3, y := 0, z := -4 }
  is_perpendicular_plane A B C (fun P => -2 * P.x + P.y - P.z + 20) :=
by
  sorry

end plane_equation_correct_l240_240174


namespace problem_inequality_l240_240529

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem problem_inequality (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≠ x2) :
  (f x2 - f x1) / (x2 - x1) < (1 + Real.log ((x1 + x2) / 2)) :=
sorry

end problem_inequality_l240_240529


namespace prime_N_k_iff_k_eq_2_l240_240596

-- Define the function to generate the number N_k based on k
def N_k (k : ℕ) : ℕ := (10^(2 * k) - 1) / 99

-- Define the main theorem to prove
theorem prime_N_k_iff_k_eq_2 (k : ℕ) : Nat.Prime (N_k k) ↔ k = 2 :=
by
  sorry

end prime_N_k_iff_k_eq_2_l240_240596


namespace sean_bought_two_soups_l240_240644

theorem sean_bought_two_soups :
  ∃ (number_of_soups : ℕ),
    let soda_cost := 1
    let total_soda_cost := 3 * soda_cost
    let soup_cost := total_soda_cost
    let sandwich_cost := 3 * soup_cost
    let total_cost := 3 * soda_cost + sandwich_cost + soup_cost * number_of_soups
    total_cost = 18 ∧ number_of_soups = 2 :=
by
  sorry

end sean_bought_two_soups_l240_240644


namespace subway_distance_per_minute_l240_240534

theorem subway_distance_per_minute :
  let total_distance := 120 -- kilometers
  let total_time := 110 -- minutes (1 hour and 50 minutes)
  let bus_time := 70 -- minutes (1 hour and 10 minutes)
  let bus_distance := (14 * 40.8) / 6 -- kilometers
  let subway_distance := total_distance - bus_distance -- kilometers
  let subway_time := total_time - bus_time -- minutes
  let distance_per_minute := subway_distance / subway_time
  distance_per_minute = 0.62 := 
by
  sorry

end subway_distance_per_minute_l240_240534


namespace area_triangle_ABC_given_conditions_l240_240285

variable (a b c : ℝ) (A B C : ℝ)

noncomputable def area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem area_triangle_ABC_given_conditions
  (habc : a = 4)
  (hbc : b + c = 5)
  (htan : Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * (Real.tan B * Real.tan C))
  : area_of_triangle_ABC a b c (Real.pi / 3) B C = 3 * Real.sqrt 3 / 4 := 
sorry

end area_triangle_ABC_given_conditions_l240_240285


namespace prescribedDosageLessThanTypical_l240_240125

noncomputable def prescribedDosage : ℝ := 12
noncomputable def bodyWeight : ℝ := 120
noncomputable def typicalDosagePer15Pounds : ℝ := 2
noncomputable def typicalDosage : ℝ := (bodyWeight / 15) * typicalDosagePer15Pounds
noncomputable def percentageDecrease : ℝ := ((typicalDosage - prescribedDosage) / typicalDosage) * 100

theorem prescribedDosageLessThanTypical :
  percentageDecrease = 25 :=
by
  sorry

end prescribedDosageLessThanTypical_l240_240125


namespace problem_one_problem_two_l240_240932

-- Define p and q
def p (a x : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := |x - 3| < 1

-- Problem (1)
theorem problem_one (a : ℝ) (h_a : a = 1) (h_pq : p a x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem_two (a : ℝ) (h_a_pos : a > 0) (suff : ¬ p a x → ¬ q x) (not_necess : ¬ (¬ q x → ¬ p a x)) : 
  (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

end problem_one_problem_two_l240_240932


namespace cost_of_paving_l240_240063

noncomputable def length : Float := 5.5
noncomputable def width : Float := 3.75
noncomputable def cost_per_sq_meter : Float := 600

theorem cost_of_paving :
  (length * width * cost_per_sq_meter) = 12375 := by
  sorry

end cost_of_paving_l240_240063


namespace value_of_product_l240_240351

theorem value_of_product : (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by sorry

end value_of_product_l240_240351


namespace sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l240_240019

-- Definition for the sum of the first n natural numbers
def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition for the sum from 1 to 60
def sum_1_to_60 : ℕ := sum_upto 60

-- Definition for the sum from 1 to 50
def sum_1_to_50 : ℕ := sum_upto 50

-- Proof problem 1
theorem sum_from_1_to_60_is_1830 : sum_1_to_60 = 1830 := 
by
  sorry

-- Definition for the sum from 51 to 60
def sum_51_to_60 : ℕ := sum_1_to_60 - sum_1_to_50

-- Proof problem 2
theorem sum_from_51_to_60_is_555 : sum_51_to_60 = 555 := 
by
  sorry

end sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l240_240019


namespace polynomial_expression_value_l240_240782

theorem polynomial_expression_value (a : ℕ → ℤ) (x : ℤ) :
  (x + 2)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 →
  ((a 1 + 3 * a 3 + 5 * a 5 + 7 * a 7 + 9 * a 9)^2 - (2 * a 2 + 4 * a 4 + 6 * a 6 + 8 * a 8)^2) = 3^12 :=
by
  sorry

end polynomial_expression_value_l240_240782


namespace max_value_x1_x2_l240_240302

noncomputable def f (x : ℝ) := 1 - Real.sqrt (2 - 3 * x)
noncomputable def g (x : ℝ) := 2 * Real.log x

theorem max_value_x1_x2 (x1 x2 : ℝ) (h1 : x1 ≤ 2 / 3) (h2 : x2 > 0) (h3 : x1 - x2 = (1 - Real.sqrt (2 - 3 * x1)) - (2 * Real.log x2)) :
  x1 - x2 ≤ -25 / 48 :=
sorry

end max_value_x1_x2_l240_240302


namespace problem_part1_problem_part2_l240_240652

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l240_240652


namespace two_a_minus_two_d_eq_zero_l240_240194

noncomputable def g (a b c d x : ℝ) : ℝ := (2 * a * x - b) / (c * x - 2 * d)

theorem two_a_minus_two_d_eq_zero (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : ∀ x : ℝ, (g a a c d (g a b c d x)) = x) : 2 * a - 2 * d = 0 :=
sorry

end two_a_minus_two_d_eq_zero_l240_240194


namespace largest_multiple_of_7_less_than_100_l240_240161

theorem largest_multiple_of_7_less_than_100 : ∃ (n : ℕ), n * 7 < 100 ∧ ∀ (m : ℕ), m * 7 < 100 → m * 7 ≤ n * 7 :=
  by
  sorry

end largest_multiple_of_7_less_than_100_l240_240161


namespace necessary_but_not_sufficient_l240_240808

-- Definitions from the conditions
def p (a b : ℤ) : Prop := True  -- Since their integrality is given
def q (a b : ℤ) : Prop := ∃ (x : ℤ), (x^2 + a * x + b = 0)

theorem necessary_but_not_sufficient (a b : ℤ) : 
  (¬ (p a b → q a b)) ∧ (q a b → p a b) :=
by
  sorry

end necessary_but_not_sufficient_l240_240808


namespace man_cannot_row_against_stream_l240_240880

theorem man_cannot_row_against_stream (rate_in_still_water speed_with_stream : ℝ)
  (h_rate : rate_in_still_water = 1)
  (h_speed_with : speed_with_stream = 6) :
  ¬ ∃ (speed_against_stream : ℝ), speed_against_stream = rate_in_still_water - (speed_with_stream - rate_in_still_water) :=
by
  sorry

end man_cannot_row_against_stream_l240_240880


namespace other_train_length_l240_240921

noncomputable def relative_speed (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

noncomputable def speed_in_km_per_sec (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr / 3600

noncomputable def total_distance_crossed (relative_speed : ℝ) (time_sec : ℕ) : ℝ :=
  relative_speed * (time_sec : ℝ)

noncomputable def length_of_other_train (total_distance length_of_first_train : ℝ) : ℝ :=
  total_distance - length_of_first_train

theorem other_train_length :
  let speed1 := 210
  let speed2 := 90
  let length_of_first_train := 0.9
  let time_taken := 24
  let relative_speed_km_per_hr := relative_speed speed1 speed2
  let relative_speed_km_per_sec := speed_in_km_per_sec relative_speed_km_per_hr
  let total_distance := total_distance_crossed relative_speed_km_per_sec time_taken
  length_of_other_train total_distance length_of_first_train = 1.1 := 
by
  sorry

end other_train_length_l240_240921


namespace find_speed_first_car_l240_240961

noncomputable def speed_first_car (v : ℝ) : Prop :=
  let t := (14 : ℝ) / 3
  let d_total := 490
  let d_second_car := 60 * t
  let d_first_car := v * t
  d_second_car + d_first_car = d_total

theorem find_speed_first_car : ∃ v : ℝ, speed_first_car v ∧ v = 45 :=
by
  sorry

end find_speed_first_car_l240_240961


namespace cos_alpha_correct_l240_240386

-- Define the point P
def P : ℝ × ℝ := (3, -4)

-- Define the hypotenuse using the Pythagorean theorem
noncomputable def r : ℝ :=
  Real.sqrt (P.1 * P.1 + P.2 * P.2)

-- Define x-coordinate of point P
def x : ℝ := P.1

-- Define the cosine of the angle
noncomputable def cos_alpha : ℝ :=
  x / r

-- Prove that cos_alpha equals 3/5 given the conditions
theorem cos_alpha_correct : cos_alpha = 3 / 5 :=
by
  sorry

end cos_alpha_correct_l240_240386


namespace relationship_between_x_y_l240_240780

def in_interval (x : ℝ) : Prop := (Real.pi / 4) < x ∧ x < (Real.pi / 2)

noncomputable def x_def (α : ℝ) : ℝ := Real.sin α ^ (Real.log (Real.cos α) / Real.log α)

noncomputable def y_def (α : ℝ) : ℝ := Real.cos α ^ (Real.log (Real.sin α) / Real.log α)

theorem relationship_between_x_y (α : ℝ) (h : in_interval α) : 
  x_def α = y_def α := 
  sorry

end relationship_between_x_y_l240_240780


namespace rectangle_width_decreased_by_33_percent_l240_240823

theorem rectangle_width_decreased_by_33_percent
  (L W A : ℝ)
  (hA : A = L * W)
  (newL : ℝ)
  (h_newL : newL = 1.5 * L)
  (W' : ℝ)
  (h_area_unchanged : newL * W' = A) : 
  (1 - W' / W) * 100 = 33.33 :=
by
  sorry

end rectangle_width_decreased_by_33_percent_l240_240823


namespace common_ratio_value_l240_240648

variable (a : ℕ → ℝ) -- defining the geometric sequence as a function ℕ → ℝ
variable (q : ℝ) -- defining the common ratio

-- conditions from the problem
def geo_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

axiom h1 : geo_seq a q
axiom h2 : a 2020 = 8 * a 2017

-- main statement to be proved
theorem common_ratio_value : q = 2 :=
sorry

end common_ratio_value_l240_240648


namespace rice_less_than_beans_by_30_l240_240420

noncomputable def GB : ℝ := 60
noncomputable def S : ℝ := 50

theorem rice_less_than_beans_by_30 (R : ℝ) (x : ℝ) (h1 : R = 60 - x) (h2 : (2/3) * R + (4/5) * S + GB = 120) : 60 - R = 30 :=
by 
  -- Proof steps would go here, but they are not required for this task.
  sorry

end rice_less_than_beans_by_30_l240_240420


namespace max_neg_integers_l240_240146

theorem max_neg_integers (
  a b c d e f g h : ℤ
) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_e : e ≠ 0)
  (h_ineq : (a * b^2 + c * d * e^3) * (f * g^2 * h + f^3 - g^2) < 0)
  (h_abs : |d| < |f| ∧ |f| < |h|)
  : ∃ s, s = 5 ∧ ∀ (neg_count : ℕ), neg_count ≤ s := 
sorry

end max_neg_integers_l240_240146


namespace ratio_avg_eq_42_l240_240278

theorem ratio_avg_eq_42 (a b c d : ℕ)
  (h1 : ∃ k : ℕ, a = 2 * k ∧ b = 3 * k ∧ c = 4 * k ∧ d = 5 * k)
  (h2 : (a + b + c + d) / 4 = 42) : a = 24 :=
by sorry

end ratio_avg_eq_42_l240_240278


namespace amy_initial_money_l240_240536

-- Define the conditions
variable (left_fair : ℕ) (spent : ℕ)

-- Define the proof problem statement
theorem amy_initial_money (h1 : left_fair = 11) (h2 : spent = 4) : left_fair + spent = 15 := 
by sorry

end amy_initial_money_l240_240536


namespace proof_problem_l240_240085

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 1

-- Define the circle C with center (h, k) and radius r
def circle_eq (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Define condition of line that intersects the circle C at points A and B
def line_eq (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Condition: OA ⊥ OB
def perpendicular_cond (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem stating the proof problem
theorem proof_problem :
  (∃ (h k r : ℝ),
    circle_eq h k r 3 1 ∧
    circle_eq h k r 5 0 ∧
    circle_eq h k r 1 0 ∧
    h = 3 ∧ k = 1 ∧ r = 3) ∧
    (∃ (a : ℝ),
      (∀ (x1 y1 x2 y2 : ℝ),
        line_eq a x1 y1 ∧
        circle_eq 3 1 3 x1 y1 ∧
        line_eq a x2 y2 ∧
        circle_eq 3 1 3 x2 y2 → 
        perpendicular_cond x1 y1 x2 y2) →
      a = -1) :=
by
  sorry

end proof_problem_l240_240085


namespace train_length_250_meters_l240_240542

open Real

noncomputable def speed_in_ms (speed_km_hr: ℝ): ℝ :=
  speed_km_hr * (1000 / 3600)

noncomputable def length_of_train (speed: ℝ) (time: ℝ): ℝ :=
  speed * time

theorem train_length_250_meters (speed_km_hr: ℝ) (time_seconds: ℝ) :
  speed_km_hr = 40 → time_seconds = 22.5 → length_of_train (speed_in_ms speed_km_hr) time_seconds = 250 :=
by
  intros
  sorry

end train_length_250_meters_l240_240542


namespace change_in_total_berries_l240_240781

-- Define the initial conditions
def blue_box_berries : ℕ := 35
def increase_diff : ℕ := 100

-- Define the number of strawberries in red boxes
def red_box_berries : ℕ := 100

-- Formulate the change in total number of berries
theorem change_in_total_berries :
  (red_box_berries - blue_box_berries) = 65 :=
by
  have h1 : red_box_berries = increase_diff := rfl
  have h2 : blue_box_berries = 35 := rfl
  rw [h1, h2]
  exact rfl

end change_in_total_berries_l240_240781


namespace nat_representable_as_sequence_or_difference_l240_240868

theorem nat_representable_as_sequence_or_difference
  (a : ℕ → ℕ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∀ n, a n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, k ≠ l ∧ (m = a k ∨ m = a k - a l) :=
by
  sorry

end nat_representable_as_sequence_or_difference_l240_240868


namespace sum_of_possible_x_values_l240_240945

theorem sum_of_possible_x_values (x : ℝ) : 
  (3 : ℝ)^(x^2 + 6*x + 9) = (27 : ℝ)^(x + 3) → x = 0 ∨ x = -3 → x = 0 ∨ x = -3 := 
sorry

end sum_of_possible_x_values_l240_240945


namespace books_total_correct_l240_240303

-- Define the constants for the number of books obtained each day
def books_day1 : ℕ := 54
def books_day2_total : ℕ := 23
def books_day2_kept : ℕ := 12
def books_day3_multiplier : ℕ := 3

-- Calculate the total number of books obtained each day
def books_day3 := books_day3_multiplier * books_day2_total
def total_books := books_day1 + books_day2_kept + books_day3

-- The theorem to prove
theorem books_total_correct : total_books = 135 := by
  sorry

end books_total_correct_l240_240303


namespace weights_are_equal_l240_240132

variable {n : ℕ}
variables {a : Fin (2 * n + 1) → ℝ}

def weights_condition
    (a : Fin (2 * n + 1) → ℝ) : Prop :=
  ∀ i : Fin (2 * n + 1), ∃ (A B : Finset (Fin (2 * n + 1))),
    A.card = n ∧ B.card = n ∧ A ∩ B = ∅ ∧
    A ∪ B = Finset.univ.erase i ∧
    (A.sum a = B.sum a)

theorem weights_are_equal
    (h : weights_condition a) :
  ∃ k : ℝ, ∀ i : Fin (2 * n + 1), a i = k :=
  sorry

end weights_are_equal_l240_240132


namespace correct_rounded_result_l240_240624

def round_to_nearest_ten (n : ℤ) : ℤ :=
  (n + 5) / 10 * 10

theorem correct_rounded_result :
  round_to_nearest_ten ((57 + 68) * 2) = 250 :=
by
  sorry

end correct_rounded_result_l240_240624


namespace bacteria_initial_count_l240_240220

noncomputable def initial_bacteria (b_final : ℕ) (q : ℕ) : ℕ :=
  b_final / 4^q

theorem bacteria_initial_count : initial_bacteria 262144 4 = 1024 := by
  sorry

end bacteria_initial_count_l240_240220


namespace initial_extra_planks_l240_240127

-- Definitions corresponding to the conditions
def charlie_planks : Nat := 10
def father_planks : Nat := 10
def total_planks : Nat := 35

-- The proof problem statement
theorem initial_extra_planks : total_planks - (charlie_planks + father_planks) = 15 := by
  sorry

end initial_extra_planks_l240_240127


namespace vertex_in_second_quadrant_l240_240875

-- Theorems and properties regarding quadratic functions and their roots.
theorem vertex_in_second_quadrant (c : ℝ) (h : 4 + 4 * c < 0) : 
  (1:ℝ) * -1^2 + 2 * -1 - c > 0 :=
sorry

end vertex_in_second_quadrant_l240_240875


namespace Mia_and_dad_time_to_organize_toys_l240_240067

theorem Mia_and_dad_time_to_organize_toys :
  let total_toys := 60
  let dad_add_rate := 6
  let mia_remove_rate := 4
  let net_gain_per_cycle := dad_add_rate - mia_remove_rate
  let seconds_per_cycle := 30
  let total_needed_cycles := (total_toys - 2) / net_gain_per_cycle -- 58 toys by the end of repeated cycles, 2 is to ensure dad's last placement
  let last_cycle_time := seconds_per_cycle
  let total_time_seconds := total_needed_cycles * seconds_per_cycle + last_cycle_time
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 15 :=
by
  sorry

end Mia_and_dad_time_to_organize_toys_l240_240067


namespace equation_of_line_l240_240414

noncomputable def line_equation_parallel (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (3 * x - 6 * y = 9) ∧ (m = 1/2)

theorem equation_of_line (m : ℝ) (b : ℝ) :
  line_equation_parallel 3 9 →
  (m = 1/2) →
  (∀ (x y : ℝ), (y = m * x + b) ↔ (y - (-1) = m * (x - 2))) →
  b = -2 :=
by
  intros h_eq h_m h_line
  sorry

end equation_of_line_l240_240414


namespace weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l240_240672

noncomputable def cost_price : ℝ := 10
noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def w (x : ℝ) : ℝ := -10 * x ^ 2 + 500 * x - 4000

-- Proof Step 1: Show the functional relationship between w and x
theorem weekly_profit_function : ∀ x : ℝ, w x = -10 * x ^ 2 + 500 * x - 4000 := by
  intro x
  -- This is the function definition provided, proof omitted
  sorry

-- Proof Step 2: Find the selling price x that maximizes weekly profit
theorem maximize_weekly_profit : ∃ x : ℝ, x = 25 ∧ (∀ y : ℝ, y ≠ x → w y ≤ w x) := by
  use 25
  -- The details of solving the optimization are omitted
  sorry

-- Proof Step 3: Given weekly profit w = 2000 and constraints on y, find the weekly sales quantity
theorem weekly_sales_quantity (x : ℝ) (H : w x = 2000 ∧ y x ≥ 180) : y x = 200 := by
  have Hy : y x = -10 * x + 400 := by rfl
  have Hconstraint : y x ≥ 180 := H.2
  have Hprofit : w x = 2000 := H.1
  -- The details of solving for x and ensuring constraints are omitted
  sorry

end weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l240_240672


namespace original_mixture_acid_percent_l240_240251

-- Definitions of conditions as per the original problem
def original_acid_percentage (a w : ℕ) (h1 : 4 * a = a + w + 2) (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : Prop :=
  (a * 100) / (a + w) = 100 / 3

-- Main theorem statement
theorem original_mixture_acid_percent (a w : ℕ) 
  (h1 : 4 * a = a + w + 2)
  (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : original_acid_percentage a w h1 h2 :=
sorry

end original_mixture_acid_percent_l240_240251


namespace solve_for_x_l240_240507

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end solve_for_x_l240_240507


namespace range_of_a_l240_240911

variable (f : ℝ → ℝ)
variable (a : ℝ)

theorem range_of_a (h1 : ∀ a : ℝ, (f (1 - 2 * a) / 2 ≥ f a))
                  (h2 : ∀ (x1 x2 : ℝ), x1 < x2 ∧ x1 + x2 ≠ 0 → f x1 > f x2) : a > (1 / 2) :=
by
  sorry

end range_of_a_l240_240911


namespace price_of_mixture_l240_240417

theorem price_of_mixture :
  (1 * 64 + 1 * 74) / (1 + 1) = 69 :=
by
  sorry

end price_of_mixture_l240_240417


namespace duration_period_l240_240690

-- Define the conditions and what we need to prove
theorem duration_period (t : ℝ) (h : 3200 * 0.025 * t = 400) : 
  t = 5 :=
sorry

end duration_period_l240_240690


namespace candy_left_proof_l240_240379

def candy_left (d_candy : ℕ) (s_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  d_candy + s_candy - eaten_candy

theorem candy_left_proof :
  candy_left 32 42 35 = 39 :=
by
  sorry

end candy_left_proof_l240_240379


namespace range_of_m_l240_240440

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^2 + 4 * x + 5)
  (h2 : ∀ x : ℝ, f (-2 + x) = f (-2 - x))
  (h3 : ∀ x : ℝ, m ≤ x ∧ x ≤ 0 → 1 ≤ f x ∧ f x ≤ 5)
  : -4 ≤ m ∧ m ≤ -2 :=
  sorry

end range_of_m_l240_240440


namespace no_integer_solutions_l240_240323

theorem no_integer_solutions (w l : ℕ) (hw_pos : 0 < w) (hl_pos : 0 < l) : 
  (w * l = 24 ∧ (w = l ∨ 2 * l = w)) → false :=
by 
  sorry

end no_integer_solutions_l240_240323


namespace display_total_cans_l240_240783

def row_num_cans (row : ℕ) : ℕ :=
  if row < 7 then 19 - 3 * (7 - row)
  else 19 + 3 * (row - 7)

def total_cans : ℕ :=
  List.sum (List.map row_num_cans (List.range 10))

theorem display_total_cans : total_cans = 145 := 
  sorry

end display_total_cans_l240_240783


namespace red_large_toys_count_l240_240258

def percentage_red : ℝ := 0.25
def percentage_green : ℝ := 0.20
def percentage_blue : ℝ := 0.15
def percentage_yellow : ℝ := 0.25
def percentage_orange : ℝ := 0.15

def red_small : ℝ := 0.06
def red_medium : ℝ := 0.08
def red_large : ℝ := 0.07
def red_extra_large : ℝ := 0.04

def green_small : ℝ := 0.04
def green_medium : ℝ := 0.07
def green_large : ℝ := 0.05
def green_extra_large : ℝ := 0.04

def blue_small : ℝ := 0.06
def blue_medium : ℝ := 0.03
def blue_large : ℝ := 0.04
def blue_extra_large : ℝ := 0.02

def yellow_small : ℝ := 0.08
def yellow_medium : ℝ := 0.10
def yellow_large : ℝ := 0.05
def yellow_extra_large : ℝ := 0.02

def orange_small : ℝ := 0.09
def orange_medium : ℝ := 0.06
def orange_large : ℝ := 0.05
def orange_extra_large : ℝ := 0.05

def green_large_count : ℕ := 47

noncomputable def total_green_toys := green_large_count / green_large

noncomputable def total_toys := total_green_toys / percentage_green

noncomputable def red_large_toys := total_toys * red_large

theorem red_large_toys_count : red_large_toys = 329 := by
  sorry

end red_large_toys_count_l240_240258


namespace gala_arrangements_l240_240400

theorem gala_arrangements :
  let original_programs := 10
  let added_programs := 3
  let total_positions := original_programs + 1 - 2 -- Excluding first and last
  (total_positions * (total_positions - 1) * (total_positions - 2)) / 6 = 165 :=
by sorry

end gala_arrangements_l240_240400


namespace blake_spent_on_apples_l240_240953

noncomputable def apples_spending_problem : Prop :=
  let initial_amount := 300
  let change_received := 150
  let oranges_cost := 40
  let mangoes_cost := 60
  let total_spent := initial_amount - change_received
  let other_fruits_cost := oranges_cost + mangoes_cost
  let apples_cost := total_spent - other_fruits_cost
  apples_cost = 50

theorem blake_spent_on_apples : apples_spending_problem :=
by
  sorry

end blake_spent_on_apples_l240_240953


namespace find_k_l240_240720

-- Define vectors a, b, and c
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 2)

-- Define the dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for perpendicular vectors
def perpendicular_condition (k : ℝ) : Prop :=
  dot_product (a.1 - k, -1) b = 0

-- State the theorem
theorem find_k : ∃ k : ℝ, perpendicular_condition k ∧ k = 0 := by
  sorry

end find_k_l240_240720


namespace equal_probabilities_l240_240528

-- Definitions based on the conditions in the problem

def total_parts : ℕ := 160
def first_class_parts : ℕ := 48
def second_class_parts : ℕ := 64
def third_class_parts : ℕ := 32
def substandard_parts : ℕ := 16
def sample_size : ℕ := 20

-- Define the probabilities for each sampling method
def p1 : ℚ := sample_size / total_parts
def p2 : ℚ := (6 : ℚ) / first_class_parts  -- Given the conditions, this will hold for all classes
def p3 : ℚ := 1 / 8

theorem equal_probabilities :
  p1 = p2 ∧ p2 = p3 :=
by
  -- This is the end of the statement as no proof is required
  sorry

end equal_probabilities_l240_240528


namespace car_speed_l240_240255

theorem car_speed (uses_one_gallon_per_30_miles : ∀ d : ℝ, d = 30 → d / 30 ≥ 1)
    (full_tank : ℝ := 10)
    (travel_time : ℝ := 5)
    (fraction_of_tank_used : ℝ := 0.8333333333333334)
    (speed : ℝ := 50) :
  let amount_of_gasoline_used := fraction_of_tank_used * full_tank
  let distance_traveled := amount_of_gasoline_used * 30
  speed = distance_traveled / travel_time :=
by
  sorry

end car_speed_l240_240255


namespace rectangle_area_l240_240283

theorem rectangle_area
    (w l : ℕ)
    (h₁ : 28 = 2 * (l + w))
    (h₂ : w = 6) : l * w = 48 :=
by
  sorry

end rectangle_area_l240_240283


namespace part1_part2_l240_240518

variable {a b c m t y1 y2 : ℝ}

-- Condition: point (2, m) lies on the parabola y = ax^2 + bx + c where axis of symmetry is x = t
def point_lies_on_parabola (a b c m : ℝ) := m = a * 2^2 + b * 2 + c

-- Condition: axis of symmetry x = t
def axis_of_symmetry (a b t : ℝ) := t = -b / (2 * a)

-- Condition: m = c
theorem part1 (a c : ℝ) (h : m = c) (h₀ : point_lies_on_parabola a (-2 * a) c m) :
  axis_of_symmetry a (-2 * a) 1 :=
by sorry

-- Additional Condition: c < m
def c_lt_m (c m : ℝ) := c < m

-- Points (-1, y1) and (3, y2) lie on the parabola y = ax^2 + bx + c
def points_on_parabola (a b c y1 y2 : ℝ) :=
  y1 = a * (-1)^2 + b * (-1) + c ∧ y2 = a * 3^2 + b * 3 + c

-- Comparison result
theorem part2 (a : ℝ) (h₁ : c_lt_m c m) (h₂ : 2 * a + (-2 * a) > 0) (h₂' : points_on_parabola a (-2 * a) c y1 y2) :
  y2 > y1 :=
by sorry

end part1_part2_l240_240518


namespace find_number_l240_240872

variable (x : ℝ)

theorem find_number (h : 0.20 * x = 0.40 * 140 + 80) : x = 680 :=
by
  sorry

end find_number_l240_240872


namespace proposition_A_proposition_B_proposition_C_proposition_D_l240_240715

theorem proposition_A (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a ≠ 1) :=
by {
  sorry
}

theorem proposition_B : (¬ ∀ x : ℝ, x^2 + x + 1 < 0) → (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by {
  sorry
}

theorem proposition_C : ¬ ∀ x ≠ 0, x + 1 / x ≥ 2 :=
by {
  sorry
}

theorem proposition_D (m : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 2) ∧ x^2 + m * x + 4 < 0) → m < -4 :=
by {
  sorry
}

end proposition_A_proposition_B_proposition_C_proposition_D_l240_240715


namespace exists_natural_2001_digits_l240_240018

theorem exists_natural_2001_digits (N : ℕ) (hN: N = 5 * 10^2000 + 1) : 
  ∃ K : ℕ, (K = N) ∧ (N^(2001) % 10^2001 = N % 10^2001) :=
by
  sorry

end exists_natural_2001_digits_l240_240018


namespace selling_price_for_given_profit_selling_price_to_maximize_profit_l240_240942

-- Define the parameters
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_decrement_per_unit_increase : ℝ := 10

-- Define the function for monthly sales based on price increment
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrement_per_unit_increase * x

-- Define the function for selling price based on price increment
def selling_price (x : ℝ) : ℝ := initial_selling_price + x

-- Define the function for monthly profit
def monthly_profit (x : ℝ) : ℝ :=
  let total_revenue := monthly_sales x * selling_price x 
  let total_cost := monthly_sales x * cost_price
  total_revenue - total_cost

-- Problem 1: Prove the selling price when monthly profit is 8750 yuan
theorem selling_price_for_given_profit : 
  ∃ x : ℝ, monthly_profit x = 8750 ∧ (selling_price x = 75 ∨ selling_price x = 65) :=
sorry

-- Problem 2: Prove the selling price that maximizes the monthly profit
theorem selling_price_to_maximize_profit : 
  ∀ x : ℝ, monthly_profit x ≤ monthly_profit 20 ∧ selling_price 20 = 70 :=
sorry

end selling_price_for_given_profit_selling_price_to_maximize_profit_l240_240942


namespace desired_interest_rate_l240_240104

def face_value : Real := 52
def dividend_rate : Real := 0.09
def market_value : Real := 39

theorem desired_interest_rate : (dividend_rate * face_value / market_value) * 100 = 12 := by
  sorry

end desired_interest_rate_l240_240104


namespace total_time_spent_l240_240039

def timeDrivingToSchool := 20
def timeAtGroceryStore := 15
def timeFillingGas := 5
def timeAtParentTeacherNight := 70
def timeAtCoffeeShop := 30
def timeDrivingHome := timeDrivingToSchool

theorem total_time_spent : 
  timeDrivingToSchool + timeAtGroceryStore + timeFillingGas + timeAtParentTeacherNight + timeAtCoffeeShop + timeDrivingHome = 160 :=
by
  sorry

end total_time_spent_l240_240039


namespace complex_ab_value_l240_240725

open Complex

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h : i = Complex.I) (h₁ : (a + b * i) * (3 + i) = 10 + 10 * i) : a * b = 8 := 
by
  sorry

end complex_ab_value_l240_240725


namespace infinitely_many_n_l240_240616

theorem infinitely_many_n (S : Set ℕ) :
  (∀ n ∈ S, n > 0 ∧ (n ∣ 2 ^ (2 ^ n + 1) + 1) ∧ ¬ (n ∣ 2 ^ n + 1)) ∧ S.Infinite :=
sorry

end infinitely_many_n_l240_240616


namespace monotonic_increase_range_of_alpha_l240_240654

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.cos (ω * x)

theorem monotonic_increase_range_of_alpha
  (ω : ℝ) (hω : ω > 0)
  (zeros_form_ap : ∀ k : ℤ, ∃ x₀ : ℝ, f ω x₀ = 0 ∧ ∀ n : ℤ, f ω (x₀ + n * (π / 2)) = 0) :
  ∃ α : ℝ, 0 < α ∧ α < 5 * π / 12 ∧ ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ α → f ω x ≤ f ω y :=
sorry

end monotonic_increase_range_of_alpha_l240_240654


namespace base7_to_base10_conversion_l240_240550

theorem base7_to_base10_conversion (n: ℕ) (H: n = 3652) : 
  (3 * 7^3 + 6 * 7^2 + 5 * 7^1 + 2 * 7^0 = 1360) := by
  sorry

end base7_to_base10_conversion_l240_240550


namespace find_angle_D_l240_240865

variables (A B C D angle : ℝ)

-- Assumptions based on the problem statement
axiom sum_A_B : A + B = 140
axiom C_eq_D : C = D

-- The claim we aim to prove
theorem find_angle_D (h₁ : A + B = 140) (h₂: C = D): D = 20 :=
by {
    sorry 
}

end find_angle_D_l240_240865


namespace max_value_fraction_l240_240516

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + 3 * y + 2) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 17 :=
by
  sorry

end max_value_fraction_l240_240516


namespace range_of_a_l240_240158

theorem range_of_a (f : ℝ → ℝ) (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) (h_ineq : f (1 - a) < f (2 * a - 1)) : a < 2 / 3 :=
sorry

end range_of_a_l240_240158


namespace frequency_of_heads_l240_240887

theorem frequency_of_heads (n h : ℕ) (h_n : n = 100) (h_h : h = 49) : (h : ℚ) / n = 0.49 :=
by
  rw [h_n, h_h]
  norm_num

end frequency_of_heads_l240_240887


namespace min_time_needed_l240_240357

-- Define the conditions and required time for shoeing horses
def num_blacksmiths := 48
def num_horses := 60
def hooves_per_horse := 4
def time_per_hoof := 5
def total_hooves := num_horses * hooves_per_horse
def total_time_one_blacksmith := total_hooves * time_per_hoof
def min_time (num_blacksmiths : Nat) (total_time_one_blacksmith : Nat) : Nat :=
  total_time_one_blacksmith / num_blacksmiths

-- Prove that the minimum time needed is 25 minutes
theorem min_time_needed : min_time num_blacksmiths total_time_one_blacksmith = 25 :=
by
  sorry

end min_time_needed_l240_240357


namespace symm_diff_A_B_l240_240540

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | abs x < 2}

-- Define set difference
def set_diff (S T : Set ℤ) : Set ℤ := {x | x ∈ S ∧ x ∉ T}

-- Define symmetric difference
def symm_diff (S T : Set ℤ) : Set ℤ := (set_diff S T) ∪ (set_diff T S)

-- Define the expression we need to prove
theorem symm_diff_A_B : symm_diff A B = {-1, 0, 2} := by
  sorry

end symm_diff_A_B_l240_240540


namespace exponentiation_equality_l240_240936

theorem exponentiation_equality :
  3^12 * 8^12 * 3^3 * 8^8 = 24 ^ 15 * 32768 := by
  sorry

end exponentiation_equality_l240_240936


namespace quadrilateral_front_view_iff_cylinder_or_prism_l240_240967

inductive Solid
| cone : Solid
| cylinder : Solid
| triangular_pyramid : Solid
| quadrangular_prism : Solid

def has_quadrilateral_front_view (s : Solid) : Prop :=
  s = Solid.cylinder ∨ s = Solid.quadrangular_prism

theorem quadrilateral_front_view_iff_cylinder_or_prism (s : Solid) :
  has_quadrilateral_front_view s ↔ s = Solid.cylinder ∨ s = Solid.quadrangular_prism :=
by
  sorry

end quadrilateral_front_view_iff_cylinder_or_prism_l240_240967


namespace students_in_donnelly_class_l240_240385

-- Conditions
def initial_cupcakes : ℕ := 40
def cupcakes_to_delmont_class : ℕ := 18
def cupcakes_to_staff : ℕ := 4
def leftover_cupcakes : ℕ := 2

-- Question: How many students are in Mrs. Donnelly's class?
theorem students_in_donnelly_class : 
  let cupcakes_given_to_students := initial_cupcakes - (cupcakes_to_delmont_class + cupcakes_to_staff)
  let cupcakes_given_to_donnelly_class := cupcakes_given_to_students - leftover_cupcakes
  cupcakes_given_to_donnelly_class = 16 :=
by
  sorry

end students_in_donnelly_class_l240_240385


namespace total_ingredient_cups_l240_240898

def butter_flour_sugar_ratio_butter := 2
def butter_flour_sugar_ratio_flour := 5
def butter_flour_sugar_ratio_sugar := 3
def flour_used := 15

theorem total_ingredient_cups :
  butter_flour_sugar_ratio_butter + 
  butter_flour_sugar_ratio_flour + 
  butter_flour_sugar_ratio_sugar = 10 →
  flour_used / butter_flour_sugar_ratio_flour = 3 →
  6 + 15 + 9 = 30 := by
  intros
  sorry

end total_ingredient_cups_l240_240898


namespace rectangular_to_polar_coordinates_l240_240619

noncomputable def polar_coordinates_of_point (x y : ℝ) : ℝ × ℝ := sorry

theorem rectangular_to_polar_coordinates :
  polar_coordinates_of_point 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := sorry

end rectangular_to_polar_coordinates_l240_240619


namespace point_on_parallel_line_with_P_l240_240933

-- Definitions
def is_on_parallel_line_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.snd = Q.snd

theorem point_on_parallel_line_with_P :
  let P := (3, -2)
  let D := (-3, -2)
  is_on_parallel_line_x_axis P D :=
by
  sorry

end point_on_parallel_line_with_P_l240_240933


namespace horse_catch_up_l240_240270

theorem horse_catch_up :
  ∀ (x : ℕ), (240 * x = 150 * (x + 12)) → x = 20 :=
by
  intros x h
  have : 240 * x = 150 * x + 1800 := by sorry
  have : 240 * x - 150 * x = 1800 := by sorry
  have : 90 * x = 1800 := by sorry
  have : x = 1800 / 90 := by sorry
  have : x = 20 := by sorry
  exact this

end horse_catch_up_l240_240270


namespace sequence_bound_l240_240904

-- Definitions and assumptions based on the conditions
def valid_sequence (a : ℕ → ℕ) (N : ℕ) (m : ℕ) :=
  (1 ≤ a 1) ∧ (a m ≤ N) ∧ (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) ∧ 
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N)

-- The main theorem to prove
theorem sequence_bound (a : ℕ → ℕ) (N : ℕ) (m : ℕ) 
  (h : valid_sequence a N m) : m ≤ 2 * Nat.floor (Real.sqrt N) :=
sorry

end sequence_bound_l240_240904


namespace Seokjin_total_problems_l240_240573

theorem Seokjin_total_problems (initial_problems : ℕ) (additional_problems : ℕ)
  (h1 : initial_problems = 12) (h2 : additional_problems = 7) :
  initial_problems + additional_problems = 19 :=
by
  sorry

end Seokjin_total_problems_l240_240573


namespace surface_area_difference_l240_240314

theorem surface_area_difference
  (larger_cube_volume : ℝ)
  (num_smaller_cubes : ℝ)
  (smaller_cube_volume : ℝ)
  (h1 : larger_cube_volume = 125)
  (h2 : num_smaller_cubes = 125)
  (h3 : smaller_cube_volume = 1) :
  (6 * (smaller_cube_volume)^(2/3) * num_smaller_cubes) - (6 * (larger_cube_volume)^(2/3)) = 600 :=
by {
  sorry
}

end surface_area_difference_l240_240314


namespace percent_of_flowers_are_daisies_l240_240764

-- Definitions for the problem
def total_flowers (F : ℕ) := F
def blue_flowers (F : ℕ) := (7/10) * F
def red_flowers (F : ℕ) := (3/10) * F
def blue_tulips (F : ℕ) := (1/2) * (7/10) * F
def blue_daisies (F : ℕ) := (7/10) * F - (1/2) * (7/10) * F
def red_daisies (F : ℕ) := (2/3) * (3/10) * F
def total_daisies (F : ℕ) := blue_daisies F + red_daisies F
def percentage_of_daisies (F : ℕ) := (total_daisies F / F) * 100

-- The statement to prove
theorem percent_of_flowers_are_daisies (F : ℕ) (hF : F > 0) :
  percentage_of_daisies F = 55 := by
  sorry

end percent_of_flowers_are_daisies_l240_240764


namespace proof_problem_1_proof_problem_2_l240_240358

noncomputable def problem_1 (a b : ℝ) : Prop :=
  ((2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3))) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6)

noncomputable def problem_2 : Prop :=
  ((2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 2^(3/4 - 1) - (-2005)^0) = 100

theorem proof_problem_1 (a b : ℝ) : problem_1 a b := 
  sorry

theorem proof_problem_2 : problem_2 := 
  sorry

end proof_problem_1_proof_problem_2_l240_240358


namespace total_weight_of_watermelons_l240_240546

theorem total_weight_of_watermelons (w1 w2 : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) :
  w1 + w2 = 14.02 :=
by
  sorry

end total_weight_of_watermelons_l240_240546


namespace car_avg_speed_l240_240839

def avg_speed_problem (d1 d2 t : ℕ) : ℕ :=
  (d1 + d2) / t

theorem car_avg_speed (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 70) (h2 : d2 = 90) (ht : t = 2) :
  avg_speed_problem d1 d2 t = 80 := by
  sorry

end car_avg_speed_l240_240839


namespace scientific_notation_of_12000000000_l240_240813

theorem scientific_notation_of_12000000000 :
  12000000000 = 1.2 * 10^10 :=
by sorry

end scientific_notation_of_12000000000_l240_240813


namespace age_ratio_in_8_years_l240_240897

-- Define the conditions
variables (s l : ℕ) -- Sam's and Leo's current ages

def condition1 := s - 4 = 2 * (l - 4)
def condition2 := s - 10 = 3 * (l - 10)

-- Define the final problem
theorem age_ratio_in_8_years (h1 : condition1 s l) (h2 : condition2 s l) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (l + x) = 3 / 2 :=
sorry

end age_ratio_in_8_years_l240_240897


namespace weighted_mean_calculation_l240_240322

/-- Prove the weighted mean of the numbers 16, 28, and 45 with weights 2, 3, and 5 is 34.1 -/
theorem weighted_mean_calculation :
  let numbers := [16, 28, 45]
  let weights := [2, 3, 5]
  let total_weight := (2 + 3 + 5 : ℝ)
  let weighted_sum := ((16 * 2 + 28 * 3 + 45 * 5) : ℝ)
  (weighted_sum / total_weight) = 34.1 :=
by
  -- We only state the theorem without providing the proof
  sorry

end weighted_mean_calculation_l240_240322


namespace abs_ineq_solution_set_l240_240025

theorem abs_ineq_solution_set (x : ℝ) :
  |x - 5| + |x + 3| ≥ 10 ↔ x ≤ -4 ∨ x ≥ 6 :=
by
  sorry

end abs_ineq_solution_set_l240_240025


namespace first_number_in_list_is_55_l240_240446

theorem first_number_in_list_is_55 : 
  ∀ (x : ℕ), (55 + 57 + 58 + 59 + 62 + 62 + 63 + 65 + x) / 9 = 60 → x = 65 → 55 = 55 :=
by
  intros x avg_cond x_is_65
  rfl

end first_number_in_list_is_55_l240_240446


namespace coplanar_k_values_l240_240597

noncomputable def coplanar_lines_possible_k (k : ℝ) : Prop :=
  ∃ (t u : ℝ), (2 + t = 1 + k * u) ∧ (3 + t = 4 + 2 * u) ∧ (4 - k * t = 5 + u)

theorem coplanar_k_values :
  ∀ k : ℝ, coplanar_lines_possible_k k ↔ (k = 0 ∨ k = -3) :=
by
  sorry

end coplanar_k_values_l240_240597


namespace Rihanna_money_left_l240_240627

theorem Rihanna_money_left (initial_money mango_count juice_count mango_price juice_price : ℕ)
  (h_initial : initial_money = 50)
  (h_mango_count : mango_count = 6)
  (h_juice_count : juice_count = 6)
  (h_mango_price : mango_price = 3)
  (h_juice_price : juice_price = 3) :
  initial_money - (mango_count * mango_price + juice_count * juice_price) = 14 :=
sorry

end Rihanna_money_left_l240_240627


namespace matrix_solution_property_l240_240405

theorem matrix_solution_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
    (h : N = Matrix.of ![![2, 4], ![1, 4]]) :
    N ^ 4 - 5 * N ^ 3 + 9 * N ^ 2 - 5 * N = Matrix.of ![![6, 12], ![3, 6]] :=
by 
  sorry

end matrix_solution_property_l240_240405


namespace annual_interest_rate_l240_240560

theorem annual_interest_rate (r : ℝ): 
  (1000 * r * 4.861111111111111 + 1400 * r * 4.861111111111111 = 350) → 
  r = 0.03 :=
sorry

end annual_interest_rate_l240_240560


namespace find_x_l240_240432

theorem find_x (x : ℝ) 
  (h1 : x = (1 / x * -x) - 5) 
  (h2 : x^2 - 3 * x + 2 ≥ 0) : 
  x = -6 := 
sorry

end find_x_l240_240432


namespace correct_option_l240_240959

theorem correct_option : ∀ (x y : ℝ), 10 * x * y - 10 * y * x = 0 :=
by 
  intros x y
  sorry

end correct_option_l240_240959


namespace exists_natural_pairs_a_exists_natural_pair_b_l240_240248

open Nat

-- Part (a) Statement
theorem exists_natural_pairs_a (x y : ℕ) :
  x^2 - y^2 = 105 → (x, y) = (53, 52) ∨ (x, y) = (19, 16) ∨ (x, y) = (13, 8) ∨ (x, y) = (11, 4) :=
sorry

-- Part (b) Statement
theorem exists_natural_pair_b (x y : ℕ) :
  2*x^2 + 5*x*y - 12*y^2 = 28 → (x, y) = (8, 5) :=
sorry

end exists_natural_pairs_a_exists_natural_pair_b_l240_240248


namespace diagonals_in_convex_polygon_l240_240187

-- Define the number of sides for the polygon
def polygon_sides : ℕ := 15

-- The main theorem stating the number of diagonals in a convex polygon with 15 sides
theorem diagonals_in_convex_polygon : polygon_sides = 15 → ∃ d : ℕ, d = 90 :=
by
  intro h
  -- sorry is a placeholder for the proof
  sorry

end diagonals_in_convex_polygon_l240_240187


namespace right_triangle_c_l240_240569

theorem right_triangle_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 4)
  (h3 : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2)) :
  c = 5 ∨ c = Real.sqrt 7 :=
by
  -- Proof omitted
  sorry

end right_triangle_c_l240_240569


namespace solve_for_x_l240_240273

theorem solve_for_x : ∃ x : ℤ, 24 - 5 = 3 + x ∧ x = 16 :=
by
  sorry

end solve_for_x_l240_240273


namespace equivalent_modulo_l240_240683

theorem equivalent_modulo :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] ∧ n = 29 := 
by
  sorry

end equivalent_modulo_l240_240683


namespace logarithmic_inequality_l240_240051

theorem logarithmic_inequality : 
  (a = Real.log 9 / Real.log 2) →
  (b = Real.log 27 / Real.log 3) →
  (c = Real.log 15 / Real.log 5) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  sorry

end logarithmic_inequality_l240_240051


namespace simplify_and_evaluate_expression_l240_240976

theorem simplify_and_evaluate_expression (x : ℝ) (h : x^2 - 2 * x - 2 = 0) :
    ( ( (x - 1)/x - (x - 2)/(x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) = 1 / 2 ) :=
by
    -- sorry to skip the proof
    sorry

end simplify_and_evaluate_expression_l240_240976


namespace sum_of_nine_consecutive_parity_l240_240476

theorem sum_of_nine_consecutive_parity (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) % 2 = n % 2 := 
  sorry

end sum_of_nine_consecutive_parity_l240_240476


namespace innovation_contribution_l240_240499

variable (material : String)
variable (contribution : String → Prop)
variable (A B C D : Prop)

-- Conditions
axiom condA : contribution material → A
axiom condB : contribution material → ¬B
axiom condC : contribution material → ¬C
axiom condD : contribution material → ¬D

-- The problem statement
theorem innovation_contribution :
  contribution material → A :=
by
  -- dummy proof as placeholder
  sorry

end innovation_contribution_l240_240499


namespace lauren_time_8_miles_l240_240373

-- Conditions
def time_alex_run_6_miles : ℕ := 36
def time_lauren_run_5_miles : ℕ := time_alex_run_6_miles / 3
def time_per_mile_lauren : ℚ := time_lauren_run_5_miles / 5

-- Proof statement
theorem lauren_time_8_miles : 8 * time_per_mile_lauren = 19.2 := by
  sorry

end lauren_time_8_miles_l240_240373


namespace quadratic_inequality_solution_l240_240334

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x - 21 ≤ 0 ↔ -3 ≤ x ∧ x ≤ 7 :=
sorry

end quadratic_inequality_solution_l240_240334


namespace range_of_x_l240_240355

theorem range_of_x {a : ℝ} : 
  (∀ a : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (x = 0 ∨ x = -2) :=
by sorry

end range_of_x_l240_240355


namespace maximum_n_l240_240928

def number_of_trapezoids (n : ℕ) : ℕ := n * (n - 3) * (n - 2) * (n - 1) / 24

theorem maximum_n (n : ℕ) (h : number_of_trapezoids n ≤ 2012) : n ≤ 26 :=
by
  sorry

end maximum_n_l240_240928


namespace problem_statement_l240_240668

theorem problem_statement : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3) ^ 2 = 4 - 2 * Real.sqrt 6 := 
by 
  sorry

end problem_statement_l240_240668


namespace calculate_expression_l240_240686

variable (a : ℝ)

theorem calculate_expression : (-a) ^ 2 * (-a ^ 5) ^ 4 / a ^ 12 * (-2 * a ^ 4) = -2 * a ^ 14 := 
by sorry

end calculate_expression_l240_240686


namespace range_of_m_l240_240634

noncomputable def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2 * x - 8 < 0}
def U : Set ℝ := Set.univ
def CU_M (m : ℝ) : Set ℝ := {x | x < -m}
def empty_intersection (m : ℝ) : Prop := (CU_M m ∩ N = ∅)

theorem range_of_m (m : ℝ) : empty_intersection m → m ≥ 2 := by
  sorry

end range_of_m_l240_240634


namespace subset_condition_l240_240561

variable {U : Type}
variables (P Q : Set U)

theorem subset_condition (h : P ∩ Q = P) : ∀ x : U, x ∉ Q → x ∉ P :=
by {
  sorry
}

end subset_condition_l240_240561


namespace Chris_age_proof_l240_240917

theorem Chris_age_proof (m c : ℕ) (h1 : c = 3 * m - 22) (h2 : c + m = 70) : c = 47 := by
  sorry

end Chris_age_proof_l240_240917


namespace exists_polynomial_triangle_property_l240_240675

noncomputable def f (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem exists_polynomial_triangle_property :
  ∀ (x y z : ℝ), (f x y z > 0 ↔ (|x| + |y| > |z| ∧ |y| + |z| > |x| ∧ |z| + |x| > |y|)) :=
sorry

end exists_polynomial_triangle_property_l240_240675


namespace reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l240_240011

-- Definition: reversing a deck of n cards in k operations
def can_reverse_deck (n k : ℕ) : Prop := sorry -- Placeholder definition

-- Proof Part (a)
theorem reverse_9_in_5_operations :
  can_reverse_deck 9 5 :=
sorry

-- Proof Part (b)
theorem reverse_52_in_27_operations :
  can_reverse_deck 52 27 :=
sorry

-- Proof Part (c)
theorem not_reverse_52_in_17_operations :
  ¬can_reverse_deck 52 17 :=
sorry

-- Proof Part (d)
theorem not_reverse_52_in_26_operations :
  ¬can_reverse_deck 52 26 :=
sorry

end reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l240_240011


namespace even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l240_240635

theorem even_ngon_parallel_edges (n : ℕ) (h : n % 2 = 0) :
  ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

theorem odd_ngon_no_two_parallel_edges (n : ℕ) (h : n % 2 = 1) :
  ¬ ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

end even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l240_240635


namespace first_shipment_weight_l240_240548

variable (first_shipment : ℕ)
variable (total_dishes_made : ℕ := 13)
variable (couscous_per_dish : ℕ := 5)
variable (second_shipment : ℕ := 45)
variable (same_day_shipment : ℕ := 13)

theorem first_shipment_weight :
  13 * 5 = 65 → second_shipment ≠ first_shipment → 
  first_shipment + same_day_shipment = 65 →
  first_shipment = 65 :=
by
  sorry

end first_shipment_weight_l240_240548


namespace root_condition_l240_240937

-- Let f(x) = x^2 + ax + a^2 - a - 2
noncomputable def f (a x : ℝ) : ℝ := x^2 + a * x + a^2 - a - 2

theorem root_condition (a : ℝ) (h1 : ∀ ζ : ℝ, (ζ > 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0) ∧ (ζ < 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0)) :
  -1 < a ∧ a < 1 :=
sorry

end root_condition_l240_240937


namespace purple_marble_probability_l240_240076

theorem purple_marble_probability (blue green : ℝ) (p : ℝ) 
  (h_blue : blue = 0.25)
  (h_green : green = 0.4)
  (h_sum : blue + green + p = 1) : p = 0.35 :=
by
  sorry

end purple_marble_probability_l240_240076


namespace nonneg_or_nonpos_l240_240160

theorem nonneg_or_nonpos (n : ℕ) (h : n ≥ 2) (c : Fin n → ℝ)
  (h_eq : (n - 1) * (Finset.univ.sum (fun i => c i ^ 2)) = (Finset.univ.sum c) ^ 2) :
  (∀ i, c i ≥ 0) ∨ (∀ i, c i ≤ 0) := 
  sorry

end nonneg_or_nonpos_l240_240160


namespace smallest_n_l240_240401

def is_perfect_fourth (m : ℕ) : Prop := ∃ x : ℕ, m = x^4
def is_perfect_fifth (m : ℕ) : Prop := ∃ y : ℕ, m = y^5

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ is_perfect_fourth (3 * n) ∧ is_perfect_fifth (2 * n) ∧ n = 6912 :=
by {
  sorry
}

end smallest_n_l240_240401


namespace original_water_amount_in_mixture_l240_240861

-- Define heat calculations and conditions
def latentHeatOfFusionIce : ℕ := 80       -- Latent heat of fusion for ice in cal/g
def initialTempWaterAdded : ℕ := 20      -- Initial temperature of added water in °C
def finalTempMixture : ℕ := 5            -- Final temperature of the mixture in °C
def specificHeatWater : ℕ := 1           -- Specific heat of water in cal/g°C

-- Define the known parameters of the problem
def totalMass : ℕ := 250               -- Total mass of the initial mixture in grams
def addedMassWater : ℕ := 1000         -- Mass of added water in grams
def initialTempMixtureIceWater : ℕ := 0  -- Initial temperature of the ice-water mixture in °C

-- Define the equation that needs to be solved
theorem original_water_amount_in_mixture (x : ℝ) :
  (250 - x) * 80 + (250 - x) * 5 + x * 5 = 15000 →
  x = 90.625 :=
by
  intro h
  sorry

end original_water_amount_in_mixture_l240_240861


namespace triangle_area_correct_l240_240208

-- Define the points (vertices) of the triangle
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (8, -3)
def point3 : ℝ × ℝ := (2, 7)

-- Function to calculate the area of the triangle given three points (shoelace formula)
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.2 * C.1 - C.2 * A.1 - A.2 * B.1)

-- Prove that the area of the triangle with the given vertices is 18 square units
theorem triangle_area_correct : triangle_area point1 point2 point3 = 18 :=
by
  sorry

end triangle_area_correct_l240_240208


namespace induction_step_l240_240438

theorem induction_step (k : ℕ) : ((k + 1 + k) * (k + 1 + k + 1) / (k + 1)) = 2 * (2 * k + 1) := by
  sorry

end induction_step_l240_240438


namespace simplify_expression_l240_240210

theorem simplify_expression (a b : ℝ) (h : a ≠ b) : 
  ((a^3 - b^3) / (a * b)) - ((a * b^2 - b^3) / (a * b - a^3)) = (2 * a * (a - b)) / b :=
by
  sorry

end simplify_expression_l240_240210


namespace markers_per_box_l240_240167

theorem markers_per_box
  (students : ℕ) (boxes : ℕ) (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ) (last_group_markers : ℕ)
  (h_students : students = 30)
  (h_boxes : boxes = 22)
  (h_group1_students : group1_students = 10)
  (h_group1_markers : group1_markers = 2)
  (h_group2_students : group2_students = 15)
  (h_group2_markers : group2_markers = 4)
  (h_last_group_markers : last_group_markers = 6) :
  (110 = students * ((group1_students * group1_markers + group2_students * group2_markers + (students - group1_students - group2_students) * last_group_markers)) / boxes) :=
by
  sorry

end markers_per_box_l240_240167


namespace part_a_l240_240957

theorem part_a : (2^41 + 1) % 83 = 0 :=
  sorry

end part_a_l240_240957


namespace geom_progression_lines_common_point_l240_240409

theorem geom_progression_lines_common_point
  (a c b : ℝ) (r : ℝ)
  (h_geom_prog : c = a * r ∧ b = a * r^2) :
  ∃ (P : ℝ × ℝ), ∀ (a c b : ℝ), c = a * r ∧ b = a * r^2 → (P = (0, 0) ∧ a ≠ 0) :=
by
  sorry

end geom_progression_lines_common_point_l240_240409


namespace sum_of_altitudes_less_than_sum_of_sides_l240_240578

-- Define a triangle with sides and altitudes properties
structure Triangle :=
(A B C : Point)
(a b c : ℝ)
(m_a m_b m_c : ℝ)
(sides : a + b > c ∧ b + c > a ∧ c + a > b) -- Triangle Inequality

axiom altitude_property (T : Triangle) :
  T.m_a < T.b ∧ T.m_b < T.c ∧ T.m_c < T.a

-- The theorem to prove
theorem sum_of_altitudes_less_than_sum_of_sides (T : Triangle) :
  T.m_a + T.m_b + T.m_c < T.a + T.b + T.c :=
sorry

end sum_of_altitudes_less_than_sum_of_sides_l240_240578


namespace problem_statement_l240_240524
noncomputable def not_divisible (n : ℕ) : Prop := ∃ k : ℕ, (5^n - 3^n) = (2^n + 65) * k
theorem problem_statement (n : ℕ) (h : 0 < n) : ¬ not_divisible n := sorry

end problem_statement_l240_240524


namespace intersection_point_l240_240033

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 3) / (2 * x - 6)
noncomputable def g (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) (h_asymp : ¬(2 = 0) ∧ (a ≠ 0 ∨ b ≠ 0)) (h_perpendicular : True) (h_y_intersect : g a b c 0 = 1) (h_intersects : f (-1) = g a b c (-1)):
  f 1 = 0 :=
by
  dsimp [f, g] at *
  sorry

end intersection_point_l240_240033


namespace unique_sequence_and_a_2002_l240_240275

-- Define the sequence (a_n)
noncomputable def a : ℕ → ℕ := -- define the correct sequence based on conditions
  -- we would define a such as in the constructive steps in the solution, but here's a placeholder
  sorry

-- Prove the uniqueness and finding a_2002
theorem unique_sequence_and_a_2002 :
  (∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k) ∧ a 2002 = 1227132168 :=
by
  sorry

end unique_sequence_and_a_2002_l240_240275


namespace find_smaller_number_l240_240746

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  sorry

end find_smaller_number_l240_240746


namespace child_current_height_l240_240776

variable (h_last_visit : ℝ) (h_grown : ℝ)

-- Conditions
def last_height (h_last_visit : ℝ) := h_last_visit = 38.5
def height_grown (h_grown : ℝ) := h_grown = 3

-- Theorem statement
theorem child_current_height (h_last_visit h_grown : ℝ) 
    (h_last : last_height h_last_visit) 
    (h_grow : height_grown h_grown) : 
    h_last_visit + h_grown = 41.5 :=
by
  sorry

end child_current_height_l240_240776


namespace general_term_formula_sum_inequality_l240_240919

noncomputable def a (n : ℕ) : ℝ := if n > 0 then (-1)^(n-1) * 3 / 2^n else 0

noncomputable def S (n : ℕ) : ℝ := if n > 0 then 1 - (-1/2)^n else 0

theorem general_term_formula (n : ℕ) (hn : n > 0) :
  a n = (-1)^(n-1) * (3/2^n) :=
by sorry

theorem sum_inequality (n : ℕ) (hn : n > 0) :
  S n + 1 / S n ≤ 13 / 6 :=
by sorry

end general_term_formula_sum_inequality_l240_240919


namespace remaining_fruits_l240_240272

theorem remaining_fruits (initial_apples initial_oranges initial_mangoes taken_apples twice_taken_apples taken_mangoes) : 
  initial_apples = 7 → 
  initial_oranges = 8 → 
  initial_mangoes = 15 → 
  taken_apples = 2 → 
  twice_taken_apples = 2 * taken_apples → 
  taken_mangoes = 2 * initial_mangoes / 3 → 
  initial_apples - taken_apples + initial_oranges - twice_taken_apples + initial_mangoes - taken_mangoes = 14 :=
by
  sorry

end remaining_fruits_l240_240272


namespace michelle_oranges_l240_240766

theorem michelle_oranges (x : ℕ) 
  (h1 : x - x / 3 - 5 = 7) : x = 18 :=
by
  -- We would normally provide the proof here, but it's omitted according to the instructions.
  sorry

end michelle_oranges_l240_240766


namespace find_a_l240_240633

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) :
  (2 * x₁ + 1 = 3) →
  (2 - (a - x₂) / 3 = 1) →
  (x₁ = x₂) →
  a = 4 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_l240_240633


namespace problem_1_problem_2_l240_240232

theorem problem_1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : (a + b) * (a^5 + b^5) ≥ 4 :=
sorry

theorem problem_2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : a + b ≤ 2 :=
sorry

end problem_1_problem_2_l240_240232


namespace number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l240_240118

-- Definition for the number of sides in the polygon
def n : ℕ := 150

-- Definition of the formula for the number of diagonals
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the formula for the sum of interior angles
def sum_of_interior_angles (n : ℕ) : ℕ :=
  180 * (n - 2)

-- Theorem statements to be proved
theorem number_of_diagonals_is_correct : number_of_diagonals n = 11025 := sorry

theorem sum_of_interior_angles_is_correct : sum_of_interior_angles n = 26640 := sorry

end number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l240_240118


namespace problem1_problem2_l240_240563

-- Definitions of sets A and B
def setA : Set ℝ := { x | x^2 - 8 * x + 15 = 0 }
def setB (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

-- Problem 1: If a = 1/5, B is a subset of A.
theorem problem1 : setB (1 / 5) ⊆ setA := sorry

-- Problem 2: If A ∩ B = B, then C = {0, 1/3, 1/5}.
def setC : Set ℝ := { a | a = 0 ∨ a = 1 / 3 ∨ a = 1 / 5 }

theorem problem2 (a : ℝ) : (setA ∩ setB a = setB a) ↔ (a ∈ setC) := sorry

end problem1_problem2_l240_240563


namespace range_of_a_l240_240515

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 1 then x else a * x^2 + 2 * x

theorem range_of_a (R : Set ℝ) :
  (∀ x : ℝ, f x a ∈ R) → (a ∈ Set.Icc (-1 : ℝ) 0) :=
sorry

end range_of_a_l240_240515


namespace find_first_term_l240_240421

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l240_240421


namespace find_k_l240_240500

theorem find_k (a k : ℝ) (h : a ≠ 0) (h1 : 3 * a + a = -12)
  (h2 : (3 * a) * a = k) : k = 27 :=
by
  sorry

end find_k_l240_240500


namespace prove_a2_minus_b2_l240_240994

theorem prove_a2_minus_b2 : 
  ∀ (a b : ℚ), 
  a + b = 9 / 17 ∧ a - b = 1 / 51 → a^2 - b^2 = 3 / 289 :=
by
  intros a b h
  cases' h
  sorry

end prove_a2_minus_b2_l240_240994


namespace gina_order_rose_cups_l240_240697

theorem gina_order_rose_cups 
  (rose_cups_per_hour : ℕ) 
  (lily_cups_per_hour : ℕ) 
  (total_lily_cups_order : ℕ) 
  (total_pay : ℕ) 
  (pay_per_hour : ℕ) 
  (total_hours_worked : ℕ) 
  (hours_spent_with_lilies : ℕ)
  (hours_spent_with_roses : ℕ) 
  (rose_cups_order : ℕ) :
  rose_cups_per_hour = 6 →
  lily_cups_per_hour = 7 →
  total_lily_cups_order = 14 →
  total_pay = 90 →
  pay_per_hour = 30 →
  total_hours_worked = total_pay / pay_per_hour →
  hours_spent_with_lilies = total_lily_cups_order / lily_cups_per_hour →
  hours_spent_with_roses = total_hours_worked - hours_spent_with_lilies →
  rose_cups_order = rose_cups_per_hour * hours_spent_with_roses →
  rose_cups_order = 6 := 
by
  sorry

end gina_order_rose_cups_l240_240697


namespace solve_eq_l240_240339

theorem solve_eq : ∃ x : ℝ, 6 * x - 4 * x = 380 - 10 * (x + 2) ∧ x = 30 := 
by
  sorry

end solve_eq_l240_240339


namespace fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l240_240665

def fractional_eq_solution_1 (x : ℝ) : Prop :=
  x + 5 / x = -6

theorem fractional_eq_solutions_1 : fractional_eq_solution_1 (-1) ∧ fractional_eq_solution_1 (-5) := sorry

def fractional_eq_solution_2 (x : ℝ) : Prop :=
  x - 3 / x = 4

theorem fractional_eq_reciprocal_sum
  (m n : ℝ) (h₀ : fractional_eq_solution_2 m) (h₁ : fractional_eq_solution_2 n) :
  m * n = -3 → m + n = 4 → (1 / m + 1 / n = -4 / 3) := sorry

def fractional_eq_solution_3 (x : ℝ) (a : ℝ) : Prop :=
  x + (a^2 + 2 * a) / (x + 1) = 2 * a + 1

theorem fractional_eq_solution_diff_square (a : ℝ) (h₀ : a ≠ 0)
  (x1 x2 : ℝ) (hx1 : fractional_eq_solution_3 x1 a) (hx2 : fractional_eq_solution_3 x2 a) :
  x1 + 1 = a → x2 + 1 = a + 2 → (x1 - x2) ^ 2 = 4 := sorry

end fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l240_240665


namespace fraction_of_work_left_l240_240320

variable (A_work_days : ℕ) (B_work_days : ℕ) (work_days_together: ℕ)

theorem fraction_of_work_left (hA : A_work_days = 15) (hB : B_work_days = 20) (hT : work_days_together = 4):
  (1 - 4 * (1 / 15 + 1 / 20)) = (8 / 15) :=
sorry

end fraction_of_work_left_l240_240320


namespace recipe_flour_amount_l240_240221

theorem recipe_flour_amount
  (cups_of_sugar : ℕ) (cups_of_salt : ℕ) (cups_of_flour_added : ℕ)
  (additional_cups_of_flour : ℕ)
  (h1 : cups_of_sugar = 2)
  (h2 : cups_of_salt = 80)
  (h3 : cups_of_flour_added = 7)
  (h4 : additional_cups_of_flour = cups_of_sugar + 1) :
  cups_of_flour_added + additional_cups_of_flour = 10 :=
by {
  sorry
}

end recipe_flour_amount_l240_240221


namespace domain_of_function_l240_240431

theorem domain_of_function :
  { x : ℝ // (6 - x - x^2) > 0 } = { x : ℝ // -3 < x ∧ x < 2 } :=
by
  sorry

end domain_of_function_l240_240431


namespace fraction_evaluation_l240_240162

theorem fraction_evaluation : (20 + 24) / (20 - 24) = -11 := by
  sorry

end fraction_evaluation_l240_240162


namespace max_mark_cells_l240_240717

theorem max_mark_cells (n : Nat) (grid : Fin n → Fin n → Bool) :
  (∀ i : Fin n, ∃ j : Fin n, grid i j = true) ∧ 
  (∀ j : Fin n, ∃ i : Fin n, grid i j = true) ∧ 
  (∀ (x1 x2 y1 y2 : Fin n), (x1 ≤ x2 ∧ y1 ≤ y2 ∧ (x2.1 - x1.1 + 1) * (y2.1 - y1.1 + 1) ≥ n) → 
   ∃ i : Fin n, ∃ j : Fin n, grid i j = true ∧ x1 ≤ i ∧ i ≤ x2 ∧ y1 ≤ j ∧ j ≤ y2) → 
  (n ≤ 7) := sorry

end max_mark_cells_l240_240717


namespace smallest_possible_value_l240_240430

theorem smallest_possible_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (⌊(a + b + c) / d⌋ + ⌊(a + b + d) / c⌋ + ⌊(a + c + d) / b⌋ + ⌊(b + c + d) / a⌋) ≥ 8 :=
sorry

end smallest_possible_value_l240_240430


namespace correct_statements_for_sequence_l240_240182

theorem correct_statements_for_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  -- Statement 1
  (S_n = n^2 + n → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 2
  (S_n = 2^n - 1 → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1)) ∧
  -- Statement 3
  (∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1) → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 4
  (¬(∀ n, n ≥ 2 → a n^2 = a (n + 1) * a (n - 1) → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1))) :=
sorry

end correct_statements_for_sequence_l240_240182


namespace fish_count_and_total_l240_240846

-- Definitions of each friend's number of fish
def max_fish : ℕ := 6
def sam_fish : ℕ := 3 * max_fish
def joe_fish : ℕ := 9 * sam_fish
def harry_fish : ℕ := 5 * joe_fish

-- Total number of fish for all friends combined
def total_fish : ℕ := max_fish + sam_fish + joe_fish + harry_fish

-- The theorem stating the problem and corresponding solution
theorem fish_count_and_total :
  max_fish = 6 ∧
  sam_fish = 3 * max_fish ∧
  joe_fish = 9 * sam_fish ∧
  harry_fish = 5 * joe_fish ∧
  total_fish = (max_fish + sam_fish + joe_fish + harry_fish) :=
by
  repeat { sorry }

end fish_count_and_total_l240_240846


namespace remainder_of_fractions_l240_240177

theorem remainder_of_fractions : 
  ∀ (x y : ℚ), x = 5/7 → y = 3/4 → (x - y * ⌊x / y⌋) = 5/7 :=
by
  intros x y hx hy
  rw [hx, hy]
  -- Additional steps can be filled in here, if continuing with the proof.
  sorry

end remainder_of_fractions_l240_240177


namespace smallest_integer_ends_3_divisible_5_l240_240256

theorem smallest_integer_ends_3_divisible_5 : ∃ n : ℕ, (53 = n ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m % 10 = 3) ∧ (m % 5 = 0) → 53 ≤ m) :=
by {
  sorry
}

end smallest_integer_ends_3_divisible_5_l240_240256


namespace boat_distance_against_stream_l240_240637

variable (v_s : ℝ)
variable (effective_speed_stream : ℝ := 15)
variable (speed_still_water : ℝ := 10)
variable (distance_along_stream : ℝ := 15)

theorem boat_distance_against_stream : 
  distance_along_stream / effective_speed_stream = 1 ∧ effective_speed_stream = speed_still_water + v_s →
  10 - v_s = 5 :=
by
  intros
  sorry

end boat_distance_against_stream_l240_240637


namespace inequality_sufficient_condition_l240_240622

theorem inequality_sufficient_condition (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (x+1)/(x-1) > 2 :=
by
  sorry

end inequality_sufficient_condition_l240_240622


namespace fraction_expression_l240_240141

theorem fraction_expression : (1 / 3) ^ 3 * (1 / 8) = 1 / 216 :=
by
  sorry

end fraction_expression_l240_240141


namespace cos_135_eq_neg_sqrt2_div_2_l240_240606

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l240_240606


namespace triangle_lattice_points_l240_240165

theorem triangle_lattice_points :
  ∀ (A B C : ℕ) (AB AC BC : ℕ), 
    AB = 2016 → AC = 1533 → BC = 1533 → 
    ∃ lattice_points: ℕ, lattice_points = 1165322 := 
by
  sorry

end triangle_lattice_points_l240_240165


namespace find_x_l240_240639

theorem find_x (x : ℕ) (hcf lcm : ℕ):
  (hcf = Nat.gcd x 18) → 
  (lcm = Nat.lcm x 18) → 
  (lcm - hcf = 120) → 
  x = 42 := 
by
  sorry

end find_x_l240_240639


namespace janet_spends_more_on_piano_l240_240133

-- Condition definitions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℝ := 52

-- Calculations based on conditions
def weekly_cost_clarinet : ℝ := clarinet_hourly_rate * clarinet_hours_per_week
def weekly_cost_piano : ℝ := piano_hourly_rate * piano_hours_per_week
def weekly_difference : ℝ := weekly_cost_piano - weekly_cost_clarinet
def yearly_difference : ℝ := weekly_difference * weeks_per_year

theorem janet_spends_more_on_piano : yearly_difference = 1040 := by
  sorry 

end janet_spends_more_on_piano_l240_240133


namespace min_value_of_sum_squares_l240_240920

noncomputable def min_value_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : ℝ :=
  y1^2 + y2^2 + y3^2

theorem min_value_of_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : 
  min_value_sum_squares y1 y2 y3 h1 h2 h3 h4 = 14400 / 29 := 
sorry

end min_value_of_sum_squares_l240_240920


namespace geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l240_240879

-- Define geometric body type
inductive GeometricBody
  | rectangularPrism
  | cylinder

-- Define the condition where both front and left views are rectangles
def hasRectangularViews (body : GeometricBody) : Prop :=
  body = GeometricBody.rectangularPrism ∨ body = GeometricBody.cylinder

-- The theorem statement
theorem geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder (body : GeometricBody) :
  hasRectangularViews body :=
sorry

end geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l240_240879


namespace initial_total_quantity_l240_240262

theorem initial_total_quantity(milk_ratio water_ratio : ℕ) (W : ℕ) (x : ℕ) (h1 : milk_ratio = 3) (h2 : water_ratio = 1) (h3 : W = 100) (h4 : 3 * x / (x + 100) = 1 / 3) :
    4 * x = 50 :=
by
  sorry

end initial_total_quantity_l240_240262


namespace determine_conflicting_pairs_l240_240558

structure EngineerSetup where
  n : ℕ
  barrels : Fin (2 * n) → Reactant
  conflicts : Fin n → (Reactant × Reactant)

def testTubeBurst (r1 r2 : Reactant) (conflicts : Fin n → (Reactant × Reactant)) : Prop :=
  ∃ i, conflicts i = (r1, r2) ∨ conflicts i = (r2, r1)

theorem determine_conflicting_pairs (setup : EngineerSetup) :
  ∃ pairs : Fin n → (Reactant × Reactant),
  (∀ i, pairs i ∈ { p | ∃ j, setup.conflicts j = p ∨ setup.conflicts j = (p.snd, p.fst) }) ∧
  (∀ i j, i ≠ j → pairs i ≠ pairs j) := 
sorry

end determine_conflicting_pairs_l240_240558


namespace find_side_y_l240_240217

noncomputable def side_length_y : ℝ :=
  let AB := 10 / Real.sqrt 2
  let AD := 10
  let CD := AD / 2
  CD * Real.sqrt 3

theorem find_side_y : side_length_y = 5 * Real.sqrt 3 := by
  let AB : ℝ := 10 / Real.sqrt 2
  let AD : ℝ := 10
  let CD : ℝ := AD / 2
  have h1 : CD * Real.sqrt 3 = 5 * Real.sqrt 3 := by sorry
  exact h1

end find_side_y_l240_240217


namespace sector_area_l240_240459

theorem sector_area (C : ℝ) (θ : ℝ) (r : ℝ) (S : ℝ)
  (hC : C = (8 * Real.pi / 9) + 4)
  (hθ : θ = (80 * Real.pi / 180))
  (hne : θ * r / 2 + r = C) :
  S = (1 / 2) * θ * r^2 → S = 8 * Real.pi / 9 :=
by
  sorry

end sector_area_l240_240459


namespace arithmetic_sequence_a2_a6_l240_240582

theorem arithmetic_sequence_a2_a6 (a : ℕ → ℕ) (d : ℕ) (h_arith_seq : ∀ n, a (n+1) = a n + d)
  (h_a4 : a 4 = 4) : a 2 + a 6 = 8 :=
by sorry

end arithmetic_sequence_a2_a6_l240_240582


namespace valerie_needs_21_stamps_l240_240474

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

end valerie_needs_21_stamps_l240_240474


namespace theater_workshop_l240_240876

-- Definitions of the conditions
def total_participants : ℕ := 120
def cannot_craft_poetry : ℕ := 52
def cannot_perform_painting : ℕ := 75
def not_skilled_in_photography : ℕ := 38
def participants_with_exactly_two_skills : ℕ := 195 - total_participants

-- The theorem stating the problem
theorem theater_workshop :
  participants_with_exactly_two_skills = 75 := by
  sorry

end theater_workshop_l240_240876


namespace complex_shape_perimeter_l240_240709

theorem complex_shape_perimeter :
  ∃ h : ℝ, 12 * h - 20 = 95 ∧
  (24 + ((230 / 12) - 2) + 10 : ℝ) = 51.1667 :=
by
  sorry

end complex_shape_perimeter_l240_240709


namespace different_result_l240_240247

theorem different_result :
  let A := -2 - (-3)
  let B := 2 - 3
  let C := -3 + 2
  let D := -3 - (-2)
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B = C ∧ B = D :=
by
  sorry

end different_result_l240_240247


namespace octopus_legs_l240_240952

/-- Four octopuses made statements about their total number of legs.
    - Octopuses with 7 legs always lie.
    - Octopuses with 6 or 8 legs always tell the truth.
    - Blue: "Together we have 28 legs."
    - Green: "Together we have 27 legs."
    - Yellow: "Together we have 26 legs."
    - Red: "Together we have 25 legs."
   Prove that the Green octopus has 6 legs, and the Blue, Yellow, and Red octopuses each have 7 legs.
-/
theorem octopus_legs (L_B L_G L_Y L_R : ℕ) (H1 : (L_B + L_G + L_Y + L_R = 28 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 27 → L_B + L_G + L_Y + L_R = 27) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 26 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 25 → L_B ≠ 7)) : 
  (L_G = 6) ∧ (L_B = 7) ∧ (L_Y = 7) ∧ (L_R = 7) :=
sorry

end octopus_legs_l240_240952


namespace correct_factoring_example_l240_240946

-- Define each option as hypotheses
def optionA (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2
def optionB (a b : ℝ) : Prop := 2 * a ^ 2 - a * b - a = a * (2 * a - b - 1)
def optionC (a b : ℝ) : Prop := 8 * a ^ 5 * b ^ 2 = 4 * a ^ 3 * b * 2 * a ^ 2 * b
def optionD (a : ℝ) : Prop := a ^ 2 - 4 * a + 3 = (a - 1) * (a - 3)

-- The goal is to prove that optionD is the correct example of factoring
theorem correct_factoring_example (a b : ℝ) : optionD a ↔ (∀ a b, ¬ optionA a b) ∧ (∀ a b, ¬ optionB a b) ∧ (∀ a b, ¬ optionC a b) :=
by
  sorry

end correct_factoring_example_l240_240946


namespace cost_price_of_one_ball_is_48_l240_240763

-- Define the cost price of one ball
def costPricePerBall (x : ℝ) : Prop :=
  let totalCostPrice20Balls := 20 * x
  let sellingPrice20Balls := 720
  let loss := 5 * x
  totalCostPrice20Balls = sellingPrice20Balls + loss

-- Define the main proof problem
theorem cost_price_of_one_ball_is_48 (x : ℝ) (h : costPricePerBall x) : x = 48 :=
by
  sorry

end cost_price_of_one_ball_is_48_l240_240763


namespace necessary_not_sufficient_x2_minus_3x_plus_2_l240_240684

theorem necessary_not_sufficient_x2_minus_3x_plus_2 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → x^2 - 3 * x + 2 ≤ 0) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ m ∧ ¬(x^2 - 3 * x + 2 ≤ 0)) →
  m ≥ 2 :=
sorry

end necessary_not_sufficient_x2_minus_3x_plus_2_l240_240684


namespace system_of_equations_value_l240_240559

theorem system_of_equations_value (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 4 * y - 10 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 96 / 13 := 
sorry

end system_of_equations_value_l240_240559


namespace veronica_pitting_time_is_2_hours_l240_240853

def veronica_cherries_pitting_time (pounds : ℕ) (cherries_per_pound : ℕ) (minutes_per_20_cherries : ℕ) :=
  let cherries := pounds * cherries_per_pound
  let sets := cherries / 20
  let total_minutes := sets * minutes_per_20_cherries
  total_minutes / 60

theorem veronica_pitting_time_is_2_hours : 
  veronica_cherries_pitting_time 3 80 10 = 2 :=
  by
    sorry

end veronica_pitting_time_is_2_hours_l240_240853


namespace justin_reads_pages_l240_240998

theorem justin_reads_pages (x : ℕ) 
  (h1 : 130 = x + 6 * (2 * x)) : x = 10 := 
sorry

end justin_reads_pages_l240_240998


namespace dina_has_60_dolls_l240_240744

variable (ivy_collectors_edition_dolls : ℕ)
variable (ivy_total_dolls : ℕ)
variable (dina_dolls : ℕ)

-- Conditions
def condition1 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := ivy_collectors_edition_dolls = 20
def condition2 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := (2 / 3 : ℚ) * ivy_total_dolls = ivy_collectors_edition_dolls
def condition3 (ivy_total_dolls dina_dolls : ℕ) := dina_dolls = 2 * ivy_total_dolls

-- Proof statement
theorem dina_has_60_dolls 
  (h1 : condition1 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h2 : condition2 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h3 : condition3 ivy_total_dolls dina_dolls) : 
  dina_dolls = 60 :=
sorry

end dina_has_60_dolls_l240_240744


namespace shekar_average_is_81_9_l240_240096

def shekar_average_marks (marks : List ℕ) : ℚ :=
  (marks.sum : ℚ) / marks.length

theorem shekar_average_is_81_9 :
  shekar_average_marks [92, 78, 85, 67, 89, 74, 81, 95, 70, 88] = 81.9 :=
by
  sorry

end shekar_average_is_81_9_l240_240096


namespace set_inclusion_interval_l240_240508

theorem set_inclusion_interval (a : ℝ) :
    (A : Set ℝ) = {x : ℝ | (2 * a + 1) ≤ x ∧ x ≤ (3 * a - 5)} →
    (B : Set ℝ) = {x : ℝ | 3 ≤ x ∧ x ≤ 22} →
    (2 * a + 1 ≤ 3 * a - 5) →
    (A ⊆ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_inclusion_interval_l240_240508


namespace gcd_91_72_l240_240649

/-- Prove that the greatest common divisor of 91 and 72 is 1. -/
theorem gcd_91_72 : Nat.gcd 91 72 = 1 :=
by
  sorry

end gcd_91_72_l240_240649


namespace inscribed_square_ratios_l240_240268

theorem inscribed_square_ratios (a b c x y : ℝ) (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sides : a^2 + b^2 = c^2) 
  (h_leg_square : x = a) 
  (h_hyp_square : y = 5 / 18 * c) : 
  x / y = 18 / 13 := by
  sorry

end inscribed_square_ratios_l240_240268


namespace regular_price_of_each_shirt_l240_240072

theorem regular_price_of_each_shirt (P : ℝ) :
    let total_shirts := 20
    let sale_price_per_shirt := 0.8 * P
    let tax_rate := 0.10
    let total_paid := 264
    let total_price := total_shirts * sale_price_per_shirt * (1 + tax_rate)
    total_price = total_paid → P = 15 :=
by
  intros
  sorry

end regular_price_of_each_shirt_l240_240072


namespace area_difference_is_correct_l240_240493

noncomputable def circumference_1 : ℝ := 264
noncomputable def circumference_2 : ℝ := 352

noncomputable def radius_1 : ℝ := circumference_1 / (2 * Real.pi)
noncomputable def radius_2 : ℝ := circumference_2 / (2 * Real.pi)

noncomputable def area_1 : ℝ := Real.pi * radius_1^2
noncomputable def area_2 : ℝ := Real.pi * radius_2^2

noncomputable def area_difference : ℝ := area_2 - area_1

theorem area_difference_is_correct :
  abs (area_difference - 4305.28) < 1e-2 :=
by
  sorry

end area_difference_is_correct_l240_240493


namespace probability_non_defective_pencils_l240_240656

theorem probability_non_defective_pencils :
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations:ℚ) / (total_combinations:ℚ) = 5 / 14 := by
  sorry

end probability_non_defective_pencils_l240_240656


namespace exponent_of_9_in_9_pow_7_l240_240003

theorem exponent_of_9_in_9_pow_7 : ∀ x : ℕ, (3 ^ x ∣ 9 ^ 7) ↔ x ≤ 14 := by
  sorry

end exponent_of_9_in_9_pow_7_l240_240003


namespace time_spent_on_Type_A_problems_l240_240443

theorem time_spent_on_Type_A_problems (t : ℝ) (h1 : 25 * (8 * t) + 100 * (2 * t) = 120) : 
  25 * (8 * t) = 60 := by
  sorry

-- Conditions
-- t is the time spent on a Type C problem in minutes
-- 25 * (8 * t) + 100 * (2 * t) = 120 (time spent on Type A and B problems combined equals 120 minutes)

end time_spent_on_Type_A_problems_l240_240443


namespace number_of_binders_l240_240886

-- Definitions of given conditions
def book_cost : Nat := 16
def binder_cost : Nat := 2
def notebooks_cost : Nat := 6
def total_cost : Nat := 28

-- Variable for the number of binders
variable (b : Nat)

-- Proposition that the number of binders Léa bought is 3
theorem number_of_binders (h : book_cost + binder_cost * b + notebooks_cost = total_cost) : b = 3 :=
by
  sorry

end number_of_binders_l240_240886


namespace commute_times_absolute_difference_l240_240166

theorem commute_times_absolute_difference
  (x y : ℝ)
  (H_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (H_var : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  abs (x - y) = 4 :=
by
  -- proof steps are omitted
  sorry

end commute_times_absolute_difference_l240_240166


namespace problem1_problem2_l240_240492

variables {p x1 x2 y1 y2 : ℝ} (h₁ : p > 0) (h₂ : x1 * x2 ≠ 0) (h₃ : y1^2 = 2 * p * x1) (h₄ : y2^2 = 2 * p * x2)

theorem problem1 (h₅ : x1 * x2 + y1 * y2 = 0) :
    ∀ (x y : ℝ), (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 → 
        x^2 + y^2 - (x1 + x2) * x - (y1 + y2) * y = 0 := sorry

theorem problem2 (h₀ : ∀ x y, x = (x1 + x2) / 2 → y = (y1 + y2) / 2 → 
    |((x1 + x2) / 2) - 2 * ((y1 + y2) / 2)| / (Real.sqrt 5) = 2 * (Real.sqrt 5) / 5) :
    p = 2 := sorry

end problem1_problem2_l240_240492


namespace weighted_average_AC_l240_240837

theorem weighted_average_AC (avgA avgB avgC wA wB wC total_weight: ℝ)
  (h_avgA : avgA = 7.3)
  (h_avgB : avgB = 7.6) 
  (h_avgC : avgC = 7.2)
  (h_wA : wA = 3)
  (h_wB : wB = 4)
  (h_wC : wC = 1)
  (h_total_weight : total_weight = 5) :
  ((avgA * wA + avgC * wC) / total_weight) = 5.82 :=
by
  sorry

end weighted_average_AC_l240_240837


namespace solve_equation_l240_240240

theorem solve_equation : ∃ x : ℝ, (1 + x) / (2 - x) - 1 = 1 / (x - 2) ↔ x = 0 := 
by
  sorry

end solve_equation_l240_240240


namespace baseball_card_distribution_l240_240840

theorem baseball_card_distribution (total_cards : ℕ) (capacity_4 : ℕ) (capacity_6 : ℕ) (capacity_8 : ℕ) :
  total_cards = 137 →
  capacity_4 = 4 →
  capacity_6 = 6 →
  capacity_8 = 8 →
  (total_cards % capacity_4) % capacity_6 = 1 :=
by
  intros
  sorry

end baseball_card_distribution_l240_240840


namespace sum_of_coordinates_of_C_and_D_l240_240626

structure Point where
  x : ℤ
  y : ℤ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def sum_coordinates (p1 p2 : Point) : ℤ :=
  p1.x + p1.y + p2.x + p2.y

def C : Point := { x := 3, y := -2 }
def D : Point := reflect_y C

theorem sum_of_coordinates_of_C_and_D : sum_coordinates C D = -4 := by
  sorry

end sum_of_coordinates_of_C_and_D_l240_240626


namespace tank_empty_time_l240_240907

theorem tank_empty_time (R L : ℝ) (h1 : R = 1 / 7) (h2 : R - L = 1 / 8) : 
  (1 / L) = 56 :=
by
  sorry

end tank_empty_time_l240_240907


namespace clock_in_2023_hours_l240_240290

theorem clock_in_2023_hours (current_time : ℕ) (h_current_time : current_time = 3) : 
  (current_time + 2023) % 12 = 10 := 
by {
  -- context: non-computational (time kept in modulo world and not real increments)
  sorry
}

end clock_in_2023_hours_l240_240290


namespace money_left_after_shopping_l240_240343

-- Conditions
def cost_mustard_oil : ℤ := 2 * 13
def cost_pasta : ℤ := 3 * 4
def cost_sauce : ℤ := 1 * 5
def total_cost : ℤ := cost_mustard_oil + cost_pasta + cost_sauce
def total_money : ℤ := 50

-- Theorem to prove
theorem money_left_after_shopping : total_money - total_cost = 7 := by
  sorry

end money_left_after_shopping_l240_240343


namespace allison_total_supply_items_is_28_l240_240157

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end allison_total_supply_items_is_28_l240_240157


namespace function_increasing_no_negative_roots_l240_240885

noncomputable def f (a x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem function_increasing (a : ℝ) (h : a > 1) : 
  ∀ (x1 x2 : ℝ), (-1 < x1) → (x1 < x2) → (f a x1 < f a x2) := 
by
  -- placeholder proof
  sorry

theorem no_negative_roots (a : ℝ) (h : a > 1) : 
  ∀ (x : ℝ), (x < 0) → (f a x ≠ 0) := 
by
  -- placeholder proof
  sorry

end function_increasing_no_negative_roots_l240_240885


namespace total_weight_collected_l240_240109

def GinaCollectedBags : ℕ := 8
def NeighborhoodFactor : ℕ := 120
def WeightPerBag : ℕ := 6

theorem total_weight_collected :
  (GinaCollectedBags * NeighborhoodFactor + GinaCollectedBags) * WeightPerBag = 5808 :=
by
  sorry

end total_weight_collected_l240_240109


namespace ratio_of_x_to_y_l240_240828

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) : 
  x / y = 37 / 16 :=
by
  sorry

end ratio_of_x_to_y_l240_240828


namespace price_of_first_candy_l240_240360

theorem price_of_first_candy (P: ℝ) 
  (total_weight: ℝ) (price_per_lb_mixture: ℝ) 
  (weight_first: ℝ) (weight_second: ℝ) 
  (price_per_lb_second: ℝ) :
  total_weight = 30 →
  price_per_lb_mixture = 3 →
  weight_first = 20 →
  weight_second = 10 →
  price_per_lb_second = 3.1 →
  20 * P + 10 * price_per_lb_second = total_weight * price_per_lb_mixture →
  P = 2.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end price_of_first_candy_l240_240360


namespace edward_work_hours_edward_work_hours_overtime_l240_240888

variable (H : ℕ) -- H represents the number of hours worked
variable (O : ℕ) -- O represents the number of overtime hours

theorem edward_work_hours (H_le_40 : H ≤ 40) (earning_eq_210 : 7 * H = 210) : H = 30 :=
by
  -- Proof to be filled in here
  sorry

theorem edward_work_hours_overtime (H_gt_40 : H > 40) (earning_eq_210 : 7 * 40 + 14 * (H - 40) = 210) : False :=
by
  -- Proof to be filled in here
  sorry

end edward_work_hours_edward_work_hours_overtime_l240_240888


namespace gcd_paving_courtyard_l240_240099

theorem gcd_paving_courtyard :
  Nat.gcd 378 595 = 7 :=
by
  sorry

end gcd_paving_courtyard_l240_240099


namespace solution_set_of_inequality_l240_240522

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x - 3) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x < 3} := 
by
  sorry

end solution_set_of_inequality_l240_240522


namespace function_has_three_zeros_l240_240855

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l240_240855


namespace find_k_no_xy_term_l240_240587

theorem find_k_no_xy_term (k : ℝ) :
  (¬ ∃ x y : ℝ, (-x^2 - 3 * k * x * y - 3 * y^2 + 9 * x * y - 8) = (- x^2 - 3 * y^2 - 8)) → k = 3 :=
by
  sorry

end find_k_no_xy_term_l240_240587


namespace combine_quadratic_radicals_l240_240058

theorem combine_quadratic_radicals (x : ℝ) (h : 3 * x + 5 = 2 * x + 7) : x = 2 :=
by
  sorry

end combine_quadratic_radicals_l240_240058


namespace number_of_integer_values_l240_240332

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 9 * x^2 + 2 * x + 17

theorem number_of_integer_values :
  (∃ xs : List ℤ, xs.length = 4 ∧ ∀ x ∈ xs, Nat.Prime (Int.natAbs (Q x))) :=
by
  sorry

end number_of_integer_values_l240_240332


namespace Maxim_is_correct_l240_240450

-- Defining the parameters
def mortgage_rate := 0.125
def dividend_yield := 0.17

-- Theorem statement
theorem Maxim_is_correct : (dividend_yield - mortgage_rate > 0) := by 
    -- Dividing the proof's logical steps
    sorry

end Maxim_is_correct_l240_240450


namespace find_e_value_l240_240301

theorem find_e_value : 
  ∃ e : ℝ, 12 / (-12 + 2 * e) = -11 - 2 * e ∧ e = 4 :=
by
  use 4
  sorry

end find_e_value_l240_240301


namespace avg_remaining_two_l240_240424

variables {A B C D E : ℝ}

-- Conditions
def avg_five (A B C D E : ℝ) : Prop := (A + B + C + D + E) / 5 = 10
def avg_three (A B C : ℝ) : Prop := (A + B + C) / 3 = 4

-- Theorem to prove
theorem avg_remaining_two (A B C D E : ℝ) (h1 : avg_five A B C D E) (h2 : avg_three A B C) : ((D + E) / 2) = 19 := 
sorry

end avg_remaining_two_l240_240424


namespace bumper_car_line_total_in_both_lines_l240_240036

theorem bumper_car_line (x y Z : ℕ) (hZ : Z = 25 - x + y) : Z = 25 - x + y :=
by
  sorry

theorem total_in_both_lines (x y Z : ℕ) (hZ : Z = 25 - x + y) : 40 - x + y = Z + 15 :=
by
  sorry

end bumper_car_line_total_in_both_lines_l240_240036


namespace solve_x_l240_240089

theorem solve_x (x : ℝ) (h : 9 - 4 / x = 7 + 8 / x) : x = 6 := 
by 
  sorry

end solve_x_l240_240089


namespace sequence_transformation_possible_l240_240525

theorem sequence_transformation_possible 
  (a1 a2 : ℕ) (h1 : a1 ≤ 100) (h2 : a2 ≤ 100) (h3 : a1 ≥ a2) : 
  ∃ (operations : ℕ), operations ≤ 51 :=
by
  sorry

end sequence_transformation_possible_l240_240525


namespace frac_add_eq_seven_halves_l240_240660

theorem frac_add_eq_seven_halves {x y : ℝ} (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 :=
by
  sorry

end frac_add_eq_seven_halves_l240_240660


namespace find_c_l240_240960

theorem find_c (a b c : ℝ) (k₁ k₂ : ℝ) 
  (h₁ : a * b = k₁) 
  (h₂ : b * c = k₂) 
  (h₃ : 40 * 5 = k₁) 
  (h₄ : 7 * 10 = k₂) 
  (h₅ : a = 16) : 
  c = 5.6 :=
  sorry

end find_c_l240_240960


namespace k_inequality_l240_240005

noncomputable def k_value : ℝ :=
  5

theorem k_inequality (x : ℝ) :
  (x * (2 * x + 3) < k_value) ↔ (x > -5 / 2 ∧ x < 1) :=
sorry

end k_inequality_l240_240005


namespace polynomial_remainder_l240_240699

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ),
  (q.eval 2 = 8) →
  (q.eval (-3) = -10) →
  ∃ c d : ℚ, (q = (Polynomial.C (c : ℚ) * (Polynomial.X - Polynomial.C 2) * (Polynomial.X + Polynomial.C 3)) + (Polynomial.C 3.6 * Polynomial.X + Polynomial.C 0.8)) :=
by intros q h1 h2; sorry

end polynomial_remainder_l240_240699


namespace count_three_digit_numbers_with_digit_sum_24_l240_240545

-- Define the conditions:
def isThreeDigitNumber (a b c : ℕ) : Prop :=
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c ≥ 100)

def digitSumEquals24 (a b c : ℕ) : Prop :=
  a + b + c = 24

-- State the theorem:
theorem count_three_digit_numbers_with_digit_sum_24 :
  (∃ (count : ℕ), count = 10 ∧ 
   ∀ (a b c : ℕ), isThreeDigitNumber a b c ∧ digitSumEquals24 a b c → (count = 10)) :=
sorry

end count_three_digit_numbers_with_digit_sum_24_l240_240545


namespace A_time_to_cover_distance_is_45_over_y_l240_240253

variable (y : ℝ)
variable (h0 : y > 0)
variable (h1 : (45 : ℝ) / (y - 2 / 3) - (45 : ℝ) / y = 3 / 4)

theorem A_time_to_cover_distance_is_45_over_y :
  45 / y = 45 / y :=
by
  sorry

end A_time_to_cover_distance_is_45_over_y_l240_240253


namespace age_difference_of_siblings_l240_240755

theorem age_difference_of_siblings (x : ℝ) 
  (h1 : 19 * x + 20 = 230) :
  |4 * x - 3 * x| = 210 / 19 := by
    sorry

end age_difference_of_siblings_l240_240755


namespace relationship_l240_240941

-- Definitions for the points on the inverse proportion function
def on_inverse_proportion (x : ℝ) (y : ℝ) : Prop :=
  y = -6 / x

-- Given conditions
def A (y1 : ℝ) : Prop :=
  on_inverse_proportion (-3) y1

def B (y2 : ℝ) : Prop :=
  on_inverse_proportion (-1) y2

def C (y3 : ℝ) : Prop :=
  on_inverse_proportion (2) y3

-- The theorem that expresses the relationship
theorem relationship (y1 y2 y3 : ℝ) (hA : A y1) (hB : B y2) (hC : C y3) : y3 < y1 ∧ y1 < y2 :=
by
  -- skeleton of proof
  sorry

end relationship_l240_240941


namespace inequality_proof_l240_240227

-- Define the inequality problem in Lean 4
theorem inequality_proof (x y : ℝ) (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : x * y = 1) : 
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l240_240227


namespace john_fouled_per_game_l240_240380

theorem john_fouled_per_game
  (hit_rate : ℕ) (shots_per_foul : ℕ) (total_games : ℕ) (participation_rate : ℚ) (total_free_throws : ℕ) :
  hit_rate = 70 → shots_per_foul = 2 → total_games = 20 → participation_rate = 0.8 → total_free_throws = 112 →
  (total_free_throws / (participation_rate * total_games)) / shots_per_foul = 3.5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end john_fouled_per_game_l240_240380


namespace scientific_notation_2150000_l240_240729

theorem scientific_notation_2150000 : 2150000 = 2.15 * 10^6 :=
  by
  sorry

end scientific_notation_2150000_l240_240729


namespace smallest_difference_l240_240900

theorem smallest_difference {a b : ℕ} (h1: a * b = 2010) (h2: a > b) : a - b = 37 :=
sorry

end smallest_difference_l240_240900


namespace min_distance_between_tracks_l240_240413

noncomputable def min_distance : ℝ :=
  (Real.sqrt 163 - 6) / 3

theorem min_distance_between_tracks :
  let RationalManTrack := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let IrrationalManTrack := {p : ℝ × ℝ | (p.1 - 2)^2 / 9 + p.2^2 / 25 = 1}
  ∀ pA ∈ RationalManTrack, ∀ pB ∈ IrrationalManTrack,
  dist pA pB = min_distance :=
sorry

end min_distance_between_tracks_l240_240413


namespace mary_groceries_fitting_l240_240638

theorem mary_groceries_fitting :
  (∀ bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice,
    bags = 2 →
    wt_green = 4 →
    wt_milk = 6 →
    wt_carrots = 2 * wt_green →
    wt_apples = 3 →
    wt_bread = 1 →
    wt_rice = 5 →
    (wt_green + wt_milk + wt_carrots + wt_apples + wt_bread + wt_rice = 27) →
    (∀ b, b < 20 →
      (b = 6 + 5 ∨ b = 22 - 11) →
      (20 - b = 9))) :=
by
  intros bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice h_bags h_green h_milk h_carrots h_apples h_bread h_rice h_total h_b
  sorry

end mary_groceries_fitting_l240_240638


namespace markers_in_desk_l240_240610

theorem markers_in_desk (pens pencils markers : ℕ) 
  (h_ratio : pens = 2 * pencils ∧ pens = 2 * markers / 5) 
  (h_pens : pens = 10) : markers = 25 :=
by
  sorry

end markers_in_desk_l240_240610


namespace product_form_l240_240537

theorem product_form (b a : ℤ) :
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := 
sorry

end product_form_l240_240537


namespace max_dot_product_OB_OA_l240_240958

theorem max_dot_product_OB_OA (P A O B : ℝ × ℝ)
  (h₁ : ∃ x y : ℝ, (x, y) = P ∧ x^2 / 16 - y^2 / 9 = 1)
  (t : ℝ)
  (h₂ : A = (t - 1) • P)
  (h₃ : P • O = 64)
  (h₄ : B = (0, 1)) :
  ∃ t : ℝ, abs (B • A) ≤ (24/5) := 
sorry

end max_dot_product_OB_OA_l240_240958


namespace sum_of_numbers_l240_240613

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def tens_digit_zero (n : ℕ) : Prop := (n / 10) % 10 = 0
def units_digit_nonzero (n : ℕ) : Prop := n % 10 ≠ 0
def same_units_digits (m n : ℕ) : Prop := m % 10 = n % 10

theorem sum_of_numbers (a b c : ℕ)
  (h1 : is_perfect_square a) (h2 : is_perfect_square b) (h3 : is_perfect_square c)
  (h4 : tens_digit_zero a) (h5 : tens_digit_zero b) (h6 : tens_digit_zero c)
  (h7 : units_digit_nonzero a) (h8 : units_digit_nonzero b) (h9 : units_digit_nonzero c)
  (h10 : same_units_digits b c)
  (h11 : a % 10 % 2 = 0) :
  a + b + c = 14612 :=
sorry

end sum_of_numbers_l240_240613


namespace sampling_probabilities_equal_l240_240336

noncomputable def populationSize (N : ℕ) := N
noncomputable def sampleSize (n : ℕ) := n

def P1 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P2 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P3 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)

theorem sampling_probabilities_equal (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  P1 N n = P2 N n ∧ P2 N n = P3 N n :=
by
  -- Proof steps will go here
  sorry

end sampling_probabilities_equal_l240_240336


namespace amusement_park_ticket_cost_l240_240860

/-- Jeremie is going to an amusement park with 3 friends. 
    The cost of a set of snacks is $5. 
    The total cost for everyone to go to the amusement park and buy snacks is $92.
    Prove that the cost of one ticket is $18.
-/
theorem amusement_park_ticket_cost 
  (number_of_people : ℕ)
  (snack_cost_per_person : ℕ)
  (total_cost : ℕ)
  (ticket_cost : ℕ) :
  number_of_people = 4 → 
  snack_cost_per_person = 5 → 
  total_cost = 92 → 
  ticket_cost = 18 :=
by
  intros h1 h2 h3
  sorry

end amusement_park_ticket_cost_l240_240860


namespace suff_but_not_nec_l240_240951

theorem suff_but_not_nec (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by {
  sorry
}

end suff_but_not_nec_l240_240951


namespace five_algorithmic_statements_l240_240461

-- Define the five types of algorithmic statements in programming languages
inductive AlgorithmicStatement : Type
| input : AlgorithmicStatement
| output : AlgorithmicStatement
| assignment : AlgorithmicStatement
| conditional : AlgorithmicStatement
| loop : AlgorithmicStatement

-- Theorem: Every programming language contains these five basic types of algorithmic statements
theorem five_algorithmic_statements : 
  ∃ (s : List AlgorithmicStatement), 
    (s.length = 5) ∧ 
    ∀ x, x ∈ s ↔
    x = AlgorithmicStatement.input ∨
    x = AlgorithmicStatement.output ∨
    x = AlgorithmicStatement.assignment ∨
    x = AlgorithmicStatement.conditional ∨
    x = AlgorithmicStatement.loop :=
by
  sorry

end five_algorithmic_statements_l240_240461


namespace mia_spent_per_parent_l240_240557

theorem mia_spent_per_parent (amount_sibling : ℕ) (num_siblings : ℕ) (total_spent : ℕ) 
  (num_parents : ℕ) : 
  amount_sibling = 30 → num_siblings = 3 → total_spent = 150 → num_parents = 2 → 
  (total_spent - num_siblings * amount_sibling) / num_parents = 30 :=
by
  sorry

end mia_spent_per_parent_l240_240557


namespace set_intersection_A_B_l240_240181

def A := {x : ℝ | 2 * x - x^2 > 0}
def B := {x : ℝ | x > 1}
def I := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_A_B :
  A ∩ B = I :=
sorry

end set_intersection_A_B_l240_240181


namespace no_real_solution_l240_240673

theorem no_real_solution (x y : ℝ) (hx : x^2 = 1 + 1 / y^2) (hy : y^2 = 1 + 1 / x^2) : false :=
by
  sorry

end no_real_solution_l240_240673


namespace find_point_P_l240_240324

noncomputable def f (x : ℝ) : ℝ := x^2 - x

theorem find_point_P :
  (∃ x y : ℝ, f x = y ∧ (2 * x - 1 = 1) ∧ (y = x^2 - x)) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end find_point_P_l240_240324


namespace integer_solutions_of_xyz_equation_l240_240874

/--
  Find all integer solutions of the equation \( x + y + z = xyz \).
  The integer solutions are expected to be:
  \[
  (1, 2, 3), (2, 1, 3), (3, 1, 2), (3, 2, 1), (1, 3, 2), (2, 3, 1), (-a, 0, a) \text{ for } (a : ℤ).
  \]
-/
theorem integer_solutions_of_xyz_equation (x y z : ℤ) :
    x + y + z = x * y * z ↔ 
    (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
    (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) ∨ 
    (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ 
    ∃ a : ℤ, (x = -a ∧ y = 0 ∧ z = a) := by
  sorry


end integer_solutions_of_xyz_equation_l240_240874


namespace sum_of_ages_l240_240121

theorem sum_of_ages (a b c : ℕ) (h₁ : a = 20 + b + c) (h₂ : a^2 = 2050 + (b + c)^2) : a + b + c = 80 :=
sorry

end sum_of_ages_l240_240121


namespace cube_painting_l240_240337

theorem cube_painting (n : ℕ) (h₁ : n > 4) 
  (h₂ : (2 * (n - 2)) = (n^2 - 2*n + 1)) : n = 5 :=
sorry

end cube_painting_l240_240337


namespace complement_intersection_l240_240345

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {2, 4}

theorem complement_intersection :
  ((U \ A) ∩ B) = {2} :=
sorry

end complement_intersection_l240_240345


namespace no_integers_satisfy_l240_240350

theorem no_integers_satisfy (a b c d : ℤ) : ¬ (a^4 + b^4 + c^4 + 2016 = 10 * d) :=
sorry

end no_integers_satisfy_l240_240350


namespace insulation_cost_l240_240714

def rectangular_prism_surface_area (l w h : ℕ) : ℕ :=
2 * l * w + 2 * l * h + 2 * w * h

theorem insulation_cost
  (l w h : ℕ) (cost_per_square_foot : ℕ)
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost : cost_per_square_foot = 20) :
  rectangular_prism_surface_area l w h * cost_per_square_foot = 1440 := 
sorry

end insulation_cost_l240_240714


namespace dot_product_v_w_l240_240787

def v : ℝ × ℝ := (-5, 3)
def w : ℝ × ℝ := (7, -9)

theorem dot_product_v_w : v.1 * w.1 + v.2 * w.2 = -62 := 
  by sorry

end dot_product_v_w_l240_240787


namespace tan_alpha_half_l240_240195

theorem tan_alpha_half (α: ℝ) (h: Real.tan α = 1/2) :
  (1 + 2 * Real.sin (Real.pi - α) * Real.cos (-2 * Real.pi - α)) / (Real.sin (-α)^2 - Real.sin (5 * Real.pi / 2 - α)^2) = -3 := 
by
  sorry

end tan_alpha_half_l240_240195


namespace area_of_shaded_region_l240_240891

theorem area_of_shaded_region :
  let width := 10
  let height := 5
  let base_triangle := 3
  let height_triangle := 2
  let top_base_trapezoid := 3
  let bottom_base_trapezoid := 6
  let height_trapezoid := 3
  let area_rectangle := width * height
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle
  let area_trapezoid := (1 / 2 : ℝ) * (top_base_trapezoid + bottom_base_trapezoid) * height_trapezoid
  let area_shaded := area_rectangle - area_triangle - area_trapezoid
  area_shaded = 33.5 :=
by
  sorry

end area_of_shaded_region_l240_240891


namespace factor_expression_l240_240556

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) :=
by
  sorry

end factor_expression_l240_240556


namespace joe_marshmallow_ratio_l240_240992

theorem joe_marshmallow_ratio (J : ℕ) (h1 : 21 / 3 = 7) (h2 : 1 / 2 * J = 49 - 7) : J / 21 = 4 :=
by
  sorry

end joe_marshmallow_ratio_l240_240992


namespace problem_div_expansion_l240_240284

theorem problem_div_expansion (m : ℝ) : ((2 * m^2 - m)^2) / (-m^2) = -4 * m^2 + 4 * m - 1 := 
by sorry

end problem_div_expansion_l240_240284


namespace square_area_from_diagonal_l240_240299

theorem square_area_from_diagonal (d : ℝ) (h_d : d = 12) : ∃ (A : ℝ), A = 72 :=
by
  -- we will use the given diagonal to derive the result
  sorry

end square_area_from_diagonal_l240_240299


namespace red_minus_white_more_l240_240924

variable (flowers_total yellow_white red_yellow red_white : ℕ)
variable (h1 : flowers_total = 44)
variable (h2 : yellow_white = 13)
variable (h3 : red_yellow = 17)
variable (h4 : red_white = 14)

theorem red_minus_white_more : 
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end red_minus_white_more_l240_240924


namespace min_students_l240_240188

theorem min_students (b g : ℕ) (hb : 1 ≤ b) (hg : 1 ≤ g)
    (h1 : b = (4/3) * g) 
    (h2 : (1/2) * b = 2 * ((1/3) * g)) 
    : b + g = 7 :=
by sorry

end min_students_l240_240188


namespace original_selling_price_is_800_l240_240505

-- Let CP denote the cost price
variable (CP : ℝ)

-- Condition 1: Selling price with a profit of 25%
def selling_price_with_profit (CP : ℝ) : ℝ := 1.25 * CP

-- Condition 2: Selling price with a loss of 35%
def selling_price_with_loss (CP : ℝ) : ℝ := 0.65 * CP

-- Given selling price with loss is Rs. 416
axiom loss_price_is_416 : selling_price_with_loss CP = 416

-- We need to prove the original selling price (with profit) is Rs. 800
theorem original_selling_price_is_800 : selling_price_with_profit CP = 800 :=
by sorry

end original_selling_price_is_800_l240_240505


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l240_240416

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l240_240416


namespace find_x_l240_240007

-- Define the condition as a theorem
theorem find_x (x : ℝ) (h : (1 + 3 + x) / 3 = 3) : x = 5 :=
by
  sorry  -- Placeholder for the proof

end find_x_l240_240007


namespace total_people_waiting_in_line_l240_240661

-- Conditions
def people_fitting_in_ferris_wheel : ℕ := 56
def people_not_getting_on : ℕ := 36

-- Definition: Number of people waiting in line
def number_of_people_waiting_in_line : ℕ := people_fitting_in_ferris_wheel + people_not_getting_on

-- Theorem to prove
theorem total_people_waiting_in_line : number_of_people_waiting_in_line = 92 := by
  -- This is a placeholder for the actual proof
  sorry

end total_people_waiting_in_line_l240_240661


namespace vector_minimization_and_angle_condition_l240_240599

noncomputable def find_OC_condition (C_op C_oa C_ob : ℝ × ℝ) 
  (C : ℝ × ℝ) : Prop := 
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  (CA.1 * CB.1 + CA.2 * CB.2) ≤ (C_op.1 * CB.1 + C_op.2 * CB.2)

theorem vector_minimization_and_angle_condition (C : ℝ × ℝ) 
  (C_op := (2, 1)) (C_oa := (1, 7)) (C_ob := (5, 1)) :
  (C = (4, 2)) → 
  find_OC_condition C_op C_oa C_ob C →
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                 (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
  cos_ACB = -4 * Real.sqrt (17) / 17 :=
  by 
    intro h1 find
    let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
    let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
    let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                   (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
    exact sorry

end vector_minimization_and_angle_condition_l240_240599


namespace LCM_of_numbers_l240_240502

theorem LCM_of_numbers (a b : ℕ) (h1 : a = 20) (h2 : a / b = 5 / 4): Nat.lcm a b = 80 :=
by
  sorry

end LCM_of_numbers_l240_240502


namespace quilt_width_is_eight_l240_240678

def length := 7
def cost_per_square_foot := 40
def total_cost := 2240
def area := total_cost / cost_per_square_foot

theorem quilt_width_is_eight :
  area / length = 8 := by
  sorry

end quilt_width_is_eight_l240_240678


namespace distance_between_trees_l240_240436

theorem distance_between_trees (num_trees : ℕ) (length_yard : ℝ)
  (h1 : num_trees = 26) (h2 : length_yard = 800) : 
  (length_yard / (num_trees - 1)) = 32 :=
by
  sorry

end distance_between_trees_l240_240436


namespace find_xy_l240_240916

theorem find_xy (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : (x - 10)^2 + (y - 10)^2 = 18) : 
  x * y = 91 := 
by {
  sorry
}

end find_xy_l240_240916


namespace possible_values_of_m_l240_240583

-- Proposition: for all real values of m, if for all real x, x^2 + 2x + 2 - m >= 0 holds, then m must be one of -1, 0, or 1

theorem possible_values_of_m (m : ℝ) 
  (h : ∀ (x : ℝ), x^2 + 2 * x + 2 - m ≥ 0) : m = -1 ∨ m = 0 ∨ m = 1 :=
sorry

end possible_values_of_m_l240_240583


namespace even_function_f_l240_240215

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem even_function_f (hx : ∀ x : ℝ, f (-x) = f x) 
  (hg : ∀ x : ℝ, g (-x) = -g x)
  (h_pass : g (-1) = 1)
  (hg_eq_f : ∀ x : ℝ, g x = f (x - 1)) 
  : f 7 + f 8 = -1 := 
by
  sorry

end even_function_f_l240_240215


namespace cube_side_length_l240_240418

theorem cube_side_length (n : ℕ) (h1 : 6 * (n^2) = 1/3 * 6 * (n^3)) : n = 3 := 
sorry

end cube_side_length_l240_240418


namespace price_of_tea_mixture_l240_240562

theorem price_of_tea_mixture 
  (p1 p2 p3 : ℝ) 
  (q1 q2 q3 : ℝ) 
  (h_p1 : p1 = 126) 
  (h_p2 : p2 = 135) 
  (h_p3 : p3 = 173.5) 
  (h_q1 : q1 = 1) 
  (h_q2 : q2 = 1) 
  (h_q3 : q3 = 2) : 
  (p1 * q1 + p2 * q2 + p3 * q3) / (q1 + q2 + q3) = 152 := 
by 
  sorry

end price_of_tea_mixture_l240_240562


namespace maximum_gcd_of_sequence_l240_240727

def a_n (n : ℕ) : ℕ := 100 + n^2

def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

theorem maximum_gcd_of_sequence : ∃ n : ℕ, ∀ m : ℕ, d_n n ≤ d_n m ∧ d_n n = 401 := sorry

end maximum_gcd_of_sequence_l240_240727


namespace hyperbola_eccentricity_l240_240185

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes_intersect : ∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 = -1 ∧ B.1 = -1 ∧
    ∀ (A B : ℝ × ℝ), ∃ x y : ℝ, (A.2 = y ∧ B.2 = y ∧ x^2 / a^2 - y^2 / b^2 = 1))
  (triangle_area : ∃ A B : ℝ × ℝ, 1 / 2 * abs (A.1 * B.2 - A.2 * B.1) = 2 * Real.sqrt 3) :
  ∃ e : ℝ, e = Real.sqrt 13 :=
by {
  sorry
}

end hyperbola_eccentricity_l240_240185


namespace number_of_red_balls_l240_240473

theorem number_of_red_balls (m : ℕ) (h1 : ∃ m : ℕ, (3 / (m + 3) : ℚ) = 1 / 4) : m = 9 :=
by
  obtain ⟨m, h1⟩ := h1
  sorry

end number_of_red_balls_l240_240473


namespace car_owners_without_motorcycles_l240_240061

theorem car_owners_without_motorcycles 
    (total_adults : ℕ) 
    (car_owners : ℕ) 
    (motorcycle_owners : ℕ) 
    (total_with_vehicles : total_adults = 500) 
    (total_car_owners : car_owners = 480) 
    (total_motorcycle_owners : motorcycle_owners = 120) : 
    car_owners - (car_owners + motorcycle_owners - total_adults) = 380 := 
by
    sorry

end car_owners_without_motorcycles_l240_240061


namespace avg_annual_growth_rate_profit_exceeds_340_l240_240692

variable (P2018 P2020 : ℝ)
variable (r : ℝ)

theorem avg_annual_growth_rate :
    P2018 = 200 → P2020 = 288 →
    (1 + r)^2 = P2020 / P2018 →
    r = 0.2 :=
by
  intros hP2018 hP2020 hGrowth
  sorry

theorem profit_exceeds_340 (P2020 : ℝ) (r : ℝ) :
    P2020 = 288 → r = 0.2 →
    P2020 * (1 + r) > 340 :=
by
  intros hP2020 hr
  sorry

end avg_annual_growth_rate_profit_exceeds_340_l240_240692


namespace house_number_is_fourteen_l240_240054

theorem house_number_is_fourteen (a b c n : ℕ) (h1 : a * b * c = 40) (h2 : a + b + c = n) (h3 : 
  ∃ (a b c : ℕ), a * b * c = 40 ∧ (a = 1 ∧ b = 5 ∧ c = 8) ∨ (a = 2 ∧ b = 2 ∧ c = 10) ∧ n = 14) :
  n = 14 :=
sorry

end house_number_is_fourteen_l240_240054


namespace base_number_mod_100_l240_240773

theorem base_number_mod_100 (base : ℕ) (h : base ^ 8 % 100 = 1) : base = 1 := 
sorry

end base_number_mod_100_l240_240773


namespace tangency_condition_l240_240377

-- Define the equation for the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 = 9

-- Define the equation for the hyperbola
def hyperbola_eq (x y m : ℝ) : Prop :=
  (x - 2)^2 - m * (y + 1)^2 = 1

-- Prove that for the ellipse and hyperbola to be tangent, m must equal 3
theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y ∧ hyperbola_eq x y m) → m = 3 :=
by
  sorry

end tangency_condition_l240_240377


namespace percentage_profit_is_35_l240_240376

-- Define the conditions
def initial_cost_price : ℝ := 100
def markup_percentage : ℝ := 0.5
def discount_percentage : ℝ := 0.1
def marked_price : ℝ := initial_cost_price * (1 + markup_percentage)
def selling_price : ℝ := marked_price * (1 - discount_percentage)

-- Define the statement/proof problem
theorem percentage_profit_is_35 :
  (selling_price - initial_cost_price) / initial_cost_price * 100 = 35 := by 
  sorry

end percentage_profit_is_35_l240_240376


namespace range_of_m_l240_240110

theorem range_of_m (m : ℝ) :
  (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ m > 0 ∧ (15 - m > 0) ∧ (15 - m > 2 * m))
  ∨ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)) →
  (¬ (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)))) →
  (0 < m ∧ m ≤ 2) ∨ (5 ≤ m ∧ m < 16/3) :=
by
  sorry

end range_of_m_l240_240110


namespace probability_of_drawing_black_ball_l240_240894

/-- The bag contains 2 black balls and 3 white balls. 
    The balls are identical except for their colors. 
    A ball is randomly drawn from the bag. -/
theorem probability_of_drawing_black_ball (b w : ℕ) (hb : b = 2) (hw : w = 3) :
    (b + w > 0) → (b / (b + w) : ℚ) = 2 / 5 :=
by
  intros h
  rw [hb, hw]
  norm_num

end probability_of_drawing_black_ball_l240_240894


namespace product_d_e_l240_240164

-- Define the problem: roots of the polynomial x^2 + x - 2
def roots_of_quadratic : Prop :=
  ∃ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0)

-- Define the condition that both roots are also roots of another polynomial
def roots_of_higher_poly (α β : ℚ) : Prop :=
  (α^7 - 7 * α^3 - 10 = 0 ) ∧ (β^7 - 7 * β^3 - 10 = 0)

-- The final proposition to prove
theorem product_d_e :
  ∀ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0) → (α^7 - 7 * α^3 - 10 = 0) ∧ (β^7 - 7 * β^3 - 10 = 0) → 7 * 10 = 70 := 
by sorry

end product_d_e_l240_240164


namespace smallest_b_factors_l240_240442

theorem smallest_b_factors (b : ℕ) (m n : ℤ) (h : m * n = 2023 ∧ m + n = b) : b = 136 :=
sorry

end smallest_b_factors_l240_240442


namespace carpet_length_l240_240346

theorem carpet_length (percent_covered : ℝ) (width : ℝ) (floor_area : ℝ) (carpet_length : ℝ) :
  percent_covered = 0.30 → width = 4 → floor_area = 120 → carpet_length = 9 :=
by
  sorry

end carpet_length_l240_240346


namespace set_intersection_l240_240608

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x^2 < 4

theorem set_intersection : {x | A x} ∩ {x | B x} = {x | 0 < x ∧ x < 2} := by
  sorry

end set_intersection_l240_240608


namespace motel_total_rent_l240_240618

theorem motel_total_rent (R40 R60 : ℕ) (total_rent : ℕ) 
  (h1 : total_rent = 40 * R40 + 60 * R60) 
  (h2 : 40 * (R40 + 10) + 60 * (R60 - 10) = total_rent - total_rent / 10) 
  (h3 : total_rent / 10 = 200) : 
  total_rent = 2000 := 
sorry

end motel_total_rent_l240_240618


namespace number_of_bicycles_l240_240151

theorem number_of_bicycles (B T : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 := by
  sorry

end number_of_bicycles_l240_240151


namespace initial_acorns_l240_240266

theorem initial_acorns (T : ℝ) (h1 : 0.35 * T = 7) (h2 : 0.45 * T = 9) : T = 20 :=
sorry

end initial_acorns_l240_240266


namespace problem_1_problem_2_l240_240988

-- Proposition p
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2 * a * x + 2 - a)

-- Proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2 * x + a ≥ 0

-- Problem 1: Prove that if p is true then a ≤ -2 or a ≥ 1
theorem problem_1 (a : ℝ) (hp : p a) : a ≤ -2 ∨ a ≥ 1 := sorry

-- Problem 2: Prove that if p ∨ q is true then a ≤ -2 or a ≥ 0
theorem problem_2 (a : ℝ) (hpq : p a ∨ q a) : a ≤ -2 ∨ a ≥ 0 := sorry

end problem_1_problem_2_l240_240988


namespace simplify_expression_l240_240483

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l240_240483


namespace pascal_triangle_ratio_l240_240226

theorem pascal_triangle_ratio (n r : ℕ) :
  (r + 1 = (4 * (n - r)) / 5) ∧ (r + 2 = (5 * (n - r - 1)) / 6) → n = 53 :=
by sorry

end pascal_triangle_ratio_l240_240226


namespace division_of_neg_six_by_three_l240_240520

theorem division_of_neg_six_by_three : (-6) / 3 = -2 := by
  sorry

end division_of_neg_six_by_three_l240_240520


namespace circle_center_radius_1_circle_center_coordinates_radius_1_l240_240909

theorem circle_center_radius_1 (x y : ℝ) : 
  x^2 + y^2 + 2*x - 4*y - 3 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 8 :=
sorry

theorem circle_center_coordinates_radius_1 : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y - 3 = 0 ∧ (x, y) = (-1, 2)) ∧ 
  (∃ r : ℝ, r = 2*Real.sqrt 2) :=
sorry

end circle_center_radius_1_circle_center_coordinates_radius_1_l240_240909


namespace prosecutor_cases_knight_or_liar_l240_240544

-- Define the conditions as premises
variable (X : Prop)
variable (Y : Prop)
variable (prosecutor : Prop) -- Truthfulness of the prosecutor (true for knight, false for liar)

-- Define the statements made by the prosecutor
axiom statement1 : X  -- "X is guilty."
axiom statement2 : ¬ (X ∧ Y)  -- "Both X and Y cannot both be guilty."

-- Lean 4 statement for the proof problem
theorem prosecutor_cases_knight_or_liar (h1 : prosecutor) (h2 : ¬prosecutor) : 
  (prosecutor ∧ X ∧ ¬Y) :=
by sorry

end prosecutor_cases_knight_or_liar_l240_240544


namespace prove_identity_l240_240762

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l240_240762


namespace range_of_m_l240_240079

theorem range_of_m (m : ℝ) :
  (∀ x : ℕ, (x = 1 ∨ x = 2 ∨ x = 3) → (3 * x - m ≤ 0)) ↔ 9 ≤ m ∧ m < 12 :=
by
  sorry

end range_of_m_l240_240079


namespace snail_kite_snails_eaten_l240_240759

theorem snail_kite_snails_eaten 
  (a₀ : ℕ) (a₁ : ℕ) (a₂ : ℕ) (a₃ : ℕ) (a₄ : ℕ)
  (h₀ : a₀ = 3)
  (h₁ : a₁ = a₀ + 2)
  (h₂ : a₂ = a₁ + 2)
  (h₃ : a₃ = a₂ + 2)
  (h₄ : a₄ = a₃ + 2)
  : a₀ + a₁ + a₂ + a₃ + a₄ = 35 := 
by 
  sorry

end snail_kite_snails_eaten_l240_240759


namespace min_value_a_plus_b_plus_c_l240_240049

theorem min_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 9 * a + 4 * b = a * b * c) : a + b + c ≥ 10 :=
by
  sorry

end min_value_a_plus_b_plus_c_l240_240049


namespace shopkeeper_oranges_l240_240114

theorem shopkeeper_oranges (O : ℕ) 
  (bananas : ℕ) 
  (percent_rotten_oranges : ℕ) 
  (percent_rotten_bananas : ℕ) 
  (percent_good_condition : ℚ) 
  (h1 : bananas = 400) 
  (h2 : percent_rotten_oranges = 15) 
  (h3 : percent_rotten_bananas = 6) 
  (h4 : percent_good_condition = 88.6) : 
  O = 600 :=
by
  -- This proof needs to be filled in.
  sorry

end shopkeeper_oranges_l240_240114


namespace minimum_value_hyperbola_l240_240503

noncomputable def min_value (a b : ℝ) (h : a > 0) (k : b > 0)
  (eccentricity_eq_two : (2:ℝ) = Real.sqrt (1 + (b/a)^2)) : ℝ :=
  (b^2 + 1) / (3 * a)

theorem minimum_value_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2:ℝ) = Real.sqrt (1 + (b/a)^2) ∧
  min_value a b (by sorry) (by sorry) (by sorry) = (2 * Real.sqrt 3) / 3 :=
sorry

end minimum_value_hyperbola_l240_240503


namespace beef_weight_after_processing_l240_240190

def original_weight : ℝ := 861.54
def weight_loss_percentage : ℝ := 0.35
def retained_percentage : ℝ := 1 - weight_loss_percentage
def weight_after_processing (w : ℝ) := retained_percentage * w

theorem beef_weight_after_processing :
  weight_after_processing original_weight = 560.001 :=
by
  sorry

end beef_weight_after_processing_l240_240190


namespace xy_plus_one_ge_four_l240_240818

theorem xy_plus_one_ge_four {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x + 1) * (y + 1) >= 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
by
  sorry

end xy_plus_one_ge_four_l240_240818


namespace line_intersects_hyperbola_l240_240851

theorem line_intersects_hyperbola (k : Real) : 
  (∃ x y : Real, y = k * x ∧ (x^2) / 9 - (y^2) / 4 = 1) ↔ (-2 / 3 < k ∧ k < 2 / 3) := 
sorry

end line_intersects_hyperbola_l240_240851


namespace min_value_proof_l240_240733

noncomputable def minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : ℝ :=
  (3 / a) + (2 / b)

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : minimum_value a b h1 h2 h3 = 25 / 2 :=
sorry

end min_value_proof_l240_240733


namespace length_of_bridge_l240_240589

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_per_hr : ℕ)
  (crossing_time_sec : ℕ)
  (h_train_length : length_of_train = 100)
  (h_speed : speed_km_per_hr = 45)
  (h_time : crossing_time_sec = 30) :
  ∃ (length_of_bridge : ℕ), length_of_bridge = 275 :=
by
  -- Convert speed from km/hr to m/s
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  -- Total distance the train travels in crossing_time_sec
  let total_distance := speed_m_per_s * crossing_time_sec
  -- Length of the bridge
  let bridge_length := total_distance - length_of_train
  use bridge_length
  -- Skip the detailed proof steps
  sorry

end length_of_bridge_l240_240589


namespace RiversideAcademy_statistics_l240_240394

theorem RiversideAcademy_statistics (total_students physics_students both_subjects : ℕ)
  (h1 : total_students = 25)
  (h2 : physics_students = 10)
  (h3 : both_subjects = 6) :
  total_students - (physics_students - both_subjects) = 21 :=
by
  sorry

end RiversideAcademy_statistics_l240_240394


namespace sixth_grade_boys_l240_240447

theorem sixth_grade_boys (x : ℕ) :
    (1 / 11) * x + (147 - x) = 147 - x → 
    (152 - (x - (1 / 11) * x + (147 - x) - (152 - x - 5))) = x
    → x = 77 :=
by
  intros h1 h2
  sorry

end sixth_grade_boys_l240_240447


namespace average_rate_of_change_is_7_l240_240304

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- Define the proof problem
theorem average_rate_of_change_is_7 : 
  ((f b - f a) / (b - a)) = 7 :=
by 
  -- The proof would go here
  sorry

end average_rate_of_change_is_7_l240_240304


namespace percentage_of_16_l240_240969

theorem percentage_of_16 (p : ℝ) (h : (p / 100) * 16 = 0.04) : p = 0.25 :=
by
  sorry

end percentage_of_16_l240_240969


namespace probability_of_rolling_five_l240_240803

-- Define a cube with the given face numbers
def cube_faces : List ℕ := [1, 1, 2, 4, 5, 5]

-- Prove the probability of rolling a "5" is 1/3
theorem probability_of_rolling_five :
  (cube_faces.count 5 : ℚ) / cube_faces.length = 1 / 3 := by
  sorry

end probability_of_rolling_five_l240_240803


namespace find_savings_l240_240211

-- Definitions and conditions from the problem
def income : ℕ := 36000
def ratio_income_to_expenditure : ℚ := 9 / 8
def expenditure : ℚ := 36000 * (8 / 9)
def savings : ℚ := income - expenditure

-- The theorem statement to prove
theorem find_savings : savings = 4000 := by
  sorry

end find_savings_l240_240211


namespace marble_problem_l240_240122

def total_marbles_originally 
  (white_marbles : ℕ := 20) 
  (blue_marbles : ℕ) 
  (red_marbles : ℕ := blue_marbles) 
  (total_left : ℕ := 40)
  (jack_removes : ℕ := 2 * (white_marbles - blue_marbles)) : ℕ :=
  white_marbles + blue_marbles + red_marbles

theorem marble_problem : 
  ∀ (white_marbles : ℕ := 20) 
    (blue_marbles red_marbles : ℕ) 
    (jack_removes total_left : ℕ),
    red_marbles = blue_marbles →
    jack_removes = 2 * (white_marbles - blue_marbles) →
    total_left = total_marbles_originally white_marbles blue_marbles red_marbles - jack_removes →
    total_left = 40 →
    total_marbles_originally white_marbles blue_marbles red_marbles = 50 :=
by
  intros white_marbles blue_marbles red_marbles jack_removes total_left h1 h2 h3 h4
  sorry

end marble_problem_l240_240122


namespace num_valid_seat_permutations_l240_240052

/-- 
  The number of ways eight people can switch their seats in a circular 
  arrangement such that no one sits in the same, adjacent, or directly 
  opposite chair they originally occupied is 5.
-/
theorem num_valid_seat_permutations : 
  ∃ (σ : Equiv.Perm (Fin 8)), 
  (∀ i : Fin 8, σ i ≠ i) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 7 then i + 1 else 0) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 8 / 2 then (i + 8 / 2) % 8 else (i - 8 / 2) % 8) :=
  sorry

end num_valid_seat_permutations_l240_240052


namespace eric_containers_l240_240444

theorem eric_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) 
  (h1 : initial_pencils = 150) (h2 : additional_pencils = 30) (h3 : pencils_per_container = 36) :
  (initial_pencils + additional_pencils) / pencils_per_container = 5 := 
by {
  sorry
}

end eric_containers_l240_240444


namespace trapezoid_circle_ratio_l240_240388

variable (P R : ℝ)

def is_isosceles_trapezoid_inscribed_in_circle (P R : ℝ) : Prop :=
  ∃ m A, 
    m = P / 4 ∧
    A = m * 2 * R ∧
    A = (P * R) / 2

theorem trapezoid_circle_ratio (P R : ℝ) 
  (h : is_isosceles_trapezoid_inscribed_in_circle P R) :
  (P / 2 * π * R) = (P / 2 * π * R) :=
by
  -- Use the given condition to prove the statement
  sorry

end trapezoid_circle_ratio_l240_240388


namespace water_height_in_tank_l240_240152

noncomputable def cone_radius := 10 -- in cm
noncomputable def cone_height := 15 -- in cm
noncomputable def tank_width := 20 -- in cm
noncomputable def tank_length := 30 -- in cm
noncomputable def cone_volume := (1/3:ℝ) * Real.pi * (cone_radius^2) * cone_height
noncomputable def tank_volume (h:ℝ) := tank_width * tank_length * h

theorem water_height_in_tank : ∃ h : ℝ, tank_volume h = cone_volume ∧ h = 5 * Real.pi / 6 := 
by 
  sorry

end water_height_in_tank_l240_240152


namespace count_valid_age_pairs_l240_240163

theorem count_valid_age_pairs :
  ∃ (d n : ℕ) (a b : ℕ), 10 * a + b ≥ 30 ∧
                       10 * b + a ≥ 35 ∧
                       b > a ∧
                       ∃ k : ℕ, k = 10 := 
sorry

end count_valid_age_pairs_l240_240163


namespace set_intersection_complement_l240_240567

open Set

def I := {n : ℕ | True}
def A := {x ∈ I | 2 ≤ x ∧ x ≤ 10}
def B := {x | Nat.Prime x}

theorem set_intersection_complement :
  A ∩ (I \ B) = {4, 6, 8, 9, 10} := by
  sorry

end set_intersection_complement_l240_240567


namespace regular_polygon_enclosure_l240_240854

theorem regular_polygon_enclosure (m n : ℕ) (h₁: m = 6) (h₂: (m + 1) = 7): n = 6 :=
by
  -- Lean code to include the problem hypothesis and conclude the theorem
  sorry

end regular_polygon_enclosure_l240_240854


namespace sum_first_five_terms_geometric_sequence_l240_240700

theorem sum_first_five_terms_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ):
  (∀ n, a (n+1) = a 1 * (1/2) ^ n) →
  a 1 = 16 →
  1/2 * (a 4 + a 7) = 9 / 8 →
  S 5 = (a 1 * (1 - (1 / 2) ^ 5)) / (1 - 1 / 2) →
  S 5 = 31 := by
  sorry

end sum_first_five_terms_geometric_sequence_l240_240700


namespace initial_hours_per_day_l240_240509

-- Definitions capturing the conditions
def num_men_initial : ℕ := 100
def num_men_total : ℕ := 160
def portion_completed : ℚ := 1 / 3
def num_days_total : ℕ := 50
def num_days_half : ℕ := 25
def work_performed_portion : ℚ := 2 / 3
def hours_per_day_additional : ℕ := 10

-- Lean statement to prove the initial number of hours per day worked by the initial employees
theorem initial_hours_per_day (H : ℚ) :
  (num_men_initial * H * num_days_total = work_performed_portion) ∧
  (num_men_total * hours_per_day_additional * num_days_half = portion_completed) →
  H = 1.6 := 
sorry

end initial_hours_per_day_l240_240509


namespace Laura_won_5_games_l240_240481

-- Define the number of wins and losses for each player
def Peter_wins : ℕ := 5
def Peter_losses : ℕ := 3
def Peter_games : ℕ := Peter_wins + Peter_losses

def Emma_wins : ℕ := 4
def Emma_losses : ℕ := 4
def Emma_games : ℕ := Emma_wins + Emma_losses

def Kyler_wins : ℕ := 2
def Kyler_losses : ℕ := 6
def Kyler_games : ℕ := Kyler_wins + Kyler_losses

-- Define the total number of games played in the tournament
def total_games_played : ℕ := (Peter_games + Emma_games + Kyler_games + 8) / 2

-- Define total wins and losses
def total_wins_losses : ℕ := total_games_played

-- Prove the number of games Laura won
def Laura_wins : ℕ := total_wins_losses - (Peter_wins + Emma_wins + Kyler_wins)

theorem Laura_won_5_games : Laura_wins = 5 := by
  -- The proof will be completed here
  sorry

end Laura_won_5_games_l240_240481


namespace simplify_expression_l240_240809

variable (q : ℝ)

theorem simplify_expression : ((6 * q + 2) - 3 * q * 5) * 4 + (5 - 2 / 4) * (7 * q - 14) = -4.5 * q - 55 :=
by sorry

end simplify_expression_l240_240809


namespace part1_part2_l240_240029

-- Definition of p: x² + 2x - 8 < 0
def p (x : ℝ) : Prop := x^2 + 2 * x - 8 < 0

-- Definition of q: (x - 1 + m)(x - 1 - m) ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Define A as the set of real numbers that satisfy p
def A : Set ℝ := { x | p x }

-- Define B as the set of real numbers that satisfy q when m = 2
def B (m : ℝ) : Set ℝ := { x | q x m }

theorem part1 : A ∩ B 2 = { x | -1 ≤ x ∧ x < 2 } :=
sorry

-- Prove that m ≥ 5 is the range for which p is a sufficient but not necessary condition for q
theorem part2 : ∀ m : ℝ, (∀ x: ℝ, p x → q x m) ∧ (∃ x: ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end part1_part2_l240_240029


namespace lily_lemonade_calories_l240_240793

def total_weight (lemonade_lime_juice lemonade_honey lemonade_water : ℕ) : ℕ :=
  lemonade_lime_juice + lemonade_honey + lemonade_water

def total_calories (weight_lime_juice weight_honey : ℕ) : ℚ :=
  (30 * weight_lime_juice / 100) + (305 * weight_honey / 100)

def calories_in_portion (total_weight total_calories portion_weight : ℚ) : ℚ :=
  (total_calories * portion_weight) / total_weight

theorem lily_lemonade_calories :
  let lemonade_lime_juice := 150
  let lemonade_honey := 150
  let lemonade_water := 450
  let portion_weight := 300
  let total_weight := total_weight lemonade_lime_juice lemonade_honey lemonade_water
  let total_calories := total_calories lemonade_lime_juice lemonade_honey
  calories_in_portion total_weight total_calories portion_weight = 201 := 
by
  sorry

end lily_lemonade_calories_l240_240793


namespace value_of_a_l240_240246

theorem value_of_a (a : ℝ) : (-2)^2 + 3*(-2) + a = 0 → a = 2 :=
by {
  sorry
}

end value_of_a_l240_240246


namespace total_cupcakes_correct_l240_240306

def cupcakes_per_event : ℝ := 96.0
def num_events : ℝ := 8.0
def total_cupcakes : ℝ := cupcakes_per_event * num_events

theorem total_cupcakes_correct : total_cupcakes = 768.0 :=
by
  unfold total_cupcakes
  unfold cupcakes_per_event
  unfold num_events
  sorry

end total_cupcakes_correct_l240_240306


namespace additional_height_last_two_floors_l240_240741

-- Definitions of the problem conditions
def num_floors : ℕ := 20
def height_per_floor : ℕ := 3
def building_total_height : ℤ := 61

-- Condition on the height of first 18 floors
def height_first_18_floors : ℤ := 18 * 3

-- Height of the last two floors
def height_last_two_floors : ℤ := building_total_height - height_first_18_floors
def height_each_last_two_floor : ℤ := height_last_two_floors / 2

-- Height difference between the last two floors and the first 18 floors
def additional_height : ℤ := height_each_last_two_floor - height_per_floor

-- Theorem to prove
theorem additional_height_last_two_floors :
  additional_height = 1 / 2 := 
sorry

end additional_height_last_two_floors_l240_240741


namespace thirty_percent_of_forty_percent_of_x_l240_240116

theorem thirty_percent_of_forty_percent_of_x (x : ℝ) (h : 0.12 * x = 24) : 0.30 * 0.40 * x = 24 :=
sorry

end thirty_percent_of_forty_percent_of_x_l240_240116


namespace even_product_divisible_by_1947_l240_240944

theorem even_product_divisible_by_1947 (n : ℕ) (h_even : n % 2 = 0) :
  (∃ k: ℕ, 2 ≤ k ∧ k ≤ n / 2 ∧ 1947 ∣ (2 ^ k * k!)) → n ≥ 3894 :=
by
  sorry

end even_product_divisible_by_1947_l240_240944


namespace students_not_picked_correct_l240_240841

-- Define the total number of students and the number of students picked for the team
def total_students := 17
def students_picked := 3 * 4

-- Define the number of students who didn't get picked based on the conditions
noncomputable def students_not_picked : ℕ := total_students - students_picked

-- The theorem stating the problem
theorem students_not_picked_correct : students_not_picked = 5 := 
by 
  sorry

end students_not_picked_correct_l240_240841


namespace ratio_of_third_layer_to_second_l240_240847

theorem ratio_of_third_layer_to_second (s1 s2 s3 : ℕ) (h1 : s1 = 2) (h2 : s2 = 2 * s1) (h3 : s3 = 12) : s3 / s2 = 3 := 
by
  sorry

end ratio_of_third_layer_to_second_l240_240847


namespace divisibility_problem_l240_240395

theorem divisibility_problem (n : ℕ) : 2016 ∣ ((n^2 + n)^2 - (n^2 - n)^2) * (n^6 - 1) := 
sorry

end divisibility_problem_l240_240395


namespace expr_div_24_l240_240930

theorem expr_div_24 (a : ℤ) : 24 ∣ ((a^2 + 3*a + 1)^2 - 1) := 
by 
  sorry

end expr_div_24_l240_240930


namespace value_of_expression_at_x_4_l240_240045

theorem value_of_expression_at_x_4 :
  ∀ (x : ℝ), x = 4 → (x^2 - 2 * x - 8) / (x - 4) = 6 :=
by
  intro x hx
  sorry

end value_of_expression_at_x_4_l240_240045


namespace sum_of_cubes_four_consecutive_integers_l240_240307

theorem sum_of_cubes_four_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 11534) :
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 74836 :=
by
  sorry

end sum_of_cubes_four_consecutive_integers_l240_240307


namespace jill_food_spending_l240_240751

theorem jill_food_spending :
  ∀ (T : ℝ) (c f o : ℝ),
    c = 0.5 * T →
    o = 0.3 * T →
    (0.04 * c + 0 + 0.1 * o) = 0.05 * T →
    f = 0.2 * T :=
by
  intros T c f o h_c h_o h_tax
  sorry

end jill_food_spending_l240_240751


namespace bianca_points_l240_240252

theorem bianca_points : 
  let a := 5; let b := 8; let c := 10;
  let A1 := 10; let P1 := 5; let G1 := 5;
  let A2 := 3; let P2 := 2; let G2 := 1;
  (A1 * a - A2 * a) + (P1 * b - P2 * b) + (G1 * c - G2 * c) = 99 := 
by
  sorry

end bianca_points_l240_240252


namespace scientific_notation_of_9280000000_l240_240761

theorem scientific_notation_of_9280000000 :
  9280000000 = 9.28 * 10^9 :=
by
  sorry

end scientific_notation_of_9280000000_l240_240761


namespace bobs_total_profit_l240_240069

theorem bobs_total_profit :
  let cost_parent_dog := 250
  let num_parent_dogs := 2
  let num_puppies := 6
  let cost_food_vaccinations := 500
  let cost_advertising := 150
  let selling_price_parent_dog := 200
  let selling_price_puppy := 350
  let total_cost_parent_dogs := num_parent_dogs * cost_parent_dog
  let total_cost_puppies := cost_food_vaccinations + cost_advertising
  let total_revenue_puppies := num_puppies * selling_price_puppy
  let total_revenue_parent_dogs := num_parent_dogs * selling_price_parent_dog
  let total_revenue := total_revenue_puppies + total_revenue_parent_dogs
  let total_cost := total_cost_parent_dogs + total_cost_puppies
  let total_profit := total_revenue - total_cost
  total_profit = 1350 :=
by
  sorry

end bobs_total_profit_l240_240069


namespace mul_fraction_eq_l240_240993

theorem mul_fraction_eq : 7 * (1 / 11) * 33 = 21 :=
by
  sorry

end mul_fraction_eq_l240_240993


namespace arithmetic_sequence_l240_240856

theorem arithmetic_sequence (a : ℕ → ℝ) 
    (h : ∀ m n, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
    ∃ d, ∀ k, a k = k * d := 
sorry

end arithmetic_sequence_l240_240856


namespace error_percentage_calc_l240_240877

theorem error_percentage_calc (y : ℝ) (hy : y > 0) : 
  let correct_result := 8 * y
  let erroneous_result := y / 8
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 := by
  sorry

end error_percentage_calc_l240_240877


namespace van_distance_l240_240228

theorem van_distance
  (D : ℝ)  -- distance the van needs to cover
  (S : ℝ)  -- original speed
  (h1 : D = S * 5)  -- the van takes 5 hours to cover the distance D
  (h2 : D = 62 * 7.5)  -- the van should maintain a speed of 62 kph to cover the same distance in 7.5 hours
  : D = 465 :=         -- prove that the distance D is 465 kilometers
by
  sorry

end van_distance_l240_240228


namespace toll_for_18_wheel_truck_l240_240769

-- Define the total number of wheels, wheels on the front axle, 
-- and wheels on each of the other axles.
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4

-- Define the formula for calculating the toll.
def toll_formula (x : ℕ) : ℝ := 2.50 + 0.50 * (x - 2)

-- Calculate the number of other axles.
def calc_other_axles (wheels_left : ℕ) (wheels_per_axle : ℕ) : ℕ :=
wheels_left / wheels_per_axle

-- Statement to prove the final toll is $4.00.
theorem toll_for_18_wheel_truck : toll_formula (
  1 + calc_other_axles (total_wheels - front_axle_wheels) other_axle_wheels
) = 4.00 :=
by sorry

end toll_for_18_wheel_truck_l240_240769


namespace min_max_ab_bc_cd_de_l240_240144

theorem min_max_ab_bc_cd_de (a b c d e : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e) (h_sum : a + b + c + d + e = 2018) : 
  ∃ a b c d e, 
  a > 0 ∧ 
  b > 0 ∧ 
  c > 0 ∧ 
  d > 0 ∧ 
  e > 0 ∧ 
  a + b + c + d + e = 2018 ∧ 
  ∀ M, M = max (max (max (a + b) (b + c)) (max (c + d) (d + e))) ↔ M = 673  :=
sorry

end min_max_ab_bc_cd_de_l240_240144


namespace number_of_sheets_l240_240601

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l240_240601


namespace total_number_of_wheels_l240_240696

-- Define the conditions as hypotheses
def cars := 2
def wheels_per_car := 4

def bikes := 2
def trashcans := 1
def wheels_per_bike_or_trashcan := 2

def roller_skates_pair := 1
def wheels_per_skate := 4

def tricycle := 1
def wheels_per_tricycle := 3

-- Prove the total number of wheels
theorem total_number_of_wheels :
  cars * wheels_per_car +
  (bikes + trashcans) * wheels_per_bike_or_trashcan +
  (roller_skates_pair * 2) * wheels_per_skate +
  tricycle * wheels_per_tricycle 
  = 25 :=
by
  sorry

end total_number_of_wheels_l240_240696


namespace no_three_digits_all_prime_l240_240580

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function that forms a three-digit number from digits a, b, c
def form_three_digit (a b c : ℕ) : ℕ :=
100 * a + 10 * b + c

-- Define a function to check if all permutations of three digits form prime numbers
def all_permutations_prime (a b c : ℕ) : Prop :=
is_prime (form_three_digit a b c) ∧
is_prime (form_three_digit a c b) ∧
is_prime (form_three_digit b a c) ∧
is_prime (form_three_digit b c a) ∧
is_prime (form_three_digit c a b) ∧
is_prime (form_three_digit c b a)

-- The main theorem stating that there are no three distinct digits making all permutations prime
theorem no_three_digits_all_prime : ¬∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  all_permutations_prime a b c :=
sorry

end no_three_digits_all_prime_l240_240580


namespace find_second_number_l240_240097

theorem find_second_number
  (a : ℝ) (b : ℝ)
  (h : a = 1280)
  (h_percent : 0.25 * a = 0.20 * b + 190) :
  b = 650 :=
sorry

end find_second_number_l240_240097


namespace average_speed_l240_240514

theorem average_speed (d1 d2 t1 t2 : ℝ) 
  (h1 : d1 = 100) 
  (h2 : d2 = 80) 
  (h3 : t1 = 1) 
  (h4 : t2 = 1) : 
  (d1 + d2) / (t1 + t2) = 90 := 
by 
  sorry

end average_speed_l240_240514


namespace continuous_at_3_l240_240749

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 3 then x^2 + x + 2 else 2 * x + a

theorem continuous_at_3 {a : ℝ} : (∀ x : ℝ, 0 < abs (x - 3) → abs (f x a - f 3 a) < 0.0001) →
a = 8 :=
by
  sorry

end continuous_at_3_l240_240749


namespace add_fractions_l240_240978

theorem add_fractions : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by sorry

end add_fractions_l240_240978


namespace chicken_cost_l240_240688
noncomputable def chicken_cost_per_plate
  (plates : ℕ) 
  (rice_cost_per_plate : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_rice_cost := plates * rice_cost_per_plate
  let total_chicken_cost := total_cost - total_rice_cost
  total_chicken_cost / plates

theorem chicken_cost
  (hplates : plates = 100)
  (hrice_cost_per_plate : rice_cost_per_plate = 0.10)
  (htotal_cost : total_cost = 50) :
  chicken_cost_per_plate 100 0.10 50 = 0.40 :=
by
  sorry

end chicken_cost_l240_240688


namespace no_such_continuous_function_exists_l240_240065

theorem no_such_continuous_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (Continuous f) ∧ ∀ x : ℝ, ((∃ q : ℚ, f x = q) ↔ ∀ q' : ℚ, f (x + 1) ≠ q') :=
sorry

end no_such_continuous_function_exists_l240_240065


namespace initial_deadline_in_days_l240_240170

theorem initial_deadline_in_days
  (men_initial : ℕ)
  (days_initial : ℕ)
  (hours_per_day_initial : ℕ)
  (fraction_work_initial : ℚ)
  (additional_men : ℕ)
  (hours_per_day_additional : ℕ)
  (fraction_work_additional : ℚ)
  (total_work : ℚ := men_initial * days_initial * hours_per_day_initial)
  (remaining_days : ℚ := (men_initial * days_initial * hours_per_day_initial) / (additional_men * hours_per_day_additional * fraction_work_additional))
  (total_days : ℚ := days_initial + remaining_days) :
  men_initial = 100 →
  days_initial = 25 →
  hours_per_day_initial = 8 →
  fraction_work_initial = 1 / 3 →
  additional_men = 160 →
  hours_per_day_additional = 10 →
  fraction_work_additional = 2 / 3 →
  total_days = 37.5 :=
by
  intros
  sorry

end initial_deadline_in_days_l240_240170


namespace measure_orthogonal_trihedral_angle_sum_measure_polyhedral_angles_l240_240017

theorem measure_orthogonal_trihedral_angle (d : ℕ) (a : ℝ) (n : ℕ) 
(h1 : d = 3) (h2 : a = π / 2) (h3 : n = 8) : 
  ∃ measure : ℝ, measure = π / 2 :=
by
  sorry

theorem sum_measure_polyhedral_angles (d : ℕ) (a : ℝ) (n : ℕ) 
(h1 : d = 3) (h2 : a = π / 2) (h3 : n = 8) 
(h4 : n * a = 4 * π) : 
  ∃ sum_measure : ℝ, sum_measure = 4 * π :=
by
  sorry

end measure_orthogonal_trihedral_angle_sum_measure_polyhedral_angles_l240_240017


namespace crayons_per_child_l240_240489

theorem crayons_per_child (total_crayons children : ℕ) (h_total : total_crayons = 56) (h_children : children = 7) : (total_crayons / children) = 8 := by
  -- proof will go here
  sorry

end crayons_per_child_l240_240489


namespace conference_handshakes_l240_240938

theorem conference_handshakes (n_leaders n_participants : ℕ) (n_total : ℕ) 
  (h_total : n_total = n_leaders + n_participants) 
  (h_leaders : n_leaders = 5) 
  (h_participants : n_participants = 25) 
  (h_total_people : n_total = 30) : 
  (n_leaders * (n_total - 1) - (n_leaders * (n_leaders - 1) / 2)) = 135 := 
by 
  sorry

end conference_handshakes_l240_240938


namespace max_perimeter_of_triangle_l240_240101

theorem max_perimeter_of_triangle (x : ℕ) 
  (h1 : 3 < x) 
  (h2 : x < 15) 
  (h3 : 7 + 8 > x) 
  (h4 : 7 + x > 8) 
  (h5 : 8 + x > 7) :
  x = 14 ∧ 7 + 8 + x = 29 := 
by {
  sorry
}

end max_perimeter_of_triangle_l240_240101


namespace total_beads_needed_l240_240199

-- Condition 1: Number of members in the crafts club
def members := 9

-- Condition 2: Number of necklaces each member makes
def necklaces_per_member := 2

-- Condition 3: Number of beads each necklace requires
def beads_per_necklace := 50

-- Total number of beads needed
theorem total_beads_needed :
  (members * (necklaces_per_member * beads_per_necklace)) = 900 := 
by
  sorry

end total_beads_needed_l240_240199


namespace hyperbola_condition_l240_240148

theorem hyperbola_condition (m : ℝ) : 
  (exists a b : ℝ, ¬ a = 0 ∧ ¬ b = 0 ∧ ( ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 )) →
  ( -2 < m ∧ m < -1 ) :=
by
  sorry

end hyperbola_condition_l240_240148


namespace problem_l240_240517

theorem problem (a b : ℚ) (h : a / b = 6 / 5) : (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := 
by 
  sorry

end problem_l240_240517


namespace largest_x_eq_neg5_l240_240311

theorem largest_x_eq_neg5 (x : ℝ) (h : x ≠ 7) : (x^2 - 5*x - 84)/(x - 7) = 2/(x + 6) → x ≤ -5 := 
sorry

end largest_x_eq_neg5_l240_240311


namespace chord_length_of_intersecting_circle_and_line_l240_240579

-- Define the conditions in Lean
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ
def line_equation (ρ θ : ℝ) : Prop := 3 * ρ * Real.cos θ - 4 * ρ * Real.sin θ - 1 = 0

-- Define the problem to prove the length of the chord
theorem chord_length_of_intersecting_circle_and_line 
  (ρ θ : ℝ) (hC : circle_equation ρ θ) (hL : line_equation ρ θ) : 
  ∃ l : ℝ, l = 2 * Real.sqrt 3 :=
by 
  sorry

end chord_length_of_intersecting_circle_and_line_l240_240579


namespace count_remainders_gte_l240_240695

def remainder (a N : ℕ) : ℕ := a % N

theorem count_remainders_gte (N : ℕ) : 
  (∀ a, a > 0 → remainder a 1000 > remainder a 1001 → N ≤ 1000000) →
  N = 499500 :=
by
  sorry

end count_remainders_gte_l240_240695


namespace second_person_fraction_removed_l240_240778

theorem second_person_fraction_removed (teeth_total : ℕ) 
    (removed1 removed3 removed4 : ℕ)
    (total_removed: ℕ)
    (h1: teeth_total = 32)
    (h2: removed1 = teeth_total / 4)
    (h3: removed3 = teeth_total / 2)
    (h4: removed4 = 4)
    (h5 : total_removed = 40):
    ((total_removed - (removed1 + removed3 + removed4)) : ℚ) / teeth_total = 3 / 8 :=
by
  sorry

end second_person_fraction_removed_l240_240778


namespace inequality_transformations_l240_240000

theorem inequality_transformations (a b : ℝ) (h : a > b) :
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) :=
by
  sorry

end inequality_transformations_l240_240000


namespace boys_planted_more_by_62_percent_girls_fraction_of_total_l240_240410

-- Define the number of trees planted by boys and girls
def boys_trees : ℕ := 130
def girls_trees : ℕ := 80

-- Statement 1: Boys planted 62% more trees than girls
theorem boys_planted_more_by_62_percent : (boys_trees - girls_trees) * 100 / girls_trees = 62 := by
  sorry

-- Statement 2: The number of trees planted by girls represents 4/7 of the total number of trees
theorem girls_fraction_of_total : girls_trees * 7 = 4 * (boys_trees + girls_trees) := by
  sorry

end boys_planted_more_by_62_percent_girls_fraction_of_total_l240_240410


namespace rectangle_square_division_l240_240970

theorem rectangle_square_division (a b : ℝ) (n : ℕ) (h1 : (∃ (s1 : ℝ), s1^2 * (n : ℝ) = a * b))
                                            (h2 : (∃ (s2 : ℝ), s2^2 * (n + 76 : ℝ) = a * b)) :
    n = 324 := 
by
  sorry

end rectangle_square_division_l240_240970


namespace inradius_of_triangle_l240_240975

variable (A : ℝ) (p : ℝ) (r : ℝ) (s : ℝ)

theorem inradius_of_triangle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_l240_240975


namespace base2_to_base4_conversion_l240_240378

/-- Definition of base conversion from binary to quaternary. -/
def bin_to_quat (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
  if n = 10 then 2 else
  if n = 11 then 3 else
  0 -- (more cases can be added as necessary)

theorem base2_to_base4_conversion :
  bin_to_quat 1 * 4^4 + bin_to_quat 1 * 4^3 + bin_to_quat 10 * 4^2 + bin_to_quat 11 * 4^1 + bin_to_quat 10 * 4^0 = 11232 :=
by sorry

end base2_to_base4_conversion_l240_240378


namespace possible_value_of_sum_l240_240064

theorem possible_value_of_sum (p q r : ℝ) (h₀ : q = p * (4 - p)) (h₁ : r = q * (4 - q)) (h₂ : p = r * (4 - r)) 
  (h₃ : p ≠ q ∧ p ≠ r ∧ q ≠ r) : p + q + r = 6 :=
sorry

end possible_value_of_sum_l240_240064


namespace find_smaller_integer_l240_240403

theorem find_smaller_integer (x : ℤ) (h1 : ∃ y : ℤ, y = 2 * x) (h2 : x + 2 * x = 96) : x = 32 :=
sorry

end find_smaller_integer_l240_240403


namespace finite_tasty_integers_l240_240271

def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (c : ℕ), (b = c * 2^a * 5^a)

def is_tasty (n : ℕ) : Prop :=
  n > 2 ∧ ∀ (a b : ℕ), a + b = n → (is_terminating_decimal a b ∨ is_terminating_decimal b a)

theorem finite_tasty_integers : 
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → ¬ is_tasty n :=
sorry

end finite_tasty_integers_l240_240271


namespace consecutive_integer_sum_l240_240914

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l240_240914


namespace isosceles_right_triangle_side_length_l240_240153

theorem isosceles_right_triangle_side_length
  (a b : ℝ)
  (h_triangle : a = b ∨ b = a)
  (h_hypotenuse : xy > yz)
  (h_area : (1 / 2) * a * b = 9) :
  xy = 6 :=
by
  -- proof will go here
  sorry

end isosceles_right_triangle_side_length_l240_240153


namespace total_metal_rods_needed_l240_240605

-- Definitions extracted from the conditions
def metal_sheets_per_panel := 3
def metal_beams_per_panel := 2
def panels := 10
def rods_per_sheet := 10
def rods_per_beam := 4

-- Problem statement: Prove the total number of metal rods required is 380
theorem total_metal_rods_needed :
  (panels * ((metal_sheets_per_panel * rods_per_sheet) + (metal_beams_per_panel * rods_per_beam))) = 380 :=
by
  sorry

end total_metal_rods_needed_l240_240605


namespace train_speed_l240_240231

theorem train_speed (length_m : ℝ) (time_s : ℝ) (h_length : length_m = 133.33333333333334) (h_time : time_s = 8) : 
  let length_km := length_m / 1000
  let time_hr := time_s / 3600
  length_km / time_hr = 60 :=
by
  sorry

end train_speed_l240_240231


namespace possible_sums_of_digits_l240_240056

-- Defining the main theorem
theorem possible_sums_of_digits 
  (A B C : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (hdiv : (A + 6 + 2 + 8 + B + 7 + C + 3) % 9 = 0) :
  A + B + C = 1 ∨ A + B + C = 10 ∨ A + B + C = 19 :=
by
  sorry

end possible_sums_of_digits_l240_240056


namespace min_value_fraction_sum_l240_240498

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (4 / (x + 2) + 1 / (y + 1)) ≥ 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l240_240498


namespace lowest_discount_l240_240922

theorem lowest_discount (c m : ℝ) (p : ℝ) (h_c : c = 100) (h_m : m = 150) (h_p : p = 0.05) :
  ∃ (x : ℝ), m * (x / 100) = c * (1 + p) ∧ x = 70 :=
by
  use 70
  sorry

end lowest_discount_l240_240922


namespace total_travel_time_correct_l240_240243

-- Define the conditions
def highway_distance : ℕ := 100 -- miles
def mountain_distance : ℕ := 15 -- miles
def break_time : ℕ := 30 -- minutes
def time_on_mountain_road : ℕ := 45 -- minutes
def speed_ratio : ℕ := 5

-- Define the speeds using the given conditions.
def mountain_speed := mountain_distance / time_on_mountain_road -- miles per minute
def highway_speed := speed_ratio * mountain_speed -- miles per minute

-- Prove that total trip time equals 240 minutes
def total_trip_time : ℕ := 2 * (time_on_mountain_road + (highway_distance / highway_speed)) + break_time

theorem total_travel_time_correct : total_trip_time = 240 := 
by
  -- to be proved
  sorry

end total_travel_time_correct_l240_240243


namespace avg_weight_A_l240_240015

-- Define the conditions
def num_students_A : ℕ := 40
def num_students_B : ℕ := 20
def avg_weight_B : ℝ := 40
def avg_weight_whole_class : ℝ := 46.67

-- State the theorem using these definitions
theorem avg_weight_A :
  ∃ W_A : ℝ,
    (num_students_A * W_A + num_students_B * avg_weight_B = (num_students_A + num_students_B) * avg_weight_whole_class) ∧
    W_A = 50.005 :=
by
  sorry

end avg_weight_A_l240_240015


namespace birds_joined_l240_240609

def numBirdsInitially : Nat := 1
def numBirdsNow : Nat := 5

theorem birds_joined : numBirdsNow - numBirdsInitially = 4 := by
  -- proof goes here
  sorry

end birds_joined_l240_240609


namespace proofSmallestM_l240_240480

def LeanProb (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 2512 →
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧ (0 < e) ∧ (0 < f) →
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))))

theorem proofSmallestM (a b c d e f : ℕ) (h1 : a + b + c + d + e + f = 2512) 
(h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < d) (h6 : 0 < e) (h7 : 0 < f) : 
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f))))):=
by
  sorry

end proofSmallestM_l240_240480


namespace sum_max_min_f_l240_240370

noncomputable def f (x : ℝ) : ℝ :=
  1 + (Real.sin x / (2 + Real.cos x))

theorem sum_max_min_f {a b : ℝ} (ha : ∀ x, f x ≤ a) (hb : ∀ x, b ≤ f x) (h_max : ∃ x, f x = a) (h_min : ∃ x, f x = b) :
  a + b = 2 :=
sorry

end sum_max_min_f_l240_240370


namespace problem_solution_l240_240242

def p : Prop := ∀ x : ℝ, |x| ≥ 0
def q : Prop := ∃ x : ℝ, x = 2 ∧ x + 2 = 0

theorem problem_solution : p ∧ ¬q :=
by
  -- Here we would provide the proof to show that p ∧ ¬q is true
  sorry

end problem_solution_l240_240242


namespace cats_given_by_Mr_Sheridan_l240_240996

-- Definitions of the initial state and final state
def initial_cats : Nat := 17
def total_cats : Nat := 31

-- Proof statement that Mr. Sheridan gave her 14 cats
theorem cats_given_by_Mr_Sheridan : total_cats - initial_cats = 14 := by
  sorry

end cats_given_by_Mr_Sheridan_l240_240996


namespace arithmetic_sequence_sum_ratio_l240_240899

theorem arithmetic_sequence_sum_ratio
  (a_n : ℕ → ℝ)
  (d a1 : ℝ)
  (S_n : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n = a1 + (n-1) * d)
  (h_sum : ∀ n, S_n n = n / 2 * (2 * a1 + (n-1) * d))
  (h_ratio : S_n 4 / S_n 6 = -2 / 3) :
  S_n 5 / S_n 8 = 1 / 40.8 :=
sorry

end arithmetic_sequence_sum_ratio_l240_240899


namespace incorrect_inequality_l240_240926

theorem incorrect_inequality (a b : ℝ) (h : a > b ∧ b > 0) :
  ¬ (1 / a > 1 / b) :=
by
  sorry

end incorrect_inequality_l240_240926


namespace minimum_expression_value_l240_240330

theorem minimum_expression_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := 
by
  sorry

end minimum_expression_value_l240_240330


namespace Morio_age_when_Michiko_was_born_l240_240511

theorem Morio_age_when_Michiko_was_born (Teresa_age_now : ℕ) (Teresa_age_when_Michiko_born : ℕ) (Morio_age_now : ℕ)
  (hTeresa : Teresa_age_now = 59) (hTeresa_born : Teresa_age_when_Michiko_born = 26) (hMorio : Morio_age_now = 71) :
  Morio_age_now - (Teresa_age_now - Teresa_age_when_Michiko_born) = 38 :=
by
  sorry

end Morio_age_when_Michiko_was_born_l240_240511


namespace prove_f_2013_l240_240703

-- Defining the function f that satisfies the given conditions
variable (f : ℕ → ℕ)

-- Conditions provided in the problem
axiom cond1 : ∀ n, f (f n) + f n = 2 * n + 3
axiom cond2 : f 0 = 1
axiom cond3 : f 2014 = 2015

-- The statement to be proven
theorem prove_f_2013 : f 2013 = 2014 := sorry

end prove_f_2013_l240_240703


namespace exists_integers_cd_iff_divides_l240_240929

theorem exists_integers_cd_iff_divides (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (a - b) ∣ (2 * a * b) := 
by
  sorry

end exists_integers_cd_iff_divides_l240_240929


namespace julia_internet_speed_l240_240280

theorem julia_internet_speed
  (songs : ℕ) (song_size : ℕ) (time_sec : ℕ)
  (h_songs : songs = 7200)
  (h_song_size : song_size = 5)
  (h_time_sec : time_sec = 1800) :
  songs * song_size / time_sec = 20 := by
  sorry

end julia_internet_speed_l240_240280


namespace weight_of_11th_person_l240_240997

theorem weight_of_11th_person
  (n : ℕ) (avg1 avg2 : ℝ)
  (hn : n = 10)
  (havg1 : avg1 = 165)
  (havg2 : avg2 = 170)
  (W : ℝ) (X : ℝ)
  (hw : W = n * avg1)
  (havg2_eq : (W + X) / (n + 1) = avg2) :
  X = 220 :=
by
  sorry

end weight_of_11th_person_l240_240997


namespace construct_triangle_condition_l240_240708

theorem construct_triangle_condition (m_a f_a s_a : ℝ) : 
  (m_a < f_a) ∧ (f_a < s_a) ↔ (exists A B C : Type, true) :=
sorry

end construct_triangle_condition_l240_240708


namespace functions_from_M_to_N_l240_240381

def M : Set ℤ := { -1, 1, 2, 3 }
def N : Set ℤ := { 0, 1, 2, 3, 4 }
def f2 (x : ℤ) := x + 1
def f4 (x : ℤ) := (x - 1)^2

theorem functions_from_M_to_N :
  (∀ x ∈ M, f2 x ∈ N) ∧ (∀ x ∈ M, f4 x ∈ N) :=
by
  sorry

end functions_from_M_to_N_l240_240381


namespace find_principal_l240_240298

theorem find_principal (R : ℝ) : ∃ P : ℝ, (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 100 :=
by {
  use 200,
  sorry
}

end find_principal_l240_240298


namespace average_movers_per_hour_l240_240655

-- Define the main problem parameters
def total_people : ℕ := 3200
def days : ℕ := 4
def hours_per_day : ℕ := 24
def total_hours : ℕ := hours_per_day * days
def average_people_per_hour := total_people / total_hours

-- State the theorem to prove
theorem average_movers_per_hour :
  average_people_per_hour = 33 :=
by
  -- Proof is omitted
  sorry

end average_movers_per_hour_l240_240655


namespace T_53_eq_38_l240_240415

def T (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem T_53_eq_38 : T 5 3 = 38 := by
  sorry

end T_53_eq_38_l240_240415


namespace visitors_that_day_l240_240726

theorem visitors_that_day (total_visitors : ℕ) (previous_day_visitors : ℕ) 
  (h_total : total_visitors = 406) (h_previous : previous_day_visitors = 274) : 
  total_visitors - previous_day_visitors = 132 :=
by
  sorry

end visitors_that_day_l240_240726


namespace find_c_l240_240869

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 + 19 * x - 84
noncomputable def g (x : ℝ) : ℝ := 4 * x ^ 2 - 12 * x + 5

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ f x = 0)
  (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ g x = 0) :
  c = -23 / 2 := by
  sorry

end find_c_l240_240869


namespace karen_cases_pickup_l240_240387

theorem karen_cases_pickup (total_boxes cases_per_box: ℕ) (h1 : total_boxes = 36) (h2 : cases_per_box = 12):
  total_boxes / cases_per_box = 3 :=
by
  -- We insert a placeholder to skip the proof here
  sorry

end karen_cases_pickup_l240_240387


namespace fraction_ordering_l240_240300

theorem fraction_ordering (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) :=
by
  sorry

end fraction_ordering_l240_240300


namespace hockey_games_in_season_l240_240981

theorem hockey_games_in_season
  (games_per_month : ℤ)
  (months_in_season : ℤ)
  (h1 : games_per_month = 25)
  (h2 : months_in_season = 18) :
  games_per_month * months_in_season = 450 :=
by
  sorry

end hockey_games_in_season_l240_240981


namespace total_cows_is_108_l240_240340

-- Definitions of the sons' shares and the number of cows the fourth son received
def first_son_share : ℚ := 2 / 3
def second_son_share : ℚ := 1 / 6
def third_son_share : ℚ := 1 / 9
def fourth_son_cows : ℕ := 6

-- The total number of cows in the herd
def total_cows (n : ℕ) : Prop :=
  first_son_share + second_son_share + third_son_share + (fourth_son_cows / n) = 1

-- Prove that given the number of cows the fourth son received, the total number of cows in the herd is 108
theorem total_cows_is_108 : total_cows 108 :=
by
  sorry

end total_cows_is_108_l240_240340


namespace initial_balance_l240_240982

theorem initial_balance (B : ℝ) (payment : ℝ) (new_balance : ℝ)
  (h1 : payment = 50) (h2 : new_balance = 120) (h3 : B - payment = new_balance) :
  B = 170 :=
by
  rw [h1, h2] at h3
  linarith

end initial_balance_l240_240982


namespace chicago_bulls_heat_games_total_l240_240896

-- Statement of the problem in Lean 4
theorem chicago_bulls_heat_games_total :
  ∀ (bulls_games : ℕ) (heat_games : ℕ),
    bulls_games = 70 →
    heat_games = bulls_games + 5 →
    bulls_games + heat_games = 145 :=
by
  intros bulls_games heat_games h_bulls h_heat
  rw [h_bulls, h_heat]
  exact sorry

end chicago_bulls_heat_games_total_l240_240896


namespace product_of_two_numbers_l240_240111

theorem product_of_two_numbers : 
  ∀ (x y : ℝ), (x + y = 60) ∧ (x - y = 10) → x * y = 875 :=
by
  intros x y h
  sorry

end product_of_two_numbers_l240_240111


namespace transformation_correct_l240_240795

theorem transformation_correct (a x y : ℝ) (h : a * x = a * y) : 3 - a * x = 3 - a * y :=
sorry

end transformation_correct_l240_240795


namespace Ms_Smiths_Class_Books_Distribution_l240_240554

theorem Ms_Smiths_Class_Books_Distribution :
  ∃ (x : ℕ), (20 * 2 * x + 15 * x + 5 * x = 840) ∧ (20 * 2 * x = 560) ∧ (15 * x = 210) ∧ (5 * x = 70) :=
by
  let x := 14
  have h1 : 20 * 2 * x + 15 * x + 5 * x = 840 := by sorry
  have h2 : 20 * 2 * x = 560 := by sorry
  have h3 : 15 * x = 210 := by sorry
  have h4 : 5 * x = 70 := by sorry
  exact ⟨x, h1, h2, h3, h4⟩

end Ms_Smiths_Class_Books_Distribution_l240_240554


namespace find_a1_l240_240053

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

noncomputable def sumOfArithmeticSequence (a d : α) (n : ℕ) : α :=
  n * a + d * (n * (n - 1) / 2)

theorem find_a1 (a1 d : α) :
  arithmeticSequence a1 d 2 + arithmeticSequence a1 d 8 = 34 →
  sumOfArithmeticSequence a1 d 4 = 38 →
  a1 = 5 :=
by
  intros h1 h2
  sorry

end find_a1_l240_240053


namespace simplify_complex_number_l240_240990

theorem simplify_complex_number (i : ℂ) (h : i^2 = -1) : i * (1 - i)^2 = 2 := by
  sorry

end simplify_complex_number_l240_240990


namespace bus_stops_for_18_minutes_l240_240478

-- Definitions based on conditions
def speed_without_stoppages : ℝ := 50 -- kmph
def speed_with_stoppages : ℝ := 35 -- kmph
def distance_reduced_due_to_stoppage_per_hour : ℝ := speed_without_stoppages - speed_with_stoppages

noncomputable def time_bus_stops_per_hour (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem bus_stops_for_18_minutes :
  time_bus_stops_per_hour distance_reduced_due_to_stoppage_per_hour (speed_without_stoppages / 60) = 18 := by
  sorry

end bus_stops_for_18_minutes_l240_240478


namespace cadence_total_earnings_l240_240594

/-- Cadence's total earnings in both companies. -/
def total_earnings (old_salary_per_month new_salary_per_month : ℕ) (old_company_months new_company_months : ℕ) : ℕ :=
  (old_salary_per_month * old_company_months) + (new_salary_per_month * new_company_months)

theorem cadence_total_earnings :
  let old_salary_per_month := 5000
  let old_company_years := 3
  let months_per_year := 12
  let old_company_months := old_company_years * months_per_year
  let new_salary_per_month := old_salary_per_month + (old_salary_per_month * 20 / 100)
  let new_company_extra_months := 5
  let new_company_months := old_company_months + new_company_extra_months
  total_earnings old_salary_per_month new_salary_per_month old_company_months new_company_months = 426000 := by
sorry

end cadence_total_earnings_l240_240594


namespace sum_k1_k2_k3_l240_240329

theorem sum_k1_k2_k3 :
  ∀ (k1 k2 k3 t1 t2 t3 : ℝ),
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  t1 = (5 / 9) * (k1 - 32) →
  t2 = (5 / 9) * (k2 - 32) →
  t3 = (5 / 9) * (k3 - 32) →
  k1 + k2 + k3 = 510 :=
by
  intros k1 k2 k3 t1 t2 t3 ht1 ht2 ht3 ht1k1 ht2k2 ht3k3
  sorry

end sum_k1_k2_k3_l240_240329


namespace tree_growth_rate_consistency_l240_240027

theorem tree_growth_rate_consistency (a b : ℝ) :
  (a + b) / 2 = 0.15 ∧ (1 + a) * (1 + b) = 0.90 → ∃ a b : ℝ, (a + b) / 2 = 0.15 ∧ (1 + a) * (1 + b) = 0.90 := by
  sorry

end tree_growth_rate_consistency_l240_240027


namespace regression_analysis_correct_l240_240814

-- Definition of the regression analysis context
def regression_analysis_variation (forecast_var : Type) (explanatory_var residual_var : Type) : Prop :=
  forecast_var = explanatory_var ∧ forecast_var = residual_var

-- The theorem to prove
theorem regression_analysis_correct :
  ∀ (forecast_var explanatory_var residual_var : Type),
  regression_analysis_variation forecast_var explanatory_var residual_var →
  (forecast_var = explanatory_var ∧ forecast_var = residual_var) :=
by
  intro forecast_var explanatory_var residual_var h
  exact h

end regression_analysis_correct_l240_240814


namespace probability_no_shaded_in_2_by_2004_l240_240669

noncomputable def probability_no_shaded_rectangle (total_rectangles shaded_rectangles : Nat) : ℚ :=
  1 - (shaded_rectangles : ℚ) / (total_rectangles : ℚ)

theorem probability_no_shaded_in_2_by_2004 :
  let rows := 2
  let cols := 2004
  let total_rectangles := (cols + 1) * cols / 2 * rows
  let shaded_rectangles := 501 * 2507 
  probability_no_shaded_rectangle total_rectangles shaded_rectangles = 1501 / 4008 :=
by
  sorry

end probability_no_shaded_in_2_by_2004_l240_240669


namespace find_f_l240_240704

theorem find_f (f : ℕ → ℕ) :
  (∀ a b c : ℕ, ((f a + f b + f c) - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) →
  (∀ n : ℕ, f n = n * n) :=
sorry

end find_f_l240_240704


namespace range_of_x_l240_240830

def star (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x (x : ℝ) (h : star x (x - 2) < 0) : -2 < x ∧ x < 1 := by
  sorry

end range_of_x_l240_240830


namespace intersect_complement_l240_240402

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Definition of set A
def A : Set ℕ := {1, 2, 3}

-- Definition of set B
def B : Set ℕ := {3, 4}

-- Definition of the complement of B in U
def CU (U : Set ℕ) (B : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- Expected result of the intersection
def result : Set ℕ := {1, 2}

-- The proof statement
theorem intersect_complement :
  A ∩ CU U B = result :=
sorry

end intersect_complement_l240_240402


namespace boat_distance_against_stream_l240_240013

-- Definitions from Step a)
def speed_boat_still_water : ℝ := 15  -- speed of the boat in still water in km/hr
def distance_downstream : ℝ := 21  -- distance traveled downstream in one hour in km
def time_hours : ℝ := 1  -- time in hours

-- Translation of the described problem proof
theorem boat_distance_against_stream :
  ∃ (v_s : ℝ), (speed_boat_still_water + v_s = distance_downstream / time_hours) → 
               (15 - v_s = 9) :=
by
  sorry

end boat_distance_against_stream_l240_240013


namespace coin_collection_problem_l240_240570

theorem coin_collection_problem (n : ℕ) 
  (quarters : ℕ := n / 2)
  (half_dollars : ℕ := 2 * (n / 2))
  (value_nickels : ℝ := 0.05 * n)
  (value_quarters : ℝ := 0.25 * (n / 2))
  (value_half_dollars : ℝ := 0.5 * (2 * (n / 2)))
  (total_value : ℝ := value_nickels + value_quarters + value_half_dollars) :
  total_value = 67.5 ∨ total_value = 135 :=
sorry

end coin_collection_problem_l240_240570


namespace antonio_weight_l240_240292

-- Let A be the weight of Antonio
variable (A : ℕ)

-- Conditions:
-- 1. Antonio's sister weighs A - 12 kilograms.
-- 2. The total weight of Antonio and his sister is 88 kilograms.

theorem antonio_weight (A: ℕ) (h1: A - 12 >= 0) (h2: A + (A - 12) = 88) : A = 50 := by
  sorry

end antonio_weight_l240_240292


namespace quarters_count_l240_240490

theorem quarters_count (total_money : ℝ) (value_of_quarter : ℝ) (h1 : total_money = 3) (h2 : value_of_quarter = 0.25) : total_money / value_of_quarter = 12 :=
by sorry

end quarters_count_l240_240490


namespace factorization_roots_l240_240494

theorem factorization_roots (x : ℂ) : 
  (x^3 - 2*x^2 - x + 2) * (x - 3) * (x + 1) = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  -- Note: Proof to be completed
  sorry

end factorization_roots_l240_240494


namespace true_proposition_l240_240575

-- Define proposition p
def p : Prop := ∀ x : ℝ, Real.log (x^2 + 4) / Real.log 2 ≥ 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ≥ 0 → x^(1/2) ≤ x^(1/2)

-- Theorem: true proposition is p ∨ ¬q
theorem true_proposition : p ∨ ¬q :=
by
  sorry

end true_proposition_l240_240575


namespace eccentricity_is_sqrt2_div2_l240_240457

noncomputable def eccentricity_square_ellipse (a b c : ℝ) : ℝ :=
  c / (Real.sqrt (b ^ 2 + c ^ 2))

theorem eccentricity_is_sqrt2_div2 (a b c : ℝ) (h : b = c) : 
  eccentricity_square_ellipse a b c = Real.sqrt 2 / 2 :=
by
  -- The proof will show that the eccentricity calculation is correct given the conditions.
  sorry

end eccentricity_is_sqrt2_div2_l240_240457


namespace mother_kept_one_third_l240_240103

-- Define the problem conditions
def total_sweets : ℕ := 27
def eldest_sweets : ℕ := 8
def youngest_sweets : ℕ := eldest_sweets / 2
def second_sweets : ℕ := 6
def total_children_sweets : ℕ := eldest_sweets + youngest_sweets + second_sweets
def sweets_mother_kept : ℕ := total_sweets - total_children_sweets
def fraction_mother_kept : ℚ := sweets_mother_kept / total_sweets

-- Prove the fraction of sweets the mother kept
theorem mother_kept_one_third : fraction_mother_kept = 1 / 3 := 
  by
    sorry

end mother_kept_one_third_l240_240103


namespace correct_calculation_l240_240721

theorem correct_calculation :
  (3 * Real.sqrt 2) * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 :=
by sorry

end correct_calculation_l240_240721


namespace correct_number_of_statements_l240_240972

noncomputable def number_of_correct_statements := 1

def statement_1 : Prop := false -- Equal angles are not preserved
def statement_2 : Prop := false -- Equal lengths are not preserved
def statement_3 : Prop := false -- The longest segment feature is not preserved
def statement_4 : Prop := true  -- The midpoint feature is preserved

theorem correct_number_of_statements :
  (statement_1 ∧ statement_2 ∧ statement_3 ∧ statement_4) = true →
  number_of_correct_statements = 1 :=
by
  sorry

end correct_number_of_statements_l240_240972


namespace bus_speed_including_stoppages_l240_240468

theorem bus_speed_including_stoppages :
  ∀ (s t : ℝ), s = 75 → t = 24 → (s * ((60 - t) / 60)) = 45 :=
by
  intros s t hs ht
  rw [hs, ht]
  sorry

end bus_speed_including_stoppages_l240_240468


namespace evaluate_difference_of_squares_l240_240254

theorem evaluate_difference_of_squares : 81^2 - 49^2 = 4160 := by
  sorry

end evaluate_difference_of_squares_l240_240254


namespace sequence_fill_l240_240145

theorem sequence_fill (x2 x3 x4 x5 x6 x7: ℕ) : 
  (20 + x2 + x3 = 100) ∧ 
  (x2 + x3 + x4 = 100) ∧ 
  (x3 + x4 + x5 = 100) ∧ 
  (x4 + x5 + x6 = 100) ∧ 
  (x5 + x6 + 16 = 100) →
  [20, x2, x3, x4, x5, x6, 16] = [20, 16, 64, 20, 16, 64, 20, 16] :=
by
  sorry

end sequence_fill_l240_240145


namespace value_of_b_l240_240657

theorem value_of_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 3) : b = 3 := 
by
  sorry

end value_of_b_l240_240657


namespace foldable_polygons_count_l240_240621

def isValidFolding (base_positions : Finset Nat) (additional_position : Nat) : Prop :=
  ∃ (valid_positions : Finset Nat), valid_positions = {4, 5, 6, 7, 8, 9} ∧ additional_position ∈ valid_positions

theorem foldable_polygons_count : 
  ∃ (valid_additional_positions : Finset Nat), valid_additional_positions = {4, 5, 6, 7, 8, 9} ∧ valid_additional_positions.card = 6 := 
by
  sorry

end foldable_polygons_count_l240_240621


namespace cakes_left_l240_240042

def cakes_yesterday : ℕ := 3
def baked_today : ℕ := 5
def sold_today : ℕ := 6

theorem cakes_left (cakes_yesterday baked_today sold_today : ℕ) : cakes_yesterday + baked_today - sold_today = 2 := by
  sorry

end cakes_left_l240_240042


namespace Jessica_victory_l240_240189

def bullseye_points : ℕ := 10
def other_possible_scores : Set ℕ := {0, 2, 5, 8, 10}
def minimum_score_per_shot : ℕ := 2
def shots_taken : ℕ := 40
def remaining_shots : ℕ := 40
def jessica_advantage : ℕ := 30

def victory_condition (n : ℕ) : Prop :=
  8 * n + 80 > 370

theorem Jessica_victory :
  ∃ n, victory_condition n ∧ n = 37 :=
by
  use 37
  sorry

end Jessica_victory_l240_240189


namespace sequence_8123_appears_l240_240785

theorem sequence_8123_appears :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) + a (n-4)) % 10) ∧
  (a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ a 4 = 4) ∧
  (∃ n, a n = 8 ∧ a (n+1) = 1 ∧ a (n+2) = 2 ∧ a (n+3) = 3) :=
sorry

end sequence_8123_appears_l240_240785


namespace hot_air_balloon_height_l240_240276

theorem hot_air_balloon_height (altitude_temp_decrease_per_1000m : ℝ) 
  (ground_temp : ℝ) (high_altitude_temp : ℝ) :
  altitude_temp_decrease_per_1000m = 6 →
  ground_temp = 8 →
  high_altitude_temp = -1 →
  ∃ (height : ℝ), height = 1500 :=
by
  intro h1 h2 h3
  have temp_change := ground_temp - high_altitude_temp
  have height := (temp_change / altitude_temp_decrease_per_1000m) * 1000
  exact Exists.intro height sorry -- height needs to be computed here

end hot_air_balloon_height_l240_240276


namespace abs_diff_ge_abs_sum_iff_non_positive_prod_l240_240821

theorem abs_diff_ge_abs_sum_iff_non_positive_prod (a b : ℝ) : 
  |a - b| ≥ |a| + |b| ↔ a * b ≤ 0 := 
by sorry

end abs_diff_ge_abs_sum_iff_non_positive_prod_l240_240821


namespace sum_of_roots_expression_involving_roots_l240_240832

variables {a b : ℝ}

axiom roots_of_quadratic :
  (a^2 + 3 * a - 2 = 0) ∧ (b^2 + 3 * b - 2 = 0)

theorem sum_of_roots :
  a + b = -3 :=
by 
  sorry

theorem expression_involving_roots :
  a^3 + 3 * a^2 + 2 * b = -6 :=
by 
  sorry

end sum_of_roots_expression_involving_roots_l240_240832


namespace ratio_of_hours_l240_240719

theorem ratio_of_hours (x y z : ℕ) 
  (h1 : x + y + z = 157) 
  (h2 : z = y - 8) 
  (h3 : z = 56) 
  (h4 : y = x + 10) : 
  (y / gcd y x) = 32 ∧ (x / gcd y x) = 27 := 
by 
  sorry

end ratio_of_hours_l240_240719


namespace total_cost_proof_l240_240086

-- Define the cost of items
def cost_of_1kg_of_mango (M : ℚ) : Prop := sorry
def cost_of_1kg_of_rice (R : ℚ) : Prop := sorry
def cost_of_1kg_of_flour (F : ℚ) : Prop := F = 23

-- Condition 1: cost of some kg of mangos is equal to the cost of 24 kg of rice
def condition1 (M R : ℚ) (x : ℚ) : Prop := M * x = R * 24

-- Condition 2: cost of 6 kg of flour equals to the cost of 2 kg of rice
def condition2 (R : ℚ) : Prop := 23 * 6 = R * 2

-- Final proof problem
theorem total_cost_proof (M R F : ℚ) (x : ℚ) 
  (h1: condition1 M R x) 
  (h2: condition2 R) 
  (h3: cost_of_1kg_of_flour F) :
  4 * (69 * 24 / x) + 3 * R + 5 * 23 = 1978 :=
sorry

end total_cost_proof_l240_240086


namespace xiaozhi_needs_median_for_top_10_qualification_l240_240422

-- Define a set of scores as a list of integers
def scores : List ℕ := sorry

-- Assume these scores are unique (this is a condition given in the problem)
axiom unique_scores : ∀ (a b : ℕ), a ∈ scores → b ∈ scores → a ≠ b → scores.indexOf a ≠ scores.indexOf b

-- Define the median function (in practice, you would implement this, but we're just outlining it here)
def median (scores: List ℕ) : ℕ := sorry

-- Define the position of Xiao Zhi's score
def xiaozhi_score : ℕ := sorry

-- Given that the top 10 scores are needed to advance
def top_10 (scores: List ℕ) : List ℕ := scores.take 10

-- Proposition that Xiao Zhi needs median to determine his rank in top 10
theorem xiaozhi_needs_median_for_top_10_qualification 
    (scores_median : ℕ) (zs_score : ℕ) : 
    (∀ (s: List ℕ), s = scores → scores_median = median s → zs_score ≤ scores_median → zs_score ∉ top_10 s) ∧ 
    (exists (s: List ℕ), s = scores → zs_score ∉ top_10 s → zs_score ≤ scores_median) := 
sorry

end xiaozhi_needs_median_for_top_10_qualification_l240_240422


namespace landmark_postcards_probability_l240_240237

theorem landmark_postcards_probability :
  let total_postcards := 12
  let landmark_postcards := 4
  let total_arrangements := Nat.factorial total_postcards
  let favorable_arrangements := Nat.factorial (total_postcards - landmark_postcards + 1) * Nat.factorial landmark_postcards
  favorable_arrangements / total_arrangements = (1:ℝ) / 55 :=
by
  sorry

end landmark_postcards_probability_l240_240237


namespace tan_of_angle_in_third_quadrant_l240_240230

theorem tan_of_angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α = -12 / 13) (h2 : π < α ∧ α < 3 * π / 2) : Real.tan α = 12 / 5 := 
sorry

end tan_of_angle_in_third_quadrant_l240_240230


namespace domain_ln_x_squared_minus_2_l240_240169

theorem domain_ln_x_squared_minus_2 (x : ℝ) : 
  x^2 - 2 > 0 ↔ (x < -Real.sqrt 2 ∨ x > Real.sqrt 2) := 
by 
  sorry

end domain_ln_x_squared_minus_2_l240_240169


namespace problem1_problem2_l240_240801

theorem problem1 : 2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  -- Proof omitted
  sorry

theorem problem2 : (Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2 :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l240_240801


namespace trajectory_equation_circle_equation_l240_240295

-- Define the variables
variables {x y r : ℝ}

-- Prove the trajectory equation of the circle center P
theorem trajectory_equation (h1 : x^2 + r^2 = 2) (h2 : y^2 + r^2 = 3) : y^2 - x^2 = 1 :=
sorry

-- Prove the equation of the circle P given the distance to the line y = x
theorem circle_equation (h : (|x - y| / Real.sqrt 2) = (Real.sqrt 2) / 2) : 
  (x = y + 1 ∨ x = y - 1) → 
  ((y + 1)^2 + x^2 = 3 ∨ (y - 1)^2 + x^2 = 3) :=
sorry

end trajectory_equation_circle_equation_l240_240295


namespace coin_problem_l240_240667

theorem coin_problem (n d q : ℕ) 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 25 * q = 410)
  (h3 : d = n + 4) : q - n = 2 :=
by
  sorry

end coin_problem_l240_240667


namespace divisible_by_12_l240_240371

theorem divisible_by_12 (a b c d : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (hpos_d : 0 < d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := 
by
  sorry

end divisible_by_12_l240_240371


namespace find_f2_l240_240419

def f (x : ℝ) : ℝ := sorry

theorem find_f2 : (∀ x, f (x-1) = x / (x-1)) → f 2 = 3 / 2 :=
by
  sorry

end find_f2_l240_240419


namespace central_angle_of_sector_l240_240702

theorem central_angle_of_sector (alpha : ℝ) (l : ℝ) (A : ℝ) (h1 : l = 2 * Real.pi) (h2 : A = 5 * Real.pi) : 
  alpha = 72 :=
by
  sorry

end central_angle_of_sector_l240_240702


namespace isosceles_triangle_perimeter_l240_240092

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 2 ∧ b = 5 ∨ a = 5 ∧ b = 2):
  ∃ c : ℕ, (c = a ∨ c = b) ∧ 2 * c + (if c = a then b else a) = 12 :=
by
  sorry

end isosceles_triangle_perimeter_l240_240092


namespace eggs_per_chicken_l240_240384

theorem eggs_per_chicken (num_chickens : ℕ) (eggs_per_carton : ℕ) (num_cartons : ℕ) (total_eggs : ℕ) 
  (h1 : num_chickens = 20) (h2 : eggs_per_carton = 12) (h3 : num_cartons = 10) (h4 : total_eggs = num_cartons * eggs_per_carton) : 
  total_eggs / num_chickens = 6 :=
by
  sorry

end eggs_per_chicken_l240_240384


namespace triplet_zero_solution_l240_240325

theorem triplet_zero_solution (x y z : ℝ) 
  (h1 : x^3 + y = z^2) 
  (h2 : y^3 + z = x^2) 
  (h3 : z^3 + x = y^2) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end triplet_zero_solution_l240_240325


namespace mean_of_xyz_l240_240050

theorem mean_of_xyz (x y z : ℚ) (eleven_mean : ℚ)
  (eleven_sum : eleven_mean = 32)
  (fourteen_sum : 14 * 45 = 630)
  (new_mean : 14 * 45 = 630) :
  (x + y + z) / 3 = 278 / 3 :=
by
  sorry

end mean_of_xyz_l240_240050


namespace carl_city_mileage_l240_240068

noncomputable def city_mileage (miles_city mpg_highway cost_per_gallon total_cost miles_highway : ℝ) : ℝ :=
  let total_gallons := total_cost / cost_per_gallon
  let gallons_highway := miles_highway / mpg_highway
  let gallons_city := total_gallons - gallons_highway
  miles_city / gallons_city

theorem carl_city_mileage :
  city_mileage 60 40 3 42 200 = 20 / 3 := by
  sorry

end carl_city_mileage_l240_240068


namespace functional_eq_solution_l240_240180

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (x) ^ 2 + f (y)) = x * f (x) + y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := sorry

end functional_eq_solution_l240_240180


namespace geometric_series_inequality_l240_240115

variables {x y : ℝ}

theorem geometric_series_inequality 
  (hx : |x| < 1) 
  (hy : |y| < 1) :
  (1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y)) :=
sorry

end geometric_series_inequality_l240_240115


namespace abs_neg_two_l240_240906

theorem abs_neg_two : abs (-2) = 2 := 
by 
  sorry

end abs_neg_two_l240_240906


namespace inverse_passes_through_3_4_l240_240784

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given that f(x) has an inverse
def has_inverse := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Given that y = f(x+1) passes through the point (3,3)
def condition := f (3 + 1) = 3

theorem inverse_passes_through_3_4 
  (h1 : has_inverse f) 
  (h2 : condition f) : 
  f⁻¹ 3 = 4 :=
sorry

end inverse_passes_through_3_4_l240_240784


namespace binomial_expansion_sum_l240_240789

theorem binomial_expansion_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 :=
sorry

end binomial_expansion_sum_l240_240789


namespace value_of_y_plus_10_l240_240129

theorem value_of_y_plus_10 (x y : ℝ) (h1 : 3 * x = (3 / 4) * y) (h2 : x = 20) : y + 10 = 90 :=
by
  sorry

end value_of_y_plus_10_l240_240129


namespace solution_l240_240698

theorem solution (x : ℝ) (h : 6 ∈ ({2, 4, x * x - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end solution_l240_240698


namespace count_triples_not_div_by_4_l240_240028

theorem count_triples_not_div_by_4 :
  {n : ℕ // n = 117 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 5 → 1 ≤ b ∧ b ≤ 5 → 1 ≤ c ∧ c ≤ 5 → (a + b) * (a + c) * (b + c) % 4 ≠ 0} :=
sorry

end count_triples_not_div_by_4_l240_240028


namespace meeting_lamppost_l240_240333

-- Define the initial conditions of the problem
def lampposts : ℕ := 400
def start_alla : ℕ := 1
def start_boris : ℕ := 400
def meet_alla : ℕ := 55
def meet_boris : ℕ := 321

-- Define a theorem that we need to prove: Alla and Boris will meet at the 163rd lamppost
theorem meeting_lamppost : ∃ (n : ℕ), n = 163 := 
by {
  sorry -- Proof goes here
}

end meeting_lamppost_l240_240333


namespace friends_division_ways_l240_240315

theorem friends_division_ways : (4 ^ 8 = 65536) :=
by
  sorry

end friends_division_ways_l240_240315


namespace meaningful_expression_range_l240_240788

theorem meaningful_expression_range (x : ℝ) : (∃ y, y = 1 / (x - 4)) ↔ x ≠ 4 := 
by
  sorry

end meaningful_expression_range_l240_240788


namespace suitable_for_comprehensive_survey_l240_240595

-- Define the four survey options as a custom data type
inductive SurveyOption
  | A : SurveyOption -- Survey on the water quality of the Beijiang River
  | B : SurveyOption -- Survey on the quality of rice dumplings in the market during the Dragon Boat Festival
  | C : SurveyOption -- Survey on the vision of 50 students in a class
  | D : SurveyOption -- Survey by energy-saving lamp manufacturers on the service life of a batch of energy-saving lamps

-- Define feasibility for a comprehensive survey
def isComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => False
  | SurveyOption.B => False
  | SurveyOption.C => True
  | SurveyOption.D => False

-- The statement to be proven
theorem suitable_for_comprehensive_survey : ∃! o : SurveyOption, isComprehensiveSurvey o := by
  sorry

end suitable_for_comprehensive_survey_l240_240595


namespace Tom_age_ratio_l240_240471

variable (T N : ℕ)
variable (a : ℕ)
variable (c3 c4 : ℕ)

-- conditions
def condition1 : Prop := T = 4 * a + 5
def condition2 : Prop := T - N = 3 * (4 * a + 5 - 4 * N)

theorem Tom_age_ratio (h1 : condition1 T a) (h2 : condition2 T N a) : (T = 6 * N) :=
by sorry

end Tom_age_ratio_l240_240471


namespace percentage_change_l240_240918

def original_income (P T : ℝ) : ℝ :=
  P * T

def new_income (P T : ℝ) : ℝ :=
  (P * 1.3333) * (T * 0.6667)

theorem percentage_change (P T : ℝ) (hP : P ≠ 0) (hT : T ≠ 0) :
  ((new_income P T - original_income P T) / original_income P T) * 100 = -11.11 :=
by
  sorry

end percentage_change_l240_240918


namespace range_of_r_l240_240367

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem range_of_r (r : ℝ) (hr: 0 < r) : (M ∩ N r = N r) → r ≤ 2 - Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_r_l240_240367


namespace sarees_shirts_cost_l240_240680

variable (S T : ℕ)

-- Definition of conditions
def condition1 : Prop := 2 * S + 4 * T = 2 * S + 4 * T
def condition2 : Prop := (S + 6 * T) = (2 * S + 4 * T)
def condition3 : Prop := 12 * T = 2400

-- Proof goal
theorem sarees_shirts_cost :
  condition1 S T → condition2 S T → condition3 T → 2 * S + 4 * T = 1600 := by
  sorry

end sarees_shirts_cost_l240_240680


namespace factorize_polynomial_l240_240126

theorem factorize_polynomial (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
by 
  sorry

end factorize_polynomial_l240_240126


namespace number_of_students_l240_240617

variables (T S n : ℕ)

-- 1. The teacher's age is 24 years more than the average age of the students.
def condition1 : Prop := T = S / n + 24

-- 2. The teacher's age is 20 years more than the average age of everyone present.
def condition2 : Prop := T = (T + S) / (n + 1) + 20

-- Proving that the number of students in the classroom is 5 given the conditions.
theorem number_of_students (h1 : condition1 T S n) (h2 : condition2 T S n) : n = 5 :=
by sorry

end number_of_students_l240_240617


namespace ratio_shorter_to_longer_l240_240269

theorem ratio_shorter_to_longer (total_length shorter_length longer_length : ℕ) (h1 : total_length = 40) 
(h2 : shorter_length = 16) (h3 : longer_length = total_length - shorter_length) : 
(shorter_length / Nat.gcd shorter_length longer_length) / (longer_length / Nat.gcd shorter_length longer_length) = 2 / 3 :=
by
  sorry

end ratio_shorter_to_longer_l240_240269


namespace complement_intersection_l240_240572

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 3}

-- Define the complement of A with respect to U
def complement_U (s : Set ℕ) : Set ℕ := {x ∈ U | x ∉ s}

-- Define the statement to be proven
theorem complement_intersection :
  (complement_U A ∩ B) = {2} :=
by
  sorry

end complement_intersection_l240_240572


namespace problem_statement_l240_240986

theorem problem_statement (x m : ℝ) :
  (¬ (x > m) → ¬ (x^2 + x - 2 > 0)) ∧ (¬ (x > m) ↔ ¬ (x^2 + x - 2 > 0)) → m ≥ 1 :=
sorry

end problem_statement_l240_240986


namespace find_fraction_l240_240404

theorem find_fraction (c d : ℕ) (h1 : 435 = 2 * 100 + c * 10 + d) :
  (c + d) / 12 = 5 / 6 :=
by sorry

end find_fraction_l240_240404


namespace solution_exists_l240_240313

noncomputable def find_A_and_B : Prop :=
  ∃ A B : ℚ, 
    (A, B) = (75 / 16, 21 / 16) ∧ 
    ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 → 
    (6 * x + 3) / ((x - 12) * (x + 4)) = A / (x - 12) + B / (x + 4)

theorem solution_exists : find_A_and_B :=
sorry

end solution_exists_l240_240313


namespace current_number_of_people_l240_240984

theorem current_number_of_people (a b : ℕ) : 0 ≤ a → 0 ≤ b → 48 - a + b ≥ 0 := by
  sorry

end current_number_of_people_l240_240984


namespace arrange_snow_leopards_l240_240308

theorem arrange_snow_leopards :
  let n := 9 -- number of leopards
  let factorial x := (Nat.factorial x) -- definition for factorial
  let tall_short_perm := 2 -- there are 2 ways to arrange the tallest and shortest leopards at the ends
  tall_short_perm * factorial (n - 2) = 10080 := by sorry

end arrange_snow_leopards_l240_240308


namespace general_formula_arithmetic_sequence_l240_240223

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

theorem general_formula_arithmetic_sequence (x : ℝ) (a : ℕ → ℝ) 
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n : ℕ, (a n = 2 * n - 4) ∨ (a n = 4 - 2 * n) :=
by
  sorry

end general_formula_arithmetic_sequence_l240_240223


namespace maximum_value_existence_l240_240263

open Real

theorem maximum_value_existence (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
    8 * a + 3 * b + 5 * c ≤ sqrt (373 / 36) := by
  sorry

end maximum_value_existence_l240_240263


namespace square_area_is_correct_l240_240915

noncomputable def find_area_of_square (x : ℚ) : ℚ :=
  let side := 6 * x - 27
  side * side

theorem square_area_is_correct (x : ℚ) (h1 : 6 * x - 27 = 30 - 2 * x) :
  find_area_of_square x = 248.0625 :=
by
  sorry

end square_area_is_correct_l240_240915


namespace projection_of_b_onto_a_l240_240344

open Real

noncomputable def e1 : ℝ × ℝ := (1, 0)
noncomputable def e2 : ℝ × ℝ := (0, 1)

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := (dot_product u v) / (magnitude u)

theorem projection_of_b_onto_a : projection b a = 2 * sqrt 5 / 5 := by
  sorry

end projection_of_b_onto_a_l240_240344


namespace area_of_third_region_l240_240020

theorem area_of_third_region (A B C : ℝ) 
    (hA : A = 24) 
    (hB : B = 13) 
    (hTotal : A + B + C = 48) : 
    C = 11 := 
by 
  sorry

end area_of_third_region_l240_240020


namespace multiple_of_other_number_l240_240002

theorem multiple_of_other_number 
(m S L : ℕ) 
(hl : L = 33) 
(hrel : L = m * S - 3) 
(hsum : L + S = 51) : 
m = 2 :=
by
  sorry

end multiple_of_other_number_l240_240002


namespace find_c_l240_240112

def conditions (c d : ℝ) : Prop :=
  -- The polynomial 6x^3 + 7cx^2 + 3dx + 2c = 0 has three distinct positive roots
  ∃ u v w : ℝ, 0 < u ∧ 0 < v ∧ 0 < w ∧ u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
  (6 * u^3 + 7 * c * u^2 + 3 * d * u + 2 * c = 0) ∧
  (6 * v^3 + 7 * c * v^2 + 3 * d * v + 2 * c = 0) ∧
  (6 * w^3 + 7 * c * w^2 + 3 * d * w + 2 * c = 0) ∧
  -- Sum of the base-2 logarithms of the roots is 6
  Real.log (u * v * w) / Real.log 2 = 6

theorem find_c (c d : ℝ) (h : conditions c d) : c = -192 :=
sorry

end find_c_l240_240112


namespace invalid_speed_against_stream_l240_240326

theorem invalid_speed_against_stream (rate_still_water speed_with_stream : ℝ) (h1 : rate_still_water = 6) (h2 : speed_with_stream = 20) :
  ∃ (v : ℝ), speed_with_stream = rate_still_water + v ∧ (rate_still_water - v < 0) → false :=
by
  sorry

end invalid_speed_against_stream_l240_240326


namespace trapezoid_circle_center_l240_240218

theorem trapezoid_circle_center 
  (EF GH : ℝ)
  (FG HE : ℝ)
  (p q : ℕ) 
  (rel_prime : Nat.gcd p q = 1)
  (EQ GH : ℝ)
  (h1 : EF = 105)
  (h2 : FG = 57)
  (h3 : GH = 22)
  (h4 : HE = 80)
  (h5 : EQ = p / q)
  (h6 : p = 10)
  (h7 : q = 1) :
  p + q = 11 :=
by
  sorry

end trapezoid_circle_center_l240_240218


namespace max_sqrt_distance_l240_240374

theorem max_sqrt_distance (x y : ℝ) 
  (h : x^2 + y^2 - 4 * x - 4 * y + 6 = 0) : 
  ∃ z, z = 3 * Real.sqrt 2 ∧ ∀ w, w = Real.sqrt (x^2 + y^2) → w ≤ z :=
sorry

end max_sqrt_distance_l240_240374


namespace eggs_in_basket_empty_l240_240590

theorem eggs_in_basket_empty (a : ℕ) : 
  let remaining_after_first := a - (a / 2 + 1 / 2)
  let remaining_after_second := remaining_after_first - (remaining_after_first / 2 + 1 / 2)
  let remaining_after_third := remaining_after_second - (remaining_after_second / 2 + 1 / 2)
  (remaining_after_first = a / 2 - 1 / 2) → 
  (remaining_after_second = remaining_after_first / 2 - 1 / 2) → 
  (remaining_after_third = remaining_after_second / 2 -1 / 2) → 
  (remaining_after_third = 0) → 
  (a = 7) := sorry

end eggs_in_basket_empty_l240_240590


namespace min_value_arith_seq_l240_240760

noncomputable def S_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem min_value_arith_seq : ∀ n : ℕ, n > 0 → 2 * S_n 2 = (n + 1) * 2 → (n = 4 → (2 * S_n n + 13) / n = 33 / 4) :=
by
  intros n hn hS2 hn_eq_4
  sorry

end min_value_arith_seq_l240_240760


namespace trigonometric_identity_l240_240130

theorem trigonometric_identity (α : ℝ) (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) : 
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3/40 :=
by
  sorry

end trigonometric_identity_l240_240130


namespace combined_resistance_parallel_l240_240196

theorem combined_resistance_parallel (x y : ℝ) (r : ℝ) (hx : x = 3) (hy : y = 5) 
  (h : 1 / r = 1 / x + 1 / y) : r = 15 / 8 :=
by
  sorry

end combined_resistance_parallel_l240_240196


namespace find_r_value_l240_240382

theorem find_r_value (m : ℕ) (h_m : m = 3) (t : ℕ) (h_t : t = 3^m + 2) (r : ℕ) (h_r : r = 4^t - 2 * t) : r = 4^29 - 58 := by
  sorry

end find_r_value_l240_240382


namespace machine_A_produces_7_sprockets_per_hour_l240_240406

theorem machine_A_produces_7_sprockets_per_hour
    (A B : ℝ)
    (h1 : B = 1.10 * A)
    (h2 : ∃ t : ℝ, 770 = A * (t + 10) ∧ 770 = B * t) : 
    A = 7 := 
by 
    sorry

end machine_A_produces_7_sprockets_per_hour_l240_240406


namespace divisor_of_44404_l240_240488

theorem divisor_of_44404: ∃ k : ℕ, 2 * 11101 = k ∧ k ∣ (44402 + 2) :=
by
  sorry

end divisor_of_44404_l240_240488


namespace find_quotient_l240_240184

-- Constants representing the given conditions
def dividend : ℕ := 690
def divisor : ℕ := 36
def remainder : ℕ := 6

-- Theorem statement
theorem find_quotient : ∃ (quotient : ℕ), dividend = (divisor * quotient) + remainder ∧ quotient = 19 := 
by
  sorry

end find_quotient_l240_240184


namespace eval_operation_l240_240204

-- Definition of the * operation based on the given table
def op (a b : ℕ) : ℕ :=
  match a, b with
  | 1, 1 => 4
  | 1, 2 => 1
  | 1, 3 => 2
  | 1, 4 => 3
  | 2, 1 => 1
  | 2, 2 => 3
  | 2, 3 => 4
  | 2, 4 => 2
  | 3, 1 => 2
  | 3, 2 => 4
  | 3, 3 => 1
  | 3, 4 => 3
  | 4, 1 => 3
  | 4, 2 => 2
  | 4, 3 => 3
  | 4, 4 => 4
  | _, _ => 0 -- Default case (not needed as per the given problem definition)

-- Statement of the problem in Lean 4
theorem eval_operation : op (op 3 1) (op 4 2) = 3 :=
by {
  sorry -- Proof to be provided
}

end eval_operation_l240_240204


namespace range_of_a_l240_240521

variable (x a : ℝ)

def p : Prop := x^2 - 2 * x - 3 ≥ 0

def q : Prop := x^2 - (2 * a - 1) * x + a * (a - 1) ≥ 0

def sufficient_but_not_necessary (p q : Prop) : Prop := 
  (p → q) ∧ ¬(q → p)

theorem range_of_a (a : ℝ) : (∃ x, sufficient_but_not_necessary (p x) (q a x)) → (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_of_a_l240_240521


namespace complex_equation_solution_l240_240743

theorem complex_equation_solution (x y : ℝ)
  (h : (x / (1 - (-ⅈ)) + y / (1 - 2 * (-ⅈ)) = 5 / (1 - 3 * (-ⅈ)))) :
  x + y = 4 :=
sorry

end complex_equation_solution_l240_240743


namespace planet_not_observed_l240_240107

theorem planet_not_observed (k : ℕ) (d : Fin (2*k+1) → Fin (2*k+1) → ℝ) 
  (h_d : ∀ i j : Fin (2*k+1), i ≠ j → d i i = 0 ∧ d i j ≠ d i i) 
  (h_astronomer : ∀ i : Fin (2*k+1), ∃ j : Fin (2*k+1), j ≠ i ∧ ∀ k : Fin (2*k+1), k ≠ i → d i j < d i k) : 
  ∃ i : Fin (2*k+1), ∀ j : Fin (2*k+1), i ≠ j → ∃ l : Fin (2*k+1), (j ≠ l ∧ d l i < d l j) → false :=
  sorry

end planet_not_observed_l240_240107


namespace pipe_B_fill_time_l240_240901

-- Definitions based on the given conditions
def fill_time_by_ABC := 10  -- in hours
def B_is_twice_as_fast_as_C : Prop := ∀ C B, B = 2 * C
def A_is_twice_as_fast_as_B : Prop := ∀ A B, A = 2 * B

-- The main theorem to prove
theorem pipe_B_fill_time (A B C : ℝ) (h1: fill_time_by_ABC = 10) 
    (h2 : B_is_twice_as_fast_as_C) (h3 : A_is_twice_as_fast_as_B) : B = 1 / 35 :=
by
  sorry

end pipe_B_fill_time_l240_240901


namespace find_f_five_l240_240031

-- Define the function f and the conditions as given in the problem.
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (h₁ : ∀ x y : ℝ, f (x - y) = f x * g y)
variable (h₂ : ∀ y : ℝ, g y = Real.exp (-y))
variable (h₃ : ∀ x : ℝ, f x ≠ 0)

-- Goal: Prove that f(5) = e^{2.5}.
theorem find_f_five : f 5 = Real.exp 2.5 :=
by
  -- Proof is omitted as per the instructions.
  sorry

end find_f_five_l240_240031


namespace horizontal_asymptote_is_3_l240_240139

-- Definitions of the polynomials
noncomputable def p (x : ℝ) : ℝ := 15 * x^5 + 10 * x^4 + 5 * x^3 + 7 * x^2 + 6 * x + 2
noncomputable def q (x : ℝ) : ℝ := 5 * x^5 + 3 * x^4 + 9 * x^3 + 4 * x^2 + 2 * x + 1

-- Statement that we need to prove
theorem horizontal_asymptote_is_3 : 
  (∃ (y : ℝ), (∀ x : ℝ, x ≠ 0 → (p x / q x) = y) ∧ y = 3) :=
  sorry -- The proof is left as an exercise.

end horizontal_asymptote_is_3_l240_240139


namespace b_should_pay_l240_240119

def TotalRent : ℕ := 725
def Cost_a : ℕ := 12 * 8 * 5
def Cost_b : ℕ := 16 * 9 * 6
def Cost_c : ℕ := 18 * 6 * 7
def Cost_d : ℕ := 20 * 4 * 4
def TotalCost : ℕ := Cost_a + Cost_b + Cost_c + Cost_d
def Payment_b (Cost_b TotalCost TotalRent : ℕ) : ℕ := (Cost_b * TotalRent) / TotalCost

theorem b_should_pay :
  Payment_b Cost_b TotalCost TotalRent = 259 := 
  by
  unfold Payment_b
  -- Leaving the proof body empty as per instructions
  sorry

end b_should_pay_l240_240119


namespace group_count_l240_240628

theorem group_count (sample_capacity : ℕ) (frequency : ℝ) (h_sample_capacity : sample_capacity = 80) (h_frequency : frequency = 0.125) : sample_capacity * frequency = 10 := 
by
  sorry

end group_count_l240_240628


namespace product_x_z_l240_240197

-- Defining the variables x, y, z as positive integers and the given conditions.
theorem product_x_z (x y z : ℕ) (h1 : x = 4 * y) (h2 : z = 2 * x) (h3 : x + y + z = 3 * y ^ 2) : 
    x * z = 5408 / 9 := 
  sorry

end product_x_z_l240_240197


namespace original_price_of_article_l240_240685

theorem original_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (hSP : SP = 374) (hprofit : profit_percent = 0.10) : 
  CP = 340 ↔ SP = CP * (1 + profit_percent) :=
by 
  sorry

end original_price_of_article_l240_240685


namespace cars_meet_time_l240_240979

theorem cars_meet_time (s1 s2 : ℝ) (d : ℝ) (c : s1 = (5 / 4) * s2) 
  (h1 : s1 = 100) (h2 : d = 720) : d / (s1 + s2) = 4 :=
by 
  sorry

end cars_meet_time_l240_240979


namespace no_such_triples_l240_240974

noncomputable def no_triple_satisfy (a b c : ℤ) : Prop :=
  ∀ (x1 x2 x3 : ℤ), 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    Int.gcd x1 x2 = 1 ∧ Int.gcd x2 x3 = 1 ∧ Int.gcd x1 x3 = 1 ∧
    (x1^3 - a^2 * x1^2 + b^2 * x1 - a * b + 3 * c = 0) ∧ 
    (x2^3 - a^2 * x2^2 + b^2 * x2 - a * b + 3 * c = 0) ∧ 
    (x3^3 - a^2 * x3^2 + b^2 * x3 - a * b + 3 * c = 0) →
    False

theorem no_such_triples : ∀ (a b c : ℤ), no_triple_satisfy a b c :=
by
  intros
  sorry

end no_such_triples_l240_240974


namespace symmetric_polynomial_identity_l240_240592

variable (x y z : ℝ)
def σ1 : ℝ := x + y + z
def σ2 : ℝ := x * y + y * z + z * x
def σ3 : ℝ := x * y * z

theorem symmetric_polynomial_identity : 
  x^3 + y^3 + z^3 = σ1 x y z ^ 3 - 3 * σ1 x y z * σ2 x y z + 3 * σ3 x y z := by
  sorry

end symmetric_polynomial_identity_l240_240592


namespace find_x2_plus_y2_l240_240903

theorem find_x2_plus_y2 (x y : ℝ) (h : (x ^ 2 + y ^ 2 + 1) * (x ^ 2 + y ^ 2 - 3) = 5) : x ^ 2 + y ^ 2 = 4 := 
by 
  sorry

end find_x2_plus_y2_l240_240903


namespace frank_initial_money_l240_240348

theorem frank_initial_money (X : ℝ) (h1 : X * (4 / 5) * (3 / 4) * (6 / 7) * (2 / 3) = 600) : X = 2333.33 :=
sorry

end frank_initial_money_l240_240348


namespace gcd_24_36_l240_240802

theorem gcd_24_36 : Int.gcd 24 36 = 12 := by
  sorry

end gcd_24_36_l240_240802


namespace mira_additional_stickers_l240_240317

-- Define the conditions
def mira_stickers : ℕ := 31
def row_size : ℕ := 7

-- Define the proof statement
theorem mira_additional_stickers (a : ℕ) (h : (31 + a) % 7 = 0) : 
  a = 4 := 
sorry

end mira_additional_stickers_l240_240317


namespace new_sequence_69th_term_l240_240347

-- Definitions and conditions
def original_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ := a n

def new_sequence (a : ℕ → ℕ) (k : ℕ) : ℕ :=
if k % 4 = 1 then a (k / 4 + 1) else 0  -- simplified modeling, the inserted numbers are denoted arbitrarily as 0

-- The statement to be proven
theorem new_sequence_69th_term (a : ℕ → ℕ) : new_sequence a 69 = a 18 :=
by
  sorry

end new_sequence_69th_term_l240_240347


namespace problem_l240_240615

open Real

def p (x : ℝ) : Prop := 2*x^2 + 2*x + 1/2 < 0

def q (x y : ℝ) : Prop := (x^2)/4 - (y^2)/12 = 1 ∧ x ≥ 2

def x0_condition (x0 : ℝ) : Prop := sin x0 - cos x0 = sqrt 2

theorem problem (h1 : ∀ x : ℝ, ¬ p x)
               (h2 : ∃ x y : ℝ, q x y)
               (h3 : ∃ x0 : ℝ, x0_condition x0) :
               ∀ x : ℝ, ¬ ¬ p x := 
sorry

end problem_l240_240615


namespace fraction_value_l240_240742

theorem fraction_value :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 :=
by
  sorry

end fraction_value_l240_240742


namespace area_of_moving_point_l240_240815

theorem area_of_moving_point (a b : ℝ) :
  (∀ (x y : ℝ), abs x ≤ 1 ∧ abs y ≤ 1 → a * x - 2 * b * y ≤ 2) →
  ∃ (A : ℝ), A = 8 := sorry

end area_of_moving_point_l240_240815


namespace trigonometric_identity_l240_240706

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
    Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -4 / 3 :=
sorry

end trigonometric_identity_l240_240706


namespace find_f_prime_at_2_l240_240890

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l240_240890


namespace common_ratio_half_l240_240207

-- Definitions based on conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n+1) = a n * q
def arith_seq (x y z : ℝ) := x + z = 2 * y

-- Theorem statement
theorem common_ratio_half (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q)
  (h_arith : arith_seq (a 5) (a 6 + a 8) (a 7)) : q = 1 / 2 := 
sorry

end common_ratio_half_l240_240207


namespace max_q_value_l240_240923

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l240_240923


namespace total_hours_eq_52_l240_240817

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l240_240817


namespace jill_study_hours_l240_240614

theorem jill_study_hours (x : ℕ) (h_condition : x + 2*x + (2*x - 1) = 9) : x = 2 :=
by
  sorry

end jill_study_hours_l240_240614


namespace number_of_girls_l240_240034

theorem number_of_girls
  (total_students : ℕ)
  (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_non_binary : ℕ)
  (h_ratio : ratio_girls = 3 ∧ ratio_boys = 2 ∧ ratio_non_binary = 1)
  (h_total : total_students = 72) :
  ∃ (k : ℕ), 3 * k = (total_students * 3) / 6 ∧ 3 * k = 36 :=
by
  sorry

end number_of_girls_l240_240034


namespace Cindy_hourly_rate_l240_240765

theorem Cindy_hourly_rate
    (num_courses : ℕ)
    (weekly_hours : ℕ) 
    (monthly_earnings : ℕ) 
    (weeks_in_month : ℕ)
    (monthly_hours_per_course : ℕ)
    (hourly_rate : ℕ) :
    num_courses = 4 →
    weekly_hours = 48 →
    monthly_earnings = 1200 →
    weeks_in_month = 4 →
    monthly_hours_per_course = (weekly_hours / num_courses) * weeks_in_month →
    hourly_rate = monthly_earnings / monthly_hours_per_course →
    hourly_rate = 25 := by
  sorry

end Cindy_hourly_rate_l240_240765


namespace fraction_savings_spent_on_furniture_l240_240964

theorem fraction_savings_spent_on_furniture (savings : ℝ) (tv_cost : ℝ) (F : ℝ) 
  (h1 : savings = 840) (h2 : tv_cost = 210) 
  (h3 : F * savings + tv_cost = savings) : F = 3 / 4 :=
sorry

end fraction_savings_spent_on_furniture_l240_240964


namespace julia_played_more_kids_on_monday_l240_240219

def n_monday : ℕ := 6
def n_tuesday : ℕ := 5

theorem julia_played_more_kids_on_monday : n_monday - n_tuesday = 1 := by
  -- Proof goes here
  sorry

end julia_played_more_kids_on_monday_l240_240219


namespace mod_pow_sum_7_l240_240467

theorem mod_pow_sum_7 :
  (45 ^ 1234 + 27 ^ 1234) % 7 = 5 := by
  sorry

end mod_pow_sum_7_l240_240467


namespace RS_segment_length_l240_240549

theorem RS_segment_length (P Q R S : ℝ) (r1 r2 : ℝ) (hP : P = 0) (hQ : Q = 10) (rP : r1 = 6) (rQ : r2 = 4) :
    (∃ PR QR SR : ℝ, PR = 6 ∧ QR = 4 ∧ SR = 6) → (R - S = 12) :=
by
  sorry

end RS_segment_length_l240_240549


namespace coin_difference_l240_240496

-- Define the coin denominations
def coin_denominations : List ℕ := [5, 10, 25, 50]

-- Define the target amount Paul needs to pay
def target_amount : ℕ := 60

-- Define the function to compute the minimum number of coins required
noncomputable def min_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the function to compute the maximum number of coins required
noncomputable def max_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the theorem to state the difference between max and min coins is 10
theorem coin_difference : max_coins target_amount coin_denominations - min_coins target_amount coin_denominations = 10 :=
  sorry

end coin_difference_l240_240496


namespace min_value_of_expression_l240_240135

theorem min_value_of_expression 
  (x y : ℝ) 
  (h : 3 * |x - y| + |2 * x - 5| = x + 1) : 
  ∃ (x y : ℝ), 2 * x + y = 4 :=
by {
  sorry
}

end min_value_of_expression_l240_240135


namespace CaitlinIs24_l240_240216

-- Definition using the given conditions
def AuntAnnaAge : ℕ := 45
def BriannaAge : ℕ := (2 * AuntAnnaAge) / 3
def CaitlinAge : ℕ := BriannaAge - 6

-- Statement to be proved
theorem CaitlinIs24 : CaitlinAge = 24 :=
by
  sorry

end CaitlinIs24_l240_240216


namespace problem1_problem2_l240_240209

-- Problem 1: Simplify the calculation: 6.9^2 + 6.2 * 6.9 + 3.1^2
theorem problem1 : 6.9^2 + 6.2 * 6.9 + 3.1^2 = 100 := 
by
  sorry

-- Problem 2: Simplify and find the value of the expression with given conditions
theorem problem2 (a b : ℝ) (h1 : a = 1) (h2 : b = 0.5) :
  (a^2 * b^3 + 2 * a^3 * b) / (2 * a * b) - (a + 2 * b) * (a - 2 * b) = 9 / 8 :=
by
  sorry

end problem1_problem2_l240_240209


namespace hyperbola_equation_l240_240206

theorem hyperbola_equation (a b c : ℝ) (e : ℝ) 
  (h1 : e = (Real.sqrt 6) / 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : (c / a) = e)
  (h5 : (b * c) / (Real.sqrt (b^2 + a^2)) = 1) :
  (∃ a b : ℝ, (a = Real.sqrt 2) ∧ (b = 1) ∧ (∀ x y : ℝ, (x^2 / 2) - y^2 = 1)) :=
by
  sorry

end hyperbola_equation_l240_240206


namespace largest_divisor_of_n4_minus_n2_is_12_l240_240149

theorem largest_divisor_of_n4_minus_n2_is_12 : ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  intro n
  -- Placeholder for proof; the detailed steps of the proof go here
  sorry

end largest_divisor_of_n4_minus_n2_is_12_l240_240149


namespace brenda_mice_left_l240_240566

theorem brenda_mice_left (litters : ℕ) (mice_per_litter : ℕ) (fraction_to_robbie : ℚ) 
                          (mult_to_pet_store : ℕ) (fraction_to_feeder : ℚ) 
                          (total_mice : ℕ) (to_robbie : ℕ) (to_pet_store : ℕ) 
                          (remaining_after_first_sales : ℕ) (to_feeder : ℕ) (left_after_feeder : ℕ) :
  litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1/6 →
  mult_to_pet_store = 3 →
  fraction_to_feeder = 1/2 →
  total_mice = litters * mice_per_litter →
  to_robbie = total_mice * fraction_to_robbie →
  to_pet_store = mult_to_pet_store * to_robbie →
  remaining_after_first_sales = total_mice - to_robbie - to_pet_store →
  to_feeder = remaining_after_first_sales * fraction_to_feeder →
  left_after_feeder = remaining_after_first_sales - to_feeder →
  left_after_feeder = 4 := sorry

end brenda_mice_left_l240_240566


namespace correct_average_l240_240884

theorem correct_average (avg_incorrect : ℕ) (old_num new_num : ℕ) (n : ℕ)
  (h_avg : avg_incorrect = 15)
  (h_old_num : old_num = 26)
  (h_new_num : new_num = 36)
  (h_n : n = 10) :
  (avg_incorrect * n + (new_num - old_num)) / n = 16 := by
  sorry

end correct_average_l240_240884


namespace sum_of_roots_l240_240010

theorem sum_of_roots (p : ℝ) (h : (4 - p) / 2 = 9) : (p / 2 = 7) :=
by 
  sorry

end sum_of_roots_l240_240010


namespace shooting_challenge_sequences_l240_240593

theorem shooting_challenge_sequences : ∀ (A B C : ℕ), 
  A = 4 → B = 4 → C = 2 →
  (A + B + C = 10) →
  (Nat.factorial (A + B + C) / (Nat.factorial A * Nat.factorial B * Nat.factorial C) = 3150) :=
by
  intros A B C hA hB hC hsum
  sorry

end shooting_challenge_sequences_l240_240593


namespace quadratic_factorization_l240_240105

theorem quadratic_factorization :
  ∃ a b : ℕ, (a > b) ∧ (x^2 - 20 * x + 96 = (x - a) * (x - b)) ∧ (4 * b - a = 20) := sorry

end quadratic_factorization_l240_240105


namespace fewest_erasers_l240_240870

theorem fewest_erasers :
  ∀ (JK JM SJ : ℕ), 
  (JK = 6) →
  (JM = JK + 4) →
  (SJ = JM - 3) →
  (JK ≤ JM ∧ JK ≤ SJ) :=
by
  intros JK JM SJ hJK hJM hSJ
  sorry

end fewest_erasers_l240_240870


namespace first_player_wins_l240_240670

-- Define the initial conditions
def initial_pile_1 : ℕ := 100
def initial_pile_2 : ℕ := 200

-- Define the game rules
def valid_move (pile_1 pile_2 n : ℕ) : Prop :=
  (n > 0) ∧ ((n <= pile_1) ∨ (n <= pile_2))

-- The game state is represented as a pair of natural numbers
def GameState := ℕ × ℕ

-- Define what it means to win the game
def winning_move (s: GameState) : Prop :=
  (s.1 = 0 ∧ s.2 = 1) ∨ (s.1 = 1 ∧ s.2 = 0)

-- Define the main theorem
theorem first_player_wins : 
  ∀ s : GameState, (s = (initial_pile_1, initial_pile_2)) → (∃ move, valid_move s.1 s.2 move ∧ winning_move (s.1 - move, s.2 - move)) :=
sorry

end first_player_wins_l240_240670


namespace find_b_l240_240071

theorem find_b (b : ℚ) (h : b * (-3) - (b - 1) * 5 = b - 3) : b = 8 / 9 :=
by
  sorry

end find_b_l240_240071


namespace little_john_spent_on_sweets_l240_240826

theorem little_john_spent_on_sweets
  (initial_amount : ℝ)
  (amount_per_friend : ℝ)
  (friends_count : ℕ)
  (amount_left : ℝ)
  (spent_on_sweets : ℝ) :
  initial_amount = 10.50 →
  amount_per_friend = 2.20 →
  friends_count = 2 →
  amount_left = 3.85 →
  spent_on_sweets = initial_amount - (amount_per_friend * friends_count) - amount_left →
  spent_on_sweets = 2.25 :=
by
  intros h_initial h_per_friend h_friends_count h_left h_spent
  sorry

end little_john_spent_on_sweets_l240_240826


namespace eliza_is_18_l240_240591

-- Define the relevant ages
def aunt_ellen_age : ℕ := 48
def dina_age : ℕ := aunt_ellen_age / 2
def eliza_age : ℕ := dina_age - 6

-- Theorem to prove Eliza's age is 18
theorem eliza_is_18 : eliza_age = 18 := by
  sorry

end eliza_is_18_l240_240591


namespace find_common_ratio_l240_240966

theorem find_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h₀ : ∀ n, a (n + 1) = q * a n)
  (h₁ : a 0 = 4)
  (h₂ : q ≠ 1)
  (h₃ : 2 * a 4 = 4 * a 0 - 2 * a 2) :
  q = -1 := 
sorry

end find_common_ratio_l240_240966


namespace simplify_expression_l240_240512

theorem simplify_expression (x : ℝ) (h1 : x^3 + 2*x + 1 ≠ 0) (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ( ((x + 2)^2 * (x^2 - x + 2)^2 / (x^3 + 2*x + 1)^2 )^3 * ((x - 2)^2 * (x^2 + x + 2)^2 / (x^3 - 2*x - 1)^2 )^3 ) = 1 :=
by sorry

end simplify_expression_l240_240512


namespace sum_of_three_numbers_is_seventy_l240_240722

theorem sum_of_three_numbers_is_seventy
  (a b c : ℝ)
  (h1 : a ≤ b ∧ b ≤ c)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 30)
  (h4 : b = 10)
  (h5 : a + c = 60) :
  a + b + c = 70 :=
  sorry

end sum_of_three_numbers_is_seventy_l240_240722


namespace power_of_x_is_one_l240_240750

-- The problem setup, defining the existence of distinct primes and conditions on exponents
theorem power_of_x_is_one (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (a b c : ℕ) (h_divisors : (a + 1) * (b + 1) * (c + 1) = 12) :
  a = 1 :=
sorry

end power_of_x_is_one_l240_240750


namespace chord_midpoint_line_eqn_l240_240737

-- Definitions of points and the ellipse condition
def P : ℝ × ℝ := (3, 2)

def is_midpoint (P E F : ℝ × ℝ) := 
  P.1 = (E.1 + F.1) / 2 ∧ P.2 = (E.2 + F.2) / 2

def ellipse (x y : ℝ) := 
  4 * x^2 + 9 * y^2 = 144

theorem chord_midpoint_line_eqn
  (E F : ℝ × ℝ) 
  (h1 : is_midpoint P E F)
  (h2 : ellipse E.1 E.2)
  (h3 : ellipse F.1 F.2):
  ∃ (m b : ℝ), (P.2 = m * P.1 + b) ∧ (2 * P.1 + 3 * P.2 - 12 = 0) :=
by 
  sorry

end chord_midpoint_line_eqn_l240_240737


namespace purely_periodic_denominator_l240_240250

theorem purely_periodic_denominator :
  ∀ q : ℕ, (∃ a : ℕ, (∃ b : ℕ, q = 99 ∧ (a < 10) ∧ (b < 10) ∧ (∃ f : ℝ, f = ↑a / (10 * q) ∧ ∃ g : ℝ, g = (0.01 * ↑b / (10 * (99 / q))))) → q = 11 ∨ q = 33 ∨ q = 99) :=
by sorry

end purely_periodic_denominator_l240_240250


namespace number_of_boxes_l240_240989

-- Define the given conditions
def total_chocolates : ℕ := 442
def chocolates_per_box : ℕ := 26

-- Prove the number of small boxes in the large box
theorem number_of_boxes : (total_chocolates / chocolates_per_box) = 17 := by
  sorry

end number_of_boxes_l240_240989


namespace problem_l240_240093

theorem problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_l240_240093


namespace correct_value_of_wrongly_read_number_l240_240288

theorem correct_value_of_wrongly_read_number 
  (avg_wrong : ℝ) (n : ℕ) (wrong_value : ℝ) (avg_correct : ℝ) :
  avg_wrong = 5 →
  n = 10 →
  wrong_value = 26 →
  avg_correct = 6 →
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  correct_value = 36 :=
by
  intros h_avg_wrong h_n h_wrong_value h_avg_correct
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  sorry

end correct_value_of_wrongly_read_number_l240_240288


namespace find_x_squared_plus_y_squared_l240_240389

open Real

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 := by
  sorry

end find_x_squared_plus_y_squared_l240_240389


namespace smallest_positive_b_l240_240707

def periodic_10 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 10) = f x

theorem smallest_positive_b
  (f : ℝ → ℝ)
  (h : periodic_10 f) :
  ∀ x, f ((x - 20) / 2) = f (x / 2) :=
by
  sorry

end smallest_positive_b_l240_240707


namespace expression_equals_4034_l240_240434

theorem expression_equals_4034 : 6 * 2017 - 4 * 2017 = 4034 := by
  sorry

end expression_equals_4034_l240_240434


namespace initial_hotdogs_l240_240892

-- Definitions
variable (x : ℕ)

-- Conditions
def condition : Prop := x - 2 = 97 

-- Statement to prove
theorem initial_hotdogs (h : condition x) : x = 99 :=
  by
    sorry

end initial_hotdogs_l240_240892


namespace kayla_apples_correct_l240_240335

-- Definition of Kylie and Kayla's apples
def total_apples : ℕ := 340
def kaylas_apples (k : ℕ) : ℕ := 4 * k + 10

-- The main statement to prove
theorem kayla_apples_correct :
  ∃ K : ℕ, K + kaylas_apples K = total_apples ∧ kaylas_apples K = 274 :=
sorry

end kayla_apples_correct_l240_240335


namespace max_common_initial_segment_l240_240466

theorem max_common_initial_segment (m n : ℕ) (h_coprime : Nat.gcd m n = 1) : 
  ∃ L, L = m + n - 2 := 
sorry

end max_common_initial_segment_l240_240466


namespace inequality_logarithms_l240_240541

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem inequality_logarithms : a > c ∧ c > b :=
by
  -- the proof will be written here
  sorry

end inequality_logarithms_l240_240541


namespace pre_image_of_f_5_1_l240_240893

def f (x y : ℝ) : ℝ × ℝ := (x + y, 2 * x - y)

theorem pre_image_of_f_5_1 : ∃ (x y : ℝ), f x y = (5, 1) ∧ (x, y) = (2, 3) :=
by
  sorry

end pre_image_of_f_5_1_l240_240893


namespace value_of_gg_neg1_l240_240625

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_gg_neg1 : g (g (-1)) = 199 := by
  sorry

end value_of_gg_neg1_l240_240625


namespace weekly_caloric_allowance_l240_240022

-- Define the given conditions
def average_daily_allowance : ℕ := 2000
def daily_reduction_goal : ℕ := 500
def intense_workout_extra_calories : ℕ := 300
def moderate_exercise_extra_calories : ℕ := 200
def days_intense_workout : ℕ := 2
def days_moderate_exercise : ℕ := 3
def days_rest : ℕ := 2

-- Lean statement to prove the total weekly caloric intake
theorem weekly_caloric_allowance :
  (days_intense_workout * (average_daily_allowance - daily_reduction_goal + intense_workout_extra_calories)) +
  (days_moderate_exercise * (average_daily_allowance - daily_reduction_goal + moderate_exercise_extra_calories)) +
  (days_rest * (average_daily_allowance - daily_reduction_goal)) = 11700 := by
  sorry

end weekly_caloric_allowance_l240_240022


namespace john_weekly_earnings_l240_240798

theorem john_weekly_earnings :
  (4 * 4 * 10 = 160) :=
by
  -- Proposition: John makes $160 a week from streaming
  -- Condition 1: John streams for 4 days a week
  let days_of_streaming := 4
  -- Condition 2: He streams 4 hours each day.
  let hours_per_day := 4
  -- Condition 3: He makes $10 an hour.
  let earnings_per_hour := 10

  -- Now, calculate the weekly earnings
  -- Weekly earnings = 4 days/week * 4 hours/day * $10/hour
  have weekly_earnings : days_of_streaming * hours_per_day * earnings_per_hour = 160 := sorry
  exact weekly_earnings


end john_weekly_earnings_l240_240798


namespace determine_a_l240_240758

theorem determine_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x, y) = (-1, 2) → 3 * x + y + a = 0) → ∃ (a : ℝ), a = 1 :=
by
  sorry

end determine_a_l240_240758


namespace cost_price_of_computer_table_l240_240454

theorem cost_price_of_computer_table
  (C : ℝ) 
  (S : ℝ := 1.20 * C)
  (S_eq : S = 8600) : 
  C = 7166.67 :=
by
  sorry

end cost_price_of_computer_table_l240_240454


namespace sequence_bound_equivalent_problem_l240_240871

variable {n : ℕ}
variable {a : Fin (n+2) → ℝ}

theorem sequence_bound_equivalent_problem (h1 : a 0 = 0) (h2 : a (n + 1) = 0) 
  (h3 : ∀ k : Fin n, |a (k.val - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : Fin (n+2), |a k| ≤ k * (n + 1 - k) / 2 := 
by
  sorry

end sequence_bound_equivalent_problem_l240_240871


namespace parallel_line_through_P_perpendicular_line_through_P_l240_240124

-- Define the line equations
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define the equations for parallel and perpendicular lines through point P
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

-- Define the point P where the lines intersect
def point_P : (ℝ × ℝ) := (2, 1)

-- Assert the proof statements
theorem parallel_line_through_P : parallel_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry
  
theorem perpendicular_line_through_P : perpendicular_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry

end parallel_line_through_P_perpendicular_line_through_P_l240_240124


namespace nondegenerate_ellipse_iff_l240_240205

theorem nondegenerate_ellipse_iff (k : ℝ) :
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = k) ↔ k > -117/4 :=
by
  sorry

end nondegenerate_ellipse_iff_l240_240205


namespace washer_cost_l240_240796

theorem washer_cost (D : ℝ) (H1 : D + (D + 220) = 1200) : D + 220 = 710 :=
by
  sorry

end washer_cost_l240_240796


namespace value_of_x_div_y_l240_240844

theorem value_of_x_div_y (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, x = t * y ∧ t = -2 := 
sorry

end value_of_x_div_y_l240_240844


namespace articles_selling_price_eq_cost_price_of_50_articles_l240_240452

theorem articles_selling_price_eq_cost_price_of_50_articles (C S : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * S) (h2 : S = 2 * C) : N = 25 := by
  sorry

end articles_selling_price_eq_cost_price_of_50_articles_l240_240452


namespace exists_quad_root_l240_240173

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l240_240173


namespace find_speed_l240_240006

noncomputable def distance : ℝ := 600
noncomputable def speed1 : ℝ := 50
noncomputable def meeting_distance : ℝ := distance / 2
noncomputable def departure_time1 : ℝ := 7
noncomputable def departure_time2 : ℝ := 8
noncomputable def meeting_time : ℝ := 13

theorem find_speed (x : ℝ) : 
  (meeting_distance / speed1 = meeting_time - departure_time1) ∧
  (meeting_distance / x = meeting_time - departure_time2) → 
  x = 60 :=
by
  sorry

end find_speed_l240_240006


namespace simplified_equation_has_solution_l240_240281

theorem simplified_equation_has_solution (n : ℤ) :
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x * y - y * z - z * x = n) →
  (∃ x y : ℤ, x^2 + y^2 - x * y = n) :=
by
  intros h
  exact sorry

end simplified_equation_has_solution_l240_240281


namespace f_800_value_l240_240433

theorem f_800_value (f : ℝ → ℝ) (f_condition : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y) (f_400 : f 400 = 4) : f 800 = 2 :=
  sorry

end f_800_value_l240_240433


namespace nonnegative_poly_sum_of_squares_l240_240632

open Polynomial

theorem nonnegative_poly_sum_of_squares (P : Polynomial ℝ) 
    (hP : ∀ x : ℝ, 0 ≤ P.eval x) 
    : ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 := 
by
  sorry

end nonnegative_poly_sum_of_squares_l240_240632


namespace number_of_solutions_l240_240810

theorem number_of_solutions : 
  ∃ n : ℕ, (∀ x y : ℕ, 3 * x + 4 * y = 766 → x % 2 = 0 → x > 0 → y > 0 → x = n * 2) ∧ n = 127 := 
by
  sorry

end number_of_solutions_l240_240810


namespace original_houses_count_l240_240155

namespace LincolnCounty

-- Define the constants based on the conditions
def houses_built_during_boom : ℕ := 97741
def houses_now : ℕ := 118558

-- Statement of the theorem
theorem original_houses_count : houses_now - houses_built_during_boom = 20817 := 
by sorry

end LincolnCounty

end original_houses_count_l240_240155


namespace choir_members_l240_240289

theorem choir_members (k m n : ℕ) (h1 : n = k^2 + 11) (h2 : n = m * (m + 5)) : n ≤ 325 :=
by
  sorry -- A proof would go here, showing that n = 325 meets the criteria

end choir_members_l240_240289


namespace solution_is_correct_l240_240724

def valid_triple (a b c : ℕ) : Prop :=
  (Nat.gcd a 20 = b) ∧ (Nat.gcd b 15 = c) ∧ (Nat.gcd a c = 5)

def is_solution_set (triples : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ a b c, (a, b, c) ∈ triples ↔ 
    (valid_triple a b c) ∧ 
    ((∃ k, a = 20 * k ∧ b = 20 ∧ c = 5) ∨
    (∃ k, a = 20 * k - 10 ∧ b = 10 ∧ c = 5) ∨
    (∃ k, a = 10 * k - 5 ∧ b = 5 ∧ c = 5))

theorem solution_is_correct : ∃ S, is_solution_set S :=
sorry

end solution_is_correct_l240_240724


namespace complex_quadrant_l240_240475

theorem complex_quadrant (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : 1 + a * i = (b + i) * (1 + i)) : 
  (a - b * i).re > 0 ∧ (a - b * i).im < 0 :=
by
  have h1 : 1 + a * i = (b - 1) + (b + 1) * i := by sorry
  have h2 : a = b + 1 := by sorry
  have h3 : b - 1 = 1 := by sorry
  have h4 : b = 2 := by sorry
  have h5 : a = 3 := by sorry
  have h6 : (a - b * i).re = 3 := by sorry
  have h7 : (a - b * i).im = -2 := by sorry
  exact ⟨by linarith, by linarith⟩

end complex_quadrant_l240_240475


namespace gray_area_correct_l240_240244

-- Define the side lengths of the squares
variable (a b : ℝ)

-- Define the areas of the larger and smaller squares
def area_large_square : ℝ := (a + b) * (a + b)
def area_small_square : ℝ := a * a

-- Define the gray area
def gray_area : ℝ := area_large_square a b - area_small_square a

-- The proof statement
theorem gray_area_correct (a b : ℝ) : gray_area a b = 2 * a * b + b ^ 2 := by
  sorry

end gray_area_correct_l240_240244


namespace intersection_point_l240_240912

theorem intersection_point (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) :
  ∃ x y, (y = a * x^2 + b * x + c) ∧ (y = a * x^2 - b * x + c + d) ∧ x = d / (2 * b) ∧ y = a * (d / (2 * b))^2 + (d / 2) + c :=
by
  sorry

end intersection_point_l240_240912


namespace equation_circle_iff_a_equals_neg_one_l240_240091

theorem equation_circle_iff_a_equals_neg_one :
  (∀ x y : ℝ, ∃ k : ℝ, a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = k * (x^2 + y^2)) ↔ 
  a = -1 :=
by sorry

end equation_circle_iff_a_equals_neg_one_l240_240091


namespace greatest_possible_value_of_x_l240_240128

theorem greatest_possible_value_of_x (x : ℝ) (h : ( (5 * x - 25) / (4 * x - 5) ) ^ 3 + ( (5 * x - 25) / (4 * x - 5) ) = 16):
  x = 5 :=
sorry

end greatest_possible_value_of_x_l240_240128


namespace series_divergence_l240_240012

theorem series_divergence (a : ℕ → ℝ) (hdiv : ¬ ∃ l, ∑' n, a n = l) (hpos : ∀ n, a n > 0) (hnoninc : ∀ n m, n ≤ m → a m ≤ a n) : 
  ¬ ∃ l, ∑' n, (a n / (1 + n * a n)) = l :=
by
  sorry

end series_divergence_l240_240012


namespace find_c_l240_240460

theorem find_c (a c : ℝ) (h1 : x^2 + 80 * x + c = (x + a)^2) (h2 : 2 * a = 80) : c = 1600 :=
sorry

end find_c_l240_240460


namespace teacher_drank_milk_false_l240_240955

-- Define the condition that the volume of milk a teacher can reasonably drink in a day is more appropriately measured in milliliters rather than liters.
def reasonable_volume_units := "milliliters"

-- Define the statement to be judged
def teacher_milk_intake := 250

-- Define the unit of the statement
def unit_of_statement := "liters"

-- The proof goal is to conclude that the statement "The teacher drank 250 liters of milk today" is false, given the condition on volume units.
theorem teacher_drank_milk_false (vol : ℕ) (unit : String) (reasonable_units : String) :
  vol = 250 ∧ unit = "liters" ∧ reasonable_units = "milliliters" → false :=
by
  sorry

end teacher_drank_milk_false_l240_240955


namespace sufficient_but_not_necessary_condition_l240_240245

def vectors_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

def vector_a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  x = 3 → vectors_parallel (vector_a x) (vector_b x) ∧
  vectors_parallel (vector_a 3) (vector_b 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l240_240245


namespace exists_unique_c_for_a_equals_3_l240_240863

theorem exists_unique_c_for_a_equals_3 :
  ∃! c : ℝ, ∀ x ∈ Set.Icc (3 : ℝ) 9, ∃ y ∈ Set.Icc (3 : ℝ) 27, Real.log x / Real.log 3 + Real.log y / Real.log 3 = c :=
sorry

end exists_unique_c_for_a_equals_3_l240_240863


namespace new_sign_cost_l240_240073

theorem new_sign_cost 
  (p_s : ℕ) (p_c : ℕ) (n : ℕ) (h_ps : p_s = 30) (h_pc : p_c = 26) (h_n : n = 10) : 
  (p_s - p_c) * n / 2 = 20 := 
by 
  sorry

end new_sign_cost_l240_240073


namespace book_pages_l240_240852

theorem book_pages (books sheets pages_per_sheet pages_per_book : ℕ)
  (hbooks : books = 2)
  (hpages_per_sheet : pages_per_sheet = 8)
  (hsheets : sheets = 150)
  (htotal_pages : pages_per_sheet * sheets = 1200)
  (hpages_per_book : pages_per_book = 1200 / books) :
  pages_per_book = 600 :=
by
  -- Proof goes here
  sorry

end book_pages_l240_240852


namespace water_level_balance_l240_240060

noncomputable def exponential_decay (a n t : ℝ) : ℝ := a * Real.exp (n * t)

theorem water_level_balance
  (a : ℝ)
  (n : ℝ)
  (m : ℝ)
  (h5 : exponential_decay a n 5 = a / 2)
  (h8 : exponential_decay a n m = a / 8) :
  m = 10 := by
  sorry

end water_level_balance_l240_240060


namespace inequality_for_M_cap_N_l240_240533

def f (x : ℝ) := 2 * |x - 1| + x - 1
def g (x : ℝ) := 16 * x^2 - 8 * x + 1

def M := {x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3}
def N := {x : ℝ | -1 / 4 ≤ x ∧ x ≤ 3 / 4}
def M_cap_N := {x : ℝ | 0 ≤ x ∧ x ≤ 3 / 4}

theorem inequality_for_M_cap_N (x : ℝ) (hx : x ∈ M_cap_N) : x^2 * f x + x * (f x)^2 ≤ 1 / 4 := 
by 
  sorry

end inequality_for_M_cap_N_l240_240533


namespace corresponding_angles_not_always_equal_l240_240140

theorem corresponding_angles_not_always_equal :
  (∀ α β c : ℝ, (α = β ∧ ¬c = 0) → (∃ x1 x2 y : ℝ, α = x1 ∧ β = x2 ∧ x1 = y * c ∧ x2 = y * c)) → False :=
by
  sorry

end corresponding_angles_not_always_equal_l240_240140


namespace geometric_sequences_common_ratios_l240_240735

theorem geometric_sequences_common_ratios 
  (k m n o : ℝ)
  (a_2 a_3 b_2 b_3 c_2 c_3 : ℝ)
  (h1 : a_2 = k * m)
  (h2 : a_3 = k * m^2)
  (h3 : b_2 = k * n)
  (h4 : b_3 = k * n^2)
  (h5 : c_2 = k * o)
  (h6 : c_3 = k * o^2)
  (h7 : a_3 - b_3 + c_3 = 2 * (a_2 - b_2 + c_2))
  (h8 : m ≠ n)
  (h9 : m ≠ o)
  (h10 : n ≠ o) : 
  m + n + o = 1 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequences_common_ratios_l240_240735


namespace kim_time_away_from_home_l240_240757

noncomputable def time_away_from_home (distance_to_friend : ℕ) (detour_percent : ℕ) (stay_time : ℕ) (speed_mph : ℕ) : ℕ :=
  let return_distance := distance_to_friend * (1 + detour_percent / 100)
  let total_distance := distance_to_friend + return_distance
  let driving_time := total_distance / speed_mph
  let driving_time_minutes := driving_time * 60
  driving_time_minutes + stay_time

theorem kim_time_away_from_home : 
  time_away_from_home 30 20 30 44 = 120 := 
by
  -- We will handle the proof here
  sorry

end kim_time_away_from_home_l240_240757


namespace recorder_price_new_l240_240154

theorem recorder_price_new (a b : ℕ) (h1 : 10 * a + b < 50) (h2 : 10 * b + a = (10 * a + b) * 12 / 10) :
  10 * b + a = 54 :=
by
  sorry

end recorder_price_new_l240_240154


namespace problem_part1_problem_part2_l240_240338

-- Define the sequences and conditions
variable {a b : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {T : ℕ → ℕ}
variable {d q : ℕ}
variable {b_initial : ℕ}

axiom geom_seq (n : ℕ) : b n = b_initial * q^n
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Problem conditions
axiom cond_geom_seq : b_initial = 2
axiom cond_geom_b2_b3 : b 2 + b 3 = 12
axiom cond_geom_ratio : q > 0
axiom cond_relation_b3_a4 : b 3 = a 4 - 2 * a 1
axiom cond_sum_S_11_b4 : S 11 = 11 * b 4

-- Theorem statement
theorem problem_part1 :
  (a n = 3 * n - 2) ∧ (b n = 2 ^ n) :=
  sorry

theorem problem_part2 :
  (T n = (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) :=
  sorry

end problem_part1_problem_part2_l240_240338


namespace total_ice_cubes_correct_l240_240437

/-- Each tray holds 48 ice cubes -/
def cubes_per_tray : Nat := 48

/-- Billy has 24 trays -/
def number_of_trays : Nat := 24

/-- Calculate the total number of ice cubes -/
def total_ice_cubes (cubes_per_tray : Nat) (number_of_trays : Nat) : Nat :=
  cubes_per_tray * number_of_trays

/-- Proof that the total number of ice cubes is 1152 given the conditions -/
theorem total_ice_cubes_correct : total_ice_cubes cubes_per_tray number_of_trays = 1152 := by
  /- Here we state the main theorem, but we leave the proof as sorry per the instructions -/
  sorry

end total_ice_cubes_correct_l240_240437


namespace calc_expression_l240_240674

theorem calc_expression : (900^2) / (264^2 - 256^2) = 194.711 := by
  sorry

end calc_expression_l240_240674


namespace proof_max_ρ_sq_l240_240732

noncomputable def max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b) 
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : ℝ :=
  (a / b) ^ 2

theorem proof_max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b)
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : (max_ρ_sq a b h₀ h₁ h₂ x y h₃ h₄ h₅ h₆ h_xy h_eq h_x_le) ≤ 9 / 5 := by
  sorry

end proof_max_ρ_sq_l240_240732


namespace sixth_term_is_sixteen_l240_240241

-- Definition of the conditions
def first_term : ℝ := 512
def eighth_term (r : ℝ) : Prop := 512 * r^7 = 2

-- Proving the 6th term is 16 given the conditions
theorem sixth_term_is_sixteen (r : ℝ) (hr : eighth_term r) :
  512 * r^5 = 16 :=
by
  sorry

end sixth_term_is_sixteen_l240_240241


namespace fraction_of_work_left_l240_240513

theorem fraction_of_work_left 
  (A_days : ℝ) (B_days : ℝ) (work_days : ℝ) 
  (A_work_rate : A_days = 15) 
  (B_work_rate : B_days = 30) 
  (work_duration : work_days = 4)
  : (1 - (work_days * ((1 / A_days) + (1 / B_days)))) = 3 / 5 := 
by
  sorry

end fraction_of_work_left_l240_240513
