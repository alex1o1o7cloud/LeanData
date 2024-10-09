import Mathlib

namespace number_of_female_students_l1545_154512

theorem number_of_female_students
  (F : ℕ) -- number of female students
  (T : ℕ) -- total number of students
  (h1 : T = F + 8) -- total students = female students + 8 male students
  (h2 : 90 * T = 85 * 8 + 92 * F) -- equation from the sum of scores
  : F = 20 :=
sorry

end number_of_female_students_l1545_154512


namespace graph_EQ_a_l1545_154544

theorem graph_EQ_a (x y : ℝ) : (x - 2) * (y + 3) = 0 ↔ x = 2 ∨ y = -3 :=
by sorry

end graph_EQ_a_l1545_154544


namespace solve_pears_and_fruits_l1545_154513

noncomputable def pears_and_fruits_problem : Prop :=
  ∃ (x y : ℕ), x + y = 1000 ∧ (11 * x) * (1/9 : ℚ) + (4 * y) * (1/7 : ℚ) = 999

theorem solve_pears_and_fruits :
  pears_and_fruits_problem := by
  sorry

end solve_pears_and_fruits_l1545_154513


namespace find_a_and_theta_find_max_min_g_l1545_154597

noncomputable def f (x a θ : ℝ) : ℝ := (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

-- Provided conditions
variable (a : ℝ)
variable (θ : ℝ)
variable (is_odd : ∀ x, f x a θ = -f (-x) a θ)
variable (f_pi_over_4 : f ((Real.pi) / 4) a θ = 0)
variable (theta_in_range : 0 < θ ∧ θ < Real.pi)

-- To Prove
theorem find_a_and_theta :
  a = -1 ∧ θ = (Real.pi / 2) :=
sorry

-- Define g(x) and its domain
noncomputable def g (x : ℝ) : ℝ := f x (-1) (Real.pi / 2) + f (x + (Real.pi / 3)) (-1) (Real.pi / 2)

-- Provided domain condition
variable (x_in_domain : 0 ≤ x ∧ x ≤ (Real.pi / 4))

-- To Prove maximum and minimum value of g(x)
theorem find_max_min_g :
  (∀ x, x ∈ Set.Icc (0 : ℝ) (Real.pi / 4) → -((Real.sqrt 3) / 2) ≤ g x ∧ g x ≤ (Real.sqrt 3) / 2)
  ∧ ∃ x_min, g x_min = -((Real.sqrt 3) / 2) ∧ x_min = (Real.pi / 8)
  ∧ ∃ x_max, g x_max = ((Real.sqrt 3) / 2) ∧ x_max = (Real.pi / 4) :=
sorry

end find_a_and_theta_find_max_min_g_l1545_154597


namespace Glorys_favorite_number_l1545_154546

variable (M G : ℝ)

theorem Glorys_favorite_number :
  (M = G / 3) →
  (M + G = 600) →
  (G = 450) :=
by
sorry

end Glorys_favorite_number_l1545_154546


namespace max_value_h3_solve_for_h_l1545_154511

-- Definition part for conditions
def quadratic_function (h : ℝ) (x : ℝ) : ℝ :=
  -(x - h) ^ 2

-- Part (1): When h = 3, proving the maximum value of the function within 2 ≤ x ≤ 5 is 0.
theorem max_value_h3 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function 3 x ≤ 0 :=
by
  sorry

-- Part (2): If the maximum value of the function is -1, then the value of h is 6 or 1.
theorem solve_for_h (h : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function h x ≤ -1) ↔ h = 6 ∨ h = 1 :=
by
  sorry

end max_value_h3_solve_for_h_l1545_154511


namespace total_population_l1545_154583

theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t :=
by
  -- We state our theorem including assumptions and goal
  sorry -- placeholder for the proof

end total_population_l1545_154583


namespace train_cross_pole_time_l1545_154515

-- Defining the given conditions
def speed_km_hr : ℕ := 54
def length_m : ℕ := 135

-- Conversion of speed from km/hr to m/s
def speed_m_s : ℤ := (54 * 1000) / 3600

-- Statement to be proved
theorem train_cross_pole_time : (length_m : ℤ) / speed_m_s = 9 := by
  sorry

end train_cross_pole_time_l1545_154515


namespace area_converted_2018_l1545_154502

theorem area_converted_2018 :
  let a₁ := 8 -- initial area in ten thousand hectares
  let q := 1.1 -- common ratio
  let a₆ := a₁ * q^5 -- area converted in 2018
  a₆ = 8 * 1.1^5 :=
sorry

end area_converted_2018_l1545_154502


namespace multiple_of_9_is_multiple_of_3_l1545_154549

theorem multiple_of_9_is_multiple_of_3 (n : ℤ) (h : ∃ k : ℤ, n = 9 * k) : ∃ m : ℤ, n = 3 * m :=
by
  sorry

end multiple_of_9_is_multiple_of_3_l1545_154549


namespace number_of_monomials_l1545_154501

def isMonomial (expr : String) : Bool :=
  match expr with
  | "-(2 / 3) * a^3 * b" => true
  | "(x * y) / 2" => true
  | "-4" => true
  | "0" => true
  | _ => false

def countMonomials (expressions : List String) : Nat :=
  expressions.foldl (fun acc expr => if isMonomial expr then acc + 1 else acc) 0

theorem number_of_monomials : countMonomials ["-(2 / 3) * a^3 * b", "(x * y) / 2", "-4", "-(2 / a)", "0", "x - y"] = 4 :=
by
  sorry

end number_of_monomials_l1545_154501


namespace susie_total_earnings_l1545_154542

def pizza_prices (type : String) (is_whole : Bool) : ℝ :=
  match type, is_whole with
  | "Margherita", false => 3
  | "Margherita", true  => 15
  | "Pepperoni", false  => 4
  | "Pepperoni", true   => 18
  | "Veggie Supreme", false => 5
  | "Veggie Supreme", true  => 22
  | "Meat Lovers", false => 6
  | "Meat Lovers", true  => 25
  | "Hawaiian", false   => 4.5
  | "Hawaiian", true    => 20
  | _, _                => 0

def topping_price (is_weekend : Bool) : ℝ :=
  if is_weekend then 1 else 2

def happy_hour_price : ℝ := 3

noncomputable def susie_earnings : ℝ :=
  let margherita_slices := 12 * happy_hour_price + 12 * pizza_prices "Margherita" false
  let pepperoni_slices := 8 * happy_hour_price + 8 * pizza_prices "Pepperoni" false + 6 * topping_price true
  let veggie_supreme_pizzas := 4 * pizza_prices "Veggie Supreme" true + 8 * topping_price true
  let margherita_whole_discounted := 3 * pizza_prices "Margherita" true - (3 * pizza_prices "Margherita" true) * 0.1
  let meat_lovers_slices := 10 * happy_hour_price + 10 * pizza_prices "Meat Lovers" false
  let hawaiian_slices := 12 * pizza_prices "Hawaiian" false + 4 * topping_price true
  let pepperoni_whole := pizza_prices "Pepperoni" true + 3 * topping_price true
  margherita_slices + pepperoni_slices + veggie_supreme_pizzas + margherita_whole_discounted + meat_lovers_slices + hawaiian_slices + pepperoni_whole

theorem susie_total_earnings : susie_earnings = 439.5 := by
  sorry

end susie_total_earnings_l1545_154542


namespace triangle_ABC_right_angled_l1545_154588
open Real

theorem triangle_ABC_right_angled (A B C : ℝ) (a b c : ℝ)
  (h1 : cos (2 * A) - cos (2 * B) = 2 * sin C ^ 2)
  (h2 : a = sin A) (h3 : b = sin B) (h4 : c = sin C)
  : a^2 + c^2 = b^2 :=
by sorry

end triangle_ABC_right_angled_l1545_154588


namespace length_of_AE_l1545_154547

theorem length_of_AE (AD AE EB EF: ℝ) (h_AD: AD = 80) (h_EB: EB = 40) (h_EF: EF = 30) 
  (h_eq_area: 2 * ((EB * EF) + (1 / 2) * (ED * (AD - EF))) = AD * (AD - AE)) : AE = 15 :=
  sorry

end length_of_AE_l1545_154547


namespace sequence_initial_value_l1545_154561

theorem sequence_initial_value (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : a 1 = 0 ∨ a 1 = 2 :=
sorry

end sequence_initial_value_l1545_154561


namespace Olga_paints_zero_boards_l1545_154590

variable (t p q t' : ℝ)
variable (rv ro : ℝ)

-- Conditions
axiom Valera_solo_trip : 2 * t + p = 2
axiom Valera_and_Olga_painting_time : 2 * t' + q = 3
axiom Valera_painting_rate : rv = 11 / p
axiom Valera_Omega_painting_rate : rv * q + ro * q = 9
axiom Valera_walk_faster : t' > t

-- Question: How many boards will Olga be able to paint alone if she needs to return home 1 hour after leaving?
theorem Olga_paints_zero_boards :
  t' > 1 → 0 = 0 := 
by 
  sorry

end Olga_paints_zero_boards_l1545_154590


namespace value_of_c_l1545_154532

theorem value_of_c (c : ℝ) :
  (∀ x y : ℝ, (x, y) = ((2 + 8) / 2, (6 + 10) / 2) → x + y = c) → c = 13 :=
by
  -- Placeholder for proof
  sorry

end value_of_c_l1545_154532


namespace rhombus_area_l1545_154526

theorem rhombus_area (x y : ℝ)
  (h1 : x^2 + y^2 = 113) 
  (h2 : x = y + 8) : 
  1 / 2 * (2 * y) * (2 * (y + 4)) = 97 := 
by 
  -- Assume x and y are the half-diagonals of the rhombus
  sorry

end rhombus_area_l1545_154526


namespace part_I_part_II_l1545_154553

theorem part_I : 
  (∀ x : ℝ, |x - (2 : ℝ)| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) :=
  sorry

theorem part_II :
  (∀ a b c : ℝ, a - 2 * b + c = 2 → a^2 + b^2 + c^2 ≥ 2 / 3) :=
  sorry

end part_I_part_II_l1545_154553


namespace angle_measure_l1545_154575

theorem angle_measure (α : ℝ) (h1 : α - (90 - α) = 20) : α = 55 := by
  -- Proof to be provided here
  sorry

end angle_measure_l1545_154575


namespace krishan_nandan_investment_ratio_l1545_154594

theorem krishan_nandan_investment_ratio
    (X t : ℝ) (k : ℝ)
    (h1 : X * t = 6000)
    (h2 : X * t + k * X * 2 * t = 78000) :
    k = 6 := by
  sorry

end krishan_nandan_investment_ratio_l1545_154594


namespace correct_calculation_l1545_154536

theorem correct_calculation (x : ℤ) (h : 20 + x = 60) : 34 - x = -6 := by
  sorry

end correct_calculation_l1545_154536


namespace five_cubic_km_to_cubic_meters_l1545_154570

theorem five_cubic_km_to_cubic_meters (km_to_m : 1 = 1000) : 
  5 * (1000 ^ 3) = 5000000000 := 
by
  sorry

end five_cubic_km_to_cubic_meters_l1545_154570


namespace find_circle_radius_l1545_154580

-- Definitions of given distances and the parallel chord condition
def isChordParallelToDiameter (c d : ℝ × ℝ) (radius distance1 distance2 : ℝ) : Prop :=
  let p1 := distance1
  let p2 := distance2
  p1 = 5 ∧ p2 = 12 ∧ 
  -- Assuming distances from the end of the diameter to the ends of the chord
  true

-- The main theorem which states the radius of the circle given the conditions
theorem find_circle_radius
  (diameter chord : ℝ × ℝ)
  (R p1 p2 : ℝ)
  (h1 : isChordParallelToDiameter diameter chord R p1 p2) :
  R = 6.5 :=
  by
    sorry

end find_circle_radius_l1545_154580


namespace outfit_combinations_l1545_154552

theorem outfit_combinations (shirts ties hat_choices : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 7) (h_hat_choices : hat_choices = 3) : shirts * ties * hat_choices = 168 := by
  sorry

end outfit_combinations_l1545_154552


namespace jenna_costume_l1545_154531

def cost_of_skirts (skirt_count : ℕ) (material_per_skirt : ℕ) : ℕ :=
  skirt_count * material_per_skirt

def cost_of_bodice (shirt_material : ℕ) (sleeve_material_per : ℕ) (sleeve_count : ℕ) : ℕ :=
  shirt_material + (sleeve_material_per * sleeve_count)

def total_material (skirt_material : ℕ) (bodice_material : ℕ) : ℕ :=
  skirt_material + bodice_material

def total_cost (total_material : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  total_material * cost_per_sqft

theorem jenna_costume : 
  cost_of_skirts 3 48 + cost_of_bodice 2 5 2 = 156 → total_cost 156 3 = 468 :=
by
  sorry

end jenna_costume_l1545_154531


namespace mario_total_flowers_l1545_154543

def hibiscus_flower_count (n : ℕ) : ℕ :=
  let h1 := 2 + 3 * n
  let h2 := (2 * 2) + 4 * n
  let h3 := (4 * (2 * 2)) + 5 * n
  h1 + h2 + h3

def rose_flower_count (n : ℕ) : ℕ :=
  let r1 := 3 + 2 * n
  let r2 := 5 + 3 * n
  r1 + r2

def sunflower_flower_count (n : ℕ) : ℕ :=
  6 * 2^n

def total_flower_count (n : ℕ) : ℕ :=
  hibiscus_flower_count n + rose_flower_count n + sunflower_flower_count n

theorem mario_total_flowers :
  total_flower_count 2 = 88 :=
by
  unfold total_flower_count hibiscus_flower_count rose_flower_count sunflower_flower_count
  norm_num

end mario_total_flowers_l1545_154543


namespace point_not_in_region_l1545_154568

theorem point_not_in_region : ¬ (3 * 2 + 2 * 0 < 6) :=
by simp [lt_irrefl]

end point_not_in_region_l1545_154568


namespace symmetric_point_x_axis_l1545_154563

theorem symmetric_point_x_axis (P : ℝ × ℝ) (hx : P = (2, 3)) : P.1 = 2 ∧ P.2 = -3 :=
by
  -- The proof is omitted
  sorry

end symmetric_point_x_axis_l1545_154563


namespace div_add_example_l1545_154599

theorem div_add_example : 150 / (10 / 2) + 5 = 35 := by
  sorry

end div_add_example_l1545_154599


namespace right_triangle_of_condition_l1545_154556

theorem right_triangle_of_condition
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_trig : Real.sin γ - Real.cos α = Real.cos β) :
  (α = 90) ∨ (β = 90) :=
sorry

end right_triangle_of_condition_l1545_154556


namespace max_a_value_l1545_154550

-- Variables representing the real numbers a, b, c, and d
variables (a b c d : ℝ)

-- Real number hypothesis conditions
-- 1. a + b + c + d = 10
-- 2. ab + ac + ad + bc + bd + cd = 20

theorem max_a_value
  (h1 : a + b + c + d = 10)
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  a ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_a_value_l1545_154550


namespace qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l1545_154591

variable (m : Int)

theorem qiqi_initial_batteries (m : Int) : 
  let Qiqi_initial := 2 * m - 2
  Qiqi_initial = 2 * m - 2 := sorry

theorem qiqi_jiajia_difference_after_transfer (m : Int) : 
  let Qiqi_after := 2 * m - 2 - 2
  let Jiajia_after := m + 2
  Qiqi_after - Jiajia_after = m - 6 := sorry

end qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l1545_154591


namespace root_of_polynomial_l1545_154592

theorem root_of_polynomial (a b : ℝ) (h₁ : a^4 + a^3 - 1 = 0) (h₂ : b^4 + b^3 - 1 = 0) : 
  (ab : ℝ) → ab * ab * ab * ab * ab * ab + ab * ab * ab * ab + ab * ab * ab - ab * ab - 1 = 0 :=
sorry

end root_of_polynomial_l1545_154592


namespace find_x_l1545_154538

variable (x : ℝ)

theorem find_x (h : 0.60 * x = (1/3) * x + 110) : x = 412.5 :=
sorry

end find_x_l1545_154538


namespace problem_1_problem_2_l1545_154596

variable (x y : ℝ)
noncomputable def x_val : ℝ := 2 + Real.sqrt 3
noncomputable def y_val : ℝ := 2 - Real.sqrt 3

theorem problem_1 :
  3 * x_val^2 + 5 * x_val * y_val + 3 * y_val^2 = 47 := sorry

theorem problem_2 :
  Real.sqrt (x_val / y_val) + Real.sqrt (y_val / x_val) = 4 := sorry

end problem_1_problem_2_l1545_154596


namespace asymptote_of_hyperbola_l1545_154535

theorem asymptote_of_hyperbola (h : (∀ x y : ℝ, y^2 / 3 - x^2 / 2 = 1)) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (sqrt6 / 2) * x ∨ y = - (sqrt6 / 2) * x) :=
sorry

end asymptote_of_hyperbola_l1545_154535


namespace simplify_and_evaluate_l1545_154534

theorem simplify_and_evaluate :
  ∀ (a b : ℤ), a = -1 → b = 4 →
  (a + b)^2 - 2 * a * (a - b) + (a + 2 * b) * (a - 2 * b) = -64 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l1545_154534


namespace factor_expression_l1545_154521

theorem factor_expression (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) :=
by
  sorry

end factor_expression_l1545_154521


namespace geometric_sequence_common_ratio_l1545_154587

theorem geometric_sequence_common_ratio (a : ℕ → ℕ) (q : ℕ) (h2 : a 2 = 8) (h5 : a 5 = 64)
  (h_geom : ∀ n, a (n+1) = a n * q) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l1545_154587


namespace chord_length_perpendicular_bisector_l1545_154530

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) :
  ∃ (CD : ℝ), CD = 10 * Real.sqrt 3 :=
by
  -- The proof is omitted.
  sorry

end chord_length_perpendicular_bisector_l1545_154530


namespace proportion_condition_l1545_154560

variable (a b c d a₁ b₁ c₁ d₁ : ℚ)

theorem proportion_condition
  (h₁ : a / b = c / d)
  (h₂ : a₁ / b₁ = c₁ / d₁) :
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ := by
  sorry

end proportion_condition_l1545_154560


namespace proof_problem_l1545_154574

noncomputable def question (a b c : ℝ) : ℝ := 
  (a ^ 2 * b ^ 2) / ((a ^ 2 + b * c) * (b ^ 2 + a * c)) +
  (a ^ 2 * c ^ 2) / ((a ^ 2 + b * c) * (c ^ 2 + a * b)) +
  (b ^ 2 * c ^ 2) / ((b ^ 2 + a * c) * (c ^ 2 + a * b))

theorem proof_problem (a b c : ℝ) (h : a ≠ 0) (h1 : b ≠ 0) (h2 : c ≠ 0) 
  (h3 : a ^ 2 + b ^ 2 + c ^ 2 = a * b + b * c + c * a ) : 
  question a b c = 1 := 
by 
  sorry

end proof_problem_l1545_154574


namespace exist_pair_lcm_gcd_l1545_154573

theorem exist_pair_lcm_gcd (a b: ℤ) : 
  ∃ a b : ℤ, Int.lcm a b - Int.gcd a b = 19 := 
sorry

end exist_pair_lcm_gcd_l1545_154573


namespace range_of_a_l1545_154503

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 0 → x^2 + 2 * x - 3 + a ≤ 0) ↔ a ≤ -12 :=
by
  sorry

end range_of_a_l1545_154503


namespace find_f_of_2_l1545_154539

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f_of_2_l1545_154539


namespace spring_stretch_150N_l1545_154576

-- Definitions for the conditions
def spring_stretch (weight : ℕ) : ℕ :=
  if weight = 100 then 20 else sorry

-- The theorem to prove
theorem spring_stretch_150N : spring_stretch 150 = 30 := by
  sorry

end spring_stretch_150N_l1545_154576


namespace triangles_from_sticks_l1545_154569

theorem triangles_from_sticks (a1 a2 a3 a4 a5 a6 : ℕ) (h_diff: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 
∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 
∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 
∧ a4 ≠ a5 ∧ a4 ≠ a6 
∧ a5 ≠ a6) (h_order: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) : 
  (a1 + a3 > a5 ∧ a1 + a5 > a3 ∧ a3 + a5 > a1) ∧ 
  (a2 + a4 > a6 ∧ a2 + a6 > a4 ∧ a4 + a6 > a2) :=
by
  sorry

end triangles_from_sticks_l1545_154569


namespace sum_of_square_areas_l1545_154517

variable (WX XZ : ℝ)

theorem sum_of_square_areas (hW : WX = 15) (hX : XZ = 20) : WX^2 + XZ^2 = 625 := by
  sorry

end sum_of_square_areas_l1545_154517


namespace overall_average_length_of_ropes_l1545_154595

theorem overall_average_length_of_ropes :
  let ropes := 6
  let third_part := ropes / 3
  let average1 := 70
  let average2 := 85
  let length1 := third_part * average1
  let length2 := (ropes - third_part) * average2
  let total_length := length1 + length2
  let overall_average := total_length / ropes
  overall_average = 80 := by
sorry

end overall_average_length_of_ropes_l1545_154595


namespace total_tape_length_l1545_154540

-- Definitions based on the problem conditions
def first_side_songs : ℕ := 6
def second_side_songs : ℕ := 4
def song_length : ℕ := 4

-- Statement to prove the total tape length is 40 minutes
theorem total_tape_length : (first_side_songs + second_side_songs) * song_length = 40 := by
  sorry

end total_tape_length_l1545_154540


namespace largest_number_l1545_154557

theorem largest_number (a b c : ℕ) (h1: a ≤ b) (h2: b ≤ c) 
  (h3: (a + b + c) = 90) (h4: b = 32) (h5: b = a + 4) : c = 30 :=
sorry

end largest_number_l1545_154557


namespace sequence_bound_l1545_154507

noncomputable def sequenceProperties (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ i, 0 ≤ a i ∧ a i ≤ c) ∧ (∀ i j, i ≠ j → abs (a i - a j) ≥ 1 / (i + j))

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) (h : sequenceProperties a c) : 
  c ≥ 1 :=
by
  sorry

end sequence_bound_l1545_154507


namespace house_height_l1545_154522

theorem house_height
  (tree_height : ℕ) (tree_shadow : ℕ)
  (house_shadow : ℕ) (h : ℕ) :
  tree_height = 15 →
  tree_shadow = 18 →
  house_shadow = 72 →
  (h / tree_height) = (house_shadow / tree_shadow) →
  h = 60 :=
by
  intros h1 h2 h3 h4
  have h5 : h / 15 = 72 / 18 := by
    rw [h1, h2, h3] at h4
    exact h4
  sorry

end house_height_l1545_154522


namespace count_4_tuples_l1545_154524

theorem count_4_tuples (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  Nat.card {abcd : ℕ × ℕ × ℕ × ℕ // (0 < abcd.1 ∧ abcd.1 < p) ∧ 
                                     (0 < abcd.2.1 ∧ abcd.2.1 < p) ∧ 
                                     (0 < abcd.2.2.1 ∧ abcd.2.2.1 < p) ∧ 
                                     (0 < abcd.2.2.2 ∧ abcd.2.2.2 < p) ∧ 
                                     ((abcd.1 * abcd.2.2.2 - abcd.2.1 * abcd.2.2.1) % p = 0)} = (p - 1) * (p - 1) * (p - 1) :=
by
  sorry

end count_4_tuples_l1545_154524


namespace value_of_d_l1545_154586

theorem value_of_d (d y : ℤ) (h₁ : y = 2) (h₂ : 5 * y^2 - 8 * y + 55 = d) : d = 59 := by
  sorry

end value_of_d_l1545_154586


namespace find_a_l1545_154545

theorem find_a (a : ℝ) : (∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5) → (∃ x : ℝ, x = 3) → a = 8 :=
by
  sorry

end find_a_l1545_154545


namespace monthly_payment_l1545_154527

noncomputable def house_price := 280
noncomputable def deposit := 40
noncomputable def mortgage_years := 10
noncomputable def months_per_year := 12

theorem monthly_payment (house_price deposit : ℕ) (mortgage_years months_per_year : ℕ) :
  (house_price - deposit) / mortgage_years / months_per_year = 2 :=
by
  sorry

end monthly_payment_l1545_154527


namespace jack_shoes_time_l1545_154555

theorem jack_shoes_time (J : ℝ) (h : J + 2 * (J + 3) = 18) : J = 4 :=
by
  sorry

end jack_shoes_time_l1545_154555


namespace total_students_l1545_154516

theorem total_students
  (T : ℝ) 
  (h1 : 0.20 * T = 168)
  (h2 : 0.30 * T = 252) : T = 840 :=
sorry

end total_students_l1545_154516


namespace sequence_properties_l1545_154572

theorem sequence_properties (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) + a n = 4 * n) →
  (∀ n : ℕ, a n = 2 * n - 1) ∧ (a 2023 = 4045) :=
by
  sorry

end sequence_properties_l1545_154572


namespace geometric_sequence_ratio_28_l1545_154518

noncomputable def geometric_sequence_sum_ratio (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :=
  S 6 / S 3 = 28

theorem geometric_sequence_ratio_28 (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_GS : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h_increasing : ∀ n m, n < m → a1 * q^n < a1 * q^m) 
  (h_mean : 2 * 6 * a1 * q^6 = a1 * q^7 + a1 * q^8) : 
  geometric_sequence_sum_ratio a1 q S := 
by {
  -- Proof should be completed here
  sorry
}

end geometric_sequence_ratio_28_l1545_154518


namespace range_of_f_l1545_154520

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x - 1

-- Define the domain
def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

-- Define the range
def range : Set ℕ := {2, 5, 8, 11}

-- Lean 4 theorem statement
theorem range_of_f : 
  {y | ∃ x ∈ domain, y = f x} = range :=
by
  sorry

end range_of_f_l1545_154520


namespace difficult_vs_easy_l1545_154500

theorem difficult_vs_easy (x y z : ℕ) (h1 : x + y + z = 100) (h2 : x + 3 * y + 2 * z = 180) :
  x - y = 20 :=
by sorry

end difficult_vs_easy_l1545_154500


namespace tessellation_coloring_l1545_154577

theorem tessellation_coloring :
  ∀ (T : Type) (colors : T → ℕ) (adjacent : T → T → Prop),
    (∀ t1 t2, adjacent t1 t2 → colors t1 ≠ colors t2) → 
    (∃ c1 c2 c3, ∀ t, colors t = c1 ∨ colors t = c2 ∨ colors t = c3) :=
sorry

end tessellation_coloring_l1545_154577


namespace parabola_intersection_l1545_154510

theorem parabola_intersection:
  (∀ x y1 y2 : ℝ, (y1 = 3 * x^2 - 6 * x + 6) ∧ (y2 = -2 * x^2 - 4 * x + 6) → y1 = y2 → x = 0 ∨ x = 2 / 5) ∧
  (∀ a c : ℝ, a = 0 ∧ c = 2 / 5 ∧ c ≥ a → c - a = 2 / 5) :=
by sorry

end parabola_intersection_l1545_154510


namespace parabola_difference_eq_l1545_154562

variable (a b c : ℝ)

def original_parabola (x : ℝ) : ℝ := a * x^2 + b * x + c
def reflected_parabola (x : ℝ) : ℝ := -(a * x^2 + b * x + c)
def translated_original (x : ℝ) : ℝ := a * x^2 + b * x + c + 3
def translated_reflection (x : ℝ) : ℝ := -(a * x^2 + b * x + c) - 3

theorem parabola_difference_eq (x : ℝ) :
  (translated_original a b c x) - (translated_reflection a b c x) = 2 * a * x^2 + 2 * b * x + 2 * c + 6 :=
by 
  sorry

end parabola_difference_eq_l1545_154562


namespace unsuitable_temperature_for_refrigerator_l1545_154571

theorem unsuitable_temperature_for_refrigerator:
  let avg_temp := -18
  let variation := 2
  let min_temp := avg_temp - variation
  let max_temp := avg_temp + variation
  let temp_A := -17
  let temp_B := -18
  let temp_C := -19
  let temp_D := -22
  temp_D < min_temp ∨ temp_D > max_temp := by
  sorry

end unsuitable_temperature_for_refrigerator_l1545_154571


namespace product_of_good_numbers_does_not_imply_sum_digits_property_l1545_154585

-- Define what it means for a number to be "good".
def is_good (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement
theorem product_of_good_numbers_does_not_imply_sum_digits_property :
  ∀ (A B : ℕ), is_good A → is_good B → is_good (A * B) →
  ¬ (sum_digits (A * B) = sum_digits A * sum_digits B) :=
by
  intros A B hA hB hAB
  -- The detailed proof is not provided here, hence we use sorry to skip it.
  sorry

end product_of_good_numbers_does_not_imply_sum_digits_property_l1545_154585


namespace perpendicular_angles_l1545_154514

theorem perpendicular_angles (α β : ℝ) (k : ℤ) : 
  (∃ k : ℤ, β - α = k * 360 + 90 ∨ β - α = k * 360 - 90) →
  β = k * 360 + α + 90 ∨ β = k * 360 + α - 90 :=
by
  sorry

end perpendicular_angles_l1545_154514


namespace eu_countries_2012_forms_set_l1545_154551

def higher_level_skills_students := false -- Condition A can't form a set.
def tall_trees := false -- Condition B can't form a set.
def developed_cities := false -- Condition D can't form a set.
def eu_countries_2012 := true -- Condition C forms a set.

theorem eu_countries_2012_forms_set : 
  higher_level_skills_students = false ∧ tall_trees = false ∧ developed_cities = false ∧ eu_countries_2012 = true :=
by {
  sorry
}

end eu_countries_2012_forms_set_l1545_154551


namespace second_polygon_sides_l1545_154506

theorem second_polygon_sides 
  (s : ℝ) -- side length of the second polygon
  (n1 n2 : ℕ) -- n1 = number of sides of the first polygon, n2 = number of sides of the second polygon
  (h1 : n1 = 40) -- first polygon has 40 sides
  (h2 : ∀ s1 s2 : ℝ, s1 = 3 * s2 → n1 * s1 = n2 * s2 → n2 = 120)
  : n2 = 120 := 
by
  sorry

end second_polygon_sides_l1545_154506


namespace distance_between_points_on_parabola_l1545_154525

theorem distance_between_points_on_parabola (x1 y1 x2 y2 : ℝ) 
  (h_parabola : ∀ (x : ℝ), 4 * ((x^2)/4) = x^2) 
  (h_focus : F = (0, 1))
  (h_line : y1 = k * x1 + 1 ∧ y2 = k * x2 + 1)
  (h_intersects : x1^2 = 4 * y1 ∧ x2^2 = 4 * y2)
  (h_y_sum : y1 + y2 = 6) :
  |dist (x1, y1) (x2, y2)| = 8 := sorry

end distance_between_points_on_parabola_l1545_154525


namespace ratio_girls_to_boys_l1545_154565

variable (g b : ℕ)

-- Conditions: total students are 30, six more girls than boys.
def total_students : Prop := g + b = 30
def six_more_girls : Prop := g = b + 6

-- Proof that the ratio of girls to boys is 3:2.
theorem ratio_girls_to_boys (ht : total_students g b) (hs : six_more_girls g b) : g / b = 3 / 2 :=
  sorry

end ratio_girls_to_boys_l1545_154565


namespace perfect_square_condition_l1545_154541

theorem perfect_square_condition (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (gcd_xyz : Nat.gcd (Nat.gcd x y) z = 1)
    (hx_dvd : x ∣ y * z * (x + y + z))
    (hy_dvd : y ∣ x * z * (x + y + z))
    (hz_dvd : z ∣ x * y * (x + y + z))
    (sum_dvd : x + y + z ∣ x * y * z) :
  ∃ m : ℕ, m * m = x * y * z * (x + y + z) := sorry

end perfect_square_condition_l1545_154541


namespace find_factor_l1545_154509

-- Defining the given conditions
def original_number : ℕ := 7
def resultant (x: ℕ) : ℕ := 2 * x + 9
def condition (x f: ℕ) : Prop := (resultant x) * f = 69

-- The problem statement
theorem find_factor : ∃ f: ℕ, condition original_number f ∧ f = 3 :=
by
  sorry

end find_factor_l1545_154509


namespace jenny_change_l1545_154582

/-!
## Problem statement

Jenny is printing 7 copies of her 25-page essay. It costs $0.10 to print one page.
She also buys 7 pens, each costing $1.50. If she pays with $40, calculate the change she should get.
-/

def cost_per_page : ℝ := 0.10
def pages_per_copy : ℕ := 25
def num_copies : ℕ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℕ := 7
def amount_paid : ℝ := 40.0

def total_pages : ℕ := num_copies * pages_per_copy

def cost_printing : ℝ := total_pages * cost_per_page
def cost_pens : ℝ := num_pens * cost_per_pen

def total_cost : ℝ := cost_printing + cost_pens

theorem jenny_change : amount_paid - total_cost = 12 := by
  -- proof here
  sorry

end jenny_change_l1545_154582


namespace number_of_valid_n_l1545_154554

-- The definition for determining the number of positive integers n ≤ 2000 that can be represented as
-- floor(x) + floor(4x) + floor(5x) = n for some real number x.

noncomputable def count_valid_n : ℕ :=
  (200 : ℕ) * 3 + (200 : ℕ) * 2 + 1 + 1

theorem number_of_valid_n : count_valid_n = 802 :=
  sorry

end number_of_valid_n_l1545_154554


namespace find_g_neg_six_l1545_154523

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l1545_154523


namespace equivalent_weeks_l1545_154537

def hoursPerDay := 24
def daysPerWeek := 7
def hoursPerWeek := daysPerWeek * hoursPerDay
def totalHours := 2016

theorem equivalent_weeks : totalHours / hoursPerWeek = 12 := 
by
  sorry

end equivalent_weeks_l1545_154537


namespace simplify_polynomial_l1545_154589

theorem simplify_polynomial : 
  (5 - 3 * x - 7 * x^2 + 3 + 12 * x - 9 * x^2 - 8 + 15 * x + 21 * x^2) = (5 * x^2 + 24 * x) :=
by 
  sorry

end simplify_polynomial_l1545_154589


namespace curve_y_all_real_l1545_154566

theorem curve_y_all_real (y : ℝ) : ∃ (x : ℝ), 2 * x * |x| + y^2 = 1 :=
sorry

end curve_y_all_real_l1545_154566


namespace arithmetic_seq_properties_l1545_154584

theorem arithmetic_seq_properties (a : ℕ → ℝ) (d a1 : ℝ) (S : ℕ → ℝ) :
  (a 1 + a 3 = 8) ∧ (a 4 ^ 2 = a 2 * a 9) →
  ((a1 = 4 ∧ d = 0 ∧ (∀ n, S n = 4 * n)) ∨
   (a1 = 1 ∧ d = 3 ∧ (∀ n, S n = (3 * n^2 - n) / 2))) := 
sorry

end arithmetic_seq_properties_l1545_154584


namespace proof_problem_l1545_154548

def pos_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
∀ n, 4 * S n = (a n + 1) ^ 2

def sequence_condition (a : ℕ → ℝ) : Prop :=
a 0 = 1 ∧ ∀ n, a (n + 1) - a n = 2

def sum_sequence_T (a : ℕ → ℝ) (T : ℕ → ℝ) :=
∀ n, T n = (1 - 1 / (2 * n + 1))

def range_k (T : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n, T n ≥ k → k ≤ 2 / 3

theorem proof_problem (a : ℕ → ℝ) (S T : ℕ → ℝ) (k : ℝ) :
  pos_sequence a S → sequence_condition a → sum_sequence_T a T → range_k T k :=
by sorry

end proof_problem_l1545_154548


namespace relay_race_time_l1545_154593

theorem relay_race_time (M S J T : ℕ) 
(hJ : J = 30)
(hS : S = J + 10)
(hM : M = 2 * S)
(hT : T = M - 7) : 
M + S + J + T = 223 :=
by sorry

end relay_race_time_l1545_154593


namespace find_c_l1545_154567

variable (y c : ℝ)

theorem find_c (h : y > 0) (h_expr : (7 * y / 20 + c * y / 10) = 0.6499999999999999 * y) : c = 3 := by
  sorry

end find_c_l1545_154567


namespace sahil_selling_price_l1545_154528

-- Defining the conditions as variables
def purchase_price : ℕ := 14000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Defining the total cost
def total_cost : ℕ := purchase_price + repair_cost + transportation_charges

-- Calculating the profit amount
def profit : ℕ := (profit_percentage * total_cost) / 100

-- Calculating the selling price
def selling_price : ℕ := total_cost + profit

-- The Lean statement to prove the selling price is Rs 30,000
theorem sahil_selling_price : selling_price = 30000 :=
by 
  simp [total_cost, profit, selling_price]
  sorry

end sahil_selling_price_l1545_154528


namespace find_y_l1545_154564

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 :=
sorry

end find_y_l1545_154564


namespace problem_statement_l1545_154508

theorem problem_statement 
  (x y z : ℝ) 
  (hx1 : x ≠ 1) 
  (hy1 : y ≠ 1) 
  (hz1 : z ≠ 1) 
  (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 :=
sorry

end problem_statement_l1545_154508


namespace athlete_weight_l1545_154559

theorem athlete_weight (a b c : ℤ) (k₁ k₂ k₃ : ℤ)
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : a = 5 * k₁)
  (h5 : b = 5 * k₂)
  (h6 : c = 5 * k₃) :
  b = 40 :=
by
  sorry

end athlete_weight_l1545_154559


namespace minimum_value_inequality_l1545_154505

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 64) :
  ∃ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 64 ∧ (x^2 + 8 * x * y + 4 * y^2 + 4 * z^2) = 384 := 
sorry

end minimum_value_inequality_l1545_154505


namespace arith_seq_sum_nine_l1545_154598

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arith_seq := ∀ n : ℕ, a n = a 0 + (n - 1) * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n / 2) * (a 0 + a (n - 1))

theorem arith_seq_sum_nine (h_seq : arith_seq a) (h_sum : sum_first_n_terms a S) (h_S9 : S 9 = 18) : 
  a 2 + a 5 + a 8 = 6 :=
  sorry

end arith_seq_sum_nine_l1545_154598


namespace smallest_a_l1545_154533

theorem smallest_a (a x : ℤ) (hx : x^2 + a * x = 30) (ha_pos : a > 0) (product_gt_30 : ∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = -a ∧ x₁ * x₂ > 30) : a = 11 :=
sorry

end smallest_a_l1545_154533


namespace advertisements_shown_l1545_154529

theorem advertisements_shown (advertisement_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) :
  advertisement_duration = 3 →
  cost_per_minute = 4000 →
  total_cost = 60000 →
  total_cost / (advertisement_duration * cost_per_minute) = 5 :=
by
  sorry

end advertisements_shown_l1545_154529


namespace find_positive_integers_l1545_154519

theorem find_positive_integers (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / m + 1 / n - 1 / (m * n) = 2 / 5) ↔ 
  (m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4) :=
by sorry

end find_positive_integers_l1545_154519


namespace distinct_ordered_pairs_l1545_154581

/-- There are 9 distinct ordered pairs of positive integers (m, n) such that the sum of the 
    reciprocals of m and n equals 1/6. -/
theorem distinct_ordered_pairs : 
  ∃ (s : Finset (ℕ × ℕ)), s.card = 9 ∧ 
  ∀ (p : ℕ × ℕ), p ∈ s → 
    (0 < p.1 ∧ 0 < p.2) ∧ 
    (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6) :=
sorry

end distinct_ordered_pairs_l1545_154581


namespace square_side_measurement_error_l1545_154578

theorem square_side_measurement_error (S S' : ℝ) (h1 : S' = S * Real.sqrt 1.0404) : 
  (S' - S) / S * 100 = 2 :=
by
  sorry

end square_side_measurement_error_l1545_154578


namespace domain_and_parity_range_of_a_l1545_154558

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a

theorem domain_and_parity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x, f a x * g a x = f a (-x) * g a (-x)) ∧ (∀ x, -1 < x ∧ x < 1) :=
sorry

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f a 1 + g a (1/4) < 1) :
  (a ∈ (Set.Ioo 0 1 ∪ Set.Ioi (3/2))) :=
sorry

end domain_and_parity_range_of_a_l1545_154558


namespace digit_864_div_5_appending_zero_possibilities_l1545_154504

theorem digit_864_div_5_appending_zero_possibilities :
  ∀ X : ℕ, (X * 1000 + 864) % 5 ≠ 0 :=
by sorry

end digit_864_div_5_appending_zero_possibilities_l1545_154504


namespace zero_in_interval_l1545_154579

open Real

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem zero_in_interval (a b : ℕ) (h1 : b - a = 1) (h2 : 1 ≤ a) (h3 : 1 ≤ b) 
  (h4 : f a < 0) (h5 : 0 < f b) : a + b = 5 :=
sorry

end zero_in_interval_l1545_154579
